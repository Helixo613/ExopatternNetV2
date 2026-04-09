"""
Data acquisition from MAST archive for real Kepler/TESS light curves
with ground truth labels from known exoplanet ephemerides.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Thread-local storage for lightkurve imports (avoids contention)
_thread_local = threading.local()


class MastDataAcquisitor:
    """
    Downloads real stellar light curves from MAST (Kepler/TESS) and creates
    per-point ground truth labels using known exoplanet ephemerides from the
    NASA Exoplanet Archive.
    """

    def __init__(self, output_dir: str = 'data/labeled', enable_cloud: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'lightcurves').mkdir(exist_ok=True)

        # Enable S3 cloud downloads — MAST data served from AWS public bucket
        # (free, no credentials needed, much faster than MAST file servers)
        if enable_cloud:
            try:
                from astroquery.mast import Observations
                import sys
                # Only enable S3 cloud in interactive (TTY) mode.
                # In parallel/background runs the astropy ProgressBarOrSpinner writes
                # to stdout from worker threads, which crashes with "I/O operation on
                # closed file" when stdout is redirected.
                if sys.stdout.isatty():
                    Observations.enable_cloud_dataset()
                    logger.info("S3 cloud downloads enabled (AWS public Kepler bucket)")
                else:
                    logger.info("Non-TTY detected — using MAST file servers (avoids S3 progress bar crash)")
            except Exception as e:
                logger.warning(f"Could not enable S3 cloud downloads: {e}. "
                               "Falling back to MAST file servers.")

    def query_confirmed_planets(self, max_targets: int = 150,
                                 mission: str = 'Kepler') -> pd.DataFrame:
        """
        Query NASA Exoplanet Archive for confirmed transiting planets.

        Returns DataFrame with columns:
            target_id, planet_name, period, epoch, duration, depth, mission
        """
        try:
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        except ImportError:
            raise ImportError("Install astroquery: pip install astroquery>=0.4.7")

        logger.info(f"Querying NASA Exoplanet Archive for confirmed planets ({mission})...")

        if mission.lower() == 'kepler':
            # Query confirmed Kepler planets with good transit parameters
            table = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name,hostname,pl_orbper,pl_tranmid,pl_trandur,pl_trandep,"
                       "disc_facility,sy_kepmag",
                where="disc_facility like '%Kepler%' "
                      "AND pl_orbper IS NOT NULL "
                      "AND pl_tranmid IS NOT NULL "
                      "AND pl_trandur IS NOT NULL "
                      "AND pl_trandep IS NOT NULL "
                      "AND sy_kepmag < 14",
                order="sy_kepmag"
            )
        elif mission.lower() == 'tess':
            table = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name,hostname,pl_orbper,pl_tranmid,pl_trandur,pl_trandep,"
                       "disc_facility,sy_tmag",
                where="disc_facility like '%TESS%' "
                      "AND pl_orbper IS NOT NULL "
                      "AND pl_tranmid IS NOT NULL "
                      "AND pl_trandur IS NOT NULL "
                      "AND pl_trandep IS NOT NULL "
                      "AND sy_tmag < 12",
                order="sy_tmag"
            )
        else:
            raise ValueError(f"Unsupported mission: {mission}. Use 'Kepler' or 'TESS'.")

        df = table.to_pandas()

        # Deduplicate by hostname (keep first/brightest planet per star)
        df = df.drop_duplicates(subset='hostname', keep='first')

        if len(df) > max_targets:
            df = df.head(max_targets)

        logger.info(f"Found {len(df)} confirmed planet hosts")

        # Standardize column names
        result = pd.DataFrame({
            'target_id': df['hostname'],
            'planet_name': df['pl_name'],
            'period': df['pl_orbper'].astype(float),
            'epoch': df['pl_tranmid'].astype(float),
            'duration_hours': df['pl_trandur'].astype(float),
            'depth_ppm': df['pl_trandep'].astype(float),
            'mission': mission,
        })

        return result.reset_index(drop=True)

    def query_non_planet_stars(self, n_stars: int = 100,
                                mission: str = 'Kepler',
                                mag_range: Tuple[float, float] = (10, 14)) -> pd.DataFrame:
        """
        Query for stars with no known planets (negative examples).

        Uses a simple approach: query bright, quiet stars not in the confirmed planet list.
        """
        try:
            import lightkurve as lk
        except ImportError:
            raise ImportError("Install lightkurve: pip install lightkurve>=2.4.0")

        logger.info(f"Querying non-planet stars for {mission}...")

        # Get list of known planet host names to exclude
        try:
            planet_hosts = self.query_confirmed_planets(max_targets=500, mission=mission)
            exclude_names = set(planet_hosts['target_id'].values)
        except Exception:
            exclude_names = set()

        # For Kepler, query the Kepler Input Catalog (KIC) via MAST Catalogs
        non_planet_ids = []
        if mission.lower() == 'kepler':
            try:
                from astroquery.mast import Catalogs
                kic = Catalogs.query_criteria(
                    catalog="Kic",
                    kp_min=mag_range[0],
                    kp_max=mag_range[1],
                    columns=["kic_kepler_id", "kic_kepmag"],
                )
                kic_df = kic.to_pandas()
                # Shuffle so we get a varied sample
                kic_df = kic_df.sample(frac=1, random_state=42).reset_index(drop=True)
                for _, krow in kic_df.iterrows():
                    kid = f"KIC {int(krow['kic_kepler_id'])}"
                    if kid not in exclude_names and len(non_planet_ids) < n_stars * 3:
                        non_planet_ids.append(kid)
            except Exception as e:
                logger.warning(f"KIC catalog query failed: {e}. "
                               "Falling back to hardcoded quiet-star list.")
                # Fallback: verified Kepler asteroseismology solar-type stars
                # All confirmed to have 15-18 quarters of long-cadence data on MAST
                _fallback_kic = [
                    "KIC 3544595", "KIC 4914423", "KIC 6196457", "KIC 6278762",
                    "KIC 6603624", "KIC 7103006", "KIC 7206837", "KIC 7670943",
                    "KIC 8379927", "KIC 8694723", "KIC 9139151", "KIC 9139163",
                    "KIC 9206432", "KIC 9353712", "KIC 9812850", "KIC 10516096",
                    "KIC 10644253", "KIC 10963065", "KIC 11244118", "KIC 11295426",
                    "KIC 12009504", "KIC 12258514", "KIC 4351319", "KIC 5184732",
                    "KIC 6116048", "KIC 6225718", "KIC 6442183", "KIC 7680114",
                    "KIC 8006161", "KIC 8150065", "KIC 8179536", "KIC 8524425",
                ]
                for kid in _fallback_kic:
                    if kid not in exclude_names:
                        non_planet_ids.append(kid)

        result = pd.DataFrame({
            'target_id': non_planet_ids[:n_stars],
            'planet_name': None,
            'period': np.nan,
            'epoch': np.nan,
            'duration_hours': np.nan,
            'depth_ppm': np.nan,
            'mission': mission,
        })

        logger.info(f"Found {len(result)} non-planet star candidates")
        return result.reset_index(drop=True)

    def download_lightcurve(self, target_id: str, mission: str = 'Kepler',
                            quarter: Optional[int] = None,
                            max_quarters: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Download a light curve from MAST via lightkurve.

        Args:
            target_id: Star identifier (e.g., 'Kepler-10', 'KIC 11904151', 'TIC 261136679')
            mission: 'Kepler' or 'TESS'
            quarter: Specific quarter/sector (None = use max_quarters or all)
            max_quarters: Max number of quarters to download (None = all). Use 3-5 for faster downloads.

        Returns:
            DataFrame with time, flux, flux_err columns, or None on failure
        """
        try:
            import lightkurve as lk
        except ImportError:
            raise ImportError("Install lightkurve: pip install lightkurve>=2.4.0")

        logger.info(f"Downloading light curve for {target_id} ({mission})...")

        try:
            import socket
            socket.setdefaulttimeout(120)  # 2-min hard timeout — prevents hung connections
            search = lk.search_lightcurve(target_id, mission=mission, cadence="long")

            if len(search) == 0:
                logger.warning(f"No light curves found for {target_id}")
                return None

            if quarter is not None:
                # Filter to specific quarter/sector
                lc_collection = search[quarter].download()
                lc = lc_collection
            elif max_quarters is not None and len(search) > max_quarters:
                # Download only first N quarters for speed
                logger.info(f"Limiting to {max_quarters} of {len(search)} available quarters")
                lc_collection = search[:max_quarters].download_all()
                if lc_collection is None or len(lc_collection) == 0:
                    logger.warning(f"Download failed for {target_id}")
                    return None
                lc = lc_collection.stitch()
            else:
                # Download all and stitch
                lc_collection = search.download_all()
                if lc_collection is None or len(lc_collection) == 0:
                    logger.warning(f"Download failed for {target_id}")
                    return None
                lc = lc_collection.stitch()

            # Remove NaN values and normalize
            lc = lc.remove_nans().normalize()

            # Convert to DataFrame
            df = pd.DataFrame({
                'time': lc.time.value,
                'flux': lc.flux.value,
                'flux_err': lc.flux_err.value if lc.flux_err is not None else np.full(len(lc), np.nan),
            })

            # Remove remaining NaN/inf
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            logger.info(f"Downloaded {len(df)} points for {target_id}")
            return df

        except Exception as e:
            logger.error(f"Error downloading {target_id}: {e}")
            return None

    def create_ground_truth_labels(self, df: pd.DataFrame, period: float,
                                    epoch: float, duration_hours: float) -> np.ndarray:
        """
        Create per-point binary ground truth labels from known transit ephemeris.

        A point is labeled 1 (anomaly/transit) if it falls within the transit window
        defined by the ephemeris parameters.

        Args:
            df: DataFrame with 'time' column (BJD or BKJD)
            period: Orbital period in days
            epoch: Transit mid-time (BJD)
            duration_hours: Transit duration in hours

        Returns:
            Binary labels array: 1 = in transit, 0 = out of transit
        """
        time = df['time'].values
        duration_days = duration_hours / 24.0
        half_duration = duration_days / 2.0

        labels = np.zeros(len(time), dtype=int)

        # Phase-fold the time series
        phase = ((time - epoch) % period) / period

        # Transit occurs near phase 0 (or equivalently phase 1)
        # Calculate phase of transit center and half-duration in phase units
        phase_half_dur = half_duration / period

        for i, p in enumerate(phase):
            # Check if point is within transit window (accounting for phase wrap)
            if p < phase_half_dur or p > (1.0 - phase_half_dur):
                labels[i] = 1

        n_transit = np.sum(labels)
        transit_frac = n_transit / len(labels) * 100
        logger.info(f"Labeled {n_transit} transit points ({transit_frac:.1f}% of {len(labels)} total)")

        return labels

    def download_and_label_target(self, target_info: pd.Series,
                                   save: bool = True,
                                   max_quarters: Optional[int] = None) -> Optional[Dict]:
        """
        Download a single target's light curve and create labels.

        Args:
            target_info: Row from query_confirmed_planets() or query_non_planet_stars()
            save: Whether to save the result to disk
            max_quarters: Max quarters to download (None = all)

        Returns:
            Dict with 'df' (DataFrame with labels), 'metadata', or None on failure
        """
        target_id = target_info['target_id']
        mission = target_info.get('mission', 'Kepler')

        df = self.download_lightcurve(target_id, mission=mission, max_quarters=max_quarters)
        if df is None:
            return None

        # Create labels if planet parameters are available
        has_planet = pd.notna(target_info.get('period'))
        if has_planet:
            labels = self.create_ground_truth_labels(
                df,
                period=target_info['period'],
                epoch=target_info['epoch'],
                duration_hours=target_info['duration_hours'],
            )
            df['label'] = labels
            label_type = 'transit'
        else:
            df['label'] = 0  # All normal for non-planet stars
            label_type = 'normal'

        metadata = {
            'target_id': target_id,
            'mission': mission,
            'n_points': len(df),
            'label_type': label_type,
            'n_anomalies': int(df['label'].sum()),
            'anomaly_fraction': float(df['label'].mean()),
        }

        if has_planet:
            metadata.update({
                'planet_name': target_info.get('planet_name'),
                'period': float(target_info['period']),
                'epoch': float(target_info['epoch']),
                'duration_hours': float(target_info['duration_hours']),
                'depth_ppm': float(target_info.get('depth_ppm', 0)),
            })

        if save:
            # Save light curve
            safe_name = target_id.replace(' ', '_').replace('+', 'p')
            filename = f"{safe_name}.csv"
            df.to_csv(self.output_dir / 'lightcurves' / filename, index=False)
            metadata['filename'] = filename

        return {'df': df, 'metadata': metadata}

    def build_dataset(self, n_planet_hosts: int = 150, n_non_planet: int = 100,
                      mission: str = 'Kepler', retry_limit: int = 3,
                      delay_between: float = 1.0,
                      max_quarters: Optional[int] = None) -> pd.DataFrame:
        """
        Build complete labeled dataset by downloading and labeling multiple targets.

        Args:
            n_planet_hosts: Number of planet-hosting stars to download
            n_non_planet: Number of non-planet stars to download
            mission: 'Kepler' or 'TESS'
            retry_limit: Max retries per target on failure
            delay_between: Seconds to wait between downloads (rate limiting)
            max_quarters: Max quarters/sectors to download per target (None = all)

        Returns:
            metadata DataFrame summarizing all downloaded targets
        """
        all_metadata = []

        # Download planet hosts
        logger.info(f"=== Downloading {n_planet_hosts} planet hosts ===")
        try:
            planet_targets = self.query_confirmed_planets(
                max_targets=n_planet_hosts, mission=mission
            )
        except Exception as e:
            logger.error(f"Failed to query confirmed planets: {e}")
            planet_targets = pd.DataFrame()

        for idx, row in planet_targets.iterrows():
            for attempt in range(retry_limit):
                try:
                    result = self.download_and_label_target(row, save=True, max_quarters=max_quarters)
                    if result is not None:
                        all_metadata.append(result['metadata'])
                        logger.info(
                            f"[{len(all_metadata)}/{n_planet_hosts + n_non_planet}] "
                            f"{row['target_id']} OK "
                            f"({result['metadata']['n_points']} pts, "
                            f"{result['metadata']['anomaly_fraction']*100:.1f}% transit)"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for {row['target_id']}: {e}")
                    if attempt < retry_limit - 1:
                        time_module.sleep(delay_between * 2)

            time_module.sleep(delay_between)

        # Download non-planet stars
        logger.info(f"=== Downloading {n_non_planet} non-planet stars ===")
        try:
            non_planet_targets = self.query_non_planet_stars(
                n_stars=n_non_planet, mission=mission
            )
        except Exception as e:
            logger.error(f"Failed to query non-planet stars: {e}")
            non_planet_targets = pd.DataFrame()

        for idx, row in non_planet_targets.iterrows():
            for attempt in range(retry_limit):
                try:
                    result = self.download_and_label_target(row, save=True, max_quarters=max_quarters)
                    if result is not None:
                        all_metadata.append(result['metadata'])
                        logger.info(
                            f"[{len(all_metadata)}/{n_planet_hosts + n_non_planet}] "
                            f"{row['target_id']} OK (normal star)"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for {row['target_id']}: {e}")
                    if attempt < retry_limit - 1:
                        time_module.sleep(delay_between * 2)

            time_module.sleep(delay_between)

        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(self.output_dir / 'metadata.csv', index=False)

        # Save summary
        summary = {
            'total_targets': len(all_metadata),
            'planet_hosts': len([m for m in all_metadata if m['label_type'] == 'transit']),
            'non_planet': len([m for m in all_metadata if m['label_type'] == 'normal']),
            'total_points': sum(m['n_points'] for m in all_metadata),
            'mission': mission,
        }
        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset complete: {summary}")
        return metadata_df

    def build_dataset_fast(self, n_planet_hosts: int = 150, n_non_planet: int = 100,
                           mission: str = 'Kepler', n_workers: int = 8,
                           max_quarters: Optional[int] = None) -> pd.DataFrame:
        """
        Optimized parallel dataset builder.

        Speed gains over build_dataset():
          1. S3 cloud downloads (enabled in __init__)
          2. Parallel search+download across n_workers threads
          3. Resume support — skips already-downloaded stars
          4. No inter-download delay (S3 handles concurrency fine)

        Args:
            n_planet_hosts: Number of planet-hosting stars
            n_non_planet: Number of non-planet stars
            mission: 'Kepler' or 'TESS'
            n_workers: Parallel download threads (default 8)
            max_quarters: Max quarters per star (None = all)

        Returns:
            metadata DataFrame
        """
        # --- 1. Query target lists (fast, single API call each) ---
        logger.info(f"=== Fast parallel download: {n_planet_hosts} planets + {n_non_planet} normal ===")

        planet_targets = pd.DataFrame()
        non_planet_targets = pd.DataFrame()

        try:
            planet_targets = self.query_confirmed_planets(
                max_targets=n_planet_hosts, mission=mission
            )
        except Exception as e:
            logger.error(f"Failed to query planet hosts: {e}")

        try:
            non_planet_targets = self.query_non_planet_stars(
                n_stars=n_non_planet, mission=mission
            )
        except Exception as e:
            logger.error(f"Failed to query non-planet stars: {e}")

        all_targets = pd.concat([planet_targets, non_planet_targets], ignore_index=True)
        total = len(all_targets)
        logger.info(f"Total targets to process: {total}")

        # --- 2. Check for already-downloaded stars (resume support) ---
        lc_dir = self.output_dir / 'lightcurves'
        existing = set()
        for f in lc_dir.glob('*.csv'):
            existing.add(f.stem.replace('_', ' '))

        to_download = []
        already_done_meta = []
        for _, row in all_targets.iterrows():
            tid = row['target_id']
            if tid in existing or tid.replace(' ', '_') in existing:
                logger.info(f"  Skipping {tid} (already downloaded)")
                # Rebuild metadata from existing file
                safe_name = tid.replace(' ', '_').replace('+', 'p')
                filename = f"{safe_name}.csv"
                fpath = lc_dir / filename
                if fpath.exists():
                    df = pd.read_csv(fpath)
                    has_planet = pd.notna(row.get('period'))
                    meta = {
                        'target_id': tid, 'mission': mission,
                        'n_points': len(df),
                        'label_type': 'transit' if has_planet else 'normal',
                        'n_anomalies': int(df['label'].sum()) if 'label' in df.columns else 0,
                        'anomaly_fraction': float(df['label'].mean()) if 'label' in df.columns else 0.0,
                        'filename': filename,
                    }
                    if has_planet:
                        meta.update({
                            'planet_name': row.get('planet_name'),
                            'period': float(row['period']),
                            'epoch': float(row['epoch']),
                            'duration_hours': float(row['duration_hours']),
                            'depth_ppm': float(row.get('depth_ppm', 0)),
                        })
                    already_done_meta.append(meta)
            else:
                to_download.append(row)

        logger.info(f"Already downloaded: {len(already_done_meta)}, remaining: {len(to_download)}")

        if not to_download:
            logger.info("All targets already downloaded!")
            metadata_df = pd.DataFrame(already_done_meta)
            metadata_df.to_csv(self.output_dir / 'metadata.csv', index=False)
            return metadata_df

        # --- 3. Parallel download + label ---
        all_metadata = list(already_done_meta)
        completed = len(already_done_meta)
        failed = []
        lock = threading.Lock()

        def _download_one(row):
            """Download and label a single target (runs in worker thread)."""
            tid = row['target_id']
            for attempt in range(3):
                try:
                    # Run the actual download in a daemon thread with a hard timeout.
                    # This is the only reliable way to abort a hung urllib/boto3 transfer
                    # since socket.setdefaulttimeout only covers connect, not read stalls.
                    _result = [None]
                    _error = [None]

                    def _do():
                        try:
                            _result[0] = self.download_and_label_target(
                                row, save=True, max_quarters=max_quarters
                            )
                        except Exception as e:
                            _error[0] = e

                    t = threading.Thread(target=_do, daemon=True)
                    t.start()
                    t.join(timeout=600)  # 10-min hard cap per attempt

                    if t.is_alive():
                        logger.warning(f"Attempt {attempt+1} timed out (600s) for {tid} — retrying")
                        continue
                    if _error[0]:
                        raise _error[0]
                    if _result[0] is not None:
                        return _result[0]['metadata']
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for {tid}: {e}")
                    time_module.sleep(1)
            return None

        logger.info(f"Starting parallel download with {n_workers} workers...")
        t0 = time_module.time()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_target = {
                executor.submit(_download_one, row): row['target_id']
                for row in to_download
            }

            for future in as_completed(future_to_target):
                tid = future_to_target[future]
                try:
                    meta = future.result()
                    with lock:
                        if meta is not None:
                            all_metadata.append(meta)
                            completed += 1
                            elapsed = time_module.time() - t0
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / rate if rate > 0 else 0
                            logger.info(
                                f"[{completed}/{total}] {tid} OK "
                                f"({meta['n_points']} pts) "
                                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
                            )
                        else:
                            failed.append(tid)
                            logger.warning(f"[FAILED] {tid}")
                except Exception as e:
                    failed.append(tid)
                    logger.error(f"[ERROR] {tid}: {e}")

        elapsed_total = time_module.time() - t0
        logger.info(
            f"\nDownload complete in {elapsed_total:.0f}s "
            f"({len(all_metadata)} OK, {len(failed)} failed)"
        )
        if failed:
            logger.warning(f"Failed targets: {failed}")

        # --- 4. Save metadata ---
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(self.output_dir / 'metadata.csv', index=False)

        summary = {
            'total_targets': len(all_metadata),
            'planet_hosts': len([m for m in all_metadata if m['label_type'] == 'transit']),
            'non_planet': len([m for m in all_metadata if m['label_type'] == 'normal']),
            'total_points': sum(m['n_points'] for m in all_metadata),
            'mission': mission,
            'download_time_seconds': elapsed_total,
        }
        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset complete: {summary}")
        return metadata_df

    def load_labeled_dataset(self) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Load a previously downloaded labeled dataset from disk.

        Returns:
            lightcurves: List of DataFrames (each with time, flux, flux_err, label)
            metadata: DataFrame with target info
        """
        metadata_path = self.output_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No dataset found at {self.output_dir}. Run build_dataset() first."
            )

        metadata = pd.read_csv(metadata_path)
        lightcurves = []

        for _, row in metadata.iterrows():
            lc_path = self.output_dir / 'lightcurves' / row['filename']
            if lc_path.exists():
                df = pd.read_csv(lc_path)
                lightcurves.append(df)
            else:
                logger.warning(f"Missing file: {lc_path}")

        logger.info(f"Loaded {len(lightcurves)} labeled light curves")
        return lightcurves, metadata
