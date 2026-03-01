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

logger = logging.getLogger(__name__)


class MastDataAcquisitor:
    """
    Downloads real stellar light curves from MAST (Kepler/TESS) and creates
    per-point ground truth labels using known exoplanet ephemerides from the
    NASA Exoplanet Archive.
    """

    def __init__(self, output_dir: str = 'data/labeled'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'lightcurves').mkdir(exist_ok=True)

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

        # For Kepler, use a catalog of quiet stars
        # We'll search for stars in a magnitude range and filter out planet hosts
        if mission.lower() == 'kepler':
            # Search for generic Kepler targets
            search_results = lk.search_lightcurve(
                f"KIC",
                mission="Kepler",
                cadence="long",
            )
            # This is a broad search; we'll pick from available targets
            # In practice, for a real dataset build, use the Kepler Input Catalog
            non_planet_ids = []
            if len(search_results) > 0:
                unique_targets = search_results.table['target_name']
                for target in unique_targets:
                    if target not in exclude_names and len(non_planet_ids) < n_stars:
                        non_planet_ids.append(target)

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
