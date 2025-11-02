"""
Light curve data loader supporting FITS and CSV formats.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightCurveLoader:
    """
    Loads and validates light curve data from FITS or CSV files.
    """

    REQUIRED_COLUMNS = ['time', 'flux']
    OPTIONAL_COLUMNS = ['flux_err']

    def __init__(self):
        self.data = None
        self.metadata = {}

    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load light curve from FITS or CSV file.

        Args:
            file_path: Path to the light curve file

        Returns:
            DataFrame with columns: time, flux, flux_err (optional)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.fits':
            return self._load_fits(file_path)
        elif suffix == '.csv':
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .fits or .csv")

    def _load_fits(self, file_path: Path) -> pd.DataFrame:
        """Load light curve from FITS file."""
        logger.info(f"Loading FITS file: {file_path}")

        try:
            with fits.open(file_path) as hdul:
                # Try to find the data in different HDUs
                data = None

                # Check for LIGHTCURVE extension (Kepler/TESS standard)
                if 'LIGHTCURVE' in [hdu.name for hdu in hdul]:
                    data = hdul['LIGHTCURVE'].data
                # Otherwise use the first binary table
                elif len(hdul) > 1 and isinstance(hdul[1], (fits.BinTableHDU, fits.TableHDU)):
                    data = hdul[1].data
                else:
                    raise ValueError("No valid data table found in FITS file")

                # Extract metadata from primary header
                if len(hdul) > 0:
                    header = hdul[0].header
                    self.metadata = {
                        'object_name': header.get('OBJECT', 'Unknown'),
                        'telescope': header.get('TELESCOP', 'Unknown'),
                        'instrument': header.get('INSTRUME', 'Unknown'),
                    }

                # Convert to DataFrame - handle common column name variations
                df = pd.DataFrame()

                # Time column (try multiple names)
                time_cols = ['TIME', 'time', 'JD', 'MJD', 'BJD', 'BTJD']
                for col in time_cols:
                    if col in data.names:
                        df['time'] = data[col]
                        break

                # Flux column (try multiple names)
                flux_cols = ['FLUX', 'flux', 'SAP_FLUX', 'PDCSAP_FLUX']
                for col in flux_cols:
                    if col in data.names:
                        df['flux'] = data[col]
                        break

                # Flux error column (optional)
                error_cols = ['FLUX_ERR', 'flux_err', 'SAP_FLUX_ERR', 'PDCSAP_FLUX_ERR', 'ERROR']
                for col in error_cols:
                    if col in data.names:
                        df['flux_err'] = data[col]
                        break

                if 'time' not in df.columns or 'flux' not in df.columns:
                    raise ValueError(f"Required columns not found. Available columns: {data.names}")

                logger.info(f"Successfully loaded {len(df)} data points from FITS")
                return self._validate_and_clean(df)

        except Exception as e:
            logger.error(f"Error loading FITS file: {str(e)}")
            raise

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load light curve from CSV file."""
        logger.info(f"Loading CSV file: {file_path}")

        try:
            # Try to read CSV with different delimiters
            for delimiter in [',', '\t', ' ', ';']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    if len(df.columns) >= 2:
                        break
                except:
                    continue
            else:
                raise ValueError("Could not parse CSV file with common delimiters")

            # Normalize column names (case-insensitive matching)
            df.columns = df.columns.str.lower().str.strip()

            # Check for required columns
            if 'time' not in df.columns or 'flux' not in df.columns:
                # Try to auto-detect if first two columns are time and flux
                if len(df.columns) >= 2:
                    old_cols = df.columns.tolist()
                    df = df.rename(columns={
                        old_cols[0]: 'time',
                        old_cols[1]: 'flux'
                    })
                    if len(old_cols) >= 3:
                        df = df.rename(columns={old_cols[2]: 'flux_err'})
                    logger.warning(f"Auto-detected columns: {old_cols} -> time, flux, flux_err")
                else:
                    raise ValueError(f"Required columns 'time' and 'flux' not found. Available: {df.columns.tolist()}")

            logger.info(f"Successfully loaded {len(df)} data points from CSV")
            return self._validate_and_clean(df)

        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the light curve data.
        """
        logger.info("Validating and cleaning data...")

        # Remove NaN and infinite values
        original_len = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['time', 'flux'])

        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} rows with NaN/inf values")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Check for minimum data points
        if len(df) < 10:
            raise ValueError(f"Insufficient data points: {len(df)}. Need at least 10.")

        # Add flux_err if not present (use std as estimate)
        if 'flux_err' not in df.columns:
            df['flux_err'] = df['flux'].std() * 0.01  # 1% of std as default error
            logger.info("Added default flux_err column")

        logger.info(f"Validation complete. Clean dataset: {len(df)} points")
        return df

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the light curve.
        """
        return {
            'n_points': len(df),
            'time_span': float(df['time'].max() - df['time'].min()),
            'time_start': float(df['time'].min()),
            'time_end': float(df['time'].max()),
            'flux_mean': float(df['flux'].mean()),
            'flux_std': float(df['flux'].std()),
            'flux_min': float(df['flux'].min()),
            'flux_max': float(df['flux'].max()),
            'flux_median': float(df['flux'].median()),
        }
