"""
Preprocessing and feature extraction for stellar light curves.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Preprocesses light curve data and extracts features for ML models.
    """

    def __init__(self):
        self.scaler_params = None

    def preprocess(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Preprocess light curve data.

        Args:
            df: DataFrame with time, flux, flux_err columns
            normalize: Whether to normalize the flux

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Remove outliers using sigma clipping
        df = self._sigma_clip(df, sigma=5)

        # Normalize flux if requested
        if normalize:
            df['flux'] = self._normalize_flux(df['flux'])

        # Fill gaps with interpolation
        df = self._fill_gaps(df)

        return df

    def _sigma_clip(self, df: pd.DataFrame, sigma: float = 5) -> pd.DataFrame:
        """Remove extreme outliers using sigma clipping."""
        flux = df['flux'].values
        median = np.median(flux)
        std = np.std(flux)

        mask = np.abs(flux - median) < sigma * std
        clipped_df = df[mask].copy()

        n_removed = len(df) - len(clipped_df)
        if n_removed > 0:
            logger.info(f"Sigma clipping removed {n_removed} outliers")

        return clipped_df

    def _normalize_flux(self, flux: pd.Series) -> pd.Series:
        """Normalize flux to zero mean and unit variance."""
        mean = flux.mean()
        std = flux.std()

        # Store for inverse transform
        self.scaler_params = {'mean': mean, 'std': std}

        normalized = (flux - mean) / std
        return normalized

    def denormalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """Inverse transform normalized flux."""
        if self.scaler_params is None:
            return flux

        return flux * self.scaler_params['std'] + self.scaler_params['mean']

    def _fill_gaps(self, df: pd.DataFrame, max_gap: Optional[float] = None) -> pd.DataFrame:
        """
        Fill small gaps in time series using linear interpolation.

        Args:
            df: DataFrame with time and flux
            max_gap: Maximum gap size to fill (in time units). If None, no limit.
        """
        # Check for gaps in time series
        time_diff = np.diff(df['time'].values)
        median_cadence = np.median(time_diff)

        # If gaps are small, interpolation is not needed
        if max_gap is not None and np.max(time_diff) < max_gap:
            return df

        # Simple forward fill for small gaps only
        df = df.interpolate(method='linear', limit=5)

        return df

    def extract_features(self, df: pd.DataFrame, window_size: int = 50) -> np.ndarray:
        """
        Extract statistical and morphological features from light curve.

        Features include:
        - Basic statistics (mean, std, skewness, kurtosis)
        - Percentiles
        - Variability metrics
        - Slope and trend features
        - Frequency domain features

        Args:
            df: DataFrame with time, flux columns
            window_size: Window size for rolling features

        Returns:
            Feature array of shape (n_windows, n_features)
        """
        flux = df['flux'].values
        time = df['time'].values

        # Global features
        global_features = self._extract_global_features(flux)

        # Window-based features
        window_features = self._extract_window_features(flux, time, window_size)

        return window_features

    def _extract_global_features(self, flux: np.ndarray) -> dict:
        """Extract global statistical features."""
        features = {
            'mean': np.mean(flux),
            'std': np.std(flux),
            'median': np.median(flux),
            'mad': np.median(np.abs(flux - np.median(flux))),  # Median Absolute Deviation
            'skewness': stats.skew(flux),
            'kurtosis': stats.kurtosis(flux),
            'p05': np.percentile(flux, 5),
            'p25': np.percentile(flux, 25),
            'p75': np.percentile(flux, 75),
            'p95': np.percentile(flux, 95),
            'iqr': np.percentile(flux, 75) - np.percentile(flux, 25),
        }
        return features

    def _extract_window_features(self, flux: np.ndarray, time: np.ndarray,
                                 window_size: int) -> np.ndarray:
        """
        Extract features using sliding windows.

        This is crucial for detecting localized anomalies like transits or flares.
        """
        n_points = len(flux)
        stride = max(1, window_size // 2)  # 50% overlap (reduced from 75% to prevent amplification)
        features_list = []

        for i in range(0, n_points - window_size + 1, stride):
            window_flux = flux[i:i + window_size]
            window_time = time[i:i + window_size]

            # Statistical features
            feat = [
                np.mean(window_flux),
                np.std(window_flux),
                np.median(window_flux),
                np.min(window_flux),
                np.max(window_flux),
                np.ptp(window_flux),  # peak-to-peak
                stats.skew(window_flux),
                stats.kurtosis(window_flux),
            ]

            # Trend features
            if len(window_time) > 1:
                slope, intercept = np.polyfit(window_time, window_flux, 1)
                feat.extend([slope, intercept])
            else:
                feat.extend([0, 0])

            # Variability features
            mad = np.median(np.abs(window_flux - np.median(window_flux)))
            rms = np.sqrt(np.mean(window_flux**2))
            feat.extend([mad, rms])

            # Difference features (rate of change)
            diff = np.diff(window_flux)
            feat.extend([
                np.mean(np.abs(diff)),
                np.max(np.abs(diff)),
            ])

            features_list.append(feat)

        return np.array(features_list)

    def detect_transits_simple(self, df: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """
        Simple transit detection using flux drops.

        Returns indices where potential transits occur.
        """
        flux = df['flux'].values
        median = np.median(flux)
        std = np.std(flux)

        # Find points significantly below median
        transit_mask = flux < (median - threshold * std)

        return np.where(transit_mask)[0]

    def smooth_lightcurve(self, flux: np.ndarray, window_length: int = 21,
                         polyorder: int = 3) -> np.ndarray:
        """
        Smooth light curve using Savitzky-Golay filter.

        Useful for detrending and removing high-frequency noise.
        """
        if len(flux) < window_length:
            window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1

        if window_length < polyorder + 2:
            return flux

        smoothed = signal.savgol_filter(flux, window_length, polyorder)
        return smoothed

    def compute_periodogram(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lomb-Scargle periodogram for frequency analysis.

        Returns:
            frequencies, power
        """
        # Remove mean
        flux_centered = flux - np.mean(flux)

        # Compute Lomb-Scargle periodogram
        from scipy.signal import lombscargle

        # Define frequency grid
        time_span = time.max() - time.min()
        min_freq = 1.0 / time_span
        max_freq = 1.0 / (2 * np.median(np.diff(time)))  # Nyquist frequency
        frequencies = np.linspace(min_freq, max_freq, 1000)

        # Compute periodogram
        power = lombscargle(time, flux_centered, frequencies, normalize=True)

        return frequencies, power
