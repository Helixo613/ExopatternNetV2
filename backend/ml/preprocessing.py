"""
Preprocessing and feature extraction for stellar light curves.

Supports ~45 features per window across 5 groups:
- statistical (14): mean, std, median, min, max, ptp, skew, kurtosis, slope, intercept, MAD, RMS, mean/max abs diff
- frequency (7): FFT amplitudes, dominant freq, spectral entropy/centroid/rolloff
- wavelet (7): CWT energy at 5 scales, wavelet entropy, max-energy scale
- autocorrelation (5): ACF at lags 1/5/10, first zero crossing, decay rate
- shape (5): asymmetry, flatness, edge gradient ratio, min position, beyond-1sigma fraction
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Preprocesses light curve data and extracts features for ML models.
    """

    def __init__(self):
        self.scaler_params = None

    def preprocess(self, df: pd.DataFrame, normalize: bool = True, sigma: float = 5.0) -> pd.DataFrame:
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
        df = self._sigma_clip(df, sigma=sigma)

        # Normalize flux if requested
        if normalize:
            df['flux'] = self._normalize_flux(df['flux'])

        # Fill gaps with interpolation
        df = self._fill_gaps(df)

        return df

    def _sigma_clip(self, df: pd.DataFrame, sigma: float = 5) -> pd.DataFrame:
        """Remove extreme outliers using sigma clipping.

        IMPORTANT: returns a DataFrame with a reset index so that downstream
        integer-positional indexing (e.g. label slicing) stays aligned with
        the clipped row positions.
        """
        flux = df['flux'].values
        median = np.median(flux)
        std = np.std(flux)

        mask = np.abs(flux - median) < sigma * std
        clipped_df = df[mask].copy().reset_index(drop=True)

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

    def extract_features(self, df: pd.DataFrame, window_size: int = 50,
                         feature_groups: Optional[List[str]] = None,
                         stride: Optional[int] = None) -> np.ndarray:
        """
        Extract features from light curve using sliding windows.

        Args:
            df: DataFrame with time, flux columns
            window_size: Window size for rolling features
            feature_groups: List of feature groups to extract. Options:
                'statistical' (14 features) - original features
                'frequency' (7 features) - FFT-based
                'wavelet' (7 features) - CWT-based
                'autocorrelation' (5 features) - ACF-based
                'shape' (5 features) - morphological
                None = all groups (~38 features per window)

        Returns:
            Feature array of shape (n_windows, n_features)
        """
        if feature_groups is None:
            feature_groups = ['statistical', 'frequency', 'wavelet',
                              'autocorrelation', 'shape']

        flux = df['flux'].values
        time = df['time'].values

        n_points = len(flux)
        if stride is None:
            stride = max(1, window_size // 4)  # 75% overlap default
        else:
            stride = max(1, stride)
        features_list = []

        for i in range(0, n_points - window_size + 1, stride):
            window_flux = flux[i:i + window_size]
            window_time = time[i:i + window_size]

            feat = []

            if 'statistical' in feature_groups:
                feat.extend(self._extract_statistical_features(window_flux, window_time))

            if 'frequency' in feature_groups:
                feat.extend(self._extract_frequency_features(window_flux, window_time))

            if 'wavelet' in feature_groups:
                feat.extend(self._extract_wavelet_features(window_flux))

            if 'autocorrelation' in feature_groups:
                feat.extend(self._extract_autocorrelation_features(window_flux))

            if 'shape' in feature_groups:
                feat.extend(self._extract_shape_features(window_flux))

            features_list.append(feat)

        return np.array(features_list)

    def extract_features_with_metadata(
        self,
        df: pd.DataFrame,
        star_id: str,
        window_size: int = 50,
        feature_groups: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extract features AND return per-window metadata and labels.

        Unlike extract_features(), this function:
        - preserves star_id, window time range, and positional indices per window
        - correctly aligns labels to the post-sigma-clip DataFrame positions
          (the label array must correspond to the already-clipped df rows)

        Args:
            df: Preprocessed (already sigma-clipped + normalised) DataFrame
                with columns time, flux. Row index must be 0-based after clipping.
            star_id: Identifier for this star (used for star-level CV grouping).
            window_size: Sliding window size in cadences.
            feature_groups: Feature groups to extract (default: all 5).
            labels: Per-point label array aligned to df rows (0=normal, 1=anomaly).
                    If None, window labels are returned as all-zero.

        Returns:
            features:  np.ndarray of shape (n_windows, n_features)
            win_labels: np.ndarray of shape (n_windows,) — 1 if any point in
                        window is labelled anomalous, else 0
            metadata:  list of dicts, one per window:
                       {star_id, window_idx, start_idx, end_idx,
                        start_time, end_time, center_time}
        """
        if feature_groups is None:
            feature_groups = ['statistical', 'frequency', 'wavelet',
                              'autocorrelation', 'shape']

        flux = df['flux'].values
        time = df['time'].values
        n_points = len(flux)
        if stride is None:
            stride = max(1, window_size // 4)  # 75% overlap default
        else:
            stride = max(1, stride)

        if labels is None:
            labels = np.zeros(n_points, dtype=int)
        else:
            labels = np.asarray(labels, dtype=int)
            if len(labels) != n_points:
                raise ValueError(
                    f"labels length {len(labels)} != df length {n_points}. "
                    "Pass labels that are already aligned to the clipped DataFrame."
                )

        features_list: List[List[float]] = []
        win_labels_list: List[int] = []
        metadata_list: List[Dict[str, Any]] = []

        window_idx = 0
        for i in range(0, n_points - window_size + 1, stride):
            window_flux = flux[i:i + window_size]
            window_time = time[i:i + window_size]
            window_lab = labels[i:i + window_size]

            feat: List[float] = []
            if 'statistical' in feature_groups:
                feat.extend(self._extract_statistical_features(window_flux, window_time))
            if 'frequency' in feature_groups:
                feat.extend(self._extract_frequency_features(window_flux, window_time))
            if 'wavelet' in feature_groups:
                feat.extend(self._extract_wavelet_features(window_flux))
            if 'autocorrelation' in feature_groups:
                feat.extend(self._extract_autocorrelation_features(window_flux))
            if 'shape' in feature_groups:
                feat.extend(self._extract_shape_features(window_flux))

            features_list.append(feat)
            win_labels_list.append(int(window_lab.max() > 0))
            metadata_list.append({
                'star_id': star_id,
                'window_idx': window_idx,
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'start_time': float(window_time[0]),
                'end_time': float(window_time[-1]),
                'center_time': float(np.mean(window_time)),
            })
            window_idx += 1

        return (
            np.array(features_list),
            np.array(win_labels_list, dtype=int),
            metadata_list,
        )

    def _extract_statistical_features(self, flux: np.ndarray,
                                       time: np.ndarray) -> List[float]:
        """
        Extract 14 statistical features from a window (original feature set).

        Returns: [mean, std, median, min, max, ptp, skew, kurtosis,
                  slope, intercept, mad, rms, mean_abs_diff, max_abs_diff]
        """
        feat = [
            np.mean(flux),
            np.std(flux),
            np.median(flux),
            np.min(flux),
            np.max(flux),
            np.ptp(flux),
            stats.skew(flux),
            stats.kurtosis(flux),
        ]

        # Trend features
        if len(time) > 1:
            slope, intercept = np.polyfit(time, flux, 1)
            feat.extend([slope, intercept])
        else:
            feat.extend([0.0, 0.0])

        # Variability features
        mad = np.median(np.abs(flux - np.median(flux)))
        rms = np.sqrt(np.mean(flux**2))
        feat.extend([mad, rms])

        # Difference features
        diff = np.diff(flux)
        feat.extend([
            np.mean(np.abs(diff)),
            np.max(np.abs(diff)),
        ])

        return feat

    def _extract_frequency_features(self, flux: np.ndarray,
                                     time: np.ndarray) -> List[float]:
        """
        Extract 7 frequency-domain features using FFT.

        Returns: [fft_amp_1, fft_amp_2, fft_amp_3, dominant_freq,
                  spectral_entropy, spectral_centroid, spectral_rolloff]
        """
        n = len(flux)
        if n < 4:
            return [0.0] * 7

        # Compute FFT (real-valued input)
        flux_centered = flux - np.mean(flux)
        fft_vals = np.fft.rfft(flux_centered)
        fft_magnitudes = np.abs(fft_vals)[1:]  # skip DC component
        freqs = np.fft.rfftfreq(n)[1:]

        if len(fft_magnitudes) == 0:
            return [0.0] * 7

        # Top 3 FFT amplitudes
        sorted_idx = np.argsort(fft_magnitudes)[::-1]
        top_amps = [0.0, 0.0, 0.0]
        for k in range(min(3, len(sorted_idx))):
            top_amps[k] = float(fft_magnitudes[sorted_idx[k]])

        # Dominant frequency
        dominant_freq = float(freqs[sorted_idx[0]]) if len(sorted_idx) > 0 else 0.0

        # Spectral entropy
        psd = fft_magnitudes ** 2
        psd_norm = psd / (psd.sum() + 1e-12)
        spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

        # Spectral centroid
        spectral_centroid = float(np.sum(freqs * psd_norm))

        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(psd_norm)
        rolloff_idx = np.searchsorted(cumulative_energy, 0.85)
        spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        return top_amps + [dominant_freq, spectral_entropy,
                           spectral_centroid, spectral_rolloff]

    def _extract_wavelet_features(self, flux: np.ndarray) -> List[float]:
        """
        Extract 7 wavelet features using Continuous Wavelet Transform.

        Returns: [cwt_energy_scale1..5, wavelet_entropy, max_energy_scale]
        """
        try:
            import pywt
        except ImportError:
            # Fallback: return zeros if PyWavelets not installed
            logger.warning("PyWavelets not installed, returning zero wavelet features")
            return [0.0] * 7

        n = len(flux)
        if n < 8:
            return [0.0] * 7

        # Define scales (corresponding to different time scales)
        # Scales roughly correspond to widths: 2, 4, 8, 16, 32 samples
        scales = np.array([2, 4, 8, 16, 32])
        # Clip scales that are too large for the window
        scales = scales[scales < n // 2]
        if len(scales) == 0:
            return [0.0] * 7

        # Compute CWT using Ricker (Mexican hat) wavelet
        flux_centered = flux - np.mean(flux)
        coefficients, _ = pywt.cwt(flux_centered, scales, 'mexh')

        # Energy at each scale
        energies = np.sum(coefficients ** 2, axis=1)
        # Pad to always have 5 values
        padded_energies = np.zeros(5)
        padded_energies[:len(energies)] = energies

        # Wavelet entropy
        total_energy = energies.sum() + 1e-12
        energy_dist = energies / total_energy
        wavelet_entropy = float(-np.sum(energy_dist * np.log2(energy_dist + 1e-12)))

        # Scale with maximum energy (1-indexed)
        max_energy_scale = float(np.argmax(energies) + 1)

        return list(padded_energies) + [wavelet_entropy, max_energy_scale]

    def _extract_autocorrelation_features(self, flux: np.ndarray) -> List[float]:
        """
        Extract 5 autocorrelation features.

        Returns: [acf_lag1, acf_lag5, acf_lag10, acf_first_zero, acf_decay_rate]
        """
        n = len(flux)
        if n < 12:
            return [0.0] * 5

        flux_centered = flux - np.mean(flux)
        variance = np.var(flux)
        if variance < 1e-12:
            return [0.0] * 5

        # Vectorized ACF via np.correlate (O(n²) but in C, not Python loops)
        max_lag = min(n // 2, 30)
        full_corr = np.correlate(flux_centered, flux_centered, mode='full')
        # full_corr has length 2n-1; the zero-lag is at index n-1
        mid = n - 1
        acf = full_corr[mid:mid + max_lag + 1] / (variance * n)

        # ACF at specific lags
        acf_lag1 = float(acf[1]) if len(acf) > 1 else 0.0
        acf_lag5 = float(acf[5]) if len(acf) > 5 else 0.0
        acf_lag10 = float(acf[10]) if len(acf) > 10 else 0.0

        # First zero crossing of ACF
        neg_mask = acf[1:] <= 0
        neg_indices = np.nonzero(neg_mask)[0]
        first_zero = float(neg_indices[0] + 1) if len(neg_indices) > 0 else float(max_lag)

        # Decay rate: fit exponential to ACF envelope
        # Simplified: ratio of acf[1] to acf[5]
        if abs(acf_lag1) > 1e-6 and len(acf) > 5:
            decay_rate = float(-np.log(abs(acf_lag5) + 1e-12) / 5.0)
        else:
            decay_rate = 0.0

        return [acf_lag1, acf_lag5, acf_lag10, first_zero, decay_rate]

    def _extract_shape_features(self, flux: np.ndarray) -> List[float]:
        """
        Extract 5 shape/morphological features.

        Returns: [asymmetry, flatness, edge_gradient_ratio, min_position, beyond_1sigma_frac]
        """
        n = len(flux)
        if n < 4:
            return [0.0] * 5

        # Asymmetry: difference between mean of first half and second half
        mid = n // 2
        asymmetry = float(np.mean(flux[:mid]) - np.mean(flux[mid:]))

        # Flatness: ratio of median to mean absolute deviation
        mad = np.median(np.abs(flux - np.median(flux)))
        flatness = float(mad / (np.std(flux) + 1e-12))

        # Edge gradient ratio: mean gradient at edges vs center
        diff = np.diff(flux)
        edge_size = max(1, n // 5)
        edge_grad = np.mean(np.abs(diff[:edge_size])) + np.mean(np.abs(diff[-edge_size:]))
        center_grad = np.mean(np.abs(diff[edge_size:-edge_size])) if n > 2 * edge_size + 1 else 1e-12
        edge_gradient_ratio = float(edge_grad / (center_grad + 1e-12))

        # Position of minimum (normalized to [0, 1])
        min_position = float(np.argmin(flux) / (n - 1))

        # Fraction of points beyond 1 sigma
        std = np.std(flux)
        mean = np.mean(flux)
        beyond_1sigma = float(np.mean(np.abs(flux - mean) > std))

        return [asymmetry, flatness, edge_gradient_ratio, min_position, beyond_1sigma]

    def extract_global_periodogram_features(self, time: np.ndarray,
                                             flux: np.ndarray) -> List[float]:
        """
        Extract 7 global periodogram features (computed once per light curve).

        Returns: [ls_max_power, ls_dominant_period, ls_second_power,
                  ls_power_ratio, bls_power, bls_period, bls_duration]
        """
        features = [0.0] * 7

        if len(time) < 50:
            return features

        try:
            # Lomb-Scargle periodogram (reuse existing method)
            frequencies, power = self.compute_periodogram(time, flux)

            if len(power) > 0:
                sorted_idx = np.argsort(power)[::-1]
                features[0] = float(power[sorted_idx[0]])  # max power
                features[1] = float(1.0 / (frequencies[sorted_idx[0]] + 1e-12))  # dominant period

                if len(sorted_idx) > 1:
                    features[2] = float(power[sorted_idx[1]])  # second peak power
                    features[3] = float(features[2] / (features[0] + 1e-12))  # ratio
        except Exception as e:
            logger.debug(f"Lomb-Scargle failed: {e}")

        # BLS (Box Least Squares) for transit detection
        try:
            from astropy.timeseries import BoxLeastSquares

            time_span = time.max() - time.min()
            min_period = max(0.5, np.median(np.diff(time)) * 10)
            max_period = min(time_span / 2, 100.0)

            if max_period > min_period:
                bls = BoxLeastSquares(time, flux)
                periods = np.linspace(min_period, max_period, 500)
                result = bls.power(periods, duration=np.linspace(0.05, 0.2, 5))

                best_idx = np.argmax(result.power)
                features[4] = float(result.power[best_idx])
                features[5] = float(result.period[best_idx])
                features[6] = float(result.duration[best_idx])
        except Exception as e:
            logger.debug(f"BLS failed: {e}")

        return features

    # --- Preserved original methods ---

    def _extract_global_features(self, flux: np.ndarray) -> dict:
        """Extract global statistical features."""
        features = {
            'mean': np.mean(flux),
            'std': np.std(flux),
            'median': np.median(flux),
            'mad': np.median(np.abs(flux - np.median(flux))),
            'skewness': stats.skew(flux),
            'kurtosis': stats.kurtosis(flux),
            'p05': np.percentile(flux, 5),
            'p25': np.percentile(flux, 25),
            'p75': np.percentile(flux, 75),
            'p95': np.percentile(flux, 95),
            'iqr': np.percentile(flux, 75) - np.percentile(flux, 25),
        }
        return features

    def detect_transits_simple(self, df: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """
        Simple transit detection using flux drops.

        Returns indices where potential transits occur.
        """
        flux = df['flux'].values
        median = np.median(flux)
        std = np.std(flux)

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
        flux_centered = flux - np.mean(flux)

        from scipy.signal import lombscargle

        time_span = time.max() - time.min()
        min_freq = 1.0 / time_span
        max_freq = 1.0 / (2 * np.median(np.diff(time)))
        frequencies = np.linspace(min_freq, max_freq, 1000)

        power = lombscargle(time, flux_centered, frequencies, normalize=True)

        return frequencies, power
