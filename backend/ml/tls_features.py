"""
Transit Least Squares (TLS) feature extraction.

Two layers of features:

1. Per-star global features (6, computed once per star):
   Appended as constant values to every window of that star.
   These enter the anomaly models and affect window-level scoring.

2. Per-event consistency features (3, computed per CandidateEvent):
   Measure how well a candidate event aligns with the TLS-inferred
   periodic signal.  These are computed AFTER candidate generation
   and only enter the final ranking score — NOT candidate generation.

Normalization of the 3 consistency features is fit on proper-training
candidates only, then frozen for calibration and test candidates.

Reference: Hippke & Heller (2019), A&A 623, A39
           https://www.aanda.org/articles/aa/abs/2019/03/aa34672-18/
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Feature names for the 6 global TLS features
TLS_GLOBAL_FEATURE_NAMES = [
    'tls_sde',
    'tls_period',
    'tls_depth',
    'tls_duration',
    'tls_odd_even',
    'tls_snr',
]

# Feature names for the 3 event-consistency features
TLS_CONSISTENCY_FEATURE_NAMES = [
    'tls_epoch_distance',
    'tls_phase_agreement',
    'tls_depth_ratio',
]

# Fallback values when TLS fails (conservative: no signal)
_TLS_FALLBACK = {
    'tls_sde':      0.0,
    'tls_period':   0.0,
    'tls_depth':    0.0,
    'tls_duration': 0.0,
    'tls_odd_even': 1.0,   # 1.0 = no odd/even mismatch (neutral)
    'tls_snr':      0.0,
    'tls_t0':       np.nan,  # internal only; not appended to window features
}


# ---------------------------------------------------------------------------
# Per-star global features
# ---------------------------------------------------------------------------

def extract_tls_features(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    Run TLS on a single star's light curve and return 6 global features.

    Args:
        time: time array (BKJD)
        flux: normalized flux array (same length as time)
        period_min: minimum search period in days (default: 0.5)
        period_max: maximum search period in days (default: time_span / 3)

    Returns:
        dict with keys: tls_sde, tls_period, tls_depth, tls_duration,
                        tls_odd_even, tls_snr, and internal tls_t0
    """
    try:
        from transitleastsquares import transitleastsquares

        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)

        # Remove NaNs
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        if len(time) < 100:
            logger.warning("Too few points for TLS — returning fallback features")
            return dict(_TLS_FALLBACK)

        # TLS requires flux normalized to mean ~1.0.
        # Divide by median to handle any upstream z-score normalization.
        flux_median = float(np.median(flux))
        if abs(flux_median) > 1e-9:
            flux = flux / flux_median

        time_span = float(time.max() - time.min())
        if period_min is None:
            period_min = 0.5
        if period_max is None:
            # Cap at 150 days — longer periods are undetectable with <3 transits
            # in typical Kepler baselines and greatly inflate the period grid.
            period_max = min(max(period_min + 0.1, time_span / 3.0), 150.0)

        # Downsample to at most 20k points for TLS to keep runtime tractable.
        # TLS runtime scales as O(n_points × n_periods); the transit shape is
        # preserved at Kepler cadence even after binning by 5-10x.
        MAX_TLS_POINTS = 10_000
        if len(time) > MAX_TLS_POINTS:
            factor = len(time) // MAX_TLS_POINTS
            n_keep = (len(time) // factor) * factor
            time_ds = time[:n_keep].reshape(-1, factor).mean(axis=1)
            flux_ds = flux[:n_keep].reshape(-1, factor).mean(axis=1)
        else:
            time_ds, flux_ds = time, flux

        model = transitleastsquares(time_ds, flux_ds)
        results = model.power(
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=1,
            duration_grid_step=1.1,
            show_progress_bar=False,
        )

        # Odd-even mismatch: ratio of odd to even transit depth
        # TLS returns these directly; guard against division by zero
        odd_depth  = float(getattr(results, 'depth_odd',  results.depth) or results.depth)
        even_depth = float(getattr(results, 'depth_even', results.depth) or results.depth)
        if even_depth > 1e-9:
            odd_even = odd_depth / even_depth
        else:
            odd_even = 1.0

        t0 = getattr(results, 'T0', None)
        if t0 is None:
            t0 = getattr(results, 'transit_time', None)
        if t0 is None:
            t0 = getattr(results, 'transit_times', None)
        if t0 is None:
            t0_value = np.nan
        elif np.isscalar(t0):
            t0_value = float(t0)
        else:
            t0_arr = np.asarray(t0).ravel()
            t0_value = float(t0_arr[0]) if t0_arr.size else np.nan

        return {
            'tls_sde':      float(results.SDE),
            'tls_period':   float(results.period),
            'tls_depth':    float(results.depth),
            'tls_duration': float(results.duration),
            'tls_odd_even': float(odd_even),
            'tls_snr':      float(results.snr),
            'tls_t0':       t0_value,
        }

    except Exception as e:
        logger.warning(f"TLS failed: {e} — returning fallback features")
        return dict(_TLS_FALLBACK)


def extract_tls_features_batch(
    metadata_csv: str,
    lightcurve_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Extract TLS global features for all stars in metadata.csv.

    Returns:
        dict mapping star_id -> {tls_sde, tls_period, ...}
    """
    import pandas as pd
    from pathlib import Path

    meta = pd.read_csv(metadata_csv)
    results: Dict[str, Dict[str, float]] = {}

    for i, row in meta.iterrows():
        sid = str(row['target_id'])
        lc_path = Path(lightcurve_dir) / row['filename']
        try:
            df = pd.read_csv(lc_path)
            feats = extract_tls_features(
                time=df['time'].values,
                flux=df['flux'].values,
            )
            results[sid] = feats
            logger.info(
                f"[{i+1}/{len(meta)}] {sid}: "
                f"SDE={feats['tls_sde']:.2f} "
                f"period={feats['tls_period']:.3f}d"
            )
        except Exception as e:
            logger.warning(f"Could not process {sid}: {e}")
            results[sid] = dict(_TLS_FALLBACK)

    return results


def append_tls_to_windows(
    features: np.ndarray,
    tls_features: Dict[str, float],
) -> np.ndarray:
    """
    Append 6 TLS global features to a star's window feature matrix.

    The 6 values are constant across all windows (global per-star features).

    Args:
        features: (n_windows, n_features) — existing window features
        tls_features: dict of 6 TLS values for this star

    Returns:
        (n_windows, n_features + 6) array
    """
    n_windows = features.shape[0]
    tls_row = np.array([tls_features[k] for k in TLS_GLOBAL_FEATURE_NAMES], dtype=float)
    tls_block = np.tile(tls_row, (n_windows, 1))
    return np.hstack([features, tls_block])


# ---------------------------------------------------------------------------
# Per-event consistency features
# ---------------------------------------------------------------------------

def compute_event_consistency(
    candidate,           # CandidateEvent
    tls_features: Dict[str, float],
    flux_dip_depth: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute 3 TLS event-consistency features for a single candidate event.

    These measure how well the candidate aligns with the TLS-inferred
    periodic signal and are computed AFTER candidate generation.

    Args:
        candidate: CandidateEvent object
        tls_features: global TLS features for the candidate's star
        flux_dip_depth: observed flux dip depth at the candidate center
                        (optional; if None, tls_depth_ratio is set to NaN)

    Returns:
        dict with tls_epoch_distance, tls_phase_agreement, tls_depth_ratio
    """
    def _coerce_float(value, default=np.nan) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    period = _coerce_float(tls_features.get('tls_period', 0.0), default=np.nan)
    depth = _coerce_float(tls_features.get('tls_depth', 0.0), default=np.nan)
    t0 = _coerce_float(tls_features.get('tls_t0', np.nan), default=np.nan)

    if not np.isfinite(period) or period <= 0:
        return {
            'tls_epoch_distance': np.nan,
            'tls_phase_agreement': np.nan,
            'tls_depth_ratio': np.nan,
        }

    center = candidate.center_time

    # tls_epoch_distance: distance from candidate center to the nearest
    # TLS-predicted transit epoch anchored by tls_t0.
    if np.isfinite(t0):
        nearest_n = int(np.round((center - t0) / period))
        nearest_epoch = t0 + nearest_n * period
        epoch_distance = abs(center - nearest_epoch)
        phase_agreement = 1.0 - (epoch_distance / (period / 2.0))
        phase_agreement = float(np.clip(phase_agreement, 0.0, 1.0))
    else:
        epoch_distance = np.nan
        phase_agreement = np.nan

    # tls_depth_ratio: observed_depth / tls_depth.
    # Values near 1.0 = consistent with TLS model.
    # Extreme values = unusual (too deep, too shallow, or sign-inverted).
    if flux_dip_depth is not None and depth > 1e-9:
        depth_ratio = float(flux_dip_depth / depth)
    else:
        depth_ratio = np.nan

    return {
        'tls_epoch_distance':  float(epoch_distance),
        'tls_phase_agreement': phase_agreement,
        'tls_depth_ratio':     depth_ratio,
    }


def attach_consistency_features(
    candidates: list,
    tls_by_star: Dict[str, Dict[str, float]],
    flux_dip_by_candidate: Optional[Dict] = None,
) -> None:
    """
    In-place: attach TLS consistency features to each CandidateEvent.

    Args:
        candidates: list of CandidateEvent objects
        tls_by_star: dict {star_id -> tls_features_dict}
        flux_dip_by_candidate: optional dict {candidate_id -> observed_depth}
    """
    for cand in candidates:
        tls_feats = tls_by_star.get(cand.star_id, dict(_TLS_FALLBACK))
        dip = None
        if flux_dip_by_candidate is not None:
            dip = flux_dip_by_candidate.get(id(cand))
        consistency = compute_event_consistency(cand, tls_feats, dip)
        cand.tls_epoch_distance  = consistency['tls_epoch_distance']
        cand.tls_phase_agreement = consistency['tls_phase_agreement']
        cand.tls_depth_ratio     = consistency['tls_depth_ratio']


# ---------------------------------------------------------------------------
# Consistency score normalizer (fit on proper-training candidates only)
# ---------------------------------------------------------------------------

class ConsistencyScoreNormalizer:
    """
    Normalizes the 3 TLS consistency features into a single [0, 1] score.

    Fit on proper-training candidates only, then frozen for calibration
    and test candidates (blueprint D6 constraint).
    """

    def __init__(self) -> None:
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._fitted = False

    def _feature_vector(self, candidate) -> np.ndarray:
        """Extract the 3 consistency features as a vector, replacing NaN with 0."""
        epoch_distance = candidate.tls_epoch_distance
        if epoch_distance is None or not np.isfinite(epoch_distance):
            epoch_distance = 0.0

        phase_agreement = candidate.tls_phase_agreement
        if phase_agreement is None or not np.isfinite(phase_agreement):
            phase_agreement = 0.5

        depth_ratio = candidate.tls_depth_ratio
        if depth_ratio is None or not np.isfinite(depth_ratio):
            depth_ratio = 1.0

        return np.array([epoch_distance, phase_agreement, depth_ratio], dtype=float)

    def fit(self, train_candidates: list) -> 'ConsistencyScoreNormalizer':
        """Fit min-max on proper-training candidate consistency features."""
        if not train_candidates:
            logger.warning("No training candidates for ConsistencyScoreNormalizer")
            self._mins = np.zeros(3)
            self._maxs = np.ones(3)
            self._fitted = True
            return self

        X = np.vstack([self._feature_vector(c) for c in train_candidates])
        self._mins = X.min(axis=0)
        self._maxs = X.max(axis=0)
        # Prevent zero range
        for i in range(3):
            if self._maxs[i] - self._mins[i] < 1e-9:
                self._maxs[i] = self._mins[i] + 1.0
        self._fitted = True
        return self

    def score(self, candidate) -> float:
        """
        Return a single consistency score in [0, 1].

        Higher = more consistent with the TLS periodic model
        (i.e., candidate aligns well with known transit epochs and depth).
        Used directly in the ranking blend: alpha*composite + (1-alpha)*consistency.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        v = self._feature_vector(candidate)
        normed = np.clip((v - self._mins) / (self._maxs - self._mins), 0.0, 1.0)

        # epoch_distance: smaller distance = more consistent, so invert
        epoch_score = 1.0 - normed[0]
        # phase_agreement: higher = more consistent (already directional)
        phase_score = normed[1]
        # depth_ratio: physically meaningful target is 1.0 (observed ≈ TLS depth).
        # Score in raw space so the peak is always at ratio=1.0, regardless of the
        # training distribution used for normalization.
        depth_ratio = v[2]  # raw value from _feature_vector
        depth_score = float(np.clip(1.0 - abs(depth_ratio - 1.0), 0.0, 1.0))

        return float(np.mean([epoch_score, phase_score, depth_score]))
