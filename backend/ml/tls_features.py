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
                        tls_odd_even, tls_snr
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

        time_span = float(time.max() - time.min())
        if period_min is None:
            period_min = 0.5
        if period_max is None:
            period_max = max(period_min + 0.1, time_span / 3.0)

        model = transitleastsquares(time, flux)
        results = model.power(
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=3,
            duration_grid_step=1.05,
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

        return {
            'tls_sde':      float(results.SDE),
            'tls_period':   float(results.period),
            'tls_depth':    float(results.depth),
            'tls_duration': float(results.duration),
            'tls_odd_even': float(odd_even),
            'tls_snr':      float(results.snr),
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
    period   = tls_features.get('tls_period', 0.0)
    depth    = tls_features.get('tls_depth',  0.0)

    if period <= 0:
        return {
            'tls_epoch_distance': np.nan,
            'tls_phase_agreement': np.nan,
            'tls_depth_ratio': np.nan,
        }

    center = candidate.center_time

    # tls_epoch_distance: distance from candidate center to nearest
    # predicted transit epoch (in days).  Small = candidate coincides
    # with a known periodic signal.
    phase = (center % period) / period          # fractional phase [0, 1)
    # Distance to nearest transit (phase 0.0), wrapping around
    phase_dist = min(phase, 1.0 - phase)        # in [0, 0.5]
    epoch_distance = phase_dist * period        # back to days

    # tls_phase_agreement: 1 - (epoch_distance / (period/2))
    # = 1 when candidate is exactly on a transit epoch
    # = 0 when candidate is halfway between transits
    phase_agreement = 1.0 - (epoch_distance / (period / 2.0))
    phase_agreement = float(np.clip(phase_agreement, 0.0, 1.0))

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
        v = np.array([
            candidate.tls_epoch_distance  if candidate.tls_epoch_distance  is not None else 0.0,
            candidate.tls_phase_agreement if candidate.tls_phase_agreement is not None else 0.5,
            candidate.tls_depth_ratio     if candidate.tls_depth_ratio     is not None else 1.0,
        ], dtype=float)
        v = np.nan_to_num(v, nan=0.0)
        return v

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

        Note: for anomaly ranking we INVERT this — a candidate that is
        INCONSISTENT with TLS (novel morphology, unusual depth, off-epoch)
        may be MORE interesting.  The inversion is handled in pipeline.py
        when combining with composite_score.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        v = self._feature_vector(candidate)
        normed = np.clip((v - self._mins) / (self._maxs - self._mins), 0.0, 1.0)

        # epoch_distance: higher is LESS consistent, so invert
        epoch_score = 1.0 - normed[0]
        phase_score = normed[1]          # tls_phase_agreement: higher = more consistent
        # depth_ratio: values near 1.0 are consistent, extreme values less so
        depth_normed = normed[2]
        depth_score  = 1.0 - abs(depth_normed - 0.5) * 2.0   # peaks at 0.5 (ratio ~1.0)
        depth_score  = float(np.clip(depth_score, 0.0, 1.0))

        return float(np.mean([epoch_score, phase_score, depth_score]))
