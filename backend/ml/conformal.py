"""
Conformal calibration for anomaly candidate ranking.

Converts raw composite anomaly scores into calibrated p-values with a
finite-sample false alarm rate guarantee (Vovk et al., 2005).

Guarantee: for any alpha in [0, 1],
  P(p_value <= alpha | candidate is null) <= alpha + 1/(n_calib + 1)

The null distribution is built from FALSE POSITIVE candidates on calibration
stars — candidates that do NOT overlap any known transit ground truth.
Calibration stars are disjoint from both training and test stars (see D3 in
BLUEPRINT.md).

Paper claim scope: calibrated candidate prioritization WITHIN transit-host
light curves.  Not a general survey-level false alarm claim.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Split conformal anomaly detector.

    Fit on null (false positive) candidate scores from calibration stars,
    then predict calibrated p-values for test candidates.
    """

    def __init__(self) -> None:
        self._null_scores: Optional[np.ndarray] = None
        self._n_calib: int = 0
        self._fitted: bool = False

    def fit(self, null_scores: np.ndarray) -> 'ConformalCalibrator':
        """
        Store the calibration null distribution.

        Args:
            null_scores: composite scores of false-positive candidates from
                         calibration stars (higher = more anomalous).
                         Must be from stars DISJOINT from both train and test.
        """
        null_scores = np.asarray(null_scores, dtype=float)
        if len(null_scores) == 0:
            logger.warning(
                "ConformalCalibrator.fit() called with empty null_scores. "
                "p-values will be uninformative (all = 1.0)."
            )
        self._null_scores = np.sort(null_scores)[::-1]   # descending for fast search
        self._n_calib = len(null_scores)
        self._fitted = True
        logger.info(f"ConformalCalibrator fitted on {self._n_calib} null candidates.")
        return self

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Compute conformal p-values for test candidate scores.

        p_value(s_test) = (|{i : s_i >= s_test}| + 1) / (n_calib + 1)

        A small p-value means the test candidate scores higher than most of
        the null distribution — strong evidence of anomalousness.

        Args:
            test_scores: composite scores of test candidates

        Returns:
            p-value array, same shape as test_scores, in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        test_scores = np.asarray(test_scores, dtype=float)
        n = self._n_calib

        if n == 0:
            # No null data — all p-values are 1.0 (no information)
            return np.ones_like(test_scores)

        p_values = np.empty_like(test_scores)
        for i, s in enumerate(test_scores):
            # Number of null scores >= s_test
            n_above = int(np.searchsorted(-self._null_scores, -s, side='right'))
            p_values[i] = (n_above + 1) / (n + 1)

        return np.clip(p_values, 0.0, 1.0)

    def threshold_at_alpha(self, alpha: float) -> float:
        """
        Return the composite score threshold that corresponds to significance
        level alpha.  Candidates with score >= threshold have p_value <= alpha.

        Args:
            alpha: desired false alarm rate (e.g. 0.05)

        Returns:
            score threshold
        """
        if not self._fitted or self._n_calib == 0:
            raise RuntimeError("No null data — threshold undefined.")
        # p_value <= alpha  <=>  score >= quantile(1 - alpha) of null distribution
        quantile = np.quantile(self._null_scores, 1.0 - alpha)
        return float(quantile)

    @property
    def n_calib(self) -> int:
        return self._n_calib

    @property
    def guarantee_slack(self) -> float:
        """Upper bound on excess FAR: 1 / (n_calib + 1)."""
        return 1.0 / (self._n_calib + 1)


class CrossConformalCalibrator:
    """
    K-fold cross-conformal calibration.

    Aggregates p-values across multiple CV folds where each fold has its own
    calibration null set.  Produces p-values for all test candidates across
    all folds without reusing any calibration data for test inference.
    """

    def __init__(self) -> None:
        self._fold_calibrators: List[ConformalCalibrator] = []

    def add_fold(
        self,
        null_scores: np.ndarray,
        test_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Add one fold: fit a calibrator on null_scores, predict p-values for
        test_scores, store the calibrator, and return the p-values.

        Args:
            null_scores: FP candidate scores from this fold's calibration stars
            test_scores: candidate scores from this fold's test stars

        Returns:
            p-values for test candidates in this fold
        """
        cal = ConformalCalibrator().fit(null_scores)
        self._fold_calibrators.append(cal)
        return cal.predict(test_scores)

    @property
    def n_folds(self) -> int:
        return len(self._fold_calibrators)

    def summary(self) -> Dict:
        """Return per-fold calibration stats."""
        return {
            f'fold_{i}': {
                'n_calib': cal.n_calib,
                'guarantee_slack': round(cal.guarantee_slack, 5),
            }
            for i, cal in enumerate(self._fold_calibrators)
        }


def apply_conformal_to_candidates(
    candidates,          # List[CandidateEvent]
    calibrator: ConformalCalibrator,
) -> None:
    """
    In-place: set p_value on each CandidateEvent using a fitted calibrator.

    Args:
        candidates: list of CandidateEvent objects
        calibrator: fitted ConformalCalibrator
    """
    if not candidates:
        return
    scores = np.array([c.composite_score for c in candidates])
    p_values = calibrator.predict(scores)
    for cand, pv in zip(candidates, p_values):
        cand.p_value = float(pv)


def calibration_diagnostics(
    null_scores: np.ndarray,
    alpha_levels: Tuple[float, ...] = (0.01, 0.05, 0.10),
) -> Dict:
    """
    Report diagnostics for a calibration null distribution.

    Args:
        null_scores: FP candidate scores used as the null
        alpha_levels: significance levels to report thresholds for

    Returns:
        dict with stats and per-alpha thresholds
    """
    if len(null_scores) == 0:
        return {'n_null': 0, 'warning': 'empty null set'}

    cal = ConformalCalibrator().fit(null_scores)
    diag = {
        'n_null': len(null_scores),
        'null_mean': float(np.mean(null_scores)),
        'null_std': float(np.std(null_scores)),
        'null_p95': float(np.percentile(null_scores, 95)),
        'null_p99': float(np.percentile(null_scores, 99)),
        'guarantee_slack': round(cal.guarantee_slack, 5),
    }
    for alpha in alpha_levels:
        diag[f'threshold_alpha_{int(alpha*100):02d}'] = round(
            cal.threshold_at_alpha(alpha), 4
        )
    return diag
