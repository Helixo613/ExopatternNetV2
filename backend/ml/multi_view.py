"""
Multi-view anomaly score aggregation.

Enforces the universal score-direction convention (higher = more anomalous)
by negating all raw model scores once at extraction time.  Nothing downstream
ever negates scores again.

Models included in the core CV ranking:
  - IsolationForest  (classical)
  - LOF              (classical)
  - OneClassSVM      (classical)

Deep models (CNN-AE, LSTM-AE) are excluded because their pre-trained artifacts
were trained on the same 150-star corpus used for CV evaluation, causing
data leakage.  They may be added as auxiliary views with an explicit caveat
after the core results are established.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Names of the three classical views used in core CV
CORE_VIEWS = ('isolation_forest', 'lof', 'ocsvm')


class MultiViewScorer:
    """
    Aggregates per-window scores from multiple anomaly detection models into a
    single composite score.

    Usage (inside one CV fold):
        scorer = MultiViewScorer()
        raw = scorer.score_windows(models, X_train)   # fit normaliser
        scorer.fit_normalizer(raw)
        train_composite = scorer.composite(raw)

        raw_test = scorer.score_windows(models, X_test)  # inference only
        test_composite = scorer.composite(raw_test)       # uses frozen normaliser
    """

    def __init__(self) -> None:
        # min/max per view, fitted on proper-training data
        self._mins: Optional[Dict[str, float]] = None
        self._maxs: Optional[Dict[str, float]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    @staticmethod
    def score_windows(
        models: Dict[str, object],
        X: np.ndarray,
        view_names: Tuple[str, ...] = CORE_VIEWS,
    ) -> Dict[str, np.ndarray]:
        """
        Call score_samples() on each model and apply the sign convention.

        All wrappers in baselines.py expose score_samples().  sklearn's
        convention is more-negative = more-anomalous, so we negate once here.
        This is the ONLY place negation happens.

        Args:
            models: dict mapping view name -> fitted model with score_samples()
            X: feature array (n_windows, n_features)
            view_names: which views to extract

        Returns:
            dict mapping view name -> score array (n_windows,),
            higher = more anomalous
        """
        scores: Dict[str, np.ndarray] = {}
        for name in view_names:
            if name not in models:
                logger.warning(f"View '{name}' not in models dict — skipping")
                continue
            model = models[name]
            if not hasattr(model, 'score_samples'):
                logger.warning(f"Model '{name}' has no score_samples() — skipping")
                continue
            try:
                raw = model.score_samples(X)
                scores[name] = -np.asarray(raw, dtype=float)   # negate once
            except Exception as e:
                logger.error(f"score_samples() failed for '{name}': {e}")
        return scores

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def fit_normalizer(self, scores: Dict[str, np.ndarray]) -> None:
        """
        Fit min-max normaliser on proper-training scores.
        Must be called before composite() on calibration or test data.

        Args:
            scores: output of score_windows() on training data
        """
        self._mins = {}
        self._maxs = {}
        for name, arr in scores.items():
            lo, hi = float(np.min(arr)), float(np.max(arr))
            if hi - lo < 1e-12:
                logger.warning(
                    f"View '{name}' has near-zero range — scores will be 0.5 after normalisation"
                )
                hi = lo + 1.0   # prevent division by zero
            self._mins[name] = lo
            self._maxs[name] = hi
        self._fitted = True

    def _normalize(self, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply frozen min-max to a set of scores."""
        if not self._fitted:
            raise RuntimeError("Call fit_normalizer() on training data first.")
        normed: Dict[str, np.ndarray] = {}
        for name, arr in scores.items():
            if name not in self._mins:
                logger.warning(f"View '{name}' not seen during fit — skipping normalisation")
                normed[name] = arr
                continue
            lo, hi = self._mins[name], self._maxs[name]
            normed[name] = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return normed

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def composite(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Normalise and mean-aggregate per-view scores into one composite score.

        Args:
            scores: output of score_windows() (un-normalised)

        Returns:
            composite score array (n_windows,), higher = more anomalous
        """
        normed = self._normalize(scores)
        if not normed:
            raise ValueError("No valid views to aggregate.")
        stack = np.stack(list(normed.values()), axis=1)   # (n_windows, n_views)
        return stack.mean(axis=1)

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def fit_threshold(
        self,
        train_composite: np.ndarray,
        quantile: float = 0.975,
    ) -> float:
        """
        Compute the flagging threshold from proper-training composite scores.

        Recall-oriented: the top (1 - quantile) fraction of training windows
        are flagged.  Default 97.5th percentile => ~2.5% of windows flagged.

        Args:
            train_composite: composite scores on proper-training windows
            quantile: percentile threshold (0-1)

        Returns:
            threshold value (stored internally and returned)
        """
        self._threshold = float(np.quantile(train_composite, quantile))
        return self._threshold

    @property
    def threshold(self) -> float:
        if not hasattr(self, '_threshold'):
            raise RuntimeError("Call fit_threshold() first.")
        return self._threshold

    def flag_windows(self, composite: np.ndarray) -> np.ndarray:
        """Return boolean mask — True where composite score exceeds threshold."""
        return composite >= self.threshold

    # ------------------------------------------------------------------
    # Persistence helpers (for saving/loading within a CV fold)
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        return {
            'mins': self._mins,
            'maxs': self._maxs,
            'threshold': getattr(self, '_threshold', None),
            'fitted': self._fitted,
        }

    def set_state(self, state: Dict) -> None:
        self._mins = state['mins']
        self._maxs = state['maxs']
        self._fitted = state['fitted']
        if state['threshold'] is not None:
            self._threshold = state['threshold']


# ------------------------------------------------------------------
# Convenience: fit and score in one shot (used in smoke tests)
# ------------------------------------------------------------------

def fit_and_score(
    models: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    quantile: float = 0.975,
    view_names: Tuple[str, ...] = CORE_VIEWS,
) -> Tuple[MultiViewScorer, np.ndarray, np.ndarray, float]:
    """
    Convenience wrapper for a single fold:
      1. Score training windows
      2. Fit normaliser + threshold
      3. Score test windows

    Returns:
        scorer, train_composite, test_composite, threshold
    """
    scorer = MultiViewScorer()
    train_scores = scorer.score_windows(models, X_train, view_names)
    scorer.fit_normalizer(train_scores)
    train_composite = scorer.composite(train_scores)
    threshold = scorer.fit_threshold(train_composite, quantile)

    test_scores = scorer.score_windows(models, X_test, view_names)
    test_composite = scorer.composite(test_scores)

    return scorer, train_composite, test_composite, threshold
