"""
End-to-end RankingPipeline for ExoPattern v3.1.

Implements the full anomaly ranking pipeline described in BLUEPRINT.md:

1. Feature extraction (38 window features + 6 TLS global = 44 features)
2. Multi-view anomaly scoring (IF, LOF, OCSVM)
3. Composite score: min-max normalize → mean
4. Candidate event generation (composite threshold at 97.5th percentile)
5. TLS event-consistency features on candidates
6. Conformal calibration (p-values from calibration-set false positives)
7. Final ranking: alpha * composite_score + (1-alpha) * consistency_score

The pipeline is designed for star-level GroupKFold CV (see evaluation.py).
For inference on new stars, call RankingPipeline.score_star() after fitting.

Score convention: higher = more anomalous, everywhere. No exceptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml.preprocessing import LightCurvePreprocessor
from backend.ml.multi_view import MultiViewScorer, CORE_VIEWS
from backend.ml.events import CandidateEvent, generate_candidates
from backend.ml.tls_features import (
    extract_tls_features, append_tls_to_windows,
    attach_consistency_features, ConsistencyScoreNormalizer,
)
from backend.ml.conformal import ConformalCalibrator, apply_conformal_to_candidates
from backend.ml.event_evaluation import load_ground_truth_events

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Hyperparameters for RankingPipeline (all blueprint-specified defaults)."""

    # Feature extraction
    window_size: int = 50
    stride: int = 12
    sigma_clip: float = 5.0
    use_tls: bool = True

    # Multi-view scoring
    view_names: Tuple[str, ...] = CORE_VIEWS   # ('isolation_forest', 'lof', 'ocsvm')

    # Candidate generation
    flag_quantile: float = 0.975
    gap_tolerance: int = 2
    max_event_windows: int = 20
    event_score_method: str = 'top3_mean'       # top3_mean | max | length_penalized

    # TLS consistency ranking
    alpha: float = 0.7   # weight on composite_score vs consistency_score

    # Conformal calibration
    use_conformal: bool = True

    # Model hyperparameters (passed to sklearn models)
    contamination: float = 0.1
    n_estimators: int = 100           # IF n_estimators
    random_state: int = 42


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class RankingPipeline:
    """
    Fit-transform pipeline for transit candidate ranking.

    Typical usage (star-level CV):

        pipeline = RankingPipeline(config)

        # Fit on proper-training stars
        pipeline.fit(train_star_dfs)

        # Collect conformal null from calibration stars
        calib_candidates = pipeline.score_stars(calib_star_dfs)
        fp_scores = [c.composite_score for c in calib_candidates
                     if c is not a GT transit]
        pipeline.fit_conformal(np.array(fp_scores))

        # Evaluate on test stars
        test_candidates = pipeline.score_stars(test_star_dfs)

    For injection-recovery, use as_inference_fn():

        pipeline_fn = pipeline.as_inference_fn()
        trial = inject_and_recover(df, params, pipeline_fn)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._preprocessor = LightCurvePreprocessor()
        self._scorer: Optional[MultiViewScorer] = None
        self._conformal: Optional[ConformalCalibrator] = None
        self._consistency_normalizer: Optional[ConsistencyScoreNormalizer] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        tls_cache: Optional[Dict[str, Dict]] = None,
    ) -> 'RankingPipeline':
        """
        Fit models and normalizers on proper-training stars.

        Args:
            train_dfs: {star_id -> DataFrame with time/flux columns}
            tls_cache: optional precomputed TLS features {star_id -> tls_dict}.
                       If None, TLS is computed here (slow).

        Returns:
            self
        """
        from backend.ml.model_registry import get_model

        cfg = self.config

        # 1. Extract features for all training stars
        logger.info(f"Extracting features for {len(train_dfs)} training stars...")
        all_features: List[np.ndarray] = []

        for star_id, df in train_dfs.items():
            feats = self._extract_star_features(df, star_id, tls_cache)
            if feats is not None and len(feats) > 0:
                all_features.append(feats)

        if not all_features:
            raise ValueError("No features extracted from training stars.")

        X_train = np.vstack(all_features)
        logger.info(f"Training feature matrix: {X_train.shape}")

        # 2. Build and fit models (each model has different accepted kwargs)
        _MODEL_KWARGS: Dict[str, Dict] = {
            'isolation_forest': {
                'contamination': cfg.contamination,
                'n_estimators': cfg.n_estimators,
                'random_state': cfg.random_state,
            },
            'lof': {
                'contamination': cfg.contamination,
            },
            'ocsvm': {
                'contamination': cfg.contamination,
            },
        }
        models: Dict[str, object] = {}
        for name in cfg.view_names:
            kwargs = _MODEL_KWARGS.get(name, {'contamination': cfg.contamination})
            m = get_model(name, **kwargs)
            m.fit(X_train)
            models[name] = m
            logger.info(f"  Fitted {name}")

        # 3. Fit multi-view scorer (normalizer)
        self._scorer = MultiViewScorer()
        raw_scores = MultiViewScorer.score_windows(models, X_train, cfg.view_names)
        self._scorer.fit_normalizer(raw_scores)
        composite = self._scorer.composite(raw_scores)

        # 4. Fit composite threshold
        self._scorer.fit_threshold(composite, quantile=cfg.flag_quantile)
        logger.info(
            f"Composite threshold (p={cfg.flag_quantile:.3f}): "
            f"{self._scorer.threshold:.4f}"
        )

        # 5. Store fitted models inside scorer
        self._scorer._models = models

        self._fitted = True
        logger.info("RankingPipeline.fit() complete.")
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_star(
        self,
        star_id: str,
        df: pd.DataFrame,
        tls_cache: Optional[Dict[str, Dict]] = None,
        gt_events: Optional[List] = None,
    ) -> List[CandidateEvent]:
        """
        Score a single star and return candidate events.

        Args:
            star_id: star identifier
            df: light curve DataFrame (time, flux)
            tls_cache: optional precomputed TLS features
            gt_events: optional ground truth events (used only for
                       conformal null collection in calibration mode)

        Returns:
            list of CandidateEvent, sorted by ranking_score descending
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score_star().")

        cfg = self.config

        # 1. Extract features + metadata
        feats, _, meta = self._preprocessor.extract_features_with_metadata(
            df, star_id=star_id,
            window_size=cfg.window_size,
        )

        if len(feats) == 0:
            logger.warning(f"{star_id}: no windows extracted")
            return []

        # 2. Append TLS global features (if enabled)
        tls_feats: Dict = {}
        if cfg.use_tls:
            tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
            feats = append_tls_to_windows(feats, tls_feats)

        # 3. Score windows
        raw_scores = MultiViewScorer.score_windows(
            self._scorer._models, feats, cfg.view_names
        )
        composite = self._scorer.composite(raw_scores)

        # 4. Generate candidates
        candidates = generate_candidates(
            composite_scores=composite,
            metadata=meta,
            threshold=self._scorer.threshold,
            per_model_scores=raw_scores,
            gap_tolerance=cfg.gap_tolerance,
            max_event_windows=cfg.max_event_windows,
            event_score_method=cfg.event_score_method,
        )

        if not candidates:
            return []

        # 5. Attach TLS event-consistency features
        if cfg.use_tls and tls_feats:
            attach_consistency_features(candidates, {star_id: tls_feats})

            # Normalize consistency and compute ranking_score
            if self._consistency_normalizer is not None:
                for cand in candidates:
                    c_score = self._consistency_normalizer.score(cand)
                    # Invert: high consistency = less novel = lower anomaly interest
                    novelty_bonus = 1.0 - c_score
                    cand.ranking_score = float(
                        cfg.alpha * cand.composite_score
                        + (1.0 - cfg.alpha) * novelty_bonus
                    )
            else:
                # No normalizer yet — use composite score only
                for cand in candidates:
                    cand.ranking_score = cand.composite_score
        else:
            for cand in candidates:
                cand.ranking_score = cand.composite_score

        # 6. Apply conformal calibration
        if cfg.use_conformal and self._conformal is not None:
            apply_conformal_to_candidates(candidates, self._conformal)

        # 7. Sort by ranking_score descending
        candidates.sort(key=lambda c: c.ranking_score, reverse=True)
        return candidates

    def score_stars(
        self,
        star_dfs: Dict[str, pd.DataFrame],
        tls_cache: Optional[Dict[str, Dict]] = None,
    ) -> List[CandidateEvent]:
        """
        Score multiple stars and return all candidates (sorted by ranking_score).
        """
        all_candidates: List[CandidateEvent] = []
        for star_id, df in star_dfs.items():
            cands = self.score_star(star_id, df, tls_cache)
            all_candidates.extend(cands)
        all_candidates.sort(key=lambda c: c.ranking_score, reverse=True)
        return all_candidates

    # ------------------------------------------------------------------
    # Conformal calibration
    # ------------------------------------------------------------------

    def fit_conformal(self, null_scores: np.ndarray) -> 'RankingPipeline':
        """
        Fit the conformal calibrator on false-positive candidate scores from
        calibration stars.

        Must be called AFTER fit() and BEFORE score_star() on test stars.

        Args:
            null_scores: composite scores of FP candidates (calibration stars)
        """
        self._conformal = ConformalCalibrator().fit(null_scores)
        logger.info(
            f"Conformal calibrator fitted: {len(null_scores)} null candidates, "
            f"guarantee_slack={self._conformal.guarantee_slack:.5f}"
        )
        return self

    # ------------------------------------------------------------------
    # Consistency normalizer
    # ------------------------------------------------------------------

    def fit_consistency_normalizer(
        self, train_candidates: List[CandidateEvent]
    ) -> 'RankingPipeline':
        """
        Fit the TLS consistency score normalizer on proper-training candidates.
        Must be called AFTER candidate generation on training stars.
        """
        self._consistency_normalizer = ConsistencyScoreNormalizer().fit(
            train_candidates
        )
        logger.info(
            f"ConsistencyScoreNormalizer fitted on {len(train_candidates)} "
            "training candidates."
        )
        return self

    # ------------------------------------------------------------------
    # Injection-recovery interface
    # ------------------------------------------------------------------

    def as_inference_fn(
        self,
        tls_cache: Optional[Dict[str, Dict]] = None,
    ):
        """
        Return a callable pipeline_fn(df) -> List[CandidateEvent] for use
        with injection_recovery.inject_and_recover().

        The returned function assigns a temporary star_id and runs the full
        scoring pipeline (without conformal p-values, which require a fitted
        calibrator).
        """
        def _pipeline_fn(df: pd.DataFrame) -> List[CandidateEvent]:
            return self.score_star(
                star_id='__injection__',
                df=df,
                tls_cache=tls_cache,
            )
        return _pipeline_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_star_features(
        self,
        df: pd.DataFrame,
        star_id: str,
        tls_cache: Optional[Dict[str, Dict]],
    ) -> Optional[np.ndarray]:
        """Extract window features (+ TLS if enabled) for one star."""
        cfg = self.config
        try:
            feats, _, _ = self._preprocessor.extract_features_with_metadata(
                df, star_id=star_id, window_size=cfg.window_size
            )
            if len(feats) == 0:
                logger.warning(f"{star_id}: no features extracted")
                return None

            if cfg.use_tls:
                tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
                feats = append_tls_to_windows(feats, tls_feats)

            return feats
        except Exception as e:
            logger.warning(f"{star_id}: feature extraction failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Helper: TLS cache lookup / compute
# ---------------------------------------------------------------------------

def _get_or_compute_tls(
    star_id: str,
    df: pd.DataFrame,
    tls_cache: Optional[Dict[str, Dict]],
) -> Dict:
    """Return TLS features from cache or compute them."""
    if tls_cache is not None and star_id in tls_cache:
        return tls_cache[star_id]
    return extract_tls_features(df['time'].values, df['flux'].values)

