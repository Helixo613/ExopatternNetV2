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

    # Disk-backed feature store (shared across folds and ablations)
    # Set to a directory path to enable; None disables the store.
    feature_store_dir: Optional[str] = None

    # Maximum training windows passed to model fitting and normaliser.
    # If the fold's feature matrix exceeds this, a random subsample is taken.
    # None = no limit (use all windows).  Bounds peak RAM for large datasets.
    # Recommended: 200_000 for 32 GB RAM; None for local dev datasets.
    max_train_windows: Optional[int] = None


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
        self._impute_medians: Optional[np.ndarray] = None  # from training
        self._fitted = False
        # Per-star feature cache: {star_id -> (features, metadata)}
        # Avoids redundant extraction across fit/score_stars/calibration calls
        self._feature_cache: Dict[str, Tuple[np.ndarray, List]] = {}
        # Disk-backed feature store (optional)
        self._feature_store = None
        self._config_hash: Optional[str] = None
        if self.config.feature_store_dir:
            from backend.ml.feature_store import StarFeatureStore
            self._feature_store = StarFeatureStore(self.config.feature_store_dir)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        tls_cache: Optional[Dict[str, Dict]] = None,
        gt_by_star: Optional[Dict[str, List]] = None,
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

        # 1. Extract features for all training stars, tracking total count to enable
        #    proportional per-star subsampling BEFORE vstack (avoids peak-RAM spike).
        logger.info(f"Extracting features for {len(train_dfs)} training stars...")
        all_features: List[np.ndarray] = []
        total_windows = 0
        total_excluded = 0

        for star_id, df in train_dfs.items():
            extracted = self._extract_star_features(df, star_id, tls_cache)
            if extracted is None:
                continue

            feats, meta = extracted
            if gt_by_star is not None:
                keep_mask = _build_training_keep_mask(meta, gt_by_star.get(star_id, []))
                excluded = int((~keep_mask).sum())
                if excluded > 0:
                    feats = feats[keep_mask]
                    total_excluded += excluded
                    logger.info(
                        f"{star_id}: excluded {excluded} GT-overlapping training windows "
                        f"({len(feats)} kept)"
                    )

            if feats is not None and len(feats) > 0:
                all_features.append(feats)
                total_windows += len(feats)

        if not all_features:
            raise ValueError("No features extracted from training stars.")

        logger.info(
            f"Collected {total_windows} training windows from {len(all_features)} stars"
            + (f" after excluding {total_excluded} GT-overlapping windows" if total_excluded else "")
        )

        mtw = cfg.max_train_windows
        if mtw is not None and total_windows > mtw:
            # Proportional pre-subsampling: each star contributes ~(n_star/total)*mtw rows.
            # Subsetting here avoids materialising the full (total_windows × n_features)
            # matrix. Deleting all_features before vstack lets GC release the list
            # references, keeping peak allocation close to 1× the subset size.
            rng = np.random.default_rng(cfg.random_state)
            sub_arrays: List[np.ndarray] = []
            for f in all_features:
                n_take = max(1, round(len(f) * mtw / total_windows))
                idx = rng.choice(len(f), size=min(n_take, len(f)), replace=False)
                sub_arrays.append(f[idx])
            del all_features          # release list before allocating subset matrix
            X_train = np.vstack(sub_arrays)
            del sub_arrays
            logger.info(
                f"fit(): pre-subsampled {total_windows} → {len(X_train)} windows "
                f"(max_train_windows={mtw}, proportional per star)"
            )
        else:
            X_train = np.vstack(all_features)
            del all_features          # release list once matrix is owned by X_train

        logger.info(f"Training matrix for models: {X_train.shape}")


        # Store training column medians for consistent NaN imputation at inference
        col_medians = np.nanmedian(X_train, axis=0)
        self._impute_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

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
        threshold_override: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[CandidateEvent]:
        """
        Score a single star and return candidate events.

        Args:
            star_id: star identifier
            df: light curve DataFrame (time, flux)
            tls_cache: optional precomputed TLS features
            gt_events: optional ground truth events (used only for
                       conformal null collection in calibration mode)
            threshold_override: if given, use this threshold instead of the
                       fitted one (useful for calibration-mode FP collection)

        Returns:
            list of CandidateEvent, sorted by ranking_score descending
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score_star().")

        cfg = self.config

        # 1. Check feature cache before expensive extraction
        tls_feats: Dict = {}
        if use_cache and star_id in self._feature_cache:
            feats_raw, meta = self._feature_cache[star_id]
            feats = _impute_nan(feats_raw.copy(), self._impute_medians)
            # Still need TLS dict for consistency features later
            if cfg.use_tls:
                tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
        else:
            # Full extraction path: check disk store before re-extracting
            _store_hit = False
            if use_cache and self._feature_store is not None:
                _stored = self._feature_store.load(star_id, self._get_config_hash())
                if _stored is not None:
                    feats_f32, meta = _stored
                    self._feature_cache[star_id] = (feats_f32, meta)
                    feats = _impute_nan(feats_f32.copy(), self._impute_medians)
                    if cfg.use_tls:
                        tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
                    _store_hit = True

            if not _store_hit:
                df_proc = self._preprocessor.preprocess(df, normalize=True, sigma=cfg.sigma_clip)
                feats, _, meta = self._preprocessor.extract_features_with_metadata(
                    df_proc, star_id=star_id,
                    window_size=cfg.window_size,
                    stride=cfg.stride,
                )

                if len(feats) == 0:
                    logger.warning(f"{star_id}: no windows extracted")
                    return []

                if cfg.use_tls:
                    tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
                    feats = append_tls_to_windows(feats, tls_feats)

                # Bypassed for injection trials (use_cache=False) to prevent
                # cache-key collisions across concurrent ThreadPoolExecutor workers.
                if use_cache:
                    feats_f32 = feats.astype(np.float32)
                    self._feature_cache[star_id] = (feats_f32, meta)
                    if self._feature_store is not None:
                        self._feature_store.save(star_id, self._get_config_hash(), feats_f32, meta)
                    feats = _impute_nan(feats_f32.copy(), self._impute_medians)
                else:
                    feats = _impute_nan(feats.astype(np.float32), self._impute_medians)

        if len(feats) == 0:
            logger.warning(f"{star_id}: no windows extracted")
            return []

        # 3. Score windows
        raw_scores = MultiViewScorer.score_windows(
            self._scorer._models, feats, cfg.view_names
        )
        composite = self._scorer.composite(raw_scores)

        # 4. Generate candidates
        eff_threshold = (
            threshold_override if threshold_override is not None
            else self._scorer.threshold
        )
        candidates = generate_candidates(
            composite_scores=composite,
            metadata=meta,
            threshold=eff_threshold,
            per_model_scores=raw_scores,
            gap_tolerance=cfg.gap_tolerance,
            max_event_windows=cfg.max_event_windows,
            event_score_method=cfg.event_score_method,
        )

        if not candidates:
            return []

        # 5. Attach TLS event-consistency features
        if cfg.use_tls and tls_feats:
            # Use raw (median-normalized) df so depth units match TLS depth (fractional)
            flux_dips = _estimate_candidate_flux_dips(candidates, df)
            attach_consistency_features(
                candidates,
                {star_id: tls_feats},
                flux_dip_by_candidate=flux_dips,
            )

            # Normalize consistency and compute ranking_score
            if self._consistency_normalizer is not None:
                for cand in candidates:
                    c_score = self._consistency_normalizer.score(cand)
                    # Blueprint ranking: alpha * composite + (1 - alpha) * consistency
                    # High consistency = strong transit alignment = higher rank
                    cand.ranking_score = float(
                        cfg.alpha * cand.composite_score
                        + (1.0 - cfg.alpha) * c_score
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
        threshold_override: Optional[float] = None,
    ) -> List[CandidateEvent]:
        """
        Score multiple stars and return all candidates (sorted by ranking_score).
        """
        all_candidates: List[CandidateEvent] = []
        for star_id, df in star_dfs.items():
            cands = self.score_star(star_id, df, tls_cache,
                                    threshold_override=threshold_override)
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
            # use_cache=False: injection trials run concurrently in a ThreadPoolExecutor.
            # All trials share star_id='__injection__', which would cause different
            # injected light curves to collide on the same cache key, corrupting
            # completeness estimates. Disabling the cache makes scoring stateless
            # and thread-safe at the cost of re-extraction per trial.
            return self.score_star(
                star_id='__injection__',
                df=df,
                tls_cache=tls_cache,
                use_cache=False,
            )
        return _pipeline_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_hash(self) -> str:
        """Return an 8-char hash encoding the feature extraction config."""
        if self._config_hash is None:
            from backend.ml.feature_store import StarFeatureStore
            cfg = self.config
            self._config_hash = StarFeatureStore.make_config_hash(
                cfg.window_size, cfg.stride, cfg.sigma_clip, cfg.use_tls
            )
        return self._config_hash

    def _extract_star_features(
        self,
        df: pd.DataFrame,
        star_id: str,
        tls_cache: Optional[Dict[str, Dict]],
    ) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """Extract window features (+ TLS if enabled) for one star.

        Check order: in-memory cache → disk feature store → full extraction.
        Populates both caches after extraction.
        """
        # 1. In-memory cache hit
        if star_id in self._feature_cache:
            feats_raw, meta = self._feature_cache[star_id]
            return _impute_nan(feats_raw.copy()), meta

        # 2. Disk feature store hit
        if self._feature_store is not None:
            stored = self._feature_store.load(star_id, self._get_config_hash())
            if stored is not None:
                feats_f32, meta = stored
                self._feature_cache[star_id] = (feats_f32, meta)
                logger.debug(f"{star_id}: features loaded from disk store")
                return _impute_nan(feats_f32.copy()), meta

        # 3. Full extraction
        cfg = self.config
        try:
            df_proc = self._preprocessor.preprocess(df, normalize=True, sigma=cfg.sigma_clip)
            feats, _, meta = self._preprocessor.extract_features_with_metadata(
                df_proc, star_id=star_id, window_size=cfg.window_size,
                stride=cfg.stride,
            )
            if len(feats) == 0:
                logger.warning(f"{star_id}: no features extracted")
                return None

            if cfg.use_tls:
                tls_feats = _get_or_compute_tls(star_id, df, tls_cache)
                feats = append_tls_to_windows(feats, tls_feats)

            feats_f32 = feats.astype(np.float32)
            self._feature_cache[star_id] = (feats_f32, meta)
            # Persist to disk store for cross-fold / cross-experiment reuse
            if self._feature_store is not None:
                self._feature_store.save(star_id, self._get_config_hash(), feats_f32, meta)
            return _impute_nan(feats_f32.copy()), meta
        except Exception as e:
            logger.warning(f"{star_id}: feature extraction failed: {e}")
            return None

    def preload_to_store(
        self,
        star_dfs: Dict[str, pd.DataFrame],
        tls_cache: Optional[Dict[str, Dict]] = None,
    ) -> int:
        """
        Precompute and persist features for a set of stars.

        Useful before injection-recovery: call this with the injection corpus
        so the disk store is warm before threaded trials start.  Trials with
        use_cache=False still re-extract for each injected LC (unavoidable —
        the flux has changed), but training-star features are ready for the
        fold-fit path that runs before injection.

        Returns the number of stars written to the store.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before preload_to_store().")
        saved = 0
        for star_id, df in star_dfs.items():
            if star_id not in self._feature_cache:
                if self._feature_store is not None and \
                        self._feature_store.has(star_id, self._get_config_hash()):
                    continue  # already on disk
            result = self._extract_star_features(df, star_id, tls_cache)
            if result is not None:
                saved += 1
        logger.info(f"preload_to_store: {saved} stars written/verified in feature store")
        return saved

    def clear_cache(self) -> None:
        """Free the per-star feature cache (call between experiments)."""
        self._feature_cache.clear()


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


def _impute_nan(X: np.ndarray, fill_values: Optional[np.ndarray] = None) -> np.ndarray:
    """Replace NaN values with fill_values (training medians) or column medians."""
    if not np.any(np.isnan(X)):
        return X
    X = X.copy()
    if fill_values is not None:
        medians = fill_values
    else:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return X


def _build_training_keep_mask(metadata: List[Dict], gt_events: List) -> np.ndarray:
    """
    Return a boolean mask that keeps only windows not overlapping known GT transits.

    GT intervals are already buffered in event_evaluation, so any direct interval
    overlap is sufficient here.
    """
    if not metadata:
        return np.zeros(0, dtype=bool)
    if not gt_events:
        return np.ones(len(metadata), dtype=bool)

    starts = np.array([m['start_time'] for m in metadata], dtype=float)
    ends = np.array([m['end_time'] for m in metadata], dtype=float)
    keep = np.ones(len(metadata), dtype=bool)

    for gt in gt_events:
        overlap = (starts <= gt.end_time) & (ends >= gt.start_time)
        keep[overlap] = False

    return keep


def _estimate_candidate_flux_dips(
    candidates: List[CandidateEvent],
    df: pd.DataFrame,
) -> Dict[int, float]:
    """
    Estimate observed fractional dip depth for each candidate.

    Depth = (local_median - local_min) / local_median, which matches the
    fractional units returned by TLS (depth ≈ 1 - min_normalized_flux).
    Must be called with raw (not z-scored) flux so units are compatible.
    """
    flux_dips: Dict[int, float] = {}
    time = df['time'].values
    flux = df['flux'].values

    for cand in candidates:
        mask = (time >= cand.start_time) & (time <= cand.end_time)
        if not np.any(mask):
            continue
        segment = flux[mask]
        if len(segment) == 0:
            continue
        local_median = float(np.median(segment))
        local_min = float(np.min(segment))
        if abs(local_median) > 1e-9:
            flux_dips[id(cand)] = max(0.0, (local_median - local_min) / abs(local_median))
        else:
            flux_dips[id(cand)] = 0.0

    return flux_dips
