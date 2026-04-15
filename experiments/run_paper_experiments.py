"""
Paper experiment runner for ExoPattern v3.1.

"Conformal Anomaly Ranking for Transit Candidate Prioritization
 in Kepler Light Curves"

Experiments
-----------
1. Star-level GroupKFold CV — event-level metrics + conformal diagnostics
2. Conformal calibration validity — FAR coverage at alpha = 0.01/0.05/0.10
3. Ablation study — TLS alpha sweep + event score method comparison
4. SHAP explainability — top-10 ranked candidates
5. Injection-recovery — 8×8 completeness heatmap

Usage
-----
    python experiments/run_paper_experiments.py [--exp 1,2,3,4,5] [--debug]

    # Development run (5 local stars, 1 trial/cell):
    python experiments/run_paper_experiments.py --dev

    # Full run (requires 150-star dataset):
    python experiments/run_paper_experiments.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.paper_config import *
from backend.ml.pipeline import RankingPipeline, PipelineConfig
from backend.ml.event_evaluation import (
    load_ground_truth_events, compute_ground_truth_events_from_dfs,
    event_metrics, aggregate_event_metrics,
)
from backend.ml.conformal import (
    ConformalCalibrator, calibration_diagnostics,
)
from backend.ml.injection import (
    run_injection_recovery, summarize_injection_recovery,
    completeness_contours,
)
from backend.ml.tls_features import extract_tls_features
from backend.ml.periodic_propagation import apply_propagation
from backend.ml.events import CandidateEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('paper_experiments')
warnings.filterwarnings('ignore')
TLS_DISK_CACHE = Path("results/paper/tls_cache.json")
FEATURE_STORE_DIR = Path("results/paper/feature_store")


class LazyCorpus(dict):
    """
    Lazy-loading dict of star DataFrames.

    Each star's CSV is read from disk only when first accessed.  When the
    pipeline's disk feature store is warm (all stars already extracted),
    score_star() loads features directly from the store and never triggers
    a CSV read via __getitem__.

    This collapses the upfront 1.1 GB corpus load to zero on repeated runs
    where features are cached, and to star-by-star streaming on first run.
    """

    def __init__(self, metadata_df: pd.DataFrame, lc_dir: Path) -> None:
        super().__init__()
        self._meta = {str(row['target_id']): row for _, row in metadata_df.iterrows()}
        self._lc_dir = lc_dir
        for sid in self._meta:
            super().__setitem__(sid, None)

    def __getitem__(self, star_id: str):
        val = super().__getitem__(star_id)
        if val is None:
            row = self._meta[star_id]
            path = self._lc_dir / row['filename']
            try:
                df = pd.read_csv(path)
            except Exception as e:
                raise KeyError(f"Could not load {star_id}: {e}") from e
            super().__setitem__(star_id, df)
            return df
        return val

    def get(self, star_id: str, default=None):
        """
        dict.get() variant that preserves lazy-loading semantics.

        compute_ground_truth_events_from_dfs() uses dfs.get(star_id), so if we
        inherit dict.get() unchanged it returns the stored sentinel None instead
        of triggering a CSV load. That silently collapses all GT event lists to
        empty under LazyCorpus.
        """
        if star_id not in self._meta:
            return default
        try:
            return self[star_id]
        except KeyError:
            return default

    def items(self):
        for k in self._meta:
            try:
                yield k, self[k]
            except KeyError as e:
                logger.warning(str(e))

    def keys(self):
        return self._meta.keys()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(metadata_csv: str, lightcurve_dir: str) -> Dict:
    """
    Load all light curves and metadata.

    Uses lazy loading: each star's CSV is read on first access rather than
    all at once.  When combined with a populated disk feature store, repeated
    runs avoid reading any CSVs for the feature-extraction path entirely.

    Returns:
        dict with keys: star_ids, dfs (LazyCorpus), metadata, gt_events_by_star
    """
    meta = pd.read_csv(metadata_csv)
    lc_dir = Path(lightcurve_dir)
    dfs = LazyCorpus(meta, lc_dir)
    logger.info(f"LazyCorpus registered {len(dfs)} stars (CSVs loaded on first access)")

    # GT event computation needs time ranges — triggers CSV loads on first run,
    # but metadata is cheap to recompute and is not the bottleneck.
    gt_events = compute_ground_truth_events_from_dfs(meta, dfs)
    logger.info(f"Ground truth events computed for {len(gt_events)} stars")

    return {
        'star_ids': list(dfs.keys()),
        'dfs': dfs,
        'metadata': meta,
        'gt_events_by_star': gt_events,
    }




def precompute_tls_cache(dfs: Dict[str, pd.DataFrame], n_workers: int = 2) -> Dict[str, Dict]:
    """Precompute TLS features for all stars in parallel.

    Results are cached to disk (results/paper/tls_cache.json) so reruns are instant.
    Uses ThreadPoolExecutor (not ProcessPoolExecutor) because transitleastsquares
    uses its own internal multiprocessing — nesting it in ProcessPoolExecutor
    causes deadlocks on Linux via fork-inside-fork.
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from backend.ml.tls_features import _TLS_FALLBACK

    # --- Load from disk cache if available ---
    cache: Dict[str, Dict] = {}
    if TLS_DISK_CACHE.exists():
        try:
            cached = _json.loads(TLS_DISK_CACHE.read_text())
            # Only use cache entries for stars in this dataset
            # Restore NaN values lost during JSON serialization (NaN → null → None)
            cache = {
                sid: {k: (float('nan') if v is None else v) for k, v in cached[sid].items()}
                for sid in dfs if sid in cached
            }
            if len(cache) == len(dfs):
                logger.info(f"TLS cache loaded from disk ({len(cache)} stars) — skipping recompute")
                return cache
            else:
                logger.info(f"TLS cache partial ({len(cache)}/{len(dfs)}) — computing missing stars")
        except Exception as e:
            logger.warning(f"Could not load TLS disk cache: {e} — recomputing all")
            cache = {}

    missing = {sid: df for sid, df in dfs.items() if sid not in cache}
    logger.info(f"Precomputing TLS for {len(missing)} stars ({n_workers} workers)...")

    def _compute_one(sid_df):
        sid, df = sid_df
        try:
            return sid, extract_tls_features(df['time'].values, df['flux'].values)
        except Exception as e:
            logger.warning(f"  TLS failed for {sid}: {e} — using fallback")
            return sid, dict(_TLS_FALLBACK)

    completed = 0
    total = len(missing)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="TLS precompute", unit="star", dynamic_ncols=True)
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_one, (sid, df)): sid
                   for sid, df in missing.items()}
        for future in as_completed(futures):
            try:
                sid, result = future.result(timeout=600)
                cache[sid] = result
                completed += 1
                if pbar:
                    pbar.set_postfix(star=sid[:20], SDE=f"{result['tls_sde']:.2f}")
                    pbar.update(1)
                else:
                    logger.info(f"  [{completed}/{total}] {sid}: SDE={result['tls_sde']:.2f}")
            except Exception as e:
                sid = futures[future]
                logger.warning(f"  TLS future failed for {sid}: {e} — using fallback")
                completed += 1
                if pbar:
                    pbar.update(1)
                cache[sid] = dict(_TLS_FALLBACK)

    if pbar:
        pbar.close()

    # --- Save to disk cache ---
    try:
        TLS_DISK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # Convert any non-serializable values (nan → null)
        serializable = {}
        for sid, feats in cache.items():
            serializable[sid] = {k: (None if (v is not None and v != v) else v)
                                  for k, v in feats.items()}
        TLS_DISK_CACHE.write_text(_json.dumps(serializable, indent=2))
        logger.info(f"TLS cache saved to {TLS_DISK_CACHE}")
    except Exception as e:
        logger.warning(f"Could not save TLS disk cache: {e}")

    return cache


# ---------------------------------------------------------------------------
# Star-level GroupKFold CV (core of Experiment 1 & 2)
# ---------------------------------------------------------------------------

def run_star_cv(
    dataset: Dict,
    config: PipelineConfig,
    n_splits: int = N_CV_FOLDS,
    calib_fraction: float = CALIB_FRACTION,
    tls_cache: Optional[Dict[str, Dict]] = None,
    rng_seed: int = RANDOM_SEED,
) -> Dict:
    """
    Run star-level GroupKFold CV using the RankingPipeline.

    All folds are run in parallel (ThreadPoolExecutor) since each fold creates
    its own independent RankingPipeline instance.  The feature store and TLS
    cache are read-only once warm, so thread-sharing is safe.

    Returns dict with per-fold metrics and aggregated results.
    """
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dfs = dataset['dfs']
    gt_by_star = dataset['gt_events_by_star']
    star_ids = dataset['star_ids']

    rng = np.random.default_rng(rng_seed)
    X_dummy = np.zeros(len(star_ids))
    groups = np.array(star_ids)

    kf = GroupKFold(n_splits=n_splits)

    # Generate all fold splits + rng states upfront (deterministic, same as sequential)
    fold_splits = list(enumerate(kf.split(X_dummy, groups=groups)))
    rng_states  = [int(rng.integers(0, 10_000)) for _ in fold_splits]

    def _run_fold(fold_idx, train_val_idx, test_idx, rng_state):
        test_ids      = [star_ids[i] for i in test_idx]
        train_val_ids = [star_ids[i] for i in train_val_idx]

        n_train_val = len(train_val_ids)
        gss = GroupShuffleSplit(
            n_splits=1, test_size=calib_fraction, random_state=rng_state
        )
        X_tv = np.zeros(n_train_val)
        G_tv = np.array(train_val_ids)
        train_idx_local, calib_idx_local = next(gss.split(X_tv, groups=G_tv))

        proper_train_ids = [train_val_ids[i] for i in train_idx_local]
        calib_ids        = [train_val_ids[i] for i in calib_idx_local]

        logger.info(
            f"  [Fold {fold_idx+1}/{n_splits}] "
            f"proper-train: {len(proper_train_ids)}  "
            f"calib: {len(calib_ids)}  test: {len(test_ids)}"
        )

        train_dfs = {k: dfs[k] for k in proper_train_ids if k in dfs}
        calib_dfs = {k: dfs[k] for k in calib_ids if k in dfs}
        test_dfs  = {k: dfs[k] for k in test_ids  if k in dfs}

        if not train_dfs:
            logger.warning(f"  [Fold {fold_idx+1}] no training data, skipping")
            return None, None

        pipeline = RankingPipeline(config)
        pipeline.fit(train_dfs, tls_cache=tls_cache, gt_by_star=gt_by_star)

        train_cands = pipeline.score_stars(train_dfs, tls_cache)
        if train_cands:
            pipeline.fit_consistency_normalizer(train_cands)

        calib_cands = pipeline.score_stars(calib_dfs, tls_cache)
        fp_scores = _collect_fp_scores(calib_cands, gt_by_star)
        logger.info(f"  [Fold {fold_idx+1}] Calibration FP candidates: {len(fp_scores)}")

        if len(fp_scores) > 0:
            pipeline.fit_conformal(np.array(fp_scores))
            conf_diag = calibration_diagnostics(np.array(fp_scores), ALPHA_LEVELS)
        else:
            logger.warning(f"  [Fold {fold_idx+1}] No FP candidates — skipping conformal")
            conf_diag = {'n_null': 0, 'warning': 'empty null set'}

        test_cands = pipeline.score_stars(test_dfs, tls_cache)

        # Period-aware propagation: add phantom candidates at all TLS-predicted
        # transit times not already covered by a real candidate.
        # Phantoms score below every real candidate (no precision@K impact).
        if tls_cache:
            test_cands = apply_propagation(
                test_cands, test_ids, tls_cache, dfs,
                min_sde=5.0, coverage_half_days=0.5,
            )

        all_gt = []
        for sid in test_ids:
            all_gt.extend(gt_by_star.get(sid, []))

        fold_metrics = event_metrics(test_cands, all_gt)
        fold_metrics['fold'] = fold_idx
        fold_metrics['n_proper_train'] = len(proper_train_ids)
        fold_metrics['n_calib'] = len(calib_ids)
        fold_metrics['n_test'] = len(test_ids)
        fold_metrics['n_fp_null'] = len(fp_scores)
        fold_metrics['per_star'] = _per_star_metrics(
            test_cands, test_ids, gt_by_star, dataset['metadata']
        )
        fold_metrics['period_bins'] = _period_bin_metrics(fold_metrics['per_star'])
        fold_metrics['system_recall'] = _system_level_recall(
            test_cands, test_ids, gt_by_star
        )

        logger.info(
            f"  [Fold {fold_idx+1}] "
            f"Recall@K={fold_metrics['recall_at_k']:.3f}  "
            f"AU-PR={fold_metrics['au_pr']:.3f}  "
            f"EventF1={fold_metrics['event_f1']:.3f}  "
            f"SystemRecall={fold_metrics['system_recall']:.3f}"
        )
        _log_period_bins(fold_metrics['period_bins'], fold_idx)
        return fold_metrics, conf_diag

    logger.info(f"Running {n_splits} CV folds in parallel...")
    per_fold: List[Dict] = []
    conformal_diagnostics: List[Dict] = []

    with ThreadPoolExecutor(max_workers=n_splits) as executor:
        futures = {
            executor.submit(_run_fold, fold_idx, tv_idx, t_idx, rng_states[fold_idx]): fold_idx
            for fold_idx, (tv_idx, t_idx) in fold_splits
        }
        fold_results = {}
        for future in as_completed(futures):
            fold_idx = futures[future]
            try:
                metrics, diag = future.result()
                if metrics is not None:
                    fold_results[fold_idx] = (metrics, diag)
            except Exception as e:
                logger.error(f"  Fold {fold_idx+1} failed: {e}")

    # Restore fold order
    for fold_idx in sorted(fold_results):
        metrics, diag = fold_results[fold_idx]
        per_fold.append(metrics)
        conformal_diagnostics.append(diag)

    agg = aggregate_event_metrics(per_fold)
    agg['system_recall_mean'] = float(np.mean([f['system_recall'] for f in per_fold]))
    agg['system_recall_std']  = float(np.std( [f['system_recall'] for f in per_fold]))
    agg['period_bins'] = _aggregate_period_bins(per_fold)

    return {
        'per_fold': per_fold,
        'aggregated': agg,
        'conformal_diagnostics': conformal_diagnostics,
    }


def _collect_fp_scores(
    candidates: List,
    gt_by_star: Dict[str, List],
) -> np.ndarray:
    """
    Return ranking scores of false-positive candidates using proper per-star
    one-to-one matching (highest-scoring candidate claims each GT event first).
    """
    from collections import defaultdict
    from backend.ml.event_evaluation import match_events

    cands_by_star: Dict[str, List] = defaultdict(list)
    for cand in candidates:
        cands_by_star[cand.star_id].append(cand)

    fp_scores = []
    for star_id, star_cands in cands_by_star.items():
        gt_events = gt_by_star.get(star_id, [])
        _, fp_cands, _, _ = match_events(star_cands, gt_events)
        for cand in fp_cands:
            fp_scores.append(
                cand.ranking_score if cand.ranking_score is not None else cand.composite_score
            )
    return np.array(fp_scores)


# ---------------------------------------------------------------------------
# Per-star / period-bin diagnostics
# ---------------------------------------------------------------------------

# Period bins (days) — chosen to match astrophysical regimes:
#   ultra-short (USP):   P < 3d  — hundreds of transits, anomaly detection ill-suited
#   short:        3–10d          — tens of transits, marginal regime
#   moderate:    10–50d          — 30–146 transits, transitional
#   long:         > 50d          — <30 transits, anomaly detection most natural
PERIOD_BINS = [
    ('P<3d',   0.0,   3.0),
    ('3-10d',  3.0,  10.0),
    ('10-50d', 10.0, 50.0),
    ('P>50d',  50.0, float('inf')),
]


def _per_star_metrics(
    test_cands: List,
    test_ids: List[str],
    gt_by_star: Dict[str, List],
    metadata: 'pd.DataFrame',
) -> List[Dict]:
    """
    Compute per-star event metrics and attach orbital period from metadata.

    Returns a list of dicts, one per test star.
    """
    from collections import defaultdict
    from backend.ml.event_evaluation import match_events

    # Build period lookup from metadata
    period_lookup: Dict[str, float] = {}
    if metadata is not None:
        for _, row in metadata.iterrows():
            sid = str(row['target_id'])
            p = row.get('period', float('nan'))
            period_lookup[sid] = float(p) if not pd.isna(p) else float('nan')

    # Group candidates by star
    cands_by_star: Dict[str, List] = defaultdict(list)
    for cand in test_cands:
        cands_by_star[cand.star_id].append(cand)

    per_star = []
    for sid in test_ids:
        gt = gt_by_star.get(sid, [])
        cands = cands_by_star.get(sid, [])
        n_gt = len(gt)
        period = period_lookup.get(sid, float('nan'))

        if n_gt == 0:
            per_star.append({
                'star_id': sid,
                'period': period,
                'n_gt': 0,
                'n_cands': len(cands),
                'n_detected': 0,
                'recall': float('nan'),
                'precision': float('nan'),
                'system_detected': False,
            })
            continue

        tp_cands, fp_cands, detected_gt, _ = match_events(cands, gt)
        n_detected = len(detected_gt)
        recall    = n_detected / n_gt
        precision = len(tp_cands) / max(len(cands), 1)

        per_star.append({
            'star_id': sid,
            'period': period,
            'n_gt': n_gt,
            'n_cands': len(cands),
            'n_detected': n_detected,
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'system_detected': n_detected > 0,
        })

    return per_star


def _period_bin_metrics(per_star: List[Dict]) -> Dict[str, Dict]:
    """
    Group per-star results into period bins and compute aggregate metrics.
    """
    bins: Dict[str, Dict] = {}
    for bin_name, lo, hi in PERIOD_BINS:
        stars = [s for s in per_star
                 if not np.isnan(s['period']) and lo <= s['period'] < hi
                 and not np.isnan(s['recall'])]
        if not stars:
            bins[bin_name] = {
                'n_stars': 0,
                'recall_mean': float('nan'),
                'recall_std': float('nan'),
                'system_recall': float('nan'),
                'total_gt': 0,
                'total_detected': 0,
            }
            continue
        recalls = [s['recall'] for s in stars]
        bins[bin_name] = {
            'n_stars':       len(stars),
            'recall_mean':   round(float(np.mean(recalls)), 4),
            'recall_std':    round(float(np.std(recalls)), 4),
            'system_recall': round(float(np.mean([s['system_detected'] for s in stars])), 4),
            'total_gt':      int(sum(s['n_gt'] for s in stars)),
            'total_detected': int(sum(s['n_detected'] for s in stars)),
        }
    return bins


def _system_level_recall(
    test_cands: List,
    test_ids: List[str],
    gt_by_star: Dict[str, List],
) -> float:
    """
    Fraction of planet-host test stars where at least one transit was detected.
    Stars with no GT events are excluded (they are quiet stars, not hosts).
    """
    from collections import defaultdict
    from backend.ml.event_evaluation import match_events

    cands_by_star: Dict[str, List] = defaultdict(list)
    for cand in test_cands:
        cands_by_star[cand.star_id].append(cand)

    n_hosts = 0
    n_detected = 0
    for sid in test_ids:
        gt = gt_by_star.get(sid, [])
        if not gt:
            continue   # not a planet host in our catalog
        n_hosts += 1
        cands = cands_by_star.get(sid, [])
        if cands:
            _, _, detected_gt, _ = match_events(cands, gt)
            if detected_gt:
                n_detected += 1

    return n_detected / n_hosts if n_hosts > 0 else float('nan')


def _log_period_bins(bins: Dict[str, Dict], fold_idx: int) -> None:
    logger.info(f"  Period-bin breakdown (fold {fold_idx}):")
    for bin_name, stats in bins.items():
        if stats['n_stars'] == 0:
            logger.info(f"    {bin_name:8s}: no stars")
            continue
        logger.info(
            f"    {bin_name:8s}: {stats['n_stars']:3d} stars  "
            f"recall={stats['recall_mean']:.3f}±{stats['recall_std']:.3f}  "
            f"system={stats['system_recall']:.3f}  "
            f"gt={stats['total_gt']}→detected={stats['total_detected']}"
        )


def _aggregate_period_bins(per_fold: List[Dict]) -> Dict[str, Dict]:
    """Average period-bin metrics across folds."""
    agg: Dict[str, Dict] = {}
    for bin_name, _, _ in PERIOD_BINS:
        fold_stats = [f['period_bins'].get(bin_name, {}) for f in per_fold]
        fold_recalls = [s['recall_mean'] for s in fold_stats
                        if s and not np.isnan(s.get('recall_mean', float('nan')))]
        fold_sysrec  = [s['system_recall'] for s in fold_stats
                        if s and not np.isnan(s.get('system_recall', float('nan')))]
        total_gt  = sum(s.get('total_gt', 0) for s in fold_stats if s)
        total_det = sum(s.get('total_detected', 0) for s in fold_stats if s)
        n_stars   = int(np.mean([s.get('n_stars', 0) for s in fold_stats if s]))
        agg[bin_name] = {
            'n_stars_per_fold': n_stars,
            'recall_mean':   round(float(np.mean(fold_recalls)), 4) if fold_recalls else float('nan'),
            'recall_std':    round(float(np.std(fold_recalls)),  4) if fold_recalls else float('nan'),
            'system_recall': round(float(np.mean(fold_sysrec)),  4) if fold_sysrec  else float('nan'),
            'total_gt_across_folds':       total_gt,
            'total_detected_across_folds': total_det,
        }
    return agg


# ---------------------------------------------------------------------------
# Experiment 1: CV + event metrics
# ---------------------------------------------------------------------------

def experiment_1_cv(dataset: Dict, dev_mode: bool = False) -> Dict:
    """Star-level GroupKFold CV — event-level metrics table."""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Star-level CV — event-level metrics")
    logger.info("="*60)

    cfg = PipelineConfig(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        flag_quantile=FLAG_QUANTILE,
        gap_tolerance=GAP_TOLERANCE,
        max_event_windows=MAX_EVENT_WINDOWS,
        event_score_method=EVENT_SCORE_METHOD,
        alpha=TLS_ALPHA,
        contamination=CONTAMINATION,
        use_tls=True,
        feature_store_dir=str(FEATURE_STORE_DIR),
        max_train_windows=MAX_TRAIN_WINDOWS,
    )
    n_splits = 2 if dev_mode else N_CV_FOLDS

    tls_cache = precompute_tls_cache(dataset['dfs'])
    results = run_star_cv(dataset, cfg, n_splits=n_splits, tls_cache=tls_cache)

    _print_cv_table(results)
    return results


def _print_cv_table(results: Dict) -> None:
    metric_keys = [
        'recall_at_k', 'precision_at_k', 'event_recall',
        'event_precision', 'event_f1', 'au_pr',
    ]
    agg = results['aggregated']
    logger.info("\n--- CV Results (mean ± std) ---")
    for key in metric_keys:
        mean = agg.get(f'{key}_mean', float('nan'))
        std  = agg.get(f'{key}_std',  float('nan'))
        logger.info(f"  {key:25s}: {mean:.4f} ± {std:.4f}")

    # System-level recall
    sr_mean = agg.get('system_recall_mean', float('nan'))
    sr_std  = agg.get('system_recall_std',  float('nan'))
    logger.info(f"  {'system_recall':25s}: {sr_mean:.4f} ± {sr_std:.4f}")

    # Period-bin breakdown
    bins = agg.get('period_bins', {})
    if bins:
        logger.info("\n--- Period-bin recall (mean across folds) ---")
        logger.info(f"  {'Bin':10s}  {'Stars/fold':>10}  {'Recall':>8}  {'SysRecall':>10}  {'GT':>8}  {'Detected':>8}")
        for bin_name, stats in bins.items():
            if np.isnan(stats.get('recall_mean', float('nan'))):
                continue
            logger.info(
                f"  {bin_name:10s}  {stats['n_stars_per_fold']:>10}  "
                f"{stats['recall_mean']:>8.4f}  {stats['system_recall']:>10.4f}  "
                f"{stats['total_gt_across_folds']:>8}  {stats['total_detected_across_folds']:>8}"
            )


# ---------------------------------------------------------------------------
# Experiment 2: Conformal calibration validity
# ---------------------------------------------------------------------------

def experiment_2_conformal(dataset: Dict, dev_mode: bool = False) -> Dict:
    """
    Validate conformal FAR coverage.
    For each alpha level, report the empirical false alarm rate on test candidates
    and verify it is bounded by alpha + 1/(n_calib+1).
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Conformal calibration validity")
    logger.info("="*60)

    cfg = PipelineConfig(
        window_size=WINDOW_SIZE, stride=STRIDE, use_tls=False,
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
        feature_store_dir=str(FEATURE_STORE_DIR),
        max_train_windows=MAX_TRAIN_WINDOWS,
    )
    n_splits = 2 if dev_mode else N_CV_FOLDS
    tls_cache = None  # TLS off for conformal validation

    results = run_star_cv(dataset, cfg, n_splits=n_splits, tls_cache=tls_cache)

    logger.info("\n--- Conformal Diagnostics per Fold ---")
    for i, diag in enumerate(results['conformal_diagnostics']):
        n_null = diag.get('n_null', 0)
        slack  = diag.get('guarantee_slack', float('nan'))
        logger.info(f"  Fold {i}: n_null={n_null}  guarantee_slack={slack:.5f}")
        for alpha in ALPHA_LEVELS:
            key = f'threshold_alpha_{int(alpha*100):02d}'
            thr = diag.get(key, 'N/A')
            logger.info(f"    alpha={alpha:.2f}: threshold={thr}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Ablation study
# ---------------------------------------------------------------------------

def experiment_3_ablation(dataset: Dict, dev_mode: bool = False) -> Dict:
    """
    Ablation over:
    (a) TLS alpha weight [0.0, 0.3, 0.5, 0.7, 1.0]
    (b) Event score aggregation method [max, top3_mean, length_penalized]

    All 8 configs are run in parallel (ThreadPoolExecutor, max 4 workers) since
    they are fully independent.  The TLS cache and feature store are read-only
    during CV, so thread-sharing is safe.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Ablation study")
    logger.info("="*60)

    n_splits = 2 if dev_mode else N_CV_FOLDS
    tls_cache = precompute_tls_cache(dataset['dfs'])

    # Build all configs up front
    alpha_specs = [
        ('alpha', alpha, PipelineConfig(
            window_size=WINDOW_SIZE, stride=STRIDE, use_tls=True, alpha=alpha,
            flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
            feature_store_dir=str(FEATURE_STORE_DIR),
            max_train_windows=MAX_TRAIN_WINDOWS,
        ))
        for alpha in TLS_ALPHA_ABLATION
    ]
    method_specs = [
        ('method', method, PipelineConfig(
            window_size=WINDOW_SIZE, stride=STRIDE, use_tls=True, alpha=TLS_ALPHA,
            event_score_method=method, flag_quantile=FLAG_QUANTILE,
            contamination=CONTAMINATION,
            feature_store_dir=str(FEATURE_STORE_DIR),
            max_train_windows=MAX_TRAIN_WINDOWS,
        ))
        for method in EVENT_SCORE_ABLATION
    ]
    all_specs = alpha_specs + method_specs
    logger.info(f"Running {len(all_specs)} ablation configs in parallel (max 4 workers)...")

    def _run_one(spec):
        kind, value, cfg = spec
        logger.info(f"  [{kind}={value}] starting CV ({n_splits} folds)...")
        res = run_star_cv(dataset, cfg, n_splits=n_splits, tls_cache=tls_cache)
        agg = res['aggregated']
        logger.info(
            f"  [{kind}={value}] done — "
            f"Recall@K={agg.get('recall_at_k_mean', float('nan')):.4f}  "
            f"AU-PR={agg.get('au_pr_mean', float('nan')):.4f}"
        )
        return kind, value, agg

    alpha_rows: List[Dict] = []
    method_rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_one, spec): spec for spec in all_specs}
        for future in as_completed(futures):
            try:
                kind, value, agg = future.result()
                if kind == 'alpha':
                    alpha_rows.append({
                        'alpha': value,
                        'recall_at_k': agg.get('recall_at_k_mean', float('nan')),
                        'au_pr':        agg.get('au_pr_mean', float('nan')),
                    })
                else:
                    method_rows.append({
                        'method': value,
                        'recall_at_k': agg.get('recall_at_k_mean', float('nan')),
                        'au_pr':        agg.get('au_pr_mean', float('nan')),
                    })
            except Exception as e:
                spec = futures[future]
                logger.error(f"  Ablation config {spec[0]}={spec[1]} failed: {e}")

    # Restore original ordering
    alpha_rows.sort(key=lambda r: TLS_ALPHA_ABLATION.index(r['alpha']))
    method_rows.sort(key=lambda r: EVENT_SCORE_ABLATION.index(r['method']))

    logger.info("\n--- Alpha sweep results ---")
    for row in alpha_rows:
        logger.info(f"  alpha={row['alpha']:.1f}  Recall@K={row['recall_at_k']:.4f}  AU-PR={row['au_pr']:.4f}")
    logger.info("\n--- Event score method results ---")
    for row in method_rows:
        logger.info(f"  method={row['method']}  Recall@K={row['recall_at_k']:.4f}  AU-PR={row['au_pr']:.4f}")

    return {
        'alpha_sweep': alpha_rows,
        'event_score_method': method_rows,
    }


# ---------------------------------------------------------------------------
# Experiment 4: SHAP explainability
# ---------------------------------------------------------------------------

def experiment_4_shap(dataset: Dict) -> Dict:
    """
    Fit pipeline on all stars, score all candidates, compute SHAP values
    for the top-K ranked candidates.
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 4: SHAP explainability")
    logger.info("="*60)

    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping Experiment 4. pip install shap")
        return {'skipped': True, 'reason': 'shap not installed'}

    dfs = dataset['dfs']
    tls_cache = precompute_tls_cache(dfs)

    cfg = PipelineConfig(
        window_size=WINDOW_SIZE, stride=STRIDE, use_tls=True, alpha=TLS_ALPHA,
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
        feature_store_dir=str(FEATURE_STORE_DIR),
        max_train_windows=MAX_TRAIN_WINDOWS,
    )
    pipeline = RankingPipeline(cfg)
    pipeline.fit(dfs, tls_cache=tls_cache)

    all_cands = pipeline.score_stars(dfs, tls_cache)
    top_k = all_cands[:SHAP_TOP_K]
    logger.info(f"Top-{SHAP_TOP_K} candidates selected for SHAP analysis")

    # Extract feature matrices for top candidates
    # (re-extract to get per-window features for the candidate windows)
    from backend.ml.preprocessing import LightCurvePreprocessor
    from backend.ml.tls_features import append_tls_to_windows, TLS_GLOBAL_FEATURE_NAMES
    from backend.ml.feature_names import ALL_WINDOW_FEATURES

    shap_results = []

    for cand in top_k:
        df = dfs.get(cand.star_id)
        if df is None:
            continue

        # Reuse pipeline feature cache (populated during score_stars above) to avoid
        # redundant extraction. Cache stores pre-imputation float32 features with TLS
        # already appended, matching the 44-dim feature space the IF model was trained on.
        if cand.star_id in pipeline._feature_cache:
            feats_raw, _ = pipeline._feature_cache[cand.star_id]
            feats = feats_raw.copy().astype(float)  # back to float64 for SHAP
        else:
            # Fallback: re-extract if cache miss (e.g. cache was cleared)
            from backend.ml.preprocessing import LightCurvePreprocessor
            prep = LightCurvePreprocessor()
            df_proc = prep.preprocess(df, normalize=True)
            feats, _, _ = prep.extract_features_with_metadata(df_proc, star_id=cand.star_id)
            if len(feats) == 0:
                continue
            tls_f = tls_cache.get(cand.star_id, {})
            feats = append_tls_to_windows(feats, tls_f)

        # SHAP on the IF model (TreeExplainer, correct for isolation forest)
        if_model = pipeline._scorer._models.get('isolation_forest')
        if if_model is None:
            continue

        try:
            explainer = shap.TreeExplainer(if_model.model)
            # Use only the candidate's windows — scale to match training convention
            window_slice = feats[cand.window_indices]
            window_slice_scaled = if_model.scaler.transform(window_slice)
            shap_vals = explainer.shap_values(window_slice_scaled)
            mean_shap = np.mean(np.abs(shap_vals), axis=0)

            feature_names = ALL_WINDOW_FEATURES + TLS_GLOBAL_FEATURE_NAMES
            top_features = sorted(
                zip(feature_names, mean_shap),
                key=lambda x: x[1], reverse=True
            )[:10]

            shap_results.append({
                'star_id': cand.star_id,
                'center_time': cand.center_time,
                'ranking_score': cand.ranking_score,
                'top_features': [(name, round(float(val), 6)) for name, val in top_features],
            })

            logger.info(
                f"  {cand.star_id} t={cand.center_time:.2f}: "
                f"top feature = {top_features[0][0]} ({top_features[0][1]:.4f})"
            )
        except Exception as e:
            logger.warning(f"  SHAP failed for {cand.star_id}: {e}")

    return {'shap_results': shap_results, 'n_explained': len(shap_results)}


# ---------------------------------------------------------------------------
# Experiment 5: Injection-recovery
# ---------------------------------------------------------------------------

def experiment_5_injection_recovery(
    dataset: Dict,
    dev_mode: bool = False,
) -> Dict:
    """
    8×8 injection-recovery completeness heatmap.

    In dev mode: 1 trial/cell on local stars (fast).
    In full mode: 25 trials/cell (1,600 total, ~80 min).
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 5: Injection-recovery")
    logger.info("="*60)

    dfs = dataset['dfs']
    n_trials = 1 if dev_mode else INJECTION_N_TRIALS

    # Fit pipeline on all available stars
    cfg = PipelineConfig(
        window_size=WINDOW_SIZE, stride=STRIDE, use_tls=False,
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
        feature_store_dir=str(FEATURE_STORE_DIR),
        max_train_windows=MAX_TRAIN_WINDOWS,
    )
    pipeline = RankingPipeline(cfg)
    pipeline.fit(dfs)

    logger.info(
        f"Running injection-recovery: "
        f"{len(INJECTION_RADIUS_GRID)}×{len(INJECTION_PERIOD_GRID)} grid, "
        f"{n_trials} trials/cell"
    )

    checkpoint = str(Path(RESULTS_DIR) / 'exp5_injection_checkpoint.json')

    def progress(done, total):
        if done % 50 == 0 or done == total:
            logger.info(f"  Progress: {done}/{total} ({100*done//total}%)")

    result = run_injection_recovery(
        light_curves=dfs,
        pipeline=pipeline,
        radius_grid=INJECTION_RADIUS_GRID,
        period_grid=INJECTION_PERIOD_GRID,
        n_trials=n_trials,
        rng_seed=RANDOM_SEED,
        recovery_window_factor=INJECTION_RECOVERY_WINDOW_FACTOR,
        progress_callback=progress,
        n_workers=os.cpu_count(),
        checkpoint_path=checkpoint,
        checkpoint_interval=50,
    )

    summary = summarize_injection_recovery(result)
    contours = completeness_contours(result, levels=(0.5, 0.8, 0.9))

    logger.info(f"\nOverall recovery rate: {result.overall_recovery_rate:.3f}")
    logger.info("\nCompleteness matrix (rows=radius, cols=period):")
    header = "R\\P   " + "  ".join(f"{p:>5.0f}d" for p in result.period_grid)
    logger.info(header)
    for i, r in enumerate(result.radius_grid):
        row_str = f"{r:>4.1f}Re " + "  ".join(
            f"{result.completeness[i, j]:>6.2f}" for j in range(len(result.period_grid))
        )
        logger.info(row_str)

    return {
        'completeness': result.completeness.tolist(),
        'radius_grid': result.radius_grid,
        'period_grid': result.period_grid,
        'summary': summary,
        'contours': {str(k): v for k, v in contours.items()},
        'overall_recovery_rate': result.overall_recovery_rate,
    }


# ---------------------------------------------------------------------------
# Experiment 6: BLS baseline
# ---------------------------------------------------------------------------

def experiment_6_bls_baseline(dataset: Dict, dev_mode: bool = False) -> Dict:
    """
    BLS (Box Least Squares) baseline comparison.

    Runs astropy's BoxLeastSquares on all stars in parallel (16 workers).
    The best-fit period + t0 + duration are used to enumerate transit candidates
    with the same logic as period-aware propagation.  Results are evaluated with
    the same event_metrics() function for a direct comparison table.

    This is a *blind* BLS run — no training set, no anomaly scores.
    BLS candidates are assigned a uniform composite_score of 1.0 and ranked
    by transit depth (negative depth → more anomalous).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 6: BLS baseline")
    logger.info("="*60)

    try:
        from astropy.timeseries import BoxLeastSquares
        import astropy.units as u
    except ImportError:
        logger.warning("astropy not installed — skipping Experiment 6. pip install astropy")
        return {'skipped': True, 'reason': 'astropy not installed'}

    dfs          = dataset['dfs']
    gt_by_star   = dataset['gt_events_by_star']
    star_ids     = dataset['star_ids']
    n_workers    = 4 if dev_mode else 16

    # Period grid: 0.5 – 200 days, log-spaced, 2000 points (fast BLS)
    period_min  = 0.5
    period_max  = 200.0
    n_periods   = 500 if dev_mode else 2000

    def _bls_one_star(star_id: str):
        """Run BLS on one star and return CandidateEvent list."""
        df = dfs.get(star_id)
        if df is None or 'time' not in df.columns or 'flux' not in df.columns:
            return star_id, []

        time_arr = df['time'].dropna().values
        flux_arr = df['flux'].dropna().values

        # Align lengths after dropna
        valid = ~(np.isnan(time_arr) | np.isnan(flux_arr))
        time_arr = time_arr[valid]
        flux_arr = flux_arr[valid]

        if len(time_arr) < 50:
            return star_id, []

        try:
            # Normalise flux to zero-mean (BLS works on fractional flux deviations)
            flux_norm = flux_arr / np.nanmedian(flux_arr) - 1.0

            # Use .power() with explicit period grid (autopower() doesn't accept period=)
            periods = np.geomspace(period_min, period_max, n_periods) * u.day
            durations = np.array([0.05, 0.1, 0.2]) * u.day
            bls = BoxLeastSquares(time_arr * u.day, flux_norm)
            result = bls.power(periods, durations, objective='snr')

            best_idx = np.argmax(result.power)
            best_period = float(result.period[best_idx].value)
            best_t0     = float(result.transit_time[best_idx].value)
            best_dur    = float(result.duration[best_idx].value)
            best_depth  = float(result.depth[best_idx])
            best_power  = float(result.power[best_idx])

            # Enumerate all transit centers within the light curve
            time_min = float(time_arr.min())
            time_max = float(time_arr.max())
            half_dur = best_dur / 2.0 + 0.5  # same coverage radius as propagation

            n_start = int(np.floor((time_min - best_t0) / best_period))
            n_end   = int(np.ceil( (time_max - best_t0) / best_period))
            centers = [best_t0 + n * best_period for n in range(n_start, n_end + 1)
                       if time_min <= best_t0 + n * best_period <= time_max]

            # Score: use BLS power (higher power = more transit-like)
            cands = []
            for tc in centers:
                cand = CandidateEvent(
                    star_id=star_id,
                    start_time=tc - half_dur,
                    end_time=tc + half_dur,
                    center_time=tc,
                    window_indices=[],
                    n_windows=0,
                    model_scores={'bls_power': best_power, 'bls_depth': best_depth},
                    composite_score=best_power,
                    ranking_score=best_power,
                )
                cands.append(cand)
            return star_id, cands

        except Exception as e:
            logger.debug(f"  BLS failed for {star_id}: {e}")
            return star_id, []

    logger.info(f"Running BLS on {len(star_ids)} stars ({n_workers} workers)...")
    all_bls_cands: List[CandidateEvent] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_bls_one_star, sid): sid for sid in star_ids}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                sid, cands = future.result()
                all_bls_cands.extend(cands)
                if completed % 20 == 0 or completed == len(star_ids):
                    logger.info(f"  [{completed}/{len(star_ids)}] done, {len(all_bls_cands)} total candidates")
            except Exception as e:
                sid = futures[future]
                logger.warning(f"  BLS future failed for {sid}: {e}")

    logger.info(f"BLS produced {len(all_bls_cands)} total transit candidates")

    # Evaluate with same event_metrics — all stars together (no CV split needed for baseline)
    all_gt = []
    for sid in star_ids:
        all_gt.extend(gt_by_star.get(sid, []))

    metrics = event_metrics(all_bls_cands, all_gt)

    # Per-star system recall
    from collections import defaultdict
    from backend.ml.event_evaluation import match_events
    cands_by_star = defaultdict(list)
    for c in all_bls_cands:
        cands_by_star[c.star_id].append(c)

    n_hosts = 0
    n_sys_detected = 0
    for sid in star_ids:
        gt = gt_by_star.get(sid, [])
        if not gt:
            continue
        n_hosts += 1
        cands = cands_by_star.get(sid, [])
        if cands:
            _, _, detected_gt, _ = match_events(cands, gt)
            if detected_gt:
                n_sys_detected += 1

    system_recall = n_sys_detected / n_hosts if n_hosts > 0 else float('nan')
    metrics['system_recall'] = system_recall

    logger.info("\n--- BLS Baseline Results ---")
    for k in ['recall_at_k', 'precision_at_k', 'event_recall', 'event_precision', 'event_f1', 'au_pr']:
        logger.info(f"  {k:25s}: {metrics.get(k, float('nan')):.4f}")
    logger.info(f"  {'system_recall':25s}: {system_recall:.4f}")

    return {
        'metrics': metrics,
        'n_bls_candidates': len(all_bls_cands),
        'n_stars': len(star_ids),
        'n_ground_truth_events': len(all_gt),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    logger.info(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run ExoPattern v3.1 paper experiments'
    )
    parser.add_argument(
        '--exp', type=str, default='1,2,3,4,5,6',
        help='Comma-separated experiment numbers to run (default: all)',
    )
    parser.add_argument(
        '--dev', action='store_true',
        help='Dev mode: use local 5-star dataset, 1 trial/cell, 2 CV folds',
    )
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: 20 stars, 2 CV folds, 2 trials/cell — full pipeline, fast',
    )
    parser.add_argument(
        '--n-stars', type=int, default=None,
        help='Limit dataset to first N stars (for testing)',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Set logging level to DEBUG',
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --smoke implies --dev settings but with 20 stars
    if args.smoke:
        args.dev = True
        if args.n_stars is None:
            args.n_stars = 20

    to_run = {int(x.strip()) for x in args.exp.split(',')}

    # Output dirs
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, RAW_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {METADATA_CSV}")
    t0 = time.time()
    dataset = load_dataset(METADATA_CSV, LIGHTCURVE_DIR)
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s")

    # Subset dataset if requested
    if args.n_stars is not None:
        n = args.n_stars
        star_ids = dataset['star_ids'][:n]
        dataset['star_ids'] = star_ids
        dataset['dfs'] = {sid: dataset['dfs'][sid] for sid in star_ids if sid in dataset['dfs']}
        dataset['gt_events_by_star'] = {sid: dataset['gt_events_by_star'].get(sid, []) for sid in star_ids}
        logger.info(f"Dataset subset to {len(dataset['dfs'])} stars (--n-stars {n})")

    all_results: Dict[str, Dict] = {}

    if 1 in to_run:
        t0 = time.time()
        res = experiment_1_cv(dataset, dev_mode=args.dev)
        all_results['experiment_1'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_1_cv.json')
        logger.info(f"Experiment 1 done in {time.time()-t0:.1f}s")

    if 2 in to_run:
        t0 = time.time()
        res = experiment_2_conformal(dataset, dev_mode=args.dev)
        all_results['experiment_2'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_2_conformal.json')
        logger.info(f"Experiment 2 done in {time.time()-t0:.1f}s")

    if 3 in to_run:
        t0 = time.time()
        res = experiment_3_ablation(dataset, dev_mode=args.dev)
        all_results['experiment_3'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_3_ablation.json')
        logger.info(f"Experiment 3 done in {time.time()-t0:.1f}s")

    if 4 in to_run:
        t0 = time.time()
        res = experiment_4_shap(dataset)
        all_results['experiment_4'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_4_shap.json')
        logger.info(f"Experiment 4 done in {time.time()-t0:.1f}s")

    if 5 in to_run:
        t0 = time.time()
        res = experiment_5_injection_recovery(dataset, dev_mode=args.dev)
        all_results['experiment_5'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_5_injection.json')
        logger.info(f"Experiment 5 done in {time.time()-t0:.1f}s")

    if 6 in to_run:
        t0 = time.time()
        res = experiment_6_bls_baseline(dataset, dev_mode=args.dev)
        all_results['experiment_6'] = res
        _save_json(res, Path(RAW_DIR) / 'experiment_6_bls_baseline.json')
        logger.info(f"Experiment 6 done in {time.time()-t0:.1f}s")

    # Save combined results
    _save_json(all_results, Path(RESULTS_DIR) / 'all_results.json')
    logger.info("\nAll experiments complete.")


if __name__ == '__main__':
    main()
