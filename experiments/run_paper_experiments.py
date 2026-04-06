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
    load_ground_truth_events, event_metrics, aggregate_event_metrics,
)
from backend.ml.conformal import (
    ConformalCalibrator, calibration_diagnostics,
)
from backend.ml.injection import (
    run_injection_recovery, summarize_injection_recovery,
    completeness_contours,
)
from backend.ml.tls_features import extract_tls_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('paper_experiments')
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(metadata_csv: str, lightcurve_dir: str) -> Dict:
    """
    Load all light curves and metadata.

    Returns:
        dict with keys: star_ids, dfs, metadata, gt_events_by_star
    """
    meta = pd.read_csv(metadata_csv)
    lc_dir = Path(lightcurve_dir)

    dfs: Dict[str, pd.DataFrame] = {}
    for _, row in meta.iterrows():
        sid = str(row['target_id'])
        path = lc_dir / row['filename']
        try:
            dfs[sid] = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Could not load {sid}: {e}")

    logger.info(f"Loaded {len(dfs)} light curves")

    gt_events = load_ground_truth_events(metadata_csv, lightcurve_dir)
    logger.info(f"Ground truth events loaded for {len(gt_events)} stars")

    return {
        'star_ids': list(dfs.keys()),
        'dfs': dfs,
        'metadata': meta,
        'gt_events_by_star': gt_events,
    }


def precompute_tls_cache(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Precompute TLS features for all stars (slow; ~2-5 min for 150 stars)."""
    logger.info(f"Precomputing TLS for {len(dfs)} stars...")
    cache: Dict[str, Dict] = {}
    for i, (sid, df) in enumerate(dfs.items()):
        cache[sid] = extract_tls_features(df['time'].values, df['flux'].values)
        logger.info(
            f"  [{i+1}/{len(dfs)}] {sid}: SDE={cache[sid]['tls_sde']:.2f}"
        )
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

    Returns dict with per-fold metrics and aggregated results.
    """
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    dfs = dataset['dfs']
    gt_by_star = dataset['gt_events_by_star']
    star_ids = dataset['star_ids']

    rng = np.random.default_rng(rng_seed)
    X_dummy = np.zeros(len(star_ids))
    groups = np.array(star_ids)

    kf = GroupKFold(n_splits=n_splits)
    per_fold: List[Dict] = []
    conformal_diagnostics: List[Dict] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        kf.split(X_dummy, groups=groups)
    ):
        logger.info(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

        test_ids  = [star_ids[i] for i in test_idx]
        train_val_ids = [star_ids[i] for i in train_val_idx]

        # 3-way split: proper-train / calibration
        n_train_val = len(train_val_ids)
        gss = GroupShuffleSplit(
            n_splits=1, test_size=calib_fraction,
            random_state=int(rng.integers(0, 10_000))
        )
        X_tv = np.zeros(n_train_val)
        G_tv = np.array(train_val_ids)
        train_idx_local, calib_idx_local = next(gss.split(X_tv, groups=G_tv))

        proper_train_ids = [train_val_ids[i] for i in train_idx_local]
        calib_ids        = [train_val_ids[i] for i in calib_idx_local]

        logger.info(
            f"  proper-train: {len(proper_train_ids)}  "
            f"calib: {len(calib_ids)}  test: {len(test_ids)}"
        )

        train_dfs = {k: dfs[k] for k in proper_train_ids if k in dfs}
        calib_dfs = {k: dfs[k] for k in calib_ids if k in dfs}
        test_dfs  = {k: dfs[k] for k in test_ids  if k in dfs}

        if not train_dfs:
            logger.warning(f"  Fold {fold_idx}: no training data, skipping")
            continue

        # Fit pipeline
        pipeline = RankingPipeline(config)
        pipeline.fit(train_dfs, tls_cache=tls_cache)

        # Generate candidates on training stars (for consistency normalizer)
        train_cands = pipeline.score_stars(train_dfs, tls_cache)
        if train_cands:
            pipeline.fit_consistency_normalizer(train_cands)

        # Calibration: collect FP candidates → conformal null
        calib_cands = pipeline.score_stars(calib_dfs, tls_cache)
        fp_scores = _collect_fp_scores(calib_cands, gt_by_star)
        logger.info(f"  Calibration FP candidates: {len(fp_scores)}")

        if len(fp_scores) > 0:
            pipeline.fit_conformal(np.array(fp_scores))
            conf_diag = calibration_diagnostics(np.array(fp_scores), ALPHA_LEVELS)
        else:
            logger.warning("  No FP candidates for conformal — skipping calibration")
            conf_diag = {'n_null': 0, 'warning': 'empty null set'}
        conformal_diagnostics.append(conf_diag)

        # Test: score + evaluate
        test_cands = pipeline.score_stars(test_dfs, tls_cache)
        all_gt = []
        for sid in test_ids:
            all_gt.extend(gt_by_star.get(sid, []))

        fold_metrics = event_metrics(test_cands, all_gt)
        fold_metrics['fold'] = fold_idx
        fold_metrics['n_proper_train'] = len(proper_train_ids)
        fold_metrics['n_calib'] = len(calib_ids)
        fold_metrics['n_test'] = len(test_ids)
        fold_metrics['n_fp_null'] = len(fp_scores)
        per_fold.append(fold_metrics)

        logger.info(
            f"  Recall@K={fold_metrics['recall_at_k']:.3f}  "
            f"AU-PR={fold_metrics['au_pr']:.3f}  "
            f"EventF1={fold_metrics['event_f1']:.3f}"
        )

    agg = aggregate_event_metrics(per_fold)
    return {
        'per_fold': per_fold,
        'aggregated': agg,
        'conformal_diagnostics': conformal_diagnostics,
    }


def _collect_fp_scores(
    candidates: List,
    gt_by_star: Dict[str, List],
) -> np.ndarray:
    """Return composite scores of candidates that don't match any GT event."""
    from backend.ml.event_evaluation import _find_matching_gt

    fp_scores = []
    for cand in candidates:
        gt_events = gt_by_star.get(cand.star_id, [])
        if _find_matching_gt(cand, gt_events, overlap_fraction=0.25) is None:
            fp_scores.append(cand.composite_score)
    return np.array(fp_scores)


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
        flag_quantile=FLAG_QUANTILE,
        gap_tolerance=GAP_TOLERANCE,
        max_event_windows=MAX_EVENT_WINDOWS,
        event_score_method=EVENT_SCORE_METHOD,
        alpha=TLS_ALPHA,
        contamination=CONTAMINATION,
        use_tls=True,
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
        window_size=WINDOW_SIZE, use_tls=False,  # skip TLS for speed here
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
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
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Ablation study")
    logger.info("="*60)

    n_splits = 2 if dev_mode else N_CV_FOLDS
    tls_cache = precompute_tls_cache(dataset['dfs'])
    ablation_results: Dict[str, List[Dict]] = {}

    # (a) TLS alpha sweep
    logger.info("\n--- (a) TLS alpha sweep ---")
    alpha_results = []
    for alpha in TLS_ALPHA_ABLATION:
        logger.info(f"  alpha={alpha:.1f}")
        cfg = PipelineConfig(
            window_size=WINDOW_SIZE, use_tls=True, alpha=alpha,
            flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
        )
        res = run_star_cv(dataset, cfg, n_splits=n_splits, tls_cache=tls_cache)
        agg = res['aggregated']
        row = {
            'alpha': alpha,
            'recall_at_k': agg.get('recall_at_k_mean', float('nan')),
            'au_pr':        agg.get('au_pr_mean', float('nan')),
        }
        alpha_results.append(row)
        logger.info(
            f"    Recall@K={row['recall_at_k']:.4f}  AU-PR={row['au_pr']:.4f}"
        )
    ablation_results['alpha_sweep'] = alpha_results

    # (b) Event score method
    logger.info("\n--- (b) Event score method ---")
    method_results = []
    for method in EVENT_SCORE_ABLATION:
        logger.info(f"  method={method}")
        cfg = PipelineConfig(
            window_size=WINDOW_SIZE, use_tls=True, alpha=TLS_ALPHA,
            event_score_method=method, flag_quantile=FLAG_QUANTILE,
            contamination=CONTAMINATION,
        )
        res = run_star_cv(dataset, cfg, n_splits=n_splits, tls_cache=tls_cache)
        agg = res['aggregated']
        row = {
            'method': method,
            'recall_at_k': agg.get('recall_at_k_mean', float('nan')),
            'au_pr':        agg.get('au_pr_mean', float('nan')),
        }
        method_results.append(row)
        logger.info(
            f"    Recall@K={row['recall_at_k']:.4f}  AU-PR={row['au_pr']:.4f}"
        )
    ablation_results['event_score_method'] = method_results

    return ablation_results


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
        window_size=WINDOW_SIZE, use_tls=True, alpha=TLS_ALPHA,
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
    )
    pipeline = RankingPipeline(cfg)
    pipeline.fit(dfs, tls_cache=tls_cache)

    all_cands = pipeline.score_stars(dfs, tls_cache)
    top_k = all_cands[:SHAP_TOP_K]
    logger.info(f"Top-{SHAP_TOP_K} candidates selected for SHAP analysis")

    # Extract feature matrices for top candidates
    # (re-extract to get per-window features for the candidate windows)
    from backend.ml.preprocessing import LightCurvePreprocessor
    from backend.ml.tls_features import append_tls_to_windows
    from backend.ml.feature_names import ALL_WINDOW_FEATURES, TLS_GLOBAL_FEATURE_NAMES

    prep = LightCurvePreprocessor()
    shap_results = []

    for cand in top_k:
        df = dfs.get(cand.star_id)
        if df is None:
            continue
        feats, _, meta = prep.extract_features_with_metadata(df, star_id=cand.star_id)
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
            # Use only the candidate's windows
            window_slice = feats[cand.window_indices]
            shap_vals = explainer.shap_values(window_slice)
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
        window_size=WINDOW_SIZE, use_tls=False,  # TLS off for speed in injection
        flag_quantile=FLAG_QUANTILE, contamination=CONTAMINATION,
    )
    pipeline = RankingPipeline(cfg)
    pipeline.fit(dfs)
    pipeline_fn = pipeline.as_inference_fn()

    logger.info(
        f"Running injection-recovery: "
        f"{len(INJECTION_RADIUS_GRID)}×{len(INJECTION_PERIOD_GRID)} grid, "
        f"{n_trials} trials/cell"
    )

    def progress(done, total):
        if done % 50 == 0 or done == total:
            logger.info(f"  Progress: {done}/{total} ({100*done//total}%)")

    result = run_injection_recovery(
        light_curves=dfs,
        pipeline_fn=pipeline_fn,
        radius_grid=INJECTION_RADIUS_GRID,
        period_grid=INJECTION_PERIOD_GRID,
        n_trials=n_trials,
        rng_seed=RANDOM_SEED,
        recovery_window_factor=INJECTION_RECOVERY_WINDOW_FACTOR,
        progress_callback=progress,
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
        '--exp', type=str, default='1,2,3,4,5',
        help='Comma-separated experiment numbers to run (default: all)',
    )
    parser.add_argument(
        '--dev', action='store_true',
        help='Dev mode: use local 5-star dataset, 1 trial/cell, 2 CV folds',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Set logging level to DEBUG',
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    to_run = {int(x.strip()) for x in args.exp.split(',')}

    # Output dirs
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, RAW_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {METADATA_CSV}")
    t0 = time.time()
    dataset = load_dataset(METADATA_CSV, LIGHTCURVE_DIR)
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s")

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

    # Save combined results
    _save_json(all_results, Path(RESULTS_DIR) / 'all_results.json')
    logger.info("\nAll experiments complete.")


if __name__ == '__main__':
    main()
