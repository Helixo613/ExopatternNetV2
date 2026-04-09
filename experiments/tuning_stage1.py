"""
Stage 1 tuning sweep: FLAG_QUANTILE × MAX_TRAIN_WINDOWS.

2-fold CV on the full corpus. Reuses the warmed disk feature store.
Results saved to results/paper/tuning_stage1.json.

Usage:
    python experiments/tuning_stage1.py 2>&1 | tee results/paper/tuning_stage1.log
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.paper_config import (
    METADATA_CSV, LIGHTCURVE_DIR, RANDOM_SEED,
    WINDOW_SIZE, STRIDE, GAP_TOLERANCE, MAX_EVENT_WINDOWS,
    EVENT_SCORE_METHOD, TLS_ALPHA, CONTAMINATION,
)
from experiments.run_paper_experiments import (
    load_dataset, precompute_tls_cache, run_star_cv,
    FEATURE_STORE_DIR, TLS_DISK_CACHE,
)
from backend.ml.pipeline import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('tuning_stage1')

# ---------------------------------------------------------------------------
# Stage 1 grid
# ---------------------------------------------------------------------------
FLAG_QUANTILES     = [0.975, 0.970, 0.950, 0.900]
MAX_TRAIN_WINDOWS  = [200_000, 400_000, 600_000]

N_FOLDS = 2   # fast tuning; final config will use 5 folds


def main() -> None:
    out_dir = Path('results/paper')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'tuning_stage1.json'

    # Load dataset (lazy corpus — CSVs loaded on demand)
    logger.info("Loading dataset...")
    dataset = load_dataset(METADATA_CSV, LIGHTCURVE_DIR)
    dfs = dataset['dfs']

    # TLS cache (already on disk from prior runs)
    logger.info("Loading TLS cache...")
    tls_cache = precompute_tls_cache(dfs)

    grid = list(product(FLAG_QUANTILES, MAX_TRAIN_WINDOWS))
    logger.info(f"Stage 1: {len(grid)} trials ({len(FLAG_QUANTILES)} quantiles × {len(MAX_TRAIN_WINDOWS)} mtw), {N_FOLDS} folds each")

    results: List[Dict] = []

    for trial_idx, (fq, mtw) in enumerate(grid):
        label = f"fq={fq:.3f}_mtw={mtw//1000}k"
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial_idx+1}/{len(grid)}: {label}")
        logger.info(f"{'='*60}")

        cfg = PipelineConfig(
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            flag_quantile=fq,
            gap_tolerance=GAP_TOLERANCE,
            max_event_windows=MAX_EVENT_WINDOWS,
            event_score_method=EVENT_SCORE_METHOD,
            alpha=TLS_ALPHA,
            contamination=CONTAMINATION,
            use_tls=True,
            feature_store_dir=str(FEATURE_STORE_DIR),
            max_train_windows=mtw,
        )

        t0 = time.time()
        try:
            cv_result = run_star_cv(
                dataset, cfg,
                n_splits=N_FOLDS,
                tls_cache=tls_cache,
            )
            elapsed = time.time() - t0
            agg = cv_result['aggregated']

            trial_result = {
                'trial': trial_idx,
                'flag_quantile': fq,
                'max_train_windows': mtw,
                'n_folds': N_FOLDS,
                'elapsed_s': round(elapsed, 1),
                'recall_at_k_mean': agg.get('recall_at_k_mean'),
                'recall_at_k_std': agg.get('recall_at_k_std'),
                'au_pr_mean': agg.get('au_pr_mean'),
                'au_pr_std': agg.get('au_pr_std'),
                'event_f1_mean': agg.get('event_f1_mean'),
                'event_f1_std': agg.get('event_f1_std'),
                'event_recall_mean': agg.get('event_recall_mean'),
                'event_precision_mean': agg.get('event_precision_mean'),
                'per_fold': cv_result['per_fold'],
            }

            logger.info(
                f"  → Recall@K={agg.get('recall_at_k_mean',0):.4f} ± {agg.get('recall_at_k_std',0):.4f}  "
                f"AU-PR={agg.get('au_pr_mean',0):.4f}  "
                f"EventF1={agg.get('event_f1_mean',0):.4f}  "
                f"({elapsed:.0f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  Trial {label} FAILED: {e}")
            trial_result = {
                'trial': trial_idx,
                'flag_quantile': fq,
                'max_train_windows': mtw,
                'error': str(e),
                'elapsed_s': round(elapsed, 1),
            }

        results.append(trial_result)

        # Save incrementally (so partial results survive crashes)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary table
    logger.info(f"\n{'='*70}")
    logger.info("STAGE 1 RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'FQ':>6} {'MTW':>6} {'Recall@K':>10} {'AU-PR':>8} {'EventF1':>9} {'Time':>6}")
    logger.info(f"{'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*9} {'-'*6}")

    best_trial = None
    best_recall = -1.0

    for r in results:
        if 'error' in r:
            logger.info(f"{r['flag_quantile']:>6.3f} {r['max_train_windows']//1000:>5}k   FAILED")
            continue
        recall = r.get('recall_at_k_mean', 0) or 0
        aupr = r.get('au_pr_mean', 0) or 0
        f1 = r.get('event_f1_mean', 0) or 0
        t = r.get('elapsed_s', 0)
        logger.info(
            f"{r['flag_quantile']:>6.3f} {r['max_train_windows']//1000:>5}k "
            f"{recall:>10.4f} {aupr:>8.4f} {f1:>9.4f} {t:>5.0f}s"
        )
        if recall > best_recall:
            best_recall = recall
            best_trial = r

    if best_trial:
        logger.info(f"\nBest: fq={best_trial['flag_quantile']}, mtw={best_trial['max_train_windows']//1000}k → Recall@K={best_recall:.4f}")

    logger.info(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
