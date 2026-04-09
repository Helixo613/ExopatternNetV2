"""
Warm the disk feature store for all local labeled stars.

Run this once after downloading data to populate results/paper/feature_store/.
Subsequent paper-experiment runs will skip feature extraction entirely for
stars already in the store, collapsing the cost to a fast .npy file load.

Usage:
    python scripts/warm_feature_store.py [--data-dir data/labeled] [--use-tls]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.pipeline import RankingPipeline, PipelineConfig
from backend.ml.feature_store import StarFeatureStore
from experiments.run_paper_experiments import (
    FEATURE_STORE_DIR,
    WINDOW_SIZE,
    CONTAMINATION,
)
try:
    from experiments.paper_config import STRIDE, SIGMA_CLIP
except ImportError:
    STRIDE = 12
    SIGMA_CLIP = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("warm_feature_store")


def load_local_dfs(data_dir: Path) -> dict:
    """Load all light curves under data_dir/lightcurves/ or data_dir/ directly."""
    lc_dirs = [data_dir / "lightcurves", data_dir]
    dfs = {}
    for lc_dir in lc_dirs:
        if not lc_dir.exists():
            continue
        for csv_path in sorted(lc_dir.glob("*.csv")):
            star_id = csv_path.stem
            if star_id not in dfs:
                try:
                    dfs[star_id] = pd.read_csv(csv_path)
                except Exception as e:
                    logger.warning(f"Could not load {csv_path}: {e}")
    return dfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm ExoPattern disk feature store")
    parser.add_argument("--data-dir", default="data/labeled", help="Local labeled data directory")
    parser.add_argument("--use-tls", action="store_true", help="Include TLS global features (slower)")
    parser.add_argument("--max-train-windows", type=int, default=None,
                        help="Cap training matrix size (e.g. 200000 for 32 GB RAM)")
    parser.add_argument("--store-dir", default=str(FEATURE_STORE_DIR),
                        help="Feature store directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    store_dir = Path(args.store_dir)

    logger.info(f"Loading light curves from {data_dir} ...")
    dfs = load_local_dfs(data_dir)
    if not dfs:
        logger.error(f"No CSV files found under {data_dir}. Run download_dataset.py first.")
        sys.exit(1)
    logger.info(f"Found {len(dfs)} stars: {sorted(dfs)[:5]}{'...' if len(dfs) > 5 else ''}")

    cfg = PipelineConfig(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        sigma_clip=SIGMA_CLIP,
        use_tls=args.use_tls,
        contamination=CONTAMINATION,
        n_estimators=10,          # minimal fit — we only want features, not model quality
        feature_store_dir=str(store_dir),
        max_train_windows=args.max_train_windows,
    )

    logger.info(
        f"Config: window_size={cfg.window_size}, stride={cfg.stride}, "
        f"sigma_clip={cfg.sigma_clip}, use_tls={cfg.use_tls}"
    )
    logger.info(f"Store directory: {store_dir}")

    store = StarFeatureStore(str(store_dir))
    chash = StarFeatureStore.make_config_hash(cfg.window_size, cfg.stride, cfg.sigma_clip, cfg.use_tls)

    already_cached = sum(1 for sid in dfs if store.has(sid, chash))
    missing = [sid for sid in dfs if not store.has(sid, chash)]
    logger.info(f"Already in store: {already_cached}/{len(dfs)}. To compute: {len(missing)}")

    if not missing:
        logger.info("Store already warm — nothing to do.")
        stats = store.stats()
        logger.info(f"Store stats: {stats['n_stars']} stars, {stats['total_mb']:.1f} MB")
        return

    # Fit pipeline (needed for impute medians; uses only local stars)
    pipeline = RankingPipeline(cfg)
    logger.info("Fitting pipeline on local stars (needed for impute medians)...")
    t0 = time.time()
    pipeline.fit(dfs)
    logger.info(f"Fit done in {time.time() - t0:.1f}s")

    # _extract_star_features writes to the store automatically during fit
    # For safety, explicitly preload remaining stars
    n_saved = pipeline.preload_to_store(dfs)
    logger.info(f"preload_to_store saved/verified {n_saved} stars")

    stats = store.stats()
    logger.info(
        f"Store warm: {stats['n_stars']} stars, {stats['total_mb']:.1f} MB "
        f"at {store_dir}"
    )


if __name__ == "__main__":
    main()
