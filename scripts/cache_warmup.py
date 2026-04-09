#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.feature_store import StarFeatureStore
from backend.ml.pipeline import PipelineConfig, RankingPipeline
from backend.ml.tls_features import _TLS_FALLBACK, extract_tls_features


WINDOW_SIZE = 50
STRIDE = 12
SIGMA_CLIP = 5.0
FEATURE_STORE_DIR = "results/paper/feature_store"
TLS_CACHE_PATH = Path("results/paper/tls_cache.json")
META_PATH = Path("data/labeled/metadata.csv")
LC_DIR = Path("data/labeled/lightcurves")


def load_rows():
    meta = pd.read_csv(META_PATH)
    return [(str(r["target_id"]), LC_DIR / r["filename"]) for _, r in meta.iterrows()]


def warm_no_tls_feature_store(rows):
    store = StarFeatureStore(FEATURE_STORE_DIR)
    cfg = PipelineConfig(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        sigma_clip=SIGMA_CLIP,
        use_tls=False,
        feature_store_dir=FEATURE_STORE_DIR,
        max_train_windows=200_000,
    )
    pipe = RankingPipeline(cfg)
    chash = store.make_config_hash(WINDOW_SIZE, STRIDE, SIGMA_CLIP, False)
    missing = [(sid, path) for sid, path in rows if not store.has(sid, chash)]
    print(f"phase1_no_tls start existing={len(rows) - len(missing)} missing={len(missing)} hash={chash}", flush=True)

    for i, (sid, path) in enumerate(missing, start=1):
        try:
            df = pd.read_csv(path)
            pipe._extract_star_features(df, sid, tls_cache=None)
            status = "ok"
        except Exception as e:
            status = f"error={e}"
        pipe.clear_cache()
        if i == 1 or i % 10 == 0 or i == len(missing):
            stats = StarFeatureStore(FEATURE_STORE_DIR).stats()
            print(
                f"phase1_no_tls [{i}/{len(missing)}] {sid} {status} "
                f"store_n={stats['n_stars']} store_mb={stats['total_mb']}",
                flush=True,
            )


def load_tls_cache():
    if not TLS_CACHE_PATH.exists():
        return {}
    raw = json.loads(TLS_CACHE_PATH.read_text())
    return {
        sid: {k: (float("nan") if v is None else v) for k, v in feats.items()}
        for sid, feats in raw.items()
    }


def save_tls_cache(cache):
    TLS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        sid: {k: (None if (v is not None and v != v) else v) for k, v in feats.items()}
        for sid, feats in cache.items()
    }
    TLS_CACHE_PATH.write_text(json.dumps(serializable, indent=2))


def warm_tls_cache(rows):
    cache = load_tls_cache()
    missing = [(sid, path) for sid, path in rows if sid not in cache]
    print(f"phase2_tls start existing={len(cache)} missing={len(missing)}", flush=True)

    for i, (sid, path) in enumerate(missing, start=1):
        try:
            df = pd.read_csv(path)
            feats = extract_tls_features(df["time"].values, df["flux"].values)
            cache[sid] = feats
            status = f"SDE={feats.get('tls_sde', float('nan')):.2f}"
        except Exception as e:
            cache[sid] = dict(_TLS_FALLBACK)
            status = f"fallback error={e}"

        if i == 1 or i % 5 == 0 or i == len(missing):
            save_tls_cache(cache)
            print(f"phase2_tls [{i}/{len(missing)}] {sid} {status} total={len(cache)}", flush=True)

    save_tls_cache(cache)
    return cache


def warm_tls_feature_store(rows, tls_cache):
    store = StarFeatureStore(FEATURE_STORE_DIR)
    cfg = PipelineConfig(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        sigma_clip=SIGMA_CLIP,
        use_tls=True,
        feature_store_dir=FEATURE_STORE_DIR,
        max_train_windows=200_000,
    )
    pipe = RankingPipeline(cfg)
    chash = store.make_config_hash(WINDOW_SIZE, STRIDE, SIGMA_CLIP, True)
    missing = [(sid, path) for sid, path in rows if not store.has(sid, chash)]
    print(f"phase3_tls_on start existing={len(rows) - len(missing)} missing={len(missing)} hash={chash}", flush=True)

    for i, (sid, path) in enumerate(missing, start=1):
        try:
            df = pd.read_csv(path)
            pipe._extract_star_features(df, sid, tls_cache=tls_cache)
            status = "ok"
        except Exception as e:
            status = f"error={e}"
        pipe.clear_cache()
        if i == 1 or i % 10 == 0 or i == len(missing):
            stats = StarFeatureStore(FEATURE_STORE_DIR).stats()
            print(
                f"phase3_tls_on [{i}/{len(missing)}] {sid} {status} "
                f"store_n={stats['n_stars']} store_mb={stats['total_mb']}",
                flush=True,
            )


def main():
    rows = load_rows()
    print(f"warmup_start stars={len(rows)}", flush=True)
    warm_no_tls_feature_store(rows)
    tls_cache = warm_tls_cache(rows)
    warm_tls_feature_store(rows, tls_cache)
    print("warmup_done", flush=True)


if __name__ == "__main__":
    main()
