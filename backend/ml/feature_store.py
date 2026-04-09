"""
Disk-backed per-star feature store for RankingPipeline.

Features are saved as float32 .npy files keyed by (star_id, config_hash).
The config_hash encodes window_size, stride, sigma_clip, use_tls — changing
any of those automatically invalidates cached entries.

Layout inside store_dir/:
    {safe_star_id}_{config_hash}.npy   # float32 (n_windows, n_features)
    {safe_star_id}_{config_hash}.json  # list of window metadata dicts

Benefits:
  - Features computed once per config; reused across all CV folds and ablations.
  - Raw CSV files are not re-read for stars already in the store.
  - Reduces peak RAM: stars whose features are cached are never fully loaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StarFeatureStore:
    """
    Disk-backed per-star feature cache, keyed by (star_id, config_hash).

    Thread-safety: reads are safe for concurrent workers.  Concurrent writes
    to the *same* star_id are possible in multi-fold runs but are safe because
    features are deterministic — last-write-wins produces the correct result.
    """

    def __init__(self, store_dir: str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config hash
    # ------------------------------------------------------------------

    @staticmethod
    def make_config_hash(
        window_size: int,
        stride: int,
        sigma_clip: float,
        use_tls: bool,
    ) -> str:
        """8-char hex digest uniquely identifying a feature extraction config."""
        payload = json.dumps(
            {
                "window_size": window_size,
                "stride": stride,
                "sigma_clip": round(float(sigma_clip), 4),
                "use_tls": bool(use_tls),
            },
            sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _safe_id(self, star_id: str) -> str:
        return star_id.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _feat_path(self, star_id: str, chash: str) -> Path:
        return self._dir / f"{self._safe_id(star_id)}_{chash}.npy"

    def _meta_path(self, star_id: str, chash: str) -> Path:
        return self._dir / f"{self._safe_id(star_id)}_{chash}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, star_id: str, chash: str) -> bool:
        return (
            self._feat_path(star_id, chash).exists()
            and self._meta_path(star_id, chash).exists()
        )

    def load(
        self, star_id: str, chash: str
    ) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """Return (features_f32, metadata_list) or None on miss/corrupt."""
        fp = self._feat_path(star_id, chash)
        mp = self._meta_path(star_id, chash)
        if not fp.exists() or not mp.exists():
            return None
        try:
            features = np.load(fp)  # float32
            with open(mp, "r") as f:
                metadata = json.load(f)
            return features, metadata
        except Exception as e:
            logger.warning(f"Feature store load failed for {star_id}: {e} — recomputing")
            return None

    def save(
        self,
        star_id: str,
        chash: str,
        features: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """Persist features + metadata; silently skips on I/O failure."""
        fp = self._feat_path(star_id, chash)
        mp = self._meta_path(star_id, chash)
        try:
            np.save(fp, features.astype(np.float32))
            with open(mp, "w") as f:
                json.dump(metadata, f, separators=(",", ":"))
        except Exception as e:
            logger.warning(f"Feature store save failed for {star_id}: {e}")
            for p in (fp, mp):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def clear(self) -> int:
        """Delete all cached entries. Returns count of stars removed."""
        n = 0
        for npy in self._dir.glob("*.npy"):
            npy.unlink()
            jsn = npy.with_suffix(".json")
            if jsn.exists():
                jsn.unlink()
            n += 1
        return n

    def stats(self) -> Dict:
        npy_files = list(self._dir.glob("*.npy"))
        total_b = sum(p.stat().st_size for p in npy_files)
        return {
            "n_stars": len(npy_files),
            "total_mb": round(total_b / 1024 ** 2, 1),
            "store_dir": str(self._dir),
        }
