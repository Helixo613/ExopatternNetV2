from __future__ import annotations

import json
import time
from pathlib import Path

import experiments.run_paper_experiments as rpe
from backend.ml.pipeline import PipelineConfig


def main() -> None:
    cfg = PipelineConfig(
        window_size=rpe.WINDOW_SIZE,
        stride=rpe.STRIDE,
        flag_quantile=0.900,
        gap_tolerance=rpe.GAP_TOLERANCE,
        max_event_windows=rpe.MAX_EVENT_WINDOWS,
        event_score_method="top3_mean",
        alpha=0.9,
        contamination=0.03,
        use_tls=True,
        feature_store_dir=str(rpe.FEATURE_STORE_DIR),
        max_train_windows=200_000,
    )

    t0 = time.time()
    dataset = rpe.load_dataset(rpe.METADATA_CSV, rpe.LIGHTCURVE_DIR)
    result = rpe.run_star_cv(dataset, cfg, n_splits=2)

    out = {
        "experiment": "exp1_root_cause_check",
        "config": {
            "flag_quantile": 0.900,
            "max_train_windows": 200_000,
            "event_score_method": "top3_mean",
            "alpha": 0.9,
            "contamination": 0.03,
            "train_gt_masking": True,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "result": result,
    }
    out_path = Path("results/paper/root_cause_exp1_masked_2fold.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps(
            {
                "elapsed_s": out["elapsed_s"],
                "aggregated": result["aggregated"],
                "out": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
