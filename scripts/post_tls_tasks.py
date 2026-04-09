#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.feature_store import StarFeatureStore
from backend.ml.tls_features import _TLS_FALLBACK
from experiments.paper_config import (
    INJECTION_N_TRIALS,
    METADATA_CSV,
    N_CV_FOLDS,
    STRIDE,
    WINDOW_SIZE,
)


RESULTS_DIR = ROOT / "results" / "paper"
WARMUP_LOG = RESULTS_DIR / "cache_warmup.log"
SMOKE_LOG = RESULTS_DIR / "post_tls_smoke.log"
REPORT_MD = RESULTS_DIR / "post_tls_report.md"
REPORT_JSON = RESULTS_DIR / "post_tls_report.json"
TLS_CACHE_PATH = RESULTS_DIR / "tls_cache.json"
FEATURE_STORE_DIR = RESULTS_DIR / "feature_store"
RAW_DIR = RESULTS_DIR / "raw"
POLL_SECONDS = 120
SIGMA_CLIP = 5.0


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_hours(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    return f"{seconds / 3600.0:.1f}h"


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(ROOT / METADATA_CSV)


def feature_store_hash(use_tls: bool) -> str:
    return StarFeatureStore.make_config_hash(WINDOW_SIZE, STRIDE, SIGMA_CLIP, use_tls)


def count_store_entries(star_ids: Iterable[str], chash: str) -> int:
    store = StarFeatureStore(str(FEATURE_STORE_DIR))
    return sum(1 for sid in star_ids if store.has(sid, chash))


def total_windows(star_ids: Iterable[str], chash: str) -> int:
    store = StarFeatureStore(str(FEATURE_STORE_DIR))
    total = 0
    for sid in star_ids:
        path = store._feat_path(sid, chash)
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        total += int(arr.shape[0])
    return total


def wait_for_warmup(star_ids: List[str]) -> Dict:
    tls_hash = feature_store_hash(True)
    no_tls_hash = feature_store_hash(False)
    total = len(star_ids)

    while True:
        log_tail = _tail(WARMUP_LOG, 20)
        tls_count = 0
        if TLS_CACHE_PATH.exists():
            try:
                tls_count = len(json.loads(TLS_CACHE_PATH.read_text()))
            except Exception:
                tls_count = -1
        tls_on_count = count_store_entries(star_ids, tls_hash)
        tls_off_count = count_store_entries(star_ids, no_tls_hash)

        print(
            f"wait warmup tls={tls_count}/{total} "
            f"store_off={tls_off_count}/{total} store_on={tls_on_count}/{total}",
            flush=True,
        )

        if "warmup_done" in log_tail:
            return {
                "tls_count": tls_count,
                "tls_on_count": tls_on_count,
                "tls_off_count": tls_off_count,
                "log_tail": log_tail,
            }

        if "Traceback" in log_tail:
            raise RuntimeError(f"Cache warm-up failed.\n{log_tail}")

        time.sleep(POLL_SECONDS)


def validate_tls_cache(star_ids: List[str]) -> Dict:
    required_keys = set(_TLS_FALLBACK.keys())
    raw = json.loads(TLS_CACHE_PATH.read_text())
    missing_stars = [sid for sid in star_ids if sid not in raw]
    extra_stars = sorted(set(raw) - set(star_ids))
    malformed = []
    missing_keys = {}
    negative_fields = []
    fallback_count = 0
    nonfallback_sde = []
    nonfallback_period = []

    for sid in star_ids:
        feats = raw.get(sid)
        if not isinstance(feats, dict):
            malformed.append(sid)
            continue
        miss = sorted(required_keys - set(feats))
        if miss:
            missing_keys[sid] = miss
            continue

        period = feats.get("tls_period")
        duration = feats.get("tls_duration")
        depth = feats.get("tls_depth")
        sde = feats.get("tls_sde")
        snr = feats.get("tls_snr")
        odd_even = feats.get("tls_odd_even")

        any_none = False
        for name, value in (
            ("tls_period", period),
            ("tls_duration", duration),
            ("tls_depth", depth),
            ("tls_sde", sde),
            ("tls_snr", snr),
        ):
            if value is None:
                negative_fields.append((sid, name, value))
                any_none = True
                continue
            if float(value) < 0:
                negative_fields.append((sid, name, value))

        if odd_even is None or not np.isfinite(float(odd_even)):
            malformed.append(sid)
            continue

        if any_none:
            malformed.append(sid)
            continue

        is_fallback = (
            float(period) == 0.0
            and float(duration) == 0.0
            and float(depth) == 0.0
            and float(sde) == 0.0
            and float(snr) == 0.0
        )
        if is_fallback:
            fallback_count += 1
        else:
            nonfallback_sde.append(float(sde))
            nonfallback_period.append(float(period))

    valid = not missing_stars and not malformed and not missing_keys and not negative_fields
    return {
        "valid": valid,
        "n_expected": len(star_ids),
        "n_actual": len(raw),
        "missing_stars": missing_stars,
        "extra_stars": extra_stars,
        "malformed_stars": malformed,
        "missing_keys": missing_keys,
        "negative_fields": negative_fields,
        "fallback_count": fallback_count,
        "nonfallback_count": len(nonfallback_sde),
        "sde_median": float(np.median(nonfallback_sde)) if nonfallback_sde else None,
        "sde_p95": float(np.percentile(nonfallback_sde, 95)) if nonfallback_sde else None,
        "period_median": float(np.median(nonfallback_period)) if nonfallback_period else None,
        "period_max": float(np.max(nonfallback_period)) if nonfallback_period else None,
    }


def run_smoke() -> Dict:
    cmd = [sys.executable, "experiments/run_paper_experiments.py", "--smoke"]
    started = time.time()
    with open(SMOKE_LOG, "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
    elapsed = time.time() - started

    text = SMOKE_LOG.read_text(errors="replace") if SMOKE_LOG.exists() else ""
    exp_times = {
        int(exp): float(sec)
        for exp, sec in re.findall(r"Experiment (\d) done in ([0-9.]+)s", text)
    }
    return {
        "exit_code": code,
        "elapsed_seconds": elapsed,
        "experiment_times_seconds": exp_times,
        "log_path": str(SMOKE_LOG),
    }


def _safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def validate_smoke_outputs() -> Dict:
    exp1 = _safe_read_json(RAW_DIR / "experiment_1_cv.json")
    exp2 = _safe_read_json(RAW_DIR / "experiment_2_conformal.json")
    exp3 = _safe_read_json(RAW_DIR / "experiment_3_ablation.json")
    exp4 = _safe_read_json(RAW_DIR / "experiment_4_shap.json")
    exp5 = _safe_read_json(RAW_DIR / "experiment_5_injection.json")

    def finite(x):
        return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))

    exp1_ok = bool(
        exp1
        and finite(exp1.get("aggregated", {}).get("recall_at_k_mean"))
        and finite(exp1.get("aggregated", {}).get("precision_at_k_mean"))
        and finite(exp1.get("aggregated", {}).get("event_f1_mean"))
        and finite(exp1.get("aggregated", {}).get("au_pr_mean"))
    )
    exp2_ok = bool(exp2 and exp2.get("conformal_diagnostics"))
    exp3_ok = bool(
        exp3
        and isinstance(exp3.get("alpha_sweep"), list)
        and isinstance(exp3.get("event_score_method"), list)
    )
    exp4_skipped = bool(exp4 and exp4.get("skipped"))
    exp4_ok = bool(
        exp4 and (exp4_skipped or isinstance(exp4.get("shap_results"), list))
    )
    exp5_ok = bool(exp5 and finite(exp5.get("overall_recovery_rate")))

    return {
        "exp1_ok": exp1_ok,
        "exp2_ok": exp2_ok,
        "exp3_ok": exp3_ok,
        "exp4_ok": exp4_ok,
        "exp4_skipped": exp4_skipped,
        "exp4_reason": exp4.get("reason") if exp4_skipped else None,
        "exp5_ok": exp5_ok,
        "all_required_ok": exp1_ok and exp2_ok and exp3_ok and exp5_ok,
        "exp1_metrics": exp1.get("aggregated") if exp1 else None,
        "exp5_overall_recovery_rate": exp5.get("overall_recovery_rate") if exp5 else None,
    }


def estimate_eta(star_ids: List[str], smoke_times: Dict[int, float]) -> Dict:
    smoke_star_ids = star_ids[:20]
    folds_smoke = 2.0
    folds_full = float(N_CV_FOLDS)
    tls_on_hash = feature_store_hash(True)
    tls_off_hash = feature_store_hash(False)

    tls_on_smoke_windows = total_windows(smoke_star_ids, tls_on_hash)
    tls_on_full_windows = total_windows(star_ids, tls_on_hash)
    tls_off_smoke_windows = total_windows(smoke_star_ids, tls_off_hash)
    tls_off_full_windows = total_windows(star_ids, tls_off_hash)

    tls_on_scale = (
        (tls_on_full_windows / tls_on_smoke_windows) * (folds_full / folds_smoke)
        if tls_on_smoke_windows
        else None
    )
    tls_off_scale = (
        (tls_off_full_windows / tls_off_smoke_windows) * (folds_full / folds_smoke)
        if tls_off_smoke_windows
        else None
    )

    avg_off_smoke = tls_off_smoke_windows / max(len(smoke_star_ids), 1)
    avg_off_full = tls_off_full_windows / max(len(star_ids), 1)
    inj_star_factor = (avg_off_full / avg_off_smoke) if avg_off_smoke else None

    estimates = {}
    if 1 in smoke_times and tls_on_scale:
        estimates["experiment_1_seconds"] = smoke_times[1] * tls_on_scale
    if 2 in smoke_times and tls_off_scale:
        estimates["experiment_2_seconds"] = smoke_times[2] * tls_off_scale
    if 3 in smoke_times and tls_on_scale:
        estimates["experiment_3_seconds"] = smoke_times[3] * tls_on_scale
    if 4 in smoke_times and tls_on_smoke_windows:
        estimates["experiment_4_seconds"] = smoke_times[4] * (tls_on_full_windows / tls_on_smoke_windows)
    if 5 in smoke_times and inj_star_factor:
        estimates["experiment_5_seconds"] = smoke_times[5] * INJECTION_N_TRIALS * inj_star_factor

    core_seconds = sum(estimates.get(f"experiment_{i}_seconds", 0.0) for i in (1, 2, 3, 4))
    full_seconds = core_seconds + estimates.get("experiment_5_seconds", 0.0)

    return {
        "smoke_windows_tls_on": tls_on_smoke_windows,
        "full_windows_tls_on": tls_on_full_windows,
        "smoke_windows_tls_off": tls_off_smoke_windows,
        "full_windows_tls_off": tls_off_full_windows,
        "tls_on_scale": tls_on_scale,
        "tls_off_scale": tls_off_scale,
        "injection_star_factor": inj_star_factor,
        "estimates_seconds": estimates,
        "core_seconds": core_seconds,
        "full_seconds": full_seconds,
        "core_range_seconds": (core_seconds * 0.8, core_seconds * 1.35) if core_seconds else None,
        "full_range_seconds": (full_seconds * 0.8, full_seconds * 1.45) if full_seconds else None,
    }


def write_report(
    warmup_state: Dict,
    tls_validation: Dict,
    smoke_run: Dict,
    smoke_validation: Dict,
    eta: Dict,
) -> None:
    report = {
        "warmup_state": warmup_state,
        "tls_validation": tls_validation,
        "smoke_run": smoke_run,
        "smoke_validation": smoke_validation,
        "eta": eta,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# Post-TLS Warm-up Report")
    lines.append("")
    lines.append("## Warm-up Completion")
    lines.append(f"- TLS entries: {warmup_state['tls_count']}")
    lines.append(f"- no-TLS feature-store coverage: {warmup_state['tls_off_count']}")
    lines.append(f"- TLS-on feature-store coverage: {warmup_state['tls_on_count']}")
    lines.append("")
    lines.append("## TLS Validation")
    lines.append(f"- Valid: {tls_validation['valid']}")
    lines.append(f"- Expected stars: {tls_validation['n_expected']}")
    lines.append(f"- Cache entries: {tls_validation['n_actual']}")
    lines.append(f"- Fallback entries: {tls_validation['fallback_count']}")
    lines.append(f"- Non-fallback entries: {tls_validation['nonfallback_count']}")
    lines.append(f"- Median SDE: {tls_validation['sde_median']}")
    lines.append(f"- 95th pct SDE: {tls_validation['sde_p95']}")
    lines.append(f"- Median period: {tls_validation['period_median']}")
    lines.append(f"- Max period: {tls_validation['period_max']}")
    if tls_validation["missing_stars"]:
        lines.append(f"- Missing stars: {len(tls_validation['missing_stars'])}")
    if tls_validation["extra_stars"]:
        lines.append(f"- Extra stars: {len(tls_validation['extra_stars'])}")
    if tls_validation["missing_keys"]:
        lines.append(f"- Stars with missing keys: {len(tls_validation['missing_keys'])}")
    if tls_validation["negative_fields"]:
        lines.append(f"- Negative-valued fields: {len(tls_validation['negative_fields'])}")
    lines.append("")
    lines.append("## Smoke Run")
    lines.append(f"- Exit code: {smoke_run['exit_code']}")
    lines.append(f"- Total smoke time: {_fmt_seconds(smoke_run['elapsed_seconds'])}")
    for exp, sec in sorted(smoke_run["experiment_times_seconds"].items()):
        lines.append(f"- Experiment {exp}: {_fmt_seconds(sec)}")
    lines.append("")
    lines.append("## Smoke Validation")
    lines.append(f"- Experiments 1,2,3,5 valid: {smoke_validation['all_required_ok']}")
    lines.append(f"- Experiment 4 valid: {smoke_validation['exp4_ok']}")
    if smoke_validation["exp4_skipped"]:
        lines.append(f"- Experiment 4 skipped reason: {smoke_validation['exp4_reason']}")
    exp1_metrics = smoke_validation.get("exp1_metrics") or {}
    if exp1_metrics:
        lines.append(f"- Smoke Recall@K mean: {exp1_metrics.get('recall_at_k_mean')}")
        lines.append(f"- Smoke Precision@K mean: {exp1_metrics.get('precision_at_k_mean')}")
        lines.append(f"- Smoke Event F1 mean: {exp1_metrics.get('event_f1_mean')}")
        lines.append(f"- Smoke AU-PR mean: {exp1_metrics.get('au_pr_mean')}")
    lines.append("")
    lines.append("## ETA")
    estimates = eta["estimates_seconds"]
    for key in sorted(estimates):
        lines.append(f"- {key}: {_fmt_hours(estimates[key])} ({_fmt_seconds(estimates[key])})")
    lines.append(f"- Core experiments 1-4 estimate: {_fmt_hours(eta['core_seconds'])}")
    if eta["core_range_seconds"]:
        lo, hi = eta["core_range_seconds"]
        lines.append(f"- Core experiments 1-4 range: {_fmt_hours(lo)} to {_fmt_hours(hi)}")
    lines.append(f"- Full suite 1-5 estimate: {_fmt_hours(eta['full_seconds'])}")
    if eta["full_range_seconds"]:
        lo, hi = eta["full_range_seconds"]
        lines.append(f"- Full suite 1-5 range: {_fmt_hours(lo)} to {_fmt_hours(hi)}")
    lines.append("")
    lines.append("## Recommendation")
    if tls_validation["valid"] and smoke_validation["all_required_ok"]:
        if smoke_validation["exp4_skipped"]:
            lines.append("- Core paper path is ready. Experiment 4 still requires `shap` if you want the full five-experiment suite.")
        else:
            lines.append("- Full paper path passed smoke on warmed caches.")
    else:
        lines.append("- Do not start the full run until the validation issues above are resolved.")

    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    meta = load_metadata()
    star_ids = [str(x) for x in meta["target_id"].tolist()]
    print(f"post_tls_start stars={len(star_ids)}", flush=True)
    warmup_state = wait_for_warmup(star_ids)
    print("warmup_complete detected", flush=True)

    tls_validation = validate_tls_cache(star_ids)
    print(f"tls_validation valid={tls_validation['valid']} fallback={tls_validation['fallback_count']}", flush=True)

    smoke_run = run_smoke()
    print(f"smoke_complete exit={smoke_run['exit_code']} elapsed={smoke_run['elapsed_seconds']:.1f}s", flush=True)

    smoke_validation = validate_smoke_outputs()
    eta = estimate_eta(star_ids, smoke_run["experiment_times_seconds"])
    write_report(warmup_state, tls_validation, smoke_run, smoke_validation, eta)
    print(f"report_written md={REPORT_MD} json={REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
