"""
Candidate event generation from per-window anomaly scores.

A candidate event is a contiguous group of flagged windows that represents
a single astrophysical signal (transit, flare, or artifact).

Pipeline position:
  window composite scores
      -> flag windows above 97.5th-percentile threshold
      -> merge consecutive flagged windows (gap_tolerance=2, max_event_windows=20)
      -> score each event using top-3-mean of window composite scores
      -> return list of CandidateEvent objects
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvent:
    """
    A candidate astrophysical event identified by the anomaly ranking pipeline.

    Scores are in the higher-is-more-anomalous convention throughout.
    p_value is None until conformal calibration is applied.
    """
    star_id: str
    start_time: float           # BKJD of the first flagged window's start
    end_time: float             # BKJD of the last flagged window's end
    center_time: float          # midpoint of start_time and end_time
    window_indices: List[int]   # indices into the star's window array
    n_windows: int              # number of windows in the event

    # Per-model peak window scores (higher = more anomalous)
    model_scores: Dict[str, float] = field(default_factory=dict)

    # Aggregated composite score (top-3-mean of window composites)
    composite_score: float = 0.0

    # TLS event-consistency features (populated later by tls_features.py)
    tls_epoch_distance: Optional[float] = None
    tls_phase_agreement: Optional[float] = None
    tls_depth_ratio: Optional[float] = None

    # Final ranking score (composite + TLS blend, populated by pipeline.py)
    ranking_score: Optional[float] = None

    # Conformal p-value (populated by conformal.py)
    p_value: Optional[float] = None

    def is_calibrated(self) -> bool:
        return self.p_value is not None


def generate_candidates(
    composite_scores: np.ndarray,
    metadata: List[Dict],
    threshold: float,
    per_model_scores: Optional[Dict[str, np.ndarray]] = None,
    gap_tolerance: int = 2,
    max_event_windows: int = 20,
    event_score_method: str = 'top3_mean',
) -> List[CandidateEvent]:
    """
    Generate candidate events from per-window composite scores.

    Steps:
      1. Flag windows with composite_score >= threshold
      2. Group windows by star_id and merge contiguous flagged runs
         (gaps of <= gap_tolerance unflagged windows are bridged)
      3. Drop merged runs longer than max_event_windows
      4. Score each event using the specified aggregation method
      5. Return one CandidateEvent per run

    Args:
        composite_scores: array of shape (n_windows,), higher = more anomalous.
                          Must be aligned with metadata list.
        metadata: list of window metadata dicts, each containing:
                  star_id, window_idx, start_time, end_time, center_time
        threshold: flagging threshold (97.5th percentile of training composites)
        per_model_scores: optional dict {model_name: score_array} for storing
                          per-model peak scores on the event
        gap_tolerance: max unflagged windows allowed inside an event
        max_event_windows: events longer than this are discarded
        event_score_method: one of
            'top3_mean' (default) — mean of top 3 window scores
            'max'                 — peak window score
            'length_penalized'    — max * min(1, 3 / n_windows)

    Returns:
        List of CandidateEvent objects, unsorted.
    """
    composite_scores = np.asarray(composite_scores, dtype=float)
    assert len(composite_scores) == len(metadata), (
        f"composite_scores length {len(composite_scores)} != metadata length {len(metadata)}"
    )

    flagged = composite_scores >= threshold

    # --- Group by star, then merge within each star ---
    # Build a mapping: star_id -> list of (window_position_in_global_array, window_metadata)
    from collections import defaultdict
    star_windows: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    for i, (meta, flag) in enumerate(zip(metadata, flagged)):
        star_windows[meta['star_id']].append((i, meta, flag))

    candidates: List[CandidateEvent] = []

    for star_id, win_list in star_windows.items():
        # win_list is already in order (metadata was built sequentially per star)
        candidates.extend(
            _merge_star_windows(
                star_id=star_id,
                win_list=win_list,
                composite_scores=composite_scores,
                per_model_scores=per_model_scores,
                gap_tolerance=gap_tolerance,
                max_event_windows=max_event_windows,
                event_score_method=event_score_method,
            )
        )

    logger.info(
        f"Generated {len(candidates)} candidate events across "
        f"{len(star_windows)} stars (threshold={threshold:.4f})"
    )
    return candidates


def _merge_star_windows(
    star_id: str,
    win_list: List[Tuple[int, Dict, bool]],
    composite_scores: np.ndarray,
    per_model_scores: Optional[Dict[str, np.ndarray]],
    gap_tolerance: int,
    max_event_windows: int,
    event_score_method: str,
) -> List[CandidateEvent]:
    """Merge consecutive flagged windows for a single star."""
    candidates: List[CandidateEvent] = []
    if not win_list:
        return candidates

    # Current run state
    in_run = False
    run_global_indices: List[int] = []     # indices into global arrays
    run_window_indices: List[int] = []     # window_idx within this star
    run_start_times: List[float] = []
    run_end_times: List[float] = []
    gap_count = 0

    def flush_run():
        nonlocal run_global_indices, run_window_indices, run_start_times, run_end_times
        if not run_global_indices:
            return
        n_win = len(run_global_indices)

        # Split oversized runs into max_event_windows-size chunks rather than
        # discarding them — prevents whole-star blackout when threshold is low.
        if n_win > max_event_windows:
            for chunk_start in range(0, n_win, max_event_windows):
                chunk_end = min(chunk_start + max_event_windows, n_win)
                gi = run_global_indices[chunk_start:chunk_end]
                wi = run_window_indices[chunk_start:chunk_end]
                st = run_start_times[chunk_start:chunk_end]
                et = run_end_times[chunk_start:chunk_end]
                win_composites = composite_scores[gi]
                ev_score = _aggregate_score(win_composites, event_score_method)
                model_scores: Dict[str, float] = {}
                if per_model_scores:
                    for model_name, arr in per_model_scores.items():
                        model_scores[model_name] = float(np.max(arr[gi]))
                candidates.append(CandidateEvent(
                    star_id=star_id,
                    start_time=min(st), end_time=max(et),
                    center_time=(min(st) + max(et)) / 2.0,
                    window_indices=list(wi), n_windows=len(gi),
                    model_scores=model_scores, composite_score=ev_score,
                ))
            run_global_indices = []
            run_window_indices = []
            run_start_times = []
            run_end_times = []
            return

        win_composites = composite_scores[run_global_indices]
        ev_score = _aggregate_score(win_composites, event_score_method)

        # Per-model peak scores
        model_scores: Dict[str, float] = {}
        if per_model_scores:
            for model_name, arr in per_model_scores.items():
                model_scores[model_name] = float(np.max(arr[run_global_indices]))

        start_time = min(run_start_times)
        end_time = max(run_end_times)

        candidates.append(CandidateEvent(
            star_id=star_id,
            start_time=start_time,
            end_time=end_time,
            center_time=(start_time + end_time) / 2.0,
            window_indices=list(run_window_indices),
            n_windows=n_win,
            model_scores=model_scores,
            composite_score=ev_score,
        ))
        run_global_indices = []
        run_window_indices = []
        run_start_times = []
        run_end_times = []

    for global_idx, meta, is_flagged in win_list:
        if is_flagged:
            in_run = True
            gap_count = 0
            run_global_indices.append(global_idx)
            run_window_indices.append(meta['window_idx'])
            run_start_times.append(meta['start_time'])
            run_end_times.append(meta['end_time'])
        elif in_run:
            gap_count += 1
            if gap_count <= gap_tolerance:
                # Bridge the gap — include this unflagged window in the run
                run_global_indices.append(global_idx)
                run_window_indices.append(meta['window_idx'])
                run_start_times.append(meta['start_time'])
                run_end_times.append(meta['end_time'])
            else:
                # Gap too large — close the current run, start fresh
                flush_run()
                in_run = False
                gap_count = 0

    flush_run()  # close any open run at end of star
    return candidates


def _aggregate_score(
    win_composites: np.ndarray,
    method: str,
) -> float:
    """
    Aggregate window-level composite scores into a single event score.

    Methods:
        top3_mean       — mean of top-3 window scores (default)
        max             — peak window score
        length_penalized — max * min(1, 3 / n_windows)
    """
    n = len(win_composites)
    if n == 0:
        return 0.0

    if method == 'max':
        return float(np.max(win_composites))

    if method == 'top3_mean':
        k = min(3, n)
        top_k = np.partition(win_composites, -k)[-k:]
        return float(np.mean(top_k))

    if method == 'length_penalized':
        peak = float(np.max(win_composites))
        return peak * min(1.0, 3.0 / n)

    raise ValueError(f"Unknown event_score_method: '{method}'. "
                     f"Choose from 'top3_mean', 'max', 'length_penalized'.")


def sort_candidates(candidates: List[CandidateEvent]) -> List[CandidateEvent]:
    """Return candidates sorted by ranking_score descending (fallback: composite)."""
    return sorted(
        candidates,
        key=lambda c: (
            c.ranking_score if c.ranking_score is not None else c.composite_score
        ),
        reverse=True,
    )


def candidates_to_records(candidates: List[CandidateEvent]) -> List[Dict]:
    """Convert candidate list to list of plain dicts (for serialisation)."""
    return [
        {
            'star_id': c.star_id,
            'start_time': c.start_time,
            'end_time': c.end_time,
            'center_time': c.center_time,
            'n_windows': c.n_windows,
            'composite_score': c.composite_score,
            'ranking_score': c.ranking_score,
            'p_value': c.p_value,
            'tls_epoch_distance': c.tls_epoch_distance,
            'tls_phase_agreement': c.tls_phase_agreement,
            'tls_depth_ratio': c.tls_depth_ratio,
            **{f'model_{k}': v for k, v in c.model_scores.items()},
        }
        for c in candidates
    ]
