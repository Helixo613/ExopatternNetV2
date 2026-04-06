"""
Event-level evaluation for transit candidate ranking.

Point-level metrics (F1, accuracy) are meaningless for transit detection:
a transit spanning 4 out of 35,000 cadences makes a "predict all normal"
classifier achieve 99.99% accuracy.

This module implements event-level evaluation where the unit of measurement
is a transit event, not a cadence.

Ground truth events are derived from NASA ephemerides (period, epoch, duration)
stored in data/labeled/metadata.csv.

Matching rule (tightened from "any 1-cadence overlap"):
  A candidate event DETECTS a ground truth event if:
    (a) the candidate center time falls within the buffered GT interval, OR
    (b) the time overlap between candidate and GT exceeds 25% of GT duration.
  One-to-one: each GT event is claimed by at most one candidate (highest score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from backend.ml.events import CandidateEvent

logger = logging.getLogger(__name__)

# Buffer added to each side of a ground truth transit window (in days)
# Kepler long-cadence = 30 min = 0.0208 days; 2 cadences = 0.0417 days
GT_BUFFER_DAYS = 0.0417


@dataclass
class GroundTruthEvent:
    """A single known transit event derived from NASA ephemeris."""
    star_id: str
    transit_number: int       # 0-indexed transit within the observation window
    center_time: float        # predicted mid-transit time (BKJD)
    start_time: float         # buffered start (center - half_dur - buffer)
    end_time: float           # buffered end   (center + half_dur + buffer)
    duration_days: float      # full transit duration (hours / 24)


# ---------------------------------------------------------------------------
# Ground truth event computation
# ---------------------------------------------------------------------------

def compute_ground_truth_events(
    star_id: str,
    time: np.ndarray,
    period: float,
    epoch: float,
    duration_hours: float,
    buffer_days: float = GT_BUFFER_DAYS,
) -> List[GroundTruthEvent]:
    """
    Enumerate all transit events of a known planet within a light curve's
    time range, using NASA ephemeris parameters.

    Args:
        star_id: star identifier
        time: time array of the light curve (BKJD)
        period: orbital period in days
        epoch: reference transit mid-time in BJD (may be outside Kepler range)
        duration_hours: full transit duration in hours
        buffer_days: extra padding added to each side of the transit window

    Returns:
        List of GroundTruthEvent objects covering the observed time range
    """
    half_dur = (duration_hours / 24.0) / 2.0
    t_min, t_max = float(time.min()), float(time.max())

    # Convert epoch to BKJD if it looks like full BJD (> 2.4e6)
    # Kepler BKJD = BJD - 2454833.0
    if epoch > 1e6:
        epoch_bkjd = epoch - 2454833.0
    else:
        epoch_bkjd = epoch

    # Find the first transit after t_min
    n_start = int(np.ceil((t_min - epoch_bkjd - half_dur) / period))
    n_end   = int(np.floor((t_max - epoch_bkjd + half_dur) / period))

    events: List[GroundTruthEvent] = []
    for n in range(n_start, n_end + 1):
        center = epoch_bkjd + n * period
        start  = center - half_dur - buffer_days
        end    = center + half_dur + buffer_days
        # Only include if the window overlaps the observation range
        if end >= t_min and start <= t_max:
            events.append(GroundTruthEvent(
                star_id=star_id,
                transit_number=n - n_start,
                center_time=center,
                start_time=start,
                end_time=end,
                duration_days=duration_hours / 24.0,
            ))

    return events


def load_ground_truth_events(
    metadata_csv: str,
    lightcurve_dir: str,
    buffer_days: float = GT_BUFFER_DAYS,
) -> Dict[str, List[GroundTruthEvent]]:
    """
    Load ground truth events for all stars in metadata.csv.

    Returns:
        dict mapping star_id -> list of GroundTruthEvent
    """
    meta = pd.read_csv(metadata_csv)
    gt_events: Dict[str, List[GroundTruthEvent]] = {}

    for _, row in meta.iterrows():
        star_id = str(row['target_id'])
        lc_path = f"{lightcurve_dir}/{row['filename']}"
        try:
            df = pd.read_csv(lc_path)
            events = compute_ground_truth_events(
                star_id=star_id,
                time=df['time'].values,
                period=float(row['period']),
                epoch=float(row['epoch']),
                duration_hours=float(row['duration_hours']),
                buffer_days=buffer_days,
            )
            gt_events[star_id] = events
            logger.debug(f"{star_id}: {len(events)} ground truth transit events")
        except Exception as e:
            logger.warning(f"Could not load ground truth for {star_id}: {e}")
            gt_events[star_id] = []

    return gt_events


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------

def match_events(
    candidates: List[CandidateEvent],
    gt_events: List[GroundTruthEvent],
    overlap_fraction: float = 0.25,
) -> Tuple[List[CandidateEvent], List[CandidateEvent],
           List[GroundTruthEvent], List[GroundTruthEvent]]:
    """
    Match candidate events to ground truth events using the tightened rule:

      A candidate DETECTS a GT event if:
        (a) candidate center falls inside the buffered GT interval, OR
        (b) time overlap > overlap_fraction * GT duration

    One-to-one: each GT event is claimed by at most one candidate (highest
    composite_score). Each candidate can match at most one GT event.

    Args:
        candidates: list of CandidateEvent (all from the same star)
        gt_events: list of GroundTruthEvent (all from the same star)
        overlap_fraction: minimum overlap as fraction of GT duration (default 0.25)

    Returns:
        tp_candidates:   candidates that matched a GT event
        fp_candidates:   candidates that matched no GT event
        detected_gt:     GT events that were matched by at least one candidate
        missed_gt:       GT events that were not matched
    """
    if not gt_events:
        return [], list(candidates), [], []
    if not candidates:
        return [], [], [], list(gt_events)

    # Sort candidates by composite_score descending so highest-scoring
    # candidate gets first claim on each GT event
    sorted_cands = sorted(candidates, key=lambda c: c.composite_score, reverse=True)

    claimed_gt: set = set()     # indices of GT events already claimed
    tp_candidates: List[CandidateEvent] = []
    fp_candidates: List[CandidateEvent] = []

    for cand in sorted_cands:
        matched_gt_idx = _find_matching_gt(cand, gt_events, overlap_fraction)
        if matched_gt_idx is not None and matched_gt_idx not in claimed_gt:
            claimed_gt.add(matched_gt_idx)
            tp_candidates.append(cand)
        else:
            fp_candidates.append(cand)

    detected_gt = [gt_events[i] for i in sorted(claimed_gt)]
    missed_gt   = [gt for i, gt in enumerate(gt_events) if i not in claimed_gt]

    return tp_candidates, fp_candidates, detected_gt, missed_gt


def _find_matching_gt(
    cand: CandidateEvent,
    gt_events: List[GroundTruthEvent],
    overlap_fraction: float,
) -> Optional[int]:
    """
    Return the index of the best-matching GT event for a candidate, or None.

    'Best' = highest overlap (to resolve ambiguous cases where a candidate
    straddles two GT events).
    """
    best_idx: Optional[int] = None
    best_overlap: float = -1.0

    for i, gt in enumerate(gt_events):
        # Rule (a): candidate center inside buffered GT interval
        center_inside = gt.start_time <= cand.center_time <= gt.end_time

        # Rule (b): time overlap > overlap_fraction * GT duration
        overlap = max(
            0.0,
            min(cand.end_time, gt.end_time) - max(cand.start_time, gt.start_time)
        )
        gt_dur = max(gt.end_time - gt.start_time, 1e-9)
        frac_overlap = overlap / gt_dur

        if center_inside or frac_overlap > overlap_fraction:
            if frac_overlap > best_overlap:
                best_overlap = frac_overlap
                best_idx = i

    return best_idx


# ---------------------------------------------------------------------------
# Event-level metrics
# ---------------------------------------------------------------------------

def event_metrics(
    candidates: List[CandidateEvent],
    gt_events: List[GroundTruthEvent],
    k_multiplier: float = 2.0,
    overlap_fraction: float = 0.25,
) -> Dict:
    """
    Compute all event-level metrics.

    Primary metric: Recall@K where K = k_multiplier * n_ground_truth_events.
    This models a realistic review workload (an astronomer inspects twice as
    many candidates as there are known events).

    Args:
        candidates: ALL candidates for the test set (unsorted)
        gt_events:  ALL ground truth events for the test set
        k_multiplier: multiplier for Recall@K (default 2.0)
        overlap_fraction: matching threshold (default 0.25)

    Returns:
        dict of metrics
    """
    n_gt = len(gt_events)
    n_cand = len(candidates)

    if n_gt == 0:
        logger.warning("No ground truth events — all metrics undefined")
        return {
            'event_recall': float('nan'), 'event_precision': float('nan'),
            'event_f1': float('nan'), 'recall_at_k': float('nan'),
            'precision_at_k': float('nan'), 'au_pr': float('nan'),
            'n_gt': 0, 'n_candidates': n_cand,
        }

    tp_cands, fp_cands, detected_gt, missed_gt = match_events(
        candidates, gt_events, overlap_fraction
    )

    n_detected = len(detected_gt)
    n_tp = len(tp_cands)

    recall    = n_detected / n_gt if n_gt > 0 else 0.0
    precision = n_tp / n_cand if n_cand > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Recall@K — sort candidates by score, take top K
    K = int(np.ceil(k_multiplier * n_gt))
    recall_at_k, precision_at_k = _recall_at_k(
        candidates, gt_events, K, overlap_fraction
    )

    # AU-PR (event level) — sweep threshold over all composite scores
    au_pr = _event_au_pr(candidates, gt_events, overlap_fraction)

    return {
        'event_recall': round(recall, 4),
        'event_precision': round(precision, 4),
        'event_f1': round(f1, 4),
        'recall_at_k': round(recall_at_k, 4),
        'precision_at_k': round(precision_at_k, 4),
        'au_pr': round(au_pr, 4),
        'n_gt': n_gt,
        'n_detected': n_detected,
        'n_candidates': n_cand,
        'n_tp': n_tp,
        'K': K,
    }


def _recall_at_k(
    candidates: List[CandidateEvent],
    gt_events: List[GroundTruthEvent],
    K: int,
    overlap_fraction: float,
) -> Tuple[float, float]:
    """Compute Recall@K and Precision@K."""
    if not candidates or not gt_events:
        return 0.0, 0.0

    top_k = sorted(candidates, key=lambda c: c.composite_score, reverse=True)[:K]
    _, _, detected_gt, _ = match_events(top_k, gt_events, overlap_fraction)

    recall_at_k    = len(detected_gt) / len(gt_events)
    precision_at_k = len([c for c in top_k
                           if _find_matching_gt(c, gt_events, overlap_fraction) is not None]) / max(K, 1)
    return recall_at_k, precision_at_k


def _event_au_pr(
    candidates: List[CandidateEvent],
    gt_events: List[GroundTruthEvent],
    overlap_fraction: float,
) -> float:
    """
    Area under the event-level Precision-Recall curve.

    Sweeps the composite score threshold from high to low, adding one
    candidate at a time (ranked by score), and records (precision, recall)
    at each step.
    """
    if not candidates or not gt_events:
        return 0.0

    sorted_cands = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    n_gt = len(gt_events)

    precisions = [1.0]
    recalls    = [0.0]

    for i in range(1, len(sorted_cands) + 1):
        subset = sorted_cands[:i]
        _, _, detected_gt, _ = match_events(subset, gt_events, overlap_fraction)
        n_detected = len(detected_gt)
        n_tp = sum(
            1 for c in subset
            if _find_matching_gt(c, gt_events, overlap_fraction) is not None
        )
        p = n_tp / i
        r = n_detected / n_gt
        precisions.append(p)
        recalls.append(r)

    # Trapezoidal integration
    precisions_arr = np.array(precisions)
    recalls_arr    = np.array(recalls)
    # Sort by recall ascending for proper integration
    order = np.argsort(recalls_arr)
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(trapz(precisions_arr[order], recalls_arr[order]))


def event_pr_curve(
    candidates: List[CandidateEvent],
    gt_events: List[GroundTruthEvent],
    overlap_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the event-level precision-recall curve.

    Returns:
        precisions, recalls, thresholds  (arrays, same convention as sklearn)
    """
    if not candidates or not gt_events:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([])

    sorted_cands = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    n_gt = len(gt_events)

    precisions_list = []
    recalls_list    = []
    thresholds_list = []

    for i in range(1, len(sorted_cands) + 1):
        subset = sorted_cands[:i]
        _, _, detected_gt, _ = match_events(subset, gt_events, overlap_fraction)
        n_tp = sum(
            1 for c in subset
            if _find_matching_gt(c, gt_events, overlap_fraction) is not None
        )
        p = n_tp / i
        r = len(detected_gt) / n_gt
        precisions_list.append(p)
        recalls_list.append(r)
        thresholds_list.append(sorted_cands[i - 1].composite_score)

    return (
        np.array(precisions_list),
        np.array(recalls_list),
        np.array(thresholds_list),
    )


# ---------------------------------------------------------------------------
# Multi-star helpers (for CV)
# ---------------------------------------------------------------------------

def aggregate_event_metrics(per_fold_metrics: List[Dict]) -> Dict:
    """
    Aggregate event-level metrics across CV folds (mean ± std).

    Args:
        per_fold_metrics: list of metric dicts, one per fold

    Returns:
        dict with {metric_mean, metric_std} for each scalar metric
    """
    scalar_keys = [
        'event_recall', 'event_precision', 'event_f1',
        'recall_at_k', 'precision_at_k', 'au_pr',
    ]
    agg: Dict = {}
    for key in scalar_keys:
        vals = [m[key] for m in per_fold_metrics
                if key in m and not np.isnan(m[key])]
        if vals:
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std']  = float(np.std(vals))
        else:
            agg[f'{key}_mean'] = float('nan')
            agg[f'{key}_std']  = float('nan')
    return agg
