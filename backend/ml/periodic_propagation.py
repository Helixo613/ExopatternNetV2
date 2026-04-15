"""
Period-aware event propagation.

After the anomaly detector generates candidates for a test star, this module
uses the TLS-detected period to add "phantom" CandidateEvent objects at all
predicted transit times NOT already covered by a real candidate.

Motivation
----------
The anomaly model flags unusual windows — not every dip.  A 3-day planet that
transits ~500 times in 4 years may only have 50 windows flagged.  If TLS
independently confirmed the period (SDE >= min_sde), we can extrapolate where
the remaining ~450 transits should be and add phantom candidates for each.

Score convention
----------------
Phantom score = min(real_scores) - 1e-6.
Phantoms always rank *below* every real candidate so they cannot pollute
precision@K — they only improve event recall / F1.

Phantom candidates are flagged with `is_phantom=True` stored in model_scores
so downstream code can distinguish them if needed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from backend.ml.events import CandidateEvent

logger = logging.getLogger(__name__)

# Minimum TLS Signal Detection Efficiency to trust the period
DEFAULT_MIN_SDE = 5.0

# A phantom covers this many days either side of the predicted transit center.
# Defaults to half_duration + 0.5 days slack; caller may override.
DEFAULT_COVERAGE_HALF_DAYS = 0.5


def propagate_periodic_candidates(
    candidates: List[CandidateEvent],
    star_id: str,
    tls_feats: Dict[str, float],
    time_min: float,
    time_max: float,
    min_sde: float = DEFAULT_MIN_SDE,
    coverage_half_days: float = DEFAULT_COVERAGE_HALF_DAYS,
) -> List[CandidateEvent]:
    """
    For one star, add phantom candidates at all TLS-predicted transit times not
    already covered by a real candidate.

    Parameters
    ----------
    candidates : existing candidates for this star (may be empty)
    star_id    : star identifier (for populating CandidateEvent.star_id)
    tls_feats  : dict with keys tls_period, tls_t0, tls_sde, tls_duration
    time_min   : earliest time in the star's light curve (BKJD)
    time_max   : latest  time in the star's light curve (BKJD)
    min_sde    : minimum TLS SDE required to trust the period
    coverage_half_days : half-width (days) used when checking existing coverage

    Returns
    -------
    Original candidates + any new phantom CandidateEvent objects.
    If TLS SDE is too low or period/t0 are NaN, returns the original list unchanged.
    """
    period  = tls_feats.get('tls_period', float('nan'))
    t0      = tls_feats.get('tls_t0',     float('nan'))
    sde     = tls_feats.get('tls_sde',    0.0)
    dur_hrs = tls_feats.get('tls_duration', float('nan'))

    # Guard: only propagate if TLS found a credible signal
    if sde < min_sde or np.isnan(period) or np.isnan(t0) or period <= 0:
        return candidates

    # Half-duration in days (used as default coverage radius if given)
    if not np.isnan(dur_hrs) and dur_hrs > 0:
        half_dur = (dur_hrs / 24.0) / 2.0
    else:
        half_dur = 0.0
    radius = max(half_dur + coverage_half_days, coverage_half_days)

    # Convert TLS t0 from full BJD to BKJD if necessary (BKJD = BJD - 2454833)
    BKJD_OFFSET = 2454833.0
    if t0 > BKJD_OFFSET:
        t0 -= BKJD_OFFSET

    # Enumerate all predicted transit centers within [time_min, time_max]
    n_start = int(np.floor((time_min - t0) / period))
    n_end   = int(np.ceil( (time_max - t0) / period))
    predicted_centers = [t0 + n * period for n in range(n_start, n_end + 1)
                         if time_min <= t0 + n * period <= time_max]

    if not predicted_centers:
        return candidates

    # Build set of covered intervals from existing real candidates
    covered: List[tuple] = [(c.start_time, c.end_time) for c in candidates
                             if c.star_id == star_id]

    def _is_covered(tc: float) -> bool:
        """True if transit center tc falls within any existing candidate window."""
        for (cs, ce) in covered:
            if cs - radius <= tc <= ce + radius:
                return True
        return False

    # Phantom score: just below the lowest real candidate score
    real_scores = [
        c.ranking_score if c.ranking_score is not None else c.composite_score
        for c in candidates
    ]
    phantom_base_score = (min(real_scores) - 1e-6) if real_scores else 0.0

    phantoms: List[CandidateEvent] = []
    for tc in predicted_centers:
        if _is_covered(tc):
            continue
        start = tc - radius
        end   = tc + radius
        phantom = CandidateEvent(
            star_id=star_id,
            start_time=start,
            end_time=end,
            center_time=tc,
            window_indices=[],
            n_windows=0,
            model_scores={'is_phantom': 1.0},  # flag for downstream use
            composite_score=phantom_base_score,
            ranking_score=phantom_base_score,
        )
        phantoms.append(phantom)

    if phantoms:
        logger.debug(
            f"  [{star_id}] propagated {len(phantoms)} phantoms "
            f"(period={period:.3f}d, SDE={sde:.2f})"
        )

    return candidates + phantoms


def apply_propagation(
    test_cands: List[CandidateEvent],
    test_ids: List[str],
    tls_cache: Dict[str, Dict],
    dfs: Dict,
    min_sde: float = DEFAULT_MIN_SDE,
    coverage_half_days: float = DEFAULT_COVERAGE_HALF_DAYS,
) -> List[CandidateEvent]:
    """
    Apply period-aware propagation across all test stars.

    For each star, looks up TLS features from tls_cache and the time range
    from dfs (used to bound phantom placement).  Stars with no TLS detection
    (SDE < min_sde) are left unchanged.

    Parameters
    ----------
    test_cands  : all CandidateEvent objects from the test set
    test_ids    : list of star IDs in the test set
    tls_cache   : dict {star_id: {tls_period, tls_t0, tls_sde, tls_duration, ...}}
    dfs         : dict-like {star_id: DataFrame with 'time' column}
    min_sde     : minimum TLS SDE to trigger propagation
    coverage_half_days : half-width used for coverage check

    Returns
    -------
    Augmented candidate list (original + phantoms for qualifying stars).
    """
    from collections import defaultdict

    if tls_cache is None:
        return test_cands

    # Bucket real candidates by star
    cands_by_star: Dict[str, List[CandidateEvent]] = defaultdict(list)
    for cand in test_cands:
        cands_by_star[cand.star_id].append(cand)

    augmented: List[CandidateEvent] = []
    total_phantoms = 0

    for star_id in test_ids:
        star_cands = cands_by_star.get(star_id, [])
        tls_feats  = tls_cache.get(star_id, {})

        # Get time bounds from DataFrame
        df = dfs.get(star_id)
        if df is None or 'time' not in df.columns or tls_feats is None:
            augmented.extend(star_cands)
            continue

        time_arr = df['time'].dropna().values
        if len(time_arr) == 0:
            augmented.extend(star_cands)
            continue

        time_min = float(time_arr.min())
        time_max = float(time_arr.max())

        result = propagate_periodic_candidates(
            candidates=star_cands,
            star_id=star_id,
            tls_feats=tls_feats,
            time_min=time_min,
            time_max=time_max,
            min_sde=min_sde,
            coverage_half_days=coverage_half_days,
        )
        n_new = len(result) - len(star_cands)
        total_phantoms += n_new
        augmented.extend(result)

    if total_phantoms:
        logger.info(
            f"  Propagation: added {total_phantoms} phantom candidates "
            f"across {len(test_ids)} test stars"
        )

    return augmented
