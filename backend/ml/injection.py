"""
Injection-recovery test framework for pipeline completeness characterization.

Generates synthetic transit signals using batman (Kreidberg 2015), injects them
into real Kepler light curves, and runs the full anomaly pipeline to determine
recovery rate as a function of (planet_radius, orbital_period).

Output: 8×8 completeness heatmap in (log P, R_p) space.

Reference: Kreidberg (2015), PASP 127, 1161
           https://www.pasp.org/article/10.1086/683602
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition (matches BLUEPRINT D7)
# ---------------------------------------------------------------------------

# Planet radius grid in R_Earth
RADIUS_GRID_REARTH = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

# Orbital period grid in days
PERIOD_GRID_DAYS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Trials per grid cell
N_TRIALS_PER_CELL = 25

# Physical constants
R_SUN_KM  = 695_700.0     # km
R_EARTH_KM = 6_371.0       # km
R_EARTH_RSUN = R_EARTH_KM / R_SUN_KM

# Limb-darkening coefficients (solar, quadratic law — Claret 2000)
LD_U1 = 0.4
LD_U2 = 0.26


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InjectionParams:
    """Parameters for a single synthetic transit injection."""
    radius_rearth: float    # planet radius in R_Earth
    period_days: float      # orbital period in days
    t0: float               # transit epoch (BKJD)
    impact_b: float         # impact parameter [0, 0.8]
    r_star_rsun: float = 1.0

    @property
    def rp_over_rs(self) -> float:
        """Radius ratio (R_p / R_star) — dimensionless."""
        return (self.radius_rearth * R_EARTH_RSUN) / self.r_star_rsun


@dataclass
class InjectionTrial:
    """Result of a single injection-recovery trial."""
    params: InjectionParams
    star_id: str
    recovered: bool
    n_candidates_generated: int
    # Time of best-matching recovered candidate center (None if not recovered)
    best_match_center: Optional[float] = None
    # Composite score of best-matching candidate (None if not recovered)
    best_match_score: Optional[float] = None


@dataclass
class InjectionRecoveryResult:
    """Aggregated injection-recovery results across all trials."""
    # Completeness matrix: shape (n_radii, n_periods), values in [0, 1]
    completeness: np.ndarray
    radius_grid: List[float]
    period_grid: List[float]
    n_trials: int
    n_recovered: int
    all_trials: List[InjectionTrial] = field(default_factory=list)

    @property
    def overall_recovery_rate(self) -> float:
        return self.n_recovered / max(self.n_trials, 1)


# ---------------------------------------------------------------------------
# Transit model
# ---------------------------------------------------------------------------

def make_transit_model(
    time: np.ndarray,
    params: InjectionParams,
) -> np.ndarray:
    """
    Compute a batman transit light curve model.

    Returns a flux array (same length as time) where 1.0 = out-of-transit,
    values < 1.0 during transit.

    Args:
        time: time array (BKJD)
        params: InjectionParams

    Returns:
        flux_model array, shape same as time
    """
    try:
        import batman

        bparams = batman.TransitParams()
        bparams.t0 = params.t0
        bparams.per = params.period_days
        bparams.rp = params.rp_over_rs
        bparams.a = _semi_major_axis_over_rs(params.period_days, params.r_star_rsun)
        bparams.inc = _inclination_deg(bparams.a, params.impact_b)
        bparams.ecc = 0.0
        bparams.w = 90.0
        bparams.u = [LD_U1, LD_U2]
        bparams.limb_dark = 'quadratic'

        m = batman.TransitModel(bparams, time)
        return m.light_curve(bparams)

    except Exception as e:
        logger.warning(f"batman model failed: {e}. Returning flat (no injection).")
        return np.ones_like(time, dtype=float)


def _semi_major_axis_over_rs(period_days: float, r_star_rsun: float) -> float:
    """
    Compute a/R_star from Kepler's third law (circular orbit, M_star = 1 M_sun).

    a (AU) = (period_years)^(2/3)
    """
    period_years = period_days / 365.25
    a_au = period_years ** (2.0 / 3.0)
    # 1 AU = 214.9 R_sun
    a_rsun = a_au * 214.9
    return float(a_rsun / r_star_rsun)


def _inclination_deg(a_over_rs: float, impact_b: float) -> float:
    """Convert impact parameter b to inclination in degrees."""
    # b = a/R_star * cos(i)
    if a_over_rs < 1.0:
        return 90.0
    cos_i = impact_b / a_over_rs
    cos_i = float(np.clip(cos_i, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_i)))


# ---------------------------------------------------------------------------
# Single injection-recovery trial
# ---------------------------------------------------------------------------

def inject_and_recover(
    df: pd.DataFrame,
    params: InjectionParams,
    pipeline_fn,
    star_id: str = "unknown",
    recovery_window_factor: float = 1.5,
) -> InjectionTrial:
    """
    Inject a synthetic transit into a real light curve and run the pipeline.

    Recovery criterion: at least one candidate center falls within
    `recovery_window_factor * transit_duration` of any injected transit mid-time.

    Args:
        df: light curve DataFrame with columns ['time', 'flux']
        params: injection parameters
        pipeline_fn: callable(df) -> List[CandidateEvent]
                     Runs the full anomaly pipeline on a light curve.
        star_id: identifier for logging
        recovery_window_factor: multiplier on transit duration for recovery window

    Returns:
        InjectionTrial result
    """
    time = np.asarray(df['time'].values, dtype=float)
    flux = np.asarray(df['flux'].values, dtype=float)

    # 1. Compute transit model
    transit_model = make_transit_model(time, params)

    # 2. Inject by multiplication
    flux_injected = flux * transit_model
    df_injected = df.copy()
    df_injected['flux'] = flux_injected

    # 3. Run pipeline
    try:
        candidates = pipeline_fn(df_injected)
    except Exception as e:
        logger.warning(f"Pipeline failed on injection trial for {star_id}: {e}")
        return InjectionTrial(
            params=params, star_id=star_id, recovered=False,
            n_candidates_generated=0,
        )

    n_candidates = len(candidates)

    # 4. Enumerate injected transit mid-times within observation window
    t_min, t_max = float(time.min()), float(time.max())
    transit_times = _injected_transit_times(params, t_min, t_max)

    if not transit_times:
        # No transit falls in the window — uninformative trial
        return InjectionTrial(
            params=params, star_id=star_id, recovered=False,
            n_candidates_generated=n_candidates,
        )

    # Estimate transit duration (days) from batman model via depth width
    approx_duration = _estimate_transit_duration_days(
        time, transit_model, params.t0, params.period_days
    )
    recovery_half_window = recovery_window_factor * approx_duration

    # 5. Check recovery: any candidate center within recovery window of any transit
    best_center: Optional[float] = None
    best_score: Optional[float] = None
    recovered = False

    for cand in candidates:
        for t_mid in transit_times:
            if abs(cand.center_time - t_mid) <= recovery_half_window:
                recovered = True
                cand_score = getattr(cand, 'composite_score', 0.0)
                if best_score is None or cand_score > best_score:
                    best_score = cand_score
                    best_center = cand.center_time

    return InjectionTrial(
        params=params,
        star_id=star_id,
        recovered=recovered,
        n_candidates_generated=n_candidates,
        best_match_center=best_center,
        best_match_score=best_score,
    )


def _injected_transit_times(
    params: InjectionParams,
    t_min: float,
    t_max: float,
) -> List[float]:
    """Return list of transit mid-times within [t_min, t_max]."""
    times = []
    n_start = int(np.floor((t_min - params.t0) / params.period_days))
    n_end   = int(np.ceil( (t_max - params.t0) / params.period_days))
    for n in range(n_start, n_end + 1):
        t = params.t0 + n * params.period_days
        if t_min <= t <= t_max:
            times.append(t)
    return times


def _estimate_transit_duration_days(
    time: np.ndarray,
    transit_model: np.ndarray,
    t0: float,
    period_days: float,
    depth_threshold: float = 0.999,
) -> float:
    """
    Estimate transit duration in days from the batman model array.
    Falls back to 0.1 days if no transit found near t0.
    """
    # Find points in-transit near t0
    in_transit = transit_model < depth_threshold
    if not np.any(in_transit):
        return 0.1

    # Find time extent of the primary transit (within period/2 of t0)
    near_t0 = np.abs(time - t0) < period_days / 2.0
    mask = in_transit & near_t0
    if not np.any(mask):
        return 0.1

    t_in = time[mask]
    return float(t_in.max() - t_in.min()) + (time[1] - time[0])  # add 1 cadence


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_injection_recovery(
    light_curves: Dict[str, pd.DataFrame],
    pipeline_fn,
    radius_grid: Optional[List[float]] = None,
    period_grid: Optional[List[float]] = None,
    n_trials: int = N_TRIALS_PER_CELL,
    rng_seed: int = 42,
    r_star_rsun: float = 1.0,
    recovery_window_factor: float = 1.5,
    progress_callback=None,
) -> InjectionRecoveryResult:
    """
    Run the full 8×8 injection-recovery grid.

    For each (radius, period) cell, injects `n_trials` synthetic transits
    (random t0 and impact parameter) into randomly selected light curves
    from `light_curves`, then recovers using `pipeline_fn`.

    Args:
        light_curves: dict {star_id -> DataFrame with time/flux columns}
        pipeline_fn: callable(df) -> List[CandidateEvent]
        radius_grid: planet radii in R_Earth (default: RADIUS_GRID_REARTH)
        period_grid: orbital periods in days (default: PERIOD_GRID_DAYS)
        n_trials: trials per cell (default 25)
        rng_seed: random seed for reproducibility
        r_star_rsun: assumed stellar radius in R_sun
        recovery_window_factor: multiplier on transit duration for recovery window
        progress_callback: optional callable(completed, total) for progress reporting

    Returns:
        InjectionRecoveryResult
    """
    if radius_grid is None:
        radius_grid = RADIUS_GRID_REARTH
    if period_grid is None:
        period_grid = PERIOD_GRID_DAYS

    rng = np.random.default_rng(rng_seed)
    star_ids = list(light_curves.keys())

    n_r = len(radius_grid)
    n_p = len(period_grid)
    completeness = np.zeros((n_r, n_p), dtype=float)
    all_trials: List[InjectionTrial] = []

    total_trials = n_r * n_p * n_trials
    completed = 0

    logger.info(
        f"Starting injection-recovery: {n_r}×{n_p} grid, "
        f"{n_trials} trials/cell, {total_trials} total"
    )

    for i, radius in enumerate(radius_grid):
        for j, period in enumerate(period_grid):
            n_recovered_cell = 0

            for _ in range(n_trials):
                # Pick a random star
                star_id = rng.choice(star_ids)
                df = light_curves[star_id]
                time = df['time'].values

                t_min, t_max = float(time.min()), float(time.max())

                # Skip if period is longer than the light curve
                if period > (t_max - t_min):
                    all_trials.append(InjectionTrial(
                        params=InjectionParams(
                            radius_rearth=radius, period_days=period,
                            t0=t_min, impact_b=0.0, r_star_rsun=r_star_rsun
                        ),
                        star_id=star_id, recovered=False,
                        n_candidates_generated=0,
                    ))
                    completed += 1
                    continue

                # Random epoch and impact parameter
                t0 = float(rng.uniform(t_min, t_min + period))
                b  = float(rng.uniform(0.0, 0.8))

                params = InjectionParams(
                    radius_rearth=radius,
                    period_days=period,
                    t0=t0,
                    impact_b=b,
                    r_star_rsun=r_star_rsun,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trial = inject_and_recover(
                        df, params, pipeline_fn,
                        star_id=star_id,
                        recovery_window_factor=recovery_window_factor,
                    )

                all_trials.append(trial)
                if trial.recovered:
                    n_recovered_cell += 1

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_trials)

            completeness[i, j] = n_recovered_cell / n_trials
            logger.debug(
                f"  R={radius:.1f} R⊕  P={period:.0f}d: "
                f"{n_recovered_cell}/{n_trials} recovered "
                f"({100*completeness[i,j]:.0f}%)"
            )

    n_total_recovered = sum(t.recovered for t in all_trials)
    logger.info(
        f"Injection-recovery complete: {n_total_recovered}/{len(all_trials)} recovered "
        f"({100*n_total_recovered/max(len(all_trials),1):.1f}%)"
    )

    return InjectionRecoveryResult(
        completeness=completeness,
        radius_grid=list(radius_grid),
        period_grid=list(period_grid),
        n_trials=len(all_trials),
        n_recovered=n_total_recovered,
        all_trials=all_trials,
    )


# ---------------------------------------------------------------------------
# Result analysis helpers
# ---------------------------------------------------------------------------

def completeness_contours(
    result: InjectionRecoveryResult,
    levels: Tuple[float, ...] = (0.5, 0.8, 0.9),
) -> Dict[float, List[Tuple[float, float]]]:
    """
    Return approximate (period, radius) contour points at given completeness levels.

    Uses simple threshold: for each radius row, find the largest period where
    completeness >= level.

    Returns:
        dict {level -> list of (period_days, radius_rearth) points}
    """
    contours: Dict[float, List[Tuple[float, float]]] = {lv: [] for lv in levels}

    for i, radius in enumerate(result.radius_grid):
        for level in levels:
            # Find the largest period index where completeness >= level
            row = result.completeness[i, :]
            idx = np.where(row >= level)[0]
            if len(idx) > 0:
                max_period = result.period_grid[int(idx.max())]
                contours[level].append((max_period, radius))

    return contours


def summarize_injection_recovery(result: InjectionRecoveryResult) -> Dict:
    """Return a summary dict for logging/reporting."""
    summary: Dict = {
        'overall_recovery_rate': round(result.overall_recovery_rate, 4),
        'n_trials': result.n_trials,
        'n_recovered': result.n_recovered,
        'n_radii': len(result.radius_grid),
        'n_periods': len(result.period_grid),
        'completeness_matrix': result.completeness.tolist(),
        'radius_grid': result.radius_grid,
        'period_grid': result.period_grid,
    }

    # Per-radius recovery rates
    for i, r in enumerate(result.radius_grid):
        summary[f'recovery_r{r:.1f}'] = round(float(result.completeness[i].mean()), 4)

    # Per-period recovery rates
    for j, p in enumerate(result.period_grid):
        summary[f'recovery_p{p:.0f}d'] = round(float(result.completeness[:, j].mean()), 4)

    return summary
