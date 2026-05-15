"""
calibration/lm_optimizer.py
----------------------------
Classical Heston calibration via Levenberg-Marquardt.

Minimises the sum of squared IV residuals between a target vol surface
and the surface implied by candidate Heston parameters.

Used as the accuracy / speed benchmark against the neural calibrator.

Author: Abdellah Kahlaoui
"""

import time
import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import Optional

try:
    from data.heston import HestonParams, PARAM_BOUNDS, price_surface_batch
    from data.surfaces import (
        VolSurface, DEFAULT_MONEYNESS, DEFAULT_MATURITIES,
        implied_vol, build_surface,
    )
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.heston import HestonParams, PARAM_BOUNDS, price_surface_batch
    from data.surfaces import (
        VolSurface, DEFAULT_MONEYNESS, DEFAULT_MATURITIES,
        implied_vol, build_surface,
    )


# ---------------------------------------------------------------------------
# Parameter bounds for LM (same as sampler)
# ---------------------------------------------------------------------------

LB = np.array([PARAM_BOUNDS["v0"][0],    PARAM_BOUNDS["kappa"][0],
               PARAM_BOUNDS["theta"][0], PARAM_BOUNDS["sigma"][0],
               PARAM_BOUNDS["rho"][0]])

UB = np.array([PARAM_BOUNDS["v0"][1],    PARAM_BOUNDS["kappa"][1],
               PARAM_BOUNDS["theta"][1], PARAM_BOUNDS["sigma"][1],
               PARAM_BOUNDS["rho"][1]])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    params:        HestonParams
    rmse:          float          # IV RMSE (in vol units, e.g. 0.01 = 1 vol pt)
    n_iter:        int
    n_evals:       int
    elapsed_ms:    float
    success:       bool
    message:       str

    def to_array(self) -> np.ndarray:
        return self.params.to_array()


# ---------------------------------------------------------------------------
# Core residual function
# ---------------------------------------------------------------------------

def _iv_residuals(
    x: np.ndarray,
    target_iv: np.ndarray,          # shape (n_strikes * n_maturities,)
    S0: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Given parameter vector x = [v0, kappa, theta, sigma, rho],
    price a surface, extract IVs, return (model_iv - target_iv).
    """
    params = HestonParams(*x)

    surface = build_surface(
        S0=S0, r=r, params=params,
        moneyness=strikes / S0,
        maturities=maturities,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        fill_missing=True,
    )
    model_iv = surface.flatten()

    # Replace any residual NaN with a large penalty
    model_iv = np.where(np.isnan(model_iv), 2.0, model_iv)

    return model_iv - target_iv


# ---------------------------------------------------------------------------
# LM calibrator
# ---------------------------------------------------------------------------

def calibrate_lm(
    target_surface: VolSurface,
    S0: float,
    r: float,
    x0: Optional[np.ndarray] = None,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-6,
    max_nfev: int = 200,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Calibrate Heston parameters to a target IV surface using
    Levenberg-Marquardt via scipy.optimize.least_squares.

    Parameters
    ----------
    target_surface : VolSurface
        The observed / reference implied vol surface to fit.
    S0 : float
        Spot price used to generate the target surface.
    r : float
        Risk-free rate.
    x0 : np.ndarray, shape (5,), optional
        Initial parameter guess [v0, kappa, theta, sigma, rho].
        Defaults to the midpoint of each parameter range.
    n_paths : int
        MC paths per function evaluation. Lower = faster but noisier.
        Recommended: 20k–50k for LM (accuracy traded for speed).
    n_steps : int
        Time steps per maturity.
    seed : int, optional
        Fixed seed for MC (keeps residuals deterministic across evals).
    ftol, xtol, gtol : float
        scipy convergence tolerances.
    max_nfev : int
        Maximum number of function evaluations.
    verbose : bool
        Print iteration info.

    Returns
    -------
    CalibrationResult
    """
    if x0 is None:
        x0 = 0.5 * (LB + UB)  # midpoint of all ranges

    target_iv = target_surface.flatten()
    # Replace NaN in target with mean (shouldn't happen after fill_missing)
    target_iv = np.where(np.isnan(target_iv), np.nanmean(target_iv), target_iv)

    t_start = time.perf_counter()

    result = least_squares(
        fun=_iv_residuals,
        x0=x0,
        bounds=(LB, UB),
        method="trf",          # Trust Region Reflective — handles bounds;
                               # pure LM (method='lm') doesn't support bounds
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
        kwargs=dict(
            target_iv=target_iv,
            S0=S0,
            r=r,
            strikes=target_surface.strikes,
            maturities=target_surface.maturities,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        ),
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    params = HestonParams(*result.x)
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))

    return CalibrationResult(
        params=params,
        rmse=rmse,
        n_iter=result.njev,
        n_evals=result.nfev,
        elapsed_ms=elapsed_ms,
        success=result.success,
        message=result.message,
    )


# ---------------------------------------------------------------------------
# Multi-start wrapper (improves robustness against local minima)
# ---------------------------------------------------------------------------

def calibrate_lm_multistart(
    target_surface: VolSurface,
    S0: float,
    r: float,
    n_starts: int = 5,
    n_paths: int = 30_000,
    n_steps: int = 50,
    seed: int = 42,
    **lm_kwargs,
) -> CalibrationResult:
    """
    Run LM from multiple random starting points; return best result (lowest RMSE).

    Useful when benchmarking: gives the classical optimizer its best shot.
    """
    rng = np.random.default_rng(seed)
    best: Optional[CalibrationResult] = None

    for i in range(n_starts):
        x0 = rng.uniform(LB, UB)
        res = calibrate_lm(
            target_surface=target_surface,
            S0=S0, r=r, x0=x0,
            n_paths=n_paths, n_steps=n_steps,
            seed=seed + i,
            **lm_kwargs,
        )
        if best is None or res.rmse < best.rmse:
            best = res
        if best.rmse < 1e-4:
            break  # good enough — stop early

    return best


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.heston import sample_params

    print("=== LM Calibrator — smoke test ===\n")

    # Ground-truth params
    true_params = sample_params(n_samples=1, enforce_feller=True, seed=99)[0]
    print(f"True params: {true_params}\n")

    # Build target surface
    print("Building target surface...")
    target = build_surface(
        S0=100.0, r=0.05, params=true_params,
        n_paths=50_000, n_steps=100, seed=0,
    )
    print(f"Surface built. Missing cells: {target.n_missing}\n")

    # Calibrate
    print("Running LM calibration (this takes a while)...")
    result = calibrate_lm(
        target_surface=target,
        S0=100.0, r=0.05,
        n_paths=20_000, n_steps=50,
        seed=1, max_nfev=100,
    )

    print(f"\n{'Param':>6}  {'True':>8}  {'Estimated':>10}  {'Error':>8}")
    print("-" * 40)
    true_arr = true_params.to_array()
    est_arr  = result.params.to_array()
    for name, t, e in zip(["v0","kappa","theta","sigma","rho"], true_arr, est_arr):
        print(f"{name:>6}  {t:>8.4f}  {e:>10.4f}  {abs(t-e):>8.4f}")

    print(f"\nRMSE (IV): {result.rmse:.6f}")
    print(f"Evals:     {result.n_evals}")
    print(f"Time:      {result.elapsed_ms:.0f} ms")
    print(f"Success:   {result.success}")