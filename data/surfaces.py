"""
data/surface.py
---------------
Implied volatility surface builder.

Takes Heston MC call prices on a (strike, maturity) grid and inverts them
to implied volatilities using Brent's method on the Black-Scholes formula.

Output: fixed-size flattened IV surface — the feature vector fed to the
neural calibrator.

Default grid: 10 strikes × 8 maturities = 80 features.

Author: Abdellah Kahlaoui
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Optional
from dataclasses import dataclass

from data.heston import HestonParams, price_surface_batch


# ---------------------------------------------------------------------------
# Default grid axes
# ---------------------------------------------------------------------------

# Moneyness (K / S0) — log-spaced around ATM
DEFAULT_MONEYNESS = np.array([0.70, 0.80, 0.85, 0.90, 0.95,
                               1.00, 1.05, 1.10, 1.20, 1.30])

# Maturities in years
DEFAULT_MATURITIES = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00])

GRID_N_STRIKES   = len(DEFAULT_MONEYNESS)   # 10
GRID_N_MATURITIES = len(DEFAULT_MATURITIES) # 8
GRID_SIZE        = GRID_N_STRIKES * GRID_N_MATURITIES  # 80


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_intrinsic(S: float, K: float, T: float, r: float) -> float:
    return max(S * np.exp(-0.0 * T) - K * np.exp(-r * T), 0.0)


# ---------------------------------------------------------------------------
# Single implied vol via Brent
# ---------------------------------------------------------------------------

def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
    tol: float = 1e-8,
) -> Optional[float]:
    """
    Invert a call price to implied volatility using Brent's method.

    Returns None if the price is outside no-arbitrage bounds or the
    root-finder fails to converge.
    """
    intrinsic = _bs_intrinsic(S, K, T, r)
    upper_bound = S  # call price can't exceed spot

    # Arbitrage filter
    if price <= intrinsic or price >= upper_bound:
        return None

    f_lo = _bs_call(S, K, T, r, sigma_lo) - price
    f_hi = _bs_call(S, K, T, r, sigma_hi) - price

    # Price not in [BS(sigma_lo), BS(sigma_hi)] — can't bracket
    if f_lo * f_hi > 0:
        return None

    try:
        iv = brentq(
            lambda sig: _bs_call(S, K, T, r, sig) - price,
            sigma_lo, sigma_hi,
            xtol=tol, rtol=tol, maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Surface builder
# ---------------------------------------------------------------------------

@dataclass
class VolSurface:
    iv_grid: np.ndarray      # shape (n_strikes, n_maturities), may contain NaN
    strikes: np.ndarray
    maturities: np.ndarray
    S0: float
    r: float
    n_missing: int           # number of NaN cells after inversion

    @property
    def moneyness(self) -> np.ndarray:
        return self.strikes / self.S0

    def flatten(self) -> np.ndarray:
        """Return the IV surface as a flat feature vector (row-major)."""
        return self.iv_grid.flatten()

    def is_valid(self, max_missing: int = 0) -> bool:
        """True if the surface has at most `max_missing` NaN cells."""
        return self.n_missing <= max_missing

    def fill_missing(self, method: str = "linear") -> None:
        """
        In-place interpolation of NaN cells.
        method: 'linear' uses 2D linear interpolation via griddata.
        """
        if self.n_missing == 0:
            return
        from scipy.interpolate import griddata
        grid = self.iv_grid
        n_k, n_t = grid.shape
        ki, ti = np.meshgrid(np.arange(n_k), np.arange(n_t), indexing="ij")
        mask_valid   = ~np.isnan(grid)
        mask_missing = np.isnan(grid)
        if mask_valid.sum() < 4:
            # Too few valid points — fill with column mean as fallback
            col_means = np.nanmean(grid, axis=0)
            for j in range(n_t):
                grid[np.isnan(grid[:, j]), j] = col_means[j]
            return
        grid[mask_missing] = griddata(
            points=(ki[mask_valid], ti[mask_valid]),
            values=grid[mask_valid],
            xi=(ki[mask_missing], ti[mask_missing]),
            method=method,
            fill_value=float(np.nanmean(grid)),
        )
        self.iv_grid = grid


def build_surface(
    S0: float,
    r: float,
    params: HestonParams,
    moneyness: np.ndarray = DEFAULT_MONEYNESS,
    maturities: np.ndarray = DEFAULT_MATURITIES,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
    fill_missing: bool = True,
) -> VolSurface:
    """
    Build an implied volatility surface for a given set of Heston parameters.

    Steps:
      1. Compute strikes from moneyness * S0
      2. Price the full grid via Heston MC
      3. Invert each price to IV using Brent on BS

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    params : HestonParams
        Heston model parameters.
    moneyness : np.ndarray
        K/S0 values (default 10-point grid).
    maturities : np.ndarray
        Time-to-maturity values in years (default 8-point grid).
    n_paths : int
        MC paths for pricing.
    n_steps : int
        Time discretisation steps.
    seed : int, optional
        RNG seed.
    fill_missing : bool
        If True, interpolate any NaN cells before returning.

    Returns
    -------
    VolSurface
    """
    strikes = moneyness * S0

    # Price grid — shape (n_strikes, n_maturities)
    price_grid = price_surface_batch(
        S0=S0,
        strikes=strikes,
        maturities=maturities,
        r=r,
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    # Invert to implied vols
    iv_grid = np.full_like(price_grid, np.nan)
    for i, (K, price_row) in enumerate(zip(strikes, price_grid)):
        for j, (T, price) in enumerate(zip(maturities, price_row)):
            iv = implied_vol(price, S0, K, T, r)
            if iv is not None:
                iv_grid[i, j] = iv

    n_missing = int(np.isnan(iv_grid).sum())
    surface = VolSurface(
        iv_grid=iv_grid,
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        n_missing=n_missing,
    )

    if fill_missing and n_missing > 0:
        surface.fill_missing()
        surface.n_missing = int(np.isnan(surface.iv_grid).sum())

    return surface


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_surfaces: int,
    S0: float = 100.0,
    r: float = 0.05,
    moneyness: np.ndarray = DEFAULT_MONEYNESS,
    maturities: np.ndarray = DEFAULT_MATURITIES,
    n_paths: int = 100_000,
    n_steps: int = 100,
    enforce_feller: bool = True,
    max_missing_cells: int = 0,
    base_seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of (IV surface, Heston params) pairs.

    Parameters
    ----------
    n_surfaces : int
        Target number of valid surfaces.
    ...

    Returns
    -------
    X : np.ndarray, shape (n_surfaces, grid_size)
        Flattened IV surfaces — neural net inputs.
    y : np.ndarray, shape (n_surfaces, 5)
        Heston parameter vectors [v0, kappa, theta, sigma, rho] — targets.
    """
    from data.heston import sample_params

    grid_size = len(moneyness) * len(maturities)
    X = np.zeros((n_surfaces, grid_size))
    y = np.zeros((n_surfaces, 5))

    generated = 0
    attempts  = 0

    while generated < n_surfaces:
        attempts += 1
        seed = base_seed + attempts * 997  # deterministic per attempt

        # Sample one parameter set
        params = sample_params(n_samples=1, enforce_feller=enforce_feller,
                               seed=seed)[0]

        surface = build_surface(
            S0=S0, r=r, params=params,
            moneyness=moneyness, maturities=maturities,
            n_paths=n_paths, n_steps=n_steps,
            seed=seed, fill_missing=True,
        )

        if not surface.is_valid(max_missing_cells):
            continue  # discard surfaces with residual NaNs

        X[generated] = surface.flatten()
        y[generated] = params.to_array()
        generated += 1

        if verbose and generated % max(1, n_surfaces // 10) == 0:
            print(f"  Generated {generated}/{n_surfaces} surfaces "
                  f"(attempts: {attempts}, yield: {generated/attempts:.1%})")

    if verbose:
        print(f"\nDone. {n_surfaces} surfaces in {attempts} attempts "
              f"({generated/attempts:.1%} yield).")

    return X, y


# ---------------------------------------------------------------------------
# Normalisation helpers (call before feeding to the neural net)
# ---------------------------------------------------------------------------

def normalize_surfaces(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise IV surfaces (per feature across the dataset).

    Returns (X_norm, mean, std). Pass mean/std from the training set when
    normalising val/test splits.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # avoid div-by-zero
    return (X - mean) / std, mean, std


def normalize_params(
    y: np.ndarray,
    bounds: Optional[dict] = None,
) -> np.ndarray:
    """
    Scale Heston parameters to [0, 1] using known parameter bounds.
    """
    from data.heston import PARAM_BOUNDS
    if bounds is None:
        bounds = PARAM_BOUNDS
    lo = np.array([bounds["v0"][0],    bounds["kappa"][0], bounds["theta"][0],
                   bounds["sigma"][0], bounds["rho"][0]])
    hi = np.array([bounds["v0"][1],    bounds["kappa"][1], bounds["theta"][1],
                   bounds["sigma"][1], bounds["rho"][1]])
    return (y - lo) / (hi - lo)


def denormalize_params(
    y_norm: np.ndarray,
    bounds: Optional[dict] = None,
) -> np.ndarray:
    """Invert normalize_params."""
    from data.heston import PARAM_BOUNDS
    if bounds is None:
        bounds = PARAM_BOUNDS
    lo = np.array([bounds["v0"][0],    bounds["kappa"][0], bounds["theta"][0],
                   bounds["sigma"][0], bounds["rho"][0]])
    hi = np.array([bounds["v0"][1],    bounds["kappa"][1], bounds["theta"][1],
                   bounds["sigma"][1], bounds["rho"][1]])
    return y_norm * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from data.heston import sample_params

    print("=== Vol Surface Builder — smoke test ===\n")

    params = sample_params(n_samples=1, enforce_feller=True, seed=7)[0]
    print(f"Params: v0={params.v0:.3f}  kappa={params.kappa:.3f}  "
          f"theta={params.theta:.3f}  sigma={params.sigma:.3f}  rho={params.rho:.3f}")
    print(f"Feller satisfied: {params.satisfies_feller()}\n")

    t0 = time.time()
    surface = build_surface(
        S0=100.0, r=0.05, params=params,
        n_paths=50_000, n_steps=100, seed=42,
    )
    elapsed = time.time() - t0

    print(f"Surface built in {elapsed:.1f}s")
    print(f"Grid shape:    {surface.iv_grid.shape}")
    print(f"Missing cells: {surface.n_missing}")
    print(f"IV range:      [{np.nanmin(surface.iv_grid):.4f}, "
          f"{np.nanmax(surface.iv_grid):.4f}]")
    print(f"\nIV grid (strikes × maturities):\n")

    header = "         " + "  ".join(f"T={t:.2f}" for t in surface.maturities)
    print(header)
    for i, m in enumerate(surface.moneyness):
        row = f"m={m:.2f}:  " + "  ".join(
            f"{surface.iv_grid[i, j]:.4f}" for j in range(len(surface.maturities))
        )
        print(row)

    flat = surface.flatten()
    print(f"\nFlattened feature vector shape: {flat.shape}")
    print(f"Any NaN: {np.isnan(flat).any()}")