"""
data/heston.py
--------------
Heston stochastic volatility model:
  - Parameter sampler (financially realistic bounds)
  - Monte Carlo pricer using Milstein discretization
  - European call/put pricing via risk-neutral MC

Heston dynamics:
  dS = r * S * dt + sqrt(v) * S * dW_S
  dv = kappa * (theta - v) * dt + sigma * sqrt(v) * dW_v
  dW_S * dW_v = rho * dt

Author: Abdellah Kahlaoui
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    v0: float       # initial variance          (0.01 – 0.50)
    kappa: float    # mean-reversion speed       (0.10 – 5.00)
    theta: float    # long-run variance           (0.01 – 0.50)
    sigma: float    # vol-of-vol                  (0.10 – 1.00)
    rho: float      # S-v correlation             (-0.95 – 0.00)

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @staticmethod
    def from_array(arr: np.ndarray) -> "HestonParams":
        return HestonParams(*arr)

    def satisfies_feller(self) -> bool:
        """Feller condition: 2 * kappa * theta > sigma^2  (variance stays > 0)."""
        return 2.0 * self.kappa * self.theta > self.sigma ** 2


# ---------------------------------------------------------------------------
# Parameter bounds (financially realistic)
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    "v0":    (0.01, 0.50),
    "kappa": (0.10, 5.00),
    "theta": (0.01, 0.50),
    "sigma": (0.10, 1.00),
    "rho":   (-0.95, 0.00),
}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_params(
    n_samples: int = 1,
    enforce_feller: bool = True,
    seed: Optional[int] = None,
) -> list[HestonParams]:
    """
    Sample Heston parameters uniformly within financially realistic bounds.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to generate.
    enforce_feller : bool
        If True, resample until Feller condition is met (2κθ > σ²).
        Keeps the variance process strictly positive almost surely.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[HestonParams]
        List of sampled parameter sets.
    """
    rng = np.random.default_rng(seed)
    results: list[HestonParams] = []

    while len(results) < n_samples:
        # How many still needed — sample in batches for efficiency
        remaining = n_samples - len(results)
        batch = max(remaining, 512)

        v0    = rng.uniform(*PARAM_BOUNDS["v0"],    size=batch)
        kappa = rng.uniform(*PARAM_BOUNDS["kappa"], size=batch)
        theta = rng.uniform(*PARAM_BOUNDS["theta"], size=batch)
        sigma = rng.uniform(*PARAM_BOUNDS["sigma"], size=batch)
        rho   = rng.uniform(*PARAM_BOUNDS["rho"],   size=batch)

        if enforce_feller:
            feller_ok = 2.0 * kappa * theta > sigma ** 2
            v0, kappa, theta, sigma, rho = (
                v0[feller_ok], kappa[feller_ok], theta[feller_ok],
                sigma[feller_ok], rho[feller_ok],
            )

        for i in range(min(len(v0), remaining)):
            results.append(HestonParams(v0[i], kappa[i], theta[i], sigma[i], rho[i]))

    return results[:n_samples]


# ---------------------------------------------------------------------------
# Black-Scholes helper (for IV inversion in surface.py)
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Milstein Monte Carlo pricer
# ---------------------------------------------------------------------------

def heston_mc_pricer(
    S0: float,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    params: HestonParams,
    n_paths: int = 100_000,
    n_steps: int = 100,
    option_type: str = "call",
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Price European options under the Heston model via Milstein MC.

    Variance process uses the full truncation scheme to keep v ≥ 0:
        v⁺ = max(v, 0)   (used in drift/diffusion, not stored)

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float or array-like
        Strike price(s).
    T : float or array-like
        Time(s) to maturity (in years).
    r : float
        Risk-free rate (continuous, annualised).
    params : HestonParams
        Model parameters.
    n_paths : int
        Number of Monte Carlo paths. If antithetic=True, half are base paths.
    n_steps : int
        Time discretisation steps per maturity.
    option_type : str
        "call" or "put".
    seed : int, optional
        RNG seed.
    antithetic : bool
        Use antithetic variates to reduce variance (recommended).

    Returns
    -------
    np.ndarray
        Option price(s), shape matching broadcast of K × T inputs.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    rng   = np.random.default_rng(seed)

    v0, kappa, theta, sigma, rho = (
        params.v0, params.kappa, params.theta, params.sigma, params.rho
    )

    # Cholesky decomposition for correlated Brownians
    # dW_v = dZ1
    # dW_S = rho * dZ1 + sqrt(1 - rho^2) * dZ2
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho ** 2)

    n_base = n_paths // 2 if antithetic else n_paths

    prices = np.zeros((len(K_arr), len(T_arr)))

    for j, Tj in enumerate(T_arr):
        dt = Tj / n_steps
        sqrt_dt = np.sqrt(dt)

        # Simulate paths
        S = np.full(n_base, S0)
        v = np.full(n_base, v0)

        if antithetic:
            S_anti = np.full(n_base, S0)
            v_anti = np.full(n_base, v0)

        for _ in range(n_steps):
            Z1 = rng.standard_normal(n_base)
            Z2 = rng.standard_normal(n_base)

            dW_v = Z1 * sqrt_dt
            dW_S = (rho * Z1 + sqrt_one_minus_rho2 * Z2) * sqrt_dt

            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Milstein for variance (CIR):
            # dv = kappa*(theta - v⁺)*dt + sigma*sqrt(v⁺)*dW_v
            #      + 0.5 * sigma^2 * (dW_v^2 - dt)    ← Milstein correction
            v = (
                v
                + kappa * (theta - v_pos) * dt
                + sigma * sqrt_v * dW_v
                + 0.5 * sigma ** 2 * (dW_v ** 2 - dt)
            )
            v = np.maximum(v, 0.0)  # full truncation

            # Log-Euler for asset (exact in log-space):
            # d(ln S) = (r - 0.5*v⁺)*dt + sqrt(v⁺)*dW_S
            S = S * np.exp((r - 0.5 * v_pos) * dt + sqrt_v * dW_S)

            if antithetic:
                dW_v_a = -dW_v
                dW_S_a = -dW_S
                v_pos_a = np.maximum(v_anti, 0.0)
                sqrt_v_a = np.sqrt(v_pos_a)
                v_anti = (
                    v_anti
                    + kappa * (theta - v_pos_a) * dt
                    + sigma * sqrt_v_a * dW_v_a
                    + 0.5 * sigma ** 2 * (dW_v_a ** 2 - dt)
                )
                v_anti = np.maximum(v_anti, 0.0)
                S_anti = S_anti * np.exp(
                    (r - 0.5 * v_pos_a) * dt + sqrt_v_a * dW_S_a
                )

        discount = np.exp(-r * Tj)

        for i, Ki in enumerate(K_arr):
            if option_type == "call":
                payoff = np.maximum(S - Ki, 0.0)
                if antithetic:
                    payoff_anti = np.maximum(S_anti - Ki, 0.0)
            else:  # put
                payoff = np.maximum(Ki - S, 0.0)
                if antithetic:
                    payoff_anti = np.maximum(Ki - S_anti, 0.0)

            if antithetic:
                avg_payoff = 0.5 * (payoff.mean() + payoff_anti.mean())
            else:
                avg_payoff = payoff.mean()

            prices[i, j] = discount * avg_payoff

    # Squeeze scalar output
    if prices.shape == (1, 1):
        return float(prices[0, 0])
    return prices.squeeze()


# ---------------------------------------------------------------------------
# Batch pricer for dataset generation
# ---------------------------------------------------------------------------

def price_surface_batch(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    params: HestonParams,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Price a full grid of (strike, maturity) pairs under Heston.

    Returns
    -------
    np.ndarray, shape (len(strikes), len(maturities))
        Call prices on the grid.
    """
    return heston_mc_pricer(
        S0=S0,
        K=strikes,
        T=maturities,
        r=r,
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        option_type="call",
        seed=seed,
        antithetic=True,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Heston MC Pricer — smoke test ===\n")

    # Sample a few parameter sets
    params_list = sample_params(n_samples=5, enforce_feller=True, seed=42)
    for i, p in enumerate(params_list):
        feller = "✓" if p.satisfies_feller() else "✗"
        print(
            f"  [{i}] v0={p.v0:.3f}  kappa={p.kappa:.3f}  theta={p.theta:.3f}"
            f"  sigma={p.sigma:.3f}  rho={p.rho:.3f}  Feller={feller}"
        )

    print()

    # Price a single option
    p = params_list[0]
    price = heston_mc_pricer(
        S0=100.0, K=100.0, T=1.0, r=0.05,
        params=p, n_paths=200_000, n_steps=200, seed=0,
    )
    print(f"ATM call (S=K=100, T=1, r=5%): {price:.4f}")

    # Price a surface grid
    strikes    = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
    maturities = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    grid = price_surface_batch(
        S0=100.0, strikes=strikes, maturities=maturities,
        r=0.05, params=p, n_paths=50_000, n_steps=100, seed=1,
    )
    print(f"\nCall price grid (strikes × maturities) shape: {grid.shape}")
    print(np.round(grid, 4))
