"""
data/sabr.py
--------------
SABR stochastic volatility model:
  - Parameter sampler (financially realistic bounds)
  - Monte Carlo pricer using Euler-Maruyama discretization for the forward price
    and exact geometric Brownian motion for the volatility process.
  - European call/put pricing via risk-neutral MC

SABR dynamics (on forward price F):
  dF = alpha * F^beta * dW_1
  dalpha = nu * alpha * dW_2
  dW_1 * dW_2 = rho * dt

Author: Abdellah Kahlaoui
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SabrParams:
    alpha0: float   # initial volatility
    beta: float     # CEV parameter (elasticity)
    nu: float       # vol of vol
    rho: float      # correlation

    def to_array(self) -> np.ndarray:
        return np.array([self.alpha0, self.beta, self.nu, self.rho])

    @staticmethod
    def from_array(arr: np.ndarray) -> "SabrParams":
        return SabrParams(*arr)


# ---------------------------------------------------------------------------
# Parameter bounds (financially realistic)
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    "alpha0": (0.01, 1.00),
    "beta":   (0.10, 1.00),
    "nu":     (0.10, 2.00),
    "rho":    (-0.95, 0.95),
}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_params(
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> list[SabrParams]:
    """
    Sample SABR parameters uniformly within financially realistic bounds.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to generate.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[SabrParams]
        List of sampled parameter sets.
    """
    rng = np.random.default_rng(seed)
    
    alpha0 = rng.uniform(*PARAM_BOUNDS["alpha0"], size=n_samples)
    beta   = rng.uniform(*PARAM_BOUNDS["beta"],   size=n_samples)
    nu     = rng.uniform(*PARAM_BOUNDS["nu"],     size=n_samples)
    rho    = rng.uniform(*PARAM_BOUNDS["rho"],    size=n_samples)

    return [SabrParams(a, b, n, r) for a, b, n, r in zip(alpha0, beta, nu, rho)]


# ---------------------------------------------------------------------------
# Euler-Maruyama Monte Carlo pricer
# ---------------------------------------------------------------------------

def sabr_mc_pricer(
    S0: float,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    params: SabrParams,
    n_paths: int = 100_000,
    n_steps: int = 100,
    option_type: str = "call",
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Price European options under the SABR model via Monte Carlo.

    The volatility process alpha is simulated exactly as a Geometric Brownian Motion.
    The forward price process F is simulated using an Euler-Maruyama scheme with
    absorption at 0 (to handle beta < 1).

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
    params : SabrParams
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

    alpha0, beta, nu, rho = (
        params.alpha0, params.beta, params.nu, params.rho
    )

    # Cholesky decomposition for correlated Brownians
    # dW_alpha = dZ1
    # dW_F = rho * dZ1 + sqrt(1 - rho^2) * dZ2
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho ** 2)

    n_base = n_paths // 2 if antithetic else n_paths

    prices = np.zeros((len(K_arr), len(T_arr)))

    for j, Tj in enumerate(T_arr):
        if Tj <= 0:
            # Option expired
            for i, Ki in enumerate(K_arr):
                if option_type == "call":
                    prices[i, j] = max(S0 - Ki, 0.0)
                else:
                    prices[i, j] = max(Ki - S0, 0.0)
            continue

        dt = Tj / n_steps
        sqrt_dt = np.sqrt(dt)

        # Forward price F(0) = S0 * exp(r * T)
        F0 = S0 * np.exp(r * Tj)
        
        F = np.full(n_base, F0)
        alpha = np.full(n_base, alpha0)

        if antithetic:
            F_anti = np.full(n_base, F0)
            alpha_anti = np.full(n_base, alpha0)

        for _ in range(n_steps):
            Z1 = rng.standard_normal(n_base)
            Z2 = rng.standard_normal(n_base)

            dW_alpha = Z1 * sqrt_dt
            dW_F = (rho * Z1 + sqrt_one_minus_rho2 * Z2) * sqrt_dt

            # Alpha is a GBM, simulated exactly
            alpha = alpha * np.exp(-0.5 * nu**2 * dt + nu * dW_alpha)

            # F uses Euler with absorption at 0
            F_pos = np.maximum(F, 0.0)
            F = F + alpha * (F_pos ** beta) * dW_F
            F = np.maximum(F, 0.0)

            if antithetic:
                dW_alpha_a = -dW_alpha
                dW_F_a = -dW_F
                
                alpha_anti = alpha_anti * np.exp(-0.5 * nu**2 * dt + nu * dW_alpha_a)
                
                F_pos_a = np.maximum(F_anti, 0.0)
                F_anti = F_anti + alpha_anti * (F_pos_a ** beta) * dW_F_a
                F_anti = np.maximum(F_anti, 0.0)

        discount = np.exp(-r * Tj)

        for i, Ki in enumerate(K_arr):
            if option_type == "call":
                payoff = np.maximum(F - Ki, 0.0)
                if antithetic:
                    payoff_anti = np.maximum(F_anti - Ki, 0.0)
            else:  # put
                payoff = np.maximum(Ki - F, 0.0)
                if antithetic:
                    payoff_anti = np.maximum(Ki - F_anti, 0.0)

            if antithetic:
                avg_payoff = 0.5 * (payoff.mean() + payoff_anti.mean())
            else:
                avg_payoff = payoff.mean()

            prices[i, j] = discount * avg_payoff

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
    params: SabrParams,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Price a full grid of (strike, maturity) pairs under SABR.

    Returns
    -------
    np.ndarray, shape (len(strikes), len(maturities))
        Call prices on the grid.
    """
    return sabr_mc_pricer(
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
    print("=== SABR MC Pricer — smoke test ===\n")

    # Sample a few parameter sets
    params_list = sample_params(n_samples=5, seed=42)
    for i, p in enumerate(params_list):
        print(
            f"  [{i}] alpha0={p.alpha0:.3f}  beta={p.beta:.3f}  nu={p.nu:.3f}"
            f"  rho={p.rho:.3f}"
        )

    print()

    # Price a single option
    p = params_list[0]
    price = sabr_mc_pricer(
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
