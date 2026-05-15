"""
evaluation/metrics.py
---------------------
Evaluation metrics and Plotly visualisations for the neural calibrator
vs the LM benchmark.

Covers:
  - Per-parameter RMSE / MAE / MaxError
  - Speed comparison table
  - 3D IV surface overlay (true vs predicted)
  - Parameter error distribution plots
  - Calibration scatter plots (true vs predicted per param)

Author: Abdellah Kahlaoui
"""

import numpy as np
import pandas as pd
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PARAM_NAMES = ["v0", "kappa", "theta", "sigma", "rho"]
PARAM_LABELS = {
    "v0":    "v₀  (initial variance)",
    "kappa": "κ   (mean reversion)",
    "theta": "θ   (long-run variance)",
    "sigma": "σ   (vol of vol)",
    "rho":   "ρ   (correlation)",
}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] = PARAM_NAMES,
) -> pd.DataFrame:
    """
    Compute per-parameter RMSE, MAE, and MaxError.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, n_params)  — real-world parameter values
    y_pred : np.ndarray, shape (N, n_params)

    Returns
    -------
    pd.DataFrame with columns [param, RMSE, MAE, MaxError, R2]
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch"
    errors = y_pred - y_true
    rows = []
    for i, name in enumerate(param_names):
        e = errors[:, i]
        t = y_true[:, i]
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({
            "param":    name,
            "RMSE":     float(np.sqrt(np.mean(e ** 2))),
            "MAE":      float(np.mean(np.abs(e))),
            "MaxError": float(np.max(np.abs(e))),
            "R2":       float(r2),
        })
    return pd.DataFrame(rows).set_index("param")


def compare_speed(
    neural_times_ms: list[float],
    lm_times_ms: list[float],
) -> pd.DataFrame:
    """
    Summarise inference speed of neural vs LM calibrator.

    Parameters
    ----------
    neural_times_ms : list of per-surface neural inference times (ms)
    lm_times_ms     : list of per-surface LM calibration times (ms)

    Returns
    -------
    pd.DataFrame
    """
    def _stats(times):
        t = np.array(times)
        return {"mean_ms": t.mean(), "median_ms": np.median(t),
                "p95_ms": np.percentile(t, 95), "min_ms": t.min()}

    ns = _stats(neural_times_ms)
    ls = _stats(lm_times_ms)
    speedup = ls["mean_ms"] / ns["mean_ms"]

    df = pd.DataFrame([
        {"method": "Neural (ours)", **ns},
        {"method": "LM baseline",  **ls},
    ]).set_index("method")
    df.loc["Speedup (×)", :] = [speedup, speedup, speedup, speedup]
    return df


def print_metrics(df: pd.DataFrame, title: str = "Calibration Metrics") -> None:
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")
    print(df.to_string(float_format=lambda x: f"{x:.6f}"))
    print()


# ---------------------------------------------------------------------------
# 3D surface overlay
# ---------------------------------------------------------------------------

def plot_surface_overlay(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_true: np.ndarray,
    iv_pred: np.ndarray,
    title: str = "IV Surface: True vs Predicted",
    show: bool = True,
) -> go.Figure:
    """
    3D Plotly surface overlay — true (blue) vs predicted/calibrated (red).

    Parameters
    ----------
    strikes    : shape (n_strikes,)
    maturities : shape (n_maturities,)
    iv_true    : shape (n_strikes, n_maturities)
    iv_pred    : shape (n_strikes, n_maturities)
    """
    K, T = np.meshgrid(maturities, strikes)   # (n_strikes, n_maturities)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=T, y=K, z=iv_true,
        name="True",
        colorscale="Blues",
        opacity=0.85,
        showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "white"}},
    ))

    fig.add_trace(go.Surface(
        x=T, y=K, z=iv_pred,
        name="Predicted",
        colorscale="Reds",
        opacity=0.75,
        showscale=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Maturity (yr)",
            zaxis_title="Implied Vol",
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
            zaxis=dict(showgrid=True, gridcolor="lightgray"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Calibration scatter: true vs predicted per parameter
# ---------------------------------------------------------------------------

def plot_calibration_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] = PARAM_NAMES,
    title: str = "True vs Predicted Parameters",
    show: bool = True,
) -> go.Figure:
    """
    One scatter subplot per parameter — diagonal = perfect calibration.
    """
    n = len(param_names)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PARAM_LABELS.get(p, p) for p in param_names],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    for idx, name in enumerate(param_names):
        row = idx // cols + 1
        col = idx  % cols + 1

        t = y_true[:, idx]
        p = y_pred[:, idx]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = (hi - lo) * 0.05

        # Scatter
        fig.add_trace(go.Scatter(
            x=t, y=p,
            mode="markers",
            marker=dict(size=3, opacity=0.4, color="#2563eb"),
            name=name,
            showlegend=False,
        ), row=row, col=col)

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            showlegend=(idx == 0),
            name="y = x",
        ), row=row, col=col)

        fig.update_xaxes(title_text="True",      row=row, col=col)
        fig.update_yaxes(title_text="Predicted", row=row, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300 * rows,
        template="plotly_white",
        margin=dict(t=80),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Error distribution histograms
# ---------------------------------------------------------------------------

def plot_error_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] = PARAM_NAMES,
    title: str = "Parameter Error Distributions",
    show: bool = True,
) -> go.Figure:
    """
    Histogram of (predicted - true) per parameter.
    """
    n = len(param_names)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PARAM_LABELS.get(p, p) for p in param_names],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    for idx, name in enumerate(param_names):
        row = idx // cols + 1
        col = idx  % cols + 1
        err = y_pred[:, idx] - y_true[:, idx]

        fig.add_trace(go.Histogram(
            x=err,
            nbinsx=50,
            marker_color="#2563eb",
            opacity=0.75,
            name=name,
            showlegend=False,
        ), row=row, col=col)

        # Zero line
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                      row=row, col=col)

        fig.update_xaxes(title_text="Error", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300 * rows,
        template="plotly_white",
        margin=dict(t=80),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    history: dict,
    title: str = "Training & Validation Loss",
    show: bool = True,
) -> go.Figure:
    """
    Plot train/val MSE loss over epochs from the history dict returned by train().
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        mode="lines", name="Train MSE",
        line=dict(color="#2563eb", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        mode="lines", name="Val MSE",
        line=dict(color="#dc2626", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.75, y=0.95),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Full benchmark report
# ---------------------------------------------------------------------------

def benchmark_report(
    y_true_neural: np.ndarray,
    y_pred_neural: np.ndarray,
    y_true_lm:     np.ndarray,
    y_pred_lm:     np.ndarray,
    neural_times_ms: list[float],
    lm_times_ms:     list[float],
    show_plots: bool = True,
) -> dict:
    """
    Full side-by-side benchmark: neural calibrator vs LM.

    Returns dict with metric DataFrames and Plotly figures.
    """
    metrics_neural = compute_metrics(y_true_neural, y_pred_neural)
    metrics_lm     = compute_metrics(y_true_lm,     y_pred_lm)
    speed_df       = compare_speed(neural_times_ms, lm_times_ms)

    print_metrics(metrics_neural, "Neural Calibrator — Test Metrics")
    print_metrics(metrics_lm,     "LM Baseline       — Test Metrics")
    print_metrics(speed_df,       "Speed Comparison")

    figs = {}
    figs["scatter_neural"] = plot_calibration_scatter(
        y_true_neural, y_pred_neural,
        title="Neural: True vs Predicted", show=show_plots,
    )
    figs["scatter_lm"] = plot_calibration_scatter(
        y_true_lm, y_pred_lm,
        title="LM: True vs Predicted", show=show_plots,
    )
    figs["errors_neural"] = plot_error_distributions(
        y_true_neural, y_pred_neural,
        title="Neural: Error Distributions", show=show_plots,
    )

    return {
        "metrics_neural": metrics_neural,
        "metrics_lm":     metrics_lm,
        "speed":          speed_df,
        "figures":        figs,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Metrics — smoke test (synthetic data) ===\n")

    rng = np.random.default_rng(0)
    N = 500

    # Fake ground truth and noisy predictions
    y_true = rng.uniform([0.01, 0.1, 0.01, 0.1, -0.95],
                         [0.50, 5.0, 0.5,  1.0,  0.0], size=(N, 5))
    y_pred_neural = y_true + rng.normal(0, 0.01, size=(N, 5))
    y_pred_lm     = y_true + rng.normal(0, 0.03, size=(N, 5))

    metrics_n = compute_metrics(y_true, y_pred_neural)
    metrics_l = compute_metrics(y_true, y_pred_lm)
    print_metrics(metrics_n, "Neural (synthetic)")
    print_metrics(metrics_l, "LM (synthetic)")

    speed = compare_speed(
        neural_times_ms=list(rng.uniform(0.5, 2.0, 100)),
        lm_times_ms=list(rng.uniform(3000, 8000, 100)),
    )
    print_metrics(speed, "Speed")

    # Plots (comment out if no display)
    # plot_calibration_scatter(y_true, y_pred_neural, show=True)
    # plot_error_distributions(y_true, y_pred_neural, show=True)
    print("Plots skipped in smoke test (call explicitly with show=True).")