"""
models/network.py
-----------------
Feedforward MLP calibrator: maps a flattened implied vol surface to
Heston model parameters.

Architecture:
  Input (80) → FC(256) → BN → ReLU → Dropout
             → FC(128) → BN → ReLU → Dropout
             → FC(64)  → BN → ReLU
             → FC(5)   → Sigmoid          ← outputs in [0, 1] (normalised params)

Author: Abdellah Kahlaoui
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# MLP block helper
# ---------------------------------------------------------------------------

def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
    )


# ---------------------------------------------------------------------------
# HestonCalibrator
# ---------------------------------------------------------------------------

class HestonCalibrator(nn.Module):
    """
    Feedforward neural calibrator for the Heston model.

    Input  : flattened normalised IV surface, shape (batch, input_dim)
    Output : normalised Heston params in [0,1], shape (batch, 5)
             order: [v0, kappa, theta, sigma, rho]

    Parameters are in [0,1] at output — use data.surface.denormalize_params
    to recover real-world values.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        output_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim

        # Build hidden layers
        layers: list[nn.Module] = []
        in_d = input_dim
        for out_d in hidden_dims:
            layers.append(_mlp_block(in_d, out_d, dropout))
            in_d = out_d

        self.hidden = nn.Sequential(*layers)

        # Output head — no BN/Dropout, Sigmoid to enforce [0,1]
        self.head = nn.Sequential(
            nn.Linear(in_d, output_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform for linear layers, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    # ------------------------------------------------------------------
    # Convenience: single-surface inference (numpy in, numpy out)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        surface_flat: "np.ndarray",
        surface_mean: Optional["np.ndarray"] = None,
        surface_std:  Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """
        Predict normalised Heston params from a single flat IV surface.

        Optionally applies z-score normalisation if mean/std are supplied
        (they should come from the training set).

        Returns np.ndarray of shape (5,) in [0, 1].
        """
        import numpy as np

        x = surface_flat.copy().astype(np.float32)
        if surface_mean is not None and surface_std is not None:
            x = (x - surface_mean) / surface_std

        tensor = torch.from_numpy(x).unsqueeze(0)  # (1, input_dim)
        self.eval()
        out = self(tensor).squeeze(0).numpy()
        return out

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)

    model = HestonCalibrator(input_dim=80, hidden_dims=(256, 128, 64),
                             output_dim=5, dropout=0.1)
    print(model)
    print(f"\nTrainable parameters: {model.count_params():,}")

    # Forward pass — batch of 32 random surfaces
    x = torch.randn(32, 80)
    y = model(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]  (should be in [0,1])")

    # Single-surface predict
    surface = np.random.rand(80).astype(np.float32)
    pred = model.predict(surface)
    print(f"\nSingle predict output: {np.round(pred, 4)}")