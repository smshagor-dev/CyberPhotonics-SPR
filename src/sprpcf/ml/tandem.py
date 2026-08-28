from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Small fully connected network with optional dropout uncertainty support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU(), nn.LayerNorm(width)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardNetwork(nn.Module):
    """Predict sensing metrics from sensor design plus analyte RI condition."""

    def __init__(self, input_dim: int = 5, metric_dim: int = 3) -> None:
        super().__init__()
        self.model = MLP(input_dim, metric_dim, hidden=(128, 128, 64))

    def forward(self, forward_inputs: torch.Tensor) -> torch.Tensor:
        return self.model(forward_inputs)


class InverseGenerator(nn.Module):
    """Generate fabrication geometry from target metrics and analyte RI condition."""

    def __init__(
        self,
        metric_dim: int = 3,
        condition_dim: int = 1,
        geometry_dim: int = 4,
        latent_dim: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.model = MLP(
            metric_dim + condition_dim + latent_dim,
            geometry_dim,
            hidden=(128, 128, 64),
            dropout=dropout,
        )

    def forward(
        self,
        metrics: torch.Tensor,
        conditions: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent is None:
            latent = torch.zeros(metrics.shape[0], self.latent_dim, dtype=metrics.dtype, device=metrics.device)
        return self.model(torch.cat([metrics, conditions, latent], dim=-1))


class TandemNetwork(nn.Module):
    """Tandem inverse-design model using a frozen forward surrogate."""

    def __init__(self, forward_model: ForwardNetwork, inverse_model: InverseGenerator) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(
        self,
        target_metrics: torch.Tensor,
        conditions: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.inverse_model(target_metrics, conditions, latent)
        predicted_metrics = self.forward_model(torch.cat([geometry, conditions], dim=-1))
        return geometry, predicted_metrics
