from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Small fully connected network with SiLU activations."""

    def __init__(self, in_features: int, out_features: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU(), nn.LayerNorm(width)])
            previous = width
        layers.append(nn.Linear(previous, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardNetwork(nn.Module):
    """Predict sensing metrics from PCF-SPR geometry."""

    def __init__(self, geometry_dim: int = 3, metric_dim: int = 3) -> None:
        super().__init__()
        self.model = MLP(geometry_dim, metric_dim, hidden=(128, 128, 64))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.model(geometry)


class InverseGenerator(nn.Module):
    """Generate geometry candidates from target sensing metrics."""

    def __init__(self, metric_dim: int = 3, geometry_dim: int = 3, latent_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.model = MLP(metric_dim + latent_dim, geometry_dim, hidden=(128, 128, 64))

    def forward(self, metrics: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is None:
            latent = torch.zeros(metrics.shape[0], self.latent_dim, device=metrics.device)
        return self.model(torch.cat([metrics, latent], dim=-1))


class TandemNetwork(nn.Module):
    """Tandem inverse-design model using a frozen forward surrogate."""

    def __init__(self, forward_model: ForwardNetwork, inverse_model: InverseGenerator) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, target_metrics: torch.Tensor, latent: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.inverse_model(target_metrics, latent)
        predicted_metrics = self.forward_model(geometry)
        return geometry, predicted_metrics
