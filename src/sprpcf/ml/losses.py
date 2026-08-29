from __future__ import annotations

import torch
import torch.nn.functional as F

from sprpcf.simulation.dispersion import plasmonic_validity_score

GEOMETRY_MIN = (0.8, 0.20, 15.0, 0.20)
GEOMETRY_MAX = (4.0, 0.90, 80.0, 1.50)


def _normalized_boundary_penalty(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    scale = max(upper - lower, 1e-6)
    return (F.relu((lower - values) / scale).pow(2) + F.relu((values - upper) / scale).pow(2)).mean()


def clamp_physical_geometry(geometry: torch.Tensor) -> torch.Tensor:
    """Project physical geometry to the documented fabrication envelope."""
    lower = torch.tensor(GEOMETRY_MIN, dtype=geometry.dtype, device=geometry.device)
    upper = torch.tensor(GEOMETRY_MAX, dtype=geometry.dtype, device=geometry.device)
    return torch.minimum(torch.maximum(geometry, lower), upper)


def geometry_constraint_loss(
    geometry: torch.Tensor,
    overlap_weight: float = 1.0,
    boundary_weight: float = 1.0,
    resonance_wavelength_nm: torch.Tensor | None = None,
    dispersion_weight: float = 0.0,
) -> torch.Tensor:
    """Differentiable fabrication penalty for PCF-SPR geometry.

    Column order: pitch_um, d_over_lambda, metal_thickness_nm, channel_radius_um.
    """
    if geometry.ndim != 2 or geometry.shape[1] != 4:
        raise ValueError("geometry must have shape [batch, 4].")

    pitch_um = geometry[:, 0]
    d_over_lambda = geometry[:, 1]
    metal_nm = geometry[:, 2]
    channel_radius_um = geometry[:, 3]
    air_diameter_um = d_over_lambda * pitch_um

    overlap_loss = F.relu((air_diameter_um - pitch_um) / torch.clamp(pitch_um, min=1e-6)).pow(2).mean()
    boundary_loss = (
        _normalized_boundary_penalty(pitch_um, 0.8, 4.0)
        + _normalized_boundary_penalty(d_over_lambda, 0.20, 0.90)
        + _normalized_boundary_penalty(metal_nm, 15.0, 80.0)
        + _normalized_boundary_penalty(channel_radius_um, 0.20, 1.50)
    )
    total = overlap_weight * overlap_loss + boundary_weight * boundary_loss
    if resonance_wavelength_nm is not None and dispersion_weight > 0.0:
        wavelength_um = resonance_wavelength_nm.reshape(-1).to(dtype=geometry.dtype, device=geometry.device) / 1000.0
        validity = plasmonic_validity_score(wavelength_um)
        dispersion_loss = F.relu(-validity).pow(2).mean()
        total = total + dispersion_weight * dispersion_loss
    return total
