from __future__ import annotations

import torch
import torch.nn.functional as F

from sprpcf.simulation.dispersion import plasmonic_validity_score


def geometry_constraint_loss(
    geometry: torch.Tensor,
    overlap_weight: float = 1.0,
    boundary_weight: float = 1.0,
    resonance_wavelength_nm: torch.Tensor | None = None,
    dispersion_weight: float = 0.0,
) -> torch.Tensor:
    """Differentiable fabrication penalty for PCF-SPR geometry.

    Column order: pitch_um, d_over_lambda, metal_thickness_nm.
    """
    pitch_um = geometry[:, 0]
    d_over_lambda = geometry[:, 1]
    metal_nm = geometry[:, 2]
    air_diameter_um = d_over_lambda * pitch_um

    overlap_loss = F.relu(air_diameter_um - pitch_um).pow(2).mean()
    boundary_loss = torch.stack(
        [
            F.relu(20.0 - metal_nm).pow(2).mean(),
            F.relu(metal_nm - 80.0).pow(2).mean(),
        ]
    ).sum()
    total = overlap_weight * overlap_loss + boundary_weight * boundary_loss
    if resonance_wavelength_nm is not None and dispersion_weight > 0.0:
        wavelength_um = resonance_wavelength_nm.reshape(-1).to(dtype=geometry.dtype, device=geometry.device) / 1000.0
        validity = plasmonic_validity_score(wavelength_um)
        dispersion_loss = F.relu(-validity).pow(2).mean()
        total = total + dispersion_weight * dispersion_loss
    return total
