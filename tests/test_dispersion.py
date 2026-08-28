from __future__ import annotations

import numpy as np
import torch

from sprpcf.ml.losses import geometry_constraint_loss
from sprpcf.simulation.dispersion import gold_permittivity_drude_lorentz, silica_refractive_index


def test_silica_sellmeier_visible_index_range() -> None:
    index = silica_refractive_index(np.array([0.633]))
    assert 1.44 < float(index[0]) < 1.47


def test_gold_drude_lorentz_is_metallic_in_visible() -> None:
    epsilon = gold_permittivity_drude_lorentz(np.array([0.8]))
    assert float(epsilon.real[0]) < 0.0


def test_geometry_loss_accepts_dispersion_penalty() -> None:
    # Current design schema: pitch_um, d_over_lambda, metal_thickness_nm, channel_radius_um.
    geometry = torch.tensor([[2.0, 0.5, 45.0, 0.6]], dtype=torch.float32)
    wavelength_nm = torch.tensor([650.0], dtype=torch.float32)
    loss = geometry_constraint_loss(geometry, resonance_wavelength_nm=wavelength_nm, dispersion_weight=0.1)
    assert torch.isfinite(loss)
