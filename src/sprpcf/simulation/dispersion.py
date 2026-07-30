from __future__ import annotations

import numpy as np
import torch


SILICA_SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)
SILICA_SELLMEIER_C_UM2 = (0.0684043**2, 0.1162414**2, 9.896161**2)
HC_EV_UM = 1.239841984


def silica_refractive_index(wavelength_um: np.ndarray | float) -> np.ndarray:
    """Return fused-silica refractive index from the three-term Sellmeier equation."""
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    wavelength_sq = wavelength**2
    n_sq = np.ones_like(wavelength_sq)
    for coefficient_b, coefficient_c in zip(SILICA_SELLMEIER_B, SILICA_SELLMEIER_C_UM2):
        n_sq = n_sq + coefficient_b * wavelength_sq / (wavelength_sq - coefficient_c)
    return np.sqrt(n_sq)


def torch_silica_refractive_index(wavelength_um: torch.Tensor) -> torch.Tensor:
    """Differentiable fused-silica Sellmeier refractive index."""
    wavelength_sq = wavelength_um.pow(2)
    n_sq = torch.ones_like(wavelength_sq)
    for coefficient_b, coefficient_c in zip(SILICA_SELLMEIER_B, SILICA_SELLMEIER_C_UM2):
        n_sq = n_sq + coefficient_b * wavelength_sq / (wavelength_sq - coefficient_c)
    return torch.sqrt(torch.clamp(n_sq, min=1e-9))


def gold_permittivity_drude_lorentz(wavelength_um: np.ndarray | float) -> np.ndarray:
    """Approximate complex Au permittivity using a compact Drude-Lorentz model.

    Parameters are adapted from commonly used Rakic-style Au fits and are intended
    for physical regularization, not as a substitute for measured optical constants.
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    energy_ev = HC_EV_UM / wavelength
    omega_p = 9.03
    epsilon_inf = 1.53
    drude_f = 0.760
    drude_gamma = 0.053
    oscillator_f = np.array([0.024, 0.010, 0.071, 0.601, 4.384])
    oscillator_gamma = np.array([0.241, 0.345, 0.870, 2.494, 2.214])
    oscillator_energy = np.array([0.415, 0.830, 2.969, 4.304, 13.32])

    epsilon = epsilon_inf - drude_f * omega_p**2 / (energy_ev * (energy_ev + 1j * drude_gamma))
    for strength, damping, resonance in zip(oscillator_f, oscillator_gamma, oscillator_energy):
        epsilon = epsilon + strength * omega_p**2 / (resonance**2 - energy_ev**2 - 1j * damping * energy_ev)
    return epsilon


def gold_refractive_index_drude_lorentz(wavelength_um: np.ndarray | float) -> np.ndarray:
    """Return complex Au refractive index from Drude-Lorentz permittivity."""
    return np.sqrt(gold_permittivity_drude_lorentz(wavelength_um))


def torch_gold_permittivity_drude(wavelength_um: torch.Tensor) -> torch.Tensor:
    """Differentiable Drude-only Au permittivity for training-time checks."""
    energy_ev = HC_EV_UM / wavelength_um
    omega_p = torch.tensor(9.03, dtype=wavelength_um.dtype, device=wavelength_um.device)
    epsilon_inf = torch.tensor(1.53, dtype=wavelength_um.dtype, device=wavelength_um.device)
    drude_f = torch.tensor(0.760, dtype=wavelength_um.dtype, device=wavelength_um.device)
    drude_gamma = torch.tensor(0.053, dtype=wavelength_um.dtype, device=wavelength_um.device)
    real = epsilon_inf - drude_f * omega_p.pow(2) / (energy_ev.pow(2) + drude_gamma.pow(2))
    imag = drude_f * omega_p.pow(2) * drude_gamma / (energy_ev * (energy_ev.pow(2) + drude_gamma.pow(2)))
    return torch.complex(real, imag)


def plasmonic_validity_score(wavelength_um: torch.Tensor) -> torch.Tensor:
    """Positive score when Au is metallic and silica remains transparent."""
    silica_n = torch_silica_refractive_index(wavelength_um)
    gold_eps = torch_gold_permittivity_drude(wavelength_um)
    metallic_margin = -gold_eps.real
    silica_margin = silica_n - 1.0
    return torch.minimum(metallic_margin, silica_margin)
