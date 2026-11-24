"""Convert native complex effective indices to modal confinement loss.

Software-only observable conversion. Not proof of a qualified PCF-SPR mode,
resonance, PML convergence, material fidelity or COMSOL verification.
The solver must establish exp(+i beta z - i omega t) before use.
"""
import math


def confinement_loss_db_per_cm(wavelength_um, neff, *, convention="exp(+i_beta_z-i_omega_t)"):
    """Power loss [dB/cm] = (20/ln(10))*k0*Im(neff)/100.

    k0 is in rad/m. Positive Im(neff) attenuates power for the stated
    propagation convention; no sign correction is performed automatically.
    """
    if convention != "exp(+i_beta_z-i_omega_t)":
        raise ValueError("unsupported propagation convention")
    if isinstance(wavelength_um, bool) or not isinstance(wavelength_um, (int, float)):
        raise ValueError("invalid wavelength")
    if not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("invalid wavelength")
    if not isinstance(neff, complex) or not math.isfinite(neff.real) or not math.isfinite(neff.imag):
        raise ValueError("finite complex neff required")
    if neff.imag < 0:
        raise ValueError("negative attenuation: verify solver convention and eigenmode")
    loss = (20 / math.log(10)) * (2 * math.pi / (wavelength_um * 1e-6)) * neff.imag / 100
    if not math.isfinite(loss):
        raise ValueError("nonfinite computed loss")
    return loss


def spectrum_loss(records):
    """Convert strictly ordered spectral records without claiming a resonance."""
    if not isinstance(records, (list, tuple)) or len(records) < 2:
        raise ValueError("at least two spectral points required")
    result = []
    previous = -math.inf
    for row in records:
        if not isinstance(row, dict) or set(row) != {"wavelength_um", "neff"}:
            raise ValueError("invalid spectrum record")
        wl = row["wavelength_um"]
        loss = confinement_loss_db_per_cm(wl, row["neff"])
        if wl <= previous:
            raise ValueError("wavelengths must be strictly increasing")
        previous = wl
        result.append({"wavelength_um": wl, "loss_db_per_cm": loss})
    return result
