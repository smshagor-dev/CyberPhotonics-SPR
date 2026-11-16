"""Unit-explicit passive complex-neff attenuation audit; not a physics solver.

Convention: fields propagate as exp(+i beta z - i omega t), so Im(beta)>=0
means decaying field amplitude. Power attenuation dB/cm is derived from
10 log10(P(0)/P(z)); never use 20 log10 on power.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LossPoint:
    wavelength_nm: float
    neff_real: float
    neff_imag: float
    loss_db_per_cm: float
    power_1cm_fraction: float


def loss_from_neff(wavelength_nm: float, neff: complex) -> LossPoint:
    """Convert passive forward-mode neff into power attenuation, dB/cm.

    beta = (2*pi/lambda_m)*neff, P(z)/P(0)=exp(-2*Im(beta)*z).
    Hence loss = (20/log(10))*Im(beta)*0.01 dB/cm.
    """
    if not isinstance(neff, (complex, float, int)) or isinstance(neff, bool):
        raise TypeError('neff must be a scalar complex number')
    if isinstance(wavelength_nm, bool) or not isinstance(wavelength_nm, (float, int)):
        raise TypeError('wavelength must be a scalar number')
    n = complex(neff)
    if not (math.isfinite(wavelength_nm) and wavelength_nm > 0):
        raise ValueError('wavelength_nm must be positive and finite')
    if not (math.isfinite(n.real) and math.isfinite(n.imag)):
        raise ValueError('neff must be finite')
    if n.imag < 0:
        raise ValueError('negative imaginary neff violates passive forward convention')
    beta_im_per_m = 2 * math.pi * n.imag / (wavelength_nm * 1e-9)
    alpha_db_per_cm = (20 / math.log(10)) * beta_im_per_m * 0.01
    if not math.isfinite(alpha_db_per_cm):
        raise ValueError('attenuation overflow')
    return LossPoint(float(wavelength_nm), n.real, n.imag,
                     alpha_db_per_cm, math.exp(-2 * beta_im_per_m * 0.01))
