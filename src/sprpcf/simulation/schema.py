from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Fabrication-relevant PCF-SPR design variables and operating condition."""

    d_over_lambda: float
    pitch_um: float
    metal_thickness_nm: float
    analyte_ri: float
    channel_radius_um: float = 0.6

    @property
    def air_hole_diameter_um(self) -> float:
        return self.d_over_lambda * self.pitch_um

    def validate(self) -> None:
        """Reject geometry values outside the supported fabrication envelope."""
        if not 0.20 <= self.d_over_lambda <= 0.90:
            raise ValueError("d_over_lambda must be within [0.20, 0.90].")
        if not 0.8 <= self.pitch_um <= 4.0:
            raise ValueError("pitch_um must be within [0.8, 4.0] um.")
        if not 15.0 <= self.metal_thickness_nm <= 80.0:
            raise ValueError("metal_thickness_nm must be within [15, 80] nm.")
        if not 0.20 <= self.channel_radius_um <= 1.50:
            raise ValueError("channel_radius_um must be within [0.20, 1.50] um.")
        if not 1.0 < self.analyte_ri < 2.0:
            raise ValueError("analyte_ri must be a physically plausible refractive index in (1, 2).")
        if self.air_hole_diameter_um >= self.pitch_um:
            raise ValueError("Air-hole diameter must be smaller than pitch.")


@dataclass(frozen=True)
class SpectralMetrics:
    """Extracted sensing metrics from a confinement-loss spectrum."""

    lambda_res_nm: float
    peak_loss_db_per_cm: float
    fwhm_nm: float
    sensitivity_nm_per_riu: float | None
    fom_per_riu: float | None
