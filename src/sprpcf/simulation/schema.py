from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Fabrication-relevant PCF-SPR design variables."""

    d_over_lambda: float
    pitch_um: float
    metal_thickness_nm: float
    analyte_ri: float
    channel_radius_um: float = 0.6

    @property
    def air_hole_diameter_um(self) -> float:
        return self.d_over_lambda * self.pitch_um


@dataclass(frozen=True)
class SpectralMetrics:
    """Extracted sensing metrics from a loss spectrum."""

    lambda_res_nm: float
    peak_loss_db_per_cm: float
    fwhm_nm: float
    sensitivity_nm_per_riu: float | None
    fom_per_riu: float | None
