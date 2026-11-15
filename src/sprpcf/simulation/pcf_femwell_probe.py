"""Experimental 2D full-vector FEM *feasibility* probe for an explicit PCF geometry.

Femwell+Gmsh is optional. All topology is intentionally labeled exploratory,
not inferred from a peer-reviewed sensor design. NO PCF-SPR publication claim
can be derived from this solver feasibility probe: no PML, convergence, mode
branch validation, independent reference, sensor performance or hardware.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any


@dataclass(frozen=True)
class ExploratoryPcfTopology:
    """Reconstructable exploratory topology, in micrometres; NOT manuscript CAD."""

    pitch_um: float = 2.0
    hole_diameter_over_pitch: float = 0.35
    rings: int = 1
    channel_x_um: float = 0.85
    channel_y_um: float = 0.0
    channel_radius_um: float = 0.32
    gold_thickness_um: float = 0.05
    domain_halfwidth_um: float = 5.5
    circle_quad_segs: int = 16

    def build(self) -> OrderedDict[str, Any]:
        """Create non-overlapping analyte/Au/air/silica regions."""
        from shapely.geometry import Point, box
        from shapely.ops import unary_union

        parameters = asdict(self)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in parameters.values()):
            raise ValueError("nonfinite topology parameter")
        if not (1 <= self.rings <= 4
                and 0.2 <= self.hole_diameter_over_pitch <= 0.65
                and 0.8 <= self.pitch_um <= 4
                and 0.015 <= self.gold_thickness_um <= 0.1):
            raise ValueError("invalid topology parameters")
        if not (8 <= self.circle_quad_segs <= 64
                and self.channel_radius_um > self.gold_thickness_um):
            raise ValueError("invalid circle resolution or channel radius")
        window = box(
            -self.domain_halfwidth_um, -self.domain_halfwidth_um,
            self.domain_halfwidth_um, self.domain_halfwidth_um,
        )
        channel_center = Point(self.channel_x_um, self.channel_y_um)
        channel = channel_center.buffer(self.channel_radius_um, quad_segs=self.circle_quad_segs)
        coating_outer = channel_center.buffer(
            self.channel_radius_um + self.gold_thickness_um,
            quad_segs=self.circle_quad_segs,
        )
        gold = coating_outer.difference(channel)
        holes: list[tuple[str, Any]] = []
        for i in range(-self.rings, self.rings + 1):
            for j in range(-self.rings, self.rings + 1):
                if not 1 <= max(abs(i), abs(j), abs(i + j)) <= self.rings:
                    continue
                center = Point(
                    self.pitch_um * (i + 0.5 * j),
                    self.pitch_um * (math.sqrt(3) / 2 * j),
                )
                air = center.buffer(
                    self.pitch_um * self.hole_diameter_over_pitch / 2,
                    quad_segs=self.circle_quad_segs,
                )
                if air.distance(coating_outer) < 0.10:
                    raise ValueError("air hole overlaps or is too close to Au-coated channel")
                if not window.contains(air):
                    raise ValueError("computational window clips air hole")
                holes.append((f"hole_{len(holes)+1:02d}", air))
        if not window.contains(coating_outer) or not holes:
            raise ValueError("sensor/window geometry invalid or empty")
        silica = window.difference(unary_union([coating_outer, *[p for _, p in holes]]))
        polygons = OrderedDict([("analyte", channel), ("gold", gold), *holes, ("silica", silica)])
        if not all(shape.is_valid and not shape.is_empty and shape.area > 0 for shape in polygons.values()):
            raise ValueError("invalid or degenerate polygon")
        if abs(sum(p.area for p in polygons.values()) - window.area) > 1e-5:
            raise ValueError("topology does not partition computational domain")
        return polygons


def silica_index_sellmeier(wavelength_um: float) -> float:
    """Three-term silica Sellmeier used in the project; valid demonstration range."""
    if not math.isfinite(wavelength_um) or not 0.45 <= wavelength_um <= 1.7:
        raise ValueError("wavelength must be within 0.45–1.7 μm")
    lam_sq = wavelength_um**2
    n2 = 1.0
    for b, c in zip(
        (0.6961663, 0.4079426, 0.8974794),
        (0.0684043**2, 0.1162414**2, 9.896161**2),
    ):
        n2 += b * lam_sq / (lam_sq - c)
    return math.sqrt(n2)


def fitted_gold_permittivity(wavelength_um: float) -> complex:
    """Repository-compatible fitted model; not independently measured source."""
    from sprpcf.simulation.open_physics import fitted_gold_epsilon
    return fitted_gold_epsilon(wavelength_um)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_femwell_pcf_probe(
    out: Path, *, topology: ExploratoryPcfTopology = ExploratoryPcfTopology(),
    wavelength_um: float = 0.8, analyte_ri: float = 1.36,
    num_modes: int = 3, gold_mesh_resolution_um: float = 0.018,
) -> dict[str, Any]:
    """Execute 2D FEM on a demonstrator topology, with *unqualified* evidence.

    Missing scientific ingredients are explicit: no PML/convergence or verified
    mode branch; no geometry from a frozen reference article; no sensor RI sweep.
    """
    if not math.isfinite(analyte_ri) or not 1 < analyte_ri < 2:
        raise ValueError("invalid analyte RI")
    if not isinstance(num_modes, int) or not 1 <= num_modes <= 8:
        raise ValueError("num_modes must be 1..8")
    if not math.isfinite(gold_mesh_resolution_um) or not 0.005 <= gold_mesh_resolution_um <= 0.03:
        raise ValueError("invalid gold mesh resolution")
    polygons = topology.build()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    import importlib.metadata
    import numpy as np
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    n_silica = silica_index_sellmeier(wavelength_um)
    eps_au = fitted_gold_permittivity(wavelength_um)
    if eps_au.real >= -analyte_ri**2 or eps_au.imag <= 0:
        raise ValueError("Au approximation not in passive plasmonic regime")

    geometry_path = out / "geometry.json"
    geometry_path.write_text(
        json.dumps({
            "topology": asdict(topology),
            "topology_type": "explicit-exploratory-not-literature-validated",
            "coordinate_system": "transverse_xy_micrometres",
            "regions": {key: float(poly.area) for key, poly in polygons.items()},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    material_path = out / "materials.json"
    material_path.write_text(
        json.dumps({
            "wavelength_um": wavelength_um, "analyte_ri": analyte_ri,
            "silica_sellmeier_ri": n_silica,
            "gold_fitted_epsilon": {"real": eps_au.real, "imag": eps_au.imag},
            "gold_is_measured_optical_constants": False,
            "gold_model": "repository-Rakic-Drude-Lorentz",
            "temporal_convention": "exp(-i omega t)",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    resolutions: dict[str, dict[str, float]] = {
        "gold": {"resolution": gold_mesh_resolution_um, "distance": 0.15},
        "analyte": {"resolution": 0.05, "distance": 0.2},
    }
    for key in polygons:
        if key.startswith("hole_"):
            resolutions[key] = {"resolution": 0.11, "distance": 0.2}
    mesh_path = out / "pcf_exploratory_mesh.msh"
    mesh_data = mesh_from_OrderedDict(
        polygons, resolutions, filename=str(mesh_path), default_resolution_max=0.5,
    )
    mesh = from_meshio(mesh_data)
    if mesh.nelements > 150000 or mesh.nelements < 100:
        raise ValueError(f"mesh contains invalid/unreasonable element count: {mesh.nelements}")

    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros(dtype=np.complex128)
    expected_eps = {
        key: (
            eps_au if key == "gold"
            else analyte_ri**2 if key == "analyte"
            else 1.0 if key.startswith("hole_")
            else n_silica**2
        ) for key in polygons
    }
    for region, value in expected_eps.items():
        epsilon[basis0.get_dofs(elements=region)] = value

    # These are unselected raw vectorial eigenmodes; no claim of sensor core mode
    # or propagation attenuation until PML, field tracking and convergence exist.
    modes = compute_modes(
        basis0, epsilon, wavelength=wavelength_um,
        num_modes=num_modes, order=1, n_guess=n_silica,
    )
    eigenpairs = []
    for i, mode in enumerate(modes):
        neff = complex(mode.n_eff)
        if not np.isfinite(neff.real) or not np.isfinite(neff.imag):
            raise ValueError("computed FEM effective index is nonfinite")
        eigenpairs.append({
            "mode_index": i, "n_eff_real": neff.real, "n_eff_imag": neff.imag,
            "mode_identity_independently_verified": False,
        })
    if not eigenpairs:
        raise ValueError("FEM solver produced no eigenmodes")

    mode_path = out / "raw_eigenvalues.json"
    mode_path.write_text(json.dumps(eigenpairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "experiment": "exploratory_2d_pcf_full_vector_fem_feasibility",
        "solver": "femwell-maxwell-waveguide",
        "solver_version": importlib.metadata.version("femwell"),
        "mesh_elements": int(mesh.nelements),
        "num_modes_computed": len(eigenpairs),
        "fem_mode_solver_executed": True,
        "boundary_type": "finite-window-no-PML",
        "Au_mesh_resolution_um": gold_mesh_resolution_um,
        "source_evidence_class": "unqualified_solver_feasibility",
        "topology_frozen_to_published_reference": False,
        "meshing_converged": False,
        "pml_validated": False,
        "modal_branch_verified": False,
        "physical_sensor_sensitivity_verified": False,
        "scientific_submission_ready": False,
        "full_vector_implementation_validated_for_pcf_spr": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "sha256": {p.name: _checksum(p) for p in (geometry_path, material_path, mesh_path, mode_path)},
    }
    (out / "FEMWELL_PILOT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
