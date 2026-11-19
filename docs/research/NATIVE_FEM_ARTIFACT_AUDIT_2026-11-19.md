# Byte-level audit of genuine measured-reference Au exploratory FEM grid

Audit date: 2026-11-19 UTC. Source: GitHub Actions run [44434895603](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/44434895603), artifact ID **14489450225** (`measured-reference-au-exploratory-2d-vector-ri-grid`), produced from PR #148 head `2aed9b0f86e02da6877639df82cb60bc0f177245`. This report is an independent post-run inspection of the downloaded ZIP's actual bytes, not a simulation rerun.

Artifact ZIP SHA256: `d22a32c950f5a80fa964fe7913aac3b6c96ebb5ee39c8e7983e80c994d359dd8`.

## Reproducible audit procedure

Download artifact ID 14489450225 from the run, verify the ZIP SHA256, extract into a fresh directory, and execute:

```python
import csv, hashlib, json
from pathlib import Path
p = Path(".")
report = json.loads((p / "GRID_READINESS.json").read_text())
assert report["completed"] and report["solver_calls_completed"] == 9
assert report["scientific_submission_ready"] is False
assert report["pml_validated"] is False
assert report["mesh_convergence_verified"] is False
for name, expected in report["artifacts_sha256"].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
rows = list(csv.DictReader((p / "raw_eigenmode_grid.csv").open()))
edges = json.loads((p / "vector_mode_overlap_diagnostics.json").read_text())
assert len(rows) == 18 and len(edges) == 24
assert all(float(r["raw_attenuation_db_per_cm_unqualified"]) >= 0 for r in rows)
assert all(not e["assignment_ambiguous_nonpositive_margin"] for e in edges)
print("Audit passed: hashes, 9 solves, 18 modes, 24 diagnostic edges")
```

## Directly observed native evidence

- ZIP contains **15 files**: mesh (.msh), protocol, geometry, readiness, eigenmode CSV, overlap JSON, and **9 compressed native E/H NPZ archives**.
- **14/14** file SHA256 entries in `GRID_READINESS.json` match the extracted native files (the readiness JSON is intentionally not self-hashed).
- **9** completed Femwell 0.1.12 solver calls, **18** raw eigenmode rows, **24** adjacent-grid E/H overlap assignment edges, **0** reported failures.
- Same mesh has **48,940** triangular elements; Au interface resolution parameter is **0.025 µm**.
- Fixed exploratory geometry: wavelengths **0.75, 0.80, 0.85 µm**, analyte RI **1.35, 1.36, 1.37**; **2** native modes per point.
- Recorded unqualified raw attenuation spans **0.0241033532–0.1453226255 dB/cm**; all 18 values are nonnegative. Recorded imaginary effective index spans **3.31241356e-8–2.26338267e-7**.
- Across the 24 reported edges, absolute vector E/H overlap spans **0.9992740924–0.9999999995**; **0** edges have nonpositive bidirectional assignment margin. These are adjacent-mode *diagnostics*, not core/SPP identity proof.
- One directly opened NPZ contains four arrays named `mode_0_E`, `mode_0_H`, `mode_1_E`, `mode_1_H`. The full nine NPZ files' physical field consistency was **not independently re-solved**.

## Evidence limitations (mandatory)

The original `GRID_READINESS.json` declares `sensor_geometry_peer_reviewed_and_frozen=false`, `pml_validated=false`, `mesh_convergence_verified=false`, `mode_tracking_physically_verified=false`, `resonance_fwhm_verified=false`, `physical_sensor_sensitivity_verified=false`, and `scientific_submission_ready=false`. Preserve all of them as **FALSE**.

Johnson–Christy bulk Au reference is marked sourced by the generating workflow; this audit checked hashes of the archived manifest, **not** the original paper against its transcription. The geometry is an exploratory finite-box one-ring model, not a literature-frozen PCF-SPR design. Three wavelength samples do not establish a resonance peak or FWHM. The observed attenuation is **not PML-converged confinement loss**. No independent full-vector solver or COMSOL model was used; the audit does not establish sensitivity, FOM, physical device behavior, or publication readiness.

Next experiments: freeze a citable topology; verify source-transcribed Au independently; add PML/domain and at least three mesh levels with branch-stable E/H tracking; resolve spectra sufficiently to bracket peaks and half-maximum crossings; compare a separate open-source full-vector solver before inverse training.
