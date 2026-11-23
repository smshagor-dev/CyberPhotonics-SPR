# Native measured-Au full-vector archive audit — 2026-11-23

Source: GitHub Actions run 45076667894, artifact 14775991240 (branch `research/measured-au-eight-wavelength-vector-campaign-20261122`, commit `be0b6bbbd5a673ff80927eef8cc9bab428df3289`).

Actual ZIP downloaded and read locally. ZIP SHA256: `b318310ceff177754835110d0fd85270a6402b096fda3c61fbf2b71815cf0d90` (matches GitHub artifact metadata).

## Executed audit

- 22 ZIP entries, including 16 native E/H NPZ archives, frozen Gmsh mesh, 32-row eigenmode CSV, and experiment/topology/overlap metadata.
- All 21 internal artifact SHA256 entries matched actual decompressed bytes.
- Each of the 16 NPZ files contains `mode_0_E`, `mode_0_H`, `mode_1_E`, `mode_1_H`, each complex128 with 97,969 coefficients.
- Zero nonfinite coefficients across the native field arrays; zero empty field arrays; zero nonfinite numeric values in inspected CSV columns.
- 2 analyte refractive indices (1.35 and 1.37), 8 wavelengths (0.74–0.88 μm, step 0.02 μm), 2 modes per point. Au reference: Johnson–Christy 1972 bulk optical data, not measured sensor film.
- Solver identity in manifest: `femwell.maxwell.waveguide.compute_modes`, version 0.1.12. Exploratory topology: one-ring, finite-box, not peer-reviewed/frozen literature geometry.

## Scientific interpretation and exclusions

This is **real solver output with verified archive byte integrity**, not fabricated/synthetic field data. Structural integrity does not establish correctness of the Maxwell solution, boundary treatment, physical core/SPP mode identity, or sensitivity.

The archive explicitly declares `pml_validated=false`, `mesh_convergence_verified=false`, `mode_tracking_physically_verified=false`, `resonance_fwhm_verified=false`, and `scientific_submission_ready=false`. No PML, independent solver agreement, domain convergence, or resolved verified resonance is established. Native modes are untracked. No COMSOL model or result is claimed.

`physics_qualified=false`, `publication_ready=false`, `comsol_verified=false`.

Reproduce by downloading artifact 14775991240 and running `python scripts/audit_native_maxwell_zip.py downloaded.zip` with NumPy installed. This audit script is software-only and deliberately never sets a physics publication gate to true.
