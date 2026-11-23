# Measured-Au 8×2 native ZIP: independent byte-level inspection (2026-11-23)

**Evidence class:** exploratory, unqualified full-vector Femwell numerical feasibility. Not COMSOL, not an experimental sensor result.

## Exact source

- GitHub Actions run: https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/45076667894
- Artifact ID: `14775991240`; 97,656,281 bytes; SHA256 `b318310ceff177754835110d0fd85270a6402b096fda3c61fbf2b71815cf0d90`
- Producer commit: `be0b6bbbd5a673ff80927eef8cc9bab428df3289`.
- Downloaded actual ZIP bytes and independently checked SHA256 and ZIP CRC (no errors).

## Inspected native contents

- 22 ZIP members: 16 E/H NPZ files, mesh, topology, protocol, raw CSV, overlap diagnostics, readiness JSON.
- All 21 SHA256 hashes in readiness JSON matched the actual corresponding native members.
- 32 eigenmode CSV rows, 16 RI/wavelength solver calls; 64 complex E/H arrays checked for finite, nonzero coefficients, each with 97,969 entries. ZIP and nested NPZ CRC checks succeeded.
- All 32 attenuation entries matched the explicit `8.685889638065037 × 2π × Im(n_eff) × 10^4 / λ_µm` identity to floating-point tolerance, with nonnegative imaginary effective index.
- Femwell 0.1.12, 48,940 mesh elements, Au interface step 0.025 µm, wavelengths 0.74–0.88 µm (8 points), analyte RI 1.35 and 1.37.

## Diagnostic warning

Native mode index **cannot be treated as a tracked physical core or SPP branch**. For RI=1.35, native mode 0 loss [dB/cm, unqualified] at successive wavelengths is:
`0.02318, 0.02539, 0.04981, 0.03327, 0.07974, 0.04492, 0.20412, 0.06905`.
Mode 1 is:
`0.03584, 0.04161, 0.02890, 0.06137, 0.03868, 0.11306, 0.05618, 0.61089`.
The alternation and large last-point excursion warrant field-based branch identification and convergence checks, **not** peak fitting or physical sensitivity inference. These numbers are diagnostic eigenvalue-derived attenuations, not qualified sensor results.

## Scientific gates (remain false)

`sensor_geometry_peer_reviewed_and_frozen=false`; `pml_validated=false`; `mesh_convergence_verified=false`; `mode_tracking_physically_verified=false`; `resonance_fwhm_verified=false`; `physical_sensor_sensitivity_verified=false`; `scientific_submission_ready=false`; `comsol_verified=false`.

**Next physics milestone:** source-frozen literature topology and independent PML/outgoing boundary implementation, matched E/H core/SPP field localization, three-level mesh and two-domain PML convergence on each spectral branch, then adaptive fixed-geometry RI-resolved spectra and independent full-vector cross-solver agreement. No journal performance claim is supported yet.
