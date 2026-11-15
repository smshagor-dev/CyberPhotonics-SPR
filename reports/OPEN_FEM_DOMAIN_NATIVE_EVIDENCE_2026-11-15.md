# Native measured-Au FEM finite-window experiment — numerical evidence record

Evidence class: UNQUALIFIED open-source full-vector numerical feasibility. Date: 2026-11-15.

## Immutable execution provenance

- Workflow: https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/43692348172
- GitHub Actions artifact ID: 14152654083, name measured-au-finite-box-domain-5p5-vs-7p0-unqualified
- Downloaded ZIP SHA256: ef2be211d0f62cbfd919032c406b482537c9c55d1f467894b353e1e4d8ef0fe7
- Frozen source: Johnson-Christy measured bulk-reference Au (1972), DOI 10.1103/PhysRevB.6.4370; Femwell Maxwell solver version 0.1.12.
- Both subarchives passed all manifest SHA256 checks on independent downloaded artifacts; both preserve native Gmsh meshes, real/imag effective eigenvalues, optical source metadata, complex E/H fields and overlap diagnostics.
- Both solver campaigns completed with zero reported solver failures. The workflow executed four vector Maxwell solves per window, eight total, two eigenmodes per solve.
- The raw source metadata and the diagnostic report both assert scientific_submission_ready = false.

| Numerical quantity | Small window | Expanded window |
|---|---:|---:|
| Square domain halfwidth (um) | 5.5 | 7.0 |
| Mesh elements | 48,940 | 53,850 |
| Wavelengths (um) | 0.75, 0.80 | 0.75, 0.80 |
| Analyte RI | 1.35, 1.36 | 1.35, 1.36 |
| Requested raw eigenmodes per sample | 2 | 2 |
| Solver calls complete | 4/4 | 4/4 |
| Raw eigenpairs | 8 | 8 |
| Native SHA256 mismatches | 0 | 0 |

Gmsh mesh SHA256 for width 5.5: ef21c1972285093f8fbac9cfa817ef1ab4fda07abbebb3ef472f6cd8665717ec.
Gmsh mesh SHA256 for width 7.0: 9e46c5929b3d8a1d330ebb51420fe350292fbaf762f5e223e089c58474ff4fae.

## Observed full-vector numerical differences (diagnostic only)

- Maximum absolute change in matched complex n_eff: **5.881247105e-4**.
- Maximum relative change in the imaginary n_eff component: **0.6088862 = 60.89%**.
- Four RI/wavelength grid points, eight diagnostic mode pairs.
- Two-mode alternative-assignment cost margins by gridpoint: approximately 3.247e-13, 1.401e-12, 3.292e-13, 1.405e-12. These are extremely small compared with the total pairing costs (~0.0010 to ~0.0012), so nearest-complex-eigenvalue pairing is effectively ambiguous. Do not track core/SPP branches by mode index or this assignment.
- The 5.5 um and 7.0 um meshes are independently regenerated. The observed change includes **both** finite-window and remeshing effects; the study cannot isolate a domain-only error.
- No PML is present and no physical core/SPP branch has been established. Im(n_eff) cannot yet be promoted to independently verified sensor attenuation, resonance FWHM, RI sensitivity or figure of merit.

## Required next physics experiments

(1) Exact, source-cited literature geometry with independent CAD verification.
(2) True complex-coordinate-stretched PML or another independently validated outgoing-wave radiation treatment.
(3) Multiple mesh orders/refinements at each *fixed* domain size to distinguish discretization, material and boundary effects, plus numerical confidence intervals.
(4) Overlap-plus-spatial-field-localization identification of core and interface-localized SPP branches.
(5) At least 11, and practically far more, adaptive spectral points around an independently verified and bracketed resonance, across multiple RI, with wavelength/FWHM uncertainty.
(6) Independently implemented second full-vector solver and matched input/output comparison, before any inverse-model scientific training dataset qualifies.

Original PR #71 updated the diagnostic software to flag tiny alternative-assignment margins explicitly; the cited raw artifact remains an immutable source. **No synthetic/fixture data were used to derive the numerical values in this report.** No COMSOL computation or experiment was performed. Publication flags remain FALSE.
