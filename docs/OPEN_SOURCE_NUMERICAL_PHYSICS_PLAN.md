# COMSOL-free scientific numerical-physics validation plan for CyberPhotonics-SPR

**Status:** Proposed workflow / implementation specification, **NOT** completed physics evidence.  
**Baseline:** `main` at `dae850c7b27e1d614dc8c262ab67463550c080ee` (2026-11-15).  
**Scientific blocker:** [#42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).  
**Goal:** A genuinely executed, independently checked, reproducible **numerical** PCF-SPR study without a COMSOL license or `.mph` model. No fabricated experimental claims.

## 0. Critical scientific boundary and selection decision

The current repository's COMSOL-only source validators and `comsol_physics` registry entries **cannot** simply be relabeled for another solver. The current `src/sprpcf/evidence/qualification.py`, `src/sprpcf/utils/readiness.py`, `src/sprpcf/publication/claim_trace.py`, and COMSOL-specific Stage 7–15 scripts have deliberate fail-closed contracts. Preserve these contracts; implement a **separate** open-source backend, verification archive and physics claim gate.

**Preferred primary computation**: two-dimensional cross-sectional **full-vector Maxwell eigenmode FEM** with complex, wavelength-dependent permittivities and a radiation boundary/PML, via [NGSolve/Netgen](https://ngsolve.org) and a validated optical-fiber eigenmode implementation (evaluate [fibermode](https://github.com/jayggg/fibermode), which documents guided/leaky modes, FEAST, PML and confinement loss). This is a proposed technical route, not a claim that these libraries already solve **this** gold-coated PCF-SPR geometry. Gate A below must demonstrate complex, lossy, negative-real-permittivity Au layers, correct mode identification and stable outgoing-wave loss; if not, implement and verify the full-vector formulation directly with NGSolve or switch to a verified implementation. **Do not substitute a scalar Helmholtz surrogate** and call it an equivalent plasmonic model.

**Independent verification**: [Meep](https://meep.readthedocs.io/en/stable/) FDTD using dispersive Au and the correctly configured nonzero out-of-plane `kz` for a fiber cross-section. Its complex temporal eigenfrequency results cannot be compared naively with fixed-frequency spatial-loss FEM results. Follow the [Meep lossy-waveguide FAQ](https://meep.readthedocs.io/en/stable/FAQ/) and [2D with out-of-plane k](https://meep.readthedocs.io/en/stable/2d_Cell_Special_kz/) to perform a controlled conversion/cross-check. This comparison is a **separate numerical method**, not experimental validation. [Femwell](https://helgegehring.github.io/femwell/) is an alternative photonic eigenmode prototype, subject to the same gold/PML/full-vector acceptance tests.

## 1. Freeze an explicit, physically reconstructable sensor

The existing `Geometry` inputs (`pitch_um`, `d_over_lambda`, `metal_thickness_nm`, `channel_radius_um`, `analyte_ri`) are not a full electromagnetic CAD description. Specify and freeze **all** omitted topology: triangular/other lattice, air-hole ring count and placement, core defect, sensing-channel orientation/position, Au-coated surface, supporting medium, symmetry assumptions, finite computational-domain size, air-hole count and material regions. Start with **one previously documented PCF-SPR topology** from a cited peer-reviewed optical design, with documented changes; do not reverse-engineer topology from synthetic ML numbers. Store a cross-section SVG/PNG, geometric source, geometry JSON/YAML, geometry units and SHA-256. Resolve fabrication overlaps, wall thickness, Au continuity and parameter bounds explicitly.

### Material source contract
- Silica: wavelength-dependent Sellmeier with source, coefficients, units and valid wavelength interval; do not use one constant index across a broadband spectrum if dispersion matters.
- Au: tabulated wavelength-dependent complex `n+ik` from a documented source such as [Johnson and Christy (1972)](https://doi.org/10.1103/PhysRevB.6.4370); calculate `epsilon_r=(n+ik)^2`, use a documented `exp(-i omega t)` sign convention, preserve input tables, source bibliographic metadata, interpolation method and checksum. No frequency-independent negative epsilon shortcut.
- Analyte: explicitly modeled dielectric RI, with assumed or measured dispersion range and temperature; state all assumptions.
- Ensure wavelength, length, and absorption units are identical across mesh, solver and artifact conversion.

## 2. Pre-register numerical protocol **before** evaluating results

Store `protocol.json` with hash and immutable external timestamp, Git commit, complete geometry/material references, intended RI range, optical wavelength window, mesh refinements, outer boundary/PML parameters, eigenmode count, mode-selection algorithm, all seed/model splits and solver settings. Start from a **published, justified spectral window**, not a synthetic 820 nm/RIU sensitivity target. Example `[1.33,1.35,1.37,1.39,1.41]` RI is only a proposed grid until justified for the selected model. Pilot coarse wavelength sampling then predeclared local refinement with the same independent rule for all methods. Pre-register numerical-uncertainty tolerances relative to the expected linewidth and grid spacing; do not tune tolerances after seeing unfavorable results.

Any missing convergence, zero/negative or unresolved loss, ambiguous modes, resonance at a wavelength boundary, or nonphysical metal response means the record is **failed/unqualified**, not published.

## 3. Gate A — implementation and analytical sanity tests (no research claims yet)

1. Create a separate isolated Linux/WSL2 Python environment for the solver. Verify the precise upstream package versions, dependency licenses and `pip freeze`/lockfile. **No experimental outputs are created by an install or CI job.**
2. Demonstrate a basic dielectric step-index-fiber guided mode against an independent analytical/semi-analytical solution; save numerical residuals and field plots.
3. Demonstrate planar metal/dielectric SPP dispersion in a geometry where `beta = k0 * sqrt(eps_m*eps_d/(eps_m+eps_d))` is applicable, including sign convention, boundary limits and complex material; use it only as a **limiting-case** test, not a prediction for thin-film PCF.
4. Demonstrate complex propagation constant, positive physically decaying loss, vector electric-field components, energy localization at the Au/analyte interface and sensible mode evolution in a metallic fiber test geometry.
5. Sweep mesh refinement, exterior-domain size and PML thickness/strength independently; reject spurious PML modes. Track core-like branch across wavelengths/RI by normalized eigenfield overlap/continuation, not by highest-loss eigenvalue at each sample. Detect avoided crossings and branch switches.
6. Run negative controls: remove Au / replace Au by lossless dielectric, disconnect RI from sensing region, and vary boundary to expose false resonance detection.
7. Verify eigenmode residuals and solver convergence; repeat at least twice from fresh processes to detect stateful path dependence.

**Gate A passes only if all physical, numerical and source tests pass.** If gold/PML/full-vector support is absent or incorrect, stop and change the backend, not the acceptance criteria.

## 4. Gate B — real 2D numerical PCF-SPR pilot

For a frozen reference geometry:
- Compute the complex guided mode at each predeclared wavelength and analyte RI, preserving **raw** `n_eff.real`, `n_eff.imag`, eigenfield, branch IDs, residual, mesh ID, PML parameters and solver logs.
- With convention `E ~ exp(i*beta*z - i*omega*t)` and `beta = 2*pi*n_eff/lambda_m`, use `loss_db_per_m = (20/ln(10)) * imag(beta)` for a decaying positive-loss branch and `loss_db_per_cm = loss_db_per_m / 100`. Reject inconsistent sign/convention rather than taking an absolute value to hide gain.
- Extract peak resonance `lambda_res_nm`, `FWHM_nm`, fixed-geometry `S_lambda = d(lambda_res_nm)/d(analyte_ri)` in nm/RIU, and `FOM = abs(S_lambda)/FWHM_nm` in RIU^-1. Full RI sweeps **must** use exactly the same geometry.
- Check resonance peak is interior, physically resolved, and stable under mesh, wavelength-grid and PML refinement. For negative controls, no fabricated sensitivity is inferred from noise.
- Use existing `src/sprpcf/simulation/metrics.py` **only after** validating numerical spectra and correct units.
- Record all solver failures in the denominator; no silent reruns or hand-picked peaks.

Expected artifacts, only **after genuine runs**:
```
outputs/open_physics/
  protocol.json
  geometry/geometry.yaml
  materials/au_optical_constants.csv
  pilot/mesh_coarse.msh
  pilot/mesh_medium.msh
  pilot/mesh_fine.msh
  pilot/raw_eigenpairs.parquet
  pilot/raw_mode_fields/
  pilot/spectra.csv
  pilot/solver_log.jsonl
  pilot/convergence_audit.json
  pilot/negative_controls.json
  pilot/physics_pilot_readiness.json
```
These are planned file paths, **not files currently in the repository**.

## 5. Gate C — independent cross-solver challenge

Choose several **precommitted** reference geometries and RI conditions (including one potential failure case) and independently construct the material and geometry in Meep. Demonstrate planar SPP and dielectric-fiber controls first. For the full PCF topology, use correct 2D+`kz` full-vector physics, Au dispersion, spatial and temporal convergence. If the FDTD spatial attenuation conversion is ill-conditioned, explicitly report only the quantities that are genuinely comparable (e.g. field localization/resonance trend) and keep attenuation agreement **unverified**. Assess agreements against **predeclared**, physically justified tolerances; investigate discrepancies rather than averaging them away.

Meep is independent numerical cross-validation, not a fabricated-device measurement. The use of two libraries does not guarantee both implementations are correct; preserve independent expert physics review.

## 6. Gate D — dataset, trained baselines, untouched test and solver-cost accounting

Once the reference mesh/material model is qualified, run a staged source-traceable campaign: (a) tiny pilot with a few geometries; (b) pre-registered expansion based on available RAM/CPU and demonstrated convergence; (c) sufficiently large **group-disjoint** training, validation and fully untouched test geometry sets. Do not claim dataset adequacy until group count, model capacity and uncertainty are justified. Freeze all solver-sourced training rows with an adjacent exact-hash `.meta.json`; never label as `comsol` or `experimental`.

Retrain the source-bound tandem checkpoint, Ridge/random-forest forward baselines and candidate-generation baselines using compatible dataset provenance; adapt existing source checks **explicitly** for an `open_fem` solver type. All inverse methods must use the same held-out target geometries, comparable candidate generation and ranking budgets, and separately itemized solver call counts/wall-clock/peak RAM. Include independent solver evaluation of **selected** inverse designs, not self-judged surrogate predictions. Include multiple **training** seeds (not only generation seeds), ablations, OOD tests, fabrication validity and per-geometry bootstrap intervals. Keep the final test untouched by thresholds, early stopping and method selection.

## 7. Required repository engineering changes (separate, reviewable PRs)

- **PR A:** Scientific geometry/material schema, analytic reference tests, solver feasibility probe; no source promotion.
- **PR B:** Genuine open-source solver adapter, raw vectorial eigenmode archive, loss normalization, mode tracking, PML/domain/mesh audit; kill false PASS on missing/fixture data.
- **PR C:** Solver-neutral frozen request/RI campaign runner, crashes and checkpoints, all-run logs; keep legacy COMSOL runner unchanged.
- **PR D:** New evidence class `open_fem_physics` and optional `fdtd_crosscheck` with source-specific validation (solver identity/version, mesh and material data, raw mode fields/eigenpairs, nonfixture native run logs, hashes and science-review flags). **Do not** map either to `comsol_physics`; add numerical-physics-based paper gate that does not require a measured hardware sensor when no hardware claim is made. Existing release `FULL_EVIDENCE_CLASSES` contains `experimental_sensor` and `device_benchmark`; that **software release** rule should remain distinct from **numerical-only paper** readiness.
- **PR E:** Independently sourced/untouched ML benchmark and cross-method physics validation; manuscript claim table maps every number to frozen experiment+source.
- **PR F:** Reviewer package, tables, failure case figures, manuscript revision, public archived outputs and Zenodo DOI **only after actual deposit**.

### Do not inherit COMSOL terminology

Current COMSOL-specific scripts (`run_comsol_verification_campaign.py`, `probe_comsol_model.py`, `register_evidence.py ... comsol`) still require genuine COMSOL files. A new open-source command path must be implemented; **do not pass fake `.mph` files, label generated spectra `comsol`, or weaken the existing admission filters**. Also retain original synthetic-demo outputs with `software_only` labels.

## 8. Reviewable scientific acceptance tests

| Gate | Minimum evidence | Invalidation condition |
|---|---|---|
| A — solver physics | Analytic dielectric/SPP checks, true full-vector Au eigenmodes and documented material validity | Scalar shortcut, nonphysical Au, PML-only modes, solver nonconvergence |
| B — PCF-SPR | Frozen topology, actual raw eigenpairs and spectra, resolved resonance, fixed-geometry RI sweeps, mesh/spectral/PML convergence | Fake fixture, branch swap, missing/failed points, boundary resonance, unqualified units |
| C — independent | Independent Meep implementation and explicit like-for-like comparisons / limitations | Unjustified conversion of temporal loss to spatial loss |
| D — ML | Fully source-bound independent solver dataset, untouched test, matched inverse baselines, multiple training seeds, uncertainty and cost | Same surrogate grades itself, split leakage, synthetic data presented as physics |
| E — paper | DOI-backed archived raw+derived data, claim matrix, methods, figures, material/model versions and limitations | Missing raw evidence, untraceable headline metrics, fabricated device claim |

The scientific physics reviewer must approve model topology, Maxwell formulation, metal dispersion, mode selection, boundary conditions, convergence and limitations. SHA-256 checks alone are not independent physics review.

## 9. CPU-only feasibility and stop conditions

Start with **one** low-resolution cross-section pilot and report actual time/peak memory/eigenproblem degrees-of-freedom; do not guess that a full PCF-Au-FDTD campaign will run on a 16 GB laptop. A thin 15–80 nm metal layer inside micron-scale geometry requires very careful local mesh refinement; full 3D high-resolution FDTD may exceed consumer resources. For a modest CPU, prefer a converged 2D cross-sectional full-vector FEM mode study, adaptive wavelength refinement and a limited independent FDTD sample. If physics fidelity or resource budget is inadequate, narrow the claim or arrange external institutional compute; never substitute cheap synthetic curves as independent numerical evidence.

**Initial proposed CLI design (NOT currently implemented):**
```text
python scripts/probe_open_fem.py --geometry configs/pcf_spr_topology.yaml --materials configs/optical_materials.yaml --out outputs/open_physics/pilot --strict
python scripts/run_open_fem_campaign.py --plan outputs/open_physics/frozen_plan --out outputs/open_physics/campaign --strict
python scripts/audit_open_fem_physics.py --campaign outputs/open_physics/campaign --out outputs/open_physics/audit --strict
```

## 10. Source references

1. NGSolve PML: https://docu.ngsolve.org/latest/how_to/pml.html
2. fibermode full-vector guided/leaky FEM with PML: https://github.com/jayggg/fibermode
3. Free FEM waveguide alternative: https://helgegehring.github.io/femwell/
4. Meep material dispersion: https://meep.readthedocs.io/en/stable/Materials/
5. Meep 2D out-of-plane propagation: https://meep.readthedocs.io/en/stable/2d_Cell_Special_kz/
6. Meep lossy waveguides: https://meep.readthedocs.io/en/stable/FAQ/
7. P. B. Johnson, R. W. Christy, *Optical Constants of the Noble Metals*, Phys. Rev. B 6, 4370 (1972), https://doi.org/10.1103/PhysRevB.6.4370
8. Vector leaky fiber Maxwell FEM context: https://doi.org/10.1137/24M1651605

**Audit statement:** As of this plan, **zero** new open-source Maxwell simulations have been executed or scientifically certified; no journal-ready physics numbers were generated. Complete A–E before representing the paper as FEM-validated.
