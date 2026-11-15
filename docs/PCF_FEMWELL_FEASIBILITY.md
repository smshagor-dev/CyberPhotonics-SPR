# Experimental 2D PCF Femwell feasibility (UNQUALIFIED)

This is an opt-in genuine CPU-only **full-vector waveguide eigenmode calculation attempt** using [Femwell](https://helgegehring.github.io/femwell/photonics/examples/waveguide_modes.html), [Gmsh](https://gmsh.info/), scikit-fem and Shapely. It is the first practical probe toward a COMSOL-free numerical Maxwell solver, **not** a source-verified research result.

## Status and hard limitations

- A reconstructable 2D transverse geometry is generated: one triangular air-hole ring (six holes), silica host, offset analyte channel and 50 nm Au shell, with parameters explicitly frozen to JSON.
- **This geometry is an exploratory feasibility demonstrator**. It is not claimed to match the true PCF topology in a published source or the manuscript.
- Au uses a fitted Drude-Lorentz approximation already present in the repo, not an independently archived Johnson/Christy optical-constants table.
- Femwell solves waveguide vector modes with complex materials; the code records raw complex n_eff but **does not infer propagation loss or identify a PCF core branch**.
- No outgoing-wave PML/transparent boundary, mesh convergence, modal identity across wavelength, resolved RI sweep, sensor sensitivity, independent cross-solver comparison, or publication-ready claim is established.
- Existing COMSOL-only physics evidence rules are unchanged. Any output is labeled unqualified.

## Run

Install native Gmsh runtime libraries on Ubuntu if necessary:

    sudo apt-get install -y libglu1-mesa libgl1

Using an isolated Python environment:

    python -m pip install "femwell>=0.1,<0.2" "shapely>=2" "meshio>=5.3" pytest
    PYTHONPATH=src pytest -q -W error tests/test_pcf_femwell_feasibility.py
    python scripts/probe_femwell_pcf.py --out outputs/open_physics/femwell_pilot --num-modes 2 --strict

The optional GitHub Actions workflow pcf-femwell-feasibility.yml executes an actual solver attempt and archives the raw mesh/metadata/eigenvalues or FEMWELL_BLOCKER.json on failure. Workflow success means **the pilot executed**, not that its mode is physically validated. Do not use this trial as a paper sensor figure.

## Planned experimental validation

1. Verify mesh region tags, lossless step-index reference, Au planar SPP/metal-film benchmarks and vectorial field components. Reject sign/mode inconsistencies before claiming a PCF core mode.
2. Select a **cited specific PCF-SPR topology**; replace this provisional channel geometry and provide all geometry/material sources.
3. Implement proper PML/absorbing boundaries and PML/domain-size sensitivity. Femwell's default finite box cannot establish physically correct confinement loss without this.
4. Commit a **pre-registered** mesh/spectral/RI plan and validate at least three mesh refinements and repeat fresh-process solver runs.
5. Extract physical wavelength-varying loss spectra from the independently tracked core branch, frozen RI sweeps and detect credible resonance before calculating sensitivity and FOM.
6. Independently cross-check with a different method (e.g. Meep or another verified FEM solver) and archive complete solver-native field/eigenpair data with hashes.
7. Only after all validation stages: use the qualified solver data for tandem inverse-design training and matched-method benchmarking; revise the manuscript.

**No synthetic or demonstration data is promoted to physical journal evidence.** The scientific blocker in issue #42 remains open until actual independent 2D Maxwell PCF-SPR validation is achieved.
