# Exploratory full-vector PCF wavelength × analyte-RI matrix (UNQUALIFIED)

An **actual** CPU-only 2D finite-element frequency-domain vector Maxwell calculation using Femwell on a single frozen mesh, rather than a synthetic resonance generator. The code varies the dispersive Au/silica material tensor with wavelength and changes analyte permittivity at fixed geometry, and archives *both* E and H eigenfield coefficient vectors for every unselected mode and sample.

## Run

    python scripts/probe_femwell_ri_wavelength.py --out outputs/open_physics/vector_grid --wavelengths-um 0.75,0.8,0.85 --analyte-ris 1.35,1.36,1.37 --num-modes 2 --strict

Optional dependencies: Gmsh runtime, femwell, meshio, shapely, scikit-fem. These are installed by GitHub Actions workflow pcf-ri-wavelength-grid.yml.

This performs 9 genuine full-vector Maxwell eigenmode solves, computes 18 **raw** complex effective indices, archives raw E/H vector coefficients for each point/mode, and compares selected neighbor pairs by the Femwell full-vector electric/magnetic Poynting-overlap diagnostic. The sample grid is a feasibility design, **not** a published sensor study. The 3-point wavelength range is too sparse to establish a resolved resonance/FWHM and cannot support a sensing FOM.

## Outputs

- topology.json and experiment_protocol.json: explicitly frozen **exploratory** geometric inputs and wavelength/RI grid; no literature provenance.
- pcf_frozen_exploratory_grid_mesh.msh: common real Gmsh mesh for all 9 points.
- raw_eigenmode_grid.csv: wavelength, RI, **native untracked mode index**, real/imag n_eff, gold permittivity and silica index for each eigenpair.
- raw_vector_eh_ri{index}_wl{index}.npz: actual complex electric and magnetic vector field coefficients on the SAME finite element mesh for each sample.
- vector_mode_overlap_diagnostics.json: full-vector neighbor overlap matched with the Hungarian algorithm. An overlap-based assignment by itself **does not prove core or SPP modal identity or physical branch tracking**.
- GRID_READINESS.json: source identity, 9 real solver calls, SHA256 hashes, errors, output counts and all explicit scientific flags FALSE.

## Not yet validated

No PML/outgoing boundary, no independent exact-source topology, no independently archived measured Au optical constants, no mode field localization classification, no spectral resonance, no FWHM, no propagation loss calibration, no physical sensitivity, no independent Meep/NGSolve challenge, no mesh/PML/wavelength convergence accepted, no physics-qualified ML training data. The nominally positive imaginary n_eff values are not sufficient to establish physical confinement loss.

Never replace the existing COMSOL-only evidence registry with fabricated or relabeled Femwell data. This pilot remains entirely outside the journal results table.

## Next physical stages

1. Select a documented published geometry with full dimensions, ensure mode confinement and Au/analyte field localization.
2. Use a proper open-source full-vector eigenmode solver with validated outgoing-wave PML; test boundary/domain/mesh convergence and film resolution.
3. Track the same physically identified core-like mode across a **much denser** adaptive wavelength grid and fixed-geometry RI set, validate a resolved plasmonic resonance and FWHM.
4. Verify signed loss and units, independent analytical controls and Meep/alternate FEM comparison.
5. Only then train and verify inverse design on geometry-disjoint fully numerical reference data and report uncertainty, ablations, compute cost and limitations.

The intended research manuscript remains **NOT READY** until those gates pass.
