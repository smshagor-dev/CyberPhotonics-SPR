# Exploratory open-source measured-Au finite-domain study

Status: UNQUALIFIED numerical feasibility; this document freezes protocol, not sensor performance.

## Optical sources and limitations

- P. B. Johnson and R. W. Christy, Optical Constants of the Noble Metals, Physical Review B 6, 4370 (1972), DOI https://doi.org/10.1103/PhysRevB.6.4370. The repository preserves a SHA256-traced CC0 transcription of reference bulk-film n,k data, which is not a measurement of any fabricated PCF Au film.
- Chao Liu et al., Numerical analysis of a photonic crystal fiber based on a surface plasmon resonance sensor with an annular analyte channel, Optics Communications 382 (2017), DOI https://doi.org/10.1016/j.optcom.2016.07.031.
- Statistical Modelling of Photonic Crystal Fibre Based Surface Plasmon Resonance Sensors Resonant Peak Wavelength for Tolerance Studies (2021), https://pmc.ncbi.nlm.nih.gov/articles/PMC8512521/, documents pitch and Au-thickness design sensitivity.

The current one-ring 2D PCF demonstrator is NOT an exact CAD reconstruction of either publication. Published precedent motivates the physical problem but does not validate this particular topology. Literature-grounded CAD remains a separate, uncompleted task.

## Frozen domain experiment

Question: How do raw full-vector Femwell complex eigenvalues change when the square outer computational domain expands, with internal sensor geometry and optical material law fixed?

- Geometry: pitch 2.0 um, hole diameter/pitch 0.35, one ring, analyte-channel center (0.85,0) um, channel radius 0.32 um, Au shell 0.05 um.
- Outer square halfwidth: 5.5 um versus 7.0 um; re-mesh each domain. This combines finite-boundary and remeshing effects; it is NOT pure domain or PML convergence.
- Gold interface mesh target size 0.025 um for both domains.
- Au source: Johnson-Christy measured-reference optical n,k; silica: three-term Sellmeier.
- Wavelengths: 0.75, 0.80 um; analyte RIs: 1.35, 1.36; eigenmodes requested per sample: 2.
- Expected 4 authentic eigen-solves and 8 raw eigenpairs per domain, 8 solves/16 eigenpairs total.
- Preserve native Gmsh mesh, exact protocol, geometry, complex neff, E/H NPZ field coefficients, SHA256, solver/version, and explicit failures.
- Compare paired eigenvalues using a Hungarian minimum-complex-distance assignment. This is a diagnostic, NOT proof of field-based mode identity. Numerical change in imaginary neff is not verified optical loss here.

Reproduce with the exact code commit, FEM dependencies and GitHub Actions artifact ID. Both output folders must pass the measured-reference Au archive audit. Comparison refuses missing/corrupted data, changed internal sensor geometry, changed RI/wavelength/material protocol, different solver versions, or identical meshes after domain change.

Local commands after installing open-source FEM/Gmsh:

    python scripts/probe_femwell_ri_wavelength.py --out /tmp/domain5p5 --wavelengths-um 0.75,0.8 --analyte-ris 1.35,1.36 --num-modes 2 --gold-source johnson_christy_1972 --domain-halfwidth-um 5.5 --strict
    python scripts/probe_femwell_ri_wavelength.py --out /tmp/domain7p0 --wavelengths-um 0.75,0.8 --analyte-ris 1.35,1.36 --num-modes 2 --gold-source johnson_christy_1972 --domain-halfwidth-um 7.0 --strict
    python scripts/compare_domain_grids.py /tmp/domain5p5 /tmp/domain7p0 --out /tmp/DOMAIN_DIAGNOSTIC.json

## Explicit scientific blockers

No outgoing-wave PML, no convergence over multiple gold-mesh resolutions, no physically verified core/SPP branches, no bracketed spectral peak/FWHM, no independent Maxwell solver cross-check, no fabricated sensor measurement, no exact peer-reviewed topology, and no geometry-disjoint source data for inverse-model training.

The 2-point spectrum cannot establish resonance, wavelength sensitivity, optical FOM, or sensor loss. These results are unqualified exploratory numerical domain/remesh diagnostics only. Never label them COMSOL. All physical-performance and publication flags remain FALSE.
