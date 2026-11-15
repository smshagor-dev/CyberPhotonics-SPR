# Johnson–Christy Au measured optical constants: source-bound reference

**This is optical material input evidence only. It does not establish the actual physical properties of any fabricated 50 nm Au PCF-SPR sensor.**

- Publication: P. B. Johnson and R. W. Christy, Optical Constants of the Noble Metals, Phys. Rev. B 6, 4370–4379 (1972), DOI https://doi.org/10.1103/PhysRevB.6.4370.
- CC0 public-domain database transcription: https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/Au/nk/Johnson.yml.
- Recorded source Git blob SHA: cf2f1b2c490d60bea818096920e323da94bbb540.
- Local source-bound package data: src/sprpcf/simulation/data/au_johnson_christy_1972.csv.
- 49 room-temperature measured n,k rows across 0.1879–1.9370 μm, with exact knots preserved, no extrapolation or smoothing beyond piecewise linear interpolation in n and k.
- Temporal convention exp(-i omega t); epsilon_r=(n+i*k)^2 and imaginary epsilon >0 checked for passive Au.
- Source filename, DOI, SHA256 and interpolation limitations are machine-readable via sprpcf.simulation.materials_reference.au_reference_manifest().

## API

    from sprpcf.simulation.materials_reference import (
        au_johnson_christy_nk, au_johnson_christy_permittivity,
        au_reference_manifest,
    )

    eps_at_800_nm = au_johnson_christy_permittivity(0.8)
    source = au_reference_manifest()

These functions intentionally reject wavelengths beyond the actual measurements. The CSV is packaged in Python wheel and source distributions.

## Scientific use and limitations

This independent measured optical-source table **replaces a major provenance gap** for bulk-gold reference material. The previous repository's fitted Drude–Lorentz Au model is useful for a model sensitivity comparison but must not be mislabeled measured optical data.

Important: Au films 15–80 nm thick may differ from the source bulk-film constants due to substrate, annealing, grain size, roughness, deposition method, temperature and other real-world conditions. Therefore, using Johnson–Christy optical constants is a documented **material assumption**, not a measurement of the hypothetical gold-coated sensor.

No gold-film sensitivity, plasmonic peak, FOM or numerical-paper readiness is established by importing the optical constants. After this PR: integrate as an optional/explicit solver input in exploratory PCF FEM scripts, rerun full raw eigenmode and mesh studies, preregister realistic material uncertainty comparisons, then qualify full Maxwell/PML physics independently.

All COMSOL and device evidence classification rules remain unchanged.
