# Dataset Card — CyberPhotonics-SPR

## Dataset types

CyberPhotonics-SPR supports three explicitly distinguished evidence sources:

- **Synthetic fixed-geometry RI sweeps** for software/pipeline validation.
- **COMSOL-generated spectra** from a user-supplied and unit-validated physics model.
- **Experimental sensor spectra** acquired or replayed through the calibrated hardware runtime.

These sources must not be relabeled as one another. Synthetic output is never sufficient by itself for an experimental optical-performance claim.

## Core schema

Design/condition columns include `pitch_um`, `d_over_lambda`, `metal_thickness_nm`, `channel_radius_um`, and `analyte_ri`. Spectral records include wavelength and confinement-loss data. Derived targets include resonance wavelength, sensitivity, FOM, FWHM, and related validation metrics when available.

## Split policy

Rows sharing the same physical geometry belong to the same split. This prevents different RI points from one fixed-geometry sweep from leaking into train and validation sets. Duplicate RI values inside a fixed-geometry sensitivity sweep are rejected.

## Provenance

Repository dataset writers generate metadata sidecars with source information and SHA-256 content hashes. Research experiments should additionally run `scripts/create_reproducibility_bundle.py` so dataset hashes, model hashes, seed, configuration, Git state, and software environment are bound to the reported result.

## Quality and limitations

COMSOL wavelength/loss units must satisfy the configured unit contract. Experimental spectra require wavelength calibration, dark/reference correction where applicable, non-extrapolating resampling to the model grid, and OOD checks. Physical conclusions remain limited by simulation fidelity, sensor calibration, fabrication tolerances, environmental drift, and the coverage of the measured RI range.

## Storage policy

Raw/processed datasets are intentionally excluded from Git except for `.gitkeep` placeholders. Publishable datasets should be deposited separately with an immutable identifier (for example a DOI) and their hash recorded in the reproducibility bundle.
