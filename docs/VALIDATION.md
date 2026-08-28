# Validation protocol

Use this checklist before reporting PCF-SPR results in a paper, preprint, thesis, or benchmark.

## Physics/data acceptance

1. Keep geometry fixed while varying analyte RI for wavelength-sensitivity estimation.
2. Use at least two unique RI values per geometry; three or more are preferred for central finite differences.
3. Confirm COMSOL wavelength and confinement-loss units through `wavelength_scale_to_nm` and `loss_scale_to_db_per_cm`.
4. Inspect failed COMSOL rows. Do not impute failed physics runs into reported sensitivity/FOM results.
5. Preserve the generated `.meta.json` sidecar and verify its SHA-256 before training.
6. Treat synthetic spectra as software/pipeline validation, not experimental evidence.

## ML acceptance

1. Split by base geometry, not by individual RI row.
2. Report physical-unit forward metrics and inverse-geometry metrics from the held-out geometry groups.
3. Run more than one seed for paper results and report mean plus dispersion.
4. Compare against simple baselines before claiming a neural-model advantage.
5. Verify generated designs with COMSOL or experimental measurement before treating them as valid discoveries.

## Active learning acceptance

1. Candidate targets must include analyte RI.
2. Use MC dropout only with a model trained with dropout layers.
3. Record selected target metrics, generated physical geometry, uncertainty score, and resulting COMSOL output.
4. Do not call a generic sweep after acquisition; run the selected generated geometries only.

## Edge deployment acceptance

1. Full INT8 export requires a representative calibration dataset.
2. Report validation metrics from the actual TFLite artifacts, not only the float Keras models.
3. Report RI MAE/R², resonance MAE/R², denoising PSNR/SSIM, model size, P50 latency, and P95 latency.
4. Benchmark on the intended target hardware before making real-time claims.

## Reproducibility package

For a publication release, archive:

- source commit SHA;
- COMSOL model and configuration where licensing permits;
- raw/processed dataset hashes and metadata sidecars;
- training seed(s) and command lines;
- checkpoints and exported ONNX/TFLite artifacts;
- validation output tables;
- environment/package lock or container specification.
