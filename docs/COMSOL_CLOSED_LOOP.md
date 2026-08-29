# COMSOL Closed-Loop Physics Validation

Phase 2 connects the inverse-design model to a real physics-validation loop:

```text
target sensing metrics
        |
        v
conditioned inverse model + MC dropout
        |
        v
predicted PCF-SPR geometry
        |
        v
fixed-geometry analyte-RI validation sweep
        |
        v
COMSOL solve -> spectra -> resonance/FWHM
        |
        v
independent fitted sensitivity + FOM + linearity
        |
        v
accept / reject against explicit tolerances
        |
        +---- accepted -> append verified rows -> optional retraining
        |
        `---- rejected -> preserve evidence for next active-learning iteration
```

The loop never treats a single resonance point as proof of wavelength sensitivity. Every selected design is evaluated over an odd, target-centered RI sweep, and sensitivity is independently fitted from resonance wavelength versus analyte RI.

## Target file

CSV or Parquet targets must contain:

```text
sensitivity_nm_per_riu
fom_per_riu
lambda_res_nm
analyte_ri
```

Example:

```csv
sensitivity_nm_per_riu,fom_per_riu,lambda_res_nm,analyte_ri
820.0,18.5,615.0,1.37
```

## COMSOL configuration template

The repository includes a tracked template at:

```text
configs/comsol_sweep.example.yaml
```

Copy it for the actual campaign and replace the study/expression names and unit scaling with the exact contract of the validated `.mph` model. The template is not evidence that a specific COMSOL model, material assignment, mesh, boundary condition, or unit convention is correct.

## Real COMSOL iteration

```powershell
python scripts/run_comsol_closed_loop.py `
  --checkpoint models/tandem.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/training.parquet `
  --backend comsol `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config configs/comsol_sweep.example.yaml `
  --out outputs/closed_loop/iteration_001 `
  --ri-span 0.04 `
  --ri-points 5 `
  --passes 64
```

The COMSOL configuration reuses the same study, wavelength-expression, loss-expression, and unit contract used by `sprpcf.simulation.comsol_sweep`.

## Acceptance gates

A design is accepted only when:

- every requested RI point solves successfully;
- the target RI is simulated exactly;
- fitted wavelength sensitivity is within `--max-sensitivity-error`;
- fitted FOM is within `--max-fom-error`;
- target-condition resonance wavelength is within `--max-lambda-error`;
- RI-to-resonance linearity is at least `--min-linearity-r2`.

Defaults are intentionally visible and configurable:

```text
max sensitivity absolute error: 150 nm/RIU
max FOM absolute error:           5 /RIU
max resonance absolute error:    30 nm
minimum linearity R^2:           0.95
```

For a manuscript experiment, choose tolerances before running the validation campaign and report them with the results instead of tuning them after observing outcomes.

## Uncertainty acquisition

The inverse generator uses trained MC dropout. Use:

```powershell
--uncertainty-threshold 0.05
```

to send only targets above the acquisition threshold to the physics backend. If the option is omitted, all supplied targets are validated.

The uncertainty value is stored beside every generated design and in the verification output.

## Optional retraining

Accepted COMSOL rows can immediately form the next training dataset and retrain the tandem model:

```powershell
python scripts/run_comsol_closed_loop.py `
  --checkpoint models/tandem.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/training.parquet `
  --backend comsol `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config configs/comsol_sweep.example.yaml `
  --out outputs/closed_loop/iteration_001 `
  --retrain `
  --retrain-epochs 50 `
  --retrain-device cpu
```

Retraining runs only when at least one new verified row is appended. The iteration manifest records whether retraining was requested, completed, or skipped because no new rows were accepted.

## Outputs

Each iteration writes:

```text
targets_with_geometry.csv
simulation_results.csv
simulation_results.csv.meta.json
verification.csv
augmented_dataset.csv|parquet
augmented_dataset.csv|parquet.meta.json
iteration_manifest.json
retrained_tandem.pt                  # only with successful --retrain
retrained_inverse_pcf_spr.onnx       # only with successful --retrain
```

`verification.csv` is the reviewer-facing decision table. It contains requested versus simulated sensitivity, FOM, resonance wavelength, absolute errors, linearity, uncertainty, generated geometry, acceptance status, and rejection reason.

`iteration_manifest.json` records input/output paths and SHA-256 hashes, backend type, thresholds, RI sweep settings, evidence class, counts, seed, and retraining metadata.

## Dataset append policy

Only rows from accepted targets with `status == "ok"` are eligible for augmentation. Geometry plus analyte RI is used as the observation key; duplicate points are replaced by the newly validated evidence instead of being counted twice.

The original dataset is not overwritten by default. The loop writes a new augmented dataset inside the iteration output directory.

## CI-safe synthetic backend

COMSOL cannot run on ordinary GitHub-hosted CI because it requires a licensed installation. The same orchestration can therefore be exercised with:

```powershell
python scripts/run_comsol_closed_loop.py `
  --checkpoint models/tandem.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/synthetic.parquet `
  --backend synthetic `
  --out outputs/closed_loop/smoke_test
```

This validates software orchestration only. The manifest labels it:

```text
evidence_class = software_only
```

Real COMSOL iterations are labeled:

```text
evidence_class = comsol_physics
```

Synthetic evidence must not be presented as experimental or COMSOL physics validation.

## Recommended publication workflow

1. Freeze the checkpoint, COMSOL `.mph` model, YAML unit contract, target set, thresholds, and random seed.
2. Record their hashes in the generated manifest.
3. Run the closed-loop campaign without changing acceptance thresholds.
4. Report accepted and rejected designs, not only successful examples.
5. Use the augmented COMSOL dataset for the next model version.
6. Re-run the Phase 1 validation pack and ablation study on the new checkpoint.
7. Keep experimental sensor validation separate from COMSOL validation.
