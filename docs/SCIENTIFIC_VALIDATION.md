# Scientific Validation Pack

This layer turns CyberPhotonics-SPR model outputs into reviewer-friendly, reproducible evidence. It does **not** convert synthetic data into physical evidence; COMSOL or experimental spectra remain required for physical claims.

## What the validation pack measures

1. **Fixed-geometry RI sweeps**
   - Fits resonance wavelength versus analyte RI independently for every geometry.
   - Reports fitted wavelength sensitivity, FOM, RI span, and linearity R².
   - Keeps this fitted slope separate from the local finite-difference sensitivity used as an ML label.

2. **Bootstrap uncertainty**
   - Reports deterministic percentile confidence intervals for mean sensitivity and FOM.
   - Uses a fixed seed so manuscript tables can be regenerated.

3. **Leakage-resistant baseline**
   - Trains a Ridge regression baseline using geometry + analyte RI.
   - Uses the same geometry-grouped validation principle as the neural pipeline.
   - Gives the tandem forward surrogate a transparent classical baseline.

4. **Tandem target satisfaction**
   - Evaluates whether inverse-generated geometry, when passed back through the frozen forward surrogate, reproduces the requested physical sensing metrics.
   - This is more meaningful for one-to-many inverse design than direct geometry R² alone.

5. **Fabrication validity**
   - Reports total valid/violation rates and per-variable bound violations.
   - Checks air-hole diameter versus pitch.
   - Reports both raw inverse output validity and post-projection validity, so hard clamping cannot hide poor unconstrained generation.

6. **MC-dropout uncertainty**
   - Repeats inverse generation with trained dropout active.
   - Reports mean geometry and sensing-metric standard deviation across stochastic passes.

7. **Provenance**
   - Stores SHA-256 hashes for the dataset and checkpoint.
   - Stores seeds, bootstrap count, MC sample count, and canonical column order.

## Run a validation pack

Dataset-only validation:

```powershell
python scripts/run_validation_pack.py `
  --data data/processed/synthetic.parquet `
  --out outputs/validation
```

Model-aware validation:

```powershell
python scripts/run_validation_pack.py `
  --data data/processed/synthetic.parquet `
  --checkpoint models/tandem.pt `
  --out outputs/validation `
  --bootstrap-resamples 5000 `
  --mc-samples 64 `
  --seed 7
```

Generated artifacts:

```text
outputs/validation/
  fixed_geometry_sweeps.csv
  summary.json
  provenance.json
  validation_report.md
  resonance_shift.png
  sensitivity_distribution.png
  model_comparison_r2.png   # when a checkpoint is supplied
```

Plots are exported at 300 dpi.

## Physics-loss ablation

The ablation runner compares:

- `physics_informed`: configured fabrication/physics penalties enabled.
- `no_physics_penalty`: all inverse-design penalty weights disabled.

It repeats both variants across deterministic seeds and records forward R², inverse target-satisfaction R², and raw fabrication-validity rates.

```powershell
python scripts/run_ablation_study.py `
  --data data/processed/synthetic.parquet `
  --out outputs/ablation `
  --seeds 7,17,29 `
  --epochs 50 `
  --device cpu
```

Generated artifacts:

```text
outputs/ablation/
  ablation_runs.csv
  ablation_summary.json
  models/
    physics_informed_seed_*/
    no_physics_penalty_seed_*/
```

For manuscript-grade results, increase the number of seeds and training epochs, archive the exact dataset/checkpoint hashes, and run the same validation on verified COMSOL or experimental data.

## Recommended manuscript evidence order

1. Data provenance and fixed-geometry RI sweep definition.
2. Resonance-shift linearity and fitted sensitivity with confidence intervals.
3. Classical Ridge baseline versus neural forward surrogate.
4. Inverse target-satisfaction metrics.
5. Fabrication-validity rate before and after projection.
6. Physics-loss ablation across seeds.
7. MC-dropout uncertainty.
8. COMSOL closed-loop confirmation of selected inverse designs.
9. Real hardware / spectrometer edge benchmark.

Synthetic outputs should be labeled as software/pipeline validation, never as experimental confirmation.
