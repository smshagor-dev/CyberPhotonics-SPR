# Advanced AI Design: Ensemble, Calibration, OOD and Pareto Search

Phase 3 upgrades CyberPhotonics-SPR from one-shot inverse generation to an auditable multi-objective design workflow.

## What is new

The advanced design layer adds four independent safety/quality signals before COMSOL validation:

1. **Latent candidate search** — generate many inverse designs per sensing target instead of trusting one neural output.
2. **Pareto ranking** — separately minimize normalized sensitivity error, FOM error, resonance-wavelength error, fabrication projection distance, and OOD score.
3. **Calibrated uncertainty** — estimate metric error radii from the leakage-resistant held-out geometry split using finite-sample conformal residual quantiles.
4. **Domain awareness** — compute a Mahalanobis distance in the standardized geometry + analyte-RI space and normalize it by a training-domain quantile. `ood_score <= 1` is inside the calibrated reference envelope.

When the checkpoint contains a forward ensemble, the prediction interval combines the held-out conformal residual radius with per-candidate ensemble disagreement. A single-forward checkpoint still works; ensemble disagreement is then zero and the conformal/OOD layers remain active.

## 1. Upgrade a tandem checkpoint with a deep forward ensemble

```powershell
python -m sprpcf.ml.ensemble `
  --checkpoint models/tandem.pt `
  --data data/processed/training.parquet `
  --out models/tandem_ensemble.pt `
  --members 5 `
  --epochs 50 `
  --device auto
```

The original forward surrogate remains ensemble member 0. Additional members use independent initialization seeds but the same leakage-resistant grouped split and checkpoint scalers. The command fails if the reference dataset/scalers do not match the checkpoint.

## 2. Generate Pareto-ranked designs

```powershell
python scripts/run_multiobjective_design.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --reference-data data/processed/training.parquet `
  --out outputs/multiobjective `
  --candidates-per-target 128 `
  --confidence 0.95 `
  --latent-scale 0.10 `
  --seed 7
```

Outputs:

```text
outputs/multiobjective/pareto_candidates.csv
outputs/multiobjective/selected_designs.csv
outputs/multiobjective/calibration.json
```

Each candidate records physical and raw inverse geometry, forward-predicted sensing metrics, normalized target errors, ensemble standard deviation, calibrated interval width, OOD score, fabrication projection distance, Pareto rank, composite risk, confidence score, and selection status.

The chosen design is the lowest-risk candidate on the best available Pareto front. Hard geometry clamping is not hidden: `raw_*` columns and `fabrication_projection_distance` preserve how much projection was required.

## 3. Run the advanced design directly into the physics closed loop

```powershell
python scripts/run_advanced_closed_loop.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/training.parquet `
  --backend comsol `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config sweep.example.yaml `
  --out outputs/advanced_closed_loop/iteration_001 `
  --candidates-per-target 128 `
  --confidence 0.95 `
  --ri-span 0.04 `
  --ri-points 5 `
  --retrain
```

The bridge sends only the Pareto-selected geometry for each target into the fixed-geometry RI sweep used by Phase 2. COMSOL remains the physics authority: a high AI confidence score does **not** bypass the sensitivity/FOM/resonance/linearity acceptance gates.

For CI/software testing without a licensed COMSOL installation, use `--backend synthetic`. Synthetic evidence remains labeled software-only and must not be reported as experimental or COMSOL physics evidence.

## Interpretation

- **Pareto rank 0**: no other candidate is at least as good on every objective and strictly better on one.
- **OOD score <= 1**: candidate lies inside the calibrated training-domain Mahalanobis envelope.
- **Confidence score**: ranking aid derived from target error, fabrication projection, calibration-domain excess, and target interval coverage. It is not a probability of physical success.
- **Prediction interval**: held-out conformal residual radius plus ensemble disagreement. It expresses surrogate uncertainty; real COMSOL/experimental validation is still required.

## Research protocol recommendation

Before manuscript experiments, freeze the ensemble member count, confidence level, candidates per target, latent scale, closed-loop acceptance thresholds, random seeds, reference dataset hash, and COMSOL model/config hashes. Do not tune them after seeing test outcomes; this avoids post-hoc threshold selection and keeps the evidence reproducible.
