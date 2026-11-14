# Stage 3 — Group-disjoint ML baselines, feature ablation and statistics

This stage adds **actual executable forward-prediction experiments**. It does not
claim that the proposed tandem inverse-design method is fully benchmarked or
that real COMSOL physics is established. See [Issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).

## Experiment design and anti-leakage protocol

The sensor geometry (pitch, diameter/pitch, metal thickness and channel radius)
is the **independent statistical unit**. RI observations from the same geometry
are correlated. For every seed, a GroupShuffleSplit partitions *entire sensor
geometries* into training and held-out test sets.

Each candidate receives **exactly the same split within a seed**. Feature
scalers are fit only to training rows. Results are given separately in physical
units for sensitivity (nm/RIU), FOM (1/RIU) and resonance wavelength (nm).

| Compared method | Purpose | Features |
|---|---|---|
| `ridge_full` | Classical linear baseline | Geometry + RI |
| `random_forest` | Nonlinear classical baseline | Geometry + RI |
| `ridge_no_ri` | RI-input feature ablation | Geometry only |
| `dummy_mean` | Trivial reference | Geometry + RI (mean response) |

**These are forward ML baselines and an RI feature ablation**, not the full
tandem generator or an inverse-design method comparison. Training these methods
does not, by itself, validate the proposed inverse-design contribution.

## Provenance-first input contract

Provide a real CSV/Parquet dataset plus an adjacent `<filename>.meta.json`
with `source` and an exact `sha256` of the dataset.

- `source: synthetic`: accepted **only** when `--allow-software-only` is explicitly passed.
- `source: comsol`: classed as `declared_physical_unqualified` until separate solver
  authenticity, FEM convergence and raw spectra are independently checked.
- `source: experimental` or `measured_sensor`: still **unqualified** without
  instrument/protocol/calibration checks.
- Missing or mismatched hash, nonfinite values, duplicated geometry+RI points,
  fewer than eight geometries or any geometry with fewer than two RI points
  prevent a run. The runner never silently drops invalid rows.

## Run

```bash
# Software-only demonstration (never use these values as measured optical results):
python scripts/generate_synthetic_dataset.py \
  --samples 24 --wavelengths 128 --seed 7 \
  --out data/processed/stage3_synthetic.parquet

python scripts/run_grouped_ml_study.py \
  --data data/processed/stage3_synthetic.parquet \
  --out outputs/research/stage3_demo \
  --seeds 7,17,29 \
  --test-fraction 0.25 \
  --trees 60 --bootstrap-resamples 500 \
  --allow-software-only

# For authentic externally generated data, do NOT use --allow-software-only:
python scripts/run_grouped_ml_study.py \
  --data /path/to/frozen_comsol_dataset.parquet \
  --out outputs/research/stage3_comsol_declared \
  --seeds 7,17,29,41,53 \
  --trees 120 --bootstrap-resamples 2000
```

The tool refuses to overwrite an existing nonempty output directory.

## Output contract

- `splits.csv`: exact train/test membership for **every original row**, group
  identity, seed and SHA-256 split signature.
- `predictions.csv`: original row index, geometry group, target, predicted and
  observed value, absolute error for each method and seed.
- `per_seed_metrics.csv`: per-target MAE, RMSE, R² for all methods on shared
  split, including train/test group counts.
- `paired_group_bootstrap.csv`: paired `method - ridge_full` MAE differences,
  using **mean error per held-out geometry** as the resampling unit. When fewer
  than three test geometries remain, no interval is reported.
- `aggregate_metrics.csv`: descriptive mean and SD across the repeated seed
  splits. Since splits may share geometries, this is **not** an independent
  between-experiment significance test.
- `STUDY_MANIFEST.json`: input data/sidecar hashes, exact config, Git state,
  package version, hashes of output tables, evidence class and remaining blockers.
- `STUDY_REPORT.md`: short interpretation with publication limitations.

All methods share split identity within each seed; no p-values or physical
model improvement claims are manufactured. The group bootstrap CI concerns
the observed held-out geometry groups, not independently replicated laboratory
sensors or random physical manufacturing batches.

## Separate tandem physics-loss ablation

The existing `scripts/run_ablation_study.py` can train physics-penalty and
no-penalty variants. You can pass its **actual** `ablation_runs.csv` to the new
runner via `--ablation-runs PATH`. This computes matched-seed differences for
forward R², inverse target satisfaction, fabrication validity and violations.

However, the old ablation file **does not record per-seed held-out split hashes**.
The resulting comparison is labelled `surrogate_model_only` and
`split_identity_verified=false`, even when the same seed is used. Fewer than
five matched seeds suppress CI reporting. Before scientific publication the
experimental runner must freeze train/test geometry identities and train both
ablated variants on those exact splits, with comparable compute budgets and
independent COMSOL re-verification.

## Remaining scientific blockers

1. A real COMSOL model and raw fixed-geometry RI spectra qualified by mesh,
   boundary, material and mode-selection review (Stage 2 external evidence).
2. Proposed **tandem inverse-design** vs relevant competing inverse algorithms
   on identical held-out target sets and matched physics-solver budgets.
3. Independent target verification with full recorded error distributions,
   accepted/rejected designs, failure cases, fabrication pre/post projection.
4. Calibrated geometry-group uncertainty coverage/interval widths; OOD stress
   tests and full physics-loss/ensemble/calibration ablations.
5. Scientific claim-to-evidence matrix, multiple valid seeds, archive DOIs and
   journal-specific manuscript validation.

**The scientific publication gate remains closed.** Green CI is an engineering
check that the methodology runs correctly, not a result of a real optical study.
