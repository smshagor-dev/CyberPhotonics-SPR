# Stage 4 — Matched-query inverse-design benchmark

This is an executable comparison of a **trained tandem generator** with two
different inverse-design baselines on the **same held-out geometry groups**:
nearest training-library retrieval and uniformly sampled valid designs. A
zero-latent tandem variant measures the effect of latent sampling diversity.

**This is a surrogate-screening benchmark only.** The ranking evaluator is
the forward model jointly trained with the tandem generator. Until independent
COMSOL fixed-geometry RI sweeps are run and reviewed, none of these results
can establish improved optical sensitivity, physical sensor accuracy, real
generalization or independently verified design superiority.

## Preflight and source contract

1. Supply a dataset with complete finite geometry, RI and target labels.
   Every geometry must have at least two distinct RI samples; at least eight
   distinct geometries are required. The dataset must carry a same-name
   `.meta.json` sidecar with an exact SHA-256 and source label.
2. **Retrain** the tandem model using the updated training pipeline. It now
   saves the training dataset hash, exact train/validation row indices, frozen
   geometry-group identities, training seed and training epochs in the
   checkpoint. Old unbound checkpoints are deliberately rejected.
3. A fresh dataset hash, different row order, mismatching train/validation
   membership, missing epochs, duplicate geometry/RI or unqualified synthetic
   input fails closed. A checkpoint metadata declaration does **not** by
   itself prove that neural training actually occurred.
4. The held-out set is the training pipeline's **validation split**. It is not
   an independent untouched external test. If model decisions were informed
   by this split, the reported results cannot be used as unbiased final
   performance evidence. An independent external cohort remains required.
5. Select one central-RI target per held-out geometry, making a geometry the
   paired bootstrap unit. Every method sees the same targets and the same
   independent sensor geometry groups.

## Run a real trained-model SOFTWARE example

```bash
python scripts/generate_synthetic_dataset.py \
  --samples 24 --wavelengths 64 --seed 7 \
  --out data/processed/stage4_demo.csv

python -m sprpcf.ml.train_tandem \
  --data data/processed/stage4_demo.csv \
  --device cpu --epochs 2 --batch-size 32 --seed 7 \
  --out models/stage4_demo.pt --onnx-out models/stage4_demo.onnx

python scripts/run_inverse_benchmark.py \
  --data data/processed/stage4_demo.csv \
  --checkpoint models/stage4_demo.pt \
  --out outputs/research/stage4_demo \
  --seeds 7,17,29 --candidates-per-target 16 \
  --bootstrap-resamples 500 --allow-software-only
```

The values in this synthetic demo are not actual optical measurements or
independent COMSOL results. The test suite also creates intentionally random-
weight checkpoints to exercise consistency code; those fixtures are not
scientific results. Use real trained checkpoints for research.

For independently generated data, omit `--allow-software-only`; note that
a source label `comsol` still remains **declared_physical_unqualified** unless
validated separately using Stage 2 archives and review.

## Budget and methods

| Method | Candidate generation | Same ranking queries per target? |
|---|---|---|
| `tandem_latent` | Trained inverse generator, seeded latent pool | Yes |
| `training_retrieval` | Closest distinct training geometries based on known targets/RI | Yes |
| `uniform_random` | Seeded uniform samples inside fabrication bounds | Yes |
| `tandem_zero_latent` | Zero-latent repeated candidate (diversity ablation) | Yes |

Each method submits exactly K projected candidate geometries to the same
frozen forward surrogate. All chosen designs are ranked by the **mean absolute
normalized surrogate-predicted error** over sensitivity, FOM and resonance.
The true loss/sensitivity of the resulting designs is **unknown** without
independent solver checks. Retrieval scores can use archived training labels
and hence has a different generation-cost structure. Candidate-query parity
is **not total compute, wall-clock or physics-solver budget parity**.

## Generated result files

- `split_membership.csv`: every row, frozen train/validation group and split hash.
- `validation_targets.csv`: one RI/metric target per held-out geometry.
- `all_candidates.csv`: K scored geometries for each method/target/generation seed.
- `selected_designs.csv`: selected geometry and explicitly
  `surrogate_pred_*` / `surrogate_abs_error_*` metrics, projected-geometry
  validity and distinct candidate count.
- `paired_geometry_bootstrap.csv`: paired group bootstrap contrast vs tandem
  for the same target group, within each candidate-generation seed. Intervals
  are descriptive, not proof of superiority.
- `PHYSICS_VERIFICATION_REQUESTS.csv`: **actual work queue** of selected
  designs to verify on a separately qualified real COMSOL solver, preserving
  source target RI, geometry and desired metrics. Every row starts with
  `verification_status = not_run`.
- `INVERSE_BENCHMARK_MANIFEST.json`: immutable input/checkpoint/split hashes,
  method budgets, seed semantics, output-table SHA-256, software provenance and
  explicit unresolved physics and statistical qualifications.
- `INVERSE_BENCHMARK_REPORT.md`: clear research limitations.

## Next scientific acceptance gates

- Run a real, independently reviewed COMSOL fixed-geometry RI sweep for
  **every** selected unique design, including full wavelength-loss spectra,
  material properties, mesh convergence, model/config hashes and solver logs.
- For the exact **same** independently evaluated targets and designs, compare
  real numerical sensitivity/FOM/resonance errors, feasibility, acceptance
  rate and raw failure cases; do not silently drop unsuccessful solver runs.
- Report an honestly matched end-to-end wall-clock/runtime and physical-solver
  budget. The Stage 4 proxy query count is not sufficient.
- Freeze external untouched test geometries separately from training and
  calibration; report independent uncertainty coverage/OOD, relevant
  inverse-method baselines and **multiple training seeds**.
- Connect each final paper table and figure to a model/data/COMSOL evidence
  hash; qualify through the existing registry and Stage 2 spectral/convergence
  consistency audits. A passing audit alone is still not proof of the actual
  electromagnetic formulation.

**Submission readiness remains false** until qualified external physics
results and a scientifically independent comparison are available.
