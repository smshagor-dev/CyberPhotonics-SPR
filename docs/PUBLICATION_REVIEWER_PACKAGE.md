# Publication and Reviewer Evidence Package

CyberPhotonics-SPR can assemble completed research artifacts into a reviewer-facing package without inventing missing results or upgrading software-only evidence into physical or experimental evidence.

The package is designed for manuscript submission support, internal review, supplementary-material preparation, and release archiving.

## 1. What the package contains

A generated package can include:

- scientific validation tables and figures,
- physics-loss ablation outputs,
- Pareto/inverse-design evidence,
- COMSOL closed-loop verification outputs,
- hardware/runtime evidence,
- reproducibility manifests and environment snapshots,
- release validation metadata,
- citation, model-card, dataset-card, license, and reproducibility documentation,
- a machine-readable artifact manifest,
- a claims-to-evidence matrix,
- SHA-256 checksums,
- reviewer quick-start guidance,
- generated release notes.

Large model binaries and COMSOL `.mph` files are not copied by default. Their provenance should be retained through hashes and the original experiment records.

## 2. Evidence classes

Every supplied source is assigned one of the following evidence classes:

| Evidence class | Appropriate interpretation |
|---|---|
| `software_only` | synthetic data, replay, software execution, methodology checks |
| `surrogate_model` | learned-model or inverse-design evidence |
| `comsol_physics` | COMSOL-backed numerical-physics verification |
| `experimental_sensor` | measured sensor evidence from a documented experiment |
| `device_benchmark` | latency/memory/throughput measured on the exact target device |
| `reproducibility` | seeds, Git state, environment, configuration, hashes |
| `release` | release validation, citation, packaging, licensing metadata |

The packager is conservative by default.

- validation defaults to `software_only`,
- design output defaults to `surrogate_model`,
- hardware output defaults to `software_only`,
- closed-loop output uses `iteration_manifest.json` when available and recognizes `comsol_physics`,
- experimental or target-device status must be supplied deliberately when the evidence is genuinely measured.

Do **not** label replayed, synthetic, workstation, or simulated results as `experimental_sensor` or `device_benchmark`.

## 3. Build a package from completed research artifacts

Example:

```powershell
python scripts/build_reviewer_package.py `
  --validation-dir outputs/validation `
  --ablation-dir outputs/ablation `
  --design-dir outputs/multiobjective `
  --closed-loop-dir outputs/closed_loop/iteration_001 `
  --reproducibility-dir outputs/repro/experiment_001 `
  --out outputs/reviewer_package
```

If the closed-loop manifest records `backend = comsol` / `evidence_class = comsol_physics`, the package preserves that classification.

For genuine measured sensor evidence:

```powershell
python scripts/build_reviewer_package.py `
  --validation-dir outputs/validation `
  --hardware-dir outputs/experimental_sensor_session `
  --hardware-class experimental_sensor `
  --reproducibility-dir outputs/repro/experimental_session `
  --out outputs/reviewer_package_experimental
```

Use `--hardware-class device_benchmark` only when latency, memory, or throughput were actually measured on the exact device named by the evidence.

## 4. Reviewer entry points

The generated package contains:

```text
REVIEWER_GUIDE.md
CLAIMS_MATRIX.md
CLAIMS_MATRIX.csv
artifact_index.csv
manifest.json
checksums.sha256
RELEASE_NOTES.md
evidence/
release/
```

Recommended inspection order:

1. `REVIEWER_GUIDE.md`
2. `CLAIMS_MATRIX.md`
3. `artifact_index.csv`
4. the relevant evidence directories
5. `manifest.json`
6. `checksums.sha256`

The claims matrix reports `not supplied` when COMSOL, experimental, or target-device evidence is absent. Missing evidence is never replaced with a guessed value.

## 5. Deterministic polished demo

A complete software-only demo can be generated with one command:

```powershell
python scripts/run_publication_demo.py `
  --out outputs/publication_demo `
  --samples 24 `
  --wavelengths 128 `
  --bootstrap-resamples 500 `
  --seed 7
```

Outputs include:

```text
outputs/publication_demo/
├── DEMO_INDEX.html
├── README.md
├── demo_summary.json
├── demo_synthetic.parquet
├── validation/
├── reproducibility/
└── reviewer_package/
```

Open `DEMO_INDEX.html` for the static reviewer/demo landing page.

The demo is explicitly labeled `software_only`. It demonstrates the publication workflow but does not provide COMSOL validation, laboratory measurements, detection limits, fabricated-sensor results, or target-device benchmarks.

## 6. Release automation

The tag-time `Research Release Validation` workflow:

1. validates release metadata,
2. runs core tests with warnings as errors,
3. builds wheel and source distributions,
4. captures a reproducibility bundle,
5. builds the deterministic publication demo,
6. uploads the demo reviewer package together with the release evidence.

The workflow does not publish a DOI, fabricate experimental evidence, or claim that a tagged software release has been experimentally validated.

GitHub release-note categories are configured in `.github/release.yml` for cleaner generated release notes.

## 7. Submission use

For a manuscript or revision, the reviewer package can be attached as supplementary computational evidence after checking journal file-size and file-type requirements.

Before submission, verify that:

- the Git commit matches the manuscript,
- dataset/model/configuration hashes match the reported experiments,
- COMSOL evidence uses the correct `.mph` model, materials, mesh, boundaries, and units,
- experimental evidence includes the measurement protocol and calibration provenance,
- device benchmarks were measured on the exact reported device,
- no synthetic/demo value is presented as a laboratory result,
- the claims matrix matches the wording used in the manuscript.

## 8. Scientific boundary

The package is an evidence organizer and provenance tool. It does not establish correctness by itself.

A hash proves artifact identity, not scientific validity. A COMSOL label is useful only when the underlying model is physically appropriate. An experimental label is useful only when the measurement protocol is valid and documented. Peer review remains necessary.
