# Research Reproducibility and Release Protocol

CyberPhotonics-SPR treats reproducibility metadata as part of the scientific evidence, not as a documentation afterthought.

## 1. Create an isolated environment

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e ".[dev,onnx]"
```

For edge deployment add `edge`; for serial acquisition add `hardware`; COMSOL automation requires `comsol` and a licensed compatible installation.

## 2. Run validation before recording evidence

```bash
ruff check . --select E9,F63,F7,F82
python -m compileall -q src main.py scripts
pytest -q -W error
python scripts/verify_release.py
```

The CI matrix repeats core validation on Python 3.10–3.13 and separately validates the Edge/LiteRT path.

## 3. Capture every experiment

```bash
python scripts/create_reproducibility_bundle.py \
  --out outputs/repro/experiment_001 \
  --name experiment_001 \
  --seed 7 \
  --data data/processed/training.parquet \
  --checkpoint models/tandem.pt \
  --config configs/experiment.yaml
```

Additional artifacts use repeatable `--artifact ROLE=PATH` arguments. The bundle writes:

- `manifest.json`: experiment name, seed, config, Git commit/dirty state, artifact SHA-256 hashes.
- `environment.json`: Python/platform metadata and installed packages.
- `environment.lock.txt`: exact installed Python package versions from that run.
- `checksums.sha256`: compact artifact checksum list.
- `REPRODUCE.md`: minimal reproduction instructions.

The bundle records hashes and metadata; it does not duplicate large model or dataset files.

## 4. Container workflow

Build the tested core/ONNX research image:

```bash
docker build -t cyberphotonics-spr:research .
```

Select additional optional extras when needed:

```bash
docker build --build-arg INSTALL_EXTRAS="dev,onnx,edge,hardware" -t cyberphotonics-spr:edge .
```

The default image uses Python 3.11 and CPU PyTorch. `.devcontainer/devcontainer.json` provides the same baseline for VS Code-compatible development environments.

## 5. Release validation

```bash
python scripts/verify_release.py
```

For a version tag, require tag/project consistency:

```bash
python scripts/verify_release.py --expected-version 0.2.0
```

The validator checks required research metadata, `pyproject.toml`/package/CFF version consistency, citation fields, and that raw/generated runtime artifacts have not accidentally been committed.

## 6. Tag/release workflow

Pushing a `v*` tag triggers `.github/workflows/release-validation.yml`. It validates metadata, runs warning-as-error tests, builds wheel/sdist packages, captures release evidence, and uploads the evidence/build outputs as GitHub Actions artifacts. It does **not** publish to PyPI or create a public GitHub release automatically.

## 7. DOI / Zenodo-ready release

Before depositing a release:

1. Generate the scientific validation and reproducibility bundles from the actual evidence source.
2. Confirm `CITATION.cff`, `MODEL_CARD.md`, and `DATASET_CARD.md` match the released version.
3. Tag the exact commit and require the release-validation workflow to pass.
4. Archive or deposit datasets/models separately when they are too large for Git; record their immutable hashes/DOIs in the release notes or publication.
5. Connect the GitHub repository to a DOI service such as Zenodo only when the release contents are final.

A DOI identifies a frozen release; it does not turn synthetic or simulated evidence into experimental validation.
