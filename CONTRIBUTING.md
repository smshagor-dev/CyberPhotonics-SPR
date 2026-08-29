# Contributing to CyberPhotonics-SPR

Thank you for improving CyberPhotonics-SPR. Contributions are expected to preserve both software quality and scientific traceability.

## Development principles

1. Keep the software path reproducible and testable.
2. Preserve the distinction between synthetic, numerical-physics, experimental, and exact-device evidence.
3. Do not weaken provenance, checksums, evidence qualification, or release gates to make a test pass.
4. Keep optional integrations optional; a missing COMSOL, hardware, XAI, dashboard, or edge extra must not break unrelated core workflows.
5. Prefer small, reviewable changes with tests over broad unverified rewrites.

## Local setup

Python 3.10-3.13 is supported. Python 3.11 is the recommended development environment.

### Windows PowerShell

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,onnx]"
python -m sprpcf.runtime.doctor
```

Install additional capabilities only when needed:

```powershell
python -m pip install -e ".[edge]"       # TensorFlow/LiteRT edge path
python -m pip install -e ".[dashboard]"  # legacy Streamlit research dashboard
python -m pip install -e ".[hardware]"   # serial hardware support
python -m pip install -e ".[comsol]"     # COMSOL mph bridge
python -m pip install -e ".[xai]"        # SHAP explainability
```

The native desktop control center is launched with:

```powershell
python main.py
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,onnx]"
python -m sprpcf.runtime.doctor
```

## Before changing code

- Search existing issues and pull requests to avoid duplicate work.
- For scientific changes, identify the evidence class affected: synthetic, COMSOL/numerical physics, experimental sensor data, or exact-device benchmark.
- For public API, model schema, evidence format, or release-contract changes, explain compatibility impact in the pull request.
- Security vulnerabilities should follow `SECURITY.md`, not the public issue workflow.

## Quality checks

Run the smallest relevant test set while developing, then run the broad checks before opening a pull request.

```bash
ruff check . --select E9,F63,F7,F82
python -m compileall -q src main.py scripts
pytest -q -W error --ignore=tests/test_edge_deployment.py --ignore=tests/test_full_pipeline.py
```

For edge work:

```bash
pytest -q -W error tests/test_edge_deployment.py tests/test_full_pipeline.py
```

For native desktop work, use an environment with PySide6 installed and run:

```bash
pytest -q -W error tests/test_desktop_gui.py tests/test_main_dashboard_default.py
```

Release-oriented changes should also preserve:

```bash
python scripts/check_system_readiness.py --profile release --expected-version 1.0.0rc1 --strict
```

## Scientific integrity requirements

A contribution must not:

- describe synthetic data as measured or experimental;
- describe workstation/CI latency as target-device performance;
- treat uncertainty/OOD scores as fabrication-success probabilities;
- silently replace failed COMSOL samples with accepted values;
- calculate sensitivity from a geometry-changing sweep when the contract requires fixed geometry;
- remove provenance or hash checks merely to accept an artifact.

If a manuscript-facing result depends on real physical evidence, provide the appropriate qualified evidence or leave the claim explicitly marked as pending.

### Checkpoint security

Current tandem checkpoints are written in a format compatible with PyTorch `weights_only=True` loading. Older checkpoints that require unrestricted pickle deserialization are intentionally rejected by current loaders. Regenerate those checkpoints with the current training pipeline rather than disabling the safe loader.

## Data, models, and generated artifacts

Do not commit large generated datasets, model weights, temporary caches, private `.mph` files, or local runtime outputs unless the repository intentionally tracks that artifact and the pull request explains why.

Prefer deterministic generation scripts, compact fixtures, metadata sidecars, and checksums. Test fixtures should be the smallest artifacts that exercise the behavior under test.

## Code style

- Follow the existing project structure and naming conventions.
- Keep functions focused and typed where practical.
- Avoid shell interpolation for user-provided values; pass subprocess arguments as lists.
- Keep optional heavy imports lazy when possible.
- Surface actionable errors rather than raw dependency tracebacks in user-facing launch paths.
- Add or update tests for fixes and behavior changes.

## Commits and pull requests

Use clear, scoped commit messages, for example:

```text
feat(edge): add calibrated streaming benchmark
fix(desktop): preserve light-mode chart contrast
test(evidence): reject mismatched artifact hashes
docs(security): document private reporting flow
```

A pull request should include:

- what changed and why;
- affected scientific/software boundary;
- validation performed;
- screenshots for meaningful desktop UI changes;
- migration or compatibility notes where relevant;
- explicit limitations or evidence still missing.

Keep unrelated formatting or generated-file churn out of the same pull request.

## Review and merge expectations

All automated checks relevant to the change should pass. Review comments should be resolved rather than hidden by unrelated rewrites. Maintainers may request additional evidence or tests for changes that affect numerical results, evidence qualification, hardware claims, packaging, or release readiness.

## Documentation

User-visible behavior, new commands, new optional dependencies, evidence contracts, and release workflow changes should be reflected in the relevant README/docs files in the same pull request.

## License

By submitting a contribution, you agree that your contribution is provided under the repository's Apache-2.0 license unless explicitly stated otherwise for a separately licensed third-party component.
