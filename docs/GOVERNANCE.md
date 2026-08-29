# Repository Governance and Security Controls

CyberPhotonics-SPR keeps software quality, research integrity, and repository administration as explicit and reviewable controls.

## Public project policies

- `SECURITY.md` defines responsible vulnerability disclosure, supported versions, security scope, secret handling, and evidence-integrity expectations.
- `CONTRIBUTING.md` defines environment setup, validation commands, scientific-integrity requirements, code style, and pull-request expectations.
- `CODE_OF_CONDUCT.md` defines community behavior and includes research-integrity expectations relevant to publication-oriented software.

## Automated maintenance

Dependabot checks both Python dependencies and GitHub Actions weekly through `.github/dependabot.yml`.

The dedicated `Security` workflow runs:

- `pip-audit` against the installed project environment;
- Bandit over Python source, scripts, and the main launcher;
- CodeQL Python analysis.

Security scanning runs for repository pushes and pull requests, on a weekly schedule, and by manual dispatch.

## Safe model checkpoints

Current tandem checkpoints are written so they can be loaded with PyTorch `weights_only=True`. NumPy scaler arrays are converted to tensors during serialization and restored to arrays after restricted loading.

Legacy checkpoints that require unrestricted pickle deserialization are rejected. Regenerate them with the current training pipeline rather than weakening the loader. This prevents an untrusted `.pt` file from becoming an arbitrary-code-execution path through the normal project loaders.

## Documentation site

`docs/index.html` is the static project homepage. `.github/workflows/docs-pages.yml` validates it on pull requests and relevant pushes. Deployment is intentionally manual until repository Pages settings are enabled, preventing an unconfigured Pages environment from making ordinary pull requests fail.

Target homepage:

```text
https://smshagor-dev.github.io/CyberPhotonics-SPR/
```

## Repository-admin settings

Branch protection, topics, homepage metadata, and initial Pages enablement are GitHub repository settings rather than tracked source files. `scripts/configure_github_repository.py` makes the intended configuration reproducible.

Preview the plan without changing GitHub:

```bash
python scripts/configure_github_repository.py
```

Apply it only with an administrator-authorized token:

```bash
GH_TOKEN=<admin-token> python scripts/configure_github_repository.py --apply
```

On Windows PowerShell:

```powershell
$env:GH_TOKEN = "<admin-token>"
python scripts/configure_github_repository.py --apply
```

The script configures:

- focused repository topics for PCF-SPR, inverse design, edge AI, COMSOL, and scientific computing;
- the project homepage URL;
- `main` branch protection that blocks force-push and deletion, requires linear history and resolved review conversations, and keeps the repository usable for a single maintainer;
- GitHub Pages with GitHub Actions as the publishing source and HTTPS enforcement when Pages already exists.

The command is a dry run unless `--apply` is supplied, and it refuses to mutate repository settings without `GH_TOKEN`.

## Scientific governance boundary

Repository governance does not change the evidence model. Synthetic validation remains software/surrogate evidence. COMSOL claims require qualified numerical-physics evidence, and experimental or exact-device claims require the corresponding real measurements and provenance.
