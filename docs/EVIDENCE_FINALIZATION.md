# Evidence-Aware Finalization Pipeline

This pipeline rebuilds reviewer and submission artifacts from the evidence that is actually qualified at finalization time. It is designed for partial as well as complete research campaigns.

## Purpose

One command produces:

- a refreshed reviewer evidence package;
- a refreshed manuscript supplementary package;
- `EVIDENCE_DELTA.json` and `EVIDENCE_DELTA.md`;
- `BLOCKER_MATRIX.csv` and `BLOCKER_MATRIX.md`;
- `FINALIZATION_MANIFEST.json`;
- whole-bundle `checksums.sha256`.

Physical claim support comes only from the hash-validated qualified evidence registry. Optional validation, design, replay, or reproducibility folders keep their software/surrogate classes and cannot satisfy missing COMSOL, experimental-sensor, or exact-device gates.

## Build a finalization bundle

```bash
python scripts/finalize_evidence_package.py \
  --evidence-registry outputs/real_validation/evidence/evidence_registry.json \
  --out outputs/real_validation/finalization \
  --validation-dir outputs/validation \
  --design-dir outputs/design \
  --reproducibility-dir outputs/reproducibility
```

Optional inputs include:

- `--journal`
- `--manuscript`
- `--ablation-dir`
- `--closed-loop-dir`
- `--hardware-dir`
- `--release-validation`

Use `--replace` only to refresh a directory previously created by this pipeline. The command refuses to delete an unrelated non-empty directory.

## Evidence delta

`EVIDENCE_DELTA.*` reports:

- qualified physical classes present;
- qualified physical classes missing;
- claim families currently supported;
- claim families still marked `not supplied`.

A missing class remains a blocker. Synthetic, surrogate, replay, or workstation output is never substituted.

## Blocker matrix

The blocker matrix independently checks:

1. qualified registry validity and artifact hashes;
2. `comsol_physics` availability;
3. `experimental_sensor` availability;
4. `device_benchmark` availability;
5. submission-package integrity;
6. whole-system full readiness;
7. stable, non-prerelease version.

This means a fully populated evidence registry can still produce `ready_for_stable_release=false` while the project version is a release candidate.

## Stable decision

To turn the command into a release gate:

```bash
python scripts/finalize_evidence_package.py \
  --evidence-registry outputs/real_validation/evidence/evidence_registry.json \
  --out outputs/real_validation/finalization \
  --replace \
  --strict-stable
```

`--strict-stable` exits non-zero unless every required blocker row passes.

The version should only be changed to a stable release after real evidence has been qualified and the manuscript claims match that evidence. This tool does not create a Git tag or publish a release.

## Integrity

The finalization bundle contains nested reviewer/submission checksums plus a whole-bundle checksum file. Any later modification to a tracked finalization artifact causes `validate_finalization_package()` to fail.

Packaging integrity is not scientific validation by itself. Reviewers still need to inspect the COMSOL model/configuration, experimental protocol/calibration, raw measurements, target-device setup, and analysis assumptions.
