# CyberPhotonics-SPR v1.0.0 Release Candidate

Current candidate: `1.0.0rc1`

Intended tag: `v1.0.0rc1`

This release candidate freezes the publication-oriented software architecture and validates the end-to-end research packaging path before any stable `v1.0.0` tag is created.

## Candidate scope

The RC includes:

- PCF-SPR synthetic/COMSOL workflow interfaces,
- conditioned and multi-objective inverse-design tooling,
- uncertainty/OOD/fabrication-aware ranking,
- COMSOL closed-loop orchestration,
- edge/hardware-ready inference interfaces,
- research dashboard,
- reproducibility/release metadata,
- reviewer evidence packaging,
- manuscript supplementary/submission packaging,
- deterministic software-only publication demo,
- whole-system readiness auditing,
- built-wheel isolated smoke validation,
- tracked COMSOL configuration contract template,
- warnings-as-errors CI across the supported Python matrix.

## v1.0 system-readiness milestone

The authoritative RC gate is:

```bash
python scripts/check_system_readiness.py \
  --profile release \
  --expected-version 1.0.0rc1 \
  --strict
```

See `docs/V1_SYSTEM_READINESS.md` for the complete definition of done.

A separate `--profile full` gate is intentionally stricter and requires real evidence classes for COMSOL physics, experimental sensor measurements, and exact-device benchmarks. This prevents software completeness from being confused with physical-claim completeness.

## Stable-release gates

Before creating stable `v1.0.0`, deliberately review:

- package/version/citation consistency,
- whole-system `release` readiness,
- wheel and source-distribution build,
- isolated smoke import of the built wheel,
- core, edge, dashboard, publication, and submission tests,
- COMSOL configuration contract and actual `.mph` model/unit validation,
- reviewer package integrity and claims matrix,
- submission package checksums and evidence readiness flags,
- release notes and citation metadata,
- known limitations and scientific-boundary wording.

Real COMSOL, laboratory, fabricated-sensor, or target-device measurements are **not fabricated by the RC process**. They are required before making the corresponding manuscript claims, but absence of those data does not prevent validation of the software release itself.

No DOI or external archival identifier is claimed by this file. Add one only after it has actually been minted.
