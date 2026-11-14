# Evidence Inventory: Stage 1

This is an **artifact discovery and provenance triage** stage, not scientific acceptance.
The scanner records files and SHA-256 hashes, checks dataset source sidecars, flags
synthetic/mock outputs, validates an optional qualified evidence registry, and emits
an explicit blocker matrix. It does **not** invent missing COMSOL or laboratory results.

## Current repository baseline (main at 4b4d99d, 2026-11-14)

| Artifact group | Observed | Scientific interpretation |
|---|---|---|
| Data | `data/dataset.parquet` is tracked | Original provenance/sidecar not established from tracked tree |
| Tracked raw files | `data/raw/.gitkeep` only | No archived measured or COMSOL raw spectra in Git |
| Tracked processed experiments | `data/processed/.gitkeep` only | No verified processed experimental dataset in Git |
| Trained weights | `models/.gitkeep` only | No archived model checkpoint in Git |
| Real experiment output | `outputs/.gitkeep` only | No completed validation package in Git |
| Benchmark | `reports/phase4_hil_benchmark.json` | `protocol: mock`: software test only |
| Solver interface | COMSOL `mph` integration and config examples | Code/template does not constitute a verified FEM run |
| Research infrastructure | Grouped split, Ridge, ablation, uncertainty, packaging tests | Implemented tools; final paper numbers not established by the source tree |
| Physical evidence registry | No tracked populated registry in expected output path | COMSOL, measured sensor and exact-device classes not established |

**Important:** The Git checkout is not an exhaustive inventory of external disks,
untracked `data/raw` files, COMSOL workstations, GitHub Actions artifacts, or DOI archives.
Run the same scanner with `--external-root` to include each real experiment archive.
Dataset contents/schemas and physics are not decoded by the Stage 1 scanner; use the
scientific validation tools in later stages.

## Run from repository root

```bash
python scripts/evidence_inventory.py \
  --json-out outputs/inventory/EVIDENCE_INVENTORY.json \
  --markdown-out outputs/inventory/EVIDENCE_INVENTORY.md

# Repeat --external-root for all existing local simulation/data archives:
python scripts/evidence_inventory.py \
  --external-root /path/to/comsol/archive \
  --external-root /path/to/sensor/measurements \
  --evidence-registry /path/to/evidence_registry.json \
  --json-out outputs/inventory/COMPLETE_LOCAL_INVENTORY.json \
  --markdown-out outputs/inventory/COMPLETE_LOCAL_INVENTORY.md

# Optional fail-closed submission check (expected exit 2 until blockers resolved)
python scripts/evidence_inventory.py --fail-on-blockers
```

The report contains `git`, `scope`, `artifacts`, `counts`,
`registry`, `requirements`, `scientific_submission_ready` and `limitations`.
Paths are relative to the scanned root labels, not absolute machine paths.
`external_N` denotes each additional root in input order.

### Interpretation of statuses

- `software_only`: synthetic/mock/replay source with the appropriate source label.
- `invalid_provenance`: content differs from recorded dataset SHA-256.
- `declared_physical_unqualified`: a file *claims* a COMSOL/measurement origin;
  the scanner has not independently established that physical origin.
- `unverified`: no sufficient provenance or unknown evidence type.
- Registry `integrity_valid=true` only verifies the registry's own hashes and
  record structure; it does **not** verify the underlying physics or apparatus.

The inventory is intentionally **never** an automatic optical-science sign-off.
A full claim-level review, independent FEM convergence study, baselines, uncertainty
analysis, and immutable paper outputs remain necessary.

## Completion and hand-off

Stage 1 engineering acceptance: repeatable artifact discovery, SHA-256 fingerprints,
provenance validation, conservative classification, missing-evidence requirements,
machine-readable/markdown outputs, regression tests and CI archive.

Scientific gaps are tracked in [issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).
Stage 2 must supply a validated COMSOL `.mph` model, mesh/boundary/material
review, independent fixed-geometry RI spectra and source-bound results.

For a numerical-only paper, fabricated-sensor measurements and device benchmarks
are optional *unless those specific claims appear in the manuscript*. Do not
confuse a software release success with scientific publication readiness.
