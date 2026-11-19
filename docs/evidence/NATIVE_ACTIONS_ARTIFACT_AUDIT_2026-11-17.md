# Independent native GitHub Actions artifact audit — 2026-11-17

**Scope:** actual ZIP bytes retrieved from [workflow 43745651955](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/43745651955), not metadata alone. This is an evidence-integrity audit, **not** a new Maxwell solve, independent COMSOL verification, or PCF-SPR qualification.

| Artifact ID | GitHub Actions ZIP SHA256 | ZIP integrity | Evidence |
|---|---|---|---|
| 14173928053 | `5803478018961721f723a518737130b86b715ee82cf38d90d80930ede276081c` | Pass, one JSON | `PHYSICS_VERIFICATION_AUDIT.json`: 0 requested, 0 completed, 0 failed; absent native verification manifest |
| 14174366132 | `8a5af0785f91690c7eb4a35f7f069c56a425b53eecb605064c855c8de3c2fcb9` | Pass, one JSON | `COMSOL_MODEL_PILOT_READINESS.json`: static-only; backend not invoked; `config:is_not_template` failed; 160 **planned**, not executed |
| 14174197147 | `eed785a2a1220b03602b9c191d2bcee139a9e02cf4d7902c4b714fcbae65239c` | Pass, eight files | `INVERSE_BENCHMARK_MANIFEST.json`: six output-file SHA256 checks **all match**; source synthetic/software-only; physics solver evaluations 0 |

## Reproduction

Download each original artifact ZIP from workflow 43745651955 (artifact IDs above). Compare raw ZIP bytes with the stated SHA256 digests, then run this Python 3 snippet with the files renamed `14173928053.zip`, `14174366132.zip`, `14174197147.zip`:

```python
import hashlib, json, zipfile
expected = {
    "14173928053": "5803478018961721f723a518737130b86b715ee82cf38d90d80930ede276081c",
    "14174366132": "8a5af0785f91690c7eb4a35f7f069c56a425b53eecb605064c855c8de3c2fcb9",
    "14174197147": "eed785a2a1220b03602b9c191d2bcee139a9e02cf4d7902c4b714fcbae65239c",
}
for artifact_id, digest in expected.items():
    path = artifact_id + ".zip"
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        if artifact_id == "14174197147":
            manifest = json.loads(z.read("INVERSE_BENCHMARK_MANIFEST.json"))
            for name, expected_hash in manifest["outputs_sha256"].items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected_hash
            assert manifest["physics_solver_evaluations"] == 0
            assert manifest["dataset_provenance"]["independently_verified_physics"] is False
        elif artifact_id == "14173928053":
            audit = json.loads(z.read("PHYSICS_VERIFICATION_AUDIT.json"))
            assert audit["counts"]["completed_requests"] == 0
            assert audit["scientific_submission_ready"] is False
        else:
            audit = json.loads(z.read("COMSOL_MODEL_PILOT_READINESS.json"))
            assert audit["pilot_solver_called"] is False
            assert audit["ready_for_full_solver_run"] is False
```

## Scientific interpretation and required work

The inverse benchmark manifest declares **12 training geometry groups, 4 held-out groups**, one checkpoint trained for **one epoch** per component, and two candidate-generation seeds sharing that checkpoint. Its own warnings say held-out groups were used for model selection, the forward surrogate selected all candidates, and budgets are not matched by wall-clock. These are **not untouched external validation** or independent physics verification.

The physical audit's missing verification manifest is a **failure of available evidence**, not a zero-loss/zero-error physical result. The COMSOL pilot's `planned_solver_evaluations=160` is a plan, not solver execution.

This audit validates only these **three** downloaded archives, not the other six artifacts in the workflow or the separate native Femwell runs in PRs #63/#64. Archive SHA256 and internal CSV hashes establish byte consistency, **not** Maxwell correctness, source authenticity, spectral convergence, or publication readiness.

**Fail-closed status:** `physics_qualified=false`, `publication_ready=false`, `comsol_verified=false`.

Next: obtain and inspect native Femwell mesh/field ZIPs; independently establish a literature-frozen geometry and independently sourced Au dispersion; run mesh/PML/domain and dense fixed-geometry RI/wavelength convergence with core/SPP mode tracking and independent solver comparisons; only then evaluate inverse models with untouched geometry-disjoint test sets.
