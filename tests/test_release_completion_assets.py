from __future__ import annotations

from pathlib import Path


REQUIRED_COMPLETION_ASSETS = (
    "configs/experimental_measurements.example.yaml",
    "docs/REAL_VALIDATION_TO_STABLE_RELEASE.md",
    "scripts/analyze_experimental_results.py",
    "scripts/build_paper_results.py",
    "scripts/prepare_stable_release.py",
    "scripts/research_completion.py",
    "src/sprpcf/publication/results.py",
    "src/sprpcf/utils/stable_release.py",
    "src/sprpcf/validation/completion.py",
    "src/sprpcf/validation/experiment.py",
)


def test_real_validation_to_stable_release_assets_are_tracked() -> None:
    missing = [path for path in REQUIRED_COMPLETION_ASSETS if not Path(path).is_file()]
    assert missing == []


def test_stable_tag_workflow_requires_promotion_certificate() -> None:
    workflow = Path(".github/workflows/release-validation.yml").read_text(encoding="utf-8")
    assert "Require qualified-evidence promotion certificate for stable tags" in workflow
    assert "--validate-certificate" in workflow
    assert "STABLE_RELEASE_EVIDENCE.json" in workflow
