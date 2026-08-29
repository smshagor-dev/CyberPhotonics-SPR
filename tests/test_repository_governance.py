from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "configure_github_repository.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("configure_github_repository", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_governance_script_defaults_to_safe_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0
    assert "Dry run only" in completed.stdout
    assert "main -> PR workflow" in completed.stdout


def test_governance_apply_requires_explicit_admin_token() -> None:
    env = os.environ.copy()
    env.pop("GH_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--apply"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 2
    assert "GH_TOKEN is required" in completed.stderr


def test_branch_protection_blocks_destructive_pushes() -> None:
    module = _load_module()
    payload = module._branch_protection_payload()
    assert payload["allow_force_pushes"] is False
    assert payload["allow_deletions"] is False
    assert payload["required_linear_history"] is True
    assert payload["required_conversation_resolution"] is True
    assert payload["required_pull_request_reviews"]["required_approving_review_count"] == 0


def test_topics_are_specific_and_lowercase() -> None:
    module = _load_module()
    assert "surface-plasmon-resonance" in module.DEFAULT_TOPICS
    assert "photonic-crystal-fiber" in module.DEFAULT_TOPICS
    assert all(topic == topic.lower() for topic in module.DEFAULT_TOPICS)
    assert len(module.DEFAULT_TOPICS) == len(set(module.DEFAULT_TOPICS))
