from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_exact(path: str, old: str, new: str, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new, count), encoding="utf-8")


def migrate_checkpoint_loaders() -> None:
    specs = {
        "src/sprpcf/ml/active_learning.py": [
            ("import torch\n", "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint\n"),
            (
                'checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(checkpoint_path)",
            ),
        ],
        "src/sprpcf/ml/ensemble.py": [
            (
                "import torch\n",
                "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint, save_tandem_checkpoint\n",
            ),
            (
                'checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(checkpoint_path)",
            ),
            ("torch.save(checkpoint, output_path)", "save_tandem_checkpoint(checkpoint, output_path)"),
        ],
        "src/sprpcf/ml/explainability.py": [
            ("import torch\n", "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint\n"),
            (
                'checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(checkpoint_path)",
            ),
        ],
        "src/sprpcf/ml/export_onnx.py": [
            ("import torch\n", "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint\n"),
            (
                'checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(checkpoint_path)",
            ),
            (
                'checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(args.checkpoint)",
            ),
        ],
        "src/sprpcf/ml/multiobjective.py": [
            ("import torch\n", "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint\n"),
            (
                'checkpoint = torch.load(path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(path)",
            ),
        ],
        "src/sprpcf/validation/benchmark.py": [
            ("import torch\n", "import torch\n\nfrom sprpcf.ml.checkpoint_io import load_tandem_checkpoint\n"),
            (
                'checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                "checkpoint = load_tandem_checkpoint(checkpoint_path)",
            ),
        ],
    }
    for path, replacements in specs.items():
        for old, new in replacements:
            replace_exact(path, old, new)

    replace_exact(
        "src/sprpcf/ml/train_tandem.py",
        "from sprpcf.ml.dataset import CONDITION_COLUMNS, DesignDataModule, GEOMETRY_COLUMNS, METRIC_COLUMNS\n",
        "from sprpcf.ml.checkpoint_io import save_tandem_checkpoint\n"
        "from sprpcf.ml.dataset import CONDITION_COLUMNS, DesignDataModule, GEOMETRY_COLUMNS, METRIC_COLUMNS\n",
    )
    replace_exact(
        "src/sprpcf/ml/train_tandem.py",
        "torch.save(checkpoint, checkpoint_out)",
        "save_tandem_checkpoint(checkpoint, checkpoint_out)",
    )


def harden_governance_transport() -> None:
    replace_exact(
        "scripts/configure_github_repository.py",
        "import urllib.request\n",
        "import urllib.request\nfrom urllib.parse import urlparse\n",
    )
    replace_exact(
        "scripts/configure_github_repository.py",
        '    data = None if payload is None else json.dumps(payload).encode("utf-8")\n',
        '    parsed = urlparse(url)\n'
        '    if parsed.scheme != "https" or parsed.hostname != "api.github.com":\n'
        '        raise ValueError("Only https://api.github.com requests are permitted.")\n'
        '    data = None if payload is None else json.dumps(payload).encode("utf-8")\n',
    )
    replace_exact(
        "scripts/configure_github_repository.py",
        "        with urllib.request.urlopen(request, timeout=30) as response:\n",
        "        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 -- URL allowlisted above\n",
    )


def write_checkpoint_tests() -> None:
    (ROOT / "tests/test_checkpoint_io.py").write_text(
        '''from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint, save_tandem_checkpoint


def test_safe_checkpoint_round_trip_converts_numpy_scalers(tmp_path: Path) -> None:
    path = tmp_path / "safe.pt"
    payload = {
        "inverse_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "geometry_mean": np.array([1.0, 2.0], dtype=np.float64),
        "metric_scale": np.array([3.0, 4.0], dtype=np.float32),
        "seed": 7,
    }
    save_tandem_checkpoint(payload, path)
    loaded = load_tandem_checkpoint(path)
    np.testing.assert_allclose(loaded["geometry_mean"], [1.0, 2.0])
    np.testing.assert_allclose(loaded["metric_scale"], [3.0, 4.0])
    assert loaded["seed"] == 7


def test_safe_loader_refuses_legacy_numpy_pickle_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"geometry_mean": np.array([1.0, 2.0])}, path)
    with pytest.raises(RuntimeError, match="Refusing to deserialize unsafe or legacy checkpoint"):
        load_tandem_checkpoint(path)
''',
        encoding="utf-8",
    )


def document_checkpoint_boundary() -> None:
    path = ROOT / "CONTRIBUTING.md"
    text = path.read_text(encoding="utf-8")
    note = '''### Checkpoint security

Current tandem checkpoints are written in a format compatible with PyTorch `weights_only=True` loading. Older checkpoints that require unrestricted pickle deserialization are intentionally rejected by current loaders. Regenerate those checkpoints with the current training pipeline rather than disabling the safe loader.

'''
    marker = "## Data, models, and generated artifacts\n"
    if note.strip() not in text:
        if marker not in text:
            raise RuntimeError("CONTRIBUTING checkpoint-security insertion marker missing")
        path.write_text(text.replace(marker, note + marker, 1), encoding="utf-8")


def remove_migration_helpers() -> None:
    for relative in (
        ".github/workflows/checkpoint-hardening-once.yml",
        "scripts/checkpoint_hardening_migration.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def main() -> None:
    migrate_checkpoint_loaders()
    harden_governance_transport()
    write_checkpoint_tests()
    document_checkpoint_boundary()
    remove_migration_helpers()


if __name__ == "__main__":
    main()
