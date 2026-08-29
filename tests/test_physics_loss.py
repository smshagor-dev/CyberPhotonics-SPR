from __future__ import annotations

import torch

from sprpcf.ml.losses import clamp_physical_geometry, geometry_constraint_loss


def test_geometry_constraint_loss_covers_all_documented_bounds() -> None:
    feasible = torch.tensor([[2.0, 0.5, 45.0, 0.6]], dtype=torch.float32)
    infeasible = torch.tensor(
        [
            [0.5, 0.5, 45.0, 0.6],
            [2.0, 0.95, 45.0, 0.6],
            [2.0, 0.5, 10.0, 0.6],
            [2.0, 0.5, 45.0, 2.0],
        ],
        dtype=torch.float32,
    )
    assert geometry_constraint_loss(feasible).item() == 0.0
    assert geometry_constraint_loss(infeasible).item() > 0.0


def test_clamp_physical_geometry_projects_to_fabrication_envelope() -> None:
    geometry = torch.tensor([[9.0, -1.0, 120.0, 3.0]], dtype=torch.float32)
    clamped = clamp_physical_geometry(geometry)[0]
    assert torch.allclose(clamped, torch.tensor([4.0, 0.2, 80.0, 1.5]))
