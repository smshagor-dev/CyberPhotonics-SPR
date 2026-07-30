from __future__ import annotations

import torch

from sprpcf.ml.losses import geometry_constraint_loss


def test_geometry_constraint_loss_penalizes_overlap_and_thickness() -> None:
    feasible = torch.tensor([[2.0, 0.5, 45.0]], dtype=torch.float32)
    infeasible = torch.tensor([[2.0, 1.2, 10.0], [2.0, 0.4, 95.0]], dtype=torch.float32)

    assert geometry_constraint_loss(feasible).item() == 0.0
    assert geometry_constraint_loss(infeasible).item() > 0.0
