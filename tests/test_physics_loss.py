import torch
from sprpcf.ml.losses import geometry_constraint_loss
def test_geometry_constraint_loss_penalizes_all_bounds():
    feasible=torch.tensor([[2.0,.5,45.,.6]]);invalid=torch.tensor([[.5,.95,10.,.05],[5.0,.1,95.,2.5]]);assert geometry_constraint_loss(feasible).item()==0.0;assert geometry_constraint_loss(invalid).item()>0.0
