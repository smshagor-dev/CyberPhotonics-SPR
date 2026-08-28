import numpy as np,pandas as pd
from sprpcf.ml.active_learning import select_uncertain_candidates
from sprpcf.ml.dataset import DESIGN_COLUMNS,METRIC_COLUMNS
from sprpcf.ml.tandem import InverseGenerator
def test_uncertainty_outputs_physical_candidate_geometry():
    c=pd.DataFrame({METRIC_COLUMNS[0]:[800.,900.],METRIC_COLUMNS[1]:[20.,25.],METRIC_COLUMNS[2]:[650.,700.],"analyte_ri":[1.34,1.36]});r=select_uncertain_candidates(InverseGenerator(),c,np.array([850.,22.,675.]),np.array([100.,5.,50.]),np.array([2.,.5,45.,.6]),np.array([.5,.1,10.,.1]),-1.0,passes=4);assert len(r.selected)==2;assert all(x in r.selected.columns for x in DESIGN_COLUMNS);assert np.all(np.isfinite(r.uncertainty))
