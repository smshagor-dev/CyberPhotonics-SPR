import numpy as np
from sprpcf.ml.dataset import FORWARD_INPUT_COLUMNS
from sprpcf.ml.explainability import attribution_matrix
from sprpcf.ml.tandem import ForwardNetwork
def test_integrated_gradients_uses_forward_schema():
    x=np.zeros((2,len(FORWARD_INPUT_COLUMNS)),dtype=np.float32);m=attribution_matrix(ForwardNetwork(input_dim=len(FORWARD_INPUT_COLUMNS)),x,steps=4);assert list(m.columns)==FORWARD_INPUT_COLUMNS and m.shape==x.shape
