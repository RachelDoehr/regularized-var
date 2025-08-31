import numpy as np
from regularized_var.utils import build_lagged_matrix

def test_build_lagged_matrix_shapes():
    X = np.arange(20).reshape(10, 2)
    Y, Z = build_lagged_matrix(X, p=2, include_const=True)
    assert Y.shape == (8, 2)
    assert Z.shape == (8, 2*2 + 1)  # 2 lags * K + const
