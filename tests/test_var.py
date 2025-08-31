import numpy as np
import pandas as pd
from regularized_var.var import VAR
from regularized_var.utils import build_lagged_matrix

def test_var_fit_predict_and_coeff_mats():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(120, 2)), columns=['a', 'b'])
    model = VAR(n_lags=2, alpha=0.1, include_const=True).fit(df)
    fcst = model.predict(steps=5)
    assert fcst.shape == (5, 2)
    A = model.coefficient_matrices()
    assert A.shape == (2, 2, 2)  # (p, K, K)

def test_var_alpha0_equals_lstsq():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(80, 3)), columns=['x', 'y', 'z'])
    p = 2
    model = VAR(n_lags=p, alpha=0.0, include_const=True).fit(df)

    Xv = df.values
    Y, Z = build_lagged_matrix(Xv, p, include_const=True)

    # VAR coefficients
    B_var = model.coef_

    # vs from np.linalg.lstsq, equation-by-equation
    B_lstsq = np.empty_like(B_var)
    for j in range(Y.shape[1]):
        b_j, *_ = np.linalg.lstsq(Z, Y[:, j], rcond=None)
        B_lstsq[:, j] = b_j
