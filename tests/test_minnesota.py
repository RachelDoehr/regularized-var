import numpy as np
import pandas as pd
from regularized_var.var import MinnesotaVAR

def test_minnesota_var_runs_and_predicts():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(150, 3)), columns=['x', 'y', 'z'])
    mvar = MinnesotaVAR(n_lags=3, alpha_own=5.0, alpha_cross=15.0, include_const=False).fit(df)
    fcst = mvar.predict(steps=4)
    assert fcst.shape == (4, 3)
    