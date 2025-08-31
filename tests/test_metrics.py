import numpy as np
from regularized_var.metrics import mse, mae, pseudo_r2

def test_basic_metrics_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    assert mse(y_true, y_pred) >= 0
    assert mae(y_true, y_pred) >= 0

    # r2 can be negative; just ensure it returns a float
    _ = pseudo_r2(y_true, y_pred)

