import numpy as np

def MSE(y_true: np.ndarray , y_pred: np.ndarray  ) -> np.float64:
    return np.mean(np.power( y_true - y_pred  , 2))