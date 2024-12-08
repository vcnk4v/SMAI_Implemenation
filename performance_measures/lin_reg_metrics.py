import numpy as np


class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def mse(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def variance(self):
        return np.var(self.y_pred - self.y_true)

    def std_deviation(self):
        return np.std(self.y_pred - self.y_true)
