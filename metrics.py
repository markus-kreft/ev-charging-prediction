import numpy as np


def absolute_percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)


def absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.sum(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    ) / len(y_true)
