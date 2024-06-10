import numpy as np


class EmptyModel:
    def __init__(self, **kwargs):
        pass

    def get_params(self, deep=False):
        # Dummy get_params function required by the pipeline
        return {}

    def score(self, x, y):
        # Dummy score function required by the pipeline
        return 1

    def fit(self, X_train, y_train):
        pass


class MeanValue(EmptyModel):
    def fit(self, X_train, y_train):
        self.mean = np.mean(y_train)

    def predict(self, X_test):
        return [self.mean] * len(X_test)


class MedianValue(EmptyModel):
    def fit(self, X_train, y_train):
        self.median = np.median(y_train)

    def predict(self, X_test):
        return [self.median] * len(X_test)


class LastDuration(EmptyModel):
    def predict(self, X_test):
        return X_test['LastDuration']


class LastConsumedkWh(EmptyModel):
    def predict(self, X_test):
        return X_test['LastConsumedkWh']
