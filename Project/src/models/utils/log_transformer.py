import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.columns is not None:
            for col in self.columns:
                X[col] = np.log1p(X[col])  # log1p is used to handle zero values
        else:
            for col in X.columns:
                X[col] = np.log1p(X[col])
        return X
