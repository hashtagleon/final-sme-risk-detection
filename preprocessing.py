"""Helper used inside the model pipeline (same ToDense step as in the notebook)."""
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


class ToDense(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sp.issparse(X) else X
