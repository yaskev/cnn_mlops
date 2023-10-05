import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from .model import Model


class BoostedTrees(Model):
    """Wrapper over gradient boosting  classifier"""
    def __init__(self, optimize_hyperparams=True):
        super().__init__(GradientBoostingClassifier(), "Boosted trees", optimize_hyperparams)

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            'max_depth': np.arange(3, 10, 3),
            'n_estimators': np.arange(20, 151, 10)
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)
