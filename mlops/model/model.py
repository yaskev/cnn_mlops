from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


class Model(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def __init__(self, basic_model, model_name: str, optimize_hyperparams: bool = True):
        self.optimize_hyperparams = optimize_hyperparams
        self.fitted = False
        self.optimized_model = None
        self.basic_model = basic_model
        self.model_name = model_name

    @abstractmethod
    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        pass

    def predict(self, test_data: pd.DataFrame):
        if not self.fitted:
            raise Exception(f'The model "{self.model_name}" has not been fitted')
        return self.optimized_model.predict(test_data)

    def fit_with_grid_search(
        self,
        design_matrix: pd.DataFrame,
        labels: np.ndarray,
        hyperparams_grid: Dict = None,
    ):
        if self.optimize_hyperparams and hyperparams_grid is not None:
            self.optimized_model = GridSearchCV(
                self.basic_model, hyperparams_grid, n_jobs=-1
            )
        else:
            self.optimized_model = self.basic_model
        self.optimized_model.fit(design_matrix, labels)
        self.fitted = True

    def get_model_parameters(self):
        if not self.fitted:
            raise Exception(f'The model "{self.model_name}" has not been fitted')
        if self.optimize_hyperparams:
            return self.optimized_model.best_estimator_.get_params()
        return self.optimized_model.get_params()
