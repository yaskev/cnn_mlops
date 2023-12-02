import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier

from .model import Model


class BoostedTrees(Model):
    """Wrapper over gradient boosting  classifier."""

    def __init__(self, cfg: DictConfig):
        super().__init__(
            GradientBoostingClassifier(),
            "Boosted trees",
            cfg.train.optimize_hyperparams,
        )
        self.cfg = cfg

    def fit(self, design_matrix: pd.DataFrame, labels: np.ndarray):
        grid = {
            "max_depth": np.arange(
                self.cfg.train.min_max_depth,
                self.cfg.train.max_max_depth,
                self.cfg.train.max_depth_step,
            ),
            "n_estimators": np.arange(
                self.cfg.train.min_estimators,
                self.cfg.train.max_estimators,
                self.cfg.train.estimators_step,
            ),
        }
        super().fit_with_grid_search(design_matrix, labels, grid)

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)
