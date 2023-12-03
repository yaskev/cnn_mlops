from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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

        if self.cfg.analytics.add_additional_charts:
            self.make_charts(design_matrix, labels, grid)

    def make_charts(
        self, design_matrix: pd.DataFrame, labels: np.ndarray, grid: Dict = None
    ):
        for one_est in grid["n_estimators"]:
            model = GradientBoostingClassifier(n_estimators=one_est, max_depth=5)
            model.fit(design_matrix, labels)
            pred = model.predict(design_matrix)
            mlflow.log_metric("Accuracy_estimators", accuracy_score(labels, pred))
            mlflow.log_metric("ROC-AUC_estimators", roc_auc_score(labels, pred))
        for one_depth_lim in grid["max_depth"]:
            model = GradientBoostingClassifier(max_depth=one_depth_lim, n_estimators=20)
            model.fit(design_matrix, labels)
            pred = model.predict(design_matrix)
            mlflow.log_metric("Accuracy_depth_limit", accuracy_score(labels, pred))
            mlflow.log_metric("ROC-AUC_depth_limit", roc_auc_score(labels, pred))

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return str(self)
