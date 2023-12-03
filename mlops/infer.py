from io import StringIO

import dvc.api
import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlops.enums import EncoderType, ScalerType
from omegaconf import DictConfig
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Runner:
    def __init__(self, design_matrix: pd.DataFrame, cfg: DictConfig):
        self.X = design_matrix
        self.cfg = cfg
        self.encoder = (
            preprocessing.OrdinalEncoder()
            if cfg.preprocessing.encoder == EncoderType.ORDINAL_ENCODER.value
            else None
        )
        self.scaler = (
            preprocessing.StandardScaler()
            if cfg.preprocessing.encoder == ScalerType.STANDARD_SCALER.value
            else None
        )

    def __do_preprocessing(self) -> None:
        if self.encoder is not None:
            non_numeric_features = self.X.select_dtypes(exclude="number")
            transformed_features = self.encoder.fit_transform(non_numeric_features)
            self.X = pd.concat(
                [
                    self.X.select_dtypes(include="number"),
                    pd.DataFrame(
                        transformed_features, columns=non_numeric_features.columns
                    ),
                ],
                axis=1,
            )
        if self.scaler is not None:
            self.X = self.scaler.fit_transform(self.X)

    def run(self) -> np.ndarray:
        self.__do_preprocessing()
        model = joblib.load(self.cfg.data.model_path)

        return model.predict(self.X)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def infer(cfg: DictConfig):
    """Run infer and print metrics."""
    mlflow.set_tracking_uri(cfg.analytics.mlflow_uri)
    mlflow.autolog()
    mlflow.log_params(cfg)

    data = dvc.api.read(cfg.data.test_path)
    dataset = pd.read_csv(StringIO(data))

    labels = dataset.loc[:, cfg.preprocessing.target_column]
    matrix = dataset.loc[:, dataset.columns != cfg.preprocessing.target_column]

    runner = Runner(matrix, cfg)
    pred = runner.run()

    metrics = set(cfg.analytics.metrics)
    if "accuracy" in metrics:
        print(f"Accuracy: {accuracy_score(labels, pred)}")
    if "precision" in metrics:
        print(f"Precision: {precision_score(labels, pred)}")
    if "recall" in metrics:
        print(f"Recall: {recall_score(labels, pred)}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {roc_auc_score(labels, pred)}")

    pred.tofile(cfg.data.results_path, sep="\n")


if __name__ == "__main__":
    infer()
