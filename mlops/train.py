from io import StringIO

import dvc.api
import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlops.enums import EncoderType, ScalerType
from mlops.model import boosting_model
from omegaconf import DictConfig
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(
        self,
        design_matrix: pd.DataFrame,
        labels: np.ndarray,
        cfg: DictConfig,
    ):
        self.X = design_matrix
        self.y = labels
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
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model = boosting_model.BoostedTrees(cfg)
        self.cfg = cfg

    def __do_preprocessing(self):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.cfg.preprocessing.test_size, random_state=42
        )

    def train(self):
        self.__do_preprocessing()
        self.model.fit(self.X_train, self.y_train)

        if self.cfg.postprocessing.save_trained_model:
            joblib.dump(self.model, self.cfg.data.model_path)

        if self.cfg.postprocessing.convert_to_onnx:
            initial_types = [("input", FloatTensorType((None, self.X_train.shape[1])))]
            model_onnx = convert_sklearn(
                self.model.optimized_model, initial_types=initial_types, target_opset=12
            )
            mlflow.onnx.save_model(model_onnx, self.cfg.data.onnx_path)

        print("Fitted")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def train(cfg: DictConfig):
    """Run train and optionally save model."""
    mlflow.set_tracking_uri(cfg.analytics.mlflow_uri)
    mlflow.log_params(cfg)

    data = dvc.api.read(cfg.data.train_path)
    dataset = pd.read_csv(StringIO(data))

    labels = dataset.loc[:, cfg.preprocessing.target_column]
    matrix = dataset.loc[:, dataset.columns != cfg.preprocessing.target_column]

    trainer = Trainer(matrix, labels, cfg)
    trainer.train()


if __name__ == "__main__":
    train()
