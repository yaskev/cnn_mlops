import joblib
import numpy as np
import pandas as pd
from mlops import config
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Runner:
    def __init__(self, design_matrix: pd.DataFrame):
        self.X = design_matrix
        self.encoder = preprocessing.OrdinalEncoder()
        self.scaler = preprocessing.StandardScaler()

    def __do_preprocessing(self) -> None:
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
        self.X = self.scaler.fit_transform(self.X)

    def run(self) -> np.ndarray:
        self.__do_preprocessing()
        model = joblib.load(config.MODEL_PATH)

        return model.predict(self.X)


if __name__ == "__main__":
    dataset = pd.read_csv(config.TEST_PATH)
    labels = dataset.loc[:, "Survived"]
    matrix = dataset.loc[:, dataset.columns != "Survived"]

    runner = Runner(matrix)
    pred = runner.run()

    print(f"Accuracy: {accuracy_score(labels, pred)}")
    print(f"Precision: {precision_score(labels, pred)}")
    print(f"Recall: {recall_score(labels, pred)}")
    print(f"ROC-AUC: {roc_auc_score(labels, pred)}")

    pred.tofile(config.RESULTS_PATH, sep="\n")
