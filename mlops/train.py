import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from mlops.model import boosting_model

import config


class Trainer:
    def __init__(self, design_matrix: pd.DataFrame, labels: np.ndarray,
                 encoder=preprocessing.OrdinalEncoder(),
                 scaler=preprocessing.StandardScaler()):
        self.X = design_matrix
        self.y = labels
        self.encoder = encoder
        self.scaler = scaler
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model = boosting_model.BoostedTrees()

    def __do_preprocessing(self):
        non_numeric_features = self.X.select_dtypes(exclude='number')
        transformed_features = self.encoder.fit_transform(non_numeric_features)
        self.X = pd.concat([self.X.select_dtypes(include='number'),
                            pd.DataFrame(transformed_features, columns=non_numeric_features.columns)],
                           axis=1)
        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def train(self):
        self.__do_preprocessing()
        self.model.fit(self.X_train, self.y_train)

        joblib.dump(self.model, config.MODEL_PATH)

        print(f'Fitted')


if __name__ == '__main__':
    dataset = pd.read_csv(config.TRAIN_PATH)
    labels = dataset.loc[:, 'Survived']
    matrix = dataset.loc[:, dataset.columns != 'Survived']

    trainer = Trainer(matrix, labels)
    trainer.train()
