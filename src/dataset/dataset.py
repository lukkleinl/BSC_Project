import pickle

import pandas as pd
import sklearn.ensemble
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluate.dataset_listener import setup_log_event_handlers
from evaluate.event import post_event


class Dataset:
    setup_log_event_handlers()

    def __init__(self, config):
        self._dataset = pd.read_csv(config.raw_data_file, sep=";")
        self._target = config.target
        self._test_size = config.test_size
        self._random_state = config.random_state
        self._processed_train = config.processed_train
        self._ensemble_model = config.ensemble_model
        self._model_n_estimators = config.model_n_estimators
        self._model_random_state = config.random_state
        self._model_path = config.model_path
        self._processed_test = config.processed_test
        self._predicted_file = config.predicted_file
        self._export_result = config.export_result
        self._train = []
        self._test = []
        self._model = []

    def transform_data(self):
        bins = (2, 6.5, 8)
        group_names = ["bad", "good"]
        self._dataset[self._target] = pd.cut(self._dataset[self._target], bins=bins, labels=group_names)
        label_quality = LabelEncoder()
        self._dataset[self._target] = label_quality.fit_transform(self._dataset[self._target])
        post_event("data_changed", self._dataset)

    def train_test_split(self):
        self._train, self._test = train_test_split(self._dataset, test_size=self._test_size,
                                                   random_state=self._random_state)
        self._train.to_csv(self._processed_train, index=False)
        self._test.to_csv(self._processed_test, index=False)

    def train(self):
        self._model = sklearn.ensemble.GradientBoostingClassifier()
        X, y, cols = self._getXY(self._train, self._target)
        self._model.fit(X, y)
        pickle.dump(self._model, open(self._model_path, 'wb'))
        self._model = (pickle.load(open(self._model_path, 'rb')))
        X, y, cols = self._getXY(self._test, self._target)
        y_predicted = self._model.predict(X)
        mean_square_error = mean_squared_error(y, y_predicted)
        print(mean_square_error)

    def _getXY(self, data, target: str):
        return data.drop(target, axis=1).to_numpy(), data[target], list(data.columns)
