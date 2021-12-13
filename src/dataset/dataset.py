import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.config import Config


class Dataset:
    def __init__(self, config):
        self.dataset = pd.read_csv(config.raw_data_file)
        self.test_size = config.test_size
        self.random_state = config.random_state
        self.processed_train = config.processed_train
        self.ensemble_model = config.ensemble_model
        self.model_n_estimators = config.model_n_estimators
        self.model_random_state = config.random_state
        self.model_path = config.model_path
        self.processed_test = config.processed_test
        self.predicted_file = config.predicted_file
        self.export_result = config.export_result
        self.train = []
        self.test = []

    def transform_data(self):
        bins = (2, 6.5, 8)
        group_names = ["bad", "good"]
        self.dataset["quality"] = pd.cut(self.dataset["quality"], bins=bins, labels=group_names)
        label_quality = LabelEncoder()
        self.dataset["quality"] = label_quality.fit_transform(self.dataset["quality"])

    def train_test_split(self):
        self.train, self.test = train_test_split(self.dataset, test_size=self.test_size, random_state=self.random_state)
        self.train.to_csv(self.processed_train, index=False)
        self.test.to_csv(self.processed_test, index=False)
