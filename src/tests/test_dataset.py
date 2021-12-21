import pandas as pd

from dataset.dataset import Dataset
from preprocessing.config import Config


def test_creating_dataset():
    conf = Config()
    data = Dataset(conf)
    data.transform_data()
    data.train_test_split()
    data.train()
    data.predict()