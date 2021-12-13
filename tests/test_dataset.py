import pytest
import pandas as pd


from dataset.dataset import Dataset
from config.config import Config


def test_creating_dataset():
    conf = Config()
    data = Dataset(conf)
    data.transform_data()
    data.train_test_split()
    data = pd.read_csv(conf.processed_train)
    x, y, cols = data.drop("quality", axis=1).to_numpy(), data["quality"], list(data.columns)
    assert cols
