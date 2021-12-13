import pytest
import pandas as pd

from dataset.dataset import Dataset
from config.config import Config


def test_creating_dataset():
    conf = Config()
    data = Dataset(conf)
    data.transform_data()
    data.train_test_split()
    data.train = pd.DataFrame(data.train)
    x, y, cols = data.train.drop("quality", axis=1).to_numpy(), data.train["quality"], list(data.train.columns)
