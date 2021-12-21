import pandas as pd

from dataset.dataset import Dataset
from preprocessing.config import Config
from preprocessing.preprocessing import transform_data


def test_creating_dataset():
    conf = Config()
    data = Dataset(conf)
    data._dataset=transform_data(data._dataset,data._target)
    data.train_test_split()
    data.train()
    data.predict()
