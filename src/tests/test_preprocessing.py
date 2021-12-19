import pandas as pd

from preprocessing.config import Config


def test_preprocess():
    conf = Config()
    # get_data(conf.url, conf.raw_data_file)
    df = pd.read_csv(conf.raw_data_file)
