import pandas as pd

from preprocessing.config import Config
from preprocessing.preprocessing import data_quick_check, get_data


def test_preprocess():
    conf = Config()
    #get_data(conf.url, conf.raw_data_file)
    df = pd.read_csv(conf.raw_data_file)
    data_quick_check(df)


