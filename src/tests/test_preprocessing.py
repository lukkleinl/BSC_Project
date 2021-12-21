import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing.config import Config
from preprocessing.data_loader import WineLoader
from preprocessing.preprocessing import transform_data


def test_preprocess():
    conf = Config()
    wine = WineLoader(conf.url, "data/wine_quality/raw_data.csv")
    wine.get_data()
    df = pd.read_csv("data/wine_quality/raw_data.csv", sep=';')

    df=transform_data(df, "quality")

    print(df.head())
