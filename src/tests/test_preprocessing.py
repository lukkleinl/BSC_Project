import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing.config import Config
from preprocessing.data_loader import WineLoader


def test_preprocess():
    conf = Config()
    wine = WineLoader(conf.url, "data/wine_quality/raw_data.csv")
    wine.get_data()
    df = pd.read_csv("data/wine_quality/raw_data.csv", sep=';')

    bins = (2, 6.5, 8)
    group_names = ["bad", "good"]
    df["quality"] = pd.cut(df["quality"], bins=bins, labels=group_names)
    label_quality = LabelEncoder()
    df["quality"] = label_quality.fit_transform(df["quality"])

    print(df.head())
