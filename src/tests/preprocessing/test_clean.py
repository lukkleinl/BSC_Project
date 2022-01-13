import os

import pandas as pd
from sklearn.model_selection import train_test_split

from data_classes.data_classes import Loader, Paths
from conversion.conversion import get_config_from_yaml
from preprocessing.transform_data import binary_classification_and_label_encoding


def test_preprocess():
    conf = get_config_from_yaml("src/data/config_files/config_rfc.yaml")
    load = Loader(**conf["Loader"])
    paths = Paths(**conf["Paths"])

    df = pd.read_csv("src/" + paths.raw_data_path + load.name_of_file, sep=';')
    df = binary_classification_and_label_encoding(df, {"target":"quality"})

    print(type(df))
    print(df.head())
    train, test = train_test_split(df, test_size=None, random_state=None)
    print(df.head())
    print(train)
    print(test)
