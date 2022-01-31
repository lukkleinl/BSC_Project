import pandas as pd
import pytest

from configuration.configuration import load_configuration
from preprocessing.preprocessing import separate_features_outcome, preprocess_data, \
    import_and_load_preprocessing_function

data_classes_rfc = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")

for dataclass in data_classes_rfc:
    if type(dataclass).__name__ == "PreprocessingSteps":
        prepr_steps = dataclass
    if type(dataclass).__name__ == "TrainTestSplit":
        train_split = dataclass


def test_separate_features_outcome():
    df = pd.read_csv("src/tests/data/raw/raw_data.csv")
    features, labels = separate_features_outcome(df,"quality")
    assert len(features) == 1599

def test_preprocess_data():
    df = pd.read_csv("src/tests/data/raw/raw_data.csv")
    with pytest.raises(FileNotFoundError):
        x_train,x_test,y_train, y_test = preprocess_data(df, prepr_steps, train_split)
