import pandas as pd
import pytest
import sklearn.ensemble

from algorithms import algorithms_creation
from algorithms import algorithms_implementation
from configuration.configuration import load_configuration
from preprocessing.preprocessing import separate_features_outcome

data_classes_rfc = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")
data_classes_converted = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_Converted_Loader.yaml")
data_classes_sgd = load_configuration("src/tests/data/config_files/config_sgd_convert_functions_loader_CSV_Loader.yaml")

for dataclass in data_classes_rfc:
    if type(dataclass).__name__ == "Model":
        model = dataclass
    if type(dataclass).__name__ == "Loader":
        loader = dataclass
    if type(dataclass).__name__ == "TrainTestSplit":
        traintestsplit = dataclass

for dataclass in data_classes_sgd:
    if type(dataclass).__name__ == "Model":
        sgd_model = dataclass
    if type(dataclass).__name__ == "Loader":
        sgd_loader = dataclass
    if type(dataclass).__name__ == "TrainTestSplit":
        sgd_traintestsplit = dataclass



def test_random_forest_classifier():
    """"""
    rfc = algorithms_implementation.RandomForestClassifier(model, "src/tests/data/model/" + model.file_name)
    assert rfc.__getattribute__("_ensemble_model") == "RandomForestClassifier"
    assert rfc.__getattribute__("_model_path") == "src/tests/data/model/" + model.file_name




def test_rfc_fit_predict():
    rfc = algorithms_implementation.RandomForestClassifier(model, "src/tests/data/model/" + model.file_name)
    df = pd.read_csv("src/tests/data/raw/raw_data.csv")
    X, y = separate_features_outcome(df, traintestsplit.target)
    rfc.fit(X, y)
    assert type(rfc.__getattribute__("_model")) == sklearn.ensemble._forest.RandomForestClassifier
    y_pred = rfc.predict(X)
    assert y.shape == y_pred.shape

    rfc._criterion = "ad"
    with pytest.raises(SystemExit):
        rfc.fit(X,y)


def test_stochastic_gradient_decent():
    """"""
    sgd = algorithms_implementation.StochasticGradientDecentClassifier(sgd_model,
                                                                       "src/tests/data/model/" + sgd_model.file_name)
    assert sgd.__getattribute__("_ensemble_model") == "StochasticGradientDecent"
    assert sgd.__getattribute__("_model_path") == "src/tests/data/model/" + sgd_model.file_name


def test_sgd_fit_predict():
    sgd = algorithms_implementation.StochasticGradientDecentClassifier(sgd_model,
                                                                       "src/tests/data/model/" + sgd_model.file_name)
    df = pd.read_csv("src/tests/data/raw/raw_data.csv")
    X, y = separate_features_outcome(df, sgd_traintestsplit.target)
    sgd.fit(X, y)
    assert type(sgd.__getattribute__("_model")) == sklearn.linear_model._stochastic_gradient.SGDClassifier
    y_pred = sgd.predict(X)
    assert y.shape == y_pred.shape


def test_algorithm_not_implemented():
    with pytest.raises(SystemExit):
        algorithms_creation.create_algorithm("Fehler")


def test_main():
    with pytest.raises(FileNotFoundError):
        algorithms_creation.main("src/tests/data/configuration/Model.json",
                                 "src/tests/data/configuration/TrainTestSplit.json")

    with pytest.raises(SystemExit):
        algorithms_creation.main("Fehler", "Fehler")


def test_Converted_algorithm():
    for dataclass in data_classes_converted:
        if type(dataclass).__name__ == "Model":
            model_converted = dataclass

    factory = algorithms_creation.create_algorithm(model_converted.ensemble_model)
    """Train and predict Model"""
    training_algorithm = factory.get_algorithm(model_converted, "data/model/" + model_converted.file_name)

    model_converted.name_of_class = "Fehler"

    with pytest.raises(SystemExit):
        training_algorithm = factory.get_algorithm(model_converted, "data/model/" + model_converted.file_name)
