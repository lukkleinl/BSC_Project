import pandas as pd
from sklearn.model_selection import train_test_split

import algorithms
from algorithms import algorithms_creation
from configuration.configuration import load_configuration
from loader.data_loader import create_loader
from preprocessing.preprocessing import separate_features_outcome

data_classes_rfc = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")

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


def test_algorithm_factory_rfc():
    fac = algorithms_creation.create_algorithm(model.ensemble_model)
    assert type(fac) == algorithms.algorithms_creation.RandomForestFactory
    algorithm = fac.get_algorithm(model, "src/tests/data/model/" + model.file_name)
    assert type(algorithm) == algorithms.algorithms_creation.RandomForestClassifier


def test_algorithm_factory_sgd():
    fac = algorithms_creation.create_algorithm(sgd_model.ensemble_model)
    assert type(fac) == algorithms.algorithms_creation.StochasticGradientDecentFactory
    algorithm = fac.get_algorithm(sgd_model, "src/tests/data/model/" + sgd_model.file_name)
    assert type(algorithm) == algorithms.algorithms_creation.StochasticGradientDecentClassifier


def test_create_train_predict():
    df = pd.read_csv("src/tests/data/raw/raw_data.csv")
    X, y = separate_features_outcome(df, traintestsplit.target)
    train_size = traintestsplit.parameter_train_test_split["test_size"]
    random_state = traintestsplit.parameter_train_test_split["random_state"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=random_state)
    """Train and predict Model"""
    factory = algorithms_creation.create_algorithm(model.ensemble_model)
    training_algorithm = factory.get_algorithm(model, "src/tests/data/model/" + model.file_name)
    training_algorithm.fit(x_train, y_train)
    y_predicted = training_algorithm.predict(x_test)
    y_test = pd.DataFrame(y_test)
    y_predicted = pd.DataFrame(y_predicted)
    assert y_test.shape == y_predicted.shape
