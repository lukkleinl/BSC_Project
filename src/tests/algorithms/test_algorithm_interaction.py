import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import algorithms
from algorithms import algorithms_creation
from main import load_configuration
from preprocessing.data_loader import create_loader
from preprocessing.transform_data import separate_features_outcome

paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter = load_configuration("src/tests/data/config_files"
                                                                                           "/test_config_rfc.yaml")

sgd_paths, sgd_loader, sgd_model, sgd_dataclass, sgd_prepr_steps_after, sgd_prepr_steps_prior, converter = load_configuration(
    "src/tests/data/config_files"
    "/test_config_sgd.yaml")


def test_algorithm_factory_rfc():
    fac = algorithms_creation.create_algorithm(model.ensemble_model)
    assert type(fac) == algorithms.algorithms_creation.RandomForestFactory
    algorithm = fac.get_algorithm(model, paths.model_path + model.file_name)
    assert type(algorithm) == algorithms.algorithms_creation.RandomForestClassifier


def test_algorithm_factory_sgd():
    fac = algorithms_creation.create_algorithm(sgd_model.ensemble_model)
    assert type(fac) == algorithms.algorithms_creation.StochasticGradientDecentFactory
    algorithm = fac.get_algorithm(sgd_model, sgd_paths.model_path + sgd_model.file_name)
    assert type(algorithm) == algorithms.algorithms_creation.StochasticGradientDecentClassifier


def test_create_train_predict():
    factory = algorithms_creation.create_algorithm(model.ensemble_model)
    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader, paths.raw_data_path)
    df = load.get_data()
    X, y = separate_features_outcome(df, {"target": dataclass.target})
    train_size = dataclass.parameter_train_test_split[0]
    random_state = dataclass.parameter_train_test_split[1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=random_state)
    """Train and predict Model"""
    training_algorithm = factory.get_algorithm(model, paths.model_path + model.file_name)
    training_algorithm.fit(x_train, y_train)
    y_predicted = training_algorithm.predict(x_test)
    mean_square_error = mean_squared_error(y_test, y_predicted)
    print(accuracy_score(y_test, y_predicted))
    print(mean_square_error)
    print(confusion_matrix(y_test, y_predicted))
    y_test = pd.DataFrame(y_test)
    y_predicted = pd.DataFrame(y_predicted)
    assert y_test.shape == y_predicted.shape
