from sklearn.model_selection import train_test_split

import algorithms
from algorithms.algorithms_interaction import RandomForestFactory, create_algorithm, create_train_predict
from main import load_configuration
from preprocessing.data_loader import create_loader
from preprocessing.transform_data import separate_features_outcome

paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior , converter= load_configuration("tests/data/config_files"
                                                                                           "/test_config_rfc.yaml")

sgd_paths, sgd_loader, sgd_model, sgd_dataclass, sgd_prepr_steps_after, sgd_prepr_steps_prior, converter = load_configuration(
    "tests/data/config_files"
    "/test_config_sgd.yaml")


def test_algorithm_factory_rfc():
    fac = create_algorithm(model.ensemble_model)
    assert type(fac) == algorithms.algorithms_interaction.RandomForestFactory
    algorithm = fac.get_algorithm(model, paths.model_path + model.file_name)
    assert type(algorithm) == algorithms.algorithms_interaction.RandomForestClassifier


def test_algorithm_factory_sgd():
    fac = create_algorithm(sgd_model.ensemble_model)
    assert type(fac) == algorithms.algorithms_interaction.StochasticGradientDecentFactory
    algorithm = fac.get_algorithm(sgd_model, sgd_paths.model_path + sgd_model.file_name)
    assert type(algorithm) == algorithms.algorithms_interaction.StochasticGradientDecentClassifier


def test_create_train_predict():
    fac = create_algorithm(model.ensemble_model)
    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader, paths.raw_data_path)
    df = load.get_data()
    X, y = separate_features_outcome(df, {"target": dataclass.target})
    train_size = dataclass.parameter_train_test_split[0]
    random_state = dataclass.parameter_train_test_split[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=random_state)
    create_train_predict(fac, X_train, y_train, X_test, y_test, model, paths.model_path)
