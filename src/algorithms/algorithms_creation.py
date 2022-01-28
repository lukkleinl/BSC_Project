import importlib
import json
import sys
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

from algorithms.algorithms_implementation import RandomForestClassifier, StochasticGradientDecentClassifier
from data_classes.data_classes import Model, Paths


class AlgorithmFactory(ABC):
    """Factory which represents the different algorithm factories"""

    @abstractmethod
    def get_algorithm(self, model: Model, out_path: str):
        """returns the specified algorithm"""


class RandomForestFactory(AlgorithmFactory):
    """Factory to get a random forest classifier"""

    def get_algorithm(self, model: Model, out_path: str):
        return RandomForestClassifier(model, out_path)


class StochasticGradientDecentFactory(AlgorithmFactory):
    """Factory to get a Stochastic Gradient Decent Classifier"""

    def get_algorithm(self, model: Model, out_path: str):
        return StochasticGradientDecentClassifier(model, out_path)


class ConvertedAlgorithmFactory(AlgorithmFactory):
    """Factory to get a Stochastic Gradient Decent Classifier"""

    def get_algorithm(self, model: Model, out_path: str):
        """load module of preprocessing step"""
        module = importlib.import_module("conversion.algorithm")

        """tries to load function with given name and execute it"""
        try:
            loader_class = module.__getattribute__("CSVLoader")
            loader_class = loader_class()
        except AttributeError as err:
            sys.exit(f"specified {err}")

        return loader_class


def create_algorithm(algorithm: str) -> AlgorithmFactory:
    """
    used to identify which factory should be used
    :param algorithm:
    :return:
    """

    """names of factories"""
    factories = {
        "RandomForestClassifier": RandomForestFactory(),
        "StochasticGradientDecent": StochasticGradientDecentFactory(),
        "ConvertedAlgorithm": ConvertedAlgorithmFactory()
    }

    if algorithm in factories:
        return factories[algorithm]
    else:
        sys.exit(f"algorithm name was not recognized {algorithm}")


def separate_features_outcome(df: pd.DataFrame, params: dict):
    """
    Separates the features from the target

    :param df: Dataframe which should be separated
    :param params: target: specifies the target column
    :return: separated Dataframes
    """
    return df.drop(params["target"], axis=1).to_numpy(), df[params["target"]]


def main(model_path: str, train_test_split_path: str):
    """Parameter Specification and train tes split"""
    with open(model_path, 'r') as file:
        model = json.load(file)

    with open(train_test_split_path, 'r') as file:
        train_test_split = json.load(file)

    model = Model(**model)

    # train_size = train_test_split.parameter_train_test_split[0]
    # random_state = train_test_split.parameter_train_test_split[1]
    print(train_test_split)

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    train_features, train_labels = separate_features_outcome(train, {"target": "quality"})
    test_features, test_labels = separate_features_outcome(test, {"target": "quality"})

    """Create Factory for specified Model"""
    factory = create_algorithm(model.ensemble_model)

    """Train and predict Model"""
    training_algorithm = factory.get_algorithm(model, "data/model/" + model.file_name)
    training_algorithm.fit(train_features, train_labels)
    y_predicted = training_algorithm.predict(test_features)
    mean_square_error = mean_squared_error(test_labels, y_predicted)
    print(accuracy_score(test_labels, y_predicted))
    print(mean_square_error)
    print(confusion_matrix(test_labels, y_predicted))
