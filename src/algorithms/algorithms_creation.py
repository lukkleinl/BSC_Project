import importlib
import json
import sys
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report

from algorithms.algorithms_implementation import RandomForestClassifier, StochasticGradientDecentClassifier
from data_classes.data_classes import Model, TrainTestSplit
from preprocessing.preprocessing import separate_features_outcome


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
        module = importlib.import_module("conversion." + model.name_of_module)

        """tries to load function with given name and execute it"""
        try:
            algorithm_class = module.__getattribute__(model.name_of_class)
            algorithm_class = algorithm_class(model, out_path)
        except AttributeError as err:
            sys.exit(f"specified {err}")

        return algorithm_class


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


def main(model_path: str, train_test_split_path: str):
    """Parameter Specification and train tes split"""
    with open(model_path, 'r') as file:
        model = json.load(file)

    with open(train_test_split_path, 'r') as file:
        train_test_split = json.load(file)

    model = Model(**model)
    train_test_split = TrainTestSplit(**train_test_split)
    # train_size = train_test_split.parameter_train_test_split[0]
    # random_state = train_test_split.parameter_train_test_split[1]

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    train_features, train_labels = separate_features_outcome(train, train_test_split.target)
    test_features, test_labels = separate_features_outcome(test, train_test_split.target)

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
    print(classification_report(test_labels, y_predicted))
