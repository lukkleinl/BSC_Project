import sys
from abc import ABC, abstractmethod

from algorithms.algorithms_impl import RandomForestClassifier, StochasticGradientDecentClassifier
from data_classes.data_classes import Model


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


def create_algorithm(algorithm: str) -> AlgorithmFactory:
    """used to identify which factory should be used"""

    factories = {
        """list of implemented factories"""
        "RandomForestClassifier": RandomForestFactory(),
        "StochasticGradientDecent": StochasticGradientDecentFactory()
    }

    if algorithm in factories:
        return factories[algorithm]

    sys.exit("algorithm name was not recognized")


def create_train_predict(fac: AlgorithmFactory, x_train, y_train, x_test, y_test, model, out_path) -> None:
    """gets the specified algorithm
        fits and predicts model on given data"""

    algorithm = fac.get_algorithm(model, out_path + model.file_name)
    algorithm.fit(x_train, y_train)
    algorithm.predict(x_test, y_test)