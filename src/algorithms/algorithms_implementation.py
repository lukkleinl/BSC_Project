import pickle
import sys
from abc import abstractmethod, ABC

from sklearn.ensemble import RandomForestClassifier as RFC_Sklearn
from sklearn.linear_model import SGDClassifier

from data_classes.data_classes import Model


class BaseAlgorithm(ABC):
    """Base class for the implemented algorithms"""

    _model = """prediction model"""
    _ensemble_model = """name of used algorithm"""
    _model_path = """path where the model gets saved"""

    def __init__(self, model: Model, out_path: str):
        self._ensemble_model = model.ensemble_model
        self._model_path = out_path

    def _save_model(self, model):
        """

        :param model:
        :return:
        """
        pickle.dump(model, open(self._model_path, 'wb'))

    def load_model(self):
        """

        :return:
        """
        _model = pickle.load(open(self._model_path, 'rb'))

    @abstractmethod
    def fit(self, features, labels):
        """"Fits the data"""

    @abstractmethod
    def predict(self, features):
        """predicts on given data"""


class RandomForestClassifier(BaseAlgorithm):
    """Random Forest Classifier algorithm"""

    """Number of estimators for the model"""
    _n_estimators = 100

    """The maximum depth of the tree"""
    _max_depth = None

    """The number of features to consider when looking for the best split"""
    _max_features = "auto"

    """The function to measure the quality of a split"""
    """gini or entropy are supported within the sklearn implementation"""
    _criterion = "gini"

    """Seed for the random estimator"""
    _random_state = None

    def __init__(self, model: Model, out_path: str):
        """

        :param model:
        :param out_path:
        """

        super().__init__(model, out_path)

        if "n_estimators" in model.model_config:
            self._model_n_estimators = model.model_config["n_estimators"]

        if "criterion" in model.model_config:
            self._criterion = model.model_config["criterion"]

        if "max_depth" in model.model_config:
            self._max_depth = model.model_config["max_depth"]

        if "max_features" in model.model_config:
            self._max_depth = model.model_config["max_features"]

        if "random_state" in model.model_config:
            self._model_random_state = model.model_config["random_state"]

    def fit(self, features, labels):
        """

        :param features:
        :param labels:
        :return:
        """
        try:
            self._model = RFC_Sklearn(n_estimators=self._n_estimators, random_state=self._random_state,
                                      criterion=self._criterion, max_depth=self._max_depth,
                                      max_features=self._max_features)
        except KeyError as err:
            sys.exit(f"  wrong key {err} for the Random Forest")

        self._save_model(self._model)

    def predict(self, features):
        """

        :param features:
        :return:
        """
        y_predicted = self._model.predict(features)
        return y_predicted


class StochasticGradientDecentClassifier(BaseAlgorithm):
    """Stochastic Gradient Decent Classifier algorithm"""

    def fit(self, features, labels):
        """
        
        :param features: 
        :param labels: 
        :return: 
        """

        self._model = SGDClassifier(penalty=None)
        self._model.fit(features, labels)
        self._save_model(self._model)

    def predict(self, features):
        """

        :param features:
        :return:
        """
        y_predicted = self._model.predict(features)
        return y_predicted
