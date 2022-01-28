import pickle
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
    def fit(self, X, y):
        """"Fits the data"""

    @abstractmethod
    def predict(self, X, y):
        """predicts on given data"""


class RandomForestClassifier(BaseAlgorithm):
    """Random Forest Classifier algorithm"""

    _model_n_estimators = """Number of estimators for the model"""
    _model_random_state = """Seed for the random estimator"""

    def __init__(self, model: Model, out_path: str):
        """

        :param model:
        :param out_path:
        """
        self._model_n_estimators = model.model_config["n_estimators"]
        self._model_random_state = model.model_config["random_state"]
        super().__init__(model, out_path)

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self._model = RFC_Sklearn()
        self._model.fit(X, y)
        self._save_model(self._model)

    def predict(self, X):
        """

        :param X:
        :param y:
        :return:
        """
        y_predicted = self._model.predict(X)
        return y_predicted


class StochasticGradientDecentClassifier(BaseAlgorithm):
    """Stochastic Gradient Decent Classifier algorithm"""

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self._model = SGDClassifier(penalty=None)
        self._model.fit(X, y)
        self._save_model(self._model)

    def predict(self, X):
        """

        :param X:
        :param y:
        :return:
        """
        y_predicted = self._model.predict(X)
        return y_predicted
