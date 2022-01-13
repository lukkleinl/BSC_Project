import pickle
from abc import abstractmethod, ABC

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier as RFC_Sklearn
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

from data_classes.data_classes import Model


class BaseAlgorithm(ABC):
    """Base class for the implemented algorithms"""

    """model variables"""
    _model = """prediction model"""
    _ensemble_model = """name of used algorithm"""
    _model_path = """path where the model gets saved"""

    def __init__(self, model: Model, out_path: str):
        self._ensemble_model = model.ensemble_model
        self._model_path = out_path

    """"Methods to save and load models"""

    def _save_model(self, model):
        pickle.dump(model, open(self._model_path, 'wb'))

    def _load_model(self):
        _model = pickle.load(open(self._model_path, 'rb'))

    """base methods to fit and predict"""

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
        self._model_n_estimators = model.model_config["n_estimators"]
        self._model_random_state = model.model_config["random_state"]
        super().__init__(model, out_path)

    def fit(self, X, y):
        self._model = RFC_Sklearn()
        self._model.fit(X, y)
        self._save_model(self._model)

    def predict(self, X, y):
        self._load_model()
        y_predicted = self._model.predict(X)
        mean_square_error = mean_squared_error(y, y_predicted)
        print(accuracy_score(y, y_predicted))
        print(mean_square_error)
        print(confusion_matrix(y, y_predicted))


class StochasticGradientDecentClassifier(BaseAlgorithm):
    """Stochastic Gradient Decent Classifier algorithm"""

    def fit(self, X, y):
        self._model = SGDClassifier(penalty=None)
        self._model.fit(X, y)
        self._save_model(self._model)

    def predict(self, X, y):
        self._load_model()
        y_predicted = self._model.predict(X)
        mean_square_error = mean_squared_error(y, y_predicted)
        print(accuracy_score(y, y_predicted))
        print(mean_square_error)
        print(confusion_matrix(y, y_predicted))
