from data_classes.data_classes import Model
from sklearn.ensemble import RandomForestClassifier as RFC_Sklearn


class RandomForestClassifier:
    """Random Forest Classifier algorithm"""

    _model_n_estimators = """Number of estimators for the model"""
    _model_random_state = """Seed for the random estimator"""

    def __init__(self, model: Model, out_path: str):
        """

        :param model:
        :param out_path:
        """
        self._model = None
        self._model_n_estimators = model.model_config["n_estimators"]
        self._model_random_state = model.model_config["random_state"]
        self._ensemble_model = model.ensemble_model
        self._model_path = out_path

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
