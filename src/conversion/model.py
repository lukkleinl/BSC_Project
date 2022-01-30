#Example Transformation into class

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from algorithms.algorithms_implementation import BaseAlgorithm
from data_classes.data_classes import Model


class grid(BaseAlgorithm):
    params_grid = {'C': [0.001, 10, 100, 1000],
                   'gamma': [1, 0.1, 0.01, 0.001],
                   'degree': [2, 3, 4, 5],
                   'coef0': [0, 1, 2, 4]

                   }
    poly_best = None
    grid_search = GridSearchCV(SVC(C='poly'), params_grid, verbose=2)

    def __init__(self, model: Model, outpath: str):
        super().__init__(model, outpath)

    def get_model(self):
        super().get_model()

    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.poly_best = self.grid_search.best_estimator_.fit(X_train, y_train)
        self._save_model(self._model)

    # In grid search definition above, if you set parameter re_fit to True, you won't have to do this
    def predict(self, X_train):
        grid_pred = self.poly_best.predict(X_train)
        return grid_pred