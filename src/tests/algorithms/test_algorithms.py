import sklearn.ensemble

from algorithms.algorithms_impl import RandomForestClassifier, StochasticGradientDecentClassifier
from main import load_configuration
from preprocessing.data_loader import create_loader
from preprocessing.transform_data import separate_features_outcome

paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter = load_configuration("src/tests/data"
                                                                                                      "/config_files"
                                                                                           "/test_config_rfc.yaml")
paths_sgd, loader_sgd, model_sgd, dataclass_sgd, prepr_steps_after_sgd, prepr_steps_prior_sgd, converter = \
    load_configuration("src/tests/data/config_files/test_config_sgd.yaml")


def test_random_forest_classifier():
    """"""
    rfc = RandomForestClassifier(model, paths.model_path + model.file_name)
    assert rfc.__getattribute__("_ensemble_model") == "RandomForestClassifier"
    assert rfc.__getattribute__("_model_path") == paths.model_path + model.file_name


def test_rfc_fit_predict():
    rfc = RandomForestClassifier(model, paths.model_path + model.file_name)
    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader, paths.raw_data_path)
    df = load.get_data()
    X, y = separate_features_outcome(df, {"target": dataclass.target})
    rfc.fit(X, y)
    assert type(rfc.__getattribute__("_model")) == sklearn.ensemble._forest.RandomForestClassifier
    y_pred = rfc.predict(X)
    assert y.shape == y_pred.shape


def test_stochastic_gradient_decent():
    """"""
    sgd = StochasticGradientDecentClassifier(model_sgd, paths_sgd.model_path + model_sgd.file_name)
    assert sgd.__getattribute__("_ensemble_model") == "StochasticGradientDecent"
    assert sgd.__getattribute__("_model_path") == paths_sgd.model_path + model_sgd.file_name


def test_sgd_fit_predict():
    sgd = StochasticGradientDecentClassifier(model_sgd, paths_sgd.model_path + model_sgd.file_name)
    loader_factory = create_loader(loader_sgd.name)
    load = loader_factory.get_loader(loader_sgd, paths_sgd.raw_data_path)
    df = load.get_data()
    X, y = separate_features_outcome(df, {"target": dataclass_sgd.target})
    sgd.fit(X, y)
    assert type(sgd.__getattribute__("_model")) == sklearn.linear_model._stochastic_gradient.SGDClassifier
    y_pred = sgd.predict(X)
    assert y.shape == y_pred.shape
