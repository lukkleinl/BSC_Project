import importlib
import sys
import urllib.error

import click
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

from algorithms import algorithms_interaction
from data_classes import data_classes
from conversion.conversion import get_parameters_from_notebook, get_config_from_yaml
from preprocessing.data_loader import create_loader
from preprocessing.transform_data import separate_features_outcome


def get_parameter(config):
    """
    loads the parameters either from yaml file or from the specified notebook"
    :param config: Configuration file
    :return: Parameters either from notebook or from data_classes file
    """

    converter = data_classes.Converter(**config["Converter"])
    paths = data_classes.Paths(**config["Paths"])

    """Deciding on where to load parameter from"""
    if converter.parameter_conversion:
        params_nb = get_parameters_from_notebook(paths.notebook_path + converter.notebook_name)
        return params_nb
    else:
        return config


def import_and_load_prepr_function(preprocessing_step: data_classes.PrepStep, df: pd.DataFrame):
    """
    First loads the module of the converted file,
       after that the specified function is loaded and executed.
       The preprocessing functions should all return a Dataframe
    :param preprocessing_step:
    :param df: Dataframe
    :return: Preprocessed Dataframe
    """

    """declaration of parameters"""
    location_of_step = preprocessing_step.location_of_step
    name_of_step = preprocessing_step.name_of_step
    name_of_module = preprocessing_step.name_of_module
    params = preprocessing_step.params

    """load module of preprocessing step"""
    module = importlib.import_module(location_of_step + name_of_module)

    """tries to load function with given name and execute it"""
    try:
        func = module.__getattribute__(name_of_step)
        df = func(df, params)
    except AttributeError as err:
        sys.exit(f"specified {err}")

    return df


def load_configuration(config_file):
    """
    loads data from specified config file
    :param config_file: location of yaml file where either the parameters are stored or the credentials
                        of the notebooks from where the parameters can load from
    :return: dataclasses which were specified in configuration file
    """
    try:
        conf = get_config_from_yaml(config_file)
        params = get_parameter(conf)
        paths = data_classes.Paths(**params["Paths"])
        loader = data_classes.Loader(**params["Loader"])
        model = data_classes.Model(**params["Model"])
        dataclass = data_classes.Dataset(**params["Dataset"])
        converter = data_classes.Converter(**params["Converter"])
        prepr_steps_prior = data_classes.PreprocessingSteps(params["Preprocessing_Steps_pre_split"])
        prepr_steps_after = data_classes.PreprocessingSteps(params["Preprocessing_Steps_after_split"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes in the following class {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    return paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter


@click.command()
@click.argument("config_file", type=str, default="data/config_files/config_rfc_no_convert.yaml")
def create_model(config_file):
    """
    Main function to create a model with the given configuration
    :param config_file:
    :return:
    """

    """Configuration"""
    paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter = load_configuration(config_file)

    """Load Data"""
    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader, paths.raw_data_path)

    try:
        df = load.get_data()
    except urllib.error.URLError as err:
        sys.exit(f"{err} \ncan not load csv from URL\nchange to another loader")

    """Preprocess Data prior to separation of features"""
    preprocessing_steps = prepr_steps_prior.get_list_of_preprocessing_steps()

    for step in preprocessing_steps:
        df = import_and_load_prepr_function(step, df)
        df.to_csv(paths.preprocessing_path + step.name_of_step + ".csv")

    """Separate Feature and Outcome + scale Data"""
    X, y = separate_features_outcome(df, {"target": dataclass.target})

    """Preprocess Data after separation of features"""
    preprocessing_steps = prepr_steps_after.get_list_of_preprocessing_steps()

    for step in preprocessing_steps:
        X = import_and_load_prepr_function(step, X)
        X = pd.DataFrame(X)
        df = X.join(y)
        df.to_csv(paths.preprocessing_path + step.name_of_step + ".csv")

    df = X.join(y)
    df.to_csv(paths.processed_path + "processed.csv")
    """Parameter Specification and train tes split"""
    train_size = dataclass.parameter_train_test_split[0]
    random_state = dataclass.parameter_train_test_split[1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=random_state)

    """join and export to csv"""
    df_train = x_train.join(y_train)
    df_test = x_test.join(y_test)
    df_train.to_csv(paths.processed_path + "train_split.csv")
    df_test.to_csv(paths.processed_path + "test_split.csv")

    """Create Factory for specified Model"""
    factory = algorithms_interaction.create_algorithm(model.ensemble_model)

    """Train and predict Model"""
    algorithm = factory.get_algorithm(model, paths.model_path + model.file_name)
    algorithm.fit(x_train, y_train)
    y_predicted = algorithm.predict(x_test, y_test)
    mean_square_error = mean_squared_error(y_test, y_predicted)
    print(accuracy_score(y_test, y_predicted))
    print(mean_square_error)
    print(confusion_matrix(y_test, y_predicted))


if __name__ == "__main__":
    create_model()
