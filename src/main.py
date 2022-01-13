import importlib
import sys
import click
import pandas as pd

from sklearn.model_selection import train_test_split

from algorithms.algorithms_interaction import create_algorithm, create_train_predict
from data_classes.data_classes import Converter, Paths, Loader, Model, Dataset, PreprocessingSteps, PrepStep
from conversion.conversion import get_parameters_from_notebook, get_config_from_yaml
from preprocessing.data_loader import create_loader
from preprocessing.transform_data import separate_features_outcome, scale_data


def get_parameter(config):
    """
    loads the parameters either from yaml file or from the specified notebook"
    :param config: Configuration file
    :return: Parameters either from notebook or from config file
    """

    converter = Converter(**config["Converter"])
    paths = Paths(**config["Paths"])

    """Deciding on where to load parameter from"""
    if converter.parameter_conversion:
        params_nb = get_parameters_from_notebook(paths.notebook_path + converter.notebook_name)
        return params_nb
    else:
        return config


def get_list_of_preprocessing_steps(prepr_steps: PreprocessingSteps):
    """
    creates a list of the specified preprocessing steps

    :param prepr_steps: Dictionary of Preprocessing steps
    :return: list of preprocessing steps
    """

    """getting the names"""
    steps_names = [test for test in prepr_steps.names_of_steps]

    """creating list of steps"""
    preprocessing_step = []
    for step in steps_names:
        """Create Preprocessing Step"""
        prep_step = PrepStep(prepr_steps.names_of_steps[step]["params"]["location_of_step"],
                             prepr_steps.names_of_steps[step]["params"]["name_of_step"],
                             prepr_steps.names_of_steps[step]["params"]["name_of_module"],
                             **prepr_steps.names_of_steps[step])

        """Append it to array"""
        preprocessing_step.append(prep_step)

    return preprocessing_step


def import_and_load_prepr_function(preprocessing_step: PrepStep, df: pd.DataFrame):
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


@click.command()
@click.argument("config_file", type=str, default="data/config_files/config_rfc_no_convert.yaml")
def create_model(config_file):
    """
    Main function to create a model with the given configuration
    :param config_file:
    :return:
    """

    """Configuration"""
    try:
        conf = get_config_from_yaml(config_file)
        params = get_parameter(conf)
        paths = Paths(**params["Paths"])
        loader = Loader(**params["Loader"])
        model = Model(**params["Model"])
        dataclass = Dataset(**params["Dataset"])
        prepr_steps_prior = PreprocessingSteps(params["Preprocessing_Steps_pre_split"])
        prepr_steps_after = PreprocessingSteps(params["Preprocessing_Steps_after_split"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes in the following class {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    """Load Data"""
    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader, paths.raw_data_path)
    df = load.get_data()

    """Preprocess Data prior to separation of features"""
    preprocessing_steps = get_list_of_preprocessing_steps(prepr_steps_prior)

    for step in preprocessing_steps:
        df = import_and_load_prepr_function(step, df)

    """Separate Feature and Outcome + scale Data"""
    X, y = separate_features_outcome(df, {"target": dataclass.target})

    """Preprocess Data after separation of features"""
    preprocessing_steps = get_list_of_preprocessing_steps(prepr_steps_after)

    for step in preprocessing_steps:
        X = import_and_load_prepr_function(step, X)

    """Parameter Specification and train tes split"""
    train_size = dataclass.parameter_train_test_split[0]
    random_state = dataclass.parameter_train_test_split[1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_size, random_state=random_state)

    """Create Factory for specified Model"""
    factory = create_algorithm(model.ensemble_model)

    """Create train and predict Model"""
    create_train_predict(factory, x_train, y_train, x_test, y_test, model, paths.model_path)


if __name__ == "__main__":
    create_model()
