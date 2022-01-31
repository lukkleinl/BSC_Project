import json
import sys

from conversion import conversion_functions
from data_classes import data_classes


def _get_parameter(config):
    """
    loads the parameters either from yaml file or from the specified notebook"
    :param config: Configuration file
    :return: config: Parameters either from notebook or from data_classes file
    """

    converter = data_classes.Converter(**config["Converter"])

    """Deciding on where to load parameter from"""
    if converter.parameter_conversion:
        params_nb = conversion_functions.get_parameters_from_notebook(converter.location_of_notebook +
                                                                      converter.notebook_name)
        return params_nb
    else:
        return config


def load_configuration(config_file):
    """
    loads data from specified config file
    and maps it to the defined data classes

    :param config_file: location of yaml file where either the parameters are stored or the credentials
                        of the notebooks from where the parameters can load from
    :return: paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter
             dataclasses which were specified in configuration file
    """
    try:
        conf = conversion_functions.get_config_from_yaml(config_file)
        params = _get_parameter(conf)
        loader = data_classes.Loader(**params["Loader"])
        model = data_classes.Model(**params["Model"])
        train_test_split = data_classes.TrainTestSplit(**params["TrainTestSplit"])
        converter = data_classes.Converter(**params["Converter"])
        preprocessing_steps = data_classes.PreprocessingSteps(params["Preprocessing_Steps"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes in the following class {err}")
    except TypeError as err:
        sys.exit(f"Missing argument {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    return [loader, model, train_test_split, preprocessing_steps, converter]


def get_config(config_file: str,path_to_data: str):
    """

    :param config_file:
    :return:
    """
    config_classes = load_configuration(config_file)

    for dataclass in config_classes:
        with open( path_to_data + type(dataclass).__name__ + ".json", 'w') as file:
            file.write(json.dumps(dataclass.__dict__))

def main(config_file:str):
    path_to_data = "data/configuration/"
    get_config(config_file,path_to_data)




