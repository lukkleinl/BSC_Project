import os
import sys

import nbformat
import yaml
from nbparameterise import extract_parameters


def get_config_from_yaml(config_path: str):
    """
    loads the configuration from the given yaml file
    :param config_path  Location of the configuration file
    :return: list of parameters
    """

    with open(config_path) as file:
        parameter_list = yaml.load(file, Loader=yaml.FullLoader)
    return parameter_list


def convert_cells(path_notebook: str, processing_step: str, output_path: str):
    """
    Converts the cells from the given notebook and saves them to the specified location

    :param path_notebook:
    :param processing_step:
    :param output_path:
    :return:
    """
    notebook = nbformat.read(path_notebook, as_version=4)
    python_string_to_convert = ""
    for cell in notebook.cells:
        tags = cell["metadata"].get("tags", None)
        if tags == [processing_step]:
            python_string_to_convert += cell["source"]
            python_string_to_convert += "\n"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    f = open(output_path + processing_step + ".py", "w")
    f.write(python_string_to_convert)
    f.close()


def get_parameters_from_notebook(path_notebook: str):
    """
    Loads the parameters from the specified notebook
    Parameters have to be in the first cell otherwise
    the parameter extraction is not woking

    :param path_notebook:
    :return: found Parameters
    """

    notebook = nbformat.read(path_notebook, as_version=4)
    found_params = {}
    try:
        orig_parameters = extract_parameters(notebook)
        for param in orig_parameters:
            found_params[param.name] = param.value
        return found_params
    except SyntaxError:
        sys.exit("no parameter in first cell at " + path_notebook)
