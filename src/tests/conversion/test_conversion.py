import importlib

import pytest
import yaml

from conversion.conversion_functions import get_config_from_yaml, convert_cells, get_parameters_from_notebook
from main import load_configuration
from tests.conversion.parameters_from_notebook import get_params_notebook

paths, loader, model, dataclass, prepr_steps_after, prepr_steps_prior, converter = load_configuration(
    "src/tests/data/config_files"
    "/test_config_rfc.yaml")


def test_get_config_from_yaml():
    params = get_config_from_yaml("src/tests/data/config_files/test_config_rfc.yaml")
    with open("src/tests/data/config_files/test_config_rfc.yaml") as file:
        test_params = yaml.load(file, Loader=yaml.FullLoader)
    assert params == test_params


def test_convert_cells():
    convert_cells(paths.notebook_path + converter.notebook_name, "transform_data_func",
                  paths.output_path_converted_files)

    """load module of preprocessing step"""
    module = importlib.import_module("conversion.transform_data_func")
    assert module
    func = module.__getattribute__("transform_data_func")


def test_parameters_from_notebook():
    params = get_parameters_from_notebook(paths.notebook_path + converter.notebook_name)
    parameter_from_function = get_params_notebook()

    assert params == parameter_from_function

    with pytest.raises(SystemExit):
        get_parameters_from_notebook(paths.notebook_path + "prediction-of-quality-of-wine_without_params.ipynb")
