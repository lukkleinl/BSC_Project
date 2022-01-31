import importlib
import sys

import pytest
import yaml

from configuration.configuration import load_configuration
from conversion.conversion_functions import get_config_from_yaml, convert_cells, get_parameters_from_notebook

data_classes_rfc = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")

for dataclass in data_classes_rfc:
    if type(dataclass).__name__ == "Converter":
        converter = dataclass


def test_get_config_from_yaml():
    params = get_config_from_yaml("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")
    with open("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml") as file:
        test_params = yaml.load(file, Loader=yaml.FullLoader)
    assert params == test_params


def test_convert_cells():
    convert_cells("src/tests/data/notebooks/" + converter.notebook_name, "transform_data_func", "src/tests/conversion/")

    """load module of preprocessing step"""
    module = importlib.import_module("conversion.transform_data_func")

    assert module.__name__ == "conversion.transform_data_func"


def test_parameters_from_notebook():
    params = get_parameters_from_notebook("src/tests/data/notebooks/" + converter.notebook_name)
    assert params["Loader"]
    assert params["Converter"]
    assert params["TrainTestSplit"]
    assert params["Model"]
    assert params["Preprocessing_Steps"]

    with pytest.raises(KeyError):
        params = get_parameters_from_notebook("src/tests/data/notebooks/" + "prediction-of-quality-of-wine_broken_params.ipynb")
        assert params["Loader"]
        assert params["Converter"]
        assert params["TrainTestSplit"]
        assert params["Model"]

    with pytest.raises(SystemExit):
        get_parameters_from_notebook("src/tests/data/notebooks/" + "prediction-of-quality-of-wine_without_params.ipynb")
