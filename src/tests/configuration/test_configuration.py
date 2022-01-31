import json

import pytest

from configuration.configuration import get_config, load_configuration, _get_parameter
from conversion import conversion_functions


def test_get_config():
    with pytest.raises(SystemExit):
        get_config("src/tests/data/config_files/config_rfc_convert_functionsa_CSV_Loader.yaml",
                   "src/tests/data/configuration")

    get_config("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml",
               "src/tests/data/configuration/")

    with open("src/tests/data/configuration/Model.json", 'r') as file:
        model = json.load(file)

    assert model


def test_load_configuration_wrong_path():
    with pytest.raises(SystemExit):
        data_classes = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loadear.yaml")


def test_load_configuration_wrong_parameters():
    with pytest.raises(SystemExit):
        data_classes = load_configuration(
            "src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader_wrong.yaml")


def test_load_configuration():
    with pytest.raises(SystemExit):
        data_classes = load_configuration(
            "src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader_wrong+.yaml")


