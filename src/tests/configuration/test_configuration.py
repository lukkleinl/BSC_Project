import pytest

from configuration.configuration import get_config, load_configuration, _get_parameter
from conversion import conversion_functions


def test_get_config():
    with pytest.raises(FileNotFoundError):
        get_config("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")


def test_load_configuration_wrong_path():
    with pytest.raises(SystemExit):
        data_classes = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loadear.yaml")


def test_load_configuration_wrong_parameters():
    with pytest.raises(SystemExit):
        data_classes = load_configuration(
            "src/tests/data/config_files/config_rfc_convert_functions_CSV_Loadear_wrong.yaml")


def test_get_parameter_without_conversion():
    config = conversion_functions.get_config_from_yaml(
        "src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")
    config = _get_parameter(config)
    conf = conversion_functions.get_config_from_yaml(
        "src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")
    print(conf)
    print(config)

def test_get_parameter_with_conversion():
    config = conversion_functions.get_config_from_yaml(
        "src/tests/data/config_files/config_rfc_convert_params_CSV_Loader.yaml")
    config = _get_parameter(config)
    conf = conversion_functions.get_config_from_yaml(
        "src/tests/data/config_files/config_rfc_convert_params_CSV_Loader.yaml")

    print("adav\n",config)
    print(conf)