from configuration.configuration import load_configuration

data_classes_rfc = load_configuration("src/tests/data/config_files/config_rfc_convert_functions_CSV_Loader.yaml")

for dataclass in data_classes_rfc:
    if type(dataclass).__name__ == "PreprocessingSteps":
        prepr_steps = dataclass


def test_list_of_processing_steps():
    array_of_steps = prepr_steps.get_list_of_preprocessing_steps()
    assert len(array_of_steps) == 2