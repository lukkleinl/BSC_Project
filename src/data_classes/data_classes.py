from array import array
from dataclasses import dataclass
from typing import Optional

import data_classes


@dataclass
class Paths:
    """stores the paths to the needed folders"""
    raw_data_path: str
    preprocessing_path: str
    processed_path: str
    model_path: str
    notebook_path: str
    output_path_converted_files: Optional[str] = None


@dataclass
class Dataset:
    """Parameters for train test split"""
    target: str
    parameter_train_test_split: Optional[array] = None


@dataclass
class Converter:
    """Specifies if a conversion is needed
       It can be chosen if we want to convert Parameters
       or preprocessing steps or both"""

    convert_preprocessing_steps: bool
    parameter_conversion: bool
    notebook_name: Optional[str] = None
    preprocessing_tags: Optional[array] = None


@dataclass
class Loader:
    """Specifies the Information for the loader
        which loader should be used and an url can be specified
        if needed to load the data"""

    name: str
    name_of_file: str
    url: Optional[str] = None


@dataclass
class Model:
    """Specifies the credentials for the ensemble model which should be used"""
    file_name: str
    ensemble_model: str
    model_config: Optional[dict] = None
    export_result: Optional[bool] = False
    conversion: Optional[bool] = False


@dataclass
class PrepStep:
    """Stores the Information of a preprocessing step
       Preprocessing steps can be either loaded from already existing modules
       or from recently converted files"""

    """location has to be specified with a dot. For instance preprocessing."""
    location_of_step: str
    name_of_step: str
    name_of_module: str
    conversion: bool
    params: Optional[dict] = None


@dataclass
class PreprocessingSteps:
    """List of Preprocessing steps"""

    names_of_steps: Optional[dict] = None

    def get_list_of_preprocessing_steps(self):
        """
        creates a list of the specified preprocessing steps

        :param prepr_steps: Dictionary of Preprocessing steps
        :return: list of preprocessing steps
        """

        """getting the names"""
        steps_names = [test for test in self.names_of_steps]

        """creating list of steps"""
        preprocessing_step = []
        for step in steps_names:
            """Create Preprocessing Step"""
            prep_step = PrepStep(self.names_of_steps[step]["params"]["location_of_step"],
                                 self.names_of_steps[step]["params"]["name_of_step"],
                                 self.names_of_steps[step]["params"]["name_of_module"],
                                 **self.names_of_steps[step])

            """Append it to array"""
            preprocessing_step.append(prep_step)

        return preprocessing_step
