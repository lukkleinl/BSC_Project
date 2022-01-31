from array import array
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainTestSplit:
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
    location_of_notebook: Optional[str] = None
    notebook_name: Optional[str] = None
    preprocessing_tags: Optional[array] = None


@dataclass
class Loader:
    """Specifies the Information for the loader
        which loader should be used and an url can be specified
        if needed to load the data"""

    """name of the desired loader"""
    name: str

    """used when loader is from conversion"""
    name_of_module: Optional[str] = None

    """used when loader is from conversion"""
    name_of_class: Optional[str] = None

    """specifies the path to the csv file where the data is stored"""
    path_to_csv: Optional[str] = None

    """specifies the URL from where the data is loaded"""
    url: Optional[str] = None


@dataclass
class Model:
    """Specifies the credentials for the ensemble model which should be used"""
    file_name: str
    ensemble_model: str
    model_config: dict
    export_result: Optional[bool] = False
    conversion: Optional[bool] = False
    name_of_class: Optional[str] = None
    name_of_module: Optional[str] = None


@dataclass
class PrepStep:
    """Stores the Information of a preprocessing step
       Preprocessing steps can be either loaded from already existing modules
       or from recently converted files"""

    """location has to be specified with a dot. For instance preprocessing."""

    name_of_step: str
    name_of_module: str
    conversion: bool
    prior_to_split: bool
    params: Optional[dict] = None


@dataclass
class PreprocessingSteps:
    """List of Preprocessing steps"""

    names_of_steps: Optional[dict] = None

    def get_list_of_preprocessing_steps(self):
        """
        creates a list of the specified preprocessing steps

        preprocessing_steps: Dictionary of Preprocessing steps

        :param
        :return: list of preprocessing steps
        """

        """getting the names"""
        steps_names = [step for step in self.names_of_steps]

        """creating list of steps"""
        preprocessing_step = []
        for step in steps_names:
            """Create Preprocessing Step"""
            prep_step = PrepStep(**self.names_of_steps[step])

            """Append it to array"""
            preprocessing_step.append(prep_step)

        return preprocessing_step
