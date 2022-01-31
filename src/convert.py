import sys

import click

from data_classes.data_classes import Converter
from conversion import conversion_functions


@click.command()
@click.argument("config_file", type=str, default="../data/config_files/Wine_quality/config_sgd_convert_functions_loader_CSV_Loader.yaml")
def convert_preprocessing_steps(config_file):
    output_path = "conversion/"

    try:
        conf = conversion_functions.get_config_from_yaml(config_file)
        converter = Converter(**conf["Converter"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    if converter.preprocessing_tags is not None:
        for processing_step in converter.preprocessing_tags:
            conversion_functions.convert_cells(converter.location_of_notebook + converter.notebook_name, processing_step, output_path)


if __name__ == "__main__":
    convert_preprocessing_steps()
