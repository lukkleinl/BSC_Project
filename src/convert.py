import sys

import click

from data_classes.data_classes import Converter
from conversion import conversion_functions


@click.command()
@click.argument("config_file", type=str, default="../data/config_files/config_rfc_no_convert_CSV_Loader.yaml")
@click.argument("notebook_path", type=str, default="../data/notebooks/")
def convert_preprocessing_steps(config_file, notebook_path):
    output_path = "conversion/"

    try:
        conf = conversion_functions.get_config_from_yaml(config_file)
        converter = Converter(**conf["Converter"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    for processing_step in converter.preprocessing_tags:
        conversion_functions.convert_cells(notebook_path + converter.notebook_name, processing_step, output_path)


if __name__ == "__main__":
    convert_preprocessing_steps()
