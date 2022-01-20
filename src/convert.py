import sys

import click

from data_classes.data_classes import Converter, Paths
from conversion import conversion_functions


@click.command()
@click.argument("config_file", type=str, default="data/config_files/config_rfc.yaml")
def convert_preprocessing_steps(config_file):
    try:
        conf = conversion_functions.get_config_from_yaml(config_file)
        converter = Converter(**conf["Converter"])
        paths = Paths(**conf["Paths"])
    except KeyError as err:
        sys.exit(f"Parameters don't match with implemented classes {err}")
    except FileNotFoundError as err:
        sys.exit(f"Specified file not found {err}")

    for processing_step in converter.preprocessing_tags:
        if paths.output_path_converted_files is not None:
            conversion_functions.convert_cells(paths.notebook_path + converter.notebook_name, processing_step,
                                               paths.output_path_converted_files)
        else:
            sys.exit("No output path for the converted files specified")


if __name__ == "__main__":
    convert_preprocessing_steps()
