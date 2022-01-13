import sys

import click

from data_classes.data_classes import Converter, Paths, PreprocessingSteps
from conversion.conversion import convert_cells, get_config_from_yaml


@click.command()
@click.argument("config_file", type=str, default="data/config_files/config_rfc.yaml")
def convert_preprocessing_steps(config_file):
    try:
        conf = get_config_from_yaml(config_file)
        converter = Converter(**conf["Converter"])
        paths = Paths(**conf["Paths"])
    except KeyError:
        sys.exit(f"Parameters don't match with implemented classes {KeyError}")
    except FileNotFoundError:
        sys.exit(f"Specified file not found ")

    for processing_step in converter.preprocessing_tags:
        if paths.output_path_converted_files is not None:
            convert_cells(paths.notebook_path + converter.notebook_name, processing_step,
                          paths.output_path_converted_files)
        else:
            sys.exit("No output path for the converted files specified")


if __name__ == "__main__":
    convert_preprocessing_steps()
