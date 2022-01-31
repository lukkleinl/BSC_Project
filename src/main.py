import click

from algorithms import algorithms_creation
from configuration import configuration
from loader import data_loader
from preprocessing import preprocessing


@click.command()
#@click.argument("config_path", type=str, default="../data/config_files/Wine_quality/config_rfc_no_convert_CSV_Loader.yaml")
@click.argument("config_path", type=str, default="../data/config_files/Wine_quality/config_sgd_convert_functions_loader_CSV_Loader.yaml")
#@click.argument("config_path", type=str, default="../data/config_files/SV_machines_class/SV_machines_class_config.yaml")
#@click.argument("config_path", type=str, default="../data/config_files/Random_Forest_config/random_forest_config.yaml")
def main(config_path):
    configuration.main(config_path)
    data_loader.main("data/configuration/Loader.json")
    preprocessing.main("data/configuration/PreprocessingSteps.json", "data/configuration/TrainTestSplit.json")
    algorithms_creation.main("data/configuration/Model.json",
                             "data/configuration/TrainTestSplit.json")


if __name__ == "__main__":
    main()
