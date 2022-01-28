import luigi

from algorithms import algorithms_creation
from configuration import configuration
from loader import data_loader
from preprocessing import preprocessing


class Configuration(luigi.Task):
    configuration_path = luigi.Parameter()

    def requires(self):
        return []

    def run(self):
        configuration.get_config(self.configuration_path)


class Loader(luigi.Task):
    loader_path = luigi.Parameter()

    def requires(self):
        return [Configuration()]

    def run(self):
        data_loader.main(self.loader_path)


class Preprocessor(luigi.Task):
    preprocessing_path = luigi.Parameter()

    def requires(self):
        return [Loader()]

    def run(self):
        preprocessing.main(self.preprocessing_path)


class Algorithm(luigi.Task):
    model_path = luigi.Parameter()
    paths_path = luigi.Parameter()
    train_test_split_path = luigi.Parameter()

    def requires(self):
        return [Preprocessor()]

    def run(self):
        algorithms_creation.main(self.model_path, self.train_test_split_path, self.paths_path)


if __name__ == '__main__':
    luigi.run()