import importlib
import json
import sys
import urllib.request
from abc import ABC, abstractmethod

import click
import pandas as pd

from data_classes.data_classes import Loader


class BaseLoader(ABC):
    """Abstract Class which specifies the core functions of a loader"""

    filename = "data/raw/raw_data.csv"

    def __init__(self, seperator: str):
        self.seperator = seperator

    @abstractmethod
    def create_raw_data(self):
        """used to load and create the raw data"""


class CSVLoader(BaseLoader):
    """loads a csv file from a given location"""

    def __init__(self, path_to_csv: str, seperator: str = ";"):
        """

        :param path_to_csv:
        :param seperator:
        """
        self.path_to_csv = path_to_csv
        super(CSVLoader, self).__init__(seperator)

    def create_raw_data(self):
        df = pd.read_csv(self.path_to_csv, sep=self.seperator)
        df.to_csv(self.filename)
        # return pd.read_csv(self.path_to_csv, sep=self.seperator)


class URLLoader(BaseLoader):
    """loads and saves data from a given url to the given location"""

    url: str

    def __init__(self, url, seperator: str = ";"):
        """

        :param url:
        :param seperator:
        """
        self.url = url
        super(URLLoader, self).__init__(seperator)

    def create_raw_data(self):
        """

        :return:
        """
        try:
            urllib.request.urlretrieve(self.url, self.filename)
        except urllib.error.HTTPError as err:
            sys.exit(f" {err} \ncan not load csv from URL\nchange to another loader")


class LoaderFactory(ABC):
    """Factory which represents the different algorithm factories"""

    @abstractmethod
    def get_loader(self, loader: Loader):
        """returns the specified algorithm"""


class CSVLoaderFactory(LoaderFactory):
    """Factory to get a random forest classifier"""

    def get_loader(self, loader: Loader):
        """

        :param loader:
        :return:
        """
        return CSVLoader(loader.path_to_csv)


class URLLoaderFactory(LoaderFactory):
    """Factory to get a Stochastic Gradient Decent Classifier"""

    def get_loader(self, loader: Loader):
        """

        :param loader:
        :return:
        """
        return URLLoader(loader.url)


class ConvertedLoaderFactory(LoaderFactory):
    """Factory to get a Stochastic Gradient Decent Classifier"""

    def get_loader(self, loader: Loader):
        """

        :param loader:
        :return:
        """
        """load module of preprocessing step"""
        module = importlib.import_module("conversion.loader")

        """tries to load function with given name and execute it"""
        try:
            loader_class = module.__getattribute__("CSVLoader")
            loader_class = loader_class()
        except AttributeError as err:
            sys.exit(f"specified {err}")

        return loader_class


def create_loader(loader_name: str):
    """used to identify which factory should be used

    :param loader_name:
    :return:
    """

    loaders = {
        "CSVLoader": CSVLoaderFactory(),
        "URLLoader": URLLoaderFactory(),
        "ConvertedLoader": ConvertedLoaderFactory()
    }

    if loader_name in loaders:
        return loaders[loader_name]
    else:
        sys.exit("loader name was not recognized")


def load_data(loader: Loader):
    """

    :param loader:
    :return:
    """

    loader_factory = create_loader(loader.name)
    load = loader_factory.get_loader(loader)
    try:
        load.create_raw_data()
    except AttributeError as err:
        sys.exit(err)


def main(loader_path):
    """

    :param loader_path:
    :return:
    """
    with open(loader_path, 'r') as file:
        loader = json.load(file)
    loader = Loader(**loader)

    load_data(loader)

# if __name__ == "__main__":
#     main()
