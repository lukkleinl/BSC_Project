import sys
import urllib.request
from abc import ABC, abstractmethod

import pandas as pd

from data_classes.data_classes import Loader


class BaseLoader(ABC):
    """Abstract Class which specifies the core functions of a loader"""
    out_path: str

    def __init__(self, out_path):
        self.out_path = out_path

    @abstractmethod
    def get_data(self):
        """used to load the data
           :return: the data from the saved location"""


class CSVLoader(BaseLoader):
    """loads a csv file from a given location"""

    def __init__(self, out_path):
        super().__init__(self, out_path)

    def get_data(self):
        return pd.read_csv(self.out_path, sep=";")


class URLLoader(BaseLoader):
    """loads and saves data from a given url to the given location"""
    # url for  wine quality
    # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    url: str

    def __init__(self, url, out_path):
        self.url = url
        BaseLoader.__init__(self, out_path)

    def get_data(self):
        urllib.request.urlretrieve(self.url, self.out_path)
        return pd.read_csv(self.out_path, sep=";")


class LoaderFactory(ABC):
    """Factory which represents the different algorithm factories"""

    @abstractmethod
    def get_loader(self, loader: Loader, out_path: str):
        """returns the specified algorithm"""


class CSVLoaderFactory(LoaderFactory):
    """Factory to get a random forest classifier"""

    def get_loader(self, loader: Loader, out_path: str):
        return CSVLoader(out_path + loader.name_of_file)


class URLLoaderFactory(LoaderFactory):
    """Factory to get a Stochastic Gradient Decent Classifier"""

    def get_loader(self, loader: Loader, out_path: str):
        return URLLoader(loader.url, out_path + loader.name_of_file)


def create_loader(loader_name: str):
    """used to identify which factory should be used"""
    """return the selected factory"""

    """names of factories"""
    loaders = {
        "CSVLoader": CSVLoaderFactory(),
        "URLLoader": URLLoaderFactory()
    }

    if loader_name in loaders:
        return loaders[loader_name]
    else:
        sys.exit("loader name was not recognized")
