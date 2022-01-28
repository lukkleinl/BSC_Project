import pandas as pd


class CSVLoader:
    """loads a csv file from a given location"""
    filename = "../data/raw/raw_data.csv"
    path_to_csv = "../../data/Wine_data/raw_data.csv"
    seperator = ";"

    def create_raw_data(self):
        df = pd.read_csv(self.path_to_csv, sep=self.seperator)
        df.to_csv(self.filename)
        # return pd.read_csv(self.path_to_csv, sep=self.seperator)
