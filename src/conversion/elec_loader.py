from loader.data_loader import BaseLoader
from sklearn.datasets import fetch_openml

class elec_loader(BaseLoader):
    def create_raw_data(self):
        elec_data = fetch_openml(name='electricity', version=1)
        df = elec_data.frame
        df.to_csv(self.filename)
