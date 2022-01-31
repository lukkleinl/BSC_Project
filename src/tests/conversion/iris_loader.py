import numpy as np
import pandas as pd
from sklearn import datasets

from loader.data_loader import BaseLoader
from sklearn.datasets import fetch_openml

class iris_loader(BaseLoader):
    def create_raw_data(self):
        iris = datasets.load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                            columns=iris['feature_names'] + ['target'])
        data.to_csv(self.filename)



class elec_loader(BaseLoader):
    def create_raw_data(self):
        elec_data = fetch_openml(name='electricity', version=1)
        df = elec_data.frame
        df.to_csv(self.filename)
