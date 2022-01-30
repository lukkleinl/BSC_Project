import numpy as np
import pandas as pd

from sklearn import datasets


class iris_loader:
    def create_raw_data(self):
        iris = datasets.load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                            columns=iris['feature_names'] + ['target'])
        return data
