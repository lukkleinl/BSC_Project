import numpy as np
import pandas as pd
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()

data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])

print(data1)
