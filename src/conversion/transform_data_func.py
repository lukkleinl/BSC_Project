#Example refactoring of Data Transformation
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def transform_data_func(wine: pd.DataFrame, params: dict):
    # Making binary classificaion for the response variable.
    # Dividing wine as good and bad by giving the limit for the quality
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    print(wine)
    wine[params["target"]] = pd.cut(wine[params["target"]], bins=bins, labels=group_names)
    # Now lets assign a labels to our quality variable
    label_quality = LabelEncoder()
    # Bad becomes 0 and good becomes 1
    wine[params["target"]] = label_quality.fit_transform(wine[params["target"]])
    return wine
