import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evaluate.event import post_event


def transform_data(df, target):
    bins = (2, 6.5, 8)
    group_names = ["bad", "good"]
    df[target] = pd.cut(df[target], bins=bins, labels=group_names)
    label_quality = LabelEncoder()
    df[target] = label_quality.fit_transform(df[target])
    post_event("data_changed", df)
    return df
