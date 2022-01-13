import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


def binary_classification_and_label_encoding(df: pd.DataFrame, params: dict):
    """
    Groups the target colum into binaray classifiers good and bad

    :param df: Dataframe whih should be transformed
    :param params: target Specifies the target column
    :return: transformed Dataset
    """
    # bins = params["bins"]
    # group_names = params["group_names"]
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    target = params["target"]
    df[target] = pd.cut(df[target], bins=bins, labels=group_names)
    label_quality = LabelEncoder()
    df[target] = label_quality.fit_transform(df[target])
    return df


def scale_data(df: pd.DataFrame, params: dict):
    """
    Scales data with the given Scalar

    :param df: Dataframe which should be scaled
    :param params: name of scalar
    :return: scaled Dataframe
    """

    scaler_names = {
        "StandardScaler": StandardScaler(),
        "RobustScalar": RobustScaler(),
        "MinMaxScalar": MinMaxScaler()
    }

    if params["scaler_name"] in scaler_names:
        scaler = scaler_names[params["scaler_name"]]
        df = scaler.fit_transform(df)
    else:
        print("Scaler doesn't exist")

    return df


def separate_features_outcome(df: pd.DataFrame, params: dict):
    """
    Separates the features from the target

    :param df: Dataframe which should be separated
    :param params: target: specifies the target column
    :return: separated Dataframes
    """
    return df.drop(params["target"], axis=1).to_numpy(), df[params["target"]]
