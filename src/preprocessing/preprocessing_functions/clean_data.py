import pandas as pd


def drop_columns(df: pd.DataFrame, params: dict):
    """
    drops specified columns
    :param df: Dataframe to drop from
    :param params: column names for columns to drop
    :return: new Dataframe
    """
    for name in params["column_names"]:
        df = df.drop(columns=name, axis=1)
    return df


def remove_na(df: pd.DataFrame, params=None):
    """
    Removes missing values

    :param df: Dataframe
    :param params:
    :return: new Dataframe
    """
    df = df.dropna(axis=0)
    return df
