import importlib
import json
import sys
import pandas as pd

from sklearn import model_selection

from data_classes import data_classes
from data_classes.data_classes import PreprocessingSteps, TrainTestSplit


def import_and_load_preprocessing_function(preprocessing_step: data_classes.PrepStep, df: pd.DataFrame):
    """
    First loads the module of the converted file,
       after that the specified function is loaded and executed.
       The preprocessing functions should all return a Dataframe
    :param preprocessing_step:
    :param df: Dataframe
    :return: df: Preprocessed Dataframe
    """

    """declaration of parameters"""
    if preprocessing_step.conversion:
        location_of_step = "conversion."
    else:
        location_of_step = "preprocessing.preprocessing_functions."

    name_of_step = preprocessing_step.name_of_step
    name_of_module = preprocessing_step.name_of_module
    params = preprocessing_step.params

    """load module of preprocessing step"""
    module = importlib.import_module(location_of_step + name_of_module)

    """tries to load function with given name and execute it"""
    try:
        func = module.__getattribute__(name_of_step)
        df = func(df, params)
    except AttributeError as err:
        sys.exit(f"specified {err}")

    return df


def separate_features_outcome(df: pd.DataFrame, target: str):
    """
    Separates the features from the target

    :param target:
    :param df: Dataframe which should be separated
    :return: separated Dataframes
    """
    features = df.drop(target, axis=1)
    labels = df[target]
    return features, labels


def preprocess_data(df, preprocessing_steps: PreprocessingSteps, traintestsplit: TrainTestSplit):
    """Preprocess Data prior to separation of features"""

    """path definition on where to output the dataframes for each step"""
    output_path_preprocessing = "data/preprocessed/"
    output_path_processed = "data/processed/"

    preprocessing_steps = preprocessing_steps.get_list_of_preprocessing_steps()

    for step in preprocessing_steps:
        if step.prior_to_split:
            df = import_and_load_preprocessing_function(step, df)
            df.to_csv(output_path_preprocessing + step.name_of_step + ".csv")

    """Separate Feature and Outcome + scale Data"""
    features, labels = separate_features_outcome(df, traintestsplit.target)

    """Preprocess Data after separation of features"""
    for step in preprocessing_steps:
        if not step.prior_to_split:
            features = import_and_load_preprocessing_function(step, features)
            features = pd.DataFrame(features)
            df = features.join(labels)
            df.to_csv(output_path_preprocessing + step.name_of_step + ".csv")

    df.to_csv(output_path_processed + "processed.csv")
    """Parameter Specification and train tes split"""
    test_size = traintestsplit.parameter_train_test_split[0]
    random_state = traintestsplit.parameter_train_test_split[1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=test_size,
                                                                        random_state=random_state)

    """join and export to csv"""
    df_train = x_train.join(y_train)
    df_test = x_test.join(y_test)
    df_train.to_csv("data/processed/train.csv")
    df_test.to_csv("data/processed/test.csv")
    return x_train, x_test, y_train, y_test


def main(preprocessing_steps_path, train_test_split_path):
    """"""
    path_to_raw_data = "data/raw/raw_data.csv"

    df = pd.read_csv(path_to_raw_data)

    with open(preprocessing_steps_path, 'r') as file:
        preprocessing_steps = json.load(file)

    with open(train_test_split_path, 'r') as file:
        train_test_split = json.load(file)

    preprocessing_steps = PreprocessingSteps(**preprocessing_steps)
    train_test_split = TrainTestSplit(**train_test_split)
    preprocess_data(df, preprocessing_steps, train_test_split)
