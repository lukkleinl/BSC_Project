import pandas as pd
from .event import subscribe


def log_changed_data(df: pd.DataFrame):
    """"""
    #logger = set_logger("./log/dataset.log")
    #log(f"data changed {df}")


def setup_data_classes_event_handlers():
    subscribe("Data classes created", log_changed_data)

