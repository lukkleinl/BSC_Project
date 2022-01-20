import pandas as pd
from evaluate.utility import set_logger
from .event import subscribe


def log_changed_data(df: pd.DataFrame):
    """"""
    #logger = set_logger("./log/dataset.log")
    #log(f"data changed {df}")


def setup_log_event_handlers():
    subscribe("data_changed", log_changed_data)


