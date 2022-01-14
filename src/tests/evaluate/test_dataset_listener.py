import pandas as pd

from evaluate.utility import set_logger
from .test_event import subscribe


def handle_changed_data(df: pd.DataFrame):
    logger = set_logger("./log/dataset.log")
    logger.info(f"data changed {df}")


def setup_log_event_handlers():
    subscribe("data_changed", handle_changed_data)

# def log_plot_data(df: pd.DataFrame):
#     logger.info(plot_data(df))
