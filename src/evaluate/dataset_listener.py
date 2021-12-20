import pandas as pd
import logging

from utility import set_logger
from .event import subscribe

logger = set_logger("./log/dataset.log")


def handle_changed_data(df: pd.DataFrame):
    logger.info(f"data changed {df}")


def setup_log_event_handlers():
    subscribe("data_changed", handle_changed_data)
