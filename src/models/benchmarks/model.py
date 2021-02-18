from abc import ABCMeta, abstractmethod
from datetime import datetime as dt
import pandas as pd


class Model(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self.creation_time = dt.today().strftime('%Y-%m-%d_%H:%M')
        self.model = None

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_time

    @abstractmethod
    def forecast(self, df: pd.DataFrame):
        pass
