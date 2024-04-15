from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel

from .column_types import ColumnTypes


class BaseDataPod(ABC):

    @abstractmethod
    def fit_transform(self, transformer: "Transformer") -> "BaseDataPod":
        pass

    @abstractmethod
    def transform(self, transformer: "Transformer") -> "BaseDataPod":
        pass

    @abstractmethod
    def append_footprint(self, transformer: "Transformer") -> "BaseDataPod":
        pass


class Transformer(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _fit_transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        pass

    @abstractmethod
    def _transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        pass

    def fit_transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        data_pod = self._fit_transform(data_pod)
        data_pod = self._call_hook(data_pod)

        return data_pod

    def transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        data_pod = self._transform(data_pod)
        data_pod = self._call_hook(data_pod)
        return data_pod

    def _call_hook(self, data_pod: BaseDataPod):
        data_pod = data_pod.append_footprint(self)
        return data_pod


class DataFrameNode(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    df: pd.DataFrame
    column_type: ColumnTypes
