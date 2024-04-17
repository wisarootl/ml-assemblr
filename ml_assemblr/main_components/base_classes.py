from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
from pydantic import BaseModel

from .column_type import ColumnType

if TYPE_CHECKING:
    from ml_assemblr.main_components.data_pod import DataPod


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

    def fit_transform(self, data_pod: Union[BaseDataPod, "DataPod"]) -> Union[BaseDataPod, "DataPod"]:
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
    column_type: ColumnType

    def top_down_features_inference(self, excluded_column_type_list: Optional[list[str]] = None):
        all_columns = list(self.df.columns)

        if not excluded_column_type_list:
            excluded_column_type_list = [
                column_type for column_type in vars(self.column_type) if column_type not in {"features"}
            ]

        excluded_columns = set(
            [
                column
                for column_type in excluded_column_type_list
                for column in getattr(self.column_type, column_type)
            ]
        )

        self.column_type.features = [column for column in all_columns if column not in excluded_columns]
