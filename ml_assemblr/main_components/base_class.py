from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
from pydantic import BaseModel

from .column_type import ColumnType

if TYPE_CHECKING:
    from ml_assemblr.main_components.data_pod import DataPod
    from ml_assemblr.main_components.data_pod_list import DataPodList


class BaseDataPod(ABC):

    @abstractmethod
    def fit_transform(self, transformer: "Transformer") -> "BaseDataPod":
        raise NotImplementedError("Subclasses must implement fit_transform method")

    @abstractmethod
    def transform(self, transformer: "Transformer") -> "BaseDataPod":
        raise NotImplementedError("Subclasses must implement transform method")

    @abstractmethod
    def append_footprint(self, transformer: "Transformer") -> "BaseDataPod":
        raise NotImplementedError("Subclasses must implement append_footprint method")


class Transformer(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _fit_transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        raise NotImplementedError("Subclasses must implement _fit_transform method")

    @abstractmethod
    def _transform(self, data_pod: BaseDataPod) -> BaseDataPod:
        raise NotImplementedError("Subclasses must implement _transform method")

    def fit_transform(
        self, data_pod: Union[BaseDataPod, "DataPod"]
    ) -> Union[BaseDataPod, "DataPod", "DataPodList"]:
        data_pod = self._call_hook_pre_fit_transform(data_pod)
        data_pod = self._fit_transform(data_pod)
        data_pod = self._call_hook(data_pod)

        return data_pod

    def transform(self, data_pod: BaseDataPod) -> Union[BaseDataPod, "DataPod", "DataPodList"]:
        data_pod = self._transform(data_pod)
        data_pod = self._call_hook(data_pod)
        return data_pod

    def _call_hook_pre_fit_transform(
        self, data_pod: BaseDataPod
    ) -> Union[BaseDataPod, "DataPod", "DataPodList"]:
        return data_pod

    def _call_hook(self, data_pod: BaseDataPod) -> Union[BaseDataPod, "DataPod", "DataPodList"]:
        data_pod = data_pod.append_footprint(self)
        return data_pod


class DataFrameNode(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    df: pd.DataFrame
    column_type: ColumnType

    def set_features_by_top_down_approach(
        self, excluded_types: Optional[set[str]] = None, excluded_col_names: Optional[set[str]] = None
    ):
        all_columns = list(self.df.columns)

        if not excluded_types:
            excluded_types = [
                column_type for column_type in vars(self.column_type) if column_type not in {"features"}
            ]

        all_excluded_col_names = set(
            [
                column
                for column_type in excluded_types
                for column in getattr(self.column_type, column_type)
            ]
        )

        if excluded_col_names:
            all_excluded_col_names = all_excluded_col_names.union(excluded_col_names)

        self.column_type.features = [
            column for column in all_columns if column not in all_excluded_col_names
        ]
