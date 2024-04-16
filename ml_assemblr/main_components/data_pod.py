import copy
from typing import Callable, Literal, Optional, Union

import pandas as pd

from ml_assemblr.utils.string_case_utils import to_screaming_snake_case

from .base_classes import BaseDataPod, DataFrameNode, Transformer
from .column_type import ColumnType


class DataPod(BaseDataPod):

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame] = {},
        column_types: dict[str, ColumnType] = {},
        main_df_name: str = "",
        clean_column_name_func: Callable = to_screaming_snake_case,
    ):
        self.main_df_name = main_df_name
        self.df_nodes: dict[str, DataFrameNode] = {}

        if dfs:
            for df_name, df in dfs.items():
                self.df_nodes[df_name] = DataFrameNode(
                    name=df_name, df=df, column_type=column_types.get(df_name, ColumnType())
                )
        else:
            self.df_nodes = {}

        from .transformer import Serializer  # import outside top level to prevent a circular import

        self.footprints: Serializer = Serializer(transformers=[])
        self.clean_column_name_func = clean_column_name_func
        self.variables = {}

    @property
    def dfs(self):
        return {df_name: df_node.df for df_name, df_node in self.df_nodes.items()}

    @property
    def column_types(self):
        return {df_name: df_node.column_type for df_name, df_node in self.df_nodes.items()}

    @property
    def main_df(self):
        return self.df_nodes[self.main_df_name].df

    @property
    def main_column_type(self):
        return self.df_nodes[self.main_df_name].column_type

    def fit_transform(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        # preventing using the same transformer with other data_pod
        transformer = copy.deepcopy(transformer)

        return transformer.fit_transform(self)

    def transform(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        return transformer.transform(self)

    def append_footprint(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        self.footprints.transformers.append(transformer)
        return self

    def slice_df(
        self,
        split: Optional[
            Literal["train", "valid", "test", "production"]
            | set[Literal["train", "valid", "test", "production"]]
        ] = None,
        columns: Optional[Literal["features", "label", "prediction"] | list[str]] = None,
        table_name: Optional[str] = None,
    ) -> pd.DataFrame:
        if not table_name:
            table_name = self.main_df_name
        df = self.dfs[table_name]
        column_type = self.column_types[table_name]

        if isinstance(split, str):
            relevant_row = df[column_type.splitters[0]] == split
        elif isinstance(split, set):
            relevant_row = df[column_type.splitters[0]].isin(split)
        else:
            relevant_row = pd.Series(True, index=df.index)

        is_list_of_str = isinstance(columns, list) and all(isinstance(x, str) for x in columns)

        if columns is None:
            return df.loc[relevant_row]
        elif columns == "features":
            return df.loc[relevant_row, column_type.features]
        elif columns == "label":
            return df.loc[relevant_row, [column_type.labels[0]]]
        elif columns == "prediction":
            return df.loc[relevant_row, [column_type.predictions[0]]]
        elif is_list_of_str:
            return df.loc[relevant_row, columns]
        else:
            raise ValueError("Invalid input for columns")
