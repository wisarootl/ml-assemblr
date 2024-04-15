import copy
from typing import Union

import pandas as pd

from .base_classes import BaseDataPod, DataFrameNode, Transformer
from .column_types import ColumnTypes


class DataPod(BaseDataPod):

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame] | None = None,
        column_types: dict[str, ColumnTypes] = {},
        main_df_name: str | None = None,
    ):
        self.main_df_name = main_df_name
        self.df_nodes: dict[str, DataFrameNode] = {}

        if not dfs:
            assert isinstance(dfs, dict)
            for df_name, df in dfs.items():
                self.df_nodes[df_name] = DataFrameNode(
                    name=df_name, df=df, column_type=column_types.get(df_name, ColumnTypes())
                )
        else:
            self.df_nodes = {}

        from .transformer import Serializer  # import outside top level to prevent a circular import

        self.footprints: Serializer = Serializer(transformers=[])

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
