from typing import TYPE_CHECKING

import pandas as pd

from ml_assemblr.main_components.column_type import ColumnType

if TYPE_CHECKING:
    from ml_assemblr.main_components.data_pod import DataPod


def dfs(self: "DataPod") -> dict[str, pd.DataFrame]:
    return {df_name: df_node.df for df_name, df_node in self.df_nodes.items()}


def column_types(self: "DataPod") -> dict[str, ColumnType]:
    return {df_name: df_node.column_type for df_name, df_node in self.df_nodes.items()}


def main_df(self: "DataPod") -> pd.DataFrame:
    return self.df_nodes[self.main_df_name].df


def main_column_type(self: "DataPod") -> ColumnType:
    return self.df_nodes[self.main_df_name].column_type
