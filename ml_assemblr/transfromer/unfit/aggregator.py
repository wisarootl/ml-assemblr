from typing import Callable, Optional, Union

import pandas as pd

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer


class Aggregator(UnfittingTransformer, DataFrameTransformer):
    source_df_name: str
    groupby_col_names: Union[str, list[str]]
    agg_functions: Union[Callable, str, dict, list]
    filter_function: Optional[Callable] = None
    target_col_prefix: str = ""

    # Learnable parameters
    target_col_names: Optional[list[str]] = None

    def _transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        target_df = data_pod.dfs[self.target_df_name]
        source_df = data_pod.dfs[self.source_df_name]
        if self.filter_function:
            source_df = source_df[self.filter_function]

        agg_source_df: pd.DataFrame = source_df.groupby(self.groupby_col_names).agg(self.agg_functions)
        self.target_col_names = [
            data_pod.clean_column_name_func(
                f"{self.target_col_prefix + '_' if self.target_col_prefix else ''}{col[0]}_{col[1]}"
            )
            for col in agg_source_df.columns
        ]

        agg_source_df.columns = self.target_col_names

        target_df = pd.merge(target_df, agg_source_df, on=self.groupby_col_names, how="left")

        data_pod.df_nodes[self.target_df_name].df = target_df
        return data_pod
