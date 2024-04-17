from typing import Callable

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer


class CustomFunctionApplier(UnfittingTransformer, DataFrameTransformer):
    target_col_name: str
    function: Callable

    def _transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        df = data_pod.dfs[self.target_df_name]
        df[self.target_col_name] = self.function(df)
        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod
