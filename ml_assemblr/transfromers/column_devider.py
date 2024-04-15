from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer


class ColumnDivider(DataFrameTransformer):
    target_col_name: str
    numerator_col_name: str
    denominator_col_name: str

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        df[self.target_col_name] = df[self.numerator_col_name] / df[self.denominator_col_name]

        assert isinstance(self.target_df_name, str)
        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod

    def _transform(self, data_pod: DataPod) -> DataPod:
        return self._fit_transform(data_pod)
