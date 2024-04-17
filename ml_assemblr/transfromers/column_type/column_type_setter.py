from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer


class ColumnTypeSetter(UnfittingTransformer, DataFrameTransformer):
    column_type_map: dict[str, list[str]]

    def _transform(self, data_pod: DataPod) -> DataPod:
        for attr, columns in self.column_type_map.items():
            data_pod.df_nodes[self.target_df_name].column_type.__setattr__(attr, columns)
        return data_pod


class ColumnTypeExtender(ColumnTypeSetter):

    def _transform(self, data_pod: DataPod) -> DataPod:
        for attr, columns in self.column_type_map.items():
            original_columns = data_pod.df_nodes[self.target_df_name].column_type.__getattribute__(attr)
            extended_columns = original_columns + columns
            data_pod.df_nodes[self.target_df_name].column_type.__setattr__(attr, extended_columns)
        return data_pod
