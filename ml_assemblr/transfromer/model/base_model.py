from typing import Any

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer


class BaseModel(FittingTransformer, DataFrameTransformer):
    pred_col_prefix: str = "pred"
    pred_col_name: str = None
    label_idx_in_column_type: int = 0

    # learnable parameter
    model: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.pred_col_name:
            self.pred_col_prefix = None

    def _get_pred_col_name(self, data_pod: DataPod) -> DataPod:
        label_col_name = data_pod.column_types[self.target_df_name].labels[self.label_idx_in_column_type]
        if not self.pred_col_name:
            self.pred_col_name = f"{self.pred_col_prefix}_{label_col_name}"
        self.pred_col_name = data_pod.clean_column_name(self.pred_col_name)

        data_pod.column_types[self.target_df_name].predictions.insert(0, self.pred_col_name)
        return data_pod

    def _call_hook_pre_fit_transform(self, data_pod: DataPod) -> DataPod:
        data_pod = super()._call_hook_pre_fit_transform(data_pod)
        data_pod = self._get_pred_col_name(data_pod)
        return data_pod
