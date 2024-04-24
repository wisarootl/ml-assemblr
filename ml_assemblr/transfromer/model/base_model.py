from typing import Any, Optional

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer


def get_model_index(dp: DataPod, order: int = 0) -> Optional[int]:
    for i, transformer in enumerate(dp.footprints.transformers):
        if isinstance(transformer, BaseModel):
            if order == 0:
                return i
            order -= 1
    return None


def get_trained_model(dp: DataPod, model_index: int) -> Any:
    model_transformer = dp.footprints.transformers[model_index]
    return model_transformer.model


class BaseModel(FittingTransformer, DataFrameTransformer):
    pred_col_prefix: str = "pred"
    pred_col_name: Optional[str] = None
    label_idx_in_column_type: int = 0

    # cross validation config
    cv_idx: Optional[int] = None
    cv_idx_map_var_name: Optional[str] = "cv_idx_map"
    cv_pred_idx_in_column_type_var_name: Optional[str] = "cv_pred_idx_in_column_type"
    cv_model_idx_in_footprints_var_name: Optional[str] = "cv_model_idx_in_footprints"
    cv_split_idx_in_column_type_var_name: Optional[str] = "cv_split_idx_in_column_type"
    split_idx_in_column_type: int = 0

    # learnable parameter
    model: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.pred_col_name:
            self.pred_col_prefix = None

    def _append_idx_to_cv_var(self, data_pod: DataPod, var_name: str, idx: int) -> DataPod:
        if var_name not in data_pod.variables[self.cv_idx_map_var_name]:
            data_pod.variables[self.cv_idx_map_var_name][var_name] = []

        data_pod.variables[self.cv_idx_map_var_name][var_name].append(idx)

        return data_pod

    def _preprocess_attrs(self, data_pod: DataPod) -> DataPod:
        label_col_name = data_pod.column_types[self.target_df_name].labels[self.label_idx_in_column_type]
        if not self.pred_col_name:
            self.pred_col_name = f"{self.pred_col_prefix}_{label_col_name}"

        if self.cv_idx is not None:
            self.split_idx_in_column_type = data_pod.variables[self.cv_idx_map_var_name][
                self.cv_split_idx_in_column_type_var_name
            ][self.cv_idx]
            self.pred_col_name = f"{self.pred_col_name}_{self.cv_idx}"

            # append cv_pred_idx_in_column_type
            data_pod = self._append_idx_to_cv_var(
                data_pod,
                self.cv_pred_idx_in_column_type_var_name,
                len(data_pod.column_types[self.target_df_name].predictions),
            )

            # append cv_model_idx_in_footprints
            data_pod = self._append_idx_to_cv_var(
                data_pod,
                self.cv_model_idx_in_footprints_var_name,
                len(data_pod.footprints.transformers),
            )

        self.pred_col_name = data_pod.clean_column_name(self.pred_col_name)

        data_pod.column_types[self.target_df_name].predictions.append(self.pred_col_name)
        return data_pod

    def _call_hook_pre_fit_transform(self, data_pod: DataPod) -> DataPod:
        data_pod = super()._call_hook_pre_fit_transform(data_pod)
        data_pod = self._preprocess_attrs(data_pod)
        return data_pod
