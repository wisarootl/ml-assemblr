from typing import Literal, Optional, Union

from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

from ml_assemblr.main_components.constant import TRAIN, VALID
from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer


class CrossValidator(DataFrameTransformer):
    split_col_name: Union[str, None] = None
    target_split_col_name_prefix: Union[str, None] = None
    cross_validate_on_split: Optional[
        Literal["train", "valid", "test", "production"]
        | set[Literal["train", "valid", "test", "production"]]
    ] = set(["train", "valid"])
    sklearn_cv: Union[BaseCrossValidator, BaseShuffleSplit]
    cv_idx_map_var_name: str = "cv_idx_map"
    cv_split_idx_in_column_type_var_name: str = "cv_split_idx_in_column_type"

    def _call_hook_pre_fit_transform(self, data_pod: DataPod) -> DataPod:
        data_pod = super()._call_hook_pre_fit_transform(data_pod)
        if not self.split_col_name:
            self.split_col_name = data_pod.column_types[self.target_df_name].splitters[0]

        if not self.target_split_col_name_prefix:
            self.target_split_col_name_prefix = data_pod.column_types[self.target_df_name].splitters[0]
        return data_pod

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        relevant_df = df[df[self.split_col_name].isin(self.cross_validate_on_split)]
        label_col_name = data_pod.column_types[self.target_df_name].labels[0]

        x = relevant_df.index
        y = relevant_df[label_col_name]

        cv_split_idx_in_column_type = []

        for i, (train_idx, valid_idx) in enumerate(self.sklearn_cv.split(x, y)):
            train_idx = relevant_df.index[train_idx]
            valid_idx = relevant_df.index[valid_idx]

            cv_split_col_name = f"{self.target_split_col_name_prefix}_{i}"
            data_pod.column_types[self.target_df_name].splitters.append(cv_split_col_name)
            cv_split_idx_in_column_type.append(
                len(data_pod.column_types[self.target_df_name].splitters) - 1
            )

            data_pod.main_df.loc[:, cv_split_col_name] = data_pod.main_df.loc[:, self.split_col_name]
            data_pod.main_df.loc[train_idx, cv_split_col_name] = TRAIN
            data_pod.main_df.loc[valid_idx, cv_split_col_name] = VALID

        cv_idx_map = {self.cv_split_idx_in_column_type_var_name: cv_split_idx_in_column_type}

        data_pod.variables[self.cv_idx_map_var_name] = cv_idx_map
        return data_pod

    def _transform(self, data_pod: DataPod) -> DataPod:
        return data_pod


def get_cv_folds(
    dp: DataPod,
    cv_idx_map_var_name: str = "cv_idx_map",
    cv_split_idx_in_column_type_var_name: str = "cv_split_idx_in_column_type",
):
    folds = []
    for idx in dp.variables[cv_idx_map_var_name][cv_split_idx_in_column_type_var_name]:
        split_col_name = dp.main_column_type.splitters[idx]
        split_col = dp.main_df[split_col_name]
        split_col = split_col[split_col.isin({"train", "valid"})].reset_index(drop=True)
        fold = tuple(split_col.index[split_col == "train"]), tuple(split_col.index[split_col == "valid"])
        folds.append(fold)
    return folds
