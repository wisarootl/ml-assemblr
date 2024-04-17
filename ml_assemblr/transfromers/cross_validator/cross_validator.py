from copy import deepcopy
from typing import Optional, Union, Literal

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.data_pod_list import DataPodList
from ml_assemblr.main_components.transformer import DataFrameTransformer
from ml_assemblr.main_components.constant import TRAIN, VALID
from sklearn.model_selection import BaseCrossValidator


class CrossValidator(DataFrameTransformer):
    splitter_col_name: Union[str, None] = None
    cross_validate_on_split: Optional[
        Literal["train", "valid", "test", "production"]
        | set[Literal["train", "valid", "test", "production"]]
    ] = set(["train", "valid"])
    sklearn_cv: BaseCrossValidator

    def _fit_transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        if not self.splitter_col_name:
            self.splitter_col_name = data_pod.column_types[self.target_df_name].splitters[0]

        dps = DataPodList(self.generate_folds(data_pod))
        return dps

    def _transform(self, data_pod: DataPod) -> DataPod:
        return data_pod

    def generate_folds(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        relevant_df = df[df[self.splitter_col_name].isin(self.cross_validate_on_split)]
        label_col_name = data_pod.column_types[self.target_df_name].labels[0]

        x = relevant_df.index
        y = relevant_df[label_col_name]

        for train_idx, valid_idx in self.sklearn_cv.split(x, y):
            child_data_pod = data_pod.copy()

            train_idx = relevant_df.index[train_idx]
            valid_idx = relevant_df.index[valid_idx]
            child_data_pod.main_df.loc[train_idx, self.splitter_col_name] = TRAIN
            child_data_pod.main_df.loc[valid_idx, self.splitter_col_name] = VALID
            yield child_data_pod
