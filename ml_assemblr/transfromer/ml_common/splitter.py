from typing import Optional

import numpy as np

from ml_assemblr.main_components.constant import PRODUCTION, TEST, TRAIN, VALID
from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer


class ShuffleSplitter(FittingTransformer, DataFrameTransformer):
    col_name: str
    test_size: float
    valid_size: float
    random_seed: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.test_size + self.valid_size > 1:
            raise ValueError("test_size + valid_size must be <= 1")

        if not (0 <= self.test_size <= 1):
            raise ValueError("test_size must be with in [0, 1] range")

        if not (0 <= self.valid_size <= 1):
            raise ValueError("valid_size must be with in [0, 1] range")

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        # add split column
        df = data_pod.dfs[self.target_df_name]
        np.random.seed(self.random_seed)
        data_pod.dfs[self.target_df_name][self.col_name] = self._split(df.shape[0])

        # update column type
        data_pod.column_types[self.target_df_name].splitters.append(self.col_name)
        return data_pod

    def _transform(self, data_pod: DataPod) -> DataPod:
        # add split column
        df = data_pod.dfs[self.target_df_name]
        df[self.col_name] = PRODUCTION

        # update column type
        data_pod.column_types[self.target_df_name].splitters.append(self.col_name)
        return data_pod

    def _split(self, total_samples):
        test_size_count = int(self.test_size * total_samples)
        valid_size_count = int(self.valid_size * total_samples)
        train_size_count = total_samples - test_size_count - valid_size_count

        splits = np.empty(total_samples, dtype=object)
        splits[:train_size_count] = TRAIN
        splits[train_size_count : train_size_count + valid_size_count] = VALID
        splits[train_size_count + valid_size_count :] = TEST

        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(splits)
        return splits
