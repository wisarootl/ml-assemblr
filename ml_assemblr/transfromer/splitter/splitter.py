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

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        np.random.seed(self.random_seed)
        data_pod.dfs[self.target_df_name][self.col_name] = self._split(df.shape[0])

        return data_pod

    def _transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        df[self.col_name] = PRODUCTION
        return data_pod

    def _split(self, total_samples):
        test_size_count = int(self.test_size * total_samples)
        valid_size_count = int(self.valid_size * total_samples)
        train_size_count = total_samples - test_size_count - valid_size_count

        splits = np.empty(total_samples, dtype=object)
        splits[:train_size_count] = TRAIN
        splits[train_size_count : train_size_count + valid_size_count] = VALID
        splits[train_size_count + valid_size_count :] = TEST

        np.random.seed(self.random_seed)
        np.random.shuffle(splits)
        return splits
