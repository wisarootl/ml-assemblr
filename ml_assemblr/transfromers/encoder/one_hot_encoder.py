from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer


class OneHotEncoder(FittingTransformer, DataFrameTransformer):
    col_names: list[str]

    # learnable parameter
    encoder: Optional[SkLearnOneHotEncoder] = None
    target_col_names: Optional[list[str]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.encoder:
            self.encoder = SkLearnOneHotEncoder()

    def _fit_transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        df = data_pod.slice_df(split=self.learn_on_split, columns=None, table_name=self.target_df_name)
        self.encoder.fit(df[self.col_names])
        self.target_col_names = [
            data_pod.clean_column_name_func(feature_name)
            for feature_name in self.encoder.get_feature_names_out()
        ]
        return self._transform(data_pod)

    def _transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]

        one_hot_encoded = self.encoder.transform(df[self.col_names])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=self.target_col_names)
        df = pd.concat([df, one_hot_encoded_df], axis=1)
        df = df.drop(columns=self.col_names)
        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod
