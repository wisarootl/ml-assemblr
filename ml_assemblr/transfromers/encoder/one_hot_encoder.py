import re
from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer


class OneHotEncoder(DataFrameTransformer):
    categorical_col_names: list[str]
    is_nan_as_category: bool = False

    # learnable parameter
    encoder: Optional[SkLearnOneHotEncoder] = None
    one_hot_encoded_feature_names: Optional[list[str]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SkLearnOneHotEncoder()

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        return self._encode(data_pod, fit=True)

    def _transform(self, data_pod: DataPod) -> DataPod:
        return self._encode(data_pod, fit=False)

    def _encode(self, data_pod: DataPod, fit: bool = False) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        assert isinstance(self.encoder, SkLearnOneHotEncoder)

        if fit:
            one_hot_encoded = self.encoder.fit_transform(df[self.categorical_col_names])
            self.one_hot_encoded_feature_names = [
                re.sub(r"[^a-zA-Z0-9]+", "_", feature_name).upper()
                for feature_name in self.encoder.get_feature_names_out()
            ]
        else:
            one_hot_encoded = self.encoder.transform(df[self.categorical_col_names])

        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded.toarray(), columns=self.one_hot_encoded_feature_names
        )

        df = pd.concat([df, one_hot_encoded_df], axis=1)
        df = df.drop(columns=self.categorical_col_names)

        assert isinstance(self.target_df_name, str)
        data_pod.df_nodes[self.target_df_name].df = df

        return data_pod
