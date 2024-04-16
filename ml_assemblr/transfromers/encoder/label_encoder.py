from copy import deepcopy
from typing import Optional

from sklearn.preprocessing import LabelEncoder as SkLearnLabelEncoder

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer


class LabelEncoder(FittingTransformer, DataFrameTransformer):
    col_names: list[str]

    # learnable parameter
    encoder: Optional[SkLearnLabelEncoder] = None
    encoders: dict[str, SkLearnLabelEncoder] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.encoder:
            self.encoder = SkLearnLabelEncoder()

    def _fit_transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        df = data_pod.slice_df(split=self.learn_on_split, columns=None, table_name=self.target_df_name)
        for col_name in self.col_names:
            self.encoders[col_name] = deepcopy(self.encoder)
            self.encoders[col_name].fit(df[col_name])
        return self._transform(data_pod)

    def _transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]

        for col_name in self.col_names:
            df[col_name] = self.encoders[col_name].transform(df[col_name])

        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod
