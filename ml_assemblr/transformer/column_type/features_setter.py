from typing import Optional

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer


class TopDownFeaturesSetter(UnfittingTransformer, DataFrameTransformer):
    excluded_types: Optional[set[str]] = None
    excluded_col_names: Optional[set[str]] = None

    def _transform(self, data_pod: DataPod) -> DataPod:
        data_pod.df_nodes[self.target_df_name].set_features_by_top_down_approach(
            excluded_types=self.excluded_types, excluded_col_names=self.excluded_col_names
        )
        return data_pod
