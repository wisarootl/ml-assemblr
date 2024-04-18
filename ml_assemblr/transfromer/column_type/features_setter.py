from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer
from typing import Optional


class TopDownFeaturesSetter(UnfittingTransformer, DataFrameTransformer):
    excluded_types: Optional[list[str]] = None

    def _transform(self, data_pod: DataPod) -> DataPod:
        data_pod.df_nodes[self.target_df_name].set_features_by_top_down_approach(
            excluded_types=self.excluded_types
        )
        return data_pod
