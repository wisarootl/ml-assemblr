from typing import Literal, Optional

from .base_class import BaseDataPod, Transformer
from .data_pod import DataPod


class DataFrameTransformer(Transformer):
    target_df_name: str = ""

    def _infer_df_name_from_main_df_name(self, data_pod: DataPod):
        if not self.target_df_name:
            self.target_df_name = data_pod.main_df_name

    def _call_hook_pre_fit_transform(self, data_pod: DataPod) -> DataPod:
        data_pod = super()._call_hook_pre_fit_transform(data_pod)
        self._infer_df_name_from_main_df_name(data_pod)
        return data_pod


class FittingTransformer(Transformer):
    fit_on_split: Optional[
        Literal["train", "valid", "test", "production"]
        | set[Literal["train", "valid", "test", "production"]]
    ] = set(["train", "valid"])


class UnfittingTransformer(Transformer):
    def _fit_transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        return self._transform(data_pod)


class Serializer(Transformer):
    transformers: list[Transformer]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flattened_transformers = []
        for transformer in self.transformers:
            if isinstance(transformer, Serializer):
                flattened_transformers.extend(transformer.transformers)
            else:
                flattened_transformers.append(transformer)
        self.transformers = flattened_transformers

    def _fit_transform(self, data_pod: BaseDataPod):
        for transformer in self.transformers:
            data_pod = data_pod.fit_transform(transformer)

        return data_pod

    def _transform(self, data_pod: BaseDataPod):
        for transformer in self.transformers:
            data_pod = transformer.transform(data_pod)
        return data_pod

    def _call_hook(self, data_pod: BaseDataPod):
        return data_pod
