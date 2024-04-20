from ml_assemblr.main_components.data_pod import DataPod

from .base_class import BaseDataPod, Transformer


class DataPodList(BaseDataPod, list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.variables = {}

    def fit_transform(self, transformer: Transformer) -> "DataPodList":
        for i in range(len(self)):
            self[i] = self[i].fit_transform(transformer)
        return self

    def transform(self, transformer: Transformer) -> "DataPodList":
        for i in range(len(self)):
            self[i] = self[i].transform(transformer)
        return self

    def append_footprint(self, transformer: Transformer) -> "DataPodList":
        for i in range(len(self)):
            self[i] = self[i].append_footprint(transformer)
        return self

    def __getitem__(self, index) -> DataPod:
        return super().__getitem__(index)

    def __setitem__(self, index, value: DataPod):
        super().__setitem__(index, value)
