import copy
from typing import Any, Callable, Union

import pandas as pd

from ml_assemblr.utils.string_case_utils import to_screaming_snake_case

from .base_classes import BaseDataPod, DataFrameNode, Transformer
from .column_type import ColumnType
from .data_pod_methods.df_methods import delete_dfs, peek_df, peek_main_df, slice_df
from .data_pod_methods.getters_and_setters import column_types, dfs, main_column_type, main_df


class DataPod(BaseDataPod):

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame] = {},
        column_types: dict[str, ColumnType] = {},
        main_df_name: str = "",
        clean_column_name_func: Callable = to_screaming_snake_case,
    ):
        self.main_df_name = main_df_name
        self.df_nodes: dict[str, DataFrameNode] = {}

        if dfs:
            for df_name, df in dfs.items():
                self.df_nodes[df_name] = DataFrameNode(
                    name=df_name, df=df, column_type=column_types.get(df_name, ColumnType())
                )
        else:
            self.df_nodes = {}

        from .transformer import Serializer  # import outside top level to prevent a circular import

        self.footprints: Serializer = Serializer(transformers=[])
        self.clean_column_name_func = clean_column_name_func
        self.variables: dict[str, Any] = {}

    # getters and setters
    dfs = property(dfs)
    column_types = property(column_types)
    main_df = property(main_df)
    main_column_type = property(main_column_type)

    def fit_transform(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        # preventing using the same transformer with other data_pod
        transformer = copy.deepcopy(transformer)

        return transformer.fit_transform(self)

    def transform(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        return transformer.transform(self)

    def append_footprint(self, transformer: Transformer) -> Union["BaseDataPod", "DataPod"]:
        self.footprints.transformers.append(transformer)
        return self

    # df methods
    slice_df = slice_df
    delete_dfs = delete_dfs
    peek_df = peek_df
    peek_main_df = peek_main_df
