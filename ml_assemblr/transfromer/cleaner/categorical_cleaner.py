from typing import Optional

import numpy as np
import pandas as pd

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, FittingTransformer

from .utils import log_na_diff_between_two_pandas_series


def get_categorical_column_names(df: pd.DataFrame, exclude: set[str] = set()) -> list[str]:
    return [column for column in df.columns if df[column].dtype == "object" and column not in exclude]


class CategoricalCleaner(FittingTransformer, DataFrameTransformer):
    clean_categorical_columns_from_column_type: bool = True
    clean_categorical_columns_from_data_inference: bool = False
    exclude_columns_from_inference: set[str] = set()
    is_update_column_type: bool = True

    # learnable parameter
    clean_categorical_columns_map: dict[str, Optional[set]] = {}
    # category_map = {
    #   "column_1": {"cat_1", "cat_2", "cat_3"},
    #   "column_2": None, # if None, it will fit from data
    #   "column_3": None
    # }

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.slice_df(split=self.fit_on_split, columns=None, table_name=self.target_df_name)

        if self.clean_categorical_columns_from_column_type:
            self.clean_categorical_columns_map = {
                **self.clean_categorical_columns_map,
                **{
                    key: None
                    for key in list(data_pod.column_types[self.target_df_name].categorical_features)
                    if key not in self.clean_categorical_columns_map
                },
            }

        if self.clean_categorical_columns_from_data_inference:
            categorical_columns_from_data_inference = get_categorical_column_names(
                df, self.exclude_columns_from_inference
            )
            self.clean_categorical_columns_map = {
                **self.clean_categorical_columns_map,
                **{
                    key: None
                    for key in categorical_columns_from_data_inference
                    if key not in self.clean_categorical_columns_map
                },
            }

        # fit category from data
        for column in self.clean_categorical_columns_map.keys():
            if self.clean_categorical_columns_map[column] is not None:
                continue

            self.clean_categorical_columns_map[column] = set(df[column].dropna().unique())

        return self._transform(data_pod)

    def _transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        for column in self.clean_categorical_columns_map.keys():
            original_column = df[column].copy()
            data_pod.dfs[self.target_df_name][column] = np.where(
                original_column.isin(self.clean_categorical_columns_map[column]),
                original_column,
                None,
            )
            new_column = df[column]

            log_na_diff_between_two_pandas_series(
                original_column,
                new_column,
                "There are additional NaN after transformation",
                {"transformer": self.__class__.__name__, "df_name": self.target_df_name},
            )

        if self.is_update_column_type:
            data_pod.column_types[self.target_df_name].categorical_features = list(
                self.clean_categorical_columns_map.keys()
            )

        return data_pod
