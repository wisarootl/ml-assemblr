from typing import Callable, Literal, Union

import pandas as pd

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer, UnfittingTransformer


def plus(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
    return series_1 + series_2


def minus(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
    return series_1 - series_2


def multiply(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
    return series_1 * series_2


def divide(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
    return series_1 * series_2


class BicolumnsOperator(UnfittingTransformer, DataFrameTransformer):
    target_col_name: str
    first_col_name: str
    second_col_name: str
    operation: Union[Literal["+", "-", "*", "/"], Callable[[pd.Series, pd.Series], pd.Series]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.operation, str):
            if self.operation == "+":
                self.operation = plus
            elif self.operation == "-":
                self.operation = minus
            elif self.operation == "*":
                self.operation = multiply
            elif self.operation == "/":
                self.operation = divide
            else:
                raise ValueError("Invalid operation. Must be one of '+', '-', '*', '/'")

    def _transform(self, data_pod: DataPod) -> DataPod:  # type: ignore[override]
        df = data_pod.dfs[self.target_df_name]
        df[self.target_col_name] = self.operation(df[self.first_col_name], df[self.second_col_name])

        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod
