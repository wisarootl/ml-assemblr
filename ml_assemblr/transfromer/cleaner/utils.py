import pandas as pd
from typing import Any
import numpy as np
from collections import Counter
import structlog


def log_na_diff_between_two_pandas_series(
    series_1: pd.Series,
    series_2: pd.Series,
    log_message: str = "There are different NaNs between 2 pandas series",
    context: dict[Any, Any] = {},
):

    series_1_na_count = series_1.isna().sum()
    series_2_na_count = series_2.isna().sum()

    if series_2_na_count > series_1_na_count:
        diff_values = series_1[np.logical_and(series_2.isna(), ~series_1.isna())]
        diff_values = dict(Counter([str(value) for value in list(diff_values)]))
        context = {
            **context,
            **{
                "column": series_1.name,
                "series_1_na_count": series_1_na_count,
                "series_2_na_count": series_2_na_count,
                "additional_na_count": series_2_na_count - series_1_na_count,
            },
        }

        structlog.get_logger().warning(log_message, **context)
        debug_context = {**context, **{"value_counts_that_transformed_to_nans": diff_values}}
        structlog.get_logger().warning(log_message, **debug_context)
