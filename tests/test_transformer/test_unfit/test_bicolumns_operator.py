from typing import Callable, Literal, Union

import pandas as pd
import pytest

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transfromer.unfit.bicolumns_operator import BicolumnsOperator, plus


@pytest.mark.parametrize(
    "operation, expected_output",
    [
        ("+", 50025),
        ("-", -49975),
        ("*", 1250000),
        ("/", 0.0005),
        (plus, 50025),
    ],
)
def test_bicolumns_operator(
    some_dp: DataPod,
    operation: Union[Literal["+", "-", "*", "/"], Callable[[pd.Series, pd.Series], pd.Series]],
    expected_output: int | float,
):
    prod_dp = some_dp.copy()

    bicolumns_operator = BicolumnsOperator(
        target_col_name="output", first_col_name="age", second_col_name="salary", operation=operation
    )

    some_dp = some_dp.fit_transform(bicolumns_operator)
    assert "output" in some_dp.main_df.columns
    assert some_dp.main_df.loc[0, "output"] == expected_output

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "output" in prod_dp.main_df.columns
    assert prod_dp.main_df.loc[0, "output"] == expected_output
