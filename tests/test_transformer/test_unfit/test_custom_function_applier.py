import pandas as pd

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.unfit.custom_function_applier import CustomFunctionApplier


def test_custom_function_applier(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    def custom_function(df: pd.DataFrame) -> pd.Series:
        return df["age"] * 2

    custom_function_applier = CustomFunctionApplier(target_col_name="output", function=custom_function)

    assert "output" not in some_dp.main_df.columns
    some_dp = some_dp.fit_transform(custom_function_applier)
    assert "output" in some_dp.main_df.columns
    assert some_dp.main_df["output"].equals(some_dp.main_df["age"] * 2)

    assert "output" not in prod_dp.main_df.columns
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "output" in prod_dp.main_df.columns
    assert prod_dp.main_df["output"].equals(prod_dp.main_df["age"] * 2)
