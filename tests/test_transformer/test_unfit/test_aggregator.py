from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.unfit.aggregator import Aggregator


def test_aggregator(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    aggregator = Aggregator(
        source_df_name="some_support_df",
        groupby_col_names="id",
        agg_functions={"hours_worked": ["sum"]},
        target_col_prefix="total_project",
    )

    some_dp = some_dp.fit_transform(aggregator)
    assert "total_project_hours_worked_sum" in some_dp.main_df.columns
    assert some_dp.main_df.loc[0, "total_project_hours_worked_sum"] == 70

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "total_project_hours_worked_sum" in prod_dp.main_df.columns
    assert prod_dp.main_df.loc[0, "total_project_hours_worked_sum"] == 70
