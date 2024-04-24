from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.data_pod_list import DataPodList
from ml_assemblr.transformer.unfit.bicolumns_operator import BicolumnsOperator


def test_data_pod_list(some_dp: DataPod):
    some_dps = DataPodList([some_dp.copy(), some_dp.copy()])

    assert isinstance(some_dps, DataPodList)
    assert isinstance(some_dps[0], DataPod)
    assert isinstance(some_dps[1], DataPod)

    bicolumns_operator = BicolumnsOperator(
        target_col_name="output", first_col_name="age", second_col_name="salary", operation="+"
    )

    some_dps = some_dps.fit_transform(bicolumns_operator)

    assert "output" in some_dps[0].main_df.columns
    assert "output" in some_dps[1].main_df.columns

    prod_dps = DataPodList([some_dp.copy(), some_dp.copy()])

    prod_dps = prod_dps.transform(some_dps[0].footprints.transformers[-1])

    assert "output" in prod_dps[0].main_df.columns
    assert "output" in prod_dps[1].main_df.columns
