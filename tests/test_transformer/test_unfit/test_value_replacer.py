from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.unfit.value_replacer import ValueReplacer


def test_custom_function_applier(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    replacing_mappers = {"gender": {"Female": "F", "Male": "M"}}

    value_replacer = ValueReplacer(replacing_mappers=replacing_mappers)

    assert some_dp.main_df.loc[0, "gender"] == "Female"
    some_dp = some_dp.fit_transform(value_replacer)
    assert some_dp.main_df.loc[0, "gender"] == "F"

    assert prod_dp.main_df.loc[0, "gender"] == "Female"
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert prod_dp.main_df.loc[0, "gender"] == "F"
