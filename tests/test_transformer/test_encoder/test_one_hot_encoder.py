from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.encoder.one_hot_encoder import OneHotEncoder


def test_one_hot_encoder(some_dps_with_splitting: tuple[DataPod, DataPod]):
    some_dp, prod_dp = some_dps_with_splitting

    assert "gender" in some_dp.main_df.columns

    one_hot_encoder = OneHotEncoder(col_names=["gender", "department"])
    some_dp = some_dp.fit_transform(one_hot_encoder)

    assert "gender" not in some_dp.main_df.columns
    assert "gender_male" in some_dp.main_df.columns
    assert some_dp.main_df.loc[0, "gender_male"] == 0
    assert some_dp.main_df.loc[0, "gender_female"] == 1

    assert "gender" in prod_dp.main_df.columns

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])

    assert "gender" not in prod_dp.main_df.columns
    assert "gender_male" in prod_dp.main_df.columns
    assert prod_dp.main_df.loc[0, "gender_male"] == 0
    assert prod_dp.main_df.loc[0, "gender_female"] == 1
