from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.encoder.label_encoder import LabelEncoder


def test_label_encoder(some_dps_with_splitting: tuple[DataPod, DataPod]):
    some_dp, prod_dp = some_dps_with_splitting

    assert "gender" in some_dp.main_df.columns

    one_hot_encoder = LabelEncoder(col_names=["gender", "department"])

    assert some_dp.main_df.loc[0, "department"] == "HR"
    some_dp = some_dp.fit_transform(one_hot_encoder)
    assert some_dp.main_df.loc[0, "department"] == 1

    assert prod_dp.main_df.loc[0, "department"] == "HR"
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert prod_dp.main_df.loc[0, "department"] == 1
