from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transfromer.model.xgb_model import XGBModel


def test_xgb_model(some_dps_with_features_setter: tuple[DataPod, DataPod]):
    some_dp, prod_dp = some_dps_with_features_setter

    xgb_model = XGBModel(xgb_params={})
    some_dp = some_dp.fit_transform(xgb_model)

    assert "pred_label" in some_dp.main_df.columns
    assert some_dp.main_df["pred_label"].isna().sum() == 0

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "pred_label" in prod_dp.main_df.columns
    assert prod_dp.main_df["pred_label"].isna().sum() == 0

    assert some_dp.main_df["pred_label"].equals(prod_dp.main_df["pred_label"])


def test_xgb_model_witv_cv(some_dps_with_cv: tuple[DataPod, DataPod]):
    some_dp, prod_dp = some_dps_with_cv
    cv_count = len(some_dp.variables["cv_idx_map"]["cv_split_idx_in_column_type"])
    for i in range(cv_count):
        xgb_model = XGBModel(xgb_params={}, fit_on_split="train", cv_idx=i)
        some_dp = some_dp.fit_transform(xgb_model)
        prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])

        assert f"pred_label_{i}" in some_dp.main_df.columns
        assert f"pred_label_{i}" in prod_dp.main_df.columns
        assert some_dp.main_df[f"pred_label_{i}"].equals(prod_dp.main_df[f"pred_label_{i}"])
