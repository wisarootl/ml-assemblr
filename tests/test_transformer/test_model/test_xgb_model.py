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
