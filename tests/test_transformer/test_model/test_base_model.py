from xgboost.core import Booster

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.model.base_model import get_model_index, get_trained_model
from ml_assemblr.transformer.model.xgb_model import XGBModel


def test_base_model_utils(some_dps_with_cv: tuple[DataPod, DataPod]):
    some_dp, _ = some_dps_with_cv
    cv_count = len(some_dp.variables["cv_idx_map"]["cv_split_idx_in_column_type"])
    for i in range(cv_count):
        xgb_model = XGBModel(xgb_params={}, fit_on_split="train", cv_idx=i)
        some_dp = some_dp.fit_transform(xgb_model)

        model_idx = get_model_index(some_dp, i)
        trained_model = get_trained_model(some_dp, model_idx)
        assert isinstance(trained_model, Booster)
