from typing import Any, Optional

import pandas as pd
import xgboost as xgb
from optuna import Study
from xgboost.core import Booster, Metric

from ml_assemblr.main_components.data_pod import DataPod

from .base_model import BaseModel


def get_xgb_config(study: Study) -> dict[str, Any]:
    study_params = {**study.best_trial.params, **study.best_trial.user_attrs}
    xbg_params = {
        key[len("param_") :]: val for key, val in study_params.items() if key.startswith("param_")
    }
    xgb_config = dict(
        xgb_params=xbg_params,
        num_boost_round=study_params["num_rounds"],
    )

    return xgb_config


class XGBModel(BaseModel):
    xgb_params: dict[str, Any]
    num_boost_round: int = 10

    # validation config
    early_stopping_rounds: Optional[int] = None
    is_maximize_metric: Optional[bool] = None
    verbose_eval: Optional[bool | int] = False
    custom_metric: Optional[Metric] = None

    # learnable parameter
    enable_categorical: Optional[bool] = None
    model: Optional[Booster] = None

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df_train_features = data_pod.slice_df(
            split=self.fit_on_split,
            columns="features",
            table_name=self.target_df_name,
            split_idx_in_column_type=self.split_idx_in_column_type,
        )
        df_train_label = data_pod.slice_df(
            split=self.fit_on_split,
            columns="label",
            table_name=self.target_df_name,
            split_idx_in_column_type=self.split_idx_in_column_type,
        )

        df_train_features = self._correct_type_for_categorical_columns(
            df_train_features, data_pod.column_types[self.target_df_name].categorical_features
        )

        dtrain = xgb.DMatrix(
            df_train_features, label=df_train_label, enable_categorical=self.enable_categorical
        )

        if self.val_on_split:
            df_valid_features = data_pod.slice_df(
                split=self.val_on_split,
                columns="features",
                table_name=self.target_df_name,
                split_idx_in_column_type=self.split_idx_in_column_type,
            )
            df_valid_label = data_pod.slice_df(
                split=self.val_on_split,
                columns="label",
                table_name=self.target_df_name,
                split_idx_in_column_type=self.split_idx_in_column_type,
            )
            df_valid_features = self._correct_type_for_categorical_columns(
                df_valid_features, data_pod.column_types[self.target_df_name].categorical_features
            )
            dval = xgb.DMatrix(
                df_valid_features, label=df_valid_label, enable_categorical=self.enable_categorical
            )

            evals = [(dtrain, "train"), (dval, "valid")]
        else:
            evals = None

        evals_result = {}

        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            maximize=self.is_maximize_metric,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
            custom_metric=self.custom_metric,
        )

        if self.val_on_split:
            self.model = self.model[: self.model.best_iteration + 1]

        return self._transform(data_pod)

    def _transform(self, data_pod: DataPod) -> DataPod:
        df_all_features = data_pod.slice_df(
            split=None, columns="features", table_name=self.target_df_name
        )

        df_all_features = self._correct_type_for_categorical_columns(
            df_all_features, data_pod.column_types[self.target_df_name].categorical_features
        )

        dall = xgb.DMatrix(data=df_all_features, enable_categorical=self.enable_categorical)

        data_pod.dfs[self.target_df_name][self.pred_col_name] = self.model.predict(dall)

        return data_pod

    def _correct_type_for_categorical_columns(
        self, df: pd.DataFrame, categorical_col_names: list[str]
    ) -> pd.DataFrame:
        if self.enable_categorical is None:
            if categorical_col_names:
                self.enable_categorical = True

        if self.enable_categorical:
            df[categorical_col_names] = df[categorical_col_names].apply(lambda col: pd.Categorical(col))

        return df
