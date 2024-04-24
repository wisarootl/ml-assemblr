from typing import Any, Optional

import pandas as pd
import xgboost as xgb
from optuna import Study
from xgboost.core import Booster

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

    # learnable parameter
    enable_categorical: Optional[bool] = None
    model: Optional[Booster] = None

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df_train = data_pod.slice_df(
            split=self.fit_on_split,
            columns=None,
            table_name=self.target_df_name,
            split_idx_in_column_type=self.split_idx_in_column_type,
        )

        label_col_name = data_pod.column_types[self.target_df_name].labels[self.label_idx_in_column_type]
        df_train_features = df_train[data_pod.column_types[self.target_df_name].features]
        df_train_label = df_train[label_col_name]

        categorical_features = data_pod.column_types[self.target_df_name].categorical_features
        if self.enable_categorical is None:
            if categorical_features:
                self.enable_categorical = True
                df_train_features = self._correct_type_for_categorical_columns(
                    df_train_features, categorical_features
                )
            else:
                self.enable_categorical = False

        dtrain = xgb.DMatrix(
            df_train_features, label=df_train_label, enable_categorical=self.enable_categorical
        )

        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            verbose_eval=False,
        )

        return self._transform(data_pod)

    def _transform(self, data_pod: DataPod) -> DataPod:
        df_all_features = data_pod.slice_df(
            split=None, columns="features", table_name=self.target_df_name
        )

        if self.enable_categorical:
            df_all_features = self._correct_type_for_categorical_columns(
                df_all_features, data_pod.column_types[self.target_df_name].categorical_features
            )

        dall = xgb.DMatrix(data=df_all_features, enable_categorical=self.enable_categorical)

        data_pod.dfs[self.target_df_name][self.pred_col_name] = self.model.predict(dall)

        return data_pod

    def _correct_type_for_categorical_columns(
        self, df: pd.DataFrame, categorical_col_names: list[str]
    ) -> pd.DataFrame:
        df[categorical_col_names] = df[categorical_col_names].apply(lambda col: pd.Categorical(col))
        return df
