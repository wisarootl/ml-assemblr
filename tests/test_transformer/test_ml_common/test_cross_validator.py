from sklearn.model_selection import ShuffleSplit

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.ml_common.cross_validator import CrossValidator, get_cv_folds


def test_cross_validator(some_dps_with_splitting: tuple[DataPod, DataPod]):
    some_dp, prod_dp = some_dps_with_splitting

    cross_validator = CrossValidator(
        sklearn_cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        cross_validate_on_split={"train", "valid"},
    )

    some_dp = some_dp.fit_transform(cross_validator)

    original_test_idx = some_dp.main_df[some_dp.main_df["split"] == "test"].index

    split_col_names = ["split_0", "split_1", "split_2"]
    for split_col_name in split_col_names:
        assert split_col_name in some_dp.main_df.columns
        value_counts = some_dp.main_df[split_col_name].value_counts()
        assert abs(value_counts["valid"] - 2) <= 1
        assert abs(value_counts["train"] - 5) <= 1
        assert abs(value_counts["test"] - 1) <= 1
        cv_test_idx = some_dp.main_df[some_dp.main_df[split_col_name] == "test"].index
        assert set(original_test_idx) == set(cv_test_idx)

    assert "cv_idx_map" in some_dp.variables
    assert "cv_split_idx_in_column_type" in some_dp.variables["cv_idx_map"]
    assert some_dp.variables["cv_idx_map"]["cv_split_idx_in_column_type"] == [1, 2, 3]

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])


def test_get_cv_folds(some_dps_with_cv):
    some_dp, _ = some_dps_with_cv
    folds = get_cv_folds(some_dp)

    for i, fold in enumerate(folds):
        train_idx, test_idx = fold
        df = some_dp.main_df
        assert set(df[df[f"split_{i}"] == "train"].index) == set(train_idx)
        assert set(df[df[f"split_{i}"] == "valid"].index) == set(test_idx)
