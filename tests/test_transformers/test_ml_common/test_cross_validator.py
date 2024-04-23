from sklearn.model_selection import ShuffleSplit

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transfromer.ml_common.cross_validator import CrossValidator


def test_cross_validator(some_dp_with_splitting: DataPod):
    some_dp = some_dp_with_splitting
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
