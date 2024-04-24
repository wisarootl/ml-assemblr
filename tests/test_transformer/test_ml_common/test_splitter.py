import numpy as np
import pytest

from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transfromer.ml_common.splitter import ShuffleSplitter


@pytest.mark.parametrize(
    "test_size, valid_size",
    [
        (test_size, valid_size)
        for test_size in np.arange(0, 1.2, 0.2)
        for valid_size in np.arange(0, 1.2, 0.2)
    ],
)
def test_shuffle_splitter(some_dp: DataPod, test_size: float, valid_size: float):
    prod_dp = some_dp.copy()

    if (test_size + valid_size > 1) or not (0 <= test_size <= 1) or not (0 <= valid_size <= 1):
        with pytest.raises(ValueError):
            ShuffleSplitter(col_name="split", test_size=test_size, valid_size=valid_size, random_seed=0)
        return

    splitter = ShuffleSplitter(
        col_name="split", test_size=test_size, valid_size=valid_size, random_seed=0
    )

    # test fit_transform
    some_dp = some_dp.fit_transform(splitter)
    assert "split" in some_dp.main_df.columns

    test_size_count = int(test_size * some_dp.main_df.shape[0])
    valid_size_count = int(valid_size * some_dp.main_df.shape[0])
    train_size_count = some_dp.main_df.shape[0] - test_size_count - valid_size_count

    value_counts = some_dp.main_df["split"].value_counts()
    assert (train_size_count == 0 and "train" not in value_counts) or (
        abs(value_counts["train"] - train_size_count) <= 1
    )
    assert (valid_size_count == 0 and "valid" not in value_counts) or (
        abs(value_counts["valid"] - valid_size_count) <= 1
    )
    assert (test_size_count == 0 and "test" not in value_counts) or (
        abs(value_counts["test"] - test_size_count) <= 1
    )
    assert some_dp.main_df["split"].isna().sum() == 0

    # test transform
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])

    assert "split" in prod_dp.main_df.columns

    value_counts = prod_dp.main_df["split"].value_counts()
    assert value_counts["production"] == prod_dp.main_df.shape[0]
