from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.cleaner.categorical_cleaner import CategoricalCleaner


def test_categorical_cleaner(
    some_dps_with_splitting: tuple[DataPod, DataPod],
):
    some_dp, prod_dp = some_dps_with_splitting
    categorical_cleaner = CategoricalCleaner(
        clean_categorical_columns_map={"gender": None, "department": None}
    )
    some_dp = some_dp.fit_transform(categorical_cleaner)

    prod_row = {
        "id": 8,
        "age": 40,
        "gender": "Unknown",
        "salary": 80000,
        "department": "Customer Services",
        "label": 0.7,
        "split": "production",
    }

    prod_dp.main_df.loc[prod_dp.main_df.index.max() + 1] = prod_row

    assert prod_dp.main_df.loc[8, "gender"] == "Unknown"
    assert prod_dp.main_df.loc[8, "department"] == "Customer Services"

    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])

    assert prod_dp.main_df.loc[8, "gender"] is None
    assert prod_dp.main_df.loc[8, "department"] is None
