from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transformer.column_type.column_type_setter import ColumnTypeSetter
from ml_assemblr.transformer.column_type.features_setter import TopDownFeaturesSetter


def test_top_down_features_setter(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    top_down_features_setter = TopDownFeaturesSetter()

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(top_down_features_setter)
    assert some_dp.main_column_type.features == ["age", "gender", "salary", "department"]

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert prod_dp.main_column_type.features == ["age", "gender", "salary", "department"]


def test_top_down_features_setter_excluded_types(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    categorical_features_setter = ColumnTypeSetter(
        column_type_map={"categorical_features": ["gender", "salary"]}
    )
    top_down_features_setter = TopDownFeaturesSetter(excluded_types={"keys", "labels"})

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(categorical_features_setter)
    assert some_dp.main_column_type.categorical_features == ["gender", "salary"]
    some_dp = some_dp.fit_transform(top_down_features_setter)
    # categorical_features was not exclude in top dow process
    assert some_dp.main_column_type.features == ["age", "gender", "salary", "department"]

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-2])
    assert prod_dp.main_column_type.categorical_features == ["gender", "salary"]
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    # categorical_features was not exclude in top dow process
    assert prod_dp.main_column_type.features == ["age", "gender", "salary", "department"]


def test_top_down_features_setter_excluded_col_names(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    top_down_features_setter = TopDownFeaturesSetter(excluded_col_names=["gender"])

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(top_down_features_setter)
    assert "gender" not in some_dp.main_column_type.features
    assert some_dp.main_column_type.features == ["age", "salary", "department"]

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "gender" not in prod_dp.main_column_type.features
    assert prod_dp.main_column_type.features == ["age", "salary", "department"]
