from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.transfromer.column_type.column_type_setter import ColumnTypeExtender, ColumnTypeSetter


def test_column_type_setter(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    column_type_map = {"features": ["age", "gender", "salary", "department"]}

    column_type_setter = ColumnTypeSetter(column_type_map=column_type_map)

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(column_type_setter)
    assert some_dp.main_column_type.features == column_type_map["features"]

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert some_dp.main_column_type.features == column_type_map["features"]


def test_column_type_setter_override(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    column_type_map_1 = {"features": ["age", "gender"]}
    column_type_map_2 = {"features": ["salary", "department"]}

    column_type_setter_1 = ColumnTypeSetter(column_type_map=column_type_map_1)
    column_type_setter_2 = ColumnTypeSetter(column_type_map=column_type_map_2)

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(column_type_setter_1)
    assert "age" in some_dp.main_column_type.features
    assert "salary" not in some_dp.main_column_type.features
    some_dp = some_dp.fit_transform(column_type_setter_2)
    assert "age" not in some_dp.main_column_type.features
    assert "salary" in some_dp.main_column_type.features

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-2])
    assert "age" in prod_dp.main_column_type.features
    assert "salary" not in prod_dp.main_column_type.features
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "age" not in prod_dp.main_column_type.features
    assert "salary" in prod_dp.main_column_type.features


def test_column_type_extender(
    some_dp: DataPod,
):
    prod_dp = some_dp.copy()

    column_type_map_1 = {"features": ["age", "gender"]}
    column_type_map_2 = {"features": ["salary", "department"]}

    column_type_extender_1 = ColumnTypeExtender(column_type_map=column_type_map_1)
    column_type_extender_2 = ColumnTypeExtender(column_type_map=column_type_map_2)

    assert some_dp.main_column_type.features == []
    some_dp = some_dp.fit_transform(column_type_extender_1)
    assert "age" in some_dp.main_column_type.features
    assert "salary" not in some_dp.main_column_type.features
    some_dp = some_dp.fit_transform(column_type_extender_2)
    assert "age" in some_dp.main_column_type.features
    assert "salary" in some_dp.main_column_type.features

    assert prod_dp.main_column_type.features == []
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-2])
    assert "age" in prod_dp.main_column_type.features
    assert "salary" not in prod_dp.main_column_type.features
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "age" in prod_dp.main_column_type.features
    assert "salary" in prod_dp.main_column_type.features
