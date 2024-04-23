import pandas as pd
import pytest
from structlog.testing import capture_logs

from ml_assemblr.main_components.column_type import ColumnType
from ml_assemblr.main_components.data_pod import DataPod


def test_df_after_init_dp(some_dp: DataPod):
    # test some_dp.df_nodes[].df
    assert isinstance(some_dp.df_nodes, dict)
    assert "some_main_df" in some_dp.df_nodes
    df_1 = some_dp.df_nodes["some_main_df"].df

    # test some_dp.dfs[]
    assert isinstance(some_dp.dfs, dict)
    assert "some_main_df" in some_dp.dfs
    df_2 = some_dp.df_nodes["some_main_df"].df

    # test some_dp.main
    assert some_dp.main_df_name == "some_main_df"
    df_3 = some_dp.main_df

    def _test_df(df: pd.DataFrame):
        assert isinstance(df, pd.DataFrame)
        assert "age" in df.columns
        assert "Age" not in df.columns
        assert df.loc[0, "age"] == 25

    _test_df(df_1)
    _test_df(df_2)
    _test_df(df_3)
    assert df_1 is df_2 is df_3


def test_column_type_after_init_dp(some_dp: DataPod):
    # test some_dp.df_nodes[].column_type
    assert isinstance(some_dp.df_nodes, dict)
    assert "some_main_df" in some_dp.df_nodes
    column_type_1 = some_dp.df_nodes["some_main_df"].column_type

    # test some_dp.dfs[]
    assert isinstance(some_dp.column_types, dict)
    assert "some_main_df" in some_dp.column_types
    column_type_2 = some_dp.df_nodes["some_main_df"].column_type

    # test some_dp.main
    assert some_dp.main_df_name == "some_main_df"
    column_type_3 = some_dp.main_column_type

    def _test_df(column_type: ColumnType):
        assert isinstance(column_type, ColumnType)
        assert "label" in column_type.labels
        assert "Label" not in column_type.labels
        assert "id" in column_type.keys
        assert "ID" not in column_type.keys
        assert not column_type.features
        assert not column_type.predictions
        assert not column_type.splitters
        assert not column_type.categorical_features

    _test_df(column_type_1)
    _test_df(column_type_2)
    _test_df(column_type_3)
    assert column_type_1 is column_type_2 is column_type_3


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        ("UnClean Column_Name", "unclean_column_name"),
        ("Another_Unclean_Name", "another_unclean_name"),
        ("Yet_Another", "yet_another"),
        ("MixedCaseName", "mixedcasename"),
        ("1234_Name", "1234_name"),
    ],
)
def test_dp_clean_column_name(some_dp: DataPod, input_name, expected_output):
    assert some_dp.clean_column_name(input_name) == expected_output


def test_dp_peek_df(some_dp: DataPod, capfd):
    expected_keywords = ["rows", "×", "columns", "id", "age", "gender", "salary", "department", "label"]

    some_dp.peek_df("some_main_df")
    captured = capfd.readouterr()
    for keyword in expected_keywords:
        assert keyword in captured.out

    some_dp.peek_main_df()
    captured = capfd.readouterr()
    for keyword in expected_keywords:
        assert keyword in captured.out


def test_dp_delete_dfs(some_dp: DataPod):
    assert "some_main_df" in some_dp.df_nodes
    assert "some_support_df" in some_dp.df_nodes
    some_dp.delete_dfs(["some_main_df"])
    assert "some_main_df" not in some_dp.df_nodes
    assert "some_support_df" in some_dp.df_nodes


def test_dp_delete_dfs_exclude(some_dp: DataPod):
    assert "some_main_df" in some_dp.df_nodes
    assert "some_support_df" in some_dp.df_nodes
    some_dp.delete_dfs(["some_main_df"], is_delete_all_except_specified_df_names=True)
    assert "some_main_df" in some_dp.df_nodes
    assert "some_support_df" not in some_dp.df_nodes


def test_dp_delete_dfs_unexisting(some_dp: DataPod):
    with capture_logs() as cap_logs:
        some_dp.delete_dfs(["unexisting_df"])
        assert cap_logs[0]["event"] == "There is no table name `unexisting_df` to delete."


def test_dp_slice_df(some_dp: DataPod):
    df = some_dp.slice_df(split=None, columns="label")
    assert df.shape[1] == 1
    assert df.columns == ["label"]
    assert some_dp.main_column_type.labels[0] == df.columns[0]

    df = some_dp.slice_df(split=None, columns="features")
    assert df.empty

    # todo: test with different split
