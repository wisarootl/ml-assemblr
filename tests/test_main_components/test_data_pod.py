import pandas as pd
import pytest

from ml_assemblr.main_components.column_type import ColumnType
from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.utils.string_case_utils import to_snake_case


@pytest.fixture()
def some_main_df() -> pd.DataFrame:
    data = {
        "ID": [0, 1, 2, 3, 4, 5, 6, 7],
        "Age": [25, 30, 35, 40, 45, 28, 32, 37],
        "Gender": ["Female", "Male", "Male", "Male", "Female", "Male", "Female", "Male"],
        "Salary": [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000],
        "Department": ["HR", "Finance", "IT", "Marketing", "Sales", "HR", "IT", "Sales"],
        "Label": [0.2, 0.5, 0.1, 0.8, 0.3, 0.6, 0.4, 0.9],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def some_support_df() -> pd.DataFrame:
    data = {
        "Employee_ID": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "Project_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Project_Name": [
            "Project A",
            "Project B",
            "Project C",
            "Project D",
            "Project A",
            "Project B",
            "Project C",
            "Project D",
            "Project A",
            "Project B",
        ],
        "Hours_Worked": [40, 30, 35, 45, 50, 20, 40, 35, 45, 40],
        "Client": [
            "Client X",
            "Client Y",
            "Client Z",
            "Client X",
            "Client Y",
            "Client Z",
            "Client X",
            "Client Y",
            "Client Z",
            "Client X",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def some_dp(some_main_df: pd.DataFrame, some_support_df: pd.DataFrame) -> DataPod:

    dfs = {"some_main_df": some_main_df, "some_support_df": some_support_df}
    column_types = {"some_main_df": ColumnType(labels=["Label"], keys=["ID"])}

    dp = DataPod(
        dfs=dfs, column_types=column_types, main_df_name="some_main_df", clean_column_name=to_snake_case
    )

    return dp


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


def test_dp_clean_column_name(some_dp: DataPod):
    assert some_dp.clean_column_name("UnClean Column_Name") == "unclean_column_name"


def test_dp_peek_df(some_dp: DataPod, capfd):
    expected = (
        "8 rows × 6 columns\n"
        "   id  age  gender  salary department  label\n"
        "0   0   25  Female   50000         HR    0.2\n"
        "1   1   30    Male   60000    Finance    0.5\n"
        "2   2   35    Male   70000         IT    0.1\n"
        "3   3   40    Male   80000  Marketing    0.8\n"
        "4   4   45  Female   90000      Sales    0.3\n"
    )

    some_dp.peek_df("some_main_df")
    captured = capfd.readouterr()
    assert captured.out == expected

    some_dp.peek_main_df()
    captured = capfd.readouterr()
    assert captured.out == expected


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


def test_dp_slice_df(some_dp: DataPod):
    df = some_dp.slice_df(split=None, columns="label")
    assert df.shape[1] == 1
    assert df.columns == ["label"]
    assert some_dp.main_column_type.labels[0] == df.columns[0]

    df = some_dp.slice_df(split=None, columns="features")
    assert df.empty

    df = some_dp.slice_df(split=None, columns="prediction")
    assert df.empty

    # todo: test with different split
