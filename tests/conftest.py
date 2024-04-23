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
