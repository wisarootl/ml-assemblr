import pandas as pd
import pytest

from ml_assemblr.main_components.data_pod import DataPod


@pytest.fixture()
def some_df():
    data = {
        "Age": [25, 30, 35, 40, 45],
        "Gender": ["Female", "Male", "Male", "Male", "Female"],
        "Salary": [50000, 60000, 70000, 80000, 90000],
        "Department": ["HR", "Finance", "IT", "Marketing", "Sales"],
    }
    return pd.DataFrame(data)


def test_data_pod(some_df: pd.DataFrame):

    assert some_df.loc[0, "Age"] == 25

    dfs = {"some_data": some_df}

    dp = DataPod(dfs=dfs, main_df_name="some_data")

    assert dp.main_df_name == "some_data"
