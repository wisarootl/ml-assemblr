from unittest.mock import patch

import pandas as pd

from ml_assemblr.eda.general import display_duplication, investigate_basic_structure_df


def test_general_eda(some_main_df: pd.DataFrame):
    with patch("matplotlib.pyplot.show"):
        investigate_basic_structure_df(some_main_df)
        display_duplication(some_main_df, "ID")
