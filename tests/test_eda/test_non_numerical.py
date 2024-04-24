from unittest.mock import patch

import pandas as pd

from ml_assemblr.eda.non_numerical import display_cardinality


def test_non_numerical(some_main_df: pd.DataFrame):
    with patch("matplotlib.pyplot.show"):
        display_cardinality(some_main_df)
