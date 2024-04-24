import sys
from unittest.mock import MagicMock

import pandas as pd

from ml_assemblr.utils.df_utils import get_categorical_column_names
from ml_assemblr.utils.notebook_utils import is_in_notebook


def test_get_categorical_column_names(some_main_df: pd.DataFrame):
    assert get_categorical_column_names(some_main_df) == ["Gender", "Department"]


def test_is_in_notebook():
    # Mock IPython module and get_ipython function
    ipython_module_mock = MagicMock()
    sys.modules["IPython"] = ipython_module_mock
    ipython_module_mock.get_ipython = MagicMock()

    # Test when IPKernelApp is in config
    ipython_module_mock.get_ipython().config = {"IPKernelApp": True}
    assert is_in_notebook()

    # Test when IPKernelApp is not in config
    ipython_module_mock.get_ipython().config = {}
    assert not is_in_notebook()

    # Test when IPython module is not present
    del sys.modules["IPython"]
    assert not is_in_notebook()
