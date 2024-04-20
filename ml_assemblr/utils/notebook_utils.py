import sys


def is_in_notebook() -> bool:
    """Check whether the code is running inside Jupyter Notebook."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            return False
    except Exception:
        return False
    return True
