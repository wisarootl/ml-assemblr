from collections.abc import Hashable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display


def investigate_basic_structure_df(df: pd.DataFrame):
    print(f"{df.shape[0]} rows × {df.shape[1]} columns")
    print("== info ==")
    df.info()
    display(df.head())


def display_duplication(df: pd.DataFrame, primary_keys: Hashable | Sequence[Hashable] | None):
    print("Duplicated rows ===")
    display(df[df.duplicated()])

    print("Duplicated primary keys ===")
    display(df[df.duplicated(subset=primary_keys, keep=False)])


def display_missing_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()
