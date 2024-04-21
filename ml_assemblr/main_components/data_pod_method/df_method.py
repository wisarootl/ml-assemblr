import logging
from typing import TYPE_CHECKING, Literal, Optional

import pandas as pd
from IPython.core.display_functions import display

if TYPE_CHECKING:
    from ml_assemblr.main_components.data_pod import DataPod

logger = logging.getLogger(__name__)


def slice_df(
    self: "DataPod",
    split: Optional[
        Literal["train", "valid", "test", "production"]
        | set[Literal["train", "valid", "test", "production"]]
    ] = None,
    columns: Optional[Literal["features", "label", "prediction"] | list[str]] = None,
    table_name: Optional[str] = None,
    split_idx_in_column_type: int = 0,
) -> pd.DataFrame:
    if not table_name:
        table_name = self.main_df_name
    df = self.dfs[table_name]
    column_type = self.column_types[table_name]

    if isinstance(split, str):
        relevant_row = df[column_type.splitters[split_idx_in_column_type]] == split
    elif isinstance(split, set):
        relevant_row = df[column_type.splitters[split_idx_in_column_type]].isin(split)
    else:
        relevant_row = pd.Series(True, index=df.index)

    is_list_of_str = isinstance(columns, list) and all(isinstance(x, str) for x in columns)

    if columns is None:
        return df.loc[relevant_row]
    elif columns == "features":
        return df.loc[relevant_row, column_type.features]
    elif columns == "label":
        return df.loc[relevant_row, [column_type.labels[0]]]
    elif columns == "prediction":
        return df.loc[relevant_row, [column_type.predictions[0]]]
    elif is_list_of_str:
        return df.loc[relevant_row, columns]
    else:
        raise ValueError("Invalid input for columns")


def delete_dfs(
    self: "DataPod",
    df_names: list[str],
    is_delete_all_except_specified_df_names: bool = False,
):
    df_names_to_delete = (
        df_names
        if not is_delete_all_except_specified_df_names
        else [df_name for df_name in self.dfs.keys() if df_name not in df_names]
    )

    for df_name in df_names_to_delete:
        if df_name in self.df_nodes:
            del self.df_nodes[df_name]
        else:
            logger.warning(f"There is no table_name {df_name} to delete.")


def peek_df(
    self: "DataPod",
    df_name: str,
    n: int = 5,
):
    df = self.dfs[df_name]
    print(f"{df.shape[0]} rows × {df.shape[1]} columns")
    display(df.head(n))


def peek_main_df(
    self: "DataPod",
    n: int = 5,
):
    self.peek_df(df_name=self.main_df_name, n=n)
