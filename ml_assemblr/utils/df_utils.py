import pandas as pd


def get_categorical_column_names(df: pd.DataFrame, exclude: set[str] = set()) -> list[str]:
    return [col for col in df.columns if df[col].dtype == "object" and col not in exclude]
