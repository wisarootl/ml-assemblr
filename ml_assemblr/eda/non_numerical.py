import matplotlib.pyplot as plt
import pandas as pd


def display_cardinality(df: pd.DataFrame):

    for column in df.columns:
        value_counts = df[column].value_counts()
        plt.figure(figsize=(5, 3))
        value_counts.plot(kind="bar")
        plt.title(f"Value Counts for {column}")
        plt.xlabel("Values")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.show()

        print("columns:", column)
        print("unique values:", len(df[column].unique()))
