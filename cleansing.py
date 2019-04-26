import pandas as pd


def cleanse_data(df: pd.DataFrame):
    for col in df:
        if df[col].dtype != float:
            df[col] = df[col].cat.rename_categories(range(df[col].nunique())).astype(int)
    return df
