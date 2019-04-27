import pandas as pd


def change_categorials(df: pd.DataFrame):
    for col in df:
        if df[col].dtype != float:
            df[col] = df[col].cat.rename_categories(range(df[col].nunique())).astype(int)
    return df
