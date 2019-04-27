import pandas as pd
import numpy as np


def median_imputation(feature: str, df: pd.DataFrame, df2: pd.DataFrame = None):
    # complete the missing values of a feature based on the avg of what all the other samples that voted the same
    if df[feature].dtype == float:
        for voting in df["Vote"].unique():
            if df2 is None:
                ss = df.groupby("Vote")[feature].mean()[voting]
            else:
                ss = df2.groupby("Vote")[feature].mean()[voting]
            df.loc[(df["Vote"] == voting) & (df[feature].isnull()), feature] = ss
    else:
        # complete the missing categorical value with the category most samples have
        if df2 is None:
            common_val = df[feature].value_counts().idxmax()
        else:
            common_val = df2[feature].value_counts().idxmax()
        df[feature].fillna(common_val, inplace=True)


def close_nieg(sample_idx: int, feature: int, df: pd.DataFrame, df2: pd.DataFrame = None):
    sample_val = df.iloc[sample_idx, feature]
    min_dist = np.inf
    min_nigh = -1

    if df2 is None:
        col = df.iloc[:, feature]
    else:
        col = df2.iloc[:, feature]
    col = pd.DataFrame(col)

    # count = 0

    if df2 is None:
        for count, (idx, row) in enumerate(col.iterrows()):
            if count != sample_idx and abs(row[0] - sample_val) < min_dist:
                min_dist = abs(row[0] - sample_val)
                min_nigh = count
            # count = count + 1
    else:
        for count, (idx, row) in enumerate(col.iterrows()):
            if abs(row[0] - sample_val) < min_dist:
                min_dist = abs(row[0] - sample_val)
                min_nigh = count
            # count = count + 1

    return min_nigh


def related_features_imputation(feature: int, df: pd.DataFrame, df2: pd.DataFrame = None):
    if feature >= 27:
        return -1
    if df2 is None:
        mat = df.corr().dropna().as_matrix()
    else:
        mat = df2.corr().dropna().as_matrix()
    max_corr = [(i, mat[i][feature]) for i in range(mat.shape[0]) if
                i != feature and df.dtypes[i] == float and abs(mat[i][feature]) > 0.5]
    # max_corr = []
    # for i in range(mat.shape[0]):
    #     if abs(mat[i][feature]) > 0.5 and i != feature:
    #         max_corr.append((i, mat[i][feature]))
    max_corr.sort(reverse=True, key=lambda tup: tup[1])

    # drop categorical values
    # to_remove = []
    # for most_corr_feat, best_corr in max_corr:
    #     if df.dtypes[most_corr_feat] != float:
    #         to_remove.append((most_corr_feat, best_corr))
    #
    # for feat in to_remove:
    #     max_corr.remove(feat)

    # if no correlation with any feature do nothing
    if len(max_corr) == 0:
        return -1

    # count = 0
    for count, (idx, row) in enumerate(df.iterrows()):
        if pd.isna(row[feature]):
            for most_corr_feat, best_corr in max_corr:
                # check the sample that is closest in the correlated feature to the idx sample
                if pd.isna(row[most_corr_feat]):
                    continue
                if df2 is None:
                    close_nieg_index = close_nieg(count, most_corr_feat, df)
                    df.iloc[count, feature] = df.iloc[close_nieg_index, feature]
                else:
                    close_nieg_index = close_nieg(count, most_corr_feat, df, df2)
                    df.iloc[count, feature] = df2.iloc[close_nieg_index, feature]
                break
        # count = count + 1


def imputation(dataset, dataset2=None):
    # do a median_imputation
    for idx, col in enumerate(dataset):
        print(idx, col)
        related_features_imputation(idx, dataset, dataset2)
        median_imputation(col, dataset, dataset2)
