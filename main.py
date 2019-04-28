import loading
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import imputation
import cleansing
import feat_selection


def main():
    df = loading.load_csv()
    # cleansing the data
    df = cleansing.cleanse(df)
    for col in df:
        if col == "Vote":
            continue
        df[col].astype(float)
    print(df.dtypes)
    plt.imshow(df.corr().as_matrix())
    plt.colorbar()
    plt.show()
    # splitting the data
    train, valid, test = loading.split_data(df)
    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)
    test = pd.DataFrame(test)

    # imputation of the data
    imputation.imputation(train)
    train.to_csv("after_imputation")

    features = df.columns.to_numpy().tolist()
    features_close = feat_selection.remove_similar_features(train, 'Vote', 10)
    print(f'features after close removal: {features_close}')
    features_far = feat_selection.remove_far_features(train, 'Vote', 10)
    print(f'features after far removal: {features_far}')
    features_close_far = feat_selection.remove_similar_features(train[features_close], 'Vote', 10)
    print(f'features after close and then far removal: {features_close_far}')
    features_far_close = feat_selection.remove_far_features(train[features_far], 'Vote', 10)
    print(f'features after far removal: {features_far_close}')
    train.to_csv("hello.csv")


"""""
    print(valid["Overall_happiness_score"].isna().sum())
    imputation.related_features_imputation(14, valid, train)
    print(valid["Overall_happiness_score"].isna().sum())
    valid.to_csv("hello.csv")"""

if __name__ == "__main__":
    main()
