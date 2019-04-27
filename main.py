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

    # TODO maybe change Looking_at_poles_results to boolean
    # TODO add all relevant types
    types = {"Vote": 'category', "Most_Important_Issue": 'category', "Looking_at_poles_results": 'category',
             "Married": 'category', "Gender": 'category', "Voting_Time": 'category',
             "Will_vote_only_large_party": 'category', "Age_group": 'category', "Main_transportation": 'category',
             "Financial_agenda_matters": 'category', "Occupation": 'category'}

    for header, type in types.items():
        df[header] = df[header].astype(type)

    train, valid, test = loading.split_data(df)
    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)
    test = pd.DataFrame(test)

    # plt.imshow(train.corr())
    # plt.colorbar()
    # plt.show()

    imputation.imputation(train)

    # plt.imshow(feat_selection.calc_MI_matrix(train))
    # plt.colorbar()
    # plt.show()

    features = df.columns.to_numpy().tolist()
    features_close = feat_selection.remove_similar_features(train, 'Vote', 10)
    print(f'features after close removal: {features_close}')
    features_far = feat_selection.remove_far_features(train, 'Vote', 10)
    print(f'features after far removal: {features_far}')
    features_close_far = feat_selection.remove_similar_features(train[features_close], 'Vote', 10)
    print(f'features after close and then far removal: {features_close_far}')
    features_far_close = feat_selection.remove_far_features(train[features_far], 'Vote', 10)
    print(f'features after far removal: {features_far_close}')

    train = cleansing.change_categorials(train)
    train.to_csv("hello.csv")


"""""
    print(valid["Overall_happiness_score"].isna().sum())
    imputation.related_features_imputation(14, valid, train)
    print(valid["Overall_happiness_score"].isna().sum())
    valid.to_csv("hello.csv")"""

if __name__ == "__main__":
    main()
