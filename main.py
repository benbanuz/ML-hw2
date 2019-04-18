import loading
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import imputation


def main():
    df = loading.load_csv()

    # TODO maybe change Looking_at_poles_results to boolean
    # TODO add all relevant types
    types = {"Vote": 'category', "Most_Important_Issue": 'category', "Looking_at_poles_results": 'category',
             "Married": 'category'
        , "Gender": 'category', "Voting_Time": 'category', "Will_vote_only_large_party": 'category',
             "Age_group": 'category', "Main_transportation": 'category', "Financial_agenda_matters": 'category',
             "Occupation": 'category'}

    for header, type in types.items():
        df[header] = df[header].astype(type)

    train, valid, test = loading.split_data(df)
    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)
    test = pd.DataFrame(test)

    print(train["Overall_happiness_score"].isna().sum())
    imputation.related_features_imputation(14, train)
    print(train["Overall_happiness_score"].isna().sum())
    train.to_csv("hello.csv")


if __name__ == "__main__":
    main()
