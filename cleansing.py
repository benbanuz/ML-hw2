import pandas as pd
import numpy as np


def change_categorials(df: pd.DataFrame):
    cleanup_nums = {"Age_group": {"Below_30": 0, "30-45": 1, "45_and_up": 2},
                    "Looking_at_poles_results": {"yes": 1, "no": 0},
                    "Married": {"yes": 1, "no": 0},
                    "Financial_agenda_matters": {"yes": 1, "no": 0},
                    "Gender": {"male": 1, "female": -1},
                    "Voting_Time": {"After_16:00": 1, "By_16:00": -1},
                    "Will_vote_only_large_party": {"yes": 1, "maybe": 0, "no": -1}
                    }
    df.replace(cleanup_nums, inplace=True)
    df = pd.get_dummies(df, columns=["Most_Important_Issue", "Main_transportation", "Occupation"],
                        prefix=["Issue", "trans", "Occ"])
    df.drop(["Most_Important_Issue", "Main_transportation", "Occupation"], axis=1, inplace=True)

    return df


def outlier_dropping(feature: int, df: pd.DataFrame, min: int = 0, max: int = np.inf, is_int: bool = False,
                     finite_range=None):
    to_drop = []
    for count, (idx, row) in enumerate(df.iterrows()):
        val = row[feature]
        if pd.isna(val):
            continue
        if is_int:
            if not np.isclose(val, int(val)):
                to_drop.append(idx)
        if val < min or val > max:
            to_drop.append(idx)
        if finite_range is not None and val not in finite_range:
            to_drop.append(idx)

    df.drop(to_drop, inplace=True)
    return df


def cleanse(df: pd.DataFrame):
    for count, feature in enumerate(df):
        if df[feature].dtype != float:
            continue
        if feature in ["%Of_Household_Income", "%Time_invested_in_work", "%_satisfaction_financial_policy"]:
            outlier_dropping(count, df, 0, 100)
        if feature == "Number_of_differnt_parties_voted_for":
            outlier_dropping(count, df, is_int=True)
        if feature == "Financial_balance_score_(0-1)":
            outlier_dropping(count, df, 0, 1)
        if feature == "Occupation_Satisfaction":
            outlier_dropping(count, df, finite_range=range(1, 11, 1))
        if feature == "Number_of_valued_Kneset_members":
            outlier_dropping(count, df, finite_range=range(0, 121, 1))
        if feature == "Num_of_kids_born_last_10_years":
            outlier_dropping(count, df, finite_range=range(0, 11, 1))
        if feature == "Last_school_grades":
            outlier_dropping(count, df, finite_range=range(0, 101, 10))
    change_categorials(df)
    return df
