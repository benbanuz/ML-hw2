import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from typing import List


def calc_MI_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    calculate the mutual info between each 2 features
    :param df: the data frame
    :return: a matrice m of size (n_features, n_features) where m[i,j] = mutual_information(feature_i, feature_j)
    """
    # get the features
    features = df.feature_names

    # start with zeroes everywhere but the diagonal, as mutual_information(feature_i, feature_i) = 1 when normalized
    mi_matrix = np.eye(len(features))

    # calculate the values, use the fact that the matrice is symmetric to skip a few iterations
    for i in range(len(features)):
        for j in range(i):
            feat_i, feat_j = features[i], features[j]
            mi = normalized_mutual_info_score(df[feat_i], df[feat_j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def remove_similar_features(df: pd.DataFrame, max_remove: int, close_thresh: int = 3, mi_thres: float = 0.05) -> List[str]:
    """
    removes features that share a lot of info
    :param df: dataframe, should **NOT** include the target
    :param max_remove: maximum number of features to remove
    :param close_thresh: how many features should be close to a feature in order for us to remove it
    :param mi_thres: mi threshold for considering features to be close
    :return: list of features after the removal (i.e the ones that weren't removed)
    """
    # get features list
    features: List[str] = df.feature_names
    # empty dictionaries for close features and the number of them
    close_features = {feature: [] for feature in features}
    num_close = {feature: 0 for feature in features}

    # calculate mutual information matrix
    mi_matrix = calc_MI_matrix(df)

    # check the mi matrix for close features, use the symmetry to remove redundant iterations
    for i in range(len(features)):
        for j in range(i):
            feat_i, feat_j = features[i], features[j]
            # if the mi value is below the threshold
            if mi_matrix[i, j] < mi_thres:
                # mark the features as close
                close_features[feat_i].append(feat_j)
                close_features[feat_j].append(feat_i)
                # and update the number accordingly
                num_close[feat_i] += 1
                num_close[feat_j] += 1

    for _ in range(max_remove):
        # if all features have less close ones then the threshold, stop removing features
        if max(num_close.values()) < close_thresh:
            break

        # get the feature with the most close features
        worst_feature = max(num_close, key=num_close.get)

        # update the close features according to the removal
        for close_feat in close_features[worst_feature]:
            close_features[close_feat].remove(worst_feature)
            num_close[close_feat] -= 1
        # remove the feature
        close_features.pop(worst_feature)

    return list(close_features.keys())


def remove_far_features(df: pd.DataFrame, target: str, max_remove: int, mi_thres: float = 0.8) -> List[str]:
    """
    remove features that don't have enough mutual information with the target
    :param df: the dataframe
    :param target: the target feature
    :param max_remove: maximum number of features to remove
    :param mi_thres: threshold for mutual information, all values above it will be considered too far away
    :return: list of features after the removal (i.e the ones that weren't removed)
    """
    # all features except for the target
    features: List[str] = df.feature_names
    features.remove(target)
    mi_vals = {feat: normalized_mutual_info_score(df[feat], df[target]) for feat in features}

    # sort by mi in relation to target
    mi_vals = sorted(mi_vals.items(), key=lambda kv: (kv[1], kv[0]))

    for i in range(max_remove):
        # if no features are far away, stop removing features
        if mi_vals[i][1] < mi_thres:
            break
        # remove the furthest feature
        features.remove(mi_vals[i][0])

    return features
