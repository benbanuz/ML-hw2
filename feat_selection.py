import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from typing import List, Tuple
from sklearn.base import TransformerMixin


def calc_MI_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    calculate the mutual info between each 2 features
    :param df: the data frame
    :return: a matrice m of size (n_features, n_features) where m[i,j] = mutual_information(feature_i, feature_j)
    """
    # get the features
    features = list(df)

    # start with zeroes everywhere but the diagonal, as mutual_information(feature_i, feature_i) = 1 when normalized
    mi_matrix = np.eye(len(features))

    # calculate the values, use the fact that the matrice is symmetric to skip a few iterations
    for i in range(len(features)):
        for j in range(i):
            feat_i, feat_j = features[i], features[j]
            mi = normalized_mutual_info_score(df[feat_i].values, df[feat_j].values)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def remove_similar_features(df: pd.DataFrame, target: str, max_remove: int, close_thresh: int = 2,
                            mi_thres: float = 0.95) -> List[str]:
    """
    removes features that share a lot of info
    :param df: the dataframe
    :param target: the target feature
    :param max_remove: maximum number of features to remove
    :param close_thresh: how many features should be close to a feature in order for us to remove it
    :param mi_thres: mi threshold for considering features to be close
    :return: list of features after the removal (i.e the ones that weren't removed)
    """
    # get features list without the target
    features: List[str] = df.columns.to_numpy().tolist()
    features.remove(target)
    # empty dictionaries for close features and the number of them
    close_features = {feature: [] for feature in features}
    num_close = {feature: 0 for feature in features}

    # calculate mutual information matrix
    mi_matrix = calc_MI_matrix(df[features])

    # check the mi matrix for close features, use the symmetry to remove redundant iterations
    for i in range(len(features)):
        for j in range(i):
            feat_i, feat_j = features[i], features[j]
            # if the mi value is below the threshold
            if mi_matrix[i, j] > mi_thres:
                # mark the features as close
                close_features[feat_i].append(feat_j)
                close_features[feat_j].append(feat_i)
                # and update the number accordingly
                num_close[feat_i] += mi_matrix[i, j]
                num_close[feat_j] += mi_matrix[i, j]

    for _ in range(max_remove):
        # if all features have less close ones then the threshold, stop removing features
        if max(num_close.values()) < close_thresh:
            break

        # get the feature with the most close features
        worst_feature = max(num_close, key=num_close.get)

        # update the close features according to the removal
        for close_feat in close_features[worst_feature]:
            close_features[close_feat].remove(worst_feature)
            num_close[close_feat] -= mi_matrix[features.index(worst_feature), features.index(close_feat)]
        # remove the feature
        close_features.pop(worst_feature)
        num_close.pop(worst_feature)

    return [target] + list(close_features.keys())


def remove_far_features(df: pd.DataFrame, target: str, max_remove: int, mi_thres: float = 0.25) -> List[str]:
    """
    remove features that don't have enough mutual information with the target
    :param df: the dataframe
    :param target: the target feature
    :param max_remove: maximum number of features to remove
    :param mi_thres: threshold for mutual information, all values above it will be considered too far away
    :return: list of features after the removal (i.e the ones that weren't removed)
    """
    # all features except for the target
    features: List[str] = df.columns.to_numpy().tolist()
    features.remove(target)
    mi_vals = {feat: normalized_mutual_info_score(df[feat].values, df[target].values) for feat in features}

    # sort by mi in relation to target
    sorted_mi_vals: List[Tuple[str, float]] = sorted(mi_vals.items(), key=lambda kv: (kv[1], kv[0]))

    for i in range(max_remove - 1, -1, -1):
        # remove all features below the threshold (only need to find the first because we already sorted it)
        if sorted_mi_vals[i][1] < mi_thres:
            sorted_mi_vals = sorted_mi_vals[i + 1:]
            break

    return [target] + list(tuple(zip(*sorted_mi_vals))[0])


def wrapper_SFS(df_train: pd.DataFrame, df_valid: pd.DataFrame, target: str, clf, cur_score: float = -np.inf,
                features: List[str] = [], max_features=17) -> List[str]:
    if len(features) == max_features:
        return features

    posssible_features: List[str] = list(set(df_train.columns.to_numpy().tolist()).difference(set(features + [target])))

    results = {}

    for feat in posssible_features:
        clf.fit(df_train[features + [feat]].values, df_train[target].values)
        pred: np.ndarray = clf.predict(df_valid[features + [feat]].values)
        results[feat] = np.sum(pred == df_valid[target].values) / pred.size

    best_feature = max(results, key=results.get)
    best_acc = results[best_feature]

    if best_acc > cur_score:
        return wrapper_SFS(df_train, df_valid, target, clf, best_acc, features + [best_feature])
    else:
        return features


def wrapper_SBS(df_train: pd.DataFrame, df_valid: pd.DataFrame, target: str, clf, cur_score: float = -np.inf,
                features: List[str] = None, max_features=17) -> List[str]:
    if features is None:
        features = df_train.columns.to_numpy().tolist()
        features.remove(target)

    results = {}

    for feat in features:
        used_features = list(set(features).difference({feat}))
        clf.fit(df_train[used_features].values, df_train[target].values)
        pred: np.ndarray = clf.predict(df_valid[used_features].values)
        results[feat] = np.sum(pred == df_valid[target].values) / pred.size

    best_feature = max(results, key=results.get)
    best_acc = results[best_feature]

    if best_acc >= cur_score or len(features) > max_features:
        used_features = list(set(features).difference({best_feature}))
        return wrapper_SBS(df_train, df_valid, target, clf, best_acc, used_features)
    else:
        return features
