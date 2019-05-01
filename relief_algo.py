import pandas as pd
import numpy as np
from typing import List


def closest_row(mat: np.ndarray, row: np.ndarray) -> int:
    dists: np.ndarray = np.sum(np.power((mat - row), 2), axis=1)
    return np.argmin(dists)


def relief(df: pd.DataFrame, target: str, max_remove: int, num_iter=10) -> List[str]:
    features = list(set(df.columns.to_numpy().tolist()).difference({target}))
    weights = np.zeros(len(features))

    for row_idx in np.random.randint(0, len(df.index), num_iter):
        target_val = df[target].iloc[row_idx]
        hit_indices = df[target] == target_val

        val = df[features].iloc[row_idx].values

        misses = df[~hit_indices][features].values

        hit_indices[row_idx] = False
        hits = df[hit_indices][features].values

        near_hit = df[features].iloc[closest_row(hits, val)]
        near_miss = df[features].iloc[closest_row(misses, val)]

        weights += np.power((val - near_miss), 2) - np.power((val - near_hit), 2)

    feat_weights = {feat: weights[i] for i, feat in enumerate(features)}
    sorted_feats = dict(sorted(feat_weights.items(), key=lambda kv: (kv[1], kv[0])))

    return list(sorted_feats.keys())[max_remove:]
