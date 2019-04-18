import pandas as ps
from sklearn.model_selection import train_test_split


def load_csv():
    data = ps.read_csv("ElectionsData.csv")
    return data


def split_data(data: ps.DataFrame, valid_ratio: float = 0.15, test_ratio: float = 0.1):
    train_ratio = 1 - (valid_ratio + test_ratio)
    assert train_ratio >= 0.5

    train, test = train_test_split(data, test_size=test_ratio)
    valid_ratio = valid_ratio / (1 - test_ratio)
    train, valid = train_test_split(train, test_size=valid_ratio)

    return train, valid, test
