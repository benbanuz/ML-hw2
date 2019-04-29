import loading
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import imputation
import cleansing
import feat_selection
from feat_selection import *
from sklearn.tree import DecisionTreeClassifier


def main():
    df = loading.load_csv()
    # # cleansing the data
    # df = cleansing.cleanse(df)
    # for col in df:
    #     if col == "Vote":
    #         df[col].astype('category')
    #     else:
    #         df[col].astype(float)
    #
    # # # splitting the data
    # train, valid, test = loading.split_data(df)
    # train = pd.DataFrame(train)
    # valid = pd.DataFrame(valid)
    # test = pd.DataFrame(test)

    # # imputation of the data
    # imputation.imputation(train)

    #
    # train.to_csv("train_after_imputation.csv", index=False)
    #
    # imputation.imputation(valid, train)
    # valid.to_csv("valid_after_imputation.csv", index=False)

    train = pd.read_csv('train_after_imputation.csv')
    valid = pd.read_csv('valid_after_imputation.csv')

    plt.imshow(train.corr().abs())
    plt.colorbar()
    plt.title("correlation after imputation")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train))
    plt.colorbar()
    plt.title("MI after imputation")
    plt.show()
    plt.close('all')

    features = df.columns.to_numpy().tolist()
    features_close = feat_selection.remove_similar_features(train, 'Vote', 8)
    print(f'features after close removal: {features_close}')
    plt.imshow(train[features_close].corr().abs())
    plt.colorbar()
    plt.title("correlation after close removal")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[features_close]))
    plt.colorbar()
    plt.title("MI after close removal")
    plt.show()
    plt.close('all')

    features_far = feat_selection.remove_far_features(train, 'Vote', 8)
    print(f'features after far removal: {features_far}')
    plt.imshow(train[features_far].corr().abs())
    plt.colorbar()
    plt.title("correlation after far removal")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[features_far]))
    plt.colorbar()
    plt.title("MI after far removal")
    plt.show()
    plt.close('all')

    features_close_far = feat_selection.remove_similar_features(train[features_close], 'Vote', 8)
    print(f'features after close and then far removal: {features_close_far}')
    plt.imshow(train[features_close_far].corr().abs())
    plt.colorbar()
    plt.title("correlation after close and then far removal")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[features_close_far]))
    plt.colorbar()
    plt.title("MI after close and then far removal")
    plt.show()
    plt.close('all')

    features_far_close = feat_selection.remove_far_features(train[features_far], 'Vote', 8)
    print(f'features after far and then close removal: {features_far_close}')
    plt.imshow(train[features_far_close].corr().abs())
    plt.colorbar()
    plt.title("correlation after far and then close removal")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[features_far_close]))
    plt.colorbar()
    plt.title("MI after far and then close removal")
    plt.show()
    plt.close('all')

    sfs_features = wrapper_SFS(train[features_close_far], valid[features_close_far], 'Vote', DecisionTreeClassifier())
    sfs_features += ['Vote']
    print(f'features after SFS: {features_far_close}')
    plt.imshow(train[sfs_features].corr().abs())
    plt.colorbar()
    plt.title("correlation after SFS")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[sfs_features]))
    plt.colorbar()
    plt.title("MI after SFS")
    plt.show()
    plt.close('all')

    sbs_features = wrapper_SBS(train[features_close_far], valid[features_close_far], 'Vote', DecisionTreeClassifier())
    sbs_features += ['Vote']
    print(f'features after SBS: {features_far_close}')
    plt.imshow(train[sbs_features].corr().abs())
    plt.colorbar()
    plt.title("correlation after SBS")
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(train[sbs_features]))
    plt.colorbar()
    plt.title("MI after SBS")
    plt.show()
    plt.close('all')

    train.to_csv("hello.csv")


"""""
    print(valid["Overall_happiness_score"].isna().sum())
    imputation.related_features_imputation(14, valid, train)
    print(valid["Overall_happiness_score"].isna().sum())
    valid.to_csv("hello.csv")"""

if __name__ == "__main__":
    main()
