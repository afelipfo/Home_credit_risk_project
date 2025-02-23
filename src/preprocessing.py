from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    binary_cols = [col for col in train_df.select_dtypes(include=['object']).columns if train_df[col].nunique() == 2] 
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(working_train_df[binary_cols])
    working_train_df[binary_cols] = ordinal_encoder.transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])

    one_hot_cols = [col for col in train_df.select_dtypes(include=['object']).columns if train_df[col].nunique() > 2] 
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(working_train_df[one_hot_cols])
    train_encoded = one_hot_encoder.transform(working_train_df[one_hot_cols])
    val_encoded = one_hot_encoder.transform(working_val_df[one_hot_cols])
    test_encoded = one_hot_encoder.transform(working_test_df[one_hot_cols])

    train_encoded_df = pd.DataFrame(train_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
    val_encoded_df = pd.DataFrame(val_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
    test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

    working_train_df = working_train_df.drop(columns=one_hot_cols).join(train_encoded_df)
    working_val_df = working_val_df.drop(columns=one_hot_cols).join(val_encoded_df)
    working_test_df = working_test_df.drop(columns=one_hot_cols).join(test_encoded_df)

    working_train_df.reset_index(drop=True, inplace=True)
    train_encoded_df.reset_index(drop=True, inplace=True)
    working_val_df.reset_index(drop=True, inplace=True)
    val_encoded_df.reset_index(drop=True, inplace=True)
    working_test_df.reset_index(drop=True, inplace=True)
    test_encoded_df.reset_index(drop=True, inplace=True)

    imputer = SimpleImputer(strategy='median')
    working_train_df = pd.DataFrame(imputer.fit_transform(working_train_df), columns=working_train_df.columns)
    working_val_df = pd.DataFrame(imputer.transform(working_val_df), columns=working_val_df.columns)
    working_test_df = pd.DataFrame(imputer.transform(working_test_df), columns=working_test_df.columns)

    scaler = MinMaxScaler()
    working_train_df = pd.DataFrame(scaler.fit_transform(working_train_df), columns=working_train_df.columns)
    working_val_df = pd.DataFrame(scaler.transform(working_val_df), columns=working_val_df.columns)
    working_test_df = pd.DataFrame(scaler.transform(working_test_df), columns=working_test_df.columns)

    train = working_train_df.to_numpy()
    val = working_val_df.to_numpy()
    test = working_test_df.to_numpy()

    return train, val, test