import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df


def apply_function(df, function=np.sum):
    return df.apply(function)


def describe_data(df):
    return df.describe()


def group_and_aggregate(df, groupby_column, agg_func):
    return df.groupby(groupby_column).agg(agg_func)


def merge_dataframes(df1, df2, on_column, how='inner'):
    return pd.merge(df1, df2, on=on_column, how=how)


def fill_missing_values(df):
    df_filled = df.copy()
    for column in df_filled.columns:
        if df_filled[column].isnull().any():
            df_filled[column].fillna(df_filled[column].median(), inplace=True)
    return df_filled


def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())


def set_and_reset_index(df):
    df = df.set_index('index')
    df = df.reset_index()
    return df


def visualize_data(df):
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            plt.plot(df[column], label=column)
    plt.legend()
    plt.show()


def apply_xgboost(df, target_column):
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, y_test


def evaluate_model(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    df = pd.read_csv('sample.csv')
    df2 = pd.read_csv('sample2.csv')

    print("Original Data: \n", df)
    sorted_df = sort_data(df)
    print("\nAfter Sorting:\n", sorted_df)
    df = apply_function(df)
    print("\nAfter Applying Function:\n", df)
    df = describe_data(df)
    print("\nData Description:\n", df)
    group_agg = group_and_aggregate(df, 'group_column', np.mean)
    print("\nAfter Grouping & Aggregating:\n", group_agg)
    merged_df = merge_dataframes(df, df2, 'join_column')
    print("\nAfter Merging:\n", merged_df)
    filled_df = fill_missing_values(df)
    print("\nAfter Filling Missing Values:\n", filled_df)
    normalized_df = normalize_data(df)
    print("\nAfter Normalization:\n", normalized_df)
    indexed_df = set_and_reset_index(df)
    print("\nAfter Setting & Resetting Index:\n", indexed_df)
    visualize_data(df)
    predictions, y_test = apply_xgboost(df, 'target_column')
    print("\nPrediction:\n", predictions)
    performance = evaluate_model(y_test, predictions)
    print("\nModel Performance (RMSE):\n", performance)


if __name__ == "__main__":
    main()