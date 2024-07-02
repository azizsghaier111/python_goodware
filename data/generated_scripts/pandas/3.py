import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df


def apply_function(df, function=np.sum):
    """Apply a function to each column of data"""
    return df.apply(function)


def describe_data(df):
    """Get the description of data"""
    return df.describe()


def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())


def set_and_reset_index(df):
    df = df.set_index('column_to_set_index_on')
    df = df.reset_index()
    return df


def visualize_data(df):
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.show()


def apply_xgboost(df, target_column):
    model = XGBRegressor()
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions


def main():
    df = pd.read_csv('sample.csv')
    
    print("Original Data: \n", df)

    sorted_df = sort_data(df)
    print("\nAfter Sorting:\n", sorted_df)

    df = apply_function(df)
    print("\nAfter Applying Function:\n", df)

    df = describe_data(df)
    print("\nData Description:\n", df)

    normalized_df = normalize_data(df)
    print("\nAfter Normalization:\n", normalized_df)

    indexed_df = set_and_reset_index(df)
    print("\nAfter Setting & Resetting Index:\n", indexed_df)

    visualize_data(df)

    predictions = apply_xgboost(df, 'target_column')
    print("\nPrediction:\n", predictions)


if __name__ == "__main__":
    main()