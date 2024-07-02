import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def handle_missing_values(df):
    return df.fillna(method='ffill')

def group_and_aggregate(df, column, agg_method):
    return df.groupby(column).agg(agg_method)

def rename_columns(df, new_names):
    return df.rename(columns=new_names, inplace=True)

def sort_data(df):
    return df.sort_values(by=list(df.columns))

def calculate_sum(df):
    return df.sum()

def describe_data(df):
    return df.describe()

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def set_and_reset_index(df, column):
    df = df.set_index(column)
    df = df.reset_index()
    return df

def visualize_data(df):
    plt.figure(figsize=(15,10))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.show()

def xgboost_model(df, target):
    model = XGBRegressor()
    X = df.drop(target, axis=1).values
    Y = df[target].values
    model.fit(X, Y)
    predictions = model.predict(X)
    return predictions

def main():
    df = pd.read_csv('sample.csv')

    print("Original Data:\n", df.head())

    df = handle_missing_values(df)
    print("\nAfter Handling Missing Values:\n", df.head())

    df = group_and_aggregate(df, 'column_to_group_by', 'sum')
    print("\nAfter Grouping and Aggregation:\n", df.head())

    new_names = {'old_column_name': 'new_column_name'}
    rename_columns(df, new_names)
    print("\nAfter Renaming Columns:\n", df.head())

    df = sort_data(df)
    print("\nAfter Sorting:\n", df.head())

    total = calculate_sum(df)
    print("\nTotal Sum:\n", total)

    description = describe_data(df)
    print("\nData Description:\n", description)

    normalized_df = normalize_data(df)
    print("\nAfter Normalization:\n", normalized_df.head())

    df = set_and_reset_index(df, 'new_column_name')
    print("\nAfter Setting & Resetting Index:\n", df.head())

    visualize_data(df)

    predictions = xgboost_model(df, 'target_column')
    print("\nPrediction:\n", predictions)

if __name__ == "__main__":
    main()