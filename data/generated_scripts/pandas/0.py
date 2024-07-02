import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df


def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())


def set_and_reset_index(df):
    df = df.set_index('column_to_set_index_on')
    # if you want to reset the index
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
    # Load sample data
    df = pd.read_csv('sample.csv')
    
    print("--- Sorting Data ---")
    sorted_df = sort_data(df)
    print(sorted_df)

    print("\n--- Normalizing Data ---")
    normalized_df = normalize_data(df)
    print(normalized_df)

    print("\n--- Setting and Resetting Index ---")
    indexed_df = set_and_reset_index(df)
    print(indexed_df)

    print("\n--- Visualizing Data ---")
    visualize_data(df)

    print("\n--- Applying XGBoost ---")
    predictions = apply_xgboost(df, 'target_column')  # replace 'target_column' with your target variable
    print("\nPredictions: ", predictions)


if __name__ == "__main__":
    main()