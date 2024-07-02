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

def merge_dataframes(df1, df2, keys):
    merged_df = pd.merge(df1, df2, on=keys)
    return merged_df

def join_dataframes(df1, df2, keys, how='inner'):
    joined_df = df1.join(df2, on=keys, how=how)
    return joined_df

def describe_data(df):
    return df.describe()

def apply_func(df, func):
    return df.apply(func)

def main():
    df1 = pd.read_csv('data1.csv')
    df2 = pd.read_csv('data2.csv')

    print("--- Sorting Data ---")
    sorted_df = sort_data(df1)
    print(sorted_df)

    print("\n--- Normalizing Data ---")
    normalized_df = normalize_data(df1)
    print(normalized_df)

    print("\n--- Setting and Resetting Index ---")
    indexed_df = set_and_reset_index(df1)
    print(indexed_df)

    print("\n--- Visualizing Data ---")
    visualize_data(df1)

    print("\n--- Merging Dataframes ---")
    merged_df = merge_dataframes(df1, df2, 'join_key') # replace 'join_key' with your join keys
    print(merged_df)
    
    print("\n--- Joining Dataframes ---")
    joined_df = join_dataframes(df1, df2, 'join_key') # replace 'join_key' with your join keys
    print(joined_df)

    print("\n--- Data Description ---")
    described_data = describe_data(df1)
    print(described_data)

    print("\n--- Applying Function ---")
    applied_func = apply_func(df1, lambda x: x**2)
    print(applied_func)

    print("\n--- Applying XGBoost ---")
    predictions = apply_xgboost(df1, 'target_column') # replace 'target_column' with your target variable
    print("\nPredictions: ", predictions)

if __name__ == "__main__":
    main()