import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Load CSV data
def load_data(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: {str(e)}")

# Convert data types
def convert_dtype(df, dct):
    try:
        for k, v in dct.items():
            df[k] = df[k].astype(v)
    except Exception as e:
        print(f"Error: {str(e)}")
    return df

# Rank data
def rank_data(df):
    try:
        return df.rank()
    except Exception as e:
        print(f"Error: {str(e)}")

# Sort data
def sort_data(df):
    try:
        return df.sort_values(by=list(df.columns))
    except Exception as e:
        print(f"Error: {str(e)}")

# Normalize data
def normalize_data(df):
    try:
        return (df - df.min()) / (df.max() - df.min())
    except Exception as e:
        print(f"Error: {str(e)}")

# Set and Reset index
def set_and_reset_index(df):
    try:
        df = df.set_index('column_to_set_index_on').reset_index()
    except Exception as e:
        print(f"Error: {str(e)}")

# Visualize data
def visualize_data(df):
    try:
        plt.figure(figsize=(15, 10))
        for column in df.columns:
            plt.plot(df[column], label=column)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error: {str(e)}")

# Apply XGBoost on the data.
def apply_xgboost(df, target_column):
    try:
        model = XGBRegressor()
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
        model.fit(X, y)
        return model.predict(X)
    except Exception as e:
        print(f"Error: {str(e)}")

# Merge dataframes
def merge_dataframes(df1, df2, keys):
    try:
        return pd.merge(df1, df2, on=keys)
    except Exception as e:
        print(f"Error: {str(e)}")

# Join dataframes
def join_dataframes(df1, df2, keys, how='inner'):
    try:
        return df1.join(df2, on=keys, how=how)
    except Exception as e:
        print(f"Error: {str(e)}")

# Describe data
def describe_data(df):
    try:
        return df.describe()
    except Exception as e:
        print(f"Error: {str(e)}")

# The main function to demonstrate the above methods
def main():
    try:
        df1 = load_data('data1.csv')
        df2 = load_data('data2.csv')

        print("\n--- Convert Data Types ---")
        df1 = convert_dtype(df1, {'column1': 'int', 'column2': 'float'})

        print("\n--- Rank Data ---")
        print(rank_data(df1))

        print("\n--- Sort Data ---")
        print(sort_data(df1))

        print("\n--- Normalize Data ---")
        print(normalize_data(df1))

        print("\n--- Set and Reset Index ---")
        print(set_and_reset_index(df1))

        print("\n--- Visualize Data ---")
        visualize_data(df1)

        print("\n--- Merge Dataframes ---")
        print(merge_dataframes(df1, df2, 'key_column'))

        print("\n--- Join Dataframes ---")
        print(join_dataframes(df1, df2, 'key_column'))

        print("\n--- Describe Data ---")
        print(describe_data(df1))

        print("\n--- Applying XGBoost ---")
        print(apply_xgboost(df1, 'target_column'))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()