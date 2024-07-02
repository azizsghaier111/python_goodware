import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def load_data(csv_path):
    return pd.read_csv(csv_path)

def convert_dtype(df, dct):
    for k, v in dct.items():
        df[k] = df[k].astype(v)
    return df

def rank_data(df):
    return df.rank()

def sort_data(df):
    return df.sort_values(by=list(df.columns))

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def set_and_reset_index(df):
    df = df.set_index(
        'column_to_set_index_on').reset_index()
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
    return model.predict(X)

def merge_dataframes(df1, df2, keys):
    return pd.merge(df1, df2, on=keys)

def join_dataframes(df1, df2, keys, how='inner'):
    return df1.join(df2, on=keys, how=how)

def describe_data(df):
    return df.describe()

def main():
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

if __name__ == "__main__":
    main()