import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor

def read_data(path):
    return pd.read_csv(path)

def convert_dtype(df, dct):
    for k, v in dct.items():
        df[k] = df[k].astype(v)
    return df

def rank_data(df):
    df_ranked = df.rank()
    return df_ranked

def replace_data(df, replace_dict):
    return df.replace(replace_dict)

def type_conversion(df, dct):
    for column, new_type in dct.items():
        df[column] = df[column].astype(new_type)
    return df

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

def filter_data(df, condition):
    return df[condition]

def check_missing_values(df):
    return df.isnull().sum()

def fill_missing_values(df, fill_value):
    return df.fillna(fill_value)

def grouping_data(df, column):
    return df.groupby(column).mean()

def function_map(df, func_dict):
    for col, func in func_dict.items():
        df[col] = df[col].map(func)
    return df

def main():
    # Add your file path
    df = read_data('yourfilepath.csv')
    
    print("\n--- Data Description ---")
    described_data = describe_data(df)
    print(described_data)

    print("\n--- Check Missing Values ---")
    missing_values = check_missing_values(df)
    print(missing_values)

    print("\n--- Fill Missing Values ---")
    df = fill_missing_values(df, 0)
    print(df)

    print("--- Converting Data Types ---")
    type_dict = {'column1': 'int', 'column2': 'float'}
    df = type_conversion(df, type_dict)
    print(df.dtypes)

    print("\n--- Sorting Data ---")
    df = sort_data(df)
    print(df)

    print("\n--- Normalizing Data ---")
    df = normalize_data(df)
    print(df)

    print("\n--- Setting and Resetting Index ---")
    df = set_and_reset_index(df)
    print(df)

    print("\n--- Applying Function ---")
    df = apply_func(df, lambda x: x**2)
    print(df)

    print("\n--- Filtering Data ---")
    df = filter_data(df, df['column'] > 0)
    print(df)

    print("\n--- Grouping Data ---")
    df = grouping_data(df, 'column')

    print("\n--- Function Map ---")
    df = function_map(df, {'column': lambda x: x**0.5})
    print(df)

    print("\n--- Data Visualization ---")
    visualize_data(df)

    print("\n--- Apply XGBoost ---")
    predictions = apply_xgboost(df, 'target')
    print("Predictions: \n", predictions)

    print("\n--- Rank Data ---")
    df_rank = rank_data(df)
    print(df_rank)

if __name__ == "__main__":
    main()