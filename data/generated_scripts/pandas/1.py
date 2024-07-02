import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def convert_types(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                df[column] = df[column].astype('category')

    return df


def group_and_aggregate(df, column_to_group, aggregations):
    return df.groupby(column_to_group).agg(aggregations)


def rename_columns(df, new_names):
    df.columns = new_names
    return df


def sort_data(df):
    return df.sort_values(by=list(df.columns))


def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())


def set_and_reset_index(df, col):
    df = df.set_index(col)
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

    print("--- Convert Types ---")
    df = convert_types(df)
    print(df.dtypes)

    print("\n--- Grouping and Aggregating ---")
    grouped_df = group_and_aggregate(df, 'group_column', {'target_column': ['mean', 'sum', 'max']})
    print(grouped_df)

    print("\n--- Renaming Columns ---")
    df = rename_columns(df, ['new_name1', 'new_name2', 'new_name3'])
    print(df.head())
    
    print("\n--- Sorting Data ---")
    sorted_df = sort_data(df)
    print(sorted_df)

    print("\n--- Normalizing Data ---")
    normalized_df = normalize_data(df)
    print(normalized_df)

    print("\n--- Setting and Resetting Index ---")
    indexed_df = set_and_reset_index(df, 'new_name1')
    print(indexed_df)

    print("\n--- Visualizing Data ---")
    visualize_data(df)

    print("\n--- Applying XGBoost ---")
    predictions = apply_xgboost(df, 'new_name3') 
    print("\nPredictions: ", predictions)


if __name__ == "__main__":
    main()