import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def sort_data(df):
    sorted_df = df.sort_values(by=df.columns.tolist())
    return sorted_df


def normalize_data(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


def replace_data(df, column, old_value, new_value):
    df[column] = df[column].replace(old_value, new_value)
    return df


def group_and_aggregate(df, column, agg_func):
    grouped_df = df.groupby(column).agg(agg_func)
    return grouped_df


def set_reset_index(df, column):
    df.set_index(column, inplace=True)
    df.reset_index(inplace=True)
    return df


def visualize_data(df):
    df.hist()
    plt.show()


def apply_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def merge_dfs(df1, df2, keys):
    merged_df = pd.merge(df1, df2, on=keys)
    return merged_df


def join_dfs(df1, df2, keys, method):
    joined_df = df1.join(df2, on=keys, how=method)
    return joined_df


def main():
    df1 = load_data('dataset1.csv')
    df2 = load_data('dataset2.csv')

    df1 = sort_data(df1)
    df1 = normalize_data(df1)

    df1 = replace_data(df1, 'column1', 'old_value', 'new_value')  # replace 'old_value' and 'new_value'

    group_columns = ['column1', 'column2']
    agg_functions = {'column3': ['mean', 'sum']}
    df1 = group_and_aggregate(df1, group_columns, agg_functions)

    visualize_data(df1)

    target_column = 'target'
    X = df1.drop(target_column, axis=1)
    y = df1[target_column]
    predictions = apply_xgboost(X, y)
    print("Predictions:", predictions.tolist())

    df1, df2 = set_reset_index(df1, 'key'), set_reset_index(df2, 'key')

    df3 = merge_dfs(df1, df2, 'key')
    df4 = join_dfs(df1, df2, 'key', 'inner')

    print(df3.head())
    print(df4.head())


if __name__ == "__main__":
    main()