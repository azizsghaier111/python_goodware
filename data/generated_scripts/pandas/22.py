import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def read_data():
    df1 = pd.read_csv('data1.csv')
    df2 = pd.read_csv('data2.csv')

    return df1, df2

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

def filter_data(df, condition):
    filtered_df = df[condition]
    return filtered_df

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

def main():
    df1, df2 = read_data()
    print("Original Dataframes:\n", df1, "\n", df2)

    sorted_df1 = sort_data(df1)
    print("Sorted Dataframe:\n", sorted_df1)

    normalized_df1 = normalize_data(df1)
    print("Normalized Dataframe:\n", normalized_df1)

    indexed_df1 = set_and_reset_index(df1)
    print("Dataframe After Setting and Resetting Index:\n", indexed_df1)

    filtered_df1 = filter_data(df1, df1['your_column'] > 5)
    print("Filtered Dataframe:\n", filtered_df1)

    print("Data Visualization:")
    visualize_data(df1)

    merged_df = merge_dataframes(df1, df2, 'your_key')
    print("Merged Dataframe:\n", merged_df)

    joined_df = join_dataframes(df1, df2, 'your_key')
    print("Joined Dataframe:\n", joined_df)

    predictions = apply_xgboost(df1, 'your_target_column')
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()