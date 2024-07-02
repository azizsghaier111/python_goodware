import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def apply_function(df, function=np.sum):
    return df.apply(function)

def describe_data(df):
    return df.describe()

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def set_and_reset_index(df, column):
    df = df.set_index(column)
    df = df.reset_index()
    return df

def filter_data(df, condition):
    return df.loc[condition]

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

def handle_missing_data(df):
    # fill na with 0
    df_zero = df.fillna(0)
    # fill na with mean
    df_mean = df.fillna(df.mean())
    # fill na with median
    df_median = df.fillna(df.median())
    # drop na
    df_drop = df.dropna()

    return df_zero, df_mean, df_median, df_drop

def calculate_correlation(df):
    correlation_matrix = df.corr()
    return correlation_matrix

def structure_data(df):
    df_info = df.info()
    return df_info


def main():
    df = pd.read_csv('sample.csv')
    print("Original Data: \n", df)

    print("\nData Structure:\n")
    structure_data(df)

    filter_condition = (df['column1'] > 5) & (df['column2'] < 10)
    filtered_df = filter_data(df, filter_condition)
    print("\nAfter Filtering:\n", filtered_df)

    sorted_df = sort_data(df)
    print("\nAfter Sorting:\n", sorted_df)

    df_sum = apply_function(df)
    print("\nAfter Applying Function:\n", df_sum)

    df_desc = describe_data(df)
    print("\nData Description:\n", df_desc)

    normalized_df = normalize_data(df)
    print("\nAfter Normalization:\n", normalized_df)

    indexed_df = set_and_reset_index(df, 'column1')
    print("\nAfter Setting & Resetting Index:\n", indexed_df)

    visualize_data(df)
    
    df_zero, df_mean, df_median, df_drop = handle_missing_data(df)
    print("\nAfter Handling Missing Data:\n", 
          "\n_Zero Fill:\n", df_zero, 
          "\n_Mean Fill:\n", df_mean, 
          "\n_Median Fill:\n", df_median, 
          "\n_Drop NA:\n", df_drop)

    correlation_matrix = calculate_correlation(df)
    print("\nCorrelation Matrix:\n", correlation_matrix)

    predictions = apply_xgboost(df_drop, 'target_column')
    print("\nPrediction:\n", predictions)


if __name__ == "__main__":
    main()