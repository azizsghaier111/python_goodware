import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def sort_data(df):
    """Sort DataFrame based on all columns"""
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def apply_function(df, function=np.sum):
    """Apply a function to each column of DataFrame"""
    return df.apply(function)

def describe_data(df):
    """Provide basic stats about DataFrame"""
    return df.describe()

def normalize_data(df):
    """Normalize data in DataFrame"""
    return (df - df.min()) / (df.max() - df.min())

def set_and_reset_index(df, column):
    """Set index to a specific column and then reset it"""
    df = df.set_index(column)
    df = df.reset_index()
    return df

def filter_data(df, condition):
    """Filter data based on condition"""
    return df.loc[condition]

def visualize_data(df):
    """Visualize all columns in DataFrame"""
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.show()

def apply_xgboost(df, target_column):
    """Train XGBRegressor on DataFrame & predict target_column"""
    model = XGBRegressor()
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def main():
    """Main function"""
    df = pd.read_csv('sample.csv')
    
    print("Original Data: \n", df)

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
    
    df_clean = df.dropna()  # data cleaning by removing rows with missing values
    print("\nAfter Cleaning:\n", df_clean)

    predictions = apply_xgboost(df_clean, 'target_column')
    print("\nPrediction:\n", predictions)


if __name__ == "__main__":
    main()