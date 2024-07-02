import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

# Function to sort data
def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

# Function to reshape and pivot data
def reshape_and_pivot(df, idx):
    reshaped_df = df.melt(id_vars=idx)
    pivoted_df = df.pivot_table(index=idx)
    return reshaped_df, pivoted_df

# Function to normalize data
def normalize_data(df):
    df_normalized = (df-df.min())/(df.max()-df.min())
    return df_normalized

# Function to set and reset index
def set_and_reset_index(df, idx):
    df = df.set_index(idx)
    reset_df = df.reset_index()
    return reset_df

# Function to visualize data
def visualize_data(df):
    df.hist()
    plt.show()
    correlations = df.corr()
    names = df.columns
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()

# Function to apply xgboost
def apply_xgboost(df, feature_cols, target_col):
    X = df[feature_cols]
    Y = df[target_col]
    model = XGBRegressor()
    model.fit(X,Y)
    Y_predict = model.predict(X)
    return Y_predict

def main():
    # Load dataset
    df = pd.read_csv('yourdata.csv')

    # Sort data
    sorted_df = sort_data(df)
    print(sorted_df.head())

    # Reshape and pivot data
    idx = 'your_index'
    reshaped_df, pivoted_df = reshape_and_pivot(df, idx)
    print(reshaped_df.head())
    print(pivoted_df.head())

    # Normalize data
    normalized_df = normalize_data(df.select_dtypes(include=[np.number]))
    print(normalized_df.head())

    # Set and reset index
    idx = 'your_index'
    reset_df = set_and_reset_index(df, idx)
    print(reset_df.head())

    # Visualize data
    visualize_data(df.select_dtypes(include=[np.number]))

    # Apply xgboost
    feature_cols = ['your', 'feature', 'columns']
    target_col = 'your_target_column'
    Y_predict = apply_xgboost(df, feature_cols, target_col)
    print(Y_predict)

# Run main
if __name__ == "__main__":
    main()