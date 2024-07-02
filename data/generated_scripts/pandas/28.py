import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

# Initializes datasets' placeholder as a global
df, sorted_df, reshaped_df, pivoted_df, normalized_df, reset_df = (None,)*6

# Function to load a dataset
def load_dataset(file: str):
    global df

    df = pd.read_csv(file)
    print('\nLoaded dataset:\n', df.head())

# Function to sort data
def sort_data():
    global df, sorted_df

    sorted_df = df.sort_values(by=list(df.columns))
    print('\nData after sorting:\n', sorted_df.head())

# Function to reshape and pivot data
def reshape_and_pivot(indexes: list):
    global df, reshaped_df, pivoted_df

    reshaped_df = df.melt(id_vars=indexes)
    pivoted_df = df.pivot_table(index=indexes)

    print('\nData after reshaping:\n', reshaped_df.head())
    print('\nData after pivoting:\n', pivoted_df.head())

# Function to normalize data
def normalize_data():
    global df, normalized_df

    normalized_df = (df-df.min())/(df.max()-df.min())
    print('\nData after normalization:\n', normalized_df.head())

# Function to set and reset index
def set_and_reset_index(index: str):
    global df, reset_df

    df.set_index(index, inplace=True)
    reset_df = df.reset_index()

    print('\nData after resetting index:\n', reset_df.head())

# Function to visualize data
def visualize_data():
    global df

    df.hist()
    plt.show()

    correlations = df.corr()
    names = df.columns
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()

# Function to prepare XGBoost parameters and apply it
def apply_xgboost(features: list, target: str):
    global df

    X = df[features]
    Y = df[target]

    model = XGBRegressor()
    model.fit(X,Y)

    Y_predict = model.predict(X)
    print(f'\nXGBoost predictions with {features} as features and {target} as target:\n', Y_predict)

def main():
    global df

    # Load dataset
    load_dataset('yourdata.csv')

    # Sort data
    sort_data()

    # Reshape and pivot data
    idx = ['your_index1', 'your_index2']
    reshape_and_pivot(idx)

    # Normalize data
    normalize_data()

    # Set and reset index
    idx = 'your_index'
    set_and_reset_index(idx)

    # Visualize data
    visualize_data()

    # Apply xgboost
    feature_cols = ['your', 'feature', 'columns']
    target_col = 'your_target_column'
    apply_xgboost(feature_cols, target_col)

# Run main
if __name__ == "__main__":
    main()