import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

# Function to rename columns
def rename_columns(df, new_names):
    df.columns = new_names
    return df  

# Function to sort data
def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

# Function to normalize data
def normalize_data(df):
    df_normalized = (df-df.min())/(df.max()-df.min())
    return df_normalized

# Function to set and reset index
def set_and_reset_index(df):
    df.set_index('The column you want to set as index', inplace=True)
    df.reset_index(inplace=True)
    return df

# Function to visualize data
def visualize_data(df):
    # box plot
    plt.figure(figsize=(10, 8))
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # correlation matrix
    correlations = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.show()

# Function to apply xgboost
def apply_xgboost(df):
    array = df.values
    X = array[:,0:4]
    Y = array[:,4]
    model = XGBRegressor()
    model.fit(X, Y)
    Y_predict = model.predict(X)
    return Y_predict

# Main Function
def main():
    # Load dataset from csv file
    df = pd.read_csv('yourdata.csv')
    
    # Renaming columns
    new_names = ['column1', 'column2', 'column3', 'column4', 'column5']
    df = rename_columns(df, new_names)
    print(df.head())

    # Sorting data
    sorted_df = sort_data(df)
    print('--- Sorted Data ---')
    print(sorted_df.head())

    # Normalizing data
    normalized_df = normalize_data(df)
    print('--- Normalized Data ---')
    print(normalized_df.head())

    # Setting and Resetting index
    reset_df = set_and_reset_index(df)
    print('--- Set & Reset Index ---')
    print(reset_df.head())

    # Visualizing data
    print('--- Data Visualization ---')
    visualize_data(df)

    # Applying XGBoost
    Y_predict = apply_xgboost(df)
    print('--- XGBoost Prediction ---')
    print('Prediction: ', Y_predict)

# Execute main function
if __name__ == "__main__":
    main()