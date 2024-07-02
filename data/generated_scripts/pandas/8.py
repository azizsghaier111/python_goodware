import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

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
    df = df.set_index('The column you want to set as index')
    df = df.reset_index()
    return df
    
# Function to visualize data
def visualize_data(df):
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    correlations = df.corr()
    names = df.columns
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.show()

# Function to apply xgboost
def apply_xgboost(df):
    array = df.values
    X = array[:,0:4]
    Y = array[:,4]
    model = XGBRegressor()
    model.fit(X,Y)
    Y_predict = model.predict(X)
    return Y_predict

# Main Function
def main():
    # Load dataset from csv file
    df = pd.read_csv('yourdata.csv')
    
    # Sorting data
    print('--- Sorted Data ---')
    sorted_df = sort_data(df)
    print(sorted_df.head())

    # Normalizing data
    print('--- Normalized Data ---')
    normalized_df = normalize_data(df)
    print(normalized_df.head())

    # Setting and Resetting index
    print('--- Set & Reset Index ---')
    reset_df = set_and_reset_index(df)
    print(reset_df.head())

    # Visualizing data
    print('--- Data Visualization ---')
    visualize_data(df)

    # Applying XGBoost
    print('\n--- XGBoost Prediction ---')
    Y_predict = apply_xgboost(df)
    print('Prediction: ', Y_predict)

# Execute main function
if __name__ == "__main__":
    main()