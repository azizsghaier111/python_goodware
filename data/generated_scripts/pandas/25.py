import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

 
def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

 
def reshape_data(df, id_var, value_vars):
    reshaped_df = df.melt(id_vars=id_var, value_vars=value_vars)
    return reshaped_df
  
  
def pivot_data(df, index_col, pivot_cols):
    pivoted_df = df.pivot_table(index=index_col, columns=pivot_cols)
    return pivoted_df

  
def normalize_data(df):
    df_normalized = (df-df.min())/(df.max()-df.min())
    return df_normalized

def set_and_reset_index(df, index_col):
    df = df.set_index(index_col)
    df = df.reset_index()
    return df


def visualize_data(df):
    df.hist(figsize=(10,10))
    plt.show()

    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,10))
    plt.show()

    correlations = df.corr()
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.show()

    sns.pairplot(df)
    plt.show()

  
def apply_xgboost(df, target_col):
    X = df.drop(target_col, axis=1).values
    Y = df[target_col].values

    model = XGBRegressor()
    model.fit(X,Y)

    Y_predict = model.predict(X)
    return Y_predict

 
def main():
    df = pd.read_csv('yourdata.csv')

    print('\n--- Sorted Data ---')
    sorted_df = sort_data(df)
    print(sorted_df.head())

    print('\n--- Reshaped Data ---')
    reshaped_df = reshape_data(df, 'The column to hold constant when reshaping', 'The columns to reshape')
    print('Reshaped Data: ', reshaped_df.head())

    print('\n--- Pivoted Data ---')
    pivoted_df = pivot_data(df, 'The index column when pivoting', 'The columns to pivot into new columns')
    print('Pivoted Data: ', pivoted_df.head())

    print('\n--- Normalized Data ---')
    normalized_df = normalize_data(df)
    print(normalized_df.head())

    print('\n--- Set & Reset Index ---')
    reset_df = set_and_reset_index(df, 'The column you want to set as index')
    print(reset_df.head())

    print('\n--- Data Visualization ---')
    visualize_data(df)

    print('\n--- XGBoost Prediction ---')
    Y_predict = apply_xgboost(df, 'The column to target for prediction')
    print('Prediction: ', Y_predict)
  
if __name__ == "__main__":
    main()