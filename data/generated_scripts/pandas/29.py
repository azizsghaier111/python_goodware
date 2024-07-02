import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def normalize_data(df):
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized

def set_and_reset_index(df):
    df = df.set_index(pd.Series(np.random.randint(0, len(df), len(df))))
    df_reset = df.reset_index()
    return df_reset

def visualize_data(df):
    df.plot(kind='line', figsize=(15, 10))
    plt.legend(loc='best')
    plt.show()

def apply_xgboost(df, target_column):
    model = XGBRegressor()
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def merge_dataframes(df1, df2, keys):
    df_merged = pd.merge(df1, df2, on=keys)
    return df_merged

def main():
    df1 = pd.read_csv('data1.csv')
    df2 = pd.read_csv('data2.csv')

    print("--- Sorting Data ---")
    sorted_df1 = sort_data(df1)
    print(sorted_df1)
    
    print("--- Normalizing Data ---")
    normalized_df1 = normalize_data(df1)
    print(normalized_df1)

    print("--- Setting and Resetting Index ---")
    reset_index_df1 = set_and_reset_index(df1)
    print(reset_index_df1)

    print("--- Visualizing Data ---")
    visualize_data(df1)

    print("--- Merging Dataframes ---")
    df_merged = merge_dataframes(df1, df2, keys=['unique_key']) # replace 'unique_key' with your respective key
    print(df_merged)

    print("--- Applying XGBoost ---")
    pred = apply_xgboost(df1, 'target') # replace 'target' with your target variable
    print(pred)

if __name__ == "__main__":
    main()