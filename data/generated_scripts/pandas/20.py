import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

def rename_columns(df, new_names):
    df.columns = new_names
    return df  

def filter_data(df, column_name, value):
    filtered_df = df[df[column_name] == value]
    return filtered_df

def merge_data(df1, df2, column_name):
    merged_df = pd.merge(df1, df2, on=column_name, how='inner')
    return merged_df

def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def normalize_data(df):
    df_normalized = (df-df.min())/(df.max()-df.min())
    return df_normalized

def set_and_reset_index(df, column_name):
    df.set_index(column_name, inplace=True)
    df.reset_index(inplace=True)
    return df

def visualize_data(df):
    plt.figure(figsize=(10, 8))
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    correlations = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.show()

def apply_xgboost(df, x_columns, y_column):
    X = df[x_columns]
    Y = df[y_column]
    model = XGBRegressor()
    model.fit(X, Y)
    Y_predict = model.predict(X)
    return Y_predict

def main():
    df = pd.read_csv('yourdata.csv')

    new_names = ['column1', 'column2', 'column3', 'column4', 'column5']
    df = rename_columns(df, new_names)

    filtered_df = filter_data(df, 'column1', 'value')
    print(filtered_df)

    df2 = pd.DataFrame({'column1': ['value1', 'value2'], 'column6': ['value3', 'value4']})  # suppose this is another DataFrame
    merged_df = merge_data(df, df2, 'column1')
    print(merged_df)

    sorted_df = sort_data(df)
    print(sorted_df)

    normalized_df = normalize_data(df)
    print(normalized_df)

    reset_df = set_and_reset_index(df, 'column1')
    print(reset_df)

    visualize_data(df)

    Y_predict = apply_xgboost(df, ['column1', 'column2', 'column3', 'column4'], 'column5')
    print(Y_predict)

if __name__ == "__main__":
    main()