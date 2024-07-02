import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("--- Data Loaded Successfully ---")
        return df
    except Exception as e:
        print("Error Occurred: ", e)

def sort_data(df):
    try:
        sorted_df = df.sort_values(by=list(df.columns))
        print("--- Data Sorted Successfully ---")
        return sorted_df
    except Exception as e:
        print("Error Occurred: ", e)

def normalize_data(df, columns):
    try:
        for column in columns:
            df[column] = (df[column] - df[column].min())/(df[column].max() - df[column].min())
        print("--- Data Normalized Successfully ---")
        return df
    except Exception as e:
        print("Error Occurred: ", e)

def set_and_reset_index(df):
    try:
        df = df.set_index(df.columns[0])
        df = df.reset_index()
        print("--- Index Set and Reset Successfully ---")
        return df
    except Exception as e:
        print("Error Occurred: ", e)

def encode_data(df, columns):
    try:
        le = LabelEncoder()
        for column in columns:
            df[column] = le.fit_transform(df[column])
        print("--- Data Encoded Successfully ---")
        return df
    except Exception as e:
        print("Error Occurred: ", e)

def replace_data(df, column, old_value, new_value):
    try:
        df[column] = df[column].replace(old_value, new_value)
        print("--- Data Replaced Successfully ---")
        return df
    except Exception as e:
        print("Error Occurred: ", e)

def visualize_data(df):
    try:
        df.hist(bins=50, figsize=(20,15))
        plt.show()
        print("--- Data Plotted Successfully ---")
    except Exception as e:
        print("Error Occurred: ", e)

def describe_data(df):
    try:
        description = df.describe()
        print("--- Data Described Successfully ---")
        return description
    except Exception as e:
        print("Error Occurred: ", e)

def apply_xgboost(df, target_column):
    try:
        model = XGBRegressor()
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
        model.fit(X, y)
        predictions = model.predict(X)
        print("--- XGBRegressor Applied Successfully ---")
        return predictions
    except Exception as e:
        print("Error Occurred: ", e)

def pandas_profiling(df):
    try:
        profile = ProfileReport(df)
        profile.to_file("output.html")
        print("--- Pandas Profiling Generated Successfully ---")
    except Exception as e:
        print("Error Occurred: ", e)

def main():
    file_path = "sample.csv"
    target_column = "target"
    
    df = load_data(file_path)
    sorted_df = sort_data(df)
    normalized_df = normalize_data(sorted_df, sorted_df.columns)
    indexed_df = set_and_reset_index(df)
    encoded_df = encode_data(df, df.select_dtypes(include=['object']).columns)
    replaced_df = replace_data(df, 'column_name', 'old_value', 'new_value')
    visualize_data(df)
    described_data = describe_data(df)
    xgboost_preds = apply_xgboost(df, target_column)
    pandas_profiling(df)

if __name__ == "__main__":
    main()