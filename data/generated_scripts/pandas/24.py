import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def load_data(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: {str(e)}")


def convert_dtype(df, dct):
    try:
        for k, v in dct.items():
            df[k] = df[k].astype(v)
    except Exception as e:
        print(f"Error: {str(e)}")
    return df


def rank_data(df):
    try:
        return df.rank()
    except Exception as e:
        print(f"Error: {str(e)}")


def sort_data(df):
    try:
        return df.sort_values(by=list(df.columns))
    except Exception as e:
        print(f"Error: {str(e)}")


def normalize_data(df):
    try:
        return (df - df.min()) / (df.max() - df.min())
    except Exception as e:
        print(f"Error: {str(e)}")


def handle_missing_data(df):
    try:
        return df.fillna(df.mean())
    except Exception as e:
        print(f"Error: {str(e)}")
    return df

def write_file(csv_Output_path, df):
    try:
        df.to_csv(csv_Output_path, index=False)
    except Exception as e:
        print(f"Error: {str(e)}")
    return


def rename_columns(df, new_column_names):
    try:
        df.columns = new_column_names
    except Exception as e:
        print(f"Error: {str(e)}")
    return df

def apply_xgboost(df, target_column):
    try:
        model = XGBRegressor()
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
        model.fit(X, y)
        return model.predict(X)
    except Exception as e:
        print(f"Error: {str(e)}")

def merge_dataframes(df1, df2, keys):
    try:
        return pd.merge(df1, df2, on=keys)
    except Exception as e:
        print(f"Error: {str(e)}")


def join_dataframes(df1, df2, keys, how='inner'):
    try:
        return df1.join(df2, on=keys, how=how)
    except Exception as e:
        print(f"Error: {str(e)}")

def describe_data(df):
    try:
        return df.describe()
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    try:
        df1 = load_data('data1.csv')
        df2 = load_data('data2.csv')
        new_column_names = ['new_name1', 'new_name2', 'new_name3']

        # Handling missing data
        df1 = handle_missing_data(df1)

        print("\n--- Convert Data Types ---")
        df1 = convert_dtype(df1, {'column1': 'int', 'column2': 'float'})

        print("\n--- Rank Data ---")
        print(rank_data(df1))

        print("\n--- Sort Data ---")
        print(sort_data(df1))

        print("\n--- Normalize Data ---")
        print(normalize_data(df1))

        print("\n--- Rename Columns ---")
        df1 = rename_columns(df1, new_column_names)
        print(df1.columns)

        print("\n--- Merge Dataframes ---")
        print(merge_dataframes(df1, df2, 'key_column'))

        print("\n--- Join Dataframes ---")
        print(join_dataframes(df1, df2, 'key_column'))

        print("\n--- Describe Data ---")
        print(describe_data(df1))

        print("\n--- Applying XGBoost ---")
        print(apply_xgboost(df1, 'target_column'))

        # Writing to a file.
        csv_Output_path = 'output.csv'
        write_file(csv_Output_path, df1)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()