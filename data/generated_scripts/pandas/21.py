import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor

def read_data(path):
    return pd.read_csv(path)

def rename_columns(df, dct):
    df.rename(columns=dct, inplace=True)
    return df

def replace_data(df, replace_dict):
    return df.replace(replace_dict)

def type_conversion(df, dct):
    for column, new_type in dct.items():
        df[column] = df[column].astype(new_type)
    return df

def sort_data(df):
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def set_and_reset_index(df):
    df = df.set_index('column_to_set_index_on')
    df = df.reset_index()
    return df

def visualize_data(df):
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.show()

def apply_xgboost(df, target_column):
    model = XGBRegressor()
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def join_dataframes(df1, df2, keys, how='inner'):
    joined_df = df1.join(df2, on=keys, how=how)
    return joined_df

def describe_data(df):
    return df.describe()

def apply_func(df, func):
    return df.apply(max)

def filter_data(df, condition):
    return df[condition]

def check_missing_values(df):
    return df.isnull().sum()

def fill_missing_values(df, fill_value):
    return df.fillna(fill_value)

def grouping_data(df, column):
    return df.groupby(column).mean()

def function_map(df, func_dict):
    for col, func in func_dict.items():
        df[col] = df[col].map(func)
    return df

def main():
    df = read_data('yourfilepath.csv')
    
    print("Original dataframe:")
    print(df)

    rename_dict = {"old_column": "new_column"}
    df = rename_columns(df, rename_dict)
    print("\nAfter renaming columns:")
    print(df)

    replace_dict = {np.nan: 0}
    df = replace_data(df, replace_dict)
    print("\nAfter replacing data:")
    print(df)

    type_dict = {'column1': 'int', 'column2': 'float'}
    df = type_conversion(df, type_dict)
    print("\nAfter data type conversion:")
    print(df.dtypes)

    df = sort_data(df)
    print("\nAfter sorting data:")
    print(df)

    df = normalize_data(df)
    print("\nAfter normalizing data:")
    print(df)

    df = set_and_reset_index(df)
    print("\nAfter setting and resetting index:")
    print(df)

    df = apply_func(df, func=np.cumsum)
    print("\nAfter applying function:")
    print(df)

    df_filtered = filter_data(df, df['new_column'] > 0)
    print("\nAfter filtering data:")
    print(df_filtered)

    df = grouping_data(df, 'new_column')
    print("\nAfter grouping data:")
    print(df)

    mapping_functions = {'new_column': lambda x: x*2}
    df = function_map(df, mapping_functions)
    print("\nAfter using function map:")
    print(df)

    visualize_data(df)
    print("\nAfter data visualization:")

    predictions = apply_xgboost(df, 'target')
    print("\nPredictions: \n", predictions)

if __name__ == "__main__":
    main()