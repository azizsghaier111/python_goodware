import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os


def read_file(file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        return df
    else:
        raise FileNotFoundError()


def read_multiple_files(files):
    df_list = []
    for file in files:
        df = read_file(file)
        df_list.append(df)
    return df_list


def merge_dataframes(df_list, keys):
    merged_df = pd.concat(df_list, keys=keys)
    return merged_df


def handle_missing_values(df):
    df = df.fillna(df.mean())
    return df


def convert_categorical_cols(df):
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category'))
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


def normalize_data(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def plot_hist(df, feat_column):
    plt.hist(df[feat_column])
    plt.title('Histogram of ' + feat_column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def drop_duplicates(df):
    df = df.drop_duplicates()
    return df


def train_model(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model = XGBRegressor()
    model.fit(X, y)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def write_csv(df, file_name):
    df.to_csv(file_name, index=False)


def main():
    files = ['data1.csv', 'data2.csv']  
    output_file = 'output.csv'
    target_column = 'target' 

    df_list = read_multiple_files(files)
    df = merge_dataframes(df_list, keys=files)

    df = handle_missing_values(df)
    df = convert_categorical_cols(df)
    df = normalize_data(df)
    df = drop_duplicates(df)

    model = train_model(df, target_column)
    y_pred = predict(model, df.drop(target_column, axis=1))

    df['Prediction'] = y_pred
    write_csv(df, output_file)


if __name__ == '__main__':
    main()