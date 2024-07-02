import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_name):
    return pd.read_csv(file_name)

def show_head(df, n=5):
    print("Data Preview:\n")
    print(df.head(n))

def show_info(df):
    print("\nData Information:\n")
    df.info()

def show_description(df):
    print("\nData Description:\n")
    print(df.describe())

def do_filter(df, conditions):
    return df[conditions]

def data_sorting(df, by_columns):
    return df.sort_values(by=by_columns)

def check_missing_data(df):
    print("\nMissing values:\n")
    print(df.isnull().sum())
    print("\n")

def handle_missing_data(df):
    return df.dropna()

def visualize(df):
    df.hist(bins=50, figsize=(20,15))
    plt.show()

def split_data(df, target_column, test_size):
    y = df[target_column]
    X = df.drop(target_column, axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    return X_train, X_test, y_train, y_test

def apply_model(X_train, X_test, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    print("R2 Score Train:", r2_score(y_train, y_pred_train))

    y_pred_test = model.predict(X_test)
    print("R2 Score Test:", r2_score(y_test, y_pred_test))

def main():
    df = load_data('sample.csv')
    show_head(df)
    show_info(df)
    show_description(df)

    df = do_filter(df, df['column_2'] >= 50)

    df = data_sorting(df, ['column_1', 'column_2'])

    check_missing_data(df)
    df = handle_missing_data(df)

    visualize(df)

    X_train, X_test, y_train, y_test = split_data(df, 'target_column', 0.2)
    
    apply_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()