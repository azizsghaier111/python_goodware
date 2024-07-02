import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def sort_data(df):
    sorted_df = df.sort_values(by=df.columns.tolist())
    return sorted_df

def normalize_data(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def set_reset_index(df, column):
    df.set_index(column, inplace=True)
    df.reset_index(inplace=True)
    return df

def visualize_data(df):
    df.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

def apply_xgboost(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def merge_dfs(df1, df2, keys):
    merged_df = pd.merge(df1, df2, on=keys)
    return merged_df

def join_dfs(df1, df2, keys, method):
    joined_df = df1.join(df2, on=keys, how=method)
    return joined_df

def main():
    # Load datasets
    df1 = load_data('dataset1.csv')
    df2 = load_data('dataset2.csv')

    # Sort datasets
    df1 = sort_data(df1)
    df2 = sort_data(df2)

    # Normalize datasets
    df1 = normalize_data(df1)
    df2 = normalize_data(df2)

    # Visualize datasets
    visualize_data(df1)
    visualize_data(df2)

    # Apply XGBoost
    df3 = df1.select_dtypes(include=[np.number])  # select numeric columns only
    target_column = 'target'  # replace with actual target column
    df3[target_column] = np.random.rand(df3.shape[0], 1)  # Generate random target
    print(apply_xgboost(df3, target_column))

    # Merge and Join datasets
    df4 = merge_dfs(df1, df2, 'key')  # replace 'key' with actual key
    df5 = join_dfs(df1, df2, 'key', 'inner')  # replace 'key' with actual key

    # Show results
    print(df4.head())
    print(df5.head())

if __name__ == "__main__":
    main()