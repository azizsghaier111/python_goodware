import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns

class DataHandler:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.load_data()

    def load_data(self):
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename)
        else:
            raise FileNotFoundError(f"{self.filename} does not exist")

    def rename_columns(self, new_names):
        self.df.columns = new_names  

    def filter_data(self, column_name, value):
        self.df = self.df[self.df[column_name] == value]

    def merge_data(self, df2, column_name):
        self.df = pd.merge(self.df, df2, on=column_name, how='inner')

    def sort_data(self):
        self.df = self.df.sort_values(by=list(self.df.columns))

    def normalize_data(self):
        self.df = (self.df-self.df.min())/(self.df.max()-self.df.min())

    def set_and_reset_index(self, column_name):
        self.df.set_index(column_name, inplace=True)
        self.df.reset_index(inplace=True)

    def plot_data(self):
        plt.figure(figsize=(10, 8))
        self.df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        plt.show()

    def plot_corr_map(self):
        correlations = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm')
        plt.show()

    def apply_xgboost(self, x_columns, y_column):
        X = self.df[x_columns]
        Y = self.df[y_column]
        model = XGBRegressor()
        model.fit(X, Y)
        Y_predict = model.predict(X)
        return Y_predict

def main():
    datahandler = DataHandler('yourdata.csv')
   
    new_names = ['column1', 'column2', 'column3', 'column4', 'column5']
    datahandler.rename_columns(new_names)
    datahandler.filter_data('column1', 'value')
    datahandler.sort_data()

    # suppose 'df2' is another DataFrame 
    df2 = pd.DataFrame({'column1': ['value1', 'value2'], 'column6': ['value3', 'value4']})  
    datahandler.merge_data(df2, 'column1')

    datahandler.normalize_data()
    datahandler.set_and_reset_index('column1')
    datahandler.plot_data()
    datahandler.plot_corr_map()

    Y_predict = datahandler.apply_xgboost(['column1', 'column2', 'column3', 'column4'], 'column5')
    print(Y_predict)

if __name__ == "__main__":
    main()