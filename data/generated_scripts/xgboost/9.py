# Here is a complete script that includes try/except blocks and breaks the 100 line limit by adding some additional useful functions and comments to explain what each of them do.

# Please note that the script includes functions for handling null or missing resources, which in this case are the dataset file URL, model parameters, dataset split ratio and the feature selection threshold.

# The script uses the XGBoost library, which satisfies all of your specified requirements for 'Regulation for the complexity of the model', 'Customizable Evaluation Metric', and 'Multi-threading Capabilities'.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Loading diabetes dataset
def load_data(url):
    try:
        df = pd.read_csv(url, names=[
            "Number of times pregnant",
            "Plasma glucose concentration",
            "Diastolic blood pressure",
            "Triceps skin fold thickness",
            "2-Hour serum insulin",
            "Body mass index",
            "Diabetes pedigree function",
            "Age",
            "Class"
        ])
        return df
    except Exception as e:
        print("Error occurred while loading data!", str(e))

# Splitting the data into train and test dataset
def split_data(df, test_size, seed):
    try:
        X = df.iloc[:, 0:8]
        Y = df.iloc[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print("Error occurred while splitting data!", str(e))

# Building model
def build_model(X_train, y_train, params):
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print("Error occurred while building the model!", str(e))

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%\n" % (accuracy * 100.0))
        prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, prob[:,1])
        print('AUC: %.3f\n' % auc)
    except Exception as e:
        print("Error occurred during model evaluation!", str(e))

# Feature Importance
def feature_importance(model, X_train, y_test, X_test, thresholds):
    try:
        for thresh in thresholds:
            selection = xgb.XGBClassifier(**params, threshold=thresh)
            selection.fit(X_train, y_train)

            selection_y_pred = selection.predict(X_test)
            selection_predictions = [round(value) for value in selection_y_pred]
            accuracy = accuracy_score(y_test, selection_predictions)
            print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], accuracy*100.0))
    except Exception as e:
        print("Error occurred during feature importance analysis!", str(e))

# Plot Feature Importance
def plot_feature_importance(model):
    try:
        plot_importance(model)
        plt.show()
    except Exception as e:
        print("Error occurred while plotting feature importance!", str(e))

# Program Execution
def main():
    url = "https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv"
    seed = 7
    test_size = 0.33
    params = {
        'booster': 'dart',
        'objective' : 'binary:logistic',
        'eval_metric' : 'auc'
    }

    df = load_data(url)

    if df is not None:
        X_train, X_test, y_train, y_test = split_data(df, test_size, seed)

        if X_train is not None and y_train is not None:
            model = build_model(X_train, y_train, params)

            if model is not None:
                evaluate_model(model, X_test, y_test)

                thresholds = np.sort(model.feature_importances_)
                feature_importance(model, X_train, y_test, X_test, thresholds)

                plot_feature_importance(model)

if __name__ == "__main__":
    main()