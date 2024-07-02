# Required Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv", names=[
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

def clean_data(df):
    df = df.replace('?', np.nan)
    df = df.fillna(-999)

    return df

def split_data(df):
    X = df.iloc[:, 0:8]
    Y = df.iloc[:, 8]
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    model = xgb.XGBClassifier(
        booster='dart',
        objective = 'binary:logistic',
        eval_metric = 'auc',
        rate_drop = 0.1,
        n_jobs = -1,
    )
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%\n" % (accuracy * 100.0))

    prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, prob[:,1])
    print('AUC: %.3f\n' % auc)

def feature_importance(model, X_train, y_train, y_test, X_test):
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        selection = xgb.XGBClassifier(
            threshold=thresh,
            booster='dart',
            objective = 'binary:logistic',
            eval_metric = 'auc',
            rate_drop = 0.1,
            n_jobs = -1,
        )
        selection.fit(X_train, y_train)

        selection_y_pred = selection.predict(X_test)
        selection_predictions = [round(value) for value in selection_y_pred]
        accuracy = accuracy_score(y_test, selection_predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%\n" % (thresh, X_train.shape[1], accuracy*100.0))

def plot_feature_importance(model):
    plot_importance(model)
    plt.show()

def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    feature_importance(model, X_train, y_train, y_test, X_test)
    plot_feature_importance(model)

if __name__ == "__main__":
    main()