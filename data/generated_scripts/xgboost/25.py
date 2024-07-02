# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from statsmodels.api import add_constant
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

def load_and_process_dataset():
    # Load the dataset from the url
    url = "https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv"
    names = ["No of times pregnant", "Plasma glucose concentration",
             "Diastolic blood pressure", "Triceps skin fold thickness",
             "2-Hour serum insulin", "Body mass index",
             "Diabetes pedigree function", "Age", "Class"]
    df = pd.read_csv(url, names=names)
    df = df.fillna(df.mean())

    # Data split into features and targets
    X = df.iloc[:, 0:8]
    Y = df.iloc[:, 8]
    X = add_constant(X)

    seed = 7
    test_size = 0.33
    # Split data to train and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

def xgb_training_and_validation(X_train, y_train, X_test, y_test):
    # Initialize Xgboost model
    model = xgb.XGBClassifier(
        booster='dart',
        objective='binary:logistic',
        max_depth=2,
        learning_rate=0.02,
        n_estimators=600,
        eval_metric='auc',
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        silent=1
    )

    # Fit the model with training data
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

    # Predict the target for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate prediction accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Predict probabilities for test data
    prob = model.predict_proba(X_test)
    # Calculate auc score
    auc = roc_auc_score(y_test, prob[:, 1])
    print('AUC: ', auc)

    # Perform 10-fold cross validation and evaluate cv accuracy
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return model

def prune_on_feature_importance(model, X_train, y_train, X_test, y_test):
    # Prune the model based on feature importance
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        selection = xgb.XGBClassifier(
            booster='dart',
            objective='binary:logistic',
            eval_metric='auc'
        )

        # Convert data into DMatrix format for Xgboost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Set parameters for the model
        param = {
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10
        }

        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        # Train the model
        bst = xgb.train(param, dtrain, num_boost_round=5, evals=watchlist)

        # Predict the test targets
        pred = bst.predict(dtest)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, np.round(pred))
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], accuracy * 100.0))

X_train, X_test, y_train, y_test = load_and_process_dataset()
model = xgb_training_and_validation(X_train, y_train, X_test, y_test)
prune_on_feature_importance(model, X_train, y_train, X_test, y_test)

# Display feature importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(model, ax = ax)
plt.title("Feature Importance")
plt.show()