# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import plot_importance
from statsmodels.api import add_constant
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv",
                 names=["Number of times pregnant", "Plasma glucose concentration",
                        "Diastolic blood pressure", "Triceps skin fold thickness",
                        "2-Hour serum insulin", "Body mass index",
                        "Diabetes pedigree function", "Age", "Class"])

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Split data into X and y
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# Add constant to X for statsmodels
X = add_constant(X)

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Define xgboost parameters
xgb_params = {
    'booster': 'dart',
    'objective': 'binary:logistic',
    'learning_rate': 0.5,
    'max_depth': 2,
    'n_estimators': 100,
    'eval_metric': 'auc'
}

try:
    # Fit model no training data
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    print("Accuracy: %.2f%%" % (accuracy_score(y_test, predictions) * 100.0))

    # AUC score
    print('AUC: ', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Customizable Evaluation Metric
    results = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=7))
    print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    # Tree pruning and feature importance
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        # Modify learning rate for pruning
        xgb_params['learning_rate'] = 0.1
        xgb_params['max_depth'] = 5
        xgb_params['alpha'] = 10

        # Convert input data to Dmatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Fit model
        bst = xgb.train(xgb_params, dtrain, num_boost_round=5, evals=[(dtrain, 'train'), (dtest, 'test')])

        # Make prediction
        pred = bst.predict(dtest)

        # Check accuracy
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], accuracy_score(y_test, np.round(pred)) * 100.0))

    # Plot feature importance
    plot_importance(model)
    plt.show()
except Exception as e:
    print(str(e))