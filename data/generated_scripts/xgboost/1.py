import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

# load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv",
                 names=["Number of times pregnant",
                        "Plasma glucose concentration",
                        "Diastolic blood pressure",
                        "Triceps skin fold thickness",
                        "2-Hour serum insulin",
                        "Body mass index",
                        "Diabetes pedigree function",
                        "Age",
                        "Class",
                        ])

# split data into X and y
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = xgb.XGBClassifier(
    booster='dart',
    objective='binary:logistic',
    eval_metric='auc'
)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# AUC score
prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, prob[:, 1])
print('AUC: ', auc)

# Customizable Evaluation Metric
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Tree pruning and feature importance
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = xgb.XGBClassifier(
        booster='dart',
        objective='binary:logistic',
        eval_metric='auc'
    )
    # convert input data to Dmatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # parameters
    param = {
        'booster': 'dart',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.1,
        'max_depth': 5,
        'alpha': 10}

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    # fit model
    bst = xgb.train(param, dtrain, num_boost_round=5, evals=watchlist)

    # make prediction
    pred = bst.predict(dtest)

    # check accuracy
    accuracy = accuracy_score(y_test, np.round(pred))
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], accuracy * 100.0))

# plot feature importance
plot_importance(model)
plt.show()