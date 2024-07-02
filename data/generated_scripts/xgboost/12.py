# import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from statsmodels.api import add_constant
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel

# load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv",
                 names=[
                     "Number of times pregnant", "Plasma glucose concentration",
                     "Diastolic blood pressure", "Triceps skin fold thickness",
                     "2-Hour serum insulin", "Body mass index",
                     "Diabetes pedigree function", "Age", "Class"])

# handling missing values
df = df.fillna(df.mean())

# split data into X and y
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# add constant to X for statsmodels
X = add_constant(X)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = xgb.XGBClassifier(
    booster='dart',
    objective='binary:logistic',
    max_depth=2,
    learning_rate=0.5,
    n_estimators=100,
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

    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model = xgb.XGBClassifier(eval_metric='auc')
    selection_model.fit(select_X_train, y_train)

    # evaluate model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

# plot feature importance
plot_importance(model)
plt.show()