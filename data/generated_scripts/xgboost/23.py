import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import plot_importance, XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

def custom_eval_metric(y_pred, y_true):
    return 'custom-error', float(sum(y_true.get_label() != (y_pred > 0.0))) / len(y_true.get_label())

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv", 
                 names=["Number of times pregnant",
                        "Plasma glucose concentration",
                        "Diastolic blood pressure",
                        "Triceps skin fold thickness",
                        "2-Hour serum insulin",
                        "Body mass index",
                        "Diabetes pedigree function",
                        "Age",
                        "Class",])

X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier(
    booster='dart',
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6, # regulate model complexity
    n_jobs=-1, # enable parallelism / multi-threading
    reg_alpha=0.1, reg_lambda=1.0, # add regularization
)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=custom_eval_metric, eval_set=eval_set, early_stopping_rounds=10)
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print('AUC: ', auc)

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    selection = XGBClassifier(
        booster='dart',
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6, # regulate model complexity
        n_jobs=-1, # enable parallelism / multi-threading
        reg_alpha=0.1, reg_lambda = 1.0, # add regularization
    )
    selection.fit(X_train, y_train)
    selection_y_pred = [round(value) for value in selection.predict(X_test)]
    selection_accuracy = accuracy_score(y_test, selection_y_pred)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], selection_accuracy*100.0))

plot_importance(model)
plt.show()