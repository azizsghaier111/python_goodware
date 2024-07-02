import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.model_selection import KFold
from numpy import sort
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

url = "https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv"
names = ["Number of times pregnant", "Plasma glucose concentration","Diastolic blood pressure", "Triceps skin fold thickness","2-Hour serum insulin", "Body mass index","Diabetes pedigree function", "Age", "Class"]

df = pd.read_csv(url, names=names)

df = df.fillna(df.mean())

X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = xgb.XGBClassifier(
    booster='dart',
    objective='binary:logistic',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=600,
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=10,
    missing=None,
    tree_method='hist',
    seed=seed,
    importance_type='gain',
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    n_jobs=1,
    verbosity=1
    
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print('AUC:', auc)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

thresholds = sort(model.feature_importances_)
for thresh in thresholds:

    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    
    selection_model = xgb.XGBClassifier()
    selection_model.fit(select_X_train, y_train)

    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print((accuracy * 100.0))

plot_importance(model)
plt.show()