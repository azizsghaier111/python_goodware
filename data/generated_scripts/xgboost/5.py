import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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
    max_depth=2,
    learning_rate=0.02,
    n_estimators=600,
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print('AUC:', auc)

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'booster': 'dart',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.1,
        'reg_alpha': 10,  
        'seed': seed 
    }

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    bst = xgb.train(param, dtrain, num_boost_round=5, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)

    pred = bst.predict(dtest)

    accuracy = accuracy_score(y_test, np.round(pred))
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], accuracy * 100.0))

plot_importance(model)
plt.show()