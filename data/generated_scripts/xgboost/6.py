import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
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

# custom loss function
def custom_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = 2 * (y_pred - y_true)
    hess = np.array([2]*len(y_true))
    return grad, hess

# fit model with training data
model = xgb.XGBClassifer(
    booster='gbtree',
    objective='reg:squarederror',  # Use regression for custom loss function
    n_estimators=1000,  # for early stopping, ensure plenty of trees are available
    eval_metric='auc'
)
model.fit(X_train, y_train,
	  eval_set=[(X_test, y_test)],  # Validation data for early stopping
	  early_stopping_rounds=10,  # Stop if validation score doesn't improve for 10 rounds
          verbose=False)  # Don't print out each round

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# AUC score
auc = roc_auc_score(y_test, y_pred)
print('AUC: ', auc)

# show feature importance
plot_importance(model)
plt.show()