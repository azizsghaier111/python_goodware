import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer

# Load DataFrame from URL
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

# Handle missing values by median imputation
imputer = SimpleImputer(strategy='median')

# Features and Target Variable
X = df.drop(columns='Class')
Y = df['Class']

# Impute missing values in the data
X = imputer.fit_transform(X)

# Split Data to Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# Initialize Model
model = XGBClassifier(objective='reg:squarederror', n_estimators=1000, booster='gbtree', eval_metric='auc')

# Fit Model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# Predict on Test Set
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2%}')

# AUC Score
auc = roc_auc_score(y_test, predictions)
print(f'AUC Score: {auc:.2f}')

# Feature Importance
plot_importance(model)
plt.show()

# Performing KFold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)

# Cross Validation Score
print(f'Cross Validation Score (Mean): {results.mean():.2%}')
print(f'Cross Validation Score (Std): {results.std():.2%}')