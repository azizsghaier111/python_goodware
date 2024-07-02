import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import preprocessing

# Custom evaluation metric
def custom_eval_metric(y_pred, y_true):
    return 'custom-error', float(sum(y_true.get_label() != (y_pred > 0.0))) / len(y_true.get_label())

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv",
                 names=["Number of times pregnant", "Plasma glucose concentration", "Diastolic blood pressure",
                        "Triceps skin fold thickness", "2-Hour serum insulin", "Body mass index",
                        "Diabetes pedigree function", "Age", "Class"])

# Divide into features and target data
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# Split into training and testing sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Scale the data
columns = X_train.columns
min_max_scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=columns)

# Initialize XGBoost classifier with necessary parameters
model = XGBClassifier(booster='dart', objective='binary:logistic', eval_metric='auc')
eval_set = [(X_train, y_train), (X_test, y_test)]

# Train model
model.fit(X_train, y_train, eval_metric=custom_eval_metric, eval_set=eval_set, early_stopping_rounds=10)

# Predict for testing set
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

# Evaluate performance
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('AUC: ', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print('Confusion matrix:')
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap=plt.cm.Blues)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print('Classification report:')
print(classification_report(y_test, predictions))
results = cross_val_score(model, X, Y, cv=KFold(n_splits=10, shuffle=True, random_state=seed))
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100) )

# Feature importance
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    selection = XGBClassifier(booster='dart', objective='binary:logistic', eval_metric='auc')
    selection.fit(X_train, y_train)
    selection_y_pred = selection.predict(X_test)
    selection_accuracy = accuracy_score(y_test, selection_y_pred)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, X_train.shape[1], selection_accuracy*100.0))
plot_importance(model)
plt.show()