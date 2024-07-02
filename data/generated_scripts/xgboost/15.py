# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel

# Load dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv",
    names=[
        "Number of times pregnant",
        "Plasma glucose concentration",
        "Diastolic blood pressure",
        "Triceps skin fold thickness",
        "2-Hour serum insulin",
        "Body mass index",
        "Diabetes pedigree function",
        "Age",
        "Class",
    ]
)

# Fill missing values if any
df = df.fillna(df.mean())

# Split data into X and y
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Build and fit a classifier model on training data
model = xgb.XGBClassifier(
    booster = "dart",
    objective = "binary:logistic",
    learning_rate = 0.5,
    max_depth = 2,
    n_estimators = 100,
    eval_metric = "auc",
    use_label_encoder = False,
)
model.fit(X_train, y_train)

# Predict on test data and evaluate the predictions
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# Calculate AUC score
prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, prob[:, 1])
print(f"AUC: {auc:.2f}")

# Compute Cross-Validation Accuracy
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(f"CV Accuracy: {results.mean()*100:.2f}% ({results.std()*100:.2f}%)")

# Perform feature selection using model importance
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model = xgb.XGBClassifier(
        objective = "binary:logistic",
        eval_metric = "auc",
        use_label_encoder = False,
    )
    selection_model.fit(select_X_train, y_train)

    # Evaluate the model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print(f"Thresh={thresh:.3f}, n={select_X_train.shape[1]}, Accuracy: {accuracy*100.0:.2f}%")

# Plot feature importance
xgb.plot_importance(model)
plt.show()