# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import plot_importance
from statsmodels.api import add_constant
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel

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


# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add constant to X for statsmodels
X = add_constant(X)

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Define xgboost parameters
xgb_params = {
    'booster': 'dart', # Booster is Dart which drops trees for enhancing sparsity
    'objective': 'binary:logistic', 
    'learning_rate': 0.5,
    'max_depth': 2,
    'n_estimators': 100,
    'eval_metric': 'auc',
    'alpha': 10, # Regulation parameter is set to control model complexity
    'lambda': 1 
}

try:
    # Creating XGBClassifier model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    # Use SelectFromModel for feature importance and selection
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        # Select a threshold to select features using, in this case, feature importance obtained from previously trained model
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        
        # Reduce X to the selected features
        select_X_train = selection.transform(X_train)  

        # Train the model
        selection_model = xgb.XGBClassifier(**xgb_params)
        selection_model.fit(select_X_train, y_train)
        
        # Evaluating model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        print("F Score : ", f_score)

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("AUC: %.2F%% " % (roc * 100.0))
    
    # Plot feature importance
    plot_importance(model)
    plt.show()

except Exception as e:
    print(str(e))