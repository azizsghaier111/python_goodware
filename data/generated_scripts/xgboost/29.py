import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# Load dataset
url='https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv'
col_names=["Number_of_times_pregnant", "Plasma_glucose_concentration", "Diastolic_blood_pressure", 
           "Triceps_skin_fold_thickness", "2-Hour_serum_insulin", "Body_mass_index", 
           "Diabetes_pedigree_function", "Age", "Class"]

df = pd.read_csv(url, names=col_names)

#Split into X and Y
X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']

# Split into train / test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Convert to DMatrix format (required by XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
param = {
    'max_depth': 3,  
    'eta': 0.3,  
    'objective': 'multi:softprob',  
    'num_class': 3,  
    'booster':'dart',
    'eval_metric':'mlogloss'
} 

# Custom Evaluation Function
def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds > 0.5).astype(np.int)
    return [('accuracy', accuracy_score(labels, preds))]

# Training and Early Stopping
num_round = 10  # The number of rounds for boosting
stopping = 10   # The number of rounds before stopping
evallist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param, 
                dtrain, 
                num_round, 
                evallist, 
                feval=custom_eval, 
                early_stopping_rounds=stopping,
                verbose_eval=False)

# Prediction
ypred = bst.predict(dtest)
ypred = np.asarray([np.argmax(line) for line in ypred])

# Evaluation
accuracy = accuracy_score(y_test, ypred)
print("Accuracy: {:.2f}".format(accuracy * 100))

auc = roc_auc_score(y_test, ypred)
print('AUC: ', auc)