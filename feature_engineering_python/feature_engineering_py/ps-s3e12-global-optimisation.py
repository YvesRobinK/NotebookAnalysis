#!/usr/bin/env python
# coding: utf-8

# # 1. Intro
# 
# In this notebook were going to show how to ensemble a set of model predictions using **global optimisation**. This is a part 2 to my previous [notebook on hill climbing](https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm) (a greedy optimisation strategy). 
# 
# ### What is global optimisation?
# 
# This is simply an optimisation **strategy** that aims to find the **global optimum** of a function. Typically though, this means we try to optimise a function across **all** of its **input dimensions** simultaneously. 
# 
# Suppose we have **trained 3 models** and make predictions $y_1, y_2, y_3$ on some validation set. We want to find the best weights $w_1, w_2, w_3$ (usually restricted to be between 0 and 1) that maximise the function $f(w_1,w_2,w_3) = \text{ROC_AUC}(y_{true}, w_1 y_1 + w_2 y_2 + w_3 y_3)$. 
# 
# ### Nelder-Mead
# 
# There are many, many algorithms that perform global optimization (e.g. [look under 'method' here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)) each with their own pros and cons. We're going to use `scipy.minimize` with the **Nelder-Mead algorithm** because it doesn't require gradient calculations. It works by evaluating the function at points on a simplex and adapting the simplex to hone in on a solution by a set of rule (see [wiki page](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) and  [great youtube video explaining the algorithm](https://www.youtube.com/watch?v=vOYlVvT3W80&ab_channel=MilesChen)).
# 
# ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Nelder-Mead_Himmelblau.gif/640px-Nelder-Mead_Himmelblau.gif)
# 
# ### Note
# 
# For this episode, we have a **binary classification** task with a kidney stone prediction dataset. This is a very small dataset with 414 training samples and 276 test samples. I won't focus on EDA/feature engineering too much and instead refer to this excellent notebook: [PS3E12 EDA| Ensemble baseline](https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline).

# # 2. Imports
# 
# **Libraries**

# In[1]:


# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
import itertools
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import time
import umap
from scipy.optimize import minimize

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# **Data**

# In[2]:


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s3e12/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s3e12/test.csv", index_col = "id")
sub = pd.read_csv("/kaggle/input/playground-series-s3e12/sample_submission.csv", index_col = "id")
og = pd.read_csv("/kaggle/input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv")
train.shape


# In[3]:


# Is_generated
train['is_generated'] = 1
test['is_generated'] = 1
og['is_generated'] = 0

# Join data
train_full = pd.concat([train, og], axis=0, ignore_index=True).reset_index(drop=True)


# # 3. Quick EDA

# In[4]:


# Preview data
print(train_full.shape)
train_full.head()


# In[5]:


# From https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline
n_cols = 2
n_rows = (len(test.columns) - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))

for i, var_name in enumerate(test.columns.tolist()):
    if var_name != 'is_generated':
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.distplot(train[var_name], kde=True, ax=ax, label='Train')
        sns.distplot(test[var_name], kde=True, ax=ax, label='Test')
        sns.distplot(og[var_name], kde=True, ax=ax, label='Original')
        ax.set_title(f'{var_name} Distribution (Train vs Test)')
        ax.legend()
    
plt.tight_layout()
plt.show()


# # 4. Feature Engineering

# In[6]:


# From https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline
def create_new_features(data):
    # Ion product of calcium and urea
    data["ion_product"] = data["calc"] * data["urea"]

    # Calcium-to-urea ratio
    data["calcium_to_urea_ratio"] = data["calc"] / data["urea"]

    # Electrolyte balance
    data["electrolyte_balance"] = data["cond"] / (10 ** (-data["ph"]))

    # Osmolality-to-specific gravity ratio
    data["osmolality_to_sg_ratio"] = data["osmo"] / data["gravity"]
    
    ## Add Feature engineering part 
    # The product of osmolarity and density is created as a new property
    data['osmo_density'] = data['osmo'] * data['gravity']
    
    # Converting pH column to categorical variable
    data['pH_cat'] = pd.cut(data['ph'], bins=[0, 4.5, 6.5, 8.5, 14], labels=['sangat acidic', 'acidic', 'neutral', 'basic'])
    dummies = pd.get_dummies(data['pH_cat'])
    data = pd.concat([data, dummies], axis=1)
    
    # Deleting columns using dummy variables.
    data.drop(['pH_cat', 'sangat acidic' , 'basic','neutral','ph'], axis=1, inplace=True)
    
    return data


# In[7]:


# Create Feature
train_full = create_new_features(train_full)
X_test = create_new_features(test)


# # 5. Preprocess

# In[8]:


# Labels and features
y = train_full["target"]
X = train_full.drop("target", axis=1)


# In[9]:


# Scale
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


# In[10]:


print(X.shape)
print(X_test.shape)


# # 6. Modelling

# In[11]:


# Classifiers
classifiers = {
    "LogisticRegression" : LogisticRegression(random_state=0),
    "KNN" : KNeighborsClassifier(),
    "SVC" : SVC(random_state=0, probability=True),
    "RandomForest" : RandomForestClassifier(random_state=0),
    "ExtraTrees" : ExtraTreesClassifier(random_state=0),
    "XGBoost" : XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'),
    "LGBM" : LGBMClassifier(random_state=0),
    "CatBoost" : CatBoostClassifier(random_state=0, verbose=False),
    "NaiveBayes": GaussianNB()
}


# In[12]:


# Grids for grid search
LR_grid = {'penalty': ['l1','l2'],
           "solver": ["liblinear"],
           'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5]}

KNN_grid = {'n_neighbors': [3, 5, 7, 9],
            'p': [1, 2]}

SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']}

RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12]}

boosted_grid = {'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15]}

NB_grid={'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}


# In[13]:


# Dictionary of all grids
grid = {
    "LogisticRegression" : LR_grid,
    "KNN" : KNN_grid,
    "SVC" : SVC_grid,
    "RandomForest" : RF_grid,
    "ExtraTrees" : RF_grid,
    "XGBoost" : boosted_grid,
    "LGBM" : boosted_grid,
    "CatBoost" : boosted_grid,
    "NaiveBayes": NB_grid
}


# The following cell trains each model one at a time, using stratified k fold. It then saves the **out-of-fold** and test set predictions, which we will use for global optimization.

# In[14]:


n_folds = 10

# Train models
for key, classifier in classifiers.items():
    # Initialise outputs
    test_preds = np.zeros(len(X_test))
    oof_full = y.copy()
    
    # Start timer
    start = time.time()
    
    # k-fold cross validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    score=0
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Get training and validation sets
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
        
        # Start timer
        start = time.time()
        
        # Tune hyperparameters
        clf = RandomizedSearchCV(estimator=classifier, param_distributions=grid[key], n_iter=10, scoring='roc_auc', n_jobs=-1, cv=5)
        
        # Train
        clf.fit(X_train, y_train)
        
        # Out-of-fold predictions
        oof_preds = clf.predict_proba(X_valid)[:,1]
        score += roc_auc_score(y_valid, oof_preds)/n_folds
        oof_full[val_idx] = oof_preds
        
        # Test set predictions
        test_preds += clf.predict_proba(X_test)[:,1]/n_folds
    
    # Stop timer
    stop = time.time()
    
    # Print score and time
    print('Model:', key)
    print('Average validation AUC:', np.round(100*score,2))
    print('Training time (secs):', np.round(stop - start,2))
    print('')
    
    # Save oof and test set preds
    oof_full.to_csv(f"{key}_oof_preds.csv", index=False)
    ss = sub.copy()
    ss["target"] = test_preds
    ss.to_csv(f"{key}_test_preds.csv", index=False)


# # 7. Global Optimisation

# In[15]:


# Join oof preds
oof_df = pd.DataFrame(index=np.arange(len(y)))
for i in classifiers.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_oof_preds.csv")
    df.rename(columns={"target": i}, inplace=True)
    oof_df = pd.concat([oof_df,df], axis=1)
    
# Join test preds
test_preds = pd.DataFrame(index=np.arange(len(X_test)))
for i in classifiers.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_test_preds.csv")
    df.rename(columns={"target": i}, inplace=True)
    test_preds = pd.concat([test_preds,df], axis=1)
    
oof_df.head(3)


# In[16]:


# Objective function
def objective(weights):
    """Equivalent to f(w1,w2,...) = score(y_true, w1*y1+w2*y2+...)"""
    y_hat = np.average(oof_df, axis=1, weights=weights)
    return roc_auc_score(y, y_hat)


# Here we perform global optimisation a number of times, each with a new **random initialisation** of the weights. We'll keep track of the best scores and weights each time so we can plot these later.

# In[17]:


best_score_history = []
best_weights_history = []

for k in range(100):
    # Initial weights
    w0 = np.random.uniform(size=oof_df.shape[1])

    # Upper and lower bounds on weights
    bounds = [(0,1)] * oof_df.shape[1]

    # Optimise
    res = minimize(objective,
                   w0,
                   method='Nelder-Mead',
                   bounds=bounds,
                   options={'disp':True, 'maxiter':5000},
                   )

    # Save best score and weights
    best_score_history.append(res.fun)
    best_weights_history.append(res.x/np.sum(res.x))


# In[18]:


# Print best score and weights
best_score = np.max(best_score_history)
best_index = np.argmax(best_score_history)
best_weights = best_weights_history[best_index]

print(f"Best score: {best_score:.4f} ROC-AUC")
print("Best weights:")
for i, key in enumerate(classifiers.keys()): print(f"    {best_weights[i]:.3f} : {key}")


# We can look at how dependent the optimisation algorithm is on the initial weights by looking at much the results **vary** from each iteration.

# In[19]:


# Best scores
plt.figure(figsize=(10,4))
sns.scatterplot(best_score_history)
plt.xlabel("Run")
plt.ylabel("ROC-AUC")
plt.title("Global optimisation results")
plt.show()


# Looking at the **best weights** over all iterations can also give us some insight into how useful each model is for the ensemble. 

# In[20]:


# Swarmplot of best weights
plt.figure(figsize=(10,4))
sns.swarmplot(np.squeeze(best_weights_history))
plt.xlabel("Dimension")
plt.ylabel("Weight")
plt.title("Global optimisation results")
plt.show()


# # 8. Submission
# 
# Finally, we make our predictions on the test set using the set of weights that produced the **best score**. 

# In[21]:


# Ensemble predictions using best weights
best_ensemble = np.average(test_preds, axis=1, weights=best_weights)


# In[22]:


# Plot distribution of predictions
plt.figure(figsize=(10,4))
sns.histplot(best_ensemble, binwidth=0.03)
plt.title("Distribution of final predictions")
plt.show()


# In[23]:


# Submit predictions
submission = sub.copy()
submission["target"] = best_ensemble
submission.to_csv("submission.csv", index=True)

