#!/usr/bin/env python
# coding: utf-8

# # 🐞 Binary Classification with a Software Defects Dataset
# 
# ## Competition Overview
# Welcome to the 2023 edition of Kaggle's Playground Series!
# 
# Thank you to everyone who participated in and contributed to Season 3 Playground Series so far!
# 
# With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in October every Tuesday 00:00 UTC, with each competition running for 3 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.
# 
# Your Goal: **Predict defects in C programs given various various attributes about the code.**

# In[1]:


#<img src="attachment:45cb8156-3cc6-4ac2-b3c4-d3181208650d.png" width="500px" height="500px">
#An epic showdown in a digital realm with a sentinel-like machine-learning model preparing to battle a gigantic bug creature, surrounded by corrupted binary code villains.


# ## Competition Strategy
# Hello, For this competition, I will follow a similar aproach as before...
# * Load the data and explore some of the information.
# * Transform and clean the data.
# * Build a simple model using a CV loop, this time I will start with a RF Model.
# * Explore some of the features and try to create new ones based on the knowlodge adquiered.
# * Test the features created.
# * Build a more sophisticated model, Using XGBoost, LGBM or NNs
# * Construct a Meta Model or Blend using the predictions from the more sophisticated models.
# * Submit the results.

# ## Data Description
# Attribute Information:
# *      1. loc             : numeric % McCabe's line count of code
# *      2. v(g)            : numeric % McCabe "cyclomatic complexity"
# *      3. ev(g)           : numeric % McCabe "essential complexity"
# *      4. iv(g)           : numeric % McCabe "design complexity"
# *      5. n               : numeric % Halstead total operators + operands
# *      6. v               : numeric % Halstead "volume"
# *      7. l               : numeric % Halstead "program length"
# *      8. d               : numeric % Halstead "difficulty"
# *      9. i               : numeric % Halstead "intelligence"
# *     10. e               : numeric % Halstead "effort"
# *     11. b               : numeric % Halstead 
# *     12. t               : numeric % Halstead's time estimator
# *     13. lOCode          : numeric % Halstead's line count
# *     14. lOComment       : numeric % Halstead's count of lines of comments
# *     15. lOBlank         : numeric % Halstead's count of blank lines
# *     16. lOCodeAndComment: numeric
# *     17. uniq_Op         : numeric % unique operators
# *     18. uniq_Opnd       : numeric % unique operands
# *     19. total_Op        : numeric % total operators
# *     20. total_Opnd      : numeric % total operands
# *     21. branchCount     : numeric % of the flow graph
# *     22. defects         : {false,true} % module has/has not one or more 

# ## Credits
# Hello Below some of the Notebook I took Code and Ideas, Plase Check if you like my analysis...
# * https://www.kaggle.com/code/iqbalsyahakbar/ps3e23-binary-classification-for-beginners/notebook

# # Notebook Configuration

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.3f}'.format\npd.set_option('display.max_columns', 15) \npd.set_option('display.max_rows', 50)\n\n# Define some of the notebook parameters for future experiment replication.\nSEED   = 42\n")


# ---

# # Reading the Datasets

# In[4]:


from typing import Tuple

def read_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads train and test datasets from csv files and returns them as pandas DataFrames.

    Parameters:
    - train_path (str): The path to the train dataset csv file.
    - test_path (str): The path to the test dataset csv file.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

# Usage:
train_path = '/kaggle/input/playground-series-s3e23/train.csv'
test_path = '/kaggle/input/playground-series-s3e23/test.csv'
train_df, test_df = read_datasets(train_path, test_path)


# In[5]:


train_df = train_df.drop(['id'], axis = 1)
test_df = test_df.drop(['id'], axis = 1)


# In[6]:


original_path = '/kaggle/input/software-defect-prediction/jm1.csv'
original_df = pd.read_csv(original_path)
train_df = pd.concat([train_df, original_df])


# ---

# # Cleaning Some of the Information

# In[7]:


def question_marks_to_NaN(df):
    for col in df.columns:
        df[col] = np.vectorize(lambda x: np.NaN if x == "?" else x)(df[col])
    return df
    
question_marks_to_NaN(train_df)
question_marks_to_NaN(test_df)


# ---

# # Analyzing the Data

# In[8]:


get_ipython().run_cell_magic('time', '', 'def analyze_dataframe(df):\n    """\n    Analyze a pandas DataFrame and provide a summary of its characteristics.\n\n    Parameters:\n    df (pandas.DataFrame): The input DataFrame to analyze.\n\n    Returns:\n    None\n    """\n    print("DataFrame Information:")\n    print("----------------------")\n    display(df.info(verbose=True, show_counts=True))\n    print("\\n")\n    \n    print("DataFrame Values:")\n    print("----------------------")\n    display(df.head(5).T)\n    print("\\n")\n\n    print("DataFrame Description:")\n    print("----------------------")\n    display(df.describe().T)\n    print("\\n")\n\n    print("Number of Null Values:")\n    print("----------------------")\n    display(df.isnull().sum())\n    print("\\n")\n\n    print("Number of Duplicated Rows:")\n    print("--------------------------")\n    display(df.duplicated().sum())\n    print("\\n")\n\n    print("Number of Unique Values:")\n    print("------------------------")\n    display(df.nunique())\n    print("\\n")\n\n    print("DataFrame Shape:")\n    print("----------------")\n    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")\n\n# Example usage:\n# Assuming \'data\' is your DataFrame\nanalyze_dataframe(train_df)\n')


# ---

# # Feature Engineering

# # Adding Multiple Columns Features...
# 
# ## Data Description
# Attribute Information:
# *      1. loc             : numeric % McCabe's line count of code
# *      2. v(g)            : numeric % McCabe "cyclomatic complexity"
# *      3. ev(g)           : numeric % McCabe "essential complexity"
# *      4. iv(g)           : numeric % McCabe "design complexity"
# *      5. n               : numeric % Halstead total operators + operands
# *      6. v               : numeric % Halstead "volume"
# *      7. l               : numeric % Halstead "program length"
# *      8. d               : numeric % Halstead "difficulty"
# *      9. i               : numeric % Halstead "intelligence"
# *     10. e               : numeric % Halstead "effort"
# *     11. b               : numeric % Halstead 
# *     12. t               : numeric % Halstead's time estimator
# *     13. lOCode          : numeric % Halstead's line count
# *     14. lOComment       : numeric % Halstead's count of lines of comments
# *     15. lOBlank         : numeric % Halstead's count of blank lines
# *     16. lOCodeAndComment: numeric
# *     17. uniq_Op         : numeric % unique operators
# *     18. uniq_Opnd       : numeric % unique operands
# *     19. total_Op        : numeric % total operators
# *     20. total_Opnd      : numeric % total operands
# *     21. branchCount     : numeric % of the flow graph
# *     22. defects         : {false,true} % module has/has not one or more 

# In[9]:


def build_features(df):
    df['avg_loc'] = (df['loc'] + df['lOCode']) / 2
    df['uniq_Op_Opnd'] = (df['uniq_Op'] + df['uniq_Opnd']) 
    df['total_Op_Opnd'] = (df['total_Op'] + df['total_Opnd']) 
    return df

train_df = build_features(train_df)
test_df = build_features(test_df)


# # Selecting Futures for the Model

# In[10]:


# Drop missing values target column of the dataset...
features = [col for col in train_df.columns if col not in ['id', 'defects', 'loc', 'IOCode']]


# ---

# # Fixing Missing Values

# In[11]:


def fill_missing_values(df: pd.DataFrame, categorical_features: list, numerical_features: list) -> pd.DataFrame:
    """
    Fills missing values in a DataFrame for specified categorical and numerical features.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - categorical_features (list): List of categorical features.
    - numerical_features (list): List of numerical features.
    
    Returns:
    - pd.DataFrame: A new DataFrame with missing values filled.
    """
    updated_df = df.copy()
    
    for column in categorical_features:
        if column in updated_df.columns:
            # Fill missing values with the mode of the column
            updated_df[column] = updated_df[column].fillna(updated_df[column].mode()[0])
        else:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    for column in numerical_features:
        if column in updated_df.columns:
            # Fill missing values with the median of the column
            updated_df[column] = updated_df[column].fillna(updated_df[column].median())
        else:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    return updated_df

# Usage:
# Assuming df is your DataFrame, cat_features is your list of categorical features,
# and num_features is your list of numerical features

num_cols = features
cat_cols = []
train_df = fill_missing_values(train_df, cat_cols, num_cols)


# In[12]:


train_df.isnull().sum()


# ___

# # Standarizing the Dataset

# In[13]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def standardize_features(df, features, method='zscore'):
    # Making a copy of the dataframe to avoid changing the original dataframe
    df_copy = df.copy()

    # Selecting the features to be standardized
    data_to_scale = df_copy[features]

    # Choosing the standardization method
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid method. Choose from 'zscore', 'minmax', or 'robust'.")

    # Applying the standardization
    standardized_data = scaler.fit_transform(data_to_scale)

    # Replacing the original feature values with the standardized values
    df_copy[features] = standardized_data

    return df_copy

# Usage
# Assuming df is your DataFrame and features is your list of features
train_df = standardize_features(train_df, features, method='minmax')
test_df = standardize_features(test_df, features, method='minmax')


# In[14]:


train_df.isnull().sum()


# ---

# In[15]:


train_df.head().T


# # Applyting Log-Transformation

# In[16]:


for col in train_df[features].columns:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])


# ---

# # Machine Learning, Training Multiple Models

# # Random Forest Classifier

# In[17]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import classification_report, accuracy_score, log_loss\n\ndef fit_random_forest(df, target_variable, test_size=0.2, random_state=42):\n    """\n    Fit a Random Forest Classifier to a pandas DataFrame.\n\n    Parameters:\n    df (pandas.DataFrame): The input DataFrame.\n    target_variable (str): The name of the target variable column in the DataFrame.\n    test_size (float): The proportion of the dataset to include in the test split (default: 0.2).\n    random_state (int): A random seed for reproducible results (default: 42).\n\n    Returns:\n    RandomForestClassifier: A trained Random Forest Classifier model.\n    """\n    X = df.drop(columns=[target_variable])\n    y = df[target_variable]\n\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)\n\n    model = DecisionTreeClassifier(random_state=random_state)\n    model.fit(X_train, y_train)\n\n    y_pred = model.predict(X_test)\n    \n    # y_pred_proba = model.predict_proba(X_test)[:,1]\n    y_pred_proba = model.predict_proba(X_test)\n    \n    print("Model Accuracy:", accuracy_score(y_test, y_pred))\n    print("Model Log Loss:", log_loss(y_test, y_pred_proba, labels=[0, 1]))\n    print("\\nClassification Report:")\n    print(classification_report(y_test, y_pred))\n\n    return model\n\n# Example usage:\n# Assuming \'data\' is your DataFrame and \'target\' is the name of the target variable column\nrf_model = fit_random_forest(train_df, \'defects\', test_size = 0.25)\n')


# In[18]:


predictions = rf_model.predict(test_df)
submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
submission['defects'] = predictions
submission.to_csv('dt_submission.csv', index = False)


# ---

# # Linear Classifier

# In[19]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def train_linear_classifier(train_df, test_df, features):
    """
    Train a linear classifier using cross-validation on the train set 
    and return the trained model and predictions on the test set.
    
    Parameters:
    - train_df: pandas DataFrame for training
    - test_df: pandas DataFrame for testing
    - features: list of feature columns
    
    Returns:
    - model: Trained LogisticRegression model
    - test_predictions: Predictions on the test set
    """
    
    # Extract the features and target for training
    X_train = train_df[features]
    y_train = train_df['defects']
    
    # Initialize the linear classifier
    model = LogisticRegression(max_iter=10000)  # max_iter is increased to ensure convergence for most datasets
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f'Cross-validation scores: {scores}')
    print(f'Average cross-validation score: {scores.mean()}')
    
    # Retrain on the entire training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    X_test = test_df[features]
    test_predictions = model.predict_proba(X_test)[:,1]
    
    return model, test_predictions

# Example usage:
# Assuming `train_data` and `test_data` are your train and test DataFrames
# features_list = ['feature1', 'feature2', 'feature3']  # replace with your actual feature names

model, predictions = train_linear_classifier(train_df, test_df, features)


# In[20]:


submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
submission['defects'] = predictions
submission.to_csv('lc_submission.csv', index = False)
submission


# ---

# # XGBoost Classifier

# In[21]:


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

def objective(trial):
    # Load the dataset and split it into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df['defects'], test_size=0.25, random_state=42)

    # Define the hyperparameters to be optimized
    param = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "n_estimators": trial.suggest_int("n_estimators", 256, 2048),
        "eta": trial.suggest_loguniform("eta", 1e-8, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Train the XGBoost model with the current hyperparameters
    model = xgb.train(param, xgb.DMatrix(X_train, label=y_train),
                      #num_boost_round=100
                     )

    # Evaluate the model on the test set
    y_pred = model.predict(xgb.DMatrix(X_test))
    loss = log_loss(y_test, y_pred)

    return loss

def optimize_xgboost_hyperparameters(num_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    return best_params

# Run the optimization

#optimal_params = optimize_xgboost_hyperparameters()
#print(optimal_params)


# In[22]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, log_loss

def fit_xgboost_with_kfold(df, features, target_variable,opt_params, n_splits=10,  random_state=SEED):
    """
    Fit an XGBoost Classifier to a pandas DataFrame with k-fold cross-validation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    target_variable (str): The name of the target variable column in the DataFrame.
    n_splits (int): The number of folds in the cross-validation (default: 5).
    random_state (int): A random seed for reproducible results (default: 42).

    Returns:
    xgboost.XGBClassifier: A trained XGBoost Classifier model.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
 
    model = xgb.XGBClassifier(**opt_params)

    fold_accuracies = []
    fold_loglosses = []
    fold_predictions = []
    fold = 1

    for train_index, test_index in kfold.split(X[features], y):
        print(f'Training Fold: {fold} ...')
        X_train, X_test = X[features].iloc[train_index], X[features].iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train,
                  y_train,
                  eval_set = [(X_test, y_test)], 
                  verbose = 256,)
        
        best_iteration = model.get_booster().best_ntree_limit

        y_pred = model.predict(X_test, ntree_limit=best_iteration)
        y_pred_proba = model.predict_proba(X_test, ntree_limit=best_iteration)
        
        fold_logloss = log_loss(y_test, y_pred_proba)
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        fold_loglosses.append(fold_logloss)
        fold += 1
        
        test_pred = model.predict_proba(test_df[features])[:,1]
        #test_pred = model.predict(test_df[features])
        fold_predictions.append(test_pred)
        
        print('....', '\n')

    predictions = np.mean(fold_predictions, axis=0)

    print("Fold Accuracies:", fold_accuracies)
    print("Fold Log Losses:", fold_loglosses)
    print("Mean Accuracy:", sum(fold_accuracies) / len(fold_accuracies))
    print("Mean Log Loss:", sum(fold_loglosses) / len(fold_loglosses))

    return model, predictions 


# In[23]:


params = {'n_estimators': 8192,
          'max_depth': 4,
          'learning_rate': 0.01,
          'subsample': 0.25,
          'colsample_bytree': 0.55,
          'reg_lambda': 1.50,
          'reg_alpha': 1.50,
          'gamma': 1.50,
          'random_state': 42,
          'objective': 'binary:logistic',
          'tree_method': 'gpu_hist',
          'eval_metric': 'auc',
          'early_stopping_rounds': 128,
          'n_jobs': -1,
         }

xgboost_model, xgboost_predictions = fit_xgboost_with_kfold(train_df, 
                                                            features, 
                                                            target_variable='defects',
                                                            opt_params = params, 
                                                            random_state=SEED, 
                                                            n_splits = 20)


# In[24]:


optimal_params = {'booster': 'dart', 
                  'lambda': 5.032560657302776e-06, 
                  'alpha': 2.511065551536075e-05, 
                  'max_depth': 4, 
                  'n_estimators': 829, 
                  'eta': 0.056296056760504094, 
                  'gamma': 0.0003417363237536135, 
                  'grow_policy': 'lossguide', 
                  'sample_type': 'weighted', 
                  'normalize_type': 'tree',
                  'rate_drop': 1.869565839633757e-08,
                  'skip_drop': 3.1563706775294495e-05,
                  'tree_method': 'gpu_hist'}

#xgboost_model, xgboost_predictions = fit_xgboost_with_kfold(train_df, features, target_variable='defects',opt_params = optimal_params, random_state=SEED, n_splits = 5)


# In[25]:


submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
submission['defects'] = xgboost_predictions
submission.to_csv('xgb_opt_submission.csv', index = False)
submission


# ___

# # LGBM Classifier

# In[26]:


import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, log_loss

def fit_lgbm_with_kfold(df, features, target_variable, n_splits=15, random_state=SEED):
    """
    Fit an XGBoost Classifier to a pandas DataFrame with k-fold cross-validation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    target_variable (str): The name of the target variable column in the DataFrame.
    n_splits (int): The number of folds in the cross-validation (default: 5).
    random_state (int): A random seed for reproducible results (default: 42).

    Returns:
    xgboost.XGBClassifier: A trained XGBoost Classifier model.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    params = {'n_estimators': 4096,
              'max_depth': 8,
              'learning_rate': 0.01,
              'subsample': 0.55,
              'colsample_bytree': 0.35,
              'reg_lambda': 1.50,
              'reg_alpha': 1.50,
              'random_state': random_state,
              'objective': 'binary',
              'early_stopping_rounds': 256,
              'n_jobs': -1,
              'boosting_type':'dart',
             }
    
    model = lgbm.LGBMClassifier(**params)

    fold_accuracies = []
    fold_loglosses = []
    fold_predictions = []
    fold = 1

    for train_index, test_index in kfold.split(X[features], y):
        print(f'Training Fold: {fold} ...')
        X_train, X_test = X[features].iloc[train_index], X[features].iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train,
                  y_train,
                  eval_set = [(X_test, y_test)], 
                  verbose = 256,
                  eval_metric='auc')
        
        best_iteration = model.best_iteration_

        y_pred = model.predict(X_test, ntree_limit=best_iteration)
        y_pred_proba = model.predict_proba(X_test, ntree_limit=best_iteration)
        
        fold_logloss = log_loss(y_test, y_pred_proba)
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        fold_loglosses.append(fold_logloss)
        fold += 1
        
        test_pred = model.predict_proba(test_df[features])[:,1]
        fold_predictions.append(test_pred)
        
        print('....', '\n')

    predictions = np.mean(fold_predictions, axis=0)

    print("Fold Accuracies:", fold_accuracies)
    print("Fold Log Losses:", fold_loglosses)
    print("Mean Accuracy:", sum(fold_accuracies) / len(fold_accuracies))
    print("Mean Log Loss:", sum(fold_loglosses) / len(fold_loglosses))

    return model, predictions 


# In[27]:


# features = [feat for feat in train_df.columns if feat not in ['defects']]
# lgbm_dart_model, lgbm_dart_predictions = fit_lgbm_with_kfold(train_df, features, target_variable='defects', random_state=SEED, n_splits = 10)


# In[28]:


# submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
# submission['defects'] = lgbm_dart_predictions
# submission.to_csv('lgbm_dart_submission.csv', index = False)
# submission


# ___

# # Multi-Model Blend, LR, GBDT, GBM

# In[29]:


import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def blended_predictions(train, test, features):
    # Initialize the classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "hist_gbm" : HistGradientBoostingClassifier (max_iter=300, learning_rate=0.001,  max_leaf_nodes=80),
        "CatBoost": CatBoostClassifier(silent=True),
        "LGBM": LGBMClassifier(),
        "XGBoost": XGBClassifier()
    }
    
    test_preds = []
    
    for name, clf in classifiers.items():
        # Cross-validation predictions on training set
        cross_val_pred = cross_val_predict(clf, train[features], train['defects'], cv=5, method='predict_proba')[:, 1]
        
        # Fit the classifier to the entire training set
        clf.fit(train[features], train['defects'])
        
        # Predict on the test set
        test_pred = clf.predict_proba(test[features])[:, 1]
        test_preds.append(test_pred)
        
        print(f"{name} done!")
    
    # Average the predictions from all classifiers
    blended_pred = np.mean(test_preds, axis=0)
    
    return blended_pred

# Example usage
# Assuming train and test dataframes already loaded with a 'target' column in the train dataset
# features = ["feature1", "feature2", "feature3"]

# predictions = blended_predictions(train_df, test_df, features)


# In[30]:


# submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
# submission['defects'] = predictions
# submission.to_csv('blend_submission.csv', index = False)
# submission


# # Multi-Model Blend, Superior Aproach

# In[31]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from category_encoders import OneHotEncoder, GLMMEncoder, TargetEncoder, CatBoostEncoder
# from sklearn import set_config
# from sklearn.inspection import permutation_importance
# from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.ensemble import RandomForestRegressor, IsolationForest
# from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, f1_score
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.base import BaseEstimator, TransformerMixin, clone
# from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.ensemble import VotingClassifier, StackingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import squareform
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# sns.set_theme(style = 'white', palette = 'viridis')
# pal = sns.color_palette('viridis')

# pd.set_option('display.max_rows', 100)
# set_config(transform_output = 'pandas')
# pd.options.mode.chained_assignment = None


# In[32]:


# seed = 42
# splits = 5
# skf = StratifiedKFold(n_splits = splits, random_state = seed, shuffle = True)
# np.random.seed(seed)


# In[33]:


# def cross_val_score(estimator, cv = skf, label = '', include_original = False):
    
#     X = train_df.copy()
#     y = X.pop('defects')
    
#     #initiate prediction arrays and score lists
#     val_predictions = np.zeros((len(X)))
#     #train_predictions = np.zeros((len(sample)))
#     train_scores, val_scores = [], []
    
#     #training model, predicting prognosis probability, and evaluating metrics
#     for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
#         model = clone(estimator)
        
#         #define train set
#         X_train = X.iloc[train_idx]
#         y_train = y.iloc[train_idx]
        
#         #define validation set
#         X_val = X.iloc[val_idx]
#         y_val = y.iloc[val_idx]
        
#         if include_original:
#             X_train = pd.concat([X_train, orig_train.drop('defects', axis = 1)])
#             y_train = pd.concat([y_train, orig_train.defects])
        
#         #train model
#         model.fit(X_train, y_train)
        
#         #make predictions
#         train_preds = model.predict_proba(X_train)[:, 1]
#         val_preds = model.predict_proba(X_val)[:, 1]
                  
#         val_predictions[val_idx] += val_preds
        
#         #evaluate model for a fold
#         train_score = roc_auc_score(y_train, train_preds)
#         val_score = roc_auc_score(y_val, val_preds)
        
#         #append model score for a fold to list
#         train_scores.append(train_score)
#         val_scores.append(val_score)
    
#     print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
#     return val_scores, val_predictions


# In[34]:


# score_list, oof_list = pd.DataFrame(), pd.DataFrame()

# models = [
#     ('log', LogisticRegression(random_state = seed, max_iter = 1000000)),
#     ('lda', LinearDiscriminantAnalysis()),
#     ('gnb', GaussianNB()),
#     ('bnb', BernoulliNB()),
#     ('knn', KNeighborsClassifier()),
#     ('rf', RandomForestClassifier(random_state = seed)),
#     ('et', ExtraTreesClassifier(random_state = seed)),
#     ('xgb', XGBClassifier(random_state = seed)),
#     ('lgb', LGBMClassifier(random_state = seed)),
#     ('dart', LGBMClassifier(random_state = seed, boosting_type = 'dart')),
#     ('cb', CatBoostClassifier(random_state = seed, verbose = 0)),
#     ('gb', GradientBoostingClassifier(random_state = seed)),
#     ('hgb', HistGradientBoostingClassifier(random_state = seed)),
# ]

# for (label, model) in models:
#     score_list[label], oof_list[label] = cross_val_score(
#         make_pipeline(SimpleImputer(), model),
#         label = label,
#         include_original = False
#     )


# In[35]:


# weights = RidgeClassifier(random_state = seed).fit(oof_list, train_df.defects).coef_[0]
# pd.DataFrame(weights, index = list(oof_list), columns = ['weight per model'])


# In[36]:


# voter = VotingClassifier(models, weights = weights, voting = 'soft')
# _ = cross_val_score(
#     make_pipeline(SimpleImputer(), voter),
#     include_original = False
# )


# In[37]:


# X = train_df.copy()
# y = X.pop('defects')


# In[38]:


# model = make_pipeline(
#     SimpleImputer(),
#     voter
# )

# model.fit(X, y)


# In[39]:


# submission = test_df.copy()
# predictions = model.predict_proba(submission)[:, 1]


# In[40]:


# submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')
# submission['defects'] = predictions
# submission.to_csv('voter_submission.csv', index = False)
# submission


# ___

# # Visualizing Future Importance

# In[41]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\n%matplotlib inline\n# Creates a plot to visualize the most important features...\n\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(train_df[features].columns, xgboost_model.feature_importances_):\n    feats[feature] = importance #add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(15,4))\nplt.show()\n")


# ---

# In[42]:


# Baseline Model, Mean Accuracy: 0.8123979241377658
# Improved Model, More Diversity: Mean Accuracy: 0.8131968841103276, # 0.8138626678002563


# In[43]:


# import pandas as pd
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_predict
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def create_model(feature_list):
#     # Create a simple model
#     model = Sequential()
#     model.add(Dense(12, input_dim=len(feature_list), activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    
#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# def build_and_evaluate_model(dataframe, feature_list, target_variable):
#     # Preprocess the data
#     X = dataframe[feature_list]
#     y = dataframe[target_variable]
    
#     # Scale the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Encode the target variable if it's categorical
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
    
#     # Wrap the Keras model with KerasClassifier
#     keras_clf = KerasClassifier(build_fn=create_model(feature_list), epochs=50, batch_size=32, verbose=0)
    
#     # Perform cross-validation and get predictions
#     predictions = cross_val_predict(keras_clf, X_scaled, y_encoded, cv=5)
    
#     return predictions

# # Assume df is your pandas DataFrame
# # Assume features is a list of feature names
# # Assume target is the name of the target variable
# predictions = build_and_evaluate_model(train_df, features, 'defects')

