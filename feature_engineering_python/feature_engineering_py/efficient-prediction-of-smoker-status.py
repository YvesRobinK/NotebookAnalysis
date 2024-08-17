#!/usr/bin/env python
# coding: utf-8

# # 🔥 Binary Prediction of Smoker Status using Bio-Signals and XGBoost
# ## Playground Series - Season 3, Episode 24
# **According to a World Health Organization report, the number of deaths caused by smoking will reach 10 million by 2030.**
# 
# 
# <img src="https://drive.google.com/uc?export=view&id=12R-zs-F0GsztmEAGj0PDK8Nur9b_OlZw" width="500px" height="500px">
# 
# **Notebook Strategy & Results of Experimentation**
# * Configure the Notebook
# * Load the Datasets and the Original Dataset **(Original dataset, helps quite a lot)**
# * Explore the Information Loaded **(Data looks good nothing needed)**
# * Clean and Transform the Data **(Nothing needed)**
# * Create some Features
#      * BMI **(Feature didn't add any value)**
#      * Kmeans Features
#          * Height (cm) **(Feature didn't add any value) -- Rank top among others...**
#          * All Features **(Feature didn't add any value)**
#      * I Created multiple ratio feature, basically all of them using a loop **(None of them make an improvement)**
#      * Once I completed training the model I tried to use pseudolabeling **(model doesn't improve the CV core improve but not the LB score)**
# * Feature Standarization
#     * Robust **(Make the model worst)**
# * Machine Learning Model
#     * XGBoost Cross Validation **(I used a 5, 10 and 20 folds; The 10 is the best option so far)**
#     * XGBoost Hyper Param Optimization **(Make the model worst from my manual calibration)**
# * Model Submission
# 
# 
# 
# **About the Dataset**
# 
# Smoking has been proven to negatively affect health in a multitude of ways.Smoking has been found to harm nearly every organ of the body, cause many diseases, as well as reducing the life expectancy of smokers in general. As of 2018, smoking has been considered the leading cause of preventable morbidity and mortality in the world, continuing to plague the world’s overall health.
# 
# According to a World Health Organization report, the number of deaths caused by smoking will reach 10 million by 2030.
# 
# Evidence-based treatment for assistance in smoking cessation had been proposed and promoted. however, only less than one third of the participants could achieve the goal of abstinence. Many physicians found counseling for smoking cessation ineffective and time-consuming, and did not routinely do so in daily practice. To overcome this problem, several factors had been proposed to identify smokers who had a better chance of quitting, including the level of nicotine dependence, exhaled carbon monoxide (CO) concentration, cigarette amount per day, the age at smoking initiation, previous quit attempts, marital status, emotional distress, temperament and impulsivity scores, and the motivation to stop smoking. However, individual use of these factors for prediction could lead to conflicting results that were not straightforward enough for the physicians and patients to interpret and apply. Providing a prediction model might be a favorable way to understand the chance of quitting smoking for each individual smoker. Health outcome prediction models had been developed using methods of machine learning over recent years.
# 
# A group of scientists are working on predictive models with smoking status as the prediction target.Your task is to help them create a machine learning model to identify the smoking status of an individual using bio-signals
# 
# **Dataset Description**
# * age : 5-years gap
# * height(cm)
# * weight(kg)
# * waist(cm) : Waist circumference length
# * eyesight(left)
# * eyesight(right)
# * hearing(left)
# * hearing(right)
# * systolic : Blood pressure
# * relaxation : Blood pressure
# * fasting blood sugar
# * Cholesterol : total
# * triglyceride
# * HDL : cholesterol type
# * LDL : cholesterol type
# * hemoglobin
# * Urine protein
# * serum creatinine
# * AST : glutamic oxaloacetic transaminase type
# * ALT : glutamic oxaloacetic transaminase type
# * Gtp : γ-GTP
# * dental caries
# * smoking
# 

# ## Other Related Work
# Hello Here are other notebooks that I have created to explore other types of models ...
# 
# NN Model: https://www.kaggle.com/cv13j0/a-neuronal-prediction-of-smoker-status/edit

# ## Notebook Configuration...

# In[1]:


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


# In[2]:


from typing import Tuple
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

import pandas as pd
from sklearn.ensemble import IsolationForest


# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.3f}'.format\npd.set_option('display.max_columns', 15) \npd.set_option('display.max_rows', 50)\n\n# Define some of the notebook parameters for future experiment replication.\nSEED   = 578\n")


# ---

# ## Reading Datasets...

# In[4]:


get_ipython().run_cell_magic('time', '', '# Create a function to read the Datasets...\ndef read_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:\n    """\n    Reads train and test datasets from csv files and returns them as pandas DataFrames.\n\n    Parameters:\n    - train_path (str): The path to the train dataset csv file.\n    - test_path (str): The path to the test dataset csv file.\n\n    Returns:\n    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.\n    """\n\n    train_df = pd.read_csv(train_path)\n    test_df = pd.read_csv(test_path)\n\n    return train_df, test_df\n\n# Usage:\ntrain_path = \'/kaggle/input/playground-series-s3e24/train.csv\'\ntest_path = \'/kaggle/input/playground-series-s3e24/test.csv\'\ntrain_df, test_df = read_datasets(train_path, test_path)\n\ntrain_df = train_df.drop(columns=[\'id\'])\ntest_df = test_df.drop(columns=[\'id\'])\n\ntrain_df[\'is_original\'] = 0\ntest_df[\'is_original\'] = 0\n')


# In[5]:


# Merge the current train data with the original dataset...
original_path = '/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv'
original_df = pd.read_csv(original_path)
original_df['is_original'] = 1
train_df = pd.concat([train_df, original_df])


# In[6]:


train_df.info()


# In[7]:


original_df.info()


# ---

# ## Exploring Loading Information...

# In[8]:


get_ipython().run_cell_magic('time', '', 'def analyze_dataframe(df):\n    """\n    Analyze a pandas DataFrame and provide a summary of its characteristics.\n\n    Parameters:\n    df (pandas.DataFrame): The input DataFrame to analyze.\n\n    Returns:\n    None\n    """\n    print("DataFrame Information:")\n    print("----------------------")\n    display(df.info(verbose=True, show_counts=True))\n    print("\\n")\n    \n    print("DataFrame Values:")\n    print("----------------------")\n    display(df.head(5).T)\n    print("\\n")\n\n    print("DataFrame Description:")\n    print("----------------------")\n    display(df.describe().T)\n    print("\\n")\n\n    print("Number of Null Values:")\n    print("----------------------")\n    display(df.isnull().sum())\n    print("\\n")\n\n    print("Number of Duplicated Rows:")\n    print("--------------------------")\n    display(df.duplicated().sum())\n    print("\\n")\n\n    print("Number of Unique Values:")\n    print("------------------------")\n    display(df.nunique())\n    print("\\n")\n\n    print("DataFrame Shape:")\n    print("----------------")\n    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")\n\n# Usage\nanalyze_dataframe(train_df)\n')


# ---

# In[9]:


import pandas as pd

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame and print the number of duplicates found and removed.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - df_no_duplicates: DataFrame with duplicates removed
    """

    # Identify duplicates
    duplicates = df[df.duplicated()]

    # Print number of duplicates found and removed
    print(f"Number of duplicates found and removed: {len(duplicates)}")

    # Remove duplicates
    df_no_duplicates = df.drop_duplicates()

    return df_no_duplicates

train_df = remove_duplicates(train_df)


# ## Feature Engineering...
# 

# In[10]:


def create_features(df):
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    df['HW_Ratio'] = df['height(cm)'] / df['waist(cm)']
    df['HA_Ratio'] = df['height(cm)'] / df['age']
    return df

#train_df = create_features(train_df)
#test_df = create_features(test_df)


# In[11]:


def massive_feature(df, ignore_list):
    features = [feat for feat in df.columns if feat not in ignore_list]

    for idx1, col_one in enumerate(features):
        for idx2, col_two in enumerate(features):
            if idx1 < idx2:
                df[col_one +'_to_'+ col_two] = df[col_one] / df[col_two]
    return df

#train_df = massive_feature(train_df, ignore_list = ['id', 'smoking', 'is_original', 'dental caries','hearing(left)','hearing(right)',])
#test_df = massive_feature(test_df, ignore_list = ['id', 'smoking', 'is_original', 'dental caries','hearing(left)','hearing(right)',])


# In[12]:


def count_outliers(df, features):
    # Subset the dataframe to only the specified features
    df_subset = df[features]
    
    # Initialize the Isolation Forest model
    clf = IsolationForest(contamination='auto')
    
    # Fit the model on the subset
    predictions = clf.fit_predict(df_subset)
    
    # Create a DataFrame to store the outlier count for each row
    outlier_count_df = pd.DataFrame({
        'Outlier_Count': [(pred == -1) for pred in predictions]
    })

    # Sum the counts for each row to get total outlier count
    total_outliers = outlier_count_df['Outlier_Count'].sum()
    
    # Attach the outlier count to the original dataframe
    df['Outlier_Count'] = outlier_count_df
    
    # Return the dataframe with the added outlier count column
    return df

# Example usage:
# Assuming 'df' is your dataframe and ['feature1', 'feature2'] are the features of interest
# df_with_outliers = count_outliers(df, ['feature1', 'feature2'])
# print(df_with_outliers)

ignore_list = ['id', 'smoking', 'is_original']
features = [feat for feat in train_df.columns if feat not in ignore_list]
train_df = count_outliers(train_df, features)
test_df = count_outliers(test_df, features)


# ---

# ## Selecting Model Features...

# In[13]:


get_ipython().run_cell_magic('time', '', "# Drop missing values target column of the dataset...\ncategorical_features = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']\nnumerical_features = [feat for feat in train_df.columns if feat not in categorical_features and feat not in ['smoking']]\n")


# In[14]:


def one_hot_encode(df, categorical_features):
    """
    One-hot encode the specified categorical features in the given DataFrame.

    Parameters:
    - df (pandas DataFrame): The input DataFrame.
    - categorical_features (list of str): List of categorical feature names to be one-hot encoded.

    Returns:
    - pandas DataFrame: A DataFrame with the specified categorical features one-hot encoded.
    """
    return pd.get_dummies(df, columns=categorical_features)

# train_df = one_hot_encode(train_df, categorical_features)
# test_df = one_hot_encode(test_df, categorical_features)


# In[15]:


import pandas as pd
from sklearn.cluster import KMeans

def kmeans_features(df, features, n_clusters=3):
    """
    Adds new features to the DataFrame based on K-means clustering.
    
    Parameters:
    - df: pandas DataFrame.
    - features: List of features to use for clustering.
    - n_clusters: Number of clusters for K-means clustering (default is 3).
    
    Returns:
    - pandas DataFrame with new features based on the clustering.
    """
    # Extract the relevant features for clustering
    X = df[features]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    
    # Create a new feature for the cluster labels
    df['cluster_label'] = kmeans.labels_
    
    # Add distance features for each cluster center
    for i, center in enumerate(kmeans.cluster_centers_):
        df[f'dist_to_center_{i}'] = ((X - center) ** 2).sum(axis=1) ** 0.5
        
    return df

# train_df = kmeans_features(train_df, numerical_features, 2)
# test_df = kmeans_features(test_df, numerical_features, 2)


# In[16]:


get_ipython().run_cell_magic('time', '', "model_features = [col for col in train_df.columns if col not in ['id', \n                                                                 'smoking',\n                                                                 #'BMI',\n                                                                 #'weight(kg)', \n                                                                 #'height(cm)'\n                                                                ]]\n")


# ---

#  ## Feature Standarization...

# In[17]:


get_ipython().run_cell_magic('time', '', 'def standardize_features(df, features, method=\'zscore\'):\n    # Making a copy of the dataframe to avoid changing the original dataframe\n    df_copy = df.copy()\n\n    # Selecting the features to be standardized\n    data_to_scale = df_copy[features]\n\n    # Choosing the standardization method\n    if method == \'zscore\':\n        scaler = StandardScaler()\n    elif method == \'minmax\':\n        scaler = MinMaxScaler()\n    elif method == \'robust\':\n        scaler = RobustScaler()\n    else:\n        raise ValueError("Invalid method. Choose from \'zscore\', \'minmax\', or \'robust\'.")\n\n    # Applying the standardization\n    standardized_data = scaler.fit_transform(data_to_scale)\n\n    # Replacing the original feature values with the standardized values\n    df_copy[features] = standardized_data\n\n    return df_copy\n\n# Usage\n# train_df = standardize_features(train_df, model_features, method = \'robust\')\n# test_df = standardize_features(test_df, model_features, method = \'robust\')\n')


# ---

# # Training an XGBoost Model...

# In[18]:


model_features


# # Optuna Hyper-Param Optimization

# In[19]:


def objective(trial):
    # Load the dataset and split it into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df[model_features], train_df['smoking'], test_size=0.25, random_state=SEED)

    # Define the hyperparameters to be optimized
    param = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0, step = 0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step = 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "n_estimators": trial.suggest_int("n_estimators", 256, 4096),
        "eta": trial.suggest_float("eta", 0.01, 0.5, step = 0.01),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "tree_method": "gpu_hist",
    }

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    #Train the XGBoost model with the current hyperparameters
    model = xgb.train(param, xgb.DMatrix(X_train, label = y_train),
                      #num_boost_round=100
                     )
    
    #model = xgb.XGBClassifier(**param)
    #model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = 512)

    # Evaluate the model on the test set
    y_pred = model.predict(xgb.DMatrix(X_test))
    loss = log_loss(y_test, y_pred)

    return loss

def optimize_xgboost_hyperparameters(num_trials=10):
    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    return best_params

# Run the optimization

optimal_params = optimize_xgboost_hyperparameters()
print('.' * 25, '\n')
print(optimal_params)


# ---

# # XG Boost Model...

# In[20]:


get_ipython().run_cell_magic('time', '', '# \ndef fit_xgboost_with_kfold(df, features, target_variable, parameters, n_splits=10,  random_state=SEED):\n    """\n    Fit an XGBoost Classifier to a pandas DataFrame with k-fold cross-validation.\n\n    Parameters:\n    df (pandas.DataFrame): The input DataFrame.\n    target_variable (str): The name of the target variable column in the DataFrame.\n    n_splits (int): The number of folds in the cross-validation (default: 5).\n    random_state (int): A random seed for reproducible results (default: 42).\n\n    Returns:\n    xgboost.XGBClassifier: A trained XGBoost Classifier model.\n    """\n    X = df.drop(columns=[target_variable])\n    y = df[target_variable]\n\n    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)\n \n    model = xgb.XGBClassifier(**parameters)\n\n    fold_rocs = []\n    fold_loglosses = []\n    fold_predictions = []\n    fold = 1\n\n    for train_index, test_index in kfold.split(X[features], y):\n        print(f\'Training Fold: {fold} ...\')\n        X_train, X_test = X[features].iloc[train_index], X[features].iloc[test_index]\n        y_train, y_test = y.iloc[train_index], y.iloc[test_index]\n\n        model.fit(X_train,\n                  y_train,\n                  eval_set = [(X_test, y_test)], \n                  verbose = 512,)\n        \n        best_iteration = model.get_booster().best_ntree_limit\n\n        y_pred = model.predict(X_test, ntree_limit=best_iteration)\n        y_pred_proba = model.predict_proba(X_test, ntree_limit=best_iteration)[:,1]\n        \n        fold_logloss = log_loss(y_test, y_pred_proba)\n        fold_roc = roc_auc_score(y_test, y_pred_proba)\n        fold_rocs.append(fold_roc)\n        fold_loglosses.append(fold_logloss)\n        fold += 1\n        \n        test_pred = model.predict_proba(test_df[features])[:,1]\n        fold_predictions.append(test_pred)\n        \n        print(\'....\', \'\\n\')\n\n    predictions = np.mean(fold_predictions, axis=0)\n\n    print("Fold Accuracies:", fold_rocs)\n    print("Fold Log Losses:", fold_loglosses)\n    print("Mean AUC:", sum(fold_rocs) / len(fold_rocs))\n    print("Mean Log Loss:", sum(fold_loglosses) / len(fold_loglosses))\n\n    return model, predictions \n')


# In[21]:


get_ipython().run_cell_magic('time', '', "# Best Model Parameters...\nparams = {'n_estimators'          : 2048,\n          'max_depth'             : 9,\n          'learning_rate'         : 0.05,\n          'booster'               : 'gbtree',\n          'subsample'             : 0.75,\n          'colsample_bytree'      : 0.30,\n          'reg_lambda'            : 1.00,\n          'reg_alpha'             : 1.00,\n          'gamma'                 : 1.00,\n          'random_state'          : SEED,\n          'objective'             : 'binary:logistic',\n          'tree_method'           : 'gpu_hist',\n          'eval_metric'           : 'auc',\n          'early_stopping_rounds' : 256,\n          'n_jobs'                : -1,\n         }\n\n\nparams = {'n_estimators'          : 2048,\n          'max_depth'             : 9,\n          'learning_rate'         : 0.045,\n          'booster'               : 'gbtree',\n          'subsample'             : 0.75,\n          'colsample_bytree'      : 0.30,\n          'reg_lambda'            : 1.00,\n          'reg_alpha'             : 0.80,\n          'gamma'                 : 0.80,\n          'random_state'          : SEED,\n          'objective'             : 'binary:logistic',\n          'tree_method'           : 'gpu_hist',\n          'eval_metric'           : 'auc',\n          'early_stopping_rounds' : 256,\n          'n_jobs'                : -1,\n         }\n\n# Not used at this point...\n# opt_params = {'booster': 'dart', \n#               'lambda': 3.386345811577273e-05, \n#               'alpha': 0.2293918168443115, \n#               'subsample': 0.8, \n#               'colsample_bytree': 1.0, \n#               'max_depth': 8, \n#               'n_estimators': 3393, \n#               'eta': 0.287678021761605, \n#               'gamma': 2.8800815977486452e-06, \n#               'grow_policy': 'lossguide', \n#               'sample_type': 'uniform', \n#               'normalize_type': 'forest', \n#               'rate_drop': 8.305338078638612e-06, \n#               'skip_drop': 0.000417122371690196,\n#               'objective': 'binary:logistic',\n#               'tree_method': 'gpu_hist',\n#               'eval_metric': 'auc',\n#               'early_stopping_rounds': 256,\n#               'n_jobs': -1}\n\n\nxgboost_model, xgboost_predictions = fit_xgboost_with_kfold(train_df, \n                                                            model_features, \n                                                            target_variable='smoking',\n                                                            parameters = params, \n                                                            random_state=SEED, \n                                                            n_splits = 10)\n")


# In[22]:


train_pred = xgboost_model.predict_proba(train_df[model_features])[:,1]
train_df['pred'] = train_pred
train_df[(train_df['smoking'] == 1) & (train_df['pred'] > 0.9)][model_features].sample(10).T


# In[23]:


get_ipython().run_cell_magic('time', '', "submission = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')\nsubmission['smoking'] = xgboost_predictions\nsubmission.to_csv('xgb_opt_submission.csv', index = False)\nsubmission\n")


# ---

# # XGBoost + Pseudo-Labels...

# In[24]:


test_df['predictions'] = xgboost_predictions
test_df.head()

cutoff = 0.95 # Probability CutOff...
pseudo_set_1 = test_df[test_df['predictions'] > cutoff]
pseudo_set_1['smoking'] = 1
pseudo_set_1.drop(columns=['predictions'], axis = 1, inplace=True)

pseudo_set_2 = test_df[test_df['predictions'] < 1 - cutoff]
pseudo_set_2['smoking'] = 0
pseudo_set_2.drop(columns=['predictions'], axis = 1, inplace=True)

pseudo_set = pd.concat([pseudo_set_1,pseudo_set_2])
pseudo_set.shape


# In[25]:


pseudo_train_df = pd.concat([train_df, pseudo_set])

params = {'n_estimators'          : 2048,
          'max_depth'             : 9,
          'learning_rate'         : 0.045,
          'booster'               : 'gbtree',
          'subsample'             : 0.75,
          'colsample_bytree'      : 0.30,
          'reg_lambda'            : 1.00,
          'reg_alpha'             : 0.80,
          'gamma'                 : 0.80,
          'random_state'          : SEED,
          'objective'             : 'binary:logistic',
          'tree_method'           : 'gpu_hist',
          'eval_metric'           : 'auc',
          'early_stopping_rounds' : 256,
          'n_jobs'                : -1,
         }

xgboost_model, xgboost_predictions = fit_xgboost_with_kfold(pseudo_train_df, 
                                                            model_features, 
                                                            target_variable='smoking',
                                                            parameters = params, 
                                                            random_state=SEED, 
                                                            n_splits = 10)


# In[26]:


get_ipython().run_cell_magic('time', '', "submission = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')\nsubmission['smoking'] = xgboost_predictions\nsubmission.to_csv('xgb_pseudo_opt_submission.csv', index = False)\nsubmission\n")


# In[27]:


# Model Scores ...
# Mean AUC: 0.872514935089370
# Mean AUC: 0.873211725798716
# Mean AUC: 0.8740200234564879
# Mean AUC: 0.8744331572784543 ... No Features Added Best ...
# Mean AUC: 0.8746445749361194 ... Added BMI Added
# Mean AUC: 0.8740744699588816 ... Added Kmeans
# Mean AUC: 0.8751253867438106 ... 0.09 LR No Features Added Best ...
# Mean AUC: 0.8755503134941037 ... 0.08 LR No Features Added Best ...
# Mean AUC: 0.8763032176590734 ... 0.07 LR No Features Added Best ...
# Mean AUC: 0.8769275947935501 ...
# Mean AUC: 0.8770171911144239
# Mean AUC: 0.8772263976643453 ...
# Mean AUC: 0.8774334209493363 ...
# Mean AUC: 0.8779683441289976 ....
# Mean AUC: 0.8779683441289976 .... 
# Mean AUC: 0.8779911796335711
# Mean AUC: 0.8780984673339569 .... Best Model... Calibrating Hyper-Params...
# Mean AUC: 0.8784115780235494
# Mean AUC: 0.87843085346659 ...
# Mean AUC: 0.878512247844634 Added IsOutlier Feature...
# Mean AUC: 0.8792021618371045 Multiple folds...
# Mean AUC: 0.9112707017734414 Using pseudolabels...


# ---

# # Linear Classifier...

# In[28]:


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
    y_train = train_df['smoking']
    
    # Initialize the linear classifier
    model = LogisticRegression(max_iter=500)  # max_iter is increased to ensure convergence for most datasets
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
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

# lr_model, lr_predictions = train_linear_classifier(train_df, test_df, model_features)


# In[29]:


# submission = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')
# submission['smoking'] = lr_predictions
# submission.to_csv('lr_submission.csv', index = False)
# submission


# ---

# # Blended Model...

# In[30]:


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
        cross_val_pred = cross_val_predict(clf, train[features], train['smoking'], cv=10, method='predict_proba')[:, 1]
        
        # Fit the classifier to the entire training set
        clf.fit(train[features], train['smoking'])
        
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

# blend_predictions = blended_predictions(train_df, test_df, model_features)


# In[31]:


# submission = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')
# submission['smoking'] = blend_predictions
# submission.to_csv('blend_submission.csv', index = False)
# submission


# ---

# # Machine Learning Explainability

# In[32]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\n%matplotlib inline\n# Creates a plot to visualize the most important features...\n\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(train_df[model_features].columns, xgboost_model.feature_importances_):\n    feats[feature] = importance #add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(10,4))\nplt.show()\n")


# In[ ]:




