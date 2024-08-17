#!/usr/bin/env python
# coding: utf-8

# <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
#     <div style="border: 2px solid #e94196; padding: 20px; text-align: center; border-radius: 10px; background-color: #ffffff;">
#         <h1 style="color: #e94196; font-size: 32px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;">Binary Prediction of Smoker Status using Bio-Signals</h1>
#         
#         <p style="font-size: 18px; color: #333333; line-height: 1.6;">
#                  Enjoying the content in this Python notebook? If you find it helpful or interesting, feel free to give it a like or follow for more updates and valuable programming insights!
#         </p>
#     </div>
# </div>
# 

# <h2 style="color: #e94196; font-weight: bold;">IMPORT LIBRARIES</h2>

# In[89]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import HTML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ROCAUC
from matplotlib.colors import ListedColormap
get_ipython().system('pip install -q plotly pandas')
get_ipython().system('pip install -q -U kaleido')
get_ipython().system('pip install -q fasteda')
from fasteda import fast_eda
import plotly.express as px
import math
get_ipython().run_line_magic('matplotlib', 'inline')
sns.color_palette("PiYG")
sns.set_style("whitegrid")

import numpy as np
import pandas as pd 
get_ipython().system('pip install -q fasteda')
get_ipython().system('pip install -q rich')
from rich import print as rich_print
from fasteda import fast_eda
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import skew
from xgboost import XGBClassifier

from gc import collect
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import time


# <h2 style="color: #e94196; font-weight: bold;">LOAD THE DATASET</h2>

# In[118]:


train = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')


# In[119]:


origin = pd.read_csv('/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv')
origin.dropna(inplace=True)


# In[120]:


origin.head()


# <h2 style="color: #e94196; font-weight: bold;">OBSERVE THE DATAFRAME</h2>

# In[121]:


pprint = lambda text: print(f"\033[1;91m{text}\033[0m")
def prGreen(text):
    print("\033[1m{}\033[0m".format(text))
prGreen("There are duplicates (train)" if train.duplicated().any() else "There are no duplicates. (train)")

def prGreen(text):
    print("\033[1m{}\033[0m".format(text))
prGreen("There are duplicates (test)" if test.duplicated().any() else "There are no duplicates. (test)")
pprint("Full train dataset shape is {}".format(train.shape))
pprint("Full test dataset shape is {}".format(test.shape))


# In[122]:


train.head()


# In[123]:


#Save the 'Id' column
train_ID = train['id']
test_ID = test['id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("id", axis = 1, inplace = True)
test.drop("id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
pprint("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
pprint("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[ ]:





# In[124]:


train = pd.concat((train, origin)).reset_index(drop=True)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.smoking.values
df = pd.concat((train, test)).reset_index(drop=True)
#df.drop(['smoking'], axis=1, inplace=True)
pprint("The size of combined data is : {}".format(df.shape))


# In[125]:


train.head()


# **SUMMARY OF COMBINED DATA**

# In[126]:


def summary(df):
    sum = pd.DataFrame(df.dtypes, columns=['dtypes'])
    sum['missing#'] = df.isna().sum()
    sum['missing%'] = (df.isna().sum())/len(df)
    sum['uniques'] = df.nunique().values
    sum['count'] = df.count().values
    return sum

summary(df).style.background_gradient(cmap='Blues')


# **NUMERICAL DATA DISTRIBUTION**

# In[127]:


df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()


# In[128]:


num_cols = df_num.shape[1]
num_rows = math.ceil(num_cols / 6)
fig, axes = plt.subplots(num_rows, 6, figsize=(16, 12))
axes = axes.ravel()

for i, column in enumerate(df_num.columns):
    sns.histplot(data=df_num, x=column, bins=50, ax=axes[i],  kde=True)
    axes[i].set_title(f'{column} Distribution', fontsize=10)
    axes[i].set_xlabel(column, fontsize=6)
    axes[i].set_ylabel('Frequency', fontsize=6)
    axes[i].tick_params(axis='both', which='both', labelsize=6)

# Remove any extra empty subplots
for i in range(num_cols, num_rows * 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# **OUTLIERS**

# In[129]:


fig, axes = plt.subplots(num_rows, 6, figsize=(16, 12))

# Flatten the axes array
axes = axes.ravel()

# Loop through numeric columns and create box plots to visualize outliers
for i, column in enumerate(df_num.columns):
    sns.boxplot(data=df_num, y=column, ax=axes[i])
    axes[i].set_title(f'{column} Box Plot', fontsize=10)
    axes[i].set_ylabel(column, fontsize=6)
    axes[i].tick_params(axis='both', which='both', labelsize=6)

# Remove any extra empty subplots
for i in range(num_cols, num_rows * 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# In[130]:


# Calculate the IQR for each column
low = 0.00
high = 1- low
Q1 = df.quantile(low)
Q3 = df.quantile(high)
IQR = Q3 - Q1

# Define the outlier detection threshold factor
outlier_threshold_factor = 1.5

# Detect outliers using the IQR method
outliers = ((df < (Q1 - outlier_threshold_factor * IQR)) | (df > (Q3 + outlier_threshold_factor * IQR)))

# Display columns with outliers
columns_with_outliers = outliers.any()
print("\033[38;2;434;18;137m"+"Columns with outliers:"+"\033[0m")
pprint(columns_with_outliers)


# In[131]:


# Remove rows with outliers
df_no_outliers = df[~outliers.any(axis=1)]
# Display the modified DataFrame
pprint( "Shape of the modeified df = " +str(df_no_outliers.shape))

df = df[~outliers.any(axis=1)]


# <h2 style="color: #e94196; font-weight: bold;">FEATURE ENGINEERING</h2>

# In[132]:


test.columns


# In[133]:


#Create a score based on age, blood pressure, cholesterol levels, and smoking status.
df['cardiovascular_risk_score'] = df.apply(lambda row: (row['age'] * 0.05) + (row['systolic'] * 0.1) + (row['Cholesterol'] * 0.002) , axis=1)

#Blood Pressure Variability
df['blood_pressure_variation'] = df['systolic'] - df['relaxation']

# Calculate BMI (Body Mass Index)
df['bmi'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
# Calculate Waist-to-Height Ratio
df['waist_to_height_ratio'] = df['waist(cm)'] / df['height(cm)']

#Kidney Function Estimate
# CKD-EPI formula for estimating GFR
alpha = -0.411
kappa = 0.9
def calculate_gfr(row):
    age_factor = 0.993 ** row['age']
    min_sc = min(row['serum creatinine'], 0.9)
    max_sc = max(row['serum creatinine'], 0.9)
    estimated_gfr = 141 * age_factor * min_sc ** alpha * max_sc ** (-1.209) * 0.993 ** 0.993 * kappa
    return estimated_gfr
df['estimated_gfr'] = df.apply(calculate_gfr, axis=1)

#Cardiovascular Health Score:
df['cardiovascular_health_score'] = (df['systolic'] + df['Cholesterol'] - df['HDL'] - 10 * df['bmi']) / 4

#cardiovascular_risk_score
#df['cardiovascular_risk_score'] = df.apply(lambda row: (row['age'] * 0.05) + (row['systolic'] * 0.1) + (row['Cholesterol'] * 0.002) + (row['smoking'] * 10), axis=1)

#Physical Fitness Level
fitness_score = (df['bmi'] * 0.3 + df['waist_to_height_ratio'] * 0.3 - df['age'] * 0.1)
df['fitness_level'] = fitness_score.apply(lambda x: 'Fit' if x < 0 else 'Not Fit')

#bmi category
conditions = [
    (df['bmi'] < 18.5),
    (df['bmi'] >= 18.5) & (df['bmi'] < 24.9),
    (df['bmi'] >= 24.9) & (df['bmi'] < 29.9),
    (df['bmi'] >= 29.9)
]
choices = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']
df['bmi_category'] = np.select(conditions, choices)

# Calculate Average Eyesight and Hearing
df['average_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
df['average_hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2

# Categorize Blood Pressure
def categorize_blood_pressure(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif 120 <= systolic < 130 or 80 <= diastolic < 90:
        return 'Elevated'
    elif 130 <= systolic < 140 or 90 <= diastolic < 100:
        return 'Hypertension Stage 1'
    else:
        return 'Hypertension Stage 2'

df['blood_pressure_category'] = df.apply(lambda row: categorize_blood_pressure(row['systolic'], row['relaxation']), axis=1)

# Calculate Cholesterol Ratio
df['cholesterol_ratio'] = df['Cholesterol'] / df['HDL']

# Calculate Liver Enzyme Ratio
df['liver_enzyme_ratio'] = (df['AST'] + df['ALT'] + df['Gtp']) / 3


# In[134]:


df = pd.get_dummies(df)


# In[135]:


df = df.apply(pd.to_numeric, errors='coerce')


# 
# <h2 style="color: #e94196; font-weight: bold;">VISUALIZE</h2>

# In[136]:


def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train = df[:ntrain]
test = df[ntrain:]
test.drop(columns="smoking", inplace=True)


# In[137]:


fast_eda(train, countplot=False, hist_box_plot=False, target="smoking")


# In[138]:


train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)


# <h2 style="color: #e94196; font-weight: bold;">MODEL</h2>

# In[140]:


X = train.drop(columns=['smoking'])
y = train['smoking']


# In[143]:


import optuna

import warnings
warnings.filterwarnings('ignore')

EDA = False
SINGLE_PARAM_OPT = False
OPTUNA = False
nfolds = 5
skfold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=0)
if OPTUNA:
    def xgb_objective(trial):
        params = {
            'booster': ['gbtree'],  # Booster type: 'gbtree' for tree based models
            'learning_rate': [0.05],
            #'colsample_bytree': [1.0],
            #'reg_alpha': [ 0.85, 0.8,0.87,0.97,0.77],
            'reg_lambda': [0.7142857142857142],
            'min_child_weight':[0],
            'eval_metric': ['auc'],  # Evaluation metric: 'auc' for Area Under the ROC Curve
           'subsample': [0.9342785731449753],
           'gamma': [0.0],  # Minimum loss reduction required to make a further partition on a leaf node
             'n_estimators':[122],
            'random_state':[42],
           'max_depth': [5],  # Maximum depth of the tree
          'objective': [ 'reg:logistic', ],
            'n_jobs': [-1]
        }

        xgb_auc_score_avg = sum(
            roc_auc_score(y[val_idx], XGBClassifier(**params).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
            for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
        ) / nfolds

        print(f'The averaged AUC score evaluated on the validation subset using XGB model:', xgb_auc_score_avg)
        return -xgb_auc_score_avg

    xgb_study = optuna.create_study()
    xgb_study.optimize(xgb_objective, n_trials=500)
    best_xgb_params = xgb_study.best_trial.params

    print('Best XGB hyperparameters:', best_xgb_params)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

# Assuming skfold and nfolds are defined previously
skfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)


def hist_gb_objective(trial):
    params = {
   'max_depth': [9],
    'learning_rate': [0.01],
    }

    hist_gb_auc_score_avg = sum(
        roc_auc_score(y[val_idx], HistGradientBoostingClassifier(**params).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
        for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
    ) / nfolds

    print(f'The averaged AUC score evaluated on the validation subset using HistGradientBoostingClassifier:', hist_gb_auc_score_avg)
    return -hist_gb_auc_score_avg

    hist_gb_study = optuna.create_study()
    hist_gb_study.optimize(hist_gb_objective, n_trials=500)
    best_hist_gb_params = hist_gb_study.best_trial.params

    print('Best HistGradientBoostingClassifier hyperparameters:', best_hist_gb_params)
from sklearn.ensemble import GradientBoostingClassifier

# Assuming you have skfold, nfolds, X, and y defined previously

# Define the parameter space for GBM optimization
gbm_params= {
   'n_estimators': [300],  # Number of boosting stages to be run
    'learning_rate': [0.005],  # Step size shrinkage used to prevent overfitting
    'max_depth': [5]  # Maximum depth of the individual estimators #güncelle
}

# Optimize GBM using Optuna
def gbm_objective(trial):
    params = {
        'max_depth': [5],
        'learning_rate': [0.005],
        'n_estimators': [300]
    }

    gbm_auc_score_avg = sum(
        roc_auc_score(y[val_idx], GradientBoostingClassifier(**params).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
        for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
    ) / nfolds

    print(f'The averaged AUC score evaluated on the validation subset using GBM model:', gbm_auc_score_avg)
    return -gbm_auc_score_avg

    # Create an Optuna study and optimize GBM hyperparameters
    gbm_study = optuna.create_study()
    gbm_study.optimize(gbm_objective, n_trials=500)
    best_gbm_params = gbm_study.best_trial.params

    print('Best GBM hyperparameters:', best_gbm_params)
if OPTUNA:
    def lgb_objective(trial):
        params = {
            'boosting_type': ['gbdt'],
            'num_leaves': [31],
            'learning_rate': [0.05],
            'colsample_bytree': [1.0] , # Put the parameter inside a list
            'reg_alpha': [0.7],
           'reg_lambda': [0.2],
                'min_child_weight': [0.001], 
                'metric': ['auc'], 
                'num_iterations': [100], 
           'subsample': [0.9342785731449753],
                    'min_split_gain': [0.0], 
                'min_child_samples': [274], 
              'num_leaves': [31], 
                'objective': ['regression'], 
                'n_jobs': [-1], 
                'n_estimators': [122]

        }

        lgb_auc_score_avg = sum(
            roc_auc_score(y[val_idx], LGBMClassifier(**params).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
            for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
        ) / nfolds

        print(f'The averaged AUC score evaluated on the validation subset using LGB model:', lgb_auc_score_avg)
        return -lgb_auc_score_avg

    lgb_study = optuna.create_study()
    lgb_study.optimize(lgb_objective, n_trials=500)
    best_lgb_params = lgb_study.best_trial.params

    print('Best LGBM parameters:', best_lgb_params)
if OPTUNA:
    def cb_objective(trial):
        params  = {
                        'depth': [9],
                        'learning_rate': [0.005],
                       'l2_leaf_reg': [3],
                        'border_count': [254],

                    }

        cb_auc_score_avg = sum(
            roc_auc_score(y[val_idx], CatBoostClassifier(**params).fit(X.iloc[train_idx], y[train_idx], verbose=0).predict_proba(X.iloc[val_idx])[:, 1])
            for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
        ) / nfolds

        print(f'The averaged AUC score evaluated on the validation subset using Catboost model:', cb_auc_score_avg)
        return -cb_auc_score_avg

    cb_study = optuna.create_study()
    cb_study.optimize(cb_objective, n_trials=500)
    best_cb_params = cb_study.best_trial.params

    print('Best Catboost parameters:', best_cb_params)
from sklearn.ensemble import AdaBoostClassifier

if OPTUNA:
    def adaboost_objective(trial):
        params = {
                        'learning_rate': [0.05],
                         'random_state':[42],
                      'base_estimator': [ 'deprecated'],
                        'n_estimators':[122],
                    'loss': ['exponential'],

                    }

        adaboost_auc_score_avg = sum(
            roc_auc_score(y[val_idx], AdaBoostClassifier(**params).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
            for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
        ) / nfolds

        print(f'The averaged AUC score evaluated on the validation subset using AdaBoost model:', adaboost_auc_score_avg)
        return -adaboost_auc_score_avg

    adaboost_study = optuna.create_study()
    adaboost_study.optimize(adaboost_objective, n_trials=500)
    best_adaboost_params = adaboost_study.best_trial.params

    print('Best AdaBoost parameters:', best_adaboost_params)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

# Define the objective function for Optuna optimization
def rf_objective(trial):
    # Define the hyperparameters to be tuned
    params = {
    'max_depth': 7,  # Maximum depth of the tree
    'min_samples_leaf': 4,     # Minimum number of samples required to be at a leaf node
    'bootstrap': True         # Whether bootstrap samples are used when building trees
                }

    
    # Perform cross-validation with StratifiedKFold
    skfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    rf_auc_score_avg = sum(
        roc_auc_score(y[val_idx], RandomForestClassifier(**params, random_state=42).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
        for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
    ) / nfolds

    print(f'The averaged AUC score evaluated on the validation subset using Random Forest model:', rf_auc_score_avg)
    return -rf_auc_score_avg

# Create an Optuna study object and optimize the objective function
#rf_study = optuna.create_study(direction='maximize')
#rf_study.optimize(rf_objective, n_trials=500)

# Get the best hyperparameters from the study
#best_rf_params = rf_study.best_trial.params

#print('Best Random Forest hyperparameters:', best_rf_params)

    rf_study = optuna.create_study()
    rf_study.optimize(rf_objective, n_trials=500)
    best_rf_params = rf_study.best_trial.params

    print('Best Random Forest hyperparameters:', best_rf_params)
import optuna
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Define the objective function for Optuna optimization
def lgb_objective(trial):
    # Define the hyperparameters with specified values
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.7,
        'reg_lambda': 0.2,
        'min_child_weight': 0.001,
        'metric': 'auc',
        'num_iterations': 100,
        'subsample': 0.9342785731449753,
        'min_split_gain': 0.0,
        'min_child_samples': 274,
        'objective': 'regression',
        'n_jobs': -1,
        'n_estimators': 122
    }
    
    # Perform cross-validation with StratifiedKFold
    skfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    lgb_auc_score_avg = sum(
        roc_auc_score(y[val_idx], lgb.LGBMClassifier(**params, random_state=42).fit(X.iloc[train_idx], y[train_idx]).predict_proba(X.iloc[val_idx])[:, 1])
        for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y))
    ) / nfolds

    print(f'The averaged AUC score evaluated on the validation subset using LightGBM model:', lgb_auc_score_avg)
    return -lgb_auc_score_avg

    # Create an Optuna study object and optimize the objective function
    lgb_study = optuna.create_study(direction='maximize')
    lgb_study.optimize(lgb_objective, n_trials=500)

    # Get the best hyperparameters from the study
    best_lgb_params = lgb_study.best_trial.params

    print('Best LightGBM hyperparameters:', best_lgb_params)
from sklearn.metrics import make_scorer, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
lgb_params = {
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0 , # Put the parameter inside a list
    'reg_alpha': 0.7,
   'reg_lambda': 0.2,
        'min_child_weight': 0.001, 
        'metric': 'auc', 
        'num_iterations': 100, 
   'subsample': 0.9342785731449753,
            'min_split_gain': 0.0, 
        'min_child_samples': 274, 
      'num_leaves': 31, 
        'objective': 'regression', 
        'n_jobs': -1, 
        'n_estimators': 122

}

xgb_params = {
    'booster': 'gbtree',  # Booster type: 'gbtree' for tree based models
    'learning_rate': 0.05,
    #'colsample_bytree': 1.0,
    #'reg_alpha':  0.85, 0.8,0.87,0.97,0.77,
    'reg_lambda': 0.7142857142857142,
    'min_child_weight':0,
    'eval_metric': 'auc',  # Evaluation metric: 'auc' for Area Under the ROC Curve
   'subsample': 0.9342785731449753,
   'gamma': 0.0,  # Minimum loss reduction required to make a further partition on a leaf node
     'n_estimators':122,
    'random_state':42,
   'max_depth': 5,  # Maximum depth of the tree
  'objective':  'reg:logistic',
    'n_jobs': -1
}

cb_params = {
    'depth': 9,
    'learning_rate': 0.005,
   'l2_leaf_reg': 3,
    'border_count': 254,

}

adaboost_params = {
    'n_estimators': 50,
    'learning_rate': 0.001,
    'random_state': 0
}

rf_params = {
    'max_depth': 7,  # Maximum depth of the tree
    'min_samples_leaf': 4,     # Minimum number of samples required to be at a leaf node
    'bootstrap': True         # Whether bootstrap samples are used when building trees
                }


histgb_params = {
   'max_depth': 9,
    'learning_rate': 0.01,
}


gbm_params = {
   'n_estimators': 300,  # Number of boosting stages to be run
    'learning_rate': 0.005,  # Step size shrinkage used to prevent overfitting
    'max_depth': 5  # Maximum depth of the individual estimators
}
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Lists to store trained models
lgb_models, xgb_models, cb_models, rf_models, adaboost_models, histgb_models, gbm_models = [], [], [], [], [], [], []

for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    print(f'Training fold {idx + 1}/5')
    train_X = X.iloc[train_idx]
    val_X = X.iloc[val_idx]
    train_y = y[train_idx]
    val_y = y[val_idx]

    # LGBMClassifier
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(train_X, train_y)
    lgb_models.append(lgb_model)

    # XGBClassifier
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(train_X, train_y)
    xgb_models.append(xgb_model)

    # CatBoost
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(train_X, train_y, verbose=0)
    cb_models.append(cb_model)

    # RandomForestClassifier
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(train_X, train_y)
    rf_models.append(rf_model)

    # AdaBoostClassifier
    adaboost_model = AdaBoostClassifier(**adaboost_params)
    adaboost_model.fit(train_X, train_y)
    adaboost_models.append(adaboost_model)

    # HistGradientBoostingClassifier
    histgb_model = HistGradientBoostingClassifier(**histgb_params)
    histgb_model.fit(train_X, train_y)
    histgb_models.append(histgb_model)
    
    # GBM (Gradient Boosting Machine)
    gbm_model = GradientBoostingClassifier(**gbm_params)
    gbm_model.fit(train_X, train_y)
    gbm_models.append(gbm_model)


# In[144]:


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

# Assuming you have lgb_models, xgb_models, cb_models, rf_models, histgb_models, adaboost_models, skfold, X, y, and nfolds defined previously

def ensemble_objective(trial):
    weights = {
        'lgb_weight': trial.suggest_float('lgb_weight', 8, 10),
        'xgb_weight': trial.suggest_float('xgb_weight', 8, 10),
        'cb_weight': trial.suggest_float('cb_weight', 0, 3),
        'rf_weight': trial.suggest_float('rf_weight', 0, 3),
        'histgb_weight': trial.suggest_float('histgb_weight', 0, 3),
        'adaboost_weight': trial.suggest_float('adaboost_weight', 0, 3),
        'gbm_weight': trial.suggest_float('gbm_weight', 0, 3)  # Add GBM weight
    }
    
    sum_weights = weights['lgb_weight'] + weights['xgb_weight'] + weights['cb_weight'] + weights['rf_weight'] + weights['histgb_weight'] + weights['adaboost_weight'] + weights['gbm_weight']
    ensemble_auc_score = 0
    
    for idx, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        train_X = X.iloc[train_idx]
        val_X = X.iloc[val_idx]
        train_y = y[train_idx]
        val_y = y[val_idx]

        # LGBMClassifier
        lgb_model = lgb_models[idx]
        lgb_prediction = lgb_model.predict_proba(val_X)[:, 1]

        # XGBClassifier
        xgb_model = xgb_models[idx]
        xgb_prediction = xgb_model.predict_proba(val_X)[:, 1]

        # CatBoost
        cb_model = cb_models[idx]
        cb_prediction = cb_model.predict_proba(val_X)[:, 1]

        # RandomForestClassifier
        rf_model = rf_models[idx]
        rf_prediction = rf_model.predict_proba(val_X)[:, 1]

        # HistGradientBoostingClassifier
        histgb_model = histgb_models[idx]
        histgb_prediction = histgb_model.predict_proba(val_X)[:, 1]

        # AdaBoostClassifier
        adaboost_model = adaboost_models[idx]
        adaboost_prediction = adaboost_model.predict_proba(val_X)[:, 1]

        # GradientBoostingClassifier
        gbm_model = gbm_models[idx]
        gbm_prediction = gbm_model.predict_proba(val_X)[:, 1]

        # Ensemble prediction using weighted average
        ensemble_prediction = (weights['lgb_weight'] * lgb_prediction +
                               weights['xgb_weight'] * xgb_prediction +
                               weights['cb_weight'] * cb_prediction +
                               weights['rf_weight'] * rf_prediction +
                               weights['histgb_weight'] * histgb_prediction +
                               weights['adaboost_weight'] * adaboost_prediction +
                               weights['gbm_weight'] * gbm_prediction) / sum_weights

        ensemble_auc_score += roc_auc_score(val_y, ensemble_prediction)
    
    # Calculate average AUC score over folds
    return -ensemble_auc_score / nfolds

# Create an Optuna study and optimize the ensemble weights
ensemble_study = optuna.create_study()
ensemble_study.optimize(ensemble_objective, n_trials=200)
best_ensemble_weights = ensemble_study.best_trial.params

print('Best ensemble weights:', best_ensemble_weights)


# In[145]:


best_lgb_weight = best_ensemble_weights['lgb_weight']
best_xgb_weight = best_ensemble_weights['xgb_weight']
best_cb_weight = best_ensemble_weights['cb_weight']
best_rf_weight = best_ensemble_weights['rf_weight']
best_histgb_weight = best_ensemble_weights['histgb_weight']
best_adaboost_weight = best_ensemble_weights['adaboost_weight']

print('Best LGBM weight:', best_lgb_weight)
print('Best XGBoost weight:', best_xgb_weight)
print('Best CatBoost weight:', best_cb_weight)
print('Best RandomForest weight:', best_rf_weight)
print('Best HistGradientBoosting weight:', best_histgb_weight)
print('Best AdaBoost weight:', best_adaboost_weight)

# Now you can use the obtained weights in your prediction code
prediction = np.zeros(len(test))

for lgb_model, xgb_model, cb_model, rf_model, histgb_model, adaboost_model in zip(lgb_models, xgb_models, cb_models, rf_models, histgb_models, adaboost_models):
    prediction += (best_lgb_weight * lgb_model.predict_proba(test)[:, 1] +
                   best_xgb_weight * xgb_model.predict_proba(test)[:, 1] +
                   best_cb_weight * cb_model.predict_proba(test)[:, 1] +
                   best_rf_weight * rf_model.predict_proba(test)[:, 1] +
                   best_histgb_weight * histgb_model.predict_proba(test)[:, 1] +
                   best_adaboost_weight * adaboost_model.predict_proba(test)[:, 1]) / (best_lgb_weight + best_xgb_weight + best_cb_weight + best_rf_weight + best_histgb_weight + best_adaboost_weight)


# In[150]:


prediction = np.zeros(len(test))

for lgb_model, xgb_model, cb_model, rf_model, histgb_model, adaboost_model, gbm_model in zip(lgb_models, xgb_models, cb_models, rf_models, histgb_models, adaboost_models, gbm_models):
    prediction += (best_lgb_weight * lgb_model.predict_proba(test)[:, 1] +
                   best_xgb_weight * xgb_model.predict_proba(test)[:, 1] +
                   best_cb_weight * cb_model.predict_proba(test)[:, 1] +
                   best_rf_weight * rf_model.predict_proba(test)[:, 1] +
                   best_histgb_weight * histgb_model.predict_proba(test)[:, 1] +
                   best_adaboost_weight * adaboost_model.predict_proba(test)[:, 1]  )/ (best_lgb_weight + best_xgb_weight + best_cb_weight + best_rf_weight + best_histgb_weight + best_adaboost_weight)


# In[152]:


sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')
sample_submission_df['smoking'] = prediction
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()


#  <div style="border:#283618 2px solid ; background-color: #FEFAE0; padding: 20px;">
# <font face="Pacifico" color="#606C38" size="5"><b>If you find value in this notebook, I would greatly appreciate it if you could consider giving it a like, following me for more updates, and sharing it with others who might also benefit from it.</b></font>
# 
# <font face="Pacifico" color="#DDA15E" size="4"><i>Your support means a lot to me and helps me continue sharing insights and knowledge within the Kaggle community.</i></font>
# 
# <font face="Pacifico" color="#BC6C25" size="4"><b>Thank you!</b></font>
# </div>
