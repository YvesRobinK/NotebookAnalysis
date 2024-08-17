#!/usr/bin/env python
# coding: utf-8

# **Created by Yang Zhou**
# 
# **[PLAYGROUND S-3,E-23]📊EDA + Modeling📈**
# 
# **3 Oct 2023**

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">Binary Classification with a Software Defects Dataset</center>
# <p><center style="color:#949494; font-family: consolas; font-size: 20px;">Playground Series - Season 3, Episode 23</center></p>
# 
# ***
# 
# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">Insights and Tricks</center>
# - Note that when using AUC score to evaluate model effectiveness, remember to use the `predict_proba()` function.
# - There are some features that are highly correlated, such as `branchCount` and `v(g)`, `total_Opnd` and `total_Op`, `total_Op` and `n`.
# - `i` doesn't seem to be related to any other feature.
# - [Oscar Aguilar](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444784) mentions a mountain-climbing strategy used to improve the weights of different base models in the Ensemble model. That is, through greedy searching, find weights that make the Ensemble model score higher. He is currently the first place in this competition, congratulations.
# - [broccoli beef](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444640) provides a tip for converting specified values to Nan with the following code:
# ```python
# original = pd.read_csv('/kaggle/input/software-defect-prediction/jm1.csv',na_values=['?'])
# ```
# - To smooth the data, we can log them with the following code:
# ```python
# num_var = [column for column in train.columns if train[column].nunique() > 10]
# for column in num_var:
#     total[column] = total[column].apply(lambda x: np.log(x) if x > 0 else 0)
# ```
# 
# **Key Observation:**
# - When I added the original data to the baseline model, **it had a lower ROC score (0.77)**, which is about 0.02 lower than using only constructed data (0.79). At this point, only the feature (`loc`) has the highest ROC score: 0.7832.
# - There seems to be a large number of features that are not useful, i found in the feature importance of LGBM Baseline that only 2 features make the average ROC score the highest: `loc`, `IOBlank`. On top of this, adding other features will reduce the score.
# - Although `loc` is the most important feature in forecasting, when used alone, its LB score is about 0.02 lower than when using full features.
# 
# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">Version Detail</center>
# | Version | Description | Best Public Score |
# |---------|-------------|-----------------|
# | Version 7 | Change Params |  |
# | Version 6 | Add Features/Change Models Params | Not Improving |
# | Version 5 | Add NN | Not Improving |
# | Version 4 | Add RFECV | Not Improving |
# | Version 3 | Add Features Importance/Hill Climbing | 0.79056 |
# | Version 2 | Ensemble Baseline/Remove raw data | 0.78953 |
# | Version 1 | Autogluon Baseline | 0.66189 |

# # 0. Imports

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import math
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, RFECV

# Models
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

# NN
import torch
import torch.nn as nn
import torch.optim as optim
if torch.cuda.is_available():
    device = torch.device("cuda")  
else:
    device = torch.device("cpu")   
from torch.utils.data import DataLoader, TensorDataset

# Metrics 
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, f1_score, auc, confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Adjusting plot style

rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
mgt = Style.BRIGHT + Fore.MAGENTA
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL


# # 1. Load Data

# In[3]:


train = pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e23/sample_submission.csv')

origin = pd.read_csv('/kaggle/input/software-defect-prediction/jm1.csv',na_values=['?'])
origin.dropna(inplace=True)

train["is_generated"] = 1
test["is_generated"] = 1
origin["is_generated"] = 0

# Drop column id
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

train_total = pd.concat([train, origin], ignore_index=True)
train_total.drop_duplicates(inplace=True)

total = pd.concat([train_total, test], ignore_index=True)
# total = pd.concat([train, test], ignore_index=True)

print('The shape of the train data:', train.shape)
print('The shape of the test data:', test.shape)

print('The shape of the origin data:', origin.shape)
print('The shape of the total data:', total.shape)


# In[4]:


total.head(3)


# In[5]:


target = 'defects'

full_features = test.columns
num_var = [column for column in train.columns if train[column].nunique() > 10]


# # 2. EDA

# In[6]:


train.describe().T\
    .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
    .background_gradient(subset=['std'], cmap='Blues')\
    .background_gradient(subset=['50%'], cmap='BuGn')


# **Some observation:**
# 1. Unlike other variables, variable `e` has a much higher range of values than other variables. This variable needs to be standardized.

# According to the introduction of the competition, the data is interpreted as follows:
# 
# - `loc`: numeric - McCabe's line count of code
# - `v(g)`: numeric - McCabe "cyclomatic complexity"
# - `ev(g)`: numeric - McCabe "essential complexity"
# - `iv(g)`: numeric - McCabe "design complexity"
# - `n`: numeric - Halstead total operators + operands
# - `v`: numeric - Halstead "volume"
# - `l`: numeric - Halstead "program length"
# - `d`: numeric - Halstead "difficulty"
# - `i`: numeric - Halstead "intelligence"
# - `e`: numeric - Halstead "effort"
# - `b`: numeric - Halstead
# - `t`: numeric - Halstead's time estimator
# - `lOCode`: numeric - Halstead's line count
# - `lOComment`: numeric - Halstead's count of lines of comments
# - `lOBlank`: numeric - Halstead's count of blank lines
# - `lOCodeAndComment`: numeric
# - `uniq_Op`: numeric - unique operators
# - `uniq_Opnd`: numeric - unique operands
# - `total_Op`: numeric - total operators
# - `total_Opnd`: numeric - total operands
# - `branchCount`: numeric - percentage of the flow graph
# - `defects`: {false, true} - module has/has not one or more reported defects
# 

# In[7]:


def summary(df):
    sum = pd.DataFrame(df.dtypes, columns=['dtypes'])
    sum['missing#'] = df.isna().sum()
    sum['missing%'] = (df.isna().sum())/len(df)
    sum['uniques'] = df.nunique().values
    sum['count'] = df.count().values
    return sum

summary(total).style.background_gradient(cmap='Blues')


# In[8]:


total['v(g)'].unique()


# **Some observation:**
# 
# 1. It seems that most of the variables are numeric, only target and is_generated are categorical.
# 2. Five missing values appear in the data, I will delete them directly.

# In[9]:


summary(test).style.background_gradient(cmap='Blues')


# ## Correlation

# In[10]:


corr_matrix = total[num_var].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='Blues', fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9} )
plt.title('Correlation Matrix', fontsize=15)
plt.show()


# **Some observation:**
# 1. Variable `i` doesn't seem to be related to any other variable.
# 2. There are some variables that are highly correlated. For example: 
#     - `branchCount` vs `v(g)`.
#     - `total_Opnd` vs `n` , `v` and `total_Op`.
#     - `n` vs `v`

# ## Scatterplot of highly correlated variables

# In[11]:


fig, axes = plt.subplots(3, 2, figsize = (20,12))

sns.scatterplot(ax = axes[0][0], data = total, x = 'uniq_Op', y = 'uniq_Opnd', hue = target)
sns.scatterplot(ax = axes[0][1], data = total, x = 'total_Opnd', y = 'n', hue = target)
sns.scatterplot(ax = axes[1][0], data = total, x = 'total_Opnd', y = 'b', hue = target)
sns.scatterplot(ax = axes[1][1], data = total, x = 'total_Op', y = 'total_Opnd', hue = target)
sns.scatterplot(ax = axes[2][0], data = total, x = 'total_Opnd', y = 'lOCode', hue = target)
sns.scatterplot(ax = axes[2][1], data = total, x = 'b', y = 'n', hue = target)


# **Some observation:**
# 
# 1. It can be noted that in features with high correlation, the higher the value, the greater the probability that the defect is true.

# ## Distribution of Target

# In[12]:


sns.countplot(data=train,x=target);


# **Some observation:**
# 
# 1. The target variable has a data imbalance that may require downsampling.

# ## Distribution of Numerical Variables

# In[13]:


num_rows = len(num_var)
num_cols = 3 

total_plots = num_rows * num_cols
plt.figure(figsize=(14, num_rows * 2.5))

for idx, col in enumerate(num_var):
    plt.subplot(num_rows, num_cols, idx % total_plots + 1)
    sns.violinplot(x=target, y=col, data=total)
    plt.title(f"{col} Distribution for target")

plt.tight_layout()
plt.show()


# **Some observation:**
# 1. When the variable `ev(g)` is small, the probability of `defect` being False is greater.
# 2. Similarly, in the variable `I`, the larger the I, the more likely the `defect` is to be True.

# ## Distribution of Numeric Variables in Train/Test set

# In[14]:


df = pd.concat([train[num_var].assign(Source = 'Train'), 
                test[num_var].assign(Source = 'Test')], 
               axis=0, ignore_index = True);

fig, axes = plt.subplots(len(num_var), 3 ,figsize = (16, len(num_var) * 4.2), 
                         gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]});

for i,col in enumerate(num_var):
    ax = axes[i,0];
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', ax = ax, linewidth = 2.1)
    ax.set_title(f"\n{col}",fontsize = 9, fontweight= 'bold');
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75);
    ax.set(xlabel = '', ylabel = '');
    ax = axes[i,1];
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,saturation = 0.90, linewidth = 0.90, fliersize= 2.25, color = '#037d97',
                ax = ax);
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Train",fontsize = 9, fontweight= 'bold');

    ax = axes[i,2];
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                saturation = 0.6, linewidth = 0.90, color = '#E4591E',
                ax = ax); 
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Test",fontsize = 9, fontweight= 'bold');

plt.tight_layout();
plt.show();


# # 3. Features Engineering

# In[15]:


# Mapping target to numbers

total[target] = total[target].map({False:0,True:1})


# In[16]:


for column in num_var:
    total[column] = total[column].apply(lambda x: np.log1p(x) if x > 0 else 0)


# In[17]:


def features_engineering(df):
    df['mean_bnv'] = (df['n'] + df['v'] + df['b']) /3;
    df['mean_uniqOpOpend'] = (df['uniq_Op'] + df['uniq_Opnd']) /2;
    df['mean_totOpOpend'] = (df['total_Op'] + df['total_Opnd']) /2;
    df['mean_brcntvg'] = (df['branchCount'] + df['v(g)']) / 2;
    return df

features_engineering(total)


# ## Baseline Model

# In[18]:


df_train = total[total[target].notna()]

df_test = total[total[target].isna()]
df_test.drop(target,axis=1,inplace=True)


# In[19]:


lgbm_baseline = LGBMClassifier(n_estimators=1000,
                     max_depth=10,
                     random_state=42)

roc_results = pd.DataFrame(columns=['Selected_Features', 'ROC'])

def evaluation(df, select_features, note):
    global roc_results
    
    X = df[select_features]
    Y = df[target]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    roc_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        
        lgbm_baseline.fit(X_train, y_train)
        y_hat = lgbm_baseline.predict_proba(X_test)[:, 1] 
        roc = roc_auc_score(y_test, y_hat)
        roc_scores.append(roc)
    
    average_roc = np.mean(roc_scores)
    new_row = {'Selected_Features': note, 'ROC': average_roc}
    roc_results = pd.concat([roc_results, pd.DataFrame([new_row])], ignore_index=True)

    print('====================================')
    print(note)
    print("Average ROC:", average_roc)
    print('====================================')
    return average_roc


# In[20]:


evaluation(df=df_train,select_features=full_features,note='Baseline')


# ## Features Importance

# In[21]:


def f_importance_plot(f_imp):
    fig = plt.figure(figsize=(12, 0.20*len(f_imp)))
    plt.title(f'Feature importances', size=16, y=1.05, 
              fontweight='bold')
    a = sns.barplot(data=f_imp, x='imp', y='feature', linestyle="-", 
                    linewidth=0.5, edgecolor="black",palette='GnBu')
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11)
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.tight_layout()
    plt.show()


# In[22]:


clf = LGBMClassifier()
clf.fit(df_train.drop(target,axis=1), df_train[target])

f_imp_df = pd.DataFrame({'feature': df_train.drop(target,axis=1).columns, 'imp': clf.feature_importances_})
f_imp_df.sort_values(by='imp',ascending=False,inplace=True)
f_importance_plot(f_imp_df)


# In[23]:


# %%time
# best_score = 0
# best_feature_num = 0
# for i in range(1,f_imp_df.shape[0]):
#     feature = f_imp_df.head(i).feature.to_list()
#     # print(f'Trying top {i} features...')
#     score = evaluation(df=df_train,select_features=feature,note=f'Top {i} Features')
#     if score > best_score:
#         best_score = score
#         best_feature_num = i


# In[24]:


best_feature_num = 1 
best_score = 0.7753025559293826
print(f'Best feature number is Top {best_feature_num}, Best score is {best_score}')


# In[25]:


best_features = f_imp_df.head(best_feature_num).feature.to_list()


# In[26]:


evaluation(df=df_train,select_features=best_features,note='Best Features')


# ## Features Selection: RFECV

# In[27]:


# clf = LGBMClassifier()
# rfe = RFECV(clf)
# X_rfe = rfe.fit_transform(df_train.drop(target,axis=1), df_train[target])
# clf.fit(X_rfe,df_train[target])

# cols = list(df_train.drop(target,axis=1).columns)
# temp = pd.Series(rfe.support_,index = cols)
# featureRFE = temp[temp==True].index.tolist()


# In[28]:


featureRFE = ['loc', 'v(g)', 'iv(g)', 'n', 'v', 'd', 'i', 'e', 'b', 'lOCode', 'lOComment', 'lOBlank', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount', 'mean_uniqOpOpend']
print("Selected features: ",featureRFE)


# In[29]:


evaluation(df=df_train,select_features=featureRFE,note='Feature RFE')


# # 4. Modeling

# In[30]:


X = df_train.drop(target,axis=1)

# X = df_train[featureRFE]
Y = df_train[target]

# df_pred = df_test[featureRFE]
df_pred = df_test


# `Mountain-climbing strategy`: 
# 
# Used to improve the weights of different base models in the Ensemble model. That is, through greedy searching, find weights that make the Ensemble model score higher. 
# 
# Code Source: [Oscar Aguilar](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444784)

# In[31]:


# Source: https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444784
# Autor: OSCAR AGUILAR

def hill_climbing(x, y, x_test):

    # Evaluating oof predictions
    scores = {}
    for col in x.columns:
        scores[col] = roc_auc_score(y, x[col])

    # Sorting the model scores
    scores = {k: v for k, v in sorted(scores.items(), key = lambda item: item[1], reverse = True)}

    # Sort oof_df and test_preds
    x = x[list(scores.keys())]
    x_test = x_test[list(scores.keys())]

    STOP = False
    current_best_ensemble = x.iloc[:,0]
    current_best_test_preds = x_test.iloc[:,0]
    MODELS = x.iloc[:,1:]
    weight_range = np.arange(-0.5, 0.51, 0.01) 
    history = [roc_auc_score(y, current_best_ensemble)]
    j = 0

    while not STOP:
        j += 1
        potential_new_best_cv_score = roc_auc_score(y, current_best_ensemble)
        k_best, wgt_best = None, None
        for k in MODELS:
            for wgt in weight_range:
                potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = roc_auc_score(y, potential_ensemble)
                if cv_score > potential_new_best_cv_score:
                    potential_new_best_cv_score = cv_score
                    k_best, wgt_best = k, wgt

        if k_best is not None:
            current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
            current_best_test_preds = (1 - wgt_best) * current_best_test_preds + wgt_best * x_test[k_best]
            MODELS.drop(k_best, axis = 1, inplace = True)
            if MODELS.shape[1] == 0:
                STOP = True
            history.append(potential_new_best_cv_score)
        else:
            STOP = True

    hill_ens_pred_1 = current_best_ensemble
    hill_ens_pred_2 = current_best_test_preds

    return [hill_ens_pred_1, hill_ens_pred_2]


# In[32]:


def training_model(clf,X_train,Y_train,X_test,Y_test,df_pred):
    clf.fit(X_train,Y_train)
    
    cv_pred = clf.predict_proba(X_test)[:, 1]
    cv_score = roc_auc_score(Y_test, cv_pred)
    pred = clf.predict_proba(df_pred)[:, 1]
    
    return cv_pred,cv_score,pred


# In[33]:


models = {
    'RF': RandomForestClassifier(
                                  n_estimators = 500, 
                                   max_depth = 7,
                                   min_samples_split = 15,
                                   min_samples_leaf = 10),
    'ET': ExtraTreesClassifier(
                               n_estimators = 500, 
                                 max_depth = 7,
                                 min_samples_split = 15,
                                 min_samples_leaf = 10),
    'Hist': HistGradientBoostingClassifier(
                                l2_regularization = 0.01,
                                 early_stopping = False,
                                 learning_rate = 0.01,
                                 max_iter = 500,
                                 max_depth = 5,
                                 max_bins = 255,
                                 min_samples_leaf = 15,
                                 max_leaf_nodes = 10),
    'LGBM': LGBMClassifier(
                            objective = 'binary',
                             n_estimators = 500,
                             max_depth = 7,
                             learning_rate = 0.01,
                             num_leaves = 20,
                             reg_alpha = 3,
                             reg_lambda = 3,
                             subsample = 0.7,
                             colsample_bytree = 0.7),
    'XGB': XGBClassifier(objective = 'binary:logistic',
                           tree_method = 'hist',
                           colsample_bytree = 0.7, 
                           gamma = 2, 
                           learning_rate = 0.01, 
                           max_depth = 7, 
                           min_child_weight = 10, 
                           n_estimators = 500, 
                           subsample = 0.7),
    'Cat': CatBoostClassifier(
                             loss_function = 'Logloss',
                                iterations = 500,
                                learning_rate = 0.01,
                                depth = 7,
                                random_strength = 0.5,
                                bagging_temperature = 0.7,
                                border_count = 30,
                                l2_leaf_reg = 5,
                                verbose = False, 
                                task_type = 'CPU')
}

model_preds = {} # To save predctions from each model


# In[34]:


ens_cv_scores, ens_preds = list(), list()
hill_ens_cv_scores, hill_ens_preds =  list(), list()

sk = RepeatedStratifiedKFold(n_splits = 30, n_repeats = 1, random_state = 42)
for i, (train_idx, test_idx) in enumerate(sk.split(X, Y)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    
    print('----------------------------------------------------------')
    
    for model_name, model in models.items():
        model_pred, model_score, model_pred_test = training_model(model, X_train, Y_train, X_test, Y_test, df_pred)
        print(f'Fold {i+1} ==> {model_name} oof ROC-AUC score is ==> {model_score}')
        
        model_preds[model_name] = (model_pred, model_pred_test)
    
    # Ensemble   
    ens_pred_1 = sum(pred[0] for pred in model_preds.values()) / len(models)
    ens_pred_2 = sum(pred[1] for pred in model_preds.values()) / len(models)
    
    ens_score_fold = roc_auc_score(Y_test, ens_pred_1)
    ens_cv_scores.append(ens_score_fold)
    ens_preds.append(ens_pred_2)
    
    # Hill Climbing Ensemble
    x = pd.DataFrame({model_name: model_preds[model_name][0] for model_name in models})
    y = Y_test

    x_test = pd.DataFrame({model_name: model_preds[model_name][1] for model_name in models})

    hill_results = hill_climbing(x, y, x_test)

    hill_ens_score_fold = roc_auc_score(y, hill_results[0])
    hill_ens_cv_scores.append(hill_ens_score_fold)
    hill_ens_preds.append(hill_results[1])

    print(f'Fold {i+1} ==> Hill Climbing Ensemble oof ROC-AUC score is ==> {hill_ens_score_fold}')
    print(f'Fold {i+1} ==> Average Ensemble oof ROC-AUC score is ==> {ens_score_fold}')

    print('The hill climbing ensemble oof ROC-AUC score over the 5-folds is', np.mean(hill_ens_cv_scores))


# In[35]:


hill_preds_test = pd.DataFrame(hill_ens_preds).apply(np.mean, axis = 0)

hill_ensemble_submission = sample_submission.copy()

hill_ensemble_submission['defects'] = hill_preds_test
hill_ensemble_submission.to_csv('Ensemble_submission.csv', index = False)


# In[36]:


hill_ensemble_submission.head()


# In[37]:


roc_results


# # 5. Hyperparameter optimization of individual models

# ## XGBoost

# In[38]:


params_xgb = {'n_estimators': 1000, 
              'learning_rate': 0.01752354328845971, 
              'booster': 'gbtree', 
              'lambda': 0.08159630121074074, 
              'alpha': 0.07564858712175693, 
              'subsample': 0.5065979400270813, 
              'colsample_bytree': 0.6187340851873067, 
              'max_depth': 4, 
              'min_child_weight': 5, 
              'eta': 0.2603059902806757, 
              'gamma': 0.6567360773618207}


# In[39]:


xgb_opt = XGBClassifier(**params_xgb).fit(X,Y)
xgb_preds = xgb_opt.predict_proba(df_pred)[:,1]

xgb_opt_submission = pd.DataFrame({'id': sample_submission['id'], 'defects': xgb_preds})
xgb_opt_submission.to_csv('xgb_opt_submission.csv',index=False)


# In[40]:


xgb_opt_submission.head(3)


# ## LGBM

# In[41]:


params_lgbm = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'goss',
    'random_state': 42,
    'colsample_bytree': 0.50,
    'subsample': 0.70,
    'learning_rate': 0.0625,
    'max_depth': 6,
    'n_estimators': 1000,
    'num_leaves': 110, 
    'reg_alpha': 0.0001,
    'reg_lambda': 2.0,
    'verbose': -1,
}


# In[42]:


lgb_opt = LGBMClassifier(**params_lgbm).fit(X,Y)
lgb_preds = lgb_opt.predict_proba(df_pred)[:,1]

lgb_opt_submission = pd.DataFrame({'id': sample_submission['id'], 'defects': lgb_preds})
lgb_opt_submission.to_csv('lgb_opt_submission.csv',index=False)


# In[43]:


lgb_opt_submission.head(3)


# # 6. NN by pytorch

# In[44]:


train_set = torch.tensor(X.values,dtype=torch.float32).to(device)
train_target = torch.tensor(Y.values,dtype=torch.float32).to(device)

pred_set = torch.tensor(df_pred.values,dtype=torch.float32).to(device)


# In[45]:


train_set.shape


# In[46]:


input_dim = train_set.shape[1]


# In[47]:


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  
        self.dropout1 = nn.Dropout(0.2)  
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.sigmoid(x)


# In[48]:


model = NN(input_dim).to(device)

criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_history = []

epochs = 3000
for epoch in range(3000):
    optimizer.zero_grad()
    outputs = model(train_set).squeeze()
    
    loss = criterion(outputs, train_target)
    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
    loss.backward()    
    optimizer.step()
train_steps = range(1, len(loss_history) + 1)


# In[49]:


plt.plot(train_steps, loss_history, label='Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Time')
plt.show()


# ## Prediction

# In[50]:


model.eval()
with torch.no_grad():
    probs = model(pred_set)
    
probs = probs.cpu().flatten().numpy()
nn_submission = pd.DataFrame({'id': sample_submission['id'], 'defects': probs})
nn_submission.to_csv('nn_submission.csv',index=False)


# In[51]:


nn_submission.head(3)


# # 7. Autogluon Baseline (Removed after version 22)
# 
# **Key Points:**
# 
# According to Autogluon's predictions, `LightGBM` model performed better, achieving an accuracy of `0.81` on the test set.

# In[52]:


# !pip install autogluon


# In[53]:


# from autogluon.tabular import TabularDataset, TabularPredictor

# predictor = TabularPredictor(label=target,).fit(df_train)
# preds = predictor.predict_proba(df_test)

# auto_submission = sample_submission.copy()
# auto_submission['defects'] = preds[1].values

# auto_submission.to_csv('auto_submission.csv',index=False)
# auto_submission.head()

