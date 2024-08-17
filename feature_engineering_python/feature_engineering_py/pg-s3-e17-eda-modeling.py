#!/usr/bin/env python
# coding: utf-8

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


# <h1 style="font-family:verdana;"> <center> 🎯PS3 E17: Binary Classification of Machine Failures: EDA and Baseline models🚀</center> </h1>
# 
# ***

# ### Version 8 Changes:
# 
# ***1. Better visuals and more plots, comes from: https://www.kaggle.com/code/yantxx/xgboost-binary-classifier-machine-failure. I need to admit that his aesthetic is much better than mine.***
# 
# ***2. Some hyperparameters.***
# 
# ***3. A simple RFE-CV to select features, the code is a bit ugly here...Because I finished it in the middle of a class...jeje***
# 
# ___
# 
# 

# ![image.png](attachment:2b7e6897-2851-4681-8407-2870701d38c8.png)

# # Imports
# ___

# In[2]:


import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna.integration.lightgbm as lgbo

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import math

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV


from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance

# Metrics
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_log_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import warnings
warnings.filterwarnings("ignore")

# Models
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


# In[3]:


# Props to @sergiosaharovskiy

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


# In[4]:


submission = pd.read_csv("/kaggle/input/playground-series-s3e17/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s3e17/train.csv")
origin = pd.read_csv("/kaggle/input/machine-failure-predictions/machine failure.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e17/test.csv")


# #### ***We used those original datas from this url: https://www.kaggle.com/datasets/dineshmanikanta/machine-failure-predictions***
# 
# #### ***It's always helpful when we have more datas.***

# In[5]:


train['generated'] = 1
origin['generated'] = 0
test['generated'] = 1


# In[6]:


train = pd.concat([train, origin], axis = 0).reset_index(drop = True)
train.drop(['id','UDI','Product ID'],axis = 1, inplace = True)
test.drop(['id','Product ID'],axis = 1, inplace = True)
train.head()


# In[7]:


print('The shape of dataset train is :', train.shape)
print('The shape of dataset test is:', test.shape)


# # EDA
# ___

# #### ***We're going to present some distribution plots.***
# 
# #### ***First of all, let's check the information of the dataset.***

# In[8]:


target = 'Machine failure'

num_var = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

bin_var = [
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]

cat_var = ['Type']


# In[9]:


train.describe().T\
        .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
        .background_gradient(subset=['std'], cmap='Blues')\
        .background_gradient(subset=['50%'], cmap='BuGn')


# In[10]:


# summary of train data

def summary(df):
    print(f'data shape is: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ

summary(train)


# In[11]:


color = ['#d44c46', '#eed5b7']

def plot_pair(df_train,num_var,target,plotname,color = ['#d44c46', '#eed5b7']):
    '''
    Funtion to make a pairplot:
    df_train: total data
    num_var: a list of numeric variable
    target: target variable
    '''
    g = sns.pairplot(data=df_train, x_vars=num_var, y_vars=num_var, hue=target, corner=True, palette=color)
    g._legend.set_bbox_to_anchor((0.8, 0.7))
    g._legend.set_title(target)
    g._legend.loc = 'upper center'
    g._legend.get_title().set_fontsize(14)
    for item in g._legend.get_texts():
        item.set_fontsize(14)

    plt.suptitle(plotname, ha='center', fontweight='bold', fontsize=25, y=0.98)
    plt.show()

plot_pair(train,num_var,target,plotname = 'Scatter Matrix with Target')


# In[12]:


def plot_histograms(df_train, df_test, original, n_cols=3):
    
    n_rows = (len(df_train.columns) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df_train.columns.tolist()):
        if var_name != 'generated':
            ax = axes[i]
            sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
            if var_name != 'Machine failure':
                sns.distplot(df_test[var_name], kde=True, ax=ax, label='Test')
            sns.distplot(original[var_name], kde=True, ax=ax, label='Original')
            ax.set_title(f'{var_name} Distribution (Train vs Test)')
            ax.legend()
            
    plt.suptitle(f'\n\n', ha='center', fontweight='bold', fontsize=25, y=0.98)
    plt.tight_layout()
    plt.show()
        
plot_histograms(train[num_var], test[num_var], origin[num_var], n_cols=3)


# In[13]:


columns = [i for i in train.columns if i not in num_var]

def plot_count(df,columns,n_cols, plotname, color=['#d44c46', '#eed5b7','#e5855d']):
    '''
    # Function to genear countplot
    df: total data
    columns: category variables
    n_cols: num of cols
    '''
    n_rows = (len(columns) - 1) // n_cols + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(17, 4 * n_rows))
    ax = ax.flatten()
    
    for i, column in enumerate(columns):
        sns.countplot(data=df, x=column, ax=ax[i], palette=color)

        # Titles
        ax[i].set_title(f'{column} Counts', fontsize=18)
        ax[i].set_xlabel(None, fontsize=16)
        ax[i].set_ylabel(None, fontsize=16)

        ax[i].tick_params(axis='x', rotation=0)

        for p in ax[i].patches:
            value = int(p.get_height())
            ax[i].annotate(f'{value:.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                           ha='center', va='bottom', fontsize=9)

    ylim_top = ax[i].get_ylim()[1]
    ax[i].set_ylim(top=ylim_top * 1.1)

    for i in range(len(columns), len(ax)):
        ax[i].axis('off')

    fig.suptitle(plotname, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
plot_count(train,columns,3,plotname = 'Categorical and Binary Features')


# In[14]:


plot_count(train[train['Machine failure']==1],columns,3, plotname = 'Categorical and Binary Features (Target = 1)')


# In[15]:


'''features = [f for f in train.columns if f in num_var]
features = columns


def boxplot(df,obj,plotname, color):
    n_rows = len(columns) // 3
    if len(columns) % 3:
        n_rows += 1
        
    fig, ax = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
    ax = ax.ravel()    
    unique_target = df[obj].unique()
    n_categories = len(unique_target)
    colors = sns.color_palette(color, n_categories)
    
    for i, column in enumerate(columns):
        data = [df[df[obj] == target][column] for target in unique_target]
        sns.boxplot(data=data, ax=ax[i], palette=colors)

        ax[i].set_title(f'{column} Distribution', fontsize=22)
        ax[i].set_xlabel(None, fontsize=18)
        ax[i].set_ylabel(None, fontsize=18)   
        
    for i in range(len(columns), len(ax)):
        ax[i].axis('off')

    fig.suptitle(plotname, ha='center', fontweight='bold', fontsize=20, y=1)
    plt.tight_layout(pad=1.0)
    plt.show()    
    
boxplot(df = train,obj = 'TWF',plotname = 'Feature Distributions by TWF',color=['#d44c46', '#eed5b7','#e5855d'])'''


# In[16]:


corr = train.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')


# 
# #### ***The skewness of some variables is large and the proportion of positive and negative samples differs very much.***
# #### ***Therefore, we will need to use some oversampling methods.***

# # Data Engineering
# 
# ___

# ### ***6 categorical variables in the dataset：`Type`,`TWF`,`HDF`,`PWF`,`OSF`,`RNF`,`generated`***
# ### ***We don't need to transfomer all of them because most of them are binary variable.***

# In[17]:


Y = train['Machine failure']
X = train.drop('Machine failure', axis=1)

X.head()


# In[18]:


var_cate = ['Type','TWF','HDF','PWF','OSF','RNF','generated']


# #### ***Here i creat some new features, the original idea is from discussion.***

# In[19]:


# onehot encoding and normalization
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def data_engineering(df,var_cate):
    # rename columns
    for i in df.columns:
        df.columns = df.columns.str.replace('[\[\]]', '', regex=True)

    
    # label encoding for Type colum
    df['Type']=df['Type'].replace({"L":0,"M":1,"H":2})

    # new features
    df['Power'] = df['Torque Nm'] * df['Rotational speed rpm']
    df['Temp_ratio'] = df['Process temperature K'] / df['Air temperature K']
    df['Process temperature C'] = df['Process temperature K'] - 273.15
    df["Air temperature C"] = df["Air temperature K"] - 273.15
    df["temp_C_ratio"] = df["Process temperature C"] / df["Air temperature C"]
    df["Failure Sum"] = (df["TWF"] + df["HDF"] + df["PWF"] + df["OSF"] + df["RNF"])
    df["tool_wear_speed"] = df["Tool wear min"] * df["Rotational speed rpm"]
    df["torque wear ratio"] = df["Torque Nm"] / (df["Tool wear min"] + 0.0001)
    df["torque times wear"] = df["Torque Nm"] * df["Tool wear min"]

    # normalization
    scaler = StandardScaler()
    for i in [i for i in df.columns if i not in ['Machine failure'] and i not in var_cate]:
        df[i] = scaler.fit_transform(df[[i]])
    
    return df
    


# In[20]:


X = data_engineering(X,var_cate)
test = data_engineering(test,var_cate)


# # Oversampling
# 
# #### ***Someone in the Discussion section suggested that Oversampling doesn't seem to do much, but I don't have much choice. In any case, more data should make the model more robust.***
# ___

# In[21]:


from imblearn.over_sampling import SMOTE

print('Shape of train data before oversampling:',X.shape)
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
print('Shape of train data after oversampling:',X_res.shape)


# #### ***Now we are ready for modeling.***

# # Features Selection

# In[22]:


'''from tqdm import tqdm
# models

results_df = pd.DataFrame()

lgb_md = LGBMClassifier()
xgb_md = XGBClassifier()


models = [lgb_md,xgb_md]

def perform_feature_selection(model, X, y):
    rfecv = RFECV(model, 
            step=1,
            min_features_to_select=10,
            cv=2,
            scoring='roc_auc', 
            n_jobs=-1)
    with tqdm(total=X.shape[1], desc="Feature Selection", leave=False) as pbar:
        rfecv.fit(X, y)
        pbar.update(rfecv.support_.sum())
    selected_features = rfecv.support_
    feature_ranking = rfecv.ranking_
    return selected_features, feature_ranking


# use lgb model to select features
lgb_selected_features, lgb_feature_ranking = perform_feature_selection(lgb_md, X,Y)

# use xgb model to select features
xgb_selected_features, xgb_feature_ranking = perform_feature_selection(xgb_md, X,Y)'''


# In[23]:


'''rank_feature = pd.DataFrame({'Feature Name': X.columns, 'LGB': lgb_feature_ranking, 'XGB': xgb_feature_ranking})
rank_feature['mean'] = (rank_feature['LGB'] + rank_feature['XGB'])/2
rank_feature'''


# ### ***I will put the result here: 'Air temperature C','Process temperature C','OSF' are features i'm going to delete because they are not performing well in REFCV . You can run the code if want.***
# 

# In[24]:


X


# In[25]:


columns_to_exclude = ['Air temperature C', 'Process temperature C', 'RNF','Type','Power',]
X = X.drop(columns=columns_to_exclude)
test = test.drop(columns=columns_to_exclude)


# # Modeling

# ### ***Here I selected some models and try to optimizar them.***
# 
# ### ***And also an Ensemble model.***

# ## Search best params for classifers
# 
# ### ***The process of finding hyperparameters is in Version 5.***

# ### LGBM

# In[26]:


import optuna

'''def obj_lgb(trial):
    params = {
        'random_state': 23,
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 10,50),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300)
    }
    
    lgb_md = LGBMClassifier(**params)  
    lgb_md.fit(X_skf_train, y_skf_train) 
    y_skf_pred = lgb_md.predict(X_skf_val)
    roc_auc = roc_auc_score(y_skf_val, y_skf_pred)

    return roc_auc

study = optuna.create_study(direction='maximize')
study.optimize(obj_lgb, n_trials=100)
display(study)

print('Best trial:', study.best_trial.params)
print('Best ROC AUC:', study.best_trial.value)

lgb_optimized_params = study.best_trial.params

## Additional params
lgb_optimized_params['objective'] = 'binary'
lgb_optimized_params['metric'] = 'binary_logloss'

lgb_md = LGBMClassifier(**lgb_optimized_params)
lgb_md
'''


# In[27]:


lgb_md = LGBMClassifier(learning_rate=0.0124415817896377, max_depth=37,
               metric='binary_error', min_child_samples=102, num_leaves=249,
               objective='binary', reg_alpha=0.00139174509988134,
               reg_lambda=0.000178964551019674, subsample=0.421482143660471,
               boosting_type = 'gbdt',subsample_freq = 4)


# ### XGB

# In[28]:


'''def obj_xgb(trial):
    params = {
        'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),    
    }
    
    xgb_md = XGBClassifier(reg_alpha = 0.18727857702097278,reg_lambda = 0.77217672456579,
                           learning_rate = 0.043011675696849064, max_depth = 15, subsample = 0.8370545840097189,
                             random_state = 23,min_child_weight = 60,**params)  
    xgb_md.fit(X_skf_train, y_skf_train) 
    y_skf_pred = xgb_md.predict(X_skf_val)
    roc_auc = roc_auc_score(y_skf_val, y_skf_pred)

    return roc_auc

study = optuna.create_study(direction='maximize')
study.optimize(obj_xgb, n_trials=50)
display(study)

print('Best trial:', study.best_trial.params)
print('Best ROC AUC:', study.best_trial.value)

xgb_optimized_params = study.best_trial.params

## Additional params

xgb_md = XGBClassifie(**xgb_optimized_params)
xgb_md '''


# In[29]:


xgb_md =  XGBClassifier(reg_alpha = 0.18727857702097278,
                        alpha = 0.0000162103492458353,
                        learning_rate = 0.00349356650247156, max_depth = 15, 
                        subsample = 0.8370545840097189, objective = 'binary:logistic',
                        random_state = 23,min_child_weight = 2, n_jobs = -1,
                        eta = 0.05, colsample_bytree = 0.244618079894501,booster = 'gbtree')


# In[30]:


gb_md = GradientBoostingClassifier(n_estimators = 500, 
                                   max_depth = 7, 
                                   learning_rate = 0.01,
                                   min_samples_split = 10, 
                                   min_samples_leaf = 20)

hist_md = HistGradientBoostingClassifier(l2_regularization = 0.01,
                                             early_stopping = False,
                                             learning_rate = 0.01,
                                             max_iter = 1000,
                                             max_depth = 15,
                                             max_bins = 255,
                                             min_samples_leaf = 30,
                                             max_leaf_nodes = 30)


# In[31]:


gb_cv_scores, gb_preds = list(), list()
hist_cv_scores, hist_preds = list(), list()
lgb_cv_scores, lgb_preds = list(), list()
xgb_cv_scores, xgb_preds = list(), list()
ens_cv_scores, ens_preds = list(), list()

skf = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True)
    
for i, (train_ix, test_ix) in enumerate(skf.split(X, Y)):
        
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
    print('---------------------------------------------------------------')
    
    ## GradientBoosting
    
    gb_md = GradientBoostingClassifier(n_estimators = 500, 
                                   max_depth = 7, 
                                   learning_rate = 0.01,
                                   min_samples_split = 10, 
                                   min_samples_leaf = 20).fit(X_train, Y_train) 
    
    gb_pred_1 = gb_md.predict_proba(X_test[X_test['generated'] == 1])[:, 1]
    gb_pred_2 = gb_md.predict_proba(test)[:, 1]
            
    gb_score_fold = roc_auc_score(Y_test[X_test['generated'] == 1], gb_pred_1)
    gb_cv_scores.append(gb_score_fold)
    gb_preds.append(gb_pred_2)
    
    print('Fold', i+1, '==> GradientBoositng oof ROC-AUC score is ==>', gb_score_fold)

    ## HistGradientBoosting 
       
    hist_md = HistGradientBoostingClassifier(l2_regularization = 0.01,
                                             early_stopping = False,
                                             learning_rate = 0.01,
                                             max_iter = 1000,
                                             max_depth = 15,
                                             max_bins = 255,
                                             min_samples_leaf = 30,
                                             max_leaf_nodes = 30).fit(X_train, Y_train)
    
    hist_pred_1 = hist_md.predict_proba(X_test[X_test['generated'] == 1])[:, 1]
    hist_pred_2 = hist_md.predict_proba(test)[:, 1]

    hist_score_fold = roc_auc_score(Y_test[X_test['generated'] == 1], hist_pred_1)
    hist_cv_scores.append(hist_score_fold)
    hist_preds.append(hist_pred_2)
    
    print('Fold', i+1, '==> HistGradient oof ROC-AUC score is ==>', hist_score_fold)
    
        
    ## LightGBM
        
    lgb_md.fit(X_train, Y_train)
    
    lgb_pred_1 = lgb_md.predict_proba(X_test[X_test['generated'] == 1])[:, 1]
    lgb_pred_2 = lgb_md.predict_proba(test)[:, 1]

    lgb_score_fold = roc_auc_score(Y_test[X_test['generated'] == 1], lgb_pred_1)    
    lgb_cv_scores.append(lgb_score_fold)
    lgb_preds.append(lgb_pred_2)
    
    print('Fold', i+1, '==> LightGBM oof ROC-AUC score is ==>', lgb_score_fold)
        
    ## XGBoost 
        
    xgb_md.fit(X_train, Y_train)
    
    xgb_pred_1 = xgb_md.predict_proba(X_test[X_test['generated'] == 1])[:, 1]
    xgb_pred_2 = xgb_md.predict_proba(test)[:, 1]

    xgb_score_fold = roc_auc_score(Y_test[X_test['generated'] == 1], xgb_pred_1)    
    xgb_cv_scores.append(xgb_score_fold)
    xgb_preds.append(xgb_pred_2)
    
    print('Fold', i+1, '==> XGBoost oof ROC-AUC score is ==>', xgb_score_fold)

    ## Ensemble 
    
    ens_pred_1 = gb_pred_1 + hist_pred_1 + lgb_pred_1 + xgb_pred_1
    ens_pred_2 = gb_pred_2 + hist_pred_2 + lgb_pred_2 + xgb_pred_2
    
    ens_score_fold = roc_auc_score(Y_test[X_test['generated'] == 1], ens_pred_1)
    ens_cv_scores.append(ens_score_fold)
    ens_preds.append(ens_pred_2)
    
    print('Fold', i+1, '==> Ensemble oof ROC-AUC score is ==>', ens_score_fold)


# In[32]:


'''
# Simple Baselines models tests

from tqdm import tqdm

model_table = pd.DataFrame(columns = ['Model Name', 'Score'])
def model_accuracy(model,model_name,X,y):
    print('Starting Iteration for',model_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores =[]
    for nfold, (train_idx, valid_idx) in enumerate(tqdm(kf.split(X, y), total=5)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
        # Train the model
        model.fit(X_train, y_train)

        # Predict on the validation set and calculate MAE
        y_pred = model.predict_proba(X_valid)[:,1]
        roc = roc_auc_score(y_valid, y_pred)
        
        print(f'Iteration : {nfold+1}  ROC AUC: {roc:.5f}')
        
        roc_auc_scores.append(roc)
    
    i=len(model_table)
    model_table.loc[i,'Model Name']=model_name
    model_table.loc[i,'Score']=np.mean(roc_auc_scores)


models = [
      ('CatBoost Classifier',CatBoostClassifier(verbose= False)),
      ('RandomForestClassifier', RandomForestClassifier(random_state=1)),
      ('KNeighborsClassifier', KNeighborsClassifier()),
      ('XGBClassifier', XGBClassifier(random_state=42))
]

for label,model in models:
    model_accuracy(model,label,X_res,Y_res)
    
    
display(model_table.sort_values(by='Score', ascending=False).style.background_gradient(cmap='summer_r'))

print("Baseline Validation done!")

selected_models = [
      ('cat', CatBoostClassifier(verbose= False)),
      ('lgb', LGBMClassifier()),
      ('xgb', XGBClassifier(random_state=42)),
      ('knn', KNeighborsClassifier()),
]

voting_clf = VotingClassifier(estimators = models, voting="soft")

model_accuracy(voting_clf,'Voting',X_res,Y_res)

display(model_table.sort_values(by='Score', ascending=False).style.background_gradient(cmap='summer_r'))


model=voting_clf
model.fit(X_res,Y_res)
preds = model.predict_proba(test)[:,1]

sample_submission['Machine failure'] = preds

sample_submission.to_csv('submission.csv',index=False)
    '''


# In[33]:


lgb_cv_score = np.mean(lgb_cv_scores)
xgb_cv_score = np.mean(xgb_cv_scores)
ens_cv_score = np.mean(ens_cv_scores)

baseline_score = pd.DataFrame(index=['cv_score'])


baseline_score['LightGBM'] = lgb_cv_score
baseline_score['XGBoost'] = xgb_cv_score
baseline_score['Ensemble'] = ens_cv_score


# In[34]:


lgb_preds_test = pd.DataFrame(lgb_preds).apply(np.mean, axis = 0)
xgb_preds_test = pd.DataFrame(xgb_preds).apply(np.mean, axis = 0)
ens_preds_test = pd.DataFrame(ens_preds).apply(np.mean, axis = 0)

submission['Machine failure'] = lgb_preds_test
submission.to_csv('LightGBM_Baseline_submission.csv', index = False)

submission['Machine failure'] = xgb_preds_test
submission.to_csv('XGBoost_Baseline_submission.csv', index = False)

submission['Machine failure'] = ens_preds_test
submission.to_csv('Ensemble_Baseline_submission.csv', index = False)


# # Hope you find this notebook helpful!
