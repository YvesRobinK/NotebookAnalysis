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


# In[2]:


df_train=pd.read_csv('/kaggle/input/playground-series-s3e26/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s3e26/test.csv')
df_train.head()
df_full=pd.concat([df_train, df_test])
df_full


# In[3]:


df_full['Status'].value_counts()


# In[4]:


df_full['Drug'].value_counts()
df_full['drug_y']=np.where(df_full['Drug']!='Placebo', 1, 0)


# In[5]:


#target is D
df_full['Status'].replace({'C':0, 'D':1, 'CL':2}, inplace=True)


# In[6]:


categorical_columns=df_full.select_dtypes(include='object')
numeric_columns=df_full.select_dtypes(exclude='object')


# In[7]:


numeric_columns.columns.values


# In[8]:


df_train=df_full[df_full['Status'].isna()==False]
df_train_cd=df_train[df_train['Status']!=2]


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
correlated_features=[]
correlation_matrix=df_train_cd[numeric_columns.columns.values].corr()[['Status']].reset_index()
correlation_matrix.columns=['Feature', 'Correlation']
for feature, cor in zip(correlation_matrix['Feature'], correlation_matrix['Correlation']):

    if (cor<=-.01) & (cor>=-.9):
        correlated_features.append(feature)
    if (cor >= .01) & (cor<=.9):
        correlated_features.append(feature)
correlated_features
        


# Status C and D have the same min and max N_days, but their distribution is very different. The 2th percentile, median, and 75th percentile are all much higher for the surviving group (as to be expected, but we can use this to our advantage)

# In[10]:


for feature in correlated_features:
    print(feature)
    display(df_train_cd[['Status', feature]].groupby('Status')[feature].describe())


# Only 94 patients who survived have Ndays <769, whereas 666 of the patients who died have N_days below 769. We will use this as a feature. 

# In[11]:


##initial pass as creating features for meaningful cutoffs
df_full['APRI']=100 * (df_full['SGOT'])/df_full['Platelets']
df_full['under769days']=np.where(df_full['N_Days']<769, 1, 0)
df_full['platelets_low']=np.where(df_full['Platelets']<300, 1, 0)
df_full['bilirubin_1.2']=np.where(df_full['Bilirubin']>1.2, 1, 0)
df_full['albumin_low']=np.where(df_full['Albumin']<2.23, 1, 0)
df_full['copper_high']=np.where(df_full['Copper']>73, 1, 0)
df_full['SGOT_high']=np.where(df_full['SGOT']>130, 1, 0)
df_full['Prothrombin_high']=np.where(df_full['Prothrombin']>11, 1, 0)
df_full['under_1000days']=np.where(df_full['N_Days']<1000, 1, 0)
df_full['Edema_0']=np.where(df_full['Edema']=='N', 0, 1)
df_full['Edema_1']=np.where(df_full['Edema']=='S', 0, 1)
df_full['Edema_2']=np.where(df_full['Edema']=='Y', 0, 1)
df_full['bilirubin_3']=np.where(df_full['Bilirubin']>3, 1, 0)
df_full['high_cholesteroal']=np.where(df_full['Cholesterol']>240, 1, 0)
df_full['age_over_70']=np.where((df_full['Age']/365)>=70, 1, 0)
df_full['abnormal_alp']=np.where(((df_full['Alk_Phos']<30 )| (df_full['Alk_Phos']>147)), 1, 0)
df_full['very_high_tri']=np.where(df_full['Tryglicerides']>500, 1, 0)
df_full['high_tri']=np.where(df_full['Tryglicerides']>200, 1, 0)
df_full['copper_deficient']=np.where(((df_full['Sex']=='F') & (df_full['Copper']<80) |(df_full['Sex']=='M') & (df_full['Copper']<70)), 1, 0)
df_full['FIB4']=(df_full['Age']/365)* (df_full['SGOT']/df_full['Platelets'])
df_full['ALBI']=.66*np.log(df_full['Bilirubin'])-.085 * df_full['Albumin']
df_full['Stage_4']=np.where(df_full['Stage']==4, 1, 0)
df_full['copper_drug']=np.where((df_full['copper_deficient']==1) & (df_full['drug_y'])==1, 1, 0)


# In[12]:


df_full.replace({'Y':1, 'N':0}, inplace=True)
df_full.replace({'M':1, 'F':0}, inplace=True)
df_full


# In[13]:


numeric_columns=df_full.select_dtypes(exclude='object')
final_features=[]
df_train=df_full[df_full['Status'].isna()==False]

correlation_matrix=numeric_columns.corr()[['Status']].reset_index()
correlation_matrix.columns=['Feature', 'Correlation']
for feature, cor in zip(correlation_matrix['Feature'], correlation_matrix['Correlation']):

    if (cor<=-.09):
        final_features.append(feature)
    if (cor >= .09):
        final_features.append(feature)

final_features
        


# In[14]:


df_fullsu=df_train[final_features]


# In[15]:


cols=['Cholesterol',                
'Copper',
'Albumin',
 'ALBI',
 'Alk_Phos',
 'SGOT',                       
'Alk_Phos',                     
'SGOT',                          
'Tryglicerides',                 
'Prothrombin',                   
'APRI',                         
    'FIB4' ]


# In[16]:


# Outlier Analysis

iqr_factor = [5]
list=[]

for factor in iqr_factor:
    count = 0
    print(f'Outliers for {factor} IQR :')
    print('-------------------------------------')
    for col in cols:
    
        IQR = df_fullsu[col].quantile(0.75) - df_fullsu[col].quantile(0.25)
        lower_lim = df_fullsu[col].quantile(0.25) - factor*IQR
        upper_lim = df_fullsu[col].quantile(0.75) + factor*IQR
    
        cond = df_fullsu[(df_fullsu[col] < lower_lim) | (df_fullsu[col] > upper_lim)].shape[0]
        
        if  cond > 0 :
            list.append(df_fullsu[(df_fullsu[col] < lower_lim) | (df_fullsu[col] > upper_lim)].index.tolist())
        
        if cond > 0: print(f'{col:<30} : ', cond); count += cond
    print(f'\nTOTAL OUTLIERS FOR {factor} IQR : {count}')
    print('')


# In[17]:


def drop_outliers(df_fullsu, col):
    IQR = df_fullsu[col].quantile(0.75) - df_fullsu[col].quantile(0.25)
    lower_lim = df_fullsu[col].quantile(0.25) - (5*IQR)
    upper_lim = df_fullsu[col].quantile(0.75) + (5*IQR)
    df_fullsu.drop(df_fullsu[df_fullsu[col]>(upper_lim)].index, inplace=True)
    df_fullsu.drop(df_fullsu[df_fullsu[col]<(lower_lim)].index, inplace=True)
    
for col in cols:
    drop_outliers(df_fullsu, col)
df_fullsu.describe()


# In[18]:


# Compare Algorithms
import pandas
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
X=df_fullsu[final_features].drop(columns='Status')



y=df_fullsu['Status']
# prepare configuration for cross validation test harness
seed = 7
# prepare models

models = []

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('XG', xgb.XGBClassifier() ))

trans = RobustScaler()


# evaluate each model in turn
results = []
names = []
scoring = 'neg_log_loss'

for name, model in models:
	pipeline = Pipeline(steps=[('t', trans), ('m', model)])
	kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
	cv_results = model_selection.cross_val_score(pipeline, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[19]:


import optuna  # pip install optuna
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y)

def objective(trial):
    """
    Objective function to be minimized.
    """
    param = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_iterations": trial.suggest_int("num_iterations", 50, 1000 ),
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_class": 3,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.05, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 100),
        "learning_rate": trial.suggest_float('learning_rate', 6e-3, .1, log=True)
    }
    gbm = LGBMClassifier(**param)
    

    X_train_s=RobustScaler().fit_transform(X_train)
    X_test_s=RobustScaler().fit_transform(X_test)
    gbm.fit(X_train_s, y_train)
    preds = gbm.predict(X_test_s)
    prob = gbm.predict_proba(X_test_s)
    logloss = log_loss(y_test, prob)
    return logloss


# In[20]:


#sampler = TPESampler(seed=5)
#study = optuna.create_study(study_name="lightgbm", direction="minimize", sampler=sampler)
#study.optimize(objective, n_trials=60, show_progress_bar=True)


# In[21]:


#print('Best parameters:', study.best_params)


# In[22]:


import optuna  # pip install optuna
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y)

def objective(trial):
    """
    Objective function to be minimized.
    """
    param = {
        "objective": "multi_logloss",
        "n_estimators": trial.suggest_int('n_estimators', 500, 750),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0), 
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0), 
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }
    gbm =xgb.XGBClassifier(**param) 

    X_train_s=RobustScaler().fit_transform(X_train)
    X_test_s=RobustScaler().fit_transform(X_test)
    gbm.fit(X_train_s, y_train)
    preds = gbm.predict(X_test_s)
    prob = gbm.predict_proba(X_test_s)
    logloss = log_loss(y_test, prob)
    return logloss

#study = optuna.create_study(study_name="xgb", direction="minimize")
#study.optimize(objective, n_trials=50,show_progress_bar=True)


# In[23]:


#print('Best parameters:', study.best_params)


# In[24]:


lgbmparams={'num_iterations': 132, 'lambda_l1': 0.18583185025563095, 'lambda_l2': 0.19800577586440474, 'num_leaves': 137, 'feature_fraction': 0.15078005255050653, 'bagging_fraction': 0.6952185463065385, 'bagging_freq': 4, 'min_child_samples': 89, 'learning_rate': 0.05288406317096321}
xgbparams={'n_estimators': 672, 'learning_rate': 0.009762134984540281, 'max_depth': 6, 'subsample': 0.7704898154169506, 'colsample_bytree': 0.3488006965992821, 'min_child_weight': 7}


# In[25]:


X=df_train[final_features]


y=df_train['Status']

model2=LGBMClassifier(**lgbmparams)


trans = RobustScaler(with_centering=False, with_scaling=True)
Xs=trans.fit_transform(X)

model2.fit(Xs, y)




# In[26]:


df_test=df_full[df_full['Status'].isna()==True]
X_test=df_test[final_features]
X_tests=trans.fit_transform(X_test)
y_pred=model2.predict(X_tests)

probs=model2.predict_proba(X_tests)

df_test['prob_D2']=probs[:,1]
df_test['prob_C2']=probs[:,0]
df_test['prob_CL2']=probs[:,2]

df_test['Status_C']=df_test['prob_C2']
df_test['Status_CL']=df_test['prob_CL2']
df_test['Status_D']=df_test['prob_D2']


# In[27]:


submission=df_test[['id','Status_C', 'Status_CL', 'Status_D']]
submission.to_csv('submission.csv', index=False)


# In[28]:


submission


# In[ ]:





# In[ ]:




