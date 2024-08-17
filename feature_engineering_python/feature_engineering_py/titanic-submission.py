#!/usr/bin/env python
# coding: utf-8

# In[131]:


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


# In[132]:


#!pip install ycimpute


# In[133]:


# Math, Calculations and Statistics
import numpy as np
import scipy
import scipy.stats as st
import statsmodels as sm
import numbers
# Data Manipulation
import pandas as pd
# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
# Machine Learning
import sklearn
import sklearn.model_selection
import sklearn.metrics
import xgboost as xgb
import lightgbm as lgb
import catboost as cbd
# Data Preprocessing
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.impute
#import ycimpute.imputer
# To Ignore Warnings
import warnings
# To Handling Missing Values
import missingno as msn

warnings.filterwarnings("ignore")


# In[134]:


sns.set_theme(palette="mako")


# In[135]:


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df = concat_df(df_train,df_test)


# In[136]:


pd.options.display.max_columns = 999


# # EDA

# In[137]:


df.head()


# In[138]:


df.tail()


# In[139]:


df.shape


# In[140]:


df.info()


# In[141]:


df.describe().T


# # Data Preprocessing

# ## Missing Values

# In[142]:


# AGE
age_by_pclass_sex = df.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(df['Age'].median()))

df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[143]:


# EMBARKED
df['Embarked'] = df['Embarked'].fillna('S')


# In[144]:


# FARE
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df['Fare'] = df['Fare'].fillna(med_fare)


# ## Feature Engineering

# In[145]:


# DECK
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')


# In[146]:


# T DECK IS CLOSE TO A
idx = df[df['Deck'] == 'T'].index
df.loc[idx, 'Deck'] = 'A'


# In[147]:


# DECK TRANSFORMATION
df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')

df['Deck'].value_counts()


# In[148]:


df.drop(['Cabin'], inplace=True, axis=1)

df_train, df_test = divide_df(df)
dfs = [df_train, df_test]

for df in dfs:
    display_missing(df)


# In[149]:


df = concat_df(df_train, df_test)
df


# In[150]:


df['Fare'] = pd.qcut(df['Fare'], 13)
df['Age'] = pd.qcut(df['Age'], 10)


# In[151]:


df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df['Family_Size_Grouped'] = df['Family_Size'].map(family_map)
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
df


# In[152]:


import string

def extract_surname(data):
    
    families = []
    
    for i in range(len(data)):
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

df['Family'] = extract_surname(df['Name'])
df_train = df.loc[:890]
df_test = df.loc[891:]
dfs = [df_train, df_test]

df = concat_df(df_train, df_test)
df


# In[153]:


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()
df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]


# In[154]:


mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

df = concat_df(df_train, df_test)
df


# In[155]:


for df in [df_train, df_test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2
    
df = concat_df(df_train, df_test)
df


# ## Encoding

# In[156]:


from sklearn.preprocessing import LabelEncoder

non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:
    for feature in non_numeric_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])

df = concat_df(df_train, df_test)
drop_cols = ['Family', 'Family_Size', 'Survived',
             'Name', 'Parch', 'PassengerId', 'SibSp', 'Ticket',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']

df.drop(columns=drop_cols, inplace=True)

df


# In[157]:


cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
#encoded_features = []

#for df in dfs:
#    for feature in cat_features:
#        encoded_feat = sklearn.preprocessing.OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
#        n = df[feature].nunique()
#        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
#        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
#        encoded_df.index = df.index
#        encoded_features.append(encoded_df)

#df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
#df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

#df = concat_df(df_train,df_test)
#df 


# In[158]:


#df = concat_df(df_train, df_test)
#drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
#             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
#            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']

#df.drop(columns=drop_cols, inplace=True)

#df


# ## Transforming

# In[159]:


y = df_train["Survived"]
X = df_train.drop(columns = drop_cols)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.20, random_state=1)

# X_train, X_valid, y_train, y_valid = train_test_split(
#    X_train, y_train, test_size=0.20, random_state=1)

print(X_train.shape)
# print(X_valid.shape)
print(X_test.shape)


# # ML Models

# In[160]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = []

models.append(('LogisticRegression', LogisticRegression()))
models.append(('GaussianNB', GaussianNB()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('BC', BaggingClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(("XGBoost", XGBClassifier()))
models.append(("LightGBM", LGBMClassifier()))
models.append(("CatBoost", CatBoostClassifier(verbose = False)))

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(name, "Score: {:.8f}".format(score))


# # LGBM Dataset

# In[161]:


d_train = lgb.Dataset(data = X_train, 
                      label = y_train, 
                      free_raw_data = False, 
                      categorical_feature = cat_features)
d_eval = lgb.Dataset(data = X_test, 
                     label = y_test, 
                     reference = d_train, 
                     free_raw_data = False, 
                     categorical_feature = cat_features)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.09,
    'force_row_wise': True,
    'verbose': 0
}

evals_result={}
lgb_model = lgb.train(params,d_train,
                      valid_sets = d_eval,
                      num_boost_round = 1000,
                      early_stopping_rounds = 200,
                      evals_result = evals_result
                     )


# In[162]:


lgb_pred = lgb_model.predict(X_test)
lgb_score = sklearn.metrics.roc_auc_score(y_test, lgb_pred)
lgb_acc_score = sklearn.metrics.accuracy_score(y_test, (lgb_pred >=0.5))
lgb_conf_mx = sklearn.metrics.confusion_matrix(y_test, (lgb_pred >= 0.5)*1)
print("ROC AUC Score: {:.8f}".format(lgb_score))
print("Accuracy Score: {:.8f}".format(lgb_acc_score))
print("Confusion Matrix:\n",lgb_conf_mx)


# In[163]:


ax = lgb.plot_importance(lgb_model, max_num_features=10)
ax.set_title('')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Features')
plt.show()


# # Optuna

# In[164]:


#import optuna.integration.lightgbm as olgb
#import optuna

#rkf = sklearn.model_selection.RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

#params = {
#        "objective": "binary",
#        "metric": "auc",
#        "verbosity": -1,
#        "boosting_type": "gbdt",                
#        "seed": 1
#    }

#d_train = lgb.Dataset(data = X_train, 
#                      label = y_train, 
#                      free_raw_data = False, 
#                      categorical_feature = cat_features)
#d_eval = lgb.Dataset(data = X_test, 
#                     label = y_test, 
#                     reference = d_train, 
#                     free_raw_data = False, 
#                     categorical_feature = cat_features)

#study_tuner = optuna.create_study(direction='maximize')
#optuna.logging.set_verbosity(optuna.logging.WARNING) 
#
#tuner = olgb.LightGBMTunerCV(params, 
#                            d_train, 
#                            categorical_feature=cat_features,
#                            study=study_tuner,
#                            verbose_eval=False,                            
#                            early_stopping_rounds=250,
#                            time_budget=19800,
#                            seed = 1,
#                            #folds=rkf,
#                            num_boost_round=10000,
#                            callbacks=[lgb.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ] #[0.1]*5 + [0.05]*15 + [0.01]*45 + 
#                           )

#tuner.run()


# In[165]:


#print("ROC AUC Score: {:.8f}".format(tuner.best_score))


# In[166]:


#tmp_best_params = tuner.best_params
#if tmp_best_params['feature_fraction']==1:
#    tmp_best_params['feature_fraction']=1.0-1e-9
#if tmp_best_params['feature_fraction']==0:
#    tmp_best_params['feature_fraction']=1e-9
#if tmp_best_params['bagging_fraction']==1:
#    tmp_best_params['bagging_fraction']=1.0-1e-9
#if tmp_best_params['bagging_fraction']==0:
#    tmp_best_params['bagging_fraction']=1e-9


# In[167]:


#d_train = lgb.Dataset(data = X_train, 
#                      label = y_train, 
#                      free_raw_data = False, 
#                      categorical_feature = cat_features)
#d_eval = lgb.Dataset(data = X_test, 
#                     label = y_test, 
#                     reference = d_train, 
#                     free_raw_data = False, 
#                     categorical_feature = cat_features)
#
#best_score = 999
#training_rounds = 10000

#def objective(trial):
#    # Specify a search space using distributions across plausible values of hyperparameters.
#    param = {
#        "objective": "binary",
#        "metric": "auc",
#        "verbosity": -1,
#        "boosting_type": "gbdt",                
#        "seed": 1,
#        "feature_pre_filter" : False,
#        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
#        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
#        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
#        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
#        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
#        'seed': 1
#    }
    
    # Run LightGBM for the hyperparameter values
#    lgbcv = lgb.cv(param,
#                   d_train,
#                   categorical_feature=cat_features,
#                   #folds=rkf,
#                   verbose_eval=False,
#                   early_stopping_rounds=250,
#                   num_boost_round=10000,
#                   callbacks=[lgb.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ]
#                  )
#    
#    cv_score = np.array(lgbcv["auc-mean"]).mean()
#    
#    return cv_score
#
#optuna.logging.set_verbosity(optuna.logging.WARNING) 

# We search for another 4 hours (3600 s are an hours, so timeout=14400).
# We could instead do e.g. n_trials=1000, to try 1000 hyperparameters chosen 
# by optuna or set neither timeout or n_trials so that we keep going until 
# the user interrupts ("Cancel run").
#study = optuna.create_study(direction='maximize')
#study.enqueue_trial(tmp_best_params)
#study.optimize(objective, timeout=600, n_trials = 50)


# In[168]:


#print(f"Best Parameters : {study.best_params}",f"Best Score : {study.best_value}",sep="\n")


# In[169]:


#best_params = {
#    "objective": "binary",
#    "metric": "auc",
#    "verbosity": -1,
#    "boosting_type": "gbdt",
#    "seed": 1}
#best_params.update(study.best_params)
#best_params


# In[170]:


#lgb_ds = lgb.train(best_params,
#                   d_train,
#                   categorical_feature=cat_features,
#                   verbose_eval=False,                   
#                   num_boost_round=training_rounds)


# # Catboost Optuna

# In[171]:


import catboost as cbd
import optuna

def objective(trial):

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.33),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "depth": trial.suggest_int("depth", 1, 10),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec","Iter"]),
        "od_wait": trial.suggest_int("od_wait", 10, 50),
        #"early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [True, False]),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cbd.CatBoostClassifier(**param)

    gbm.fit(X_train, y_train, verbose=0, early_stopping_rounds=200)

    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[172]:


cbd_model = cbd.CatBoostClassifier(**trial.params).fit(X_train, y_train, verbose=0, early_stopping_rounds=100)
cbd_pred = cbd_model.predict(X_test)
cbd_score = sklearn.metrics.accuracy_score(y_test, cbd_pred)
print("Score : {:.8f}".format(cbd_score))


# # Submission

# In[173]:


drop_cols = ['Family', 'Family_Size', 'Survived',
             'Name', 'Parch', 'SibSp', 'Ticket',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']
df_test = df_test.drop(columns = drop_cols)


# In[174]:


df_test["Survived"] = np.zeros(418)


# In[175]:


df_test.sort_values(by=["PassengerId"],inplace=True)


# In[176]:


test_pred = cbd_model.predict(df_test.drop(["Survived","PassengerId"],axis=1))
df_test["Survived"] = test_pred
df_test


# In[177]:


submission = df_test[["PassengerId","Survived"]]
submission["Survived"] = submission["Survived"].astype("int")
submission.to_csv(r'submission.csv', index = False)


# In[178]:


submission

