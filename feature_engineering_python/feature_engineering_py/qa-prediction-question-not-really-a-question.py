#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge)

# # Prediction a some QA features on other features by 15 ML models (without NLP)
# ### EDA, FE, tuning and comparison of models from my kernels: 
# * https://www.kaggle.com/vbmokin/bod-prediction-in-river-by-15-regression-models
# * [FE & EDA with Pandas Profiling](https://www.kaggle.com/vbmokin/fe-eda-with-pandas-profiling)
# * [Feature importance - xgb, lgbm, logreg, linreg](https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg)

# The analysis shows that some features are very poorly predicted using NLP. It is worth looking for different approaches to prediction.
# 
# One such feature is **"question_not_really_a_question"**.
# 
# This study is devoted to its prediction using other features:
# * presence of words that begin the question ("Can", "Did", "What", ...),
# * presence of the "?",
# * length of the question,
# * length of the answer etc.
# 
# I hope this is helpful.

# In[1]:


target_name = 'question_not_really_a_question'


# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [EDA & FE](#3)
# 1. [Preparing to modeling](#4)
# 1. [Tuning models](#5)
#     -  [Linear Regression](#5.1)
#     -  [Support Vector Machines](#5.2)
#     -  [Linear SVR](#5.3)
#     -  [MLPRegressor](#5.4)
#     -  [Stochastic Gradient Descent](#5.5)
#     -  [Decision Tree Regressor](#5.6)
#     -  [Random Forest with GridSearchCV](#5.7)
#     -  [XGB](#5.8)
#     -  [LGBM](#5.9)
#     -  [GradientBoostingRegressor with HyperOpt](#5.10)
#     -  [RidgeRegressor](#5.11)
#     -  [BaggingRegressor](#5.12)
#     -  [ExtraTreesRegressor](#5.13)
#     -  [AdaBoost Regressor](#5.14)
#     -  [VotingRegressor](#5.15)
# 1. [Models comparison](#6)
# 1. [Prediction](#7)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import eli5
from eli5 import show_prediction

# preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp

# models
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_columns',100)
pd.set_option('max_rows',100)


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


valid_part = 0.2


# In[5]:


train0 = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test0 = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')


# In[6]:


train0['t_len'] = train0['question_title'].str.len()
train0['q_len'] = train0['question_body'].str.len()
train0['a_len'] = train0['answer'].str.len()


# In[7]:


test0['t_len'] = test0['question_title'].str.len()
test0['q_len'] = test0['question_body'].str.len()
test0['a_len'] = test0['answer'].str.len()


# In[8]:


train0['qa_len'] = train0['q_len'] + train0['a_len']
test0['qa_len'] = test0['q_len'] + test0['a_len']


# In[9]:


train0.head(1)


# In[10]:


test0.head(1)


# In[11]:


pd.set_option('max_colwidth', 600)


# In[12]:


train0[train0[target_name] == 0].head(3)


# In[13]:


train0[train0[target_name] == 1].head(3)


# In[14]:


train0.info()


# In[15]:


answ_start = ["Can", "Could", "Did", "Does", "Do", "Has", "How", "Will", "Is", "What", "Where", "Why", "When"]
answ_dict = dict.fromkeys(answ_start, 0)
answ_dict_test = dict.fromkeys(answ_start, 0)


# In[16]:


train0['Is_question_in_body'] = False
for i in range(len(train0)):
    if i % 1000 == 0:
        print(i)
    if train0.loc[i, target_name] > 0:
        for s in answ_start:
            if train0.loc[i, 'question_body'].find(s) > 0:
                train0.loc[i,'Is_question_in_body'] = True
                answ_dict[s] += 1
answ_dict


# In[17]:


test0['Is_question_in_body'] = False
for i in range(len(test0)):
    for s in answ_start:
        if test0.loc[i, 'question_body'].find(s) > 0:
            test0.loc[i,'Is_question_in_body'] = True
            answ_dict_test[s] += 1
answ_dict_test


# In[18]:


train0['Is_question_body_sign'] = train0['question_body'].str.contains("?", regex=False)
train0['Is_question_body_sign'].sum()


# In[19]:


test0['Is_question_body_sign'] = test0['question_body'].str.contains("?", regex=False)
test0['Is_question_body_sign'].sum()


# In[20]:


train0 = train0[['Is_question_in_body', 'Is_question_body_sign', 'qa_len', 'question_not_really_a_question']]
test0 = test0[['Is_question_in_body', 'Is_question_body_sign', 'qa_len']]


# In[21]:


train0.head(1)


# In[22]:


test0.head(1)


# ## 3. EDA & FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# [](http://)This code is based on my kernel "[FE & EDA with Pandas Profiling](https://www.kaggle.com/vbmokin/fe-eda-with-pandas-profiling)"

# In[23]:


train0 = reduce_mem_usage(train0)


# In[24]:


test0 = reduce_mem_usage(test0)


# In[25]:


pp.ProfileReport(train0)


# In[26]:


pp.ProfileReport(test0)


# In[27]:


train0 = train0.fillna(-1)
test0 = test0.fillna(-1)


# ## 4. Preparing to modeling <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[28]:


seed = 0


# In[29]:


# For boosting model
train0b = train0
train_target0b = train0b[target_name]
train0b = train0b.drop([target_name], axis=1)

# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=seed)


# In[30]:


# For boosting model
test0b = test0


# In[31]:


train_target0 = train0[target_name]
train0 = train0.drop([target_name], axis=1)


# In[32]:


#For models from Sklearn
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)
test0 = pd.DataFrame(scaler.fit_transform(test0), columns = test0.columns)


# In[33]:


train0.head(3)


# In[34]:


len(train0)


# In[35]:


# Synthesis valid as test for selection models
train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=seed)


# In[36]:


train.head(3)


# In[37]:


test.head(3)


# In[38]:


train.info()


# In[39]:


test.info()


# In[40]:


acc_train_r2 = []
acc_test_r2 = []
acc_train_d = []
acc_test_d = []
acc_train_rmse = []
acc_test_rmse = []


# In[41]:


def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5


# In[42]:


def acc_boosting_model(num,model,train,test,num_iteration=0):
    # Calculation of accuracy of boosting model by different metrics
    
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration = num_iteration)  
        ytest = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain = model.predict(train)  
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)


# In[43]:


def acc_model(num,model,train,test):
    # Calculation of accuracy of model from Sklearn by different metrics   
  
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)


# ## 5. Tuning models and test for all features <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning, we can narrow down our choice of models to a few. These include:
# 
# - Linear Regression
# - Support Vector Machines and Linear SVR
# - Stochastic Gradient Descent, GradientBoostingRegressor, RidgeCV, BaggingRegressor
# - Decision Tree Regression, Random Forest, XGBRegressor, LGBM, ExtraTreesRegressor
# - MLPRegressor (Deep Learning)
# - VotingRegressor

# ### 5.1 Linear Regression <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **Linear Regression** is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression. Reference [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# In[44]:


# Linear Regression

linreg = LinearRegression()
linreg.fit(train, target)
acc_model(0,linreg,train,test)


# In[45]:


coeff_linreg = pd.DataFrame(train.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[46]:


# Eli5 visualization
eli5.show_weights(linreg, feature_names = train.columns.tolist())


# ### 5.2 Support Vector Machines <a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **Support Vector Machines** are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).

# In[47]:


# Support Vector Machines

svr = SVR()
svr.fit(train, target)
acc_model(1,svr,train,test)


# ### 5.3 Linear SVR <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# **Linear SVR** is a similar to SVM method. Its also builds on kernel functions but is appropriate for unsupervised learning. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Support-vector_clustering_(svr).

# In[48]:


# Linear SVR

linear_svr = LinearSVR()
linear_svr.fit(train, target)
acc_model(2,linear_svr,train,test)


# ### 5.4 MLPRegressor <a class="anchor" id="5.4"></a>
# 
# [Back to Table of Contents](#0.1)

# The **MLPRegressor** optimizes the squared-loss using LBFGS or stochastic gradient descent by the Multi-layer Perceptron regressor. Reference [Sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor).

# Thanks to:
# * https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# * https://stackoverflow.com/questions/44803596/scikit-learn-mlpregressor-performance-cap

# In[49]:


# MLPRegressor

mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.01],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, verbose=True, pre_dispatch='2*n_jobs')
mlp_GS.fit(train, target)
acc_model(3,mlp_GS,train,test)


# ### 5.5 Stochastic Gradient Descent <a class="anchor" id="5.5"></a>
# 
# [Back to Table of Contents](#0.1)

# **Stochastic gradient descent** (often abbreviated **SGD**) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable). It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data). Especially in big data applications this reduces the computational burden, achieving faster iterations in trade for a slightly lower convergence rate. Reference [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

# In[50]:


# Stochastic Gradient Descent

sgd = SGDRegressor()
sgd.fit(train, target)
acc_model(4,sgd,train,test)


# ### 5.6 Decision Tree Regressor <a class="anchor" id="5.6"></a>
# 
# [Back to Table of Contents](#0.1)

# This model uses a **Decision Tree** as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).

# In[51]:


# Decision Tree Regression

decision_tree = DecisionTreeRegressor()
decision_tree.fit(train, target)
acc_model(5,decision_tree,train,test)


# ### 5.7 Random Forest <a class="anchor" id="5.7"></a>
# 
# [Back to Table of Contents](#0.1)

# **Random Forest** is one of the most popular model. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators= [100, 300]) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).

# In[52]:


# Random Forest

random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=10)
random_forest.fit(train, target)
print(random_forest.best_params_)
acc_model(6,random_forest,train,test)


# ### 5.8 XGB<a class="anchor" id="5.8"></a>
# 
# [Back to Table of Contents](#0.1)

# **XGBoost** is an ensemble tree method that apply the principle of boosting weak learners (CARTs generally) using the gradient descent architecture. XGBoost improves upon the base Gradient Boosting Machines (GBM) framework through systems optimization and algorithmic enhancements. Reference [Towards Data Science.](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)

# In[53]:


xgb_clf = xgb.XGBRegressor({'objective': 'reg:squarederror'}) 
parameters = {'n_estimators': [200, 300], 
              'learning_rate': [0.01, 0.02, 0.03],
              'max_depth': [10, 12]}
xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
print("Best score: %0.3f" % xgb_reg.best_score_)
print("Best parameters set:", xgb_reg.best_params_)
acc_boosting_model(7,xgb_reg,trainb,testb)


# ### 5.9 LGBM <a class="anchor" id="5.9"></a>
# 
# [Back to Table of Contents](#0.1)

# **Light GBM** is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms. It splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’. Reference [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/).

# In[54]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


# In[55]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }
modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=2000,verbose_eval=500, valid_sets=valid_set)


# In[56]:


acc_boosting_model(8,modelL,trainb,testb,modelL.best_iteration)


# ### 5.10 GradientBoostingRegressor with HyperOpt<a class="anchor" id="5.10"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/kabure/titanic-eda-model-pipeline-keras-nn

# **Gradient Boosting** builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced. The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data and max_features=n_features, if the improvement of the criterion is identical for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during fitting, random_state has to be fixed. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

# In[57]:


def hyperopt_gb_score(params):
    clf = GradientBoostingRegressor(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[58]:


params = space_eval(space_gb, best)
params


# In[59]:


# Gradient Boosting Regression

gradient_boosting = GradientBoostingRegressor(**params)
gradient_boosting.fit(train, target)
acc_model(9,gradient_boosting,train,test)


# ### 5.11 RidgeRegressor <a class="anchor" id="5.11"></a>
# 
# [Back to Table of Contents](#0.1)

# Tikhonov Regularization, colloquially known as **Ridge Regression**, is the most commonly used regression algorithm to approximate an answer for an equation with no unique solution. This type of problem is very common in machine learning tasks, where the "best" solution must be chosen using limited data. If a unique solution exists, algorithm will return the optimal value. However, if multiple solutions exist, it may choose any of them. Reference [Brilliant.org](https://brilliant.org/wiki/ridge-regression/).

# In[60]:


# Ridge Regressor

ridge = RidgeCV(cv=10)
ridge.fit(train, target)
acc_model(10,ridge,train,test)


# ### 5.12 BaggingRegressor <a class="anchor" id="5.12"></a>
# 
# [Back to Table of Contents](#0.1)

# Bootstrap aggregating, also called **Bagging**, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach. Bagging leads to "improvements for unstable procedures", which include, for example, artificial neural networks, classification and regression trees, and subset selection in linear regression. On the other hand, it can mildly degrade the performance of stable methods such as K-nearest neighbors. Reference [Wikipedia](https://en.wikipedia.org/wiki/Bootstrap_aggregating).

# In[61]:


# Bagging Regressor

bagging = BaggingRegressor()
bagging.fit(train, target)
acc_model(11,bagging,train,test)


# ### 5.13 ExtraTreesRegressor <a class="anchor" id="5.13"></a>
# 
# [Back to Table of Contents](#0.1)

# **ExtraTreesRegressor** implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The default values for the parameters controlling the size of the trees (e.g. max_depth, min_samples_leaf, etc.) lead to fully grown and unpruned trees which can potentially be very large on some data sets. To reduce memory consumption, the complexity and size of the trees should be controlled by setting those parameter values. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html). 
# 
# In extremely randomized trees, randomness goes one step further in the way splits are computed. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/ensemble.html#Extremely%20Randomized%20Trees).

# In[62]:


# Extra Trees Regressor

etr = ExtraTreesRegressor()
etr.fit(train, target)
acc_model(12,etr,train,test)


# ### 5.14 AdaBoost Regressor <a class="anchor" id="5.14"></a>
# 
# [Back to Table of Contents](#0.1)

# The core principle of **AdaBoost** is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. The data modifications at each so-called boosting iteration consist of applying N weights to each of the training samples. Initially, those weights are all set to 1/N, so that the first step simply trains a weak learner on the original data. For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. As iterations proceed, examples that are difficult to predict receive ever-increasing influence. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/ensemble.html#adaboost).

# In[63]:


# AdaBoost Regression

Ada_Boost = AdaBoostRegressor()
Ada_Boost.fit(train, target)
acc_model(13,Ada_Boost,train,test)


# ### 5.15 VotingRegressor <a class="anchor" id="5.15"></a>
# 
# [Back to Table of Contents](#0.1)

# A **Voting Regressor** is an ensemble meta-estimator that fits base regressors each on the whole dataset. It, then, averages the individual predictions to form a final prediction. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor).

# Thanks for the example of ensemling different models from 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor

# In[64]:


Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])
Voting_Reg.fit(train, target)
acc_model(14,Voting_Reg,train,test)


# ## 6. Models comparison <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# We can now compare our models and to choose the best one for our problem.

# In[65]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 
              'MLPRegressor', 'Stochastic Gradient Decent', 
              'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',
              'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 
              'AdaBoostRegressor', 'VotingRegressor'],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'd_train': acc_train_d,
    'd_test': acc_test_d,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
                     })


# In[66]:


pd.options.display.float_format = '{:,.2f}'.format


# In[67]:


print('Prediction accuracy for models by R2 criterion - r2_test')
models.sort_values(by=['r2_test', 'r2_train'], ascending=False)


# In[68]:


print('Prediction accuracy for models by relative error - d_test')
models.sort_values(by=['d_test', 'd_train'], ascending=True)


# In[69]:


print('Prediction accuracy for models by RMSE - rmse_test')
models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)


# In[70]:


# Plot
plt.figure(figsize=[25,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['r2_train'], label = 'r2_train')
plt.plot(xx, models['r2_test'], label = 'r2_test')
plt.legend()
plt.title('R2-criterion for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('R2-criterion, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# In[71]:


# Plot
plt.figure(figsize=[25,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['d_train'], label = 'd_train')
plt.plot(xx, models['d_test'], label = 'd_test')
plt.legend()
plt.title('Relative errors for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Relative error, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# In[72]:


# Plot
plt.figure(figsize=[25,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['rmse_train'], label = 'rmse_train')
plt.plot(xx, models['rmse_test'], label = 'rmse_test')
plt.legend()
plt.title('RMSE for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('RMSE, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# ## 7. Prediction <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# The best model is XGB.

# In[73]:


pred = xgb_reg.predict(test0)


# I hope you find this kernel useful and enjoyable.

# [Go to Top](#0)
