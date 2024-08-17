#!/usr/bin/env python
# coding: utf-8

# <h3>Our teacher (Ambrosm) gave us an homework after his exciting lesson :-)
#     
#     
# https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart
#     
# Please upvote, I would like my degree  :-)

# In[1]:


import numpy as np
import pandas as pd
import lightgbm
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
import dateutil.easter as easter


# <h2> Exactly the same Features engineering from AmbrosM

# In[2]:


original_train_df = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
original_test_df = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
gdp_df = pd.read_csv('../input/tps-2022-1-gdp/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv')


# In[3]:


gdp_df.index=gdp_df['year']
gdp_df.drop('year',axis=1,inplace=True)


# In[4]:


def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200


# <h2> Feature engineering

# In[5]:


def get_gdp(row):
    """Return the GDP based on row.country and row.date.year"""
    country = 'GDP_' + row.country
    return gdp_df.loc[row.date.year, country]

le_dict = {feature: LabelEncoder().fit(original_train_df[feature]) for feature in ['country', 'product', 'store']}

def engineer(df):
    """Return a new dataframe with the engineered features"""
    
    new_df = pd.DataFrame({'gdp': df.apply(get_gdp, axis=1),
                           'dayofyear': df.date.dt.dayofyear,
                           'wd4': df.date.dt.weekday == 4, # Friday
                           'wd56': df.date.dt.weekday >= 5, # Saturday and Sunday
                          })

    new_df.loc[(df.date.dt.year != 2016) & (df.date.dt.month >=3), 'dayofyear'] += 1 # fix for leap years
    
    for feature in ['country', 'product', 'store']:
        new_df[feature] = le_dict[feature].transform(df[feature])
        
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    new_df['days_from_easter'] = (df.date - easter_date).dt.days.clip(-5, 65)
    
    # Last Sunday of May (Mother's Day)
    sun_may_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-5-31')),
                                         2016: pd.Timestamp(('2016-5-29')),
                                         2017: pd.Timestamp(('2017-5-28')),
                                         2018: pd.Timestamp(('2018-5-27')),
                                         2019: pd.Timestamp(('2019-5-26'))})
    #new_df['days_from_sun_may'] = (df.date - sun_may_date).dt.days.clip(-1, 9)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    new_df['days_from_wed_jun'] = (df.date - wed_june_date).dt.days.clip(-5, 5)
    
    # First Sunday of November (second Sunday is Father's Day)
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    new_df['days_from_sun_nov'] = (df.date - sun_nov_date).dt.days.clip(-1, 9)
    
    return new_df

train_df = engineer(original_train_df)
train_df['date'] = original_train_df.date # used in GroupKFold
train_df['num_sold'] = original_train_df.num_sold.astype(np.float32)
train_df['target'] = np.log(train_df['num_sold'] / train_df['gdp'])
test_df = engineer(original_test_df)

features = test_df.columns.difference(['gdp'])


# <h2> Parameters from Optuna tunings

# In[6]:


lgb_params = {
        'objective': 'regression',
        'force_row_wise': True,
        'verbosity': -1,
        'seed': 1,
        'learning_rate': 0.03,
        'lambda_l1': 5e-05,
        'lambda_l2': 1e-06,
        'num_leaves': 20,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.43,
        'bagging_freq': 5,
        'min_child_samples': 17,
        }                        

cat_params = {
        'eval_metric': 'SMAPE', 
        'use_best_model': True,
        'learning_rate': 0.04421730001498909,
        'depth': 6,
        'l2_leaf_reg': 0.24960109471113703,
        'random_strength': 2.1314060037536735,
        'grow_policy': 'SymmetricTree',
        'max_bin': 406,
        'min_data_in_leaf': 77,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.7392707417524894}

xgb_params = {
        'tree_method': 'hist',
        'grow_policy' : 'lossguide',
        'learning_rate': 0.03399878704233446,
        'max_depth': 5,
        'reg_alpha': 0.7814373604498039,
        'reg_lambda': 0.00018093104956619317,
        'max_delta_step': 2,
        'min_child_weight': 14,
        'colsample_bytree': 0.6489299778623602,
        'subsample': 0.6033298718112065,
        'max_leaves': 187,  
        }


# <h2> Training and predictions

# In[7]:


import warnings
warnings.filterwarnings('ignore')

def training_prediction() :
    
    run = 5 # for seeds blending
    test_pred_list = []
    cat_test_pred_list = []
    all_score_list = []
    lgb_score_list = []
    cat_score_list = []
    xgb_score_list = []

    xgb_test = xgb.DMatrix(test_df[features])

    kf = GroupKFold(n_splits=4)

    for i in range(run):

        lgb_params['seed'] = i
        cat_params['random_seed'] = i
        xgb_params['seed'] = i

        print(25*'-',"RUN",i,25*'-')

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(train_df,
            groups = train_df.date.dt.year)):

            X_tr = train_df.iloc[train_idx]
            X_va = train_df.iloc[val_idx]

            # Preprocess the train data
            X_tr_f = X_tr[features]
            y_tr = X_tr.target.values

            lgb_data_tr = lightgbm.Dataset(
                        X_tr[features],
                        label = y_tr,
                        categorical_feature = ['country',
                                             'product',
                                             'store'])

            xgb_data_tr = xgb.DMatrix(
                        X_tr[features],
                        label=y_tr)

            # Preprocess the validation data
            X_va_f = X_va[features]
            y_va = X_va.target.values

            lgb_data_va = lightgbm.Dataset(
                        X_va[features], 
                        label = y_va)

            xgb_data_va = xgb.DMatrix(
                        X_va[features],
                        label = y_va)
            evallist = [(xgb_data_va, 'eval'), 
                        (xgb_data_tr, 'train')]

            # Training  
            lgb_model = lightgbm.train(
                        lgb_params,
                        lgb_data_tr,
                        num_boost_round=2000,
                        categorical_feature =['country',
                                             'product',
                                             'store'])

            cat_model = CatBoostRegressor(**cat_params) 
            cat_model.fit(
                        X_tr_f,
                        y_tr,eval_set =[( X_va_f,y_va)],
                        verbose = 0,
                        early_stopping_rounds = 200)

            xgb_model = xgb.train(
                        xgb_params, 
                        xgb_data_tr,
                        num_boost_round=2000, 
                        evals = evallist,
                        verbose_eval = 0,
                        early_stopping_rounds = 200)

            # Predictions
            lgb_y_va_pred = np.exp(lgb_model.predict(X_va_f)) * X_va['gdp']
            test_pred_list.append(np.exp(lgb_model.predict(test_df[features])) * test_df['gdp'].values)

            cat_y_va_pred = np.exp(cat_model.predict(X_va_f)) * X_va['gdp']
            test_pred_list.append(np.exp(cat_model.predict(test_df[features])) * test_df['gdp'].values)
            cat_test_pred_list.append(np.exp(cat_model.predict(test_df[features])) * test_df['gdp'].values)

            xgb_y_va_pred = np.exp(xgb_model.predict(xgb_data_va)) * X_va['gdp']
            test_pred_list.append(np.exp(xgb_model.predict(xgb_test)) * test_df['gdp'].values)

            del  xgb_data_tr, xgb_data_va, lgb_data_va

            # Score list for each algo
            lgb_smape = np.round(np.mean(smape_loss(X_va.num_sold, lgb_y_va_pred)),4)
            lgb_score_list.append(lgb_smape)

            cat_smape = np.round(np.mean(smape_loss(X_va.num_sold, cat_y_va_pred)),4)
            cat_score_list.append(cat_smape)

            xgb_smape = np.round(np.mean(smape_loss(X_va.num_sold, xgb_y_va_pred)),4)
            xgb_score_list.append(xgb_smape)

            # list for total average score (mean)
            all_score_list += lgb_score_list + cat_score_list + xgb_score_list

        print('RUN', i,"Cumulative Average SMAPE", np.round(sum(all_score_list) / len(all_score_list),4))
    print(40*'*')   
    print("TOTAL Average SMAPE   :", np.round(sum(all_score_list) / len(all_score_list),4))
    print(40*'*') 
    print("\nOf which :")
    print("LGB Average SMAPE   :", np.round(sum(lgb_score_list) / len(lgb_score_list),4))
    print("CAT Average SMAPE   :", np.round(sum(cat_score_list) / len(cat_score_list),4))
    print("XGB Average SMAPE   :", np.round(sum(xgb_score_list) / len(xgb_score_list),4),'\n\n')


    # Training scores visualization
    plt.figure(figsize = (12,7))
    plt.plot(lgb_score_list,label ='LightGBM')
    plt.plot(cat_score_list,label ='CatBoost')
    plt.plot(xgb_score_list,label ='XgBoost')

    for i in range(run-1):
        plt.axvline(x = (i+1) * 4, 
                    label = 'RUN' +str(i+1),
                    linewidth = 2, 
                    color ='black',
                    linestyle = 'dotted')
    plt.axvline(x = 19, 
                    label ='RUN' +'4',
                    linewidth = 2, 
                    color ='black',
                    linestyle = 'dotted')

    plt.ylabel('SMAPE')
    plt.xlabel('RUN x FOLDS')
    plt.title('Comparison between runs and Algo',fontsize = 20)
    plt.legend()
    plt.show()
    
    return lgb_score_list, cat_score_list, xgb_score_list, all_score_list, test_pred_list, cat_test_pred_list


# <h2> First training

# In[8]:


lgb_score_list, cat_score_list, xgb_score_list, all_score_list, test_pred_list, cat_test_pred_list = training_prediction()


# <h4> We can see that SEED does not provide very different results between each run.
# Catboost provides the best result and xgboost the worse, some work has to be done for xgb tuning ...

# <h2> Second training with pseudo labelling

# In[9]:


#https://www.kaggle.com/andrej0marinchenko/tps-jan-2022-automated-ensembling :
pseudo = pd.read_csv('../input/best-submission/submission_best.csv')
pseudo_test = original_test_df.copy()
pseudo_test['num_sold'] = pseudo['num_sold']
pseudo_df = engineer(pseudo_test)
pseudo_df['date'] = pseudo_test.date 
pseudo_df['num_sold'] = pseudo_test.num_sold.astype(np.float32)
pseudo_df['target'] = np.log(pseudo_df['num_sold'] / pseudo_df['gdp'])

train_df = pd.concat([train_df,pseudo_df],axis=0)
train_df = train_df.reset_index(drop = True)


# In[10]:


p_lgb_score_list, p_cat_score_list, p_xgb_score_list, p_all_score_list, p_test_pred_list, p_cat_test_pred_list = training_prediction()


# <h4> We can see that algo are crushed by pseudo labels...no benefits

# <h2> Submissions

# In[11]:


sub = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
sub['num_sold'] = sum(p_test_pred_list) / len(p_test_pred_list)
sub.to_csv('submission_cat_lgb_xgb.csv', index = False)
pd.read_csv('submission_cat_lgb_xgb.csv').head(2)


# In[12]:


sub['num_sold'] = sum(p_cat_test_pred_list) / len(p_cat_test_pred_list)
sub.to_csv('submission_cat.csv', index = False)
pd.read_csv('submission_cat.csv').head(2)

