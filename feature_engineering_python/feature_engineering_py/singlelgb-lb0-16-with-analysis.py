#!/usr/bin/env python
# coding: utf-8

# Thinks to notebook: [ICR 2023: Single LGBM 0.12 CV 0.16 LB](https://www.kaggle.com/code/chaitanyagiri/icr-2023-single-lgbm-0-12-cv-0-16-lb) This notebook is based on the aforementioned one and has been modified as follows:
# 1. Abandoning the use of weight coefficients to weigh results from different folds, and instead calculating the average of all fold outputs directly. The corresponding result has improved by 0.01, from the original score of 0.17 to 0.16 after the adjustment.
# 2. use "bags" to make result more stable, code from [RAPIDS cuML SVC Baseline - [LB 0.27] - [CV 0.35]](https://www.kaggle.com/code/cdeotte/rapids-cuml-svc-baseline-lb-0-27-cv-0-35)
# 3. Visual analysis was conducted on logloss, balanced_logloss, and feature importance.
# 
# I hope you will like the changes I have made. If you do, please remember to upvote.
# 
# ---------------------
# 
# updated:
# 1. add model_pred analysis
# 
# 

# In[1]:


VER = 5

print(f'ver: {VER}')


# ## Imports

# In[2]:


if_kaggle = True


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import math

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

import os
join = os.path.join


# ## CFG

# In[4]:


class cfg:
    
    base_path = r'' # local path
    data_path = join(base_path, 'data_ori')
    
    cat_col = ['EJ']
    
    notfea_col = ['Id', 'Class']

CFG = cfg


# In[5]:


bag_num = 20
n_fold = 5


# ## utils: Metric

# In[6]:


def competition_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    return (log_loss_0 + log_loss_1)/2

def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss/(N_0+N_1)


# In[7]:


def lgb_metric(y_true, y_pred):
    return 'balanced_log_loss', balanced_log_loss(y_true, y_pred), False


# ## utils: tools

# In[8]:


def gen_fea_imp_plot(df_imp, tp='Gain'):
    
    # 
    df_imp_grouped_sorted = df_imp.groupby('Feature').mean().sort_values(tp, ascending=False)

    # 
    fig, ax = plt.subplots(figsize=(12, 8))

    # 
    sns.boxplot(x='Feature', y=tp,  data=df_imp, order=df_imp_grouped_sorted.index ,ax=ax)
    # sns.boxplot(x='Feature', y='Gain',  data=df, ax=ax)
    plt.xlabel('Feature')
    plt.ylabel(tp)
    plt.title(f'Feature Importance: {tp}',size=20)
    plt.xticks(rotation=90)
    plt.show()

    


# In[9]:


def plot_model_ana_y0(oof_df_p):
    '''
    真实标签为 1 时，y_pred的概率分布
    '''

    # 选择特定条件下的数据
    data = oof_df_p.loc[(oof_df_p['y_true'] == 0)]['y_pred']

    # 绘制直方图
    plt.hist(data, bins=100, edgecolor='black', color='lightblue')

    # 添加标题和标签
    plt.title("Distribution of y_pred for y_true = 0")
    plt.xlabel("y_pred")
    plt.ylabel("Frequency")

    # 显示图形
    plt.show()


# In[10]:


def plot_model_ana_y1(oof_df_p):
    '''
    真实标签为 1 时，y_pred的概率分布
    '''

    # 选择特定条件下的数据
    data = oof_df_p.loc[(oof_df_p['y_true'] == 1)]['y_pred']

    # 绘制直方图
    plt.hist(data, bins=100, edgecolor='black', color='lightblue')

    # 添加标题和标签
    plt.title("Distribution of y_pred for y_true = 1")
    plt.xlabel("y_pred")
    plt.ylabel("Frequency")

    # 显示图形
    plt.show()


# ## Data Prep

# In[11]:


if if_kaggle:

    COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
    train = pd.read_csv(f"{COMP_PATH}/train.csv")
    test = pd.read_csv(f"{COMP_PATH}/test.csv")
    sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
    greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")
    
else:
    train = pd.read_csv(join(CFG.data_path, 'train.csv'))
    test = pd.read_csv(join(CFG.data_path, 'test.csv'))
    greeks = pd.read_csv(join(CFG.data_path, 'greeks.csv'))
    sample_submission_df = pd.read_csv(join(CFG.data_path, 'sample_submission.csv'))


# In[12]:


# from datetime import datetime
# times = greeks.Epsilon.copy()
# times[greeks.Epsilon != 'Unknown'] = greeks.Epsilon[greeks.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
# times[greeks.Epsilon == 'Unknown'] = np.nan


# In[13]:


# train = pd.concat((train, times), axis=1)
# test_time = pd.DataFrame(np.zeros((len(test), 1)) + train.Epsilon.max() + 1, columns=['Epsilon'])
# test = pd.concat([test, test_time], axis=1)


# In[14]:


# train['Epsilon'] = train['Epsilon'].astype('float')
# test['Epsilon'] = test['Epsilon'].astype('float')


# ## Feature Engineering

# In[15]:


# Label encoding
train['EJ'] = train['EJ'].map({'A': 0, 'B': 1})
test['EJ']  = test['EJ'].map({'A': 0, 'B': 1})

df, test_df = train.copy(), test.copy()


# In[16]:


feas_cols = [i for i in df.columns if i not in CFG.notfea_col]


# ## CV

# ### model_training

# In[17]:


# 
log_losses = []
balanced_log_losses = []

feature_importance_df_total = pd.DataFrame()

models = {}  # key: bag, val: [model] * fold_num


# In[18]:


lgbm_params = {"boosting_type": 'goss'
               , "learning_rate": 0.06733232950390658
               , "n_estimators": 50000
               , "early_stopping_round": 300
               , "random_state": 42
               , "subsample": 0.6970532011679706
               , "colsample_bytree": 0.6055755840633003
               , "class_weight": 'balanced'
               , "metric": 'none'
               , "is_unbalance": True
               , "max_depth": 8}


# In[19]:


oof_df =  pd.DataFrame()

for bag in range(bag_num):
    
    print(f'########################## bag: {bag} ##########################')

    kf = StratifiedKFold(n_splits=n_fold, random_state=42 * bag, shuffle=True)
    
    models[bag] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(df, greeks['Alpha'])):
        # df.loc[test_idx, 'fold'] = fold

        train_df = df.iloc[train_idx]
        valid_df = df.iloc[test_idx]
        valid_ids = valid_df.Id.values.tolist()

        X_train, y_train = train_df[feas_cols], train_df['Class']
        X_valid, y_valid = valid_df[feas_cols], valid_df['Class']

        lgb = LGBMClassifier(**lgbm_params)

        lgb.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False,
                eval_metric=lgb_metric)
        
        # 
        feature_importances = lgb.feature_importances_
        # 
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Split': feature_importances, 'Gain': lgb.booster_.feature_importance(importance_type='gain')})
        feature_importance_df['bag'] = bag
        feature_importance_df['fold'] = fold
        
        feature_importance_df_total = pd.concat([feature_importance_df_total, feature_importance_df], axis=0)
        
        y_pred = lgb.predict_proba(X_valid)[:, 1]
        
        oof_parts = pd.DataFrame(zip(y_valid.values, y_pred), columns=['y_true', 'y_pred'])
        oof_parts['fold'] = fold
        oof_parts['bags'] = bag
        oof_parts['id'] = test_idx
        oof_df = pd.concat([oof_df, oof_parts], axis=0)

        logloss = log_loss(y_valid, y_pred)
        balanced_logloss = balanced_log_loss(y_valid, y_pred)
        log_losses.append(logloss)
        balanced_log_losses.append(balanced_logloss)
        
        models[bag].append(lgb)

        print(f"Bags: {bag}, Fold: {fold}, log loss: {round(logloss, 3)}, balanced los loss: {round(balanced_logloss, 3)}")


# ### model_infoes

# #### model_pred ana

# In[20]:


oof_df_one = oof_df.loc[oof_df['bags'] == 0]

plot_model_ana_y0(oof_df_one)
print()
plot_model_ana_y1(oof_df_one)


# In[21]:


oof_df_one = oof_df.loc[oof_df['bags'] == 0]

plot_model_ana_y0(oof_df_one)
print()
plot_model_ana_y1(oof_df_one)


# The poorly predicted labels have a certain intersection between different bags.

# In[22]:


oof_df_one = oof_df.loc[oof_df['bags'] == 3]

print(oof_df_one.loc[(oof_df_one['y_true'] == 0) & (oof_df_one['y_pred'] > 0.8)].shape[0])
# print(oof_df_one.loc[(oof_df_one['y_true'] == 0) & (oof_df_one['y_pred'] > 0.8), 'id'].values)
print()
print(oof_df_one.loc[(oof_df_one['y_true'] == 1) & (oof_df_one['y_pred'] < 0.2)].shape[0])
# print(oof_df_one.loc[(oof_df_one['y_true'] == 1) & (oof_df_one['y_pred'] < 0.2), 'id'].values)


# #### total loss infoes

# In[23]:


print()
print("Log Loss")
# print(log_losses)
print(np.mean(log_losses), np.std(log_losses))
print()
print("Balanced Log Loss")
# print(balanced_log_losses)
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()


# #### log_loss infoes

# In[24]:


plt.hist(log_losses, bins=100) 
plt.axvline(x=np.mean(log_losses), color='red', linestyle='dashed', label='Mean: {:.2f}'.format(np.mean(log_losses))) 
plt.axvline(x=np.median(log_losses), color='blue', linestyle='dashed', label='Median: {:.2f}'.format(np.median(log_losses))) 
plt.legend() 
plt.title('Histogram of log_losses std: {:.2f}'.format(np.std(log_losses)),size=20) 
plt.show()


# #### balanced_log_loss infoes

# In[25]:


plt.hist(balanced_log_losses, bins=100) 
plt.axvline(x=np.mean(balanced_log_losses), color='red', linestyle='dashed', label='Mean: {:.2f}'.format(np.mean(balanced_log_losses))) 
plt.axvline(x=np.median(balanced_log_losses), color='blue', linestyle='dashed', label='Median: {:.2f}'.format(np.median(balanced_log_losses))) 
plt.legend() 
plt.title('Histogram of balanced_log_losses std: {:.2f}'.format(np.std(balanced_log_losses)),size=20) 
plt.show()


# #### fea_imp: Gain

# In[26]:


gen_fea_imp_plot(feature_importance_df_total, 'Gain')


# #### fea_imp: split

# In[27]:


gen_fea_imp_plot(feature_importance_df_total, 'Split')


# ## inference & Submit

# In[28]:


preds = np.zeros(len(test_df))

for bag in range(bag_num):
    for clf in models[bag]:
        preds += clf.predict_proba(test[feas_cols])[:,1] / n_fold / bag_num


# In[29]:


submission = test[['Id']].copy()
submission['Class_0'] = 1-preds
submission['Class_1'] = preds
submission.to_csv('submission.csv',index=False)
submission.head()


# In[ ]:




