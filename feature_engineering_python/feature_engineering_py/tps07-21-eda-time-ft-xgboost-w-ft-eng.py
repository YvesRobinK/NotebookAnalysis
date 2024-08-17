#!/usr/bin/env python
# coding: utf-8

# # <center>Tabular Playground Series - July/2021<center>
# ## <center>EDA (on time features) + XGBoost with Feature Engineering<center>
# ---
# Notebook created on the last days of competition for practice purposes, with no intent to aim at the top of the leaderboard (given the data leakage issue). It consists of basic feature engineering based on some EDA. No pseudolabeling (except for using carbon monoxide as a feature for nitrogen oxides) and no external sources.
#     
# Notebooks that were helpful for this work:
# * [XGBoost & LeaveOneGroupOut & Ensembling](https://www.kaggle.com/mehrankazeminia/1-tps-jul-21-xgboost-leaveonegroupout/notebook) by [@mehrankazeminia](https://www.kaggle.com/mehrankazeminia) & [@somayyehgholami](https://www.kaggle.com/somayyehgholami) (Leave one group out)
# * [TPS July EDA+LGBM+Models](https://www.kaggle.com/ankitp013/tps-july-eda-lgbm-models) by [@ankitp013](https://www.kaggle.com/ankitp013) (Feature Engineering) 

# ## Importing Libraries and Datasets

# In[1]:


import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_log_error

from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor


# In[2]:


df_train = pd.read_csv('../input/tabular-playground-series-jul-2021/train.csv', index_col='date_time', parse_dates=['date_time'])
X_test = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv', index_col='date_time', parse_dates=['date_time'])
submission = pd.read_csv('../input/tabular-playground-series-jul-2021/sample_submission.csv')


target = [col for col in df_train.columns if 'target_' in col]
Y_train = df_train[target].copy()
#X_train = df_train.copy().drop(target, axis = 1)


# In[3]:


df_train


# In[4]:


df_train.info()


# In[5]:


df_train.describe().T


# In[6]:


X_test


# In[7]:


X_test.info()


# In[8]:


X_test.describe()


# ## Exploring the Data

# In[9]:


df_train['month'] = df_train.index.month
#df_train['weekofyear'] = df_train.index.isocalendar().week
df_train['dayofweek'] = df_train.index.dayofweek
df_train['hour'] = df_train.index.hour

X_test['month'] = X_test.index.month
#X_test['weekofyear'] = X_test.index.isocalendar().week
X_test['dayofweek'] = X_test.index.dayofweek
X_test['hour'] = X_test.index.hour


# Note: Day of week values go from 0(monday) to 6(sunday)

# In[10]:


df_train.head()


# In[11]:


X_test.head()


# In[12]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_train.index, y="target_carbon_monoxide", data=df_train, label='carbon_monoxide', color = 'red')

plt.legend()
plt.xticks(rotation=25)
plt.show()


# In[13]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_train.index, y="target_benzene", data=df_train, label='benzene', color = 'blue')

plt.legend()
plt.xticks(rotation=25)
plt.show()


# In[14]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_train.index, y="target_nitrogen_oxides", data=df_train, label='nitrogen_oxides', color = 'green')

plt.legend()
plt.xticks(rotation=25)
plt.show()


# In[15]:


df_month = pd.DataFrame()
df_month['target_carbon_monoxide'] = df_train.groupby(['month'])['target_carbon_monoxide'].mean()
df_month['target_benzene'] = df_train.groupby(['month'])['target_benzene'].mean()
df_month['target_nitrogen_oxides'] = df_train.groupby(['month'])['target_nitrogen_oxides'].mean()
#df_month

df_dayofweek = pd.DataFrame()
df_dayofweek['target_carbon_monoxide'] = df_train.groupby(['dayofweek'])['target_carbon_monoxide'].mean()
df_dayofweek['target_benzene'] = df_train.groupby(['dayofweek'])['target_benzene'].mean()
df_dayofweek['target_nitrogen_oxides'] = df_train.groupby(['dayofweek'])['target_nitrogen_oxides'].mean()
#df_dayofweek

df_hour = pd.DataFrame()
df_hour['target_carbon_monoxide'] = df_train.groupby(['hour'])['target_carbon_monoxide'].mean()
df_hour['target_benzene'] = df_train.groupby(['hour'])['target_benzene'].mean()
df_hour['target_nitrogen_oxides'] = df_train.groupby(['hour'])['target_nitrogen_oxides'].mean()
#df_hour


# In[16]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_month.index, y="target_carbon_monoxide", data=df_month, label='carbon_monoxide', color = 'red')

plt.legend()
plt.xticks(df_month.index)
plt.show()


# In[17]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_month.index, y="target_benzene", data=df_month, label='benzene', color = 'blue')

plt.legend()
plt.xticks(df_month.index)
plt.show()


# In[18]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_month.index, y="target_nitrogen_oxides", data=df_month, label='nitrogen_oxides', color = 'green')

plt.legend()
plt.xticks(df_month.index)
plt.show()


# In[19]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_dayofweek.index, y="target_carbon_monoxide", data=df_dayofweek, label='carbon_monoxide', color = 'red')

plt.legend()
plt.xticks(df_dayofweek.index)
plt.show()


# In[20]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_dayofweek.index, y="target_benzene", data=df_dayofweek, label='benzene', color = 'blue')

plt.legend()
plt.xticks(df_dayofweek.index)
plt.show()


# In[21]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_dayofweek.index, y="target_nitrogen_oxides", data=df_dayofweek, label='nitrogen_oxides', color = 'green')

plt.legend()
plt.xticks(df_dayofweek.index)
plt.show()


# In[22]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_hour.index, y="target_carbon_monoxide", data=df_hour, label='carbon_monoxide', color = 'red')

plt.legend()
plt.xticks(df_hour.index)
plt.show()


# In[23]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_hour.index, y="target_benzene", data=df_hour, label='benzene', color = 'blue')

plt.legend()
plt.xticks(df_hour.index)
plt.show()


# In[24]:


plt.figure(figsize=(15,5))

sns.lineplot(x=df_hour.index, y="target_nitrogen_oxides", data=df_hour, label='nitrogen_oxides', color = 'green')

plt.legend()
plt.xticks(df_hour.index)
plt.show()


# Observations:
# - Seasonal influence on targets;
# - All targets have lower values on weekend;
# - Higher values on commute and work time.
# 
# Creating new features based on the previous plots.

# In[25]:


#Seasons
df_train['is_winter'] = df_train['month'].apply(lambda x: 1 if (x == 12 or x == 1 or x == 2) else 0)
X_test['is_winter'] = X_test['month'].apply(lambda x: 1 if (x == 12 or x == 1 or x == 2) else 0)

df_train['is_spring'] = df_train['month'].apply(lambda x: 1 if (x == 3 or x == 4 or x == 5) else 0)
X_test['is_spring'] = X_test['month'].apply(lambda x: 1 if (x == 3 or x == 4 or x == 5) else 0)

df_train['is_summer'] = df_train['month'].apply(lambda x: 1 if (x == 6 or x == 7 or x == 8) else 0)
X_test['is_summer'] = X_test['month'].apply(lambda x: 1 if (x == 6 or x == 7 or x == 8) else 0)

df_train['is_autumn'] = df_train['month'].apply(lambda x: 1 if (x == 9 or x == 10 or x == 11) else 0)
X_test['is_autumn'] = X_test['month'].apply(lambda x: 1 if (x == 9 or x == 10 or x == 11) else 0)

#Weekend or not
df_train['is_weekend'] = df_train['dayofweek'].apply(lambda x: 1 if x >= 5  else 0)
X_test['is_weekend'] = X_test['dayofweek'].apply(lambda x: 1 if x >= 5  else 0)

#Commute/work periods
df_train['is_commute_m'] = df_train['hour'].apply(lambda x: 1 if (x == 8 or x == 9)  else 0)
X_test['is_commute_m'] = X_test['hour'].apply(lambda x: 1 if (x == 8 or x == 9)  else 0)
                                                  
df_train['is_work'] = df_train['hour'].apply(lambda x: 1 if (x >=10 and x < 18)  else 0)
X_test['is_work'] = X_test['hour'].apply(lambda x: 1 if (x >=10 and x < 18)  else 0)

df_train['is_commute_e'] = df_train['hour'].apply(lambda x: 1 if (x >=18 and x <= 20)  else 0)
X_test['is_commute_e'] = X_test['hour'].apply(lambda x: 1 if (x >=18 and x <= 20)  else 0)                                                  


# In[26]:


df_train.head()


# In[27]:


X_test.head()


# In[28]:


season = ['is_winter','is_spring','is_summer','is_autumn']

plt.figure(figsize=(16,12))

for i,col in enumerate(df_train[season]):    
    plt.subplot(2,2,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_carbon_monoxide"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[29]:


plt.figure(figsize=(16,12))

for i,col in enumerate(df_train[season]):    
    plt.subplot(2,2,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_benzene"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[30]:


plt.figure(figsize=(16,12))

for i,col in enumerate(df_train[season]):    
    plt.subplot(2,2,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_nitrogen_oxides"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[31]:


work_hour = ['is_commute_e','is_work','is_commute_e']

plt.figure(figsize=(16,6))

for i,col in enumerate(df_train[work_hour]):    
    plt.subplot(1,3,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_carbon_monoxide"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[32]:


plt.figure(figsize=(16,6))

for i,col in enumerate(df_train[work_hour]):    
    plt.subplot(1,3,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_benzene"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[33]:


plt.figure(figsize=(16,6))

for i,col in enumerate(df_train[work_hour]):    
    plt.subplot(1,3,i + 1)

    sns.boxplot(x=df_train.loc[:,col], y=df_train["target_nitrogen_oxides"], palette = 'BuPu')    
    plt.ylabel("")
    plt.yticks(fontsize = 7)

plt.show()


# In[34]:


plt.figure(figsize=(5,6))

sns.boxplot(x=df_train['is_weekend'], y=df_train["target_carbon_monoxide"], palette = 'BuPu')    
plt.ylabel("")
plt.yticks(fontsize = 7)

plt.show()


# In[35]:


plt.figure(figsize=(5,6))

sns.boxplot(x=df_train['is_weekend'], y=df_train["target_benzene"], palette = 'BuPu')    
plt.ylabel("")
plt.yticks(fontsize = 7)

plt.show()


# In[36]:


plt.figure(figsize=(5,6))

sns.boxplot(x=df_train['is_weekend'], y=df_train["target_nitrogen_oxides"], palette = 'BuPu')    
plt.ylabel("")
plt.yticks(fontsize = 7)

plt.show()


# In[37]:


plt.figure(figsize=(20,10))
sns.heatmap(df_train.corr().round(2), vmin=-1, vmax=1, center=0, annot=True, cmap='viridis')
plt.show()


# In[38]:


df_train.corr('spearman')[['target_carbon_monoxide']].sort_values(['target_carbon_monoxide'], ascending=False).style.background_gradient('viridis')


# In[39]:


df_train.corr('spearman')[['target_benzene']].sort_values(['target_benzene'], ascending=False).style.background_gradient('viridis')


# In[40]:


df_train.corr('spearman')[['target_nitrogen_oxides']].sort_values(['target_nitrogen_oxides'], ascending=False).style.background_gradient('viridis')


# In[41]:


X_train = df_train.copy().drop(target, axis = 1)


# In[42]:


X_train


# In[43]:


Y_train1 = Y_train.iloc[:, 0].copy()
Y_train2 = Y_train.iloc[:, 1].copy()
Y_train3 = Y_train.iloc[:, 2].copy()


# In[44]:


Y_train1


# In[45]:


Y_train2


# In[46]:


Y_train3


# In[47]:


groups = X_train['month']
groups.value_counts()


# In[48]:


X_train['month'] = X_train['month'].replace(1,12)
groups = X_train['month']
groups.value_counts()


# ## Making Predictions

# In[49]:


#Month feature harms the performance.
X_train = X_train.drop('month', axis = 1)
X_test = X_test.drop('month', axis = 1)


# In[50]:


def prediction (X_train, Y_train, model, X_test):
        
    #kfold = KFold(n_splits = 10)
    logo = LeaveOneGroupOut()

    y_pred = np.zeros(len(X_test))
    train_oof = np.zeros(len(X_train))
    
    #for idx in kfold.split(X=X_train, y=Y_train):
    for idx in logo.split(X=X_train, y=Y_train, groups=groups):
        train_idx, val_idx = idx[0], idx[1]
        xtrain = X_train.iloc[train_idx]
        ytrain = Y_train.iloc[train_idx]
        xval = X_train.iloc[val_idx]
        yval = Y_train.iloc[val_idx]
        
        # fit model for current fold
        model.fit(xtrain, ytrain, 
            early_stopping_rounds = 100, eval_set = [(xval,yval)], verbose = False)

        #create predictions
        #y_pred += model.predict(X_test)/kfold.n_splits
        y_pred += model.predict(X_test)/10 #logo.n_splits
        print(y_pred)
               
        val_pred = model.predict(xval)
        # getting out-of-fold predictions on training set
        val_pred[val_pred < 0] = 0 #few negative values
        train_oof[val_idx] = val_pred

        # calculate and append rmsle
        msle = mean_squared_log_error(yval,val_pred)
        rmsle = np.sqrt(msle)
        print('RMSLE : {}'.format(rmsle))
  
    return y_pred, train_oof


# In[51]:


#lgbm_model = LGBMRegressor(objective = 'regression', metric = 'rmse', n_estimators = 3000, learning_rate = 0.02, random_state = 42,
#                           subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)

xgb_model = XGBRegressor(objective="reg:squarederror", eval_metric = 'rmsle', n_estimators = 3000, learning_rate = 0.02, random_state = 42,
                           subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)


# In[52]:


#pred_1, train_oof_1  = prediction (X_train, Y_train1, lgbm_model, X_test)
pred_1, train_oof_1  = prediction (X_train, Y_train1, xgb_model, X_test)


# In[53]:


#pred_2, train_oof_2  = prediction (X_train, Y_train2, lgbm_model, X_test)
pred_2, train_oof_2  = prediction (X_train, Y_train2, xgb_model, X_test)


# In[54]:


#Using carbon_monoxide to improve nitrogen_oxides prediction
X_train['target_carbon_monoxide'] = df_train['target_carbon_monoxide']
X_test['target_carbon_monoxide'] = pred_1


# In[55]:


#pred_3, train_oof_3  = prediction (X_train, Y_train3, lgbm_model, X_test)
pred_3, train_oof_3  = prediction (X_train, Y_train3, xgb_model, X_test)


# In[56]:


print("RMSLE: {0:0.6f}".format(np.sqrt(mean_squared_log_error(Y_train1,train_oof_1))))
print("RMSLE: {0:0.6f}".format(np.sqrt(mean_squared_log_error(Y_train2,train_oof_2))))
print("RMSLE: {0:0.6f}".format(np.sqrt(mean_squared_log_error(Y_train3,train_oof_3))))


# In[57]:


submission['target_carbon_monoxide'] = pred_1
submission['target_benzene'] = pred_2
submission['target_nitrogen_oxides'] = pred_3

submission


# In[58]:


submission.to_csv("submission.csv", index=False)
submission

