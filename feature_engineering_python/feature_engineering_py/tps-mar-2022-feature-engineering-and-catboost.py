#!/usr/bin/env python
# coding: utf-8

# #  ***[Tabular Playground Series - Mar 2022] Catboost Regressor***

# <img src="https://ak.picdn.net/shutterstock/videos/6521480/thumb/1.jpg" width="500">

# # Import train data and test data

# In[1]:


import pandas as pd
train_df = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv')


# In[2]:


train_df.dtypes


# # Data preprocessing

# ### 1. Feature engineering

# In[3]:


all_df = pd.concat([train_df, test_df])


# In[4]:


def feature_engineering(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['is_month_start'] = data['time'].dt.is_month_start.astype('int')
    data['is_month_end'] = data['time'].dt.is_month_end.astype('int')
    data['is_weekend'] = (data['time'].dt.dayofweek > 5).astype('int')
    data['is_afternoon'] = (data['time'].dt.hour > 12).astype('int')
    data['road'] = data['x'].astype(str) + data['y'].astype(str) + data['direction']
    data['yesterday'] = data.groupby(['x','y','direction','hour','minute'])['congestion'].transform(lambda x: x.shift(1))
    
    data['moment']  = data['time'].dt.hour * 6 + data['time'].dt.minute // 10
    
    data = data.drop(['row_id', 'direction'], axis=1)
    
    return data


# In[5]:


all_df = feature_engineering(all_df)


# In[6]:


train_df = all_df[:len(train_df)]
test_df = all_df[-len(test_df):]


# create congestion Min, Max, Median columns group by 'road', 'weekday', 'hour', 'minute'

# In[7]:


mins = pd.DataFrame(train_df.groupby(['road', 'weekday', 'moment']).congestion.min().astype(int)).reset_index()
mins = mins.rename(columns={'congestion':'min'})
train_df = train_df.merge(mins, on=['road', 'weekday', 'moment'], how='left')
test_df = test_df.merge(mins, on=['road', 'weekday', 'moment'], how='left')


# In[8]:


maxs = pd.DataFrame(train_df.groupby(['road', 'weekday', 'moment']).congestion.max().astype(int)).reset_index()
maxs = maxs.rename(columns={'congestion':'max'})
train_df = train_df.merge(maxs, on=['road', 'weekday', 'moment'], how='left')
test_df = test_df.merge(maxs, on=['road', 'weekday', 'moment'], how='left')


# In[9]:


medians = pd.DataFrame(train_df.groupby(['road', 'weekday', 'moment']).congestion.median().astype(int)).reset_index()
medians = medians.rename(columns={'congestion':'median'})
train_df = train_df.merge(medians, on=['road', 'weekday', 'moment'], how='left')
test_df = test_df.merge(medians, on=['road', 'weekday', 'moment'], how='left')


# ### 3. Label Encoding

# In[10]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(train_df['road'])
train_df['road'] = le.transform(train_df['road'])
test_df['road'] = le.transform(test_df['road'])


# In[11]:


pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 18)
train_df.head()


# In[12]:


test_df = test_df.drop('congestion', axis=1)
test_df.head()


# # Modelling

# ### 1. Split train_df to train data and valid data

# In[13]:


tst_start = pd.to_datetime('1991-09-23 12:00')
tst_finish = pd.to_datetime('1991-09-23 23:40')

X_train = train_df[train_df['time'] < tst_start]
y_train = X_train['congestion']
X_train = X_train.drop(['congestion', 'time'], axis=1)

X_valid = train_df[(train_df['time'] >= tst_start) & (train_df['time'] <= tst_finish)]
y_valid = X_valid['congestion']
X_valid = X_valid.drop(['time', 'congestion'], axis=1)


# In[14]:


from catboost import Pool
train_pool = Pool(X_train, y_train)
validate_pool = Pool(X_valid)


# ### 2. Define model and check the validation score

# In[15]:


from catboost import CatBoostRegressor

model_cat = CatBoostRegressor(logging_level='Silent', depth=8,
                              eval_metric='MAE', loss_function='MAE', n_estimators=800)


# In[16]:


from catboost import Pool
train_pool = Pool(X_train, y_train)
validate_pool = Pool(X_valid)


# In[17]:


from sklearn.metrics import mean_absolute_error

model_cat.fit(train_pool)
y_pred = model_cat.predict(validate_pool)
mean_absolute_error(y_valid, y_pred)


# ### 3. Train the model

# In[18]:


y_train = train_df['congestion']
train_df = train_df.drop(['congestion', 'time'], axis=1)
test_df = test_df.drop('time', axis=1)


# In[19]:


train_pool = Pool(train_df, y_train)
test_pool = Pool(test_df)

model_cat.fit(train_df, y_train)
cat_prediction = model_cat.predict(test_pool)


# In[20]:


feature_importance_df = pd.DataFrame(model_cat.get_feature_importance(prettified=True))
feature_importance_df


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6));
sns.barplot(x='Importances', y='Feature Id', data=feature_importance_df);
plt.title('CatBoost features importance:');


# # Create submission data

# In[22]:


submission = pd.read_csv('../input/tabular-playground-series-mar-2022/sample_submission.csv')


# In[23]:


submission['congestion'] = cat_prediction
submission['congestion'] = submission['congestion'].round().astype(int)
submission.to_csv('submission.csv', index=False)


# In[24]:


submission


# In[ ]:





# In[ ]:




