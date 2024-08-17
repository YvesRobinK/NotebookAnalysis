#!/usr/bin/env python
# coding: utf-8

# # LightGBM starter with feature engineering idea
# ***I am creating this note mostly for myself to re-organize my ideas. Please leave your comments and/or advice if you find the room I can improve my coding/model builing to go further.***
# 
# 
# We will predict **the realized volatility of the next ten-minutes time window** with two data sets of the last ten minutes (600 seconds).One dataset contains ask and bid prices of almost each second, which allows us to calculate the realized volatility of the last ten minutes.The other dataset contains the actual record of stock trading, which is more sparse.
# 
# Please look at this notebook for the detailed explanation: https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data
# 
# As for EDA, you may find this notebook useful: https://www.kaggle.com/chumajin/optiver-realized-eda-for-starter-english-version
# 
# Thank you!

# ### Contribution from the commmunity
# **The codes for showing Feature Importance is kindly prepared by this expert: https://www.kaggle.com/something4kag
# and extracted by this notebook: https://www.kaggle.com/something4kag/lightgbm-starter-with-fe-and-importance**

# ## My approach(work in progress)
# ### Feature Engineering
# 
# Here are my thoughts on feature engieering with my background knowledge on financial market. 
# 
#  - price_spread: the difference between ask price and bid price. Wide spread means low liquidity, leading to high volatility.
#  - volume: the sum of the ask/bid size. Low volume means low liquidity, leading to high volatility
#  - volume_imbalance: the difference between ask size and bid size. Large imbalance means low liquidity for one side, leading to high volatility
#  
# Also, I created features only using last XX seconds to capture the dynamics of volatility further.
# 
# 
# ### Model Building
# - optimize the weight for RMSPE: see this discussion https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/250324
# - one model for all stocks: model by stock_id does not work well. I am afraid of overfitting as well. stock_id is used as categorical and for target mean encoding.

# ## Preparation

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_rows', 300)
pd.set_option('max_columns', 300)

import os
import glob


# In[2]:


# data directory
data_dir = '../input/optiver-realized-volatility-prediction/'


# ## Functions for preprocess

# In[3]:


def calc_wap(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1'])/(df['bid_size1'] + df['ask_size1'])
    return wap
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2'])/(df['bid_size2'] + df['ask_size2'])
    return wap


# In[4]:


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 


# In[5]:


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


# In[6]:


def count_unique(series):
    return len(np.unique(series))


# In[7]:


book_train = pd.read_parquet(data_dir + "book_train.parquet/stock_id=15")
book_train.head()


# ## Main function for preprocessing book data

# In[8]:


def preprocessor_book(file_path):
    df = pd.read_parquet(file_path)
    #calculate return etc
    df['wap'] = calc_wap(df)
    df['log_return'] = df.groupby('time_id')['wap'].apply(log_return)
    
    df['wap2'] = calc_wap2(df)
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(log_return)
    
    df['wap_balance'] = abs(df['wap'] - df['wap2'])
    
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    #dict for aggregate
    create_feature_dict = {
        'log_return':[realized_volatility],
        'log_return2':[realized_volatility],
        'wap_balance':[np.mean],
        'price_spread':[np.mean],
        'bid_spread':[np.mean],
        'ask_spread':[np.mean],
        'volume_imbalance':[np.mean],
        'total_volume':[np.mean],
        'wap':[np.mean],
            }

    #####groupby / all seconds
    df_feature = pd.DataFrame(df.groupby(['time_id']).agg(create_feature_dict)).reset_index()
    
    df_feature.columns = ['_'.join(col) for col in df_feature.columns] #time_id is changed to time_id_
        
    ######groupby / last XX seconds
    last_seconds = [300]
    
    for second in last_seconds:
        second = 600 - second 
    
        df_feature_sec = pd.DataFrame(df.query(f'seconds_in_bucket >= {second}').groupby(['time_id']).agg(create_feature_dict)).reset_index()

        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns] #time_id is changed to time_id_
     
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))

        df_feature = pd.merge(df_feature,df_feature_sec,how='left',left_on='time_id_',right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'],axis=1)
    
    #create row_id
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['time_id_'],axis=1)
    
    return df_feature


# In[9]:


get_ipython().run_cell_magic('time', '', 'file_path = data_dir + "book_train.parquet/stock_id=0"\npreprocessor_book(file_path)\n')


# In[10]:


trade_train = pd.read_parquet(data_dir + "trade_train.parquet/stock_id=0")
trade_train.head(15)


# ## Main function for preprocessing trade data

# In[11]:


def preprocessor_trade(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    
    
    aggregate_dictionary = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    df_feature = df.groupby('time_id').agg(aggregate_dictionary)
    
    df_feature = df_feature.reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]

    
    ######groupby / last XX seconds
    last_seconds = [300]
    
    for second in last_seconds:
        second = 600 - second
    
        df_feature_sec = df.query(f'seconds_in_bucket >= {second}').groupby('time_id').agg(aggregate_dictionary)
        df_feature_sec = df_feature_sec.reset_index()
        
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        
        df_feature = pd.merge(df_feature,df_feature_sec,how='left',left_on='time_id_',right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'],axis=1)
    
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['trade_time_id_'],axis=1)
    
    return df_feature


# In[12]:


get_ipython().run_cell_magic('time', '', 'file_path = data_dir + "trade_train.parquet/stock_id=0"\npreprocessor_trade(file_path)\n')


# ## Combined preprocessor function

# In[13]:


def preprocessor(list_stock_ids, is_train = True):
    from joblib import Parallel, delayed # parallel computing to save time
    df = pd.DataFrame()
    
    def for_joblib(stock_id):
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
            
        df_tmp = pd.merge(preprocessor_book(file_path_book),preprocessor_trade(file_path_trade),on='row_id',how='left')
     
        return pd.concat([df,df_tmp])
    
    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
        )

    df =  pd.concat(df,ignore_index = True)
    return df


# In[14]:


list_stock_ids = [0,1]
preprocessor(list_stock_ids, is_train = True)


# ## Training set

# In[15]:


train = pd.read_csv(data_dir + 'train.csv')


# In[16]:


train_ids = train.stock_id.unique()


# In[17]:


get_ipython().run_cell_magic('time', '', 'df_train = preprocessor(list_stock_ids= train_ids, is_train = True)\n')


# In[18]:


train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_train = train.merge(df_train, on = ['row_id'], how = 'left')


# In[19]:


df_train.head()


# ## Test set

# In[20]:


test = pd.read_csv(data_dir + 'test.csv')


# In[21]:


test_ids = test.stock_id.unique()


# In[22]:


get_ipython().run_cell_magic('time', '', 'df_test = preprocessor(list_stock_ids= test_ids, is_train = False)\n')


# In[23]:


df_test = test.merge(df_test, on = ['row_id'], how = 'left')


# ## Target encoding by stock_id

# In[24]:


from sklearn.model_selection import KFold
#stock_id target encoding
df_train['stock_id'] = df_train['row_id'].apply(lambda x:x.split('-')[0])
df_test['stock_id'] = df_test['row_id'].apply(lambda x:x.split('-')[0])

stock_id_target_mean = df_train.groupby('stock_id')['target'].mean() 
df_test['stock_id_target_enc'] = df_test['stock_id'].map(stock_id_target_mean) # test_set

#training
tmp = np.repeat(np.nan, df_train.shape[0])
kf = KFold(n_splits = 10, shuffle=True,random_state = 19911109)
for idx_1, idx_2 in kf.split(df_train):
    target_mean = df_train.iloc[idx_1].groupby('stock_id')['target'].mean()

    tmp[idx_2] = df_train['stock_id'].iloc[idx_2].map(target_mean)
df_train['stock_id_target_enc'] = tmp


# ## Model Building

# In[25]:


df_train.head()


# In[26]:


df_test.head()


# In[27]:


DO_FEAT_IMP = False
if len(df_test)==3:
    DO_FEAT_IMP = True


# ## LightGBM

# In[28]:


import lightgbm as lgbm


# In[29]:


# ref https://www.kaggle.com/corochann/permutation-importance-for-feature-selection-part1
def calc_model_importance(model, feature_names=None, importance_type='gain'):
    importance_df = pd.DataFrame(model.feature_importance(importance_type=importance_type),
                                 index=feature_names,
                                 columns=['importance']).sort_values('importance')
    return importance_df


def plot_importance(importance_df, title='',
                    save_filepath=None, figsize=(8, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    importance_df.plot.barh(ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()


# In[30]:


df_train['stock_id'] = df_train['stock_id'].astype(int)
df_test['stock_id'] = df_test['stock_id'].astype(int)


# In[31]:


X = df_train.drop(['row_id','target'],axis=1)
y = df_train['target']


# In[32]:


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def feval_RMSPE(preds, lgbm_train):
    labels = lgbm_train.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False

params = {
      "objective": "rmse", 
      "metric": "rmse", 
      "boosting_type": "gbdt",
      'early_stopping_rounds': 30,
      'learning_rate': 0.01,
      'lambda_l1': 1,
      'lambda_l2': 1,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
  }


# ### Cross Validation

# In[33]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=19901028, shuffle=True)
oof = pd.DataFrame()                 # out-of-fold result
models = []                          # models
scores = 0.0                         # validation score

gain_importance_list = []
split_importance_list = []


# In[34]:


get_ipython().run_cell_magic('time', '', 'for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):\n\n    print("Fold :", fold+1)\n    \n    # create dataset\n    X_train, y_train = X.loc[trn_idx], y[trn_idx]\n    X_valid, y_valid = X.loc[val_idx], y[val_idx]\n    \n    #RMSPE weight\n    weights = 1/np.square(y_train)\n    lgbm_train = lgbm.Dataset(X_train,y_train,weight = weights)\n\n    weights = 1/np.square(y_valid)\n    lgbm_valid = lgbm.Dataset(X_valid,y_valid,reference = lgbm_train,weight = weights)\n    \n    # model \n    model = lgbm.train(params=params,\n                      train_set=lgbm_train,\n                      valid_sets=[lgbm_train, lgbm_valid],\n                      num_boost_round=5000,         \n                      feval=feval_RMSPE,\n                      verbose_eval=100,\n                      categorical_feature = [\'stock_id\']                \n                     )\n    \n    # validation \n    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)\n\n    RMSPE = round(rmspe(y_true = y_valid, y_pred = y_pred),3)\n    print(f\'Performance of the\u3000prediction: , RMSPE: {RMSPE}\')\n\n    #keep scores and models\n    scores += RMSPE / 5\n    models.append(model)\n    print("*" * 100)\n    \n    # --- calc model feature importance ---\n    if DO_FEAT_IMP:    \n        feature_names = X_train.columns.values.tolist()\n        gain_importance_df = calc_model_importance(\n            model, feature_names=feature_names, importance_type=\'gain\')\n        gain_importance_list.append(gain_importance_df)\n\n        split_importance_df = calc_model_importance(\n            model, feature_names=feature_names, importance_type=\'split\')\n        split_importance_list.append(split_importance_df)\n')


# In[35]:


scores


# In[36]:


def calc_mean_importance(importance_df_list):
    mean_importance = np.mean(
        np.array([df['importance'].values for df in importance_df_list]), axis=0)
    mean_df = importance_df_list[0].copy()
    mean_df['importance'] = mean_importance
    return mean_df


# In[37]:


if DO_FEAT_IMP:
    mean_gain_df = calc_mean_importance(gain_importance_list)
    plot_importance(mean_gain_df, title='Model feature importance by gain')
    mean_gain_df = mean_gain_df.reset_index().rename(columns={'index': 'feature_names'})
    mean_gain_df.to_csv('gain_importance_mean.csv', index=False)


# In[38]:


if DO_FEAT_IMP:
    mean_split_df = calc_mean_importance(split_importance_list)
    plot_importance(mean_split_df, title='Model feature importance by split')
    mean_split_df = mean_split_df.reset_index().rename(columns={'index': 'feature_names'})
    mean_split_df.to_csv('split_importance_mean.csv', index=False)


# # Test set

# In[39]:


df_test.columns


# In[40]:


df_train.columns


# In[41]:


y_pred = df_test[['row_id']]
X_test = df_test.drop(['time_id', 'row_id'], axis = 1)


# In[42]:


X_test


# In[43]:


target = np.zeros(len(X_test))

#light gbm models
for model in models:
    pred = model.predict(X_test[X_valid.columns], num_iteration=model.best_iteration)
    target += pred / len(models)


# In[44]:


y_pred = y_pred.assign(target = target)


# In[45]:


y_pred


# In[46]:


y_pred.to_csv('submission.csv',index = False)

