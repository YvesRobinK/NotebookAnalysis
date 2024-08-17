#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# This notebook is intended to be a starter for anyone who wants to make attempt to this challenge.  
# 
# Many people are struggling to manage with datasets or they are relying on datasets created by other Kagglers. Some also try to move datasets to ther cloud services to have more compute, memory and storage.
# 
# **I wanted to have a noebook that we can run on Kaggle only without need of any other cloud service or datasets.**
# 
# Here is a humble attempt. I request my fellow Kagglers to give suggestions and improve performance.

# # **How to Use This Notebook**

# If you are running this notebook for very first time, then follow below steps:-
# * Uncomment preprocessing steps for level 1.
# * Save Version with "Save & Run All" option. This will run notebook in background.
# * Once you receive notification, output folder will have processed training and test data.
# * In case your session is reset and you lose files in output folder, you can add notbook output datasets using "Add Data" option.
# 
# # **If you are using supporting datasets for this notebook then start from section "Train Model".**
# 
# 
# 

# # **Setup**

# In[38]:


import vaex
vaex.multithreading.thread_count_default = 8
import vaex.ml

import pandas as pd
import numpy  as np 

import os
import gc
import psutil
import glob


# # **Utility Functions**

# In[39]:


def remove_output_files(file_pattern):
    fileList = glob.glob(file_pattern)
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


# In[40]:


def fill_and_convert_floats(ddf):
    for c in ddf.columns:
        if ddf[c].dtype == 'float64':
            ddf[c] = ddf[c].fillna(0.0).astype('float32')
    return ddf

def encode_cat_features(df):
    cat_features = ['D_63','D_64']
    label_encoder = vaex.ml.LabelEncoder(features=cat_features)
    df = label_encoder.fit_transform(df)
    df.drop(cat_features, inplace=True)
    df.rename('label_encoded_D_63','D_63')
    df.rename('label_encoded_D_64','D_64')
    df['D_64'] = df['D_64'].astype('float32')
    df['D_63'] = df['D_63'].astype('float32')
    df['B_31'] = df['B_31'].astype('float32')
    return df

def get_last_statement(df):
    return df.groupby(['customer_ID']).agg({col: vaex.agg.last(col) for col in df.get_column_names() if col not in ["customer_ID"]})


# In[41]:


def get_last_statement_ex(df_test):
    delinquency_features = [col for col in df_test if col.startswith('D_')] 
    df = df_test.groupby(['customer_ID']).agg({col: vaex.agg.last(col) for col in df_test.get_column_names() if col not in delinquency_features + ["customer_ID"]})
    df.export_hdf5('./last-statement-p1.hdf5')
    del df
    gc.collect()
    delinquency_features = ['S_2'] + [col for col in df_test if col.startswith('D_') and len(col) == 4] 
    delinquency_features2 = ['S_2'] + [col for col in df_test if col.startswith('D_') and len(col) == 5]
    df_2 = df_test.groupby(['customer_ID']).agg({col: vaex.agg.last(col) for col in df_test.get_column_names() if col in delinquency_features})
    df_2.export_hdf5('./last-statement-p2.hdf5')
    del df_2
    gc.collect()
    df_3 = df_test.groupby(['customer_ID']).agg({col: vaex.agg.last(col) for col in df_test.get_column_names() if col in delinquency_features2})
    df_3.export_hdf5('./last-statement-p3.hdf5')
    del df_3
    gc.collect()
    last_statement_p1 = vaex.open('./last-statement-p1.hdf5')
    last_statement_p2 = vaex.open('./last-statement-p2.hdf5')
    last_statement_p3 = vaex.open('./last-statement-p3.hdf5')
    last_statement_p1 = last_statement_p1.drop('S_2')
    last_statement_p2 = last_statement_p2.drop('S_2')
    last_statement_p3 = last_statement_p3.drop('S_2')
    gc.collect()
    last_statement_p1 = last_statement_p1.join(last_statement_p2, how="inner", on='customer_ID')
    df_last_statement = last_statement_p1.join(last_statement_p3, how="inner", on='customer_ID')
    del last_statement_p1
    del last_statement_p2
    del last_statement_p3
    gc.collect()
    statement_path = './last-statement-p*.hdf5'
    remove_output_files(statement_path)
    return df_last_statement


# In[42]:


def process_data(data):
    for i, df in enumerate(vaex.from_csv(f'../input/amex-default-prediction/{data}.csv', chunk_size=500_000)):
        df['S_2'] = df['S_2'].str.replace('-','').astype('float32')
        #df['R_26'] = df['R_26'].astype('int16')
        df = fill_and_convert_floats(df)
        df = encode_cat_features(df)
        df = get_last_statement(df)
        export_path = f'./{data}_{i:02}.hdf5'    
        df.export_hdf5(export_path)
        del df
        gc.collect()
    import_path = f'./{data}_*.hdf5'
    df = vaex.open(import_path)
    df.export_hdf5(f'./{data}.hdf5')
    del df
    gc.collect()
    remove_output_files(import_path)
        


# In[43]:


def process_data_level2(df, data, flag):
    if flag == 1:
        df = get_last_statement_ex(df)     
    else:
        df = get_last_statement(df)
        df.drop('S_2', inplace=True)   
    label_encoder = vaex.ml.LabelEncoder(features=['customer_ID'])
    df = label_encoder.fit_transform(df)
    df_customer_map = df[['label_encoded_customer_ID', 'customer_ID']]
    df.drop('customer_ID', inplace=True)
    df.rename('label_encoded_customer_ID','customer_ID')
    df_customer_map.export_hdf5(f'./{data}_customer_map.hdf5')
    df.export_hdf5(f'./{data}v2.hdf5')
    del df
    del df_customer_map
    gc.collect()


# In[44]:


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# In[45]:


def amex_metric_np(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric_np(y_true, y_pred), True


# # **Preprocessing - Level 1 - Reduce Train and Test Datasets**

# In this step train and test datasets will be processed to reduce the size so we can perform feature engineering and build models without worrying about memory, CPU and hard disk constraints.
# 
# Uncomment below lines only if you need to create reduced version of data. **If size reduced data is already availlable then skip these steps.**
# 
# Below changes are performed on datasets:-
# * S_2 date feature is changed to float32
# * All float64 features are converted to float32
# * Missing values in all numeric features are set to 0.0
# * Categorical features D_63 and D_64 are encoded - This was required to extract last statement
# * Categorical features B_31 is converted to float32 - This was required to extract last statement
# * For each customer only last statement is kept as available in each chunk
# 

# **Uncomment below two lines to generate level 1 data. Alternatively you can use this [dataset](https://www.kaggle.com/datasets/mirfanazam/amex-prediction-starter-level-1).**

# In[46]:


# process_data('train_data')


# In[47]:


# process_data('test_data')


# # **Read Train and Test Data Preprocessed at Level 1**

# Use below lines to use train and test data from output folder. This is required when you want to run this notebook as a whole and process data, train model, make prediction and make submission.

# In[48]:


# df_train = vaex.open('./train_data.hdf5')
# df_test = vaex.open('./test_data.hdf5')


# Use below lines if you want to load level 1 data from input folder.

# In[49]:


# df_train = vaex.open('../input/amex-prediction-starter-level-1/train_data.hdf5')
# df_test = vaex.open('../input/amex-prediction-starter-level-1/test_data.hdf5')


# # **Preprocessing - Level 2**

# In this step both training and test datasets will be processed: -
# 
# * Keep only last statement for each customer
# * Remove date feature S_2
# * Encode Customer ID
# 

# **Uncomment below two lines to generate level 2 data. Alternatively you can use this [dataset](https://www.kaggle.com/datasets/mirfanazam/amex-prediction-starter-level-2).**

# In[50]:


# process_data_level2(df_train, 'train_data', 0)


# In[51]:


# process_data_level2(df_test, 'test_data', 1)


# # **Read Train and Test Data Preprocessed at Level 2**

# Use below lines to use train and test data from output folder. This is required when you want to run this notebook as a whole and process data, train model, make prediction and make submission.

# In[52]:


# df_train = vaex.open('./train_datav2.hdf5')
# df_test = vaex.open('./test_datav2.hdf5')
# df_train_map = vaex.open('./train_data_customer_map.hdf5')
# df_test_map = vaex.open('./test_data_customer_map.hdf5')


# Use below lines if you want to load level 2 data from input folder.

# In[53]:


df_train = vaex.open('../input/amex-prediction-starter-level-2/train_datav2.hdf5')
df_test = vaex.open('../input/amex-prediction-starter-level-2/test_datav2.hdf5')
df_train_map = vaex.open('../input/amex-prediction-starter-level-2/train_data_customer_map.hdf5')
df_test_map = vaex.open('../input/amex-prediction-starter-level-2/test_data_customer_map.hdf5')


# # **Prepare Data to Train Model**

# In[54]:


df_train_labels = vaex.open('../input/amex-default-prediction/train_labels.csv')
df_train_labels = df_train_labels.join(df_train_map, how="inner", on="customer_ID")

all_features = [col for col in df_train]

df_customer = df_train[all_features]
df_customer = df_customer.join(df_train_labels, left_on='customer_ID', right_on='label_encoded_customer_ID', how='inner')
df_customer.drop(['label_encoded_customer_ID'], inplace=True)


# # **Convert Vaex DataFrames to Pandas DataFrames**

# **There is not much information available for Veax ML wrappers. So we will convert our dataframes from Vaex to Pandas.**

# In[55]:


df_train = df_train.to_pandas_df()
df_test = df_test.to_pandas_df()
df_train_map = df_train_map.to_pandas_df()
df_test_map = df_test_map.to_pandas_df()

df_train_labels = df_train_labels.to_pandas_df()
df_customer = df_customer.to_pandas_df()


# # **Train Model**

# In[56]:


import lightgbm as lgb
from lightgbm import log_evaluation
from sklearn.model_selection import train_test_split


# In[57]:


y = df_customer.pop('target')
# c = df_customer.pop('customer_ID')
# model_features = [col for col in df_customer if col not in ["customer_ID"]]
model_features = [col for col in df_customer]
X = df_customer[model_features]


# In[58]:


# X.drop(columns={'D_139', 'D_103'},inplace=True)


# In[59]:


X.shape


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[61]:


dtrain = lgb.Dataset(
    data=X_train,
    label=y_train
)

dvalid = lgb.Dataset(
    data=X_test,
    label=y_test,
    reference=dtrain
)


# In[62]:


categorical_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
df_customer['D_117'] = df_customer['D_117'] + 1
df_customer['D_126'] = df_customer['D_126'] + 1


# In[63]:


# df_customer[categorical_features].describe()


# In[74]:


lgb_params={
    "objective": "binary",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "reg_lambda": 50,
    "min_child_samples": 2400,
    "num_leaves": 220,
    "colsample_bytree": 0.19,
#     device='gpu',
    "random_state": 1,
    'verbose': -1
}
    
model = lgb.train(
    params=lgb_params,
    train_set=dtrain,
    valid_sets=[dvalid],
    feval=lgb_amex_metric,
    callbacks=[log_evaluation(100)]
)


# In[65]:


gc.collect()


# In[66]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred


# # **Make Prediction**

# In[70]:


model_features = [col for col in df_customer]
df_customer_test = df_test[model_features]


# In[71]:


# df_customer_test = df_customer_test.drop(columns={'D_139', 'D_103'})


# In[72]:


df_customer_test_pred = model.predict(df_customer_test)


# In[73]:


df_customer_test_pred


# # **Make Submission**

# In[ ]:


df_customer_test


# In[ ]:


df_customer_test = pd.merge(df_customer_test, df_test_map, how="inner", left_on="customer_ID", right_on="label_encoded_customer_ID")


# In[ ]:


df_customer_test = df_customer_test.drop(columns={'customer_ID_x','label_encoded_customer_ID'})


# In[ ]:


df_customer_test_pred = pd.DataFrame(df_customer_test_pred.tolist())


# In[ ]:


df_customer_test['prediction'] = df_customer_test_pred


# In[ ]:


df_customer_test


# In[ ]:


df_customer_test = df_customer_test.rename(columns={"customer_ID_y": "customer_ID"})


# In[ ]:


final_prediction = df_customer_test[["customer_ID", "prediction"]]


# In[ ]:


final_prediction.to_csv("./submission.csv",index=False)

