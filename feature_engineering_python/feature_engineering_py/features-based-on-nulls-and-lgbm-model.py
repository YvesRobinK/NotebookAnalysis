#!/usr/bin/env python
# coding: utf-8

# > Understanding Null Values

# How missing values are handled in the data is a very important aspect of Machine learning problems. Some Data Scientists recoomend that if more than 70-75%  data in a feature/column are missing it's better to drop those features from the model. However if the entire data set has a high proportion of missing data then deleting individual features with higher proportion may work adversely in developing the model. The IEE-CIS Fraud Detection Competion data is one such case. As shown below 41%  of all the data entries in the Train_transaction file are null values.

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import os
import time
import sys
import datetime
print(os.listdir("../input"))


# In[2]:


#train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
#test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[3]:


nrows = len(train_transaction)
ncols= len(train_transaction.columns)
tot_null = train_transaction.isnull().sum().sum()
print('Number of rows in Train_transaction file : ',nrows)
print('Number of columns in Train_transaction file : ',ncols)
print('Number of entries in Train_transaction file : ',ncols*nrows)
print('Number of Null value entries in Train_transaction file : ',tot_null)
print('Percentage  of Null value entries in Train_transaction file : ',round(tot_null/(ncols*nrows)*100,2))


# Let's look at which columns have highest number of null values. It's interesting to observe ,from the plot of percentage of missing values against the Number of columns ,that many columns have exactly the same number of missing values. Columns with same percent of missing values can be grouped.

# In[4]:


def percent_na(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing},index=None)
    #missing_value_df.sort_values('percent_missing', inplace=True)
    missing_value_df=missing_value_df.reset_index().drop('index',axis=1)
    return missing_value_df
train_transaction_na = percent_na(train_transaction)


# In[5]:


sns.set(rc={'figure.figsize':(16,10)})
col_na_count=train_transaction_na.percent_missing.value_counts()[train_transaction_na.percent_missing.value_counts()>1]
col_na_count.index = col_na_count.index.to_series().apply(lambda x: np.round(x,2))
plot=sns.barplot(col_na_count.index,col_na_count.values)
for p in plot.patches:
             plot.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='top', fontsize=12, color='black', xytext=(0, 20),
                 textcoords='offset points')
plot=plot.set(xlabel='% of Missing Values in a column ',ylabel= 'Number of columns')


# Shown below are  the column_groups with exactly the same number of missing values. It's interesting to note that these  each group is almost a continous series of column names.

# In[6]:


pd.options.display.max_colwidth =300
col_na_group= train_transaction_na.groupby('percent_missing')['column_name'].unique().reset_index()
num_columns=[]
for i in range(len(col_na_group)):
    num_columns.append(len(col_na_group.column_name[i]))
col_na_group['num_columns']=num_columns
col_na_group = col_na_group.loc[(col_na_group['num_columns']>1) & (col_na_group['percent_missing']>0),].sort_values(by='percent_missing',ascending=False).reset_index()
col_na_group


# Since all columns  within the  a column group have the same missing value percentage ,in a particular row of the dataframe all columns will have null values or all columns will have non null values. Let's do a value_count of data entries in all the rows for the the column group with 46 columns and missing value percentage  77.913435 to confirm this. 

# In[7]:


col_group = col_na_group.column_name[4]
col_group_row_na = train_transaction[col_group].apply(lambda x: x.count(), axis=1)
col_group_row_na.value_counts()


# As expected in every row either none of the columns within a group have a  null value or all the columns within a group have null values.
# 
# Overall Fraud(isFraud=1) and non-Fraud(isFraud=0) plot is shown below.
# 

# In[8]:


sns.set(rc={'figure.figsize':(10,4)})
prop= train_transaction['isFraud'].value_counts(normalize=True).mul(100).round(2)
plot= sns.barplot(x=prop.index,y=prop.values)
for p in plot.patches:
    plot.annotate("%10.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()/2),
               ha='center', va='center', fontsize=8, color='black', xytext=(0, 20),
                 textcoords='offset points')


# What would be interesting is to check within a particular group of columns whether there is any difference in Proportion of Fraud transactions between the rows with all values missing for the columns in that group and rows with values for columns in that group compared to the overall distribution. The graph below shows that distribution

# In[9]:


sns.set(rc={'figure.figsize':(12,80)})
plrow= len(col_na_group)
x=1
for num,group in enumerate(col_na_group.column_name):
    col_group_row_na = train_transaction[group].apply(lambda x: x.count(), axis=1)
    plt.subplot(plrow,2,x)
    prop= train_transaction.loc[col_group_row_na==0,]['isFraud'].value_counts(normalize=True).mul(100).round(2)
    plot= sns.barplot(x=prop.index,y=prop.values)
    for p in plot.patches:
            plot.annotate("%10.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()/2),
               ha='center', va='center', fontsize=8, color='black', xytext=(0, 20),
                 textcoords='offset points')
    per = round(col_na_group['percent_missing'][num],3)
    plot.set_title(label= ' By rows with no values in Columns with ' +str(per) +"% missing values")
    plot.set(xlabel= 'Number of rows with all missing values : ' +str(len(train_transaction.loc[col_group_row_na==0,])))
    plt.subplot(plrow,2,x+1)
    prop= train_transaction.loc[col_group_row_na!=0,]['isFraud'].value_counts(normalize=True).mul(100).round(2)
    plot= sns.barplot(x=prop.index,y=prop.values)
    for p in plot.patches:
            plot.annotate("%10.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()/2),
               ha='center', va='center', fontsize=8, color='black', xytext=(0, 20),
                 textcoords='offset points')
    plot.set_title(label= 'By rows with values in Columns with ' +str(per) +"% missing values")
    plot.set(xlabel= 'Number of rows with values : ' +str(len(train_transaction.loc[col_group_row_na!=0,])))
    x= x+2
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# It looks like incase of some column groups there appears to be a significant difference in the proportion of fraud cases between non null value rows and the null value rows.
# 
# 

# **Conclusion**

# * The data set has multiple groups of related features/columns that possibly can be combined to create new features.
# * Presence or absence of data in these related feature/column groups may be an important feature in developing models.
# 

# > Feature Engineering

# Let us create some new features based on the group of columns with  same percent of null values
# 

# In[10]:


for num,alpha in enumerate(col_na_group.column_name):
    train_transaction['group_mean_'+str(num)] = train_transaction[alpha].mean(axis=1)
    train_transaction['group_std_'+str(num)] = train_transaction[alpha].std(axis=1)
    test_transaction['group_mean_'+str(num)] = test_transaction[alpha].mean(axis=1)
    test_transaction['group_std_'+str(num)] = test_transaction[alpha].std(axis=1)


# Among the C Columns some of the columns are Pseudo categorical in nature . Refer kernel https://www.kaggle.com/rajeshcv/exploring-c-columns  for details. A new feature is developed based on this. Based on transaction date transaction hour and weekday features are also created.

# In[11]:


train_transaction['Trans_hour']=pd.to_datetime(train_transaction['TransactionDT'],unit='s').dt.hour
train_transaction['Trans_weekday']=pd.to_datetime(train_transaction['TransactionDT'],unit='s').dt.weekday
train_transaction['groupCpseudoCat_mean'] = train_transaction[['C4','C7','C8','C10','C12']].mean(axis=1)
test_transaction['Trans_hour']=pd.to_datetime(test_transaction['TransactionDT'],unit='s').dt.hour
test_transaction['Trans_weekday']=pd.to_datetime(test_transaction['TransactionDT'],unit='s').dt.weekday
test_transaction['groupCpseudoCat_mean'] = test_transaction[['C4','C7','C8','C10','C12']].mean(axis=1)


# > Label Encoding

# In[12]:


y =train_transaction['isFraud']
train_transaction.drop(['TransactionID','isFraud'],axis=1,inplace=True)
test_id = test_transaction['TransactionID']
test_transaction.drop(['TransactionID'],axis=1,inplace=True)


# Let's encode the object columns to replace string values with numericals. Here the labe encoding id done without changing null values.
# 
# A check is done to find if the values in the object type columns of train and test dataset are the same

# In[13]:


objcols=[cname for cname in train_transaction.columns if train_transaction[cname].dtype == "object"]
for num,alpha in enumerate(objcols):       
    missing = [x for x in test_transaction[alpha].unique().tolist() if x not in train_transaction[alpha].unique().tolist()]
    if len(missing) >0:
        print(alpha) 
        print(missing)


# P_emaildomain has a value 'scranton.edu' in test dataset which is not in the train dataset. We will replace this with null value

# In[14]:


test_transaction.loc[test_transaction.P_emaildomain=='scranton.edu','P_emaildomain'] =np.nan


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoders = dict()
for col_name in train_transaction[objcols].columns:
    series = train_transaction[col_name]
    label_encoder = LabelEncoder()
    train_transaction[col_name] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    )
    encoders[col_name] = label_encoder


# In[16]:


for num,alpha in enumerate(objcols):
    series = test_transaction[alpha]
    test_transaction[alpha] = pd.Series(
        encoders[alpha].transform(series[series.notnull()]),
        index=series[series.notnull()].index
    )


# > Model Building

# In[17]:


features = [c for c in train_transaction.columns]


# In[18]:


param = {
        'num_leaves': 10,
        'max_bin': 127,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False
       
    }


# In[19]:


import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1337)
oof1 = np.zeros(len(train_transaction))
predictions1 = np.zeros(len(test_transaction))
feature_importance_df1 = pd.DataFrame()

start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_transaction.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_transaction.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train_transaction.iloc[val_idx][features], label=y.iloc[val_idx])

    num_round = 10000
    clf1 = lgb.train(param, trn_data, num_boost_round=10000,valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 500)
    oof1[val_idx] = clf1.predict(train_transaction.iloc[val_idx][features], num_iteration=clf1.best_iteration)
    
    fold_importance_df1 = pd.DataFrame()
    fold_importance_df1["feature"] = features
    fold_importance_df1["importance"] = clf1.feature_importance()
    fold_importance_df1["fold"] = fold_ + 1
    feature_importance_df1 = pd.concat([feature_importance_df1, fold_importance_df1], axis=0)
    predictions1 += clf1.predict(test_transaction[features], num_iteration=clf1.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(y,oof1)))


# > Feature importance

# Feature importance by top 100 features is shown below

# In[20]:


cols = (feature_importance_df1[["feature", "importance"]]
          .groupby("feature")
          .mean()
          .sort_values(by="importance", ascending=False)[:100].index)
best_features = feature_importance_df1.loc[feature_importance_df1.feature.isin(cols)]
plt.figure(figsize=(14,25))
sns.barplot(x="importance",
             y="feature",
             data=best_features.sort_values(by="importance",
                                             ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()


# > Results

# In[21]:


submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud']= predictions1
submission.to_csv('submission_LGBM.csv',index=False)

