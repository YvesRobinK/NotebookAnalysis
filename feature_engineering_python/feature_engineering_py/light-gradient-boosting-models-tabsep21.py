#!/usr/bin/env python
# coding: utf-8

# # Light Gradient Boosting Model testing
# Aim of this notebook is to review the light gradient boosting model which can be used during a binary classification challenge.

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Light Gradient Boosting

# In[2]:


# Import modules for model analysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Import lightgbm modules
import lightgbm as lgb


# In[3]:


# Read in the data
train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv',index_col=0)
test  = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv', index_col=0)

train.head()


# In[4]:


# Check the memory consumed by the DataFrame
train.info(memory_usage='deep')


# In[5]:


# Memory usage by variable in MB
train.memory_usage(deep=True) * 1e-6


# In[6]:


# Lets reduce the memory usage of the features
# First - check the integer values and downcast
def int_downcast(df):
    int_cols = df.select_dtypes(include=['int64'])

    for col in int_cols.columns:
        print(col, 'min:',df[col].min(),'; max:',df[col].max())
        df[col] = pd.to_numeric(df[col], downcast ='integer')
    return df

int_downcast(train)
train.memory_usage(deep=True) * 1e-6


# In[7]:


# Second - check the float values and downcast. Method will have to be applied to the train and test DataFrames
def float_downcast(df):
    float_cols = df.select_dtypes(include=['float64'])

    for col in float_cols.columns:
#         print(col, 'min:',df[col].min(),'; max:',df[col].max())
        df[col] = pd.to_numeric(df[col], downcast ='float')
    return df

float_downcast(train)
float_downcast(test)


# In[8]:


# Check the memory usage by feature
train.memory_usage(deep=True) * 1e-6
test.memory_usage(deep=True) * 1e-6


# In[9]:


# Review the memory usage by DataFrame
train.info(memory_usage='deep')
test.info(memory_usage='deep')


# # Missing value treatment

# In[10]:


# Check for missing values
train.isnull().sum()
test.isnull().sum()

# Add a dummy missing value for a row with missing data
features = [x for x in train.columns.values if x[0]=="f"]
train['n_missing'] = train[features].isna().sum(axis=1)
test['n_missing'] = test[features].isna().sum(axis=1)


# # Model Analysis

# In[11]:


X = train.drop('claim', axis=1)
y = train['claim']


# In[12]:


# Prepare the data to be used within the model. Make use of the lgb.Dataset() method to optimise the memory usage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)


# # Explore Hyperparameters

# In[13]:


# Evaluate models
def eval_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)


# ## Number of Trees

# In[14]:


# # List of models
# def list_models():
#     models = dict()
#     trees = [10, 50, 100, 500, 1000, 5000]
#     for n in trees:
#         models[str(n)] = LGBMClassifier(
#             device='gpu',
#             objective='binary',
#             n_estimators=n
#         )
#     return models


# In[15]:


# # Models to review
# models = list_models()
# # Evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
#     pred = eval_model(model)
#     results.append(pred)
#     names.append(name)
#     print(f'Trees = {name} {np.mean(pred)} ({np.std(pred)})')
# # plot comparisons
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()


# 500 Trees appears to be the right value to go with

# In[16]:


# # List of models
# def list_models():
#     models = dict()
#     for n in range(1, 11):
#         models[str(n)] = LGBMClassifier(
#             device='gpu',
#             objective='binary',
#             max_depth=n,
#             num_leaves=2**n
#         )
#     return models


# ## Learning Rate

# In[17]:


# # List of models
# def list_models():
#     models = dict()
#     rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
#     for r in rates:
#         key = '%.4f' % r
#         models[key] = LGBMClassifier(
#             device='gpu',
#             objective='binary',
#             learning_rate = r
#         )
#     return models


# ## Boosting Type

# In[18]:


# # List of models
# def list_models():
#     models = dict()
#     types = ['gbdt', 'dart', 'goss']
#     for t in types:
#         models[t] = LGBMClassifier(
#             device='gpu',
#             objective='binary',
#             boosting_type = t
#         )
#     return models


# # Using LGB dataset method and train

# In[19]:


# Review using the LGB dataset and model build methods
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

print(f'{type(lgb_train)}')
print(f'{lgb_train.data.info()}')


# In[20]:


print(type(lgb_train))
lgb_train.data.head()


# In[21]:


# Specify the configurations as a dict
params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0,
    'device': 'gpu'
}

# train - verbose_eval option switches off the log outputs
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=lgb_eval,
    early_stopping_rounds=100,
    verbose_eval=-1,
)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# Compute and print metrics
print(f"AUC : {roc_auc_score(y_test, y_pred)}")


# In[22]:


# Feature importance
lgb.plot_importance(gbm, max_num_features=15);
plt.show()


# In[23]:


# Let's create a function to allow for future quick reviews of the same baseline model. Will allow for easy review of feature engineering and selection processing steps
def base_model(train, dep):
    
    # Create feature variables
    X = train
    y = dep
    
    # Prepare the data to be used within the model. Make use of the lgb.Dataset() method to optimise the memory usage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)
    
    # Review using the LGB dataset and model build methods
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    # Run the model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0,
        'device': 'gpu'
    }

    # train - verbose_eval option switches off the log outputs
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=lgb_eval,
        early_stopping_rounds=100,
        verbose_eval=-1,
    )

    # predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # Compute and print metrics
    print(f"AUC : {roc_auc_score(y_test, y_pred)}")
    return model


# # Make submission

# In[24]:


def submission_sample(model, df_test, model_name):
    sample = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
    sample['claim'] = model.predict(df_test)
    return sample.to_csv(f'submission_{model_name}.csv',index=False)


# In[25]:


# Baseline submission - original code versions
# submission_sample(lgbm_clf, 'lgbm_base')
# Using the hyperparameter tuning
# submission_sample(pipe_lgbm_clf, 'lgbm_hyper')
# submission_sample(lgbm_clf_t2, 'lgbm_t2')

submission_sample(gbm, test, 'lgb_base')


# ## Feature Engineering
# ***
# After creating the initial baseline model we can start to perform some feature engineering steps. With feature engineering we are aiming to see if additional variables can be created that will help to improve the model.
# ***
# 1. Binning
#     * Create binned values (quantiles, deciles)
# 2. Feature scaling
#     * MinMax scaling
#     * Standardization
#     * Winsorizing 
# 3. Statistical transformations
#     * Log
#     * Polynomials
# 4. Feature Interactions
#     * Use PolynomialFeatures
# ***
# Prior to this feature engineering we can review teh missing value replacement assessment.
# * Replace with mean / median / mode
# * End of tail imputation - works best with normally distributed features

# Lets go back to reviewing the Train and Test DataFrames
# 

# In[26]:


# Lets confirm the feature data types
print(f'Train : \n{train.dtypes.value_counts()}')
print(f'Test : \n{test.dtypes.value_counts()}')


# ### Review missing value replacement

# In[27]:


# Take copies of the original DataFrames so the originals are not overwritten
train_miss = train.copy()
test_miss = test.copy()


# In[28]:


train_miss = train_miss.drop('claim', axis=1)
print(train_miss.shape, test_miss.shape)


# In[29]:


# Missing values by features
train_miss.isnull().sum()


# In[30]:


# List of column names for review
column_names = [col for col in train_miss.columns]


# In[31]:


# Create function for the missing value review
def impute_miss_values(df_train, df_test, strategy='mean'):
    # create the imputer, the strategy can be mean and median.
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

    # fit the imputer to the train data
    imputer.fit(df_train)

    # apply the transformation to the train and test
    train_imp = pd.DataFrame(imputer.transform(df_train), columns=column_names)
    test_imp = pd.DataFrame(imputer.transform(df_test), columns=column_names)
    return train_imp, test_imp


# In[32]:


# Create the DataFrame's as a test run
# train_miss_mean, test_miss_mean = impute_miss_values(train_miss, test_miss)


# In[33]:


train_miss_median, test_miss_median = impute_miss_values(train_miss, test_miss, strategy='median')
# train_miss_mode, test_miss_mode = impute_miss_values(train_miss, test_miss, strategy='most_frequent')


# In[34]:


# NOTE: method doesn't appear to be working. Need to review

# # Run each of the missing value options
# miss_val_methods = ['mean', 'median', 'most_frequent']
# train_str = ['train_miss_' + method for method in miss_val_methods]
# test_str = ['test_miss_' + method for method in miss_val_methods]

# for miss, train_tb, test_tb in zip(miss_val_methods, train_str, test_str):
#     print(miss, train_tb, test_tb)
#     train_tb, test_tb = impute_miss_values(train_miss, test_miss, strategy=miss)


# In[35]:


# Check impact on the model
# lgb_miss_mean = base_model(train_miss_mean, dep=train['claim'])
lgb_miss_median = base_model(train_miss_median, dep=train['claim'])
# lgb_miss_mode = base_model(train_miss_mode, dep=train['claim'])


# Median value replacement has helped to benefit the score accuracy the most. Lets review end of tail imputation as a comparison

# In[36]:


# pip install feature_engine


# In[37]:


# # Import module required
# from feature_engine.missing_data_imputers import EndTailImputer

# # Create function for the missing value review
# def impute_end_tail(df_train, df_test):
#     # create the imputer
#     imputer = EndTailImputer(distribution='gaussian', tail='right')

#     # fit the imputer to the train data
#     imputer.fit(df_train)

#     # apply the transformation to the train and test
#     train_imp = pd.DataFrame(imputer.transform(df_train), columns=column_names)
#     test_imp = pd.DataFrame(imputer.transform(df_test), columns=column_names)
#     return train_imp, test_imp


# In[38]:


# train_miss_eot, test_miss_eot = impute_end_tail(train_miss, test_miss)


# In[39]:


# submission_sample(lgb_miss_mean, test_miss_mean, 'lgb_miss_mean')
submission_sample(lgb_miss_median, test_miss_median, 'lgb_miss_median')
# submission_sample(lgb_miss_mode, test_miss_mode, 'lgb_miss_mode')


# ### Binning

# In[40]:


# Lets keep the median imputation prior to the binning
train_miss_median.head()


# In[41]:


# As all columns are numeric we don't have to specify data type
def binning(df, cut=4):
    for col in df.columns:
        df[col+'_grp'] = pd.qcut(df[col], cut, duplicates='drop', labels=False)
    return df


# In[42]:


# NOTE: notebook is running out of memory 

# train_bin0 = train_miss_median.copy()
# test_bin0 = test_miss_median.copy()
# # Use default bins e.g. quartiles
# train_bin = binning(train_bin0)
# test_bin = binning(test_bin0)
# # Try using decile bins
# train_bin_dec = binning(train_bin0, cut=10)
# test_bin_dec = binning(test_bin0, cut=10)


# In[43]:


# Check the model outputs
# lgb_bin = base_model(train_bin, dep=train['claim'])
# lgb_bin_dec = base_model(train_bin_dec, dep=train['claim'])


# In[44]:


# train.loc[:, ['n_missing', 'n_missing_q']].head()


# # Feature Selection
# ***
# Aims to reduce the dimensionality of the dataset
# ***
# 1. Remove co-linear features
# 2. Remove features with large number of missing values
# 3. Keep importance features

# In[ ]:




