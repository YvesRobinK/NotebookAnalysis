#!/usr/bin/env python
# coding: utf-8

# ### **DEMAND FORECASTING**
# 
# ##### Mission is to create a 3-month demand forecasting model for the relevant store chain using the following time series and machine learning techniques:
# * Random Noise
# * Lag Shifted Features
# * Rolling Mean Features
# * Exponentially Weighted Mean Features
# * Custom Cost Function
# * Model Validation with LightGBM 
# 
# Dataset is here: https://www.kaggle.com/c/demand-forecasting-kernels-only

# In[1]:


# Import necessary libraries and make necessary arrangements
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


# In[2]:


# HELPER FUNCTIONS (UTILS)

# Check dataframe
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Date Features
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

# Random Noise
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# Lag/Shifted Features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# Rolling Mean Features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

# Custom Cost Function
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Feature Importance
def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
        
# Kaggle input part
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


########################
# Loading the data
########################
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])
test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('../input/demand-forecasting-kernels-only/sample_submission.csv')
df = pd.concat([train, test], sort=False)


# #### **EXPLORATORY DATA ANALYSIS**

# In[4]:


# Let's check the time periods of train and test sets
df["date"].min(), df["date"].max()  


# In[5]:


train["date"].min(), train["date"].max()   


# In[6]:


test["date"].min(), test["date"].max()  


# In[7]:


check_df(train)


# In[8]:


check_df(test)


# In[9]:


check_df(df)


# In[13]:


# Distribution of sales
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])


# In[14]:


# Number of stores
df[["store"]].nunique()


# In[15]:


# Number of items
df[["item"]].nunique()


# In[16]:


# Number of unique items for each store
df.groupby(["store"])["item"].nunique()


# In[17]:


# Sales distribution per store and item
df.groupby(["store", "item"]).agg({"sales": ["sum"]})


# In[18]:


# Sales statistics per store and item
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


# #### **FEATURE ENGINEERING**

# In[20]:


########################
# Date Features
########################
df = create_date_features(df)
check_df(df)


# In[21]:


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


# In[22]:


########################
# Lag/Shifted Features
########################
# Below sort_values() is so important!
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)


# In[23]:


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


# In[24]:


check_df(df)


# In[25]:


########################
# Rolling Mean Features
########################
df = roll_mean_features(df, [365, 546])
df.tail()


# In[26]:


########################
# Exponentially Weighted Mean Features
########################
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)


# In[27]:


########################
# One-Hot Encoding
########################
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


# In[29]:


########################
# Converting sales to log(1+sales)
########################
df['sales'] = np.log1p(df["sales"].values)


# #### **MODEL**
# 
# * MAE: mean absolute error
# * MAPE: mean absolute percentage error
# * SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

# In[30]:


########################
# Time-Based Validation Sets
########################
# Train set till the beginning of 2017
train = df.loc[(df["date"] < "2017-01-01"), :]

# Validation set including first 3 months of 2017 (as we will forecast the first 3 months of 2018)
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]


# In[31]:


cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]


# In[32]:


# Define dependent variable and independent variables 
Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]


# In[33]:


# Observe the shapes
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


# In[34]:


########################
# LightGBM Model
########################
# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


# In[35]:


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


# In[36]:


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


# In[37]:


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)


# In[38]:


smape(np.expm1(y_pred_val), np.expm1(Y_val))


# In[39]:


########################
# Feature importance
########################
plot_lgb_importances(model, num=30, plot=True)


# In[40]:


########################
# Final Model
########################
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]


# In[41]:


lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# In[42]:


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)


# In[43]:


model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


# In[44]:


test_preds = model.predict(X_test, num_iteration=model.best_iteration)


# In[45]:


########################
# Submission
########################
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)


# #### **REFERENCES**
# * Data Science and Machine Learning Bootcamp, 2021, https://www.veribilimiokulu.com/
