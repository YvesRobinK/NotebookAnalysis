#!/usr/bin/env python
# coding: utf-8

# # **WARNINGS**!

# ## Reference

# * https://www.kaggle.com/code/metathesis/feature-engineering-training-with-ta

# 
# 
# ## Which features ?
# 

# *** Merging training features and some of stock list features**

# In[ ]:


#best_features


# ## Score ?

# * **Validation Score = 0.52**

# ## When ı use only training features, Score?

# * **Validation Score = 0.1**

# ## Where is the Feature İmportance ?

# * **You can find the feature importance at the end of the notebook**

# ## Select best features (18) and parameter tuning

# * **Score : 0.64**

# In[ ]:


get_ipython().system('pip install --upgrade ta')


# In[66]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[67]:


import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from tqdm import tqdm

import ta
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')


# In[69]:


# set base_dir to load data
base_dir = "../input/jpx-tokyo-stock-exchange-prediction"
# There are three types of stock_price.csv
# We use one in the train_files folder for this notebook.
train_files_dir = f"{base_dir}/train_files"


# In[70]:


def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price


# In[71]:


# load stock price data
df_price = pd.read_csv(f"{train_files_dir}/stock_prices.csv")

# generate AdjustedClose
df_price = adjust_price(df_price)


# In[72]:


from ta import add_all_ta_features
from ta.utils import dropna


# # Pre-processing for model building
# 
# This notebook presents a simple model using LightGBM.
# 
# First, the features are generated using the price change and historical volatility described above.

# # Add Stock List Features to Train Features, but How? 

# In[73]:


df_stocklist = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
df_stocklist=df_stocklist[df_stocklist["Universe0"]==True]

df_stocklist.replace("-", np.nan,inplace=True)
df_stocklist.drop(columns=["Name"],inplace=True)

for col in ['33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode']:
        df_stocklist[col] = df_stocklist[col].astype(float)
        
obj_df = df_stocklist.select_dtypes(include=['object']).copy()

df_stocklist = pd.get_dummies(df_stocklist, columns=obj_df.columns)


# In[74]:


df_price.reset_index(inplace=True)


# In[75]:


_stock_list = df_stocklist.copy()
_stock_list.rename(columns={'Close': 'Close_x'}, inplace=True)
df_price = df_price.merge(_stock_list, on='SecuritiesCode', how="left")


# In[76]:


df_price.set_index("Date",inplace=True)


# In[77]:


liste = list(df_price.columns[:10])+list(df_price.columns[11:])  # get rid of Target (10)


# In[79]:


def get_features_for_predict(price, code):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
        code (int)  : A local code for a listed company
    Returns:
        feature DataFrame (pd.DataFrame)
    """
    close_col = "AdjustedClose"
    
    feats = price.loc[price["SecuritiesCode"] == code].copy()
    
    liste = list(df_price.columns[:10])+list(df_price.columns[11:])
    
    # Adds all 42 features
    feats = df_price.loc[df_price["SecuritiesCode"] == code, liste].copy()

    # To only add specific features
    # Example: https://github.com/bukosabino/ta/blob/master/examples_to_use/bollinger_band_features_example.py
    # df['bb_bbm'] = indicator_bb.bollinger_mavg()
    # df['bb_bbh'] = indicator_bb.bollinger_hband()
    # df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # filling data for nan and inf
    feats = feats.fillna(0)
    feats = feats.replace([np.inf, -np.inf], 0)
    # drop AdjustedClose column
    feats = feats.drop([close_col], axis=1)

    return feats


# In[80]:


# fetch prediction target SecuritiesCodes
codes = sorted(df_price["SecuritiesCode"].unique())
len(codes)


# In[81]:


# generate feature
buff = []
for code in tqdm(codes):
    feat = get_features_for_predict(df_price, code)
    buff.append(feat)
feature = pd.concat(buff)


# # Label creation
# 
# Next, we obtain the labels to be used for training the model (this is where we load and split the label data).

# In[82]:


def get_label(price, code):
    """ Labelizer
    Args:
        price (pd.DataFrame): dataframe of stock_price.csv
        code (int): Local Code in the universe
    Returns:
        df (pd.DataFrame): label data
    """
    df = price.loc[price["SecuritiesCode"] == code].copy()
    df.loc[:, "label"] = df["Target"]

    return df.loc[:, ["SecuritiesCode", "label"]]


# In[83]:


# split data into TRAIN and TEST
TRAIN_END = "2019-12-31"
# We put a week gap between TRAIN_END and TEST_START
# to avoid leakage of test data information from label
TEST_START = "2020-01-06"

def get_features_and_label(price, codes, features):
    """
    Args:
        price (pd.DataFrame): loaded price data
        codes  (array) : target codes
        feature (pd.DataFrame): features
    Returns:
        train_X (pd.DataFrame): training data
        train_y (pd.DataFrame): label for train_X
        test_X (pd.DataFrame): test data
        test_y (pd.DataFrame): label for test_X
    """
    # to store splited data
    trains_X, tests_X = [], []
    trains_y, tests_y = [], []

    # generate feature one by one
    for code in tqdm(codes):

        feats = features[features["SecuritiesCode"] == code].dropna()
        labels = get_label(price, code).dropna()

        if feats.shape[0] > 0 and labels.shape[0] > 0:
            # align label and feature indexes
            labels = labels.loc[labels.index.isin(feats.index)]
            feats = feats.loc[feats.index.isin(labels.index)]

            assert (labels.loc[:, "SecuritiesCode"] == feats.loc[:, "SecuritiesCode"]).all()
            labels = labels.loc[:, "label"]

            # split data into TRAIN and TEST
            _train_X = feats[: TRAIN_END]
            _test_X = feats[TEST_START:]

            _train_y = labels[: TRAIN_END]
            _test_y = labels[TEST_START:]
            
            assert len(_train_X) == len(_train_y)
            assert len(_test_X) == len(_test_y)

            # store features
            trains_X.append(_train_X)
            tests_X.append(_test_X)
            # store labels
            trains_y.append(_train_y)
            tests_y.append(_test_y)
            
    # combine features for each codes
    train_X = pd.concat(trains_X)
    test_X = pd.concat(tests_X)
    # combine label for each codes
    train_y = pd.concat(trains_y)
    test_y = pd.concat(tests_y)

    return train_X, train_y, test_X, test_y


# In[84]:


# generate feature/label
train_X, train_y, test_X, test_y = get_features_and_label(
    df_price, codes, feature
)


# # Building a simple model
# 
# Using the a selected subset of features and labels, build a model using the following procedure

# In[ ]:


feature.columns


# ## LGBM was reworked by choosing the best feature in the top 18 in feature importance and parameter tuning 

# ## Validation Score 0.55

# In[101]:


best_features = feature_imp.sort_values(by="Value", ascending=False)[0:18].values[:,1]


# depens o

# In[ ]:


# (boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0, min_child_weight=0.001, min_child_samples=20, subsample=1, subsample_freq=0, colsample_bytree=1, reg_alpha=0, reg_lambda=0, random_state=None, n_jobs=- 1, silent=True, importance_type='split',


# In[ ]:





# In[103]:


lgbm_params = { 
    'num_leaves': 10,
    "seed": 42,
    'max_depth': 10,
    'n_estimators': 20,
    
}
"""lgbm_params = { 
    "seed": 42,

    
}"""
feat_cols = best_features


# In[104]:


# initialize model
pred_model = LGBMRegressor(**lgbm_params)
# train
pred_model.fit(train_X[feat_cols].values, train_y)
# prepare result data
result = test_X[["SecuritiesCode"]].copy()
# predict
result.loc[:, "predict"] = pred_model.predict(test_X[feat_cols])
# actual result
result.loc[:, "Target"] = test_y.values

def set_rank(df):
    """
    Args:
        df (pd.DataFrame): including predict column
    Returns:
        df (pd.DataFrame): df with Rank
    """
    # sort records to set Rank
    df = df.sort_values("predict", ascending=False)
    # set Rank starting from 0
    df.loc[:, "Rank"] = np.arange(len(df["predict"]))
    return df

result = result.sort_values(["Date", "predict"], ascending=[True, False])
result = result.groupby("Date").apply(set_rank)


# In[ ]:


result.tail()


# # Evaluation
# 
# Input the output of the forecasts of the constructed model into the evaluation function and plot the daily returns.
# 
# The evaluation function for this competition is as follows.
# 
# Please read [here](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition) to know the evaluation function more.

# In[90]:


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# ## After getting best_features

# In[105]:


# calc spread return sharpe
calc_spread_return_sharpe(result, portfolio_size=200)


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(pred_model.feature_importances_,train_X[feat_cols].columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[106]:


def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): spread return
    """
    assert df['Rank'].min() == 0
    assert df['Rank'].max() == len(df['Rank']) - 1
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short

df_result = result.groupby('Date').apply(_calc_spread_return_per_day, 200, 2)


# In[107]:


df_result


# In[108]:


df_result.plot(figsize=(20, 8))


# We also show a cumulative spread return of the mode

# In[109]:


df_result.cumsum().plot(figsize=(20, 8))


# The model in this notebook is now complete! Try different features and training methods through trial and error!

# # Saving model

# You need to save your model parameter to use created model for your submission.

# ### Please don't forget to link if you found this notebook useful 🙏 🥰
