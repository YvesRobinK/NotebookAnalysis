#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange

from sklearn.model_selection import KFold
import lightgbm as lgb


# In[2]:


ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "godaddy-microbusiness-density-forecasting"


# In[3]:


train = pd.read_csv(DATA / "train.csv")
train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
test = pd.read_csv(DATA / "test.csv")
test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
# census = pd.read_csv(DATA / "census_starter.csv")


# ## Quick EDA

# In[4]:


train.info()


# In[5]:


train.head().T


# In[6]:


test.info()


# In[7]:


test.head().T


# ### cfips

# In[8]:


train.cfips.nunique()


# In[9]:


test.cfips.nunique()


# In[10]:


test.cfips.isin(train.cfips).all()


# ### Period

# In[11]:


_ = train.first_day_of_month.value_counts().plot(kind="bar")


# In[12]:


_ = test.first_day_of_month.value_counts().plot(kind="bar")


# ### target

# In[13]:


train.microbusiness_density.describe()


# In[14]:


_ = train.microbusiness_density.hist(bins=1000)


# ### mean target time series

# In[15]:


mean_target = train.groupby("first_day_of_month")["microbusiness_density"].mean().reset_index()


# In[16]:


mean_target.head()


# In[17]:


_ = mean_target.plot(kind="line", x="first_day_of_month", y="microbusiness_density")


# ## Training & Inference

# ### feature engineering

# In[18]:


def make_feature(df):
    feat = pd.DataFrame()
    feat["contry_code"] = df["cfips"] // 100
    feat["state_code"] = df["cfips"] % 100
    feat["year"] = df["first_day_of_month"].dt.year
    feat["month"] = df["first_day_of_month"].dt.month
    
    return feat


# In[19]:


train_feat = make_feature(train)
test_feat = make_feature(test)


# In[20]:


train_feat.head()


# In[21]:


test_feat.head()


# ### split CV folds

# In[22]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_val_splits = list(kf.split(train))


# ### Training

# In[23]:


def smape(y_true, y_pred):
    nume = np.abs(y_true - y_pred)
    deno = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    return 100 * np.mean(nume / deno)


# In[24]:


MODEL_PARAM = {
    "boosting_type": "gbdt",
    "objective": "mse",

    "learning_rate": 0.3,
    "max_depth": -1,
    "colsample_bytree": .85,
    "subsample": .85,

    "random_state": 42,
    "n_jobs": -1,
}
FIT_PARAM = {
    "num_boost_round": 50000,
    "early_stopping_rounds": 200,
    "verbose_eval": 100,
}


# In[25]:


y = train.microbusiness_density.values
y_log1p = np.log1p(y)


# In[26]:


oof_pred = np.zeros((len(train),))
test_preds = np.zeros((5, len(test)))

for fold_id, (trn_idx, val_idx) in enumerate(train_val_splits):
    
    X_trn, X_val = train_feat.iloc[trn_idx], train_feat.iloc[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]
    y_log1p_trn, y_log1p_val = y_log1p[trn_idx], y_log1p[val_idx]
    
    trn_data = lgb.Dataset(X_trn, y_log1p_trn)
    val_data = lgb.Dataset(X_val, y_log1p_val)
    
    model = lgb.train(
        train_set=trn_data, valid_sets=[trn_data, val_data],
        params=MODEL_PARAM, **FIT_PARAM)
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(test_feat)
    
    oof_pred[val_idx] = np.expm1(val_pred)
    test_preds[fold_id] = np.expm1(test_pred)
    
    print(f"[fold {fold_id}] smape: {smape(y_val, oof_pred[val_idx])}")


# In[27]:


print(f"[oof] smape: {smape(y, oof_pred)}")


# ## make submit

# In[28]:


sub = pd.read_csv(DATA / "sample_submission.csv")


# In[29]:


sub["microbusiness_density"] = test_preds.mean(axis=0)

sub.to_csv("submission.csv", index=False)


# In[30]:


sub.head()


# ## EOF
