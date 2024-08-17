#!/usr/bin/env python
# coding: utf-8

# # Blend My 5 Notebooks
# 
# 
# - [TPS Oct 2021 - Basic XGBoost CV](https://www.kaggle.com/mmellinger66/tps-oct-2021-basic-xgboost-cv)
# - [TPS Oct 2021 - Basic CV LGBM](https://www.kaggle.com/mmellinger66/tps-oct-2021-basic-cv-lgbm)
# - [TPS Oct 21 - LGBM - SelectKBest](https://www.kaggle.com/mmellinger66/tps-oct-21-lgbm-selectkbest)
# - [TPS OCT 21/ XGBClassifier with optuna](https://www.kaggle.com/boneacrabonjac/tps-oct-21-xgbclassifier-with-optuna)
# - [TPS Oct 2021 - XGBoost/ CV / Feature Engineering](https://www.kaggle.com/mmellinger66/tps-oct-2021-xgboost-cv-feature-engineering)
# 
# # Versions
# 
# - V9: 5th Notebook: [TPS Oct 21 - LGBM - SelectKBest](https://www.kaggle.com/mmellinger66/tps-oct-21-lgbm-selectkbest)
# - V8: 4th Notebook: [TPS OCT 21/ XGBClassifier with optuna](https://www.kaggle.com/boneacrabonjac/tps-oct-21-xgbclassifier-with-optuna)
# - V7: ...
# - V6: [3rd Notebook](https://www.kaggle.com/mmellinger66/tps-oct-2021-xgboost-cv-feature-engineering) with different params
# - V5: column was called pred_1 instead of pred_3
# - V4: Trying a 3rd notebook
# - V3: Back to KFold
# - V2: Trying StratifiedKFold
# - V1: Original submission
# 
# # References
# 
# - [competition part-5: blending 101](https://www.kaggle.com/abhishek/competition-part-5-blending-101)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

from pathlib import Path

from sklearn.linear_model import LinearRegression


# # Configuration

# In[2]:


data_dir = Path('../input/tabular-playground-series-oct-2021') # Change me every month


# # Load Data

# In[3]:


train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(f"test data : Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")


# # Load Original Notebooks' Validation Results

# In[4]:


df1 = pd.read_csv("../input/tps-oct-2021-basic-xgboost-cv/train_pred_1.csv")
# df1 = df1.rename(columns={"target": "pred_1"})

df2 = pd.read_csv("../input/tps-oct-2021-basic-cv-lgbm/train_pred_2.csv")
# df2 = df2.rename(columns={"target": "pred_2"})

df3 = pd.read_csv("../input/tps-oct-2021-xgboost-cv-feature-engineering/train_pred_3.csv")

# df3 = df3.rename(columns={"pred_1": "pred_3"}) # Fix Typo
df4 = pd.read_csv("../input/tps-oct-21-xgbclassifier-with-optuna/train_pred_4.csv")

df5 = pd.read_csv("../input/tps-oct-21-lgbm-selectkbest/train_pred_5.csv")


# # Load Original Notebooks' Test Results

# In[5]:


df_test1 = pd.read_csv("../input/tps-oct-2021-basic-xgboost-cv/test_pred_1.csv")
df_test2 = pd.read_csv("../input/tps-oct-2021-basic-cv-lgbm/test_pred_2.csv")
df_test3 = pd.read_csv("../input/tps-oct-2021-xgboost-cv-feature-engineering/test_pred_3.csv")
df_test4 = pd.read_csv("../input/tps-oct-21-xgbclassifier-with-optuna/test_pred_4.csv")
df_test5 = pd.read_csv("../input/tps-oct-21-lgbm-selectkbest/test_pred_5.csv")

df_test1 = df_test1.rename(columns={"target": "pred_1"})
df_test2 = df_test2.rename(columns={"target": "pred_2"})
df_test3 = df_test3.rename(columns={"target": "pred_3"})
df_test4 = df_test4.rename(columns={"target": "pred_4"})
df_test5 = df_test5.rename(columns={"target": "pred_5"})


# In[6]:


df_test1.head()


# In[7]:


train_df = train_df.merge(df1, on="id", how="left")
train_df = train_df.merge(df2, on="id", how="left")
train_df = train_df.merge(df3, on="id", how="left")
train_df = train_df.merge(df4, on="id", how="left")
train_df = train_df.merge(df5, on="id", how="left")


# In[8]:


train_df.head()


# In[9]:


df_test1.head()


# In[10]:


test_df = test_df.merge(df_test1, on="id", how="left")
test_df = test_df.merge(df_test2, on="id", how="left")
test_df = test_df.merge(df_test3, on="id", how="left")
test_df = test_df.merge(df_test4, on="id", how="left")
test_df = test_df.merge(df_test5, on="id", how="left")


# In[11]:


test_df.head()


# # Only Using pred_1, ..., pred_n

# In[12]:


useful_features = ["pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]
test_df = test_df[useful_features]


# ## Seed and folds must match in all models
# 

# In[13]:


NFOLDS = 5
SEED = 42


# # Blend the Results with Linear Regression

# In[14]:


final_predictions = []
scores = []

kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
    xtrain =  train_df.iloc[train_idx].reset_index(drop=True)
    xvalid = train_df.iloc[valid_idx].reset_index(drop=True)
    
    xtest = test_df.copy()
    
    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    model = LinearRegression()
    model.fit(xtrain, ytrain)
    
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)

print(np.mean(scores), np.std(scores))


# # Submission File

# In[15]:


sample_submission.target = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission.to_csv("submission.csv", index=False)
sample_submission

