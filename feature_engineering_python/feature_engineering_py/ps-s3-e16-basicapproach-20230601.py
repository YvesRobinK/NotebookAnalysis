#!/usr/bin/env python
# coding: utf-8

# # Dear Kagglers!

# **in this notebook, we introduce basic approach to Playground Series competition (with regression task). so, we proceed with the following steps:**
# - Data handling and EDA
# - Feature engineering
# - Training with validation
# - Prediction and evaluation
# - Submission

# **and what we DO NOT do in this notebook are the following:**
# - Model ensembling
# - Hyperparameter tuning
# 
# these are very important but it is better to focus on more basic steps (e.g., EDA, Feature engineering, Validation, and more) early in the competition. so, these are the next steps in the tasks introduced in this notebook.

# # Data handling and EDA

# **in this step, we proceed tasks as the following:**
# - import libraries
# - read datasets
# - check data characteristics (with visualization)

# In[1]:


# import libraries
# add other libraries if needed
import gc
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
import optuna

import warnings
warnings.simplefilter("ignore")

print("imported.")


# In[2]:


# read datasets
# we use original-dataset and synthetic-dataset (self-generated) in addition to train/test datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s3e16/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e16/test.csv")
df_original = pd.read_csv("/kaggle/input/crab-age-prediction/CrabAgePrediction.csv")
df_sync = pd.read_csv("/kaggle/input/ps-s3-e16-synthetic-train-data/train_synthetic.csv")

num_cols = ["Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Age"]
cat_cols = ["Sex"]
print(df_train.shape, df_test.shape, df_original.shape, df_sync.shape)


# In[3]:


# check data characteristics
# histgram of train-data
df_train_hist = df_train[num_cols]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
sturges = int(math.floor(math.log2(len(df_train)) + 1)) # 17
cnt = 0
for row in range(2):
    for col in range(4):
        axs[row, col].hist(df_train_hist.iloc[:, cnt], bins=sturges*2)
        axs[row, col].set_title(df_train_hist.columns[cnt])
        cnt += 1

plt.show()


# In[4]:


# histgram of test-data
cols = num_cols.copy()
cols.pop()
df_test_hist = df_test[cols]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
cnt = 0
for row in range(2):
    for col in range(4):
        axs[row, col].hist(df_test_hist.iloc[:, cnt], bins=sturges*2)
        axs[row, col].set_title(df_test_hist.columns[cnt])
        cnt += 1
        if cnt > 6:
            break

plt.show()


# In[5]:


# histgram of original-data
df_original_hist = df_original[num_cols]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
cnt = 0
for row in range(2):
    for col in range(4):
        axs[row, col].hist(df_original_hist.iloc[:, cnt], bins=sturges*2)
        axs[row, col].set_title(df_original_hist.columns[cnt])
        cnt += 1

plt.show()


# In[6]:


# histgram of synthetic-data
df_sync_hist = df_sync[num_cols]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
cnt = 0
for row in range(2):
    for col in range(4):
        axs[row, col].hist(df_sync_hist.iloc[:, cnt], bins=sturges*2)
        axs[row, col].set_title(df_sync_hist.columns[cnt])
        cnt += 1

plt.show()


# In[7]:


# histgram of target (train/original/synthetic)
df_train_target = df_train["Age"]
df_original_target = df_train["Age"]
df_sync_target = df_train["Age"]

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(df_train_target, bins=(int(math.ceil(sturges*1.5))))
axs[0].set_title("Age - train")
axs[1].hist(df_original_target, bins=(int(math.ceil(sturges*1.5))))
axs[1].set_title("Age - original")
axs[2].hist(df_sync_target, bins=(int(math.ceil(sturges*1.5))))
axs[2].set_title("Age - synthetic")

plt.show()


# **it seems that four datasets are very similar each other, so we check if these datasets are the same.**

# In[8]:


# label encoding (feature "Sex" is categorical)
le = LabelEncoder()
df_train["Sex"] = le.fit_transform(df_train["Sex"])
df_test["Sex"] = le.transform(df_test["Sex"])
df_original["Sex"] = le.transform(df_original["Sex"])
df_sync["Sex"] = le.transform(df_sync["Sex"])
print("prepared.")


# In[9]:


# <Adversarial Validation>
# check similarity between two datasets - train/test/original/synthetic datasets
# use RandomForestClassifier and calculate ROC-AUS score
# if ROC-AUC score is around 0.5, it is difficult to distinguish their two datasets (i.e., their two datasets may be similar).

# prepare datasets for prediction
train = df_train.drop(columns=["id", "Age"], axis=1)
test = df_test.drop(columns=["id"], axis=1)
original = df_original.drop(columns=["Age"], axis=1)
sync = df_sync.drop(columns=["id", "Age"], axis=1)

X1 = train.append(test)
y1 = [0] * len(train) + [1] * len(test)
X2 = train.append(original)
y2 = [0] * len(train) + [1] * len(original)
X3 = train.append(sync)
y3 = [0] * len(train) + [1] * len(sync)
X4 = test.append(original)
y4 = [0] * len(test) + [1] * len(original)
X5 = test.append(sync)
y5 = [0] * len(test) + [1] * len(sync)
X6 = sync.append(original)
y6 = [0] * len(sync) + [1] * len(original)
print("prepared.")

# prediction by Random Forest
model = RandomForestClassifier()

cv_preds1 = cross_val_predict(model, X1, y1, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(train-test): {:.3f}".format(roc_auc_score(y_true=y1, y_score=cv_preds1[:, 1])))

cv_preds2 = cross_val_predict(model, X2, y2, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(train-original): {:.3f}".format(roc_auc_score(y_true=y2, y_score=cv_preds2[:, 1])))

cv_preds3 = cross_val_predict(model, X3, y3, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(train-sync): {:.3f}".format(roc_auc_score(y_true=y3, y_score=cv_preds3[:, 1])))

cv_preds4 = cross_val_predict(model, X4, y4, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(test-original): {:.3f}".format(roc_auc_score(y_true=y4, y_score=cv_preds4[:, 1])))

cv_preds5 = cross_val_predict(model, X5, y5, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(test-sync): {:.3f}".format(roc_auc_score(y_true=y5, y_score=cv_preds5[:, 1])))

cv_preds6 = cross_val_predict(model, X6, y6, cv=5, n_jobs=-1, method="predict_proba")
print("ROC-AUC score(sync-original): {:.3f}".format(roc_auc_score(y_true=y6, y_score=cv_preds6[:, 1])))

del train, test, original, sync
del X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6, model
del cv_preds1, cv_preds2, cv_preds3, cv_preds4, cv_preds5, cv_preds6
gc.collect()


# **it seems that train/test/synthetic datasets are similar each other (scores are around 0.5), but original dataset and train/test datasets are slightly different. so it is better to be careful to use original dataset - in this notebook, we use original dataset for model fitting as trial.**

# In[10]:


# check data characteristics
# correlation between features in train-data
df_train_cor = df_train.drop(columns=["id"], axis=1)
plt.figure(figsize=(12, 6))
colormap = plt.cm.RdBu
sns.heatmap(df_train_cor.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)


# In[11]:


# correlation between features in test-data
df_test_cor = df_test.drop(columns=["id"], axis=1)
plt.figure(figsize=(12, 6))
colormap = plt.cm.RdBu
sns.heatmap(df_test_cor.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)


# In[12]:


# correlation between features in original-data
df_original_cor = df_original.copy()
plt.figure(figsize=(12, 6))
colormap = plt.cm.RdBu
sns.heatmap(df_original_cor.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)


# In[13]:


# correlation between features in synthetic-data
df_sync_cor = df_sync.drop(columns=["id"], axis=1)
plt.figure(figsize=(12, 6))
colormap = plt.cm.RdBu
sns.heatmap(df_sync_cor.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)


# **According to heat-maps, there are very strong positive correlations (over 0.9) between features except for "Sex". it would be better to use more other features and we try to generate them.**

# # Feature engineering

# **Feature engineering is very important, essential, and deep. so we will try basic-level feature engineering what we can in this notebook.**
# 
# As an assumption, we temporally use the GBDT model (especially LightGBM) in this notebook, so Data Scaling/Standardization and filling null-values are basically not necessary.

# In[14]:


# preparation
df_train["Data Type"] = 0
df_test["Data Type"] = 1
df_original["Data Type"] = 2
df_sync["Data Type"] = 3

ids = []
for i in range(len(df_original)):
    ids.append(i + 123419)

df_original["id"] = ids
df_sync["id"] += 127312

# concatenate datasets
df_concat = pd.concat([df_train, df_original, df_sync], ignore_index=True)
df_concat = df_concat.drop_duplicates()
df_all = pd.concat([df_concat, df_test], ignore_index=True)
df_all


# In[15]:


df_all.describe()


# **min of "Height" is zero, so maybe we need some handling. in this notebook, we try to predict value of height by other features using Random Forest Regressor, and replace zero to these predicted values.**

# In[16]:


# prediction of Height by Random Forest Regressor
h1 = df_all[df_all["Height"] != 0]
h0 = df_all[df_all["Height"] == 0]
print(h1.shape, h0.shape)

x_h1 = h1.drop(columns=["id", "Height", "Age", "Data Type"], axis=1)
y_h1 = h1["Height"]
x_h0 = h0.drop(columns=["id", "Height", "Age", "Data Type"], axis=1)

rfr = RandomForestRegressor(n_jobs=-1, random_state=28)
rfr.fit(x_h1, y_h1)
preds_height = rfr.predict(x_h0)

cnt = 0
for i in range(len(df_all)):
    if df_all.loc[i, "Height"] == 0:
        df_all.loc[i, "Height"] = preds_height[cnt]
        cnt += 1

df_all["Height"].describe()


# In[17]:


# generate additional features
df_all["Viscera Ratio"] = df_all["Viscera Weight"] / df_all["Weight"]
df_all["Shell Ratio"] = df_all["Shell Weight"] / df_all["Weight"]

# Surface/Volume/Density of Crabs
df_all["Surface Area"] = 2 * (df_all["Length"] * df_all["Diameter"] + df_all["Length"] * df_all["Height"] + df_all["Diameter"] * df_all["Height"])
df_all["Volume"] = df_all["Length"] * df_all["Diameter"] * df_all["Height"]
df_all["Density"] = df_all["Weight"] / df_all["Volume"]

df_all


# **please let us know if you could generate other effective features, thanks!**

# # Training with validation & Prediction

# **we try to do the following tasks:**
# - set datasets for training and prediction
# - set parameters for model
# - define function of training and prediction
# - train and predict
# - evaluate the results

# In[18]:


# x/y datasets for train
train = df_all[df_all["Data Type"] != 1]
train.sort_values("id", inplace=True)
train.reset_index(drop=True, inplace=True)

y_train = train["Age"].astype(int)
x_train = train.drop(columns=["id", "Age", "Data Type"], axis=1)
x_train


# In[19]:


# dataset for test
x_test = df_all[df_all["Data Type"] == 1]
x_test.sort_values("id", inplace=True)
x_test.reset_index(drop=True, inplace=True)
x_test.drop(columns=["id", "Age", "Data Type"], inplace=True)
x_test


# In[20]:


# set basic-parameters
SPLITS = 10
ESTIMATORS = 10000
ES_ROUNDS = 300
VERBS = 500
R_STATE = 28

print("set basic-parameters.")


# In[21]:


# set LightGBM-parameters
# single-commented-out features will be tuned later
lgb_params = {
    "objective": "regression_l1", # ="mae"
    "metric": "mae",
    "learning_rate": 0.03, # 0.1
    "n_estimators": ESTIMATORS,
    "max_depth": 8, # -1, 1-16(3-8)
    "num_leaves": 255, # 31, 2-2**max_depth
    "feature_fraction": 0.4, # 1.0, 0.1-1.0, 0.4
    "min_data_in_leaf": 256, # 20, 0-300
    "subsample": 0.4, # 1.0, 0.01-1.0
    "reg_alpha": 0.1, # 0.0, 0.0-10.0, 0.1
    "reg_lambda": 0.1, # 0.0, 0.0-10.0, 0.1
    ###"subsample_freq": 0, # 0-10
    ###"max_bin": 255, # 32-512
    ###"min_gain_to_split": 0.0, # 0.0-15.0
    ###"subsample_for_bin": 200000, # 30-len(x_train)
    ###"boosting": "dart", # "gbdt"
    ###"device_type": "gpu", # "cpu"
}

print("set lgb-parameters.")


# In[22]:


# define function of training by LightGBM
def train_lgb(X, y, test_data, params, es_rounds, verb, r_state):
    kf = list(KFold(n_splits=SPLITS, shuffle=True, random_state=r_state).split(X, y))
    preds, models = [], []
    oof = np.zeros(len(X))
    imp = pd.DataFrame()
    
    for nfold in np.arange(SPLITS):
        print("-"*30, "fold:", nfold, "-"*30)
        
        # set train/valid data
        idx_tr, idx_va = kf[nfold][0], kf[nfold][1]
        x_tr, y_tr = X.loc[idx_tr, :], y.loc[idx_tr]
        x_va, y_va = X.loc[idx_va, :], y.loc[idx_va]
        
        # training
        model = lgb.LGBMRegressor(**params)
        model.fit(x_tr, y_tr,
                eval_set=[(x_tr, y_tr), (x_va, y_va)],
                early_stopping_rounds=es_rounds,
                verbose=verb,
        )
        models.append(model)
        
        # validation
        pred_va = model.predict(x_va)
        oof[idx_va] = pred_va
        print("MAE(valid)", nfold, ":", "{:.4f}".format(mean_absolute_error(y_va, pred_va)))
        
        # prediction
        pred_test = model.predict(test_data)
        preds.append(pred_test)
        
        # importance
        _imp = pd.DataFrame({"features": X.columns, "importance": model.feature_importances_, "nfold": nfold})
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)
    
    imp = imp.groupby("features")["importance"].agg(["mean", "std"])
    imp.columns = ["importance", "importance_std"]
    imp["importance_cov"] = imp["importance_std"] / imp["importance"]
    imp = imp.reset_index(drop=False)
    display(imp.sort_values("importance", ascending=False, ignore_index=True))
    
    return preds, models, oof, imp

print("defined.")


# In[23]:


# training
preds_lgb, models_lgb, oof_lgb, imp_lgb = train_lgb(x_train, y_train, x_test, lgb_params, ES_ROUNDS, VERBS, R_STATE)

# MAE for LightGBM
oof_lgb_round = np.zeros(len(oof_lgb), dtype=int)
for i in range(len(oof_lgb)):
    oof_lgb_round[i] = int((oof_lgb[i] * 2 + 1) // 2)

print("MAE(int):", "{:.4f}".format(mean_absolute_error(y_train, oof_lgb_round)))
print("MAE(float):", "{:.4f}".format(mean_absolute_error(y_train, oof_lgb)))

# visualization of predictions by test-data
mean_preds_lgb = np.mean(preds_lgb, axis=0)
mean_preds_lgb_round = np.zeros(len(mean_preds_lgb), dtype=int)
for i in range(len(mean_preds_lgb_round)):
    mean_preds_lgb_round[i] = int((mean_preds_lgb[i] * 2 + 1) // 2)

sns.countplot(x=mean_preds_lgb_round)


# In[24]:


# predictions (float ver.)
sns.histplot(x=mean_preds_lgb)


# **[Additional] sometimes Seed-Averaging will be effective. we post the code below, but we do not run it in this notebook for simplicity.**

# In[25]:


# Additional

# training with seed-averaging
#seeds = 10
#train_state, o_state = [], []
#for state in range(seeds):
#    print("#"*40, "State:", state, "#"*40)
#    p, m, o, i = train_lgb_seed_avg(x_train, y_train, x_test, lgb_params, ES_ROUNDS, 0, state)
#    train_state.append(np.mean(p, axis=0))
#    
#    o_round = np.zeros(len(o), dtype=int)
#    for i in range(len(o)):
#        o_round[i] = int((o[i] * 2 + 1) // 2)
#    
#    o_state.append(o_round)
#    print("")
#    print("MAE:", "{:.4f}".format(mean_absolute_error(y_train, o_round)))
#    print("")

# MAE for LightGBM
#mean_oof = np.mean(o_state, axis=0)
#print("MAE:", "{:.4f}".format(mean_absolute_error(y_train, mean_oof)))

# visualization of predictions by test-data
#mean_preds = np.mean(train_state, axis=0)
#preds = np.zeros(len(mean_preds), dtype=int)
#for i in range(len(preds)):
#    preds[i] = int((mean_preds[i] * 2 + 1) // 2)

#sns.countplot(x=preds)


# In[26]:


# Additional

# dataset for submission
#sample_sub = pd.read_csv("/kaggle/input/playground-series-s3e16/sample_submission.csv")
#df_submit = pd.DataFrame({"id": sample_sub["id"], "Age": preds})
#display(df_submit.info())
#display(df_submit.describe())
#df_submit


# # Submission

# In[27]:


# dataset for submission
sample_sub = pd.read_csv("/kaggle/input/playground-series-s3e16/sample_submission.csv")
df_submit = pd.DataFrame({"id": sample_sub["id"], "Age": mean_preds_lgb_round})
display(df_submit.info())
display(df_submit.describe())
df_submit


# In[28]:


# submission
df_submit.to_csv("submission.csv", index=None)
print("completed.")


# # Good luck!
# **Best regards, e271828**
