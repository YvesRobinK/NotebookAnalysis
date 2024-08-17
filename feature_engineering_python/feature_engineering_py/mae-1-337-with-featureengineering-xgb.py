#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Predicting the Age of Crabs: Enhancing Predictive Performance with Feature Engineering
# 
# In a previous notebook, we took a preliminary stab at predicting crab ages based on a set of basic biological and physical features. We used a dataset containing information on various aspects such as the crab's gender, body dimensions (length, diameter, height), weight (total, shucked, viscera, shell), and age. While we achieved a decent prediction accuracy with these basic features, there's potential for enhancement.
# 
# Previous Notebook - https://www.kaggle.com/code/pandeyg0811/mae-1-33-eda-ensemble

# ## Introduction
# 
# In this notebook, we're delving deeper into the world of crustacean life-cycles, specifically crabs. Crabs are fascinating creatures with diverse biological characteristics, playing crucial roles in marine ecosystems. A key attribute that's often challenging to estimate but crucial for biological and ecological studies is the age of a crab. Understanding the age distribution of a crab population can significantly contribute to population dynamics, growth rates, lifecycle understanding, and conservation strategies.

# ## Import Libraries

# In[2]:


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


# # Importing Dataset

# In[3]:


df_train = pd.read_csv("/kaggle/input/playground-series-s3e16/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e16/test.csv")
df_original = pd.read_csv("/kaggle/input/crab-age-prediction/CrabAgePrediction.csv")

df_train["Data Type"] = 0
df_test["Data Type"] = 1
df_original["Data Type"] = 2


# label encoding (feature "Sex" is categorical)
le = LabelEncoder()
df_train["Sex"] = le.fit_transform(df_train["Sex"])
df_test["Sex"] = le.transform(df_test["Sex"])
df_original["Sex"] = le.transform(df_original["Sex"])

# concatenate datasets
df_concat = pd.concat([df_train.drop('id',axis=1), df_original], ignore_index=True)
df_concat = df_concat.drop_duplicates()
df_all = pd.concat([df_concat, df_test.drop('id',axis=1)], ignore_index=True)
df_all


# ## Imputing Height variable

# In[4]:


# repalce some wrong data (Height=0) with random forest prediction
h1 = df_all[df_all["Height"] != 0]
h0 = df_all[df_all["Height"] == 0]
print(h1.shape, h0.shape)

x_h1 = h1.drop(columns=[ "Height", "Age", "Data Type"], axis=1)
y_h1 = h1["Height"]
x_h0 = h0.drop(columns=[ "Height", "Age", "Data Type"], axis=1)

rfr = RandomForestRegressor(n_jobs=-1, random_state=28)
rfr.fit(x_h1, y_h1)
preds_height = rfr.predict(x_h0)

cnt = 0
for i in range(len(df_all)):
    if df_all.loc[i, "Height"] == 0:
        df_all.loc[i, "Height"] = preds_height[cnt]
        cnt += 1

df_all["Height"].describe()


# ## Modelling

# In[5]:


lgb_params = {
    "objective": "regression_l1", # ="mae"
    "metric": "mae",
    "learning_rate": 0.03, # 0.1
    "n_estimators": 10000,
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


# ### Train and Test Data Split

# In[6]:


train = df_all[df_all['Data Type'] != 1]
train.reset_index(drop=True, inplace=True)

y_train = train["Age"].astype(int)
x_train = train.drop(columns=["Age", "Data Type"], axis=1)

x_test = df_all[df_all["Data Type"] == 1]
x_test.reset_index(drop=True, inplace=True)
x_test.drop(columns=["Age", "Data Type"], inplace=True)


# ### LightGBM Model

# In[7]:


def LightGBM(X, y, test_data, params):
    kf = list(KFold(n_splits=10, shuffle=True, random_state=100).split(X, y))
    preds, models = [], []
    oof = np.zeros(len(X))
    imp = pd.DataFrame()
    
    for nfold in np.arange(10):
        print("-"*30, "fold:", nfold, "-"*30)
        
        # set train/valid data
        idx_tr, idx_va = kf[nfold][0], kf[nfold][1]
        x_tr, y_tr = X.loc[idx_tr, :], y.loc[idx_tr]
        x_va, y_va = X.loc[idx_va, :], y.loc[idx_va]
        
        # training
        model = lgb.LGBMRegressor(**params)
        model.fit(x_tr, y_tr,
                eval_set=[(x_tr, y_tr), (x_va, y_va)],
                early_stopping_rounds=300,
                verbose=500,
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


# ## Model without Feature Engineering

# In[8]:


# Training
preds_lgb, models_lgb, oof_lgb, imp_lgb = LightGBM(x_train, y_train, x_test, lgb_params)

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


# # Reducing Error with with Feature Engineering
# 
# Now, we are ready to take our analysis a step further. In this notebook, we're going to incorporate feature engineering - a powerful tool that allows us to create new features from existing ones, thereby injecting our domain knowledge into the models, enhancing the richness of our data, and potentially boosting the predictive performance of our models.
# 
# Feature engineering is often what separates a good model from a great one. It can help us uncover complex patterns in the data that basic models might overlook. Given the intricacy of biological entities like crabs, there's a great deal of potential in engineering new features that better capture the nuanced aspects of a crab's biology and life history.
# 
# We'll be exploring a range of feature engineering techniques such as creating ratio features to capture relative size differences, geometric features to encapsulate physical properties, polynomial features to capture non-linear relationships, logarithmic transformations to manage extreme values, binning to simplify relationships with the target variable, and creating new weight-related features to provide a deeper look into weight distribution.
# 
# By the end of this notebook, we will have a much-improved model for predicting crab age, setting the stage for more accurate and informed biological and ecological studies. Let's dive in!

# 
# ### 1. Ratio-based features
# 
# - **Viscera Ratio**: The proportion of the crab's weight that comes from the viscera. This might be useful in understanding how the internal organ development correlates with the crab's age. 
#    * Formula: `Viscera Ratio = Viscera Weight / Weight`
# 
# ### 2. Geometric features
# 
# - **Surface Area**: Surface area of the crab computed as if it were a box. This feature could encapsulate the crab's overall size in a different way that the individual dimensions do not capture.
#    * Formula: `Surface Area = 2 * (Length * Diameter + Length * Height + Diameter * Height)`
# - **Volume**: Volume of the crab computed as if it were a box. Similar to the surface area, this provides a holistic measure of the crab's size.
#    * Formula: `Volume = Length * Diameter * Height`
# - **Density**: Density of the crab computed based on its weight and volume. It might help understand if older crabs tend to be denser or lighter for their size.
#    * Formula: `Density = Weight / Volume`
# 
# ### 3. Weight-related features
# 
# - **Shell-to-Body Ratio**: Ratio of the shell weight to the sum of total weight and shell weight. This can help understand if the shell development has a correlation with the crab's age.
#    * Formula: `Shell-to-Body Ratio = Shell Weight / (Weight + Shell Weight)`
# - **Meat Yield**: Ratio of the shucked weight to the sum of total weight and shell weight. It may capture if older crabs tend to have more or less meat relative to their total size.
#    * Formula: `Meat Yield = Shucked Weight / (Weight + Shell Weight)`
# - **Weight_wo_Viscera**: Weight of the crab without the viscera. It can help to examine how much of the crab's weight comes from non-viscera parts and if that changes with age.
#    * Formula: `Weight_wo_Viscera = Shucked Weight - Viscera Weight`
# 
# ### 4. Polynomial features
# 
# - **Length^2**: Squared length of the crab. It might capture any non-linear relationships between length and age.
#    * Formula: `Length^2 = Length ** 2`
# - **Diameter^2**: Squared diameter of the crab. Similar to length squared, this might capture any non-linear relationships between diameter and age.
#    * Formula: `Diameter^2 = Diameter ** 2`
# 
# ### 5. Logarithmic transformations
# 
# - **Log Weight**: Logarithm of the crab's weight. This might help deal with any extreme or skewed weight values and can sometimes help linearize relationships with the target variable.
#    * Formula: `Log Weight = log(Weight + 1)`
# 
# ### 6. Binning
# 
# - **Length Bins**: Discretization of the length into 4 bins. This simplifies the relationship of length with age and helps deal with any irregularities or noise in the relationship.
#    * Formula: `Length Bins = pd.cut(Length, bins=4, labels=False)`

# ## Feature Engineering

# In[9]:


df_all["Viscera Ratio"] = df_all["Viscera Weight"] / df_all["Weight"]
df_all["Shell Ratio"] = df_all["Shell Weight"] / df_all["Weight"]
df_all["Surface Area"] = 2 * (df_all["Length"] * df_all["Diameter"] + df_all["Length"] * df_all["Height"] + df_all["Diameter"] * df_all["Height"])
df_all["Volume"] = df_all["Length"] * df_all["Diameter"] * df_all["Height"]
df_all["Density"] = df_all["Weight"] / df_all["Volume"]
df_all['Shell-to-Body Ratio'] = df_all['Shell Weight'] / (df_all['Weight'] + df_all['Shell Weight'])
df_all['Meat Yield'] = df_all['Shucked Weight'] / (df_all['Weight'] + df_all['Shell Weight'])
df_all['Body Condition Index'] = np.sqrt(df_all['Length'] * df_all['Weight'] * df_all['Shucked Weight'])
df_all['Pseudo BMI']=df_all['Weight']/(df_all['Height']**2)
df_all['Len-to-Diam']=df_all['Length']/df_all['Diameter']
df_all['wieght-to-Viswieght']=df_all['Weight']/df_all['Viscera Weight']
df_all['wieght-to-Shellwieght']=df_all['Weight']/df_all['Shell Weight']
df_all['wieght-to-Shckwieght']=df_all['Weight']/df_all['Shucked Weight']
df_all["Weight_wo_Viscera"] = df_all['Shucked Weight'] - df_all['Viscera Weight']
df_all['Length^2'] = df_all['Length'] ** 2
df_all['Diameter^2'] = df_all['Diameter'] ** 2
df_all['Log Weight'] = np.log(df_all['Weight'] + 1) 
df_all['Length Bins'] = pd.cut(df_all['Length'], bins=4, labels=False)


# In[10]:


train = df_all[df_all['Data Type'] != 1]
train.reset_index(drop=True, inplace=True)

y_train = train["Age"].astype(int)
x_train = train.drop(columns=["Age", "Data Type"], axis=1)

x_test = df_all[df_all["Data Type"] == 1]
x_test.reset_index(drop=True, inplace=True)
x_test.drop(columns=["Age", "Data Type"], inplace=True)


# ## Model with Feature Engineering

# In[11]:


# Training
preds_lgb_fe, models_lgb_fe, oof_lgb_fe, imp_lgb_fe = LightGBM(x_train, y_train, x_test, lgb_params)

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


# ***
# 
# <BR>
#     
#     
# <div style="text-align: center;">
#    <span style="font-size: 4.5em; font-weight: bold; font-family: Arial;">THANK YOU!</span>
# </div>
# 
# <div style="text-align: center;">
#     <span style="font-size: 5em;">✔️</span>
# </div>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.4em; font-weight: bold; font-family: Arial; max-width:1200px; display: inline-block;">
#        These features helped me achieved reduced error rate and better leaderboard ranking. If you also found them helpful, please provide an upvote!
# 
#    </span>
# </div>
# 
# <br>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.2em; font-weight: bold;font-family: Arial;">@Gaurav Pandey</span>
# </div>

# In[ ]:




