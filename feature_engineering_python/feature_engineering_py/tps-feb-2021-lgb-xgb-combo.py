#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel combines XGBoost and LightGBM models together. It uses hand tuned XGBoost and LightGBM models to do so. It searches for the optimal combination of models to produce a blended combination of both. No extensive feature engineering is performed. In this instance we'll use 3 different models:
# 
# * LightGBM with `LabelEncode`.
# * XGBoost with `LeaveOneOut` encoding.
# * XGBoost with `LeaveOneOut` encoding and `MEstimateEncoder`.
# 
# ## Credits
# 
# * [LGBM Goes brrr!](https://www.kaggle.com/maunish/lgbm-goes-brrr) for tuned LGBM parameters.

# In[1]:


import pandas as pd
import numpy as np

train = pd.read_csv("../input/tabular-playground-series-feb-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-feb-2021/test.csv")
train


# # Feature Engineering

# In[2]:


cat_features = [
    "cat0", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", 
    "cat8", "cat9"
]

cont_features = [
    "cont0", "cont1", "cont2", "cont3", "cont4",
    "cont5", "cont6", "cont7", "cont8", "cont9", "cont10", 
    "cont11", "cont12", "cont13"
]


# In[3]:


get_ipython().system('pip install category-encoders')


# In[4]:


from sklearn.preprocessing import LabelEncoder
from category_encoders import LeaveOneOutEncoder
from category_encoders import MEstimateEncoder

lgb_cat_features = []
xgb1_cat_features = []
xgb2_cat_features = []

def label_encode(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column])
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    return new_feature

def loo_encode(train_df, test_df, column):
    loo = LeaveOneOutEncoder()
    new_feature = "{}_loo".format(column)
    loo.fit(train_df[column], train_df["target"])
    train_df[new_feature] = loo.transform(train_df[column])
    test_df[new_feature] = loo.transform(test_df[column])
    return new_feature

def mee_encode(train_df, test_df, column):
    mee = MEstimateEncoder()
    new_feature = "{}_mee".format(column)
    mee.fit(train_df[column], train_df["target"])
    train_df[new_feature] = mee.transform(train_df[column])
    test_df[new_feature] = mee.transform(test_df[column])
    return new_feature

for feature in cat_features:
    lgb_cat_features.append(label_encode(train, test, feature))

for feature in cat_features:
    xgb1_cat_features.append(loo_encode(train, test, feature))
    
xgb2_cat_features.extend(xgb1_cat_features)
for feature in cat_features:
    xgb2_cat_features.append(mee_encode(train, test, feature))

train


# # Build Models
# 
# Here we will build 3 models - one LightGBM model and two XGBoost models. We'll save the results of each model separately so we can try different combinations of both later. We'll use out-of-fold predictions for the test set for better results.

# In[5]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

n_folds = 10

skf = KFold(n_splits=n_folds, random_state=2021, shuffle=True)

lgb_train_oof = np.zeros((300000,))
xgb1_train_oof = np.zeros((300000,))
xgb2_train_oof = np.zeros((300000,))

lgb_test_preds = 0
xgb1_test_preds = 0
xgb2_test_preds = 0

full_features = []
full_features.extend(lgb_cat_features)
full_features.extend(xgb2_cat_features)
full_features.extend(cont_features)

lgb_features = []
lgb_features.extend(lgb_cat_features)
lgb_features.extend(cont_features)

xgb1_features = []
xgb1_features.extend(xgb1_cat_features)
xgb1_features.extend(cont_features)

xgb2_features = []
xgb2_features.extend(xgb2_cat_features)
xgb2_features.extend(cont_features)

lgb_params = {
    "random_state": 2021,
    "metric": "rmse",
    "n_jobs": -1,
    "early_stopping_round": 200,
    "cat_features": [x for x in range(len(lgb_cat_features))],
    "reg_alpha": 9.03513073170552,
    "reg_lambda": 0.024555737897445917,
    "colsample_bytree": 0.2185112060137363,
    "learning_rate": 0.003049106861273527,
    "max_depth": 65,
    "num_leaves": 51,
    "min_child_samples": 177,
    "n_estimators": 1600000,
    "cat_smooth": 93.60968300634175,
    "max_bin": 537,
    "min_data_per_group": 117,
    "bagging_freq": 1,
    "bagging_fraction": 0.6709049555262285,
    "cat_l2": 7.5586732660804445,
}

xgb1_params = {
    "seed": 2021,
    "n_estimators": 10000,
    "verbosity": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist", 
    "gpu_id": 0,
    "colsample_bytree": 0.23116058185789234, 
    "gamma": 2.0737266506535375, 
    "lambda": 8.76288374058159, 
    "learning_rate": 0.01126802018395814, 
    "max_depth": 11, 
    "min_child_weight": 1.4477515824904934, 
    "subsample": 0.4898608703522127,
    "alpha": 9.206528646529561,
    "max_bin": 658,
}

xgb2_params = {
    "seed": 2021,
    "n_estimators": 10000,
    "verbosity": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist", 
    "gpu_id": 0,
    "colsample_bytree": 0.21428906671248488, 
    "gamma": 2.116597615713485, 
    "lambda": 6.683336764226128, 
    "learning_rate": 0.008098678089211109, 
    "max_depth": 11, 
    "min_child_weight": 7.587302445130359, 
    "subsample": 0.6309499238527587,
    "alpha": 8.078271293205107,
    "max_bin": 570,
}

for fold, (train_index, test_index) in enumerate(skf.split(train, train["target"])):
    print("-------> Fold {} <--------".format(fold + 1))
    x_train, x_valid = pd.DataFrame(train.iloc[train_index]), pd.DataFrame(train.iloc[test_index])
    y_train, y_valid = train["target"].iloc[train_index], train["target"].iloc[test_index]
    
    x_train_features = pd.DataFrame(x_train[full_features])
    x_valid_features = pd.DataFrame(x_valid[full_features])

    print(": Build LightGBM model")
    lgb_model = LGBMRegressor(
        **lgb_params
    )
    lgb_model.fit(
        x_train_features[lgb_features], 
        y_train,
        eval_set=[(x_valid_features[lgb_features], y_valid)],
        verbose=500,
    )
    
    print("")
    print(": Build XGBoost model 1")
    xgb1_model = XGBRegressor(
        **xgb1_params
    )
    xgb1_model.fit(
        x_train_features[xgb1_features], 
        y_train,
        eval_set=[(x_valid_features[xgb1_features], y_valid)],
        verbose=500,
        early_stopping_rounds=200,
    )

    print("")
    print(": Build XGBoost model 2")
    xgb2_model = XGBRegressor(
        **xgb2_params
    )
    xgb2_model.fit(
        x_train_features[xgb2_features], 
        y_train,
        eval_set=[(x_valid_features[xgb2_features], y_valid)],
        verbose=500,
        early_stopping_rounds=200,
    )
    
    lgb_oof_preds = lgb_model.predict(x_valid_features[lgb_features])
    lgb_train_oof[test_index] = lgb_oof_preds

    xgb1_oof_preds = xgb1_model.predict(x_valid_features[xgb1_features])
    xgb1_train_oof[test_index] = xgb1_oof_preds

    xgb2_oof_preds = xgb2_model.predict(x_valid_features[xgb2_features])
    xgb2_train_oof[test_index] = xgb2_oof_preds

    lgb_test_preds += lgb_model.predict(test[lgb_features]) / n_folds
    xgb1_test_preds += xgb1_model.predict(test[xgb1_features]) / n_folds
    xgb2_test_preds += xgb2_model.predict(test[xgb2_features]) / n_folds
    print("")
    
print("--> Overall results for out of fold predictions")
print(": LGB RMSE = {}".format(mean_squared_error(lgb_train_oof, train["target"], squared=False)))
print(": XGB 1 RMSE = {}".format(mean_squared_error(xgb1_train_oof, train["target"], squared=False)))
print(": XGB 2 RMSE = {}".format(mean_squared_error(xgb2_train_oof, train["target"], squared=False)))


# # Find Best Combo
# 
# Now we'll search for the best split between the LightGBM model and the XGBoost model. We'll plot the results and make sure there aren't any additional useful combinations to check out besides the best looking one.

# In[6]:


best_split_rmse = 9999999.9
best_split_combo = -1

split_results = []
split_percentages = []
split_strings = []
index = 0

print("--> Calculating best combination")
splits = []
for x in range(100, 0, -5):
    if 100 - x == 0:
        splits.append([x / 100., 0., 0.])
    for y in range(100 - x, 0, -5):
        if 100 - x - y == 0 and x + y == 100:
            splits.append([x / 100., y / 100., 0.])
            splits.append([x / 100., 0., y / 100.])
        for z in range(100 - x - y, 0, -5):
            if x + y + z == 100:
                splits.append([x /100., y / 100., z / 100.])
            
for index, x in enumerate(splits):
    combo_preds = (x[0] * lgb_train_oof) + (x[1] * xgb1_train_oof) + (x[2] * xgb2_train_oof)
    rmse = mean_squared_error(combo_preds, train["target"], squared=False)
    split_results.append(rmse)
    split_strings.append("LGB {:0.3} + XGB1 {:0.3} + XGB2 {:0.3}".format(x[0], x[1], x[2]))
    split_percentages.append(x)
    if rmse < best_split_rmse:
        best_split_rmse = rmse
        best_split_combo = index
    
print(": Best combo is {}".format(split_strings[best_split_combo]))
print(": Best RMSE is {}".format(best_split_rmse))


# In[7]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

labels = [x if index % 5 == 0 else " " for index, x in enumerate(split_strings)]
plt.figure(figsize=(20, 10))
plt.plot(split_strings, split_results)
plt.xticks(labels, rotation=90)
plt.xlabel('LGB / XGB1 / XGB2 Split Amounts')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.show()


# # Generate Submission
# 
# Generate the submission file using the best combination of values.

# In[8]:


preds = (lgb_test_preds * split_percentages[best_split_combo][0])
preds += (xgb1_test_preds * split_percentages[best_split_combo][1])
preds += (xgb2_test_preds * split_percentages[best_split_combo][2])
preds = preds.tolist()
test_ids = test["id"].tolist()

submission = pd.DataFrame({"id": test_ids, "target": preds})
submission.to_csv("submission.csv", index=False)

