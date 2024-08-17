#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# A simple LightGBM model using `LabelEncoder` for categorical values. Further tuned by hand to find a good balance between fit and LB score. Note that the model is very likely overfit. This model may be most useful if combined with other ensemble methods. Note that no significant feature engineering was employed by this model.
# 
# ## Credits
# 
# * [LGBM Goes brrr!](https://www.kaggle.com/maunish/lgbm-goes-brrr) for initial tuning of LGBM parameters.

# In[1]:


import pandas as pd
import numpy as np

train = pd.read_csv("../input/tabular-playground-series-feb-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-feb-2021/test.csv")
train


# # Feature Definitions

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


# Categories must be converted to `int` types for LightGBM to use them as categorical. We'll use `LabelEncoder` to do this.

# In[3]:


from sklearn.preprocessing import LabelEncoder

new_cat_features = []

def label_encode(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column])
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    return new_feature

for feature in cat_features:
    new_cat_features.append(label_encode(train, test, feature))

train


# # Generate Model
# 
# Here we'll use `KFold` cross validation, but produce predictions out-of-fold for each fold.

# In[4]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

n_folds = 10

skf = KFold(n_splits=n_folds, random_state=2021, shuffle=True)

train_oof = np.zeros((300000,))
test_preds = 0

full_features = []
full_features.extend(new_cat_features)
full_features.extend(cont_features)

lgbm_params = {
    "random_state": 2021,
    "metric": "rmse",
    "n_jobs": 6,
    "early_stopping_round": 200,
    "cat_features": [x for x in range(len(new_cat_features))],
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

for fold, (train_index, test_index) in enumerate(skf.split(train, train["target"])):
    print("-------> Fold {} <--------".format(fold + 1))
    x_train, x_valid = pd.DataFrame(train.iloc[train_index]), pd.DataFrame(train.iloc[test_index])
    y_train, y_valid = train["target"].iloc[train_index], train["target"].iloc[test_index]
    
    x_train_features = pd.DataFrame(x_train[full_features])
    x_valid_features = pd.DataFrame(x_valid[full_features])

    model = LGBMRegressor(
        **lgbm_params
    )
    model.fit(
        x_train_features[full_features], 
        y_train,
        eval_set=[(x_valid_features[full_features], y_valid)],
        verbose=100,
    )
    oof_preds = model.predict(x_valid_features[full_features])
    test_preds += model.predict(test[full_features]) / n_folds
    train_oof[test_index] = oof_preds
    print("")
    
print("--> Overall results for out of fold predictions")
print(": RMSE = {}".format(mean_squared_error(train_oof, train["target"], squared=False)))


# # Generate Results
# 
# The test predictions were generated from each fold. Collect them here and build submission file.

# In[5]:


preds = test_preds.tolist()
test_ids = test["id"].tolist()

submission = pd.DataFrame({"id": test_ids, "target": preds})
submission.to_csv("submission.csv", index=False)


# If you find this kernel useful, please upvote!
