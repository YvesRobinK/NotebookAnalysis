#!/usr/bin/env python
# coding: utf-8

# # Please upvote if you find this helpful!! :)

# # Setup / Load Train Data

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


# In[2]:


train_df = pd.read_csv("/kaggle/input/playground-series-s3e17/train.csv")
train_df.head()


# # EDA

# In[3]:


train_df.describe()


# In[4]:


train_df.columns


# In[5]:


train_df.info()


# Below: `Type` column has a small number of classes in comparison to the total number of rows, so we can encode it.
# 
# `Product ID` has a letter and a number. The letter is the same as the letter in the `Type` column, and it may be beneficial to split `ProductID` into its letter and number.

# In[6]:


train_df.shape


# In[7]:


print(len(pd.unique(train_df["Product ID"])))
print(len(pd.unique(train_df["Type"])))


# Let's see the correlation of the different variables.

# In[8]:


import pandas as pd
import numpy as np

corr = train_df.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')


# # Encoding Categorical Columns

# In[9]:


train_df["Type"].unique()


# In[10]:


from sklearn.preprocessing import OrdinalEncoder

categorical_cols = ["Type"]

# Low, medium, high, in order (0, 1, 2)
encoder = OrdinalEncoder(categories=[['L','M','H']])
train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols])
train_df.describe()


# # Fix Column Names

# XGBoost doesn't work with column names that have "\[" or "\]" in them.

# In[11]:


train_df.columns = train_df.columns.str.replace('[\[\]]', '', regex=True)
train_df


# # Feature Engineering

# One feature that came to my mind was **Power**, defined as torque times the rotational speed. We can create this feature as shown below.

# In[12]:


train_df["Power"] = train_df["Torque Nm"] * train_df["Rotational speed rpm"]
train_df["Power"].head()


# Another feature could be the ratio of the process temperature to the air temperature.

# In[13]:


train_df["temp_ratio"] = train_df["Process temperature K"] / train_df["Air temperature K"]
train_df["temp_ratio"].head()


# In[14]:


train_df["Process temperature C"] = train_df["Process temperature K"] - 273.15
train_df["Process temperature C"].head()


# In[15]:


train_df["Air temperature C"] = train_df["Air temperature K"] - 273.15
train_df["Air temperature C"].head()


# In[16]:


train_df["temp_C_ratio"] = train_df["Process temperature C"] / train_df["Air temperature C"]
train_df["temp_C_ratio"].head()


# In[17]:


train_df["Failure Sum"] = (train_df["TWF"] +
                            train_df["HDF"] +
                            train_df["PWF"] +
                            train_df["OSF"] +
                            train_df["RNF"])
                
train_df["Failure Sum"].head()


# In[18]:


train_df["tool_wear_speed"] = train_df["Tool wear min"] * train_df["Rotational speed rpm"]
train_df["tool_wear_speed"].head()


# In[19]:


train_df["torque wear ratio"] = train_df["Torque Nm"] / (train_df["Tool wear min"] + 0.0001)
train_df["torque times wear"] = train_df["Torque Nm"] * train_df["Tool wear min"]
train_df.head()


# In[20]:


train_df["torque wear ratio"] = train_df["Torque Nm"] / (train_df["Tool wear min"] + 0.0001)
train_df["torque times wear"] = train_df["Torque Nm"] * train_df["Tool wear min"]
train_df.head()


# In[21]:


train_df["product_id_num"] = pd.to_numeric(train_df["Product ID"].str.slice(start=1))
train_df[["Product ID", "product_id_num"]].head()


# In[22]:


train_df["product_id_num"].plot.hist()


# In[23]:


train_df


# # Grouping features - Thanks to https://www.kaggle.com/erokhinvitaly for the idea!

# In[24]:


# some grouping features

# median for each Product ID
median_features = train_df.drop(["Machine failure","id"],axis=1).groupby("Product ID",as_index=False).transform('median')
print("done with median features")

# mean for each Product ID
mean_features = train_df.drop(["Machine failure","id"],axis=1).groupby("Product ID",as_index=False).transform('mean')
print("done with mean features")

# z score for each product relative to its product ID
z_features = train_df.drop(["Machine failure","id"],axis=1).groupby("Product ID",as_index=False).transform(lambda x : (x - x.mean()) / (x.std() + 0.0001))
print("done with z score features")

# range of each Product ID
range_features = train_df.drop(["Machine failure","id"],axis=1).groupby("Product ID",as_index=False).transform(lambda x: x.max() - x.min())
print("done with range features")

median_features.columns = median_features.columns + "_median"
mean_features.columns = mean_features.columns + "_mean"
z_features.columns = z_features.columns + "_z"
range_features.columns = range_features.columns + "_range"

train_df = train_df.merge(median_features, left_index=True, right_index=True, how="left")
train_df = train_df.merge(mean_features, left_index=True, right_index=True, how="left")
train_df = train_df.merge(z_features, left_index=True, right_index=True, how="left")
train_df = train_df.merge(range_features, left_index=True, right_index=True, how="left")


# In[25]:


train_df.columns[train_df.isna().any()].tolist()


# In[26]:


train_df.fillna(0, inplace=True)


# # Split into X/y

# In[27]:


y = train_df.pop("Machine failure")
X = train_df.drop(["id", "Product ID"], axis=1)


# # Split into Train/Validation for XGBoost

# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8)


# # Create and Train Individual Classifiers

# In this section, we will train the following models:
# 
# * Random Forest
# * Logistic Regression
# * Gradient Boosting Classifier
# * XGBoost (using a custom class to ensure compatibility with sklearn's `VotingClassifier`)
# 
# We will evaluate each model's performance. Later, we will combine them into a single voting classifier.

# In[29]:


from sklearn.metrics import roc_auc_score


# In[30]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500)
rf.fit(X_train, y_train)

print(roc_auc_score(y_val, [x[1] for x in rf.predict_proba(X_val)]))


# In[31]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print(roc_auc_score(y_val, [x[1] for x in lr.predict_proba(X_val)]))


# In[32]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=200)
gbc.fit(X_train, y_train)

print(roc_auc_score(y_val, [x[1] for x in gbc.predict_proba(X_val)]))


# The following Python code is how I determined the parameters for XGBoost
# ```
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = {
#     'learning_rate': [0.1, 0.01],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.2]
# }
# 
# xgb = XGBClassifier(n_estimators = 1500, tree_method='gpu_hist', predictor='gpu_predictor')
# # xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
# 
# grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=3, scoring="roc_auc")
# grid_search.fit(X_train, y_train)
# 
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# ```

# In[33]:


from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    
    def __init__(self, **params):
        
        super().__init__(**params)
        self.eval_set = params['eval_set']
    
    def fit(self, X, y):
        super().fit(X, y, eval_set=self.eval_set, verbose=100) # output progress every 100 iterations


# In[34]:


xgb = CustomXGBClassifier(n_estimators = 1500,
                          gamma=0,
                          learning_rate=0.01,
                          max_depth=3,
                          min_child_weight=1,
                          tree_method='gpu_hist',
                          predictor='gpu_predictor',
                          early_stopping_rounds=15,
                          eval_set=[(X_val, y_val)])
xgb.fit(X_train, y_train)

print(roc_auc_score(y_val, [x[1] for x in xgb.predict_proba(X_val)]))


# **LightGBM**

# Similarly to XGBoost, LightGBM works well when it can cross-validate with the validation data after each iteration of training.
# 
# For some reason, trying to include LightGBM regressor as part of the ensemble didn't work out, so I decided to train LightGBM separately and use a weighted average for the final prediction.

# Below are the hyperparameters to use for training.

# <div class="alert alert-block alert-warning">
# <b>Note:</b> These hyperparameters are NOT optimized. Even better results can be achieved using hyperparameter optimization techniques like Optuna or Hyperopt.
# </div>

# In[35]:


hyper_params = {
    'objective': 'regression',
    'metric': 'auc',
    'learning_rate': 0.005,
    'verbose': -1,
    'n_estimators': 100000,
    'random_state' : 0,
    'device' : 'gpu'
}


# In[36]:


import lightgbm

lgbm = lightgbm.LGBMClassifier(**hyper_params)

lgbm.fit(X_train,
         y_train,
         eval_set = [(X_val, y_val)],
         callbacks = [lightgbm.early_stopping(stopping_rounds = 20)]
)


# # Train EnsembleVoter

# In[37]:


from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('xgb', xgb),
                                     ('rf', rf),
                                     ('gbc', gbc)],
                        voting='soft')

model.fit(X_train, y_train)

print(roc_auc_score(y_val, [x[1] for x in model.predict_proba(X_val)]))


# # Process Testing Data

# Here I condense all the processing into one code cell.

# In[38]:


test_df = pd.read_csv("/kaggle/input/playground-series-s3e17/test.csv")
test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])
test_df.columns = test_df.columns.str.replace('[\[\]]', '', regex=True)
test_df["Power"] = test_df["Torque Nm"] * test_df["Rotational speed rpm"]
test_df["temp_ratio"] = test_df["Process temperature K"] / test_df["Air temperature K"]
test_df["Process temperature C"] = test_df["Process temperature K"] - 273.15
test_df["Air temperature C"] = test_df["Air temperature K"] - 273.15
test_df["temp_C_ratio"] = test_df["Process temperature C"] / test_df["Air temperature C"]
test_df["Failure Sum"] = (test_df["TWF"] +
                            test_df["HDF"] +
                            test_df["PWF"] +
                            test_df["OSF"] +
                            test_df["RNF"])

test_df["tool_wear_speed"] = test_df["Tool wear min"] * test_df["Rotational speed rpm"]
test_df["torque wear ratio"] = test_df["Torque Nm"] / (test_df["Tool wear min"] + 0.0001)
test_df["torque times wear"] = test_df["Torque Nm"] * test_df["Tool wear min"]
test_df["product_id_num"] = pd.to_numeric(test_df["Product ID"].str.slice(start=1))

# some grouping features

# median for each Product ID
median_features = test_df.drop(["id"],axis=1).groupby("Product ID",as_index=False).transform('median')

# mean for each Product ID
mean_features = test_df.drop(["id"],axis=1).groupby("Product ID",as_index=False).transform('mean')

# z score for each product relative to its product ID
z_features = test_df.drop(["id"],axis=1).groupby("Product ID",as_index=False).transform(lambda x: (x - x.mean()) / x.std())

# range of each Product ID
range_features = test_df.drop(["id"],axis=1).groupby("Product ID",as_index=False).transform(lambda x: x.max() - x.min())

median_features.columns = median_features.columns + "_median"
mean_features.columns = mean_features.columns + "_mean"
z_features.columns = z_features.columns + "_z"
range_features.columns = range_features.columns + "_range"

test_df = test_df.merge(median_features, left_index=True, right_index=True, how="left")
test_df = test_df.merge(mean_features, left_index=True, right_index=True, how="left")
test_df = test_df.merge(z_features, left_index=True, right_index=True, how="left")
test_df = test_df.merge(range_features, left_index=True, right_index=True, how="left")
test_df.fillna(0, inplace=True)

ids = test_df["id"]
test_X = test_df.drop(["id", "Product ID"], axis=1)


# # Prediction and Submission

# In[39]:


ensemble_preds = model.predict_proba(test_X)
lgbm_preds = lgbm.predict_proba(test_X)


# In[40]:


predicted_probs_ensemble = np.array([pred[1] for pred in ensemble_preds])
predicted_probs_lgbm = np.array([pred[1] for pred in lgbm_preds])

predicted_probs_ensemble[:5]


# In[41]:


predicted_probs_lgbm[:15]


# Calculate weighted predictions by multiplying each element of `ensemble_preds` by 0.75 and each element of `lgbm_preds` by 0.25.
# 
# This is equivalent to making the LightGBM model a part of the `VotingRegressor`.

# In[42]:


predicted_prob = predicted_probs_ensemble * 0.75 + predicted_probs_lgbm * 0.25


# In[43]:


submission_df = pd.DataFrame({
"id" : ids,
"Machine failure": predicted_prob   
})
submission_df.shape


# In[44]:


submission_df.head()


# In[45]:


submission_df.describe()


# In[46]:


submission_df.to_csv("submission.csv", index=False)

