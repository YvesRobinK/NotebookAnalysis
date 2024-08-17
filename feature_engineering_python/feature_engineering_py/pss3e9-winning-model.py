#!/usr/bin/env python
# coding: utf-8

# # Concrete strength prediction
# 
# This notebook predicts concrete strength for the Playground Series Season 3 Episode 9 competition:
# - Feature engineering is derived from the [EDA which makes sense](https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense).
# - `AgeInDays` is target-encoded as if it were a categorical variable.
# - The original data is not used.
# - The final model is a blend of `GradientBoostingRegressor`, `LGBMRegressor`, `RandomForestRegressor` and `Ridge`.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, lightgbm, math, itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

np.set_printoptions(linewidth=150, edgeitems=5)
result_list = []


# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv', index_col='id')
# original = pd.read_csv('/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv')
# original.rename(columns={"CementComponent ": "CementComponent"}, inplace=True)

target = 'Strength'
original_features = list(test.columns)

print(f"Length of train: {len(train)}")
print(f"Length of test:  {len(test)}")
# print(f"Length of original: {len(original)}")
print()

temp1 = train.isna().sum().sum()
temp2 = test.isna().sum().sum()
if temp1 == 0 and temp2 == 0:
    print('There are no null values in train and test.')
else:
    print(f'There are {temp1} null values in train')
    print(f'There are {temp2} null values in train')
print()

print('Sample lines from train:')
train.tail(3)


# # Feature engineering

# In[3]:


for df in [train, test]:
#     df['Water_Cement'] = df['WaterComponent']/df['CementComponent'] # useless
#     df['Aggregate'] = df['CoarseAggregateComponent'] + df['FineAggregateComponent'] # useless
#     df['Aggregate_Cement'] = df['Aggregate']/df['CementComponent'] # useless
#     df['Slag_Cement'] = df['BlastFurnaceSlag']/df['CementComponent'] # useless
#     df['Ash_Cement'] = df['FlyAshComponent']/df['CementComponent'] # useless
#     df['Plastic_Cement'] = df['SuperplasticizerComponent']/df['CementComponent'] # useless
    df['Age_Water'] = df['AgeInDays'] / df['WaterComponent']
    df['Age_Cement'] = df['AgeInDays'] / df['CementComponent']
    df['Coarse_Fine'] = df['CoarseAggregateComponent'] / df['FineAggregateComponent']
    df['youngCementComponent'] = df.CementComponent * (df.AgeInDays < 40)
    df['youngSuperplasticizerComponent'] = df.SuperplasticizerComponent * (df.AgeInDays < 10)
    df['clippedAge'] = df.AgeInDays.clip(None, 40)
    df['clippedWater'] = df.WaterComponent.clip(195, None)
    df['hasBlastFurnaceSlag'] = df.BlastFurnaceSlag != 0
    df['hasFlyAshComponent'] = df.FlyAshComponent != 0
    df['hasSuperplasticizerComponent'] = df.SuperplasticizerComponent != 0
    


# In[4]:


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes the AgeInDays values by their average target value"""
    def fit(self, X, y):
        # VotingRegressor forwards y as an array
        if type(y) == np.ndarray:
            y = pd.Series(y, index=X.index)
        self.encodings_ = y.groupby(X['AgeInDays'].apply(self.replace_rare)).mean()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['AgeInDays'] = self.encodings_.reindex(X['AgeInDays'].apply(self.replace_rare)).values
        return X

    @staticmethod
    def replace_rare(x):
        """Replace the rare AgeInDays values by nearby values"""
        if x == 1: return 3
        if x == 11: return 14
        if x == 49: return 56
#         if x == 91: return 90
#         if x == 120: return 100
        if x == 360: return 365
        return x



# # Cross-validation

# In[5]:


def score_model(model, features_used, label=None):
    """Cross-validate a model with selected features"""
    score_list = []
    oof = np.zeros_like(train[target])
    kf = KFold(shuffle=True, random_state=333)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train,
                                                     groups=train[original_features].apply(tuple, axis=1))):
        X_tr = train.iloc[idx_tr][features_used]
        X_va = train.iloc[idx_va][features_used]
        y_tr = train.iloc[idx_tr][target]
        y_va = train.iloc[idx_va][target]
        
#         X_tr = pd.concat([X_tr, original[features_used]], axis=0)
#         y_tr = pd.concat([y_tr, original[target]], axis=0)
        
        model.fit(X_tr, y_tr)
        trmse = mean_squared_error(y_tr, model.predict(X_tr), squared=False)
        y_va_pred = model.predict(X_va)
        rmse = mean_squared_error(y_va, y_va_pred, squared=False)
        if type(model) == Pipeline and type(model.steps[-1][1]) == Ridge:
            print('Weights:', model.steps[-1][1].coef_.round(2))
        print(f"Fold {fold}: trmse = {trmse:.3f}   rmse = {rmse:.3f}")
        oof[idx_va] = y_va_pred
        score_list.append(rmse)

    rmse = sum(score_list) / len(score_list)
    print(f"Average rmse: {rmse:.3f} {label if label is not None else ''}")
    if label is not None:
        global result_list
        result_list.append((label, rmse, oof))
    idxs = np.argsort(oof)
    oof_i = oof[idxs]
    y_true_i = train[target].iloc[idxs]
    s = pd.Series(y_true_i.values, index=oof_i)
    s = s.rolling(100, center=True).mean()
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 2, 1) # y_true vs. y_pred
    plt.scatter(oof, train[target], s=3)
    ax.scatter(s.index, s, s=2, c='r')
    ax.plot([s.index.min(), s.index.max()], [s.index.min(), s.index.max()], c='y', lw='2')
    ax.set_aspect('equal')
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    plt.subplot(1, 2, 2) # histogram
    plt.hist(oof, bins=100)
    plt.xlabel('y_pred')
    plt.ylabel('count')
    plt.show()


# In[6]:


ridge_features = ['CementComponent', 'BlastFurnaceSlag', 'WaterComponent', 'SuperplasticizerComponent',
                     'CoarseAggregateComponent', 'FineAggregateComponent', 'AgeInDays', 'hasBlastFurnaceSlag',
                     'hasSuperplasticizerComponent', 'clippedWater', 'Coarse_Fine', 'clippedAge',
                     'youngCementComponent', 'youngSuperplasticizerComponent'
                    ]
score_model(model=make_pipeline(TargetEncoder(), StandardScaler(), Ridge(30)),
            features_used=ridge_features,
            label='Ridge')


# In[7]:


get_ipython().run_cell_magic('time', '', "rf_features = original_features + ['Age_Water', 'Age_Cement']\nscore_model(model=make_pipeline(TargetEncoder(), RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=1)),\n            features_used=rf_features,\n            label='Random Forest')\n")


# In[8]:


get_ipython().run_cell_magic('time', '', "lgbm_params = {\n        'learning_rate': 0.0005,\n        'n_estimators': 20000,\n        'num_leaves': 7,\n        'colsample_bytree': 0.4,\n        'subsample': 0.5,\n        'subsample_freq': 6,\n        'min_child_samples': 25,\n    }\n\nscore_model(model=lightgbm.LGBMRegressor(**lgbm_params, random_state=1),\n            features_used=original_features,\n            label='LGBM')\n")


# In[9]:


gbr_params = {'n_estimators': 600,
              'max_depth': 4,
              'learning_rate': 0.01,
              'min_samples_leaf': 40 ,
              'max_features': 3}
score_model(model=make_pipeline(TargetEncoder(), GradientBoostingRegressor(**gbr_params, random_state=2)),
            features_used=original_features,
            label='GradientBoostingRegressor')



# # Ensemble

# In[10]:


get_ipython().run_cell_magic('time', '', "ensemble_model = VotingRegressor([('gb', make_pipeline(TargetEncoder(),\n                                                       ColumnTransformer([('pt', 'passthrough', original_features)]),\n                                                       GradientBoostingRegressor(**gbr_params, random_state=1))),\n                                  ('rf', make_pipeline(TargetEncoder(),\n                                                       ColumnTransformer([('pt', 'passthrough', rf_features)]),\n                                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=1))),\n                                  ('ridge', make_pipeline(TargetEncoder(),\n                                                          ColumnTransformer([('pt', 'passthrough', ridge_features)]),\n                                                          StandardScaler(),\n                                                          Ridge(30))),\n                                 ],\n                                 weights=[0.35, 0.3, 0.35])\nscore_model(model=ensemble_model,\n            features_used=test.columns,\n            label='GradientBoostingRegressor + RF + Ridge')\n")


# In[11]:


get_ipython().run_cell_magic('time', '', "ensemble_model = VotingRegressor([('gb', make_pipeline(TargetEncoder(),\n                                                       ColumnTransformer([('pt', 'passthrough', original_features)]),\n                                                       GradientBoostingRegressor(**gbr_params, random_state=1))),\n                                  ('lgbm', make_pipeline(ColumnTransformer([('pt', 'passthrough', original_features)]),\n                                                         lightgbm.LGBMRegressor(**lgbm_params, random_state=1))),\n                                  ('rf', make_pipeline(TargetEncoder(),\n                                                       ColumnTransformer([('pt', 'passthrough', rf_features)]),\n                                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=1))),\n                                  ('ridge', make_pipeline(TargetEncoder(),\n                                                          ColumnTransformer([('pt', 'passthrough', ridge_features)]),\n                                                          StandardScaler(),\n                                                          Ridge(30))),\n                                 ],\n                                 weights=[0.2, 0.2, 0.3, 0.3])\nscore_model(model=ensemble_model,\n            features_used=test.columns,\n            label='GradientBoostingRegressor + LGBM + RF + Ridge')\n")


# # Final comparison

# In[12]:


result_df = pd.DataFrame(result_list, columns=['label', 'rmse', 'oof'])
result_df.drop_duplicates(subset='label', keep='last', inplace=True)


# In[13]:


result_df.sort_values('rmse', inplace=True)
with pd.option_context("precision", 3):
    display(result_df[['label', 'rmse']])
plt.figure(figsize=(6, len(result_df) * 0.3))
plt.title('Final comparison')
plt.barh(np.arange(len(result_df)), result_df.rmse, color='orange')
plt.gca().invert_yaxis()
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.xticks(np.linspace(12.0, 12.2, 5))
plt.xlabel('RMSE')
plt.xlim(12.0, 12.2)
plt.show()


# # Prediction and submission

# In[14]:


get_ipython().run_cell_magic('time', '', 'ensemble_model.fit(train[test.columns], train[target])\ny_pred = ensemble_model.predict(test[test.columns])\npd.Series(y_pred, index=test.index, name=target).to_csv(f"submission.csv")\ny_pred.round(1)\n')


# In[ ]:




