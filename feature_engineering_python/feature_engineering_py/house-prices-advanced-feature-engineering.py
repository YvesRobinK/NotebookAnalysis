#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# Hello everyone!
# 
# The key objective is to use feature engineering to improve performance for tree-based and non-tree models.

# ## Load packages and data

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)
import numpy as np
import seaborn as sns

from scipy import stats


# In[2]:


X_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


# In[3]:


X_train.shape, X_test.shape


# In[4]:


X_train.head()


# In[5]:


X_test.head()


# ## EDA

# In[6]:


correlation_train=X_train.corr()
sns.set(font_scale=1.2)
mask = np.triu(correlation_train.corr())
plt.figure(figsize = (20,20))
ax = sns.heatmap(correlation_train, annot=True,annot_kws={"size": 11},fmt='.1f', linewidths=.5, square=True, mask=mask)


# ### Target log-transformation

# In[7]:


y = X_train.SalePrice.reset_index(drop=True)
X_train.drop(['SalePrice'], axis=1, inplace=True)


# In[8]:


plt.figure(figsize=(12,6))
sns.distplot(y)


# In[9]:


stats.probplot(y, plot=plt)
print(f"Skewness: {y.skew():.3f}")


# In[10]:


y = np.log1p(y)            
plt.figure(figsize=(12,6))
sns.distplot(y)


# In[11]:


stats.probplot(y, plot=plt)
print(f"Skewness: {y.skew():.3f}")


# ### Detecting missing values

# In[12]:


train_test = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
train_test.shape


# In[13]:


# Find Missing Ratio of Dataset
missing = (train_test.isnull().sum() / len(train_test)) * 100
missing = missing.drop(missing[missing == 0].index).sort_values(ascending=False)[:35]


# In[14]:


f, ax = plt.subplots(figsize=(12, 10))
plt.xticks(rotation='90')
sns.barplot(x=missing.index, y=missing)
plt.xlabel('Features')
plt.ylabel('%')
plt.title('Percentage of missing values');


# ### Imputing nominal categorical features

# In[15]:


train_test['MSSubClass'] = train_test['MSSubClass'].astype(str)
train_test['MoSold'] = train_test['MoSold'].astype(str)
train_test['YrSold'] = train_test['YrSold'].astype(str)


# In[16]:


none = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'GarageType','MasVnrType']
for col in none:
    train_test[col].replace(np.nan, 'None', inplace=True)


# In[17]:


train_test['MSZoning'] = train_test.groupby('MSSubClass')['MSZoning'].transform(
    lambda x: x.fillna(x.mode()[0]))


# In[18]:


freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd',
    'SaleType', 'Utilities'
]
for col in freq_cols:
    train_test[col].replace(np.nan, train_test[col].mode()[0], inplace=True)


# ### Encoding ordinal categorical features

# In[19]:


qualcond = ['GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual']
for f in qualcond:
    train_test[f] = train_test[f].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
train_test['BsmtExposure'] = train_test['BsmtExposure'].replace({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})
train_test['GarageFinish'] = train_test['GarageFinish'].replace({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})
basement = ['BsmtFinType1', 'BsmtFinType2']
for f in basement:
    train_test[f] = train_test[f].replace({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})


# In[20]:


functional = {'Typ': 3, 'Min1': 2.5, 'Min2': 2, 'Mod': 1.5, 'Maj1': 1, 'Maj2': 0.5, 'Sev': 0, 'Sal': 0}
train_test['Functional'] = train_test['Functional'].replace(functional)
train_test['CentralAir'] = train_test['CentralAir'].replace({'Y':1, 'N':0})


# In[21]:


train_test.isnull().sum().sort_values(ascending=False)[:22]


# Dropping features with one value - Utilities, Street, PoolQC

# In[22]:


train_test = train_test.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)


# ### Removing outliers

# In[23]:


outliers = [ 30, 462, 523, 588, 632, 1298, 1324]
train_test = train_test.drop(train_test.index[outliers])
linear_train_test = train_test.copy()
y = y.drop(y.index[outliers])


# ### Encoding nominal categorical features and Imputing missing values (Tree-based)

# In[24]:


cat_cols = [cname for cname in train_test.columns if  train_test[cname].dtype == "object"]
cat_cols
train = train_test.iloc[:1453]
test = train_test.iloc[1453:]


# CatBoostEncoder replaces a categorical value with the average value of the target from the rows before it. It works well with XGBoost and LightGBM

# In[25]:


from category_encoders import CatBoostEncoder
cbe = CatBoostEncoder()
train[cat_cols] = cbe.fit_transform(train[cat_cols], y)
test[cat_cols] = cbe.transform(test[cat_cols])


# In[26]:


train_test = pd.concat([train, test]).reset_index(drop=True)


# Each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set. Two samples are close if the features that neither is missing are close.

# In[27]:


from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=7, weights='distance', missing_values=np.nan)
imp_train_test = imp.fit_transform(train_test)


# In[28]:


train_test = pd.DataFrame(imp_train_test, columns=train_test.columns)


# In[29]:


missing = ['GarageCars', 'BsmtFinSF1', 'GarageArea', 'BsmtUnfSF', 'KitchenQual',
       'BsmtFinSF2', 'TotalBsmtSF', 'Functional', 'BsmtHalfBath',
       'BsmtFullBath', 'MasVnrArea', 'BsmtFinType1', 'BsmtFinType2',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageQual', 'GarageFinish',
       'GarageYrBlt', 'GarageCond', 'LotFrontage', 'FireplaceQu']


# In[30]:


train_test[missing] = train_test[missing].apply(lambda x: np.round(x))
train_test


# Feature Generation

# In[31]:


train_test['YearsSinceBuilt'] = train_test['YrSold'].astype(int) - train_test['YearBuilt']
train_test['YearsSinceRemod'] = train_test['YrSold'].astype(int) - train_test['YearRemodAdd']
train_test['TotalSF'] = train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF']

train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test['TotalPorchArea'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] +
                              train_test['EnclosedPorch'] + train_test['ScreenPorch'] +
                              train_test['WoodDeckSF'])
train_test['TotalOccupiedArea'] = train_test['TotalSF'] + train_test['TotalPorchArea']
train_test['OtherRooms'] = train_test['TotRmsAbvGrd'] - train_test['BedroomAbvGr'] - train_test['KitchenAbvGr']
train_test['haspool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['has2ndfloor'] = train_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasgarage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasbsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasfireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[32]:


print(train_test.shape)
train = train_test.iloc[:1453]
test = train_test.iloc[1453:]
print(train.shape, test.shape, y.shape)


# ### Encoding nominal categorical features, Imputing missing values and Scaling (Non-Tree)

# In[33]:


ohe = pd.get_dummies(linear_train_test).reset_index(drop=True)

imp = KNNImputer(n_neighbors=7, weights='distance', missing_values=np.nan)
imp_linear_train_test = imp.fit_transform(ohe)

linear_train_test = pd.DataFrame(imp_linear_train_test, columns=ohe.columns)

linear_train_test[missing] = linear_train_test[missing].apply(lambda x: np.round(x))


# In[34]:


linear_train_test['TotalSF'] = linear_train_test['TotalBsmtSF'] + linear_train_test['1stFlrSF'] + linear_train_test['2ndFlrSF']

linear_train_test['Total_Bathrooms'] = (linear_train_test['FullBath'] + (0.5 * linear_train_test['HalfBath']) +
                               linear_train_test['BsmtFullBath'] + (0.5 * linear_train_test['BsmtHalfBath']))

linear_train_test['TotalPorchArea'] = (linear_train_test['OpenPorchSF'] + linear_train_test['3SsnPorch'] +
                              linear_train_test['EnclosedPorch'] + linear_train_test['ScreenPorch'] +
                              linear_train_test['WoodDeckSF'])
linear_train_test['TotalOccupiedArea'] = linear_train_test['TotalSF'] + linear_train_test['TotalPorchArea']
linear_train_test['OtherRooms'] = linear_train_test['TotRmsAbvGrd'] - linear_train_test['BedroomAbvGr'] - linear_train_test['KitchenAbvGr']
linear_train_test['haspool'] = linear_train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
linear_train_test['has2ndfloor'] = linear_train_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
linear_train_test['hasgarage'] = linear_train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
linear_train_test['hasbsmt'] = linear_train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
linear_train_test['hasfireplace'] = linear_train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
linear_train_test


# In[35]:


from sklearn.preprocessing import StandardScaler

lin_train = linear_train_test.iloc[:1453]
lin_test = linear_train_test.iloc[1453:]


Scaler = StandardScaler()
scaled_train = Scaler.fit_transform(lin_train)
scaled_test = Scaler.transform(lin_test)

scaled_train = pd.DataFrame(scaled_train, columns=linear_train_test.columns)
scaled_test = pd.DataFrame(scaled_test, columns=linear_train_test.columns)


# In[36]:


print(scaled_train.shape, scaled_test.shape, y.shape)


# ## Feature Selection (Tree-based)

# We'll build a CatBoost model and find best features with SHAP Values

# In[37]:


from catboost import CatBoostRegressor, Pool
model = CatBoostRegressor(iterations=2500,
                            learning_rate=0.03,
                            depth=6,
                            loss_function='RMSE',
                            random_seed = 10,
                            bootstrap_type='Bernoulli',
                            subsample=0.66,
                            rsm=0.7
                         )


# In[38]:


model.fit(train, y, verbose=False, plot=False);


# In[39]:


import shap
shap.initjs()

shap_values = model.get_feature_importance(Pool(train, y), type='ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

shap.force_plot(expected_value, shap_values[0,:], train.iloc[0,:])


# ### SHAP importance

# In[40]:


shap.summary_plot(shap_values, train, max_display=88,  plot_type='bar')


# In[41]:


shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([train.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df.tail(35)


# We'll drop features with less than **1.5e-3** importance(you can change this threshold)

# In[42]:


drop = importance_df[importance_df['shap_importance'] < 1.5e-3].iloc[:,0].tolist()


# In[43]:


train_drop = train.drop(drop, axis=1)
test_drop = test.drop(drop, axis=1)
train_drop


# ## Feature Selection (Non-Tree-based)

# We'll use L1 Regularization with alpha 0.001

# In[44]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

def selected_features_l1(X,y,alpha):
    logistic = Lasso(random_state=20, alpha=alpha).fit(X,y)
    model = SelectFromModel(logistic, prefit=True)
    X_new = model.transform(X)
    selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                    index=X.index,
                                    columns=X.columns)
    features = selected_features.columns[selected_features.var() != 0]
    drop = selected_features.columns[selected_features.var() == 0]
    return features, drop


# In[45]:


features, drop = selected_features_l1(scaled_train, y, 0.001)
print(drop)


# In[46]:


selected_train =  scaled_train.drop(drop, axis=1)
selected_test = scaled_test.drop(drop, axis=1)


# ## Hyperparameter optimization

# We'll use Tree-structured Parzen Estimater (TPE), which is a form of Bayesian Optimization.
# 
# ![](https://miro.medium.com/max/700/1*tYWqO5BwNDVaM3kP3w1IAg.png)
# 
# We'll define hyperparameters and ranges with `trial`, perform 10-fold CV, and set direction with number of trials for optimization `n_trial` to 200. We'll use tree_method `gpu_hist` for faster computation.

# In[ ]:


import xgboost as xgb

import optuna


def objective(trial):
    dtrain = xgb.DMatrix(train_drop, label=y)

    param = {
        'seed': 20,
        'tree_method': 'gpu_hist',
        'max_depth': trial.suggest_int("max_depth", 3, 8),
        'eta' : trial.suggest_uniform("eta", 1e-3, 5e-2),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "gamma": trial.suggest_uniform("gamma", 1e-8, 1e-4),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.3, 1.0),        
    }
    if param['grow_policy']=="lossguide":
        param['max_leaves'] =  trial.suggest_int('max_leaves',2, 32)
    bst = xgb.cv(param, dtrain, num_boost_round=5000, nfold=10, early_stopping_rounds=50,  metrics='rmse', seed=20)
    score = bst['test-rmse-mean'].tail(1).values[0]
    return score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)
print(study.best_trial)


# After optimization, our best result will look like this:
# 
# Finished trial#110 with value: **0.1036024** with parameters:
# {'max_depth': 3, 'eta': 0.01423912926193527, 'grow_policy': 'lossguide', 'gamma': 2.804584764149306e-05, 
# 'colsample_bytree': 0.2403604834036041, 'subsample': 0.38141269740154965, 'max_leaves': 6}.
# 
# We can see our hyperparameter's history

# In[48]:


from optuna.visualization import plot_parallel_coordinate

plot_parallel_coordinate(study)


# ### Plot hyperparameter importance

# In[49]:


import optuna

importances = optuna.importance.get_param_importances(study)
importance_values = list(importances.values())
param_names = list(importances.keys())
params = pd.DataFrame([param_names, importance_values]).T
params.columns = ['param_name', 'importance']
params = params.sort_values('importance', ascending=False)
sns.catplot(x='param_name', y='importance', data=params, kind='bar')
plt.xticks(rotation='45');


# Looking at the charts above, we can conclude that we can change boundaries for hyperparameters.
# 
# For example: `eta: 7e-3, 2.2e-2`, `max_leaves: 2,16`, `gamma: 1e-6, 7e-5`, `colsample_bytree: 0.1, 0.4`, `subsample: 0.3, 0.6`
# 
# After another optimization, our best result improved to **0.1031473**

# It was an example of one model optimization, and we can apply this steps to any model.
