#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary libraries
import numpy as np
import math
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing

import pandas as pd
pd.set_option('display.max_colwidth', None)

# Data cleaning
from scipy import stats
from scipy.special import inv_boxcox

# Data visualization for exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Feature engineering
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans

# Model and evaluation
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

data = fetch_california_housing()
train_path = '/kaggle/input/playground-series-s3e1/train.csv'
test_path = '/kaggle/input/playground-series-s3e1/test.csv'
train = pd.read_csv(train_path)
# sk_ds = pd.DataFrame(data=data.data, columns=data.feature_names,)
# sk_ds['MedHouseVal'] = pd.Series(data.target)
# train = pd.concat([train, sk_ds], ignore_index=True)
test = pd.read_csv(test_path)


# # Exploratory Data Analysis

# In[2]:


train


# What do I know about the data?
# * `MedInc` - median income in block group.
# * `HouseAge` - median house age in block group.
# * `AveRooms` - average number of rooms per household.
# * `AveBedrms` - average number of bedrooms per household.
# * `Population` - block group population.
# * `AveOccup` - average number of household members.
# * `Latitude` - block group latitude.
# * `Longitude` - block group longitude.
# * `MedHouseVal` (target) - median house value in block group (in $100,000s).
# * `Household` - a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

# ## Finding correlations between columns

# In[3]:


corr = train.drop(columns=['id']).corr()
plt.figure(figsize=(12, 9))
ax = sns.heatmap(corr, annot=True)
ax.invert_yaxis(); ax.invert_xaxis()


# From the heatmap here, the only useful insight is that `MedInc` is highly correlated with out target, `MedHouseVal`.

# ## Let's see each of the column's distribution to make sure it's normally distributed or not.

# In[4]:


def remove_outliers(col):
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = stats.iqr(train[col])
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    mask = (train[col] > lower) & (train[col] < upper)
    train.drop(train[~mask].index, inplace=True)
    
def boxcox_tf(col):
    boxcox_col = f'{col}_boxcox'
    train[boxcox_col], _ = stats.boxcox(train[col])
    test[boxcox_col], _ = stats.boxcox(test[col])


# ### MedHouseVal (target)
# Let's start by analyzing our target. How is the distribution? Is it skewed? If so, we need to do some transformation in order to make the data more normally distributed.

# In[5]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
target = 'MedHouseVal'
for i in range(3):
    if i == 0:
        sns.histplot(train[target], ax=axes[i], kde=True)
    elif i == 1:
        stats.probplot(train[target], plot=axes[i])
    else:
        sns.boxplot(train[target], ax=axes[i])
skewness = train[target].skew()
print("===== Skewness =====")
print(f'{target} = {skewness}')


# Ow, that's quite skewed and there are some outliers too (which is bad for our model)! I need to transform the data into something (from trial and error, boxcox works the best for me). After that, I will remove the outliers so that our model can perform better!

# In[6]:


train['MedHouseVal_boxcox'], train_lambda = stats.boxcox(train.MedHouseVal)
remove_outliers('MedHouseVal_boxcox')


# Okay, transforming and removing outliers done! Let's see how the data looks now.

# In[7]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
target = 'MedHouseVal_boxcox'
# fig.subplots_adjust(hspace=0.35)
for i in range(3):
    if i == 0:
        sns.histplot(train[target], ax=axes[i], kde=True)
    elif i == 1:
        stats.probplot(train[target], plot=axes[i])
    else:
        sns.boxplot(train[target], ax=axes[i])
skewness = train[target].skew()
print("===== Skewness =====")
print(f'{target} = {skewness}')


# Wow, that's much better than before. Not the best looking distribution, but still there are improvements and that's great.

# ### Feature Columns
# Now let's see the columns that we have here. How will out feature's distribution look like?

# In[8]:


columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
skewness = [0 for _ in range(6)]
fig, axes = plt.subplots(3, 6, figsize=(18, 12))
fig.subplots_adjust(hspace=0.35)
for i in range(3):
    for j in range(6):
        if not i:
            sns.histplot(train[columns[j]], ax=axes[i, j], kde=True)
        elif i == 1:
            stats.probplot(train[columns[j]], plot=axes[i, j])
        else:
            sns.boxplot(train[columns[j]], ax=axes[i, j])
        skewness[j] = train[columns[j]].skew()
print("===== Skewness =====")
for column, skew in zip(columns, skewness):
    print(f'{column} = {skew}')


# That's... horrible, just horrible. Skewness 145?? Look at those outliers!!! They need some cleaning, seriously.

# In[9]:


columns_boxcox = [f'{col}_boxcox' for col in columns]
for col, bc in zip(columns, columns_boxcox):
    train[bc], _ = stats.boxcox(train[col])
    test[bc], _ = stats.boxcox(test[col])
    remove_outliers(bc)


# Okay, I've done the transformation and cleaning of outliers. Let's see how the data looks now!

# In[10]:


skewness = [0 for _ in range(6)]
fig, axes = plt.subplots(3, 6, figsize=(18, 12))
fig.subplots_adjust(hspace=0.35)
for i in range(3):
    for j in range(6):
        if not i:
            sns.histplot(train[columns_boxcox[j]], ax=axes[i, j], kde=True)
        elif i == 1:
            stats.probplot(train[columns_boxcox[j]], plot=axes[i, j])
        else:
            sns.boxplot(train[columns_boxcox[j]], ax=axes[i, j])
        skewness[j] = train[columns_boxcox[j]].skew()
print("===== Skewness =====")
for column, skew in zip(columns_boxcox, skewness):
    print(f'{column} = {skew}')


# That's much better. Those graphs are looking far cleaner than before, so much that I can't tell that they are the same data from before!

# ### Latitude and longitude mapping to map
# Since latitude and longitude data is different from other numerical data, I figured that it's better to analyze them separately. It's also cooler to map them to real world maps hehe, and I can also get some insight from it. Let's plot and color them based on the median income of the block.

# In[11]:


df = train[['Latitude', 'Longitude', 'MedInc_boxcox']]
fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="MedInc_boxcox",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5,
                  mapbox_style="carto-positron", opacity=0.4)
fig.update_layout(mapbox_center=dict(lat=36, lon=-120))
fig.show()


# Looks like most of the high median income blocks are near the shores. Noted, maybe it will come in handy on the feature engineering.

# # Feature Engineering
# Let's make features in order to improve our model's score! Here are the records of CV score before feature engineering score (the first one) and after feature engineering (second and third one):
# * `xgboost`: 0.5033229179121079 -> 0.5032321486046868 -> 0.5019607432320747
# * `lightgbm`: 0.2488430000278897 -> 0.24978599758661496 -> 0.2500165285657254

# I'll start by making a new dataframe so we can add or remove features without needing to worry about changing the original dataset.

# In[12]:


features = [col for col in train.columns if 'boxcox' in str(col)]
features.remove('MedHouseVal_boxcox')
features.append('Longitude'); features.append('Latitude');

X = train[features]
X_test = test[features]

y = train.MedHouseVal_boxcox


# ## Mutual Information
# Let's see how good each features are to our target! Maybe from that we will get some insight on what to do with the data that we have to engineer a new feature!

# In[13]:


mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


# ## Making features from the numerical columns
# This made the model perform horribly so I'm commenting this out lol

# In[14]:


# # Make a feature of total income from the multiplication of Population and MedInc
# train['TotalIncome'] = train.Population * train.MedInc
# train['TotalIncome_boxcox'], _ = stats.boxcox(train.TotalIncome)
# test['TotalIncome'] = test.Population * test.MedInc
# test['TotalIncome_boxcox'], _ = stats.boxcox(test.TotalIncome)

# # Make a feature of block house density (houses per block) from Population/AveOccup
# train['BlockDensity'] = train.Population / train.AveOccup
# train['BlockDensity_boxcox'], _ = stats.boxcox(train.BlockDensity)
# test['TotalIncome'] = test.Population * test.MedInc
# test['TotalIncome_boxcox'], _ = stats.boxcox(test.TotalIncome)


# ### Making features from latitude and longitude using clustering
# From the mutual information I see that `Latitude`, `Longitude`, and `MedInc` have the biggest score. I'm thinking of creating clusters based on their location and median income, that makes sense too right? The higher the incomes are, the higher the house value will be.

# In[15]:


coor = train[['Latitude', 'Longitude', 'MedInc_boxcox']].values
coor_test = test[['Latitude', 'Longitude', 'MedInc_boxcox']].values

kmeans = KMeans(n_clusters=5)
kmeans.fit(coor)

X['Cluster'] = kmeans.predict(coor)
X_test['Cluster'] = kmeans.predict(coor_test)

X['Cluster'] = X['Cluster'].astype('category')
X_test['Cluster'] = X_test['Cluster'].astype('category')


# In[16]:


df = X[['Latitude', 'Longitude', 'Cluster']]
fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Cluster",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5,
                  mapbox_style="carto-positron")
fig.update_layout(mapbox_center=dict(lat=36, lon=-120))
fig.show()


# ## Target encoding
# Now that the clustering is done, why don't I do some target encoding also? I'll add a feature that will gives the mean of`MedInc_boxcox` of their cluster, maybe that will help our model predict!

# In[17]:


X['Cluster_MedInc'] = X.groupby('Cluster')['MedInc_boxcox'].transform('mean')
X_test['Cluster_MedInc'] = X_test.groupby('Cluster')['MedInc_boxcox'].transform('mean')


# # Modelling
# Finally, the most tiring part is done. Time for some modelling. I'm planning to do stacking and ensemble later, but for now I'll just try some models and use the best one.

# In[18]:


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)


# ## XGBoost Model

# In[19]:


xgb_params = {
    'alpha': 9.996447466463037,
    'colsample_bytree': 0.5,
    'eval_metric': 'rmse',
    'lambda': 0.03775079369258294,
    'learning_rate': 0.0484609306,
    'n_estimators': 670,
}
xgb_model = xgb.XGBRegressor(
    **xgb_params,
    n_jobs=4,
#     tree_method='gpu_hist',
    verbosity=0
)
xgb_model.fit(X, y)
xgb.plot_importance(xgb_model)


# In[20]:


cv_params = {
    "early_stopping_rounds": 10,
    "nfold": 5,
    "metrics": 'rmse',
    "num_boost_round": 670,
}

data_dmatrix = xgb.DMatrix(data=X, label=inv_boxcox(y, train_lambda))
xgb_cv = xgb.cv(dtrain=data_dmatrix, params=xgb_params, **cv_params)
xgb_cv['test-rmse-mean'].iloc[-1]


# ## LGBM Model

# In[21]:


lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    "min_child_weight":10,
    'zero_as_missing':True
}
lgb_cv = lgb.cv(
    params = lgbm_params,
    train_set = lgb.Dataset(X, y),
    num_boost_round=2000,
    stratified=False,
    nfold = 5,
    verbose_eval=50,
    seed = 23,
    early_stopping_rounds=75
)


# In[22]:


optimal_rounds = np.argmin(lgb_cv['rmse-mean'])
best_cv_score = min(lgb_cv['rmse-mean'])
print(f'{optimal_rounds} rounds: {best_cv_score}')


# In[23]:


lgb_model = lgb.LGBMRegressor(n_estimators=optimal_rounds, **lgbm_params)
lgb_model.fit(X, y)
lgb.plot_importance(lgb_model)


# ## Catboost

# In[24]:


cb_data = cb.Pool(data=X, label=y)
params = {
    "iterations": 1000,
    "depth": 2,
    "loss_function": "RMSE",
    "verbose": False
}

scores = cb.cv(cb_data, params, nfold=10)

cb_model = cb.CatBoostRegressor()
cb_model.fit(X, y)


# In[25]:


feature_importance = cb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')


# # Submission

# LGBM scored better in the crossvalidation, so I guess I'll use that model.

# In[26]:


prediction = lgb_model.predict(X_test)
submission = pd.DataFrame({
    'id': test.id,
    'MedHouseVal': inv_boxcox(prediction, train_lambda)
})
submission.to_csv('submission.csv', index=False)
print('Successfully made a prediction!')

