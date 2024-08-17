#!/usr/bin/env python
# coding: utf-8

# # **Feature Engineering & CATBoostRegressor**
# 
# Getting started with competitive data science can be quite intimidating. So I wrote this quick overview of how I made top 12% on the Advanced Regression Techniques competition. If there is interest, I’m happy to do deep dives into the intuition behind the feature engineering and models used in this kernel.
# 
# If you like this kernel, please give it an upvote. Thank you!
# 
# # **The Goal**
# 
# - Each row in the dataset describes the characteristics of a house.
# 
# - Our goal is to predict the SalePrice, given these features.
# 
# - Our models are evaluated on the Root-Mean-Squared-Error (RMSE) between the log of the SalePrice predicted by our model, and the log of the actual SalePrice. Converting RMSE errors to a log scale ensures that errors in predicting expensive houses and cheap houses will affect our score equally.

# # **Table of Contents**
# 
# * [1. IMPORTING PACKAGES](#section1)
# * [2. LOADING DATASETS](#section2)        
# * [3. DATA CLEANING](#section3)
# * [4. ESTABLISH BASELINE](#section4)
# * [5. FEATURE MI SCORES](#section5)
# * [6. OUTLIERS](#section6)
# * [7. FEATURE ENGINEERING](#section7)
#     * [7.1 Create Mathematical Transforms](#section7.1)
#     * [7.2 Count Feature](#section7.2)
#     * [7.3 Use a Grouped Transform](#section7.3)
#     * [7.4 Square Root of Area Features](#section7.4)  
# * [8. LOG TRANFORMATION](#section8) 
# * [9. HYPERPARAMETER TUNING](#section9) 
# * [10.TRAIN MODEL AND SUBMISSION](#section10)

# # **1) IMPORTING PACKAGES** <a class="anchor"  id="section1"></a>

# In[1]:


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from catboost import CatBoostRegressor

# Filter Warnings
from warnings import simplefilter
simplefilter('ignore')

# Set Matplotlib defaults:
plt.style.use('seaborn-whitegrid')
sns.set_context('talk')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)


# # **2) LOADING DATASETS** <a class="anchor"  id="section2"></a>

# In[2]:


# Read the data
path = '../input/house-prices-advanced-regression-techniques/'
df_train = pd.read_csv(path + 'train.csv', index_col='Id')
df_test = pd.read_csv(path + 'test.csv', index_col='Id')

df = pd.concat([df_train, df_test])


# # **3) DATA CLEANING** <a class="anchor"  id="section3"></a>
# 
# Define two defenitions for data cleaning, one is for imputing missing values, and another for encoding categorical features.
# 
# Thess defenitions will return the concatenated dataframe after do below basic cleanings:
# 
# **1- For numerical columns, fill missing values by 0.**
# 
# **2- For categorical columns, fill missing values by 'None'**
# 
# **3- Encode the Statistical Data Type:** The numeric features are already encoded correctly (`float` for continuous, `int` for discrete), but the categoricals we'll need to do ourselves. Note in particular, that the `MSSubClass` feature is read as an `int` type, but is actually a (nominative) categorical.

# In[3]:


# The nominative (unordered) categorical features
features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
                "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle",
                "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating",
                "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]

# The ordinal (ordered) categorical features 

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(1,11))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"]
}
# Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()}

def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df

def impute(df):
    
    for col in df.select_dtypes(['int64','float64']).columns:
        df[col].fillna(0,inplace= True, axis=0)
    
    for col in df.select_dtypes(['object']).columns:
        df[col].fillna('None',inplace= True, axis=0)
        
    return df

def label_encode(df):

    for colname in df.select_dtypes(["category"]):
        df[colname] = df[colname].cat.codes
    return df

df = impute(df)
df = encode(df)
df = label_encode(df)

# Re-seperate df to df_train and df_test 
df_train = df.loc[df_train.index, :]
df_test = df.loc[df_test.index,:].drop('SalePrice', axis=1)


# # **4) ESTABLISH BASELINE** <a class="anchor"  id="section4"></a>
# 
# Finally, let's establish a baseline score to judge our feature engineering against.
# 
# Here is the function that will compute the cross-validated RMSLE score for a feature set. I've used CatBoostRegressor for the model.
# 
# We can reuse this scoring function anytime we want to try out a new feature set. We'll run it now on the processed data with no additional features and get a baseline score.

# In[4]:


def score_dataset(X, y, model = CatBoostRegressor(silent=True)):
    
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return np.round(score,4)

X = df_train.copy()
y = X.pop('SalePrice')

print('The Baseline RMSLE is: ',score_dataset(X,y))


# # **5) FEATURE MI SCORES** <a class="anchor"  id="section5"></a>
# 
# I will use mutual information to compute a utility score for a feature, this will give us an indication of how much potential the feature has. The next cell defines make_mi_score function.

# In[5]:


def make_mi_score(X,y):
    
    discrete_features = np.array(X.dtypes != 'float')
    mi_score = mutual_info_regression(X, y, discrete_features = discrete_features, random_state=0)
    mi_score = pd.Series(mi_score, index = X.columns, name = 'MI Scores').sort_values(ascending=False)

    return mi_score

mi_scores = np.round(make_mi_score(X,y),4)
display(mi_scores)


# You can see that we have a number of features that are highly informative and also some that don't seem to be informative at all (at least by themselves). The top scoring features will usually pay-off the most during feature development, so it could be a good idea to focus efforts on those. On the other hand, training on uninformative features can lead to overfitting. So, the features with 0.0 scores we'll drop entirely:

# In[6]:


cols_to_drop = list(mi_scores[mi_scores==0].index)
X = X.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)

print('RMSLE after drop uninformative features is: ',score_dataset(X,y))


# # **6) OUTLIERS**<a class="anchor"  id="section6"></a>
# 
# Outliers is also something that we should be aware of. Why? Because outliers can markedly affect our models and can be a valuable source of information, providing us insights about specific behaviours.
# 
# Outliers is a complex subject and it deserves more attention. Here, we'll just do a quick analysis through  a set of scatter plots for area columns which are more suspected to have outliers.

# In[7]:


fig, axes = plt.subplots(3, 2, figsize=(30, 25))
axes = axes.flatten()
suspected_outliers = ['LotArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea']
no=[0,1,2,3,4]

for n, i in zip(no,suspected_outliers):
    sns.scatterplot(ax = axes[n],data=X,x=i, y=y)
plt.show()


# Below are suspected outliers at the data, so I will drop them at he next code:
# 
# - `LotArea` > 200000.
# - `BsmtFinSF1` > 5000.
# - `TotalBsmtSF` > 6000.
# - `1stFlrSF` > 4000. 
# - `GrLivArea` > 4500.

# In[8]:


rows_to_drop = X.loc[(X['LotArea']>200000) | (X['BsmtFinSF1']>5000) |(X['TotalBsmtSF']>6000) |(X['1stFlrSF']>4000) |(X['GrLivArea']>4500)]

X = X.drop(rows_to_drop.index, axis=0)
y = y.drop(rows_to_drop.index)

print('RMSLE After Drop Outliers is: ',score_dataset(X,y))


# # **7) FEATURE ENGINEERING** <a class="anchor"  id="section7"></a>
# 
# Define a function for extracting additional features that can help in model training accuracy increasing.
# 
# This defenition will return the following new features based in previous dataframe in both of training / testing datasets:
#  
# **7.1) Create Mathematical Transforms** <a class="anchor"  id="section7.1"></a>
# 
#  Create the following features:
#  
#  - `InsideRatioQual`: the ratio of inside properties to `LotArea` multiplied by `OverallQual`
#  - `OutsideRatioQual`: the sum of outside properties multiplied by `OverallQual`
#  - `QualCondProduct`: the product of`OverallQual` & `OverallCond`
#  - `GarageRatio`: the product of`GarageQual` & `GarageArea`
#  - `BldgTypeArea`:`GrLivArea` multiplied by `BldgType` 
#  - `Total_Bathrooms`:the sum of all bathrooms 
# 
# **7.2) Count Feature** <a class="anchor"  id="section7.2"></a>
# 
# Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has. Create a feature `PorchTypes` that counts how many of the following are greater than 0.0:
# ```
# WoodDeckSF
# OpenPorchSF
# EnclosedPorch
# Threeseasonporch
# ScreenPorch
# ```
# **7.3) Use a Grouped Transform** <a class="anchor"  id="section7.3"></a>
# 
# The value of a home often depends on how it compares to typical homes in its neighborhood. Create a feature `MedNhbdArea` that describes the *median* of `GrLivArea` grouped on `Neighborhood`, also create a feature `CntNhbdQual` that describes the *count* of `OverallQual` grouped on `Neighborhood`
# 
# **7.4) Square Root of Area Features** <a class="anchor"  id="section7.4"></a>
# 
# This would convert units of square feet to just feet.

# In[9]:


def feature_engineering(X):
    
    #4.1) Create Mathematical Transforms
    X_1 = pd.DataFrame()
    X_1['InsideRatioQual'] = (X[['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea']].sum(axis=1)/ X['LotArea'])*X['OverallQual']
    X_1['OutsideRatioQual'] = X[['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis=1)*X['OverallQual']
    X_1['QualCondProduct'] = X['OverallQual'] * X['OverallCond']
    X_1['GarageRatio'] = X['GarageArea'] * X['GarageQual']
    X_1['BldgTypeArea'] = X['GrLivArea'] * X['BldgType']
    X_1['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) +
                               X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
    
    #4.2) Count Feature
    X_2 = pd.DataFrame()
    X_2["PorchTypes"] = X[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch']].gt(0).sum(axis=1)

    #4.3) Use a Grouped Transform
    X_3 = pd.DataFrame()
    X_3["MedNhbdArea"] = X.groupby('Neighborhood')['GrLivArea'].transform('median')
    X_3["CntNhbdQual"] = X.groupby('Neighborhood')['OverallQual'].transform('count')

    #4.4) Square root of Area Features
    area_features = ['LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF',
                     '2ndFlrSF', 'GrLivArea', 'GarageArea']
    
    X_4 = pd.DataFrame()
    for feat in area_features:
        X_4['SQRT'+feat] = np.sqrt(X[feat])

    # Concat and Return
    X = pd.concat([X,X_1,X_2,X_3,X_4], axis=1)
    return X

X = feature_engineering(X)
df_test = feature_engineering(df_test)

# Drop some features after we extracted them to more informative features
#X.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1,inplace=True)
#df_test.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1,inplace=True)

# scoring after feature engineering:
print('RMSLE after Feature Engineering is: ',score_dataset(X,y))


# # **8) LOG TRANSFORMATION** <a class="anchor"  id="section8"></a>
# 
# Logarithms of numeric features. If a feature has a skewed distribution, applying a logarithm can help normalize it.

# In[10]:


skewed_features = X.select_dtypes(['int64','float64']).skew().sort_values(ascending=False)
skewness = pd.Series(skewed_features)
skewness = skewness[skewness > 0.75]
print("There are {} skewed numerical features to Log transform".format(skewness.shape[0]))

for feat in skewness.index:
    X[feat] = np.log1p(X[feat])
    df_test[feat] = np.log1p(df_test[feat])
    
print('RMSLE after Log Transformation is: ',score_dataset(X,y))


# # **9) HYPERPARAMETER TUNING** <a class="anchor"  id="section9"></a>
# 
# At this stage, you might like to do some hyperparameter tuning with CatBoost before creating final submission.

# In[11]:


cat_params = dict(
    max_depth=6,           # maximum depth of each tree - try 2 to 7
    learning_rate=0.001,    # effect of each tree - try 0.0001 to 0.01
    iterations=4000,     # number of trees (that is, boosting rounds) - try 2000 to 8000
    min_data_in_leaf=3,    # minimum number of houses in a leaf - try 3 to 9
    colsample_bylevel=0.7,  # fraction of features (columns) per tree - try 0.4 to 1.0
    subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_lambda=0.1,        # L2 regularization (like Ridge) - try 0.0 to 1.0
    silent=True         
)

cat = CatBoostRegressor(**cat_params)
score_dataset(X, y, cat)


# Just tuning these by hand can give great results. However, you might like to try using one of scikit-learn's automatic hyperparameter tuners. Or you could explore more advanced tuning libraries like Optuna or scikit-optimize.
# 
# Here is how we can use Optuna with CatBoost:

# In[12]:


import optuna

def objective(trial):
    cat_params = dict(
        depth=trial.suggest_int("depth", 2, 7),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        iterations=trial.suggest_int("iterations", 2000, 8000),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 3, 9),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        subsample=trial.suggest_float("subsample", 0.2, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e-2, log=True),
        model_size_reg=trial.suggest_float("model_size_reg", 0.2, 1.0),
        silent=True
    )
    cat = CatBoostRegressor(**cat_params)
    return score_dataset(X, y, cat)

#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=20)
#cat_params = study.best_params

# Best params found by optuna
best_params = {'max_depth': 5, 'learning_rate': 0.005, 'iterations': 7000,
               'min_data_in_leaf': 5, 'colsample_bylevel': 0.9, 'silent': True,
               'subsample': 0.4, 'reg_lambda': 0.002}


# Uncomment if you'd like to use it, but be aware that it will take quite a while to run. After it's done, I found the best hyperparameters were:
# best_params = {'max_depth': 5, 'learning_rate': 0.005, 'iterations': 7000,
#                'min_data_in_leaf': 5, 'colsample_bylevel': 0.9, 'silent': True,
#                'subsample': 0.4, 'reg_lambda': 0.002, 'model_size_reg': 0.35}

# # **10) TRAIN MODEL AND SUBMISSION** <a class="anchor"  id="section10"></a>

# In[13]:


cat = CatBoostRegressor(**best_params)
cat.fit(X,np.log(y))

test_pred = cat.predict(df_test)
test_pred = np.exp(test_pred)

submission = pd.DataFrame({'Id': df_test.index,
                       'SalePrice': test_pred})
submission.to_csv('submission.csv', index=False)


# -------------------------------------------------------------------------------------
