#!/usr/bin/env python
# coding: utf-8

# # Import essential libraries

# In[1]:


import numpy as np
import pandas as pd


# # Load the data

# In[2]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')  #traning data
y = train["SalePrice"]  #target
X = train.drop(labels=["SalePrice"], axis=1)  #traning set
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')  #testing data


# # View the data

# Testing set

# In[3]:


X


# Target

# In[4]:


y


# testing set

# In[5]:


test


# # Devide data by types
# > 

# Categorical features

# In[6]:


cat_features = ["MSZoning", "MSSubClass", "Street", "Alley",
                "LotShape", "LandContour", "Utilities", "LotConfig",
                "LandSlope", "Neighborhood","Condition1", 
                "Condition2", "BldgType", "HouseStyle", 
                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",
                "Electrical", "Functional", "FireplaceQu", "GarageType",
                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
                "SaleType", "SaleCondition", "PoolQC", "Fence", "MiscFeature",
                "KitchenQual"] 


# Ordinal features

# In[7]:


ordinal = ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", 
 "GarageYrBlt", "GarageCars", "MoSold", "YrSold"]


# Absolute features

# In[8]:


absolute = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 
 "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
 "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", 
 "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", 
 "Fireplaces", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
 "3SsnPorch", "ScreenPorch", "PoolArea"]


# # Check null values for each datatype

# In[9]:


X[absolute].isnull().sum()


# In[10]:


X[cat_features].isnull().sum()


# In[11]:


X[ordinal].isnull().sum()


# # Categorical values

# To deal with categorical values I will be using CatBoost Encoder <br>
# https://contrib.scikit-learn.org/category_encoders/catboost.html

# Install the encoder

# In[12]:


get_ipython().system('pip install category_encoders')


# In[13]:


import category_encoders as ce
cb_enc = ce.CatBoostEncoder(cols=cat_features, verbose=1)

X = X.drop(labels=cat_features, axis=1).join(cb_enc.fit_transform(X[cat_features], y))  #transform cat_feature columns
test = test.drop(labels=cat_features, axis=1).join(cb_enc.transform(test[cat_features]))
print(len(test))


# In[14]:


X


# # Imputations

# Absolute values have a lot of Nan values thats why they need to be replaced with some values. To impute the values I'll use the K Nearest Heighbors Imputer.

# In[15]:


from sklearn.impute import KNNImputer, SimpleImputer
knn_imp = KNNImputer()
simp_imp = SimpleImputer(strategy="median")

X = pd.DataFrame(simp_imp.fit_transform(X), columns=X.columns)
test = pd.DataFrame(simp_imp.fit_transform(test), columns=test.columns)


# In[16]:


X[absolute].isnull().sum()


# In[17]:


X[ordinal].isnull().sum()


# # Scaling

# In[18]:


# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# columns = X.columns
# X = pd.DataFrame(ss.fit_transform(X), columns=columns)
# test = pd.DataFrame(ss.transform(test), columns=columns)


# # Select best features

# Not all features make the model more accurate. Some may even decrease it. To prevent our model from such a situation we will select best features for CatBoostRegressor

# In[19]:


print("Quantity of features:", f"train: {len(X.columns)}", f"test: {len(test.columns)}", sep="\n")


# CatBoost features

# In[20]:


# from sklearn.feature_selection import SelectFromModel

# cbr = CatBoostRegressor().fit(X, y)
# model = SelectFromModel(cbr, prefit=True)

# X_transformed = model.transform(X)

# best_features = pd.DataFrame(
#     model.inverse_transform(X_transformed), 
#     index=X.index,
#     columns=X.columns
# )

# best_features = best_features.columns[best_features.var() != 0]


# In[21]:


# # Get the valid dataset with the selected features.
# X = X[best_features]
# test = test[best_features]


# In[22]:


X


# In[23]:


print("Quantity of features:", f"X: {len(X.columns)}", f"test: {len(test.columns)}", sep="\n")


# # Model

# ![](http://camo.githubusercontent.com/1ba204e6a09e6f13c919dcf961fe5a9a7f2d6e30/687474703a2f2f73746f726167652e6d64732e79616e6465782e6e65742f6765742d646576746f6f6c732d6f70656e736f757263652f3235303835342f636174626f6f73742d6c6f676f2e706e67)

# In[24]:


from vecstack import StackingTransformer
from sklearn.pipeline import Pipeline


# In[25]:


from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# L1 Stack

# In[26]:


# 1st level estimator parameters
d1 = 3
d2 = 7
d3 = 11
d4 = 13
rs = 42
              
# L1 StackingTransformer
L1_stack = StackingTransformer(
    estimators=[
        (f'CatBoost depth={d1}', CatBoostRegressor(verbose=False, eval_metric="RMSE", random_seed=rs, depth=d1)),
        (f'CatBoost depth={d2}', CatBoostRegressor(verbose=False, eval_metric="RMSE", random_seed=rs, depth=d2)),
        (f'CatBoost depth={d3}', CatBoostRegressor(verbose=False, eval_metric="RMSE", random_seed=rs, depth=d3)),
        
        (f'XGBoost depth={d1}', XGBRegressor(seed=rs, max_depth=d1)),
        (f'XGBoost depth={d2}', XGBRegressor(seed=rs, max_depth=d2)),
        (f'XGBoost depth={d3}', XGBRegressor(seed=rs, max_depth=d3)),
        
        (f'RandomForest depth={d1}', RandomForestRegressor(random_state=rs, max_depth=d1)),
        (f'RandomForest depth={d2}', RandomForestRegressor(random_state=rs, max_depth=d2)),
        (f'RandomForest depth={d3}', RandomForestRegressor(random_state=rs, max_depth=d3))
    ], 
    regression=True,
    verbose=2, 
    random_state=42, 
    n_folds=5, 
    variant="A"
)


# In[27]:


# 2n level estimator
from sklearn.linear_model import LinearRegression

final_estimator = LinearRegression()


# Stacking Pipeline

# In[28]:


model = Pipeline(
    steps=[
        ("L1_stack", L1_stack),
        ("Final_estimator", final_estimator)
    ]
)


# Fit the pipeline

# In[29]:


model.fit(X, y)


# Make prediction

# In[30]:


y_pred = model.predict(test)


# # Submit

# In[31]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission["SalePrice"] = y_pred
submission.to_csv('submission.csv', index=False)

