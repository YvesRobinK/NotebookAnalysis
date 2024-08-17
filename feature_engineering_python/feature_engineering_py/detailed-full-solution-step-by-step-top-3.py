#!/usr/bin/env python
# coding: utf-8

# ### Detailed and Full Solution (Step by Step)
# 
# Hello kagglers ..
# 
# This notebook designed to be as detailed as possible solution for the Houses pricing problem, I tried to make it typical, clear, tidy and **beginner-friendly**.
# 
# If you find this notebook useful press the **UPVOTE** button, This helps me a lot ^-^.  
# I hope you find it helpful.
# 
# 
# 
# 
# 
# 
# 
# <center><img src="https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png" alt="drawing"/></center>
# 

# In[1]:


#=======================================================================================
# Importing the libaries:
#=======================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import math 
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000
#=======================================================================================


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b>1 ) Importing the data:</b>
#         </p>
# </div>

# In[2]:


def read_data():
    train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
    print("Train data imported successfully!!")
    print("-"*50)
    test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
    print("Test data imported successfully!!")
    return train_data , test_data


# In[3]:


train_data , test_data = read_data()


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 2 ) Discovering the data:</b>
#         </p>
# </div>

# In[4]:


train_data.head()


# In[5]:


test_data.head()


# Let's see our features:
# 

# In[6]:


print(train_data.columns.values)


# Wow It's like a mountain of features !!!, Let's see more details.

# In[7]:


train_data.info()
print("-"*50)
test_data.info()


# In[8]:


print("Train data shape = " , train_data.shape)
print("Test data shape = " , test_data.shape)


# ### Feature Types:
# - Categorical Features:
#     - **Nominal:** MSSubClass, MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation, BsmtExposure, Heating, CentralAir, Electrical, Functional, GarageType, PavedDrive, Fence, MiscFeature, SaleType, SaleCondition.
#     - **Ordinal:** OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, FireplaceQu, GarageFinish, GarageQual, GarageCond, PoolQC.
#    
#    
# - Numerical Features:
#     - **Continuous:** LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, GarageArea, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal.
#     - **Descrete:** YearBuilt, YearRemodAdd, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, Bedroom, Kitchen, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, MoSold, YrSold.
#  

# In[9]:


# Save the 'Id' column
test_ID = test_data['Id']

# Now drop the 'Id' column since it's unnecessary for  the prediction process.
train_data.drop("Id", axis = 1, inplace = True)
test_data.drop("Id", axis = 1, inplace = True)

print("\nThe train data size after dropping Id feature is : {} ".format(train_data.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test_data.shape))


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 3 ) Exploratory Data Analysis (EDA):</b>
#         </p>
# </div>
# 
# 

# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Sale Price Feature:</b>
#         </p>
# </div>
# 

# In[10]:


train_data['SalePrice'].describe()


# It seems that there are no weired or wrong values.

# In[11]:


plt.figure(figsize= (10 , 6))
g = sns.histplot(train_data.SalePrice , kde = True)


# In[12]:


print('SalePrice Skewness is = ' , train_data.SalePrice.skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())


# It's obviouse that SalePrice has Deviate from the normal distribution. we have right skewness (We will apply a **log transformation** in the next step)

# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Relationship with Numerial Variables:</b>
#         </p>
# </div>
# 

# In[13]:


plt.figure(figsize= (15 , 15))
sns.heatmap(train_data.corr(),cmap="Blues")


# In[14]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_features = []
for numerical_feature in train_data.columns:
    if train_data[numerical_feature].dtype in numeric_dtypes:
        numerical_features.append(numerical_feature)
        
plot , ax = plt.subplots(12,3 , figsize = (20 , 90))
for index in range(len(numerical_features)-1):
    sns.scatterplot(data = train_data , y = "SalePrice" , x = numerical_features[index] , alpha=0.4 , ax = ax[math.floor(index/3)][index%3])
plt.show()


# From the plots above I decided to:
# - Convert MsSubClass to categorical feature. (There is no clear distribution)
# - Convert MoSold to categorical feature. (There is no clear distribution)
# - Convert YrSold to categorical feature. (There is no clear distribution)
# - Fix Outliers in OverallQual, LotFrontage, GrLiveArea, GarageArea, LotArea, YearBuilt, TotalBsmtSF, 1stFlrSF. (Good Features)
# 

# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b>4 ) Data Processing:</b>
#         </p>
# </div>
# 
# 

# In[15]:


all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
sale_price = train_data["SalePrice"]
all_data.drop(columns = ["SalePrice"] , inplace = True)
all_data.shape


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Filling Missed Values:</b>
#         </p>
# </div>
# 

# In[16]:


def check_missed_values(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    return missing_data


# In[17]:


missing_data = check_missed_values(all_data)
missing_data.head(35)


# In[18]:


plt.figure(figsize = (16,5))
plt.xticks(rotation='90')
ax = sns.barplot(x = missing_data.index , y = missing_data["Missing Ratio"] )


# Note: Sometimes missing values mean that the house does not have the feature, So i will replace it by "None":

# In[19]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data.drop(columns = ["Utilities"] , inplace = True)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
print("No More Missed Values !!!")


# In[20]:


missing_data = check_missed_values(all_data)
missing_data.head()


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Numerical to Categorical Features:</b>
#         </p>
# </div>
# 
# 
# 

# In[21]:


# Some of the non-numeric predictors are stored as numbers; convert them into strings 
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Detect Outliers:</b>
#         </p>
# </div>
# 

# We will fix OverallQual, LotFrontage, GrLivArea, GarageArea, LotArea, YearBuilt, TotalBsmtSF, 1stFlrSF Outliers, The outlier points colored in **RED**:

# In[22]:


train_data = all_data[:len(train_data)]
train_data["SalePrice"] = sale_price
test_data = all_data[len(train_data):]


# **1 ) OverallQual:**

# In[23]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["OverallQual"] == 10) & (train_data["SalePrice"] <= 250000)
sns.scatterplot(data = train_data ,x = "OverallQual", y = "SalePrice", c = ["red" if is_outlier else "blue" for is_outlier in outliers] ,ax = ax[0])
train_data.drop(train_data[(train_data["OverallQual"] == 10) & (train_data["SalePrice"] <=250000)].index , inplace = True)
sns.scatterplot(data = train_data ,x = "OverallQual", y = "SalePrice", ax = ax[1] , c = ["blue"])
plt.show()


# **2 ) LotFrontage:**

# In[24]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["LotFrontage"] >250)
sns.scatterplot(data = train_data ,x = "LotFrontage", y = "SalePrice" , c = ["red" if is_outlier else "blue" for is_outlier in outliers] ,ax = ax[0])
train_data.drop(train_data[train_data["LotFrontage"] >250].index , inplace = True)
sns.scatterplot(data = train_data ,x = "LotFrontage", y = "SalePrice", ax = ax[1] , c = ["blue"])
plt.show()


# **3 ) GrLivArea:**

# In[25]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
sns.scatterplot(data = train_data ,x = "GrLivArea", y = "SalePrice" ,ax = ax[0])
sns.scatterplot(data = train_data ,x = "GrLivArea", y = "SalePrice", ax = ax[1] , c = ["blue"])
plt.show()


# seems that i deleted them before !!

# **4 ) GarageArea:**

# In[26]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["GarageArea"] > 1200 ) & (train_data["SalePrice"] <= 300000)
sns.scatterplot(data = train_data ,x = "GarageArea", y = "SalePrice" , c = ["red" if is_outlier else "blue" for is_outlier in outliers],ax = ax[0])
train_data.drop(train_data[(train_data["GarageArea"] > 1200 ) & (train_data["SalePrice"] <= 300000)].index , inplace = True)
sns.scatterplot(data = train_data ,x = "GarageArea", y = "SalePrice", ax = ax[1] , c = ["blue"])
plt.show()


# seems that i deleted them before !!

# **5 ) LotArea:**

# In[27]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["LotArea"] >= 100000)
sns.scatterplot(data = train_data ,x = "LotArea", y = "SalePrice" ,  c = ["red" if is_outlier else "blue" for is_outlier in outliers] ,ax = ax[0])
train_data.drop(train_data[ (train_data["LotArea"] >= 100000)].index , inplace = True)
sns.scatterplot(data = train_data ,x = "LotArea", y = "SalePrice", ax = ax[1])
plt.show()


# **6 ) Year Built:**

# In[28]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["YearBuilt"] < 1900) & (train_data["SalePrice"] >= 400000)
sns.scatterplot(data = train_data ,x = "YearBuilt", y = "SalePrice" ,c = ["red" if is_outlier else "blue" for is_outlier in outliers] ,  ax = ax[0])
train_data.drop(train_data[outliers].index , inplace = True)
sns.scatterplot(data = train_data ,x = "YearBuilt", y = "SalePrice", ax = ax[1])
plt.show()


# **7 ) TotalBsmtSF:**

# In[29]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
sns.scatterplot(data = train_data ,x = "TotalBsmtSF", y = "SalePrice" ,  ax = ax[0])
sns.scatterplot(data = train_data ,x = "TotalBsmtSF", y = "SalePrice", ax = ax[1])
plt.show()


# seems that i deleted them before !!

# **8 ) 1stFlrSF :**

# In[30]:


plot , ax = plt.subplots(1 , 2 , figsize = (12 , 5))
outliers = (train_data["1stFlrSF"] > 2700)
sns.scatterplot(data = train_data ,x = "1stFlrSF", y = "SalePrice" ,c = ["red" if is_outlier else "blue" for is_outlier in outliers] ,  ax = ax[0])
train_data.drop(train_data[outliers].index , inplace = True)
sns.scatterplot(data = train_data ,x = "1stFlrSF", y = "SalePrice", ax = ax[1])
plt.show()


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Fix Features Skewness:</b>
#         </p>
# </div>
# 

# In[31]:


# ==================================================================
# Sale Price
# ==================================================================

plot , ax = plt.subplots(2 , 2 , figsize = (12 , 8))
g = sns.histplot(sale_price , kde = True , ax = ax[0][0])
res = stats.probplot(sale_price, plot= ax[1][0])
sale_price = np.log1p(train_data["SalePrice"])
g = sns.histplot(sale_price , kde = True , ax = ax[0][1])
res = stats.probplot(sale_price, plot= ax[1][1])


# In[32]:


all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
all_data.drop(columns = ["SalePrice"] , inplace = True)
all_data.shape


# In[33]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[34]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Encode categorical features:</b>
#         </p>
# </div>
# 

# In[35]:


all_data = pd.get_dummies(all_data).reset_index(drop=True)
all_data.shape


# In[36]:


all_data.head()


# In[37]:


train_data = all_data[:len(train_data)]
test_data = all_data[len(train_data):]


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b>5 ) Modeling:</b>
#         </p>
# </div>
# 
# 

# In[38]:


# Models

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Misc
from sklearn.model_selection import GridSearchCV , learning_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


import warnings
warnings.filterwarnings(action="ignore")


# In[39]:


target = sale_price
train = train_data


# In[40]:


# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)
scores = {}


# In[41]:


# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=train , Y=target):
    rmse = np.sqrt(-cross_val_score(model,X, Y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# ### Decision Tree Regressor:

# In[42]:


decision_tree_model = DecisionTreeRegressor()
score = cv_rmse(decision_tree_model)
print("Decision Tree Model: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[43]:


clf = GridSearchCV(decision_tree_model , {
    "max_depth" : [6,7,8,9,10,11,12],
    "min_samples_split": [6,7,8,9,10],
    "min_samples_leaf" : [5,7,8,9,10]
},verbose = 1)
clf.fit(train , target)
clf.best_estimator_


# In[44]:


score = cv_rmse(clf.best_estimator_)
print("Decision Tree Model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['decision_tree'] = (score.mean(), score.std())


# ### 2) Random Forest Regressor:

# In[45]:


# Random Forest Regressor
random_forest_model = RandomForestRegressor(random_state=42)
score = cv_rmse(random_forest_model)
print("Random Forest Model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['random_forest'] = (score.mean(), score.std())


# ### 3) Gradient Boosting Regressor:

# In[46]:


# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(loss='huber',random_state=42)  
score = cv_rmse(gbr)
print("gradient_boosting: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gradient boosting'] = (score.mean(), score.std())


# ### 4 ) XGBoost Regressor:

# In[47]:


# XGBoost Regressor
xgboost = XGBRegressor(objective='reg:squarederror',random_state=42)
score = cv_rmse(xgboost)
print("xgboost_model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Xgboost'] = (score.mean(), score.std())


# ### 5 ) Ridge Regressor:

# In[48]:


# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
score = cv_rmse(ridge)
print("Ridge Regressor: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge_model'] = (score.mean(), score.std())


# ### 6 ) Support Vector Machine:

# In[49]:


# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
score = cv_rmse(svr)
print("Support Vector Machine: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Support Vector Machine'] = (score.mean(), score.std())


# ### 7 ) Light Gradient Boosting Regressor:

# In[50]:


# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', verbose=1,random_state=42)
score = cv_rmse(lightgbm)
print("Light Gbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Lgbm'] = (score.mean(), score.std())


# ### Train on Full Data:

# In[51]:


decision_tree_model_full_data = decision_tree_model.fit(train , target)
random_forest_model_full_data = random_forest_model.fit(train , target)
gbr_full_data = gbr.fit(train , target)
xgboost_full_data = xgboost.fit(train , target)
ridge_full_data = ridge.fit(train , target)
svr_full_data = svr.fit(train , target)
lightgbm_full_data = lightgbm.fit(train , target)


# ### 8 ) Blended Model:

# In[52]:


# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * random_forest_model_full_data.predict(X)) + \
            (0.2 * gbr_full_data.predict(X)) + \
            (0.1 * xgboost_full_data.predict(X)) + \
            (0.2 * ridge_full_data.predict(X)) + \
            (0.1 * lightgbm_full_data.predict(X)) + \
            (0.3 * svr_full_data.predict(X)))


# In[53]:


# Get final precitions from the blended model
blended_score = rmsle(target, blended_predictions(train))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# In[54]:


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(16, 8))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# In[55]:


# Read in sample_submission dataframe
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape


# In[56]:


# Append predictions from blended models
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(test_data)))


# In[57]:


# Fix outleir predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)


# In[58]:


# Scale predictions
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression2.csv", index=False)


# #### My Other Useful Notebooks:
# - [Titanic - Machine Learning From Disaster](https://www.kaggle.com/code/odaymourad/detailed-and-typical-solution-ensemble-modeling).
# - [Spaceship Titanic](https://www.kaggle.com/code/odaymourad/detailed-and-full-solution-step-by-step-80-score).
# - [Feature Selection and Data Engineering](https://www.kaggle.com/code/odaymourad/feature-selection-data-engineering-step-by-step).
# - [Learn Overfitting and Underfitting](https://www.kaggle.com/code/odaymourad/learn-overfitting-and-underfitting-79-4-score).
# - [Random Forest Algorithm](https://www.kaggle.com/code/odaymourad/random-forest-model-clearly-explained).
# - [NLP with Disaster tweets](https://www.kaggle.com/code/odaymourad/detailed-and-full-solution-78-4-score).
