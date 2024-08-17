#!/usr/bin/env python
# coding: utf-8

# # <center>**House Prices - Advanced Regression Techniques**</center>

# # **Introduction**
# ### **Objective:** 
# The objective of the project is to perform advance regression techniques to predict the house price in Boston.
# ### **Data Description:**
# - train.csv - the training set
# - test.csv - the test set
# - data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
# - sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms
# 
# ### **Table of Content:**
# 1. Fetch Dataset
# 2. Install & Import Libraries
# 3. Load Datasets
# 4. Exploratory Data Analysis
# 5. Feature Engineering
# 6. Model Development
# 7. Find Prediction
# 
# 
# 
# 

# # **1. Fetch datasets from kaggle**

# # **2. Install & Import Libraries**

# In[1]:


# use to visualize missing value
get_ipython().system('pip install missingno')


# In[2]:


# use for hyper parameter tuning
get_ipython().system('pip install optuna')


# In[3]:


# use to choose best algorithms for our dataset 
get_ipython().system('pip install lazypredict==0.2.7')


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)

from scipy import stats
from scipy.stats import norm, skew # for some statistics
import warnings # to ignore warning
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import lazypredict
from lazypredict.Supervised import LazyRegressor
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, RidgeCV

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib

print("Library Imported!!")


# # **3. Load Datasets**

# In[5]:


# load train and test dataset
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

# combined train and test datasets
combined_df = pd.concat([train_df,test_df],axis=0)


# # **4. Exploratory Data Analysis**
# ### 4.1. Train Data Exploration
# 
# For both train and test dataset, We'll explore following things
# 
# - First 5 rows
# - Data shape
# - Data information
# - Data types
# - Null value

# ### 4.1.1. First 5 records

# In[6]:


train_df.head()


# ### 4.1.2. Data Shape - Train Data

# In[7]:


train_df.shape


# ### 4.1.3. Data Information - Train Data

# In[8]:


train_df.info()


# ### 4.1.4. Data Type - Train Data

# In[9]:


train_dtype = train_df.dtypes
train_dtype.value_counts()


# ### 4.1.5. Null Value - Train Data

# In[10]:


train_df.isnull().sum().sort_values(ascending = False).head(20)


# ### 4.1.6. Visualize missing value using **Misingno** - Train Data

# In[11]:


msno.matrix(train_df)


# - The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another.

# In[12]:


msno.heatmap(train_df)


# ### 4.2. Test Data Exploration
# 
# ### 4.2.1. First 5 rows - Test Data

# In[13]:


test_df.head()


# ### 4.2.2. Data Type - Test Data

# In[14]:


test_dtype = test_df.dtypes
test_dtype.value_counts()


# ### 4.2.3. Data Information - Test Data

# In[15]:


test_df.info()


# ### 4.2.4. Data Shape - Test Data

# In[16]:


test_df.shape


# ### 4.2.5. Null Data - Test Data

# In[17]:


test_df.isnull().sum().sort_values(ascending = False).head(20)


# ### 4.2.6. Visualize missing value using **Misingno** - Test Data

# In[18]:


msno.matrix(test_df)


# In[19]:


msno.heatmap(test_df)


# ### 4.3. Train & Test Data Comparison
# 
# Here we'll compare below things between train and test dataset.
# - Data Type
# - Null values
# - Data Distribution

# ### 4.3.1. Data Type Comparison

# In[20]:


# as 'SalePrice' Column is not available in test dataset. So we'll delete it.
trn_dtype = train_dtype.drop('SalePrice')
trn_dtype.compare(test_dtype)


# In[21]:


test_df["TotalBsmtSF"].head()


# - Here We can see some columns have inconsistent data types i.e int64 & float64. It's not a problem.

# ### 4.3.2. Null Value Comparison

# In[22]:


null_train = train_df.isnull().sum()
null_test = test_df.isnull().sum()
null_train = null_train.drop('SalePrice')
null_comp_df = null_train.compare(null_test).sort_values(['self'],ascending = [False])
null_comp_df  


# - Here we can see that columns like **"Alley", "Fence", "LotFrontage", "FireplaceQu"**  have maximum number of null value. So we will consider to drop these columns.

# ### 4.3.3. Distribution Comparison

# In[23]:


numerical_features = [col for col in train_df.columns if train_df[col].dtypes != 'O']
discrete_features = [col for col in numerical_features if len(train_df[col].unique()) < 25 and col not in ['Id']]
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+['Id']]
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'O']

print("Total Number of Numerical Columns : ",len(numerical_features))
print("Number of discrete features : ",len(discrete_features))
print("No of continuous features are : ", len(continuous_features))
print("Number of discrete features : ",len(categorical_features))


# In[24]:


combined_df["Label"] = "test"
combined_df["Label"][:1460] = "train"


# ### 4.3.3.1. Distribution Comparison - Discrete

# In[25]:


f, axes = plt.subplots(3,6 , figsize=(30, 10), sharex=False)
for i, feature in enumerate(discrete_features):
    sns.histplot(data=combined_df, x = feature, hue="Label",ax=axes[i%3, i//3]) 


# Above distribution shows that:
# - Some features can be reclassified as 'Categorical', such as **'MSSubClass'**.
# - Some features are dominated by 0/null **(eg:PoolArea, LowQualFinSF, 3SsnPorch, MiscVal )**, thus we can consider to drop.

# ### 4.3.3.2. Distribution Comparison - Continuous

# In[26]:


f, axes = plt.subplots(4,6 , figsize=(30, 15), sharex=False)
for i, feature in enumerate(continuous_features):
    sns.histplot(data=combined_df, x = feature, hue="Label",ax=axes[i%4, i//4]) 


# Above distribution shows that:
# - The distribution of train and test data are similar for most continous features.

# ### 4.3.3.3. Linearity Check
# Here we'll see the linearity between all features and the target variable.

# In[27]:


f, axes = plt.subplots(7,6 , figsize=(30, 30), sharex=False)
for i, feature in enumerate(numerical_features):
    sns.scatterplot(data=combined_df, x = feature, y= "SalePrice",ax=axes[i%7, i//7])


# We notice that some features are not linear towards target feature.
# 
# - 'SalePrice' VS.'BsmtUnfSF',
# - 'SalePrice' VS.'TotalBsmtSF',
# - 'SalePrice' VS.'GarageArea',
# - 'SalePrice' VS.'LotArea',
# - 'SalePrice' VS.'LotFrontage',
# - 'SalePrice' VS.'GrLivArea',
# - 'SalePrice' VS.'1stFlrSF',
# 
# 

# ### 4.3.3.4. Distribution Comparison - Categorical 

# In[28]:


f, axes = plt.subplots(7,7 , figsize=(30, 30), sharex=False)
for i, feature in enumerate(categorical_features):
    sns.countplot(data = combined_df, x = feature, hue="Label",ax=axes[i%7, i//7])


# Above distribution shows that:
# 
# - The distribution of train and test data are similar for most categorical features.
# - Some features have dominant items, we can combine some minor items into a group otherwise we can drop these columns.
# - Ex: **'RoofMatl','Street','Condition2','Utilities','Heating'** (These columns should be dropped)
# - Ex: 'Fa' & 'Po' in 'HeatingQC', 'FireplaceQu', 'GarageQual' and 'GarageCond'
# 
# Now let's conform that the items we want to combine has similar prices(SalePrices value).
# 

# In[29]:


f, axes = plt.subplots(7,7 , figsize=(30, 30), sharex=False)
for i, feature in enumerate(categorical_features):
    sort_list = sorted(combined_df.groupby(feature)['SalePrice'].median().items(), key= lambda x:x[1], reverse = True)
    order_list = [x[0] for x in sort_list ]
    sns.boxplot(data = combined_df, x = feature, y = 'SalePrice', order=order_list, ax=axes[i%7, i//7])
plt.show()


# Here, we could see that sale prices for 'Fa' & 'Po' in 'HeatingQC', 'FireplaceQu', 'GarageQual' and 'GarageCond' are similar, so we can combine these items.

# ### 4.4. Find Suitable value for missing values - Numerical 
# 
# ### 4.4.1. Fill Mean Value

# In[30]:


# check the normal distribution of columns having null values by filling with the mean value
null_features_numerical = [col for col in combined_df.columns if combined_df[col].isnull().sum() > 0 and col not in categorical_features]
plt.figure(figsize=(30,20))
sns.set()

warnings.simplefilter("ignore")
for i,var in enumerate(null_features_numerical):
  plt.subplot(4,3,i+1)
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':3,'color':'red'},label="original")
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':2,'color':'yellow'},label="mean")


# ### 4.4.2. Fill Median Value

# In[31]:


plt.figure(figsize=(30,20))
sns.set()
warnings.simplefilter("ignore")
for i,var in enumerate(null_features_numerical):
  plt.subplot(4,3,i+1)
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':3,'color':'red'},label="original")
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':2,'color':'yellow'},label="median")


# - From the above visualization we saw that mean and median value both maintain the same destribution. So we can choose one of them to fill the missing values.

# ### 4.4.3 Find Suitable value for missing values - Categorical 

# In[32]:


# ---------------- do -----------------------


# In[33]:


# ---------------- do -----------------------


# ### 4.5. Temporal Variable Analysis

# In[34]:


# variables which contain year information
year_feature = [col for col in combined_df.columns if 'Yr' in col or 'Year' in col]
year_feature


# Check is there any relation betwwn **"Year Sold"** and **"Sales price"**

# In[35]:


combined_df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('House Price')
plt.title('House price vs YearSold')


# Here we'll see how the temporal variables(Year features) affect to House Price

# In[36]:


for fet in year_feature:
  if fet != 'YrSold':
    hs = combined_df.copy()
    hs[fet] = hs['YrSold'] - hs[fet]
    plt.scatter(hs[fet],hs['SalePrice'])
    plt.xlabel(fet)
    plt.ylabel('SalePrice')
    plt.show()


# ### 4.6. Data Correlation

# In[37]:


training_corr = train_df.corr(method='spearman')
plt.figure(figsize=(20,10))
sns.heatmap(training_corr, cmap="YlGnBu", linewidths=.5)


# #**5. Feature Engineering**

# ### 5.1. Drop Columns
# Here we'll drop columns like
# - ID
# - Column having more missing value
# - Column dominated by 0/null or single value 
# 

# In[38]:


drop_columns = ["Id", "Alley", "Fence", "LotFrontage", "FireplaceQu", "PoolArea", "LowQualFinSF", "3SsnPorch", "MiscVal", 'RoofMatl','Street','Condition2','Utilities','Heating','Label']
#  Drop columns
print("Number of columns before dropping : ",len(combined_df.columns))
print("Number of dropping columns : ",len(drop_columns))
combined_df.drop(columns=drop_columns, inplace=True, errors='ignore')
print("Number of columns after dropping : ",len(combined_df.columns))


# ### 5.2. Temporal Variable Change

# In[39]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

    combined_df[feature]=combined_df['YrSold']-combined_df[feature]

combined_df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# ### 5.3.1. Fill Missing Values - Numerical Feature

# In[40]:


for col in null_features_numerical:
  if col not in drop_columns:    
    # combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
    combined_df[col] = combined_df[col].fillna(0.0)


# ### 5.3.2. Fill Missing Values - Categorical Feature

# In[41]:


null_features_categorical = [col for col in combined_df.columns if combined_df[col].isnull().sum() > 0 and col in categorical_features]
cat_feature_mode = ["SaleType", "Exterior1st", "Exterior2nd", "KitchenQual", "Electrical", "Functional"]

for col in null_features_categorical:
  if col != 'MSZoning' and col not in cat_feature_mode:
    combined_df[col] = combined_df[col].fillna('NA')
  else:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])


# ### 5.4. Convert Numerical feature to Categorical

# In[42]:


# Convert "numerical" feature to categorical
convert_list = ['MSSubClass']
for col in convert_list:
  combined_df[col] = combined_df[col].astype('str')


# ### 5.5. Apply PowerTransformer to columns
# - We saw in distribution of continuous features that some features are not linear towards target feature. So we need to transform this. 
# - Lets check the skewness of all distributions
# 

# In[43]:


numeric_feats = combined_df.dtypes[combined_df.dtypes != 'object'].index
# get the features except object types

# check the skew of all numerical features
skewed_feats = combined_df[numeric_feats].apply(lambda x : skew(x.dropna())).sort_values(ascending = False)
print('\n Skew in numberical features: \n')
skewness_df = pd.DataFrame({'Skew' : skewed_feats})
print(skewness_df.head(10))


# In[44]:


# Apply PowerTransformer to columns
log_list = ['BsmtUnfSF', 'LotArea', '1stFlrSF', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
# log_list = ['LotArea', 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch', 'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF']
# log_list = skewness_df[abs(skewness_df) > 1].dropna().index


for col in log_list:
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    combined_df[[col]] = power.fit_transform(combined_df[[col]]) # fit with combined_data to avoid overfitting with training data?

print('Number of skewed numerical features got transform : ', len(log_list))


# ### 5.6. Regroup Features

# In[45]:


# Regroup features
regroup_dict = {
#     'LotConfig': ['FR2','FR3'],
#     'LandSlope':['Mod','Sev'],
#     'BldgType':['2FmCon','Duplex'],
#     'RoofStyle':['Mansard','Flat','Gambrel'],
#     'Electrical':['FuseF','FuseP','FuseA','Mix'],
#     'SaleCondition':['Abnorml','AdjLand','Alloca','Family'],
#     'BsmtExposure':['Min','Av'],
#     'Functional':['Min1','Maj1','Min2','Mod','Maj2','Sev'],
#     'LotShape':['IR2','IR3'],
    'HeatingQC':['Fa','Po'],
    # 'FireplaceQu':['Fa','Po'],
    'GarageQual':['Fa','Po'],
    'GarageCond':['Fa','Po'],
}
 

for col, regroup_value in regroup_dict.items():
    mask = combined_df[col].isin(regroup_value)
    combined_df[col][mask] = 'Other'


# ### 5.7. Encoding Categorical Features
# 
# ### 5.7.1. LabelEncoder

# In[46]:


# print('Shape combined_df before LabelEncoder : {}'.format(combined_df.shape))

# labelencoder_cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
#         'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
#         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
#         'YrSold', 'MoSold']

# # process columns, apply LabelEncoder to categorical features 
# for c in labelencoder_cols:
#   if c in combined_df.columns:
#       lbl = LabelEncoder()
#       lbl.fit(list(combined_df[c].values))
#       combined_df[c] = lbl.transform(list(combined_df[c].values))
    
# # shape
# print('Shape combined_df after LabelEncoder : {}'.format(combined_df.shape))


# ### 5.7.2. Get-Dummies

# In[47]:


# Generate one-hot dummy columns
combined_df = pd.get_dummies(combined_df).reset_index(drop=True)


# In[48]:


new_train_data = combined_df.iloc[:len(train_df), :]
new_test_data = combined_df.iloc[len(train_df):, :]
X_train = new_train_data.drop('SalePrice', axis=1)
y_train = np.log1p(new_train_data['SalePrice'].values.ravel())
X_test = new_test_data.drop('SalePrice', axis=1)


# In[49]:


pre_precessing_pipeline = make_pipeline(RobustScaler(), 
                                        # VarianceThreshold(0.001),
                                       )

X_train = pre_precessing_pipeline.fit_transform(X_train)
X_test = pre_precessing_pipeline.transform(X_test)

print(X_train.shape)
print(X_test.shape)


# # **6. Model Development**

# ### 6.1. Find best algorithms using LazyPredict

# In[50]:


x_train1,x_test1,y_train1,y_test1=train_test_split(X_train,y_train,test_size=0.25)

reg= LazyRegressor(verbose=0,ignore_warnings=True,custom_metric=None)
train,test=reg.fit(x_train1,x_test1,y_train1,y_test1)
test


# Here we can see which algorithms give best accuracy in less time.
# - **Gradient Boosting Regressor**
# - **XGBRegressor**
# - **LGBMRegressor**
# - **Lasso**
# - **Ridge**

# ### 6.2. Hyperparameter Tuning using Optuna

# In[51]:


RANDOM_SEED = 23

# 10-fold CV
kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)


# In[52]:


def tune(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params


# ### 6.3. Ridge Regression

# In[53]:


def ridge_objective(trial):

    _alpha = trial.suggest_float("alpha", 0.1, 20)

    ridge = Ridge(alpha=_alpha, random_state=RANDOM_SEED)

    score = cross_val_score(
        ridge,X_train,y_train, cv=kfolds, scoring="neg_root_mean_squared_error"
    ).mean()
    return score



# Best score: -0.13586760243668033 
ridge_params = {'alpha': 19.997759851201025}


# In[54]:


ridge = Ridge(**ridge_params, random_state=RANDOM_SEED)
ridge.fit(X_train,y_train)


# ### 6.4. Lasso Regression

# In[55]:


def lasso_objective(trial):

    _alpha = trial.suggest_float("alpha", 0.0001, 1)

    lasso = Lasso(alpha=_alpha, random_state=RANDOM_SEED)

    score = cross_val_score(
        lasso,X_train,y_train, cv=kfolds, scoring="neg_root_mean_squared_error"
    ).mean()
    return score




# Best score: -0.13319435700230317 
lasso_params = {'alpha': 0.0006224224345371836}


# In[56]:


lasso = Lasso(**lasso_params, random_state=RANDOM_SEED)
lasso.fit(X_train,y_train)


# ### 6.5. Gradient Boosting Regressor

# In[57]:


def gbr_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 50, 2000)
    _learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    _max_depth = trial.suggest_int("max_depth", 1, 20)
    _min_samp_split = trial.suggest_int("min_samples_split", 2, 20)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
    _max_features = trial.suggest_int("max_features", 10, 50)

    gbr = GradientBoostingRegressor(
        n_estimators=_n_estimators,
        learning_rate=_learning_rate,
        max_depth=_max_depth, 
        max_features=_max_features,
        min_samples_leaf=_min_samples_leaf,
        min_samples_split=_min_samp_split,
        
        random_state=RANDOM_SEED,
    )

    score = cross_val_score(
        gbr, X_train,y_train, cv=kfolds, scoring="neg_root_mean_squared_error"
    ).mean()
    return score





# Best score: -0.12736574760627803 
gbr_params = {'n_estimators': 1831, 'learning_rate': 0.01325036780847096, 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 2, 'max_features': 29}


# In[58]:


gbr = GradientBoostingRegressor(random_state=RANDOM_SEED, **gbr_params)
gbr.fit(X_train,y_train)


# ### 6.6. XGBRegressor 

# In[59]:


def xgb_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 50, 2000)
    _max_depth = trial.suggest_int("max_depth", 1, 20)
    _learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    _gamma = trial.suggest_float("gamma", 0.01, 1)
    _min_child_weight = trial.suggest_float("min_child_weight", 0.1, 10)
    _subsample = trial.suggest_float('subsample', 0.01, 1)
    _reg_alpha = trial.suggest_float('reg_alpha', 0.01, 10)
    _reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10)

    
    xgbr = xgb.XGBRegressor(
        n_estimators=_n_estimators,
        max_depth=_max_depth, 
        learning_rate=_learning_rate,
        gamma=_gamma,
        min_child_weight=_min_child_weight,
        subsample=_subsample,
        reg_alpha=_reg_alpha,
        reg_lambda=_reg_lambda,
        random_state=RANDOM_SEED,
    )
    

    score = cross_val_score(
        xgbr, X_train,y_train, cv=kfolds, scoring="neg_root_mean_squared_error"
    ).mean()
    return score





xgb_params = {'n_estimators': 847, 'max_depth': 7, 'learning_rate': 0.07412279963454066, 'gamma': 0.01048697764796929, 'min_child_weight': 5.861571837417184, 'subsample': 0.7719639391828977, 'reg_alpha': 2.231609305115769, 'reg_lambda': 3.428674606766844}
#  . Best is trial 34 with value: -0.13193488071216425.


# In[60]:


xgbr = xgb.XGBRegressor(random_state=RANDOM_SEED, **xgb_params)
xgbr.fit(X_train,y_train)


# ### 6.7. LGBMRegressor

# In[61]:


import lightgbm as lgb

def lgb_objective(trial):
    _num_leaves = trial.suggest_int("num_leaves", 50, 100)
    _max_depth = trial.suggest_int("max_depth", 1, 20)
    _learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    _n_estimators = trial.suggest_int("n_estimators", 50, 2000)
    _min_child_weight = trial.suggest_float("min_child_weight", 0.1, 10)
    _reg_alpha = trial.suggest_float('reg_alpha', 0.01, 10)
    _reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10)
    _subsample = trial.suggest_float('subsample', 0.01, 1)


    
    lgbr = lgb.LGBMRegressor(objective='regression',
                             num_leaves=_num_leaves,
                             max_depth=_max_depth,
                             learning_rate=_learning_rate,
                             n_estimators=_n_estimators,
                             min_child_weight=_min_child_weight,
                             subsample=_subsample,
                             reg_alpha=_reg_alpha,
                             reg_lambda=_reg_lambda,
                             random_state=RANDOM_SEED,
    )
    

    score = cross_val_score(
        lgbr, X_train,y_train, cv=kfolds, scoring="neg_root_mean_squared_error"
    ).mean()
    return score

# Best score: -0.12497294451988177 
# lgb_params = tune(lgb_objective)
lgb_params = {'num_leaves': 81, 'max_depth': 2, 'learning_rate': 0.05943111506493225, 'n_estimators': 1668, 'min_child_weight': 4.6721695700874015, 'reg_alpha': 0.33400189583009254, 'reg_lambda': 1.4457484337302167, 'subsample': 0.42380175866399206}


# Best score: -0.012014396001532427 
# lgb_params = {'num_leaves': 84, 'max_depth': 15, 'learning_rate': 0.3765620685374334, 'n_estimators': 1363, 'min_child_weight': 2.933698765978165, 'reg_alpha': 0.025700686948561362, 'reg_lambda': 9.02451400894547, 'subsample': 0.9947557511368282}


# In[62]:


lgbr = lgb.LGBMRegressor(objective='regression', random_state=RANDOM_SEED, **lgb_params)
lgbr.fit(X_train,y_train)


# ### 6.8. StackingRegressor

# In[63]:


# stack models
stack = StackingRegressor(
    estimators=[
        ('ridge', ridge),
        ('lasso', lasso),
        ('gradientboostingregressor', gbr),
        ('xgb', xgbr),
        ('lgb', lgbr),
        # ('svr', svr), # Not using this for now as its score is significantly worse than the others
    ],
    cv=kfolds)
stack.fit(X_train,y_train)


# ### 6.9. Save the Model

# In[64]:


# # joblib.dump(stack, "prediction_model.pkl")
# model=joblib.load("prediction_model.pkl")
# model


# In[65]:


# def cv_rmse(model):
#     rmse = -cross_val_score(model, X_train,y_train,
#                             scoring="neg_root_mean_squared_error",
#                             cv=kfolds)
#     return (rmse)


# In[66]:


# def compare_models():
#     models = {
#         'Ridge': ridge,
#         'Lasso': lasso,
#         'Gradient Boosting': gbr,
#         'XGBoost': xgbr,
#         'LightGBM': lgbr,
#         'Stacking': stack, 
#         # 'SVR': svr, # TODO: Investigate why SVR got such a bad result
#     }

#     scores = pd.DataFrame(columns=['score', 'model'])

#     for name, model in models.items():
#         score = cv_rmse(model)
#         print("{:s} score: {:.4f} ({:.4f})\n".format(name, score.mean(), score.std()))
#         df = pd.Series(score, name='score').to_frame()
#         df['model'] = name
#         scores = scores.append(df)

#     plt.figure(figsize=(20,10))
#     sns.boxplot(data = scores, x = 'model', y = 'score')
#     plt.show()
    
# compare_models()


# # **7. Find Prediction**

# In[67]:


print('Predict submission')
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.expm1(stack.predict(X_test))

submission.to_csv('submission_file.csv', index=False)


# ## If you find this notebook useful,don't forget to **"UPVOTE"**👏
# 
# ## Follow me on [github](https://github.com/sidharth178),i used to upload good data science projects.

# In[ ]:





# In[ ]:




