#!/usr/bin/env python
# coding: utf-8

# I am new to machine learning, feature engineering let's do some coding together and learn!
# We are going the implement regression model to predict the house prices.
# 
# Contents of this notebook:
#  <a id="toc"></a>
# 
# 1. [Implementing necessary libraries](#1)
# 
# 2. [Load the data](#2)
# 
# 3. [Feature engineering](#3)
# 
# 4. [Models](#4)
#     
# 5. [Submission](#5)
# 
# 
# # 1. Importing necessary libraries  <a id="1"></a>
# 

# In[1]:


get_ipython().system('pip install feature-engine')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce



import warnings
warnings.filterwarnings('ignore')


# # 2. Load the data  <a id="2"></a>
# 
# 
# Load the train data as `df_train` and test data as `df_test`

# In[3]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


df_train.head()


# In[5]:


df_train.shape


# In[6]:


df_train.columns


# In[7]:


df_train.info()


# We have 1460 rows and 81 columns in our train data we will predict `SalePrice` of the houses. We will look the relations between `SalePrice` feature and other features. Since there is so much feature we need to analyze the important features.

# In[8]:


df_test.shape


# In[9]:


test_ID = df_test['Id']


# # 3. Feature engineering  <a id="3"></a>
# 

# In[10]:


# make list of variables types
# numerical: discrete and continuous
discrete = [
    var for var in df_train.columns if df_train[var].dtype != 'O' and var != 'survived'
    and df_train[var].nunique() < 10
]
continuous = [
    var for var in df_train.columns
    if df_train[var].dtype != 'O' and var != 'survived' and var not in discrete
]

# categorical
categorical = [var for var in df_train.columns if df_train[var].dtype == 'O']

print('There are {} discrete variables'.format(len(discrete)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(categorical)))


# ## 3.1 `SalePrice`

# In[11]:


df_train['SalePrice'].describe()


# In[12]:


df_train['SalePrice'].isna().sum()


# There is no missing values in `SalePrice` that is great!

# In[13]:


plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=df_train['SalePrice'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('SalePrice Histogram')
plt.show()


# Let's look at the heatmap!

# In[14]:


corr = df_train.corr()
#correlation matrix
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# `GrLivArea`,`TotalBsmtSF`,`OverallQual` , `GarageCars` etc. seems to have high correlation with `SalePrice`.

# In[15]:


# plot between GrLivArea and SalePrice
plt.figure(figsize=[10,8])
plt.scatter(df_train['SalePrice'], df_train['GrLivArea'])
plt.title('SalePrice and GrLivArea')
plt.show()


# There is a positive correlation between the variables.

# In[16]:


# plot between TotalBsmtSF and SalePrice
plt.figure(figsize=[10,8])
plt.scatter(df_train['SalePrice'], df_train['TotalBsmtSF'])
plt.title('SalePrice and TotalBsmtSF')
plt.show()


# There is also a positive correlation between the variables.

# In[17]:


plt.figure(figsize=[10,8])
ax = sns.boxplot(x='OverallQual', y='SalePrice', data=df_train)


# In[18]:


plt.figure(figsize=[16,10])
ax = sns.boxplot(x='YearBuilt', y='SalePrice', data=df_train)


# `YearBuilt` and `OverallQual` variables seems to be positively correlated to `SalePrice`

# ## 3.2 Outliers
# 
# An outlier is a data point that is significantly different from the remaining data.

# In[19]:


# plot between GrLivArea and SalePrice
plt.figure(figsize=[10,8])
plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
plt.title('SalePrice and GrLivArea')
plt.show()


# In[20]:


# Function to create a histogram, and a boxplot and scatter plot.
def diagnostic_plots(df, variable,target):
    # The function takes a dataframe (df) and
    # the variable of interest as arguments.

    # Define figure size.
    plt.figure(figsize=(20, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30,color = 'r')
    plt.title('Histogram')


    # scatterplot
    plt.subplot(1, 3, 2)
    plt.scatter(df[variable],df[target],color = 'g')
    plt.title('Scatterplot')
    
    
    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable],color = 'b')
    plt.title('Boxplot')
    
    plt.show()


# In[21]:


diagnostic_plots(df_train, 'GrLivArea','SalePrice')


# The last two points seems to be outlier also skewed.

# In[22]:


df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]


# In[23]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# We got rid of these last two outliers.

# In[24]:


diagnostic_plots(df_train, 'MoSold','SalePrice')


# Let's detect highly correlated features and make our prediction easier!
# * Houses are mostly sold in summer and spring but there is almost no relationship between the sale month of the houses and the sale price, as seen in the correlation map down below.

# In[25]:


corr = df_train.corr()
corr['SalePrice'].sort_values(ascending=False)


# We can calculate the difference in years between the date the house was built and the date it was sold, and we can get a new feature. Let's examine the relationship between the selling price of this new feature.

# In[26]:


df_train['YearGap'] = df_train['YrSold'] - df_train['YearBuilt']
df_test['YearGap'] = df_test['YrSold'] - df_test['YearBuilt']


# In[27]:


corr = df_train.corr()
corr['SalePrice'].sort_values(ascending=False)


# In[28]:


diagnostic_plots(df_train,'YearGap','SalePrice')


# `SalePrice` and `YearGap` is negatively correlated! 
# * A large negative correlation is just as useful as a large positive correlation. The only difference is that for a positive correlation, as the feature increases, the target will increase. For a negative correlation, as the feature decreases, the target will increase.
# 
# 

# In[29]:


corr[corr['SalePrice'].gt(0.26) |  corr['SalePrice'].lt(-0.5) ].index


# In[30]:


df_train = df_train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice','YearGap']]
df_test = df_test[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF','YearGap']]


# In[31]:


df_train.shape


# Now, we have 20 colums to work with!

# ## 3.3 Normality

# In[32]:


df_train.hist(figsize=(20,20),color='pink')
plt.show()


# In[33]:


diagnostic_plots(df_train, 'LotFrontage','SalePrice')


# In[34]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[35]:


# Define figure size.
#not transformed histogram and normal probability plot
plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.distplot(df_train['SalePrice'], fit=norm,color = 'g')    
plt.subplot(1, 2, 2)
stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()


# In[36]:


#transformed histogram and normal probability plot
# Define figure size.
plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.distplot(df_train['SalePrice'], fit=norm,color = 'g')
plt.subplot(1, 2, 2)
stats.probplot(df_train['SalePrice'], plot=plt) 
plt.show()


# In[37]:


plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.distplot(df_train['GrLivArea'], fit=norm,color = 'g')    
plt.subplot(1, 2, 2)
stats.probplot(df_train['GrLivArea'], plot=plt)  
plt.show()


# In[38]:


df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[39]:


#transformed histogram and normal probability plot
plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.distplot(df_train['GrLivArea'], fit=norm,color = 'g')    
plt.subplot(1, 2, 2)
stats.probplot(df_train['GrLivArea'], plot=plt)  
plt.show()


# ## 3.4 Missing Values

# In[40]:


# separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    df_train.drop('SalePrice', axis=1),  # predictors
    df_train['SalePrice'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# In[41]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = df_train.isnull().mean().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# We have missing values in `LotFrontage` `GarageYrBlt` `MasVnrArea`. We have to impute these values.
# * `LotFrontage` is Linear feet of street connected to property

# In[42]:


df_train.dtypes


# `LotFrontage` `GarageYrBlt` `MasVnrArea` features are numerical values we can impute them using `MeanMedianImputer()`.

# In[43]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

# We fit the imputer to the train set.
# The imputer will learn the median of all variables.
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


# In[44]:


#Check remaining missing values if any 
all_data_na = (df_test.isnull().sum() / len(df_test)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[45]:


for col in ( 'GarageArea', 'GarageCars'):
    df_test[col] = df_test[col].fillna(0)
for col in ('BsmtFinSF1','TotalBsmtSF'):
    df_test[col] = df_test[col].fillna(0)


# In[46]:


df_test = imputer.transform(df_test)


# ## 3.6 Dummy variables

# In[47]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)
df_test = enc.transform(df_test)


# # 4. Models  <a id="4"></a>
# 

# In[48]:


# Instantiate DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 100)

# Instantiate Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.05, 'loss': 'ls'}

clf = GradientBoostingRegressor(**params)

# Instantiate svr
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')

# Instantiate random forest regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)


# Instantiate XGBRegressor
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# Instantiate LightGBM
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=500,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



# Instantiate Lasso
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# Define the list classifiers
classifiers = [
    ('DecisionTreeRegressor', dtreg),
    ('Gradient Boosting Regression', clf),
    ('XGBRegressor', model_xgb),
    ('Support Vector Machine Regression', svr),
    ('LightGBM',model_lgb)
]


# In[49]:


from sklearn.metrics import accuracy_score
from sklearn import metrics
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)
    
    # Predict y_pred
    y_pred = clf.predict(X_test)
    y_pred= y_pred.reshape(-1,1)
    # Calculate accuracy
    error = metrics.mean_squared_error(y_test, y_pred)
    # Evaluate clf's accuracy on the test set
    print('{:s} MSE : {:.3f}'.format(clf_name, error))


# In[50]:


lgb_pred = np.expm1(model_lgb.predict(df_test))

clf_pred = np.expm1(clf.predict(df_test))

model_xgb_pred = np.expm1(model_xgb.predict(df_test))


# In[51]:


ensemble = lgb_pred*0.70 + clf_pred*0.10 + model_xgb_pred*0.20


# In[52]:


ensemble = pd.DataFrame(ensemble, columns=['SalePrice'])


# In[53]:


test_ID = pd.DataFrame(test_ID, columns=['Id'])
result = pd.concat([test_ID, ensemble], axis=1)
result.head()


# # 5. Submission <a id="5"></a>

# In[54]:


result.to_csv('submission.csv',index=False)


# ![](https://i.pinimg.com/550x/dd/f0/1c/ddf01cd827ea8db0c44c2cabc638bf1a.jpg)
# 
# Hope you liked it! Please leave a comment and upvote 😊
# 
