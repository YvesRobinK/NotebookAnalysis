#!/usr/bin/env python
# coding: utf-8

# # Import of necessary libraries and data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


# In[2]:


Train_house=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
Test_house=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[3]:


Train_house.shape


# In[4]:


Test_house.shape


# In[5]:


House=pd.concat([Train_house.drop(['SalePrice'],axis=1),Test_house]).set_index('Id')


# In[6]:


y=Train_house.SalePrice


# # Let's do a mini-EDA

# In[7]:


numeric_features = Train_house.select_dtypes(include=[np.number])
numeric_features.columns


# In[8]:


year_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]
year_feature


# In[9]:


for feature in year_feature:
    if feature!='YrSold':
        data=Train_house.copy()
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# Let's look at the relationship between numeric variables and the target variable

# In[10]:


numeric_features = Train_house.select_dtypes(include=[np.number])
numeric_features.columns


# In[11]:


discrete_feature=[feature for feature in numeric_features if len(Train_house[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[12]:


for feature in discrete_feature:
    data=Train_house.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Target')
    plt.title(feature)
    plt.show()


# In[13]:


continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous Feature Count {}".format(len(continuous_feature)))


# In[14]:


for feature in continuous_feature:
    data=Train_house.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# Estimate the distribution of the target variable

# In[15]:


y = Train_house['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)


# In[16]:


target = np.log(Train_house['SalePrice'])
target.skew()
plt.hist(target,color='blue')


# Let's build a heat map of correlations between numerical variables in the training dataset

# In[17]:


correlation = Train_house.corr()
print(correlation['SalePrice'].sort_values(ascending = False),'\n')


# In[18]:


k= 11
cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(Train_house[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# Let us estimate the pairwise correlation of numerical variables

# In[19]:


sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(Train_house[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


# # Execute Future Engineering

# Divide the total data into categorical and numerical variables to fill in information gaps

# In[20]:


numeric_features = House.select_dtypes(include=[np.number])
numeric_features.columns


# In[21]:


categorical_features = House.select_dtypes(include=[np.object])
categorical_features.columns


# Let's start with simple categorical variables. It is necessary to estimate the number of missing information, as well as the number of unique values in each variable.

# In[22]:


cat_features_with_na=[features for features in categorical_features.columns if categorical_features[features].isnull().sum()>0]
for feature in cat_features_with_na:
    print(feature, np.round(100*categorical_features[feature].isnull().sum()/2919, 4),  ' % of Missing Values')


# Let's estimate the number of unique values in each variable of categorical features

# In[23]:


for column_name in categorical_features.columns:
    unique_category = len(categorical_features[column_name].unique())
    print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name,
                                                                                         unique_category=unique_category))


# Remove 'MiscFeature','Fence','PoolQC','Alley','FireplaceQu' from categorical variables.

# In[24]:


categorical_features=categorical_features.drop(['MiscFeature','Fence','PoolQC','Alley','FireplaceQu'],axis=1)


# Because the number of missing values for most of the variables is small, let's fill them with the mode. When filling variables with a mod using a loop, an error occurs. I decided to fill everything in manually.

# In[25]:


categorical_features


# In[26]:


categorical_features.MSZoning.unique()


# In[27]:


categorical_features['MSZoning'].fillna('RL',inplace=True)
categorical_features['Utilities'].fillna('AllPub',inplace=True)
categorical_features['MasVnrType'].fillna('None',inplace=True)
categorical_features['BsmtQual'].fillna('TA',inplace=True)
categorical_features['BsmtCond'].fillna('TA',inplace=True)
categorical_features['BsmtExposure'].fillna('No',inplace=True)
categorical_features['BsmtFinType1'].fillna('Unf',inplace=True)
categorical_features['BsmtFinType2'].fillna('Unf',inplace=True)
categorical_features['Functional'].fillna('Typ',inplace=True)
categorical_features['GarageType'].fillna('Attchd',inplace=True)
categorical_features['GarageFinish'].fillna('Unf',inplace=True)
categorical_features['GarageQual'].fillna('TA',inplace=True)
categorical_features['GarageCond'].fillna('TA',inplace=True)
categorical_features['Exterior1st'].fillna('VinylSd',inplace=True)
categorical_features['Exterior2nd'].fillna('VinylSd',inplace=True)
categorical_features['Electrical'].fillna('SBrkr',inplace=True)
categorical_features['KitchenQual'].fillna('TA',inplace=True)
categorical_features['SaleType'].fillna('WD',inplace=True)


# Let's estimate the gaps in information and the number of unique values in numeric variables. Because we have categorical variables and in numerical ones, then we will fill in the gaps of information in a different way.

# In[28]:


num_features_with_na=[features for features in numeric_features.columns if numeric_features[features].isnull().sum()>0]
for feature in num_features_with_na:
    print(feature, np.round(100*numeric_features[feature].isnull().sum()/2919, 4),  ' % of Missing Values')


# In[29]:


for column_name in numeric_features.columns:
    unique_values = len(numeric_features[column_name].unique())
    print("Feature '{column_name}' has '{unique_values}' unique values".format(column_name = column_name,
                                                                                         unique_values=unique_values))


# Variables 'BsmtFullBath' and 'BsmtHalfBath', 'GarageCars' mod, rest medium

# In[30]:


numeric_features['LotFrontage'].fillna(69.30579531442663,inplace=True)
numeric_features['MasVnrArea'].fillna(102.20131215469613,inplace=True)
numeric_features['GarageYrBlt'].fillna(1978,inplace=True)
numeric_features['BsmtFullBath'].fillna(0,inplace=True)
numeric_features['BsmtHalfBath'].fillna(0,inplace=True)
numeric_features['GarageCars'].fillna(2,inplace=True)
numeric_features['BsmtFinSF1'].fillna(441.4232350925291,inplace=True)
numeric_features['BsmtFinSF2'].fillna(49.58224811514736,inplace=True)
numeric_features['BsmtUnfSF'].fillna(560.7721041809458,inplace=True)
numeric_features['TotalBsmtSF'].fillna(1051.7775873886224,inplace=True)
numeric_features['GarageArea'].fillna(0,inplace=True)


# We observe a strong cross-correlation between some variables, remove duplicates from the data

# In[31]:


numeric_features=numeric_features.drop(['GarageArea','1stFlrSF'],axis=1)


# Merge Tables and Encode Categorical Variables One Hot Encoding

# In[32]:


House_data=numeric_features.merge(categorical_features,on='Id')


# In[33]:


cat_features=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition']


# In[34]:


for name in cat_features:
    Dummies=pd.get_dummies(House_data[name]).add_prefix(name)
    House_data=House_data.merge(Dummies,on='Id')
    House_data=House_data.drop([name],axis=1)


# In[35]:


Train=House_data[:1460]
Valid=House_data[1460:]


# In[36]:


Tr=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv").set_index('Id')


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(Train,
                                                    Tr.SalePrice,
                                                    test_size=0.33,
                                                    random_state=42)


# # Search for the best model and predict the target variable

# Let's build model pipelines

# In[38]:


models = [RandomForestRegressor(), LinearRegression(),ElasticNet(), KNeighborsRegressor(),xgb.XGBRegressor()]
scores = dict()

for m in models:
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    print(f'model: {str(m)}')
    print(f'RMSE: {round(np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred))), 3)}')
    print(f'MAE: {round(mean_absolute_error(y_test, y_pred), 3)}')
    print('-'*30, '\n')


# Let's start improving the XGBRegressor and RandomForestRegressor hyperparameters. Then we average the values of the final forecast

# In[39]:


clf = xgb.XGBRegressor()
parametres={'base_score':[0.1],
            'learning_rate':[0.1],
           'max_depth':[5,6,7],
           'n_estimators':[100,90,110]}
grid_search_cv_clf=GridSearchCV(clf,parametres,cv=5)
grid_search_cv_clf.fit(X_train,y_train)
best_clf1=grid_search_cv_clf.best_estimator_
y_pred1=best_clf1.predict(X_test)
print(f'RMSE: {round(np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred))), 3)}')


# In[40]:


clf1 = RandomForestRegressor()
parametres={'max_depth':[1,2,4,8],
           'min_samples_split':[2,4,8],
           'n_estimators':[10,20,40,80],
           'n_jobs':[-1]}
grid_search_cv_clf=GridSearchCV(clf1,parametres,cv=5)
grid_search_cv_clf.fit(X_train,y_train)
best_clf2=grid_search_cv_clf.best_estimator_
y_pred2=best_clf2.predict(X_test)
print(f'RMSE: {round(np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred))), 3)}')


# In[41]:


y_predicted_prob1=best_clf1.predict(Valid)
y_predicted_prob2=best_clf2.predict(Valid)


# In[42]:


summ=(y_predicted_prob1+y_predicted_prob2)/2


# In[43]:


Tit=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[44]:


submissions = pd.concat([Tit.Id,pd.Series(summ)],axis=1)


# In[45]:


submissions=submissions.rename(columns={0:'SalePrice'})


# In[46]:


submissions.to_csv('submissionhouse.csv',index=False)

