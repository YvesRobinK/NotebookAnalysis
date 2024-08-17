#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yellowbrick


# In[2]:


import os
import warnings
import numpy as np 
import pandas as pd 
import xgboost
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

sns.set()


# In[3]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[4]:


train.head(5)


# In[5]:


train.describe()


# In[6]:


train.info(verbose = True, null_counts = True)


# In[7]:


# Visualize the correlation between the number of missing values in different columns of dataset as a heatmap 
msno.heatmap(train);


# In[8]:


#Plotting the distribution of sales Price
plt.figure(figsize=(20,5))
sns.distplot(train.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density");


# In[9]:


# correlation heatmap
plt.figure(figsize=(10,8))
cor = train.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()


# In[10]:


#Plotting the correlation values with the sales price 
train.corrwith(train.SalePrice).plot.bar(figsize = (20, 10), title = "Correlation with class", fontsize = 15,
                                         rot = 90, grid = True);


# In[11]:


plt.figure(figsize=[20,10])
plt.subplot(331)
sns.distplot(train['LotFrontage'].dropna().values)
plt.subplot(332)
sns.distplot(train['GarageYrBlt'].dropna().values)
plt.subplot(333)
sns.distplot(train['MasVnrArea'].dropna().values)
plt.suptitle("Distribution of data before Filling NA'S");


# **Impute Missing Values**
# 
# As we can see in LotFrontage plot the distribution is approimately Noraml. So we can use either mean or mode to replace the missing values.
# In GarageYrBlt plot the distribution is skewed and median is favourable to impute the missing values in this case.
# And in final MasVnrArea distribution which is also skewed, we'll use median to impute missing values. 

# In[12]:


train['LotFrontage']=train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['GarageYrBlt']=train.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
train['MasVnrArea']=train.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# Now apply the same in Test dataset

# In[13]:


test['LotFrontage']=test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['GarageYrBlt']=test.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
test['MasVnrArea']=test.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# Now we'll made some new features that sounds obvious but not so certain at first. When we actually look for a house, we keep all these features mentioned below in our mind. Some wise person went through all these details in the discussion. Let's apply them here.

# In[14]:


train['cond*qual'] = (train['OverallCond'] * train['OverallQual']) / 100.0
train['home_age_when_sold'] = train['YrSold'] - train['YearBuilt']
train['garage_age_when_sold'] = train['YrSold'] - train['GarageYrBlt']
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'] 
train['total_porch_area'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch'] 
train['Totalsqrfootage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])
train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))


# In[15]:


test['cond*qual'] = (test['OverallCond'] * test['OverallQual']) / 100.0
test['home_age_when_sold'] = test['YrSold'] - test['YearBuilt']
test['garage_age_when_sold'] = test['YrSold'] - test['GarageYrBlt']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF'] 
test['total_porch_area'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch'] 
test['Totalsqrfootage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])
test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))


# In[16]:


old_cols=['OverallCond','OverallQual','YrSold','YearBuilt','YrSold','GarageYrBlt','TotalBsmtSF','1stFlrSF','2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']


# Let's delete the features used in making the new features. They're not useful anymore.

# In[17]:


final_cols=[]
for cols in train.columns:
    if cols not in old_cols and cols!='SalePrice':
        final_cols.append(cols)
train_clean=train[final_cols]


# In[18]:


final_cols=[]
for i in test.columns:
    if i not in old_cols and i!='SalePrice':
        final_cols.append(i)
test_clean=test[final_cols]


# In[19]:


train_clean.columns


# * Again let's check the correlation in our new features.

# In[20]:


# Bivariate plot between all the areas
plt.figure(figsize=(16,10))
plt.subplot(3,3,1)
sns.regplot(data=train,x='TotalBsmtSF',y='GrLivArea',x_jitter=0.3,scatter_kws={'alpha':1/5});
plt.subplot(3,3,2)
sns.regplot(data=train,x='TotalBsmtSF',y='GarageArea',x_jitter=0.3,scatter_kws={'alpha':1/5});
plt.subplot(3,3,3)
sns.regplot(data=train,x='GrLivArea',y='GarageArea',x_jitter=0.3,scatter_kws={'alpha':1/5});


# In[21]:


#price range correlation
corr=train.corr()
corr=corr.sort_values(by=["SalePrice"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15,20))
sns.barplot(x=corr.values, y =corr.index.values);
plt.title("Correlation Plot");


# In[22]:


y = train.SalePrice


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(train_clean, y, test_size=0.3)


# In[24]:


def Change(x):
    for col in x.select_dtypes(include=['object']).columns:
               x[col] = x[col].astype('category')
    for col in x.select_dtypes(include=['category']).columns: 
               x[col] = x[col].cat.codes
    return x  


# In[25]:


X_train = Change(X_train)
X_test = Change(X_test)


# In[26]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[27]:


# from scipy import stats
# from sklearn.model_selection import GridSearchCV

# score_calc = 'neg_mean_squared_error'

# folds = 5

# param_grid = {'learning_rate' : [0.01,0.001], 'n_estimators' : [40,200,500,5000], 'random_state': [5],
#               'max_depth' : [4,8,12,16], 'gamma' : [0.5, 0.6], 'reg_alpha' : [0.4, 0.5], 'reg_lambda' : [0.45, 0.6],
#               'min_child_weight' : [1.5, 2, 1], 'tree_method' : ['gpu_hist']}
# grid_xgb = GridSearchCV(XGBRegressor(objective ='reg:squarederror'), param_grid, cv = folds, refit=True, verbose = 1, scoring = score_calc)
# grid_xgb.fit(X_train, y_train)


# In[28]:


#xgb = get_best_score(grid_xgb)


# ** Let's Build Our XGBoost Model **

# In[29]:


model = XGBRegressor(colsample_bytree=1,
                 gamma=0.5,                 
                 learning_rate=0.005,
                 max_depth=9,
                 min_child_weight=1.5,
                 n_estimators=5000,                                                                    
                 reg_alpha=0.4,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)  


# In[30]:


model.fit(X_train, y_train)
model.score(X_test,y_test)*100


# In[31]:


visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
visualizer.poof();


# In[32]:


feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[33]:


SalePrice = pd.DataFrame(model.predict(Change(test_clean)))
Id = pd.DataFrame(test_clean.Id)
result = pd.concat([Id, SalePrice], axis=1)
result.columns = ['Id', 'SalePrice']


# In[34]:


result.to_csv('submission.csv',index=False)


# If you think, you can modify this kernel to achieve a better score than you're welcome to do so. But please comment the approch in the commenting session.
