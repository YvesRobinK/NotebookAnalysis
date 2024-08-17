#!/usr/bin/env python
# coding: utf-8

# # House Prices With Advanced Regression Techniques

# In this notebook I have performed a solution with all the best practices.
# And I have covered life cycle of a data science projects
# 
# In this Problem we have to predict house prices based on various features
# 
# Table of Content:
# 1. Importing Dataset
# 2. Data Analysis
# 3. Feature Engineering
# 4. Feature Selection
# 5. Modelling data
# 6. HyperParameter Tuning(if required)
# 7. Prediction metrics
# 8. Saving the submission file
# 
# This notebook is fully explained and this approach can be used for any regression technique.
# 
# **plz upvote and show ur appreciation and comment down if any queries**

# # Importing Dataset

# In[1]:


#imported the necessary libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#imported the train.csv
house_data= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[3]:


house_data.head()


# In[4]:


house_data.shape


# In[5]:


house_data.info()


# # Exploratory Data Analysis

# **Missing Values**

# In[6]:


sns.heatmap(house_data.isnull(),yticklabels=False,cbar=False,cmap= 'Oranges')


# In[7]:


pd.set_option('display.max_rows', None)
house_data.isnull().sum()


# In[8]:


#extracting features with any null values using list comprehension
null_features= [features for features in house_data.columns if house_data[features].isnull().any()==True]


# In[9]:


for feature in null_features:
    data=house_data.copy()
    
    #converting null features values with 0: non null values and 1: null values in features
    data[feature]=np.where(data[feature].isnull(),1,0)
    
    #now we will plot the features wrt Median Sales price for 0 and 1 values in features
    data.groupby(feature)['SalePrice'].median().plot.bar(color=['darkorange','lightblue'])
    print(data.groupby(feature)['SalePrice'].median())
    plt.show()


# **insights**
# So we can infer from this relationship between null features and dependent variable that null values in feature are also contributing towards SalePrice and even more than non null values in some features so we cant drop them we need fill them.

# **Numerical Features**

# In[10]:


#now we will extract all the numerical features from the dataset
numerical_features= [feature for feature in house_data.columns if house_data[feature].dtypes !='O']

print('Number of Numerical Features:',len(numerical_features))

house_data[numerical_features].head(5)


# **Datetime Features**

# In[11]:


#now we will extract datatime features from the dataset
year_feature=[feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]

print('Number of Yearly Features:',len(year_feature))
house_data[year_feature].head(5)


# In[12]:


#now we will analyze yearly features wrt SalePrice which is our independent feature
for feature in year_feature:
    data=house_data.copy()
    
    data.groupby(feature)['SalePrice'].median().plot()
    plt.show()


# **Insights**
# * Now what we can deduce from here is that newly built or remodelled houses and newly built garage house has more median Sale Price
# * And overtime the median Sale Price has decreased for houses sold in recent years

# now in Numerical Feature we have two types of variables continuos and discrete
# 
# **Discrete Feature**

# In[13]:


discrete_feature=[feature for feature in numerical_features if len(house_data[feature].unique())<25 and feature not in year_feature+['Id']]

print('Number of discrete feature:',len(discrete_feature))

house_data[discrete_feature].head(5)


# In[14]:


#Lets find the relationship between discrete feature and SalePrice
for feature in discrete_feature:
    data=house_data.copy()
    
    data.groupby(feature)['SalePrice'].median().plot.bar(color=['red','orange','green','skyblue','purple','turquoise','blue','darkorange'])
    plt.ylabel('SalePrice')
    plt.show()


# We can draw insights from the above given graphs that
# 1. In MSSubClass type of dwelling which is labled as 60 has the highest average Sale Price
# 2. Average Sale Price has exponentially increased for Overall quality and finish of the houses
# 3. In Overall Condition the average Sale price is higher for Label 5 which means Average condition of the house
# 
# and this is how we can draw insights form all of the graphs

# **Continuous Features**

# In[15]:


#now we will extract Continuos feature
continuous_features= [feature for feature in numerical_features if feature not in discrete_feature +year_feature +['Id']]

print('Number of Continuous Feature:',len(continuous_features))

house_data[continuous_features].head(5)


# In[16]:


#now lets do data analysis for continuous feature with the help of histograms
for feature in continuous_features:
    data=house_data.copy()
    
    data[feature].hist(bins=15)
    plt.ylabel('count')
    plt.xlabel(feature)
    plt.show()


# So as we can see only few of the features follows gausian distribution while other are skewed distribution
# we will apply normalization to fix this

# In[17]:


for feature in continuous_features:
    data=house_data.copy()
    data[feature]=np.log1p(data[feature])
    data['SalePrice']=np.log1p(data['SalePrice'])
    if feature=='SalePrice':
        pass
    else:
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# we have log normalised all the features and then plotting a relationship b/w features and SalePrice

# In[18]:


for feature in continuous_features:
    data=house_data.copy()
    data[feature]=np.log1p(data[feature])
    data.boxplot(column=feature)
    plt.ylabel(feature)
    plt.show()


# **Categorical variable**

# In[19]:


categorical_features=[feature for feature in house_data.columns if house_data[feature].dtypes=='O']

print('Number of categorical features:',len(categorical_features))
house_data[categorical_features].head(5)


# In[20]:


for feature in categorical_features:
    data=house_data.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar(color=['red','orange','green','skyblue','purple','turquoise','blue','darkorange'])
    plt.show()
    


# In[21]:


house_data.head(5)


# Handling categorical feature

# In[22]:


cat_features=[feature for feature in house_data.columns if house_data[feature].dtypes=='O']

house_data[cat_features].head(5)


# In[23]:


# % of missing values in categorical features
pct_miss=house_data[cat_features].isnull().sum()/len(house_data)*100


# In[24]:


#dropping categorical features where missing values is more than half
miss_features=pct_miss[pct_miss>70]
#now we need to drop thest columns 
miss_features


# In[25]:


for feature in miss_features.index:
    house_data.drop([feature],axis=1,inplace=True)


# In[26]:


null_features=[feature for feature in house_data.columns if house_data[feature].isnull().sum().any()==True]
null_features


# In[27]:


numerical_features=[feature for feature in null_features if house_data[feature].dtypes!='O']
house_data[numerical_features].isnull().sum()/len(house_data)*100


# so we dont have enough null values to drop so we will perform its feature engineering after split

# # Feature Engineering

# In[28]:


house_data.head()


# In feature Engineering we will handle:
# 1. numerical missing values
# 4. year_features missing values
# 3. Categorical missing values
# 4. And apply standard scaler to standardise the values of all the features

# But before performing all these steps we need to first split our house prices data which we got from train.csv into train and validation data to avoid overfitting and data leakage it is a good practice to split data first so that data doesnt get leaked.
# 
# to know more about data leakage refer to this below url
# Data Leakage:https://machinelearningmastery.com/data-leakage-machine-learning/

# In[29]:


X=house_data.drop(['SalePrice'],axis=1)
y=house_data['SalePrice']


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[31]:


X_train.shape,X_test.shape


# In[32]:


train=pd.concat([X_train,y_train],axis=1)
test=pd.concat([X_test,y_test],axis=1)


# **featurea with nan values**

# In[33]:


#features with nan values in training set
null_features= [features for features in train.columns if train[features].isnull().any()==True]


# In[34]:


null_features


# **Numerical Feature**

# In[35]:


null_numerical=[feature for feature in null_features if train[feature].dtypes!='O']

print('Number of null numerical feature:',len(null_numerical))

train[null_numerical].head()


# In[36]:


train[null_numerical].isnull().sum()


# In[37]:


#replacing nan values in numerical feature
for feature in null_numerical:
    train[feature].fillna(train[feature].median(),inplace=True)
    
train[null_numerical].isnull().sum()


# In[38]:


train[null_numerical].head()


# In[39]:


train.head(5)


# log normalise skewed Numerical feature which we have seen during our data analysis of continuous feature distribution

# In[40]:


skew_num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea','SalePrice']

for feature in skew_num_features:
    train[feature]=np.log(train[feature])


# **Year Features**

# In[41]:


train[year_feature].isnull().sum()


# no null values but we have to handle ['YearBuilt','YearRemodAdd','GarageYrBlt'] on the basis of year it was sold

# In[42]:


train[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']].head()


# In[43]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    train[feature]=train['YrSold']-train[feature]


# In[44]:


train[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']].head()


# now we have converted the year values to numerical values on the basis of YrSold Feature like Built recently or built 47 years ago or remodelled or garage built 47 years ago

# **Categorical Feature**

# In[45]:


null_categorical_feature=[feature for feature in null_features if train[feature].dtypes=='O']

print('number of null categorical features:',len(null_categorical_feature))
train[null_categorical_feature].head()


# In[46]:


train[null_categorical_feature].isnull().sum()/len(train)


# In[47]:


#replacing nan values in categorical feature with a new label
for feature in null_categorical_feature:
    mode_value=train[feature].mode()[0]
    train[feature].fillna(mode_value,inplace=True)
train[null_categorical_feature].isnull().sum()


# In[48]:


categorical_features=[feature for feature in train.columns if train[feature].dtypes=='O']


# In[49]:


#now we will perform feature label encoding
for feature in categorical_features:
    labels_order=train.groupby(feature)['SalePrice'].mean().sort_values().index
    labels_order={k:i for i,k in enumerate(labels_order,0)}
    train[feature]=train[feature].map(labels_order)


# In[50]:


train.head(5)


# In[51]:


train.isnull().sum()


# Now we will repeat all the steps for Test data set to avoid data leakage

# **Feature engineering on test data**

# In[52]:


#featurea with nan values
null_features= [features for features in test.columns if test[features].isnull().any()==True]
null_features


# **Numerical Feature** for test data

# In[53]:


null_numerical=[feature for feature in null_features if test[feature].dtypes!='O']

print('Number of null numerical feature:',len(null_numerical))

test[null_numerical].isnull().sum()


# In[54]:


#replacing nan values in numerical feature
for feature in null_numerical:
    test[feature].fillna(test[feature].median(),inplace=True)
    
test[null_numerical].isnull().sum()


# log normalise skewed Numerical feature which we have seen during our data analysis of continuous feature distribution

# In[55]:


skew_num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea','SalePrice']

for feature in skew_num_features:
    test[feature]=np.log(test[feature])


# In[56]:


test.head(5)


# **Year Features**

# In[57]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    test[feature]=test['YrSold']-test[feature]

test[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']].head()


# **Categorical features**

# In[58]:


null_categorical_feature=[feature for feature in null_features if test[feature].dtypes=='O']

print('number of null categorical features:',len(null_categorical_feature))

test[null_categorical_feature].head()


# In[59]:


#replacing nan values in categorical feature with a new label
for features in null_categorical_feature:
    mode_value=test[features].mode()[0]
    test[features].fillna(mode_value,inplace=True)
        
test[null_categorical_feature].isnull().sum()


# In[60]:


categorical_features=[feature for feature in test.columns if test[feature].dtypes=='O']

#now we will perform feature label encoding
for feature in categorical_features:
    labels_order=test.groupby(feature)['SalePrice'].mean().sort_values().index
    labels_order={k:i for i,k in enumerate(labels_order,0)}
    test[feature]=test[feature].map(labels_order)


# In[61]:


pd.set_option('display.max_columns',None)
test.head(5)


# In[62]:


test.isnull().sum()


# # Feature Scaling

# In[63]:


#lets perform Feature Scaling on train data
scale_feature=[feature for feature in train.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_scaled=scaler.fit_transform(train[scale_feature])


# In[64]:


#lets perform Feature Scaling on test data
test_scaled=scaler.transform(test[scale_feature])


# In[65]:


X=pd.DataFrame(train_scaled,columns=scale_feature)


# In[66]:


train=pd.concat([X,train['SalePrice'].reset_index(drop=True)],axis=1)


# In[67]:


train.head(5)


# In[68]:


train.isnull().sum()


# In[69]:


X1=pd.DataFrame(test_scaled,columns=scale_feature)


# In[70]:


test=pd.concat([X1,test['SalePrice'].reset_index(drop=True)],axis=1)


# In[71]:


test.head(5)


# In[72]:


test.isnull().sum()


# # Feature Selection

# In[73]:


#importing libraries to be used for feature selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[74]:


X_train=train.drop(['SalePrice'],axis=1)
y_train=train['SalePrice']


# In[75]:


#now we will use Lasso regression model
#and use the SelectFromModel this will select the features with non-zero coefficients
feature_sel_model = SelectFromModel(Lasso(alpha=0.01,random_state=0))
feature_sel_model.fit(X_train, y_train)


# In[76]:


#.get_support() will show u which all features are important
feature_sel_model.get_support()


# In[77]:


# list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

print('selected features:',len(selected_feat))


# In[78]:


selected_feat


# In[79]:


X_train=X_train[selected_feat]


# In[80]:


X_train.head(5)


# In[81]:


X_test=test[selected_feat]
y_test=test['SalePrice']


# In[82]:


X_test.head(5)


# In[83]:


X_test.isnull().sum()


# # Fitting model to dataset

# In[84]:


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()
rf_reg.fit(X_train,y_train)


# In[85]:


prediction=rf_reg.predict(X_test)


# In[86]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE:', mean_absolute_error(y_test, prediction))
print('MSE:', mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))


# In[87]:


#test data which is required to generate output submission file
house_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[88]:


house_test.head(5)


# **performing feature engineering on house test dataset for output**

# **featurea with nan values**

# In[89]:


#features with nan values
null_features= [features for features in house_test.columns if house_test[features].isnull().any()==True]
null_features


# **numerical feature for house_test data**

# In[90]:


null_numerical=[feature for feature in null_features if house_test[feature].dtypes!='O']

print('Number of null numerical feature:',len(null_numerical))

house_test[null_numerical].isnull().sum()


# In[91]:


#replacing nan values in numerical feature
for feature in null_numerical:
    house_test[feature].fillna(house_test[feature].median(),inplace=True)
    
house_test[null_numerical].isnull().sum()


# **log normalise skewed Numerical feature which we have seen during our data analysis of continuous feature distribution**

# In[92]:


skew_num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in skew_num_features:
    house_test[feature]=np.log(house_test[feature])


# **year features**

# In[93]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    house_test[feature]=house_test['YrSold']-house_test[feature]

house_test[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']].head()


# In[94]:


null_categorical_feature=[feature for feature in null_features if house_test[feature].dtypes=='O']

print('number of null categorical features:',len(null_categorical_feature))

null_categorical_feature


# In[95]:


#percentage of missing values in each categorical column
pct=house_test[null_categorical_feature].isnull().sum()/len(house_test)


# In[96]:


miss_feature=pct[pct>0.7]
miss_feature.index


# In[97]:


for feature in miss_feature.index:
    house_test.drop([feature],inplace=True,axis=1)


# In[98]:


house_test.head()


# In[99]:


null_feature=[feature for feature in house_test.columns if house_test[feature].isnull().sum().any()==True]
null_feature


# In[100]:


null_categorical_feature=[feature for feature in null_feature if house_test[feature].dtypes=='O']
#replacing nan values in categorical feature with a new label
for feature in null_categorical_feature:
    mode_value=house_test[feature].mode()[0]
    house_test[feature]=house_test[feature].fillna(mode_value)


# In[101]:


house_test.isnull().sum()


# In[102]:


house_test.head()


# In[103]:


categorical_features=[feature for feature in house_test.columns if house_test[feature].dtypes=='O']
#performing label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in categorical_features:
    house_test[feature]=le.fit_transform(house_test[feature])


# In[104]:


house_test.head(5)


# **performing feature scaling in house test data**

# In[105]:


house_test_scaled=scaler.transform(house_test[scale_feature])


# In[106]:


X_house=pd.DataFrame(house_test_scaled,columns=scale_feature)


# In[107]:


X_house.head(5)


# In[108]:


X_house=X_house[selected_feat]


# In[109]:


X_house.head(5)


# In[110]:


price_prediction=rf_reg.predict(X_house)


# In[111]:


price_prediction


# In[112]:


np.exp(price_prediction)


# # Prediction Metrics

# In[113]:


sample=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[114]:


y_test=sample['SalePrice']


# In[115]:


print('MAE:', mean_absolute_error(np.log(y_test),price_prediction))
print('MSE:', mean_squared_error(np.log(y_test), price_prediction))
print('RMSE:', np.sqrt(mean_squared_error(np.log(y_test), price_prediction)))


# In[116]:


house_test['SalePrice']=np.exp(price_prediction)


# In[117]:


submission=house_test[['Id','SalePrice']]


# In[118]:


submission.to_csv('./submission1.csv',index=False)


# In[ ]:




