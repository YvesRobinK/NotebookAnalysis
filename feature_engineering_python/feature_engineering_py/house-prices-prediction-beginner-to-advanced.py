#!/usr/bin/env python
# coding: utf-8

# **Goal: Predict the sales price for each house in the test set. Main evaluation metric is Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.**
# 
# This notebook will be specially helpful to those who are from a non coding background. I have used simple visualizations, to objectively understand the data for the purposes of modelling.

# # Importing the relevant libraries

# In[1]:


# lets import the relevat files first
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# **importing the test and train datasets**
# 
# We will be importing both the test and train datasets.
# 
# After importing both the test and the train datasets, we will process them for missing values, and other data hygiene
# 
# The processing on the test data set will be the dropping those columns from the test, which have been dropped from the train, so as to keep the columns of both the datasets aligned, as well as filling the missing values in the test

# In[3]:


train_original = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_original = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


train = train_original.copy()
test = test_original.copy()


# # Data Preprocessing
# 
# Lets check the data that we have read,in the respective dataframes. 
# We will basically check for number of rows and columns, and basic data hygiene

# In[5]:


train.head(10)


# In[6]:


train.info()


# In[7]:


# MSSubClass is actually a categorical variable, hence converting it to the categorical one and dropping ID
train['MSSubClass'] = train['MSSubClass'].astype('object')
train.drop(['Id'],axis=1, inplace=True)
test['MSSubClass'] = test['MSSubClass'].astype('object')
test.drop(['Id'],axis=1, inplace=True)

train['MoSold'] = train['MoSold'].astype('object')
test['MoSold'] = test['MoSold'].astype('object')

train['YrSold'] = train['YrSold'].astype('object')
test['YrSold'] = test['YrSold'].astype('object')
# Simultaneously we will make a note of this in a separate notebook, so that we can make use of it later on if required.


# Lets differentiate between the categorical and continuous variables and store these in separate lists. This will come handy later

# In[8]:


cat_cols_train = []
cont_cols_train = []

for i in train.columns:
    if train[i].dtypes == 'object':
        cat_cols_train.append(i)
    else:
        cont_cols_train.append(i)


# Lets do the same for the test dataset. We are doing this to see if there is any discrepancy between the two datasets. In case there is no discrepancy between the two data sets, we can then combine the two datasets

# In[9]:


cat_cols_test = []
cont_cols_test = []

for i in test.columns:
    if test[i].dtypes == 'object':
        cat_cols_test.append(i)
    else:
        cont_cols_test.append(i)


# Lets now proceed with the EDA 1.0

# # EDA 1.0
# 
# the purpose of EDA is to identify the following:
# 1. Understand the business context
# 1. See if any feature engineering might be required
# 1. check for the skewness in the data.
# 1. Identify if the outliers and missing values are genuine and whether they should be treated
# 1. Parameters to treat the outliers and the missing values
# 1. Whether scaling of the data will be required
# 1. If there's any multicollinearity present among the variables and whether some variables should be dropped.
# 

# ## Checking the target variable
# 
# 

# In[10]:


sns.boxplot(train['SalePrice'])
plt.show()


# In[11]:


from scipy.stats import norm
(avge, std_dev) = norm.fit(train['SalePrice'])
plt.figure(figsize = (20,10))
sns.distplot(a=train['SalePrice'],hist=True,kde=True,fit=norm)
plt.title('SalePrice distribution vs Normal Distribution', fontsize = 13)
plt.xlabel('Sale Price in US$')
plt.legend(['Sale Price ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(avge, std_dev)],
            loc='best')
plt.show()


# As we can see that the SalePrice is not normal. Lets further check this by way of qq plots

# In[12]:


# qq plot for SalePrice
# qq plots can be imported from the statsmodels library
import statsmodels.api as sm 
sm.qqplot(train['SalePrice'], line='s')
plt.show()


# In[13]:


# We can also draw a probability plot to check the same
# probplot can be imported from scipy.stats
import scipy.stats as stats
import pylab
stats.probplot(train['SalePrice'], dist='norm', plot=pylab)
pylab.show()


# Lets convert this into the natural log and then see the distribution

# In[14]:


train['SalePrice'] = np.log(train['SalePrice'])


# In[15]:


train['SalePrice'].head()


# In[16]:


(avge, std_dev) = norm.fit(train['SalePrice'])
plt.figure(figsize = (20,10))
sns.distplot(a=train['SalePrice'],hist=True,kde=True,fit=norm)
plt.title('SalePrice distribution after log vs Normal Distribution', fontsize = 13)
plt.xlabel('Sale Price in US$')
plt.legend(['Sale Price ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(avge, std_dev)],
            loc='best')
plt.show()


# In[17]:


# qqplot
sm.qqplot(train['SalePrice'],line='s')
plt.show()


# In[18]:


#Probplot
stats.probplot(train['SalePrice'],dist='norm', plot=pylab)
pylab.show()


# ## Outliers in the target variable
# lets check for the outliers in the target variable.

# In[19]:


sns.boxplot(train['SalePrice'], orient='v')
plt.show()


# lets remove these outliers as these can really have a detrimental effect on the linear regression models that we are trying to build.

# In[20]:


def outliers(variable):
  sorted(train[variable])
  Q1,Q3 = np.percentile(train[variable],[25,75])
  IQR = Q3-Q1
  lr = Q1 - (1.5*IQR)
  ur = Q3 + (1.5*IQR)
  return ur,lr


# In[21]:


ur,lr = outliers('SalePrice')


# In[22]:


train = train.drop(train[(train['SalePrice']<lr ) | (train['SalePrice']>ur)].index)


# In[23]:


train.shape


# In[24]:


sns.boxplot(train['SalePrice'], orient='v')
plt.show()


# In[25]:


#Probplot
stats.probplot(train['SalePrice'],dist='norm', plot=pylab)
pylab.show()


# In[26]:


(avge, std_dev) = norm.fit(train['SalePrice'])
plt.figure(figsize = (20,10))
sns.distplot(a=train['SalePrice'],hist=True,kde=True,fit=norm)
plt.title('SalePrice distribution after log vs Normal Distribution', fontsize = 13)
plt.xlabel('Sale Price in US$')
plt.legend(['Sale Price ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(avge, std_dev)],
            loc='best')
plt.show()


# In[ ]:





# ## lets draw the histograms to understand the data distribution in the continuous variables

# In[27]:


train[cont_cols_train].hist(figsize=(20,20))
plt.show()


# In the above we can notice the following:
# 
# 1. for most continuous variables, the scales vary widely, hence we will need to standardise the data.
# 1. Variables such as YearBuilt, GarageYrBlt are left skewed, but still more and more houses are build in the recent years and more and more garages are built in the later years. Hence we will not check or treat any ourliers for these variables.
# 1. Similarly, variables like EnclosedPorch, OpenPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal have overwhelming number of values close to 0. This means they actually may be significant for determining the Sale Price.
# 1. However before dropping these variables we will look at their value counts as well as correlation martix.

# We have the following variables for which the values are heavily right skewed.
# 
# 'LoTArea',  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtHalfBath', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
# 
# Lets check the descriptive values for these variables

# In[28]:


list1=['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
desc1 = train[list1].describe().transpose()
desc1['coeff_of_var'] = desc1['std']/desc1['mean']


# In[29]:


desc1


# Here we can see that there are several variables, where the coefficient of Variation (std/mean) is extremely high accompanied by very few non zero values. These data in these variables have very high variability.
# 

# While we will typically include data with high variability, but in this case we will ignore those variables where the upper quantile is also 0 and the cofeeicient of variation is above 3. Its very mych like having the missing values

# In[30]:


desc1[desc1['coeff_of_var']>3].T.columns


# We will be dropping ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal'].from the train dataset and well as the test dataset, and store this in a list

# In[31]:


dropped_columns = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal']

train.drop(dropped_columns,axis=1, inplace=True)
test.drop(dropped_columns,axis=1, inplace=True)


# In[32]:


cat_cols= []
cont_cols = []

for i in test.columns:
    if test[i].dtypes == 'object':
        cat_cols.append(i)
    else:
        cont_cols.append(i)


# In[33]:


cat_cols


# Lets check the correlation between various variables

# In[34]:


# Correlation Matrix

f, ax = plt.subplots(figsize=(30, 25))
corr_matrix = train.corr('pearson')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(300, 50, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0, annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[ ]:





# # Lets check for the missing values

# ## Lets check the categorical values first

# In[35]:


for i in train.columns:
    if train[i].isnull().sum()>0:
        if train[i].dtypes == 'object':
            print(i)
            print('Total null values:',train[i].isnull().sum())
            print('Null values as a % of total:',round((train[i].isnull().sum()*100)/train['SalePrice'].count(),1)) 
            print()


# In[36]:


for i in test.columns:
    if test[i].isnull().sum()>0:
        if test[i].dtypes == 'object':
            print(i)
            print('Total null values:',test[i].isnull().sum())
            print('Null values as a % of total:',round((test[i].isnull().sum()*100)/train['SalePrice'].count(),1)) 
            print()


# Lets check for the barplot of the categorical variables

# In[37]:


f, axes = plt.subplots(12, 4, figsize=(20, 40))
for ax, col in zip(axes.ravel(), cat_cols):
    y = train[col].value_counts()
    ax.bar(y.index, y)
    ax.set_title(col)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


# Here we can see that there are a few variables where there are an overwhelming number of missing values. Lets check for those variables where missing values exceed 40% 

# In[38]:


for i in train.columns:
    if train[i].dtypes == 'object':
        if train[i].isnull().sum()>0:
            missing_val_perc = round((train[i].isnull().sum()*100)/train['SalePrice'].count(),1)
            if missing_val_perc > 40:
                print(i)
                print(train[i].value_counts())
                print('Total null values:',train[i].isnull().sum())
                print('Null values as a % of total:',round((train[i].isnull().sum()*100)/train['SalePrice'].count(),1)) 
                print()


# In the above we have 2 scenarios
# 1. either the houses dont have these attributes hence their value has been left out
# 2. Or the houses have these attributes but their values have not been filled.
# 
# However it seems really rare that all the houses will have all the 50+ attributes. Hence we can say that these missing values actually correspond to Not Applicable, except PoolQC, where we actually have the Pool are available. So we can drop PoolQC
# 

# In[39]:


train.drop('PoolQC',axis=1,inplace=True)
test.drop('PoolQC',axis=1,inplace=True)
dropped_columns.append('PoolQC')


# For the other variables, where there are missing values lets fill the missing values by median or most frequent, whereever applicable 

# In[40]:


cat_cols= []
cont_cols = []

for i in test.columns:
    if test[i].dtypes == 'object':
        cat_cols.append(i)
    else:
        cont_cols.append(i)


# For 'Alley','FireplaceQu','Fence','MiscFeature', lets fill the missing values by Not_Applicable for others lets fill the missing values by most frequently occurring values

# In[41]:


list2 =['Alley','FireplaceQu','Fence','MiscFeature']

for i in list2:
    train[i].fillna('Not_Applicable', inplace=True)
    test[i].fillna('Not_Applicable', inplace=True)


# In[42]:


for i in cat_cols:
    if train[i].isnull().sum()>0:
        train[i].fillna(train[i].value_counts().index[0], inplace=True)


# In[43]:


for i in cat_cols:
    if test[i].isnull().sum()>0:
        test[i].fillna(train[i].value_counts().index[0], inplace=True)


# In[44]:


train[cat_cols].isnull().sum()


# In[45]:


test[cat_cols].isnull().sum()


# # Lets now check for the continuous variables

# In[46]:


for i in train.columns:
    if train[i].isnull().sum()>0:
        if train[i].dtypes != 'object':
            print(i)
            print('Total null values:',train[i].isnull().sum())
            print('Null values as a % of total:',round((train[i].isnull().sum()*100)/train['SalePrice'].count(),1)) 
            print()


# In[47]:


for i in test.columns:
    if test[i].isnull().sum()>0:
        if test[i].dtypes != 'object':
            print(i)
            print('Total null values:',test[i].isnull().sum())
            print('Null values as a % of total:',round((test[i].isnull().sum()*100)/train['SalePrice'].count(),1)) 
            print()


# We can fill missing values in the train dataset by their median since a very few of them are missing values

# In[48]:


for i in cont_cols:
    if train[i].isnull().sum()>0:
        train[i].fillna(train[i].median(), inplace=True)


# In[49]:


# lets check for the missing values again

train.isnull().sum()


# In[50]:


for i in cont_cols:
    if test[i].isnull().sum()>0:
        test[i].fillna(test[i].median(), inplace=True)


# In[51]:


# lets check for the missing values again

train.isnull().sum()


# # EDA 2.0 
# Lets Check how independent variables Vary with the Log SalePrice

# In[52]:


for i in cont_cols:
  plt.figure(figsize=(10,5))
  sns.scatterplot(x=train['SalePrice'], y=train[i])
  plt.show()


# Lets Now check the variability of the SalePrice with respect to Categorical Variables

# In[53]:


len(cat_cols)


# In[54]:


for i in cat_cols:
  plt.figure(figsize=(15,5))
  f = sns.stripplot(x=train[i], y=train['SalePrice'])
  f.set_xticklabels(f.get_xticklabels(),rotation=45)
  plt.show()
  plt.figure(figsize=(15,5))
  g = sns.boxplot(x=train[i], y=train['SalePrice'])
  g.set_xticklabels(g.get_xticklabels(),rotation=45)
  plt.show()


# ## Insights from EDA 2.0

# Continuous Variables:
# 1. LotFrontage and LotArea do not show any significant correlation with the Target Variable
# 1. However OverQual and the yearbuilt and YearRemodadd show considerable correlation. It will be an interesting thing to see if the age of the house has any thing to do with the sale price and locality.
# 1. Features related to Basement shows some correlation with the SalePrice. Lets try to do some Feature engineering to see if the basement related features have a significant impact on the Sale Price
# 1. Similarly, Greater liv area, 1st floor area and the second floor area too have significant impact on the SalePrice.
# 1. Similarly, total rooms and the garage related variables seem to have good correlation on the SalePrice.
# 
# Categorical variables:
# 1. Variables such as MsSubClass, Neighborhood, MSzoning, Condition, building type, House style, Exterior, Foundation, Heating, Central Airconditioning, kitchen quality, Garage Quality, Garage condition, Saletype seem to have significant variation with the saleprice. Lets try to capture these in the correlation matrix via feature engineering

# In[55]:


plt.figure(figsize=(15,5))
f = sns.stripplot(x=train['OverallQual'], y=train['SalePrice'])
f.set_xticklabels(f.get_xticklabels(),rotation=45)
plt.show()
plt.figure(figsize=(15,5))
g = sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.show()


# # Feature Engineering

# * There are a Categorical variables where we can make 
# 
# * the comparison among the values. Example MSSubClass, MSZoning,  ExterQual, FireplaceQual, GarageCond, Condition of Sale etc. 
# 
# * Then we know that around the world, the prices vary as per the location and hence we can say that one type of location in the city is better than others. 
# 
# * Similarly we can say that proximity to main road will fetch higher price(which again may or may not depend on the neighborhood)
# 
# * So we will need to create continuous variables for all those which can be compared.
# 
# * Also we will need to find a proxy for location.
# 
# To sum it up we need to do the following:
# 
# 1. House Quality -> we have several variables but none of them tend to show the overall house quality. SO first we will translate all the quality and conditions variables into ordinal ones and then see if we need further feature engineering.
# 1. Creating a Location variable:
# > 1. The problems with having a dummy of each
# neighbourhood are: 
# > 1. there are only a handful of observations for some neighbourhoods, with less than 30 for 8 neighbourhoods, and less than 100 for the majority of them;
# > 1. there would be significant multicollinearity between certain neighbourhoods that share similar characteristics.
# 1. To do this a very simplistic approach would be to assign ordinal values to the neighbourhoods based on the mean saleprice of each locality but again the main idea behind ranking localities is their desirability. Hence we also need to take into account the quality, condition, proximity to the main road/railroad etc.
# 
# However we will not create this location variable immediately lets first convert the others into their ordinal codes and then check the correlation.

# lets create ordinal values for the following variables:
# 
# ['BsmtCond','BsmtFinType1','BsmtFinType2','BsmtQual','ExterCond','ExterQual','Fence','FireplaceQu','Functional','GarageCond','GarageType','SaleCondition'
# ]

# In[56]:


list1 =['BsmtFinType1','BsmtFinType2','BsmtQual','ExterCond','ExterQual','Fence','FireplaceQu','Functional','GarageCond','GarageQual','SaleCondition','KitchenQual']


# In[57]:


# defining a function for ordinal encoding of the certain variables
def ordinal_encoding(variable):
  df = train[[variable,'SalePrice']]
  df1 = df.groupby(by=variable,axis=0).median()
  df1 = df1.sort_values(by='SalePrice', axis=0, ascending=True)
  df1 = df1.reset_index()
  df1[variable+'_codes'] = df1['SalePrice'].astype('category').cat.codes
  df1[variable+'_codes'].astype('int')
  df1[[variable+'_codes']] = df1[[variable+'_codes']]+1
  df1.drop(['SalePrice'],axis=1,inplace=True)
  df2 = train.merge(df1, on=variable, how='left')
  return df2


# In[58]:


# adding the ordinal variables to the train dataframe
train_final = ordinal_encoding('BsmtCond')
for i in list1:
  df1 = ordinal_encoding(i)
  j=i+'_codes'
  df2 = df1[j]
  train_final = pd.concat([train_final,df2],axis=1)


# In[59]:


train_final


# In[60]:


# lets do the same for the test dataset as well
def ordinal_encoding_test(variable):
  df = train_final[variable+'_codes'].groupby([train_final[variable]]).mean().sort_values()
  df = df.reset_index()
  df2 = test.merge(df, on=variable, how='left')
  return df2


# In[61]:


# adding the ordinal variables to the test dataframe
test_final = ordinal_encoding_test('BsmtCond')
for i in list1:
  df1 = ordinal_encoding_test(i)
  j=i+'_codes'
  df2 = df1[j]
  test_final = pd.concat([test_final,df2],axis=1)


# In[62]:


test_final.head()


# In[63]:


# Since we have added the ordinal variables for the certain variables, lets remove the original 
# from both test and train datasets
# list1.append('BsmtCond')
for i in list1:
  train_final.drop([i],axis=1,inplace=True)
  test_final.drop([i],axis=1, inplace=True)
  dropped_columns.append(i)


# In[64]:


dropped_columns


# In[65]:


# lets check if our operation is successful
train_final.head()


# In[66]:


test_final.head()


# Lets create 2 more variables -> Squarefeet per room -> this is indicative of the fact that properties with bigger rooms fetch larger prices. 
# 
# However for a standard number of rooms, bathrooms and kitchen this should correlate with the Total Living Area.
# 
# Lets do this and see. We can easily drop it later if there is high correlation between this variable and GrLivArea

# In[67]:


# taking squarefeet per room
train_final["SqFtPerRoom"] = train_final["GrLivArea"] / (train_final["TotRmsAbvGrd"] + 
                                         train_final["FullBath"] +
                                         train_final["HalfBath"] + 
                                         train_final["KitchenAbvGr"])

# taking the total number of bathrooms in the house
train_final['Total_Bathrooms'] = (train_final['FullBath'] + 
                                  (0.5 * train_final['HalfBath']) +
                                  train_final['BsmtFullBath'])

# Similarly doing the same for the test dataset

# taking squarefeet per room
test_final["SqFtPerRoom"] = test_final["GrLivArea"] / (test_final["TotRmsAbvGrd"] + 
                                         test_final["FullBath"] +
                                         test_final["HalfBath"] + 
                                         test_final["KitchenAbvGr"])

# taking the total number of bathrooms in the house
test_final['Total_Bathrooms'] = (test_final['FullBath'] + 
                                 (0.5 * test_final['HalfBath']) +
                                 test_final['BsmtFullBath'])


# In[68]:


plt.figure(figsize = (30,30))
sns.heatmap(train_final.corr(),annot=True)
plt.show()


# Now from the above heatmap, i would want to drop those variables:
# 1. which do not seem to be a good predictor of the target variable
# 1. Which are highly correlated with other variables

# In[69]:


# lets see which are those variables which have low correlation with the SalePrice
# its better to remove them since these are not good predictors of the SalePrice and most likely will add noise
df4 = train_final.corr()
df4.loc['SalePrice'][df4['SalePrice']<.2]


# In[70]:


list3=['OverallCond','BedroomAbvGr','KitchenAbvGr','EnclosedPorch','BsmtCond_codes','BsmtFinType2_codes','ExterCond_codes','Functional_codes','GarageCond_codes']
train_final.drop(list3,axis=1,inplace=True)
test_final.drop(list3,axis=1,inplace=True)
for i in list3:
  dropped_columns.append(i)


# In[71]:


# lets check the heatmap once again
plt.figure(figsize = (30,30))
sns.heatmap(train_final.corr(),annot=True)
plt.show()


# In[72]:


# finding those pairs where correlation is >0.6, to identify and remove multicollinearity
for i in train_final.corr().columns:
  for j in train_final.corr().columns:
    train_corr= train_final[[i,j]].corr()
    x=train_corr.iloc[0,1]
    if (x >.6)& (x<1):
      sns.pairplot(train_final[[i,j]])
      plt.show()
      print("(",i,",",j,")")
      print('correlation value is',x)
      print()


# Now lets check the pairs one by one:
# 1. OverallQual-> we would like to keep this variable since it is highly correlated with the sale prices. Hence we would be better off removing the following:
# >* BsmtQual_codes
# >* KitchenQual_codes
# >* ExterQual_codes
# 
# 2. Yearbuilt: We would like to keep this in the model since its correlation with the SalePrice is high. Hence we would be better off removing:
# >* GarageYrBlt
# 
# Similarly we will be removing the following variables as well
# 
# 3. BsmtFullBath
# 4. 1stFlrSF
# 5. 2ndFlrSF
# 6. FullBath
# 7. TotRmsAbvGrd
# 8. SqFtPerRoom
# 9. GarageArea
# 
# 
# 
# 

# In[73]:


dropped_columns


# In[74]:


list4=['BsmtQual_codes','KitchenQual_codes','ExterQual_codes','GarageYrBlt','BsmtFullBath','1stFlrSF','2ndFlrSF','FullBath','TotRmsAbvGrd','SqFtPerRoom','GarageArea',]
train_final.drop(list4,axis=1,inplace=True)
test_final.drop(list4,axis=1,inplace=True)
for i in list4:
  dropped_columns.append(i)


# In[75]:


# we missed Fireplaces lets drop that variable as well.
train_final.drop(['Fireplaces'],axis=1,inplace=True)
test_final.drop(['Fireplaces'],axis=1,inplace=True)
dropped_columns.append('Fireplaces')


# In[76]:


# lets check the correlation heatmap once again
plt.figure(figsize = (30,30))
sns.heatmap(train_final.corr(),annot=True)
plt.show()


# In[77]:


# now we are ready for next step which is feature scaling and train test split
# but before that lets make sure that everything is in order
train_final.head()


# In[78]:


train_final.shape


# In[79]:


test_final.shape


# In[80]:


test_final.info()


# In[81]:


train_final.info()


# # Scaling the continuous variables

# From EDA we learnt that there are different scales of various features. 
# 
# Not scaling these features might result in serious biases in the final model.
# 
# So lets go ahead and scale the features using standard scaler.
# 
# In this we will be able to scale only those features which are not of object type so before that lets update the lists of categorical and continuous variables

# In[82]:


list5 = train_final.columns.drop(['SalePrice'])


# In[83]:


cat_cols=[]
cont_cols=[]
for i in list5:
  if train_final[i].dtypes =='object':
    cat_cols.append(i)
  else:
    cont_cols.append(i)


# In[84]:


#importing StandardScaler from SciKit Learn
from sklearn.preprocessing import StandardScaler


# In[85]:


scaler = StandardScaler()


# In[86]:


scaled_features = train_final.copy()
scaled_train = scaled_features[cont_cols]
scaled_train = scaler.fit_transform(scaled_train)


# In[87]:


df_tr=train_final.copy()
df_tr[cont_cols]=scaled_train


# In[88]:


scaled_features_test = test_final.copy()
scaled_test = scaled_features_test[cont_cols]
scaled_test = scaler.fit_transform(scaled_test)


# In[89]:


df_test=test_final.copy()
df_test[cont_cols]=scaled_test


# # One hot encoding or dummy encoding for the categorical variables

# In[90]:


df_tr_encoded = pd.get_dummies(df_tr, drop_first = True, columns = cat_cols )


# In[91]:


df_test_encoded = pd.get_dummies(df_test, drop_first = True, columns = cat_cols )


# In[92]:


df_tr_encoded.head()


# In[93]:


df_tr_encoded.shape


# In[94]:


df_test_encoded.shape


# Lets remove special characters from the column names and make them conducive for analysis

# In[95]:


df_tr_encoded.columns = df_tr_encoded.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'perc').str.replace('/', '_').str.replace('-', '_').str.replace('.', 'p').str.replace('[', '_').str.replace(']', '').str.replace('&', '').str.replace('$', '').str.replace('#', '')
df_test_encoded.columns = df_test_encoded.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'perc').str.replace('/', '_').str.replace('-', '_').str.replace('.', 'p').str.replace('[', '_').str.replace(']', '').str.replace('&', '').str.replace('$', '').str.replace('#', '')


# # Checking compatibility between the test and the train data
# 
# Here since we have different sets of data for both train and test set, we need to see if the values in categorical variables in the train and test are same. If not, this will cause the model to not run on the test dataset.
# 
# example.
# 
# suppose there is a Variable named Payment_methods. The unique values in train dataset are:
# 
# 1. credit_card
# 1. debit_card
# 1. cod
# 1. Wallet_paytm
# 
# after one hot / dummy encoding, this will transform into 4 variables:
# 
# 1. Payment_methods:credit_card
# 1. Payment_methods:debit_card
# 1. Payment_methods:cod
# 1. Payment_methods:Wallet_paytm
# 
# Now lets say that the test dataset has the following values
# 
# 1. credit_card
# 1. debit_card
# 1. cod
# 1. Wallet_freecharge
# 
# after one hot / dummy encoding, this will transform into 4 variables:
# 
# 1. Payment_methods:credit_card
# 1. Payment_methods:debit_card
# 1. Payment_methods:cod
# 1. Payment_methods:Wallet_freecharge.
# 
# So we can see above that there will be a mismatch and the regression wont run on the test dataset. for this purpose we would like to see which all variables are there in the test and not in train and viceversa.

# In[96]:


train_columns = df_tr_encoded.columns
train_columns


# In[97]:


test_columns = df_test_encoded.columns
test_columns


# Now here we can see that there are a lot of variables in the train which are not there in the test and vice versa could also be possible.

# In[98]:


train_col_list = df_tr_encoded.columns.sort_values()
test_col_list = df_test_encoded.columns.sort_values()


# Lets make the list of columns which are present in both the test and the train dataset 

# In[99]:


compatible_list = set(train_col_list).intersection(test_col_list)


# In[100]:


df_tr_encoded_2 = df_tr_encoded[compatible_list]


# In[101]:


df_tr_encoded_2.head()


# In[102]:


df_tr_encoded_2.shape


# In[103]:


df_test_encoded_2 = df_test_encoded[compatible_list]


# In[104]:


df_test_encoded_2.head()


# In[105]:


df_test_encoded_2.shape


# Now the test and train Dataframes are perfectly compatible. We will now proceed for creating a feature set and the outcome variable on the train dataset

# # Creating A feature set (X) and Outcome Variable (Y)

# In[106]:


import statsmodels.api as sm


# In[107]:


# copying all predictor variables into X and Target variable in Y
X = df_tr_encoded_2
Y = df_tr_encoded['SalePrice']


# In[108]:


X.head()


# In[109]:


Y.head()


# # Train Test Split

# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size = 0.2, random_state=42) 


# In[112]:


# invoking the LinearRegression function and find the bestfit model on training data
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(train_X, train_Y)


# In[113]:


regression_model.coef_


# In[114]:


# Let us explore the coefficients for each of the independent attributes

for i, col_name in enumerate(train_X.columns):
    print("The coefficient for",col_name, "is", regression_model.coef_[i])


# In[115]:


# Let us check the intercept for the model

intercept = regression_model.intercept_

print("The intercept for our model is", intercept)


# In[116]:


regression_model.score(train_X, train_Y)


# In[117]:


regression_model.score(test_X, test_Y)


# In[118]:


# finding RSME
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(train_Y, regression_model.predict(train_X)))


# In[119]:


np.sqrt(mean_squared_error(test_Y, regression_model.predict(test_X)))


# # Linear Regression using Statsmodels
# 
# * using statsmodels.formula.api => this does not require us to add a constant to the train values
# 
# R^2 is not a reliable metric as it always increases with addition of more attributes even if the attributes have no influence on the predicted variable. 
# 
# Instead we use adjusted R^2 which removes the statistical chance that improves R^2. 
# 
# Scikit does not provide a facility for adjusted R^2, so we use statsmodel, a library that gives results similar to what you obtain in R language. This library expects the X and Y to be given in one single dataframe

# In[120]:


data_train = pd.concat([train_X, train_Y], axis=1)
data_train.head()


# In[121]:


data_train.columns


# In[122]:


reg_expression = 'SalePrice ~ MSZoning_RH+HouseStyle_SFoyer+MSZoning_FV+Neighborhood_Blueste+Exterior1st_Stucco+MSZoning_RL+MSSubClass_85+Exterior1st_HdBoard+LotShape_IR3+BldgType_Twnhs+LotConfig_CulDSac+Neighborhood_NridgHt+RoofStyle_Shed+Neighborhood_NoRidge+GarageType_BuiltIn+SaleType_WD+MoSold_10+SaleType_Oth+Neighborhood_BrkSide+Neighborhood_Somerst+Condition2_Norm+MSSubClass_60+Exterior2nd_Stone+Condition1_RRNn+WoodDeckSF+MiscFeature_Othr+Neighborhood_StoneBr+BsmtCond_Po+MSSubClass_45+MasVnrType_None+Neighborhood_Sawyer+LotConfig_Inside+MoSold_12+GarageFinish_RFn+Exterior2nd_Plywood+Neighborhood_Crawfor+Alley_Pave+Foundation_PConc+YearRemodAdd+MSSubClass_50+LotConfig_FR2+Neighborhood_ClearCr+BsmtFinSF1+Neighborhood_NPkVill+Electrical_FuseF+SaleType_CWD+YrSold_2010+TotalBsmtSF+GarageType_Basment+RoofStyle_Hip+Neighborhood_MeadowV+MSSubClass_90+Condition1_RRAe+CentralAir_Y+MoSold_9+Condition1_PosN+MSSubClass_40+Condition1_RRAn+Electrical_FuseP+Exterior2nd_Wd_Shng+Electrical_SBrkr+Foundation_CBlock+Heating_Grav+MSSubClass_80+Neighborhood_Edwards+LandSlope_Mod+Neighborhood_Timber+MasVnrType_Stone+HouseStyle_1Story+OverallQual+SaleType_Con+Foundation_Stone+FireplaceQu_codes+Neighborhood_NAmes+Total_Bathrooms+Exterior1st_CBlock+Exterior1st_MetalSd+Neighborhood_BrDale+YearBuilt+HeatingQC_TA+PavedDrive_P+Fence_codes+Neighborhood_CollgCr+HouseStyle_2Story+Condition2_PosA+Foundation_Slab+SaleType_New+MSSubClass_75+LandSlope_Sev+Condition1_PosA+MoSold_5+Heating_Wall+OpenPorchSF+LotFrontage+GrLivArea+HeatingQC_Po+Neighborhood_SawyerW+BsmtExposure_Mn+LotArea+GarageType_Attchd+Condition2_PosN+BsmtFinType1_codes+BldgType_2fmCon+BldgType_TwnhsE+Exterior2nd_CmentBd+GarageType_CarPort+RoofMatl_TarGrv+LotConfig_FR3+LotShape_IR2+Exterior1st_Plywood+MoSold_4+RoofMatl_WdShngl+RoofStyle_Gable+HalfBath+LandContour_Lvl+Neighborhood_Mitchel+Exterior1st_WdShing+Exterior2nd_Wd_Sdng+Foundation_Wood+Condition1_Norm+MSSubClass_180+MasVnrType_BrkFace+YrSold_2007+GarageType_Detchd+Alley_Not_Applicable+Exterior2nd_HdBoard+Exterior2nd_ImStucc+BsmtExposure_No+MiscFeature_Not_Applicable+SaleType_ConLD+SaleType_ConLw+Exterior2nd_Brk_Cmn+Street_Pave+Neighborhood_OldTown+MiscFeature_Shed+GarageFinish_Unf+RoofMatl_WdShake+Exterior2nd_Stucco+Neighborhood_Veenker+BsmtCond_Gd+Exterior1st_Wd_Sdng+LotShape_Reg+MoSold_11+MSZoning_RM+LandContour_HLS+MSSubClass_70+Exterior2nd_MetalSd+GarageCars+MoSold_6+LandContour_Low+Heating_GasW+MasVnrArea+YrSold_2008+Exterior2nd_VinylSd+MSSubClass_30+Exterior2nd_AsphShn+BldgType_Duplex+Exterior2nd_CBlock+HouseStyle_2p5Unf+RoofStyle_Gambrel+PavedDrive_Y+Neighborhood_SWISU+HeatingQC_Fa+MoSold_3+Exterior1st_BrkFace+HouseStyle_1p5Unf+MoSold_7+MSSubClass_160+Neighborhood_Gilbert+Neighborhood_NWAmes+Exterior1st_BrkComm+MoSold_8+BsmtExposure_Gd+Condition1_Feedr+YrSold_2009+Exterior1st_AsphShn+SaleType_ConLI+HeatingQC_Gd+Exterior1st_VinylSd+HouseStyle_SLvl+MSSubClass_120+BsmtCond_TA+Exterior2nd_BrkFace+Neighborhood_IDOTRR+BsmtUnfSF+RoofStyle_Mansard+MSSubClass_190+SaleCondition_codes+Exterior1st_CemntBd+Condition1_RRNe+Condition2_Feedr+GarageQual_codes+MoSold_2'


# Lets check the coefficients of the variables in the regression equation

# In[123]:


import statsmodels.formula.api as smf
model1 = smf.ols(formula=reg_expression, data=data_train).fit()
# displaying first 5 parameters
model1.params.head()


# In the above , though have the coefficients of the  regression variables, we dont know if these coefficients are significant or not. So lets print the Model summary. Here id the P values are greater than 0.05 that would mean that the coefficient is not significant in predicting the target variable.
# 
# Hence we would drop such variables, this will be reflected in the decrease of Mean absolute error and the RMSE

# In[124]:


print(model1.summary())


# # Calculating the Mean Square Error

# In[125]:


# calculating the Mean square error
mse = np.mean((model1.predict(data_train.drop('SalePrice',axis=1))- data_train['SalePrice'])**2)


# In[126]:


np.sqrt(mse)


# Testing the model on the test data

# In[127]:


data_test = pd.concat([test_X, test_Y], axis=1)
data_test.head()


# In[128]:


# calculating the Mean square error
mse_test = np.mean((model1.predict(data_test.drop('SalePrice',axis=1))- data_test['SalePrice'])**2)


# In[129]:


# RMSE for the test data
np.sqrt(mse_test)


# Here we can see that RMSE values for the test and train samples are close. However there are variables where p-values for a lot of coefficients are very high hence lets see if by removing them the RMSE gets better.
# 
# Hence we will remove those variables from the linear regression expression where the P value is greater than 0.05

# In[130]:


reg_expression2 = 'SalePrice ~ Neighborhood_Blueste+LotShape_IR3+Neighborhood_NridgHt+Neighborhood_NoRidge+GarageType_BuiltIn+WoodDeckSF+Neighborhood_StoneBr+Neighborhood_Crawfor+YearRemodAdd+MSSubClass_50+GarageType_Basment+Neighborhood_MeadowV+CentralAir_Y+Condition1_PosN+Condition1_RRAn+Electrical_SBrkr+Neighborhood_Edwards+HouseStyle_1Story+OverallQual+FireplaceQu_codes+Total_Bathrooms+HeatingQC_TA+HouseStyle_2Story+Condition2_PosA+Foundation_Slab+LandSlope_Sev+Condition1_PosA+MoSold_5+LotFrontage+GrLivArea+LotArea+GarageType_Attchd+Condition2_PosN+MoSold_4+LandContour_Lvl+Condition1_Norm+GarageType_Detchd+BsmtCond_Gd+LandContour_HLS+GarageCars+MoSold_6+MSSubClass_30+HouseStyle_1p5Unf+MoSold_7+BsmtExposure_Gd+HeatingQC_Gd+BsmtCond_TA+SaleCondition_codes+GarageQual_codes'


# In[131]:


model2 = smf.ols(formula=reg_expression2,data=data_train).fit()
# Displaying top 5 parameters
model2.params.head()


# In[132]:


print(model2.summary())


# In[133]:


# Calculating MSE
MSE2 = np.mean((model2.predict(data_train.drop(['SalePrice'],axis=1))- data_train['SalePrice'])**2)


# In[134]:


#RMSE
np.sqrt(MSE2)


# In[135]:


# MSE on the test data
MSE2_test = np.mean((model2.predict(data_test.drop(['SalePrice'],axis=1))- data_test['SalePrice'])**2)


# In[136]:


# RMSE on the test Data
np.sqrt(MSE2_test)


# We have seen that the RMSE has not improved, instead, it has become worse. However, from the regression equation above lets further remove those Variables where P values exceed 0.05, and then see if the values improve. Else we will select model1
# 
# 

# In[137]:


reg_expression3 = 'SalePrice ~ Neighborhood_Blueste+LotShape_IR3+Neighborhood_NridgHt+Neighborhood_NoRidge+GarageType_BuiltIn+WoodDeckSF+Neighborhood_StoneBr+Neighborhood_Crawfor+YearRemodAdd+GarageType_Basment+Neighborhood_MeadowV+CentralAir_Y+Condition1_PosN+Neighborhood_Edwards+HouseStyle_1Story+OverallQual+FireplaceQu_codes+Total_Bathrooms+HeatingQC_TA+Foundation_Slab+MoSold_5+GrLivArea+LotArea+GarageType_Attchd+Condition2_PosN+MoSold_4+LandContour_Lvl+Condition1_Norm+GarageType_Detchd+BsmtCond_Gd+GarageCars+MoSold_6+MSSubClass_30+MoSold_7+BsmtExposure_Gd+HeatingQC_Gd+BsmtCond_TA+SaleCondition_codes+GarageQual_codes'


# In[138]:


model3 = smf.ols(formula=reg_expression3,data=data_train).fit()
model3.params.head()


# In[139]:


print(model3.summary())


# In[140]:


# Calculating MSE
MSE3 = np.mean((model3.predict(data_train.drop(['SalePrice'],axis=1))- data_train['SalePrice'])**2)


# In[141]:


#RMSE
np.sqrt(MSE3)


# In[142]:


# MSE on the test data
MSE3_test = np.mean((model3.predict(data_test.drop(['SalePrice'],axis=1))- data_test['SalePrice'])**2)


# In[143]:


# RMSE on the test Data
np.sqrt(MSE3_test)


# In[144]:


# lets drop one more variable where the p value is greater than 0.05 and see if the RMSE further improves:
reg_expression4 = 'SalePrice ~ Neighborhood_Blueste+LotShape_IR3+Neighborhood_NridgHt+Neighborhood_NoRidge+GarageType_BuiltIn+WoodDeckSF+Neighborhood_StoneBr+Neighborhood_Crawfor+YearRemodAdd+GarageType_Basment+Neighborhood_MeadowV+CentralAir_Y+Condition1_PosN+Neighborhood_Edwards+HouseStyle_1Story+OverallQual+FireplaceQu_codes+Total_Bathrooms+HeatingQC_TA+Foundation_Slab+MoSold_5+GrLivArea+LotArea+GarageType_Attchd+Condition2_PosN+MoSold_4+LandContour_Lvl+Condition1_Norm+GarageType_Detchd+BsmtCond_Gd+GarageCars+MSSubClass_30+MoSold_7+BsmtExposure_Gd+HeatingQC_Gd+BsmtCond_TA+SaleCondition_codes+GarageQual_codes'


# In[145]:


model4 = smf.ols(formula=reg_expression4,data=data_train).fit()
model4.params.head()


# In[146]:


print(model4.summary())


# In[147]:


RMSE4 = np.sqrt(np.mean((model4.predict(data_train.drop(['SalePrice'],axis=1))- data_train['SalePrice'])**2))
RMSE4


# In[148]:


RMSE4_test = np.sqrt(np.mean((model4.predict(data_test.drop(['SalePrice'],axis=1))- data_test['SalePrice'])**2))
RMSE4_test


# We get the best RMSE scores from the model 1 hence we will be using model 1 

# In[ ]:





# # Regularisation using Ridge and Lasso
# 
# Lets go for regularisation to further improve the regression models
# we will be doing:
# L1 regularisation: also called Lasso
# L2 regularisation: also called Ridge

# In[149]:


# Import linear models
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# Create lasso and ridge objects
lasso = linear_model.Lasso()
ridge = linear_model.Ridge()
# Fit the models
lasso.fit(train_X, train_Y)
ridge.fit(train_X, train_Y)
# Print scores, MSE, and coefficients
print("lasso score:", lasso.score(train_X, train_Y))
print("ridge score:",ridge.score(train_X, train_Y))
print("lasso RMSE:", np.sqrt(mean_squared_error(test_Y, lasso.predict(test_X))))
print("ridge RMSE:", np.sqrt(mean_squared_error(test_Y, ridge.predict(test_X))))
print("lasso coef:", lasso.coef_)
print("ridge coef:", ridge.coef_)


# In[150]:


# Import library for visualization
import matplotlib.pyplot as plt
coefsLasso = []
coefsRidge = []
# Build Ridge and Lasso for 200 values of alpha and write the coefficients into array
alphasLasso = np.arange (0, 25, 0.1)
alphasRidge = np.arange (0, 250, 1)
for i in range(250):
    lasso = linear_model.Lasso(alpha=alphasLasso[i])
    lasso.fit(train_X, train_Y)
    coefsLasso.append(lasso.coef_)
    ridge = linear_model.Ridge(alpha=alphasRidge[i])
    ridge.fit(train_X, train_Y)
    coefsRidge.append(ridge.coef_[0])

# Build Lasso and Ridge coefficient plots
plt.figure(figsize = (16,7))

plt.subplot(121)
plt.plot(alphasLasso, coefsLasso)
plt.title('Lasso coefficients')
plt.xlabel('alpha')
plt.ylabel('coefs')

plt.subplot(122)
plt.plot(alphasRidge, coefsRidge)
plt.title('Ridge coefficients')
plt.xlabel('alpha')
plt.ylabel('coefs')

plt.show()


# In[151]:


# model1 predicts the best RMSE scores for OLS method
test_predicted_ols = model1.predict(df_test_encoded_2)
test_predicted_ols


# In[152]:


test_predicted_ridge = ridge.predict(df_test_encoded_2)
test_predicted_ridge


# Among the 3 i.e., OLS, Ridge and Lasso, Ridge has the best RMSE scores. However after submission with ridge, the score is 0.16153
# hence choosing to go with OLS score now.

# In[153]:


test_pred = test_predicted_ols.copy()


# In[154]:


test_pred


# In[155]:


sns.distplot(data_train['SalePrice'],color = 'blue', label='train')
sns.distplot(test_predicted_ols,color = 'red', label='test')
sns.distplot(test_predicted_ridge,color = 'green', label='test')
plt.show()


# Well it seems that we have a decent prediction. The distribution of OLS predicion SalePrices looks closer to the train dataset.

# # Submission

# In[156]:


submission = test_original['Id']
test_pred = np.expm1(test_pred)
test_pred = pd.DataFrame(test_pred)
submission = pd.concat([submission,test_pred],axis=1)
submission.rename({0:'SalePrice'},axis=1,inplace=True)


# In[157]:


submission.head()


# In[158]:


submission.to_csv("result.csv", index = False, header = True)


# In[ ]:




