#!/usr/bin/env python
# coding: utf-8

# My first Kernels-practice Kernels

# <a id="1"></a>
# <div style="padding:20px;
#             color:white;
#             margin:10;
#             font-size:200%;
#             text-align:left;
#             display:fill;
#             border-radius:5px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700">If you like this work, please upvote this kernel  and share the kernel with others so we can all benefit</div>
# 
# https://www.linkedin.com/in/krishnaraj-palaniselvaraj/

# # What is Feature Engineering ?
# 
# Applying your knowledge of the data and the model you are using, to create better feature to train your model with.
# 
# -Which features should I use?
# 
# -Do I need to transform these features in some way?
# 
# -How do I handle missing data?
# 
# -Should I create new features from the existing ones?

# # The Curse of Dimensionality
# 
# Too many features can be a problem that will lead to sparse data. And have to be remember that every feature is a new dimension -equal- more compute resources.
# 
# Much of feature engineering is selecting the features most relevant to the problem at hand, this is often where domain knowledge comes into play.
# 
# Unsupervised dimensionality reduction techniques can also be employed to distill many features into fewer features : Using Principal Component Analysis (PCA) AND/OR K-Means.

# **Important step as Data Scientist is to work with Data. Below is the step by step to work with CSV data:**
# 
# Understand the problem. We'll look at each variable and do a philosophical analysis about their meaning and importance for this problem.
# 
# Univariable study. We'll just focus on the dependent variable ('SalePrice') and try to know a little bit more about it.
# 
# Multivariate study. We'll try to understand how the dependent variable and independent variables relate.
# 
# Feature Engineering We'll clean the dataset and handle the missing data, outliers and categorical variables.
# 
# Test assumptions. We'll check if our data meets the assumptions required by most multivariate techniques.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir('../input'))


# In[2]:


df_train = pd.read_csv('../input/train.csv')


# In[3]:


df_train.info()


# In[4]:


df_train.head(10)


# In[5]:


df_train.columns


# In[6]:


#check if there any zero in minimal of the price
df_train['SalePrice'].describe()


# In[7]:


#https://seaborn.pydata.org/tutorial/distributions.html#plotting-univariate-distributions
sns.distplot(df_train['SalePrice'], bins=20, rug=True);


# In[8]:


#Skew = ambience distributions data (0=evenly distributed)
#Kurt = to check the outlier data (3=standart value)
print("Skewness: %f" %df_train['SalePrice'].skew())
print("Kurtosis: %f" %df_train['SalePrice'].kurt())


# In[9]:


#Annotation Heatmap https://seaborn.pydata.org/examples/heatmap_annotation.html

Heatmap_Annotation = df_train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(Heatmap_Annotation, 
            vmax=.8, square=True)


# In[10]:


#Diagonal Correlation https://seaborn.pydata.org/examples/many_pairwise_correlations.html

sns.set(style='white')
Diagonal_Corr = df_train.corr()
mask = np.zeros_like(Diagonal_Corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(Diagonal_Corr, mask=mask, cmap=cmap,
           vmax=.8, center=0,square=True,
           linewidths=5, cbar_kws={"shrink":.5})


# In[11]:


#SalePrice Correlation Matrix
k=10
sns.set(font_scale=1.25)
corrmat=df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.heatmap(cm, annot=True, square=True,
           fmt='.2f', annot_kws={'size':10},
           yticklabels=cols.values,
           xticklabels=cols.values)


# In[12]:


#Scatterplot https://seaborn.pydata.org/tutorial/regression.html
cols = ['SalePrice', 'OverallQual','GrLivArea',
       'GarageCars', 'GarageArea', 'TotalBsmtSF',
       '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
       'YearBuilt']
sns.pairplot(df_train[cols], size =5)


# In[13]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[14]:


f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=total.index, y=total)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values',fontsize=15)
plt.title('Percent missing data by feature',fontsize=15)


# In[15]:


#drop columns/keys that have more than 50% of null values
df_train = df_train.drop((missing_data[missing_data['Percent'] > 50 ]).index,1)
df_train.isnull().sum().sort_values(ascending=False) #check


# In[16]:


#FireplaceQu : data description says Null means "no fireplace"
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('None')


# In[17]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
# 
# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
# 
# 

# In[18]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2',
            'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond'):
    df_train[col] = df_train[col].fillna('None')


# In[19]:


#GarageYrBlt replacing missing data with 0
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(0)


# In[20]:


df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)


# In[21]:


#Electrical : It has one NA value. 
#Since this feature has mostly 'SBrkr', we can set that for the missing value.
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])


# In[22]:


df_train.isnull().sum().sort_values(ascending=False) #check


# In[23]:


#SalePrice Correlation Matrix
k=10
sns.set(font_scale=1.5)
corrmat=df_train.corr()
cols = corrmat.nsmallest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.heatmap(cm, annot=True, square=True,
           fmt='.2f', annot_kws={'size':10},
           yticklabels=cols.values,
           xticklabels=cols.values)




# In[24]:


#deleting uncorrelate colomns
Uncor = ['EnclosedPorch', 'OverallCond', 
        'YrSold', 'LowQualFinSF', 'Id', 
         'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']
df_train.drop(Uncor, axis=1, inplace=True)
df_train.info()


# In[25]:


#More features engineering
#Transforming some numerical variables that are really
df_train['MSSubClass'] = df_train['MSSubClass'].astype(str)
df_train['MoSold'] = df_train['MoSold'].astype(str)


# In[26]:


# Adding total sqfootage feature 
df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']


# In[27]:


#Univariate analysis
#Detect and exclude outlier in numeric dtype
#low 0.05 and high 0.90 quantile
from pandas.api.types import is_numeric_dtype
def remove_outlier(df_train):
    low = .05
    high = .90
    quant_df = df_train.quantile([low, high])
    for name in list(df_train.columns):
        if is_numeric_dtype(df_train[name]):
            df_train = df_train[(df_train[name] > quant_df.loc[low, name]) & (df_train[name] < quant_df.loc[high, name])]
    return df_train

remove_outlier(df_train).head()


# In[28]:


#check the standardizing data
for name in list(df_train.columns):
    if is_numeric_dtype(df_train[name]):
        saleprice_scaled = StandardScaler().fit_transform(df_train[name][:,np.newaxis]);
        low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:5]
        high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-5:]
        print('outer range (low) of the distribution:',name)
        print(low_range)
        print('\nouter range (high) of the distribution:',name)
        print(high_range)


# In[29]:


#Bivariate/Multivariate outlier checking with scatter plot
for name in list(df_train.columns):
    if is_numeric_dtype(df_train[name]):
        data = pd.concat([df_train['SalePrice'], df_train[name]], axis=1)
        data.plot.scatter(x=name, y='SalePrice', ylim=(0,800000))


# In[30]:


#Dropping the outlier
#Only on the Feature that perform linear regression dot in the scatter plot
df_train = df_train.drop(df_train[df_train['LotFrontage'] > 300].index)
df_train = df_train.drop(df_train[df_train['LotArea'] > 60000].index)
df_train = df_train.drop(df_train[(df_train['OverallQual'] > 9) & (df_train['SalePrice'] < 200000)].index)
df_train = df_train.drop(df_train[df_train['MasVnrArea'] > 1500].index)
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index)
df_train = df_train.drop(df_train[df_train['1stFlrSF'] > 2500].index)
df_train = df_train.drop(df_train[df_train['BsmtFullBath'] > 2.5].index)
df_train = df_train.drop(df_train[df_train['HalfBath'] > 1.5].index)
df_train = df_train.drop(df_train[df_train['BedroomAbvGr'] > 4].index)
df_train = df_train.drop(df_train[df_train['TotRmsAbvGrd'] > 13].index)
df_train = df_train.drop(df_train[df_train['Fireplaces'] > 2.5].index)
df_train = df_train.drop(df_train[df_train['GarageCars'] > 3].index)
df_train = df_train.drop(df_train[df_train['GarageArea'] >= 1250].index)


# In[31]:


#skewed features
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(20)


# In[32]:


sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[33]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #df_train[feat] += 1
    df_train [feat] = boxcox1p(df_train[feat], lam)
    
#df_train[skewed_features] = np.log1p(df_train[skewed_features])


# In[34]:


#check
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[35]:


from sklearn.preprocessing import LabelEncoder
colomns = df_train.dtypes[df_train.dtypes == "object"].index
# process columns, apply LabelEncoder to categorical features
for name in colomns:
    lbl = LabelEncoder() 
    lbl.fit(list(df_train[name].values)) 
    df_train[name] = lbl.transform(list(df_train[name].values))

# shape        
print('Shape of df_train: {}'.format(df_train.shape))


# In[36]:


#Dummy categorical features
df_train = pd.get_dummies(df_train)
print(df_train.shape)
df_train.head(20) #please compare the data after engineering and before engineering


# <a id="1"></a>
# <div style="padding:20px;
#             color:white;
#             margin:10;
#             font-size:200%;
#             text-align:left;
#             display:fill;
#             border-radius:5px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700">Thank you All, I will appreciate you to check and share feedback on my other notebooks in your free time</div>
# 
# 
# 
# https://www.kaggle.com/code/krishnaraj30/visualization-decision-tree-classification
# 
# https://www.kaggle.com/code/krishnaraj30/decision-tree-sklearn-hyper-parameter-tuning
# 
# https://www.kaggle.com/code/krishnaraj30/house-price-explore-with-python-visualization
# 
# https://www.kaggle.com/code/krishnaraj30/clustering-segmentation-k-means
# 
# https://www.kaggle.com/code/krishnaraj30/exploratory-data-analysis-matplotlib-seaborn
