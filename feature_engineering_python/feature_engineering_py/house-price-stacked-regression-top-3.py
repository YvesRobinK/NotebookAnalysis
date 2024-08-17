#!/usr/bin/env python
# coding: utf-8

# # House price: feature select, stacking, blending (top 3%)
# (Hoang Pham)
# 
# * [**1. Getting familiar with data**](#1)
#     * [**1.1 Missing values**](#1.1)
#     * [**1.2 Numeric features**](#1.2)
#     * [**1.3 Categorical features**](#1.3)
#     * [**1.4 Target column distribution**](#1.4)
# * [**2. Feature engineering**](#2)
#     * [**2.1 Binning continuous features**](#2.1)
#     * [**2.2 Construct new useful features**](#2.2)
#     * [**2.3 Feature selection**](#2.3)
#         * [**2.3.1 Select categorical features**](#2.3.1)
#         * [**2.3.2 Mismatched value between train & test set in categorical features**](#2.3.2)
#         * [**2.3.3 Select contunious features**](#2.3.3)
#     * [**2.4 Features transformation**](#2.4)
#         * [**2.4.1 Highly skewed numeric features**](#2.4.1)
#         * [**2.4.2 One-hot encoding categorical features**](#2.4.2)
# * [**3. Modeling**](#3)
#     * [**3.1 Base models**](#3.1)
#     * [**3.2 Stacking model**](#3.2)
#     * [**3.3 Blending model**](#3.3)
# * [**4 Submision**](#4)

# ## Introduction
# Kaggle describes this competition as follows:
# 
# Ask a home buyer to describe their dream house, and they probably won’t begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition’s dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# ## Excutive summary:
# I started this competition with the purpose of obtaining an intuitive knowledge about feature selection, stacking model and blending model to solve the regression problem. After completing the competition, there are a lot of interesting and useful things that I'd like to share with all of Kagglers.
# - For features selection in this kernel, I use Pearson's correlation values to choose the numeric features based on their linear association with the target. And "Forward feature selection" technique is used for choosing suitable categorical features
# - While using based models only, the average values of rmse is nearly 0.13 on LB. But when I applied stacking model, the rmse value moved down directly to only 0.124 on LB. It worths trying!
# - I've also implement the short version of this notebook for someone who want to go through a quick start of this competion. Link: [Quick start house price competition](https://www.kaggle.com/hoangphamviet/beginner-quick-start-house-price-competition)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats
from sklearn import preprocessing
from sklearn import feature_selection
import warnings
warnings.filterwarnings('ignore')
SEED = 42


# <a name='1'></a>
# # 1. Getting familiar with data
# - Training set has 1460 rows, testing set has 1459 rows
# - There are 81 features in training set and 80 features in testing set
# - One extra feature in training set is "SalePrice" which is the target feature
# - "df_all" the is concatenated dataframe between training and testing data for more convenience preprocessing. And we also should be careful about the "data leak" problem

# In[2]:


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:1459], all_data.loc[1460:]

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
y_train = df_train.SalePrice
id_val = df_train.Id
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_all = concat_df(df_train, df_test).drop(['SalePrice', 'Id'], axis=1)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 

dfs = [df_train, df_test]

print(f'Number of Training Examples = {df_train.shape[0]}')
print(f'Number of Test Examples = {df_test.shape[0]}\n')
print(f'Training X Shape = {df_train.shape}')
print(f'Training y Shape = {df_train["SalePrice"].shape[0]}\n')
print(f'Test X Shape = {df_test.shape}')
print(f'Test y Shape = {df_test.shape[0]}\n')
print(df_train.columns)


# In[3]:


df_train.head()


# <a name='1.1'></a>
# ## 1.1 Missing values

# In[4]:


# Visualize the general missing values of data
msno.matrix(df_all)
plt.show()


# In[5]:


for df in dfs:
    print(f'Only features contained missing value in {df.name}')
    temp = df.isnull().sum()
    print(temp.loc[temp!=0], '\n')


# - For features having missing value smaller than 100 -> I'll fill numeric features with the corresponding median & categorical features with the corresponding most frequent values
# - For features having missing value larger than 1000 -> Removing these features might be a good choice
# - For other features having missing value -> I'll fill them with "Null" value (b.c all the other features are object features)

# In[6]:


null_features = df_all.isnull().sum()

# For features having smaller than 100 missing values
null_100 = df_all.columns[list((null_features < 100) & (null_features != 0))]
num = df_all[null_100].select_dtypes(include=np.number).columns
non_num = df_all[null_100].select_dtypes(include='object').columns
# Numerous features
df_all[num] = df_all[num].apply(lambda x: x.fillna(x.median()))
# Object features
df_all[non_num] = df_all[non_num].apply(lambda x: x.fillna(x.value_counts().index[0]))

# For features having larger than 1000 missing values
null_1000 = df_all.columns[list(null_features > 1000)]
df_all.drop(null_1000, axis=1, inplace=True)
df_all.drop(['GarageYrBlt', 'LotFrontage'], axis=1, inplace=True)


# In[7]:


# For other features having missing values
# GarageCond
df_all['GarageCond'] = df_all['GarageCond'].fillna('Null')
# GarageFinish
df_all['GarageFinish'] = df_all['GarageFinish'].fillna('Null')
# GarageQual
df_all['GarageQual'] = df_all['GarageQual'].fillna('Null')
# GarageType
df_all['GarageType'] = df_all['GarageType'].fillna('Null')


# In[8]:


# Update training & testing data
df_train, df_test = divide_df(df_all)
df_train = pd.concat([df_train, y_train], axis=1)

# Checking existing missing value or not
print('If the result is zero means not exist any missing values in dataset')
print(df_all.isnull().any().sum())


# <a name='1.2'></a>
# ## 1.2 Numeric features
# - There are totally 34 numeric features in the dataset
# - We'll visualize the features having the high level of correlation with target feature (SalePrice)

# In[9]:


# Seeing the correlation between features and target
df_train_corr = df_train.corr()['SalePrice'].sort_values(ascending=False).drop(['SalePrice'])
df_train_corr.head(10)


# Visualize the 16 features which has the highest correlation with the target

# In[10]:


fig, axs = plt.subplots(4, 4, figsize=(18, 18))
plt.subplots_adjust(right=1.3, top=1.3)
axs = axs.flatten()
for i, col in enumerate(list(df_train_corr.index[:16])):
    sns.scatterplot(y='SalePrice', x=col, ax=axs[i], data=df_train)
    axs[i].set_xlabel('SalePrice')
    axs[i].set_ylabel(col)
plt.show()


# OverallQual is a categorical feature. Therefore, the box plot should be suitable in this case to clearly show the high correlation characteristic between Overall quality ("OverallQual") of the house with the its price ("SalePrice").

# In[11]:


# Corr of "OverallQual": 0.7909
fig = plt.figure(figsize=(8, 8))
sns.boxplot(df_train['OverallQual'], df_train['SalePrice'])
plt.show()


# In[12]:


# Corr of "GrLivArea": 0.708
fig = plt.figure(figsize=(8, 8))
sns.scatterplot(df_train['GrLivArea'], df_train['SalePrice'])
plt.show()


# <a name='1.3'></a>
# ## 1.3 Categorical features
# - There are totally 38 categorical features in the dataset
# - Be careful that there are some categorical features in training dataset containing some values which do not exist in  the same features of testing dataset (E.g "Condition2" feature). We'll detect them and fix them later

# In[13]:


df_train_cate = df_train.select_dtypes(include=['object', 'category'])
df_train_cate.head()


# - Now we'll visualize the relationship between categorical features and target feature (SalePrice) using violin plot
# - Violin plot is effective to categorical features, it shows some important statistic terms so that we can compare between each other
# 
# <img src="https://miro.medium.com/max/650/1*TTMOaNG1o4PgQd-e8LurMg.png" style="width:400px;height:400px;">

# In[14]:


data = pd.melt(pd.concat([df_train_cate, y_train], axis=1),
               id_vars=['SalePrice'], value_vars=df_train_cate.columns, var_name='features')
g = sns.FacetGrid(data, col='features', col_wrap=2, sharex=False, sharey=False, size=5)
g.map(sns.violinplot, 'value', 'SalePrice')


# <a name='1.4'></a>
# ## 1.4 Target column distribution
# - Target feature is a heavy-tailed distribution. It'd be problematic to input the small-range features to predict the large-range feature (SalePrice). Therefore, "Box-cox transformation" technique to transform target feature to normal distribution might be appropriate in this occasion
# - There're also some large-scale input features & we'll normalize them later 👍👍

# In[15]:


# There're some features having the mean smaller than 1 -> Problematic :<
df_train.describe()


# In[16]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
y_train.hist(bins=100, ax=ax1)
ax1.set_ylabel('Occurences')
ax1.set_xlabel('SalePrice')

stats.probplot(y_train, dist=stats.norm, plot=ax2)
ax2.set_ylabel('SalePrice')
plt.show()


# - Target feature is a heavy-tailed distribution --> So Box-Cox transformation should be useful to bring the target from heavy-tailed to normal distribution

# In[17]:


# Using Box-Cot transformation on target feature
org_y_train = y_train
y_train = pd.Series(stats.boxcox(y_train, lmbda=0), name='SalePrice')

# Visualize target after box-cox transformation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
y_train.hist(bins=100, ax=ax1)
ax1.set_ylabel('Occurences')
ax1.set_xlabel('SalePrice')

stats.probplot(y_train, dist=stats.norm, plot=ax2)
ax2.set_ylabel('SalePrice')
plt.show()


# **Skewness**
# - Is the degree of distortion from the symmetrical normal curve --> Skewness of normal distribution is "0"
# - **Positive skewness** means the tail on the right side of the distribution is longer and fatter
# - **Negative skewness** means the tail on the left side of the ditribution is longer and fatter
# 
# **Kurtosis**
# - In probability theory and statistics, **Kurtosis** is the measure of extreme values (outliers) presented in the distribution

# In[18]:


# Compute Skewness & Kurtosis
print(f'Skewness before transformation: {stats.skew(org_y_train)}')
print(f'Kurtosis before transformation: {stats.kurtosis(org_y_train)}\n')

print(f'Skewness after transformation: {stats.skew(y_train)}')
print(f'Kurtosis after transformation: {stats.kurtosis(y_train)}')


# <a name='2'></a>
# # 2. Feature engineering

# <a name='2.1'></a>
# ## 2.1 Binning continuous features
# - Using "Bin" technique to all features representing "year" value (Ex: 2000, 1999,...)
# - And after that encoding them into continuous features

# In[19]:


# Using binned technique for "YearBuilt", "YearRemodAdd" & "YrSold"
df_all['YearBuilt'] = pd.qcut(df_all['YearBuilt'], 10, duplicates='drop')
df_all['YearRemodAdd'] = pd.qcut(df_all['YearRemodAdd'], 10, duplicates='drop')
df_all['YrSold'] = pd.qcut(df_all['YrSold'], 10, duplicates='drop')


# In[20]:


# Encode categorical features to numeric feature
for cate_col in ['YearBuilt', 'YearRemodAdd', 'YrSold']:
    df_all[cate_col] = preprocessing.LabelEncoder().fit_transform(df_all[cate_col].values)


# Next we'll visualize the year features after applying the binning technique

# In[21]:


fig, axs = plt.subplots(2, 1, figsize=(15, 10))
sns.countplot(df_all['YearBuilt'], ax=axs[0])
sns.countplot(df_all['YearRemodAdd'], ax=axs[1])
plt.show()


# And then transform some numeric features that are actually the categorical feature

# In[22]:


# Transform numeric features that are really the categorical features
df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)
df_all['OverallCond'] = df_all['OverallCond'].astype(str)
df_all['MoSold'] = df_all['MoSold'].astype(str)


# <a name='2.2'></a>
# ## 2.2 Construct new useful features
# - There are some features that we can concatunate them together to get more useful features
# - After constructing new features, all the recipe features might be removed b.c these features and the new one both represent the same type of infomation. Therefore, they would not be more effective to be together than to be alone

# In[23]:


# Generating new features
# Total square foot
df_all['TotalSF'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']

# Total number of bathroom
df_all['TotalBath'] = (df_all['FullBath'] + (0.5 * df_all['HalfBath']) +
                               df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath']))
df_all['TotalBsmtbath'] = df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath'])

# Total square feet of porch in a house
df_all['TotalPorchSF'] = (df_all['OpenPorchSF'] + df_all['3SsnPorch'] +
                            df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF'])

# Check the exist of each infrastructure (Ex: basement, bath,...) in a house
df_all['IsRemodel'] = df_all[['YearBuilt', 'YearRemodAdd']].apply(lambda x: 1 if x[0] != x[1] else 0, axis=1)
df_all['HasPool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['Has2ndFloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasBsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[24]:


# Drop all the recipe features
remove_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'FullBath', 'HalfBath', 'BsmtFullBath',
              'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
df_all.drop(remove_cols, axis=1, inplace=True)


# In[25]:


# List of categorical features
cate_features = list(df_all.select_dtypes(include=['object', 'category']).columns)

# List of numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_features = list(df_all.select_dtypes(include=numeric_dtypes).columns)


# In[26]:


# Update training and testing dataset
df_train, df_test = divide_df(df_all)


# <a name='2.3'></a>
# ## 2.3 Feature selection

# <a name='2.3.1'></a>
# ### 2.3.1 Select categorical features
# - Using **forward feature selection** to select the categorical features
# - **Forward selection** is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model until the technique finishs choosing k features as we set
# - RandomForest Regression can be used as a model to filter the features
# - There are totally 38 numerical features, I'll use forward feature selection technique to select the most 30 correlated categorical features with the target
# - Because this forward selection technique trains on categorical data to filter features. Therefore, I need to encode the categorical features using "label encoding" technique first, but only for this step not for actual training
# 
# **(NOTE)** Because of the iterative training process, so I commented the training code and print out the result below for avoiding time processing. If you're curious, you can uncomment it and try by yourself 

# In[27]:


from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[28]:


def forward_feature_selection(df_train, cate_features):
    '''Activate the "forward feature selection" technique to select the most appropriate features
    Arg:
        cate_features: list of string names of all categorical features in dataset
    Return:
        SFS object
        '''
    # df prepared for inputing into the technique
    pre_ffs = pd.DataFrame(columns=cate_features)

    # Encode categorical features to numeric feature for utilize "forward selection feature"
    for cate_col in cate_features:
        pre_ffs[cate_col] = preprocessing.LabelEncoder().fit_transform(df_train[cate_col].values)
        
    # Step forward feature selection
    sfs1 = SFS(RandomForestRegressor(),
               k_features=2,
               forward=True,
               floating=False,
               verbose=2,
               scoring='r2',
               cv=3)
    
    sfs1 = sfs1.fit(np.array(pre_ffs[cate_features]), np.array(y_train))
    return sfs1

# (UNCOMMENT HERE TO TRY)
# Choose categorical features using SFS technique
# sfs1 = forward_feature_selection(df_train, cate_features)

# Print out the chosen categorical feature
# cate_features = list(df_all[cate_features].columns[list(sfs1.k_feature_idx_)])
# cate_features


# In[29]:


cate_features =  ['BldgType', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical',
 'ExterCond', 'ExterQual', 'Exterior2nd', 'Functional', 'GarageCond', 'GarageType', 'Heating', 'HouseStyle', 'KitchenQual',
 'LandContour', 'LandSlope', 'LotShape', 'MSSubClass', 'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle',
 'SaleCondition', 'SaleType', 'Street', 'Utilities']


# <a name='2.3.2'></a>
# ### 2.3.2 Mismatched value between train & test set in categorical features
# - In some case, some columns in train dataset contained values which do not exist in testing dataset, we called them **missmatched data** between train and test
# - This would be an serious problem if we plan to perform one-hot encoding for categorical features in the future because of the different number of features between train and test data
# - Below are an example of missmatched data in "Electrical" feature. The value "mix" exist in training set but not in testing set

# In[30]:


# "Electrical" is an example of mismatched feature values
print('In training dataset\n', df_train['Electrical'].value_counts(), '\n')
print('In testing dataset\n', df_test['Electrical'].value_counts())


# In[31]:


cate_mismatch = list()

# Determine features contained mismatched values
for cate_col in cate_features:
    train_cate = df_train[cate_col].value_counts().index
    test_cate = df_test[cate_col].value_counts().index
    check_len = len(np.setdiff1d(train_cate, test_cate)) + len(np.setdiff1d(test_cate, train_cate))
    if check_len != 0:
        cate_mismatch.append(cate_col)
        
print('List of mismatched value features: \n', cate_mismatch)


# - For dealing with mismatched values, I'll replace them by the values having the highest frequency in each feature. 
# - With some features having high number of different values, I think removing them might be a good choice

# In[32]:


# "Electrical" features
df_train['Electrical'].loc[df_train['Electrical']=='Mix'] = 'SBrkr'

# "Exterior2nd" features
df_train['Exterior2nd'].loc[df_train['Exterior2nd']=='Other'] = 'VinylSd'

# "Heating" features
df_train['Heating'].loc[df_train['Heating']=='OthW'] = 'GasA'
df_train['Heating'].loc[df_train['Heating']=='Floor'] = 'GasA'

# "HouseStyle" features
df_train['HouseStyle'].loc[df_train['HouseStyle']=='2.5Fin'] = '1.5Fin'

# "MSSubClass" features
df_test['MSSubClass'].loc[df_test['MSSubClass']=='150'] = '160'

# "Condition2" feature
temp = [True if ((val=='RRNn') | (val=='RRAn') | (val=='RRAe')) else False
        for val in df_train['Condition2']]
df_train['Condition2'].loc[temp] = 'Norm'

# "Utilities" is a constant-value feature --> Delete it
# "RoofMatl" has high number of different values --> Delete it
cate_drop = ['Utilities', 'RoofMatl']
df_train.drop(cate_drop, axis=1, inplace=True)
df_test.drop(cate_drop, axis=1, inplace=True)

# Update the cate_features list also
cate_features = [col for col in cate_features if col not in cate_drop]


# In[33]:


# Check "Condition2" feature
print('In training dataset\n', df_train['Electrical'].value_counts(), '\n')
print('In testing dataset\n', df_test['Electrical'].value_counts())


# Great!! All the mismatched values are fixed. Now let's move to selecting the continuous features

# <a name='2.3.3'></a>
# ### 2.3.3 Select contunious features
# - The Pearson correlation coeficient is a statistical measure of the strength of a linear association between 2 continuous features. This technique is suitable for linear correlation, or rank-based methods for a non linear correlation
# - Therefore, Pearson could be a compeling choice for choosing the features having high correlation to the SalePrice target
# 
# Reference link: [here](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

# In[34]:


# Find feature correlation with target using pearson's coeficient
pearson = dict()
for col in num_features:
    pear_val = stats.pearsonr(np.array(df_train[col]), np.array(y_train))[0]
    pearson[col] = pear_val
    
pearson = pd.Series(pearson).abs().sort_values(ascending=False)
# Choose only feature having correlation larger than 0.2
num_features = list(pearson.loc[pearson > 0.2].index)
num_features


# In[35]:


chosen_cols = num_features + cate_features

# Visualizing the correlation table
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pd.concat([df_train[chosen_cols], y_train], axis=1).corr(), square=True,
            cmap='mako', annot_kws={'size': 14})


# In[36]:


df_train = df_train[chosen_cols]
df_test = df_test[chosen_cols]
df_all = concat_df(df_train, df_test)


# <a name='2.4'></a>
# ## 2.4 Features transformation

# <a name='2.4.1'></a>
# ### 2.4.1 Highly skewed numeric features
# - Highly skewed numeric features are the heavy-tail features like our target features
# - We decide whether a feature is skewness or not based on the value of "skewness" statistics measurement
# - All skewed features will be normalize by Box-cox normalization technique

# In[37]:


# Normalize skewness feature using Log function
skew_features = df_all[num_features].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
skew_features = skew_features[abs(skew_features) > 0.75]
print(skew_features)           

# Apply Box cox for skewness > 0.75
for feat in skew_features.index:
    df_all[feat] = np.log1p(df_all[feat])


# In[38]:


df_train, df_test = divide_df(df_all)


# <a name='2.4.2'></a>
# ### 2.4.2 One-hot encoding categorical features
# Now after finishing preprocessing all the categorical features, we should encode them to numeric features to be successfully inputted into the model by "one-hot encoding" technique. 

# In[39]:


print(df_train.shape, df_test.shape)


# In[40]:


# Transform categorical feature to dummies features
encoded_features = list()

for df in [df_train, df_test]:
    for feature in cate_features:
        # Change to array after encoding b.c want to add columns when change back to df
        encoded_feat = preprocessing.OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        # "n": Number of unique value in each feature
        n = df[feature].nunique()
        # "feature_uniqueVal" are the col's names in df after One-hot encoding
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)
        
df_train = pd.concat([df_train, *encoded_features[:len(cate_features)]], axis=1)
df_test = pd.concat([df_test, *encoded_features[len(cate_features):]], axis=1)


# After encoding all the categorical features, we need to remove the original ones to prevent training the same type of information. Because the orginal categorical features and their one-hot encoding version represent the same infomation

# In[41]:


# Drop original category features
df_train.drop(cate_features, axis=1, inplace=True)
df_test.drop(cate_features, axis=1, inplace=True)

df_all = concat_df(df_train, df_test)


# In[42]:


print(df_train.shape, df_test.shape)


# Now the number of features in training and testing dataset are the same & all the preprocessing steps are finished. The data is ready for training!!

# In[43]:


df_train.head()


# <a name='3'></a>
# # 3. Modeling

# In[44]:


from sklearn.model_selection import KFold # for repeated K-fold cross validation
from sklearn.model_selection import cross_val_score # score evaluation
from sklearn.model_selection import cross_val_predict # prediction
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
SEED = 42


# In[45]:


# Repeated K-fold cross validation
kfolds = KFold(n_splits=10, shuffle=True, random_state=SEED)

# Return root mean square error of model prediction (Used for test prediction)
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# Return root mean square error applied cross validation (Used for training prediction)
def evaluate_model_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


# <a name='3.1'></a>
# ## 3.1 Base models
# We'll plan to construct these below based model
# - **Ridge** is regression model applying l2 regularization technique. ([Link here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html))
# - **Lasso** stands for Least Absolute Shrinkage and Selection Operator that is a linear regression model applied l1 regularization technique ([Link here](https://www.statisticshowto.com/lasso-regression/))
# - **elasticnet** is a penalized linear regression model that includes both the L1 and L2 penalties during training ([Link here](https://machinelearningmastery.com/elastic-net-regression-in-python/))
# - **svr** stands for Support Vector Regression is a type of "SVM" model using for regression problem ([Link here](https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2))
# - **gbr** is gradient boosting model for regression problem ([Link here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html))
# - **lgbm** is a fast, distributed, high-performance gradient boosting framework that uses a tree-based learning algorithm ([Link here](https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/))
# - **xgboost** is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework ([Link here](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d#:~:text=What%20is%20XGBoost%3F,all%20other%20algorithms%20or%20frameworks.))

# In[46]:


def construct_models():
    # Initialize parameters for models
    alphas_ridge = [0.005, 0.01, 0.1, 1, 5, 10, 15]
    alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas_elas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio_elas = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    
    # Constructing the models
    models = dict()
    
    models['ridge'] = RidgeCV(alphas=alphas_ridge, cv=kfolds)
    models['lasso'] = LassoCV(alphas=alphas_lasso, random_state=SEED, cv=kfolds)
    models['elasticnet'] = ElasticNetCV(alphas=e_alphas_elas, cv=kfolds, l1_ratio=e_l1ratio_elas)
    models['svr'] = SVR(C = 20, epsilon = 0.008, gamma =0.0003)
    models['gbr'] = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, 
                                              max_depth=4, max_features='sqrt',
                                              min_samples_leaf=15, min_samples_split=10, 
                                              loss='huber',random_state =SEED) 
    models['lgbm'] = LGBMRegressor(objective='regression', num_leaves=4,
                                   learning_rate=0.01, n_estimators=5000,
                                   max_bin=200, bagging_fraction=0.75,
                                   bagging_freq=5, bagging_seed=7,
                                   feature_fraction=0.2,
                                   feature_fraction_seed=7, verbose=-1,
                                  colsample_bytree=None, subsample=None, subsample_freq=None)
    models['xgboost'] = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7, verbosity = 0,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=SEED, reg_alpha=0.00006)
    return models

# Construct the set of model
models = construct_models()


# After design and construct the based model, we need to fit the training data to the model and compute the root mean square error (rmse) result to validate models after the training
# - **Note that:** numpy array is recommended as an input to the training model instead of Dataframe. Because numpy has a hugh benefit relating to the time consuming than pandas. Link for more infomation [Here](https://towardsdatascience.com/speed-testing-pandas-vs-numpy-ffbf80070ee7)

# In[47]:


for name, model in models.items():
    # Start counting time
    start = time.perf_counter()
    
    model = model.fit(np.array(df_train), np.array(y_train))
    rmse_result = rmse(y_train, model.predict(np.array(df_train)))
    print(f'{name}\'s rmse after training: {rmse_result}')
    
    # Compute time for executing each algo
    run = time.perf_counter() - start
    print(f'Computational runtime of this algo: {round(run, 2)} seconds\n')


# - "Overfitting" problem happens when the model overlearns the detail of training dataset so that it'll negatively impact the performance of model on the testing dataset. In short, when the performance on training dataset is much more higher than performance on testing dataset
# - Computing rmse applying cross validation technique is effective to prevent the "Overfitting" problem

# In[48]:


cv_rmse_result = dict()
cv_rmse_mean = dict()
cv_rmse_std = dict()

for name, model in models.items():
    # Start counting time
    start = time.perf_counter()
    
    cv_rmse_result[name] = evaluate_model_cv(model, np.array(df_train), np.array(y_train))
    cv_rmse_mean[name] = cv_rmse_result[name].mean()
    cv_rmse_std[name] = cv_rmse_result[name].std()
    print(f'Finish {name}\'s model')
    
    # Compute time for executing each algo
    run = time.perf_counter() - start
    print(f'Computational runtime of this algo: {round(run, 2)} seconds\n')


# In[49]:


ML_cv = pd.DataFrame({'cv_rsme_mean' : cv_rmse_mean, 'cv_rmse_std' : cv_rmse_std})
ML_cv


# <a name='3.2'></a>
# ## 3.2 Stacking model
# - In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than the learning algorithm alone. **Stacking model** is an ensemble one
# - It uses a meta-learning algorithm to learn how to best combine the predictions from two or more base machine learning algorithms.
# 
# Reference link: [Stacking model](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

# In[50]:


# Type 1 stacking model
stack_model = StackingCVRegressor(regressors=(models['ridge'], models['lasso'], models['xgboost'],
                                              models['elasticnet'], models['gbr'], models['lgbm']),
                                  meta_regressor=models['xgboost'], use_features_in_secondary=True)


# In[51]:


# Time performance counter
start = time.perf_counter()

stack_model = stack_model.fit(np.array(df_train), np.array(y_train))
print('Finish training')

# Compute rmse with cross-validation technique
# rmse_stack_cv = evaluate_model_cv(stack_model, np.array(df_train), y_train)
# print(f'stack_model\'s rmse (using cv) after training: {rmse_stack_cv.mean()}')

# Compute rmse without cross-validation technique
rmse_stack = rmse(y_train, stack_model.predict(np.array(df_train)))
print(f'stack_model\'s rmse (using cv) after training: {rmse_stack}')

# Compute time for executing each algo
run = time.perf_counter() - start
print(f'Computational runtime of this algo: {round(run, 2)} seconds\n')


# <a name='3.3'></a>
# ## 3.3 Blending model
# - **Blending** is an ensemble machine learning technique that uses a machine learning model to learn how to best combine the predictions from multiple contributing ensemble member models
# - For more understanding, the link [Blending model](https://mlwave.com/kaggle-ensembling-guide/) might be useful!

# In[52]:


def blend_models_predict(X):
    return ((0.05 * models['ridge'].predict(np.array(X))) + \
            (0.05 * models['lasso'].predict(np.array(X))) + \
            (0.05 * models['elasticnet'].predict(np.array(X))) + \
            (0.15 * models['gbr'].predict(np.array(X))) + \
            (0.15 * models['lgbm'].predict(np.array(X))) + \
            (0.25 * models['xgboost'].predict(np.array(X))) + \
            (0.3 * stack_model.predict(np.array(X))))


# In[53]:


print('RMSLE score on train data:')
print(rmse(y_train, blend_models_predict(np.array(df_train))))


# <a name='4'></a>
# # 4 Submision

# In[54]:


# Get the id feature from testing dataset
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_id = test['Id']


# In[55]:


# Submission set
submit = pd.concat((test_id, pd.Series(np.exp(blend_models_predict(df_test)), 
                                       name='SalePrice')), axis=1)
submit.to_csv('Submission.csv', index=False)


# ## Credit & resource
# - Credit should be extended for [Serigne](https://www.kaggle.com/serigne) and [Prashant Banerjee](https://www.kaggle.com/prashant111/comprehensive-guide-on-feature-selection) notebooks to help me to gain knowledge about features selection, blending model and give me some hint to finish this notebook. Great thanks for them
# - There are 2 supper informative and comprehensive sources to understand the concept [ensemble model](https://mlwave.com/kaggle-ensembling-guide/) and learn how to code them [ensemble model code](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
# - Great notebook about features selection [Feature selections](https://www.kaggle.com/prashant111/comprehensive-guide-on-feature-selection). I've learned a lot from it!
# 
# ### Please give it an upvote, if you found this notebook helpful, thank for reading!

# In[ ]:




