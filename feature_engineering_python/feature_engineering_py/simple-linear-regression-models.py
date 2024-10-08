#!/usr/bin/env python
# coding: utf-8

# # First (baby) steps
# *Rocío Byron, Jan. 2018*
# 
# The goal of this notebook is to test out how far one can get with the simplest tools available: basic feature engineering and linear regression modeling. And on the way, learn how to work on Jupyter Notebook.
# 
# ***
# ## Inspirations
# 
# There are hundreds of kernels on this competition, and I have probably read half of them at some point or another. Two kernels, however, have inspired me the most:
# 
# 1. [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) by Serigne
# 2. The ubiquous [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) by Pedro Marcelino
# 
# ## Outline
# 
# 1. Preprocessing
#     1. Understanding the problem and the data available
#     2. Normality and skewness
#     3. Missing values
#     4. Dummy encoding
#     5. Rescaling
# 2. Regression
#     1. Linear regression
#     2. L1 regularisation
#     3. L2 regularisation
#     4. ElasticNet regularisation

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Preprocessing 
# 
# The main goal here is to get rid of (very) spurious data points and prepare the dataset for learning. 
# 
# This is a very delicate process: go to far, and you will be introducing bias in your data. Go to short, and you will be introducing rubbish in your learning process.

# ## A. The dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
testID = test['Id']

data = pd.concat([train.drop('SalePrice', axis=1), test], keys=['train', 'test'])
data.drop(['Id'], axis=1, inplace=True)


# In[ ]:


data.head(2)


# ### Sanity check
# 
# Before we go on and process this data, we need to be sure it actually makes sense. There are three "low hanging fruits" in this sense:
# - Features that represent years should not go take values larger than 2018
# - Areas, distances and prices should not take negative values
# - Months should be between 1 and 12

# In[ ]:


years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']


# In[ ]:


data[years].max()


# In[ ]:


mask = (data[years] > 2018).any(axis=1) # take any index with a twisted year value
data[mask]


# In[ ]:


data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']


# In[ ]:


mask = (data[metrics] < 0).any(axis=1)
data[mask]


# In[ ]:


mask = ((data['MoSold'] > 12) | (data['MoSold'] < 1))
data[mask]


# ### Data types
# 
# In terms of data type, there are four big groups:
# 
# 1. Continuous numerical features: lengths, areas, prices
# 2. Discrete numerical features: numerical scores, number of bedrooms, years; they support order and arithmetic operations, so they can be treated as numerical
# 3. Ordinal categorical features: features with qualitative scores (such as 'Excellent' or 'Slightly Irregular'); They support ordering ('Gentle slope' < 'Severe slope') but not arithmetic operations (how much is 'Sever slope' - 'Gentle slope'?)
# 4. Purely categorical features: a few examples are 'MSSubClass' or 'SaleType'
# 
# After some trial and error, I decide to separate numerical (both continuous and discrete) from categorical.
# 
# The pros:
# - We keep the relationship between discrete numerical features
# - We end up with less features, hence less risk of overfitting
# 
# The cons: 
# - We add some arbitrariness (if a house is an "8" is it really twice as better than a house that scores a 4?)

# In[ ]:


# Numerical features
num_feats = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
             'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
             'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal',
             'YrSold']    

# We need to convert literal grades to a numerical scale
grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
          'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
num = [9, 7, 5, 3, 2]
G = dict(zip(literal, num))

data[grades] = data[grades].replace(G)

# Categorical features: everything that is not 'numerical'
cat_feats = data.drop(num_feats, axis=1).columns


# In[ ]:


cat_feats


# ## B. Normality and skewness
# 
# Many regression models are more comfortable with normally distributed variables (or at least something close to it). 
# 
# We will, however, skip the discrete numerical features because:
# - The results will be more readable this way
# - Most of the discrete numerical features only take a few different values, so hoping for normality is a waste of time

# In[ ]:


#log transform the target:
price = np.log1p(train['SalePrice'])

#log transform skewed continuous numerical features:
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True)) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

data[skewed_feats] = np.log1p(data[skewed_feats])


# ## C. Missing values
# 
# There are two main types of NaN here:
# * Missing values: just some of the values that where not recorded (usually a small number of them)
# * Missing feature in the house: such as when there is no basement or no garage

# In[ ]:


data.isnull().sum()[data.isnull().sum() > 0]


# ### MSZoning, Utilities, Exteriors, Electrical, Functional, Utilities and SaleType
# 
# There are not many missing values, so we will just go with the mode of the neighbourhood.

# In[ ]:


feats = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional',
         'SaleType']
model = data.loc['train'].groupby('Neighborhood')[feats].apply(lambda x: x.mode().iloc[0])

for f in feats:
    data[f].fillna(data['Neighborhood'].map(model[f]), inplace=True)


# ### Lot frontage
# 
# My best guess is that it depends somewhat strongly on the configuration of the lot (inside/corner/cul/2-frontage/3-frontage).

# In[ ]:


plt.subplots(figsize=(15,5))
boxdata = data.loc['train'].groupby('LotConfig')['LotFrontage'].median().sort_values(ascending=False)
order = boxdata.index
sns.boxplot(x='LotConfig', y='LotFrontage', order=order, data=data.loc['train'])


# In[ ]:


data['LotFrontage'] = data['LotFrontage'].fillna(data.loc['train', 'LotFrontage'].median())


# ### KitchenQual
# 
# Again, very few missing values. We will substitute in this case with the 'OverallQual' value.

# In[ ]:


data['KitchenQual'].fillna(data['OverallQual'], inplace=True)


# ### Basement, garage, fireplaces and other features
# 
# We can interpret an NA in all these things as "the house does not have this feature".

# In[ ]:


bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath',
        'BsmtHalfBath', 
        'TotalBsmtSF']
fire = ['Fireplaces', 'FireplaceQu']
garage = ['GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageCars', 
          'GarageArea', 'GarageYrBlt']
masn = ['MasVnrType', 'MasVnrArea']
others = ['Alley', 'Fence', 'PoolQC', 'MiscFeature']

cats = data.columns[data.dtypes == 'object']
nums = list(set(data.columns) - set(cats))

# Be sure the category 'None' is also handled here
data['MasVnrType'].replace({'None': np.nan}, inplace=True)

data[cats] = data[cats].fillna('0')
data[nums] = data[nums].fillna(0)


# In[ ]:


data.isnull().sum().sum()


# ### Adjusting the type of variable
# 
# First, there are a few features that are not represented with the right type of variable:
# 
# - 'MSSubClass': represented as an integer, when it is just a category label (we will use 'object' for now)
# - 'MoSold': represented as an integer, a month is just a category label out of 12 possibilities (we will use 'object' for now)
# - 'BsmtFullBath', 'BsmtHalfBath': these two represent integers and not floats (or I at least I do not know what a third of half bathroom is)
# - years: a year, in the context of this dataset, is an integer, and not a float
# - 'GarageCars': represented as a float, it is an actual integer (nobody wants to have 0.5 car at home)

# In[ ]:


data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)
data[years] = data[years].astype('int64', copy=False)


# ### Categorical data with few samples per bin
# 
# Some categories of the categorical features are so unrrepresented in the dataset that drawing conclusions from them would lead to a noisy result. Instead, we will group those in one single category.

# In[ ]:


categorical_data = pd.concat((data.loc['train'][cat_feats], price), axis=1)


# In[ ]:


low = 0.05 * data.loc['train'].shape[0] # at least 5% of the dataset should have this value

for feat in cat_feats:        
    # we will group the categories based on the average sale price
    order = ((categorical_data.groupby(feat).mean()).sort_values(by='SalePrice', 
                                                      ascending=False).index.values.tolist())
    for i in range(0, len(order)):
        N = (categorical_data[categorical_data[feat] == order[i]]
             .count().max())
        j = i
        while (N < low) & (N != 0):
            j += 1

            if (j > len(order) - 1):
                # if reached the end of list, go back to last
                # useful category of the 'order' list
                j = i - 1
                break
            else: 
                N += (categorical_data[categorical_data[feat] == order[j]]
                      .count().max())
        if j < i:
            lim = len(order)
        else:
            lim = j

        for k in range(i, lim):
            categorical_data.replace({feat: {order[k]: order[j]}},
                                 inplace=True)
            data.replace({feat: {order[k]: order[j]}},
                                     inplace=True)            
    uniD = data[feat].unique()
    order = categorical_data[feat].unique()

    for i in uniD:
        if i not in order:
            ind = np.argsort(order - i)[0]
            data.replace({feat: {i: order[ind]}}, inplace=True)


# In[ ]:


data.columns


# ## D. Dummy encoding
# 
# ### Categorical data as categories
# 
# First, we need to turn those features with two categories into 0-1 encoding (the get_dummies method would convert them into two separate features otherwise, feat_0 and feat_1).

# In[ ]:


# Remove columns with just one category
for feat in categorical_data.columns[:-1]:    
    uni = categorical_data.groupby(feat).mean().sort_values(by='SalePrice').index
    if (len(uni) < 2):
            data.drop(feat, axis=1, inplace=True)
    elif len(uni) < 3:
        print("{}: {}".format(feat, uni))
        data[feat].replace({uni[0]: 0, uni[1]: 1}, inplace=True)
        data[feat] = data[feat].astype('int8')
    else:
        data[feat] = data[feat].astype('category')
        


# In[ ]:


finaldata = pd.get_dummies(data)


# The variables that express a "I don't have this" feature should not treat the '0' as a normal category. Instead, it would be cleaner (and less overfitting) to encode the zero in the other possible options.

# In[ ]:


black_list = bsmt + fire + garage + masn + others
for feat in finaldata.columns:
    if ('_0' in feat) and (feat.split("_")[0] in black_list):
        finaldata.drop(feat, axis=1, inplace=True)


# In[ ]:


finaldata.shape


# ## E. Rescaling
# 
# I will just separate the data and normalise it to make the regressors run smoother. As @Gennadi mentioned in the comments, I have to be careful not to leak any data from the test set into my training set. 
# 
# That is: I can only use the mean and standard dev of my train set to normalise.

# In[ ]:


# Training/testing sets
X_test = finaldata.loc['test']
X_train = finaldata.loc['train']

y_train = price


# In[ ]:


m = X_train.mean()
std = X_train.std()

X_train = (X_train - m) / std
X_test = (X_test - m) / std


# # 2. Regression
# 

# ## A. Linear regression (without regularisation)

# In[ ]:


# Create linear regression object
LR = LinearRegression()

# Train the model using the training sets
LR.fit(X_train, y_train)


# ### Top influencers

# In[ ]:


maxcoef = np.argsort(-np.abs(LR.coef_))
coef = LR.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# The coefficients indicate that this model is <u>very</u> overfitted. We can blame this on the correlation of several of the features:
# 
# 1. Dummy features are by definition correlated (i.e. if LotShape_Reg = 1, we can be certain that LotShape_IR1 = 0 for that house)
# 2. There might be other correlated features: 'GrLivArea' is probably very close to the sum of the first and second floor areas.

# ## B. Linear regression, L1 regularisation

# In[ ]:


# Create linear regression object
Ls = LassoCV()

# Train the model using the training sets
Ls.fit(X_train, y_train)


# ### Top influencers

# In[ ]:


maxcoef = np.argsort(-np.abs(Ls.coef_))
coef = Ls.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# This looks a lot more reasonable. When one thinks about the price of a house, lot area and living area are usually the first guess.

# ## C. Linear regression, L2 regularisation

# In[ ]:


# Create linear regression object
Rr = RidgeCV()

# Train the model using the training sets
Rr.fit(X_train, y_train)


# ### Top influencers

# In[ ]:


maxcoef = np.argsort(-np.abs(Rr.coef_))
coef = Rr.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# This is also very consistent with the L1 regularisation.

# ## D. Linear regression, elastic net
# 
# Okay, so one last attempt is to linearly combine Lasso and Ridge regularisations together (what they fancily call elastic net). The advantage of the elastic net is that if two features are correlated, it will retain both instead of just one (remember that in L1 regularisation, most of the features are set to parameter 0). 

# In[ ]:


# Create linear regression object
EN = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 5)) # we are essentially smashing most of the Rr model here

# Train the model using the training sets
train_EN = EN.fit(X_train, y_train)


# ### Top influencers

# In[ ]:


maxcoef = np.argsort(-np.abs(EN.coef_))
coef = EN.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# ## Comparison

# In[ ]:


model = [Ls, Rr, EN]
M = len(model)
CV = 5
score = np.empty((M, CV))
for i in range(0, M):
    score[i, :] = cross_val_score(model[i], X_train, y_train, cv=CV)


# In[ ]:


print(score.mean(axis=1))


# In[ ]:


submit = pd.DataFrame({'Id': testID, 'SalePrice': np.exp(EN.predict(X_test))})
submit.to_csv('submission.csv', index=False)


# A few conclusions from here:
# 
# - Linear regression, without regularisation, cannot fit this data "as-is". I believe it is because of the strong correlation between some of the variables with each other
# - Lasso and Ridge regression work both fairly well. The elastic net does not improve the performance that much

# ### Future improvements
# 
# - Feature engineering: in general, it is not a good idea to feed correlated features to a linear regression model. A way forward could be some deeper analysis of the variables
# - These are the simplest supervised learning models out there. My next try will probably be tree-based solutions, which tend to do better in this kind of datasets
# 
# And, that's it. If you read this and have some ideas for improvement, please let me know!
