#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
import xgboost as xg
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import missingno as msno

import optuna

# import pandas_profiling as pp

import warnings
warnings.filterwarnings('ignore')


# In[3]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape, test.shape)


# In[4]:


# pp.ProfileReport(train)


# When we look at this, we can see that "Utilities" and "Street" and so on consist of almost the same variables. It seems to be almost useless for analysis.<br>Later, we address on this problem.

# In[5]:


train.head(10)


# In[6]:


train.describe().T


# In[7]:


train.info()


# In[8]:


train.dtypes.value_counts()


# ### Missing Train values

# In[9]:


msno.matrix(train);


# In[10]:


total = train.isnull().sum().sort_values(ascending = False)[train.isnull().sum().sort_values(ascending = False) != 0]
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False)[(train.isnull().sum() / train.isnull().count()).sort_values(ascending = False) != 0]
missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
print(missing)


# In[11]:


def plotting_3_chart(df, feature):
    style.use("fivethirtyeight")
    
    fig = plt.figure(constrained_layout = True, figsize = (15, 8))#constrained_layout:auto adjust object placement.
    grid = gridspec.GridSpec(ncols = 3, nrows = 2, figure = fig)
    
    #Histogram
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:, feature], norm_hist = True, ax = ax1, color = 'g')
    
    #Probability Plot
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Probability Plot')
    stats.probplot(df.loc[:, feature], plot = ax2)
    
    #Box Plot
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:, feature], orient = 'v', ax = ax3)


# In[12]:


plotting_3_chart(train, 'SalePrice')


# <ul>
# <li>Out target variable, <b>SalePrice</b> is not normally distributed.
# <li>Our target variable is right-skewed.
# <li>There are multiple outliers in the variable.
# </ul>

# It looks like there are quite a bit Skewness and Kurtosis in the target variable. Let's talk about those a bit. 
# 
# <b>Skewness</b> 
# * is the degree of distortion from the symmetrical bell curve or the normal curve. 
# * So, a symmetrical distribution will have a skewness of "0". 
# * There are two types of Skewness: <b>Positive and Negative.</b> 
# * <b>Positive Skewness</b>(similar to our target variable distribution) means the tail on the right side of the distribution is longer and fatter. 
# * In <b>positive Skewness </b> the mean and median will be greater than the mode similar to this dataset. Which means more houses were sold by less than the average price. 
# * <b>Negative Skewness</b> means the tail on the left side of the distribution is longer and fatter.
# * In <b>negative Skewness </b> the mean and median will be less than the mode. 
# * Skewness differentiates in extreme values in one versus the other tail. 
# 
# Here is a picture to make more sense.  
# ![image](https://cdn-images-1.medium.com/max/1600/1*nj-Ch3AUFmkd0JUSOW_bTQ.jpeg)

# In[13]:


print("Skewness: ", train['SalePrice'].skew())


# **Kurtosis**<br>
# 
# According to Wikipedia, 
# 
# *In probability theory and statistics, **Kurtosis** is the measure of the "tailedness" of the probability. distribution of a real-valued random variable.* So, In other words, **it is the measure of the extreme values(outliers) present in the distribution.** 
# 
# * There are three types of Kurtosis: <b>Mesokurtic, Leptokurtic, and Platykurtic</b>. 
# * Mesokurtic is similar to the normal curve with the standard value of 3. This means that the extreme values of this distribution are similar to that of a normal distribution. 
# * Leptokurtic: Example of leptokurtic distributions are the T-distributions with small degrees of freedom.
# * Platykurtic: Platykurtic describes a particular statistical distribution with thinner tails than a normal distribution. Because this distribution has thin tails, it has fewer outliers (e.g., extreme values three or more standard deviations from the mean) than do mesokurtic and leptokurtic distributions. 
# 
# ![image](https://i2.wp.com/mvpprograms.com/help/images/KurtosisPict.jpg?resize=375%2C234)
# 
# 
# You can read more about this from [this](https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa) article. 

# In[14]:


print('kurtosis: ', train['SalePrice'].kurt())


# We can fix this by using different types of transformation(more on this later). However, before doing that, I want to find out the relationships among the target variable and other predictor variables. Let's find out.

# In[15]:


(train.corr()**2)['SalePrice'].sort_values(ascending = False)#.index[:5]


# These are the predictor variables sorted in a descending order starting with the most correlated one <b>OverallQual</b>. <br>Let's put these in a scatter plot and check how it looks.

# #### SalePrice vs OverallQual

# In[16]:


def customized_scatterplot(y, x):
    style.use('fivethirtyeight')
    plt.subplots(figsize = (12, 8))
    sns.scatterplot(y = y, x = x)


# In[17]:


customized_scatterplot(train.SalePrice, train.OverallQual)


# Because OverallQual is categorical variable, so scatter plot is not suitable. But it is certain that there is a relationship between the two variables.<br>Let's check other variables.

# #### SalePrice vs GvLivArea

# In[18]:


customized_scatterplot(train.SalePrice, train.GrLivArea)


# #### SalePrice vs GarageArea

# In[19]:


customized_scatterplot(train.SalePrice, train.GarageArea)


# #### SalePrice vs TotalBsmtSF

# In[20]:


customized_scatterplot(train.SalePrice, train.TotalBsmtSF)


# #### SalePrice vs 1stFlrSF

# In[21]:


customized_scatterplot(train.SalePrice, train['1stFlrSF'])


# #### SalePrice vs MasVnrArea

# In[22]:


customized_scatterplot(train.SalePrice, train.MasVnrArea)


# There are multiple outliers in some figures. We'll address this problem later.

# In[23]:


train_ = train[train.GrLivArea < 4500]
train_.reset_index(drop = True, inplace = True)
previous_train = train_.copy()
customized_scatterplot(train_.SalePrice, train_.GrLivArea)


# The two on the top-right edge of above figure seem to follow a trend, which can be explained by saying that "As the prices increased, this did too", so we leave this.

# In[24]:


fig, (ax1, ax2) = plt.subplots(figsize = (12, 8), ncols = 2, sharey = False)
sns.scatterplot(x = train_.GrLivArea, y = train_.SalePrice, ax = ax1);
sns.regplot(x = train_.GrLivArea, y = train_.SalePrice, ax = ax1, color = 'b');

sns.scatterplot(x = train_.MasVnrArea, y = train_.SalePrice, ax = ax2);
sns.regplot(x = train_.MasVnrArea, y = train_.SalePrice, ax = ax2, color = 'r');


# **Residual plot** tell us how is the error variance across the true line.

# In[25]:


plt.subplots(figsize = (12, 8))
sns.residplot(train_.GrLivArea, train_.SalePrice);


# In[26]:


plotting_3_chart(train_, 'SalePrice')


# The reason why outliers was removed was because linear regression analysis was susceptible to outliers.<br>Also, linear regression analysis requires normality. Therefore, we will apply the **log function** to solve these problems.

# In[27]:


train_['SalePrice'] = np.log1p(train_['SalePrice'])
plotting_3_chart(train_, 'SalePrice')


# In[28]:


fig, (ax1, ax2) = plt.subplots(figsize = (15, 6), ncols = 2, sharey = False, sharex = False)
sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1);
sns.residplot(x = train_.GrLivArea, y = train_.SalePrice, ax = ax2, color = 'r');


# Wow!<br>Here, we can see that the pre-transformed chart on the left has heteroscedasticity, and the post-transformed chart on the right has homoscedasticity(almost an equal amount of variance across the zero lines).

# But problem still remains. The name is **multicollinearity**.<br><br>Multicollinearity is occur when there is a strong correlation between independent variables. Linear regression or multilinear regression requires independent variables to have little or no related features.If there is multicollinearity, the accuracy of analysis will be reduced.<br>So we use Heatmap. this is an excellent way to identify whether there is multicollinearity or not. The best way to solve multicollinearity is to use regularization methods like <a href = '#section_ridge_other'>Ridge or Lasso.</a>

# In[29]:


style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30, 20))

mask = np.zeros_like(train_.corr(), dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(train_.corr(),
            cmap = sns.diverging_palette(20, 220, n = 200),
            mask = mask,
            annot = True,
            center = 0);


# But this time, we leave these variables for later study. There is algorithms as scikit-learn modules that can give us better outcome.

# ## Feature engineering

# In[30]:


train_.drop(columns = ['Id'], axis = 1, inplace = True)
test.drop(columns = ['Id'], axis = 1, inplace = True)


# In[31]:


#as y_train
y = train_['SalePrice'].reset_index(drop = True)


# In[32]:


previous_train = train_.copy()


# In[33]:


all_data = pd.concat((train_, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)


# ### Dealing with Missing Values

# In[34]:


ratio = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False)[(all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False) != 0]
total = all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]
concat_ = pd.concat([total, ratio], axis = 1, keys = ['Total', 'Ratio'])
concat_


# In[35]:


all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))


# In[36]:


missing_val_col = ["Alley", 
                   "PoolQC", 
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType']

for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None')


# In[37]:


missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath', 
                    'BsmtHalfBath', 
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea']

for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0)


# In[38]:


all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

#The mode function returns 'pandas.Series'. Therefore, we use [0] to extract the element.
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[39]:


all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])


# In[40]:


# all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
# all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
# all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)


# In[41]:


if len(all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]) == 0:
    print('there is no null')


# In[42]:


numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
skewed_features = all_data[numeric_features].apply(lambda x: skew(x)).sort_values(ascending = False)
skewed_features


# In[43]:


sns.distplot(all_data['1stFlrSF']);


# In[44]:


## Fixing skewed features using boxcox transformation
def fixing_skewness(df):
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    
    skewed_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending = False)
    high_skew = skewed_features[abs(skewed_features) > 0.5] #abs means absolute value
    skewed_features = high_skew.index
    
    for feature in skewed_features:
        df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))
        
fixing_skewness(all_data)


# In[45]:


sns.distplot(all_data['1stFlrSF']);


# Compare to the above figure, we can see that skewness has been solved!

# ### Creating New Features

# In[46]:


all_data['TotalSF'] = (all_data['TotalBsmtSF'] 
                       + all_data['1stFlrSF'] 
                       + all_data['2ndFlrSF'])

all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] 
                                 + all_data['BsmtFinSF2'] 
                                 + all_data['1stFlrSF'] 
                                 + all_data['2ndFlrSF']
                                )
                                 

all_data['Total_Bathrooms'] = (all_data['FullBath'] 
                               + (0.5 * all_data['HalfBath']) 
                               + all_data['BsmtFullBath'] 
                               + (0.5 * all_data['BsmtHalfBath'])
                              )
                               

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] 
                              + all_data['3SsnPorch'] 
                              + all_data['EnclosedPorch'] 
                              + all_data['ScreenPorch'] 
                              + all_data['WoodDeckSF']
                             )


# In[47]:


all_data['hasapool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[48]:


all_data.shape


# ### unnecessary feature drop

# In[49]:


drop_features = []
for i in all_data.columns:
    counts = all_data[i].value_counts(ascending = False)
    zeros = counts.iloc[0]
    if zeros / len(all_data) > 0.995:
        print(i)
        drop_features.append(i)


# In[50]:


all_data = all_data.drop(drop_features, axis = 1)


# In[51]:


all_data.shape


# In[52]:


feature_candidates = pd.get_dummies(all_data).reset_index(drop = True)
print(feature_candidates.head(5))


# In[53]:


X = feature_candidates.iloc[:len(y), :]
X_sub = feature_candidates.iloc[len(y):, :]


# In[54]:


print(X.shape, X_sub.shape)


# In[55]:


def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit


overfitted_features = overfit_reducer(X)


# In[56]:


X = X.drop(overfitted_features, axis=1)
X_sub = X_sub.drop(overfitted_features, axis=1)


# In[57]:


X.shape, y.shape, X_sub.shape


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Modeling the Data

# First, let's check the variance of SalePrice and GrLivArea, these are correlated with each other as you can see above.

# In[59]:


sample_train = previous_train.sample(300)
plt.subplots(figsize = (15, 8))
ax = plt.gca()
ax.scatter(sample_train.GrLivArea.values, sample_train.SalePrice.values);
plt.title('Chart with Data Points');


# What happens if we take the average of the data points?

# In[60]:


plt.subplots(figsize = (15, 8))
ax = plt.gca()
ax.scatter(sample_train.GrLivArea.values, sample_train.SalePrice.values);
ax.plot((sample_train.GrLivArea.values.min(), sample_train.GrLivArea.values.max()), (sample_train.SalePrice.values.mean(), sample_train.SalePrice.values.mean()));


# This way is the most efficient way to estimate the price of houses. But this doesn't represent all datapoint trend.<br>So, we use **MSE**.

# In[61]:


sample_train['mean_sale_price'] = sample_train.SalePrice.mean()
sample_train['mse'] = np.square(sample_train.mean_sale_price - sample_train.SalePrice)
sample_train.mse.mean()# this is mse. the closer this value of MSE is to 0, the better.


# The detail of MSE is [here](https://towardsdatascience.com/https-medium-com-chayankathuria-regression-why-mean-square-error-a8cad2a1c96f). <br>In a nutshell, the closer the value of MSE is to "0", the better. We want to minimize this error. In the process of reducing MSE, we use the powerful model, **Linear Regression**.

# ### $$ y = \beta_0 + \beta_1 x + \epsilon \\ $$
# <hr>

# ##### $$ \hat{\beta}_1 = r_{xy} \frac{s_y}{s_x}$$
# ##### $$ \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} $$
# ##### $$ r_{xy}= \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sqrt{\sum(x_i - \bar{x})^2{\sum(y_i - \bar{y})^2}}}$$
# <br>Here...
# - $\bar{y}$ : the sample mean of observed values $Y$
# - $\bar{x}$ : the sample mean of observed values $X$
# - $s_y$ : the sample standard deviation of observed values $Y$
# - $s_x$ : the sample standard deviation of observed values $X$
# - $ \epsilon$ : error or residual. In above figure, the distance of datapoint and redline.

# In[62]:


y_bar = sample_train.SalePrice.mean()
x_bar = sample_train.GrLivArea.mean()
std_y = sample_train.SalePrice.std()
std_x = sample_train.GrLivArea.std()
r_xy = sample_train.corr().loc['GrLivArea', 'SalePrice']#pearson correlation coefficient 相関係数


# In[63]:


beta_1 = r_xy * (std_y / std_x)
beta_0 = y_bar - beta_1 * x_bar

sample_train['Linear_Yhat'] = beta_0 + beta_1 * sample_train['GrLivArea']


# In[64]:


sample_train.head(10)


# In[65]:


fig = plt.figure(figsize = (15, 7))
ax = plt.gca()

ax.scatter(sample_train.GrLivArea, sample_train.SalePrice, c = 'b');
ax.plot(sample_train.GrLivArea, sample_train.Linear_Yhat);


# In[66]:


print('mean squared error for regression line is {}'.format(np.square(sample_train['SalePrice'] - sample_train['Linear_Yhat']).mean()))
# print('mean squared error for regression line is {}'.format(mean_squared_error(sample_train['SalePrice'], sample_train['Linear_Yhat'])))


# In[67]:


fig = plt.figure(constrained_layout = True, figsize = (15, 5))
grid = gridspec.GridSpec(ncols = 2, nrows = 1, figure = fig)
ax1 = fig.add_subplot(grid[0, :1])

# ax1 = fig.gca()
ax1.scatter(x = sample_train['GrLivArea'], y = sample_train['SalePrice'], c = 'b');
ax1.plot(sample_train['GrLivArea'], sample_train['mean_sale_price'], color = 'k');

for _, row in sample_train.iterrows():
    plt.plot((row['GrLivArea'], row['GrLivArea']), (row['SalePrice'], row['mean_sale_price']), 'r-')
    

ax2 = fig.add_subplot(grid[0, 1:])
ax2.scatter(x = sample_train['GrLivArea'], y = sample_train['SalePrice'], c = 'b');
ax2.plot(sample_train['GrLivArea'], sample_train['Linear_Yhat'], color = 'k');

for _, row in sample_train.iterrows():
    plt.plot((row['GrLivArea'], row['GrLivArea']), (row['SalePrice'], row['Linear_Yhat']), 'r-')


# This makes it obvious.<br>The MSE is getting smaller and close to its goal. Now, let's actually try to predict the SalePrice using **Linear Legression**.<br>This time, we have two features for simplicity. If there is more than one variable, it will look like this, 
# ### $$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n $$
# Conveniently, the library is ready to go. We have to be grateful for this.

# In[68]:


lin_reg = LinearRegression(normalize = True, n_jobs = -1)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)


# In[69]:


print('%.2f'%mean_squared_error(y_test, y_pred))


# In[70]:


lin_reg = LinearRegression()
cv = KFold(shuffle = True, random_state = 2, n_splits = 10)
scores = cross_val_score(lin_reg, X, y, cv = cv, scoring = 'neg_mean_absolute_error')


# In[71]:


print('%.8f'%scores.mean())


# Above we've seen is the simplest way to construct the model. However, there are more complex regression, so let's dive deep into some of it.

# <a id = 'section_ridge_other'></a>
# ## Regularization Models
# 
# What makes regressionmodel more effective is its ability of regularizing. *Regularization* is the ability to structurally prevent overfitting by imposing a penalty on the coefficients.
# 
# There are three types of regularizations.
# <ul>
# <li>Ridge
# <li>Lasso
# <li>Elastic Net
# </ul>

# ### <b>Ridge</b><br>
# 
# ### $$ \text{minimize:}\; RSS+Ridge = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_2\sum_{j=1}^p \beta_j^2$$<br>
# One of the benefits of regularization of using *Ridge* is that it deals with **multicollinearity**(high correlation between predictor variables) well especially.<br>(Lasso deals with multicollinearity more brutally by penalizing related coefficient and force them to become zero, hence removing them.)<br>We have some points we need to be aware of.
# <ul>
# <li>It is essential to standardize the predictor variables before constructing the models.
# <li>It is important to chec for multicollinearity.
# </ul>

# In[72]:


alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}

for i in alpha_ridge:
    ridge = Ridge(alpha = i, normalize = True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    temp_mse[i] = mse
    
    rss = sum((y_pred - y_test) ** 2)
    temp_rss[i] = rss


# In[73]:


for key, value in sorted(temp_mse.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[74]:


for key, value in sorted(temp_rss.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[75]:


temp_rss_ = {}
temp_mse_ = {}

for i in alpha_ridge:
    lasso_reg = Lasso(alpha = i, normalize = True)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    temp_mse_[i] = mse
    
    rss = sum((y_pred - y_test) ** 2)
    temp_rss_[i] = rss


# ### <b>Lasso</b>
# 
#     
# ### $$ \text{minimize:}\; RSS + Lasso = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_1\sum_{j=1}^p |\beta_j|$$

# In[76]:


temp_rss_ = {}
temp_mse_ = {}

for i in alpha_ridge:
    lasso_reg = Lasso(alpha = i, normalize = True)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    temp_mse_[i] = mse
    
    rss = sum((y_pred - y_test) ** 2)
    temp_rss_[i] = rss


# In[77]:


for key, value in sorted(temp_mse_.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[78]:


for key, value in sorted(temp_rss_.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# ### <b>Elastic Net</b>
# Elastic Net is the combination of both Ridge and Lasso.<br>
# ### $$ \text{minimize:}\; RSS + Ridge + Lasso = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_1\sum_{j=1}^p |\beta_j| + \lambda_2\sum_{j=1}^p \beta_j^2$$

# In[79]:


_temp_rss = {}
_temp_mse = {}

for i in alpha_ridge:
    lasso_reg = ElasticNet(alpha = i, normalize = True)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    _temp_mse[i] = mse
    
    rss = sum((y_pred - y_test) ** 2)
    _temp_rss[i] = rss


# In[80]:


for key, value in sorted(_temp_mse.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[81]:


for key, value in sorted(_temp_rss.items(), key = lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[82]:


kfolds = KFold(n_splits = 10, shuffle = True, random_state = 42)


# In[83]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = (0.01, 1e-05, 1e-02), cv = kfolds))#The larger the alphas, the stronger the regularization.
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter = 1e7, alphas = (0.01, 1e-05, 1e-02), random_state = 42, cv = kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter = 1e7, alphas = (0.01, 0.01, 1), cv = kfolds))
svr = make_pipeline(RobustScaler(), SVR(C = 20, epsilon = 0.008, gamma = 'auto'))


# In[84]:


'''
gbr = GradientBoostingRegressor(n_estimators = 3000,
                                learning_rate = 0.05, 
                                max_depth = 4, 
                                max_features = 'sqrt', 
                                min_samples_leaf = 15, 
                                min_samples_split = 10, 
                                loss = 'huber',
                                random_state = 42)
                                '''


# In[85]:


lightgbm = LGBMRegressor(objective = 'regression', 
                         num_leaves = 4,
                         learning_rate = 0.01,
                         n_estimators = 5000,
                         max_bin = 200,
                         bagging_fraction = 0.75, 
                         bagging_freq = 5,
                         bagging_seed = 7, 
                         feature_fraction = 0.2,
                         feature_fraction_seed = 7, 
                         verbose = -1)


# In[86]:


'''xgboost = xg.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)'''


# In[87]:


'''stack_gen = StackingCVRegressor(regressors = (ridge, lasso, elasticnet, lightgbm), 
                                meta_regressor = lightgbm,
                                use_features_in_secondary = True)'''


# In[88]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = kfolds))
    return rmse


# In[89]:


score = cv_rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(gbr)
# print("gradient_boosting_regressor: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# score = cv_rmse(xgboost)
# print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[90]:


# score = cv_rmse(stack_gen)
# print("stacking_cv_legressor: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[91]:


elastic_model_full_data = elasticnet.fit(X_train, y_train)
lasso_model_full_data = lasso.fit(X_train, y_train)
ridge_model_full_data = ridge.fit(X_train, y_train)
svr_model_full_data = svr.fit(X_train, y_train)
lgb_model_full_data = lightgbm.fit(X_train, y_train)
# stack_gen_model = stack_gen.fit(np.array(X), np.array(y))


# In[92]:


# abc


# In[93]:


def blend_models_predict(X):
    return (
        (0.2 * elastic_model_full_data.predict(X)) + \
        (0.2 * lasso_model_full_data.predict(X)) + \
        (0.2 * ridge_model_full_data.predict(X)) + \
        (0.2 * svr_model_full_data.predict(X)) + \
        (0.2 * lgb_model_full_data.predict(X))# + \
#         (0.1 * stack_gen_model.predict(np.array(X)))
    )


# In[94]:


print("RMSLE score on train data: ", rmsle(y, blend_models_predict(X)))


# In[95]:


submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:, 1] = np.floor(np.expm1(blend_models_predict(X_sub)))


# In[96]:


submission.to_csv("submission.csv", index = False)


# In[97]:


submission


# ## Thank you!!!
# 
# If you find this notebook useful, please **upvote**!
# <br>And if you have any questions, please ask on the **comments**!
# 

# ## Other Work( please **upvote**＜(_ _)＞ )
# 
# * [The power of normality and visualization
# ](https://www.kaggle.com/fightingmuscle/the-power-of-normality-and-visualization)
# * [【Over 0.8!】Titanic_more_technical_EDA(ENG&JPN)](https://www.kaggle.com/fightingmuscle/over-0-8-titanic-more-technical-eda-eng-jpn)
# 
# > These notebooks were put together by me as a beginner, so I believe anyone can learn from them!
# 
# * [How did I get the silver medal?(0.717)【Infer】](https://www.kaggle.com/fightingmuscle/how-did-i-get-the-silver-medal-0-717-infer/comments)
# 
# > I got silver medal for the first time! I published my inference code and trained models. My training code will be available soon, please wait. 
