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


get_ipython().system('python -m pip install matplotlib==3.5.0')
get_ipython().system('python -m pip install seaborn==0.11.0')


# In[3]:


import numpy as np
import pandas as pd
import datetime
import random
import math

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV


# In[4]:


from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


# In[5]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings(action = 'ignore')
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os


# In[6]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.shape, test.shape


# In[7]:


train.head(10)


# In[8]:


train.info()


# In[9]:


sns.set_style("white")
sns.set_color_codes(palette = 'deep')
f, ax = plt.subplots(figsize = (8, 7))
sns.distplot(train['SalePrice'], color = 'g');
ax.xaxis.grid(False)
ax.set(ylabel = "Frequency")
ax.set(title = "SalePrice distribution")
sns.despine(trim = True, left = True)
plt.show()


# In[10]:


print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# <a id = "section1"></a>
# <h2>Detail of the numeric variables</h2>

# In[11]:


import sys
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []

for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        numeric.append(i)
print('numeric variables', numeric)
print('-- numeric variables --\n')


fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (12, 120))
plt.subplots_adjust(right = 2)
plt.subplots_adjust(top = 2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if feature == 'MiscVal':
        break
        
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x = feature, y = 'SalePrice', hue = 'SalePrice', palette = 'Greens', data = train)
    
    plt.xlabel('{}'.format(feature), size = 15, labelpad = 12.5)
    plt.ylabel('SalePrice', size = 15, labelpad = 12.5)
    
    plt.tick_params(axis = 'x', labelsize = 12)
    plt.tick_params(axis = 'y', labelsize = 12)
        
    plt.legend(loc = 'best', prop = {'size' : 10})
    
plt.show()


# If we enumerate the variances of the numerical variables as above, we can clearly check the variables that are correlated with the 'SalePrice' of Target variable.

# <h2>the power of visualization</h2>

# **Various Plot**

# In[12]:


f, ax = plt.subplots(4, 1, figsize = (12, 15), sharex = True)

data = pd.concat([train['SalePrice'], train['OverallQual']], axis = 1)
fig1 = sns.boxplot(x = train['OverallQual'], y = "SalePrice", data = data, ax = ax[0])
fig1.axis(ymin = 0, ymax = 800000)

fig2 = sns.stripplot(x = train['OverallQual'], y = "SalePrice", data = data, ax = ax[1])
fig2.axis(ymin = 0, ymax = 800000)

fig3 = sns.swarmplot(x = train['OverallQual'], y = "SalePrice", data = data, ax = ax[2])
fig3.axis(ymin = 0, ymax = 800000)

fig4 = sns.violinplot(x = train['OverallQual'], y = "SalePrice", data = data, ax = ax[3])
fig4.axis(ymin = 0, ymax = 800000);


# **Various Plot of time series variables**

# In[13]:


f, ax = plt.subplots(4, 1, figsize = (16, 16))

data = pd.concat([train['SalePrice'], train['YearBuilt']], axis = 1)
fig1 = sns.boxplot(x = train['YearBuilt'], y = 'SalePrice', data = data, ax = ax[0])
fig1.axis(ymin = 0, ymax = 800000);
ax[0].xaxis.set_tick_params(rotation = 90);


fig2 = sns.regplot(x = train['YearBuilt'], y = "SalePrice", data = data, ax = ax[1])
fig2.axis(ymin = 0, ymax = 800000);
# ax[1].xaxis.set_tick_params(rotation = 90)

built = data['YearBuilt'].value_counts().sort_index()

ax[2].step(built.index, built, color = "#4a4a4a")
for s in ['top', 'right']:
    ax[2].spines[s].set_visible(False)
    
ax[2].grid()

color_ = ['#4a4a4a' if val != max(built) else "#e3120b" for val in built]
ax[3].bar(built.index, built, color = color_)

for s in ['top', 'right']:
    ax[3].spines[s].set_visible(False)
ax[3].grid()


# You can draw multiple graphs for the same time period by doing the following.

# In[14]:


train['HouseStyle'].value_counts()


# For clarity, we will restrict the amount of features.

# In[15]:


train['__HouseStyle'] = train['HouseStyle'].apply(lambda x: 'ETC' if x in ['SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'] else x)


# In[16]:


train['__HouseStyle'].value_counts()


# In[17]:


color_ = ["#00798c", '#d1495b', '#edae49', '#66a182']
fig, ax = plt.subplots(4, 1, figsize = (20, 12), sharex = True)

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts()
    ax[i].bar(hs_built.index, hs_built, color = color_[i], label = hs)
    ax[i].set_ylim(0, 50)
    ax[i].legend(loc = 'upper left')
    for s in ['top', 'right']:
        ax[i].spines[s].set_visible(False)

plt.show()


# <h3>Overlapping in bar graph</h3>
# In this case, multiple graphs is difficult for absolute comparison.<br>
# By doing overlapping, we can compare multiple graphs easily!

# In[18]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts()
    ax.bar(hs_built.index, hs_built, color = color_[i], label = hs, alpha = 0.4, edgecolor = color_[i])

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set_ylim(0, 50)
ax.legend(loc = 'upper left')
plt.show()


# <h3>Stack in bar graph</h3>

# These values can be **stacked**.

# In[19]:


#cumsum is Cumulative sum
data_sub = train.groupby('__HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC', '1.5Fin', '2Story', '1Story']].cumsum(axis = 0).T
data_sub


# In[20]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = data_sub[hs]
    ax.bar(hs_built.index, hs_built, color = color_[i], label = hs)
    
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.legend(loc = 'upper left')
ax.grid()
plt.show()


# <h3>Ratio in bar graph</h3><br>

# They are both easy to see! And you can even visualize the **ratios**

# In[21]:


data_sub = train.groupby('__HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC', '1.5Fin', '2Story', '1Story']].T
data_sum = data_sub.sum(axis = 1)
data_sub = (data_sub.T / data_sum).cumsum().T#(4, 112), (112, )


# In[22]:


data_sub


# In[23]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = data_sub[hs]
    ax.bar(hs_built.index, hs_built, color = color_[i], label = hs)
    
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
ax.legend(loc = 'upper left')
ax.grid()
plt.show()


# > **Line** graphs are better than bar graph for time series. 

# In[24]:


fig, ax = plt.subplots(4, 1, figsize = (20, 12), sharex = True)

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts().sort_index()
    ax[i].plot(hs_built.index, hs_built, color = color_[i], label = hs)
    ax[i].set_ylim(0, 50)
    ax[i].legend(loc = 'upper left')
    for s in ['top', 'right']:
        ax[i].spines[s].set_visible(False)
    
plt.show()


# In[25]:


fig, ax = plt.subplots(4, 1, figsize = (18, 5), sharex = True)

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts().sort_index()
    ax[i].plot(hs_built.index, hs_built, color = color_[i], label = hs)
    ax[i].fill_between(hs_built.index, 0, hs_built, color = color_[i])
    ax[i].set_ylim(0, 50)
    ax[i].legend(loc = 'upper left')

plt.subplots_adjust(hspace = 0.1)
plt.show()


# <h3>Overlapping in line graph</h3>

# As with bar, it is easier to plot them in the same place for comparison.

# In[26]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts().sort_index()
    ax.plot(hs_built.index, hs_built, color = color_[i], label = hs)

ax.set_ylim(0, 50)
ax.legend(loc = 'upper left')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
ax.grid()
plt.show()


# You can change <b>linestyle</b>!

# In[27]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))
linestyles = ['-', '--', '-.', ':']

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts().sort_index()
    ax.plot(hs_built.index, hs_built, color = color_[i], linestyle = linestyles[i], label = hs)
    
ax.set_ylim(0, 50)
ax.legend(loc = 'upper left')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.grid()#表中の格子
plt.show()


# visualize the **overlapped area**

# In[28]:


fig, ax = plt.subplots(1, 1, figsize = (18, 6))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = train[train['__HouseStyle'] == hs]['YearBuilt'].value_counts().sort_index()
    ax.plot(hs_built.index, hs_built, color = color_[i], label = hs)
    ax.fill_between(hs_built.index, 0, hs_built, color = color_[i], alpha = 0.4)
    
ax.set_ylim(0, 50)
ax.legend(loc = 'upper left')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
ax.grid()

plt.show()


# <h3>Stacking in line graph</h3>

# Line graphs can be compared in the same way by **stacking**!

# In[29]:


#unstack(): Series ←→ DataFrame
data_sub = train.groupby('__HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC', '1.5Fin', '2Story', '1Story']].cumsum().T

fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = data_sub[hs]
    ax.plot(hs_built.index, hs_built, color = color_[i], label = hs)
    ax.fill_between(hs_built.index, 0, hs_built, color = color_[i])
    
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
ax.legend(loc = 'upper left')
ax.grid()
plt.show()


# <h3>Stream graph in line graph</h3>

# A streamgraph, or stream graph, is a type of stacked area graph which is displaced around a central axis, resulting in a flowing, organic shape.

# In[30]:


data_sub = train.groupby('__HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].cumsum().T
data_sub.insert(0, "base", np.zeros(len(data_sub)))
data_sub = data_sub.add(-train['YearBuilt'].value_counts() / 2, axis = 0)


# In[31]:


fig, ax = plt.subplots(1, 1, figsize = (18, 5))
_color = color_[::-1]#reverse
hs_list = data_sub.columns

for i, hs in enumerate(hs_list):
    if i == 0: continue

    ax.fill_between(data_sub.index, data_sub.iloc[:, i-1], data_sub.iloc[:, i], color = _color[i-1])

for s in ['top', 'right', 'bottom', 'left']:
    ax.spines[s].set_visible(False)
    
ax.set_yticks([])
ax.grid()
plt.show()


# <h3>Ratio in line graph</h3>

# In[32]:


# sns.set_style('white')

data_sub = train.groupby('__HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].T
data_sum = data_sub.sum(axis = 1)
data_sub = (data_sub.T / data_sum).cumsum().T

fig, ax = plt.subplots(1, 1, figsize = (18, 5))

for i, hs in enumerate(train['__HouseStyle'].value_counts().index):
    hs_built = data_sub[hs]
    ax.fill_between(hs_built.index, 0, hs_built, color = color_[i])
    
ax.grid()
ax.set_ylim(0, 1)
ax.set_xlim(1872, 2010)
plt.show()


# <h2>Variances of correlated variables with target variable</h2>

# Let's look at <a href = "#section1">Detail of the numeric variables</a> and check the variance of the variables that are strongly correlated with target variable.

# In[33]:


data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis = 1)
data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', alpha = 0.3, ylim = (0, 800000));


# By using **plotly**, we can check details from figure.<by>Hover the cursor over the figure. You will be able to see it, right?

# In[34]:


data = pd.concat([train['SalePrice'], train['LotArea']], axis = 1)
fig = px.scatter(data, x = "LotArea", y = "SalePrice", trendline = "ols")
fig.show();


# In[35]:


data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', alpha = 0.3, ylim = (0, 800000));


# In[36]:


corr = train.corr()
plt.subplots(figsize = (15, 12))
sns.heatmap(corr, vmax = 0.9, square = True);


# This makes it hard to see, doesn't it?

# In[37]:


plt.figure(figsize = (12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], cmap = 'viridis', vmax = 0.9, linewidth = 0.1, annot = True, annot_kws = {'size': 8}, square = True);


# There is another way to focus only on the more useful features. In this notebook, we will take a look at this one.

# In[38]:


k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index#class 'pandas.core.indexes.base.Index'
print('top 10 relative variables \n', train[cols].columns)
print('-- top 10 relative variables-- \n')
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cmap = 'Greens', annot = True, cbar = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)


# **OverallQual** was found to be very correlated with SalePrice.<br> Of the variables that are strongly correlated with SalePrice, GarageCars and GarageArea are **similar**. Therefore, **GarageCars**, which has a stronger correlation, is retained.<br>TotalBsmtSF and 1stFloor are also similar. Therefore, we leave **TotalBsmtSF**, which has a stronger correlation.<br>TotRmsAbvGrd and GrLivArea are also similar. Therefore, we leave **GrLivArea**.

# In[39]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()


# In[40]:


total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False)
missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing.head(20)


# In[41]:


df_train = train.drop(missing[missing['Total'] > 1].index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max())


# In[42]:


print(len(df_train.columns))
df_train.head(10)


# In[43]:


print(np.newaxis)


# In[44]:


saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('low\n')
print(low_range)
print('\n')
print('high\n')
print(high_range)


# In[45]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));


# In[46]:


df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[47]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000));


# In[48]:


sns.distplot(df_train['SalePrice'], fit = norm);
# mu, sigma = norm.fit(df_train['SalePrice'])
# print(mu, sigma)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot = plt)


# The ML model doesn't work well with non-normally scattered data.<br>Therefore, we apply *log(1 + x)* function to solve the problem.

# In[49]:


df_train["SalePrice"] = np.log1p(df_train["SalePrice"])


# In[50]:


sns.distplot(df_train["SalePrice"], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train["SalePrice"], plot = plt)


# Compare with above figure, there is appropriate normality by adopting log function.<br>Let's check at some other variables!

# In[51]:


sns.distplot(df_train['GrLivArea'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot = plt)


# In[52]:


df_train['GrLivArea'] = np.log1p(df_train['GrLivArea'])


# In[53]:


sns.distplot(df_train['GrLivArea'], fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot = plt)


# In[54]:


sns.distplot(df_train['TotalBsmtSF'], fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot = plt)


# There is a problem. The value zero doesn't allow us to do log transformations.<br>Therefore, we process the data like following.

# In[55]:


# df_train['TotalBsmtSF'] = np.log1p()
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1


# In[56]:


df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log1p(df_train['TotalBsmtSF'])


# In[57]:


sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)


# In[58]:


plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# Compared to the bivariate scatter plot before the log transformation, it has a nice shape without a cone. <br>***This is the power of normality!***

# In[59]:


plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice']);


# <h2>Thank you !!!!!</h2>

# If you find this notebook useful, please **upvote**!<br>
# And if you have any questions, please ask on the **comments**!

# <h2>Other work</h2>

# * [Over 0.8!】Titanic_more_technical_EDA(ENG&JPN)](https://www.kaggle.com/fightingmuscle/over-0-8-titanic-more-technical-eda-eng-jpn)
# <br>This is my notebook.If you read this notebook, you can learn data-science **from scratch**!<br>
# * [EDA more technical🔥](https://www.kaggle.com/fightingmuscle/eda-more-technical)<br> This notebook can score 0.12 on House-Price problem with <b>detail explanation</b>.
# 
# * [How did I get the silver medal?(0.717)【Infer】](https://www.kaggle.com/fightingmuscle/how-did-i-get-the-silver-medal-0-717-infer/comments)
# 
# > I got silver medal for the first time! I published my inference code and trained models. My training code will be available soon, please wait. 
