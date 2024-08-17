#!/usr/bin/env python
# coding: utf-8

# <h1 style='text-align: center;'>HousePrices Regression</h1><br>
# 
# # Introduction
# <p>Hello!<br><br>This is my first notebook. After studying some notebooks of this competition, I summarized the overall basic process of regression analysis. This kernel introduces the following.</p>
# 
# * [EDA](#section1)<br>
# The process of searching for data is the most important step in analysis.<br>
# I will introduce how to define the type of each variable and how to identify and visualize the form.
# * [Feature Engineering](#section2)<br>
# This part is the step of preprocessing each variable based on the EDA results.<br>
# I will introduce preprocessing tasks such as outlier processing, missing value processing, derivative variable generation, and variable conversion.
# * [Optimization (GridSearch, Optuna)](#section3)<br>
# The more complex the model is, the more complex the hyperparameter setting is required.<br>
# Hyperparameter combinations can have a significant impact on the performance of the model.<br>
# I will introduce how to optimize the hyperparameters of the model using GridSearchCV and Optuna.</li>
# * [Modeling](#section4)<br>
# This part is the step of creating and learning basic models that can be used for regression analysis.<br>
# I will use linear regression models such as Lasso and Ridge, SVM, and some tree-based algorithms. And I will use Stacking to maximize generalization performance.</li>

# <h2 style="text-align:center;">Module import</h2>

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', 5000)
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('darkgrid')
from scipy.stats import norm, skew, probplot
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge as krr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr, GradientBoostingRegressor as gbr
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.cluster import DBSCAN

import optuna
from functools import partial


# <h2 style="text-align:center;">Load Data</h2>

# In[2]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


print(f'train size: {train.shape}')
print(f'test size: {test.shape}')


# <a id="section1"></a>
# # EDA

# <h3 style="text-align:center;">1. Exploring dependent variables</h3><br>
# First, I tried to understand the dependent variable.<br>
# I calculated skewness and kurtosis to determine the normality of the dependent variable, and wrote a histogram and a QQ plot.<br>
# <h4 style='color:crimson'>Normality</h4>
# Normality means that the distribution of variables <b style='color:crimson'>follows a normal distribution.</b><br>
# The easiest way to test normality is to draw a <b style='color:crimson'>Histogram and a QQ plot.</b><br>
# It is recommended that the histogram form a bell shape. It is good to understand to see skewness and kurtosis together.<br>
# If there is a shape extending along the baseline to the top right in the QQ plot, we can see that the data has normality.<br><br>
# If it violates normality, <b style='color:crimson'>log transformation</b> or boxcox transformation can be applied.

# In[6]:


f = plt.figure(facecolor='whitesmoke', figsize=(20, 5))

ax_left = f.add_axes([0,0,.2,1], facecolor='whitesmoke')
ax_left.axis('off')
ax_left.text(.4, .9, 'SalePrice', color='crimson', size=22, weight='bold')
ax_left.text(.1, .8, f'Skew: {train.SalePrice.skew():.2f}', size=20)
ax_left.text(.1, .7, f'kurt: {train.SalePrice.kurt():.2f}', size=20)
ax_left.text(.1, .6, f'missing count: {train.SalePrice.isnull().sum()}', size=20)
ax_left.text(.1, .4, 'conclusion: ', color='crimson', size=20)
ax_left.text(.1, .3, 'It is necessary to convert', size=20)
ax_left.text(.1, .2, 'such as log transformation.', size=20)

ax_right1 = f.add_axes([0.25,0,.3,.8], facecolor='whitesmoke')
sns.distplot(train.SalePrice, fit=norm, ax=ax_right1)
ax_right1.spines[['top', 'right']].set_visible(False)
ax_right1.set_title('histogram', color='crimson', weight='bold', size=15)

ax_right2 = f.add_axes([.57,0,.3,.8], facecolor='whitesmoke')
probplot(train.SalePrice, plot=ax_right2)
ax_right2.spines[['top', 'right']].set_visible(False)
ax_right2.set_title('QQ plot', color='crimson', weight='bold', size=15)

plt.show()


# <h3 style="text-align:center;">2. Exploring independent variables - Define Type</h3><br>
# <p>
#     I classified the variables into <b style='color:crimson'>categorical variables and numeric variables.</b><br>
#     We can read the description of the variable, determine the type of variable, and further derive ideas for generating derived variables and converting data.<br><br>
#     The types of variables are classified as follows.<br>
#     <ol>
#         <li><b style='color:crimson'>Categorical</b></li>
#         <ol>
#             <li>Nominal vars</li>
#             <li>Order(Rank) vars</li>
#         </ol>
#         <li><b style='color:crimson'>Numerical</b></li>
#         <ol>
#             <li>Interval vars</li>
#             <li>Ratio vars</li>
#         </ol>
#     </ol>
#     The method of searching and preprocessing is determined by the type of variable.<br>
#     <br>
#     By using Pandas, we can find out the data type (numerical type and object) of the variable.<br>
#     However, not all numerical variables can be determined as continuous variables.<br>
#     I divided them subjectively by referring to <b style='color:crimson'>data_description file.</b><br>
#     <br>
# </p>
# Check numerical and object variables!

# In[7]:


num_vars = train.columns[train.dtypes != 'object']
obj_vars = train.columns[train.dtypes == 'object']

print('\nNumerical vars: ')
print(num_vars.values)
print('\nObject vars: ')
print(obj_vars.values)


# <b>However, we can see that MSSubClass is a Nominal variable.<br>
# <b>And we can see that OverallQual, QverallCond, and Year, Month are ordered (or ranked) variables.

# <h3 style="text-align:center;">3. Exploring independent variables - Check the missing values</h3>

# In[8]:


all_data = pd.concat((train, test)).drop(['SalePrice'], axis=1)
cnt_missing = all_data.isnull().sum().sort_values(ascending=False)
cnt_percent = cnt_missing / all_data.shape[0] * 100
missing_table = pd.DataFrame([cnt_missing, cnt_percent], 
                             index=['missing count', 'missing percent']).T
missing_table = missing_table[missing_table['missing count'] > 0]
missing_table = missing_table.reset_index()
missing_table['missing count'] = missing_table['missing count'].astype(int)


# In[9]:


color_list=[['whitesmoke', 'white', 'white']]

fig = plt.figure(facecolor='whitesmoke')
ax1 = fig.add_axes([0, 0, 1, 0.1]) 
ax2 = fig.add_axes([1.5, -2.3, 1, 2.3], facecolor='whitesmoke') 

ax2.spines[['top', 'right']].set_visible(False)

ax1.set_axis_off()

table=ax1.table(cellText = missing_table.values[:20], colLabels=missing_table.columns,
                  colColours=['crimson']*3, cellColours=color_list*20)
table.auto_set_font_size(False) 
table.set_fontsize(16)  
table.scale(1.5, 2.7) 
ax1.text(0.67, .9, 'Missing count and percent', color='crimson', fontsize=20, fontweight='bold')
ax1.text(1.4, .9, 'by values', fontsize=20, fontweight='bold')

sns.barplot(y=missing_table['index'], x=missing_table['missing percent'], orient = "h", ax=ax2)
plt.show()


# <h4>Many variables have missing values. Some variables have extremely large amounts of missing values.</h4>

# <h3 style="text-align:center;">4. Exploring independent variables - distribution</h3>

# In[10]:


f, ax = plt.subplots(7, 6, figsize=(25, 25), facecolor='whitesmoke')
num_vars = all_data.columns[all_data.dtypes != 'object']
for i, c in enumerate(num_vars):
    g = sns.distplot(all_data[c], fit=norm, ax=ax[i//6, i%6], color='crimson')
    g.set_facecolor('whitesmoke')
    g.spines[['top', 'right']].set_visible(False)
f.text(0.4, .92, 'Distribution of numerical vars', size=20, weight='bold', color='crimson')
plt.show()


# Some variables seem to be able to follow a normal distribution through log transformation or box cox transformation. Some variables have zero. It seems that +1 should be done when converting.
# 
# Draw a bar chart for categorical variables.
# 
# Some variables were extremely biased toward one value (0). So they don't seem to be very important variables.

# In[11]:


f, ax = plt.subplots(9, 6, figsize=(25, 26), facecolor='whitesmoke')
cat_vars = all_data.columns[all_data.dtypes == 'object']
for i, c in enumerate(cat_vars):
    g = sns.barplot(data=pd.DataFrame(all_data[c].value_counts()).reset_index(), x='index', y=c, ax=ax[i//6, i%6], color='crimson')
    g.set(xticks=[])
    g.set(title=c)
    g.set_facecolor('whitesmoke')
    g.spines[['top', 'right']].set_visible(False)
f.text(0.4, .92, 'Distribution of categorical vars', size=20, weight='bold', color='crimson')
plt.show()


# <h3 style="text-align:center;">5. Bivariate search - correlation analysis, heat map, finding important variables</h3><br>
# <h4>Outline</h4>
# Correlation analysis is a technique to find out the correlation between the two variables.<br>
# We can find variables with relatively strong <b>'linearity'</b> using the correlation coefficient with the dependent variable.<br>
# I designated these variables as relatively important variables and tested <b>'homogeneity'</b>.<br>
# In addition, I tested <b>'independence'</b> through the correlation between independent variables.<br>
# <br>
# <h4>Some assumptions for a good regression model</h4>
# Previously, we checked the normality of the dependent variable and the residual.<br>
# In addition, several more assumptions are needed.<br>
# <ol>
#     <li>
#         <b>Linearity</b>:<br>
#         It is recommended that the independent variable and the dependent variable have a linear relationship.<br>
#         The 'variable transformation' or 'dimensional increase' method can help to have linearity.
#     </li>
#     <li>
#         <b>homogeneity:</b>:<br>
#         The variance of the residuals must be constant.<br>
#         Drawing a residual diagram for an independent variable can test the equal variance.<br>
#         If the points follow randomly based on the baseline, they satisfy the equal variance.
#     </li>
#     <li>
#         <b>Independence</b>:<br>
#         Independence means that there should be no correlation between independent variables.<br>
#         The high correlation between independent variables causes multicollinearity. As a result, the performance of the model becomes incredible.
#     </li>
#     <li>
#         <b>Irregularity</b>:<br>
#         There should be no correlation between residuals.<br>
#         Durbin-Watson' helps test for non-correlation.
#     </li>
# </ol>

# In[12]:


f, ax = plt.subplots(figsize=(10, 7))
highcorr_vars = (abs(train.corr().SalePrice).sort_values(ascending=False)[:7]).index
sns.heatmap(train[highcorr_vars].corr(), annot=True)
plt.show()


# <h4>I selected only the top 10 variables with the highest correlation with the dependent variable and conducted the correlation analysis again.<br>
# "OverallQual" and "GrLiv Area" are the strongest variables.<br>
# Garage Cars and Garage Area are also highly correlated. High correlation between independent variables is not good because it causes multicollinearity.<br><br>
# I used the following chart for some tests.<br>
#     <b style='color:crimson'>Scatter, Boxplot, resid plot, histogram, QQplot</b></h4>

# In[13]:


def hypo_test(x, y, cat=False):
    f, ax = plt.subplots(1, 4, figsize=(25, 5), facecolor='whitesmoke')
    if cat:
        sns.boxplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
    else:
        sns.scatterplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
        sns.regplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
    sns.residplot(x=train[x], y=train[y], ax=ax[1], color='crimson')
    sns.distplot(train[x], fit=norm, ax=ax[2], color='crimson')
    probplot(train[x], plot=ax[3])
    ax[0].set_facecolor('whitesmoke')
    ax[1].set_facecolor('whitesmoke')
    ax[2].set_facecolor('whitesmoke')
    ax[3].set_facecolor('whitesmoke')
    ax[0].spines[['top', 'right']].set_visible(False)
    ax[1].spines[['top', 'right']].set_visible(False)
    ax[2].spines[['top', 'right']].set_visible(False)
    ax[3].spines[['top', 'right']].set_visible(False)
    
    f.suptitle(f'{x}', color='crimson', weight='bold', size=20)
    
    plt.show()


# In[14]:


hypo_test('OverallQual', 'SalePrice', True)


# In[15]:


hypo_test('GrLivArea', 'SalePrice')


# In[16]:


hypo_test('GarageArea', 'SalePrice')


# In[17]:


skews = abs(all_data.skew()).sort_values(ascending=False)
kurts = abs(all_data.kurt()).sort_values(ascending=False)
skew_kurt_table = pd.DataFrame([skews, kurts], index=['skew', 'kurt']).T
ntv = skew_kurt_table[skew_kurt_table['skew'] > 0.5].index

plt.subplots(figsize=(15, 5))
sns.barplot(x=skew_kurt_table.loc[ntv].index, y=skew_kurt_table.loc[ntv]['skew'])
plt.xticks(rotation='90')
plt.title('skew by variable', size=20)
plt.xlabel('vars', size=15)
plt.ylabel('skew', size=15)
plt.show()


# <p>
#     <font color='red'>Conclusion:</font><br>
#     I analyzed the correlation of each variable for SalePrice and confirmed that variables such as OverallQual, GrLivArea, and CarArea had high linearity.<br>
#     Linearity was visualized using a box plot and scatter plot.<br>
#     And I thought these variables were important variables, so I drew a residual diagram for the homodis variance test, a histogram for the normality test, and a QQ plot for them.<br>
#     <br>
#     Each variable does not satisfy the equal variance because the points of the residual degree are not randomly sprayed and have some pattern or shape.<br>
#     I could see that each variable had no ideal normality through the results of the histogram and QQplot.<br>
#     The above problems may be solved through log transformation or boxcox transformation.<br>
#     <br>
#     It was also confirmed that some independent variables were correlated with each other. As expected, I visualized it as a scatter plot.<br>
#     The high correlation between independent variables causes multicollinearity. The explanatory power of the model loses its reliability.<br>
#     I decided to use regulation rather than choosing variables or using dimension reduction right away.<br>
#     The linear model may solve the above problem through regulation (normal1, normal2).<br>
#     <br>
#     It was difficult to visualize all variables, so I looked at the independent variables that required conversion through skewness and kurtosis.
# </p>

# <a id="section2"></a>
# # Feature Engineering

# <h3 style="text-align:center;">1. Remove ID</h3><br>
# <h4>Outline</h4>
# I removed the ID variable that was not needed for analysis.

# In[18]:


train_id = train.Id
test_id = test.Id
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# <h3 style="text-align:center;">2. Clensing - Outlier</h3><br>
# <h4>Outline</h4>
# A dataset with outliers can degrade the performance of the model.<br>
# It is optional to remove outliers existing in the training data. I found two outliers in the scatter plot of the variable with a strong correlation with the dependent variable.<br>
# The two points were located in a place far off the straight line.<br><br>
# 
# <h4>Finding outliers</h4>
# <ol>
#     <li><b>Statistics for univariate - normal distribution, IQR:</b><br>For independent variables, points outside the threshold can be judged as outliers using normal distributions, likelihood functions, IQR, etc.<br>
#         However, it cannot be judged that the point outside the threshold is always an outlier. For example, the scatterplot of GrLiv Area has two points apart at the top right.<br>
#         These differ greatly in value from other points, but they are important because they can prove linearity.<br>Therefore, it is necessary to carefully deal with the determination of outliers for univariate quantities.</li>
#     <li><b>Scattering point:</b><br>If you draw a scatterplot of two variables with a pattern (e.g., a linear relationship), you can intuitively find outliers.<br>
#         I previously found some variables that have a strong correlation with the dependent variable. Two outliers were found through GrLiv Area's scatter plot.</li>
#     <li><b>Clustering - DBSCAN:</b><br>DBSCAN can detect outliers using distance.<br>
#         DBSCAN has a set range (epsilon) and required peripheral points (min_samples), and generates clusters by calculating key points and peripheral points.<br>
#         Points that do not have key points around and do not have the minimum required neighboring points are outliers.<br>
#         I applied scaling to GrLiv Area and SalePrice and tried DBSCAN.<br></li>
# </ol>

# In[19]:


plt.subplots(figsize=(20, 10))
outlier_idx = train.GrLivArea.sort_values(ascending=False)[:2].index
sns.scatterplot(x=train['GrLivArea'], y=train.SalePrice)
sns.scatterplot(x=train.iloc[outlier_idx]['GrLivArea'], y=train.iloc[outlier_idx].SalePrice, color='r', s=300, alpha=.6)
plt.show()


# In[20]:


scaled_data = pd.DataFrame(StandardScaler().fit_transform(train[['GrLivArea', 'SalePrice']]), columns=['GrLivArea', 'SalePrice'])
dbscan_model = DBSCAN(eps=1.5, min_samples=3).fit(scaled_data)
tmp = pd.concat((scaled_data, pd.DataFrame(dbscan_model.labels_, columns=['label'])), axis=1)


# In[21]:


tmp.label.value_counts()


# In[22]:


sns.pairplot(tmp, hue='label', size=5)
plt.show()


# In[23]:


train.drop(train.GrLivArea.sort_values(ascending=False)[:2].index, axis=0, inplace=True)


# <p>
#     <font color='red'>Conclusion:</font><br>
#     Two outliers were found through the scatterplot. Two points were removed.<br>
#     In the case of DBSCAN, a total of 4 points were judged as outliers.
# </p>

# <h3 style="text-align:center;">3. Train, Test merge, separating dependent variables.</h3><br>
# <h4>Outline</h4>
# I removed outliers from the training data. Now I combine with the test data. The dependent variable is separated separately.

# In[24]:


train_size = train.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test), axis=0).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# In[25]:


train.shape, test.shape, all_data.shape, y_train.shape


# <h3 style="text-align:center;">4. Clensing - Missing Value</h3><br>
# <h4>Outline</h4>
# Missing values are objects that must be removed. The missing values are processed based on the understanding of each independent variable.
# <br><br>
# <h4>How to deal with missing values.</h4>
# <ol>
#     <li><b>Delete:</b><br>
#         It is the easiest and most powerful way. Clear rows or columns. However, avoid erasing rows because missing values may also exist in the test data.<br>
#         Clearing the columns risks removing important variables, so avoid them if possible.</li>
#     <li><b>Replace a specific value:</b><br>
#         It is a way to try if you have knowledge of variables. For example, if there are extremely many missing values of iso-interstitial variables that can have negative meanings,<br>
#         Missing values are likely to have a negative meaning.
#     <li><b>Replacement of central propensity:</b><br>
#         It can be replaced with a central tendency value such as an average, a median value, and a minimum value. It is possible to replace the continuous variable with the median value and the category variable with the lowest value.</li>
#     <li><b>Other than that: Simple probability replacement, multiple confrontation method, etc.</b><br>
# </ol>
# After looking at the explanation of each variable, I treated it as follows.<br>
# <ul>
#     <li>
#         Continuous: Select one relevant category variable and replace the median for each category.<br>or
#         Alternate to 0 if it is bound to be a missing value.
#     </li>
#     <li>
#         Category type: value_counts to identify the distribution and replace it with None if there is zero negative meaning, such as NA or POOL. or<br>Select one related category variable, identify the distribution of values for each category, and replace the poorest value.
#     </li>
# </ul>

# <p>
#     Variables that are currently missing.
# </p>

# In[26]:


all_data.columns[all_data.isnull().sum() > 0]


# ex1) LotFrontage is replaced by the median value per Neighborhood.

# In[27]:


sns.scatterplot(x=all_data.Neighborhood, y=all_data.LotFrontage)
plt.xticks(rotation='90')
plt.show()


# ex2) Ally is a ranking variable that can have NA, and since these values have been treated as missing values, replace them with None.

# In[28]:


all_data.Alley.value_counts()


# ex3) Utilities are extremely skewed to AllPub, so replace them with the lowest value.

# In[29]:


all_data.Utilities.value_counts()


# ex4) Garage Area is a missing value in the absence of Garage, so it is replaced by 0.

# In[30]:


all_data.GarageArea.value_counts()


# In[31]:


for c in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC",
          'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
    all_data[c] = all_data[c].fillna('None')
    
for c in ['GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1',
          'BsmtFinSF2','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
          'BsmtUnfSF','TotalBsmtSF']:
    all_data[c] = all_data[c].fillna(0)

for c in ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']:
    all_data[c] = all_data[c].fillna(all_data[c].mode()[0])

all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['Functional'] = all_data['Functional'].fillna('Typ')


# In[32]:


all_data.isnull().sum().sum()


# <h3 style="text-align:center;">5. Transform numeric variables - LabelEncoder, Boxcox, Log</h3><br>
# Log and boxcox transformations are applied to secure the normality and equal variance of continuous variables.<br>
# Label encoding is applied to ordered (priority) variables.<br>
# One-hot encoding is applied to equal variables.<br>
# Log transformation is applied to the dependent variable to secure the normality of the dependent variable.

# In[33]:


train[['MSSubClass', 'GarageYrBlt', 'MoSold', 'OverallCond', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'YrSold']] =\
train[['MSSubClass', 'GarageYrBlt', 'MoSold', 'OverallCond', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'YrSold']].astype('str')

test[['MSSubClass', 'GarageYrBlt', 'MoSold', 'OverallCond', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'YrSold']] =\
test[['MSSubClass', 'GarageYrBlt', 'MoSold', 'OverallCond', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'YrSold']].astype('str')

order_vars = [
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageYrBlt', 'GarageQual',
    'GarageCond', 'PoolQC', 'MoSold', 'YrSold'
]


# In[34]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness.Skew) > 0.5]
skewness


# In[35]:


skewness = skewness[abs(skewness) > 0.5]

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], boxcox_normmax(all_data[feat] + 1))


# In[36]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness.Skew) > 0.5]
skewness


# In[37]:


for c in order_vars:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


# In[38]:


y_train = np.log1p(y_train)


# <h3 style="text-align:center;">6. Derivative variable generation</h3><br>
# 
# Derivative variables are methods that can improve the quality of analysis. There are several ideas for how to generate derivative variables.
# 
# It is a continuous variable and means an observation of an object, and if this value is 0, it means that there is no object.
# Therefore, it is possible to add categorical variables as to whether or not the object is present. Categorical variables with binary values can be stored as 0 and 1.
# 
# If you can express continuous variables of the same series in association, use a four-line operation.
# New variables can be created. It is similar to calculating BMI with height and weight.
# 
# The above judgment can be used using min and max values after describe.
# 
# 

# In[39]:


all_data['Haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['Hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data['TotalSF'] = (all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'])
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))                              
all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])


# In[40]:


all_data = pd.get_dummies(all_data)


# <h3 style="text-align:center;">7. Select variables</h3><br>
# Using variable selection can help improve the performance of the model.<br>
# There are various approaches to the variable selection method, but I removed only a few variables that were most simply biased to one side.<br>

# In[41]:


all_data.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1, inplace=True)


# In[42]:


all_data.shape


# <h3 style="text-align:center;">8. Training, test data division</h3><br>
# Divide the training data and test data again.

# In[43]:


X_train, X_test = all_data.iloc[:train_size, :], all_data.iloc[train_size:, :]


# In[44]:


X_train.shape, X_test.shape, y_train.shape


# <a id="section3"></a>
# # Optimization (GridSearch, Optuna)<br>
# In order to maximize the performance of the model, the process of finding the appropriate hyperparameters is required. I used GridSearch to find hyperparameters for each model.
# <br><br>
# Sklearn's GridSearchCV was used for Laso, Ridge, ElasticNet, and SVM, and Optuna was used for tree-based models.

# In[45]:


def rmsle_cv(model):
    return np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring='neg_mean_squared_error',
                   cv=5, verbose=0, n_jobs=-1))


# <h3 style="text-align:center;">1. GridSearch Cross Validation</h3><br>

# In[46]:


# model_lasso = Pipeline([
#     ('scaler', RobustScaler()),
#     ('model', Lasso())
# ])
# model_elasticNet = Pipeline([
#     ('scaler', RobustScaler()),
#     ('model', ElasticNet(max_iter=5000))
# ])
# model_krr = Pipeline([
#     ('scaler', RobustScaler()),
#     ('model', krr())
# ])

# model_svr = Pipeline([
#     ('scaler', RobustScaler()),
#     ('model', SVR())
# ])

# grid_param_lasso = {
#     'model__alpha': 0.0001 * np.arange(1, 100)
# }
# grid_param_elasticNet = {
#     'model__alpha': 0.0001 * np.arange(1, 100),
#     'model__l1_ratio': 0.001 * np.arange(1, 10)
# }
# grid_param_krr = {
#     'model__alpha': 0.0001 * np.arange(1, 100),
#     'model__degree': [1, 2, 3],
#     'model__kernel': ['polynomial'],
#     'model__coef0': [2.5]
# }
# grid_param_svr = {
#     'model__C': [0.001, 0.1, 1, 10, 20],
#     'model__gamma': [.0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, .001],
#     'model__epsilon': [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
# }

# best_params = {}


# In[47]:


# search_lasso = GridSearchCV(model_lasso, grid_param_lasso, scoring='neg_mean_squared_error',
#                            cv=5, n_jobs=-1, verbose=0).fit(X_train, y_train)
# best_params['Lasso'] = search_lasso.best_params_


# In[48]:


# search_elasticNet = GridSearchCV(model_elasticNet, grid_param_elasticNet, scoring='neg_mean_squared_error',
#                            cv=5, n_jobs=-1, verbose=0).fit(X_train, y_train)
# best_params['ElasticNet'] = search_elasticNet.best_params_


# In[49]:


# search_krr = GridSearchCV(model_krr, grid_param_krr, scoring='neg_mean_squared_error',
#                            cv=5, n_jobs=-1, verbose=0).fit(X_train, y_train)
# best_params['KernelRidge'] = search_krr.best_params_


# In[50]:


# search_svr = GridSearchCV(model_svr, grid_param_svr, scoring='neg_mean_squared_error',
#                            cv=5, n_jobs=-1, verbose=0).fit(X_train, y_train)
# best_params['SVR'] = search_svr.best_params_


# <h3 style="text-align:center;">2. Optuna</h3><br>
# 
# <p>
# The hyperparameters of tree models are diverse and have many combinations. Their optimization takes a lot of time.
# 
# So, I looked for XGBoost's hyperparameters using the Optuna package using early stopping and cross validation.
# 
# The hyperparameters of tree models are diverse and have many combinations. Their optimization takes a lot of time.
# 
# I looked for XGBoost's hyperparameters using the Optuna package using early stopping and cross validation.
# 
# Preparations: 'optuna', 'functions-partial', objective function
# 
# Pre-understanding:
# 
# Optuna is a framework that helps optimize hyperparameters. An objective function is required.
# 
# Optuna's objective function selects a new hyperparameter combination of the model every trial.
# 
# Optuna's study object is an object that performs optimization. The optimization of the study object requires a partial object and the number of attempts.
# 
# The partial object is an object that binds X, y with the objective function to be used by optuna.
# 
# Study objects store results that meet the purpose for each trial. Finally, remember the most purposeful hyperparameter combination.
# 
# The trial factor of objective embeds the function of specifying the range and value of hyperparameters. It has a hyperparameter name, range or list as a factor in common.
#     <ol>
#         <li><b>Suggest_int:</b> Select an integer value within the range.</li>
#         <li><b>Suggest_uniform:</b> Select an equal distribution value within a range.</li>
#         <li><b>Suggest_discrete_uniform:</b> Select a discrete uniform distribution value within the range.</li>
#         <li><b>Suggest_loguniform:</b> Select a logarithmic function linear value within a range.</li>
#         <li><b>Suggest_category:</b> Select a value in the list.</li>
#     </ol>
# </p>

# Create Object Function.<br>
# Hyperparameters and cross-validation were implemented inside the function.

# In[51]:


# def objective_xgb(trial, X, y):
#     param = {
#         'n_estimators': 1900,
#         'max_depth': trial.suggest_int('max_depth', 3, 11),
#         'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.01),
#         'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         'colsample_bylevel': trial.suggest_categorical('colsample_bylevel', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 100),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 100),
#         'n_jobs': -1
#     }
#     train_scores, test_scores = [], []
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     model = XGBRegressor(**param)
#     for train_idx, test_idx in kf.split(X):
#         tmp_X_train, tmp_X_test = X_train.iloc[train_idx, :], X_train.iloc[test_idx, :]
#         tmp_y_train, tmp_y_test = y_train[train_idx], y_train[test_idx]
#         model.fit(tmp_X_train, tmp_y_train,
#                  eval_set=[(tmp_X_test, tmp_y_test)], eval_metric=['rmse'],
#                  early_stopping_rounds=30, verbose=0,
#                  callbacks=[optuna.integration.XGBoostPruningCallback(trial, observation_key='validation_0-rmse')])
#         train_score = np.sqrt(mse(tmp_y_train, model.predict(tmp_X_train)))
#         test_score = np.sqrt(mse(tmp_y_test, model.predict(tmp_X_test)))
#         train_scores.append(train_score)
#         test_scores.append(test_score)
#     train_score = np.array(train_scores).mean()
#     test_score = np.array(test_scores).mean()
#     print(f'train score: {train_score}')
#     print(f'test score: {test_score}')
#     return test_score


# In[52]:


# def objective_lgbr(trial, X, y):
#     param = {
#         'n_estimators': 2000,
#         'max_depth': trial.suggest_int('max_depth', 3, 11),
#         'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.01),
#         'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 100),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 100),
#         'n_jobs': -1
#     }
#     train_scores, test_scores = [], []
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     model = LGBMRegressor(**param)
#     for train_idx, test_idx in kf.split(X):
#         tmp_X_train, tmp_X_test = X_train.iloc[train_idx, :], X_train.iloc[test_idx, :]
#         tmp_y_train, tmp_y_test = y_train[train_idx], y_train[test_idx]
#         model.fit(tmp_X_train, tmp_y_train,
#                  eval_set=[(tmp_X_test, tmp_y_test)], eval_metric=['rmse'],
#                  early_stopping_rounds=30, verbose=0,
#                  callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmse')])
#         train_score = np.sqrt(mse(tmp_y_train, model.predict(tmp_X_train)))
#         test_score = np.sqrt(mse(tmp_y_test, model.predict(tmp_X_test)))
#         train_scores.append(train_score)
#         test_scores.append(test_score)
#     train_score = np.array(train_scores).mean()
#     test_score = np.array(test_scores).mean()
#     print(f'train score: {train_score}')
#     print(f'test score: {test_score}')
#     return test_score


# Create partial objects using the object function and create study objects of optuna.<br>
# By optimizing study, parameters can be optimized.

# In[53]:


# optimizer_xgbr = partial(objective_xgb, X=X_train, y=y_train)
# study_xgbr = optuna.create_study(direction='minimize')
# study_xgbr.optimize(optimizer_xgbr, n_trials=100)
# best_params['XGBoost'] = study_xgbr.best_params


# In[54]:


# optimizer_lgbr = partial(objective_lgbr, X=X_train, y=y_train)
# study_lgbr = optuna.create_study(direction='minimize')
# study_lgbr.optimize(optimizer_lgbr, n_trials=100)
# best_params['LightGBM'] = study_lgbr.best_params


# Optuna can be used to visualize the optimal value for each trial or variable.

# In[55]:


# optuna.visualization.plot_optimization_history(study_xgbr)


# In[56]:


# optuna.visualization.plot_slice(study_xgbr)


# In[57]:


# optuna.visualization.plot_optimization_history(study_lgbr)


# In[58]:


# optuna.visualization.plot_slice(study_lgbr)


# <a id="section4"></a>
# # Modeling

# Create each model using the hyperparameter combination found earlier (Some parameters are corrected through several trials and errors).

# In[59]:


# best_params


# In[60]:


model_lasso = Pipeline([
    ('scaler', RobustScaler()),
    ('model', Lasso(alpha=0.0002))
])

model_enet = Pipeline([
    ('scaler', RobustScaler()),
    ('model', ElasticNet(alpha=0.0078000000000000005, l1_ratio=0.009000000000000001, random_state=3))
])

model_krr = Pipeline([
    ('scaler', RobustScaler()),
    ('model', krr(alpha=0.99,
                        kernel='polynomial',
                        degree=1,
                        coef0=2.5))
])
model_svr = Pipeline([
    ('scaler', RobustScaler()),
    ('model', SVR(C= 20, epsilon= 0.05, gamma=0.0005))])

model_gbr = gbr(n_estimators=3000, learning_rate=0.009995774699700678,
                                   max_depth=8, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5)
#model_xgbr = XGBRegressor(n_estimators=2200, random_state =42, **study_xgbr.best_params)
model_xgbr = XGBRegressor(colsample_bytree=0.4, learning_rate=0.00898718134841855, max_depth=8, 
                             n_estimators=2200, reg_alpha=0.036142628805195254, reg_lambda=0.03188665185506858,
                             subsample=0.6, random_state =42)
#model_lgbm = LGBMRegressor(objective='regression', n_estimators=2200, **study_lgbr.best_params)
model_lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4, learning_rate=0.01, n_estimators=9000,
                                       max_bin=200, bagging_fraction=0.75, bagging_freq=5, 
                                       bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7,
                                       verbose=-1)
stack_gen = StackingCVRegressor(regressors=(model_lgbm, model_lasso, model_enet, model_krr, model_svr, model_gbr),
                               meta_regressor=model_xgbr,
                               use_features_in_secondary=True)


# Let's check the cross-validation results of each model.

# In[61]:


models = [
    model_lasso, model_enet, model_krr, model_svr, model_gbr, model_xgbr, model_lgbm
]
cross_score = {
    'Lasso': 0,
    'ElasticNet': 0,
    'Kernel Ridge': 0,
    'SVR': 0,
    'GradientBoosting': 0,
    'XGBoost': 0,
    'LightGBM': 0,
}

for idx, model in enumerate(models):
    cross_score[list(cross_score.keys())[idx]] = rmsle_cv(model).mean()


# In[62]:


cross_score


# After creating a blend function that can harmonize the results of multiple models, traning each model

# In[63]:


def blend(X):
    return ((0.10 * model_lasso.predict(X)) + \
            (0.10 * model_enet.predict(X)) + \
            (0.10 * model_krr.predict(X)) + \
            (0.10 * model_svr.predict(X)) + \
            (0.10 * model_xgbr.predict(X)) + \
            (0.10 * model_lgbm.predict(X)) + \
            (0.40 * stack_gen.predict(np.array(X))))


# In[64]:


for model in models:
    model = model.fit(X_train, y_train)


# In[65]:


stack_gen = stack_gen.fit(X_train, y_train)


# In[66]:


np.sqrt(mse(y_train, blend(X_train)))


# In[67]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = score = np.expm1(blend(X_test))
sub.to_csv('submission.csv',index=False)

