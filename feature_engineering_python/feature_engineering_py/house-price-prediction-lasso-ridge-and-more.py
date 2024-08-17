#!/usr/bin/env python
# coding: utf-8

# # <center style="font-family:Arial">1. Introduction </center>
# 
# <div class="alert alert-block alert-info"
#      style="color:black;
#            display:fill;
#            background-color:#e8f4f8;
#            font-size:130%;
#            font-family:Arial"><center>
# <b> 📌 My goal is to predict, in the best possible way, the sales price of the houses based on their characteristics using different linear regression models.</b></center>
#     </div>
#     
#  <div style="color:black;
#            font-size:120%;
#            font-family:Arial">
# In this notebook, I'll be working with the Ames Housing dataset, a complete dataset containing every aspect of residential homes in Ames, Iowa. If you want to know more about the data, you can click <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data" target="_blank"> here</a>.
# </div>
# 
# <hr style="height: 0.5px; border: 0; background-color: 'Black'">
# 
# 
# ![imageHouses](https://static.vecteezy.com/system/resources/previews/002/197/652/original/house-hand-drawn-set-design-illustration-isolated-on-white-background-free-vector.jpg)
# 

# ## <center style="font-family:Arial">Importing the Data </center>
# 
# 

# In[1]:


get_ipython().system('pip install proplot')
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


import pandas as pd
import numpy as np

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()


# # <center style="font-family:Arial">2. EDA </center>

# In[3]:


# Defining plots design
def plots_design():
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.yaxis.set_label_coords(0, 0)
    ax.grid(color='white', linewidth=2)
    # Remove ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Remove axes splines
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # Font
    mpl.rcParams['font.family'] = 'Source Sans Pro'


# In[4]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import proplot as pplt

corr = train[train.columns].corr()['SalePrice'][:].sort_values(ascending=True).to_frame()
corr = corr.drop(corr[corr.SalePrice > 0.99].index)

# Visualization
fig, ax = plt.subplots(figsize =(9, 9))

ax.barh(corr.index, corr.SalePrice, align='center', color = np.where(corr['SalePrice'] < 0, 'crimson', '#89CFF0'))

plots_design()

plt.text(-0.12, 39, "Correlation", size=24, color="grey", fontweight="bold");
plt.text(0.135, 39, "of", size=24, color="grey");
plt.text(0.185, 39, "SalePrice", size=24, color="#89CFF0", fontweight="bold");
plt.text(0.4, 39, "to", size=24, color="grey");
plt.text(0.452, 39, "Other Features", size=24, color="grey", fontweight="bold");

# Author
plt.text(0.9, -7, "@miguelfzzz", fontsize=11, ha="right", color='grey');


# <div style="font-size:120%">In this plot, we can see the correlation of sales price with the rest of the numerical features. These are the highest positive correlations:</div>
# 
# * <div style="font-size:120%"><code>OverallQual</code>: Overall material and finish quality</div>
# * <div style="font-size:120%"><code>GrLivArea</code>: Above grade (ground) living area in square feet</div>
# * <div style="font-size:120%"><code>GarageCars</code>: Size of garage by car capacity</div>
# * <div style="font-size:120%"><code>GarageArea</code>: Size of garage in square feet</div>
# * <div style="font-size:120%"><code>TotalBsmtSF</code>: Total square feet of basement area</div>
# * <div style="font-size:120%"><code>1stFlrSF</code>: First Floor square feet</div>

# In[5]:


# pairplot top 10 correlation features + target
top_corr = corr['SalePrice'].sort_values(ascending=False).head(10).index
top_corr = top_corr.union(['SalePrice'])

sns.pairplot(train[top_corr]);


# # <center style="font-family:Arial">3. Data Processing and Cleaning </center>

# In[6]:


print('Training Shape:', train.shape)
print('Test Shape:', test.shape)


# In[7]:


# let's save the ID of each dataset
train_id = train['Id']
test_id = test['Id']
del train['Id']
del test['Id']


# ## <center style="font-family:Arial">Outliers</center>

# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">Focusing on the target variable (SalePrice), we can see that there are some outliers in features such as <code>GarageArea</code>, <code>GrLivArea</code> and <code>TotalBsmtSF</code>. </div>

# In[8]:


train1 = train.copy()
train1 = train1.drop(train1[(train1['GarageArea']>1200) & (train1['SalePrice']<300000)].index)
train1 = train1.drop(train1[(train1['GrLivArea']>4000) & (train1['SalePrice']<300000)].index)
train1 = train1.drop(train1[(train1['TotalBsmtSF']>5000)].index)


# In[9]:


print('Outliers removed =' , train.shape[0] - train1.shape[0])


# ## <center style="font-family:Arial">Split X and y</center>

# In[10]:


# Split X and y (in train dataset)
X = train1.drop('SalePrice', axis=1)
y = train1['SalePrice'].to_frame()

# Add variable
X['train'] = 1
test['train'] = 0

# Combining train and test for data cleaning 
df = pd.concat([test, X])


# In[11]:


print('Count of Features per Data Type:')
df.dtypes.value_counts()  


# In[12]:


# Do we have duplicates?
print('Number of Duplicates:', len(df[df.duplicated()]))

# Do we have missing values?
print('Number of Missing Values:', df.isnull().sum().sum())


# # <center style="font-family:Arial">4. Feature Engineering </center>

# ## <center style="font-family:Arial">Missing values</center>

# In[13]:


print('Missing Values per Column:')
df.isnull().sum().sort_values(ascending=False).head(25)


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>PoolQC</code> refers to the pool quality of the house. Data description says that having a NaN in this category means that the house doesn't have a pool.</div>

# In[14]:


df['PoolQC'] = df['PoolQC'].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>MiscFeature</code> refers to miscellaneous features of the house. Data description says that having a NaN in this category means that the house doesn't have any.</div>

# In[15]:


df['MiscFeature'] = df['MiscFeature'].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>Alley</code> refers to the type of alley access to the property. Data description says that having a NaN in this category means that the house doesn't have any.</div>

# In[16]:


df['Alley'] = df['Alley'].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>Fence</code> refers to the type of fencing around the property. Data description says that having a NaN in this category means that the house doesn't have a fence.</div>
# 

# In[17]:


df['Fence'] = df['Fence'].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>FireplaceQu</code> refers to the quality of the fireplace. Data description says that having a NaN in this category means that the house doesn't have a fireplace.</div>

# In[18]:


df['FireplaceQu'] = df['FireplaceQu'].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>LotFrontage</code> refers to the distance in feet between the street and the property. Let's impute the missing values with the median of the neighborhood.</div>

# In[19]:


df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda i: i.fillna(i.median()))


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial">All the features that start with <code>Garage</code> and contain NaN means that those houses don't have a garage.</div>

# In[20]:


# Let's take a look at the "Garage" features
garage_cols = [col for col in df if col.startswith('Garage')]
df[garage_cols]


# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">We can see that some features are categorical and others numerical. Let's replace the NaN with None in the categorical features and in the numerical features with 0.</div>

# In[21]:


# For the numerical features:
for i in df[garage_cols].select_dtypes(exclude='object').columns:
    df[i] = df[i].fillna(0)

# For the categorical features:
for i in df[garage_cols].select_dtypes(include='object').columns:
    df[i] = df[i].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial">All the features that start with <code>Bsmt</code> and contain NaN means that those houses don't have a basement.</div>

# In[22]:


bsmt_cols = [col for col in df if col.startswith('Bsmt')]

# For the numerical features:
for i in df[bsmt_cols].select_dtypes(exclude='object').columns:
    df[i] = df[i].fillna(0)

# For the categorical features:
for i in df[bsmt_cols].select_dtypes(include='object').columns:
    df[i] = df[i].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial">All the features that start with <code>Mas</code> and contain NaN means that those houses don't have a masonry veneer.</div>

# In[23]:


mas_cols = [col for col in df if col.startswith('Mas')]

# For the numerical features:
for i in df[mas_cols].select_dtypes(exclude='object').columns:
    df[i] = df[i].fillna(0)

# For the categorical features:
for i in df[mas_cols].select_dtypes(include='object').columns:
    df[i] = df[i].fillna('None')


# * <div style="color:black;
#            font-size:120%;
#            font-family:Arial"><code>MSZoning</code> refers to the general zoning classification of the sale. Let's impute the missing values with the most common category of the neighborhood.</div>

# In[24]:


df['MSZoning'] = df.groupby('Neighborhood')['MSZoning'].transform(lambda i: i.fillna(i.value_counts().index[0]))


# In[25]:


print('Missing Values left:')
df.isnull().sum().sort_values(ascending=False).head(10)


# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">The rest of the <b>missing values</b> are minimal. I'm going to transform the remaining NaN to the mode of each column.</div>

# In[26]:


# replace missing values for mode of each column
df = df.fillna(df.mode().iloc[0])


# ## <center style="font-family:Arial">Transforming some numerical categories into categorical</center>
# 
# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">Reading the data description shows very clearly that some numerical features represent a specific category.</div>

# In[27]:


df.describe().T


# In[28]:


df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)           # months is always categorical
df['YrSold'] = df['YrSold'].astype(str)           # year sold just have 5 years


# ## <center style="font-family:Arial">Adding relevant features</center>
# 
# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">Adding relevant features can increase the accuracy of the prediction.</div>

# In[29]:


df['Total_House_SF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Total_Home_Quality'] = (df['OverallQual'] + df['OverallCond'])/2
df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))


# ## <center style="font-family:Arial">Skewed features</center>
# 
# <div style="font-size:120%">Outliers are silent killers in prediction models. In this section, I'll imput the features that are not normally distributed.</div>
# 
# * <div style="font-size:120%">First, I'll select the features that have a skew higher than 0.5.</div>

# In[30]:


numeric_cols = df.select_dtypes(exclude='object').columns

skew_limit = 0.5
skew_vals = df[numeric_cols].skew()

skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {0}'.format(skew_limit)))

skew_cols


# In[31]:


# Font
mpl.rcParams['font.family'] = 'Source Sans Pro'
mpl.rcParams['font.size'] = 12

fig, (ax_positive, ax_negative) = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('black')
ax_positive.patch.set_facecolor('black')
ax_negative.patch.set_facecolor('black')

sns.histplot(df['BsmtUnfSF'],kde=True, stat='density', linewidth=0, color = '#236AB9', ax=ax_positive)
sns.histplot(df['YearBuilt'], kde=True, stat='density', linewidth=0,color='#B85B14', ax=ax_negative)

ax_positive.tick_params(axis='x', colors='white')
ax_positive.tick_params(axis='y', colors='white')
ax_negative.tick_params(axis='x', colors='white')
ax_negative.tick_params(axis='y', colors='white')

ax_positive.set(ylabel='Frequency', xlabel='Value');
ax_negative.set(ylabel='Frequency', xlabel='Value');

ax_positive.xaxis.label.set_color('white')
ax_positive.yaxis.label.set_color('white')
ax_negative.xaxis.label.set_color('white')
ax_negative.yaxis.label.set_color('white')

ax_positive.set_title('Positive Skew (BsmtUnfSF)', color='white', fontsize= 15)
ax_negative.set_title('Negative Skew (YearBuilt)', color='white', fontsize= 15)



# Remove axes splines
for i in ['top', 'bottom', 'left', 'right']:
    ax_positive.spines[i].set_visible(False)

for i in ['top', 'bottom', 'left', 'right']:
    ax_negative.spines[i].set_visible(False)


# <div style="color:black;
#            font-size:120%;
#            font-family:Arial">In my case, I'll use the Box-Cox transformation to transform all the skew features into a normal distribution.</div>

# In[32]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Normalize skewed features
for col in skew_cols.index:
    df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))


# ## <center style="font-family:Arial">Transforming target</center>

# In[33]:


import matplotlib.ticker as ticker

# Font
mpl.rcParams['font.family'] = 'Source Sans Pro'
mpl.rcParams['font.size'] = 10

# Visualization
fig, ax = plt.subplots(figsize =(9, 6))
fig.patch.set_facecolor('black')
ax.patch.set_facecolor('black')

sns.histplot(y['SalePrice'], stat='density', linewidth=0, color = '#ff7f50', kde=True, alpha=0.3);

# Remove ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Remove axes splines
for i in ['top', 'bottom', 'left', 'right']:
    ax.spines[i].set_visible(False)

# Remove grid
plt.grid(b=None)

# Setting thousands with k
ax.xaxis.set_major_formatter(ticker.EngFormatter())

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

# Font
mpl.rcParams['font.family'] = 'Source Sans Pro'

plt.xlabel('SalePrice', fontsize=11);

plt.text(230000, 0.0000088, "SalePrice", size=22, color="#ff7f50", fontweight="bold");
plt.text(380000, 0.0000088, "Distribution", size=22, color="grey", fontweight="bold");


# In[34]:


# log(1+x) transform
y["SalePrice"] = np.log1p(y["SalePrice"])


# In[35]:


import matplotlib.ticker as ticker

# Font
mpl.rcParams['font.family'] = 'Source Sans Pro'
mpl.rcParams['font.size'] = 10

# Visualization
fig, ax = plt.subplots(figsize =(9, 6))
fig.patch.set_facecolor('black')
ax.patch.set_facecolor('black')

sns.histplot(y['SalePrice'], stat='density', linewidth=0, color = '#ff7f50', kde=True, alpha=0.3);

# Remove ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Remove axes splines
for i in ['top', 'bottom', 'left', 'right']:
    ax.spines[i].set_visible(False)

# Remove grid
plt.grid(b=None)

# Setting thousands with k
ax.xaxis.set_major_formatter(ticker.EngFormatter())

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

# Font
mpl.rcParams['font.family'] = 'Source Sans Pro'

plt.xlabel('SalePrice', fontsize=11);

plt.text(11.27, 1.25, "SalePrice", size=22, color="#ff7f50", fontweight="bold");
plt.text(11.92, 1.25, "Distribution", size=22, color="grey", fontweight="bold");


# ## <center style="font-family:Arial">Encoding categorical features</center>

# In[36]:


categ_cols = df.dtypes[df.dtypes == np.object]        # filtering by categorical variables
categ_cols = categ_cols.index.tolist()                # list of categorical fields

df_enc = pd.get_dummies(df, columns=categ_cols, drop_first=True)   # One hot encoding


# In[37]:


X = df_enc[df_enc['train']==1]
test = df_enc[df_enc['train']==0]
X.drop(['train'], axis=1, inplace=True)
test.drop(['train'], axis=1, inplace=True)


# # <center style="font-family:Arial">5. Modelling </center>

# In[38]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)


# In[39]:


def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))


# ## <center style="font-family:Arial">Lasso Regression + Cross-Validation</center>
#     
# * <div style="font-size:120%">Lasso Regression is a linear model that minimizes its cost function.</div>
# 
# * <div style="font-size:120%">The cost funtion has a regularization parameter -<b>L1 penalty</b>- with an alpha that tunes the intensity of this penalty term. </div>
# 
# * <div style="font-size:120%">This penalty reduces some features to zero, which makes it easier to understand and interpret the prediction.</div>
# 
# * <div style="font-size:120%">The larger the value of alpha, the more coefficients are forced to be zero.</div>
# 
# * <div style="font-size:120%">The Lasso regression helps reduce over-fitting and feature selection.</div>
# 
# 

# In[40]:


lasso = Lasso(max_iter = 100000, normalize = True)

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)

print('The Lasso I:')
print("Alpha =", lassocv.alpha_)
print("RMSE =", rmse(y_test, lasso.predict(X_test)))


# In[41]:


# Let's try the same. This time setting up alpha...
alpha = np.geomspace(1e-5, 1e0, num=6)
lasso_cv_model = LassoCV(alphas = alpha, cv = 10, max_iter = 100000, normalize = True).fit(X_train,y_train)
lasso_tuned = Lasso(max_iter = 100000, normalize = True).set_params(alpha = lasso_cv_model.alpha_).fit(X_train,y_train)
print('The Lasso II:')
print("Alpha =", lasso_cv_model.alpha_)
print("RMSE =", rmse(y_test, lasso_tuned.predict(X_test)))


# ## <center style="font-family:Arial">Ridge Regression + Cross-Validation</center>
# 
# * <div style="font-size:120%">The Ridge Regression is similar to the Lasso Regression: it's also a linear model that minimizes its cost function and has a regularization parameter -<b>L2 penalty</b>-.</div>
# 
# * <div style="font-size:120%">The lower the value of the alpha, the more linear the model will be.</div>
# 
# * <div style="font-size:120%">This model doesn't force some features to zero. </div>
# 
# * <div style="font-size:120%">The Ridge Regression shrinks the coefficients, and it helps to reduce the model complexity and multi-collinearity.</div>

# In[42]:


alphas = np.geomspace(1e-9, 5, num=100)

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)

ridge = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge.fit(X_train, y_train)

print('Ridge Regression:')
print("Alpha =", ridgecv.alpha_)
print("RMSE =", rmse(y_test, ridge.predict(X_test)))


# ## <center style="font-family:Arial">Support Vector Regression (SVR) + Cross-Validation</center>

# In[43]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVR


kf = KFold(shuffle=True, random_state=1234, n_splits=10)

X_train_scale = RobustScaler().fit_transform(X_train)
X_test_scale = RobustScaler().fit_transform(X_test)

parameters = {'C':[20, 30, 40], 'gamma': [1e-4, 3e-4, 5e-4],'epsilon':[0.1, 0.01, 0.05]}
svr = SVR(kernel='rbf')
clf = GridSearchCV(svr, parameters, cv=kf)
clf.fit(X_train_scale,y_train)
clf.best_params_


# In[44]:


svr = SVR(kernel ='rbf', C= 20, epsilon= 0.01, gamma=0.0003)
svr.fit(X_train_scale,y_train)

print('SVR Regression:')
print("RMSE =", rmse(y_test, svr.predict(X_test_scale)))


# # <center style="font-family:Arial">6. Interpretation </center>
# 
# <div style="font-size:120%">As mentioned previously, the Lasso regression is a very easy model to interpret, and that's why for this notebook I'll base the features importance on the <b>Lasso model</b>.</div>

# In[45]:


print('Out of {} coefficients, {} are non-zero with Lasso.'
     .format(len(lasso_tuned.coef_), len(lasso_tuned.coef_.nonzero()[0])))


# In[46]:


# Selecting features importance
coefs = pd.Series(lasso_tuned.coef_, index = test.columns)

lasso_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])

lasso_coefs = pd.DataFrame(lasso_coefs, columns=['importance'])

# Visualization
fig, ax = plt.subplots(figsize =(11, 9))

ax.barh(lasso_coefs.index, lasso_coefs.importance, align='center', 
        color = np.where(lasso_coefs['importance'] < 0, 'crimson', '#89CFF0'))

plots_design()

plt.text(-0.22, 20.5, "Feature Importance", size=24, color="grey", fontweight="bold");
plt.text(-0.063, 20.5, "using", size=24, color="grey");
plt.text(-0.0182, 20.5, "Lasso Model", size=24, color="#89CFF0", fontweight="bold");

# Author
plt.text(0.2, -3.3, "@miguelfzzz", fontsize=12, ha="right", color='grey');


# <div style="font-size:120%">All the features in blue positively affect the sale price of the house, which means this characteristic increases the price of the house. Vice versa, all the features in red negatively affect the sale price of the house.</div>

# <center style="font-size: 24px">If you liked this notebook, please don't forget to comment and upvote. Thank you!</center>
