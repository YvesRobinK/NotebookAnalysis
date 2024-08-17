#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>1.1 Background</b></p>
# </div>
# 
# This months TPS competition is on **forecasting** the number of **book sales** during 2021. In particular, we need to predict the number of sales of **4 books** that **2 companies** sold accross **6 countries**. It is reminiscent of the [TPS January 22](https://www.kaggle.com/competitions/tabular-playground-series-jan-2022) competition that happened earlier this year. 
# 
# <center>
# <img src='https://media.istockphoto.com/photos/open-book-close-up-at-the-library-picture-id1302676874?b=1&k=20&m=1302676874&s=170667a&w=0&h=MCUK_eGWI-FkGrUgv1sPW_5D3gZa0sMTHhRCd_wqUxQ=' width=600>
# </center>
# <br>
#   
# This competition is particularly interesting because we are not only dealing with the usual **trends** and **seasonality** patterns, but now are forecasting during 2021 - a **volatile** year influenced by many factors, e.g. **COVID-19**. 
# 
# If you are **new** to time series analysis, I highly **recommend** you check out [kaggles timeseries course](https://www.kaggle.com/learn/time-series) as it is very short but **covers all the essentials**.
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>1.2 Libraries</b></p>
# </div>

# In[1]:


# Core
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
from itertools import product as cartessian_product
import math
import statistics
import scipy.stats
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import chi2
from scipy.stats import poisson
import time
from datetime import datetime
import matplotlib.dates as mdates
import dateutil.easter as easter
import plotly.express as px
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

# Models
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# # 2. Data
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.1 Load data</b></p>
# </div>
# 
# We are working with **timeseries** data.

# In[2]:


# Save to df
train = pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv', index_col='row_id')
test = pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv', index_col='row_id')

# Convert date to datetime format
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])

# Replace problem characters
train['product'] = train['product'].str.replace(':','-')
test['product'] = test['product'].str.replace(':','-')

# Shape and preview
print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
train.head(3)


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.2 Missing values</b></p>
# </div>
# 
# There are **no missing values**.

# In[3]:


print('Train set missing values:', train.isna().sum().sum())
print('')
print('Test set missing values:', test.isna().sum().sum())


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.3 Duplicates</b></p>
# </div>
# 
# There are **no duplicated values**.

# In[4]:


print(f'Duplicates in train set: {train.duplicated().sum()}, ({np.round(100*train.duplicated().sum()/len(train),1)}%)')
print('')
print(f'Duplicates in test set: {test.duplicated().sum()}, ({np.round(100*test.duplicated().sum()/len(test),1)}%)')


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.4 Data types</b></p>
# </div>
# 
# There are **3 categorical** and one **date-time** feature. The target is **discrete**.

# In[5]:


train.dtypes


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.5 Data cardinality</b></p>
# </div>
# 
# The **cardinality** of a categorical feature is defined to be the **number of unique categories**. 

# In[6]:


train.nunique()


# In[7]:


print('Countries:', list(train['country'].unique()),'\n')
print('Stores:', list(train['store'].unique()),'\n')
print('Products:', list(train['product'].unique()))


# # 3. EDA
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.1 Time range</b></p>
# </div>
# 
# * The train set covers **2017-2020** (4 years).
# * The test set covers **2021** only.

# In[8]:


print('TRAIN:')
print('Min date', train['date'].min())
print('Max date', train['date'].max())
print('')
print('TEST:')
print('Min date', test['date'].min())
print('Max date', test['date'].max())


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.2 Product distribution</b></p>
# </div>
# 
# There are **4 book** titles being sold.

# In[9]:


plt.figure(figsize=(16,5))
ax = sns.barplot(data=train, x='product', y='num_sold', hue='country')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Product distribution grouped by country')
plt.show()


# *Observations:*
# 
# * The **product** and **country** features appear to be **independent**.

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.3 Store</b></p>
# </div>
# 

# In[10]:


plt.figure(figsize=(12,5))
sns.lineplot(data=train.groupby(['date','store']).sum(), x='date', y='num_sold', hue='store')
plt.title('Sales by store')
plt.show()


# *Observations:*
# 
# * The two stores are **highly correlated** (r2 = 0.9812).
# * It appears one is simply a **multiplicative factor** of the other (2.8837). (This was the case in January's competition)
# * To **avoid overfitting**, we could replace one of the stores by this multiplicative factor.

# In[11]:


store_corr = pearsonr(train.loc[train['store']=='KaggleMart','num_sold'], train.loc[train['store']=='KaggleRama','num_sold'])[0]
print(f'Store correlation: {store_corr:.4f}')

mult_factor = train.loc[train['store']=='KaggleMart','num_sold'].sum()/train.loc[train['store']=='KaggleRama','num_sold'].sum()
print(f'Multiplicative factor: {mult_factor:.4f}')


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.4 Country</b></p>
# </div>

# In[12]:


fig, ax = plt.subplots(4, 1, figsize=(18, 20))
ax = ax.flatten()

for i, product in enumerate(train['product'].unique()):
    subset = train[train['product'] == product]
    sns.lineplot(ax=ax[i], data=subset.groupby(['date','country']).sum(), x='date', y='num_sold', hue='country')
    plt.ylim([0,1400])
    
    ax[i].set_title(product)
    
    ax[i].legend(loc='upper right')
    if i!=2:
        ax[i].legend().remove()

fig.tight_layout()


# *Observations:*
# 
# * We can see strong **seasonality** affects (year, month and week long). 
# * There appears to be a weak **long-term trend**. (In January's competition this was correlated to GDP)
# * At the beginning of **2020** all the country sales **coallesce together** and seasonality **trends change**.
# * We might consider **dropping** the **country** feature altogether.

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.5 Product</b></p>
# </div>

# In[13]:


fig, ax = plt.subplots(2, 1, figsize=(18, 10))
ax = ax.flatten()

for i, store in enumerate(train['store'].unique()):
    subset = train[train['store'] == store]
    sns.lineplot(ax=ax[i], data=subset.groupby(['date','product']).sum(), x='date', y='num_sold', hue='product')
    plt.ylim([0,4500])
    
    ax[i].set_title(store)
    
    ax[i].legend(loc='upper right')
    if i!=1:
        ax[i].legend().remove()

fig.tight_layout()


# *Observations:*
# 
# * Again we see the **high correlation** between the two stores. 
# * The products appear to have been **affected by COVID-19**. (Look close at March 2020 - July 2020)
# * We should search for **external data** (like covid-19 case numbers) that might be **correlated** to this data.

# # 4. Seasonality
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.1 Product ratio</b></p>
# </div>
# 
# This pattern was found by [ehekatlact](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/discussion/349737). If we plot the **ratio** of sales by **product**, we find the **underlying seasonality pattern**.

# In[14]:


plt.figure(figsize=(12,5))
ax = sns.lineplot(data=train.groupby(['product','date']).sum()/train.groupby(['date']).sum(), x='date', y='num_sold', hue='product')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Ratio of sales by product')
plt.ylabel('Ratio')
plt.show()


# *Observations:*
# 
# * The proportion of books sold by product has a **strong seasonal pattern** with a **time period of 2 years** (look at red curve).
# * More importantly, this pattern **doesn't change** during the year **2020**.
# * We can **extrapolate** this trend to the year **2021** using Fourier features.

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.2 Modelling seasonality</b></p>
# </div>
# 
# As discussed in this [post](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/discussion/349737) by broccoli beef, this suggests our data follows a **multiplicative model**
# 
# $$
# y_t^{(c, s, p)} = S_t^{(c,s)} \cdot T_t^{(p)}
# $$
# 
# where $y_t^{(c, s, p)}$ are the **sales** on day $t$ corresponding to country $c$, store $s$ and product $p$. The **seasonal trend**, which we model as being only dependent on $p$ is denoted by $T_t^{(p)}$ and satisfies $0 \leq T_t^{(p)} \leq 1$ and $\sum_{p} T_t^{(p)} = 1$. The **product-independent** time series is denoted by $S_t^{(c,s)}$. 
# 
# To explore this idea further, we will plot $S_t^{(c,s)} = y_t^{(c, s, p)}/T_t^{(p)}$ to see if there is any seasonality left in the detrended data.

# In[15]:


# Calculate S and T from model
ratios = (train.groupby(['product','date']).sum()/train.groupby(['date']).sum())['num_sold'].to_dict()
train['T'] = train.set_index(['product', 'date']).index.map(ratios.get)
train['S'] = train['num_sold']/train['T']

# Plot S from model
fig, ax = plt.subplots(2, 1, figsize=(18, 10))
ax = ax.flatten()

for i, store in enumerate(train['store'].unique()):
    subset = train[train['store'] == store]
    sns.lineplot(ax=ax[i], data=subset.groupby(['date','country']).sum(), x='date', y='S', hue='country')
    
    ax[i].set_title(store)
    
    ax[i].legend(loc='upper right')
    if i!=1:
        ax[i].legend().remove()

fig.tight_layout()


# *Observations:*
# 
# * We have completely **removed** the **yearly-seasonal trend**. 
# * There is still a **weekly-seasonal trend** present. 

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.3 Country ratio</b></p>
# </div>
# 
# Now let's compute the **ratio** of sales by **country** on the **detrended** time series. 

# In[16]:


plt.figure(figsize=(12,5))
ax = sns.lineplot(data=pd.DataFrame(train.groupby(['country','date']).sum()['S']/train.groupby(['date']).sum()['S']), x='date', y='S', hue='country')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Ratio of sales by country on detrended series')
plt.ylabel('Ratio')
plt.show()


# In[17]:


# Get ratio of sales on a country-by-country basis
def get_ratios(df):
    df_ratio = pd.DataFrame(df.groupby(['country','date']).sum()['S']/df.groupby(['date']).sum()['S'])
    df_ratio = df_ratio.reset_index()
    df_2017 = df_ratio.loc[df_ratio['date'].dt.year==2017]
    df_2018 = df_ratio.loc[df_ratio['date'].dt.year==2018]
    df_2019 = df_ratio.loc[df_ratio['date'].dt.year==2019]
    df_2020 = df_ratio.loc[df_ratio['date'].dt.year==2020]
    df_out = pd.concat([df_2017.groupby('country').mean(), df_2018.groupby('country').mean(), df_2019.groupby('country').mean(), df_2020.groupby('country').mean()],axis=1)
    df_out.columns=['Ratio of sales (2017)','Ratio of sales (2018)','Ratio of sales (2019)','Ratio of sales (2020)']

    return df_out

ratios_df = get_ratios(train)
ratios_df


# *Observations:*
# 
# * The **ratios** are **piecewise constant**.
# * Beginning in 2020, the ratios **coalesce** to become **exactly 1/6** = 0.1666... 
# * We can **scale** the sales **pre-2020** (on a yearly basis) so that all countries have the **same ratio** of sales. (This assumes the ratio of 1/6 will continue into the test set.)

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.4 Balancing</b></p>
# </div>
# 
# We first scale each countries sales so they have the **same ratio** of sales as in 2020. We then balance the total sales on a yearly basis so they are the same as in 2020. This is experimental.

# **Balance (detrended) sales**

# In[18]:


# Calculate scale factors to balance sale ratios
ratios_df['Balancing scale factor (2017)'] = ratios_df['Ratio of sales (2020)'] / ratios_df['Ratio of sales (2017)']
ratios_df['Balancing scale factor (2018)'] = ratios_df['Ratio of sales (2020)'] / ratios_df['Ratio of sales (2018)']
ratios_df['Balancing scale factor (2019)'] = ratios_df['Ratio of sales (2020)'] / ratios_df['Ratio of sales (2019)']
ratios_dict_2017 = ratios_df['Balancing scale factor (2017)'].to_dict()
ratios_dict_2018 = ratios_df['Balancing scale factor (2018)'].to_dict()
ratios_dict_2019 = ratios_df['Balancing scale factor (2019)'].to_dict()

# Map scale factors to countries
train['balance_factor_2017'] = train.set_index(['country']).index.map(ratios_dict_2017.get)
train['balance_factor_2018'] = train.set_index(['country']).index.map(ratios_dict_2018.get)
train['balance_factor_2019'] = train.set_index(['country']).index.map(ratios_dict_2019.get)

# Only scale corresponding years
train.loc[train['date']>pd.Timestamp('2018-01-01 00:00:00'),'balance_factor_2017']=1
train.loc[(train['date']>pd.Timestamp('2019-01-01 00:00:00'))|(train['date']<=pd.Timestamp('2018-01-01 00:00:00')),'balance_factor_2018']=1
train.loc[(train['date']>pd.Timestamp('2020-01-01 00:00:00'))|(train['date']<=pd.Timestamp('2019-01-01 00:00:00')),'balance_factor_2019']=1

# Balance detrended sales
train['S_balanced'] = train['S'] * train['balance_factor_2017']
train['S_balanced'] = train['S_balanced'] * train['balance_factor_2018']
train['S_balanced'] = train['S_balanced'] * train['balance_factor_2019']

# Drop features
train.drop(['balance_factor_2017','balance_factor_2018','balance_factor_2019'], axis=1, inplace=True)


# In[19]:


# Plot new ratios
plt.figure(figsize=(12,5))
ax = sns.lineplot(data=pd.DataFrame(train.groupby(['country','date']).sum()['S_balanced']/train.groupby(['date']).sum()['S_balanced']), x='date', y='S_balanced', hue='country')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Balanced ratio of sales')
plt.ylabel('Ratio')
plt.ylim([0.05,0.26])
plt.show()


# *Observations:*
# 
# * This **balanced** version of the sales should help us **generalise better** on the test set.
# * By not aggregating the countries together, we can still learn **holiday patterns** (which seem to have a time period of **2 years** weirdly).
# * **Poland** is still showing some odd behaviour (we will need to investigate further).

# **Scale total sales**
# 
# This part is experimental. Since there are a lot more sales in 2020 than in 2017-2019, my models tend to underpredict the number of sales. I'm going to scale it so the total number of sales in each year is the same as in 2020.

# In[20]:


# Calculate scale factors to balance total sales
train['year'] = train['date'].dt.year
total_sales_df = train.groupby('year').sum()
total_sales_df['total_sales_factor'] = train.groupby('year').sum().loc[2020,'S_balanced']/total_sales_df['S_balanced']
total_sales_dict = total_sales_df['total_sales_factor'].to_dict()

# Map scale factors to years
train['total_sales_factor'] = train.set_index(['year']).index.map(total_sales_dict.get)
train.drop('year', axis=1, inplace=True)

# Balance total sales
train['S_balanced'] = train['S_balanced'] * train['total_sales_factor']

# Drop feature
train.drop('total_sales_factor', axis=1, inplace=True)


# In[21]:


fig, ax = plt.subplots(2, 1, figsize=(18, 10))
ax = ax.flatten()

for i, store in enumerate(train['store'].unique()):
    subset = train[train['store'] == store]
    #subset = subset[(subset['date']>=pd.Timestamp('2020-06-01 00:00:00')) | (subset['date']<pd.Timestamp('2020-03-01 00:00:00'))]
    sns.lineplot(ax=ax[i], data=subset.groupby(['date','country']).sum(), x='date', y='S_balanced', hue='country')
    
    ax[i].set_title(store)
    
    ax[i].legend(loc='upper right')
    if i!=1:
        ax[i].legend().remove()

fig.tight_layout()


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.5 Store ratio</b></p>
# </div>
# 
# Let's continue with the **ratio theme** and calculate the ratio of sales between the two **stores** on each day.

# In[22]:


plt.figure(figsize=(12,5))
sns.lineplot(data=pd.DataFrame(train[train['store']=='KaggleMart'].groupby(['date']).mean()['S_balanced']/train[train['store']=='KaggleRama'].groupby(['date']).mean()['S_balanced']), x='date', y='S_balanced')
plt.title('Ratio of store sales on balanced detrended series')
plt.ylabel('Ratio')
plt.ylim([0,4])
plt.show()


# *Observations:*
# 
# * Apart from the noise, the ratio is **constant**.
# * We could predict the **total sales** of both stores combined and later split the sales according to the **ratio** of sales each store makes. This would also have a nice **regularising effect**.

# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.6 Weekend effect</b></p>
# </div>
# 
# Let's look at the **weekly seasonal pattern** left in our detrended time series.

# In[23]:


# Based on https://www.kaggle.com/code/jcaliz/tps-sep22-eda-baseline-you-were-looking-for
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
ax = ax.flatten()

for i, store in enumerate(train['store'].unique()):
    df_to_plot = train[train['store'] == store]
    df_to_plot['day_of_week'] = df_to_plot['date'].dt.dayofweek
    
    sns.lineplot(data=pd.melt(df_to_plot, id_vars=['store', 'day_of_week'],value_vars=['S'],value_name='S'), x='day_of_week', y='S', hue='store', ax=ax[i])
    ax[i].set_title(f'{store}')
    
    if i!=1:
        ax[i].legend().remove()

plt.suptitle(f'Seasonality by week', fontsize=16)
plt.tight_layout()


# *Observations:*
# 
# * The pattern is very **consistent**. Sales are low during the week (Monday to Thursday), they increase a bit on Friday and are highest at the weekend (Saturday and Sunday).

# # 5. Feature engineering
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.1 Date features</b></p>
# </div>
# 
# Feature engineering for time series data is **essential**.
# 

# In[24]:


def get_date_features(df):
    # Extract year, month, day, etc
    #df['year'] = df['date'].dt.year                   # 2017 to 2021
    df['day_of_week'] = df['date'].dt.dayofweek       # 0 to 6
    df['day_of_month'] = df['date'].dt.day            # 1 to 31
    df['day_of_year'] = df['date'].dt.dayofyear         # 1 to 366
    #df.loc[(df['date'].dt.year==2020) & (df['day_of_year']>60), 'day_of_year'] -= 1   # 1 to 365
    df['week']=df['date'].dt.isocalendar().week       # 1 to 53
    df['week']=df['week'].astype('int')               # int64
    df['month']=df['date'].dt.month                   # 1 to 12
    return df


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.2 Fourier features</b></p>
# </div>
# 
# Fourier features allow use **sine** and **cosine** waves to model **seasonality** patterns in the data. We can change the **time period** to model yearly, monthly and weekly seasonal patterns. The **motivation** for this is that people spend money differently depending on whether it is the summer vs winter, or weekday vs weekend.
# 
# <center>
# <img src="https://undergroundmathematics.org/trigonometry-triangles-to-functions/from-stars-to-waves/images/sin-cos-wave.png" width=500>
# <\center>
#     
# Check out my other [notebook](https://www.kaggle.com/code/samuelcortinhas/tps-sept-22-fourier-analysis) where I identified the frequencies in the time series using **Fourier Analysis**. 

# In[25]:


def get_fourier_features(df):
    # Time period = 2 years
    dayofbiyear = df['date'].dt.dayofyear + 365*(1-(df['date'].dt.year%2))  # 1 to 730
    
    # k=1 -> 2 years, k=2 -> 1 year, k=4 -> 6 months
    for k in [1, 2, 4]:
        df[f'sin{k}'] = np.sin(2 * np.pi * k * dayofbiyear / (2* 365))
        df[f'cos{k}'] = np.cos(2 * np.pi * k * dayofbiyear / (2* 365))
        
        # Different products have different seasonality patterns
        for product in df['product'].unique():
            df[f'sin_{k}_{product}'] = df[f'sin{k}'] * (df['product'] == product)
            df[f'cos_{k}_{product}'] = df[f'cos{k}'] * (df['product'] == product)
        
        df = df.drop([f'sin{k}', f'cos{k}'], axis=1)
    
    return df


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.3 Holidays</b></p>
# </div>
# 
# Holidays can be **indicators of higher sales**. The idea is that if people don't have to go to work they could be using that time to buy things. Or, for holidays like Christmas, they could be buying presents for their family and friends.

# In[26]:


# https://www.kaggle.com/code/ducanger/4-47-smape-lasso-oof-predict
def get_holidays(df):
    important_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 124, 125, 126, 127, 140, 141,142, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 203, 230, 231, 232, 233, 234, 282, 289, 290, 307, 308, 309, 310, 311, 312, 313, 317, 318, 319, 320, 360, 361, 362, 363, 364, 365]
    df["important_dates"] = df["day_of_year"].apply(lambda x: x if x in important_dates else 0)
    return df


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.4 GDP</b></p>
# </div>
# 
# The idea is that **Gross Domestic Product** (GDP) measures a countries **economic output** and so this might be correlated to the **sales** in each country. 
# 
# It turns out the GDP (per capita) is **highly correlated** to the target between the years **2017-2019** only. And then this correlation disappears completely in 2020. ([extra details](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/discussion/350137))

# In[27]:


# Calculate correlations on detrended data
def get_GDP_corr(df):
    # Initialise output
    feat_corr=[]
    
    # Load data
    df['year']=df['date'].dt.year
    GDP = pd.read_csv("../input/gdp-of-european-countries/GDP_table.csv",index_col="year")
    GDP_PC = pd.read_csv("../input/gdp-of-european-countries/GDP_per_capita_table.csv",index_col="year")
    GDP_dict = GDP.unstack().to_dict()
    GDP_PC_dict = GDP_PC.unstack().to_dict()
    df['GDP'] = df.set_index(['country', 'year']).index.map(GDP_dict.get)
    df['GDP_PC'] = df.set_index(['country', 'year']).index.map(GDP_PC_dict.get)
    
    # Compute pairwise correlations
    for country in df['country'].unique():
        subset=df[(df['country']==country)&(df['year']<=2019)].groupby(['year']).agg(S=('S','sum'), GDP=('GDP','mean'), GDP_PC=('GDP_PC','mean'))

        r1 = pearsonr(subset['S'],subset['GDP'])[0]
        r2 = pearsonr(subset['S'],subset['GDP_PC'])[0]

        feat_corr.append([f'{country}', r1, r2])

    return pd.DataFrame(feat_corr, columns=['Country', 'GDP_corr', 'GDP_PC_corr'])

corr_df = get_GDP_corr(train)
corr_df


# In[28]:


# A more detailed breakdown of correlations
def get_GDP_corr2(df):
    # Initialise output
    feat_corr=[]

    # Compute pairwise correlations
    for country in df['country'].unique():
        for product in df['product'].unique():
            subset=df[(df['country']==country)&(df['product']==product)&(df['year']<=2019)].groupby(['year']).agg(S=('S','sum'), GDP=('GDP','mean'), GDP_PC=('GDP_PC','mean'))

            r1 = pearsonr(subset['S'],subset['GDP'])[0]
            r2 = pearsonr(subset['S'],subset['GDP_PC'])[0]

            feat_corr.append([f'{country}, {product}', r1, r2])

    df.drop(['GDP','GDP_PC','year'], axis=1, inplace=True)
            
    return pd.DataFrame(feat_corr, columns=['Features', 'GDP_corr', 'GDP_PC_corr'])

corr_df2 = get_GDP_corr2(train)
corr_df2.head()


# **GDP per capita** seems to be more correlated to sales than GDP (except for Italy). We can use this feature to **further detrend** the years 2017-2019. (to do)

# In[29]:


def get_GDP_data(df): # (work in progress)
    # Load data
    #GDP = pd.read_csv("../input/gdp-of-european-countries/GDP_table.csv",index_col="year")
    GDP_PC = pd.read_csv("../input/gdp-of-european-countries/GDP_per_capita_table.csv",index_col="year")
    
    # Create a dictionary
    #GDP_dict = GDP.unstack().to_dict()
    GDP_PC_dict = GDP_PC.unstack().to_dict()

    # Create new columns
    df['GDP_PC'] = df.set_index(['country', 'year']).index.map(GDP_PC_dict.get)
    #df['GDP'] = df.set_index(['country', 'year']).index.map(GDP_dict.get)
    
    # GDP_PC except for Italy
    #df['GDP_feature'] = df['GDP_PC']
    #df.loc[df['country']=='Italy','GDP_feature'] = df.loc[df['country']=='Italy','GDP']
    
    # Fill 2020 with average sales (since GDP uncorrelated)
    #df.loc[(df['year']==2020),'GDP_feature'] = df[(df['year']==2020)].agg(S=('S','sum'))
    
    # Drop intermediate columns
    #df.drop(['GDP','GDP_PC'], axis=1, inplace=True)
    
    return df


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.5 Put the pieces together</b></p>
# </div>
# 
# We are going to build a **hybrid model** (more details below). For this we need **two versions** of the train/test sets.

# In[30]:


# Labels
y_1 = train[['date','store','product','T']].drop_duplicates()['T']
y_2 = train['S_balanced']

# Features
X = train.drop(['num_sold','T','S','S_balanced'], axis=1)
X_test = test.copy()


# In[31]:


# Feature set for trend model
def FeatEng_X1(df):
    df = df[['date','store','product']].drop_duplicates()
    df = get_fourier_features(df)
    df = pd.get_dummies(df, columns=['store','product'], drop_first=True)
    df = df.drop(['date'], axis=1)
    return df

# Feature set for interactions model
def FeatEng_X2(df):
    df = get_date_features(df)
    df = get_holidays(df)
    df = df.drop(['date','product'], axis=1)
    df = pd.get_dummies(df, columns=['country','store'], drop_first=True)
    return df

# Apply feature engineering
X_train_1 = FeatEng_X1(X)
X_train_2 = FeatEng_X2(X)
X_test_1 = FeatEng_X1(X_test)
X_test_2 = FeatEng_X2(X_test)


# # 6. Modelling
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>6.1 Linear model</b></p>
# </div>
# 
# * **Part 1:** Train a **linear model** to predict seasonal trend.
# * **Part 2:** Train **tree-based model** to learn interactions on **detrended** series.
# 
# **Linear regression** is good at **extrapolating trends** but poor at learning interactions. Conversely, **tree-based algorithms** like XGBoost are very good at learning interactions but can't extrapolate trends. A **hybrid model** tries to take the **best of both worlds** by first learning the trend with linear interpolation and then learning the interactions on the detrended time series.

# In[32]:


# Create df to store predictions
data = pd.concat([train,test],axis=0)

# Model for trend
model1 = LinearRegression()
model1.fit(X_train_1, y_1)
preds1 = model1.predict(X_test_1)


# In[33]:


# Save trend predictions (in Belgium)
data.loc[X_test_1.index,'T'] = preds1

# Extend predictions to all countries
for country in train['country'].unique():
    idx = data.loc[(data['country']==country)&(data['num_sold'].isna())].index
    data.loc[idx,'T'] = data.loc[(data['country']=='Belgium')&(data['num_sold'].isna()),'T'].values


# **Visualise trend predictions**

# In[34]:


plt.figure(figsize=(12,5))
for product in train['product'].unique():
    ax = sns.lineplot(data=data.loc[(data['country']=='Belgium')&(data['product']==product)].groupby(['date']).sum(), x='date', y='T', label=product)
plt.title('Ratio of sales by store (including predictions)')
plt.ylabel('Ratio')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>6.2 Tree-based model</b></p>
# </div>
# 
# **CatBoostRegressor** has been found to be working well on this dataset. We'll use grid search to find the best parameters for our model.

# In[35]:


get_ipython().run_cell_magic('time', '', "\n# Cross validation\ntscv = TimeSeriesSplit(n_splits=3)\n\n# Grid for grid search\ngrid = {'n_estimators': [75, 100, 125],\n        'max_depth': [4, 5],\n        'learning_rate': [0.075, 0.1, 0.125]}\n\n# Model for interactions\nmodel2 = CatBoostRegressor(random_state=0, verbose=False)\ngrid_model = GridSearchCV(estimator=model2, cv=tscv, param_grid=grid)\n\n# Train model using GridSearch\ngrid_model.fit(X_train_2, y_2)\n\n# Make predictions\npreds2 = grid_model.predict(X_test_2)\n\n# Print best parameters\nprint('Best parameters:', grid_model.best_params_)\n")


# In[36]:


# Save predictions
data.loc[data['num_sold'].isna(),'S_balanced'] = preds2

# Put trend back in
data['TS_balanced'] = data['T'] * data['S_balanced']


# **Feature importances**

# In[37]:


# Feature importances
pd.DataFrame({'Feature': X_train_2.columns,'Importance': grid_model.best_estimator_.get_feature_importance()}).sort_values(by=['Importance'], ascending=False).reset_index(drop=True)


# # 7. Submission
# 
# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>7.1 Post-process</b></p>
# </div>

# In[38]:


# From https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison
def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where(result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)), np.ceil(arr), result_array)
    return result_array

# Round predictions to nearest integer
data.loc[data['num_sold'].isna(),'TS_balanced'] = data.loc[data['num_sold'].isna(),'TS_balanced'].apply(lambda x:geometric_round(x))


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>7.2 Save to csv</b></p>
# </div>

# In[39]:


# Save predictions to csv
submission = pd.DataFrame({'row_id': test.index, 'num_sold': data.loc[data['num_sold'].isna(),'TS_balanced'].values})
submission.to_csv('submission.csv', index=False)

# Check format
submission.head(3)


# <div style="color:white;display:fill;
#             background-color:#00bbe0;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>7.3 Visualise predictions</b></p>
# </div>
# 
# This is to get insight into what our predictions look like. (And make sure nothing is going too wrong.)

# In[40]:


def plot_predictions(store, country, product, series_train=[], series_test=[]):
    
    # Training set subset
    train_subset=train[(train['store']==store)&(train['country']==country)&(train['product']==product)]
    
    # Predictions
    plot_index_train=train[(train['store']==store)&(train['country']==country)&(train['product']==product)].index
    plot_index_test=test[(test['store']==store)&(test['country']==country)&(test['product']==product)].index
    
    if len(series_train)>0:
        pred_subset_train=series_train[series_train['row_id'].isin(plot_index_train)].reset_index(drop=True)
    
    if len(series_test)>0:
        pred_subset_test=series_test[series_test['row_id'].isin(plot_index_test)].reset_index(drop=True)

    # Plot
    plt.figure(figsize=(12,5))
    n1=len(train_subset['num_sold'])
    plt.plot(np.arange(n1),train_subset['num_sold'], label='Training', c='C0')
    if len(series_train)>0:
        plt.plot(np.arange(n1),pred_subset_train['num_sold'], c='C1')
    if len(series_test)>0:
        n2=len(pred_subset_test['num_sold'])
        plt.plot(np.arange(n1,n1+n2),pred_subset_test['num_sold'], label='Prediction', c='C1')
    
    plt.title('\n'+f'Store:{store}, Country:{country}, Product:{product}')
    plt.legend()
    plt.xlabel('Days since 2017-01-01')
    plt.ylabel('num_sold')


# In[41]:


for store in train['store'].unique():
    for country in train['country'].unique():
        for product in train['product'].unique():
            plot_predictions(store, country, product, series_test=submission)
        # Remove break to see all plots
        break


# # 8. References
# 
# * [TPS Jan 22 - quick EDA + Hybrid model](https://www.kaggle.com/code/samuelcortinhas/tps-jan-22-quick-eda-hybrid-model) by Samuel Cortinhas.
# * [TPS Sep22: EDA & Baseline You Were Looking For](https://www.kaggle.com/code/jcaliz/tps-sep22-eda-baseline-you-were-looking-for) by JC.
# * [TPS2209_Ridge_LGBM_EDA_TopDownApproach](https://www.kaggle.com/code/ehekatlact/tps2209-ridge-lgbm-eda-topdownapproach) by ehekatlact.
# * [[TPS-SEP-22] EDA and Linear Regression Baseline](https://www.kaggle.com/code/cabaxiom/tps-sep-22-eda-and-linear-regression-baseline) by Cabaxiom.
