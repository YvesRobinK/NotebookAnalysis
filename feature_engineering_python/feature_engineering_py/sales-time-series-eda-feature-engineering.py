#!/usr/bin/env python
# coding: utf-8

# # **Store Sales - Time Series Forecasting**
# Using machine learning to predict grocery sales
# <hr>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.stats import linregress
from sklearn import preprocessing
from scipy import stats
import warnings
import math
import datetime
sns.set()
sns.set_style('whitegrid')
# plt.style.use("dark_background")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# # I. Getting Started
# <hr>

# In[2]:


path = '../input/store-sales-time-series-forecasting/'


# In[3]:


train = pd.read_csv(path + 'train.csv', parse_dates=['date'], infer_datetime_format=True)
train.head(5)


# The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.
#  - **store_nbr** (id) identifies the store at which the products are sold.
#  - **family** (categorical) identifies the type of product sold.
#  - **sales** (discrete) gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
#  - **onpromotion** (discrete) gives the total number of items in a product family that were being promoted at a store at a given date.

# In[4]:


test = pd.read_csv(path + 'test.csv', parse_dates = ['date'], infer_datetime_format=True)
ids = test['id']
pd.concat([test.head(1), test.tail(1)], axis = 0)


# Need to predict the sales between August 16 - 31.

# In[5]:


# creating a combined dataset with both test and train rows.
n_train = train.shape[0]
n_test = test.shape[0]
df = pd.concat([train, test], axis = 0)


# In[6]:


stores = pd.read_csv(path + 'stores.csv')
stores.head(5)


# - Store metadata, including city, state, type, and cluster.
# - cluster is a grouping of similar stores.

# In[7]:


oil = pd.read_csv(path + 'oil.csv', parse_dates=['date'], infer_datetime_format=True)
oil.tail(5)


# Daily Price of oil in Ecuador. We have the data between 16 - 31 August 2017, so this can be used as a feature.

# In[8]:


holidays = pd.read_csv(path + 'holidays_events.csv', parse_dates=['date'], infer_datetime_format=True)
holidays.tail(5)


# Holidays and Events, with metadata
# - NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
# - Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

# In[9]:


transactions = pd.read_csv(path + 'transactions.csv', parse_dates=['date'], infer_datetime_format=True)
transactions.tail(5)


# We don't have transaction data between 16 - 31 August 2017, so it's this data can't be used for predictions.

# ### Additional Notes
# - Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
# - A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

# ### Creating necessary columns using date.

# In[10]:


df["year"],df["month"], df["day"] = pd.DatetimeIndex(df['date']).year, pd.DatetimeIndex(df['date']).month, pd.DatetimeIndex(df['date']).day

df['month'].replace([var for var in range (1, 13)],['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'],inplace=True)
df['month'] = pd.Categorical(df['month'],
                             categories=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'],
                             ordered=True)


# In[11]:


df = df.set_index('date')
df['dayofyear'] = df.index.dayofyear
df['dayofweek'] = df.index.dayofweek
df['week'] = df.index.week


# In[12]:


train = df.iloc[ : n_train, ]
test = df.iloc[n_train : , ]


# #### Let's first analyse the missing values.

# In[13]:


def get_missingvalues(data):
	stats = data.isnull().sum()
	if stats.max() == 0:
		print("No missing values :)")
	else:
		for feature in stats.index:
			if stats[feature] > 0:
				print('{} has {} values missing.'.format(feature, stats[feature]))

print("1. Main dataframe: ")
get_missingvalues(df)
print("2. Oil dataframe: ")
get_missingvalues(oil)
print("3. Holidays dataframe: ")
get_missingvalues(holidays)
print("4. Transactional dataframe: ")
get_missingvalues(transactions)
print("5. Stores dataframe: ")
get_missingvalues(stores)


# **Oil** is the only dataframe with missing values.

# # II. Exploratory Data Analysis
# <hr>

# ### A) Distribution

# In[14]:


# Let's check distribution of sales - target variable.
sns.distplot(train['sales'], kde = True)
print("Skew : {} Kurtosis : {}".format(train['sales'].skew(), train['sales'].kurt()))
plt.title("Distribution of feature - sales");


# Most of the sales are close to 0.

# In[15]:


# Let's check yearly sales. 
plt.figure(figsize = (18, 9))
sns.boxenplot(data = train, x = 'year', y = 'sales', palette="RdPu_r")
plt.title("Distribution of sales in a year", fontsize = 20);


# ### B) Time Analysis - **Trend**

# In[16]:


# let's plot oil prices.
fig, ax = plt.subplots(figsize = (18, 10))
data = oil.set_index('date')
trend = data.rolling(window=7, center = True, min_periods = 3).mean()
ax.plot(trend, linewidth = 3, color = 'red')
sns.scatterplot(data = oil, x = 'date', y = 'dcoilwtico', color = '0.5', ax=ax)
sns.lineplot(data = oil, x = 'date', y = 'dcoilwtico', color = '0.5', ax=ax, linewidth = 0.5)
ax.set_title("Oil Prices", fontsize = 18);


# Prices fell sharply in 2014.

# In[17]:


# Let's start by plotting sales price. 
fig, ax = plt.subplots(figsize = (16, 10))
data = train.loc[ : , 'sales']
data = data.groupby('date').sum()
trend = data.rolling(window=30, center=True, min_periods=15).mean()
ax.plot(trend, aa = True, color = '#C94B94')
trend = data.rolling(window=365, center=True, min_periods=184).mean()
ax.plot(trend, color = '#490E5E', linewidth = 3)
ax.legend(['30 day rolling window', '365 day rolling window'])
ax.set_title("Trend - sales", fontsize = 18);


# Upward trend present. Maybe related to lower oil prices? 

# ### C) Time Analysis - **Seasonality**

# In[18]:


fig, axs = plt.subplots(3, 1, figsize = (16, 15))

for year, color in zip(train.year.unique(), sns.color_palette("RdPu_r")):
#     yearly = train[train.year == year]
    sns.lineplot(data = train[train.year == year].groupby('dayofyear')['sales'].mean(), color=color,ax = axs[0], linewidth = 1.5, label = str(year))
sns.lineplot(data = train.groupby('dayofyear')['sales'].mean(), color = 'black',ax = axs[0], linewidth = 6, label = 'mean')   

    
axs[0].set_title("Yearly Sales", fontsize = 18)
    
for month, color in zip(train.month.unique(), sns.color_palette("winter", n_colors = 12)):
#     monthly = train[train.month == month]
    sns.lineplot(data = train[train.month == month].groupby('day')['sales'].mean(), color=color,ax = axs[1], linewidth = 1.5, label = month)             
sns.lineplot(data = train.groupby('day')['sales'].mean(), color = 'black',ax = axs[1], linewidth = 6, label = 'mean')   

axs[1].set_title("Monthly Sales", fontsize = 18)

for week, color in zip(train.week.unique(), sns.color_palette('summer', n_colors = 53)):
    sns.lineplot(data = train[train.week == week].groupby('dayofweek')['sales'].mean(), color=color, ax = axs[2], linewidth = 1.5)
sns.lineplot(data = train.groupby('dayofweek')['sales'].mean(), color = 'black', ax = axs[2], linewidth = 6, label = 'mean')    

axs[2].set_title("Weekly Sales", fontsize = 18)

plt.tight_layout()


# #### Analysis : 
# - Sales start slow but pick up as the year ends. Maybe because of holidays like christmas? 
# - Montly sales are highest during the start and end of the month, and a slight uptick is present during the 15th. Could be because of public sector salaries.
# - Weekly sales have a strong seasonality, where sales dip during the middle of the week and peak at the ends.

# #### Plotting Periodogram
# Creating a periodogram will give us a better understanding of the exact time periods for seasons.

# In[19]:


# Creating a periodogram.
from scipy.signal import periodogram
fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
freqencies, spectrum = periodogram(
    train['sales'],
    fs=fs,
    detrend='linear',
    window="boxcar",
    scaling='spectrum',
)
fig, ax = plt.subplots(figsize = (16, 5))
ax.step(freqencies, spectrum, color="purple")
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
ax.set_xticklabels(
    [
        "Annual (1)",
        "Semiannual (2)",
        "Quarterly (4)",
        "Bimonthly (6)",
        "Monthly (12)",
        "Biweekly (26)",
        "Weekly (52)",
        "Semiweekly (104)",
    ],
    rotation=60,
)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel("Variance")
ax.set_title("Periodogram", fontsize = 18);


# Monthly, Biweekly, Weekly, Semiweekly seasonality present.

# ### D) **C → T** (Categorical vs Target Analysis)
# Categorical features are - store_nbr and family.

# In[20]:


data = train.groupby('store_nbr')['sales'].mean().sort_values(ascending = False)
plt.figure(figsize = (18, 10))
sns.barplot(data=data, x = data.index.astype("str"), y = data, palette = "RdPu_r",   errcolor=".2", edgecolor=".2")
plt.title("Sales vs Store", fontsize = 18);


# Store has a strong effect on sales.

# In[21]:


fig, axs = plt.subplots(1, 2, figsize = (18, 10))

data_family = train.groupby('family')['sales'].sum().sort_values(ascending = False)
sns.barplot(y=data_family.index, x=data_family.values, palette = "RdPu_r",   errcolor=".2", edgecolor=".2", ax = axs[0])
axs[0].set_title("Sales vs Item Family", fontsize = 18)

others = data_family[-20:].sum()
data_family = data_family[:13]
data_family['OTHERS'] = others
plt.title("Distribution of sales", fontsize = 18)
data_family.plot.pie()
plt.tight_layout()


# Some products are more poplular than others.

# ### E) **D → T** (Discrete vs Target analysis)
# Let's analyize the only discrete independent variable - onpromotion.

# In[22]:


fig, axs = plt.subplots(1, 2, figsize = (18, 7))
sns.heatmap(train[['sales', 'onpromotion']].corr(), square = True, annot = True, cmap = "RdPu", vmax=1, vmin=-1, fmt = ".1f", ax = axs[0]);
axs[0].set_title("Correlation Plot")

sns.scatterplot(data = train, x = 'onpromotion', y = 'sales',ax = axs[1], ci = None, color = '#490E5E')
axs[1].set_title("Linear Relation")
fig.suptitle('D → T (sales vs onpromotion)', fontsize = 18);


# Strong correlation (0.4)

# ### F) Holidays

# In[23]:


# Lets get all unique holidays and their dates.
# there are 103 holidays. Let's focus on national holidays.
holidays = holidays.query("locale in ['National']").set_index('date')


# Any holiday with a +/- is a day leading upto the holiday. Let's get rid of that to reduce dimensions.

# In[24]:


for date in holidays.index:
    name=holidays.loc[date]['description']
    if '+' in name or '-' in name:
        holidays.drop(index = date, inplace = True)


# ### Let's see how sales change in 2016 because of holidays

# In[25]:


dates = [var for var in holidays.loc["2016", ].index if var in train.index]
fig, ax = plt.subplots(figsize = (18, 7))
sns.lineplot(data = train[train['year'] == 2016].sales, estimator = 'sum', ax = ax, color='#C94B94', label = 'sales')
data = train.groupby(by = train.index).sum().loc[dates, 'sales']
plt.scatter(data.index, data, s = 150, color = '#490E5E', label = 'Holiday');


# #### No clear trend. We'll have to deseason the data first.

# # III. Feature Engineering
# <hr>

# ## Adding Lag Features for _sales_

# **NOTE :** We can't just simple use the shift function to get lags here. That will give us the sales of some other family of products sold that day.
# <br> We need to take the sales of the same family and store_nbr.

# In[26]:


df = df.reset_index()
df = df.set_index(['date', 'store_nbr', 'family'])
entries_perday = len(df.loc['2013-01-01'])
df = df.reset_index().set_index('date')
print('Number of entries in a day across all stores : {}'.format(entries_perday))


# There are 1782 entries in a day. So if row 1 is about store 1, family - 'AUTOMOTIVE', row 1783 will be about the same store, and family but on the next day.

# In[27]:


for lag in range(1, 11):
    df['sales_lag' + str(lag)] = df['sales'].shift(1782 * lag)
df.head(3)


# Successfully added lag features.

# In[28]:


train = df.iloc[:n_train, ]
test = df.iloc[n_train:, ]


# ### Analysis of lagged sales

# ##### Corr-plot

# In[29]:


plt.figure(figsize = (16, 8))
columns = [col for col in train.columns if 'sales' in col]
sns.heatmap(data = train[columns].corr(), square = True,
            annot = True, cmap = "Reds", vmax=1, vmin=.83, fmt = ".3f")
plt.xticks(rotation = 40)
plt.title('Correlation - sales vs lagged sales', fontsize = 18);


# ##### Scatter Plot

# In[30]:


fig, axs = plt.subplots(2, 5, figsize = (15, 6))
axs = axs.flatten()
for i in range(1, 11):
    feature = 'sales_lag' + str(i)
    sns.scatterplot(x=train[feature], y=train['sales'], ax=axs[i - 1], s=5)
    axs[i - 1].set_title(feature, fontsize = 14)

plt.suptitle('Scatterplot - Lags vs Sales', fontsize = 18)
plt.tight_layout();


# Looks like a linear relation, but with a lot of outliers, because of the large number of rows.

# In[31]:


df.drop(columns = ['sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag4', 'sales_lag5', 'sales_lag6', 'sales_lag8', 'sales_lag9', 'sales_lag10'], inplace = True)


# ### Let's create a combined dataset.

# In[32]:


# Cleaning oil price dataset.
oil = oil.set_index('date')
oil = oil.rename(columns = {'dcoilwtico' : 'oilprice'})


# #### Removing duplicate holidays 
# In certain dates many holidays coincide with each other. This will cause problems while joining the datasets. So i'm removing duplicates.

# In[33]:


holidays = holidays.groupby(holidays.index).first()


# #### Concatenating everything

# In[34]:


df = df.join(oil, on = 'date', how = 'left')
df = df.join(holidays, on = 'date', how = 'left')
df['oilprice'] = df['oilprice'].fillna(method = 'bfill')
df.head(5)


# In[35]:


df.drop(columns = ['id', 'dayofyear', 'week', 'day', 'year', 'month'], inplace = True)


# #### Dealing with categorical variables
# Replacing holidays with a single feature, which contains weather the day is a holiday or a work day can reduce dimensionality.

# In[36]:


df['work_day'] = 1
df.loc[df['dayofweek'] > 4, 'work_day'] = 0
df.loc[df['description'].notnull(), 'work_day'] = 0
df.loc[df.type == 'Bridge', 'work_day'] = 0
df.loc[df.type == 'Work Day', 'work_day'] = 1
df.loc[df.type == 'Transfer', 'work_day'] = 0
df.loc[(df.type == 'Holiday') & (df.transferred == False), 'work_day'] = 0
df.loc[(df.type == 'Holiday') & (df.transferred == True), 'work_day'] = 0


# In[37]:


df.drop(columns = ['locale', 'locale_name', 'description', 'transferred'],  inplace = True)


# In[38]:


df = pd.get_dummies(df, columns=['type'], drop_first=False)


# Creating dummy variables for month, and day of week

# In[39]:


df.dayofweek = df.dayofweek.astype('str')
df = pd.get_dummies(df, columns = ['dayofweek'], drop_first=True)


# In[40]:


train = df.iloc[:n_train, ]
test = df.iloc[n_train:, ]


# ### Creating a LinearModel using only time.
# Let's create a model without using other features like oil-price, store-nbr, family etc, to get a idea about how trends and seasons look in our data.

# #### Creating Statsmodel Deterministic Processs

# In[41]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
# choosing order = 4 because monthly, biweekly, and weekly periodicity was observered in the periodogram.
fourier = CalendarFourier(freq="W", order=4)
data = train.reset_index().set_index(['store_nbr', 'family', 'date'])
y = data['sales'].unstack(['store_nbr', 'family'])

dp = DeterministicProcess(
    index= y.index,
    order=1,
    seasonal=False,
    constant=False,
    additional_terms = [fourier],
    drop = True
)

X = dp.in_sample()
X.shape


# In[42]:


# Let's look at our sin/cos waves created by fourier trainsform.
fig, axs = plt.subplots(1, 2, figsize = (18, 5))

for feature, color in zip(X.columns[3: ], sns.color_palette('winter', n_colors=8)):
    if feature.find('sin') != -1:
        axs[0].plot(X[feature].head(8), color = color)
    elif feature.find('cos') != -1:
        axs[1].plot(X[feature].head(8), color = color)

axs[0].set_title('sin', fontsize = 18)
axs[1].set_title('cos', fontsize = 18)

plt.suptitle('Fourier Waves', fontsize = 24);


# In[43]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
y_pred = pd.DataFrame(data = model.predict(X),
                      index = y.index,
                      columns=y.columns)


# ### Visualizing

# In[44]:


fig, axs = plt.subplots(4, 1, figsize = (18, 15))
data = y.mean(axis = 1)
y_pred = y_pred.mean(axis = 1)

# plotting overall trend
trend = data.rolling(window=1, center=True, min_periods=0).mean()
axs[0].plot(trend, linewidth = 1, color = (0.0, 0.3333333333333333, 0.8333333333333334))
axs[0].plot(y_pred,  linewidth = 2, color = (0.3333333333333333, 0.6666666666666666, 0.4))
axs[0].set_title("Overall Prediction", fontsize = 18);
axs[0].legend(['Actual Values - Daily rolling mean', 'Time Series Prediction'])

# plotting yearly trend
data = data.loc['2016']
trend = data.rolling(window=1, center=True, min_periods=0).mean()
axs[1].plot(trend, linewidth = 2, color = (0.0, 0.3333333333333333, 0.8333333333333334))
axs[1].plot(y_pred.loc['2016'], linewidth = 3, color =  (0.3333333333333333, 0.6666666666666666, 0.4))
axs[1].set_title("Yearly Prediction - 2016", fontsize = 18);
axs[1].legend(['Actual Values - Daily rolling mean', 'Time Series Prediction'])

# plotting montly trend
data = data.loc['2016-01']
trend = data.rolling(window=1, center=True, min_periods=0).mean()
axs[2].plot(trend, linewidth = 2, color = (0.0, 0.3333333333333333, 0.8333333333333334))
axs[2].plot(y_pred.loc['2016-01'], linewidth = 3, color =  (0.3333333333333333, 0.6666666666666666, 0.4))
axs[2].set_title("Monthly Prediction - Jan 2016", fontsize = 18);
axs[2].legend(['Actual Values - Daily rolling mean', 'Time Series Prediction']);

# plotting weekly trend
data = data.loc['2016-01'].iloc[3:10]
trend = data.rolling(window=1, center=True, min_periods=0).mean()
axs[3].plot(trend, linewidth = 2, color = (0.0, 0.3333333333333333, 0.8333333333333334))
axs[3].plot(y_pred.loc['2016-01-04' : '2016-01-10'], linewidth = 3, color = (0.3333333333333333, 0.6666666666666666, 0.4))
axs[3].set_title("Weekly Prediction - Jan First week 2016", fontsize = 18);
axs[3].legend(['Actual Values - Daily rolling mean', 'Time Series Prediction']);

plt.tight_layout();


# #### Our model follows all seasons reasonably well!

# ### Next, let's check the effect of holidays on Prediction.

# In[45]:


delta = y.mean(axis=1) - y_pred


# In[46]:


dates = [var for var in holidays.loc["2016", ].index if var in train.index]

points = delta.to_frame(name = 'sales').loc['2016']
points = points.groupby(by = points.index).mean()

fig, ax = plt.subplots(figsize = (18, 8))
ax.plot(points, color = 'lightgrey')
plt.scatter(x = points.index, y = points['sales'], color='#C94B94', s = 5)
ax.scatter(x = dates, y = points.loc[dates, 'sales'], s = 100,  color = (0.0, 0.3333333333333333, 0.8333333333333334))
plt.plot([datetime.datetime(2016, 1, 1), datetime.datetime(2017, 1, 1)], [0, 0], color = 'black', linewidth = 1)
plt.title('Error Explained by Holidays', fontsize = 16);


# Certain Holidays like New Years day can reduce errors.

# # IV. Modelling
# <hr>

# #### Getting started

# In[47]:


# Helper function for model evaluation
from sklearn import metrics
def visualize(predict_data):
    fig, ax = plt.subplots(figsize = (8, 8))
    
    sns.scatterplot(data = predict_data, x = "Actual Values", y = "Predicted", color='#ff846b', s=.5)
    sns.lineplot(x = [0, 20000], y = [0, 20000], color = 'r', linewidth = 1)
    ax.set_xlim([0, 20000])
    ax.set_ylim([0, 20000])
    
    mse = metrics.mean_squared_error(predict_data['Actual Values'], predict_data['Predicted'])
    r2 = metrics.r2_score(predict_data['Actual Values'], predict_data['Predicted'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
    r'mse=%.2f' % (mse, ),
    r'r2 score=%.2f' % (r2, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', bbox=props)
    plt.show()


# In[48]:


train.drop(columns = ['store_nbr', 'family'], inplace = True)
train = train.groupby(by = train.index).first()
X = X.join(train, how = 'left')
X = X.fillna(0)
X.drop(columns = ['sales', 'work_day'], inplace = True)


# #### Splitting into test and train datasets

# In[49]:


X, y = X.head(int(len(X) * 0.5)), y.head(int(len(y) * 0.5))


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## _Ridge Regression_

# In[51]:


from sklearn.linear_model import  Ridge
ridge_reg = Ridge(random_state=1)
ridge_reg.fit(X_train, y_train)
y_pred = pd.DataFrame(ridge_reg.predict(X_test), index=y_test.index, columns=y_test.columns)


# In[52]:


from sklearn.metrics import mean_squared_log_error
y_pred   = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_test.stack(['store_nbr', 'family']).reset_index().copy()

y_pred.columns = ['date', 'store_nbr', 'family', 'sales']
y_target.columns = ['date', 'store_nbr', 'family', 'sales']
y_target['sales_pred'] = y_pred['sales'].clip(0.)


# In[53]:


predict_data = pd.DataFrame({"Actual Values" : y_target["sales"], "Predicted" : y_pred["sales"]})
visualize(predict_data)


# ## _Multiple Linear Regression_

# In[54]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = pd.DataFrame(lr.predict(X_test), index=y_test.index, columns=y_test.columns)


# In[55]:


from sklearn.metrics import mean_squared_log_error
y_pred   = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_test.stack(['store_nbr', 'family']).reset_index().copy()

y_pred.columns = ['date', 'store_nbr', 'family', 'sales']
y_target.columns = ['date', 'store_nbr', 'family', 'sales']
y_target['sales_pred'] = y_pred['sales'].clip(0.)


# In[56]:


predict_data = pd.DataFrame({"Actual Values" : y_target["sales"], "Predicted" : y_pred["sales"]})
visualize(predict_data)


# ## _Support Vector Regressor_

# In[57]:


from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
svr = MultiOutputRegressor(SVR(kernel='poly'))
svr.fit(X_train, y_train)
y_pred = pd.DataFrame(svr.predict(X_test), index=y_test.index, columns=y_test.columns)


# In[58]:


y_pred   = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_test.stack(['store_nbr', 'family']).reset_index().copy()

y_pred.columns = ['date', 'store_nbr', 'family', 'sales']
y_target.columns = ['date', 'store_nbr', 'family', 'sales']
y_target['sales_pred'] = y_pred['sales'].clip(0.)


# In[59]:


predict_data = pd.DataFrame({"Actual Values" : y_target["sales"], "Predicted" : y_pred["sales"]})
visualize(predict_data)


# ## _Random Forest Regressor_

# In[60]:


from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators = 200, random_state = 0)
random_forest.fit(X_train, y_train)
y_pred = pd.DataFrame(random_forest.predict(X_test), index=y_test.index, columns=y_test.columns)


# In[61]:


y_pred   = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_test.stack(['store_nbr', 'family']).reset_index().copy()

y_pred.columns = ['date', 'store_nbr', 'family', 'sales']
y_target.columns = ['date', 'store_nbr', 'family', 'sales']
y_target['sales_pred'] = y_pred['sales'].clip(0.)


# In[62]:


predict_data = pd.DataFrame({"Actual Values" : y_target["sales"], "Predicted" : y_pred["sales"]})
visualize(predict_data)


# ## _XG Boost_

# In[63]:


import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
xgb_reg = MultiOutputRegressor(xgb.XGBRegressor(n_estimators = 100))
xgb_reg.fit(X_train, y_train)
y_pred = pd.DataFrame(xgb_reg.predict(X_test), index=y_test.index, columns=y_test.columns)


# In[64]:


y_pred   = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_test.stack(['store_nbr', 'family']).reset_index().copy()

y_pred.columns = ['date', 'store_nbr', 'family', 'sales']
y_target.columns = ['date', 'store_nbr', 'family', 'sales']
y_target['sales_pred'] = y_pred['sales'].clip(0.)


# In[65]:


predict_data = pd.DataFrame({"Actual Values" : y_target["sales"], "Predicted" : y_pred["sales"]})
visualize(predict_data)

