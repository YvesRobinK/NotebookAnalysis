#!/usr/bin/env python
# coding: utf-8

# ## Store Sales - Getting Started with Time Series
# 
# #### Thanks for checking out my Notebook! Feel free to copy and edit on you own : )
# 
# #### Getting started with Time Series might seem *overwhelming* with so many new concepts.
# 
# #### I made this notebook to illustarte a simple work flow for solving a typical Time Series problem.
# 
# #### The solution is mostly taken from [Andrej Marinchenko](https://www.kaggle.com/code/andrej0marinchenko/hyperparamaters), [BIZEN](https://www.kaggle.com/code/hiro5299834/store-sales-ridge-voting-bagging-et-bagging-rf), and [KDJ2020](https://www.kaggle.com/code/dkomyagin/simple-ts-ridge-rf/notebook).
# 
# #### I clenaed up their code and reformated some plots to make the notebook shorter and easier to read.
# 
# #### If it's helpful to your learning process, please upvote so that more people can see it.
# 
# #### All comments and feedbacks are welcome!
# 
# #### If you need more explanations about some theories, feel free to check out this [Kaggle Course](https://www.kaggle.com/learn/time-series).
# 
# #### Note: it will take a bit more than 30 minutes to run the entire notebook.

# # Setting Things Up

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, style
import seaborn as sns
from plotly import express as px, graph_objects as go
rcParams['figure.figsize'] = (10, 6)

from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor

import gc
gc.enable()
from warnings import filterwarnings, simplefilter
filterwarnings('ignore')
simplefilter('ignore')


# # Train and Test Data

# In[2]:


train = pd.read_csv('../input/store-sales-time-series-forecasting/train.csv',
                    parse_dates = ['date'], infer_datetime_format = True,
                    dtype = {'store_nbr' : 'category',
                             'family' : 'category'},
                   usecols = ['date', 'store_nbr', 'family', 'sales'])

train['date'] = train.date.dt.to_period('D')
train = train.set_index(['date', 'store_nbr', 'family']).sort_index()
train.head()


# In[3]:


test = pd.read_csv('../input/store-sales-time-series-forecasting/test.csv',
                   parse_dates = ['date'], infer_datetime_format = True)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['date', 'store_nbr', 'family']).sort_values('id')
test.head()


# # Oil Data

# In[4]:


# Using the full date range
calendar = pd.DataFrame(index = pd.date_range('2013-01-01', '2017-08-31')).to_period('D')
oil = pd.read_csv('../input/store-sales-time-series-forecasting/oil.csv',
                  parse_dates = ['date'], infer_datetime_format = True,
                  index_col = 'date').to_period('D')
oil['avg_oil'] = oil['dcoilwtico'].rolling(7).mean()
calendar = calendar.join(oil.avg_oil)
calendar['avg_oil'].fillna(method = 'ffill', inplace = True)
calendar.dropna(inplace = True)
calendar.head()


# In[5]:


# Plotting oil price
_ = sns.lineplot(data = oil.dcoilwtico.to_timestamp())


# In[6]:


# Plotting the partial autocorrelation function
_ = plot_pacf(calendar.avg_oil, lags = 12)


# In[7]:


# Adding lages based on the auto correlation plot above (up to 5 will be reasonable)
n_lags = 3
for l in range(1, n_lags + 1) :
    calendar[f'oil_lags_{l}'] = calendar.avg_oil.shift(l)
calendar.dropna(inplace = True)
calendar.head()


# In[8]:


# Checking the correlation plot with different lags
lag1, lag2, lag3 = 'oil_lags_1', 'oil_lags_2', 'oil_lags_3'

fig = plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
sns.regplot(x = calendar[lag1], y = calendar.avg_oil)
plt.title(f'corr {calendar.avg_oil.corr(calendar[lag1])}')
plt.subplot(1,3,2)
sns.regplot(x = calendar[lag2], y = calendar.avg_oil)
plt.title(f'corr {calendar.avg_oil.corr(calendar[lag2])}')
plt.subplot(1,3,3)
sns.regplot(x = calendar[lag3], y = calendar.avg_oil)
plt.title(f'corr {calendar.avg_oil.corr(calendar[lag3])}');


# # Holiday Data

# In[9]:


hol = pd.read_csv('../input/store-sales-time-series-forecasting/holidays_events.csv',
                  parse_dates = ['date'], infer_datetime_format = True,
                  index_col = 'date').to_period('D')
hol = hol[hol.locale == 'National'] # Only taking National holiday so there's no false positive.
hol = hol.groupby(hol.index).first() # Removing duplicated holiday at the same date
hol.head()


# In[10]:


# Feature Engineering
calendar = calendar.join(hol) # Joining calendar with holiday dataset
calendar['dofw'] = calendar.index.dayofweek # Weekly day
calendar['wd'] = 1
calendar.loc[calendar.dofw > 4, 'wd'] = 0 # If it's saturday or sunday then it's not workday
calendar.loc[calendar.type == 'Work Day', 'wd'] = 1 # If it's Work Day event then it's a workday
calendar.loc[calendar.type == 'Transfer', 'wd'] = 0 # If it's Transfer event then it's not a workday
calendar.loc[calendar.type == 'Bridge', 'wd'] = 0 # If it's Bridge event then it's not a workday
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == False), 'wd'] = 0 # If it's holiday and the holiday is not transferred then it's holiday
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == True), 'wd'] = 1 # If it's holiday and transferred then it's not holiday
calendar = pd.get_dummies(calendar, columns = ['dofw'], drop_first = True) # One-hot encoding (Make sure to drop one of the columns by 'drop_first = True')
calendar = pd.get_dummies(calendar, columns = ['type']) # One-hot encoding for type holiday (No need to drop one of the columns because there's a "No holiday" already)
calendar.drop(['locale', 'locale_name', 'description', 'transferred'], axis = 1, inplace = True) # Unused columns
calendar.head()


# # Visualization - Sales of Each Product

# In[11]:


y = train.unstack(['store_nbr', 'family']).loc['2013':'2017']
family = {c[2] for c in train.index}
for f in family :
    ax = y.loc(axis = 1)['sales', :, f].plot(legend = None)
    ax.set_title(f)


# # Defining the Training Date

# In[12]:


# Start and end of training date (based on plots above)
sdate = '2017-04-30' 
edate = '2017-08-15'


# In[13]:


# Adding a feature for school fluctuations
school_season = [] 
for i, r in calendar.iterrows() :
    if i.month in [4, 5, 8, 9] :
        school_season.append(1)
    else :
        school_season.append(0)
calendar['school_season'] = school_season
calendar.head()


# # Deterministic Process

# In[14]:


y = train.unstack(['store_nbr', 'family']).loc[sdate:edate]
fourier = CalendarFourier(freq = 'W', order = 3)
dp = DeterministicProcess(index = y.index,
                          order = 1,
                          seasonal = False,
                          constant = False,
                          additional_terms = [fourier],
                          drop = True)
x = dp.in_sample()
x = x.join(calendar)
x.head()


# In[15]:


# Predicting for the next 16 days
x_test = dp.out_of_sample(steps = 16)
x_test = x_test.join(calendar)
x_test.head()


# # Linear and SVR Model

# In[16]:


# Using LinearRegression and SVR to make a generalized line
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae

lnr = LinearRegression(fit_intercept = True, n_jobs = -1, normalize = True)
lnr.fit(x, y)

yfit_lnr = pd.DataFrame(lnr.predict(x), index = x.index, columns = y.columns).clip(0.)
ypred_lnr = pd.DataFrame(lnr.predict(x_test), index = x_test.index, columns = y.columns).clip(0.)

svr = MultiOutputRegressor(SVR(C = 0.2, kernel = 'rbf'), n_jobs = -1)
svr.fit(x, y)

yfit_svr = pd.DataFrame(svr.predict(x), index = x.index, columns = y.columns).clip(0.)
ypred_svr = pd.DataFrame(svr.predict(x_test), index = x_test.index, columns = y.columns).clip(0.)

yfit_mean = pd.DataFrame(np.mean([yfit_svr.values, yfit_lnr.values], axis = 0), index = x.index, columns = y.columns).clip(0.)
ypred_mean = pd.DataFrame(np.mean([ypred_lnr.values, ypred_svr.values], axis = 0), index = x_test.index, columns = y.columns).clip(0.)

y_ = y.stack(['store_nbr', 'family'])
y_['lnr'] = yfit_lnr.stack(['store_nbr', 'family'])['sales']
y_['svr'] = yfit_svr.stack(['store_nbr', 'family'])['sales']
y_['mean'] = yfit_mean.stack(['store_nbr', 'family'])['sales']

print('LNR RMSLE :', np.sqrt(msle(y, yfit_lnr)))
print('SVR RMSLE :', np.sqrt(msle(y, yfit_svr)))
print('Mean RMSLE :', np.sqrt(msle(y, yfit_mean)),'\n')

print('LNR MAE :', mae(y, yfit_lnr))
print('SVR MAE :', mae(y, yfit_svr))
print('Mean MAE :', mae(y, yfit_mean))


# In[17]:


# Concatenating linear regression's prediction with the training data (blending)
ymean = yfit_lnr.append(ypred_lnr)
school = ymean.loc(axis = 1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES']
ymean = ymean.join(school.shift(1), rsuffix = 'lag1') # I'm also adding school lag for its yearly cycle.
x = x.loc['2017-05-01':]
x = x.join(ymean) # Concatenating linear result
x_test = x_test.join(ymean)
y = y.loc['2017-05-01':]


# # Final Model

# In[18]:


from joblib import Parallel, delayed
import warnings

# Import necessary library
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor

# SEED for reproducible result
SEED = 5

class CustomRegressor():
    
    def __init__(self, n_jobs=-1, verbose=0):
        
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.estimators_ = None
        
    def _estimator_(self, X, y):
    
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        if y.name[2] == 'SCHOOL AND OFFICE SUPPLIES': # SCHOOL AND OFFICE SUPPLIES has weird trend, we use decision tree instead.
            r1 = ExtraTreesRegressor(n_estimators = 225, n_jobs=-1, random_state=SEED)
            r2 = RandomForestRegressor(n_estimators = 225, n_jobs=-1, random_state=SEED)
            b1 = BaggingRegressor(base_estimator=r1,
                                  n_estimators=10,
                                  n_jobs=-1,
                                  random_state=SEED)
            b2 = BaggingRegressor(base_estimator=r2,
                                  n_estimators=10,
                                  n_jobs=-1,
                                  random_state=SEED)
            model = VotingRegressor([('et', b1), ('rf', b2)]) # Averaging the result
        else:
            ridge = Ridge(fit_intercept=True, solver='auto', alpha=0.75, normalize=True, random_state=SEED)
            svr = SVR(C = 0.2, kernel = 'rbf')
            
            model = VotingRegressor([('ridge', ridge), ('svr', svr)]) # Averaging result
        model.fit(X, y)

        return model

    def fit(self, X, y):
        from tqdm.auto import tqdm
        
        
        if self.verbose == 0 :
            self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                  verbose=0,
                                  )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in range(y.shape[1]))
        else :
            print('Fit Progress')
            self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                  verbose=0,
                                  )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in tqdm(range(y.shape[1])))
        return
    
    def predict(self, X):
        from tqdm.auto import tqdm
        if self.verbose == 0 :
            y_pred = Parallel(n_jobs=self.n_jobs, 
                              verbose=0)(delayed(e.predict)(X) for e in self.estimators_)
        else :
            print('Predict Progress')
            y_pred = Parallel(n_jobs=self.n_jobs, 
                              verbose=0)(delayed(e.predict)(X) for e in tqdm(self.estimators_))
        
        return np.stack(y_pred, axis=1)


# In[19]:


get_ipython().run_cell_magic('time', '', 'model = CustomRegressor(n_jobs=-1, verbose=1)\nmodel.fit(x, y)\ny_pred = pd.DataFrame(model.predict(x), index=x.index, columns=y.columns)\n')


# # Evaluation

# In[20]:


from sklearn.metrics import mean_squared_log_error
y_pred = y_pred.stack(['store_nbr', 'family']).clip(0.)
y_ = y.stack(['store_nbr', 'family']).clip(0.)

y_['pred'] = y_pred.values
print(y_.groupby('family').apply(lambda r : np.sqrt(np.sqrt(mean_squared_log_error(r['sales'], r['pred'])))))
print('RMSLE : ', np.sqrt(np.sqrt(msle(y_['sales'], y_['pred']))))


# In[21]:


y_sub = pd.DataFrame(model.predict(x_test), index = x_test.index, columns = y.columns).clip(0.)
y_sub.head()


# In[22]:


y_sub = y_sub.stack(['store_nbr', 'family'])
y_sub.head()


# # Submission

# In[23]:


sub = pd.read_csv('../input/store-sales-time-series-forecasting/sample_submission.csv')
sub['sales'] = y_sub.values
sub.head()
sub.to_csv('submission.csv', index = False)


# #### Don't forget to submit the result to the contest!
# 
# #### Also, please upvote to support my work : )
