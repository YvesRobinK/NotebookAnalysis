#!/usr/bin/env python
# coding: utf-8

# # Covid-19 ML forecasting with Linear Regression and ARIMA

# # 1. Introduction

# In this notebook we will analyze the globally spreading and development of Covid-19 which was first discovered in December 2019 in Wuhan. As of 12 April 2020, more than 1.2 million cases have been reported in 210 countries. We will try to forecast the development from 15 April 2020 to 15 May 2020 which is part of an ongoing Kaggle competition. I think it is very important to mention the idea behind this which was very well explained from my point of view by the Kaggle team:
# 
# > We understand this is a serious situation, and in no way want to trivialize the human impact this crisis is causing by predicting fatalities. Our goal is to provide better methods for estimates that can assist medical and governmental institutions to prepare and adjust as pandemics unfold.
# 
# In the first part of the notebook, we will explore the data in terms of information and quality. After we will clean the data and go on with feature engineering. In the third section we will have a look at different prediction methods, namely linear regression and arima. At the end, we will have a short stop and a look back before we create our submission file for the competetion itself. Even though it is a very serious topic, I hope we learn something together. Stay healthy!

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from xgboost import XGBRegressor
le = preprocessing.LabelEncoder()
from sklearn import linear_model

import plotly.express as px
import plotly.graph_objects as go

import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# In[3]:


import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

#pio.write_html(fig, file="index.html", auto_open=True)


# In[4]:


train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")


# # 2. Explorative data analysis
# ## 2.1 Confirmed Cases

# Before we start, let's take a quick look at the data structure. We have two data sets. The training data set contains six columns:
# * ID: Unique identifier
# * Province_state: Provinces and states of a specific country, e.g. Washington in the United States
# * Country_region: Countries as Germany, France or Spain
# * Date: Datestamp for the respective row
# * ConfirmedCases: The number of confirmed cases of Covid-19
# * Fatalities: The number of registered deaths by Covid-19

# In[5]:


display(train_df.isnull().sum()/len(train_df)*100)
display(test_df.isnull().sum()/len(test_df)*100)


# In[6]:


print("The lowest date in the train data set is {} and the highest {}.".format(train_df['Date'].min(),train_df['Date'].max()))
print("The lowest date in the test data set is {} and the highest {}.".format(test_df['Date'].min(),test_df['Date'].max()))


# As you can see we have a lot of missings in the Province_State column and there is an overlapping time period in the test and train data set. Our first task is to fix both issues.

# In[7]:


#just some cosmetic renaming
train_df.rename(columns={'Province_State':'State','Country_Region':'Country'}, inplace=True)
test_df.rename(columns={'Province_State':'State','Country_Region':'Country'}, inplace=True)


# In[8]:


#function for replacing all the missings in the state column
def missings(state, country):
    return country if pd.isna(state) == True else state


# In[9]:


#if there are no states specified for a country, the missing is replaced with the country´s name
train_df['State'] = train_df.apply(lambda x: missings(x['State'],x['Country']),axis=1)
test_df['State'] = test_df.apply(lambda x: missings(x['State'],x['Country']),axis=1)


# In[10]:


print("In our data set are {} countries and {} states.".format(train_df['Country'].nunique(),train_df['State'].nunique()))


# In[11]:


df_confirmedcases = train_df.groupby(['Country','State']).max().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index().drop(columns='Id')
df_confirmedcases[:20].set_index('Country').style.background_gradient(cmap='Oranges')


# The most Covid-19 cases are by far in the United States. It is followed by the four largest EU member states and China, the country where Covid-19 was first registered.

# In[12]:


import plotly.express as px

cases = 10

countries = df_confirmedcases[:cases]['Country'].unique().tolist()
plot = train_df.loc[(train_df['Country'].isin(countries))].groupby(['Date', 'Country', 'State']).max().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()
plot2 = train_df.groupby(['Date'])['ConfirmedCases'].sum().reset_index()

fig = px.bar(plot, x="Date", y="ConfirmedCases", color="Country", barmode="stack")

fig.add_scatter(x=plot2['Date'], y=plot2['ConfirmedCases'],name='Global Trend') # Not what is desired - need a line 

fig.update_layout(title='Confirmed Cases - Top {} Countries - {}'.format(cases,train_df['Date'].max()))
fig.show()
pio.write_html(fig, file="diagram_1.html", auto_open=True)


# In the diagram above we plotted the amount of confirmed cases of top 10 countries and additional the global development line. An interesting observation is that these 10 countries account for a large part of global development (see global trend line). At first glance it looks as if the curve is flattening, which is an indication that we are leaving the phase of exponential growth.

# In[13]:


from plotly.subplots import make_subplots


development = train_df.groupby(['Date'])['ConfirmedCases'].sum().sort_values(ascending=False)[:50].reset_index()
development['ConfirmedCases_Previous'] = development['ConfirmedCases'].shift(-1)
development['Difference'] = development['ConfirmedCases'] - development['ConfirmedCases_Previous']
development['Differece_Previous'] = development['Difference'].shift(-1)
development['Increase_quota'] = development['Difference'] / development['Differece_Previous']

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=development["Date"], y=development["Difference"], name="Absolute increase in cases"))
fig.add_scatter(x=development['Date'], y=development['Increase_quota'], name="Increase quota in %", secondary_y=True)

fig.update_layout(title='Absolute and relative Increase per Day')

fig.update_yaxes(title_text="<b>Absolute</b> increase", secondary_y=False)
fig.update_yaxes(title_text="<b>Relative</b> increase in %", secondary_y=True)

fig.show()
pio.write_html(fig, file="diagram_2.html", auto_open=True)


# We created two new variables: difference and increase quota. The first variable contains the difference in absolute cases from t0 and t-1. The second variable, increase quota, is the quotient of the difference t0 and t-1. As we can see, both developments flatten out over time. This can have two reasons:
# - We are entering the **next phase** of the corona epidemic and leaving the exponential growth phase...
# - The **data quality is insufficient** and only a fraction of the actual infections are counted and statistically recorded
# 
# To get a better picture, we now examine the number of fatalities.

# ## 2.2 Fatalities

# In[14]:


import plotly.express as px

cases = 10

countries = df_confirmedcases[:cases]['Country'].unique().tolist()
plot = train_df.loc[(train_df['Country'].isin(countries))].groupby(['Date', 'Country', 'State']).max().groupby(['Date', 'Country']).sum().sort_values(by='Fatalities', ascending=False).reset_index()
plot2 = train_df.groupby('Date')['Fatalities'].sum().sort_values(ascending=False).reset_index()

fig = px.bar(plot, x="Date", y="Fatalities", color="Country", barmode="stack")

fig.add_scatter(x=plot2['Date'], y=plot2['Fatalities'],name='Global Trend') # Not what is desired - need a line 

fig.update_layout(title='Fatalities - Top {} Countries - {}'.format(cases, train_df['Date'].max()))
fig.show()
pio.write_html(fig, file="diagram_3.html", auto_open=True)


# Basically the picture is the same as for the confirmed cases. On April 12, 2020 there are more than 114,000 reported fatalities. The majority of the deaths come from the 10 states that we have seen before in the confirmed cases. At first sight the curve looks a bit steeper than in the confirmed cases, so we will compare them.

# In[15]:


import plotly.express as px

plot = train_df.groupby('Date')['Fatalities'].sum().sort_values(ascending=False).reset_index()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=plot["Date"], y=plot["Fatalities"], name="Fatalities"))
fig.add_scatter(x=development['Date'], y=development['ConfirmedCases'], name="Confirmed Cases", secondary_y=True)

fig.update_layout(title='Confirmed Cases and Fatalities - {}'.format(train_df['Date'].max()))

fig.update_yaxes(title_text="Confirmed Cases", secondary_y=False)
fig.update_yaxes(title_text="Fatalities", secondary_y=True)

fig.show()
pio.write_html(fig, file="diagram_4.html", auto_open=True)


# Our impression is confirmed: The curve of the confirmed cases is recently flatter than that of the fatalities, but the difference is not enormous. If the growth of confirmed cases really does slow down, this development is plausible, since people do not die directly from corona, but later.

# ## 2.3 Geographic Spread of Covid-19

# In[16]:


import plotly.express as px

df_plot = train_df.loc[: , ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()

#df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")
df_plot.loc[:, 'Size'] = np.power(df_plot["ConfirmedCases"]+1,0.3)-1 #np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*300)

fig = px.scatter_geo(df_plot,
                     locations="Country",
                     locationmode = "country names",
                     hover_name="Country",
                     color="ConfirmedCases",
                     animation_frame="Date", 
                     size='Size',
                     projection="natural earth",
                     title="Global Spread of Covid-19",
                    width=1500, height=800)
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 5

fig.show()

#pio.write_html(fig, file="diagram_5.html", auto_open=True)
#py.plot(fig, filename = 'diagram_5', auto_open=True)


# The animation shows the spread of Covid-19 in the period from 01.01.2020 to 12.04.2020. Until 20.02. there were only single, smaller spots besides China, after that there was rapid growth first in Europe, then in the USA and finally in Africa. At present, almost every region in the world is affected by Covid 19.

# ## 2.4 Conclusion

# Let us briefly summarize the key findings:
# * There are around 1,8 million confirmed cases and 114,000 fatalities worldwide. The top 10 countries in terms of confirmed cases and fatalities make up the majority.
# * According to the current data, the spread seems to be slowing down somewhat.
# * Almost every region in the world is affected by Covid-19
# 
# However, experts assume that the number of unreported cases is significantly higher and only a fraction of the cases are recorded. Therefore, our observation may also be due to the fact that testing capacities in many countries are exhausted. We cannot answer this question with the available data material. For further investigations we have to assume that the available figures reflect the actual development.

# # 3. Feature Engineering
# ## 3.1 Country, State and Date
# In this chapter we will prepare the data in such a way that we can then make forecasts using various statistical methods. For this purpose we will clean up the data and create features. Feature Engineerung is an very interesting factor in time series analysis. If we assume that we want to forecast the confirmed cases and fatalities, we will only have the states, provinces and the date as inputs for our models. This seems a little too little to extrapolate the data points in a meaningful way, but more on this later in this chapter. Let us first take a look at the training and test data set.

# In[17]:


data_leak = pd.merge(train_df,test_df, how='inner', on='Date')['Date'].unique().tolist()
data_leak.append('2020-04-01')
data_leak.sort()
print("Both data sets contain the following dates: {}".format(data_leak))


# As we have already seen in the first chapter, there is an overlapping period of time. The rules for [this](https://www.kaggle.com/c/covid19-global-forecasting-week-4/overview/evaluation) competition state that only data prior to 2020-04-01 may be used. For this reason we delete the data from the training data.

# In[18]:


#removing overlapping dates from our trainings data set
train_df_fix = train_df.loc[~train_df['Date'].isin(data_leak)]
df_all = pd.concat([train_df_fix, test_df], axis = 0, sort=False)

#filling up the "new" NAs which were created by the concat process
df_all['ConfirmedCases'].fillna(0, inplace=True)
df_all['Fatalities'].fillna(0, inplace=True)
df_all['Id'].fillna(-999, inplace=True)
df_all['ForecastId'].fillna(-999, inplace=True)


# We have already noticed that we do not have many features available. Let's first make the date meaningful. Algorithms cannot read anything from the date, so we extract various features from it, such as the day and week. From the combination you could for example read weekly trends. In Germany I noticed that on Thursday and Friday on average more cases are reported and on Saturday and Sunday significantly less. This may have something to do with the reporting process: On Saturday and Sunday there are fewer people working in the relevant offices and not all cases are reported reliably.
# 
# Additionally we use SciKit´s LabelEncoder() for states and provinces, because most algorithms can't do anything with strings. The Label Encoder assigns a number to each different case in the respective column.

# In[19]:


#year is commented out because it is the same for every case
def create_features(df):
    df['Day_num'] = le.fit_transform(df['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.week
    df['Month'] = df['Date'].dt.month
    #df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    df['Country'] = le.fit_transform(df['Country'])
    country_dict = dict(zip(le.inverse_transform(df['Country']), df['Country'])) 
    
    df['State'] = le.fit_transform(df['State'])
    state_dict = dict(zip(le.inverse_transform(df['State']), df['State']))
    
    return df, country_dict, state_dict


# In[20]:


df_all, country_dict, state_dict = create_features(df_all)


# ## 3.2 Confirmed Cases and Fatalities - Lags and Trends

# If we want to predict a data point in the future, currently only Country, State and the date are known. We have no information about the number of confirmed cases or fatalities in the future. However, we do have information about the number of cases in the past and we should use this information. In time series analysis we speak of lags when the previous development of the target (e.g. Confirmed Cases) is recorded as features in the data set.
# 
# In addition, trends can be recorded, i.e. the short-term development. This can be described for a trend between two consecutive data points as (t0 -t1)/ t1.

# In[21]:


def lag_feature(df,target,lags):
    for lag in lags:
        lag_col = target + "_{}".format(lag)
        df[lag_col] = df.groupby(['Country','State'])[target].shift(lag, fill_value=0)
    return df

def trend_feature(df,target,trends):
    for trend in trends:
        trend_col = "Trend_" + target + "_{}".format(trend)
        df[trend_col] = (df.groupby(['Country','State'])[target].shift(0, fill_value=0) - df.groupby(['Country','State'])[target].shift(trend, fill_value=0))/ df.groupby(['Country','State'])[target].shift(trend, fill_value=0.0001)
    return df


# # 4. Modelling

# Now we have all necessary functions and the data is prepared accordingly. We will try two different approaches: 1) linear regressions and 2) ARIMA. We will start with the linear regression. You can find more information about the implementation and parameters of linear regression in SkyCit [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
# 
# ## 4.1 Linear Regression (not used for submission)
# Unlike other machine learning projects, we can’t just draw random samples via train-test-split function, because the data points are chronologically dependent on each other. Therefore we write our own function. In the training-dataset we have data available until 31.03.2020. This is the 69th day since 22.01.2020 (the first day in our data set). We will train our data until 31.03.2020 and then we will see how close our prediction is to the actual development. We will vary in the following:
# 
# * the number of lags used
# * the period of time we include for the training
# 
# A few notes on our linear regression function:
# * We train the algorithm for each country individually. If there are several states in a country, we also train for each of them separately. Since the states are in different phases of the Covid-19 pandemic, we will not be able to make good predictions with an algorithm that is trained on all data points.
# * In a first step we calculate the lags for the respective country or state. The right number of lags is obtained by trial and error. Too few lags lead to an overinterpretation of the short-term trend, too many lags mean that we do not take the short-term trend into account enough.
# * Then we logarithmise the targets and lags. A linear regression is not suitable for extrapolating exponential trends. Logarithms allow our algorithm to better interpret and process the data.
# 
# **Note**: The use of lags creates a new problem. These are only available for our training data. So we have only logs available for the first prediction. After that we have to write this prediction back into the training data set, recalculate the lag and make a new prediction. This process must be repeated for each data point.
# 
# **Credits**: Most of the code for the implementation comes from [this great notebook](https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions). I can also generally recommend reading it for theoretical assumptions about the further process. It helped me a lot to learn about forecasting.

# In[22]:


#different trainings- and testsets for confirmed cases and fatalities
def train_test_split_extend(df,d,day,filter_col_confirmed,filter_col_fatalities):
    
    df=df.loc[df['Day_num'] >= day]
    df_train = df.loc[df['Day_num'] < d]
    X_train = df_train
    
    Y_train_1 = df_train['ConfirmedCases']
    Y_train_2 = df_train['Fatalities']
    
    X_train_1 = X_train.drop(columns=filter_col_fatalities).drop(columns='ConfirmedCases')
    X_train_2 = X_train.drop(columns=filter_col_confirmed).drop(columns='Fatalities')
    
    df_test = df.loc[df['Day_num'] == d]
    x_test = df_test
    
    x_test_1 = x_test.drop(columns=filter_col_fatalities).drop(columns='ConfirmedCases')
    x_test_2 = x_test.drop(columns=filter_col_confirmed).drop(columns='Fatalities')
    
    x_test.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)
    
    X_train_1.drop('Id', inplace=True, errors='ignore', axis=1)
    X_train_1.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    X_train_2.drop('Id', inplace=True, errors='ignore', axis=1)
    X_train_2.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    x_test_1.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test_1.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    x_test_2.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test_2.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    return X_train_1, X_train_2, Y_train_1, Y_train_2, x_test_1, x_test_2


# In[23]:


def lin_reg(X_train, Y_train, x_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    pred = regr.predict(x_test)
    return regr, pred


# In[24]:


def country_calculation(df_all,country,date,day):
    df_country = df_all.copy()
    df_country = df_country.loc[df_country['Date'] >= date]
    df_country = df_country.loc[df_country['Country'] == country_dict[country]]
    features = ['Id', 'State', 'Country','ConfirmedCases', 'Fatalities', 'Day_num']
    df_country = df_country[features]
    
    # Lags
    df_country = lag_feature(df_country, 'ConfirmedCases',range(1, 40))
    df_country = lag_feature(df_country, 'Fatalities', range(1,20))

    filter_col_confirmed = [col for col in df_country if col.startswith('Confirmed')]
    filter_col_fatalities= [col for col in df_country if col.startswith('Fataliti')]
    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)
    
    # Apply log transformation
    df_country[filter_col] = df_country[filter_col].apply(lambda x: np.log1p(x))
    df_country.replace([np.inf, -np.inf], 0, inplace=True)
    df_country.fillna(0, inplace=True) ####
    
    # Start/end of forecast
    start = df_country[df_country['Id']==-999].Day_num.min()
    end = df_country[df_country['Id']==-999].Day_num.max()
    #
    for d in range(start,end+1):
   
        X_train_1, X_train_2, Y_train_1, Y_train_2, x_test_1, x_test_2 = train_test_split_extend(df_country,d,day,filter_col_confirmed,filter_col_fatalities)
        
        regr_1, pred_1 = lin_reg(X_train_1, Y_train_1, x_test_1)
        df_country.loc[(df_country['Day_num'] == d) & (df_country['Country'] == country_dict[country]), 'ConfirmedCases'] = pred_1[0]
        
        regr_2, pred_2 = lin_reg(X_train_2, Y_train_2, x_test_2)
        df_country.loc[(df_country['Day_num'] == d) & (df_country['Country'] == country_dict[country]), 'Fatalities'] = pred_2[0]
        
        df_country = lag_feature(df_country, 'ConfirmedCases',range(1, 40))
        df_country = lag_feature(df_country, 'Fatalities', range(1,20))

        df_country.replace([np.inf, -np.inf], 0, inplace=True)
        df_country.fillna(0, inplace=True)
        
    print("Calculation done.")
    return df_country


# In[25]:


def country_state_calculation(df_all,country, state, date,day):
    df_country = df_all.copy()
    df_country = df_country.loc[df_country['Date'] >= date]
    df_country = df_country.loc[df_country['Country'] == country_dict[country] & (df_country['State']==state_dict[state])]
    features = ['Id', 'State', 'Country','ConfirmedCases', 'Fatalities', 'Day_num']
    df_country = df_country[features]
    
    # Lags
    df_country = lag_feature(df_country, 'ConfirmedCases',range(1, 40))
    df_country = lag_feature(df_country, 'Fatalities', range(1,20))

    filter_col_confirmed = [col for col in df_country if col.startswith('Confirmed')]
    filter_col_fatalities= [col for col in df_country if col.startswith('Fataliti')]
    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)
        
    # Apply log transformation
    df_country[filter_col] = df_country[filter_col].apply(lambda x: np.log1p(x))
    df_country.replace([np.inf, -np.inf], 0, inplace=True)
    df_country.fillna(0, inplace=True) ####
    
    # Start/end of forecast
    start = df_country[df_country['Id']==-999].Day_num.min()
    end = df_country[df_country['Id']==-999].Day_num.max()
    #
    for d in range(start,end+1):
        X_train_1, X_train_2, Y_train_1, Y_train_2, x_test_1, x_test_2 = train_test_split_extend(df_country,d,day,filter_col_confirmed,filter_col_fatalities)
        
        regr_1, pred_1 = lin_reg(X_train_1, Y_train_1, x_test_1)
        df_country.loc[(df_country['Day_num'] == d) & (df_country['Country'] == country_dict[country]) & (df_country['State'] == state_dict[state]), 'ConfirmedCases'] = pred_1[0]
        
        regr_2, pred_2 = lin_reg(X_train_2, Y_train_2, x_test_2)
        df_country.loc[(df_country['Day_num'] == d) & (df_country['Country'] == country_dict[country]), 'Fatalities'] = pred_2[0]
        
        df_country = lag_feature(df_country, 'ConfirmedCases',range(1, 10))
        df_country = lag_feature(df_country, 'Fatalities', range(1,8))
        
        df_country.replace([np.inf, -np.inf], 0, inplace=True)
        df_country.fillna(0, inplace=True)
        
    print("Calculation done.")
    return df_country


# In[26]:


def plot_check(df_check,country):
    crosscheck = train_df[(train_df['Country'] == country) & (train_df['Date'] >= '2020-03-10')].reset_index()
    df_check = df_check[df_check['Day_num'] >= 48]
    df_check2 = df_check[:len(crosscheck)].reset_index()
    df_check2['CC_Crosscheck'] = crosscheck['ConfirmedCases']
    df_check2['Fat_Crosscheck'] = crosscheck['Fatalities']
    df_check2['ConfirmedCases_In'] = df_check2['ConfirmedCases']
    df_check2['Fatalities_In'] = df_check2['Fatalities']
    df_check2['ConfirmedCases_In'] = df_check2['ConfirmedCases'].apply(lambda x: np.expm1(x)).astype(int)
    df_check2['Fatalities_In'] = df_check2['Fatalities'].apply(lambda x: np.expm1(x)).astype(int)
    return df_check2


# In[27]:


def check_plot(df_check2,test_con):

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['ConfirmedCases_In'], name='Confirmed Cases - Prediction')
    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['CC_Crosscheck'], name='Confirmed Cases - Official')

    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['Fatalities_In'], name='Fatalities - Prediction', secondary_y=True)
    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['Fat_Crosscheck'], name='Fatalities - Official', secondary_y=True)

    fig.add_annotation(
            x=69,
            y=0,
            xref="x",
            yref="y",
            text="Split of train- and test dataset",
            showarrow=True,
            font=dict(
                #family="Courier New, monospace",
                size=12,
                color="#ffffff"
                ),
            align="center",
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=-0,
            ay=-345,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=1
            )

    fig.update_layout(title=test_con + ' Comparison of Predicted and Real Number of Cases',
                       xaxis_title='Number of Days since 2020-01-22',
                       yaxis_title='Confirmed Cases')

    fig.update_yaxes(title_text="Confirmed Cases", secondary_y=False)
    fig.update_yaxes(title_text="Fatalities", secondary_y=True)
    
    #py.plot(fig, filename = test_con, auto_open=True)


    return fig.show()


# I will not show the results for all the different parameters now. You can test them yourself in the linked notebook. In the following we will have a look at the results for the following parameters:
# * 40 Lags for ConfirmedCases
# * 20 Lags for Fatalities
# * Start of training date 10.03.2020
# 
# The Confirmed Cases are on the left Y-axis and the Fatalities on the right Y-axis.

# In[28]:


test_con = 'Germany'

df_check = country_calculation(df_all, test_con, '2020-03-10', 48)
df_check2 = plot_check(df_check,test_con)
check_plot(df_check2,test_con)


# In[29]:


test_con = 'Spain'

df_check = country_calculation(df_all, test_con, '2020-03-10', 48)
df_check2 = plot_check(df_check,test_con)
check_plot(df_check2,test_con)


# In[30]:


test_con = 'Algeria'

df_check = country_calculation(df_all, test_con, '2020-03-10', 48)
df_check2 = plot_check(df_check,test_con)
check_plot(df_check2,test_con)


# In[31]:


test_con = 'Andorra'

df_check = country_calculation(df_all, test_con, '2020-03-10', 48)
df_check2 = plot_check(df_check,test_con)
check_plot(df_check2,test_con)


# In[32]:


## Inputs
##day_start = 39 
##lag_size = 30
#
#date = '2020-03-10'
#day = 48
#
#train3 = train_df.copy()
##train3.Province_State.fillna("None", inplace=True)
#
#results_df = pd.DataFrame()
#
#import time
#tp = time.time()
#
## Main loop for countries
#for country in train3['Country'].unique():
#
#    # List of provinces
#    provinces_list = train3[train3['Country']==country]['State'].unique()
#        
#    # If the country has several Province/State informed
#    if len(provinces_list)>1:
#        for province_name in provinces_list:
#            pred_province = country_state_calculation(df_all,country,province_name,date,day)
#            results_df = pd.concat([results_df, pred_province])
#
#    else:
#        pred_country = country_calculation(df_all,country,date,day)
#        results_df = pd.concat([results_df, pred_country])
#        
#results_df_submit = results_df.copy()
#results_df_submit['ConfirmedCases'] = results_df_submit['ConfirmedCases'].apply(lambda x: np.expm1(x))
#results_df_submit['Fatalities'] = results_df_submit['Fatalities'].apply(lambda x: np.expm1(x))
#        
##get_submission(results_df_submit.loc[results_df_submit['ForecastId']!=-1], 'ConfirmedCases', 'Fatalities')
#print("Complete process finished in ", time.time()-tp)


# In[33]:


##submission
#real_result = results_df_submit[(results_df_submit['Id'] == -999)].replace([np.inf, -np.inf], 0)
#real_result['ConfirmedCases'] = real_result['ConfirmedCases'].astype(int)
#real_result['Fatalities'] = real_result['Fatalities'].astype(int)
#real_result_cc = real_result['ConfirmedCases'].tolist()
#real_result_f = real_result['Fatalities'].tolist()
#submission = pd.DataFrame({'ForecastId':test_df['ForecastId'], 'ConfirmedCases':real_result_cc, 'Fatalities': real_result_f})
#submission.to_csv('submission.csv', index=False)


# ## 4.2 Linear Regression — Conclusion
# For Germany and Spain, the forecast initially looks good, but it can be seen that the gap between the actual figures and the forecast is widening. We have also looked at Algeria, as a country with few cases so far. Here too, the forecast and actual figures are drifting apart. I chose Andorra because there are no cases here so far. The prognosis is completely absurd and would ruin the whole score for us at Kaggle. We could intercept this phenomenon by hard-coding that countries without confirmed cases will continue to be predicted at zero. Instead, we will continue with the ARIMA method.

# ## 4.3 ARIMA (used for submission)
# ARIMA stands for AutoRegressive (AR) Integrated (I) Moving Average (MA). The provided data as input must be an univariate series, since ARIMA calculates future datapoints from the past. That is exactly what we were trying to do with linear regression as well. ARIMA basically has three important parameters:
# * p: The autoregressive part of the model. Simplified one can say that the model assumes that if there were many confirmed cases yesterday and the day before, there will be many confirmed cases today and tomorrow.
# * d -> The integrated part of the model that describes the amount of differentiation. If the available data are not stationary and contain trends, ARIMA can extract this seasonality.
# * q -> The moving average part of the model. By forming moving averages, random effects can be smoothed.
# 
# **An important note**: ARIMA is not able to take any external factors into account. The data points are only extrapolated based on historical data. If there are bans of going out imposed overnight in a state, our model will not provide good forecasts.
# 
# **Credits**: Most of the code for the implementation comes from [this great notebook](https://www.kaggle.com/hossein2015/covid-19-week-3-sarima-x-approach).

# In[34]:


get_ipython().system('pip install pyramid.arima')
from pyramid.arima import auto_arima


# In[35]:


# Import the necessary libraris #
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
warnings.filterwarnings(action='ignore')
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
# Define the directory for the input files (train + test + submission) #
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[36]:


def RMSLE(pred,actual):
    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
pd.set_option('mode.chained_assignment', None)
# Import the train & test data for COVID-19 (Week 3) #
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
# Replace the missing values in the train & test data sets #
train['Province_State'].fillna('', inplace=True)
test['Province_State'].fillna('', inplace=True)
# Convert the "Date" Variable in the training & test sets #
train['Date'] =  pd.to_datetime(train['Date'])
#train =  train.loc[train['Date'] <= '2020-04-01']
test['Date'] =  pd.to_datetime(test['Date'])
# Sort values in the training & test sets #
train = train.sort_values(['Country_Region','Province_State','Date'])
test = test.sort_values(['Country_Region','Province_State','Date'])


# In[37]:


# Defining key dates for reference purposes #
feature_day = [1,20,50,100,200,500,1000]
def CreateInput(data):
    feature = []
    for day in feature_day:
        data.loc[:,'Number day from ' + str(day) + ' case'] = 0
        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        
        else:
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       
        for i in range(0, len(data)):
            if (data['Date'].iloc[i] > fromday):
                day_denta = data['Date'].iloc[i] - fromday
                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 
        feature = feature + ['Number day from ' + str(day) + ' case']
    
    return data[feature]


# In[38]:


pred_data_all = pd.DataFrame()
for country in train['Country_Region'].unique():
    for province in train[(train['Country_Region'] == country)]['Province_State'].unique():
        print(country + ' and ' + province)
        #create dataframe for a specific country
        df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
        df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        #create features -> number of cases on a specific date
        X_train = CreateInput(df_train)
        #last 12 confirmed cases in train data set
        y_train_confirmed = df_train['ConfirmedCases'].ravel()
        #last 12 confirmed fatalities in train data set
        y_train_fatalities = df_train['Fatalities'].ravel()
        #create features in test dataset-> number of cases on a specific date
        X_pred = CreateInput(df_test)
        #creates reversed list of the possible features
        for day in sorted(feature_day,reverse = True):
            #check for the column in the list
            feature_use = 'Number day from ' + str(day) + ' case'
            #check the 0-dimension of the array (similiar to length of a dataframe)
            idx = X_train[X_train[feature_use] == 0].shape[0]     
            #if there are more than 20 values for a column, the loop will be interruped
            if (X_train[X_train[feature_use] > 0].shape[0] >= 20):
                break
                
        #[TRAIN] - cuts the value of idx from the top of the dataframe; selects the input column (e.g. Number day from 1000 case); brings it into a horizontal array
        adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)
        #[TRAIN] - get the respective confirmed cases
        adjusted_y_train_confirmed = y_train_confirmed[idx:]
        #[TRAIN] - get the respective fatalities
        adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)
        
        #[TEST] - selects for the extracted feature column and get the length (0)
        idx = X_pred[X_pred[feature_use] == 0].shape[0]
        
        #[TEST] - creates a clean array also for the prediction
        adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)
        
        #[TEST] - gets the extract from the test dataset for a specific country/ region
        pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        
        #latest date from the trainings data set and earliest date from the test data set
        max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()
        min_test_date = pred_data['Date'].min()
        
        if len(adjusted_y_train_confirmed) < 1:
            adjusted_y_train_confirmed = np.zeros(3)
        else:
            if len(adjusted_y_train_confirmed) < 2:
                adjusted_y_train_confirmed = np.append(adjusted_y_train_confirmed,adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1],adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1])
            else:
                if len(adjusted_y_train_confirmed) < 3:
                    adjusted_y_train_confirmed = np.append(adjusted_y_train_confirmed,adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1])
                else:
                    pass
        
        #[CONFIRMED CASES] - prediction and modelling
        model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 
                        measurement_error=True).fit(disp=False)
        y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values
        y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)

        if len(adjusted_y_train_fatalities) < 1:
            adjusted_y_train_fatalities = np.zeros(3)
        else:
            if len(adjusted_y_train_fatalities) < 2:
                adjusted_y_train_fatalities = np.append(adjusted_y_train_fatalities,adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1],adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1])
            else:
                if len(adjusted_y_train_fatalities) < 3:
                    adjusted_y_train_fatalities = np.append(adjusted_y_train_fatalities,adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1])
                else:
                    pass
                
        #[FATALITIES] - prediction and modelling
        model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 
                        measurement_error=True).fit(disp=False)
        y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values
        y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)
        pred_data['ConfirmedCases_hat'] =  y_hat_confirmed
        pred_data['Fatalities_hat'] = y_hat_fatalities
        pred_data_all = pred_data_all.append(pred_data)


# In[39]:


df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')
df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0
df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0
df_val_3 = df_val.copy()
submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('submission.csv', index=False)


# In[40]:


#country = 'France'
#province = 'Saint Pierre and Miquelon'
#
#
#pred_data_all = pd.DataFrame()
#print(country)
##create dataframe for a specific country
#df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
#df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
##create features -> number of cases on a specific date
#X_train = CreateInput(df_train)
##last 12 confirmed cases in train data set
#y_train_confirmed = df_train['ConfirmedCases'].ravel()
##last 12 confirmed fatalities in train data set
#y_train_fatalities = df_train['Fatalities'].ravel()
##create features in test dataset-> number of cases on a specific date
#X_pred = CreateInput(df_test)
#
##creates reversed list of the possible features
#for day in sorted(feature_day,reverse = True):
#    #check for the column in the list
#    feature_use = 'Number day from ' + str(day) + ' case'
#    #check the 0-dimension of the array (similiar to length of a dataframe)
#    idx = X_train[X_train[feature_use] == 0].shape[0]
#    #print(idx)
#    #if there are more than 20 values for a column, the loop will be interruped
#    if (X_train[X_train[feature_use] > 0].shape[0] >= 20):
#        break
#        
##[TRAIN] - cuts the value of idx from the top of the dataframe; selects the input column (e.g. Number day from 1000 case); brings it into a horizontal array
#adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)
##[TRAIN] - get the respective confirmed cases
#adjusted_y_train_confirmed = y_train_confirmed[idx:]
##[TRAIN] - get the respective fatalities
#adjusted_y_train_fatalities = y_train_fatalities[idx:]
#
##[TEST] - selects for the extracted feature column and get the length (0)
#idx = X_pred[X_pred[feature_use] == 0].shape[0]
##[TEST] - creates a clean array also for the prediction
#adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)
#
##[TEST] - gets the extract from the test dataset for a specific country/ region
#pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
#
##latest date from the trainings data set
#max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()
##earliest date from the test data set
#min_test_date = pred_data['Date'].min()
#
#if len(adjusted_y_train_confirmed) < 1:
#    adjusted_y_train_confirmed = np.zeros(3)
#else:
#    if len(adjusted_y_train_confirmed) < 2:
#        adjusted_y_train_confirmed = np.append(adjusted_y_train_confirmed,adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1],adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1])
#    else:
#        if len(adjusted_y_train_confirmed) < 3:
#            adjusted_y_train_confirmed = np.append(adjusted_y_train_confirmed,adjusted_y_train_confirmed[len(adjusted_y_train_confirmed)-1])
#        else:
#            pass
#
##[CONFIRMED CASES] - prediction and modelling
#model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 
#                measurement_error=True).fit(disp=False)
#y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
#y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values
#y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)
#
#if len(adjusted_y_train_fatalities) < 1:
#    adjusted_y_train_fatalities = np.zeros(3)
#else:
#    if len(adjusted_y_train_fatalities) < 2:
#        adjusted_y_train_fatalities = np.append(adjusted_y_train_fatalities,adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1],adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1])
#    else:
#        if len(adjusted_y_train_fatalities) < 3:
#            adjusted_y_train_fatalities = np.append(adjusted_y_train_fatalities,adjusted_y_train_fatalities[len(adjusted_y_train_fatalities)-1])
#        else:
#            pass
#
##[FATALITIES] - prediction and modelling
#model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 
#                measurement_error=True).fit(disp=False)
#
#y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
#y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values
#y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)
#pred_data['ConfirmedCases_hat'] =  y_hat_confirmed
#pred_data['Fatalities_hat'] = y_hat_fatalities
#pred_data_all = pred_data_all.append(pred_data)


# In[41]:


def crosscheck_sarima(country):
    crosscheck = train_df[(train_df['Country'] == country) & (train_df['Date'] >= '2020-04-02')].reset_index()
    arima = pred_data_all[(pred_data_all['Country_Region'] == country)].reset_index()
    arima['ConfirmedCases_In'] = arima['ConfirmedCases_hat']
    arima['Fatalities_In'] = arima['Fatalities_hat']
    arima['CC_Crosscheck'] = crosscheck['ConfirmedCases']
    arima['Fat_Crosscheck'] = crosscheck['Fatalities']
    arima['Day_num'] = arima['Date']
    return arima


# In[42]:


def crosscheck_sarima_cs(country,state):
    crosscheck = train_df[(train_df['Country'] == country) & (train_df['State'] == state) & (train_df['Date'] >= '2020-04-02')].reset_index()
    arima = pred_data_all[(pred_data_all['Country_Region'] == country) & (pred_data_all['Province_State'] == state)].reset_index()
    arima['ConfirmedCases_In'] = arima['ConfirmedCases_hat']
    arima['Fatalities_In'] = arima['Fatalities_hat']
    arima['CC_Crosscheck'] = crosscheck['ConfirmedCases']
    arima['Fat_Crosscheck'] = crosscheck['Fatalities']
    arima['Day_num'] = arima['Date']
    return arima


# In[43]:


def check_plot_2(df_check2,test_con,state=''):

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['ConfirmedCases_In'], name='Confirmed Cases - Prediction')
    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['CC_Crosscheck'], name='Confirmed Cases - Official')

    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['Fatalities_In'], name='Fatalities - Prediction', secondary_y=True)
    fig.add_scatter(x=df_check2['Day_num'], y=df_check2['Fat_Crosscheck'], name='Fatalities - Official', secondary_y=True)

    if state=='':
        fig.update_layout(title='ARIMA '+ test_con + ' Forecast',
                           xaxis_title='Number of Days since 2020-01-22',
                           yaxis_title='Confirmed Cases')
    else:
        fig.update_layout(title='Arima_'+ test_con + ', ' + state + ' Forecast',
                       xaxis_title='Number of Days since 2020-01-22',
                       yaxis_title='Confirmed Cases')    

    fig.update_yaxes(title_text="Confirmed Cases", secondary_y=False)
    fig.update_yaxes(title_text="Fatalities", secondary_y=True)
    
    #if state=='':
    #    py.plot(fig, filename = 'SARIMA_' + test_con, auto_open=True)
    #else:
    #    py.plot(fig, filename = 'SARIMA_' + test_con + '_' + state, auto_open=True)
        

    return fig.show()


# In[44]:


#pred_data_all[(pred_data_all['Country_Region'] == country_dict[country]) & (pred_data_all['Date'] <= '2020-04-02')].reset_index()


# In[45]:


#import statsmodels.api as sm
#
#decomposition = sm.tsa.seasonal_decompose(train_df['value'], model='additive', 
#                            extrapolate_trend='freq') #additive or multiplicative is data specific
#fig = decomposition.plot()
#plt.show()


# The Confirmed Cases are on the left Y-axis and the Fatalities on the right Y-axis.

# In[46]:


test_con = 'Germany'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[47]:


test_con = 'Spain'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[48]:


test_con = 'Italy'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[49]:


test_con = 'Algeria'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[50]:


test_con = 'Andorra'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[51]:


test_con = 'Iran'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[52]:


test_con = 'Russia'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[53]:


test_con = 'US'
test_st = 'New York'
df_check2 = crosscheck_sarima_cs(test_con,test_st)
check_plot_2(df_check2,test_con,test_st)


# In[54]:


test_con = 'US'
test_st = 'Washington'
df_check2 = crosscheck_sarima_cs(test_con,test_st)
check_plot_2(df_check2,test_con,test_st)


# In[55]:


test_con = 'China'
test_st = 'Shanghai'
df_check2 = crosscheck_sarima_cs(test_con,test_st)
check_plot_2(df_check2,test_con,test_st)


# In[56]:


test_con = 'China'
test_st = 'Beijing'
df_check2 = crosscheck_sarima_cs(test_con,test_st)
check_plot_2(df_check2,test_con,test_st)


# In[57]:


test_con = 'Korea, South'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[58]:


test_con = 'South Africa'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[59]:


test_con = 'Ghana'
df_check2 = crosscheck_sarima(test_con)
check_plot_2(df_check2,test_con)


# In[60]:


train_df['Country'].unique()


# ## 4.4 ARIMA — Conclusion
# The forecasts for the individual countries look much more plausible over time. Data points for countries with more confirmed cases and a longer history such as Germany can be extrapolated well. If the figures are reported erratically, as in Shanghai, our algorithm will not be able to make good predictions.
# The implementation process compared to linear regression was also much easier. The first test submission to Kaggle had an RMLSE of about 0.4. The competition for week 4 ends today and the evaluation phase begins. I will add the final results.

# # 5. Summary
# First we looked at the developments around Covid-19, cleaned up the data and prepared it accordingly. We then tried out two different methods for predicting time series: 1) linear regression and 2) ARIMA. The results of the linear regression were very volatile and the predictions did not seem particularly accurate. With ARIMA, we achieved significantly better and more plausible results. A final evaluation of the results is possible on 15.05.2020. However, the submission for the test period achieved a good score.
# I hope that despite the seriousness of the topic you were able to take something with you. Thanks for reading! Stay at home and healthy!
