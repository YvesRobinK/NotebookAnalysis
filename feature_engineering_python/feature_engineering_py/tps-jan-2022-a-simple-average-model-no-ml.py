#!/usr/bin/env python
# coding: utf-8

# ### Tabular Playground Series - Jan 2022: A simple average model
# In the [Tabular Playground Series - Jan 2022](https://www.kaggle.com/c/tabular-playground-series-jan-2022) competition we are tasked with predicting the sales of three different products (the `Kaggle Mug`, the `Kaggle Hat` and the `Kaggle Sticker`, all highly sought-after products) in two different stores (`KaggleMart` and `KaggleRama`) in three different countries (`Finland`, `Sweden` and `Norway`) for the year 2019. We are provided with training data for the years 2015 to 2018.

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})
from datetime import datetime
from datetime import timedelta


# ### SMAPE metric
# This competition is evaluated using the [symmetric mean absolute percentage error (SMAPE)](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error). Such an evaluation metric was used in the kaggle [Web Traffic Time Series Forecasting competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting/), and [CPMP](https://www.kaggle.com/cpmpml) kindly provided us with the python code to evaluate this metric in the topic ["SMAPE_Python"](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/36414):

# In[2]:


# https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/36414

def SMAPE(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.round(np.mean(diff),5)


# In the excellent notebook ["SMAPE Weirdness"](https://www.kaggle.com/cpmpml/smape-weirdness), again by CPMP, we see that in a similar fashion to the better known root mean squared logarithmic error (RMSLE) metric the SMAPE is asymmetric; penalizing much more the underestimated predictions than the overestimated predictions (figure adapted from the aforementioned notebook, using 3 as the ground truth value):

# In[3]:


# true value
y_true = np.array(3)
# predictions
y_pred = np.ones(1)
x = np.linspace(0,10,100)
result = [SMAPE(y_true, i * y_pred) for i in x]
# plot
plt.rcParams["figure.figsize"] = (5, 3)
plt.plot(x, result)
plt.xlabel('prediction')
plt.ylabel('SMAPE')
plt.show()


# ### Train-Test split and feature engineering
# We shall read in the training data, and create a new training dataset out of the year 2017 (*i.e.* use a 'look-back' window of 1 year), and a test dataset out of the 2018 data

# In[4]:


train_all_years = pd.read_csv("../input/tabular-playground-series-jan-2022/train.csv",parse_dates=['date'],index_col=["row_id"])

train = train_all_years[train_all_years.date.between('2017-01-01', '2017-12-31')].copy()
test  = train_all_years[train_all_years.date.between('2018-01-01', '2018-12-31')].copy()


# we shall now create some new features, namely the day of the week and the month

# In[5]:


train['day_of_the_week'] = train['date'].dt.day_name()
test['day_of_the_week']  = test['date'].dt.day_name()

train['month'] = train['date'].dt.month_name()
test['month']  = test['date'].dt.month_name()


# ### Model 1: Simple mean
# This is a simple model, averaging over the `country`, `store` and `product`

# In[6]:


# calculate the mean values
train_means        = train.groupby(['country','store','product'])['num_sold'].mean().to_dict()
test["prediction"] = test.set_index(['country','store','product']).index.map(train_means.get)


# now calculate the SMAPE score for this model

# In[7]:


SMAPE(test["num_sold"], test["prediction"])


# ### Model 2: Day of the week model

# In[8]:


train_means        = train.groupby(['country','store','product','day_of_the_week'])['num_sold'].mean().to_dict()
test["prediction"] = test.set_index(['country','store','product','day_of_the_week']).index.map(train_means.get)


# In[9]:


SMAPE(test["num_sold"], test["prediction"])


# Let us take a look at our predictions for `KaggleMart` stores in `Norway`, our predictions are the thick solid lines, and the test data are the thinner dashed lines

# In[10]:


country = "Norway"
store   = "KaggleMart"
one_country_and_store = test.query("country == @country & store == @store").copy()

fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="num_sold", hue="product", linewidth = 2, linestyle='--')
sns.lineplot(data=one_country_and_store,  x="date", y="prediction", hue="product", linewidth = 3.5)
plt.legend([],[], frameon=False);


# ### Model 3: Month and day of the week model

# In[11]:


train_means  = train.groupby(['country','store','product','day_of_the_week','month'])['num_sold'].mean().to_dict()
test["prediction"] = test.set_index(['country','store','product','day_of_the_week','month']).index.map(train_means.get)


# In[12]:


SMAPE(test["num_sold"], test["prediction"])


# We now have an even better score. Let us take a look

# In[13]:


one_country_and_store = test.query("country == @country & store == @store").copy()

fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="num_sold", hue="product", linewidth = 2, linestyle='--')
sns.lineplot(data=one_country_and_store,  x="date", y="prediction", hue="product", linewidth = 3.5)
plt.legend([],[], frameon=False);


# We can see that on the whole things are better, but there is plenty of room for improvement, in particular for December, but also around [Easter time](https://en.wikipedia.org/wiki/Easter#Date). Here is a plot of the difference between our predictions and the actual values

# In[14]:


one_country_and_store["residual"] =  one_country_and_store["num_sold"] - one_country_and_store["prediction"] 
fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="residual", hue="product", linewidth = 1.5)
plt.ylim(-200,1000)
plt.legend([],[], frameon=False);


# ### Model 4: Daily model of Christmas
# Let us focus only on the month of December. Visual inspection indicates that the month of December behaves as most other months up until the week of Christmas (25th-31st). In view of this we shall treat December as two different parts; a day-of-the-week model up to, and including, December 24, and the we shall look at each day individually for the last week.

# In[15]:


# make a new 'day' feature
train['day'] = train['date'].dt.day
test['day']  = test['date'].dt.day


# In[16]:


# December model part 1: day_of_the_week model for pre-Christmas
train_December = train.query("month == 'December' & day < 25").copy()
test_December  = test.query("month == 'December'& day < 25").copy()
train_means        = train_December.groupby(['country','store','product','day_of_the_week'])['num_sold'].mean().to_dict()
test_December["prediction"]=test_December.set_index(['country','store','product','day_of_the_week']).index.map(train_means.get)
test.update(test_December)

# December model part 2: a daily model for the last week
train_December = train.query("month == 'December' & day >= 25").copy()
test_December  = test.query("month == 'December'& day >= 25").copy()
train_means                 = train_December.groupby(['country','store','product','day'])['num_sold'].mean().to_dict()
test_December["prediction"] = test_December.set_index(['country','store','product','day']).index.map(train_means.get)
test.update(test_December)


# let us take a look

# In[17]:


test_December  = test.query("month == 'December'").copy()
one_country_and_store = test_December.query("country == @country & store == @store").copy()

fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="num_sold", hue="product", linewidth = 2, linestyle='--')
sns.lineplot(data=one_country_and_store,  x="date", y="prediction", hue="product", linewidth = 3.5)
plt.legend([],[], frameon=False);


# Let us update the **Month and day of the week model** with the **Daily model of December** and calculate the new score

# In[18]:


SMAPE(test["num_sold"], test["prediction"])


# now take a look at the combined model

# In[19]:


one_country_and_store = test.query("country == @country & store == @store").copy()

fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="num_sold", hue="product", linewidth = 2, linestyle='--')
sns.lineplot(data=one_country_and_store,  x="date", y="prediction", hue="product", linewidth = 3.5)
plt.legend([],[], frameon=False);


# ### Specific day model: The 1st of January
# In the above plot we can see that we even may wish to focus on a specific day, for example on the 1st of January we can see a peak where people who have been saving up all year to buy their kaggle merchandise make the most of the January sales:

# In[20]:


train_1_January = train.query("month == 'January' & day == 1").copy()
test_1_January  = test.query("month == 'January' & day == 1").copy()
train_means     = train_1_January.groupby(['country','store','product'])['num_sold'].mean().to_dict()
test_1_January["prediction"] = test_1_January.set_index(['country','store','product']).index.map(train_means.get)
test.update(test_1_January)
# calculate the score
SMAPE(test["num_sold"], test["prediction"])


# ...and for those who were way too hungover to make it to their local Kaggle stores on New Year's Day, we shall also look at the sales on the 2nd of January too:

# In[21]:


train_2_January = train.query("month == 'January' & day == 2").copy()
test_2_January  = test.query("month == 'January' & day == 2").copy()
train_means     = train_2_January.groupby(['country','store','product'])['num_sold'].mean().to_dict()
test_2_January["prediction"] = test_2_January.set_index(['country','store','product']).index.map(train_means.get)
test.update(test_2_January)
# calculate the score
SMAPE(test["num_sold"], test["prediction"])


# ### Moveable holidays: Easter
# A visual inspection of the training data indicates that there are substantially more sales during Easter Sunday all the way up to Første pinsedag 🇳🇴, Helluntaipäivä 🇫🇮, and Pingstdagen 🇸🇪, the first Saturday after Pentecost (Whit Sunday), [especially in Norway](https://www.lifeinnorway.net/pinse-holiday/). We shall make use of [`dateutil.easter`](https://dateutil.readthedocs.io/en/stable/easter.html) to provide the date of Easter Sunday onwards for each year of interest:

# In[22]:


from dateutil.easter import easter

def get_Easter_dates(years):
    Easter_dates = []
    Easter_index = []
    Easter_date_index = []
    for year in years:
        Easter_dates.append(easter(year))
        Easter_index.append(0)
        Easter_date_index.append([easter(year),0])
        # also calculate the dates of the days following Easter
        for day in range(1, 56):
            Easter_dates.append(easter(year)+timedelta(days=day))       
            Easter_index.append(day)
            Easter_date_index.append([easter(year)+timedelta(days=day),day])  
    return Easter_dates,Easter_index,Easter_date_index

years = [2015,2016,2017,2018,2019]
Easter_dates,Easter_index,Easter_date_index = get_Easter_dates(years)


# now calculate the averages for these dates w.r.t. Easter Sunday, insert them into our model, and see our new score

# In[23]:


train_Easter = train.query("date == @Easter_dates").copy()
test_Easter  = test.query("date == @Easter_dates").copy()

# create the "day_of_Easter" feature
mapping = dict(Easter_date_index)
train_Easter["day_of_Easter"] = train_Easter["date"].map(mapping)
test_Easter["day_of_Easter"]  = test_Easter["date"].map(mapping)

train_means   = train_Easter.groupby(['country','store','product','day_of_Easter'])['num_sold'].mean().to_dict()
test_Easter["prediction"] = test_Easter.set_index(['country','store','product','day_of_Easter']).index.map(train_means.get)

test.update(test_Easter)
SMAPE(test["num_sold"], test["prediction"])


# ### Scaling by year-over-year growth
# So far we have assumed that overall the 2018 test data will behave just as the 2017 training data. However, that may not be the case. For example, here is a table and plot of the total number of units sold annually for the whole training set provided:

# In[24]:


train_all_years['year'] = train_all_years['date'].dt.year
pivot_table = pd.pivot_table(train_all_years, index=['year'], values=['num_sold'], aggfunc=sum)
pivot_table


# In[25]:


plt.rcParams["figure.figsize"] = (5, 2)
pivot_table.plot(kind='bar').legend(loc='center left',bbox_to_anchor=(1.0, 0.5));


# we can see that the overall number of units sold in 2018 was substantially greater than those sold in 2017. Let us look at the effect on our score of scaling by this growth. Note that this is a form of [data leakage](https://en.wikipedia.org/wiki/Leakage_%28machine_learning%29) as we are now using information from the test set (*i.e.* information from the future is being leaked back into our training data) in our model. If one actually wished to make use of such a scaling one should first construct a model of the yearly growth *only* from the training data.

# In[26]:


units_sold_2017 = train['num_sold'].sum()
units_sold_2018 = test['num_sold'].sum()
yearly_growth   = units_sold_2018 / units_sold_2017
# scale our predictions
test["prediction"] = test["prediction"] * yearly_growth
SMAPE(test["num_sold"], test["prediction"])


# We can see that a good model of the year-over-year growth does indeed have the potential to significantly improve our score. When making such a model of the yearly growth one could make use of additional economic data, for example the [gross domestic product for each of the countries Finland, Norway, and Sweden](https://www.kaggle.com/carlmcbrideellis/gdp-of-finland-norway-and-sweden-2015-2019).
# 
# Finally let us now look again at the difference between our predictions and the actual values, on the same scale as the plot above:

# In[27]:


one_country_and_store = test.query("country == @country & store == @store").copy()
one_country_and_store["residual"] =  one_country_and_store["num_sold"] - one_country_and_store["prediction"] 

fig, ax = plt.subplots(figsize=(20, 7))
sns.lineplot(data=one_country_and_store, x="date", y="residual", hue="product", linewidth = 1.5)
plt.ylim(-200,1000)
plt.legend([],[], frameon=False);


# ### Now we shall predict the 2019 data
# Let us now make a prediction, using the combined **(Month-Day of the Week)+(December)+(Moveable holidays)+(Specific Day)** model. If we use all of the training data that we have available (4 years) we obtain a Public Leaderboard score of 7.16. Here, in a rather *ad hoc* fashion, we shall use a look-back window of the last three years:

# In[28]:


# select the last three years of training data
train = train_all_years.query("date >= '2016-01-01' ").copy()
test  = pd.read_csv("../input/tabular-playground-series-jan-2022/test.csv",parse_dates=['date'])

# create the new features
train['day_of_the_week'] = train['date'].dt.day_name()
test['day_of_the_week']  = test['date'].dt.day_name()

train['month'] = train['date'].dt.month_name()
test['month']  = test['date'].dt.month_name()

train['day'] = train['date'].dt.day
test['day']  = test['date'].dt.day

# Overall model
train_means      = train.groupby(['country','store','product','day_of_the_week','month'])['num_sold'].mean().to_dict()
test["num_sold"] = test.set_index(['country','store','product','day_of_the_week','month']).index.map(train_means.get)

# December model part 1: day_of_the_week model for pre-Christmas
train_December = train.query("month == 'December' & day < 25").copy()
test_December  = test.query("month == 'December'  & day < 25").copy()
train_means    = train_December.groupby(['country','store','product','day_of_the_week'])['num_sold'].mean().to_dict()
test_December["num_sold"] = test_December.set_index(['country','store','product','day_of_the_week']).index.map(train_means.get)
test.update(test_December)

# December model part 2: a daily model for the last week
train_December = train.query("month == 'December' & day >= 25").copy()
test_December  = test.query("month == 'December'  & day >= 25").copy()
train_means                 = train_December.groupby(['country','store','product','day'])['num_sold'].mean().to_dict()
test_December["num_sold"] = test_December.set_index(['country','store','product','day']).index.map(train_means.get)
test.update(test_December)

# Easter Sunday and the following days up to the saturday after Whit Sunday
train_Easter = train.query("date == @Easter_dates").copy()
test_Easter  = test.query("date == @Easter_dates").copy()
train_Easter["day_of_Easter"] = train_Easter["date"].map(mapping)
test_Easter["day_of_Easter"]  = test_Easter["date"].map(mapping)
train_means             = train_Easter.groupby(['country','store','product','day_of_Easter'])['num_sold'].mean().to_dict()
test_Easter["num_sold"] = test_Easter.set_index(['country','store','product','day_of_Easter']).index.map(train_means.get)
test.update(test_Easter)

# Specific day model: January the 1st
train_1_January = train.query("month == 'January' & day == 1").copy()
test_1_January  = test.query("month == 'January' & day == 1").copy()
train_means                = train_1_January.groupby(['country','store','product'])['num_sold'].mean().to_dict()
test_1_January["num_sold"] = test_1_January.set_index(['country','store','product']).index.map(train_means.get)
test.update(test_1_January)

# Specific day model: January the 2nd
train_2_January = train.query("month == 'January' & day == 2").copy()
test_2_January  = test.query("month == 'January' & day == 2").copy()
train_means                = train_2_January.groupby(['country','store','product'])['num_sold'].mean().to_dict()
test_2_January["num_sold"] = test_2_January.set_index(['country','store','product']).index.map(train_means.get)
test.update(test_2_January)


# Let us take a look at our prediction for 2019

# In[29]:


one_country_and_store_train = train.query("country == @country & store == @store")
one_country_and_store_test  = test.query("country == @country & store == @store")

fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(data=one_country_and_store_train, x="date", y="num_sold", hue="product", linewidth = 1.5)
sns.lineplot(data=one_country_and_store_test,  x="date", y="num_sold", hue="product", linewidth = 1.5)
plt.text(datetime.strptime("2017-03-01", '%Y-%m-%d'), 1300, "training data")
plt.text(datetime.strptime("2019-04-01", '%Y-%m-%d'), 1300, "2019 predictions")
plt.legend([],[], frameon=False);


# ### 2019 GDP Scaling
# If we were situated on the 31st of December 2018, and we were asked to predict the `num_sold` for the year 2019 then the following operation **would not be possible**, indeed this would be a prime example of [look-ahead bias](https://www.investopedia.com/terms/l/lookaheadbias.asp). However, as this is a kaggle competition situated in 2022 we shall proceed. We have seen above that scaling by year-over-year growth can significantly improve our score. We obviously do not know the sales data for 2019, however we could use the 2019 GDP data *per capita* as an ersatz indicator. First we shall load in the data from the dataset [GDP per capita: Finland, Norway, Sweden (2015-19)](https://www.kaggle.com/samuelcortinhas/gdp-per-capita-finland-norway-sweden-201519) created by [Samuel Cortinhas](https://www.kaggle.com/samuelcortinhas):

# In[30]:


GDP_data = pd.read_csv("../input/gdp-per-capita-finland-norway-sweden-201519/GDP_per_capita_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")
GDP_data.style.bar(subset=['Finland','Norway','Sweden'], align='left', vmin=0,  color='#A0D6F0')


# Now for each country we shall calculate a scaling factor for 2019 with respect to the mean GDP of the years 2016, 2017, and 2018, then apply that scaling factor to our predictions for 2019, for each of the countries:

# In[31]:


scale_Finland = GDP_data.iloc[4,0] / GDP_data.iloc[1:4,0].mean()
scale_Norway  = GDP_data.iloc[4,1] / GDP_data.iloc[1:4,1].mean()
scale_Sweden  = GDP_data.iloc[4,2] / GDP_data.iloc[1:4,2].mean()

mask    = (test['country']=='Finland')
Finland = test[mask]
test.loc[mask,'num_sold'] = Finland["num_sold"] * scale_Finland

mask    = (test['country']=='Norway')
Norway  = test[mask]
test.loc[mask,'num_sold'] = Norway["num_sold"] * scale_Norway

mask    = (test['country']=='Sweden')
Sweden  = test[mask]
test.loc[mask,'num_sold'] = Sweden["num_sold"] * scale_Sweden


# ### Now create a `submission.csv` file

# In[32]:


submission = pd.DataFrame({'row_id': test.row_id, 'num_sold': test.num_sold})
submission['row_id'] = submission['row_id'].astype('int32')
submission.to_csv('submission.csv', index=False)


# ### <center style="background-color:Gainsboro; width:60%;">Interesting reading</center>
# * [Rob J. Hyndman and George Athanasopoulos "*Forecasting: Principles and Practice*", (3rd Edition)](https://otexts.com/fpp3/)
# * [Fotios Petropoulos, *et al. "Forecasting: Theory and Practice*", arXiv:2012.03854 (2020)](https://arxiv.org/pdf/2012.03854.pdf)
