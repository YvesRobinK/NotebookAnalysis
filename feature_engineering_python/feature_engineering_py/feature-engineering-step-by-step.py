#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering for Customer Revenue Prediction
# 
# The purpose of this Kernel is to ingest the raw data, extract features, and engineer new ones to provide community with a ready-to-model flat dataset. The kernel is split into three part:
# 1. [Data Ingestion](#Data Ingestion)
# 2. [Feature Extraction](#Feature Extraction)
# 3. Feature Engineering - WIP
# <a id='Data Ingestion'></a>
# ### 1. Data Ingestion
# First, I import the libraries required to extraction.

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize


# Next, I use the function defined by [Julian Peller](https://www.kaggle.com/julian3833) to read ingest the raw data. Note that the function will automatically process nested JSON data elements and populate each into a separate column. 

# In[ ]:


# Read in the raw traininig dataset
# Credit: https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(csv_path='../train.csv'):

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},
                     parse_dates=['date']) # Note: added this line to Julian's code to parse dates on ingestion. It slows the process a bit
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df

raw_train = load_df("../input/train.csv")


# Similar to other Kernels, I drop the columns that are either all missing values or have just one unique value filled in (looking at you _trafficSource.campaign_ :) )

# In[ ]:


# Drop columns with just one value or all unknown
cols_to_drop = [col for col in raw_train.columns if raw_train[col].nunique() == 1]
raw_train.drop(columns = cols_to_drop, inplace=True)

# Drop campaign colum as it only has one non-null value
raw_train.drop(['trafficSource.campaign'], axis=1, inplace=True)


# To make column names nicer, I'll rename a few to avoid the nesting dot notation. To do this I will split each column name on the dot (.) character and select the last string in the list to be the new column name.

# In[ ]:


# Rename long column names to be more concise
raw_train.rename(columns={col_name: col_name.split('.')[-1] for col_name in raw_train.columns}, inplace = True)


# At the end, I am left with 30 columns, in a dataset of 903,653 rows. One of the columns is the dependent variable we are interested in predicting the log of accross all user visits - _transactionRevenue_.

# In[ ]:


print('Number of columns: {:}\nNumber of rows: {:}'.format(raw_train.shape[1], raw_train.shape[0]))

# Fill transactionRevenue with zeroes and convert its type to numeric
raw_train['transactionRevenue'].fillna(0, inplace=True)
raw_train['transactionRevenue'] = pd.to_numeric(raw_train['transactionRevenue'])


# <a id='Feature Extraction'></a>
# ### 2. Feature Extraction & Cleaning
# In this section I will constuct a dataset with the grain of sessionId, which in turn is just a concatenation of _fullVisitorId_ and _visitId_. The following table gives the names of the features I extract, the type of extraction (numeric or converting to dummy variables), and the source column. You can learn more about the meaning of each column [here](https://support.google.com/analytics/answer/3437719?hl=en).
# 
# 
# | Feature | Type | Description | Source Column |
# |:------|:------|:---|:---|
# |Month  | Numeric | Month of the visit. Values 1 to 12. | _date_ |
# |Week   | Numeric | Week of the year of the visit. Values 1 to 52. | _date_ |
# |Weekday| Numeric | Day of the week of the visit. Values 1 to 7. | _date_|
# |Hour| Numeric | Hour of day of the start of the visit. Values 0 to 24. | _visitStartTime_|
# |Channel_X| Dummy | Three dummy columns indicating the channel the visit came in from. Values 0 or 1. | _channelGrouping_ |
# |visitNumber| Numeric | The count of the current visit for this user. Values 1 to n. | _vistiNumber_ |
# |Browser_X| Dummy | Dummy columns for each of the major browsers and additional column for "other". Values 0 or 1. | _browser_ |
# |Device| Dummy | Three dummy columns indicating type of device. Values 'desktop', 'mobile', 'tablet'. | _deviceCategory_ |
# |OS| Dummy | Dummy columns for each major operating system and additional column for "other". Values 0 or 1. |_operatingSystem_|
# |SubContinent| Dummy | Dummy columns for each subcontient, combinig some of the smaller ones. Values 0 or 1. |_subContinent_|
# |pageViews| Numeric | Number of pageviews that are generated by user thus far. Every row in the dataset adds 1 to user count. Values 1 to n. |_pageviews_|
# |hits| Numeric| Superset of user activity count that also includes _pageviews_. Captures district interactions between user and webpage. Values 1 to n. |_hits_|
# |Medium| Dummy | Dummy columns to indicate what type of marketing brought the user to the site. Values 0 or 1.|_medium_|
# 
# 
# Note that the following columns I did not include in the feature extraction, and the reasons are as follows:
# * _isMobile_ is already encoded as either mobile or tablet, so there is little need for a separate column.
# * _sessionId_ is nothing more than a concatenation of _fullVisitorId_ and _visitId_
# * _Year_ - there is only 12 months worth of data in both train and test datasets, so Month and Week are plenty to capture the timeframe.
# * _city_ has almost 650 unique values, which is too many to encode as dummy variables. In the Feature engineering part I'll see what can be done with this information. 
# * _continent_ is a superset of _subContinent_ which I'm using as the base location variable. So, _continent_ is redundant.
# * _country_ likely provides a good amount of information about the demographic of the user, but I'll tackle than in feature engineering section.
# * _metro_ is rarely filled in and is highly correlated with _city_, so it's a good candidate to drop entirely.
# * _networkDomain_ could be interesting from feature engineering perspective, but I'll drop it for now.
# * _region_ can be through of as a "state" or "province", but I will again leave that for feature engineering.
# * _adNetwork_ only appears to be filled in when _adContent_ is filled in. Since the vast majority of _adNetwork_ values are "Google Search", this becomes a redundant feature.
# * _gclId_ seems to be an internal ID used by google for tracking purposes. I don't think it will be of any use for modeling.
# * _page_ onlye appears to be filled in when _adContent_ is filled in. Vast majority of values are 1 (i.e. ad appeared on page 1). I will drop the column for now.
# * _slot_ means either an ad on top of the screen or RHS (right hand side) ad. I'll drop it for now.
# * _keyword_ may be useful in the feature engineering, but in its raw form it's too cumbersome to include.
# * _referralPath_ requires some digging and feature engineering, but it will probably be a useful column to explore.
# * _source_ same as above - requires some digging and feature engineering.
# 
# 
# Phew, that took a while to writeup. Hopefully, it'll save you some work and you would have learned something in the process. Let's get "extracting"! I will use the existing dataframe as the base for pulling out the features listed in the table above. I will not drop the columns that would not be used for model training (e.g. date), but I'll add comented out code at the bottom of this Kernel to remove them if you wish.
# 
# Start with the date fields that are relatively straightforward.

# In[ ]:


# Get the month value from date
raw_train['Month'] = raw_train['date'].dt.month

# Get the week value from date
raw_train['Week'] = raw_train['date'].dt.week

# Get the weekday value from date
raw_train['Weekday'] = raw_train['date'].dt.weekday

# Get the hour value from visitStartTime
raw_train['Hour'] = pd.to_datetime(raw_train['visitStartTime'], unit='s').dt.hour


# Next, let's dummify the _challenGrouping_ and _device_ variables. This is a grouping of the sources of web traffic that lead to the GStore and the device that the user is on. The dummifying operation is pretty straightforward here.

# In[ ]:


# Dummify challenGrouping into 8 separate binary columns
raw_train = pd.get_dummies(raw_train, columns = ['channelGrouping', 'deviceCategory'])


# Who would have though that there are so many different browsers out there. I mean, have you ever heard of [Puffin](https://www.puffinbrowser.com/) or [Lunascape](https://www.lunascape.tv/)? For the purposes of training a model, I believe it will suffice to lump all of the little known browsers into one bucket called "Other" and call it a day. The major browsers I leave in their own dummy columns are Chrome, Safari, Firefox, IE, Edge, Android, Safari, Opera, UC Browser (marker for Asian market), Coc coc (marker for Vietnameese market).

# In[ ]:


# Group all little known broswers into "Other" bucket
raw_train.loc[~raw_train['browser'].isin(['Chrome', 'Safari', 'Firefox', 'Internet Explorer', 
                                          'Edge', 'Android Webview', 'Safari (in-app)', 'Opera Mini', 
                                          'Opera', 'UC Browser', 'Coc Coc']), ['browser']] = 'Other'

# Dummify browser into separage binary columns
raw_train = pd.get_dummies(raw_train, columns = ['browser'])


# For operating system, again I will take just the top 7 values and set the rest to "Other".

# In[ ]:


# Group all less common operating systems into "Other" bucket, including where it's (not set)
raw_train.loc[~raw_train['operatingSystem'].isin(['Windows', 'Macintosh', 'Android', 'iOS', 
                                                   'Linux', 'Chrome OS', 'Windows Phone']), ['operatingSystem']] = 'Other'

# Dummify operatingSytem into separate binary columns
raw_train = pd.get_dummies(raw_train, columns = ['operatingSystem'])


# Similarly, I'll combine Polynesia, Micronesia, Melanesia into "Other", cutting down on a few unnecessary dummy columns.

# In[ ]:


# Group all less populated parts of the world into "Other" bucket, including where it's (not set)
raw_train.loc[raw_train['subContinent'].isin(['Polynesia', 'Micronesian Region', 
                                              'Melanesia', '(not set)']), ['subContinent']] = 'Other'

# Dummify subContinent into separate binary columns
raw_train = pd.get_dummies(raw_train, columns = ['subContinent'])


# Finally, let's look at the marketing medium. Here we have values like CPM (cost per thousand impressions), CPC (cost per click), affiliate (from affiliate site), referral (more targeted link (e.g. share)), organic (user finds page themselves). I will turn each of these into dummy variables and combine (not set) and (none) into "Other" bucket.

# In[ ]:


# Group unknown marketing mediums into "Other" bucket
raw_train.loc[raw_train['medium'].isin(['(not set)', '(none)']), ['medium']] = 'Other'

# Dummify operatingSytem into separate binary columns
raw_train = pd.get_dummies(raw_train, columns = ['medium'])


# With that the dataset grew quite a bit in its width: from 30 columns to 85. However, remember that there are some columns I won't be using at this time (they require further engineering). So, I'll leave you with the code that will drop these columns from the dataframe. That way there is a clean dataset to plug and plan into the model. Of course, normalization will need to be done on the variables and the log transform on the target variable

# In[ ]:


print('Number of columns: {:}'.format(raw_train.shape[1]))


# In[ ]:


# Drop columns that will not be used at this point in time
raw_train.drop(['date', 'isMobile', 'sessionId', 'visitStartTime', 
                'city', 'continent', 'country', 'metro', 'networkDomain', 
                'region', 'adContent', 'adNetworkType', 'gclId', 'page', 
                'slot', 'keyword', 'referralPath', 'source'], axis=1, inplace = True)


# ### 3. Feature Engineering - Work in Progress...
# In this section I will derive more advanced features such as the following:
# 1. Hours/minutes/seconds since last visit
# 2. Combine continent and country into a useful set of dummy variables
# 3. Group adContnet into useful categories 
# 4. Further explore keyword, medium, referral path, and source
