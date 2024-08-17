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


import warnings
from collections import Counter
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_log_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use(style='ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
sample = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
test_id = test['id']
target = train['revenue']


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


sample.head()


# In[7]:


train.shape, test.shape


# # EDA

# # Belongs To Collection

# In[8]:


for i in range(10):
    print(train['belongs_to_collection'][i])


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


train.drop(['belongs_to_collection'], axis=1, inplace=True)
test.drop(['belongs_to_collection'], axis=1, inplace=True)


# # Genres

# In[12]:


for i in range(10):
    print(train['genres'][i])


# ### We Should just extract the useful info (in this case we just need the genre of the movie)

# In[13]:


print(train.iloc[1, :])


# In[14]:


def get_dict(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


# In[15]:


gn = pd.DataFrame(columns=['genre_name'])
gn


# In[16]:


train = pd.concat([train, gn], axis=1)
train.head()


# In[17]:


test = pd.concat([test, gn], axis=1)
test.head()


# In[18]:


j = 0
for i in train['genres']:
    d = get_dict(i)
    if d != {}:
        train['genre_name'][j] = d[0]['name'] 
    else:
         train['genre_name'][j] = np.NaN
    j += 1

j = 0
for i in test['genres']:
    d = get_dict(i)
    if d != {}:
        test['genre_name'][j] = d[0]['name'] 
    else:
         test['genre_name'][j] = np.NaN
    j += 1


# In[19]:


train.drop(['genres'], axis=1, inplace=True)
test.drop(['genres'], axis=1, inplace=True)


# # Budget

# In[20]:


train['budget'].unique()


# ## Zeros in budget mean unknown, we will treat it as missing values later

# In[21]:


train[train['budget'] == 0].shape[0]


# In[22]:


plt.subplots(figsize=(12, 9))
plt.scatter(x=train['budget'], y=train['revenue'])


# # Homepage

# In[23]:


train['homepage'].value_counts(dropna=False)


# In[24]:


has_hompage = pd.DataFrame(columns=['has_homepage'])


# In[25]:


train = pd.concat([train, has_hompage], axis=1)
test = pd.concat([test, has_hompage], axis=1)


# In[26]:


j = 0
for i in train['homepage']:
    if str(train['homepage'][j]) == 'nan':
        train['has_homepage'][j] = 0
    else:
        train['has_homepage'][j] = 1
    j += 1

j = 0
for i in test['homepage']:
    if str(test['homepage'][j]) == 'nan':
        test['has_homepage'][j] = 0
    else:
        test['has_homepage'][j] = 1
    j += 1


# In[27]:


sns.catplot(x='has_homepage', y='revenue', data=train)


# ## The scatter plot above seems interesting, this really shows that having a wepage for the movie is really affects its revenue

# In[28]:


train.drop(['homepage'], axis=1, inplace=True)
test.drop(['homepage'], axis=1, inplace=True)
train.shape, test.shape


# # Imdb Id

# In[29]:


train['imdb_id']


# In[30]:


train.drop(['imdb_id'], axis=1, inplace=True)
test.drop(['imdb_id'], axis=1, inplace=True)
train.shape, test.shape


# # Original Language

# In[31]:


print(len(train['original_language'].value_counts(dropna=False)))
train['original_language'].value_counts(dropna=False)


# In[32]:


print(len(test['original_language'].value_counts(dropna=False)))
test['original_language'].value_counts(dropna=False)


# In[33]:


plt.subplots(figsize=(12, 9))
sns.boxplot(x=train['original_language'], y=train['revenue'])


# ## It would be great to make a new feature (Is Original Language is English)

# In[34]:


isOrgEn = pd.DataFrame(columns=['is_en_original_language'])
train = pd.concat([train, isOrgEn], axis=1)
test = pd.concat([test, isOrgEn], axis=1)

print(train['original_language'][0])


# In[35]:


j = 0
for i in train['original_language']:
    if i == 'en':
        train['is_en_original_language'][j] = 1
    else:
        train['is_en_original_language'][j] = 0
    j += 1
    
j = 0
for i in test['original_language']:
    if i == 'en':
        test['is_en_original_language'][j] = 1
    else:
        test['is_en_original_language'][j] = 0
    j += 1


# In[36]:


sns.catplot(x='is_en_original_language', y='revenue', data=train)


# ### The feature we generated is great indeed

# # Original Title

# In[37]:


len(train['original_title'].value_counts())


# In[38]:


train.drop(['original_title'], axis=1, inplace=True)
test.drop(['original_title'], axis=1, inplace=True)


# # Overview

# In[39]:


for i in range(5): 
    print(train['overview'][i])
    print("--------------------")


# ### I don't think using overview is gonna help us

# In[40]:


train.drop(['overview'], axis=1, inplace=True)
test.drop(['overview'], axis=1, inplace=True)
train.shape, test.shape


# In[41]:


# Popularity 


# In[42]:


train['popularity'].unique()


# In[43]:


train['popularity'].isnull().sum()


# In[44]:


plt.subplots(figsize=(12, 9))
# sns.catplot(x='revenue', y='popularity', data=train)
plt.scatter(x=train['popularity'], y=train['revenue'])
plt.xlabel('popularity')
plt.ylabel('revenue')
plt.show()


# ## This would be useful

# # Poster Path

# In[45]:


train.drop(['poster_path'], axis=1, inplace=True)
test.drop(['poster_path'], axis=1, inplace=True)
train.shape, test.shape


# # Production Companies

# In[46]:


for i in range(5):
    print(train['production_companies'][i])
    print("-------")


# ### Most of films have 1-2 production companies, cometimes 3-4. But there are films with 10+ companies! Let's have a look at some of them.
# ### Not sure yet what i'm gonna do with these data

# ## I guess my best move for now to get the number of production companies, The most 50 common production companies

# # Number of prod companies

# In[47]:


num_prod_com = pd.DataFrame(columns=['num_production_companies'])
train = pd.concat([train, num_prod_com], axis=1)
test = pd.concat([test, num_prod_com], axis=1)


# In[48]:


d = get_dict(train['production_companies'][0])
# This has 3 production companies
print(len(d))
d


# In[49]:


j = 0
for i in train['production_companies']:
    d = get_dict(i)
    if len(d) != 0:
        train['num_production_companies'][j] = len(d)
    else:
        train['num_production_companies'][j] = np.NaN
    j += 1

j = 0
for i in test['production_companies']:
    d = get_dict(i)
    if len(d) != 0:
        test['num_production_companies'][j] = len(d)
    else:
        test['num_production_companies'][j] = np.NaN
    j += 1


# In[50]:


sns.catplot(x='num_production_companies', y='revenue', data=train)


# # Production Company

# In[51]:


list_of_companies = []
for i in train['production_companies']:
    d = get_dict(i)
    if d != {}:
        for j in range(len(d)):
            list_of_companies.append(d[j]['name'])
list_of_companies


# In[52]:


# Top 50 production companies
top_companies_cnt = list(Counter(list_of_companies).most_common(50))
top_companies = []
for i in top_companies_cnt:
    top_companies.append(i[0])
temp = 0
for i in top_companies_cnt:
    temp += i[1]
temp


# In[53]:


print(top_companies_cnt)


# In[54]:


print(top_companies)


# In[55]:


prod_company = pd.DataFrame(columns=['production_company'])
train = pd.concat([train, prod_company], axis=1)
test = pd.concat([test, prod_company], axis=1)


# In[56]:


j = 0
cnt = 0
chk = False
for i in train['production_companies']:
    d = get_dict(i)
#     print(len(d))
    if d != {}:
        if len(d) > 1:
            for k in range(len(d)):
                company_name = d[k]['name']
                if company_name in top_companies:
                    train['production_company'][j] = company_name
                    cnt += 1
                    chk = True
                    break
            
            if chk is False:
                    train['production_company'][j] = d[0]['name']
                
        else:
            train['production_company'][j] = d[0]['name']
    else:
        train['production_company'][j] = np.NaN
    j += 1

print("THE NUMBER OF TOP COMPANIES AT TRAIN:", cnt)

j = 0
cnt = 0
chk = False
for i in test['production_companies']:
    d = get_dict(i)
#     print(len(d))
    if d != {}:
        if len(d) > 1:
            for k in range(len(d)):
                company_name = d[k]['name']
                if company_name in top_companies:
                    test['production_company'][j] = company_name
                    cnt += 1
                    chk = True
                    break
            
            if chk is False:
                    test['production_company'][j] = d[0]['name']
                
        else:
            test['production_company'][j] = d[0]['name']
    else:
        test['production_company'][j] = np.NaN
    j += 1

print("THE NUMBER OF TOP COMPANIES AT TEST:", cnt)


# In[57]:


train['production_company'].value_counts(dropna=False)


# ## I guess using this column is just useless, we are gonna drop it

# In[58]:


train.drop(['production_companies', 'production_company'], axis=1, inplace=True)
test.drop(['production_companies', 'production_company'], axis=1, inplace=True)
train.shape, test.shape


# # Production Country

# In[59]:


prod_country = pd.DataFrame(columns=['production_country'])
train = pd.concat([train, prod_country], axis=1)
test = pd.concat([test, prod_country], axis=1)


# In[60]:


j = 0
for i in train['production_countries']:
    d = get_dict(i)
    if d != {}:
        if len(d) > 1:
            countires = []
            for k in range(len(d)):
                countires.append(d[k]['name'])
            if 'United States of America' in countires:
                train['production_country'][j] = 'United States of America'
        else:
            train['production_country'][j] = d[0]['name']
    else:
        train['production_country'][j] = np.NaN
    j += 1
    
j = 0
for i in test['production_countries']:
    d = get_dict(i)
    if d != {}:
        if len(d) > 1:
            countires = []
            for k in range(len(d)):
                countires.append(d[k]['name'])
            if 'United States of America' in countires:
                test['production_country'][j] = 'United States of America'
        else:
            test['production_country'][j] = d[0]['name']
    else:
        test['production_country'][j] = np.NaN
    j += 1


# In[61]:


sns.catplot(x='production_country', y='revenue', data=train)


# ### Some feature like is USA Production sould be useful here, let's see

# In[62]:


is_usa_production = pd.DataFrame(columns=['is_usa_production'])
train = pd.concat([train, is_usa_production], axis=1)
test = pd.concat([test, is_usa_production], axis=1)


# In[63]:


j = 0
for i in train['production_country']:
    if i == 'United States of America':
        train['is_usa_production'][j] = 1
    elif str(i) == 'nan':
        train['is_usa_production'][j] = np.NaN
    else:
        train['is_usa_production'][j] = 0
    j += 1
    
j = 0
for i in test['production_country']:
    if i == 'United States of America':
        test['is_usa_production'][j] = 1
    elif str(i) == 'nan':
        test['is_usa_production'][j] = np.NaN
    else:
        test['is_usa_production'][j] = 0
    
    j += 1


# In[64]:


sns.catplot(x='is_usa_production', y='revenue', data=train)


# In[65]:


train.drop(['production_countries'], axis=1, inplace=True)
test.drop(['production_countries'], axis=1, inplace=True)


# # Release Date

# In[66]:


train['release_date'][:5]


# In[67]:


print(list(train['release_date'][1]))
print(train['release_date'][1])


# # Split the Release date to (Release Day, Release Month, Release Year)

# In[68]:


train[['release_month', 'release_day', 'release_year']] = train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)
test[['release_month', 'release_day', 'release_year']] = test['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)


# In[69]:


train.drop(['release_date'], axis=1, inplace=True)
test.drop(['release_date'], axis=1, inplace=True)
train.shape, test.shape


# In[70]:


sns.catplot(x='release_month', y='revenue', data=train)


# In[71]:


sns.catplot(x='release_day', y='revenue', data=train)


# In[72]:


plt.subplots(figsize=(12, 9))
plt.scatter(x=train['release_year'], y=train['revenue'])


# # RunTime

# In[73]:


print(train['runtime'].isnull().sum())
print(test['runtime'].isnull().sum())


# In[74]:


plt.subplots(figsize=(12, 9))
plt.scatter(x=train['runtime'], y=train['revenue'])


# In[75]:


# Zero running time means null value
print(len(train[train['runtime'] == 0]))
print(len(train[train['runtime'] == 0]))


# # Spoken Languages

# In[76]:


train['spoken_languages'][:5]


# ## Let's generate some features like (how many languages have been spoken in the movie, is English has been spoken)

# # Number of Spoken Languages

# In[77]:


number_of_spoken_languages = pd.DataFrame(columns=['number_of_spoken_languages'])
train = pd.concat([train, number_of_spoken_languages], axis=1)
test = pd.concat([test, number_of_spoken_languages], axis=1)


# In[78]:


j = 0
for i in train['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        train['number_of_spoken_languages'][j] = len(d)
    else:
        train['number_of_spoken_languages'][j] = np.NaN
    j += 1
    
j = 0
for i in test['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        test['number_of_spoken_languages'][j] = len(d)
    else:
        test['number_of_spoken_languages'][j] = np.NaN
    j += 1


# In[79]:


sns.catplot(x='number_of_spoken_languages', y='revenue', data=train)


# Clear pattern in the graph above

# # Is English a Spoken Language

# In[80]:


is_en_spoken = pd.DataFrame(columns=['is_en_spoken'])
train = pd.concat([train, is_en_spoken], axis=1)
test = pd.concat([test, is_en_spoken], axis=1)


# In[81]:


j = 0
for i in train['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        langs = []
        for k in range(len(d)):
            lang = d[k]['name']
            langs.append(lang)
        if 'English' in langs:
            train['is_en_spoken'][j] = 1
        else:
            train['is_en_spoken'][j] = 0
    else:
        train['is_en_spoken'][j] = np.NaN
    j += 1
        
j = 0
for i in test['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        langs = []
        for k in range(len(d)):
            lang = d[k]['name']
            langs.append(lang)
        if 'English' in langs:
            test['is_en_spoken'][j] = 1
        else:
            test['is_en_spoken'][j] = 0
    else:
        test['is_en_spoken'][j] = np.NaN
    j += 1


# In[82]:


sns.catplot(x='is_en_spoken', y='revenue', data=train)


# This feature is really useful

# # Spoken Language

# In[83]:


spoken_language = pd.DataFrame(columns=['spoken_language'])
train = pd.concat([train, spoken_language], axis=1)
test = pd.concat([test, spoken_language], axis=1)


# In[84]:


j = 0
for i in train['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        langs = []
        for k in range(len(d)):
            lang = d[k]['name']
            langs.append(lang)
        if 'English' in langs:
            train['spoken_language'][j] = 'English'
        else:
            train['spoken_language'][j] = langs[0]
    else:
        train['spoken_language'][j] = np.NaN
    j += 1
    
j = 0
for i in test['spoken_languages']:
    d = get_dict(i)
    if d != {}:
        langs = []
        for k in range(len(d)):
            lang = d[k]['name']
            langs.append(lang)
        if 'English' in langs:
            test['spoken_language'][j] = 'English'
        else:
            test['spoken_language'][j] = langs[0]
    else:
        test['spoken_language'][j] = np.NaN
    j += 1


# In[85]:


print(len(train['spoken_language'].value_counts()))
print(len(test['spoken_language'].value_counts()))


# In[86]:


sns.catplot(x='spoken_language', y='revenue', data=train)


# In[87]:


train.drop(['spoken_languages'], axis=1, inplace=True)
test.drop(['spoken_languages'], axis=1, inplace=True)
train.shape, test.shape


# # Status

# In[88]:


print(train['status'].value_counts(dropna=False))
print(test['status'].value_counts(dropna=False))


# # Is Released

# In[89]:


is_released = pd.DataFrame(columns=['is_released'])
train = pd.concat([train, is_released], axis=1)
test = pd.concat([test, is_released], axis=1)


# In[90]:


j = 0
for i in train['status']:
    if i == 'Released':
        train['is_released'][j] = 1
    else:
        train['is_released'][j] = 0
    j += 1
    
j = 0
for i in test['status']:
    if i == 'Released':
        test['is_released'][j] = 1
    else:
        test['is_released'][j] = 0
    j += 1


# In[91]:


print(train['is_released'].value_counts(dropna=False))
print(test['is_released'].value_counts(dropna=False))


# In[92]:


sns.catplot(x='is_released', y='revenue', data=train)


# In[93]:


train.drop(['status'], axis=1, inplace=True)
test.drop(['status'], axis=1, inplace=True)
train.shape, test.shape


# # Tagline

# In[94]:


train['tagline'][:10]


# In[95]:


train.drop(['tagline'], axis=1, inplace=True)
test.drop(['tagline'], axis=1, inplace=True)
train.shape, test.shape


# # Keywords

# In[96]:


for i in range(10): 
    print(train['Keywords'][i])
    print("-------")


# In[97]:


keyword = pd.DataFrame(columns=['keyword'])
train = pd.concat([train, keyword], axis=1)
test = pd.concat([test, keyword], axis=1)


# In[98]:


j = 0
for i in train['Keywords']:
    d = get_dict(i)
    if d != {}:
        train['keyword'][j] = d[0]['name']
    else:
        train['keyword'][j] = np.NaN
    j += 1
    
j = 0
for i in test['Keywords']:
    d = get_dict(i)
    if d != {}:
        test['keyword'][j] = d[0]['name']
    else:
        test['keyword'][j] = np.NaN
    j += 1


# In[99]:


train['keyword'].value_counts(dropna=False)


# In[100]:


# We can not handle this much of keywords
train.drop(['Keywords', 'keyword'], axis=1, inplace=True)
test.drop(['Keywords', 'keyword'], axis=1, inplace=True)
train.shape, test.shape


# # Cast

# In[101]:


for i in range(1):
    print(train['cast'][i])
    print("----------")


# # Size of cast (number of actors)

# In[102]:


size_of_cast = pd.DataFrame(columns=['size_of_cast'])
train = pd.concat([train, size_of_cast], axis=1)
test = pd.concat([test, size_of_cast], axis=1)


# In[103]:


j = 0
for i in train['cast']:
    d = get_dict(i)
    if d != {}:
        train['size_of_cast'][j] = len(d)
    else:
        train['size_of_cast'][j] = np.NaN
    j += 1
    
j = 0
for i in test['cast']:
    d = get_dict(i)
    if d != {}:
        test['size_of_cast'][j] = len(d)
    else:
        test['size_of_cast'][j] = np.NaN
    j += 1


# In[104]:


plt.subplots(figsize=(12, 9))
plt.scatter(x=train['size_of_cast'], y=train['revenue'])


# In[105]:


train.drop(['cast'], axis=1, inplace=True)
test.drop(['cast'], axis=1, inplace=True)
train.shape, test.shape


# # Title, Crew

# In[106]:


train.drop(['id', 'crew', 'title'], axis=1, inplace=True)
test.drop(['id', 'crew', 'title'], axis=1, inplace=True)
train.shape, test.shape


# # Null values

# ### Budget

# In[107]:


print(train['budget'].isnull().sum())
print(test['budget'].isnull().sum())
print(len(train[train['budget'] == 0]))
print(len(test[test['budget'] == 0]))


# In[108]:


plt.scatter(x=train['budget'], y=train['revenue'])


# In[109]:


ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat([train, test], axis=0)
train.shape, test.shape, all_data.shape


# In[110]:


all_data['budget'].dtype


# In[111]:


all_data['budget'] = all_data['budget'].replace(0, all_data['budget'].mean())


# ### Original Language

# In[112]:


print(all_data['original_language'].isnull().sum())
print(len(all_data[all_data['original_language'] == 0]))


# ### Popularity

# In[113]:


print(all_data['popularity'].isnull().sum())
print(len(all_data[all_data['popularity'] == 0]))


# ### Runtime

# In[114]:


print(all_data['runtime'].isnull().sum())
print(len(all_data[all_data['runtime'] == 0]))


# In[115]:


all_data['runtime'].mean()


# In[116]:


all_data['runtime'] = all_data['runtime'].replace(0, all_data['runtime'].mean())
all_data['runtime'] = all_data['runtime'].fillna(all_data['runtime'].mean())


# ### Genre Name

# In[117]:


print(all_data['genre_name'].isnull().sum())
print(len(all_data[all_data['genre_name'] == 0]))


# In[118]:


print(all_data['genre_name'].value_counts())


# In[119]:


all_data['genre_name'] = all_data['genre_name'].fillna(all_data['genre_name'].mode()[0])


# ### Number of production companies

# In[120]:


print(all_data['num_production_companies'].isnull().sum())
print(len(all_data[all_data['num_production_companies'] == 0]))


# In[121]:


all_data['num_production_companies'] = all_data['num_production_companies'].fillna(all_data['num_production_companies'].mean().round())


# In[122]:


train.head()


# In[ ]:




