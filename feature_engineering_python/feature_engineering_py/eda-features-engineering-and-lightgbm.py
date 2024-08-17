#!/usr/bin/env python
# coding: utf-8

# # General information
# This kernel is dedicated to extensive EDA of Avito Demand Prediction Challenge competition as well as feature engineering. Only a simple model is used due to kernel memory constraint.

# In[ ]:


import os
import pandas_profiling as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import datetime
import pandas_profiling as pp
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
stop = set(stopwords.words('russian'))
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
periods_train = pd.read_csv('../input/periods_train.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# ## Data overview

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe(include='all')


# In[ ]:


periods_train.head()


# In[ ]:


pp.ProfileReport(train)


# We can see several things from this overview:
# 
# * There is a variety of features: numerical, categorical, text and date;
# * Some columns have missing values: not all users define additional parameters of items or descriptions, sometimes they don't even provide descriptions. In some cases there are no photos of the wares;
# * As expected, there are a lot of unique users and most of them don't post a lot of ads, but there are outliers with 600+ ads;
# * There are 9 categories in parent_category_name and "Личные вещи" (Private things) have 46,4% of values;
# * price will require a careful processing - the values are skewered and there are some outliers with huge values;
# * It is possible to use images in the analysis, but I'll simply use the fact whether there was image or not;

# ## Feature analysis
# We saw a lot of information about features, so let's now analyze each of them in more details.

# ### activation_date
# 
# At first let's create new features based on activation_date: date, weekday and day of month.

# In[ ]:


train['activation_date'] = pd.to_datetime(train['activation_date'])
train['date'] = train['activation_date'].dt.date
train['weekday'] = train['activation_date'].dt.weekday
train['day'] = train['activation_date'].dt.day
count_by_date_train = train.groupby('date')['deal_probability'].count()
mean_by_date_train = train.groupby('date')['deal_probability'].mean()

test['activation_date'] = pd.to_datetime(test['activation_date'])
test['date'] = test['activation_date'].dt.date
test['weekday'] = test['activation_date'].dt.weekday
test['day'] = test['activation_date'].dt.day
count_by_date_test = test.groupby('date')['item_id'].count()


# In[ ]:


fig, (ax1, ax3) = plt.subplots(figsize=(26, 8), ncols=2, sharey=True)
count_by_date_train.plot(ax=ax1, legend=False, label='Ads count')
ax1.set_ylabel('Ads count', color='b')
ax2 = ax1.twinx()
mean_by_date_train.plot(ax=ax2, color='g', legend=False, label='Mean deal_probability')
ax2.set_ylabel('Mean deal_probability', color='g')
count_by_date_test.plot(ax=ax3, color='r', legend=False, label='Ads count test')
plt.grid(False)

ax1.title.set_text('Trends of deal_probability and number of ads')
ax3.title.set_text('Trends of number of ads for test data')
ax1.legend(loc=(0.8, 0.35))
ax2.legend(loc=(0.8, 0.2))
ax3.legend(loc="upper right")


# As we can see, we don't only several weeks of data in train and a little more than a week in test
# 
# * For most of the data in train the number of ads is quite high (100 000 or more) and mean deal_probability is around 0.14, but after March 28 the number of ads drastically falls so deal_probability fluctuates. I wonder if decreased number of ads is intentional;
# * In test we have a reasonable number of ads up to April 18 and then number of ads become negligible - 64 and 1;
# * As a result I'd suggest not to use train data with too low number of ads (since 2017-03-29);

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Ads count and deal_probability by day of week.")
sns.countplot(x='weekday', data=train, ax=ax1)
ax1.set_ylabel('Ads count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
sns.pointplot(x="weekday", y="deal_probability", data=train, ci=99, ax=ax2, color='black')
ax2.set_ylabel('deal_probability', color='g')
plt.legend(['deal_probability'], loc=(0.875, 0.9))
plt.grid(False)


# We can see that there is a little difference in deal_probability if we look at deal_probability by weekday.

# ## Categories
# 

# In[ ]:


a = train.groupby(['parent_category_name', 'category_name']).agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'mean')], ascending=False).reset_index(drop=True)
a


# We can see that "Услуги" (services) is the category with the highest deal_probability. Other "good" categories are about animals or electronics/cars.
# Least successful are various accessories or expensive things.

# ## city

# In[ ]:


city_ads = train.groupby('city').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'mean')], ascending=False).reset_index(drop=True)
print('There are {0} cities in total.'.format(len(train.city.unique())))
print('There are {1} cities with more that {0} ads.'.format(100, city_ads[city_ads['deal_probability']['count'] > 100].shape[0]))
print('There are {1} cities with more that {0} ads.'.format(1000, city_ads[city_ads['deal_probability']['count'] > 1000].shape[0]))
print('There are {1} cities with more that {0} ads.'.format(10000, city_ads[city_ads['deal_probability']['count'] > 10000].shape[0]))


# It seems that most of the cities have little ads posted and only in 33 of them there a lot of ads. Let's see the best and the worst cities by mean deal_probability.

# In[ ]:


city_ads[city_ads['deal_probability']['count'] > 1000].head()


# In[ ]:


city_ads[city_ads['deal_probability']['count'] > 1000].tail()


# I think that it could be interesting to see what is sold in Лабинск and Миллерово

# In[ ]:


print('Лабинск')
train.loc[train.city == 'Лабинск'].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head(5)


# Most popular categories are "Автомобили" (cars) and "Телефоны" (telephones).

# In[ ]:


print('Миллерово')
train.loc[train.city == 'Миллерово'].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head()


# Most popular categories are "Автомобили" (cars). And it seems that second-hand wares are least popular.

# ## deal_probability

# In[ ]:


plt.hist(train['deal_probability']);
plt.title('deal_probability');


# On the one hand the distribution of the target value is highly skewered towards zero, on the other hand, there is a spike at about 0.8.

# ## title

# In[ ]:


text = ' '.join(train['title'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for title')
plt.axis("off")
plt.show()


# ## description

# In[ ]:


train['description'] = train['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))


# In[ ]:


text = ' '.join(train['description'].values)
text = [i for i in ngrams(text.lower().split(), 3)]
print('Common trigrams.')
Counter(text).most_common(40)


# We can see that sellers try to tell buyers that their wares are great and also tell about possibilities of delivery. But there are some strange values, let's have a look...

# In[ ]:


train[train.description.str.contains('↓')]['description'].head(10).values


# It seems that some authors really like emotional text with a lot of symbols or even words in upper case! We will use these features.

# ## image
# In this kernel I won't use the images themselves, but I'll create a feature showing wheather there is an image or not

# In[ ]:


train['has_image'] = 1
train.loc[train['image'].isnull(),'has_image'] = 0
print('There are {} ads with images. Mean deal_probability is {:.3}.'.format(len(train.loc[train['has_image'] == 1]), train.loc[train['has_image'] == 1, 'deal_probability'].mean()))
print('There are {} ads without images. Mean deal_probability is {:.3}.'.format(len(train.loc[train['has_image'] == 0]), train.loc[train['has_image'] == 0, 'deal_probability'].mean()))


# It is interesting, but ads without images are more likely to be bought.

# ## item_seq_number

# In[ ]:


plt.scatter(train.item_seq_number, train.deal_probability, label='item_seq_number vs deal_probability');
plt.xlabel('item_seq_number');
plt.ylabel('deal_probability');


# It seems that there are many users who post a lot of ads and number of ads posted isn't really correlated with deal_probability. There is some descreading trend, but we can't be sure.

# ## Params
# There are three fields with additional information, let's combine it into one. Technically it is possible to treat these features as categorical, but there would be too many of them

# In[ ]:


train['params'] = train['param_1'].fillna('') + ' ' + train['param_2'].fillna('') + ' ' + train['param_3'].fillna('')
train['params'] = train['params'].str.strip()
text = ' '.join(train['params'].values)
text = [i for i in ngrams(text.lower().split(), 3)]
print('Common trigrams.')
Counter(text).most_common(40)


# Most of params belong to clothes or cars.

# ## user_type 
# There are three main user_types. Let's see prices of their wares, where prices are below 100000.

# In[ ]:


sns.set(rc={'figure.figsize':(15, 8)})
train_ = train[train.price.isnull() == False]
train_ = train.loc[train.price < 100000.0]
sns.boxplot(x="parent_category_name", y="price", hue="user_type",  data=train_)
plt.title("Price by parent category and user type")
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# We can see that shops usually have higher prices than companies and private sellers usually have the lowest price - maybe because they are usually second-hand.

# ## price
# 
# The first question is how to deal with missing values.
# I have decided to do the following:
# 
# - at first fill missing values with median by city and category;
# - then missing values which are left are filled with region by region and category;
# - the remaining missing values are filled with median by category;

# In[ ]:


train['price'] = train.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = train.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = train.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
plt.hist(train['price']);


# Let's use boxcox transformation to get rid of skewness

# In[ ]:


plt.hist(stats.boxcox(train['price'] + 1)[0]);


# ## Feature engineering
# 

# In[ ]:


#Let's transform test in the same way as train.
test['params'] = test['param_1'].fillna('') + ' ' + test['param_2'].fillna('') + ' ' + test['param_3'].fillna('')
test['params'] = test['params'].str.strip()

test['description'] = test['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))
test['has_image'] = 1
test.loc[test['image'].isnull(),'has_image'] = 0

test['price'] = test.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
test['price'] = test.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
test['price'] = test.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = stats.boxcox(train.price + 1)[0]
test['price'] = stats.boxcox(test.price + 1)[0]


# ## Aggregate features
# I'll create a number of aggregate features.

# In[ ]:


train['user_price_mean'] = train.groupby('user_id')['price'].transform('mean')
train['user_ad_count'] = train.groupby('user_id')['price'].transform('sum')

train['region_price_mean'] = train.groupby('region')['price'].transform('mean')
train['region_price_median'] = train.groupby('region')['price'].transform('median')
train['region_price_max'] = train.groupby('region')['price'].transform('max')

train['region_price_mean'] = train.groupby('region')['price'].transform('mean')
train['region_price_median'] = train.groupby('region')['price'].transform('median')
train['region_price_max'] = train.groupby('region')['price'].transform('max')

train['city_price_mean'] = train.groupby('city')['price'].transform('mean')
train['city_price_median'] = train.groupby('city')['price'].transform('median')
train['city_price_max'] = train.groupby('city')['price'].transform('max')

train['parent_category_name_price_mean'] = train.groupby('parent_category_name')['price'].transform('mean')
train['parent_category_name_price_median'] = train.groupby('parent_category_name')['price'].transform('median')
train['parent_category_name_price_max'] = train.groupby('parent_category_name')['price'].transform('max')

train['category_name_price_mean'] = train.groupby('category_name')['price'].transform('mean')
train['category_name_price_median'] = train.groupby('category_name')['price'].transform('median')
train['category_name_price_max'] = train.groupby('category_name')['price'].transform('max')

train['user_type_category_price_mean'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
train['user_type_category_price_median'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
train['user_type_category_price_max'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('max')


# In[ ]:


test['user_price_mean'] = test.groupby('user_id')['price'].transform('mean')
test['user_ad_count'] = test.groupby('user_id')['price'].transform('sum')

test['region_price_mean'] = test.groupby('region')['price'].transform('mean')
test['region_price_median'] = test.groupby('region')['price'].transform('median')
test['region_price_max'] = test.groupby('region')['price'].transform('max')

test['region_price_mean'] = test.groupby('region')['price'].transform('mean')
test['region_price_median'] = test.groupby('region')['price'].transform('median')
test['region_price_max'] = test.groupby('region')['price'].transform('max')

test['city_price_mean'] = test.groupby('city')['price'].transform('mean')
test['city_price_median'] = test.groupby('city')['price'].transform('median')
test['city_price_max'] = test.groupby('city')['price'].transform('max')

test['parent_category_name_price_mean'] = test.groupby('parent_category_name')['price'].transform('mean')
test['parent_category_name_price_median'] = test.groupby('parent_category_name')['price'].transform('median')
test['parent_category_name_price_max'] = test.groupby('parent_category_name')['price'].transform('max')

test['category_name_price_mean'] = test.groupby('category_name')['price'].transform('mean')
test['category_name_price_median'] = test.groupby('category_name')['price'].transform('median')
test['category_name_price_max'] = test.groupby('category_name')['price'].transform('max')

test['user_type_category_price_mean'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
test['user_type_category_price_median'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
test['user_type_category_price_max'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('max')


# ## Categorical features
# 
# I'll use target encoding to deal with categorical features.

# In[ ]:


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return ft_trn_series, ft_tst_series


# In[ ]:


train['parent_category_name'], test['parent_category_name'] = target_encode(train['parent_category_name'], test['parent_category_name'], train['deal_probability'])
train['category_name'], test['category_name'] = target_encode(train['category_name'], test['category_name'], train['deal_probability'])
train['region'], test['region'] = target_encode(train['region'], test['region'], train['deal_probability'])
train['image_top_1'], test['image_top_1'] = target_encode(train['image_top_1'], test['image_top_1'], train['deal_probability'])
train['city'], test['city'] = target_encode(train['city'], test['city'], train['deal_probability'])
train['param_1'], test['param_1'] = target_encode(train['param_1'], test['param_1'], train['deal_probability'])
train['param_2'], test['param_2'] = target_encode(train['param_2'], test['param_2'], train['deal_probability'])
train['param_3'], test['param_3'] = target_encode(train['param_3'], test['param_3'], train['deal_probability'])


# In[ ]:


train.drop(['date', 'day', 'user_id'], axis=1, inplace=True)
test.drop(['date', 'day', 'user_id'], axis=1, inplace=True)


# ## Text features
# 
# We have several features with text data and they need to be processed in different ways. But at first let's create new features based on texts: length of text (symbols) and number of words. Also let's calculate counts of punctuation and some of strange symbols.

# In[ ]:


train['len_title'] = train['title'].apply(lambda x: len(x))
train['words_title'] = train['title'].apply(lambda x: len(x.split()))
train['len_description'] = train['description'].apply(lambda x: len(x))
train['words_description'] = train['description'].apply(lambda x: len(x.split()))
train['len_params'] = train['params'].apply(lambda x: len(x))
train['words_params'] = train['params'].apply(lambda x: len(x.split()))

train['symbol1_count'] = train['description'].str.count('↓')
train['symbol2_count'] = train['description'].str.count('\*')
train['symbol3_count'] = train['description'].str.count('✔')
train['symbol4_count'] = train['description'].str.count('❀')
train['symbol5_count'] = train['description'].str.count('➚')
train['symbol6_count'] = train['description'].str.count('ஜ')
train['symbol7_count'] = train['description'].str.count('.')
train['symbol8_count'] = train['description'].str.count('!')
train['symbol9_count'] = train['description'].str.count('\?')
train['symbol10_count'] = train['description'].str.count('  ')
train['symbol11_count'] = train['description'].str.count('-')
train['symbol12_count'] = train['description'].str.count(',')

test['len_title'] = test['title'].apply(lambda x: len(x))
test['words_title'] = test['title'].apply(lambda x: len(x.split()))
test['len_description'] = test['description'].apply(lambda x: len(x))
test['words_description'] = test['description'].apply(lambda x: len(x.split()))
test['len_params'] = test['params'].apply(lambda x: len(x))
test['words_params'] = test['params'].apply(lambda x: len(x.split()))

test['symbol1_count'] = test['description'].str.count('↓')
test['symbol2_count'] = test['description'].str.count('\*')
test['symbol3_count'] = test['description'].str.count('✔')
test['symbol4_count'] = test['description'].str.count('❀')
test['symbol5_count'] = test['description'].str.count('➚')
test['symbol6_count'] = test['description'].str.count('ஜ')
test['symbol7_count'] = test['description'].str.count('.')
test['symbol8_count'] = test['description'].str.count('!')
test['symbol9_count'] = test['description'].str.count('\?')
test['symbol10_count'] = test['description'].str.count('  ')
test['symbol11_count'] = test['description'].str.count('-')
test['symbol12_count'] = test['description'].str.count(',')


# Now let's start transforming texts. Titles have little number of unique words, so we can use default values for TfidfVectorizer (only add stopwords). I have to limit max_features due to memory constraints. I won't use descriptions and parameters due to kernel limits.

# In[ ]:


vectorizer=TfidfVectorizer(stop_words=stop, max_features=2000)
vectorizer.fit(train['title'])
train_title = vectorizer.transform(train['title'])
test_title = vectorizer.transform(test['title'])


# In[ ]:


train.drop(['title', 'params', 'description', 'user_type', 'activation_date'], axis=1, inplace=True)
test.drop(['title', 'params', 'description', 'user_type', 'activation_date'], axis=1, inplace=True)


# In[ ]:


pd.set_option('max_columns', 60)
train.head()


# One of possible ideas is creating meta-features. It means that we use some features to build a model and use the predictions in another model. I'll use ridge regression to create a new feature based on tokenized title and then I'll combine it with other features.

# In[ ]:


get_ipython().run_cell_magic('time', '', "X_meta = np.zeros((train_title.shape[0], 1))\nX_test_meta = []\nfor fold_i, (train_i, test_i) in enumerate(kf.split(train_title)):\n    print(fold_i)\n    model = Ridge()\n    model.fit(train_title.tocsr()[train_i], train['deal_probability'][train_i])\n    X_meta[test_i, :] = model.predict(train_title.tocsr()[test_i]).reshape(-1, 1)\n    X_test_meta.append(model.predict(test_title))\n    \nX_test_meta = np.stack(X_test_meta)\nX_test_meta_mean = np.mean(X_test_meta, axis = 0)\n")


# In[ ]:


X_full = csr_matrix(hstack([train.drop(['item_id', 'deal_probability', 'image'], axis=1), X_meta]))
X_test_full = csr_matrix(hstack([test.drop(['item_id', 'image'], axis=1), X_test_meta_mean.reshape(-1, 1)]))

X_train, X_valid, y_train, y_valid = train_test_split(X_full, train['deal_probability'], test_size=0.20, random_state=42)


# ## Building a simple model
# 

# In[ ]:


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[ ]:


#took parameters from this kernel: https://www.kaggle.com/the1owl/beep-beep
params = {'learning_rate': 0.05, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'regression', 'metric': ['auc','rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 63, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)


# In[ ]:


pred = model.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv('sub.csv', index=False)


# ## Additional ideas
# - It is really necessary to use a machine with more CPU. I used tf-idf on descriptions and params and got 0.227 on leaderboard with this lgb model;
# - Price is a tricky feature and needs careful preprocessing;
# - Texts are really interesting. I'm sure there are a lot of features which can be created based on them. Also russian embeddings can be used;
# - And of course there are pictures. We can try extract some interesting features, or build CNN models on  
