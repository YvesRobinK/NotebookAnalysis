#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <span style="font-size:21px;">Ladies and gentlemen 😃 dear members of the Kaggle community 💙, instead of spending hours watching Youtube videos wouldn't it be a good idea to work on the platform's data itself? <br>
#     The 'Predict Youtube Video Likes (Pog Series #1)' competition has been the perfect opportunity for me to replace the long hours I spent on that platform😅. Just looking through the data description, a prethera of ideas and approachs came to my mind. So I'll try to include most of them in this notebook, and I hope you like it.

# # The Journey (aka EDA, Data Preprocessing and Feature Engineering)

# In[1]:


import numpy as np
import pandas as pd
import re

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# In[2]:


train = pd.read_parquet("../input/kaggle-pog-series-s01e01/train.parquet")
test = pd.read_parquet("../input/kaggle-pog-series-s01e01/test.parquet")


# <span style="font-size:18px;">Let us start by doing a general analysis of our data ; how many features and rows does it contain, their types, number of unique values, null values...

# In[3]:


print("The shape of the trainset is : ",train.shape)
print("The shape of the testset is : ",test.shape)


# In[4]:


train.info()


# In[5]:


test.info()


# <span style="font-size:18px;">So there is a difference between the features provided in the trainset and in the testset.

# In[6]:


train.columns.difference(test.columns)


# In[7]:


(train.likes/train.view_count-train.target).unique()


# In[8]:


test.columns.difference(train.columns)


# In[9]:


test["isTest"].unique()


# <span style="font-size:18px;">
# We can see that we have one regression target to deal with in our problem with 2 features leaking our target ("likes" and "view_count"), so we should be careful approaching them if we are not dropping them to avoid overfitting.<br>
# The test data contain an additional (not informative) feature which is "isTest" that we will need to drop.

# In[10]:


train.loc[1063,"thumbnail_link"]


# <span style="font-size:18px;">
# Dealing with images is out of the scope of this notebook (I'll try to use them in a future one), so I'll be dropping the "thumbnail_link" variable.<br>
# In addition, I'll be doing only simple operations on the textual data (sorry BERT, ELMO, Word2vec and even Tf-idf fans but maybe in a future notebook 😁).<br>
# Now for the date/time variables📅, they are a goldmine that we will exploit during the Feature Engineering step (but first we will need to change the "trending_date" type to datetime).
# </span>

# ## Working on date/time variables

# In[11]:


train.trending_date = pd.to_datetime(train.trending_date)
test.trending_date = pd.to_datetime(test.trending_date)


# <span style="font-size:18px;">First we will extract various possible features from the date/time features :

# In[12]:


dates_df_train = pd.DataFrame()
dates_df_test = pd.DataFrame()
dates_df_train["published_day"] = train["publishedAt"].dt.day
dates_df_test["published_day"] = test["publishedAt"].dt.day
dates_df_train["published_month"] = train["publishedAt"].dt.month
dates_df_test["published_month"] = test["publishedAt"].dt.month
dates_df_train["trending_day"] = train["trending_date"].dt.day
dates_df_test["trending_day"] = test["trending_date"].dt.day

# I am not including the treniding month and the year(for both the publishing and the trending) variable
# because there is only one trending month(12) and only one year (2021) in the testset for both variables.
# If you want you can uncomment the following lines.
"""
dates_df_train["trending_month"] = train["trending_date"].dt.month
dates_df_test["trending_month"] = test["trending_date"].dt.month

dates_df_train["published_year"] = train["publishedAt"].dt.year
dates_df_test["published_year"] = test["publishedAt"].dt.year
dates_df_train["trending_year"] = train["trending_date"].dt.year
dates_df_test["trending_year"] = test["trending_date"].dt.year
"""

dates_df_train["days_before_trending"] = (train["trending_date"]-pd.to_datetime(train["publishedAt"].dt.date)).dt.days
dates_df_test["days_before_trending"] = (test["trending_date"]-pd.to_datetime(test["publishedAt"].dt.date)).dt.days
dates_df_train["published_weekday"] = train["publishedAt"].dt.weekday
dates_df_test["published_weekday"] = test["publishedAt"].dt.weekday
dates_df_train["published_weekend"] = (train["publishedAt"].dt.weekday >= 5)*1
dates_df_test["published_weekend"] = (test["publishedAt"].dt.weekday >= 5)*1
dates_df_train["trending_date_weekday"] = train["trending_date"].dt.weekday
dates_df_test["trending_date_weekday"] = test["trending_date"].dt.weekday
dates_df_train["trending_date_weekend"] = (train["trending_date"].dt.weekday >= 5)*1
dates_df_test["trending_date_weekend"] = (test["trending_date"].dt.weekday >= 5)*1

dates_df_train["published_hour"] = train["publishedAt"].dt.hour
dates_df_test["published_hour"] = test["publishedAt"].dt.hour


# In[13]:


dates_df_train.describe()


# In[14]:


dates_df_test.describe()


# <span style="font-size:18px;">Now we can drop the date variables as we've extracted all the necessary information :

# In[15]:


train = train.drop(["publishedAt", "trending_date"], axis = 1)
test = test.drop(["publishedAt", "trending_date"], axis = 1)


# <span style="font-size:18px;">And we define our different kinds of parameters :

# In[16]:


cat = train.nunique()[train.nunique()<16].index
text = train.select_dtypes(include=["object"]).columns.drop('id')
dates_related = dates_df_train.columns
cont = train.columns.difference(cat).difference(text).difference(["id","target"]).difference(dates_related)


# <span style="font-size:18px;">And of course we need to add the newely created variables into the train and the test sets :

# In[17]:


train = pd.concat([train,dates_df_train], axis = 1)
test = pd.concat([test,dates_df_test], axis = 1)


# <span style="font-size:18px;">Before moving on, there is something with the new dates/time variables that is bothering me ; with the current representation doesn't December seem too far from January (12 and 1) 🤔⁉ The same goes for the weekday, monthday and hour of the day.<br>
#     In fact, if we feed this data to any existing model, it wouldn't be able to understand the periodicity of such variables.<br>
#     So the idea consists of exploiting the periodicity of the trigonometric functions by applying cos or sin to the concerned variables :

# ![image.png](attachment:a0dc8b1f-9e9b-4428-a8f1-dcf60ff392c4.png)

# In[18]:


train["published_hour"] = np.cos(train["published_hour"]*np.pi/12)
test["published_hour"] = np.cos(test["published_hour"]*np.pi/12)
train["published_weekday"] = np.cos(train["published_weekday"]*np.pi/7)
test["published_weekday"] = np.cos(test["published_weekday"]*np.pi/7)
train["published_month"] = np.cos(train["published_month"]*np.pi/6)
test["published_month"] = np.cos(test["published_month"]*np.pi/6)


train["trending_date_weekday"] = np.cos(train["trending_date_weekday"]*np.pi/7)
test["trending_date_weekday"] = np.cos(test["trending_date_weekday"]*np.pi/7)

# Now for the monthday it's quite particular due to the difference of the total number of days per month
train.loc[train["published_month"].isin([1,3,5,7,8,10,12]),"published_day"] = np.cos(train[train["published_month"].\
                                                                                isin([1,3,5,7,8,10,12])]["published_day"]\
                                                                                *2*np.pi/31)
train.loc[train["published_month"].isin([4,6,9,11]),"published_day"] = np.cos(train[train["published_month"].\
                                                                                isin([4,6,9,11])]["published_day"]\
                                                                                *2*np.pi/30)
train.loc[train["published_month"].isin([2]),"published_day"] = np.cos(train[train["published_month"].\
                                                                                isin([2])]["published_day"]\
                                                                                *2*np.pi/28)


# <span style="font-size:18px;">Let's do some plotting and see how our new date/time variables influence our target :

# In[19]:


fig, axes = plt.subplots(3,3, figsize = (24,24))
axes = axes.flatten()
for i, x in enumerate(dates_df_train.columns):
    sns.histplot(data = train, x = x, ax = axes[i])


# In[20]:


fig, axes = plt.subplots(3,3, figsize = (24,24))
axes = axes.flatten()
for i, x in enumerate(dates_df_train.columns):
    sns.histplot(data = train, x = x, y = "target", ax = axes[i])


# ## Dealing with the textual data

# <span style="font-size:18px;">As mentioned earlier, nothing fancy will be used on the textual data. All the features that I'll add are based on the length of the string or its percentage of uppercase letters (Youtubers are known to love their capital letters.)

# In[21]:


train["description"].str.findall(r'[A-Z]')


# In[22]:


text_df_train = pd.DataFrame()
text_df_test = pd.DataFrame()

text_df_train["len_title"] = train["title"].str.len()
text_df_test["len_title"] = test["title"].str.len()

text_df_train["upper_percent_title"] = train["title"].str.findall(r'[A-Z]').apply(lambda x: len(x))/text_df_train["len_title"]
text_df_test["upper_percent_title"] = test["title"].str.findall(r'[A-Z]').apply(lambda x: len(x))/text_df_test["len_title"]

text_df_train["len_channel_title"] = train["channelTitle"].str.len()
text_df_test["len_channel_title"] = test["channelTitle"].str.len()

text_df_train["upper_percent_channel_title"] = train["channelTitle"].str.findall(r'[A-Z]').apply(lambda x: len(x))/text_df_train["len_channel_title"]
text_df_test["upper_percent_channel_title"] = test["channelTitle"].str.findall(r'[A-Z]').apply(lambda x: len(x))/text_df_test["len_channel_title"]

text_df_train["len_description"] = train["description"].str.len()
text_df_test["len_description"] = test["description"].str.len()

text_df_train["upper_percent_description"] = train["description"].str.findall(r'[A-Z]').apply(lambda x: len(x) if (x is not None) else 0)/text_df_train["len_description"]
text_df_test["upper_percent_description"] = test["description"].str.findall(r'[A-Z]').apply(lambda x: len(x) if (x is not None) else 0)/text_df_test["len_description"]

text_df_train["number_tags"] = train["tags"].str.split('|').apply(lambda x: len(x))
text_df_test["number_tags"] = test["tags"].str.split('|').apply(lambda x: len(x))

nbr_cap_tags_train = train["tags"].str.split('|').apply(lambda x: sum(1 for i in x if i.isupper()))
nbr_cap_tags_test = test["tags"].str.split('|').apply(lambda x: sum(1 for i in x if i.isupper()))
text_df_train["cap_tags_percent"] = nbr_cap_tags_train/text_df_train["number_tags"]
text_df_test["cap_tags_percent"] = nbr_cap_tags_train/text_df_test["number_tags"]


# In[23]:


text_df_train.describe()


# <span style="font-size:18px;">Now we can drop all the text features after extracting all the necessary information :

# In[24]:


train = train.drop(["title", "channelTitle", "tags", "description"], axis = 1)
test = test.drop(["title", "channelTitle", "tags", "description"], axis = 1)


# <span style="font-size:18px;">And of course we will need to add the new variables to the train and test sets :

# In[25]:


text_related = text_df_train.columns


# In[26]:


train = pd.concat([train,text_df_train], axis = 1)
test = pd.concat([test,text_df_test], axis = 1)


# <span style="font-size:18px;">We can't pass without doing some visualization :

# In[27]:


fig, axes = plt.subplots(2,4, figsize = (24,12))
axes = axes.flatten()
for i, x in enumerate(text_df_train.columns):
    sns.histplot(data = train, x = x, ax = axes[i])


# In[28]:


fig, axes = plt.subplots(2,4, figsize = (24,12))
axes = axes.flatten()
for i, x in enumerate(text_df_train.columns):
    sns.scatterplot(data = train, x = x, y = "target", ax = axes[i])


# <span style="font-size:18px;"> We will be dropping the video and the channel's ids as they represent similar information as the respective titles

# In[29]:


train = train.drop(["video_id", "channelId", "thumbnail_link"], axis = 1)
test = test.drop(["video_id", "channelId", "thumbnail_link", "isTest"], axis = 1)


# In[30]:


print("The shape of the trainset is : ",train.shape)
print("The shape of the testset is : ",test.shape)


# ## Misc Variables 

# <span style="font-size:18px;"> Let us explore our categorical variables :

# In[31]:


cat


# In[32]:


train["comments_disabled"] = train["comments_disabled"]*1
test["comments_disabled"] = test["comments_disabled"]*1


# In[33]:


fig, axes = plt.subplots(1,4, figsize = (24,6))
axes = axes.flatten()
for i, x in enumerate(cat):
    sns.countplot(data = train, x = x, ax = axes[i])
plt.show()


# In[34]:


fig, axs = plt.subplots(1,4,figsize = (24,6))
axs = axs.flatten()
x = "target"
for i, y in enumerate(cat):
    sns.histplot(data = train, x = x, hue = y, stat = "percent", common_norm = False, ax = axs[i])
plt.show()


# <span style="font-size:18px;"> Waiit a second, there is something fishy🐟 with the "ratings_disabled" feature.

# In[35]:


train[train["ratings_disabled"]==True]["target"].unique()


# In[36]:


len(test[test["ratings_disabled"]==True])


# <span style="font-size:18px;"> Niiiice 😃, we can deduce that the videos with ratings disabled have 0 as their targets, we can drop the rows with disabled ratings from our trainset to avoid any kind of bias and predict 0 directly for the rows with disabled ratings in the testset.

# In[37]:


train = train.drop(train[train["ratings_disabled"]==True].index)
test_0 = test[test["ratings_disabled"]==True]
test = test.drop(test[test["ratings_disabled"]==True].index)


# In[38]:


train = train.drop("ratings_disabled", axis = 1)
test = test.drop("ratings_disabled", axis = 1)


# In[39]:


cat = cat.drop("ratings_disabled")


# In[40]:


train[train["target"]==0]


# <span style="font-size:18px;"> 
#     It seems that we don't have any other row with 0 as a target for the remaining of the trainset.<br>
#     Thus we wouldn't have any problem log scaling our right-skewed target (don't worry I know that we could avoid that issue by applying log1p but I just like to mention it😜)

# In[41]:


"""fig, axs = plt.subplots(5,3,figsize = (24,6))
axs = axs.flatten()
for i, c in enumerate(cont):
    sns.histplot(data = train, x = c, stat = "percent",
                 common_norm = False, ax = axs[3*i])
    sns.histplot(x = np.log1p(train[c]), stat = "percent",
                 common_norm = False, ax = axs[3*i+1])
    sns.histplot(x = np.sqrt(train[c]), stat = "percent",
                 common_norm = False, ax = axs[3*i+2])
                 """


# In[42]:


fig, axs = plt.subplots(1,3,figsize = (24,6))
axs = axs.flatten()
c = "target"
i = 0
sns.histplot(data = train, x = c, stat = "percent",
                 common_norm = False, ax = axs[3*i])
sns.histplot(x = np.log1p(train[c]), stat = "percent",
                 common_norm = False, ax = axs[3*i+1])
sns.histplot(x = np.sqrt(train[c]), stat = "percent",
                 common_norm = False, ax = axs[3*i+2])


# <span style="font-size:18px;"> Comparing the results of different transformations for our continuous variables, we can see that the square root gave us the best one for the target meanwhile the logarithm seems like it does a better job for the rest. (Although for some models, the scaling may not matter that much.) <br>
#     And of course we shouldn't forget to apply the inverse of the transformation to the target after doing predictions in the end.
# 

# In[43]:


cont


# In[44]:


for c in cont:
    train[c] = np.log1p(train[c])
    if (c == "duration_seconds"):
        test[c] = np.log1p(test[c])
train["target"] = np.sqrt(train["target"])


# In[45]:


train.groupby(by="has_thumbnail")["target"].agg(["mean", "std"])


# In[46]:


train.groupby(by="comments_disabled")["target"].agg(["mean", "std"]).\
                rename(columns = {"mean":"mean_target_per_comments_disabled",
                                  "std":"std_target_per_comments_disabled"})


# In[47]:


train.groupby(by="categoryId").agg(mean_tar_per_cat = ("target", "mean"),
                                  std_tar_per_cat = ("target", "std"))


# <span style="font-size:18px;"> We can see that the "has_thumbnail" doesn't affect the target that much despite its imbalance unlike the other categorical variables, so it may be a good idea to drop it.

# In[48]:


train = train.drop("has_thumbnail", axis = 1)
test = test.drop("has_thumbnail", axis = 1)
cat = cat.drop("has_thumbnail")


# In[49]:


def groupby_encoding(train, test, encode, by, strategy, folds):
    """
    This function can encode various variables in your train and test sets using values in other variables
    by given strategies.
    train : DataFrame containing the trainset, on which cross-validation will be used.
    test : DataFrame containing the testset, on which the learnt encoding will be applied.
    encode : Iterable of variables to be encoded.
    by : Iterable of variables to be used during encoding.
    strategy : Iterable of strategies to be used for aggregation during the encoding.
    folds : Number of folds for the cross-validation.
    """
    kf = KFold(n_splits=5,random_state=48,shuffle=True)
    for x in encode:
        for y in by:
            new_train = train.copy()
            new_test = test.copy()
            for k, (trn_idx, test_idx) in enumerate(kf.split(train)):
                tr = train.iloc[trn_idx]
                df = tr.groupby(by=x)[y].agg(strategy).\
                        rename(columns = {s: s+"_"+y+"_per_"+x for s in strategy})
                new_train.loc[new_train.index[test_idx],df.columns] = train.join(df, on = x).iloc[test_idx][df.columns]
                if (k==0):
                    new_test = test.join(df/folds, on = x)
                else:
                    new_test[df.columns] = new_test[df.columns] + test.join(df/folds, on = x)[df.columns]
            train = new_train.copy()
            test = new_test.copy()
    return train, test


# In[50]:


train_copy = train.copy()
test_copy = test.copy()
encode = cat
by = cont
strategy = ["mean", "std"]
folds = 5
new_train, new_test = groupby_encoding(train_copy, test_copy, encode, by, strategy, folds)


# In[51]:


train = new_train
test = new_test


# <span style="font-size:18px;">So what I have done in the previous cells was to encode our categorical variables using the available continuous variables by different groupings, and I've applied it through cross-validation to avoid any overfitting issues.<br>
#     In addition, I haven't used the target in this encoding step as it is already described by the 'likes' and the 'view_count' variables.

# In[52]:


grouping_variables = [i for i in train.columns if '_per_' in i]


# <span style="font-size:18px;">Now I can drop the additional training features ('view_count', 'likes', 'comment_count' and 'dislikes) :

# In[53]:


train.columns.difference(test.columns).difference(["target"])


# In[54]:


train = train.drop(columns = train.columns.difference(test.columns).difference(["target"]))


# In[55]:


print("The shape of the trainset is : ",train.shape)
print("The shape of the testset is : ",test.shape)


# ## Dealing with Missing Values

# In[56]:


train.isnull().sum(axis=0)[train.isnull().sum(axis=0)!=0]*100/train.shape[0]


# In[57]:


test.isnull().sum(axis=0)[test.isnull().sum(axis=0)!=0]*100/test.shape[0]


# <span style="font-size:18px;">Fortunately, we don't really have that mcu missing values in this dataset, so I don't think the imputation technique would make that much of a difference.<br>
#     The one that I will be using is LGBM Imputer.<br>
#     You can find more details on imputation techniques in this notebook : https://www.kaggle.com/robikscube/handling-with-missing-data-youtube-stream

# In[58]:


# !rm -r kuma_utils
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')


# In[59]:


import sys
sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer


# In[60]:


cat


# In[61]:


train_light = train.copy()
test_light = test.copy()


# In[62]:


get_ipython().run_cell_magic('time', '', 'lgbm_imtr = LGBMImputer(cat_features = cat, n_iter=100, verbose=True)\n\ntrain_lgbmimp = pd.DataFrame(lgbm_imtr.fit_transform(train_light[train.columns.difference(["id", "target"])]))\ntest_lgbmimp = pd.DataFrame(lgbm_imtr.transform(test_light[test.columns.difference(["id"])]))\n')


# In[63]:


fig, axes = plt.subplots(1,2, figsize = (12,6))
axes = axes.flatten()
sns.histplot(x=train["duration_seconds"], ax = axes[0])
sns.histplot(train_lgbmimp["duration_seconds"], ax = axes[1])


# In[64]:


fig, axes = plt.subplots(1,2, figsize = (12,6))
axes = axes.flatten()
sns.histplot(x=train["len_description"], ax = axes[0])
sns.histplot(train_lgbmimp["len_description"], ax = axes[1])


# In[65]:


fig, axes = plt.subplots(1,2, figsize = (12,6))
axes = axes.flatten()
sns.histplot(x=train["upper_percent_description"], ax = axes[0])
sns.histplot(train_lgbmimp["upper_percent_description"], ax = axes[1])


# <span style="font-size:18px;">Great 😀, as we can see the imputer didn't change our distributions that much.

# # Conclution

# <span style="font-size:18px;">Coming to the end of this notebook, I tried to leave no stone unturned in terms of our data. Whether it is a date or a text or an ordinary variable, I explored the data and did all the preprocessing that I've seen necessary.<br>
#     If there is anything unclear or you have an advice you are all welcome to write it down in the comments. And if you've found the notebook useful, an upvote will be much appreciated 😁.<br>
#     In my next notebook, I'll be looking for the best model to fit this data while using Optuna to find the best hyperparameters. Make sure to stay tuned for it.
