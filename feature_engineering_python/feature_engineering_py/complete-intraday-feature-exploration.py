#!/usr/bin/env python
# coding: utf-8

# # Intraday feature exploration - feature engineering
# 
# As discussed [here](https://www.kaggle.com/c/jane-street-market-prediction/discussion/207709), feature 41-45, being constant for each 'stock' trough the day can help make the links between opportunities that regard the same stock appears. (We don't know if they are actually stocks but it's probably clearer than 'underlyings'). Isolating one stock in one given day we can give a hard look at all the available features, without being parasited by the other stocks.
# 
# There seems to be multiple main results in this notebook:
# - Once you isolate one stock some time series start to appear, allowing us to use a whole lot of new tools
# - Most features seems to be redundant in a way or another at the daily x stock level (It is still a bit unclear as to why they are not at a higher level)
# - Some features exhibit very weird patterns so that we should be able to devellop some original engineered features
# 
# The main drawback of this approach is that we have 500 days and around 700 'stocks' a day. So the conclusions we might draw from one stock over one day might not hold, especially if we do our exploration work on underlyings with higher number of opportunities... This notebooks might require a lot of additional work to explore the data set. (Well to be honest this is the main reason why I share the notebook: I can't possibly explore everything by myself, so I hope some of you will try it on other stocks / days and share their findings).
# 
# Once you have read the notebook feel free to fork it and run it for another stock or another day and share your results if they are significantly different. And if you don't want to, that's ok as long as you don't forget to upvote 😜. If you still have time and want to go deeper you might want to check my other notebooks (If you want to go further, you can check my other works (about [Running algos for fast inference](https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference),[Target Engineering](https://www.kaggle.com/lucasmorin/target-engineering-patterns-denoising), and [using yfinance to download financial data in Ptyhon](https://www.kaggle.com/lucasmorin/downloading-market-data)). Feel free to upvote / share those too.
# Lucas
# 
# Best,
# Lucas

# ## Features
# 
# - [Main features](#Main_features)
# - [Features 1-2](#features_1_2)
# - [Features 3-8](#features_3_8)
# - [Features 9-16](#features_9_16)
# - [Features 17-38](#features_17_38)
# - [Features 39-40](#features_39_40)
# - [Features 41-45](#features_41_45)
# - [Features 46-54](#features_46_54)
# - [Features 55-59](#features_55_59)
# - [Features 60-68](#features_60_68)
# - [Features 69-71](#features_69_71)
# - [Features 72-119](#features_72_119)
# - [Features 120-129](#features_120_129)

# # Loading base packages
# 
# Nothing fancy here : just basic packages and removal of annoying warnings

# In[1]:


import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# # Loading data 
# 
# Loading a pickle file. Check this notebook [pickling](https://www.kaggle.com/quillio/pickling) if you haven't pickled your data set yet. Check this notebook [one liner to halve your memory usage](https://www.kaggle.com/jorijnsmit/one-liner-to-halve-your-memory-usage) if you want to reduce memory usage before pickling.

# In[2]:


get_ipython().run_cell_magic('time', '', "train_pickle_file = '/kaggle/input/pickling/train.csv.pandas.pickle'\ntrain_data = pickle.load(open(train_pickle_file, 'rb'))\ntrain_data.info()\n")


# # Building Target
# 
# Nothing fancy here. I try to keep the raw result (resp * weight) as it might be interesting.

# In[3]:


train_data['action'] = (train_data['resp'].values > 0).astype('int')
train_data['profit'] = train_data['resp'].multiply(train_data['weight'])


# <a id='number_trades'></a>
# # Study the number of trades a day
# For each date, isolate the number of unique values for feature 41.

# In[4]:


plt.plot(train_data.groupby('date').feature_41.nunique())


# So we have approximately 700 unique underlyings a day. It is interesting to note that on a given day, the repartition is usually quite unbalanced: on day 0, 82 stocks make for half the opportunities.

# In[5]:


nb_opportunities = train_data.query('date==0').feature_41.shape[0]
count_f41 = train_data.query('date==0').feature_41.value_counts()
nb_stock = count_f41.values.shape[0]
print('number of opportunities: ' + str(nb_opportunities))
print('number of stocks: ' + str(nb_stock))
print('number of stocks accounting for at least 50% of trades: ' + str(np.argmax(count_f41.cumsum().values/nb_opportunities > 0.5)))
plt.plot(count_f41.cumsum().values/nb_opportunities)


# # Isolate the most common underlying of the first day
# You can esaily change the date and the number of the stock if you want to participate in the exploration effort. 

# In[6]:


nth_day = 0
train_data_fd = train_data.query('date=='+str(nth_day)).copy()

nth_most_common = 0
value = train_data_fd.feature_41.value_counts().index[nth_most_common]
df_md = train_data_fd[train_data_fd.feature_41 == value]


# Some plotting function that will make our life easier. The color show if we need to take action or not, the marker reflect feature 0.

# In[7]:


def plot_cols(data, columns):
    for i in columns:
        fig, ax = plt.subplots()
        colors = {0:'red', 1:'blue'}
        markers = {-1:'x', 1:'o'}

        x = data.index
        y = data[i]
        c = data['action'].map(colors)
        m = data['feature_0'].map(markers)

        unique_markers = set(m)  # or yo can use: np.unique(m)

        for um in unique_markers:
            mask = m == um 
            # mask is now an array of booleans that can be used for indexing  
            ax.scatter(x[mask], y[mask], c=c[mask], marker=um)
            ax.set_title(str(i))

        plt.show()


# In[8]:


#plot_cols(df_md, df_md.columns)


# <a id='Main_features'></a>
# # Main features

# In[9]:


plot_cols(df_md, ['date','weight','resp_1','resp_2','resp_3','resp_4','resp','ts_id','action','profit'])


# Nothing too surprising here. But already see that feature_0 have a lot of importance for the problem. so we define a second dataframe multiplied by feature 0 (except for the main data so as not to get weird results).

# In[10]:


df_md_f0 = df_md.multiply(df_md['feature_0'], axis=0)
df_md_f0['action'] = df_md['action']
df_md_f0['ts_id'] = df_md['ts_id']
df_md_f0['feature_0'] = df_md['feature_0']
df_md_f0['weight'] = df_md['weight']
df_md_f0['profit'] = df_md['profit']


# In[11]:


plot_cols(df_md_f0, ['resp_1','resp_2','resp_3','resp_4','resp'])


# Already seems like a significant improvement and a step towards the usage of time series.

# <a id='features_1_2'></a>
# ### Feature 1 and 2

# In[12]:


plot_cols(df_md, ['feature_1','feature_2'])


# They look oddly similar. Let's see what a scatter plot look like. (defining a fonction might be useful here)

# In[13]:


def plot_scatter(data, columns1, columns2):

    fig, ax = plt.subplots()
    colors = {0:'red', 1:'blue'}
    markers = {-1:'x', 1:'o'}

    x = data[columns1]
    y = data[columns2]
    c = data['action'].map(colors)
    m = data['feature_0'].map(markers)

    unique_markers = set(m)  # or yo can use: np.unique(m)

    for um in unique_markers:
        mask = m == um 
        # mask is now an array of booleans that can be used for indexing  
        ax.scatter(x[mask], y[mask], c=c[mask], marker=um)
        ax.set_title(columns2 + ' v.s. ' + columns1)

    plt.show()


# In[14]:


plot_scatter(df_md, 'feature_1', 'feature_2')


# hu. weird. Maybe the ratio would be a good feature.

# In[15]:


plt.plot((df_md['feature_2'].divide(df_md['feature_1']).clip(-5,5)))


# It seems a bit bad because we have very small values around 0 that create spikes (or maybe I ma doing the division wrong).
# In this discussion https://www.kaggle.com/c/jane-street-market-prediction/discussion/214321 LinDada proposes the following modification :

# In[16]:


plt.plot((df_md['feature_2'].divide(df_md['feature_1']+1e-5).clip(-5,5)))


# <a id='features_3_8'></a>
# ### Feature 3 to 8

# In[17]:


min_f = 3
max_f = 8

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# Oddly symetrical with respect to feature 0.

# In[18]:


min_f = 3
max_f = 8

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md_f0, f)


# Looks like we could do time series here. Very similar features both in terms of normal feature and features multiplied by f0 : 

# In[19]:


plot_scatter(df_md, 'feature_3', 'feature_4')
plot_scatter(df_md, 'feature_3', 'feature_5')
plot_scatter(df_md, 'feature_3', 'feature_6')


# In[20]:


plot_scatter(df_md_f0, 'feature_3', 'feature_4')
plot_scatter(df_md_f0, 'feature_3', 'feature_5')
plot_scatter(df_md_f0, 'feature_3', 'feature_6')


# In[21]:


plot_scatter(df_md, 'feature_7', 'feature_8')
plot_scatter(df_md_f0, 'feature_7', 'feature_8')


# <a id='features_9_16'></a>
# ### Feature 9 to 16

# In[22]:


min_f = 9
max_f = 16

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# No obvious symettry here, but they all look pretty similar :

# In[23]:


plot_scatter(df_md, 'feature_9', 'feature_10')
plot_scatter(df_md, 'feature_11', 'feature_12')
plot_scatter(df_md, 'feature_13', 'feature_14')
plot_scatter(df_md, 'feature_15', 'feature_16')


# Not sure what to do with that. ratios ?

# <a id='features_17_38'></a>
# ### Feature 17 to 38

# In[24]:


min_f = 17
max_f = 38

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# Very much symmetric trough feature 0 :

# In[25]:


min_f = 17
max_f = 38

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md_f0, f)


# In[26]:


for i in range(17, 39, 2):
    plot_scatter(df_md, 'feature_'+str(i), 'feature_'+str(i+1))


# Still not sure what to do with those relationships.

# <a id='features_39_40'></a>
# ### Feature 39 to 40
# 
# two odds features that don't resemble previous or next ones. We just got ou of feature_6 tag.

# In[27]:


min_f = 39
max_f = 40

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# pretty much symmetric in f_0:

# In[28]:


min_f = 39
max_f = 40

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md_f0, f)


# not surprisingly they are similare to one another :

# In[29]:


plot_scatter(df_md, 'feature_39', 'feature_40')


# More surprinsingly they seems to be really close to something we already saw : feature 3 to 6 :

# In[30]:


plot_scatter(df_md, 'feature_39', 'feature_3')
plot_scatter(df_md, 'feature_40', 'feature_3')


# (This doesn't seems to hold at all when changing day or stock wtf ?)

# <a id='features_41_45'></a>
# ### feature 41 to 45
# Those feature are mostly constant over the day. Those are the ones that are used to find stock intraday.
# There is not much to do with those intra-day. But there are still some questions to be answered imo.
# Notably, if there are some discrepancies between them. Or maybe their ratio ?

# In[31]:


min_f = 41
max_f = 45

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# So not really usefull intra-day. We might want to keep an eye on some discrepencies. 

# <a id='features_46_54'></a>
# ### Feature 46 to 54

# Those feature look quite similar. May be with some exception for feature 51 - 52. Let's have a look at them.

# In[32]:


min_f = 46
max_f = 54

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# No obvious symmetry overall. 

# In[33]:


plot_scatter(df_md, 'feature_46', 'feature_47')
plot_scatter(df_md, 'feature_46', 'feature_48')
plot_scatter(df_md, 'feature_46', 'feature_49')
plot_scatter(df_md, 'feature_46', 'feature_50')


# There are two notable execeptions :
# - Features 51 looks like the others but drunk. I suspect that feature 51 is feature 50 multiplied by something. Would that something make sense ?
# - feature 52 look like symmetrized. Can we find its relationship to others ?

# In[34]:


plot_scatter(df_md, 'feature_50', 'feature_51')


# In[35]:


df_md['feature_51-50']=df_md['feature_51']-df_md['feature_50']
plot_cols(df_md, ['feature_51-50'])


# the difference is not clear ... yet ?

# In[36]:


plot_scatter(df_md, 'feature_46', 'feature_52')


# There seems to be some pattern but not sure what.

# There seems to be some link between feature 53, 54 and the previous ones. 53 and 54 oblivously have a link between them.

# In[37]:


plot_scatter(df_md, 'feature_46', 'feature_53')
plot_scatter(df_md, 'feature_53', 'feature_54')


# <a id='features_55_59'></a>
# ### Feature 55 - 59
# Things start to get weird white feature 55.

# In[38]:


plot_cols(df_md,['feature_55'])


# So uh, just like that a well behaved time series ?
# There seems to be some link with feature 56 to 59, that appears to be some sort of times series too.

# In[39]:


min_f = 56
max_f = 59

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# In[40]:


plot_scatter(df_md, 'feature_55', 'feature_56')
plot_scatter(df_md, 'feature_55', 'feature_57')
plot_scatter(df_md, 'feature_55', 'feature_58')
plot_scatter(df_md, 'feature_55', 'feature_59')


# Not very clear what we are dealing with. But as feature_55 is a time series. I suspect that the pattern we can see in feature 57, 58 of trails are indicative of some sort of lag. See below a lagged feature 55 against itself that exhibit similar trails than the plot of 55 v.s. 57. The main problem is that the lag used is probably expressed in real time and not in ticks. I haven't found a proper way to lag variables acocunting for real time (feature 64 ?). Feel free to comment if you know something that work properly.

# In[41]:


plt.scatter(df_md['feature_55'],df_md['feature_55'].rolling(10).mean(), c = df_md['action'].map({0:'red', 1:'blue'}))


# <a id='features_60_68'></a>
# ### Feature 60 to 68
# Things are getting weirder with feature 60 to 63. They have this stepwise aspect. Their proximity with feature 64 made me wonder if they are time related but I can't see anything obvious. Notably because their behavior seems to change when stocks are changing.
# Feature 64 is believed to be time. Let's see if we can see something more.

# In[42]:


min_f = 60
max_f = 63

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# Definitely weird. Maybe they are indicating periods of the day. I suspect we might be interested in jumps.

# In[43]:


df_md['feature_60_jump'] = df_md['feature_60'].diff()
plot_cols(df_md, ['feature_60_jump'])


# We can see (between index) 1500 and 1750 that some jump are not hard jumps. Definitely need to be investigated.

# The features are related, the difference (or ratio) is unclear, but seems to be interesting for feature 60-61, a bit less for feature 62-63 as it only take two values.
# Maybe they should be investgated trough mutliple days.

# In[44]:


plot_scatter(df_md, 'feature_60', 'feature_61')
plot_scatter(df_md, 'feature_62', 'feature_63')


# In[45]:


df_md['feature_61d60']=df_md['feature_61'].divide(df_md['feature_60'])
plot_cols(df_md, ['feature_61d60'])


# In[46]:


df_md['feature_63d62']=df_md['feature_63'].divide(df_md['feature_62'])
plot_cols(df_md, ['feature_63d62'])


# 63/62 do not appear interesting, but 61/60 appears to be. 
# 
# 
# Note : this happens to depend on the day. For day 1 instead of day 0 the role of 60 and 61 are inversed with 62 and 63.

# <a id='features_64'></a>
# ### feature 64 
# Finally one that is pretty clear and somewhat well behaved.

# In[47]:


plot_cols(df_md,['feature_64'])


# is it related to feature 60 - feature 63 ? I am under the impression that the first group of point trought the feature 64 correspond to the one in feature 62-63.

# In[48]:


plot_scatter(df_md, 'feature_64', 'feature_60')
plot_scatter(df_md, 'feature_64', 'feature_61')
plot_scatter(df_md, 'feature_64', 'feature_62')
plot_scatter(df_md, 'feature_64', 'feature_63')


# Maybe feature 60-63 are group of time, I suspect they indicate fairly different market conditions such as pre-market (feature 62-63). Maybe feature 60-61 are the complete day cycle in term of market phase like [pre market](https://www.investopedia.com/terms/p/premarket.asp) /[after hours trading](https://www.investopedia.com/terms/a/afterhourstrading.asp) /[extended session](https://www.investopedia.com/terms/e/extended_trading.asp).

# Feature 65 to 68 also seems related to time. Relationship between 67, 68 are pretty direct to 64. Now I wonder what would be the difference:

# In[49]:


plot_cols(df_md, ['feature_64','feature_67','feature_68'])


# Main difference seems to be a single trade... what would that mean ?

# In[50]:


df_md['feature_68-64']=df_md['feature_68']-df_md['feature_64']
plot_cols(df_md, ['feature_68-64'])


# There seems to be more to it than a single trade ...
# Now into the very weird feature 65 and 66 :

# In[51]:


min_f = 65
max_f = 66

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# In[52]:


plot_scatter(df_md, 'feature_66', 'feature_65')


# They are definitely related. It seems that only one point (in blue this time) is outside from the line. What could that be ?

# In[53]:


df_md['feature_66d65']=df_md['feature_66'].divide(df_md['feature_65'])
plot_cols(df_md, ['feature_66d65'])


# It seems that feature 65 - 66 are time modulo something, like removing the hour from the time of the day ? It seems weird that it take negative values. It feels like we removed something constant by parts from feature 64. Feature 60 maybe ?

# In[54]:


df_md['feature_64-60']=df_md['feature_64']-df_md['feature_60']
plot_cols(df_md, ['feature_64-60'])
plot_cols(df_md, ['feature_65'])


# That definitely look like feature 65, but the relationship doesn't seems obvious as of now :

# In[55]:


plot_scatter(df_md, 'feature_64-60', 'feature_65')


# (Note : as of now the relationship appears to be mixed when changing day : when using day one instead, it seems that the role of some features are inverted between which ones are the 'stepwise' and which one are the 'minutes').

# <a id='features_69_71'></a>
# ### Feature 69,70,71
# Now we are encountering some features that are hard to interpret. Those features make me especially curious as most features appears by group of 2, not of 3. Feature 69,70,71 appears to be linked to feature 53 and 54 trough their tag. Let's investigate that.

# In[56]:


min_f = 69
max_f = 71

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# In[57]:


min_f = 53
max_f = 54

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# In[58]:


plot_scatter(df_md, 'feature_53', 'feature_69')
plot_scatter(df_md, 'feature_53', 'feature_70')
plot_scatter(df_md, 'feature_53', 'feature_71')

plot_scatter(df_md, 'feature_54', 'feature_69')
plot_scatter(df_md, 'feature_54', 'feature_70')
plot_scatter(df_md, 'feature_54', 'feature_71')


# Not sure what we can see from the scatter plots, but we can make one observation about range of feature 69,70,71 (basically the range of 69 is twice of those of 70-71), so I'll try for the sum and difference:

# In[59]:


df_md['feature_71-70']=df_md['feature_71']-df_md['feature_70']
plot_cols(df_md, ['feature_71-70'])


# In[60]:


plt.scatter(df_md['feature_71-70'],df_md['feature_69'], c = df_md['action'].map({0:'red', 1:'blue'}))


# Yeah there is probably some link, we are definitely onto something, but a bit diffcult to interpret. now the sum : 

# In[61]:


df_md['feature_71+70']=df_md['feature_71']+df_md['feature_70']
plot_cols(df_md, ['feature_71+70'])


# May be that how they relate to feature 53 and 54 ?

# In[62]:


plt.scatter(df_md['feature_71+70'],df_md['feature_54'], c = df_md['action'].map({0:'red', 1:'blue'}))


# No luck this time.
# Note : these realtionships appears to be changing heavily over stocks / days. 

# <a id='features_72_119'></a>
# ### Feature 72 to 119
# That a good amount of features. From tag it seems that there is a 6 features repeating pattern. We will try to investigate that pattern on feature 72 to 77 and apply it to other features.

# In[63]:


min_f = 72
max_f = 77

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# That look like different time series starting from 77 to 72 and others... maybe if we shuffle a bit the order ?

# In[64]:


plot_cols(df_md, ['feature_77','feature_72','feature_76','feature_75','feature_74','feature_73'])


# It seems to be a time series features that slowly delves into chaos. It seems a bit curious because it en up being quite symmetric.

# In[65]:


a = (df_md['action'] -0.5)*2

df_md_a = df_md.multiply(a, axis=0)
df_md_a['action'] = df_md['action']
df_md_a['ts_id'] = df_md['ts_id']
df_md_a['feature_0'] = df_md['feature_0']
df_md_a['weight'] = df_md['weight']
df_md_a['profit'] = df_md['profit']

plot_cols(df_md_a, ['feature_77','feature_72','feature_76','feature_75','feature_74','feature_73'])


# There is definitely some link, but it is unclear what : 

# In[66]:


plot_scatter(df_md, 'feature_72', 'feature_77')
plot_scatter(df_md, 'feature_73', 'feature_76')


# Seems like some smoothing or time difference. Pretty unconclusive if you ask me. Maybe we should look intra tags 24-28. That is we have tag 27 that group features 72 to 83. So le's look at feature 78 to 83 and compare it to 72 to 77. 

# In[67]:


min_f = 72
max_f = 77

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# In[68]:


min_f = 78
max_f = 83

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# pretty much the same thing if you ask me. Main point is that we might use some time series tools here.
# 
# Let's have a look at corresponding values (feature number +6) :

# In[69]:


for i in range(72,78):
    plot_scatter(df_md, 'feature_'+str(i), 'feature_'+str(i+6))


# yay ! back to weird lines we don't really know how to deal with.

# The other link we have are trough tag 0-5, basically feature 72 is linked with features 72,78,84,90,96,102,108,114. We might even add 120 and 121 that do not follow the previsou 6 patter but are on a 12 patterns trough tag 0-5.

# In[70]:


plot_cols(df_md, ['feature_72','feature_78','feature_84','feature_90','feature_102','feature_108','feature_114','feature_120','feature_121'])


# In[71]:


for i in range(72,114,6):
     plot_scatter(df_md, 'feature_'+str(i), 'feature_'+str(i+6))


# So we have an alternating pattern of straight line and more chaotic ones, depending on tags. I think we need to investigate, notably if those pattern hold for other days / stocks.

# <a id='features_120_129'></a>
# ### feature 120 to 129
# Finally some more interesting thing to appears are features at the end.
# 
# Naturally they are linked to one another. And their difference patterns appears to be interesting.

# In[72]:


min_f = 120
max_f = 129

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md, f)


# They appear to have very 'symmetric' behavior towards features 0 (look at dots versus crosses). So we can have a look at the features multiplied by feature_0.

# In[73]:


min_f = 120
max_f = 129

f = ['feature_'+str(i) for i in range(min_f,max_f+1)]

plot_cols(df_md_f0, f)


# In[74]:


plot_scatter(df_md, 'feature_120', 'feature_121')
df_md['feature_121-120']=df_md['feature_121']-df_md['feature_120']
plot_cols(df_md, ['feature_121-120'])


# In[75]:


plot_scatter(df_md, 'feature_122', 'feature_123')
df_md['feature_123-122']=df_md['feature_123']-df_md['feature_122']
plot_cols(df_md, ['feature_123-122'])


# In[76]:


plot_scatter(df_md, 'feature_124', 'feature_125')
df_md['feature_125-124']=df_md['feature_125']-df_md['feature_124']
plot_cols(df_md, ['feature_125-124'])


# In[77]:


plot_scatter(df_md, 'feature_126', 'feature_127')
df_md['feature_127-126']=df_md['feature_127']-df_md['feature_126']
plot_cols(df_md, ['feature_127-126'])


# In[78]:


plot_scatter(df_md, 'feature_128', 'feature_129')
df_md['feature_129-128']=df_md['feature_129']-df_md['feature_128']
plot_cols(df_md, ['feature_129-128'])


# there even seems to be some pattern when comparing the difference :

# In[79]:


plt.scatter(df_md['feature_129-128'],df_md['feature_127-126'], c = df_md['action'].map({0:'red', 1:'blue'}))


# <a id='Conclusion'></a>
# ### Conclusion
# 
# Congrats for reaching the end of the notebook ! 
# 
# There seems to be a lot of patterns in the data. If we undertsand them we might be able to build better features, then get better scores. 
# I am specifficaly intrigued by the multiples 'lines' we can see above. How would they matter ? are the difference between consecutives values informatives ? or are they some artefact of an anonymisation process ? It seems that when changing the day / stock we get different lines. What info lies in the evolution of the form of the lines trough the days ?
# 
# I think we should probably start by looking at other days and stocks. Feel free to pick a day at random, some common stock and look if the relationships holds.
