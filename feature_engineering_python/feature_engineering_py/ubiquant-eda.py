#!/usr/bin/env python
# coding: utf-8

# <img src="https://media-exp1.licdn.com/dms/image/C4E1BAQEigPyX9PIjFA/company-background_10000/0/1646016950408?e=2147483647&v=beta&t=Bd6r8nEG57vbf-KEFtqZxPr9rSlRUdZf5ZgAfEN44-s" width="700"><br>
# [Image Source](https://media-exp1.licdn.com/dms/image/C511BAQENQ8r6ZpXxEw/company-background_10000/0/1554262004191?e=2159024400&v=beta&t=1dQus0FUOa_dUfJi4zaMrR2VjPWspepFP3nnM_XCD-I)
# 
# # Ubiquant Market Prediction Dataset
#     
# ## Contents
# 1. [Overview](#1.-Overview)<br>
# 2. [The Data](#2.-The-Data)<br>
#     2.1 [time_id, row_id and investment_id](#2.1.-time_id,-row_id-and-investment_id)<br>
#     2.2 [Target](#2.2.-Target)<br>
#     2.3 [Anonymized Features](#2.3.-Anonymized-Features)<br>
# 3. [Model - TBD](#3.-Model)<br>   
# 4. [Results and Conclusion - TBD](#4.-Results-and-Conclusion)<br>
# 
# ***Please remember to upvote if you find this Notebook helpful!***

# # 1. Overview
# Ubiquant Investment (Beijing) Co., Ltd is a leading domestic quantitative hedge fund based in China. Established in 2012, they rely on international talents in math and computer science along with cutting-edge technology to drive quantitative financial market investment. Overall, Ubiquant is committed to creating long-term stable returns for investors.
# 
# In this competition, the goal is to build a time-series model that forecasts an investment's return rate. The dataset contains historical prices to train and test our models.
# 
# The dataset contains 300 features to describe the samples, however, they are all anonymized. Below is a summary of the dataset:
# 
# * **time_id** - The ID code for the time the data was gathered. The time IDs are in order, but the real-time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.
# * **investment_id** - The ID code for an investment. Not all investments have data in all-time IDs.
# * **target** - The target.
# * **f_0 to f_299** - Anonymized features generated from market data.

# In[1]:


import numpy as np
import pandas as pd
import gc
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def basic_EDA(df):
    size = df.shape
    sum_null = df.isnull().sum().sum()
    return print("Number of Samples: %d,\nNumber of Features: %d,\nNull Entries: %d" %(size[0],size[1], sum_null))


# # 2. The Data
# 
# First we start with an overview of the Dataset. Due to the dataset size, we will use the Parquet version of the data for faster outputs compared to the **.csv** version. We appreciate @Rob Mulla for the creation of this version of the dataset [Ubiquant Parquet](https://www.kaggle.com/datasets/robikscube/ubiquant-parquet).
# 
# Now we import the data and have a look at the first rows of the dataset:
# 

# In[3]:


df = pd.read_parquet('../input/ubiquant-parquet/train.parquet')
df.head()


# In[4]:


basic_EDA(df)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>First impressions / Ideas</h3>
# <ul>
# <li>Huge dataset - analysis will have to be performed on samples
# <li>No Null Values - This makes life easier
# <li>Anonymized Features - difficult to have insights from this kind of data. However, it is always a good opportunity to research some feature engineering methods    
# <li>Understand time_id and investment_id
# <li>Focus on Target Feature analysis 
# <li>Plot some of the anonymized features against the target and time_id to see any patterns
# <ul>
# </blockquote>

# # 2.1. time_id, row_id and investment_id
# It is possible to see that for each time_id we have several entries. It is not clear if each time_id is a day or an hour at this point. Let's check:
# 
# * How many unique time_id's are present in the dataset
# * How many entries for each time_id? Does it vary largely according to the time_id?
# * Are investment_id related to different row_id or time_id samples?

# In[5]:


print("Unique time_id's: ",len(df['time_id'].unique()))


# There are only 1211 unique time_id's. A relatively small number compared to the number of samples. Now let's understand how many entries (samples) we have for each time_id.

# In[6]:


time_ids = df.groupby('time_id').size()
plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.lineplot(data=time_ids, x=time_ids.index, y=time_ids.values)
sns.despine(top=True, right=True, left=False, bottom=False)

ax.set_xlabel('time_id',fontsize = 14,weight = 'bold')
ax.set_ylabel('Count of row_id',fontsize = 14,weight = 'bold')
plt.title('Number of row_ids for Each time_id Entry',fontsize = 16,weight = 'bold');
plt.grid()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>The number of entries for each time_id shows an upward trend, it is not clear why this number would raise over time;
# <li>Notice the big drops in the number of entries between time_id 350 to 500. This pattern will probably make the predictions more challening in this period due to the smaller number of samples and odd behaviour;
# <li>The drastic drop in the number of entries in these specific "date" range could be some kind of Holiday period or some time of the day where only some particular stock markets are open;
# <li> After time_id 600 the number of entries seems to stabilize to at least 2000 entries per time_id. There are periods where the number of row_id drop as low as 1500
# <ul>
# </blockquote>
# 
# Now we have a look at the **investment_id** feature. 

# In[7]:


inv_id = df.groupby(['investment_id']).size()

print("Number of unique investment_id's: ",len(df['investment_id'].unique()))
print("Index of investment_id with most entries: %d \nNumber of transactions with this investment_id: %d" % (inv_id.idxmax(), inv_id.values.max()))


# Right, so we have over 3.500 different investment ID's. One investment that we have an entry for every **time_id** is one called **2140**. Let's plot this investment over time:

# In[8]:


mean = round(df[df['investment_id'] == 2140]['target'].mean(), 3)
plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.lineplot(data=df[df['investment_id'] == 2140], x='time_id', y='target')
ax.axhline(mean, ls='--', c = 'b')
ax.text(-50,(0.15 + mean),("Mean \n" + str(mean)),
            bbox=dict(facecolor='white', edgecolor='none'))
sns.despine(top=True, right=True, left=False, bottom=False)

ax.set_xlabel('time_id',fontsize = 14,weight = 'bold')
ax.set_ylabel('target',fontsize = 14,weight = 'bold')
plt.title('target over time_id for Investment ID 2140',fontsize = 16,weight = 'bold');
plt.grid()


# Since we have a continuous sample for this investment_id, it will probably be slightly easier to predict its behaviour. 
# 
# Now we will look at investments that do not have as many samples. Also, we can use the KDE plot to understand how many investments we usually have by time_id.

# In[9]:


print("Index of investment_id with least number entries: %d \nNumber of transactions with this investment_id: %d" % (inv_id.idxmin(), inv_id.values.min()))


# In[10]:


plt.figure(figsize=(20,5))
ax = sns.kdeplot(data = inv_id, x = inv_id.values, linewidth=1,alpha=.3, fill = True,palette = 'husl') 
ax.set_xlabel('investment_id')
plt.title('KDE Plot - investment_id', fontsize = 16,weight = 'bold',pad=20);  
sns.despine(top=True, right=True, left=False, bottom=False)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>The distribution is Bimodal, displaying two peaks. Indicates a group of investments with a mean of around 400 entries (or time_id's) and another group where we have one entry for almost every time_id;
# <li>This first small peak could be the cause of that severe drop displayed in the Number of row_ids for Each time_id Entry graph;
# <li>Important to keep in mind that there are investments with less than 100 entries, e.g. inv_id 1415 has only two 2 entries. Those might also pose a challenge to predict. One idea for the stocks with a small number of samples, we can find investments that have high similarity to improve prediction performance 
# <ul>
# </blockquote>
# 
# To compare the different groups, let's plot over time some investment_id's of the first group (approx. 400 entries) and others from the group containing more entries (approx. 1000).

# In[11]:


#extracting 3 investment_ids that contain over 1000 time_id entries
inv_id_idx_larger = inv_id[inv_id.values > 1000].sample(n = 3, random_state = 0).index
#extracting 3 investment_ids that contain less than 500 time_id entries
inv_id_idx_smaller = inv_id[(inv_id.values > 350) & (inv_id.values < 450)].sample(n = 3, random_state = 0).index

inv_ids_idx = list(inv_id_idx_larger)
inv_ids_idx.extend(list(inv_id_idx_smaller))


# In[12]:


for inv in inv_ids_idx:
    df_temp = df[df['investment_id'] == inv][['time_id', 'investment_id','target']]
    plt.figure(figsize=(20,4))
    sns.set(style="ticks", font_scale = 1)
    ax = sns.lineplot(data=df_temp, x='time_id', y='target')
    sns.despine(top=True, right=True, left=False, bottom=False)

    ax.set_xlabel('time_id',fontsize = 14,weight = 'bold')
    ax.set_ylabel('target',fontsize = 14,weight = 'bold')
    
    ax.set_xlim(xmin=0)
    plt.title('target over time_id for Investment ID ' + str(inv),fontsize = 16,weight = 'bold');
    plt.grid()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>The investments we selected present a similar profile, with a mean close to zero and comparable min/max values;
# <li>The data present blank intervals even for the investments with a larger number of samples (top three graphs);
# <li>One interesting thing is that now we understand why the graph on row_id and time_id showed that upward trend earlier. New investments are being added later on our time_id timeframe causing the number of row_ids to increase;
# <li>The investments with fewer samples (randomly selected) start around time_id 800. That is probably a coincidence, but something to look at later on
# <ul>
# </blockquote>

# ## 2.2. Target
# 
# As we saw previously, the target is a numerical variable and the Mean, Min and Max values were in a similar range for the investments we plotted in the graph. We use the **.describe()** method and a **KDE plot** just to have an overview of this feature and verify if our initial insights are correct.

# In[13]:


df['target'].describe().apply("{0:.5f}".format)


# In[14]:


plt.figure(figsize=(20,5))
ax = sns.kdeplot(data = df, x = 'target', linewidth=1,alpha=.3, fill = True,palette = 'husl') 
ax.set_xlabel('investment_id')
plt.title('KDE Plot - target', fontsize = 16,weight = 'bold',pad=20);  
sns.despine(top=True, right=True, left=False, bottom=False)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>The good news is that the target variable shows symmetrical shape, i.e. is not overly skewed to one side or the other. However, it does not look like our classic Normal distribution due to the sharp peak. We can test for kurtosis and skewness in the next step. Ideally, we want a normal distribution as our statistical models usually perform best for this kind of distribution;
# <li>Most samples have values around or close to zero, as our previous plots have also shown;
# <li>The Min and Max values are a bit far from the 25% and 75% percentiles. We can have a better look at these samples and check if we can treat them as outliers. Sometimes smoothing the data can help with the model outcome;
# <ul>
# </blockquote>

# In[15]:


print('Skewness of Target Feature: ', pd.DataFrame(df['target']).skew().values)
print('Kurtosis of Target Feature: ', pd.DataFrame(df['target']).kurtosis().values)


# > For reference, a **Normal** Distribution presents skewness values close to 0 (perfect symmetry) and kurtosis near the value of 3. Even though our values are not far off from the Normal reference, sometimes transforming the data can be helpful. Something else to keep in mind ... 
# 
# Now let's analyse the data by separating the target values into positive and negative values: 
# * First we create a new Feature called **target_positive**. It will contain the value 1 if the target value is positive, 0 if the value is negative;
# 
# The goal is to understand what is the % between positive and negative target values for each day (or time_id). In a way, we can understand if there is a balance between positive and negative target values. For example, is it 50-50 or do we have a higher percentage of positive outcomes? Is there a specific period where there is a change in this %?

# In[16]:


#creating the new feature
df['target_positive'] = np.where(df.target > 0,1,0)


# In[17]:


df.head()


# In[18]:


inv_pos_out = df.groupby('time_id')['target_positive'].sum() / time_ids.values


# In[19]:


plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.lineplot(data=inv_pos_out, x=inv_pos_out.index, y=inv_pos_out.values, color = 'r')
ax.axhline(0.5, ls='--', c = 'black')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('time_id',fontsize = 14,weight = 'bold')
ax.set_ylabel('Sum target_positive', fontsize = 14,weight = 'bold')
plt.title('% of Positive target Values by time_id',fontsize = 16,weight = 'bold');
plt.grid()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>Overall, the percentage of positive target values seldom surpasses the 50% mark in the graph. As such, there are usually a higher participation of negative target values in a time_id;
# <li>It seems like in the period from time_id 800 to 1200 we have more positive outcomes compared to the initial periods from 0 to 400. See how the number of peaks above 0.5 is a bit denser by the end of the graph;
# <li>Once again we see weird behaviour in the period around 400. There are periods where 80% of positive target outcomes are presented. In contrast, there are also times when we have less than 10% positive values;
# <li>The data is not heavily unbalanced towards negatives values, since we can see a mean of 45% positive against 55% negative for most time_ids. However, the (almost) consistent exceeding number of negative target values over positive target values might result in model bias, i.e. the model might tend to assign predict negative target values;
# <ul>
# </blockquote>

# # 2.3. Anonymized Features
# 
# The features are anonymized so there aren't major insights we can extract from interpreting their behaviour. Here's a list of things we can look at:
# 
# * Does the features show a different distribution for positive / negative target values?
# * Do the features present any type of linear relationship with the target?
# * Are they normally distributed? What is their value range?
# 
# Since there are many features, here is displayed only a few for guidance. 

# In[20]:


#code to extract some random feature names
feat_list = np.random.choice(df.iloc[:,4:-1].columns, size=12, replace=False, p=None)
#feat_list = ['f_169', 'f_254', 'f_239', 'f_210', 'f_83', 'f_281']
df_sample = df.sample(frac = 0.1, random_state = 0)


# In[21]:


i = 0
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,6,figsize=(18,8));

for feature in feat_list:
    i += 1
    plt.subplot(2,6,i)
    sns.kdeplot(x = feature, data = df_sample, hue = 'target_positive')
    plt.xlabel(feature, fontsize=9)
    locs, labels = plt.xticks()
    plt.tick_params(axis='x', which='major', labelsize=6, pad = 0)
    plt.tick_params(axis='y', which='major', labelsize=6, pad = 0)
    plt.ylabel("Density",fontsize=8)
    
fig.tight_layout(pad=3.0)


# In[22]:


i = 0
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,6,figsize=(18,8));

for feature in feat_list:
    i += 1
    plt.subplot(2,6,i)
    sns.scatterplot(y = feature,x = 'target', data = df_sample, s = 10)
    plt.xlabel('Target', fontsize=9)
    plt.ylabel(feature, fontsize=9)
    locs, labels = plt.xticks()
    plt.tick_params(axis='x', which='major', labelsize=6, pad = 0)
    plt.tick_params(axis='y', which='major', labelsize=6, pad = 0)

fig.tight_layout(pad=3.0)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <h3>Insights</h3>
# <ul>
# <li>From the KDE plots we can notice that there is no distinction in the distribution between positive or negative target values. The positive values only show a smaller density, but the same shape;
# <li>The variable values range vary significantly;
# <li>The scatter plots are useful to analyse possible relationships between variables, clusters, data gaps, outliers...As a rule of thumb, round-like scatters usually mean that the variables are not correlated;
# <li>For the features we plotted above, the data points are symmetrical considering the vertical axis;
# <li>The features present high variability in their value range when target values are close to 0, note how most of the scatter has a vertically elongated shape;
# <li>For example, for f_0, f_204 and f_24 we see a trend that target values close to 0 are more likely to have negative values. For f_167, f_123 and f_63 is the exact opposite. For f_5 and f_91 the graph is symmetrical for all quadrants;
# <li>For the analysed features, there's no clear linear (or non-linear) relationship that can be extracted. 
# <ul>
# </blockquote>
#     
# One could also use correlation to examine the LINEAR relationship between the target and feature variables. Usually, features that are highly correlated to the target are useful for the model. However, it does not indicate that lower correlated features should be discarded. 

# ... Thank you for reading so far. Currently working on improving this Notebook. 
# 
# ## Work in Progress ##

# In[ ]:




