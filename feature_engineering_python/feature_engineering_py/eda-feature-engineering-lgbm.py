#!/usr/bin/env python
# coding: utf-8

# --# This is a BEGINNER'S kernel, please go easy on me. Your comments, suggestions and tips will be highly appreciated and if you liked this kernel please upvote. Thank you very much.

# > #  Costa Rican Household : EDA - Feature Selection - Prediction

# <b>Background:</b>
# 
# Many social programs have a hard time making sure the right people are given enough aid. It’s especially tricky when a program focuses on the poorest segment of the population. The world’s poorest typically can’t provide the necessary income and expense records to prove that they qualify.
#  
# Beyond Costa Rica, many countries face this same problem of inaccurately assessing social need. If Kagglers can generate an improvement, the new algorithm could be implemented in other countries around the world.
# 
# To address this problem we will try to find an optimal solution to classify the income group of families based on household attributes attributes like the material of their walls and ceiling, or the assets found in the home to classify them and predict their level of need. 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Exploration</a></span><ul class="toc-item"><li><span><a href="#Missing-values" data-toc-modified-id="Missing-values-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Missing values</a></span><ul class="toc-item"><li><span><a href="#v2a1-column-" data-toc-modified-id="v2a1-column--1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>v2a1 column <a id="v2a1_c"></a></a></span></li><li><span><a href="#v18q1-column" data-toc-modified-id="v18q1-column-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>v18q1 column</a></span></li><li><span><a href="#rez_esc-column" data-toc-modified-id="rez_esc-column-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>rez_esc column</a></span></li><li><span><a href="#meaneduc-column" data-toc-modified-id="meaneduc-column-1.1.4"><span class="toc-item-num">1.1.4&nbsp;&nbsp;</span>meaneduc column</a></span></li><li><span><a href="#SQBmeaned-column" data-toc-modified-id="SQBmeaned-column-1.1.5"><span class="toc-item-num">1.1.5&nbsp;&nbsp;</span>SQBmeaned column</a></span></li><li><span><a href="#Age-column" data-toc-modified-id="Age-column-1.1.6"><span class="toc-item-num">1.1.6&nbsp;&nbsp;</span>Age column</a></span></li></ul></li><li><span><a href="#Uniform-values" data-toc-modified-id="Uniform-values-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Uniform values</a></span></li><li><span><a href="#Variable-analysis" data-toc-modified-id="Variable-analysis-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Variable analysis</a></span><ul class="toc-item"><li><span><a href="#The-target!" data-toc-modified-id="The-target!-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>The target!</a></span></li><li><span><a href="#Ownership-by-income-group" data-toc-modified-id="Ownership-by-income-group-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Ownership by income group</a></span></li><li><span><a href="#Sex-(the-gender)-and-age-by-income-group" data-toc-modified-id="Sex-(the-gender)-and-age-by-income-group-1.3.3"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>Sex (the gender) and age by income group</a></span></li><li><span><a href="#House-quality" data-toc-modified-id="House-quality-1.3.4"><span class="toc-item-num">1.3.4&nbsp;&nbsp;</span>House quality</a></span></li><li><span><a href="#Services" data-toc-modified-id="Services-1.3.5"><span class="toc-item-num">1.3.5&nbsp;&nbsp;</span>Services</a></span></li><li><span><a href="#Overcrowding-problem" data-toc-modified-id="Overcrowding-problem-1.3.6"><span class="toc-item-num">1.3.6&nbsp;&nbsp;</span>Overcrowding problem</a></span></li></ul></li></ul></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Modelling</a></span><ul class="toc-item"><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Feature engineering</a></span></li><li><span><a href="#Feature-selection" data-toc-modified-id="Feature-selection-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Feature selection</a></span></li><li><span><a href="#Feature-Scaling" data-toc-modified-id="Feature-Scaling-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Feature Scaling</a></span></li><li><span><a href="#Sampling-and-splitting" data-toc-modified-id="Sampling-and-splitting-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Sampling and splitting</a></span><ul class="toc-item"><li><span><a href="#XGboost" data-toc-modified-id="XGboost-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>LGBM</a></span></li></ul></li></ul></li><li><span><a href="#Submission" data-toc-modified-id="Submission-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Submission</a></span></li><li><span><a href="#Summary,-conclusion-and-recommendation" data-toc-modified-id="Summary,-conclusion-and-recommendation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary, conclusion and recommendation</a></span></li></ul></div>

# In[ ]:


# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
sns.set()


# ## Data Exploration

# In[ ]:


# load dataset

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# train_df = pd.read_csv('../inputs/Costa Rica household/train.csv')
# test_df = pd.read_csv('../inputs/Costa Rica household/test.csv')


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# For now let us focus on the training data.

# In[ ]:


train_df.columns.values


# In[ ]:


train_df.head()


# That's a lot of column, we may want to remove some of them later

# ### Missing values

# First thing that we would like to do is to apply proper treatment on these missing values.

# In[ ]:


train_df.isnull().sum()[train_df.isnull().sum() > 0]


# In[ ]:


import missingno as msno


# In[ ]:


msno.matrix(train_df[['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']], color = (0.211, 0.215, 0.274))
plt.show()


# Let's do an investigation regarding the missing values from these columns

# #### v2a1 column <a id = 'v2a1_c'></a>

# v2a1 is defined as the monthly rent payment of the household. I will assume that those with missing values of v2a1 already owns the house, let's investigate how many of those missing values already owns the house using the tipovivi1 column.

# In[ ]:


sns.countplot(train_df.loc[(pd.isnull(train_df['v2a1'])), 'tipovivi1'])
plt.title("Ownership")
plt.show()


# Most of the missing monthly rent payment already owns the house, let's impute zero on these rows

# In[ ]:


train_df.loc[(pd.isnull(train_df['v2a1']) & train_df['tipovivi1'] == 1), 'v2a1'] = 0


# Let us remove the remaining missing values for now. 

# In[ ]:


train_df = train_df.loc[pd.notnull(train_df['v2a1'])]


# #### v18q1 column

# v18q1 is defined as the number of tablets household owns. It looks too easy that these missing values are 'zeros', hmmmm just to make sure I will check the value counts for v18q1

# In[ ]:


train_df['v18q1'].dropna().value_counts()


# No zero value, which means the missing values really means zero tablets owned.

# In[ ]:


train_df.loc[(pd.isnull(train_df['v18q1'])), 'v18q1'] = 0


# #### rez_esc column

# rez_esc is defined as the Years behind in school. As usual let's check for the value count

# In[ ]:


train_df['rez_esc'].dropna().value_counts()


# Yikes! There are zero values already. Let's investigate further

# In[ ]:


# statistical measures of those with rez_esc

train_df.loc[pd.notnull(train_df['rez_esc']),('age')].describe()


# In[ ]:


# statistical measures of those with missing rez_esc

train_df.loc[pd.isnull(train_df['rez_esc']),('age')].describe()


# We could infer from the table above that those with 'Years behind school' data are teenagers and kids with ages 7-17 and those with missing values are much older. This may be due to older people can't remember how many years they are behind already. Unfortunately, we may not be able to perform proper treatment on the missing values of this column, so let's drop this column.
# <hr>
# Also, the minimum value of age for those with missing rez_esc is zero. I'll have to deal with that later

# In[ ]:


# train_df.drop(columns='rez_esc', inplace = True)
train_df.loc[pd.isnull(train_df['rez_esc']), 'rez_esc'] = 0


# #### meaneduc column

# meaneduc is defined as average years of education for adults (18+)

# In[ ]:


train_df.loc[pd.isnull(train_df['meaneduc']), ('edjefa', 'edjefe', 'escolari', 'meaneduc')]


# For meaneduc let's just copy escolari (years of education) to the meaneduc

# In[ ]:


train_df.loc[pd.isnull(train_df['meaneduc']), 'meaneduc'] = train_df.loc[pd.isnull(train_df['meaneduc']), 'escolari']


# #### SQBmeaned column

# SQBmeaned is just square of meaneduc

# In[ ]:


train_df.loc[pd.isnull(train_df['SQBmeaned']), 'SQBmeaned'] = train_df.loc[pd.isnull(train_df['SQBmeaned']), 'meaneduc']**2


# #### Age column

# In[ ]:


len(train_df.loc[train_df['age'] == 0].index)


# We could predict the age of this rows, but that would be on a different Kernel. Let's delete these rows for the meantime

# In[ ]:


train_df = train_df.loc[train_df['age']!=0]


# In[ ]:


msno.matrix(train_df[['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']], color = (0.211, 0.215, 0.274))
plt.show()


# We're done imputing missing values here is the summary:
# <ol>
#     <li>We've imputed zero values for v2a1 pertaining that a household does not have a monthly rent because they own the house.</li>
#     <li>We've also imputed zero values for v18q1. Not everyone owns a tablet at their homes.</li>
#     <li>Unfortunately, we've removed rez_vec because it has too much missing values and imputation is not ideal even more so deletion of rows.</li>
#     <li>While exploring rez_vec, we've noticed zero values on the age column, this is not possible so for the meantime we've removed those rows.</li>
#     <li>meaneduc column is imputed by the value of escolari (years of education). And SQBmeaned is computed </li>
# </ol>

# ### Uniform values

# We wan't our variables to be in uniform, meaning if it's numeric, then keep it numeric afterall predictive models only understand numeric values.

# In[ ]:


train_df.groupby('dependency').size()


# You could see above that dependency column contains numerical values and 'no' and 'yes'. <br>
# For the 'no' values let's impute zero, while for 'yes' let's impute the mode or the most frequent value.

# In[ ]:


mode = train_df.loc[(train_df['dependency'] != 'yes') & (train_df['dependency'] != 'no'), 'dependency'].astype(float).mode()
mode


# In[ ]:


def mutate_columns(df):
    df.loc[df['dependency']=='no', 'dependency'] =  np.sqrt(df['SQBdependency'])
    df.loc[df['dependency']=='yes', 'dependency'] = np.sqrt(df['SQBdependency'])
    df['dependency'] = df['dependency'].astype('float16')
    
    #The same applies with edjefe and edjefa EXCEPT that the 'yes' value doesn't make any sense? Does it to you? 
    #Anyway, let's just impute for 'yes' values
    
    df.loc[df['edjefe']=='no', 'edjefe'] = 0
    df.loc[df['edjefe']=='yes', 'edjefe'] = 4
    df['edjefe'] = df['edjefe'].astype('uint8')
    
    df.loc[df['edjefa']=='no', 'edjefa'] = 0
    df.loc[df['edjefa']=='yes', 'edjefa'] = 4
    df['edjefa'] = df['edjefa'].astype('uint8')


# In[ ]:


mutate_columns(train_df)
mutate_columns(test_df)


# In[ ]:





# ### Variable analysis

# #### The target!

# Our target variable is named Target, duhh. Let's explore our target variable

# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='Target', data=train_df, palette="OrRd_r")
plt.xticks([0,1,2,3],['extreme poverty','moderate poverty','vulnerable households','non vulnerable households'])
plt.xlabel('')
plt.ylabel('')
plt.title("Poverty Levels", fontsize = 14)

plt.show()


# There seems to be a big imbalance in our data, we may want to apply sampling later on.

# #### Ownership by income group

# In[ ]:


tdf = train_df[['Target']]
n_train_df = train_df
for col in ['v18q', 'refrig', 'computer', 'television', 'mobilephone', 'v14a']:
    n_train_df[col] = n_train_df[col].astype('category')
dfcat = pd.get_dummies(n_train_df[[ 'v18q', 'refrig', 'computer', 'television', 'mobilephone', 'v14a']])
df_ = pd.concat([dfcat, tdf], axis=1)


# In[ ]:





# In[ ]:


df_ = df_.groupby(['Target']).sum()
df_.reset_index(inplace = True)


# In[ ]:


plt.figure(figsize=(12,4))
groups = ['extreme','moderate','vulnerable','non-vulnerable']

ordered_df = df_.sort_values(by='Target')
my_range=range(1,len(df_.index)+1)


plt.scatter(ordered_df['v18q_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['v18q_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['v18q_1'], xmax=ordered_df['v18q_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Tablet ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['refrig_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['refrig_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['refrig_1'], xmax=ordered_df['refrig_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Refrigerator ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['computer_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['computer_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['computer_1'], xmax=ordered_df['computer_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Computer ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['television_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['television_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['television_1'], xmax=ordered_df['television_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Television ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['mobilephone_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['mobilephone_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['mobilephone_1'], xmax=ordered_df['mobilephone_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Mobile phone ownership", fontsize = 14)
plt.show()


# #### Sex (the gender) and age by income group

# In[ ]:


df_ = train_df[['Target', 'male', 'female', 'age']]
df_.loc[(train_df['male'] == 1), 'sex'] = 'male'
df_.loc[(train_df['female'] == 1), 'sex'] = 'female'

plt.figure(figsize = (10,8))
sns.violinplot(x='Target',y='age', data=df_, hue='sex', split=True)
plt.xticks(np.arange(0,5),groups)
plt.show()


# As your income group goes up, so is your age. <br>
# A lot of data under age 20 for the extreme, moderate, and vulnerable income groups. Non-vulnerable on the other hand contains a lot of data with over 20 years of age.

# In[ ]:


plt.figure(figsize = (15,15))
gs = gridspec.GridSpec(4, 2, hspace=0.3)

plt.subplot(gs[0,0])
g = sns.countplot(train_df['r4h3'], hue=train_df['Target'], color="#3274a1")
plt.title("Total males in a household", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
legend = g.get_legend()
legend.set_title("Income group")
new_labels = ['extreme', 'moderate', 'vulnerable', 'non-vulnerable']
for t, l in zip(legend.texts, new_labels): t.set_text(l)

plt.subplot(gs[0,1])
g = sns.countplot(train_df['r4m3'], hue=train_df['Target'], color="#d32d41")
plt.title("Total females in a household", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
legend = g.get_legend()
legend.set_title("Income group")
new_labels = ['extreme', 'moderate', 'vulnerable', 'non-vulnerable']
for t, l in zip(legend.texts, new_labels): t.set_text(l)

plt.subplot(gs[1,0]) 
sns.countplot(train_df['r4h1'], hue=train_df['Target'], color="#3274a1")
plt.title("Males < 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')


plt.subplot(gs[1,1]) 
sns.countplot(train_df['r4m1'], hue=train_df['Target'], color="#d32d41")
plt.title("Females < 12 years old", fontsize = 14)
plt.xlabel('')
plt.legend('')

plt.subplot(gs[2,0]) 
sns.countplot(train_df['r4h2'], hue=train_df['Target'], color="#3274a1")
plt.title("Males >= 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')

plt.subplot(gs[2,1]) 
sns.countplot(train_df['r4m2'], hue=train_df['Target'], color="#d32d41")
plt.title("Females >= 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')

plt.show()


# #### House quality

# In[ ]:


df_q = train_df[['Target', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']]
df_q.loc[df_q['epared1'] == 1, 'wall'] = 'Bad'
df_q.loc[df_q['epared2'] == 1, 'wall'] = 'Regular'
df_q.loc[df_q['epared3'] == 1, 'wall'] = 'Good'

df_q.loc[df_q['etecho1'] == 1, 'roof'] = 'Bad'
df_q.loc[df_q['etecho2'] == 1, 'roof'] = 'Regular'
df_q.loc[df_q['etecho3'] == 1, 'roof'] = 'Good'

df_q.loc[df_q['eviv1'] == 1, 'floor'] = 'Bad'
df_q.loc[df_q['eviv2'] == 1, 'floor'] = 'Regular'
df_q.loc[df_q['eviv3'] == 1, 'floor'] = 'Good'

df_q = df_q[['Target', 'wall', 'roof', 'floor']]


# In[ ]:


print("Roof quality")
print("==============================================================================================================================")
df_q.loc[df_q['Target'] == 1, 'Target'] = 'Extreme'
df_q.loc[df_q['Target'] == 2,'Target'] = 'Moderate'
df_q.loc[df_q['Target'] == 3,'Target'] = 'Vulnerable'
df_q.loc[df_q['Target'] == 4,'Target'] = 'Non-Vulnerable'
ax = sns.catplot(x = 'roof', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()

print("Wall quality")
print("==============================================================================================================================")

ax = sns.catplot(x = 'wall', col = 'Target', data = df_q, kind="count" ,col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable'], order = ['Bad', 'Regular', 'Good']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()

print("Floor quality")
print("==============================================================================================================================")

ax = sns.catplot(x = 'floor', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()


# **Unfortunatelly for some reason, I cannot install the latest version of seaborn in this kernel which has the catplot. Please leave a comment if you know the answer to this problem of mine.**

# #### Services

# In[ ]:


# bars1 = [12, 28, 1, 8, 22]
# bars2 = [28, 7, 16, 4, 10]
# bars3 = [25, 3, 23, 25, 17]
 
# # Heights of bars1 + bars2 (TO DO better)
# bars = [40, 35, 17, 12, 32]
 
# # The position of the bars on the x-axis
# r = [0,1,2,3,4]
 
# # Names of group and bar width
# names = ['A','B','C','D','E']
# barWidth = 1
 
# # Create brown bars
# plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# # Create green bars (middle), on top of the firs ones
# plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# # Create green bars (top)
# plt.bar(r, bars3, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)
 
# # Custom X axis
# plt.xticks(r, names, fontweight='bold')
# plt.xlabel("group")


# In[ ]:





# #### Overcrowding problem 

# In[ ]:


plt.figure(figsize = (10,6))
plt.subplot(111)
sns.boxplot(x = 'Target', y = 'overcrowding', data = train_df)
plt.title('Person per room')
plt.xlabel('')
plt.ylabel('')
plt.xticks(np.arange(0,4), ['extreme','moderate','vulnerable','non-vulnerable'])
plt.show()


# ## Modelling

# ### Feature engineering

# In[ ]:


def feature_engineer(x):
    x['escolari_age'] = x['escolari'] / x['age']
    x['refrig'] = x['refrig'].astype(int)
    x['computer'] = x['computer'].astype(int)
    x['television'] = x['television'].astype(int)
    x['mobilephone'] = x['mobilephone'].astype(int)
    x['v14a'] = x['v14a'].astype(int)
    x['v18q'] = x['v18q'].astype(int)
    x['epared1'] = x['epared1'].astype(int)
    x['epared2'] = x['epared2'].astype(int)
    x['epared3'] = x['epared3'].astype(int)
    x['etecho1'] = x['etecho1'].astype(int)
    x['etecho2'] = x['etecho2'].astype(int)
    x['etecho3'] = x['etecho3'].astype(int)
    
    x['eviv1'] = x['eviv1'].astype(int)
    x['eviv2'] = x['eviv2'].astype(int)
    x['eviv3'] = x['eviv3'].astype(int)
    x['abastaguadentro'] = x['abastaguadentro'].astype(int)
    x['abastaguafuera'] = x['abastaguafuera'].astype(int)
    x['abastaguano'] = x['abastaguano'].astype(int)
    x['abastaguano'] = x['abastaguano'].astype(int)
    x[['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']] = x[['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']].apply(pd.to_numeric)
    
    x['appliances'] = (x['refrig'] + x['computer'] + x['television'])

    x['rent_by_hhsize'] = x['v2a1'] / x['hhsize'] # rent by household size
    x['rent_by_people'] = x['v2a1'] / x['r4t3'] # rent by people in household
    x['rent_by_rooms'] = x['v2a1'] / x['rooms'] # rent by number of rooms
    x['rent_by_living'] = x['v2a1'] / x['tamviv'] # rent by number of persons living in the household
    x['rent_by_minor'] = x['v2a1'] / x['hogar_nin']
    x['rent_by_adult'] = x['v2a1'] / x['hogar_adul']
    x['rent_by_dep'] = x['v2a1'] / x['dependency']
    x['rent_by_head_educ'] = x['v2a1'] / (x['edjefe'] + x['edjefa'])
    x['rent_by_educ'] = x['v2a1'] / x['meaneduc']
    x['rent_by_numPhone'] = x['v2a1'] / x['qmobilephone']
    x['rent_by_gadgets'] = x['v2a1'] / (x['computer'] + x['mobilephone'] + x['v18q'])
    x['rent_by_num_gadgets'] = x['v2a1'] / (x['v18q1'] + x['qmobilephone'])
    x['rent_by_appliances'] = x['v2a1'] / x['appliances']
    
    x['under12'] = x['r4t1']/x['r4t3']
    x['under12_male'] = x['r4h1']/x['r4t3']
    x['under12_female'] = x['r4m1']/x['r4t3']
    x['Proportion_male'] = x['r4h3']/x['r4t3']
    x['Proportion_female'] = x['r4m3']/x['r4t3']
    
    x['tablet_density'] = x['v18q1'] / x['r4t3']
    x['phone_density'] = x['qmobilephone'] / x['r4t3']
    
    x['wall_qual'] = x['epared3'] - x['epared1']
    x['roof_qual'] = x['etecho3'] - x['etecho1']
    x['floor_qual'] = x['eviv3'] - x['eviv1']
    x['water_qual'] = x['abastaguadentro'] - x['abastaguano']
    
    x['house_qual'] = x['wall_qual'] + x['roof_qual'] + x['floor_qual']
    
    x['person_per_room'] = x['hhsize'] / x['rooms']
    x['person_per_appliances'] = x['hhsize'] / x['appliances']
    
    x['educ_qual'] = (1 * x['instlevel1']) + (2 * x['instlevel2']) + (3 * x['instlevel3']) + (4 * x['instlevel4']) + (5 * x['instlevel5']) + (6 * x['instlevel6']) + ( 7 * x['instlevel7']) + (8 * x['instlevel8']) + (9 * x['instlevel9'])
    x['educ_by_individual'] = x['escolari']/x['r4t3']
    x['educ_by_adult'] = x['escolari']/(x['r4t3'] - x['r4t1'])
    x['educ_by_child'] = x['escolari']/x['r4t1']
    
    x['max_educ'] = np.max(x[['edjefa','edjefe']])
    
    def reverse_label_encoding(row, df):
        for c in df.columns:
            if row[c] == 1:
                return int(c[-1])
            
    def rate_sanitary(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'sanitario2':
            return 3
        elif row[c] == 'sanitario3':
            return 2
        elif row[c] == 'sanitario5':
            return 1
        else:
            return 0
        
    def rate_cooking(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'energcocinar2':
            return 3
        elif row[c] == 'energcocinar3':
            return 2
        elif row[c] == 'energcocinar4':
            return 1
        else:
            return 0
        
    def rate_rubbish(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'elimbasu1':
            return 1
        elif row[c] == 'elimbasu2':
            return 2
        else:
            return 0
            
    x['sanitary'] = x.apply(lambda q: reverse_label_encoding(q, x[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']]), axis=1)
    x['cooking'] =  x.apply(lambda q: reverse_label_encoding(q, x[['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']]), axis=1)
    x['rubbish'] = x.apply(lambda q: reverse_label_encoding(q, x[['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']]), axis=1)
    x['region'] = x.apply(lambda q: reverse_label_encoding(q, x[['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']]), axis=1)
    
    x['sanitary_i'] = x.apply(lambda q: rate_sanitary(q, x[['sanitary']]), axis = 1)
    x['cooking_i'] = x.apply(lambda q: rate_cooking(q, x[['cooking']]), axis = 1)
    x['rubbish_i'] = x.apply(lambda q: rate_rubbish(q, x[['rubbish']]), axis = 1)
    
    x['zone'] = x['area1'] - x['area2']

    x.replace([np.inf, -np.inf], 0, inplace = True)
    x.fillna(0, inplace = True)


# In[ ]:


feature_engineer(train_df)
feature_engineer(test_df)


# In[ ]:


train_df.shape


# In[ ]:


# def agg_features(y):
#     agg_feat = ['hacdor', 'v18q1', 'dis', 'r4h3', 'r4m3', 'age', 'hogar_nin', 'hogar_adul', 'hogar_total', 'dependency',
#                 'appliances', 'phone_density', 'tablet_density', 'house_qual', 'person_per_appliances', 'educ_qual'
#                ]
#     # https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
#     for group in ['idhogar', 'zone', 'region']:
#         for feat in agg_feat:
#             for agg_m in ['mean','sum']:
#                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()[[feat, group]]
# #                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
#                 new_col = feat + '_' + agg_m + '_' + group 
#                 id_agg.rename(columns = {feat : new_col} , inplace = True)
#                 y = y.merge(id_agg, how = 'left', on = group)

    
#     drop_ = ['sanitary', 'cooking', 'rubbish', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
#             'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',
#             'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
    
#     y.drop((drop_), inplace = True, axis = 1)
#     y.replace([np.inf, -np.inf], 0, inplace = True)
#     y.fillna(0, inplace = True)
#     return y


# In[ ]:


def agg_features(y):
    mean_list = []
    #o_list = ['escolari', 'age', 'escolari_age', 'phone_density', 'rez_esc', 'dis', 'male', 'female','v2a1','house_qual', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total']
    count_list = ['escolari_age', 'phone_density', 'rez_esc', 'dis', 'male', 'female','v2a1','house_qual', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'escolari','estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7','parentesco1', 'parentesco2', 'parentesco3',
                 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
        
#     for group in ['idhogar', 'region']:
#         for feat in o_list:
#             for agg_m in ['sum']:
#                 id_agg = y[feat].groupby(y[group]).agg(agg_m)#.reset_index()[[feat, group]]
# #                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
#                 new_col = feat + '_' + agg_m + '_' + group 
#                 #id_agg.rename(columns = {feat : new_col} , inplace = True)
#                 #y = y.merge(id_agg, how = 'left', on = group)
#                 y[new_col] = id_agg
                
    for group in ['idhogar']:            
        for item in count_list:
            for agg_m in ['mean','std','min','max','sum']:
                id_agg = y[item].groupby(y[group]).agg(agg_m)#.reset_index()[[feat, group]]
    #                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
                new_col = item + '_' + agg_m + '_' + group 
                    #id_agg.rename(columns = {feat : new_col} , inplace = True)
                    #y = y.merge(id_agg, how = 'left', on = group)
                y[new_col] = id_agg
                
    drop_ = ['sanitary', 'cooking', 'rubbish', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
            'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'tamhog', 'tamviv', 'hhsize', 'v18q', 
             'v14a', 'agesq','mobilephone', 'female', 'estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7', 'parentesco2', 'parentesco3',
                 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 
             'lugar5', 'lugar6', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned',
            'r4t1', 'r4t2', 'r4t3']
    
    y.drop((drop_), inplace = True, axis = 1)
    y.replace([np.inf, -np.inf], 0, inplace = True)
    y.fillna(0, inplace = True)
    return y
            
            
# for item in aggr_mean_list:
#     group_train_mean = train_set[item].groupby(train_set['idhogar']).mean()
#     group_test_mean = test_set[item].groupby(test_set['idhogar']).mean()
#     new_col = item + '_aggr_mean'
#     df_train[new_col] = group_train_mean
#     df_test[new_col] = group_test_mean


# In[ ]:


train_df = agg_features(train_df)
test_df = agg_features(test_df)


# In[ ]:


train_df = train_df.loc[train_df['parentesco1'] == 1]
train_df.fillna(value=0, inplace=True)


# In[ ]:


train_df.shape


# In[ ]:


y = train_df[['Target']]
x = train_df.drop(['Target','Id','idhogar'], axis = 1)


# In[ ]:


train_df.columns.tolist()


# ### Feature selection

# For selecting the most appopriate columns for our model we will perform two operations. <br>
# <ol>
#     <li>Random Forest for feature importance</li>
#     <li>Eliminate highly correlated values</li>
# </ol>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# https://www.kaggle.com/grfiv4/plotting-feature-importances

# In[ ]:


clf = RandomForestClassifier()
clf.fit(x, y)

imp = clf.feature_importances_
name = np.array(x.columns.values.tolist())


df_imp = pd.DataFrame({'feature':name, 'importance':imp})
df_imp = df_imp.sort_values(by='importance', ascending=False)


# In[ ]:


# https://www.kaggle.com/skooch/lgbm-with-random-split/notebook

def feature_importance(forest, X_train):
    ranked_list = []
    
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]) + " - " + X_train.columns[indices[f]])
        ranked_list.append(X_train.columns[indices[f]])
    
    return ranked_list    


# In[ ]:


plt.figure(figsize=(8,20))
sns.barplot(df_imp.loc[(df_imp['importance'] > 0.0005),'importance'], y = df_imp.loc[(df_imp['importance'] > 0.0005),'feature'])
plt.title('Important features')
plt.show()


# In[ ]:


important_cols = df_imp.loc[(df_imp['importance']>0),'feature']


# In[ ]:


x_ = x[important_cols]
# plt.figure(figsize = (20,16))
# sns.heatmap(x_.corr(), cmap='YlOrRd')
# plt.show()


# SQBmeaned and meaneduc are highly correlated with each other, so we will just use meaneduc. The same with SQBovercrowding and overcrowding, just use overcrowding. hogar_nin is selected ahead of SQBhogar_nin, age ahead of SQBage and agesq, escolari ahead of SQBescolari, v2a1 ahead of rent_by_rooms. tamhog, tamviv, r4t3 and SQBhogar_total, hhsize are highly correlated with each other let's use tamviv because it has higher importance.
# <hr>
# Actually, just kidding XGBosst and LGBM are immune to multicollinearity.

# In[ ]:


# x_ = x[['meaneduc', 'dependency', 'person_per_room', 'qmobilephone', 'overcrowding', 'hogar_nin', 'age', 'r4t2', 'rooms', 'cielorazo', 'r4h3', 'r4h2', 'r4m3', 'v2a1', 'rent_by_hhsize', 'r4t1', 'escolari', 'v18q', 'r4m1', 'bedrooms', 'edjefe', 'eviv3', 'epared3', 'hogar_adul', 'etecho3', 'r4m2', 'tamviv']]


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


features = [c for c in x_.columns if c not in ['Target']]
target = train_df['Target']

# scaler = StandardScaler()
# x_ = scaler.fit_transform(x_)


# ### Sampling and splitting

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_sample(x_, target)


# In[ ]:


# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_sample(x_, target)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_resampled,  y_resampled, test_size=0.1, random_state=1)
#X_train, X_valid, y_train, y_valid = train_test_split(x_,  target, test_size=0.1, random_state=1)


# In[ ]:


pd.options.display.width =300
X_resampled = pd.DataFrame(X_resampled)
X_resampled.columns = features

y_resampled = pd.DataFrame(y_resampled)
y_resampled.columns = ['Target']

X_train_df = pd.DataFrame(X_train)
X_train_df.columns = features


# 

# #### LGBM

# In[ ]:


import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True)


# In[ ]:


# heavily influenced by https://www.kaggle.com/skooch/lgbm-with-random-split/notebook

# xgboost = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.2, max_depth = 5, n_classes = 4,
#                            objective = 'multi:softprob', colsample_bytree = 0.8)

lgbm = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.88, min_child_samples = 90, num_leaves = 16, subsample = 0.94)


# In[ ]:


# imp = final.feature_importances_
# name = np.array(X_train_df.columns.tolist())


# df_imp = pd.DataFrame({'feature':name, 'importance':imp})
# df_imp = df_imp.sort_values(by='importance', ascending=False)
# feat_to_remove = df_imp.loc[df_imp['importance']<=0]


# In[ ]:


predicts_lgb = []
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_t, X_v = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_t, y_v = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    
    lgbm.fit(X_t, y_t)
    predicts_lgb.append(lgbm.predict(test_df[features]))


# In[ ]:


# predicts_xgb = []
# for train_index, test_index in skf.split(x_, target):
#     X_t, X_v = x_.iloc[train_index], x_.iloc[test_index]
#     y_t, y_v = target.iloc[train_index], target.iloc[test_index]
    
#     xgboost.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=50)
#     predicts_xgb.append(xgboost.predict(test_df[features]))


# ## Submission

# In[ ]:


# predict = vc.predict(test_df[features].values)


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test_df['Id']
submission['Target'] = np.array(predicts_lgb).mean(axis=0).round().astype(int)
#submission['Target'] = np.array(predict)


# In[ ]:


submission.to_csv('submissions.csv', index=False)


# ## Summary, conclusion and recommendation

# ----- TO be continued

# In[ ]:




