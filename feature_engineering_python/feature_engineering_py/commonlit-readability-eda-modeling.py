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


# # Read Data

# In[2]:


train = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
test = pd.read_csv('../input/commonlitreadabilityprize/test.csv')


# In[3]:


# Lets explore the Train dataset
train.head()


# In[4]:


test.head()


# Feature Details:
# 1. id - unique value of the excerpt
# 2. url_legal - Source of Url. there are some black cells as well. 
# 3. licencse - License of source material
# 4. target - ease of reading, -ve is hard and +ve is easy. 
# 5. standar_error - measures the spread of score across multiple rater.

# In[5]:


train['excerpt'][0]


# In[6]:


train.shape


# # Check for Null value

# In[7]:


train.isnull().sum()


# Out of 2834 rows, url_legal & license has null values for 2004 rows. dropping null value would reduce the dataset size largely. So, we need to check if the feaure has realtionship in target feature. 

# In[8]:


# Unique value in url_legal
train['url_legal'].value_counts()


# In[9]:


train['license'].value_counts()


# ## Creative Common License types
# The Creative Commons License Options
# There are six different license types, listed from most to least permissive here:
# 
# CC BY: This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use.
# CC BY includes the following elements:
# BY  – Credit must be given to the creator
# 
#  
# 
# CC BY-SA: This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.
# CC BY-SA includes the following elements:
# BY  – Credit must be given to the creator
# SA  – Adaptations must be shared under the same terms
# 
#  
# 
# CC BY-NC: This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. 
# It includes the following elements:
# BY  – Credit must be given to the creator
# NC  – Only noncommercial uses of the work are permitted
# 
#  
# 
# CC BY-NC-SA: This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms. 
# CC BY-NC-SA includes the following elements:
# BY  – Credit must be given to the creator
# NC  – Only noncommercial uses of the work are permitted
# SA  – Adaptations must be shared under the same terms
# 
#  
# 
# CC BY-ND: This license allows reusers to copy and distribute the material in any medium or format in unadapted form only, and only so long as attribution is given to the creator. The license allows for commercial use. 
# CC BY-ND includes the following elements:
# BY  – Credit must be given to the creator
# ND  – No derivatives or adaptations of the work are permitted
# 
#  
# 
# CC BY-NC-ND: This license allows reusers to copy and distribute the material in any medium or format in unadapted form only, for noncommercial purposes only, and only so long as attribution is given to the creator. 
# CC BY-NC-ND includes the following elements:
# BY  – Credit must be given to the creator
# NC  – Only noncommercial uses of the work are permitted
# ND  – No derivatives or adaptations of the work are permitted

# # Feature Engineering
# Let us try to extract the url_legal home page and categories the license type with in 6 category as mentioned above

# In[10]:


import re
def extract_license(t):
    if t==0:
        return 0
    else: 
        return re.split("\d",t)[0].replace('CC-','CC ').replace('BY ','BY-').strip()


# In[11]:


train['license'].fillna(value=0, axis=0, inplace=True)
train['license']=train['license'].apply(lambda x: extract_license(x))


# In[12]:


train['license'].value_counts()


# we have successfully engineered the url_legal & licence features. let us use the EDA to explore more

# In[13]:


train.isna().sum()


# In[14]:


train.info()


# In[15]:


train.drop(['id'], axis=1, inplace=True)


# # EDA

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
colors=['#C7663E','#948078','#FA6767','#A0FAA0','#81C75B']
sns.set(palette=colors, font='San', style='white', rc={'axes.facecolor':'whitesmoke', 'figure.facecolor':'whitesmoke'})
sns.despine(left=False, right=False)
sns.palplot(colors)


# ### Univariated analysis

# In[17]:


train.info()


# In[18]:


train['url_legal'].value_counts().nlargest(5)


# In[19]:


fig, ax= plt.subplot_mosaic("""ac
                                bd
                                ee""", figsize=(20,8), constrained_layout=True)
plt.suptitle("Univariated analysis of each features", size=20, weight='bold')

#ax['a'].set_title('url_legal feaure details (excluding 0 value)', size=10, weight='bold')
#sns.countplot(data=train[train['url_legal']!=0], x='url_legal', ax=ax['a'], order=train[train['url_legal']!=0]['url_legal'].value_counts().index)


#for i,j in enumerate(ax['a'].patches):
#    ax['a'].text(x=j.get_x(),y=10, s=ax['a'].get_xticklabels()[i].get_text(), rotation=90)
#ax['a'].set_xticks([])

ax['c'].set_title('legal feaure details (excluding 0 value)', size=10, weight='bold')
sns.countplot(data=train[train['license']!=0], x='license', ax=ax['c'], order=train[train['license']!=0]['license'].value_counts().index)
for i,j in enumerate(ax['c'].patches):
    ax['c'].text(x=j.get_x(),y=10, s=ax['c'].get_xticklabels()[i].get_text(), rotation=90)
ax['c'].set_xticks([])

ax['b'].set_title('Histogram for target feaure', size=10, weight='bold')
sns.histplot(data=train, x='target', ax=ax['b'], kde=True)
ax['d'].set_title('Histogram for standar_error feaure', size=10, weight='bold')
sns.histplot(data=train, x='standard_error', ax=ax['d'], kde=True)

ax['e'].text(x=0, y=0.8, s="1. url_legal - most of the contents are taken from wikipedia and followed by frontiersin, Africanstorybook etc")
ax['e'].text(x=0, y=0.6, s="2. license - most of the contents are under CC BY and CC BY-SA licensed, except the contents without license")
ax['e'].text(x=0, y=0.4, s="3. target - target feature explain about the content ease of reading, -ve value is tough and +ve is easy to read")
ax['e'].text(x=0, y=0.2, s="4. standard_error - standar_error is the spread of scores by different raters for each content")
for i in ['left','right','top','bottom']:
    ax['e'].spines[i].set_visible(False)
    ax['a'].spines[i].set_visible(False)
    ax['b'].spines[i].set_visible(False)
    ax['c'].spines[i].set_visible(False)
    ax['d'].spines[i].set_visible(False)
ax['e'].set_xticks([])
ax['e'].set_yticks([])


# In[20]:


fix, ax=plt.subplots(ncols=2, nrows=1, figsize=(15,8))
sns.boxplot(data=train, x='target',ax=ax[0])
sns.boxplot(data=train, x='standard_error', ax=ax[1])


# **Observations**
# 1. There are outliers in standar_error features, which we need to check and address
# 2. around 50% of the rating is around -2 to 0. so, most contents are predicted as moderate difficult.

# # Multivariated Analysis

# **Observation**
# 1. url_legal values with no url is distibured from -4 to +4, wikipedia & frontiersin excerpt aslo contains the readbility from easy to difficult, middle value is at -1.
# 2. median value for the excerpt from africanstorybook, ck12, freekidsbooks, digitallibrary, google, osu, wikibooks are mostly bove 0, which means difficulty of reading this contents are less compared to other excerpts

# In[21]:


#Since target feature is the target variable to predict, so let us explore with target feature with other features
fig=plt.figure(figsize=(15,8))
ax=sns.boxplot(data=train, x='license',y='target')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60);


# **Observations:**
# 1. excerpts from the license category CC BY, CC BY-NC, CC BY-NC-SA, CC BY_NC-ND, GNU free document license are comparitively less difficult compared to other license formats

# In[22]:


fig=plt.figure(figsize=(8,8))
ax=sns.scatterplot(data=train, x='standard_error', y='target')
#ax.set_xlim(0.4,0.7)


# In[23]:


fig= plt.figure(figsize=(15,8))
sns.jointplot(data=train, x='standard_error', y='target',kind='hex', xlim=(0.3, 0.7))


# **Observations:**
# when the target value is close to -1 the stardard error is also less. and the standard error starts to increase for the target feature distribution is above -2 & 1. so, the raters have predicted the difficult for the less difficult & not so easy contents. beyond that, the rating differs.  
# Outlier in Standar_error features seems to be the only 0 error excerpt. let us find more info.

# In[24]:


print(train[train['standard_error']==0]['excerpt'])
print(train[train['target']==0]['excerpt'])


# we clearly see that the excerpt is same when the target & standard_error value is 0, which means that this particular excerpt is set as baseline to rate other excerpt. is this because fo the len of the sentance? let us explore further.

# In[25]:


import math
train['length']=train['excerpt'].apply(lambda x: len(x.split()))


# In[26]:


train.corr()


# In[27]:


fig, ax=plt.subplots(ncols=3, nrows=1, figsize=(20,8))
sns.regplot(data=train, x='length',y='target',line_kws={'color':'black'}, ax=ax[0] )
sns.kdeplot(data=train, x='length', fill=True, ax=ax[1])
ax[1].axvline(train[train['target']==0]['length'].values)
ax[1].axvline(train[train['target']==train['target'].max()]['length'].values, ls='--')
sns.boxplot(data=train, x='length', ax=ax[2])


# Length/no of words doesn't give much information on target feature. number of words are distributed from 140 to 200

# In[28]:


train[train['standard_error']==train['standard_error'].min()]
#train[train['target']==0]


# target feature & standar_error feature has only one 0 values. while creating model, let us try to check the accuracy after removing the 0 value. let see if this helps.

# ### Try clustering to find the paterns

# In[29]:


import plotly.express as ex
from sklearn.cluster import KMeans
X=train.drop(['excerpt','url_legal','license'], axis=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
ex.scatter_3d(data_frame=X, x='standard_error',y='target',z='length', color=y_kmeans)


# We can see some paterns. but it is helpful to predict target?

# In[30]:


train['cluster']=y_kmeans
train.corr()


# ### Hypothesis Testing

# In[31]:


from scipy.stats import f_oneway
from statsmodels.formula.api import ols
import statsmodels.api as sm
cl0 = train[train['cluster']==0]['target']
cl1 = train[train['cluster']==1]['target']
cl2 = train[train['cluster']==2]['target']

sta, p_value=f_oneway(cl0,cl1,cl2, axis=0)
print(p_value)
if p_value <0.05:
    print(f"{np.round(p_value,5)} cluster has significant differnce in the target feature")
else:
    print("cluster has significant no differnce in the target feature")
    
    
formula = 'target ~ C(cluster)'
model = ols(formula, train).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(np.round(anova_table),3)


# so, we can say that the length of the sentence has significant impact on the target prediction at 5% significance. 

# # Topic Modeling

# # lets try to identify the topic based on excerpt

# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X=train['excerpt']
X_vect = vect.fit_transform(X)
from sklearn.decomposition import LatentDirichletAllocation
decom = LatentDirichletAllocation(n_components=6, random_state=42)
X_decm=decom.fit_transform(X_vect)


# In[33]:


train['topic']=X_decm.argmax(axis=1)
for index,topic in enumerate(decom.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([vect.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
train


# ***Most topics are in 2,4 & 5 category***  
# topic 2 - talks about galaxy, milky way, planets ets - only very few excerpts are in topic 2.  
# topic 4 - talks about people, water, man, time, food etc  
# topic 5 - talks about gas, computer, people, information, history, cells etc.  

# In[34]:


print(train[train['topic']==4]['excerpt'].iloc[0])
print("\n\n")
print(train[train['topic']==4]['excerpt'].iloc[1])


# In[35]:


print(train[train['topic']==5]['excerpt'].iloc[0])
print("\n\n")
print(train[train['topic']==5]['excerpt'].iloc[1])


# ### Excerpt Cleanup

# In[36]:


import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_word = stopwords.words('english')

def clean_excerpt(x):
    t = ' '.join(w for w in x.split() if w not in stop_word)
    return t

def clean_punct(x):
    t=' '.join(w for w in x.split() if w.isalnum())
    return t

train['excerpt_clean']=train['excerpt'].apply(clean_excerpt)
train['excerpt_clean']=train['excerpt_clean'].apply(clean_punct)


# In[37]:


train


# In[38]:


#lets do Stemming

from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer(language='english')

def clean_stem(x):
    t= ' '.join(stemmer.stem(w) for w in x.split())
    return t

train['excerpt_clean']=train['excerpt_clean'].apply(clean_stem)


# In[39]:


train['clean_length']=train['excerpt_clean'].apply(lambda x: len(x.split()))


# In[40]:


sns.heatmap(train.corr(), annot=True, linewidth=2)


# # Modeling - Baseline model

# In[41]:


X


# In[42]:


train.head()
train=train[['target','license','excerpt_clean','clean_length']]
train['license']=train['license'].apply(lambda x: 'no_license' if x==0 else x)


# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
train['license']=encode.fit_transform(train['license'])

from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(max_features=400)
X_vect=vect.fit_transform(train['excerpt_clean'])
X=pd.concat([train,pd.DataFrame(X_vect.toarray())], axis=1,ignore_index=True)

X=X.drop([0,2], axis=1)
y=train['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

model = LinearRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print(f"model train accuracy: {model.score(X_train, y_train)}")
print(f"model test accuracy: {model.score(X_test, y_test)}")
print(f"RSME: {np.sqrt(mean_squared_error(y_test, pred))}")
print(f"R-Sq: {r2_score(y_test, pred)}")


# In[44]:


X_train.shape


# In[45]:


residual = y_test-pred
fig, ax=plt.subplots(ncols=2, nrows=1, figsize=(15,4))
ax[0].scatter(y=y_test, x=pred)
ax[0].axhline(y=0, c='black', ls='--')
ax[1]=sns.kdeplot(residual)


# In[46]:


test


# In[47]:


test


# In[48]:


test


# In[49]:


test= pd.read_csv('../input/commonlitreadabilityprize/test.csv')
X_vect=vect.fit_transform(train['excerpt_clean'])
X=pd.concat([train,pd.DataFrame(X_vect.toarray())], axis=1,ignore_index=True)


X=X.drop([0,2], axis=1)
y=train['target']
test.drop(['id','url_legal'], axis=1, inplace=True)

model.fit(X,y)
print(f"model train accuracy: {model.score(X, y)}")

test['license'].fillna(value=0, axis=0, inplace=True)
test['license']=test['license'].apply(lambda x: 'no_license' if x==0 else x)
test['license']=test['license'].apply(lambda x: extract_license(x))

test['license']=encode.transform(test['license'])
test['excerpt_clean']=test['excerpt'].apply(clean_excerpt)
test['excerpt_clean']=test['excerpt_clean'].apply(clean_punct)
test['excerpt_clean']=test['excerpt_clean'].apply(clean_stem)
test['clean_length']=test['excerpt_clean'].apply(lambda x: len(x.split()))
test.drop(['excerpt'], axis=1, inplace=True)

X_test_vect=vect.transform(test['excerpt_clean'])
X_test=pd.concat([test,pd.DataFrame(X_test_vect.toarray())], axis=1,ignore_index=True)
X_test.drop([1], axis=1, inplace=True)
pred=model.predict(X_test)


# In[50]:


test= pd.read_csv('../input/commonlitreadabilityprize/test.csv')
test['target']=pred
sample_df=test[['id','target']]
sample_df.to_csv('submission.csv', index=False)


# In[51]:


sample_df


# ***Please review and provide your inputs to improve the score. appriciate your expert opinion*** 
