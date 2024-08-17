#!/usr/bin/env python
# coding: utf-8

# 
# # Tabular Playground Series: Feb 2021
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/25225/logos/header.png?t=2021-01-27-17-34-26)
# 
# ## Introduction:
# 
# Starting from January this year, the kaggle competition team is offering a month-long tabulary playground competitions. This series aims to bridge between inclass competition and featured competitions with a friendly and approachable datasets.
# 
# For this month, kaggle is offering a dataset which is synthetic but based on a real dataset and generated using a CTGAN. The original dataset, this synthetic dataset is derived from, deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features.
# 
# The data has: 
# 
# * 10 categorical variables: **cat0** to **cat9**
# * 14 continuous variables: **cont0** to **cont13**
# * 1 numericat **target** column
# 
# Files provides:
# 
# - train.csv - the training data with the target column
# - test.csv - the test set; you will be predicting the target for each row in this file
# - sample_submission.csv - a sample submission file in the correct format
# 
# The goal of the competition is to predict a continuous **target** based on the given categorical and continuous features. However, the goal of **this notebook** is to explore (EDA) and visualize the given data. And when possible try to discover (engineer) *potentially usefull* features for further data modelling and prediction.

# # Set-up

# In[1]:


import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from matplotlib.offsetbox import AnchoredText
import random
from matplotlib.ticker import MaxNLocator
import pylab as p

import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the data

# In[2]:


train_ = pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/train.csv', index_col='id')
test = pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/test.csv', index_col='id')

submission= pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/sample_submission.csv', index_col='id')


# # Explore the data

# In[3]:


train = train_.copy()


# In[4]:


print('Train data of shape {}'.format(train.shape))
display(train.head())
print('Test data of shape {}'.format(test.shape))
display(test.head())


# In[5]:


display(train.describe().T)


# In[6]:


target = train.pop('target')


# In[7]:


cat_features =[]
num_features =[]

for col in train.columns:
    if train[col].dtype=='object':
        cat_features.append(col)
    else:
        num_features.append(col)
print('Catagoric features: ', cat_features)
print('Numerical features: ', num_features)


# ## No null-values in the data

# In[8]:


print('Number of NA values in train data is {}'.format(train.isna().sum().sum()))
print('Number of NA values in test data is {}'.format(test.isna().sum().sum()))


# ## Categorical features
# 

# In[9]:


for col in cat_features:
    print('{} unique values in {}'.format(train[col].nunique(), col))


# ## Unique values in categorical features (train vs test)
# 
# <div class="alert alert-block alert-danger">  
# >>> Cat6 in train data has one more catagory than test data!!
# </div>
# 

# In[10]:


# train_data
unique_cat_train = []
for col in train[cat_features]:
    unique_train = train[col].nunique()  
    dict1 ={
        'Features' : col,
        'Unique cats (train)': unique_train,        
    }
    unique_cat_train.append(dict1)
DF1 = pd.DataFrame(unique_cat_train, index=None).sort_values(by='Unique cats (train)',ascending=False)

# test_data
unique_cat_test = []
for col in test[cat_features]:
    unique_test = test[col].nunique()    
    dict2 ={
        'Features' : col,
        'Unique cats (test)': unique_test,        
    }
    unique_cat_test.append(dict2)
DF2 = pd.DataFrame(unique_cat_test, index=None).sort_values(by='Unique cats (test)',ascending=False)

pd.merge(DF1, DF2, how='outer', on=['Features']).style.format(None, na_rep="-")


# # Data Visualization

# In[11]:


def count_plot_pct(data, features, titleText):
    i = 1
    plt.figure()
    fig, ax = plt.subplots(5, 2,figsize=(18, 22))
    fig.subplots_adjust(top=0.95)
    for feature in features:
        total = float(len(data)) 
        plt.subplot(5, 2, i)
        ax = sns.countplot(x=feature, palette='coolwarm', data=data)        
        ylabels = ['{:.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
        ax.set_yticklabels(ylabels)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f} %'.format((height/total)*100),ha="center")
        i += 1
    plt.suptitle(titleText ,fontsize = 24)
    plt.show()    
    


# ### Train data categorical features

# In[12]:


count_plot_pct(train, cat_features, 'Train data Categorical features: distribution of each category')


# ### Test data categorical features

# In[13]:


count_plot_pct(test, cat_features, 'Test data Categorical features: percentage of each category')


# In[14]:


# Install additional data-visualization library
get_ipython().system('pip install ptitprince')


# In[15]:


import ptitprince as pt
def raincloud_plot(data, features, titleText='Title'):    
    i = 1
    plt.figure()
    fig, ax = plt.subplots(5, 2,figsize=(18, 28))
    fig.subplots_adjust(top=0.95)
    for feature in features:
        plt.subplot(5, 2, i)
        ax = pt.RainCloud(x = data[feature], y = target, 
                  data = train, 
                  width_viol = 0.8,
                  width_box = 0.4,
                  orient = 'h',
                  move = 0.0)
        i += 1
    plt.suptitle(titleText ,fontsize=24)
    plt.show()


# In[16]:


raincloud_plot(train, cat_features, 'Raincloud plot: train data, categorical features')


# ## Numerical features 
# 

# In[17]:


def sns_box_plot1(data, features, titleText='Title'):
    i = 1
    
    L = len(num_features)
    nrow= int(np.ceil(L/3))
    ncol= 3
    
    remove_last= (nrow * ncol) - L
    
    fig, ax = plt.subplots(nrow, ncol,figsize=(18, 22))
    ax.flat[-remove_last].set_visible(False)
    fig.subplots_adjust(top=0.95) 
    
    for feature in features:
        plt.subplot(5, 3, i)
        ax = sns.boxplot(x=data[feature], palette='coolwarm')
        ax = sns.violinplot(x=data[feature], inner=None, palette='viridis')
        plt.xlabel(feature, fontsize=10)
        i += 1
    plt.suptitle(titleText, fontsize=24,)
    plt.show()


# In[18]:


sns_box_plot1(train, num_features, 'Box+violin plots: train data, numerical features')


# In[19]:


plt.figure()
fig, ax = plt.subplots(4, 3,figsize=(20, 20))
fig.subplots_adjust(top=0.95)
i = 1
for feature in num_features:
    plt.subplot(5, 3, i)
    ax = sns.histplot(train[feature],color="cyan", kde=True,bins=120, label='train')
    ax = sns.histplot(test[feature], color="red", kde=True,bins=120, label='test')
    ylabels = ['{:.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
    ax.set_yticklabels(ylabels)
    plt.xlabel(feature, fontsize=9)
    plt.legend()
    i += 1
plt.suptitle('Histogram of numerical features', fontsize=20)
plt.show()


# In[20]:


plt.figure()
fig, ax = plt.subplots(4, 3,figsize=(20, 20))
fig.subplots_adjust(top=0.95)
i = 1
for feature in num_features:
    plt.subplot(5, 3, i)
    sns.scatterplot(data= train, x=train[feature], y=target, color="blue", label='train')
    plt.xlabel(feature, fontsize=9);
    plt.legend()
    i += 1
plt.suptitle('Scatter plot of numerical features', fontsize=20)
plt.show()


# # Target variable distribution

# In[21]:


plt.figure(figsize=(12, 8))
ax = sns.kdeplot(target, shade=True, color='black', edgecolor='red', alpha=0.85)
plt.title('Target Distribution', fontsize=20)


# # Skewness & Kurtosis

# In[22]:


kurt = []
ske = []
for cont in num_features:
    x = train[cont]     
    kurt.append(kurtosis(x))
    ske.append(skew(x))
    
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(num_features, kurt, '*', markersize = 15, color= 'red', label="kurtosis")
ax.plot(num_features, ske, 'o', markersize = 15, color='blue',label="skewness" )
ax.hlines(y=0, xmin=0, xmax=13, colors='black', linestyles='solid', label='Normal-dist')
ax.set_xlabel('Numerical features', fontsize=12)
ax.set_ylabel('Skewness & kurtosis', fontsize=12)
ax.set_title('Skewness and kurtosis of the numerical features', fontsize=20)
ax.grid()
ax.legend()
plt.show()


# # Correlation matrix

# In[23]:


corr = train_.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(14, 10))
cmap = sns.diverging_palette(230, 0, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-.1, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.75})


# In[24]:


corr = pd.concat((train_[cat_features], target), axis=1).apply(lambda x : pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(12, 10))
cmap = sns.diverging_palette(230, 0, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-.1, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.75})


# # Feature engineering: numerical features
# (See a sample plot of engineered features)

# In[25]:


def cont_feature_engg_plot(cont, titleText, data=train):    
    L = len(num_features)
    nrow= int(np.ceil(L/3))
    ncol= 3
    
    remove_last= (nrow * ncol) - L
    
    fig, ax = plt.subplots(nrow, ncol,figsize=(18, 22))
    ax.flat[-remove_last].set_visible(False)
    fig.subplots_adjust(top=0.92)
    i = 1
    for feature in num_features:
        plt.subplot(nrow, ncol, i)
        ax = sns.kdeplot((data[cont] + data[feature]), shade=True, color='#D5DBDB', edgecolor='black',
                         alpha=0.9, label= str(cont) +'+'+ str(feature))
        
        # target is scaled only for plotting purpose. We are interested in the shape of the 
        # engineered feature in relation to the target 
        ax = sns.kdeplot(target/7, shade=True, color='red', edgecolor='black', label= 'target')
        ax.set_xlabel(None)
                
        # correlation value
        correlation = np.round(np.array(train[[cont, feature]].corr())[0][1], 3)
        
        # Anchored text for the correlation value between the two features
        at = AnchoredText(correlation,
                  prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        ax.legend()
     
        i += 1
    plt.suptitle(titleText ,fontsize = 20)
    plt.show()    


# ## Continuous feature cont{0} added to cont{x}
# (How does the resulting kde-shape look like compared to target?)
# 

# In[26]:


cont_feature_engg_plot('cont0', titleText='Shape (kde-plot) of engineered features: cont{0} + cont{x}'
                       '\n(comparision with target)\n[upper left corner: corr b/n feats]', data=train)


# ## Uncomment and run the cell below to plot the rest of the features (cont{x} + cont{y})

# In[27]:


# for cont in num_features[1:]:    
#     cont_feature_engg_plot(cont, titleText='Shape (kde-plot) of engineered features:' + cont + ' + cont{x}'
#                        '\n(comparision with target)\n[upper left corner: corr b/n feats]', data=train)
    


# ## Mutual information regression
# 
# As a beginner, Kaggle learn is a great source of information and knowledge for me. In the course [Feature Engineering](https://www.kaggle.com/learn/feature-engineering), the description of mutual information reads like this: 
# > Mutual information describes relationships in terms of uncertainty. The mutual information (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target?
# 
# So from the MI score knowing the values of cont8, cat1, cont0 and cat9 will increase confidence level of your target value associated with these features.   

# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le_train = train.copy()
le_test = test.copy()

for col in cat_features:
    le_train[col] = le.fit_transform(train[col])
    le_test[col] = le.transform(test[col])
train = le_train
test = le_test


# In[29]:


# the following two code snippets are adapted from the "feature engineering kaggle min-course"

from sklearn.feature_selection import mutual_info_regression
features = train.dtypes == int

def make_mi_scores(train, y, discrete_features):
    mi_scores = mutual_info_regression(train, target, discrete_features=features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(train, target, features)
mi_scores


# In[30]:


def plot_utility_scores(scores):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y, color='red', alpha=0.3)
    plt.yticks(width, ticks)
    #plt.grid()
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_utility_scores(mi_scores)


# ### Thank you very much for reading this notebook!

# In[ ]:




