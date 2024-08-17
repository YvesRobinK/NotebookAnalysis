#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.figure_factory as ff
import numpy as np


#background = text
bg_text = [['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e'],
           ['t', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', 'n', 'd', 's', 'e', 'r', 'i', 'e', 's'],        
        ['2', '0', '2', '1' , 't', 'a', 'b', 'u', 'l', 'a', 'r', 'p', 'l', 'a', 'y', 'g', 'r', 'o', 'u', '@', 'd', 'e', 's'],]
          
    
text_1 = text_2 = bg_text

z = [[.0, .0, 0, 0, 0, 0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0 ],
     [.0, .0, 0, 0, 0,.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0 ],
     [.0, .0, .8, .8, .8, .0, .8, .8, .8, .0, .8, .8, .8, .0, .0, .0, .8, .8, .8, .0, .8, .0, .0],
     [.0, .0, .8, .0, .8, .0, .8, .0, .0, .0, .0, .8, .0, .0, .0, .0, .0, .0, .8, .0, .8 ,.0, .0],
     [.0, .0, .8, .0, .8, .0, .8, .0, .0, .0, .0, .8, .0, .8, .8, .0, .8, .8, .8, .0, .8, .0, .0],
     [.0, .0, .8, .0, .8, .0, .8, .0, .0, .0, .0, .8, .0, .0, .0, .0, .8, .0, .0, .0, .8, .0, .0],
     [.0, .0, .8, .8, .8, .0, .8, .8, .8, .0, .0, .8, .0, .0, .0, .0, .8, .8, .8, .0, .8, .0, .0],
     [.0, .0, 0, 0, 0,.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0 ],
     [.0, .0, 0, 0, 0,.0, .5, .5, .5, .0,  .5, .5, .5, .0, .5, .5, .5, .0, .0, .0, .0, .0, .0],
     [.0, .0,.0,.0,.0,.0, .0, .5, .0, .0,  .5, .0, .5, .0, .5, .0, .0, .0, .0, .0, .0, .0, .0],
     [.0, .0, 0, 0, 0, 0, .0, .5, .0, .0,  .5, .5, .5, .0, .5, .5, .5, .0, .0, .0, .0, .0, .0],
     [.0, .0, 0,.0,.0,.0, .0, .5, .0, .0,  .5, .0, .0, .0, .0, .0, .5, .0, .0, .0, .0, .0, .0],
     [.0, .0, 0, 0, 0, 0, .0, .5, .0,  0,  .5, .0, .0, .0, .5, .5, .5, .0, .0, .0, .0, .0, .0],
     [.0, .0, 0, 0, 0, 0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0 ],
     ]
     
    
# Display something on hover
hover=[]
for x in range(len(bg_text)):
    hover.append([i + '<br>' + 'TPS: October 2021' + str(j)
                      for i, j in zip(text_1[x], text_2[x])])

# Invert Matrices
bg_text = bg_text[::-1]
hover =hover[::-1]
z = z[::-1]

# # Set Colorscale
# colorscale=[[0.0, '#d4d4d4'], [.2, '#996515'],
#             [.4, '#D4AF37'], [.6, 'gold'],
#             [.8, 'seagreen'],[1, '#996515']]


# Make Annotated Heatmap
fig = ff.create_annotated_heatmap(z, annotation_text=bg_text, text=hover,
                                 colorscale='Greens', font_colors=['white'], hoverinfo='text')
fig.update_layout(width=750,
                  height=450,
                 )                

fig.show()


# <a id="top"></a>    
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" role="tab" aria-controls="home">Table of Contents</h3>
# 
# * [Introduction](#1)
# * [EDA](#2)
#     * [Data Overview](#2.1)
#     * [Correlation heatmap](#2.2)
#     * [Dimension reduction using PCA](#2.3)
#     * [EDA summary](#2.4)
# * [Modeling](#3)
#     * [xgboost](#3.1)
#     * [lgbm](#3.2)
#     * [Submission](#3.3)
# 
# 
# <a id="1"></a>
# <font color="lightseagreen" size=+2.5><b>1. Introduction</b></font>
# 
# Starting from January this year, the kaggle competition team is offering a month-long tabulary playground competitions. This series aims to bridge between inclass competition and featured competitions with a friendly and approachable datasets.
# 
# For this competition, you will be predicting a binary target based on a number of feature columns given in the data. The columns are a mix of scaled continuous features and binary features.
# 
# The data is synthetically generated by a GAN that was trained on **real-world molecular response data**.
# 
# Files to work with:
# 
# * train.csv - the training data with the target column
# * test.csv - the test set; you will be predicting the target for each row in this file (the probability of the binary target)
# * sample_submission.csv - a sample submission file in the correct format
# 
# **Evaluation**: 
# 
# Submissions are evaluated on area under the **ROC curve** between the predicted probability and the observed target.

# <a id="2"></a>
# <font color="lightseagreen" size=+2.5><b>2. EDA</b></font>
# ### Imports

# In[2]:


import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
from matplotlib.ticker import FormatStrFormatter

import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="2.1"></a>
# <font color="lightseagreen" size=+1.5><b>2.1 Data overview</b></font>
# 
# #### Data size
# - Train data has 1000000 rows and 286 features including the target variable
# - Test dataset has 500000 rows and 285 features.
# 
# #### Missing Values
# - No missing values in both train and test datasets!
# 
# #### Features
# - There are 45  categorical features. All binary.
# - And 240 numerical features
# 
# #### Target
# - Binary target (1, 0)
# - Target distribution is balanced.

# In[3]:


train = pd.read_csv(r'/kaggle/input/tabular-playground-series-oct-2021/train.csv', index_col='id', nrows=100000)
test = pd.read_csv(r'/kaggle/input/tabular-playground-series-oct-2021/test.csv', index_col='id', nrows=50000)
submission= pd.read_csv(r'/kaggle/input/tabular-playground-series-oct-2021/sample_submission.csv', index_col='id', nrows=50000)


# In[4]:


print(train.shape)
print(test.shape)


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


display(train.isna().sum().sum())
display(test.isna().sum().sum())


# In[10]:


train.describe().T


# ### Target
# > Target distribution is balanced.

# In[11]:


target = train['target']


# In[12]:


pal = ['#6495ED','#ff355d']
#pal =['#ffd700', '#3299ff']
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=target, palette=pal)
ax.set_title('Target variable distribution', fontsize=20, y=1.05)

sns.despine(right=True)
sns.despine(offset=10, trim=True)


# ### Features
# * There are 45  categorical features. All binary.
# * And 240 numerical features

# In[13]:


cat_features =[]
num_features =[]

for col in train.columns:
    if train[col].dtype=='float64':
        num_features.append(col)
    elif col != 'target':
        cat_features.append(col)
#print('Catagoric features: ', cat_features)
display(len(cat_features))
#print('Numerical features: ', num_features)
display(len(num_features))


# In[14]:


train_ = train.sample(10000, random_state=2021)
test_ = test.sample(5000, random_state=2021)


# ### Categorical features

# In[15]:


def count_plot_testTrain(data1, data2, features, titleText):
    L = len(features)
    nrow= int(np.ceil(L/9))
    ncol= 9

    fig, ax = plt.subplots(nrow, ncol,figsize=(22, 12), sharey=True, facecolor='#dddddd')
    ax.flatten()
    fig.subplots_adjust(top=0.92)
    i = 1
    for feature in features:
        plt.subplot(nrow, ncol, i)
        ax = sns.countplot(x=feature, color=pal[0], data=data1, label='train')
        ax = sns.countplot(x=feature, color=pal[1], data=data2, label='test')
        ax.set_xlabel(feature)
        ax.set_ylabel('')
        ax.set_yticks([]) 
        ax.xaxis.set_label_position('top') 
        
        i += 1            
        
    lines, labels = fig.axes[-1].get_legend_handles_labels()    
    fig.legend(lines, labels, loc = 'upper right',borderaxespad= 3.0, title='data set') 
    plt.suptitle(titleText ,fontsize = 20)
    plt.show()


# In[16]:


count_plot_testTrain(train_, test_, cat_features, titleText='Train & test data categorical features count plots ')


# f22     0.137828
# f179    0.016654
# f83     0.009828
# f69     0.009279
# f74     0.008446
# f91     0.007811
# f57     0.007594
# f139    0.007479
# f268    0.007462
# f80     0.007010
# f78     0.006985
# f15     0.006949
# f16     0.006888
# f199    0.006850
# f87     0.006564
# f7      0.006350
# f17     0.006154
# f157    0.005762
# f150    0.005458
# f159    0.005427

# ### Target distribution (categorical features)
# * Note f22! The only categorical feature that seem to show some correlattion with the taget variable. (This was also highlighted in the discussion by Mikolaj and Craig Thomas)

# In[17]:


def count_plot(data, features, titleText, hue=None):
    
    L = len(features)
    nrow= int(np.ceil(L/9))
    ncol= 9
    
    fig, ax = plt.subplots(nrow, ncol,figsize=(22, 12), 
                           sharey=True, facecolor='#dddddd')
    fig.subplots_adjust(top=0.92)
    i = 1
    for feature in features:
        total = float(len(data)) 
        plt.subplot(nrow, ncol, i)
        ax = sns.countplot(x=feature, palette=pal, data=data, hue=hue)
        ax.set_xlabel(feature)
        ax.set_ylabel('')
        ax.xaxis.set_label_position('top')
        ax.set_yticks([]) 
        ax.get_legend().remove()
        i += 1
        if feature == 'f22':
            ax.set_facecolor('cyan')
          
    lines, labels = fig.axes[-1].get_legend_handles_labels()  
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right',borderaxespad= 3.0, title='target')
    
    plt.suptitle(titleText ,fontsize = 20)
    plt.show()    


# In[18]:


count_plot(train, cat_features, 'Train data cat_feats: target distribution (count plot)', hue=target)


# #### Numerical features
# #### The first 60 features

# In[19]:


def density_plotter(a, b, title):    
    L = len(num_features[a:b])
    nrow= int(np.ceil(L/10))
    ncol= 10
    fig, ax = plt.subplots(nrow, ncol,figsize=(24, 12), sharey=False, facecolor='#dddddd')

    fig.subplots_adjust(top=0.90)
    i = 1
    for feature in num_features[a:b]:
        plt.subplot(nrow, ncol, i)
        ax = sns.kdeplot(train_[feature], shade=True,  color='#6495ED',  alpha=0.85, label='train')
        ax = sns.kdeplot(test_[feature], shade=True, color='#ff355d',  alpha=0.85, label='test')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('')
        ax.set_yticks([])        
        ax.set_xticks([])
        i += 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()    
    fig.legend(lines, labels, loc = 'upper center',borderaxespad= 4.0, title='data set') 

    plt.suptitle(title, fontsize=20)
    plt.show()


# In[20]:


density_plotter(a=0, b=60, title='Density plot of numerical features: train & test data (first 60 feats)')


# #### The second 60 features

# In[21]:


density_plotter(a=60, b=120, title='Density plot of numerical features: train & test data (second 60 feats)')


# #### The third 60 features

# In[22]:


density_plotter(a=120, b=180, title='Density plot of numerical features: train & test data (third 60 feats)')


# #### The last 60 features

# In[23]:


density_plotter(a=180, b=len(train.columns), title='Density plot of numerical features: train & test data (last 60 feats)')


# <a id="2.2"></a>
# <font color="lightseagreen" size=+1.5><b>2.2 Correlation heatmap</b></font>

# In[24]:


df_num = pd.concat([train[num_features], train['target']], axis=1)


# In[25]:


fig, ax = plt.subplots(1, 1, figsize=(16 , 16))
corr = train.sample(10000, random_state=2021).corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, ax=ax, square=True, center=0, linewidth=1, vmax=0.0251, vmin=-0.0251,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        cbar_kws={"shrink": .85}, mask=mask ) 

ax.set_title('Correlation heatmap: Numerical features', fontsize=24, y= 1.05);


# In[26]:


df_cat = pd.concat([train[cat_features], train['target']], axis=1)
fig, ax = plt.subplots(1, 1, figsize=(16 , 16))
corr = df_cat.sample(10000, random_state=2021).corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, ax=ax, square=True, center=0, linewidth=1,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),vmax=0.5, vmin=-0.5,
        cbar_kws={"shrink": .85}, mask=mask )
ax.set_title('Correlation heatmap: Categorical features', fontsize=24, y= 1.05)
plt.show()


# 
# <a id="2.3"></a>
# <font color="lightseagreen" size=+1.5><b>2.3  Dimension reduction using PCA</b></font>
# 
# 

# In[27]:


from sklearn.decomposition import PCA
import plotly.express as px

df = train.sample(10000, random_state=2021)
pca = PCA(n_components=2)
components = pca.fit_transform(df)

fig = px.scatter(components, x=0, y=1,color=df['target'])
fig.update_layout(title_text='PCA (2-component): All features ', 
                  paper_bgcolor='rgb(230, 230, 230)',
                  plot_bgcolor='rgb(230, 230, 230)',
                  xaxis = {'showgrid': False},
                  yaxis = {'showgrid': False},
                  width=550, height=400,
                  titlefont={'color':'black', 'size': 24, 'family': 'San-Serif'}, 
                  )
fig.layout.coloraxis.colorbar.title = 'target'
fig.show()


# In[28]:


pca = PCA(n_components=2)
components = pca.fit_transform(df_num.sample(10000, random_state=2021))

fig = px.scatter(components, x=0, y=1,color=df['target'])
fig.update_layout(title_text='PCA (2-component): Numerical features', 
                  paper_bgcolor='rgb(230, 230, 230)',
                  plot_bgcolor='rgb(230, 230, 230)',
                  xaxis = {'showgrid': False},
                  yaxis = {'showgrid': False},
                  width=550, height=400,
                  titlefont={'color':'black', 'size': 24, 'family': 'San-Serif'}, 
                  )
fig.layout.coloraxis.colorbar.title = 'target'
fig.show()


# ### Explained variance 
# - Reducing the features space to 50 components could 'only' explain 79% of the variance. 
# - 100 features could explain around 91% of the total variance.

# In[29]:


pca = PCA(n_components=50, random_state=2021)
pca.fit_transform(df)
print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())


# In[30]:


n_comp = np.arange(25, 286, 25)
exp_variance =[]

for n in n_comp:
    pca = PCA(n, random_state=2021)
    pca.fit_transform(df)
    exp_variance.append(pca.explained_variance_ratio_.sum())

zip(n_comp, exp_variance)

plt.figure(figsize=(8, 6))
plt.scatter(n_comp, exp_variance)
plt.xlabel('n_compenents')
plt.ylabel('% explainde variance')
plt.grid()
plt.title('Explained variance by n-features', fontsize=20, y=1.05);


# <a id="2.4"></a>
# <font color="lightseagreen" size=+1.5><b>2.4 EDA summary</b></font>
# 
# - There are no missing values in both train and test dataset.
# - The distribution of target variable is balanced, almost 50%-50%.
# - The distribution of features in both train and test dataset is similar.
# - Correlation of most features with target variables is week. Only f22 (cat_feature) is visible on the heatmap with correlation coeficient of around |0.5|.
# 

# <a id="3"></a>
# <font color="lightseagreen" size=+2.5><b>3. Modeling</b></font>
# 
# * Using the start code from last month by [kaggle competition team](https://www.kaggle.com/ryanholbrook/getting-started-september-2021-tabular-playground)
# * 5-fold average

# In[31]:


import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate
import warnings 
warnings.filterwarnings('ignore')

data_dir = Path('../input/tabular-playground-series-oct-2021/')

df_train = pd.read_csv(
    data_dir / "train.csv",
    index_col='id',
     nrows=50000,
)

X_test = pd.read_csv(
    data_dir / "test.csv",
    index_col='id',
    nrows=50000,
)

FEATURES = df_train.columns[:-1]
TARGET = df_train.columns[-1]

X = df_train.loc[:, FEATURES]
y = df_train.loc[:, TARGET]

seed = 0
fold = 5


# In[32]:


cat_features =[]
num_features =[]

for col in X.columns:
    if X[col].dtype=='float64':
        num_features.append(col)
    elif col != 'target':
        cat_features.append(col)

display(len(cat_features))
display(len(num_features))


# ## Feature Engineering
# There are several option to try while expermenting in creating new features (feature engineering). Here we two methods: features created from generic statistics and creating new feaures by combining existing. 
# 
# ### 1. Make features form generic statistics (min, max, std, mean)

# In[33]:


# few aditional features
X['max'] = X[num_features].max(axis=1)
X['min'] = X[num_features].min(axis=1)
X['mean'] = X[num_features].mean(axis=1)
X['std'] = X[num_features].std(axis=1)

X_test['max'] = X_test[num_features].max(axis=1)
X_test['min'] = X_test[num_features].min(axis=1)
X_test['mean'] = X_test[num_features].mean(axis=1)
X_test['std'] = X_test[num_features].std(axis=1)


# ### 2. Feature combination
# - mutual information

# In[34]:


get_ipython().run_cell_magic('time', '', 'from sklearn.feature_selection import mutual_info_classif\n\nfeatures = X.columns == int\n\ndef make_mi_scores(X, y, discrete_features):\n    mi_scores = mutual_info_classif(X, y, discrete_features=features)\n    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)\n    mi_scores = mi_scores.sort_values(ascending=False)\n    return mi_scores\n\nmi_scores = make_mi_scores(X, y, features)\n')


# In[35]:


mi_scores[:25]


# In[36]:


mi_scores[0:25].index


# In[37]:


ff = ['f208', 'f207', 'f90', 'f205', 'f204', 'f91', 'f93', 'f97', 'f197',
       'f181', 'f99', 'f102', 'f194', 'f106', 'f107', 'f108', 'f190', 'f110',
       'f187', 'f186', 'f185', 'f184', 'f112', 'f182', 'f142']
 
gg = ['f22', 'f179', 'f69', 'f87', 'f57', 'f58', 'f86', 'f0', 'f136', 'f160',
       'f80', 'f21', 'f271', 'f10', 'f270', 'f1', 'f78', 'f245', 'f172',
       'f163', 'f103', 'f261', 'f164', 'f202', 'f18']


# In[38]:


def plot_utility_scores(scores):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y, color='#ff355d', alpha=0.9)
    plt.yticks(width, ticks)
    plt.grid()
    plt.title("Mutual Information Scores (top 25)")


plt.figure(dpi=100, figsize=(8, 5), facecolor='lightgray')
plot_utility_scores(mi_scores[:25])


# - Trials at feature creation

# In[39]:


X['f22_179_139'] = X['f22'] * X['f179'] * X['f139']
X_test['f22_179_139'] = X_test['f22'] * X_test['f179'] *X_test['f139']

X['bb'] = X['f22'] * X['f83'] * X['f139']
X_test['bb'] = X_test['f122'] * X_test['f83'] *X_test['f139']

X['cc'] = X['f69'] * X['f74'] *X['f139']
X_test['cc'] = X_test['f69'] * X_test['f74'] *X_test['f139']

X['dd'] = X['f7'] * X['f17'] *X['f139']
X_test['dd'] = X_test['f7'] * X_test['f17'] *X_test['f139']


# In[40]:


X.shape, X_test.shape


# <a id="3.1"></a>
# <font color="lightseagreen" size=+1.5><b>3.1 XGBoost</b></font>
# 

# In[41]:


model_xgb = XGBClassifier(max_depth=3,
    subsample = 0.7,
    colsample_bylevel = 0.6,
    min_child_weight= 156,
    reg_lambda = 0.1,
    reg_alpha = 10,
    colsample_bytree=.2,
    n_jobs=-1,
    tree_method='gpu_hist',
    sampling_method='gradient_based', 
    random_state= seed,
)
def score(X, y, model_xgb, cv):
    scoring = ["roc_auc"]
    scores = cross_validate(
        model_xgb, X, y, scoring=scoring, cv=cv, return_train_score=True
    )
    scores = pd.DataFrame(scores).T
    return scores.assign(
        mean = lambda x: x.mean(axis=1),
        std = lambda x: x.std(axis=1),
    )

scores = score(X, y, model_xgb, cv=fold)
display(scores)


# In[42]:


model_xgb.fit(X, y, eval_metric='auc')
y_pred_xgb = pd.Series(
    model_xgb.predict_proba(X_test)[:, 1],
    index=X_test.index,
    name=TARGET,
)
y_pred_xgb.to_csv("submission_xgb.csv")


# <a id="3.2"></a>
# <font color="lightseagreen" size=+1.5><b>3.2 LGBM</b></font>
# 

# In[43]:


model_lgbm = LGBMClassifier(
    max_depth=3,
    boosting_type ='gbdt',
    num_leaves=500,
    objective = "binary",
    reg_lambda = 1.0,
    reg_alpha = 0.5,
    colsample_bytree=.2,
    n_jobs=-1,
    feature_pre_filter = False,  
    device_type = 'gpu',
    )
def score(X, y, model_lgbm, cv):
    scoring = ["roc_auc"]
    scores = cross_validate(
        model_lgbm, X, y, scoring=scoring, cv=cv, return_train_score=True
    )
    scores = pd.DataFrame(scores).T
    return scores.assign(
        mean = lambda x: x.mean(axis=1),
        std = lambda x: x.std(axis=1),
    )

scores = score(X, y, model_lgbm, cv=fold)
display(scores)


# <a id="3.3"></a>
# <font color="lightseagreen" size=+1.5><b>3.3 Submission</b></font>
# 

# In[44]:


model_lgbm.fit(X, y, eval_metric='auc')
y_pred_lgbm = pd.Series(
    model_lgbm.predict_proba(X_test)[:, 1],
    index=X_test.index,
    name=TARGET,
)
y_pred_lgbm.to_csv("submission_lgbm.csv")


# In[45]:


sub_file = pd.read_csv("submission_lgbm.csv")
sub_file


# ### End of notebook! 
# 

# <!-- # feature_important = model_xgb.get_booster().get_score(importance_type='weight')
# # keys = list(feature_important.keys())
# # values = list(feature_important.values())
# 
# # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
# # data.nlargest(20, columns="score").plot(kind='barh', figsize = (16,8)) -->

# In[ ]:




