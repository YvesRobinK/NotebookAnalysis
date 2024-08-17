#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# <a id="table-of-contents"></a>
# 1. [Introduction](#introduction)
# 2. [Preparation](#preparation)
# 3. [General](#general)
#     * 3.1. [No of rows and columns](#rows_columns)
#     * 3.2. [No of missing values](#missing_values)
#     * 3.3. [First 5 rows](#first_5_rows)
#     * 3.4. [Basic statistics on continuous features](#basic_statistics_cont)
#     * 3.5. [Count of categorical features](#count_cat)
# 4. [Features & Target Correlation](#features_target_correlation)
#     * 4.1. [Correlation between features](#features_correlation)
#     * 4.2. [Correlation with target](#target_correlation)
# 5. [Features Engineering](#features_engineering)
#     * 5.1. [Continuous Features](#fe_continuous)
# 6. [Target Encoding](#target_encoding)
#     * 6.1. [Mean Encoding](#mean_encoding)
#     * 6.2. [Minimum Encoding](#min_encoding)
#     * 6.3. [Maximum Encoding](#max_encoding)
# 7. [Winners Solutions](#winners_solutions)

# [back to top](#table-of-contents)
# <a id="introduction"></a>
# # 1. Introduction
# 
# Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.
# 
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features.

# [back to top](#table-of-contents)
# <a id="preparation"></a>
# # 2. Preparation

# In[1]:


import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-feb-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-feb-2021/test.csv')


# [back to top](#table-of-contents)
# <a id="general"></a>
# # 3. General
# 
# **Observations:**
# * Train set has 300,000 rows while test set has 200,000 rows.
# * There are 10 categorical features from `cat0` - `cat9` and 14 continuous features from `cont0` - `cont13`.
# * There is no missing values in the train and test dataset but there is no category `G` in `cat6` test dataset.
# * Categorical features ranging from alphabet `A` - `O` but it varies from each categorical feature with `cat0`, `cat1`, `cat3`, `cat5` and `cat6` are dominated by one category.
# * Continuous features on train anda test dataset ranging from -0.1 to 1 which are a multimodal distribution and they are resemble each other.
# * `target` has a range between 0 to 10.3 and has a bimodal distribution.
# 
# **Ideas:**
# * Drop features that are dominated by one category `cat0`, `cat1`, `cat3`, `cat5` and `cat6` as they don't give variation to the dataset but further analysis still be needed.

# In[3]:


cat_features = [feature for feature in train_df.columns if 'cat' in feature]
cont_features = [feature for feature in train_df.columns if 'cont' in feature]


# [back to top](#table-of-contents)
# <a id="rows_columns"></a>
# ## 3.1. No of rows and columns

# In[4]:


print('Rows and Columns in train dataset:', train_df.shape)
print('Rows and Columns in test dataset:', test_df.shape)


# [back to top](#table-of-contents)
# <a id="missing_values"></a>
# ## 3.2. No of missing values

# In[5]:


print('Missing values in train dataset:', sum(train_df.isnull().sum()))
print('Missing values in test dataset:', sum(test_df.isnull().sum()))


# [back to top](#table-of-contents)
# <a id="first_5_rows"></a>
# ## 3.3. First 5 rows

# **First 5 rows in the train dataset**

# In[6]:


train_df.head()


# **First 5 rows in the test dataset**

# In[7]:


test_df.head()


# [back to top](#table-of-contents)
# <a id="basic_statistics_cont"></a>
# ## 3.4. Basic statistics on continuous features

# **Train dataset**

# In[8]:


fig = plt.figure(figsize=(15, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 4):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.3, 5.3, 'Continuous Features Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.3, 4.7, 'Continuous features have multimodal', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cont_features:
    sns.kdeplot(train_df[col], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(col, fontsize=10, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.2, 1.2)
    locals()["ax"+str(run_no)].set_xlabel('')
    run_no += 1
    
ax14.remove()
ax15.remove()


# In[9]:


train_df[cont_features].describe()


# In[10]:


fig = plt.figure(figsize=(10, 3.5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(background_color)
ax0.set_yticklabels([])
ax0.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

ax0.text(-0.5, 0.5, 'Target Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'Target has a bimodal distribution', fontsize=15, fontweight='light', fontfamily='serif')        

sns.kdeplot(train_df['target'], ax=ax0, shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
ax0.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlim(-0.5, 10.5)
ax0.set_xlabel('')
ax0.set_ylabel('')

plt.show()


# In[11]:


print('Target')
train_df['target'].describe()


# **Test dataset**

# In[12]:


fig = plt.figure(figsize=(15, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 4):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.3, 5.3, 'Continuous Features Distribution on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.3, 4.7, 'Continuous features on test dataset resemble train dataset', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cont_features:
    sns.kdeplot(test_df[col], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(col, fontsize=10, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.2, 1.2)
    locals()["ax"+str(run_no)].set_xlabel('')
    run_no += 1
    
ax14.remove()
ax15.remove()


# In[13]:


test_df[cont_features].describe()


# [back to top](#table-of-contents)
# <a id="count_cat"></a>
# ## 3.5. Count of categorical features

# In[14]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.8, 115, 'Count of categorical features on Train dataset (%)', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.8, 107, 'Some features are dominated by one category', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(train_df[col].value_counts() / len(train_df) * 100)
    sns.barplot(x=chart_df.index, y=chart_df[col], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1


# In[15]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.8, 109, 'Count of categorical features on Test dataset (%)', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.8, 101, 'Some features are dominated by one category', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(test_df[col].value_counts() / len(test_df) * 100)
    sns.barplot(x=chart_df.index, y=chart_df[col], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1


# [back to top](#table-of-contents)
# <a id="features_target_correlation"></a>
# # 4. Features & Target Correlation
# **Observations:**
# * Highest correlation between features is 0.6.
# * Correlation between features on train and test dataset are quite similar.
# * There is no continuous features that has correlation with `target` above/below +/- 0.04.
# * `cont9` has the lowest correlation with target, almost reaching 0.
# * There is a distinct separation on `cont1` relative to the `target`.

# [back to top](#table-of-contents)
# <a id="features_correlation"></a>
# # 4.1. Correlation between features

# In[16]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(18, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(0, -1, 'Features Correlation on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(0, -0.4, 'Highest correlation in the dataset is 0.6', fontsize=13, fontweight='light', fontfamily='serif')

ax1.set_facecolor(background_color)
ax1.text(-0.1, -1, 'Features Correlation on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax1.text(-0.1, -0.4, 'Features in test dataset resemble features in train dataset ', 
         fontsize=13, fontweight='light', fontfamily='serif')

sns.heatmap(train_df[cont_features].corr(), ax=ax0, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

sns.heatmap(test_df[cont_features].corr(), ax=ax1, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

plt.show()


# [back to top](#table-of-contents)
# <a id="target_correlation"></a>
# # 4.2. Correlation with target
# 
# ### 4.2.1 Continuous Features

# In[17]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(12, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(-1.1, 0.048, 'Correlation of Continuous Features with Target', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.1, 0.045, 'There is no features that pass 0.04 correlation with target', fontsize=13, fontweight='light', fontfamily='serif')

chart_df = pd.DataFrame(train_df[cont_features].corrwith(train_df['target']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
ax0.grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

plt.show()


# In[18]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in cont_features:
        sns.scatterplot(x=train_df[feature], y=train_df['target'] ,ax=locals()["ax"+str(run_no)], color='#2f5586', linewidth=0.3, edgecolor='black')
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        run_no += 1
        
ax0.text(-0.5, 14, 'Features and Target Relation', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 12.4, 'cont1 has a distinct separation', fontsize=13, fontweight='light', fontfamily='serif')

ax14.remove()
ax15.remove()

plt.show()


# ### 4.2.2 Categorical Features

# In[19]:


cat = 'cat0'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat0" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[20]:


cat = 'cat1'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat1" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[21]:


cat = 'cat2'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat2" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[22]:


cat = 'cat3'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat3" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[23]:


cat = 'cat4'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat4" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[24]:


cat = 'cat5'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(60, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, len(value)):
    locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(run_no)].set_facecolor(background_color)
    locals()["ax"+str(run_no)].set_yticklabels([])
    locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
    for s in ["top","right", 'left']:
        locals()["ax"+str(run_no)].spines[s].set_visible(False)
    run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat5" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[25]:


cat = 'cat6'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(40, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 2):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat6" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[26]:


cat = 'cat7'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(40, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 2):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat7" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# In[27]:


cat = 'cat8'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(40, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 2):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.5, 0.52, 'Target Distribution on "cat8" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1
    
ax7.remove()


# In[28]:


cat = 'cat9'
value = pd.Series(train_df[cat].value_counts().sort_index().index)

fig = plt.figure(figsize=(30, (len(value)*2.5)), facecolor='#f6f5f5')
gs = fig.add_gridspec(len(value), 5)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 3):
    for row in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.5, 0.6, 'Target Distribution on "cat9" feature ', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.54, 'To see how target is distributed across each value', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for val in value:
    sns.kdeplot(train_df[train_df[cat]==val]['target'], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(val, fontsize=20, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1.015, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.5, 10.5)
    run_no += 1


# [back to top](#table-of-contents)
# <a id="features_engineering"></a>
# # 5. Features Engineering
# 
# This section will try to create a new features from existing features and see the relation with the target. The new features haven't been implemented to a model and still unknown it's effectiveness.
# 
# <a id="fe_continuous"></a>
# ## 5.1. Continuous Features 

# In[29]:


train_fe_df = train_df.copy()
for col in cont_features:
    train_fe_df[col] = np.log(train_fe_df[col])


# In[30]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in cont_features:
        sns.scatterplot(x=train_fe_df[feature], y=train_fe_df['target'] ,ax=locals()["ax"+str(run_no)], color='#2f5586', linewidth=0.3, edgecolor='black')
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        run_no += 1
        
ax0.text(-12, 14, 'Log of Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-12, 12.8, 'Create a log of continuous feature and compare it with the target', fontsize=13, 
         fontweight='light', fontfamily='serif')

ax14.remove()
ax15.remove()

plt.show()


# In[31]:


train_fe_df = train_df.copy()
train_fe_df['min'] = train_fe_df[cont_features].min(axis=1)


# In[32]:


fig = plt.figure(figsize=(10, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.tick_params(axis='y', which=u'both',length=0)
ax1.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(-0.2, 12, 'Minimum and Log Minimum of All Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.2, 11.3, 'Minimum and log minimum of all continuous features compared with target ', fontsize=15, 
         fontweight='light', fontfamily='serif')        

sns.scatterplot(x=train_fe_df['min'], y=train_fe_df['target'] ,ax=ax0, color='#2f5586', linewidth=0.3, edgecolor='black')
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlabel('Min of all feature')  

sns.scatterplot(x=np.log(train_fe_df['min']), y=train_fe_df['target'] ,ax=ax1, color='#2f5586', linewidth=0.3, edgecolor='black')
ax1.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.set_xlabel('Log min of all feature')

plt.show()


# In[33]:


train_fe_df = train_df.copy()
train_fe_df['max'] = train_fe_df[cont_features].max(axis=1)


# In[34]:


fig = plt.figure(figsize=(10, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.tick_params(axis='y', which=u'both',length=0)
ax1.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(0.2, 12, 'Maximum and Log Maximum of All Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(0.2, 11.3, 'Maximum and log maximum of all continuous features compared with target ', fontsize=15, 
         fontweight='light', fontfamily='serif')        

sns.scatterplot(x=train_fe_df['max'], y=train_fe_df['target'] ,ax=ax0, color='#2f5586', linewidth=0.3, edgecolor='black')
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlabel('Max of all feature')  

sns.scatterplot(x=np.log(train_fe_df['max']), y=train_fe_df['target'] ,ax=ax1, color='#2f5586', linewidth=0.3, edgecolor='black')
ax1.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.set_xlabel('Log max of all feature')

plt.show()


# In[35]:


train_fe_df = train_df.copy()
train_fe_df['cont_sum'] = train_fe_df[cont_features].sum(axis=1)


# In[36]:


fig = plt.figure(figsize=(10, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.tick_params(axis='y', which=u'both',length=0)
ax1.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(2, 12, 'Sum and Log Sum of All Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(2, 11.3, 'Sum and log sum of all continuous features compared with target ', fontsize=15, 
         fontweight='light', fontfamily='serif')        

sns.scatterplot(x=train_fe_df['cont_sum'], y=train_fe_df['target'] ,ax=ax0, color='#2f5586', linewidth=0.3, edgecolor='black')
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlabel('Sum of all feature')  

sns.scatterplot(x=np.log(train_fe_df['cont_sum']), y=train_fe_df['target'] ,ax=ax1, color='#2f5586', linewidth=0.3, edgecolor='black')
ax1.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.set_xlabel('Log sum of all feature')

plt.show()


# In[37]:


train_fe_df = train_df.copy()
train_fe_df['cont_multiply'] = 1
for col in cont_features:
    train_fe_df['cont_multiply'] = train_fe_df[col] * train_fe_df['cont_multiply']


# In[38]:


fig = plt.figure(figsize=(10, 5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.tick_params(axis='y', which=u'both',length=0)
ax1.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(-0.01, 12, 'Multiplication and Log Multiplication of All Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.01, 11.3, 'Multiplication and multiplication log of all continuous features compared with target ', fontsize=15, 
         fontweight='light', fontfamily='serif')        

sns.scatterplot(x=train_fe_df['cont_multiply'], y=train_fe_df['target'] ,ax=ax0, color='#2f5586', linewidth=0.3, edgecolor='black')
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlabel('Multiplication of all feature')  

sns.scatterplot(x=np.log(train_fe_df['cont_multiply']), y=train_fe_df['target'] ,ax=ax1, color='#2f5586', linewidth=0.3, edgecolor='black')
ax1.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.set_xlabel('Log multiplication of all feature')

plt.show()


# In[39]:


train_fe_df = train_df.copy()
train_fe_df['cont_sum'] = train_fe_df[cont_features].sum(axis=1)
for col in cont_features:
    train_fe_df[col] = train_fe_df[col] / train_fe_df['cont_sum']
train_fe_df = train_fe_df.drop('cont_sum', axis=1)


# In[40]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in cont_features:
        sns.scatterplot(x=train_fe_df[feature], y=train_fe_df['target'] ,ax=locals()["ax"+str(run_no)], color='#2f5586', linewidth=0.3, edgecolor='black')
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        run_no += 1
        
ax0.text(-0.1, 14, 'Prorate Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.1, 12.8, 'Prorate continuous feature to percentage in a row and compare it with target', fontsize=13, 
         fontweight='light', fontfamily='serif')

ax14.remove()
ax15.remove()

plt.show()


# In[41]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in cont_features:
        sns.scatterplot(x=np.log(train_fe_df[feature]), y=train_fe_df['target'] ,ax=locals()["ax"+str(run_no)], color='#2f5586', linewidth=0.3, edgecolor='black')
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        run_no += 1
        
ax0.text(-13, 14, 'Log Prorate Continuous Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-13, 12.8, 'Log Prorate continuous feature to percentage in a row and compare it with target', fontsize=13, 
         fontweight='light', fontfamily='serif')

ax14.remove()
ax15.remove()

plt.show()


# [back to top](#table-of-contents)
# <a id="mean_encoding"></a>
# # 6. Target Encoding
# 
# **Observations:**
# * There is no disctinct `target mean` in each categorical features which mostly around 7.
# * `Target minimum` varies among categorical features but mostly are below 6.
# * `Target maximum` is in range 8 - 10 and quite consistent among categorical features.
# 
# <a id="mean_encoding"></a>
# ## 6.1. Mean Encoding 

# In[42]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.7, 9, 'Mean Encoding by Categorical Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.7, 8.3, 'There is no distinct mean differences in the target', fontsize=13, fontweight='light', fontfamily='serif')
run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(train_df.groupby(col)['target'].mean()).reset_index()
    sns.barplot(x=chart_df[col], y=chart_df['target'], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1


# [back to top](#table-of-contents)
# <a id="min_encoding"></a>
# ## 6.2. Minimum Encoding 

# In[43]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.7, 4.1, 'Minimum Encoding by Categorical Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.7, 3.8, 'Target minimum varies among categorial features', fontsize=13, fontweight='light', fontfamily='serif')
run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(train_df.groupby(col)['target'].min()).reset_index()
    sns.barplot(x=chart_df[col], y=chart_df['target'], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1


# [back to top](#table-of-contents)
# <a id="max_encoding"></a>
# ## 6.3. Maximum Encoding 

# In[44]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.7, 12.3, 'Maximum Encoding by Categorical Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.7, 11.3, 'Target maximum is quite flat among categorial features', fontsize=13, fontweight='light', fontfamily='serif')
run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(train_df.groupby(col)['target'].max()).reset_index()
    sns.barplot(x=chart_df[col], y=chart_df['target'], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1


# [back to top](#table-of-contents)
# <a id="winners_solutions"></a>
# # 7. Winners Solutions
# Congratulations for all the winners and thank you for sharing your solution. Below are the winners and their solutions:
# * 1st place position: [Ren](https://www.kaggle.com/ryanzhang) - [1st place DAE training code](https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745)
# * 2nd place position: [Dave E](https://www.kaggle.com/davidedwards1) - [#2 LB Approach](https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222762)
# * 3rd place position: [Ken](https://www.kaggle.com/kntyshd) - [3rd place solution (just ensembling GBDTs)](https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/223455)
