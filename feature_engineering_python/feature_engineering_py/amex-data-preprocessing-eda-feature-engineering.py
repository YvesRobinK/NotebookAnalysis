#!/usr/bin/env python
# coding: utf-8

# # <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 48px; color: #CD5C5C; text-align: center; line-height: 1.25;"><b>🏬🔧 Data Pre-processing,<span style="color: #40E0D0"> EDA & Feature Engineering 📉</span></b><br><span style="color: #DE3163; font-size: 24px">American Express </span></h1>
# <hr>

# # <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">1. | Introduction 👋</div>
# <center>
#     <img src="https://www.americanexpress.com/content/dam/amex/be/nl/kaarten/gold-kaart/960-608-chg-gold-card.png" alt="Mart" width="80%">
# </center>
# <br>

# ## <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">Data Set Problems 🤔</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 American Express is a globally integrated payments company. The largest payment card issuer in the world, they provide customers with access to products, insights, and experiences that enrich lives and build business success. <br>
#     👉 <mark>We’ll be apply our machine learning skills to predict credit default</mark> which allows lenders to optimize lending decisions. <br> 
#     👉 <mark><b>Data pre-processing and feature engineering will be performed to prepare the dataset</b></mark> before it is used by the machine learning model.
# </div>
# 
# ## <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">Objectives of Notebook 📌</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 <b>This notebook aims to:</b>
#     <ul>
#         <li> Perform <mark><b>initial data exploration</b></mark>.</li>
#         <li> Perform <mark><b>data pre-processing</b></mark>.</li>
#         <li> Perform <mark><b>EDA</b></mark> and <mark><b>hypothesis testing (statistical and non-statistical)</b></mark> in cleaned data set.</li>
#         <li> Perform <mark><b>feature engineering (one-hot encoding, label encoding, and binning)</b></mark>.</li>
#     </ul>
# </div>

# ## <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">Data Set Description 🧾</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories::
#     <ul>
#         <li> <mark><b>D_* </b></mark> = Delinquency variables,</li>
#         <li> <mark><b>S_* </b></mark> = Spend variables,</li>
#         <li> <mark><b>P_* </b></mark> = Payment variables </li>
#         <li> <mark><b>B_* </b></mark> = Balance variables </li>
#         <li> <mark><b>R_* </b></mark> = = Risk variables  </li>        
#     </ul><br>
#     
# with the following features being categorical:
# 
# ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']   
# 
# Our task is to predict, for each customer_ID, the probability of a future payment default (target = 1).
# **Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.**
#     

# # <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">2. | Importing Libraries 📚</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 <b>Importing libraries</b> that will be used in this notebook.
# </div>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
import missingno as mso
import plotly.graph_objects as go
import plotly.offline as po
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import random
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# # <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">3. | Reading Dataset 👓</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 After importing libraries, <b>the dataset that will be used will be imported</b>.
# </div>

# In[2]:


train_data=pd.read_feather("/kaggle/input/amexfeather/train_data.ftr")
train_labels=pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")


# In[3]:


# --- Reading Dataset ---
train_data.head().style.background_gradient(cmap='Greens').set_properties(**{'font-family': 'Segoe UI'}).hide_index()


# In[4]:


# --- Reading Dataset ---
train_labels.head().style.background_gradient(cmap='Greens').set_properties(**{'font-family': 'Segoe UI'}).hide_index()


# # <div style="font-family: Trebuchet MS; background-color: #CD5C5C; color: #FFFFFF; padding: 12px; line-height: 1.5;">4. | Initial Data Exploration 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>initial data exploration</b> before pre-process the data.
# </div>

# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1 | Train Labels Data Exploration 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Train Labels data set exploration</b> before pre-process the data.
# </div>

# In[5]:


# Shape of the train label data set

print('\033[1m'"Shape of the train label data file\n"'\033[0m',train_labels.shape)


# In[6]:


## Data Type
print('\033[1m'"Data types of each column in train label data file\n"'\033[0m',train_labels.dtypes)


# In[7]:


#Missing value in the table 

print('\033[1m'"Missing value present in each column of train label data file\n"'\033[0m',train_labels.isna().any())


# In[8]:


# Check for the duplicated values 

print('\033[1m'"Duplicate value present in each column of train label data file\n"'\033[0m',train_labels.customer_ID.duplicated().any())



# In[9]:


# No of unique customers 

print('\033[1m'"No of Unique customers in the  train label data file\n"'\033[0m',train_labels.customer_ID.nunique())


# In[10]:


# Count of the Target

print('\033[1m'"Value count of the target column in the train label data file\n"'\033[0m',train_labels.target.value_counts())


# In[11]:


target=train_labels.target.value_counts(normalize=True)
target.rename(index={1:'Default',0:'Paid'},inplace=True)
colors = ['#17becf', '#E1396C']
data = go.Pie(
values= target,
labels= target.index,
marker=dict(colors=colors),
textinfo='label+percent'
)
layout = go.Layout(
title=dict(text = "Target Distribution",x=0.46,y=0.95,font_size=20)
)
fig = go.Figure(data=data,layout=layout)
fig.show()


# In[12]:


del train_labels


# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1 | Train Data Exploration 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Train  data set exploration</b> before pre-process the data.
# </div>

# In[13]:


# Shape of the train data set

print('\033[1m'"Shape of the train  data file\n"'\033[0m',train_data.shape)


# In[14]:


## Data Type
print('\033[1m'"Data types of each column in train label data file\n"'\033[0m',train_data.dtypes)


# In[15]:


train_data.info(max_cols=200, show_counts=True)


# Following are the categorical variables 'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'. We are going to analyze it w.r.t to Target

# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1.1 | Analyzing the Balance categorical variables 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Analyzing the Balance categorical variables</b> before pre-process the data.
# </div>
#  
#  

# In[16]:


train_data["B_30"].value_counts()


# In[17]:


# --- Setting Colors, Labels, Order ---
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
cyan_grad = ['#142459', '#176BA0', '#19AADE', '#1AC9E6', '#87EAFA']
colors=cyan_grad
labels=train_data['B_30'].dropna().unique()
order=train_data['B_30'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16, 8))
plt.suptitle('B_30 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='B_30', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+100,rect.get_height(), horizontalalignment='center',
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))

plt.xlabel('B_30 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['B_30'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%',
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 29)
print('\033[1m'+'.: B_30 Content Total :.'+'\033[0m')
print('\033[36m*' * 29+'\033[0m')
train_data.B_38.value_counts(dropna=False)


# In[18]:


# --- Setting Colors, Labels, Order ---
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
cyan_grad = ['#142459', '#176BA0', '#19AADE', '#1AC9E6', '#87EAFA']
colors=cyan_grad
labels=train_data['B_38'].dropna().unique()
order=train_data['B_38'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(16, 8))
plt.suptitle('B_38 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='B_38', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+100,rect.get_height(), horizontalalignment='center',
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))

plt.xlabel('B_38 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['B_38'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%',
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre)

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 29)
print('\033[1m'+'.: B_38 Content Total :.'+'\033[0m')
print('\033[36m*' * 29+'\033[0m')
train_data.B_38.value_counts(dropna=False)


# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1.2| Analyzing the Delinquency categorical variables 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Analyzing the Delinquency categorical variables</b> before pre-process the data.
# </div>
# 
# 
# 
# 

# In[19]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_114'].dropna().unique()
order=train_data['D_114'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_114 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_114', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_114 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_114'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_114 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_114.value_counts(dropna=False)


# In[20]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_116'].dropna().unique()
order=train_data['D_116'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_116 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_116', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_116 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_116'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_116 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_116.value_counts(dropna=False)


# In[21]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_117'].dropna().unique()
order=train_data['D_117'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_117 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_117', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_117 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_117'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_117 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_117.value_counts(dropna=False)


# In[22]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_120'].dropna().unique()
order=train_data['D_120'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_120 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_120', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_120 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_120'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_120 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_120.value_counts(dropna=False)


# In[23]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_126'].dropna().unique()
order=train_data['D_126'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_126 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_126', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_126 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_126'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_126 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_126.value_counts(dropna=False)


# In[24]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_63'].dropna().unique()
order=train_data['D_63'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_63 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_63', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_63 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_63'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_63 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_63.value_counts(dropna=False)


# In[25]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_64'].dropna().unique()
order=train_data['D_64'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_64 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_64', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_64 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_64'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_64 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_64.value_counts(dropna=False)


# In[26]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_66'].dropna().unique()
order=train_data['D_66'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_66 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_66', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_66 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_66'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_66 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_66.value_counts(dropna=False)


# In[27]:


# --- Setting Colors, Labels, Order ---
purple_grad = ['#491D8B', '#6929C4', '#8A3FFC', '#A56EFF', '#BE95FF']
colors=purple_grad
labels=train_data['D_68'].dropna().unique()
order=train_data['D_68'].value_counts().index

# --- Size for Both Figures ---
plt.figure(figsize=(18, 8))
plt.suptitle('D_68 Distribution', fontweight='heavy', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])

# --- Histogram ---
countplt = plt.subplot(1, 2, 1)
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_68', data=train_data, palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, rect.get_height()+20,rect.get_height(), horizontalalignment='center', 
             fontsize=12, bbox=dict(facecolor='none', edgecolor=black_grad[0], linewidth=0.15, boxstyle='round'))
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_68 Distribution', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
countplt

# --- Pie Chart ---
plt.subplot(1, 2, 2)
plt.title('Pie Chart', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
plt.pie(train_data['D_68'].value_counts(), colors=colors, labels=order, pctdistance=0.67, autopct='%.2f%%', 
        wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]), textprops={'fontsize':12})
centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
plt.gcf().gca().add_artist(centre);

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('\033[36m*' * 30)
print('\033[1m'+'.: D_68 Total :.'+'\033[0m')
print('\033[36m*' * 30+'\033[0m')
train_data.D_68.value_counts(dropna=False)


# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1.3| Analyzing the categorical variables w.r.t Target 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Analyzing the categorical variables w.r.t Target</b> before pre-process the data.
# </div>
# 

# In[28]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='B_30', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('B_30 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[29]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='B_38', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('B_38 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[30]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_114', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_114 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[31]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_116', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_116 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[32]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_117', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_117 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[33]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_120', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_120 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[34]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_126', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_126 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[35]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_66', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_66 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# In[36]:


plt.figure(figsize=(12,10))
plt.title('Histogram', fontweight='bold', fontsize=14, fontfamily='sans-serif', color=black_grad[0])
ax = sns.countplot(x='D_68', data=train_data,hue='target', palette=colors, order=order, edgecolor=black_grad[2], alpha=0.85)
plt.tight_layout(rect=[0, 0.04, 1, 0.965])
plt.xlabel('D_68 Distribution w.r.t target', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', color=black_grad[1])
plt.grid(axis='y', alpha=0.4)
plt.show()


# # <div style="font-family: Trebuchet MS; background-color: #db8a8a; color: #FFFFFF; padding: 12px; line-height: 1.5;">4.1.4| Analyzing the numerical attributes 🔍</div>
# <div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
#     👉 This section will focused on <b>Analyzing the numerical attributes</b> before pre-process the data.
# </div>
# 

# ## Data analysis on the Balance attributes

# In[37]:


# Categorcal columns 

categorical_cols=['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68',
          'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'target']


# In[38]:


cols=[col for col in train_data.columns if (col.startswith(('B','T'))) & (col not in categorical_cols[:-1])]

train_balance_cols =train_data[cols]


# In[39]:


train_balance_cols.select_dtypes(exclude='object').describe().T.style.background_gradient(cmap='RdPu').set_properties(**{'font-family': 'Segoe UI'})


# In[40]:


# --- Plot Missing Values ---
mso.bar(train_balance_cols, fontsize=9, color=[purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0],
                               purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[1], purple_grad[1]], 
        figsize=(15, 8), sort='descending', labels=True)

# --- Title & Subtitle Settings ---
plt.suptitle('Missing Values in each Columns', fontweight='heavy', x=0.124, y=1.22, ha='left',fontsize='16', 
             fontfamily='sans-serif', color=black_grad[0])
plt.title('Almost all columns have  missing value.\n\nThe total of missing values in each column is less than 25%, which means that imputation can still be done to fill in the missing values in the\ntwo columns.', 
          fontsize='8', fontfamily='sans-serif', loc='left', color=black_grad[1], pad=5)
plt.grid(axis='both', alpha=0);

# --- Total Missing Values in each Columns ---
print('\033[36m*' * 43)
print('\033[1m'+'.: Total Missing Values in each Columns :.'+'\033[0m')
print('\033[36m*' * 43+'\033[0m')
train_balance_cols.isnull().sum()


# In[41]:


#plot distribution for selected B variables with respect to target label
nrows = 8
ncols = 5
fig, axes = plt.subplots(figsize=(24,22)) 
b_columns=['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10',
       'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19',
       'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28',
       'B_29', 'B_31', 'B_32', 'B_33', 'B_36', 'B_37', 'B_39', 'B_40', 'B_41',
       'B_42']
for i, col in enumerate(b_columns):
    ax=fig.add_subplot(nrows, ncols, i+1)
    sns.kdeplot(x=train_data[col],hue=train_data['target'], multiple="stack",palette=["#FF3333" ,"#00CC00"],ax=ax)
    
fig.tight_layout()  
plt.show()


# In[42]:


del train_data[col]


# In[43]:


corr=train_balance_cols.corr()
plt.figure(figsize=(24,22))
ax = sns.heatmap(corr,cmap="YlGnBu", linewidths=.5, vmin=-1, vmax=1, center=0, annot=True, annot_kws={'fontsize':12,'fontweight':'bold'}, cbar=False,fmt='.2f' )
plt.yticks(rotation=0)
plt.show()


# In[44]:


del train_balance_cols


# ## Data analysis on the Spend attributes

# In[45]:


cols=[col for col in train_data.columns if (col.startswith(('S','T'))) & (col not in categorical_cols[:-1])]

train_spend_cols =train_data[cols]


# In[46]:


train_spend_cols.select_dtypes(exclude='object').describe().T.style.background_gradient(cmap='RdPu').set_properties(**{'font-family': 'Segoe UI'})


# In[47]:


# --- Plot Missing Values ---
mso.bar(train_spend_cols, fontsize=9, color=[purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0],
                               purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[1], purple_grad[1]], 
        figsize=(15, 8), sort='descending', labels=True)

# --- Title & Subtitle Settings ---
plt.suptitle('Missing Values in each Columns', fontweight='heavy', x=0.124, y=1.22, ha='left',fontsize='16', 
             fontfamily='sans-serif', color=black_grad[0])
plt.title('Almost all columns have no missing value except \n\nThe total of missing values in each column is less than 25%, which means that imputation can still be done to fill in the missing values in the\nthe columns.', 
          fontsize='8', fontfamily='sans-serif', loc='left', color=black_grad[1], pad=5)
plt.grid(axis='both', alpha=0);

# --- Total Missing Values in each Columns ---
print('\033[36m*' * 43)
print('\033[1m'+'.: Total Missing Values in each Columns :.'+'\033[0m')
print('\033[36m*' * 43+'\033[0m')
train_spend_cols.isnull().sum()


# In[48]:


#plot distribution for selected B variables with respect to target label
nrows = 5
ncols = 5
fig, axes = plt.subplots(figsize=(24,22)) 
s_columns=['S_2', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13',
       'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24',
       'S_25', 'S_26', 'S_27']
for i, col in enumerate(s_columns):
    ax=fig.add_subplot(nrows, ncols, i+1)
    sns.kdeplot(x=train_data[col],hue=train_data['target'], multiple="stack",palette=["#FF3333" ,"#00CC00"],ax=ax)
    
fig.tight_layout()  
plt.show()


# In[49]:


del train_data[col]


# In[50]:


corr=train_spend_cols.corr()
plt.figure(figsize=(24,22))
ax = sns.heatmap(corr,cmap="YlGnBu", linewidths=.5, vmin=-1, vmax=1, center=0, annot=True, annot_kws={'fontsize':12,'fontweight':'bold'}, cbar=False,fmt='.2f' )
plt.yticks(rotation=0)
plt.show()


# In[51]:


del train_spend_cols


# ## Data analysis on the Deliquency attributes

# In[52]:


cols=[col for col in train_data.columns if (col.startswith(('D','T'))) & (col not in categorical_cols[:-1])]

train_deliquency_cols =train_data[cols]


# In[53]:


train_deliquency_cols.select_dtypes(exclude='object').describe().T.style.background_gradient(cmap='RdPu').set_properties(**{'font-family': 'Segoe UI'})


# In[54]:


# --- Plot Missing Values ---
mso.bar(train_deliquency_cols, fontsize=9, color=[purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0],
                               purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[0], purple_grad[1], purple_grad[1]], 
        figsize=(24, 22), sort='descending', labels=True)

# --- Title & Subtitle Settings ---
plt.suptitle('Missing Values in each Columns', fontweight='heavy', x=0.124, y=1.22, ha='left',fontsize='16', 
             fontfamily='sans-serif', color=black_grad[0])
plt.title('Almost all columns have no missing value except \n\nThe total of missing values in each column is less than 25%, which means that imputation can still be done to fill in the missing values in the\nthe columns.', 
          fontsize='8', fontfamily='sans-serif', loc='left', color=black_grad[1], pad=5)
plt.grid(axis='both', alpha=0);

# --- Total Missing Values in each Columns ---
print('\033[36m*' * 43)
print('\033[1m'+'.: Total Missing Values in each Columns :.'+'\033[0m')
print('\033[36m*' * 43+'\033[0m')
train_deliquency_cols.isnull().sum()


# In[55]:


#plot distribution for selected B variables with respect to target label
nrows = 18
ncols = 5
fig, axes = plt.subplots(figsize=(24,22)) 
d_columns=['D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
       'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58',
       'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72',
       'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81',
       'D_82', 'D_83', 'D_84', 'D_86', 'D_88', 'D_89', 'D_91', 'D_92',
       'D_93', 'D_94', 'D_96', 'D_102', 'D_103', 'D_104', 'D_105',
       'D_107', 'D_108', 'D_109', 'D_110', 'D_111', 'D_112', 'D_113', 'D_115',
       'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127',
       'D_128', 'D_129', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135',
       'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143',
       'D_144', 'D_145']
for i, col in enumerate(d_columns):
    ax=fig.add_subplot(nrows, ncols, i+1)
    sns.kdeplot(x=train_data[col],hue=train_data['target'], multiple="stack",palette=["#FF3333" ,"#00CC00"],ax=ax)
    
fig.tight_layout()  
plt.show()


# In[56]:


del train_data[col]


# In[57]:


del train_deliquency_cols


# ## Work In progress

# In[ ]:




