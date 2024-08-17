#!/usr/bin/env python
# coding: utf-8

# 
# <p style='text-align: left;'><span style="color: #0D0D0D; font-family: Segoe UI; font-size: 2.4em; font-weight: 300;">THE TITANIC WRECKAGE MAY COMPLETELY VANISH BY 2030
# BUT, LET US ALL KEEP THE LEGACY ALIVE</span></p>
# 
# 
# ![](https://i.pinimg.com/originals/a4/08/ef/a408efc7cea165569dbd57826278fc8d.jpg)
# 

# <span style="color: #221E1F; font-family: Trebuchet MS; font-size: 2.2em;">Contents</span>
# 
# 
# * [1. Introduction](#introduction)
# * [2. Environment Preparation](#envprep)
# * [3. A bit of Exploratory Data Analysis](#eda)
#     - [3.1 Analysis of Age](#aoa)
#     - [3.2 Exploration of Fare ](#fare)
#     - [3.4 Analysis of Pclass & Sex](#pclasssex)
#     - [3.5 Analysis of SibSp & Panrh](#sibpar)
#     - [3.6 Few more plots of Feature densities](#density)
#     - [3.7 Exploration of Feature Relationships](#rel)
# * [4. Feature Engineering & EDA Extended](#fe)
#     - [4.1 Encoding of Sex](#ensex)
#     - [4.2 Let's Analyze & Feature Engineer Name](#name)
#         - [4.2.1 Derive & Plot the Title Feature](#name)
#         - [4.2.2 Extract Name Length Feature from Name](#length)
#     - [4.3 One-hot Encode Embarked & Label Encode Title](#oneone)
#     - [4.4 Derive Family Size Feature](#feparsib)
#     - [4.5 Label Encoding of Family Size](#encfam)
#     - [4.6 Extract Family_Name Feature from Name](#famname)
#     - [4.7 Derive Friends & Family Survival Rate Feature](#famsurv)
# * [5. Data Cleaning & More Feature Engineering](#morefe)
#     - [5.1 Cleaning & Encoding of the Cabin](#morefe)
#     - [5.2 Cleaning the Ticket](#cltik)
#     - [5.3 Derive the Ticket Frequency](#ticfea)
#     - [5.4 One-hot Encoding Ticket](#onetick)
#     - [5.5 Fare into Categorical Bins ](#farecat)
#     - [5.6 Additional Derived Features from Feature Relationships](#der)
#     - [5.7 Imputation of Missing Age Values](#mis)
#     - [5.8 Obtain Features for Children & Seniors](#chilsen)
#     - [5.9 Exploration of Derived Features](#eder)
#     - [5.10 Pickle & Store Dataframes for Later](#pik)
#     - [5.11 Standard Scaling Data](#sca)
#     - [5.12 Select Features for Training](#sel)
# * [6. Checking Feature Importance by Correlation Analysis](#corr)
# * [7. Preparation of Train & Test Data](#trte)
# * [8. Model Development](#mdev)
#     - [8.1 Model Architecture Definition](#mbul)
#     - [8.2 Setting Cross-validation Scheme & Model Training](#cross)
#     - [8.3 Plot the Model Metric Trends](#mloss)
# * [9. Submission File Generation	](#subfil)

# <a id="introduction"></a>
# # 1. Introduction
# 
# **The reason for using a Multi-layer Perceptron or a Feed-forward Neural network was to exploit and showcase its amazing potential and capabilities if it is tuned just right with sufficient regularization.**
# 
# Througout the notebook, I have detailed out the Exploratory Data Analysis and the Feature Engineering that was carried out to best bring out the capabilities of MLPs.
# 
# Tuning a Neural Network with just the right amount of regularization and giving the right features to obtain an optimal low variance and bias to maximize the accuracy is a very complex and time-consuming task especially given the limited samples and nature of the dataset.
# 
# However, I admit that investing time tuning an XGboost, KKN or other ensemble models is much more worthwhile in time-bound situations and Neural Networks may not be the best choice but, proper application of techniques with good feature engineering and driving the Model to a good local minimum with a lot of regularization yielded a great performance.
# 
# **I tried Neural Network Ensembling, Mean Encoding, Grid Search, Bayesian Optimization and several other approaches to improve the model and find the best Hyperparameters at times but, the lack of options to find a representative cross-validation scheme which mimics the Public leaderboard split to represent some sort of dependancy in the model metrics was a difficult task.**
# 
# **Finally, I ended up babysitting the model with manual tuning techniques to intuitively find the best hyperparameters and a single model tuned and regularized well was capable of representing the non-linear decision boundaries of the data and give the best performance.**
# 
# <span style="color: #056e94; font-family: Trebuchet MS; font-size: 1.3em;">I will keep updating this notebook with further details of the implementation. Thanks to the Kaggle commmunity and staff for all the support.</span>
# 
# <span style="color: #056e94; font-family: Trebuchet MS; font-size: 1.3em;">Please upvote and comment if you like my work :)</span>
# 
# 
# 
# [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg?style=flat-square&logo=kaggle)](https://www.kaggle.com/sreevishnudamodaran)
# 
# 
# 
# 
# ![TPU!](https://img.shields.io/badge/Accelerator-GPU-orange?style=flat-square&logo=kaggle)
# 
# ![Upvote!](https://img.shields.io/badge/Upvote-If%20you%20like%20my%20work-07b3c8?style=for-the-badge&logo=kaggle)
# 
# 
# 
# #### Details & Description of Features:
# 
# * PassengerID
# * Survived - (0 = No, 1 = Yes)
# * Pclass - Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
# * Name
# * Sex
# * Age
# * SibSp - Number of Siblings/Spouses Aboard
# * Parch - Number of Parents/Children Aboard
# * Ticket - Ticket Number
# * Fare - Passenger Fare in British pound
# * Cabin - Cabin Number
# * Embarked - Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
# <br />
# <br />
# 
# #### Additional Notes:
# * Pclassis a proxy for socio-economic status (SES) : 1st ~ Upper, 2nd ~ Middle, 3rd ~ Lower
# * Fare is in Pre-1970 British Pounds : Conversion Factors:  £1 = 12s (shillings) = 240d (pence) and 1s = 20d 
# * Sibling : Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# * Spouse : Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# * Parent : Mother or Father of Passenger Aboard Titanic
# * Child : Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# 

# <a id="envprep"></a>
# # 2. Environment Preparation
# ### Updating Seaborn to the Version 0.11.0
# 

# In[1]:


get_ipython().system('pip install seaborn==0.11.0')


# ### Library Imports

# In[27]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.ticker as ticker
print(sns.__version__)

from matplotlib import rcParams
sns.set(rc={"font.size":18,"axes.titlesize":30,"axes.labelsize":18,
            "axes.titlepad":22, "axes.labelpad":18, "legend.fontsize":15,
            "legend.title_fontsize":15, "figure.titlesize":35})

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs.layout import Scene


# In[4]:


#Load Train and Test Data
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head(5)


# <a id="eda"></a>
# # 3. A bit of Exploratory Data Analysis
# ### Converting Appropriate Columns to Categorical Type

# In[5]:


for col in ['Sex', 'Cabin', 'Ticket', 'Embarked']:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
df_train.info(verbose=True)


# <a id="aoa"></a>
# ## 3.1 Analysis of Age
# 
# We see that survival rate is is different for passengers below age 9 and above age 74. We will use these limits later to categorize and derive new features.

# In[6]:


fig = plt.figure(figsize=(22,8))
kde = sns.kdeplot(x="Age", data=df_train, cut=0, hue="Survived",
                  fill=True, legend=True, palette="plasma_r")

kde.xaxis.set_major_locator(ticker.MultipleLocator(1))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("AGE BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=39);


# In[7]:


fig = plt.figure(figsize=(22,8))
hist = sns.histplot(df_train['Age'], color="springgreen", kde=True, bins=50, label='Train')
hist = sns.histplot(df_test['Age'], color="gold", kde=True, bins=50, label='Test')

title = fig.suptitle("DISTRIBUTION OF AGE IN TRAIN & TEST", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=39)

hist.xaxis.set_major_locator(ticker.MultipleLocator(1))
hist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()


# <a id="fare"></a>
# ## 3.2 Exploration of Fare
# 

# In[8]:


fig = plt.figure(figsize=(22,8))
kde = sns.kdeplot(x="Fare", data=df_train, cut=0, hue="Survived", fill=True, legend=True, palette="mako_r")

kde.xaxis.set_major_locator(ticker.MultipleLocator(10))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("FARE BY SURVIVED", x=0.125, y=1.01
            , ha='left',fontweight=100, fontfamily='Lato', size=39);


# 
# Behaviour of fare seems different before and after the value 39

# In[9]:


fig = plt.figure(figsize=(20,8))
kde = sns.kdeplot(x="Fare", data=df_train, cut=0, clip=[0,180], hue="Survived", fill=True, legend=True, palette="mako_r")

kde.xaxis.set_major_locator(ticker.MultipleLocator(4))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("FARE BY SURVIVED - CLIPPED TO REMOVE OUTLIERS", x=0.12, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37);


# In[10]:


fig = plt.figure(figsize=(20,8))
dist = sns.histplot(df_train[(df_train.Fare > 0) & (df_train.Fare <=180)]['Fare'],
                    color="gold", kde=True, bins=50, label='Train')
dist = sns.histplot(df_test[(df_test.Fare > 0) & (df_test.Fare <=180)]['Fare'],
                    color="crimson", kde=True, bins=50, label='Test')

title = fig.suptitle("DISTRIBUTION OF FARE IN TRAIN & TEST", x=0.12, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37)

dist.xaxis.set_major_locator(ticker.MultipleLocator(4))
dist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()


# <a id="pclasssex"></a>
# ## 3.3 Analysis of Pclass & Sex
# 
# We see differences in survival rate for Class 1 and Class 3 passengers.
# 
# We also notice that the Men seems to have a low survival rate.

# In[11]:


fig, ax = plt.subplots(ncols=2, figsize=(20,8))
for i, col in enumerate(['Pclass', 'Sex',]):
    sns.histplot(x=col, data=df_train, hue="Survived", fill=True, ax=ax[i], palette="afmhot_r", kde=True)
    #ax[i].title.set_text(col+" by Survived")
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37)


# In[12]:


fig, ax = plt.subplots(ncols=2, figsize=(20,8))
for i, col in enumerate(['Pclass', 'Sex',]):
    n_bins =df_train[col].unique().shape[0]
    sns.histplot(df_train[col], color="gold", kde=True, bins=n_bins,
                 label='Train', ax=ax[i], legend=True)
    sns.histplot(df_test[col], color="crimson", kde=True, bins=n_bins,
                 label='Test', ax=ax[i], legend=True)
    #ax[i].title.set_text(col+" by Survived")
    ax[i].set_title("DISTRIBUTION OF {} IN TRAIN & TEST".format(col.upper())
                    , x=0.0, y=1.01, ha='left', fontweight=100, fontfamily='Lato', size=25)
    ax[i].legend(loc='upper left')


# <a id="sibpar"></a>
# ## 3.4 Analysis of SibSp & Panch
# 
# Passengers without Siblings/Spouse and Parents/Children accompanying them seems to have a low survival rate.

# In[13]:


fig, ax = plt.subplots(ncols=2, figsize=(20,8))
for i, col in enumerate(['SibSp', 'Parch']):
    sns.histplot(x=col, data=df_train, hue="Survived", fill=True, ax=ax[i], palette="hsv_r", kde=True)
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.03, ha='left',
             fontweight=100, fontfamily='Lato', size=37)


# In[14]:


fig, ax = plt.subplots(ncols=2, figsize=(20,8))
for i, col in enumerate(['SibSp', 'Parch']):
    n_bins = df_train[col].unique().shape[0]
    hist1 = sns.histplot(df_train[col], color="orangered", kde=True, bins=n_bins,
                 label='Train', ax=ax[i], legend=True)
    hist2 = sns.histplot(df_test[col], color="gold", kde=True, bins=n_bins,
                 label='Test', ax=ax[i], legend=True)
    #ax[i].title.set_text(col+" by Survived")
    ax[i].set_title("DISTRIBUTION OF {} IN TRAIN & TEST".format(col.upper())
                    , x=0.0, y=1.01, ha='left', fontweight=100, fontfamily='Lato', size=25)
    ax[i].legend(loc='upper left')
    
    hist1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    hist2.xaxis.set_major_formatter(ticker.ScalarFormatter())


# <a id="density"></a>
# ## 3.5 Few more plots of Feature densities
# 

# In[15]:


fig, ax = plt.subplots(ncols=3, figsize=(20,8))
for i, col in enumerate(['Pclass', 'Parch', 'SibSp']):
    sns.violinplot(x="Survived", y=col, data=df_train, ax=ax[i], palette="Spectral_r", orient="v")
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.03, ha='left',
         fontweight=200, fontfamily='Lato', size=33)


# In[16]:


fig, ax = plt.subplots(ncols=2, figsize=(20,8))
for i, col in enumerate(['Sex', 'Embarked']):
    sns.violinplot(y="Survived", x=col, data=df_train, ax=ax[i], palette="bone_r", orient="v")
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.03, ha='left',
         fontweight=200, fontfamily='Lato', size=37)


# <a id="rel"></a>
# ## 3.6 Exploration of Feature Relationships
# 
# The relationships of features and their behaviour can be used later to derive new features.

# In[29]:


fig = px.scatter(df_train,
                 x="Fare", y="Age", color="Survived", size="Pclass",
                 log_x=True, size_max=8, color_continuous_scale=['crimson', 'cyan'],
                 marginal_x='violin', marginal_y='histogram',
                 template='plotly', title='<span style="font-weight: 100;">AGE VS FARE BY SURVIVED</span>')

fig.update_layout(
    title_x=0.08,
    title_font_size=32,
    title_font_color='black'
)

fig.show()


# In[30]:


trace1=go.Scatter3d(x=df_train['Age'], y=df_train['Pclass'], z=df_train['Fare'],
                   mode='markers', marker=dict(size=3,colorscale='rdylbu',color=df_train['Survived']),
                                             opacity=0.8, 
                    scene = 'scene')
layout = go.Layout(
    scene = Scene(
        xaxis=go.layout.scene.XAxis(title='Age'),
        yaxis=go.layout.scene.YAxis(title='Pclass'),
        zaxis=go.layout.scene.ZAxis(title='Fare')
    ),
    title='<span style="font-weight: 100;">AGE VS FARE VS PCLASS BY SURVIVED</span>',
    title_x=0.08,
    title_font_size=32,
    title_font_color='black',
    font=dict(
        size=10,
        color="RebeccaPurple"
    )  
)

fig = Figure(data=trace1, layout=layout)
fig.show()


# In[31]:


fig = px.histogram(df_train, x="Age", y="Fare", color="Survived",
                   facet_row="Pclass", facet_col="Sex")

fig.update_layout(
    title='<span style="font-weight: 100;">AGE VS FARE VS PCLASS BY SURVIVED</span>',
    title_x=0.08,
    height=500,
    title_font_size=32,
    title_font_color='black'
)

fig.show()


# In[32]:


fig = px.scatter(df_train, x="Age", y="Fare", color="Survived",
                   facet_row="Pclass", facet_col="Sex", color_continuous_scale='PiYg')

fig.update_layout(
    title='<span style="font-weight: 100;">AGE VS FARE VS PCLASS BY SURVIVED</span>',
    title_x=0.08,
    height=500,
    title_font_size=32,
    title_font_color='black'
)

fig.show()


# In[33]:


fig = px.histogram(df_train, x="SibSp", y="Parch", color="Survived",
                   facet_row="Pclass", facet_col="Survived", color_discrete_sequence=['maroon', 'mediumaquamarine'])

fig.update_layout(
    title='<span style="font-weight: 100;">SIBSP VS PARCH VS PCLASS BY SURVIVED</span>',
    title_x=0.08,
    width=900,
    height=700,
    title_font_size=32,
    title_font_color='black'
)

fig.show()


# In[35]:


features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']

pair_plt = sns.pairplot(df_train[features], hue="Survived", palette="twilight_shifted",
                 diag_kind="kde", height=2.5)
tmp = pair_plt.fig.suptitle("FEATURES GROUPED BY SURVIVED", x=0.085, y=1.05, ha='left',
             fontweight=100, fontfamily='Lato', size=33)


# <a id="fe"></a>
# # 4. Feature Engineering & EDA Extended
# ### Merge Train & Test for Tranformations 

# In[36]:


full_df = pd.concat([df_train, df_test]).reset_index(drop=True)

train_shape = df_train.shape
test_shape = df_test.shape


# <a id="ensex"></a>
# ## 4.1 The Encoding of Sex

# In[37]:


# Label Encoding
full_df.loc[:, 'Sex'] = (full_df.loc[:, 'Sex'] == 'female').astype(int)


# <a id="name"></a>
# ## 4.2 Let's Analyze & Feature Engineer Name
# ### 4.2.1 Derive & Plot the Title Feature

# In[40]:


full_df['Title'] = full_df['Name']
full_df['Title'] = full_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

c1 = sns.catplot(x="Title", hue="Survived", kind="count", data=full_df[:train_shape[0]],
                 aspect = 3.5, legend=True, palette="YlGnBu")

title = c1.fig.suptitle("COUNT BY TITLE", x=0.04, y=1.12, ha='left',
             fontweight=100, fontfamily='Lato', size=42)

# Replacing rare titles 
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 
           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}
           
full_df.replace({'Title': mapping}, inplace=True)

c2 = sns.catplot(x="Title", hue="Survived", kind="count", data=full_df[:train_shape[0]],
                 aspect = 3.5, legend=True, palette="YlGnBu")
c2.fig.suptitle("COUNT BY TITLE AGGREGATED", x=0.04, y=1.12, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="length"></a>
# ### 4.2.2 Extract Name Length Feature from Name
# 
# The basic intuition behind this feature is that people with longer names tends to be of a higher class and thus would have likely survived. 

# In[41]:


full_df["Name_Length"] = full_df.Name.str.replace("[^a-zA-Z]", "").str.len()

fig, ax = plt.subplots(ncols=1, figsize=(20,8))
kde = sns.kdeplot(x="Name_Length", data=full_df[:train_shape[0]], cut=True,
                  hue="Survived", fill=True, ax=ax, palette="mako_r")

kde.xaxis.set_major_locator(ticker.MultipleLocator(1))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("NAME_LENGTH BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="oneone"></a>
# ## 4.3 One-hot Encode Embarked & Label Encode Title
# 
# Although Title is one-hot encodeded, it is also label encoded to a categorical feature to help derive other features later.

# In[42]:


full_df['Title_C'] = full_df['Title']

full_df = pd.get_dummies(full_df, columns=["Embarked","Title_C"],\
                         prefix=["Emb","Title"], drop_first=False)

title_dict = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Other': 4, 'Royal': 5, 'Master': 6}
full_df['Title'] = full_df['Title'].map(title_dict).astype('int')


# <a id="feparsib"></a>
# ## 4.4 Derive Family Size Feature

# In[43]:


# New feature : Family_size
full_df['Family_Size'] = full_df['Parch'] + full_df['SibSp'] + 1

full_df['Fsize_Cat'] = full_df['Family_Size'].map(lambda val: 'Alone' if val <= 1 else ('Small' if val < 5 else 'Big'))

fig, ax = plt.subplots(ncols=2, figsize=(22,8))
for i, col in enumerate(['Family_Size', 'Fsize_Cat']):
    sns.histplot(x=col, data=full_df[:train_shape[0]], hue="Survived", fill=True, ax=ax[i], palette="afmhot_r", kde=True)
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.03, ha='left',
         fontweight=200, fontfamily='Lato', size=39)
    if(col=='Family_Size'):
        ax[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[i].xaxis.set_major_formatter(ticker.ScalarFormatter())


# <a id="encfam"></a>
# ## 4.5 Label Encoding Family Size

# In[44]:


Fsize_dict = {'Alone':3, 'Small':2, 'Big':1}
full_df['Fsize_Cat'] = full_df['Fsize_Cat'].map(Fsize_dict).astype('int')


# <a id="famname"></a>
# ## 4.6 Extract Family_Name Feature from Name

# Using Regex to get the Surname or the Last Name from the Name Feature

# In[45]:


full_df['Family_Name'] = full_df['Name'].str.extract('([A-Za-z]+.[A-Za-z]+)\,', expand=True)


# <a id="famsurv"></a>
# ## 4.7 Derive Friends & Family Survival Rate Feature

# This seems to be one of the key features that improves scores after a lot of submissions engineering different features each time. 
# 
# If passengers with the same Last names are present, we group them and attach a calculated survival rate based on the train survival data. 
# 
# For no matching last names, the Ticket feature is used to group and calculate the survival rate in the same way.
# 
# From the data, we see Tickets are given to groups travelling together and they all have the same Ticket number.

# In[46]:


MEAN_SURVIVAL_RATE = round(np.mean(df_train['Survived']), 4)

full_df['Family_Friends_Surv_Rate'] = MEAN_SURVIVAL_RATE
full_df['Surv_Rate_Invalid'] = 1

for _, grp_df in full_df[['Survived', 'Family_Name', 'Fare', 'Ticket', 'PassengerId']].groupby(['Family_Name', 'Fare']):                       
    if (len(grp_df) > 1):
        if(grp_df['Survived'].isnull().sum() != len(grp_df)):
            for ind, row in grp_df.iterrows():
                full_df.loc[full_df['PassengerId'] == row['PassengerId'],
                            'Family_Friends_Surv_Rate'] = round(grp_df['Survived'].mean(), 4)
                full_df.loc[full_df['PassengerId'] == row['PassengerId'],
                            'Surv_Rate_Invalid'] = 0

for _, grp_df in full_df[['Survived', 'Family_Name', 'Fare', 'Ticket', 'PassengerId', 'Family_Friends_Surv_Rate']].groupby('Ticket'):
    if (len(grp_df) > 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Friends_Surv_Rate'] == 0.) | (row['Family_Friends_Surv_Rate'] == MEAN_SURVIVAL_RATE):
                if(grp_df['Survived'].isnull().sum() != len(grp_df)):
                    full_df.loc[full_df['PassengerId'] == row['PassengerId'],
                                'Family_Friends_Surv_Rate'] = round(grp_df['Survived'].mean(), 4)
                    full_df.loc[full_df['PassengerId'] == row['PassengerId'],
                                'Surv_Rate_Invalid'] = 0


# In[47]:


fig, ax = plt.subplots(figsize=(23,8))
sns.histplot(x='Family_Friends_Surv_Rate', data=full_df[:train_shape[0]], hue="Survived", fill=True, ax=ax, palette="inferno_r")
fig.suptitle("FAMILY_FRIENDS_SURV_RATE BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# In[48]:


fig, ax = plt.subplots(figsize=(23,8))
sns.barplot(y='Survived', x='Family_Friends_Surv_Rate', data=full_df[:train_shape[0]], ax=ax, palette="Set2_r")
fig.suptitle("FAMILY_FRIENDS_SURV_RATE BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="morefe"></a>
# # 5. Data Cleaning & More Feature Engineering
# ## 5.1 Cleaning & Encoding of the Cabin

# In[49]:


# Replace missing values with 'U' for Cabin
full_df['Cabin'] = full_df['Cabin'].astype('category')
full_df['Cabin'] = full_df['Cabin'].cat.add_categories('U')
full_df['Cabin_Clean'] = full_df['Cabin'].fillna('U')
full_df['Cabin_Clean'] = full_df['Cabin_Clean'].str.strip(' ').str[0]
# Label Encoding
cabin_dict = {'A':9, 'B':8, 'C':7, 'D':6, 'E':5, 'F':4, 'G':3, 'T':2, 'U':1}
full_df['Cabin_Clean'] = full_df['Cabin_Clean'].map(cabin_dict).astype('int')


# In[50]:


fig, ax = plt.subplots(ncols=1, figsize=(23,8))
sns.histplot(x="Cabin_Clean", data=full_df[:train_shape[0]], hue="Survived", fill=True, ax=ax, palette="nipy_spectral", kde=True)
fig.suptitle("CABIN_CLEAN BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="cltik"></a>
# ## 5.2 Cleaning the Ticket

# In[51]:


import re
def clean_ticket(each_ticket):
    prefix = re.sub(r'[^a-zA-Z]', '', each_ticket)
    if(prefix):
        return prefix
    else:
        return "NUM"

full_df["Tkt_Clean"] = full_df.Ticket.apply(clean_ticket)

fig, ax = plt.subplots(ncols=1, figsize=(23,8))
sns.countplot(x="Tkt_Clean", data=full_df[:train_shape[0]], hue="Survived", fill=True, ax=ax, palette="bwr_r")
fig.suptitle("TKT_CLEAN BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="ticfea"></a>
# ## 5.3 Derive the Ticket Frequency

# In[52]:


full_df['Ticket_Frequency'] = full_df.groupby('Ticket')['Ticket'].transform('count')
fig, ax = plt.subplots(ncols=1, figsize=(23,8))
sns.countplot(x="Ticket_Frequency", data=full_df[:train_shape[0]], hue="Survived", fill=True, ax=ax, palette="PiYG_r")

fig.suptitle("TICKET_FREQUENCY BY SURVIVED", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=42);


# <a id="onetick"></a>
# ## 5.4 One-hot Encoding Ticket

# In[53]:


full_df = pd.get_dummies(full_df, columns=["Tkt_Clean"],\
                          prefix=["Tkt"], drop_first=True)


# <a id="farecat"></a>
# ## 5.5 Fare into Categorical Bins
# 
# Kernal density estimation plot of Fare gave us some insights on its distribution and impact on survival. We will use those to add a derived categorical feature from Fare.

# In[54]:


def fare_cat(fare):
    if fare <= 7.0:
        return 1
    elif fare <= 39 and fare > 7.0:
        return 2
    else:
        return 3

full_df.loc[:, 'Fare_Cat'] = full_df['Fare'].apply(fare_cat).astype('int')


# <a id="der"></a>
# ## 5.6 Additional Derived Features from Feature Relationships

# In[55]:


full_df.loc[:, 'Fare_Family_Size'] = full_df['Fare']/full_df['Family_Size']

full_df.loc[:, 'Fare_Cat_Pclass'] = full_df['Fare_Cat']*full_df['Pclass']
full_df.loc[:, 'Fare_Cat_Title'] = full_df['Fare_Cat']*full_df['Title']

full_df.loc[:, 'Fsize_Cat_Title'] = full_df['Fsize_Cat']*full_df['Title']
full_df.loc[:, 'Fsize_Cat_Fare_Cat'] = full_df['Fare_Cat']/full_df['Fsize_Cat'].astype('int')

full_df.loc[:, 'Pclass_Title'] = full_df['Pclass']*full_df['Title']
full_df.loc[:, 'Fsize_Cat_Pclass'] = full_df['Fsize_Cat']*full_df['Pclass']


# ### Remove Constant Columns

# In[56]:


colsToRemove = []
cols = ['Tkt_AQ', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
         'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE', 'Tkt_LP',
         'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
         'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
         'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ',
         'Tkt_SP', 'Tkt_STONO', 'Tkt_STONOQ', 'Tkt_SWPP', 'Tkt_WC', 
         'Tkt_WEP', 'Fare_Cat', 'Fare_Family_Size', 'Fare_Cat_Pclass',
         'Fare_Cat_Title', 'Fsize_Cat_Title', 'Fsize_Cat_Fare_Cat', 
         'Pclass_Title', 'Fsize_Cat_Pclass']

for col in cols:
    if full_df[col][:train_shape[0]].std() == 0: 
        colsToRemove.append(col)

# remove constant columns in the training set
full_df.drop(colsToRemove, axis=1, inplace=True)
print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)


# <a id="mis"></a>
# ## 5.7 Imputation of Missing Age Values

# In[57]:


# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import BayesianRidge
# imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, sample_posterior=False, 
#                                  max_iter=4000, tol=0.001, verbose=1,
#                                  n_nearest_features=4, initial_strategy='median')
from sklearn.impute import KNNImputer
imp_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title',
                 'Name_Length', 'Emb_C', 'Emb_Q', 'Emb_S','Family_Size',
                 'Fsize_Cat', 'Family_Friends_Surv_Rate', 'Surv_Rate_Invalid',
                 'Cabin_Clean','Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
                 'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE',
                 'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
                 'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
                 'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ',
                 'Tkt_SP', 'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 
                 'Tkt_WEP', 'Fare_Cat', 'Fare_Family_Size', 'Fare_Cat_Pclass',
                 'Fare_Cat_Title', 'Fsize_Cat_Title', 'Fsize_Cat_Fare_Cat', 
                 'Pclass_Title', 'Fsize_Cat_Pclass']

imputer = KNNImputer(n_neighbors=10, missing_values=np.nan)
# full_df[imp_features] = pd.DataFrame(imputer.fit_transform(full_df[imp_features]), index=full_df.index, columns = imp_features)
imputer.fit(full_df[imp_features])


# In[58]:


full_df.loc[:, imp_features] = pd.DataFrame(imputer.transform(full_df[imp_features]), index=full_df.index, columns = imp_features)


# In[59]:


plt.figure(figsize=(22,8))
kde = sns.kdeplot(x="Age", data=df_train, cut=0, hue="Survived", fill=True, legend=True, palette="terrain_r")
#kde.title.set_text("Age by Survived Before Imputation")
title = kde.set_title("AGE BY SURVIVED BEFORE IMPUTATION", x=0.0, y=1.03, ha='left',
             fontweight=100, fontfamily='Lato',
             size=41)

kde.xaxis.set_major_locator(ticker.MultipleLocator(1))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())


# In[60]:


plt.figure(figsize=(23,8))
kde = sns.kdeplot(x="Age", data=full_df[:train_shape[0]], cut=0, hue="Survived", fill=True, legend=True, palette="terrain_r")

title = kde.set_title("AGE BY SURVIVED AFTER IMPUTATION", x=0.0, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=41)

kde.xaxis.set_major_locator(ticker.MultipleLocator(1))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())


# ### Comparing Before and After Imputed Dataframes

# In[61]:


df_train[df_train.Age.isnull()].head(5)


# In[62]:


tmp = full_df[:train_shape[0]]
age_nan_indices = df_train[df_train.Age.isnull()].index.tolist()
tmp.iloc[age_nan_indices, :].head(5)


# <a id="chilsen"></a>
# ## 5.8 Obtain Features for Children & Seniors

# In[63]:


full_df['Child'] = full_df['Age'].map(lambda val:1 if val<18 else 0)
full_df['Senior'] = full_df['Age'].map(lambda val:1 if val>70 else 0)

fig, ax = plt.subplots(ncols=2, figsize=(23,8))

for i, col in enumerate(['Child', 'Senior']):
    sns.histplot(x=col, hue='Survived', data=full_df[:train_shape[0]],
                 ax=ax[i], fill=True, palette="Paired_r", kde=True)
    ax[i].set_title(col.upper()+" BY SURVIVED", x=0.0, y=1.01, ha='left',
         fontweight=200, fontfamily='Lato', size=39)


# ### Split Data back to Train and Test

# In[64]:


df_train_final = full_df[:train_shape[0]]
df_test_final = full_df[train_shape[0]:]


# <a id="eder"></a>
# ## 5.9 Exploration of Derived Features

# In[65]:


viz_features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
                'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin',
                'Title', 'Name_Length', 'Family_Friends_Surv_Rate', 
                'Ticket_Frequency']

train_viz = df_train_final[viz_features]


# In[66]:


fig = px.scatter(train_viz,
                 x="Name_Length", y="Family_Friends_Surv_Rate", color="Survived", size="Pclass",
                 size_max=8, color_continuous_scale=['maroon', 'mediumaquamarine'],
                 marginal_x='histogram', marginal_y='histogram',
                 template='plotly', title='<span style="font-weight: 100;">NAME_LENGTH VS SURV_RATE BY SURVIVED</span>')

fig.update_layout(
    title_x=0.08,
    title_font_size=30,
    title_font_color='black'
)

fig.show()


# In[67]:


fig = px.scatter(train_viz,
                 x="Name_Length", y="Ticket_Frequency", color="Survived", size="Pclass",
                 size_max=8, color_continuous_scale=['goldenrod', 'cadetblue'],
                 marginal_x='histogram', marginal_y='histogram',
                 template='plotly', title='<span style="font-weight: 100;">NAME_LENGTH VS TKT_FREQ BY SURVIVED</span>')

fig.update_layout(
    title_x=0.08,
    title_font_size=30,
    title_font_color='black'
)

fig.show()


# <a id="pik"></a>
# ## 5.10 Pickle & Store Dataframes for Later

# In[68]:


full_df.to_pickle("full_df")
df_train_final.to_pickle("df_train_final")
df_test_final.to_pickle("df_test_final")


# In[ ]:


# df_train_final = pd.read_pickle("../input/titanic-test/df_train_final")
# df_test_final = pd.read_pickle("../input/titanic-test/df_test_final")


# <a id="sca"></a>
# ## 5.11 Standard Scaling Data

# In[69]:


from sklearn.preprocessing import StandardScaler
scaler_cols = ['Age', 'Fare', 'Name_Length', 'Family_Size', 'Name_Length',
               'Ticket_Frequency', 'Fare_Family_Size', 'Fare_Cat_Pclass']
std = StandardScaler()
std.fit(df_train_final[scaler_cols])
df_train_final.loc[:, scaler_cols] = pd.DataFrame(std.transform(df_train_final[scaler_cols]), index=df_train_final.index, columns = scaler_cols)
df_test_final.loc[:, scaler_cols] = pd.DataFrame(std.transform(df_test_final[scaler_cols]), index=df_test_final.index, columns = scaler_cols)


# In[70]:


df_train_final.describe()


# <a id="sel"></a>
# ## 5.12 Select Features for Training

# In[71]:


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Name_Length', 'Emb_C',
       'Emb_Q', 'Emb_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Other', 'Title_Royal', 'Family_Size', 'Fsize_Cat',
       'Family_Friends_Surv_Rate', 'Surv_Rate_Invalid', 'Cabin_Clean',
       'Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
       'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE', 
       'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
       'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
       'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ', 'Tkt_SP',
       'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 'Tkt_WEP', 'Fare_Cat',
       'Fare_Family_Size', 'Fare_Cat_Pclass', 'Fare_Cat_Title',
       'Fsize_Cat_Title', 'Fsize_Cat_Fare_Cat', 'Pclass_Title',
       'Fsize_Cat_Pclass', 'Child', 'Senior']
features_train = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Name_Length', 'Emb_C',
       'Emb_Q', 'Emb_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Other', 'Title_Royal', 'Family_Size', 'Fsize_Cat',
       'Family_Friends_Surv_Rate', 'Surv_Rate_Invalid', 'Cabin_Clean',
       'Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
       'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE',
       'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
       'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
       'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ', 'Tkt_SP',
       'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 'Tkt_WEP', 'Fare_Cat',
       'Fare_Family_Size', 'Fare_Cat_Pclass', 'Fare_Cat_Title',
       'Fsize_Cat_Title', 'Fsize_Cat_Fare_Cat', 'Pclass_Title',
       'Fsize_Cat_Pclass', 'Child', 'Senior']

df_train_final = df_train_final[features_train]
df_test_final = df_test_final[features]


# <a id="corr"></a>
# # 6. Checking Feature Importance by Correlation Analysis

# In[72]:


corr_mat = df_train_final.astype(float).corr()
corr_mat_fil = corr_mat.loc[:, 'Survived'].sort_values(ascending=False)
corr_mat_fil = pd.DataFrame(data=corr_mat_fil[1:])


# In[73]:


plt.figure(figsize=(15,14))
bar = sns.barplot(x=corr_mat_fil.Survived, y=corr_mat_fil.index, data=corr_mat_fil, palette="Spectral")
title = bar.set_title("FEATURE CORRELATION", x=0.0, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=30)


# <a id="trte"></a>
# # 7. Preparation of Train & Test Data

# In[74]:


features = df_test_final.columns.to_list()
X_train = df_train_final[features]
Y_train = df_train_final['Survived']
X_test = df_test_final


# <a id="mdev"></a>
# # 8. Model Development

# ### Import Libraries

# In[75]:


from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout, BatchNormalization,Concatenate, concatenate
from tensorflow.keras.optimizers import SGD, RMSprop, Adamax, Adagrad, Adam, Nadam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import *


# <a id="mbul"></a>
# ## 8.1 Model Architecture Definition

# In[76]:


metrics = ['accuracy', 
           Precision(),
           Recall()]


# In[77]:


def create_model():
    model = Sequential()
    model.add(Input(shape=X_train.shape[1], name='Input_'))
    model.add(Dense(8, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001)))
    model.add(Dense(16, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))

    model.summary()
    optimize = Adam(lr = 0.0001)
    model.compile(optimizer = optimize, 
                       loss = 'binary_crossentropy', 
                       metrics = metrics)
    return model


# <a id="cross"></a>
# ## 8.2 Setting Cross-validation Scheme & Model Training

# In[78]:


estimator = KerasClassifier(build_fn = create_model, epochs = 600, batch_size = 32, verbose = 1)
kfold = StratifiedKFold(n_splits = 3)
results = cross_val_score(estimator, X_train, Y_train, cv = kfold)


# ### Train the model on full data

# In[79]:


train_history = estimator.fit(X_train, Y_train, epochs = 600, batch_size = 32)


# In[80]:


print(train_history.history.keys())


# <a id="mloss"></a>
# ## 8.3 Plot the Model Metric Trends

# In[82]:


fig = plt.figure(figsize=(22,8))
hist = sns.lineplot(data=train_history.history['accuracy'], color="darkturquoise", label='Accuracy')
hist = sns.lineplot(data=train_history.history['loss'], color="chocolate", label='Loss')
hist = sns.lineplot(data=train_history.history['recall'], color="indianred", label='Recall')

title = fig.suptitle("ACCURACY VS LOSS VS RECALL CUREVES", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37)

hist.xaxis.set_major_locator(ticker.MultipleLocator(20))
hist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()


# <a id="subfil"></a>
# # 9. Submission File Generation

# In[83]:


y_preds = estimator.predict(X_test)
submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')
submission['Survived'] = y_preds.astype(int)
submission.to_csv('submission.csv')

