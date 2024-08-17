#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:lightseagreen'>|</span> Introduction</b>
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# In[1]:


import os
import warnings
from pathlib import Path
from IPython.display import clear_output
from IPython.display import display
from pandas.api.types import CategoricalDtype
from IPython.core.display import display, HTML, Javascript

# Basic libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pp
import seaborn as sns
import math
import string

# Clustering
from sklearn.cluster import KMeans

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

#Mutual Information
from sklearn.feature_selection import mutual_info_regression

# Cross Validation
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, learning_curve, train_test_split, GridSearchCV

# Encoders
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Algorithms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Optuna - Bayesian Optimization 
import optuna
from optuna.samplers import TPESampler

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go

# Metric
from sklearn.metrics import plot_confusion_matrix

# Permutation Importance
import eli5
from eli5.sklearn import PermutationImportance

warnings.filterwarnings('ignore')

df_test = pd.read_csv("../input/titanic/test.csv")
df_train = pd.read_csv("../input/titanic/train.csv")
df_data = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)
dfs = [df_train, df_test]
clear_output()
pp.ProfileReport(df_data)


# # <b>2 <span style='color:lightseagreen'>|</span> Exploratory Data Analysis</b>
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | General Analysis</b></p>
# </div>

# In[2]:


# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

# "coffee" pallette turqoise-gold.
f1 = "#a2885e"
f2 = "#e9cf87"
f3 = "#f1efd9"
f4 = "#8eb3aa"
f5 = "#235f83"
f6 = "#b4cde3"

def plot_box(fig, feature, r, c):
    fig.add_trace(go.Box(x=df_data[feature].astype(object), y=df_data.Survived, marker = dict(color= px.colors.sequential.Viridis_r[5])), row =r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False,row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
def plot_scatter(fig, feature, r, c):
    fig.add_trace(go.Scatter(x=df_data[feature], y=df_data.SalePrice, mode='markers', marker = dict(color=np.random.randn(10000), colorscale = px.colors.sequential.Viridis)), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
def plot_hist(fig, feature, r, c):
    fig.add_trace(go.Histogram(x=df_data[feature], name='Distribution', marker = dict(color = px.colors.sequential.Viridis_r[5])), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
# chart
fig = make_subplots(rows=2, cols=3, column_widths=[0.34, 0.33, 0.33], 
                    vertical_spacing=0.1, horizontal_spacing=0.1, subplot_titles=('Age Distribution','PClass Count','Fare Distribution','Parch Count',
                    'SibSp Count','Survival Count'))

plot_hist(fig, 'Age', 1,1)

pclass_df = pd.DataFrame(df_data['Pclass'].value_counts())
fig.add_trace(go.Bar(x=pclass_df.index, y=pclass_df['Pclass'], marker = dict(color = [primary_blue3, px.colors.sequential.Viridis_r[5], px.colors.sequential.Viridis_r[5]]), name='PClass Count'), row=1, col=2)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=2)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=2)

plot_hist(fig, 'Fare', 1,3)

parch_df = pd.DataFrame(df_data['Parch'].value_counts())
fig.add_trace(go.Bar(x=parch_df.index, y=parch_df['Parch'], marker = dict(color = [primary_blue3, px.colors.sequential.Viridis_r[5], px.colors.sequential.Viridis_r[5]]), name='Parch Count'), row=2, col=1)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=2, col=1)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=2, col=1)

plot_hist(fig, 'SibSp',2,2)

survived_df = pd.DataFrame(df_data['Survived'].value_counts())
fig.add_trace(go.Bar(x=survived_df.index, y=survived_df['Survived'], marker = dict(color = [primary_blue3, px.colors.sequential.Viridis_r[5], px.colors.sequential.Viridis_r[5]]), name='Parch Count'), row=2, col=3)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=2, col=3)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=2, col=3)

# General Styling
fig.update_layout(height=750, bargap=0.2,
                  margin=dict(b=50,r=50,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>General Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | Heatmap</b></p>
# </div>

# In[3]:


fig = px.imshow(df_data[df_data['Survived'].isnull() == False].corr(), color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect='auto')
fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False)
fig.update_yaxes(showgrid = True, gridcolor='gray',gridwidth=0.5, linecolor='gray',linewidth=2, zeroline = False)

# General Styling
fig.update_layout(height=500, bargap=0.2,
                  margin=dict(b=50,r=30,l=100, t=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Heatmap - Numerical Features</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
fig.show()


# # <b>2 <span style='color:lightseagreen'>|</span> Missing Values</b>
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Age</b></p>
# </div>
# 
# To address the problem of missing values for the **<span style='color:lightseagreen'>Age</span>** field we will proceed as follows. Since **<span style='color:lightseagreen'>PClass</span>** is the variable that is **<span style='color:lightseagreen'>most correlated</span>** with both Age and Survived, we will group passengers according to the class they belong to. What we will do is replace the missing values with the **<span style='color:lightseagreen'>median</span>** of each group. In fact, what is more, within each of the existing classes we will make a **<span style='color:lightseagreen'>gender distinction</span>**. We do this because, as we will see below, the median of Age varies according to whether the passenger is male or female.

# In[4]:


df_heatmap = pd.DataFrame(df_data.corr()['Age'].abs())
f,ax = plt.subplots(figsize=(10,1.5),facecolor='white')
sns.color_palette("rocket", as_cmap=True)          # Esta paleta es la que viene por defecto
sns.heatmap(df_heatmap.transpose(),annot = True,square=True, linewidths=1.5, cmap='rocket')


# In[5]:


mediana = df_data.groupby(['Sex', 'Pclass']).median()['Age']
for i in range(0,mediana.shape[0]):
    if i<3: 
        print('Edad mediana para mujeres de la clase {}: {}'.format(i+1,mediana[i]))
    else:
        print('Edad mediana para hombres de la clase {}: {}'.format(i+1-3,mediana[i]))
df_data['Age'] = df_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
print('Missing values for Age: {}'.format(df_data.Age.isnull().sum()))


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | Embarked</b></p>
# </div>
# 
# With respect to Embarked we will replace the missing data by the **<span style='color:lightseagreen'>mode</span>**, i.e. the most repeated value.

# In[6]:


df_data.Embarked.value_counts()


# In[7]:


moda = 'S'
df_data.Embarked = df_data.Embarked.replace(np.nan,moda)
pd.isnull(df_data).sum()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3 | Cabin</b></p>
# </div>
# 
# **<span style='color:lightseagreen'>Cabin</span>** feature is little bit tricky and it needs further exploration. The large portion of the Cabin feature is missing and the feature itself **<span style='color:lightseagreen'>can't be ignored completely because some the cabins might have higher survival rates</span>**. It turns out to be the first letter of the Cabin values are the decks in which the cabins are located. Those decks were mainly separated for one passenger class, but some of them were used by multiple passenger classes.
# ![alt text](https://vignette.wikia.nocookie.net/titanic/images/f/f9/Titanic_side_plan.png/revision/latest?cb=20180322183733)
# * On the Boat Deck there were **6** rooms labeled as **T, U, W, X, Y, Z** but only the **T** cabin is present in the dataset
# * **A**, **B** and **C** decks were only for 1st class passengers
# * **D** and **E** decks were for all classes
# * **F** and **G** decks were for both 2nd and 3rd class passengers
# * From going **A** to **G**, **<span style='color:lightseagreen'>distance to the staircase increases which might be a factor of survival</span>**

# In[8]:


# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
df_data['Deck'] = df_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_data_decks = df_data.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 
                                                                        'Fare', 'Embarked', 'Cabin']).rename(columns={'Name': 'Count'}).transpose()

def get_pclass_dist(df):
    
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]    
    
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count 
            except KeyError:
                deck_counts[deck][pclass] = 0
                
    df_decks = pd.DataFrame(deck_counts)    
    deck_percentages = {}

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]
        
    return deck_counts, deck_percentages

def display_pclass_dist(percentages):
    
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85
    
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
    plt.show()    

all_deck_count, all_deck_per = get_pclass_dist(df_data_decks)
display_pclass_dist(all_deck_per)


# 📌 **Interpret:** for this graph we'll be grouping by **<span style='color:lightseagreen'>Deck</span>** and **<span style='color:lightseagreen'>PClass</span>** atribute. We are able to appreciate that A,B and C are fully occupied by passengers of 1st class. Moreover, as there is just one person in deck T, which class 1 we are going to group it with deck A. Deck of type D is mainly occupied with 1st class passengers, concretely a 85%. The rest are from 2nd class. To conclude, the remaining decks have all passengers from each of the classes.

# In[9]:


# Passenger in the T deck is changed to A
idx = df_data[df_data['Deck'] == 'T'].index
df_data.loc[idx, 'Deck'] = 'A'


# In[10]:


df_all_decks_survived = df_data.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin']).rename(columns={'Name':'Count'}).transpose()

def get_survived_dist(df):
    
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]    

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
            
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]
        
    return surv_counts, surv_percentages

def display_surv_dist(percentages):
    
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85    

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.show()

all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)
display_surv_dist(all_surv_per)


# 📌 **Interpret:** for this graph we'll be grouping by **<span style='color:lightseagreen'>Deck</span>** and **<span style='color:lightseagreen'>Survived</span>** atribute, in order to see the survival rate that each Deck has. We are able to appreciate that as expected survival rates are different for every type of Deck. **<span style='color:lightseagreen'>B</span>**, **<span style='color:lightseagreen'>D</span>** and **<span style='color:lightseagreen'>E</span>** are the ones with highest. On the other hand, **<span style='color:lightseagreen'>A</span>** and **<span style='color:lightseagreen'>M</span>** are the ones with lowest. 
# 
# Due to what we have just seen before, we are going to label decks in the following way: 
# * A, B and C decks, as they all have 1st class passengers, are going to be labeled as ABC
# * **D** and **E** decks are labeled as **DE** because both of them have similar passenger class distribution and same survival rate
# * Following the previous criterion we labeled FG
# * M remains equal because it's quite different from the others and it's the one with lowest survival rate.

# In[11]:


df_data['Deck'] = df_data['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_data['Deck'] = df_data['Deck'].replace(['D', 'E'], 'DE')
df_data['Deck'] = df_data['Deck'].replace(['F', 'G'], 'FG')

df_data['Deck'].value_counts()


# In[12]:


df_data = df_data.drop('Cabin',axis=1)
df_data.isnull().sum()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.4 | Fare</b></p>
# </div>
# 
# We have one missing value for **<span style='color:lightseagreen'>Fare</span>**, belonging to one male of the testing dataset. We can assume that it is related to **<span style='color:lightseagreen'>FamilySize</span>** and **<span style='color:lightseagreen'>PClass</span>**. Median Fare value of a male with a third class ticket and no family is a logical choice to fill the missing value.

# In[13]:


df_data[df_data.Fare.isnull()]


# In[14]:


mediana = df_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df_data.Fare = df_data.Fare.fillna(mediana)


# # <b>3 <span style='color:lightseagreen'>|</span> Feature Engineering</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Family</b></p>
# </div>
# 
# We will start by creating fields related to the family unit. The first of these will come from the **<span style='color:lightseagreen'>SibSp and Parch</span>** fields, which we can remove later. This will reflect the **<span style='color:lightseagreen'>size of passengers' family</span>**. We will also enter a field to indicate whether the passenger is travelling **<span style='color:lightseagreen'>alone</span>** or not.

# In[15]:


df_data['FamilySize'] = df_data.Parch + df_data.SibSp + 1
df_data['IsAlone'] = 0
df_data.loc[df_data['FamilySize'] == 1, 'IsAlone'] = 1
df_data = df_data.drop(['Parch','SibSp'],axis = 1)


# In[16]:


fig, axes = plt.subplots(figsize=(22,8), nrows = 1, ncols = 2)
ax = sns.countplot(x = 'FamilySize', hue='Survived', data = df_data, palette = ['#334550','#6D83AA'], ax = axes[0])
ax.set_title('Survival Rate per Family Size')
ax = sns.countplot(x = 'FamilySize', data = df_data, palette = ['#334550','#334668','#394184','#496595','#6D83AA','#91A2BF','#C8D0DF'], ax = axes[1])
_ = ax.set_title('Family Size Countplot')


# Taking this into account, I have decided to group **<span style='color:lightseagreen'>Family Size</span>** into 4 different groups. They are the following: 
# 
# * Alone: for people travelling with no member of his/her family. 
# * Small: for people travelling with 3 members of family
# * Medium: travelling with 4 or 5 members of family
# * Large: travelling with 6+ members of family

# In[17]:


family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_data['FamilySizeGrouped'] = df_data['FamilySize'].map(family_map)
df_data.head(5)


# Let's plot again graphs for previous contents:

# In[18]:


fig, axes = plt.subplots(figsize=(22,8), nrows = 1, ncols = 2)
ax = sns.countplot(x = 'FamilySizeGrouped', hue='Survived', data = df_data, palette = ['#334550','#6D83AA'], ax = axes[0])
ax.set_title('Survival Rate per Family Group')
ax = sns.countplot(x = 'FamilySizeGrouped', data = df_data, palette = ['#334550','#394184','#6D83AA','#C8D0DF'], ax = axes[1])
_ = ax.set_title('Family Group Countplot')


# 📌 **Interpret:** On the left part, on the one hand we can observe that lonely people tend to die. On the other hand, **<span style='color:lightseagreen'>small families members are most likely to survive</span>**. Appreciating the countplot, we can observe that most people is travelling alone or in small families. Travelling as a member of a medium/large family is very unusual.

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.2 | Passenger's Name</b></p>
# </div>
# 
# Next we are going to add a column that will be, in part, related to the **<span style='color:lightseagreen'>Name</span>** field:

# In[19]:


df_data.head().style.set_properties(subset=['Name'], **{'background-color': '#F1C40F'})


# In[20]:


df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(df_data['Title'], df_data['Sex']).transpose()


# We can replace many titles with a more common name or classify them as Rare.

# In[21]:


df_data['Title'] = df_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_data['Title'] = df_data['Title'].replace('Mlle', 'Miss')
df_data['Title'] = df_data['Title'].replace('Ms', 'Miss')
df_data['Title'] = df_data['Title'].replace('Mme', 'Mrs')

df_data['Is_Married'] = 0
df_data['Is_Married'].loc[df_data['Title'] == 'Mme'] = 1

df_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().transpose()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.3 | Fare</b></p>
# </div>
# 
# In order to binning continuous features we are going to use **<span style='v:#F1C40F'>13 quantile base bins</span>**. Even though the bins are too much, they provide a decent amount of information gain, as it would be seen in next section. We'll create `df_data_no_quart` in order to have a DataFrame with the discrete values of Fare, to make it easier to plotting it later.

# In[22]:


df_data_no_quart = df_data.copy()
names = ['1', '2', '3', '4', '5', '6', '7','8','9','10','11','12','13']
df_data['Fare'] = pd.qcut(df_data['Fare'], 13, labels = names)
df_data.Fare = pd.to_numeric(df_data.Fare, errors = 'coerce')


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.4 | Age</b></p>
# </div>
# 
# Let's keep binning continuous features. For **<span style='color:lightseagreen'>Age</span>** we are going to use **<span style='color:lightseagreen'>10 quantile base bins</span>**. Even though the bins are too much, they provide a decent amount of information gain, as it would be seen in next section.

# In[23]:


names = ['1','2','3','4','5','6','7','8','9','10']
df_data['Age'] = pd.qcut(df_data['Age'], 10, labels = names)


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.5 | Frequency Encoding</b></p>
# </div>
# 
# As seen in first part of this Feature Engineering section, **<span style='color:lightseagreen'>FamilySize</span>** could have a **<span style='color:lightseagreen'>huge effect on survival prediction</span>**, as rates are quite different between each other.

# In[24]:


df_data['Ticket_Frequency'] = df_data.groupby('Ticket')['Ticket'].transform('count')
df_data.head().style.set_properties(subset=['Ticket_Frequency'], **{'background-color': '#F1C40F'})


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.6 | Surname</b></p>
# </div>
# 
# We are going to group **<span style='color:lightseagreen'>passengers in the same family</span>**. `extract_surname` function is used for extracting surnames of passengers from the Name feature. Family feature is created with the extracted surname.

# In[25]:


def extract_surname(data):    
    families = []
    for i in range(len(data)):        
        name = data.iloc[i]
        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
        families.append(family)
    return families


# In[26]:


df_data['Family'] = extract_surname(df_data['Name'])
df_train = df_data.loc[:890]
df_test = df_data.loc[891:]
dfs = [df_train, df_test]


# **<span style='color:lightseagreen'>Family_Survival_Rate</span>** is calculated from families in training set since there is no Survived feature in test set. A list of family names that are occuring in both training and test set (non_unique_families), is created. The survival rate is calculated for families with more than 1 members in that list, and stored in Family_Survival_Rate feature.
# 
# An extra binary feature **<span style='color:lightseagreen'>Family_Survival_Rate_NA</span>** is created for families that are unique to the test set. This feature is also necessary because there is no way to calculate those families' survival rate. This feature implies that family survival rate is not applicable to those passengers because there is no way to retrieve their survival rate.
# 
# Ticket_Survival_Rate and Ticket_Survival_Rate_NA features are also created with the same method. Ticket_Survival_Rate and Family_Survival_Rate are averaged and become Survival_Rate, and Ticket_Survival_Rate_NA and Family_Survival_Rate_NA are also averaged and become Survival_Rate_NA.

# In[27]:


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','FamilySize'].median()
df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
        
mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

for df in [df_train, df_test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2


# # <b>4 <span style='color:#F1C40F'>|</span> Data Visualization</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | Heat Map</b></p>
# </div>
# 
# 📌 **Interpret:** On the one hand, the variables with the highest correlations are: **<span style='color:lightseagreen'>PClass</span>** and **<span style='color:lightseagreen'>Fare</span>**. On the other hand, we can appreciate that both, Age and FamilySize are the ones with less correlation coefficient with respect to Survived.

# In[28]:


df_heatmap = pd.DataFrame(df_data.corr()['Survived'].abs())
f,ax = plt.subplots(figsize=(16,1.5),facecolor='white')
sns.color_palette("rocket", as_cmap=True)          # Esta paleta es la que viene por defecto
sns.heatmap(df_heatmap.transpose(),annot = True,square=True, linewidths=1.5, cmap='rocket')


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Group of Age Visualization</b></p>
# </div>
# 
# 📌 **Interpret:** We have created several **<span style='color:lightseagreen'>age groups</span>**, in order to make it easier to make the graphs. On the left size, we have the average fare per each group of age. As observed, **<span style='color:lightseagreen'>older group (48-80)</span>** is the group which has payed more. At the bottom part of the bar graph we can find groups to which teenagers and young adults belong. On the right, I have made a pie plot in order to find which percentage of people from each group survive, and people from which group is more likely to survive. We can see that **<span style='color:lightseagreen'>kids (0.169-16)</span>** are the ones with more survival rate (14.6%). Moreover, we also observe that half of the kids survived. Concretely, a 58.8% of kids survived. On the other hand, older people and youngsters between 21-22 years old bear the brunt as they are the least likely to survive. Indeed, from the whole group of old people just 34% survived aproximately.

# In[29]:


#names = ['0-8', '9-15', '16-18', '19-25', '26-40', '41-60', '61-100']
oneHot_train_graph = df_data_no_quart.copy()
names = ['0.169-16','16-21','21-22','22-25','25-26','26-29.5','29.5-34','34-40','40-48','48-80']
oneHot_train_graph['Age'] = pd.qcut(oneHot_train_graph['Age'], 10, labels = names)
df_bar = oneHot_train_graph.groupby('Age').agg({'Fare':'mean'}).reset_index().sort_values(by='Fare',ascending=False).set_index('Age')
df_pie = oneHot_train_graph.groupby('Age').agg({"Survived" : "mean"}).reset_index().sort_values(by='Survived', ascending=False).set_index('Age')

fig = make_subplots(rows=1, cols=2, 
                    specs=[[{"type": "bar"}, {"type": "pie"}]],                          
                    column_widths=[0.7, 0.3], vertical_spacing=0, horizontal_spacing=0.02,
                    subplot_titles=("Average Fare per Group of Age", "Survival Percentage per Group of Age"))

fig.add_trace(go.Bar(x=df_bar['Fare'], y=df_bar.index, marker=dict(color=['#334550','#334550','#394184','#394184','#6D83AA','#6D83AA','#C8D0DF','#C8D0DF','#C8D0DF','#C8D0DF']),
                     name='Fare', orientation='h'), 
                     row=1, col=1)
fig.add_trace(go.Pie(values=df_pie['Survived'], labels=df_pie.index, name='Age',
                     marker=dict(colors=['#334550','#334668','#394184','#496595','#6D83AA','#91A2BF','#C8D0DF']), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label'), 
                    row=1, col=2)
# styling
fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_layout(height=500, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Group of Age Analysis",
                  template="plotly_white",
                  title_font=dict(size=29, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | PClass Visualization</b></p>
# </div>
# 
# 📌 **Interpret:** for this graph we'll be grouping by **<span style='color:lightseagreen'>PClass</span>** atribute. I have plotted 3 different graphs relating PClass with **<span style='color:lightseagreen'>Title</span>**, **<span style='color:lightseagreen'>Loneliness</span>** and **<span style='color:lightseagreen'>Family Size</span>**. We are able to appreciate that **<span style='color:lightseagreen'>lonely people</span>** have chosen class 3 more than any other. Indeed, both Class 3 and Class 2 have been the preference for more lonely people than accompanied. Class 1 as seen, is more likely to be chosen by people accompanied by one familiar. Families with 3 or more people on board are uniformly distributed between different classes.

# In[30]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20, 8))

sns.set_style('whitegrid')
ax = sns.countplot(x = "Pclass", hue='Title', data = df_data,ax = axes[0], palette=['#334550','#394184','#6D83AA','#91A2BF','#C8D0DF']);
ax.set_title('PClass per Title')

sns.set_style('whitegrid')
ax = sns.countplot(x = "Pclass", hue='IsAlone', data = df_data,ax = axes[1], palette=['#334550','#C8D0DF']);
_ = ax.set_title('PClass per Loneliness')

sns.set_style('whitegrid')
ax = sns.countplot(x = "Pclass", hue='FamilySize', data = df_data,ax = axes[2], palette=['#334550','#394184','#6D83AA','#91A2BF','#C8D0DF']);
__ = ax.set_title('PClass per Family Size')


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.4 | Fare Visualization</b></p>
# </div>
# 
# 📌 **Interpret:** The groups at the left side of the graph has the lowest survival rate and the groups at the right side of the graph has the highest survival rate. This high survival rate was not visible in the distribution graph. There is also an unusual group (15.742, 23.25] in the middle with high survival rate that is captured in this process.

# In[31]:


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Fare', hue='Survived', data=df_data, palette=['#334550','#C8D0DF'])

plt.xlabel('Fare', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

plt.show()


# # <b>5 <span style='color:#F1C40F'>|</span> Feature Transformation</b>
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.1 | Labeling Non-Numerical Features</b></p>
# </div>
# 
# Using **<span style='color:lightseagreen'>LabelEncoder</span>**, we are going to convert non-numerical features to numerical type. LabelEncoder basically labels the classes from **<span style='color:lightseagreen'>0 to n</span>**. This process is necessary for models to learn from those features.

# In[32]:


df_data.dtypes


# In[33]:


non_numerical_cols =  [col for col in df_data.columns if df_data[col].dtype == 'object']
non_numerical_cols.append('Age')

for df in dfs:
    for feature in non_numerical_cols:        
        df[feature] = LabelEncoder().fit_transform(df[feature])


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.2 | One Hot Encoding</b></p>
# </div>
# 
# To finish with, we are going to one hot encoded non-ordinal features. Those features are **<span style='color:lightseagreen'>Embarked, Sex, Deck, Title and PClass</span>**. `Age` and `Fare` as ordinal features are not converted.

# In[34]:


cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'FamilySizeGrouped']
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)


# In[35]:


df_data_OH = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)
drop_cols = ['Deck', 'Embarked', 'Family', 'FamilySize', 'FamilySizeGrouped',
             'Name', 'PassengerId', 'Pclass', 'Sex', 'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA','IsAlone']

df_data_OH.drop(columns=drop_cols, inplace=True)

df_data_OH.head()


# # <b>6 <span style='color:#F1C40F'>|</span> Modeling</b>
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.1 | Cross Validation</b></p>
# </div>
# 
# For the modeling part we will compare 10 known algorithms, and proceed to evaluate their average accuracy by a **<span style='color:lightseagreen'>stratified kfold cross validation</span>** procedure:
# * SVC
# * Decision Tree
# * AdaBoost
# * Random Forest
# * Extra Trees
# * Gradient Boosting
# * Multiple layer perceprton (neural network)
# * KNN
# * Logistic regression
# * Linear Discriminant Analysis
# * XGBoost Classifier
# 
# To begin with, we are going to create a cross validate model with Kfold stratified. Then we'll test each of the algorithms that I have mentioned before.

# In[36]:


x_train = df_data_OH[df_data_OH.Survived.isnull() == False].drop('Survived',axis=1)
y_train = df_data_OH[df_data_OH.Survived.isnull() == False].Survived


# In[37]:


kfold = StratifiedKFold(n_splits=10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LinearDiscriminantAnalysis",'XGBClassifier']})
cv_res = cv_res.sort_values(by='CrossValMeans',ascending = False)


# In[38]:


fig = make_subplots(rows=1, cols=1, 
                    specs=[[{"type": "bar"}]])

fig.add_trace(go.Bar(x=cv_res['CrossValMeans'], y=cv_res.Algorithm, marker=dict(color=['#334550','#334550','#334668','#334668','#496595','#496595','#6D83AA','#6D83AA','#91A2BF','#C8D0DF']),
                     name='Fare', orientation='h'), 
                     row=1, col=1)
# styling
fig.update_yaxes(showgrid=True, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=True, row=1, col=1)
fig.update_layout(height=500, bargap=0.1,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Cross Validation Scores",
                  template="plotly_white",
                  title_font=dict(size=29, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.2 | Hyperparameter tuning</b></p>
# </div>
# 
# For the ensemble modeling we are going to use: 
# * ExtraTreesClassifier
# * SVC
# * AdaBoost 
# * RandomForest
# * GradientBoosting
# In order to make execution quicker, we set **<span style='color:lightseagreen'>n_jobs to -1</span>**. This means that we are going to use every CPU we have in the computer.

# In[39]:


# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsadaDTC.fit(x_train,y_train)
ada_best = gsadaDTC.best_estimator_

# Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]}
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsGBC.fit(x_train,y_train)
GBC_best = gsGBC.best_estimator_

# SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsSVMC.fit(x_train,y_train)
SVMC_best = gsSVMC.best_estimator_

#ExtraTrees 
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsExtC.fit(x_train,y_train)
ExtC_best = gsExtC.best_estimator_

# RandomForest
rfc_single = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1) 
rfc_single.fit(x_train, y_train)
clear_output()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.3 | Ensemble Modeling and Prediction</b></p>
# </div>
# 
# For the final part of this project I have chosen **<span style='color:lightseagreen'>VotingClassifier</span>**. We'll fit the model and then proceed to make the predictions. At the final part of this section you'll find some graphs related to survival predictions made.

# In[40]:


votingC = VotingClassifier(estimators=[('rfc', rfc_single), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC.fit(x_train, y_train)
x_test = df_data_OH[df_data_OH.Survived.isnull() == True].drop('Survived',axis=1)
predictions_survived = votingC.predict(x_test)
clear_output()


# In[41]:


predictions = pd.DataFrame({'Survived' : predictions_survived},index = df_data[df_data.Survived.isnull() == True].PassengerId)
predictions['Survived'] = predictions.Survived.astype(int)
#predictions.to_csv('submission.csv')


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.4 | Simple Model</b></p>
# </div>
# 
# Now, as an extension, we are going to make predictions with both a single model, **<span style='color:lightseagreen'>RandomForestClassifier</span>**. Hereafter, we will make some studies on **<span style='color:lightseagreen'>features importance</span>** given by model.

# In[42]:


import eli5
from eli5.sklearn import PermutationImportance

rfc_single = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1) 
predictions_train = rfc_single.fit(x_train, y_train)
perm = PermutationImportance(rfc_single, random_state=1).fit(x_train, y_train)
#mediana = x_test.Fare.describe()[6]
#x_test.Fare = x_test.Fare.fillna(mediana)
predictions_survived = rfc_single.predict(x_test)
predictions = pd.DataFrame({'Survived' : predictions_survived},index = df_data[df_data.Survived.isnull() == True].PassengerId)
predictions['Survived'] = predictions.Survived.astype(int)
predictions.to_csv('submission.csv')
clear_output()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.5 | Permutation Importance</b></p>
# </div>
# 
# One of the most basic questions we might ask of a model is: **<span style='color:lightseagreen'>What features have the biggest impact on predictions?</span>** This concept is called feature importance. There are multiple ways to measure feature importance. Some approaches answer subtly different versions of the question above. Other approaches have documented shortcomings. In this section, we'll focus on permutation importance. Compared to most other approaches, permutation importance is:
# 
# * Fast to calculate,
# * Widely used and understood, and
# * Consistent with properties we would want a feature importance measure to have.

# In[43]:


eli5.show_weights(perm, feature_names = x_test.columns.tolist())


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.6 | Partial Plots</b></p>
# </div>
# 
# Like permutation importance, partial dependence plots are calculated **<span style='color:lightseagreen'> after a model has been fit </span>**. The model is fit on real data that has not been artificially manipulated in any way. While feature importance shows what variables most affect predictions, partial dependence plots show **<span style='color:lightseagreen'> how a feature affects predictions </span>**.

# In[44]:


from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
feature_names = [i for i in x_test.columns if x_test[i].dtype in [np.int64]]
pdp_goals = pdp.pdp_isolate(model=rfc_single, dataset=x_test, model_features=x_test.columns, feature='Fare')
clear_output()


# In[45]:


# plot it
pdp.pdp_plot(pdp_goals, 'Age')
plt.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.7 | SHAP Values</b></p>
# </div>
# 
# SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the **<span style='color:lightseagreen'> impact of each feature </span>**. Where could you use this?
# 
# * A model says a bank shouldn't loan someone money, and the bank is legally required to explain the basis for each loan rejection
# * A healthcare provider wants to identify what factors are driving each patient's risk of some disease so they can directly address those risk factors with targeted health interventions
# 
# We'll use SHAP Values to explain individual predictions in this lesson. In this section, we'll see how these can be aggregated into powerful model-level insights.

# In[46]:


X_test = df_data_OH[df_data_OH.Survived.isnull() == True].drop('Survived',axis=1)
X_test['Survived'] = predictions['Survived']
X_test = X_test.drop(891,axis=0)


# In[47]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(rfc_single)

# Calculate Shap values
shap_values = explainer.shap_values(X_test.iloc[48])      # random row
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[48])


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.8 | Confusion Matrix</b></p>
# </div>

# In[48]:


plot_confusion_matrix(rfc_single, x_train, y_train)  
clear_output()
plt.show()

