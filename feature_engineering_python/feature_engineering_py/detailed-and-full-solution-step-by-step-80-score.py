#!/usr/bin/env python
# coding: utf-8

# ## Detailed and Full Solution (Step by Step , > 80% score)
# #### By: Oday Mourad
# ##### 13 -  8 - 2022
# 
# 
# Hello kagglers ..
# 
# This notebook designed to be as detailed as possible solution for the Houses pricing problem, I tried to make it typical, clear, tidy and **beginner-friendly**.
# 
# If you find this notebook useful press the **UPVOTE** button, This helps me a lot ^-^.  
# 
# I hope you find it helpful.
# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg" width="600"/>
# 

# #### Table Of Content: <a class = "anchor" id = "toc" ></a>
# - [1 - Introduction](#introduction)
# - [2 - Importing](#import)
# - [3 - Descovering the data](#dtd)
# - [4 - Exploratory Data Analysis](#eda)
#     - [target](#eda_target)
#     - [categorical features with target](#cat_with_tar)
#     - [numerical features with target](#num_with_tar)
#     - [correlation between numerical features](#num_features)
#     - [correlation between categorical and numerical features](#cat_and_num)
# - [5 - Data Processing](#dp)
#     - [Filling Missed Values](#fmv)
#     - [Data Engineering](#de)
#     - [Preparing For Trainging](#prfortr)
# - [6 - Modeling](#modeling)
# 

# <a class="anchor" id="introduction">
#     <div style="color:#00ADB5;
#                display:fill;
#                border-radius:5px;
#                background-color:#393E46;
#                font-size:20px;
#                font-family:sans-serif;
#                letter-spacing:0.5px">
#             <p style="padding: 10px;
#                   color:white;">
#                 <b>1 ) Introduction:</b>
#             </p>
#     </div>
# </a>

# The competition is organised by **Kaggle** and is in the GettingStarted Prediction Competition series.
# 
# In this competition, you are supposed to predict predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# 
# Submissions are evaluated on **Classification Accuracy.**

# <a class="anchor" id="import">
#     <div style="color:#00ADB5;
#                display:fill;
#                border-radius:5px;
#                background-color:#393E46;
#                font-size:20px;
#                font-family:sans-serif;
#                letter-spacing:0.5px">
#             <p style="padding: 10px;
#                   color:white;">
#                 <b>2 ) Importing:</b>
#             </p>
#     </div>
# </a>

# In[1]:


#=======================================================================================
# Importing the libaries:
#=======================================================================================
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# In[2]:


#=======================================================================================
# Importing the data:
#=======================================================================================

def read_data():
    train_data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
    print("Train data imported successfully!!")
    print("-"*50)
    test_data = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
    print("Test data imported successfully!!")
    return train_data , test_data


# In[3]:


train_data , test_data = read_data()


# 
# <a class = "anchor"  id = "dtd"  >
#     <div style="color:#00ADB5;
#                display:fill;
#                border-radius:5px;
#                background-color:#393E46;
#                font-size:20px;
#                font-family:sans-serif;
#                letter-spacing:0.5px">
#             <p style="padding: 10px;
#                   color:white;">
#                 <b> 3 ) Discovering the data:</b>
#             </p>
#     </div>
# </a>
# 

# In[4]:


train_data.head()


# In[5]:


test_data.head()


# That's an interesting PassengerId .. maybe we can use it ..
# 

# In[6]:


print(train_data.columns.values)


# In[7]:


train_data.info()
print("-"*50)
test_data.info()


# Transported feature is object, I will convert it to int for visualization step.

# In[8]:


train_data["Transported"] = train_data["Transported"].astype("int")


# In[9]:


print("Train data shape = " , train_data.shape)
print("Test data shape = " , test_data.shape)


# The test data is about **50%** of the training data.

# In[10]:


print("Missed Data in train data:")
print(train_data.isnull().sum())
print("-" * 50)
print("Missed Data in test data:")
test_data.isnull().sum()


# There are a lot of missed data .. we are going to process them in the data processing step.

# In[11]:


train_data.describe()


# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>insights:</u></b><br>
#  
# * <i> There are an approximately <b>equal</b> number of transported passengers and non-transported passengers.</i><br>
# * <i> More than <b>75%</b> of the passengers are under the age of <b>38</b> and there some passengers are over <b>70</b> years old.</i><br>
# * <i> More than <b>50%</b> of the passengers didn't spend any money for RoomService, FoodCourt, ShoppingMall, Spa, VRDeck.  </i><br>
# * <i> here are too high outliers in RoomService, FoodCourt, ShoppingMall, Spa, VRDeck.</i><br>
# </div>

# In[12]:


train_data.describe(include = ["O"])


# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>insights:</u></b><br>
#  
# * <i> <b>Earth</b> is the most common HomePlanet.</i><br>
# * <i>Most of the passengers were not put into a cryosleep state.</i><br>
# * <i>There are many passengers with same Cabin (they shared the same cabin). </i><br>
# * <i> Most of the passengers going to <b>TRAPPIST-1e</b>.</i><br>
# * <i>only <b>199</b> passengers are VIP.</i><br>
# </div>

# In[13]:


# saving the test ids:
Test_Id = test_data.PassengerId


# ##### I will do a little of data engineering on **PassengerId** and **Cabin** to use it in EDA Step.

# In[14]:


combine = [train_data , test_data]


for dataset in combine:
    
    # =======================================================================
    # Extract Passenger Group:
    # =======================================================================

    dataset["PassengerGroup"] = dataset["PassengerId"].str.split('_' , expand = True)[1].astype(int).astype(str)
    dataset.drop(columns = ["PassengerId"] , inplace = True)
    # =======================================================================
    # Extract Cabin num, deck, side:
    # =======================================================================

    dataset["deck"] = (dataset.Cabin.str.split('/' , expand = True))[0]
    dataset["num"] = np.nan_to_num(dataset.Cabin.str.split('/', expand = True)[1].astype(float)).astype(int)
    dataset["side"] = dataset.Cabin.str.split('/', expand = True)[2]
    dataset.drop(columns = ["Cabin"] , inplace = True)
    
print(f"Available decks are ({train_data.deck.unique().shape[0]} decks): {train_data.deck.unique()}")
print(f"Available nums are ({train_data.num.unique().shape[0]} nums): {train_data.num.unique()}")
print(f"Available sides are ({train_data.side.unique().shape[0]} sides): {train_data.side.unique()}")


# <a href="#toc" role="button" aria-pressed="true" >Back to Table of Contents  ⬆️</a>

# <a class = "anchor" id = "eda">
#     <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 4 ) Exploratory Data Analysis (EDA):</b>
#         </p>
# </div>
# </a>
# 
# 
# 
# 
# 

# In[15]:


# Helper functions:
# ====================================================================
def survived_bar_plot(feature , ax = None , font_scale = 0.8):
    sns.set(font_scale=font_scale)  
    data = train_data[[feature, "Transported"]].groupby([feature], as_index=False).mean().sort_values(by='Transported', ascending=False)
    plot = sns.barplot(data = data , x = feature , y = "Transported" ,ci=None , ax = ax )
    plot.set_title(f"{feature} Vs Transported")
    plot.set(xlabel=None)
    plot.set(ylabel=None)
    sns.set(font_scale=font_scale)  
    plot.bar_label(plot.containers[0],fmt='%.2f')
# ====================================================================

def survived_table(feature):
    return train_data[[feature, "Transported"]].groupby([feature], as_index=False).mean().sort_values(by='Transported', ascending=False).style.background_gradient(low=0.75,high=1)
def survived_hist_plot(feature , bin_width = 5):
    plt.figure(figsize = (6,4))
    sns.histplot(data = train_data , x = feature , hue = "Transported",binwidth=bin_width,palette = sns.color_palette(["yellow" , "green"]) ,multiple = "stack" ).set_title(f"{feature} Vs Transported")
    plt.show()


# <a class = "anchor" id = eda_target >
#     <div style="color:black;
#                border-radius:0px;
#                background-color:#00ADB5;
#                font-size:14px;
#                font-family:sans-serif;
#                letter-spacing:0.5px">
#             <p style="padding: 6px;
#                   color:white;">
#                 <b>Target:</b>
#             </p>
#     </div>
# </a>
# 

# In[16]:


# ===================================================================
# Count of Transported Passengers:
# ===================================================================
f,ax=plt.subplots(1,2,figsize=(8,4))
train_data['Transported'].replace({0:"Not Transported",1:"Transported"}).value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_ylabel('')
sns.countplot(x = train_data["Transported"].replace({0:"Not Transported",1:"Transported"}) , ax = ax[1])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
plt.show()


# There are approximately equal number of transported and non-transported passengers.

# <a class = "anchor" id = "cat_with_tar">
# </a>
# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Categorical features with target:</b>
#         </p>
# </div>
# 

# In[17]:


fig , ax = plt.subplots(3,3 , figsize=(18 , 15))
survived_bar_plot("HomePlanet" , ax[0][0])
survived_bar_plot('CryoSleep' , ax[0][1])
survived_bar_plot('Destination' , ax[0][2])
survived_bar_plot('VIP'  , ax[1][0])
survived_bar_plot('PassengerGroup' , ax[1][1] , font_scale=0.7)
survived_bar_plot('deck' , ax[1][2],font_scale=0.7)
survived_bar_plot('side' , ax[2][0])


# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>insights:</u></b><br>
#  
# * <i>Passengers who came from Europa Planet is most common to be transported, then Mars then Earth.</i><br>
# * <i>Passengers going to 55 "Cancri e" are most common to be transported.</i><br>
# * <i>The non-VIP passengers are most common to be transported.</i><br>
# * <i>There are varying Transported proportions between different passenger groups.</i><br>
# * <i>As shown in the deck plot .. The highest Transport proportion is in "B" and "C", And the lowest in "T".</i><br>
# * <i>Passengers in the "S" side is most common to be transported than the "P" side. </i>
# </div>
# 
# 

# <a class = "anchor" id = "num_with_tar">
#     <div style="color:black;
#                border-radius:0px;
#                background-color:#00ADB5;
#                font-size:14px;
#                font-family:sans-serif;
#                letter-spacing:0.5px">
#             <p style="padding: 6px;
#                   color:white;">
#                 <b>Numerical features with target:</b>
#             </p>
#     </div>                          
# </a>
# 
# 

# **1 ) Age:**

# **Note:** This plot is a stack plot.

# In[18]:


sns.set_style("dark") # to remove the grid.
survived_hist_plot("Age") 


# children below 10 years old age are most common to be Transported. I am going to make is_child in Data Engineering step.

# **2 ) RoomService, FoodCourt, ShoppingMall, Spa, VrDeck:**

# In[19]:


plot , ax = plt.subplots(2 , 3, figsize = (18,8))
sns.boxplot(data = train_data , x = "Transported" , y = "RoomService" , ax = ax[0][0]).set_title("RoomService")
sns.boxplot(data = train_data , x = "Transported" , y = "FoodCourt" , ax = ax[0][1]).set_title("FoodCourt")
sns.boxplot(data = train_data , x = "Transported" , y = "ShoppingMall" , ax = ax[0][2]).set_title("ShoppingMall")
sns.boxplot(data = train_data , x = "Transported" , y = "Spa" , ax = ax[1][0]).set_title("Spa")
sns.boxplot(data = train_data , x = "Transported" , y = "VRDeck" , ax = ax[1][1]).set_title("VRDeck")
plt.subplots_adjust(wspace=0.4,hspace=0.4)


# As we saw above, The most values of these features is 0. and There are many outliers.

# <a class = "anchor" id = "num_features">
# 
# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Correlation between numerical features:</b>
#         </p>
# </div>
# </a>
# 
# 
# 

# In[20]:


sns.set(font_scale=0.8)
plt.figure(figsize = (8,8))
sns.heatmap(train_data.corr(),annot=True,fmt='.2f',cmap="Blues")


# insights:
# - luxury features haves some positive correlation with each other.
# - There are negtive correlation between luxury features and target feature.

# <a class = "anchor" id = "cat_and_num"><div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Correlation between numerical and categorical features:</b>
#         </p>
# </div></a>
# 
# 

# **1 ) Age:**

# In[21]:


plot , ax  = plt.subplots(2,3 , figsize = (16,6))
sns.boxplot(data = train_data , y = "Age" , x = "HomePlanet"  , ax = ax[0][0])
sns.boxplot(data = train_data , y = "Age" , x = "Destination"  , ax = ax[0][1])
sns.boxplot(data = train_data , y = "Age" , x = "CryoSleep"  , ax = ax[0][2])
sns.boxplot(data = train_data , y = "Age" , x = "VIP"  , ax = ax[1][0])
sns.boxplot(data = train_data , y = "Age" , x = "PassengerGroup"  , ax = ax[1][1])
sns.boxplot(data = train_data , y = "Age" , x = "VIP"  , ax = ax[1][0])


# from the plots above we can see that PassengerGroup is good to use for filling Age missed data.

# <a href="#toc" role="button" aria-pressed="true" >Back to Table of Contents  ⬆️</a>

# <a class = "anchor" id = "dp">
#     
# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 4 ) Data Processing:</b>
#         </p>
# </div>
# </a>
# 
# 
# 

# In[22]:


transported = train_data["Transported"]
all_data = pd.concat([train_data , test_data]).reset_index(drop = True)
all_data.drop(columns = ["Transported"] , inplace = True)


# <a class = "anchor" id = "fmv"><div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Filling Missed Values:</b>
#         </p>
# </div></a>
# 
# 

# In[23]:


all_data.isnull().sum()


# In[24]:


# Filling HomePlanet, CryoSleep, Destination, VIP:
all_data["HomePlanet"] = all_data["HomePlanet"].fillna(all_data["HomePlanet"].mode()[0]) 
all_data["CryoSleep"] = all_data["CryoSleep"].fillna(all_data["CryoSleep"].mode()[0]) 
all_data["Destination"] = all_data["Destination"].fillna(all_data["Destination"].mode()[0]) 
all_data["VIP"] = all_data["VIP"].fillna(all_data["VIP"].mode()[0]) 


# In[25]:


# Filling Age Feature by PassengerGroup:
PassengerGroups = ["1" , "2" , "3" , "4" , "5" , "6" , "7" , "8"]
median_ages = {}
for passengerGroup in PassengerGroups :
    median_ages[passengerGroup] = all_data.loc[all_data["PassengerGroup"] == passengerGroup , ["Age"]].median()

for index , passenger in all_data.iterrows():
    if pd.isna(passenger["Age"]):
        all_data.at[index , "Age"] = median_ages[passenger["PassengerGroup"]]


# In[26]:


# Filling RoomService, FoodCourt, ShoppingMall, Spa, VRDeck:
all_data["RoomService"] = all_data["RoomService"].fillna(all_data["RoomService"].mode()[0]) 
all_data["FoodCourt"] = all_data["FoodCourt"].fillna(all_data["FoodCourt"].mode()[0]) 
all_data["ShoppingMall"] = all_data["ShoppingMall"].fillna(all_data["ShoppingMall"].mode()[0]) 
all_data["Spa"] = all_data["Spa"].fillna(all_data["Spa"].mode()[0]) 
all_data["VRDeck"] = all_data["VRDeck"].fillna(all_data["VRDeck"].mode()[0]) 


# In[27]:


# Filling Cabin information:
all_data.deck = all_data.deck.fillna(all_data.deck.mode()[0])
all_data.num = all_data.num.fillna(all_data.num.mode()[0])
all_data.side = all_data.side.fillna(all_data.side.mode()[0])


# In[28]:


# Filling Name feature:
all_data.Name = all_data.Name.fillna("None")


# In[29]:


all_data.isnull().sum()


# No More Missed Data !!

# <a class = "anchor" id = "de"><div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Data Engineering:</b>
#         </p>
# </div>
#     </a>
# 

# **1 ) Family Size:**

# In[30]:


all_data["LastName"] = all_data.Name.str.split(" ",expand = True)[1]
last_name_count = all_data.Name.str.split(" ",expand = True)[1].value_counts()
all_data["FamilySize"] = [last_name_count[x] if not pd.isna(x) else None for x in all_data["LastName"]]
all_data["FamilySize"] = all_data["FamilySize"].fillna(0)
all_data.drop(columns = ["LastName" , "Name"] , inplace = True)


# **2 ) MoneySpent:**

# In[31]:


all_data["MoneySpent"] = all_data["RoomService"] + all_data["FoodCourt"] + all_data["ShoppingMall"] + \
all_data["Spa"] + all_data["VRDeck"] 


# **3 ) Spend Category:**

# In[32]:


all_data['SpendCategory'] = ''
all_data.loc[all_data['MoneySpent'].between(0, 1, 'left'), 'SpendCategory'] = 'Zero_Spend'
all_data.loc[all_data['MoneySpent'].between(1, 800, 'both'), 'SpendCategory'] = 'Under_800'
all_data.loc[all_data['MoneySpent'].between(800, 1200, 'right'), 'SpendCategory'] = 'Median_1200'
all_data.loc[all_data['MoneySpent'].between(1200, 2700, 'right'), 'SpendCategory'] = 'Upper_2700'
all_data.loc[all_data['MoneySpent'].between(2700, 100000, 'right'), 'SpendCategory'] = 'Big_Spender'
all_data['SpendCategory'] = all_data['SpendCategory'].astype('category')


# **4 ) Any_Spend:**

# In[33]:


all_data["AnySpend"] = all_data["MoneySpent"] > 0


# **5 ) Is Child:**

# In[34]:


all_data["IsChild"] = all_data["Age"] <= 10


# In[35]:


all_data.head()


# <a class = "anchor" id  = "prfortr">
# 
# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Preparing for Training:</b>
#         </p>
# </div>
# 
# 
# </a>
# 
# 
# 

# In[36]:


# =========================================================================
#  Converting Bool to Int
# =========================================================================

all_data["CryoSleep"] = all_data["CryoSleep"].astype(int)
all_data["VIP"] = all_data["VIP"].astype(int)
all_data["IsChild"] = all_data["IsChild"].astype(int)
all_data["FamilySize"] = all_data["FamilySize"].astype(int)
all_data["AnySpend"] = all_data["AnySpend"].astype(int)


# In[37]:


all_data = pd.get_dummies(all_data)


# In[38]:


all_data.head()


# In[39]:


train_data = all_data[:len(train_data)]
test_data = all_data[len(train_data):]


# <a href="#toc" role="button" aria-pressed="true" >Back to Table of Contents  ⬆️</a>

# 
# <a class = "anchor" id = "modeling">
# 
# 
# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 5 ) Modeling:</b>
#         </p>
# </div>
# 
# 
# 
# </a>
# 
# 
# 
# 
# 

# In[40]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier


# In[41]:


# ==================================================================================
# Preparing Data For Training:
# ==================================================================================

Y_train = transported
X_train = train_data
X_test = test_data
print(f"X_train shape is = {X_train.shape}" )
print(f"Y_train shape is = {Y_train.shape}" )
print(f"Test shape is = {X_test.shape}" )


# In[42]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=12)


# In[43]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[44]:


# Modeling step Test different algorithms 
random_state = 2
classifiers = []
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier(random_state = random_state))
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression",
                                                                                      "LinearDiscriminantAnalysis" ,"XGBoost"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[45]:


results = pd.DataFrame({"Model" : ["DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis" ,"XGBoost"],"Score" : cv_means , "Std" : cv_std})
results.sort_values("Score" , ascending = False)


# In[46]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier(random_state=random_state)

# gb_param_grid = {'loss' : ["deviance"],
#               'n_estimators' : [600 , 800],
#               'learning_rate': [0.01],
#               'max_depth': [14 , 16],
#               'min_samples_leaf': [20 , 25],
#               'max_features': [0.03 , 0.05 ,0.1] 
#               }

gb_param_grid = {
              'learning_rate': [0.01],
                "max_depth" : [14],
              'min_samples_leaf': [25],
              'max_features': [0.05] , 
                "n_estimators" : [600]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=5, scoring="accuracy", verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

print("The best Model Parameters is :")
print(GBC_best)
print(f"With Cross Validation Score = {gsGBC.best_score_}")


# In[47]:


plot_learning_curve(GBC_best , "Gradient Boosting" , X_train , Y_train)


# In[48]:


# RFC Parameters tunning 
RFC = RandomForestClassifier(random_state=random_state)

## Search grid for optimal parameters
# rf_param_grid = {"max_depth": [16],
#               "max_features": [0.2 ],
#               "min_samples_split": [5],
#               "min_samples_leaf": [15],
#               "bootstrap": [True , False],
#               "n_estimators" :[550 , 600],
#               "criterion": ["gini"]}

rf_param_grid = {"max_depth": [16],
              "max_features": [0.2],
              "min_samples_leaf": [15],
              "min_samples_split": [5],
              "bootstrap": [False],
              "n_estimators" :[560],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=5, scoring="accuracy", verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

print("The best Model Parameters is :")
print(RFC_best)
print(f"With Cross Validation Score = {gsRFC.best_score_}")


# In[49]:


plot_learning_curve(RFC_best , "Random Forest" , X_train , Y_train)


# In[50]:


#ExtraTrees 
ExtC = ExtraTreesClassifier(random_state=random_state)

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [18],
              "max_features": [0.5],
              "min_samples_split": [5],
              "min_samples_leaf": [15],
              "bootstrap": [False],
              "n_estimators" :[550],
              "criterion": ["gini"],
                }

# ex_param_grid = {"max_depth": [18],
#               "max_features": [10 , ],
#               "min_samples_split": [10 , 8],
#               "min_samples_leaf": [3 , 5],
#               "bootstrap": [False],
#               "n_estimators" :[300],
#               "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=5, scoring="accuracy", verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

print("The best Model Parameters is :")
print(ExtC_best)
print(f"With Cross Validation Score = {gsExtC.best_score_}")


# In[51]:


plot_learning_curve(ExtC_best , "Extra Trees" , X_train , Y_train)


# In[52]:


# Logistic Regression: 
LogReg = LogisticRegression(random_state=random_state)

## Search grid for optimal parameters
log_reg_param_grid = {
"C":[0.005]   ,
    "max_iter" : [600]
}

gsLogReg = GridSearchCV(LogReg,param_grid = log_reg_param_grid, cv=5, scoring="accuracy", verbose = 1)

gsLogReg.fit(X_train,Y_train)

LogReg_best = gsLogReg.best_estimator_

print("The best Model Parameters is :")
print(LogReg_best)
print(f"With Cross Validation Score = {gsLogReg.best_score_}")


# In[53]:


plot_learning_curve(LogReg_best , "Logistic Regression" , X_train , Y_train)


# In[54]:


from catboost import CatBoostClassifier
cbc = CatBoostClassifier(verbose=0, n_estimators=600)
cbc.fit(X_train, Y_train)


# In[55]:


cross_val_score(cbc, X_train, y = Y_train, scoring = "accuracy", cv = 5).mean()


# In[56]:


# ============================================================
# Train on all Data
# ============================================================
GBC_all_data = GBC_best.fit(X_train , Y_train)
RFC_all_data = RFC_best.fit(X_train , Y_train)
ExtC_all_data = ExtC_best.fit(X_train , Y_train)
LogReg_all_data = LogReg_best.fit(X_train , Y_train)
cbc_all_data = cbc.fit(X_train , Y_train)


# In[57]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                                       ('gbc',GBC_best) , ("logreg" ,LogReg_best ) , ("catboost" , cbc)], voting='soft')

votingC = votingC.fit(X_train, Y_train)


# In[58]:


predictions = pd.Series(votingC.predict(X_test).astype(bool), name="Transported")

results = pd.concat([Test_Id,predictions],axis=1)

results.to_csv("submission.csv",index=False)

