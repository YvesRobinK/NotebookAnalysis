#!/usr/bin/env python
# coding: utf-8

# # SPACESHIP TITANIC🚀 ~ 0.81 % XGBoost - EDA + ML approaches
# **ML COMPETITION**

# *author*: **Giacomo Cavalca** - PhD student in Data science
# 
# *mail*: gcavalcaphd@gmail.com

# <img src= "https://img.freepik.com/premium-photo/spaceship-grunge-interior-with-view-planet-earth_117023-176.jpg?w=2000" alt ="Titanic" style='width: 75%; margin-left: 10%
# '>

# <div style="color:white;
#            display:fill;
#            background-color:#b0c4de;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            flex-direction: row;">
# 
# <h1 style="padding: 2rem;
#           color:white;
#           text-align:center;
#           margin:0 auto;
#           font-size:3rem;">
#    TABLE OF CONTENTS:
# </h1>
#  
# </div>
# <div style="
#            display:fill;
#            background-color:#b0c4de;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            flex-direction: row;
#            justify-content: center"
#      >
# <ul style="
#            background-color:#b0c4de;
#            margin-left: 2rem;
#                ">
# 
# <li style="color: white;
#            font-size:1.75rem;">
# <text >
#  SETTINGS AND DATA LOADING
# </text>
#     <ul>
#       <li>Import libraries</li>
#     </ul>
# </li>
# <li style="color: white;
#            font-size:1.75rem;">
# 
# <text >
#  EXPLORATORY DATA ANALYSIS
# </text>
#     <ul>
#       <li>Missing values visualization</li>
#       <li>Feature engineering</li>
#       <li>Filling NaN values</li>
#       <li>Bivariate analysis</li>
#     </ul>
# </li>
# 
#     
# <li style="color: white;
#            font-size:1.75rem;">
# <text >
#   DATA PREPROCESSING
# </text>
# </li>
# 
#  <li style="color: white;
#            font-size:1.75rem;">
# <text >
#   ML - XGBoost 
# </text>
# </li> 
#     
# </ul>
# </div>

# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#b0c4de;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#             justify-content:center;">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
#     SETTINGS AND DATA LOADING
# </h2>
# </div>

# # IMPORT LIBRARIES

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier


# In[2]:


root_dir = '/kaggle/input/spaceship-titanic'
files = os.path.join(root_dir)
filenames = os.listdir(files)
print(filenames)


# In[3]:


#TRAIN DATA
train_data = pd.read_csv(os.path.join(root_dir,'train.csv'))
train_data.head()


# In[4]:


#TEST DATA
test_data = pd.read_csv(os.path.join(root_dir,'test.csv'))
test_data.head()


# In[5]:


print(colored(f"Training data","blue"),"-> ROWS:",train_data.shape[0],"COLUMNS:",train_data.shape[1])
print(colored(f"Test data","red"),"-> ROWS:",test_data.shape[0],"COLUMNS:",test_data.shape[1])


# In[6]:


train_data.info()


# In[7]:


train_data.describe(include='all')


# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#b0c4de;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#             justify-content:center;">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
#     EXPLORATORY DATA ANALYSIS
# </h2>
# </div>

# # MISSING VALUES VISUALIZATION

# In[8]:


train_data.isna().any()


# In[9]:


def percentageOfnull(df):
    percentage = ((df.isna().sum()/df.isna().count())*100).sort_values(ascending=False)
    count = df.isna().sum().sort_values(ascending=False)
    diff = pd.concat([count,percentage],axis=1,keys=['Null Count','Null Percentage'])
    return diff


# In[10]:


nan_feats_tr = train_data.columns[train_data.isna().any()].tolist()
n_nans_tr = train_data[nan_feats_tr].isna().sum()
print(f"TRAINING SET\nMissing values:\n{n_nans_tr}")
plt.figure(figsize=(14,5))
plt.title("TRAINING SET - missing values")
sns.barplot(y=n_nans_tr,x=nan_feats_tr)


# In[11]:


plt.figure(figsize=(14,5))
sns.displot(
    data=train_data.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.title("TRAINING SET - missing values distribution")


# In[12]:


percentageOfnull(train_data)


# In[13]:


nan_feats_te = test_data.columns[test_data.isna().any()].tolist()
n_nans_te = test_data[nan_feats_te].isna().sum()
print(f"TESTING SET\nMissing values:\n{n_nans_te}")
plt.figure(figsize=(14,5))
plt.title("TESTING SET - missing values")
sns.barplot(y=n_nans_te,x=nan_feats_te)


# In[14]:


plt.figure(figsize=(14,5))
sns.displot(
    data=test_data.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.title("TESTING SET - missing values distribution")


# In[15]:


percentageOfnull(test_data)


# # FEATURE ENGINEERING

# Let's take a look at the features

# ***PASSENGER ID***

# PassengerId is a unique Id for each passenger so in general it could be discarded, but in this case it contains two subfeatures:
# - gggg which indicates a group a passenger is travelling with
# - pp which is the number of the passenger within the group

# **TRAINING SET**

# In[16]:


#Create 2 new features: gggg and pp
gggg_pp = train_data['PassengerId'].apply(lambda x: x.split('_')).values
gggg = list(map(lambda x: x[0], gggg_pp))
pp = list(map(lambda x: x[1], gggg_pp))
train_data['gggg'] = gggg
train_data['pp'] = pp
train_data['pp'] = train_data['pp'].astype('int64')


# In[17]:


mode = train_data["gggg"].mode()[0]
maxP_inGroup = len(train_data[train_data["gggg"] == mode])
print("The maximum number of passengers in a single group is",maxP_inGroup)


# In[18]:


train_data[train_data["gggg"] == mode] #proof


# In[19]:


train_data['group_size'] = 0
for i in range(maxP_inGroup):
    curr_gggg = train_data[train_data['pp'] == i + 1]['gggg'].to_numpy()
    train_data.loc[train_data['gggg'].isin(curr_gggg), ['group_size']] = i + 1

plt.figure(figsize=(12,8))
print(colored("Value Counts based on the group size:\n", 'cyan', attrs=['underline', 'bold']))
print(colored("Gr. size, Count", 'blue', attrs=['bold']))
print(train_data['group_size'].value_counts())
sns.barplot(y=train_data['group_size'].value_counts(), x=np.unique(train_data['pp']), palette='viridis')
plt.show()
sns.catplot(x="group_size",  kind="count", hue='Transported', data=train_data, palette='viridis').set(title='Group Size and Transported Count')
plt.show()


# **TESTING SET**

# In[20]:


gggg_pp = test_data['PassengerId'].apply(lambda x: x.split('_')).values
gggg = list(map(lambda x: x[0], gggg_pp))
pp = list(map(lambda x: x[1], gggg_pp))
test_data['gggg'] = gggg
test_data['pp'] = pp
test_data['pp'] = test_data['pp'].astype('int64')


# In[21]:


mode = test_data["gggg"].mode()[0]
maxP_inGroup = len(test_data[test_data["gggg"] == mode])
print("The maximum number of passengers in the same group is",maxP_inGroup)


# In[22]:


test_data['group_size'] = 0
for i in range(maxP_inGroup):
    curr_gggg = test_data[test_data['pp'] == i + 1]['gggg'].to_numpy()
    test_data.loc[test_data['gggg'].isin(curr_gggg), ['group_size']] = i + 1

plt.figure(figsize=(12,8))
print(colored("Value Counts based on the group size:\n", 'cyan', attrs=['underline', 'bold']))
print(colored("Gr. size, Count", 'blue', attrs=['bold']))
print(test_data['group_size'].value_counts())
sns.barplot(y=test_data['group_size'].value_counts(), x=np.unique(test_data['pp']), palette='viridis')
plt.show()


# In[23]:


train_data.head()


# In[24]:


print(len(train_data[train_data["group_size"] == 1]))
print(len(train_data[train_data["group_size"] != 1]))


# In[25]:


#Create a new feature InGroup to indicate if a passenger is alone or in group
train_data["InGroup"] = train_data["group_size"]==1
test_data["InGroup"] = test_data["group_size"]==1


# In[26]:


train_data.head()


# In[27]:


test_data.head()


# **CABIN**

# In[28]:


train_data["Deck"] = train_data["Cabin"].apply(lambda x: str(x).split("/")[0] if(np.all(pd.notnull(x))) else x)
test_data["Deck"] = test_data["Cabin"].apply(lambda x: str(x).split("/")[0] if(np.all(pd.notnull(x))) else x)
train_data["Num"] = train_data["Cabin"].apply(lambda x: int(str(x).split("/")[1]) if(np.all(pd.notnull(x))) else x)
test_data["Num"] = test_data["Cabin"].apply(lambda x: int(str(x).split("/")[1]) if(np.all(pd.notnull(x))) else x)
train_data["Side"] = train_data["Cabin"].apply(lambda x: str(x).split("/")[2] if(np.all(pd.notnull(x))) else x)
test_data["Side"] = test_data["Cabin"].apply(lambda x: str(x).split("/")[2] if(np.all(pd.notnull(x))) else x)


# In[29]:


#Save PassengerId for the submission
Id_test_list = test_data["PassengerId"].tolist()

#Delete useless features
train_data.drop("PassengerId",axis=1,inplace=True)
test_data.drop("PassengerId",axis=1,inplace=True)
train_data.drop("Cabin",axis=1,inplace=True)
test_data.drop("Cabin",axis=1,inplace=True)
train_data.drop("Name",axis=1,inplace=True)
test_data.drop("Name",axis=1,inplace=True)
train_data.drop("gggg",axis=1,inplace=True)
test_data.drop("gggg",axis=1,inplace=True)
train_data.drop("pp",axis=1,inplace=True)
test_data.drop("pp",axis=1,inplace=True)
train_data.drop("group_size",axis=1,inplace=True)
test_data.drop("group_size",axis=1,inplace=True)


# In[30]:


train_data.head()


# In[31]:


test_data.head()


# In[32]:


train_data.dtypes


# # FILLING NAN VALUES

# In[33]:


num_feats = list(train_data.select_dtypes(include='number'))
categ_feats = list(train_data.select_dtypes(exclude='number'))
test_categ_feats = list(test_data.select_dtypes(exclude='number'))
print(colored("Numerical features:","blue"),num_feats)
print(colored("Categorical features (training set):","red"),categ_feats)
print(colored("Categorical features (testing set):","red"),test_categ_feats)


# In[34]:


#replace missing values in each numerical feature with the median
for feat in num_feats:
    train_data[feat].fillna(train_data[feat].median(), inplace=True)
    test_data[feat].fillna(test_data[feat].median(), inplace=True)

#replace missing values in each categorical feature with the most frequent value
for feat in categ_feats:
    train_data[feat].fillna(train_data[feat].value_counts().index[0],inplace=True)
    
for feat in test_categ_feats:
    test_data[feat].fillna(test_data[feat].value_counts().index[0],inplace=True)


# In[35]:


#just as proof
print(colored("TRAINING SET\n",attrs=['bold']),percentageOfnull(train_data),end="\n\n")
print(colored("TESTING SET\n",attrs=['bold']),percentageOfnull(test_data),end="\n\n")


# In[36]:


#print categories of each categorical column after removing unnecessary columns
for col in train_data.select_dtypes(exclude=['number']):
  print(f'{col:-<30},{train_data[col].unique()}')


# In[37]:


#Plot distribution of numerical columns
fig = plt.figure(figsize=(12,4))
for i,col in enumerate(num_feats):
    ax = fig.add_subplot(2,4,i+1)
    sns.distplot(train_data[col])
    
fig.tight_layout()
plt.show()


# In[38]:


#Plot distribution of categorical columns
colors =sns.color_palette("inferno_r", 7)

fig = plt.figure(figsize= (15,10))
for i, col in enumerate(categ_feats):
    ax=fig.add_subplot(3, 3, i+1)
    sns.countplot(x=train_data[col],palette=colors, ax=ax)

fig.tight_layout()  
plt.show()


# In[39]:


#Plot donut chart for each categorical column
fig = plt.figure(figsize= (16,10))
for i, col in enumerate(categ_feats):
    
    ax=fig.add_subplot( 3, 3, i+1)
    
    train_data[col].value_counts().plot.pie(autopct='%.0f%%', pctdistance=0.80, colors=sns.color_palette("inferno_r", 7))
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig1 = plt.gcf()
    # Adding Circle in Pie chart
    fig1.gca().add_artist(centre_circle)
fig.tight_layout()  
plt.show()


# # BIVARIATE ANALYSIS

# **NUMERICAL - NUMERICAL analysis**

# In[40]:


print("Number of numerical features:",len(num_feats))
num_feats


# In[41]:


#plot the pair plot of numerical features
sns.pairplot(data = train_data, vars=num_feats, diag_kind="kde")


# In[42]:


sns.set()
fig, axes = plt.subplots(len(num_feats), len(num_feats),figsize=(20, 20))
fig.suptitle('Correlation between numerical features', fontsize=24)
for i,col1 in enumerate(num_feats):
    for j,col2 in enumerate(num_feats):
        sns.regplot(x=col1,y=col2,data=train_data,color='blue', scatter_kws={
                    "color": "deepskyblue"}, line_kws={"color": "red"}, ax=axes[i,j])
fig.tight_layout()
plt.subplots_adjust(top=0.90)


# In[43]:


plt.figure(figsize=(15,12))
sns.heatmap(train_data[num_feats].corr(),cmap='BuPu',annot=True)
plt.title ('Correlation HeatMap', fontsize=20)
plt.show()


# It seems that most of the numerical features are not strongly correlated

# **NUMERICAL - CATEGORICAL analysis**

# In[44]:


fig = plt.figure(figsize= (16,10))
for i, col in enumerate(num_feats):
    
    ax=fig.add_subplot( 2, 4, i+1)

    train_data.groupby(['Transported'])[col].mean().plot(kind='bar',color=["#7AC5CD","#53868B"])
    ax.set_ylabel(col)
fig.tight_layout()  
plt.show()


# We can see that:
# * the significant majority of the passengers who spent more on RoomService, Spa and VRDeck were not transported
# * among the passengers who have spent a lot in FoodCourt, there is a slight majority were transported

# In[45]:


Transported_df=train_data[train_data['Transported']==True]
NotTransported_df=train_data[train_data['Transported']==False]
fig = plt.figure(figsize= (16,10))
for i, col in enumerate(num_feats):
    
    ax=fig.add_subplot( 2, 4, i+1)
    
    sns.distplot(Transported_df[col],label='Transp')
    sns.distplot(NotTransported_df[col],label='not_Transp')
    plt.legend()
    
fig.tight_layout()  
plt.show()


# The distribution of each numerical feature for Transported and NoTransported passengers seems different. Therefore, these variables affect chances of being transported.

# **CATEGORICAL - CATEGORICAL analysis**

# In[46]:


sns.countplot(x="Deck",data=train_data,hue="Transported")


# In[47]:


sns.countplot(x="Side",data=train_data,hue="Transported")


# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#b0c4de;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#             justify-content:center;">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
#     DATA PREPROCESSING
# </h2>
# </div>

# In[48]:


train_data.columns


# In[49]:


train_data.dtypes


# In[50]:


train_data.head()


# In[51]:


test_data.head()


# **OBJECT FEATURES**:
# Encode target labels with value between 0 and n_classes-1.
# 
# **BOOL FEATURES**:
# Cast to 'int'
# 
# **FLOAT FEATURES**:
# - normalization on Age
# - standardization on the others

# In[52]:


LABELS = test_data.columns
encoder = LabelEncoder()
for col in LABELS:
    # Check if object
    if train_data[col].dtype == 'O':
        train_data[col] = encoder.fit_transform(train_data[col]) #fit label encoder and return encoded labels.
        test_data[col] = encoder.transform(test_data[col]) #transform labels to normalized encoding
        
    elif train_data[col].dtype == 'bool':
        train_data[col] = train_data[col].astype('int')
        test_data[col] = test_data[col].astype('int')

train_data['Transported'] = train_data['Transported'].astype('int')
LABELS_MM = ['Age']
LABELS_SS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
mm_scaler = MinMaxScaler() #default=(0,1)
ss_scaler = StandardScaler()
# Apply Min-Max Scaling
train_data[LABELS_MM] = mm_scaler.fit_transform(train_data[LABELS_MM])
test_data[LABELS_MM] = mm_scaler.transform(test_data[LABELS_MM])
# Apply Standard Scaling
train_data[LABELS_SS] = ss_scaler.fit_transform(train_data[LABELS_SS])
test_data[LABELS_SS] = ss_scaler.transform(test_data[LABELS_SS])


# In[53]:


train_data.head()


# In[54]:


test_data.head()


# In[55]:


x=train_data[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'InGroup',
       'Deck', 'Num', 'Side']]
y=train_data['Transported']


# In[56]:


x.head()


# In[57]:


plt.figure(figsize=(20, 15))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm")


# In[58]:


sns.regplot(train_data['CryoSleep'], train_data['Transported'], data=train_data)


# There is a strong correlation between CryoSleep and Transported

# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#b0c4de;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#             justify-content:center;">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
#     ML - XGBoost
# </h2>
# </div>

# In[59]:


Xtr,Xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=1)
Xtr.dtypes


# **BOOSTING**

# In[60]:


params = { 'max_depth': [3,6,10,12],
          'gamma': [0,1],
          'learning_rate': [0.01, 0.02, 0.05, 0.1],
          'n_estimators': [100, 200, 500, 1000],
          'colsample_bytree': [0.3, 0.5, 0.7]}

xgb_clf = XGBClassifier(seed = 20)

xgb_grid = GridSearchCV(estimator=xgb_clf, param_grid=params, cv=2)

xgb_grid.fit(Xtr,ytr)


# In[61]:


acc = accuracy_score(xgb_grid.predict(Xte),yte)
print(acc)
print(xgb_grid.best_params_)


# **PREDICTION**

# In[62]:


pred=pd.Series(xgb_grid.predict(test_data)).map({0:False, 1:True})
len(pred)


# In[63]:


submission = pd.DataFrame({'PassengerId': Id_test_list,
                       'Transported': pred})
submission.head()


# In[64]:


submission.to_csv("submission.csv", index=False)


# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#b0c4de;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#             justify-content:center;">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
#     END
# </h2>
# </div>

# # Thank you for your attention and please let me know what you think. Any doubts or suggestions are welcome! 🙂🚀
