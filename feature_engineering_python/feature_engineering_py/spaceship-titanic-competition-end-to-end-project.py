#!/usr/bin/env python
# coding: utf-8

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# 🚀 Spaceship Titanic Kaggle Competition Complete Guide
# </p>
# </div>

# <center>
# <img src="https://cdn.mos.cms.futurecdn.net/AKbyqTKUkicsYGx3xwe3HA.jpg" width=800 height=500 />
# </center>

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>Notebook- </b> Goal </h2>

# <div style="font-family:Verdana; background-color:aliceblue; padding:30px; font-size:17px;color:#034914">
# 
# 💡 In this project we will do Binary Classification on Titanic Spaceship Dataset. (Kaggle Competition)<br>
# 
# 💡 The objective of this project is to predict whether a person will be transported to an alternate dimension or not.<br>
# </div>

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>Notebook- </b> Content </h2>

# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# 💡 Basic Understanding of Data.<br>
# 
# 💡 Exploratory Data Analysis (EDA).<br>
# 
# 💡 Feature Engineering.<br>
# 
# 💡 Data Preprocessing.<br>
# 
# 💡 Model Building.<br>
# 
# 💡 Model Performance Check.<br>
# 
# 💡 Model Hyper Parameter Tunning.<br>
# 
# 💡 Predicting Test Data using best Model.<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Importing Libraries
# </p>
# </div>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

from imblearn.over_sampling import SMOTE


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Loading Datasets
# </p>
# </div>

# In[2]:


train_df = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>Data- </b> Description </h2>

# > * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.<br>
# > * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.<br>
# > * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.<br>
# > * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.<br>
# > * **Destination** - The planet the passenger will be debarking to.<br>
# > * **Age** - The age of the passenger.<br>
# > * **VIP** - Whether the passenger has paid for special VIP service during the voyage.<br>
# > * **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.<br>
# > * **Name** - The first and last names of the passenger.<br>
# > * **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.<br>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Basic Understanding of Data
# </p>
# </div>

# ### 1.Checking Dimensions of Data

# In[3]:


print("Training Dataset shape is: ",train_df.shape)
print("Testing Dataset shape is: ",test_df.shape)


# ### 2. Showing Training & Testing Data

# In[4]:


train_df.head()


# In[5]:


test_df.head()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
#  💡 We can observe that in our testing dataset we don't have Transported feature but in training data we have that feature.<br>
#  💡 So, we have to build model using training data and have to do prediction for our testing data.<br>
# </div>

# ### 3. Checking Duplicates Data

# In[6]:


print(f"Duplicates in Train Dataset is:{train_df.duplicated().sum()},({100*train_df.duplicated().sum()/len(train_df)})%")
print(f"Duplicates in Test Dataset is:{test_df.duplicated().sum()},({100*test_df.duplicated().sum()/len(test_df)})%")


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
#     
# <b>Observation</b><br>
# 💡 We can observe that we don't have any duplicates values in our both training & testing datasets.<br>
# 💡 So we dont have any type of Data Lekage in our DataSet.<br>
# </div>

# ### 4. Checking Data-Types of Training & Testing Data

# In[7]:


print("Data Types of features of Training Data is:")
print(train_df.dtypes)
print("\n"+"-"*100)
print("\nData types of features of Testing Data is:")
print(test_df.dtypes)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that CryoSleep & VIP features contains boolean values but their data type is object so we have to convert their data-type to bool.<br>
# 💡 We will convert their Data-Types when we will do Data-Preprocessing.<br>
# </div>

# ### 5. Checking Total Number & Percentage of Missing Values in Training Dataset

# In[8]:


df1 = (train_df.isnull().sum()[train_df.isnull().sum()>0]).to_frame().rename(columns={0:"Number of Missing values"})
df1["% of Missing Values"] = round((100*train_df.isnull().sum()[train_df.isnull().sum()>0]/len(train_df)),2)
df1


# ### 6. Checking Total Number & Percentage of Missing Values in Testing Data.

# In[9]:


df2 = (test_df.isnull().sum()[test_df.isnull().sum()>0]).to_frame().rename(columns={0:"Number of Missing values"})
df2["% of Missing Values"] = round((100*test_df.isnull().sum()[test_df.isnull().sum()>0]/len(test_df)),2).values
df2


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that there is very less % of missing values in both training & testing data.<br>
# 💡 So instead of dropping those missing values we will fill/replace those missing values with best suitable values according to the data.<br>
# </div>

# ### 7. Checking Cardinality of Categorical features.

# In[10]:


print("cardinality of categorical features in training datasets is:")
print(train_df.select_dtypes(include="object").nunique())
print("\n","-"*70)
print("\nCardinality of categorical features in testing datsets is:")
print(test_df.select_dtypes(include="object").nunique())


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that PassengerId, Cabin & Name feature of both datasets are having high cardinality<br>
# 💡 We normally drop the features having high cardinality but in this project we will do Feature Engineering and will create new features from this features.<br>
# 💡 Because more amount of data leads to better predictions by model.<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Exploratory Data Analysis (EDA)
# </p>
# </div>

# ### 1. Visualizing Target Feature "Transported"

# In[11]:


plt.figure(figsize=(10,6))
plt.pie(train_df["Transported"].value_counts(),labels=train_df["Transported"].value_counts().keys(),autopct="%1.1f%%",
       textprops={"fontsize":20,"fontweight":"black"},colors=sns.color_palette("Set2"))
plt.title("Transported Feature Distribution");


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
#     
# <b>Observation</b><br>
# 💡 We can observe that our Transported Feature is highly balanced.<br>
# 💡 So we don't have to use techniques like under_sampling or over_sampling<br>
# </div>

# ### 2.Visualizing AGE Feature

# In[12]:


plt.figure(figsize=(16,6))
sns.histplot(x=train_df["Age"],hue="Transported",data=train_df,kde=True,palette="Set2")
plt.title("Age Feature Distribution");


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 Most of the passengers were between age 18-32.<br>
# 💡 Age from 0-18 passengers are highly transported when compared with not transported passengers espically for those who were new born.<br>
# 💡 Age from 18-32 passengers are comparatively less transported when compared to not transported passengers. <br>
# 💡 Age above 32 seems to be equally transported when compared to not transported passengers.<br>
# 
# <b>Insights</b><br>
# 💡 We can create a new feature Age-Catgeory from age in which we can split ages into different categories.<br>
# </div>

# ### 3. Visualizing All Expenditure Features ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")

# In[13]:


exp_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

plt.figure(figsize=(14,10))
for idx,column in enumerate(exp_cols):
    plt.subplot(3,2,idx+1)
    sns.histplot(x=column, hue="Transported", data=train_df,bins=30,kde=True,palette="Set2")
    plt.title(f"{column} Distribution")
    plt.ylim(0,100)
    plt.tight_layout()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that most of the passengers doesn't seems to expend any money.<br>
# 💡 Since most of the expenses are 0 so the values with higher expenses are kind of outliers in our data.<br>
# 💡 We can observe that RoomService,Spa & VRDeck seems to have similar distributions.<br>
# 💡 We can also observe that FoodCourt & ShoppingMall are having kind of similar distributions.<br>
# 💡 All the expenditure features distribution is Right-Skewed.<br>
# 💡 Passengers having less expenses are more likely to be transported than passengers having high expenses.<br>
# 
# <b>Insights</b><br>
# 💡 Since, all expenditure features are having right-skewed distribution. So before Model Building we will transform these features to normal distribution using log-transformation<br>
# 
# 💡 We can create a new feature Total Expenditure indicating the total expenses of all different expenditures done by the passengers.<br>
# 💡 Since, most people expense is 0 so we can create a new boolean feature No Spending indicating whether the passenger total expense is 0 or not.<br>
# 💡 We can split Total Expenditure into different categories of expenditure like Low , Medium & High Expenses and create one more new feature Expenditure Category<br>
# </div>

# 
# ### 4. Visualizing Categorical Features ("HomePlanet", "CryoSleep", "Destination", "VIP")

# In[14]:


cat_cols = ["HomePlanet","CryoSleep","Destination","VIP"]

plt.figure(figsize=(12,20))
for idx,column in enumerate(cat_cols):
    plt.subplot(4,1,idx+1)
    sns.countplot(x=column, hue="Transported", data=train_df, palette="Set2")
    plt.title(f"{column} Distribution")
    plt.tight_layout()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 In HomePlanet feature we can observe that most of passenger are from Earth but passenger from Earth are Comparatively Less Transported, passenger from Mars are Equally Transported, and passengers from Europa are Highly Transported.<br>
# 💡 In Destination feature we can observe that most of the passengers are transported to Trappist-1e.<br>
# 💡 In VIP feature we can observe that one cateogry is dominating other category too much. So it doesn't seem to be usefull feature because it can lead to overfitting in our model.<br>
# 💡 So it's better to drop VIP feature before Model building.<br>
# </div>

# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>NOTE</b><br>
# 💡 We have visualized all the features expect PassengerId, Name, Cabin features. We can't visualize this features because they are having high cardinality. <br>
# 💡 We will visualize this feature after creating new features from this old features.<br> </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Feature Engineering
# </p>
# </div>

# ### 1. Creating New Feature From "PassengerId" Feature.

# In[15]:


train_df["PassengerId"].head().to_frame()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>How will we do feature engineering on PassengerId</b><br>
# 
# 💡 We know that each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number of people within the group.<br>
# 
# 💡 So we can create a new feature Group_Size which will indicate total number of members present in each group.<br>
# 💡 We can create one more new feature Travelling Solo indicating whether the passenger is travelling solo or in a group.<br>
# </div>

# In[16]:


def passengerid_new_features(df):
    
    #Splitting Group and Member values from "PassengerId" column.
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Member"] =df["PassengerId"].apply(lambda x: x.split("_")[1])
    
    #Grouping the "Group" feature with respect to "member" feature to check which group is travelling with how many members
    x = df.groupby("Group")["Member"].count().sort_values()
    
    #Creating a set of group values which are travelling with more than 1 members.
    y = set(x[x>1].index)
    
    #Creating a new feature "Solo" which will indicate whether the person is travelling solo or not.
    df["Travelling_Solo"] = df["Group"].apply(lambda x: x not in y)
    
    #Creating a new feature "Group_size" which will indicate each group number of members.
    df["Group_Size"]=0
    for i in x.items():
        df.loc[df["Group"]==i[0],"Group_Size"]=i[1]


# In[17]:


passengerid_new_features(train_df)
passengerid_new_features(test_df)


# **We don't require Group & Member features any more so we will drop those feature from both datasets**

# In[18]:


train_df.drop(columns=["Group","Member"],inplace=True)
test_df.drop(columns=["Group","Member"],inplace=True)


# ### Visualizing "Group_Size" & "Travelling_Solo" Features.

# In[19]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.countplot(x="Group_Size", hue="Transported", data=train_df,palette="Set2")
plt.title("Group_Size vs Transported")

plt.subplot(1,2,2)
sns.countplot(x="Travelling_Solo", hue="Transported", data=train_df,palette="Set2")
plt.title("Travelling Solo vs Transported")
plt.tight_layout()
plt.show()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 From Group_Size feature we can observe that most the passengers are travelling alone.<br>
# 💡 From Travelling_Solo feature we can observe that passengers travelling solo are comparatively less transported when compared with passenger travelling in group.<br>
# </div>

# ## 2. Creating New Feature using "Cabin" Feature

# In[20]:


train_df["Cabin"].head().to_frame()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>How will we do feature engineering on Cabin</b><br>
# 
# 💡 We know that cabin feature consists of deck/num//side , where deck is deck loacation, num is deck_number and side can be P for port or S for Starboard.<br>
# 💡 We can separate all these 3 values from cabin & create three new features Cabin_Deck, Cabin_Number & Cabin_Side.<br>
# 💡 We also know that Cabin feature is having NaN values so to avoid error while splitting we have to replace it in such a way taht we can split those NaN Values in all three new features respectively.<br>
# </div>

# In[21]:


def cabin_new_feature(df):
    df["Cabin"].fillna("np.nan/np.nan/np.nan",inplace=True)  #In this way we can split NaN values into all three categories
    
    df["Cabin_Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0])
    df["Cabin_Number"]  = df["Cabin"].apply(lambda x: x.split("/")[1])
    df["Cabin_Side"] = df["Cabin"].apply(lambda x: x.split("/")[2])
    
    #Replacing string nan values to numpy nan values..
    cols = ["Cabin_Deck","Cabin_Number","Cabin_Side"]
    df[cols]=df[cols].replace("np.nan",np.nan)
    
    #Filling Missing Values in new features created.
    df["Cabin_Deck"].fillna(df["Cabin_Deck"].mode()[0],inplace=True)
    df["Cabin_Side"].fillna(df["Cabin_Side"].mode()[0],inplace=True)
    df["Cabin_Number"].fillna(df["Cabin_Number"].median(),inplace=True)


# In[22]:


cabin_new_feature(train_df)
cabin_new_feature(test_df)


# ### Visualizing "Cabin_Deck" & "Cabin_Side" Feature.

# In[23]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x="Cabin_Deck",hue="Transported", data=train_df, palette="Set2",order=["A","B","C","D","E","F","G","T"])
plt.title("Cabin_Deck Distribution")

plt.subplot(1,2,2)
sns.countplot(x="Cabin_Side", hue="Transported", data=train_df, palette="Set2")
plt.title("Cabin_Side Distribution")
plt.tight_layout()
plt.show()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Insights</b><br>
# 💡 From Cabin_Deck we can observe that most of the people are from F & G Deck.<br>
# 💡 There are very few passengers in Cabin_Deck ,T.<br>
# 💡 Passengers from Cabin Deck B & C are very highly transported. <br>
# 💡 From Cabin_Side we can observe that almost half passengers were from cabin side S and half from cabin side P.<br>
# 💡 But passenger from cabin_side S are Highly Transported but passengers from cabin_side P are Equally Transported<br>
# </div>

# ### Visualizing "Cabin_Number" Feature.

# In[24]:


train_df["Cabin_Number"]=train_df["Cabin_Number"].astype(int)
test_df["Cabin_Number"]=test_df["Cabin_Number"].astype(int)


# **Before visualizing let's do some Statistical analysis on Cabin_Number Feature**

# In[25]:


print("Total Unique values present in Cabin_Number feature is:",train_df["Cabin_Number"].nunique())
print("The Mean of Cabin_Number Feature is: ",train_df["Cabin_Number"].mean())
print("The Median of Cabin_Number Feature is:",train_df["Cabin_Number"].median())
print("The Minimum value of Cabin_Number feature is:",train_df["Cabin_Number"].min())
print("The Maximum value of Cabin_number Feature is:",train_df["Cabin_Number"].max())


# In[26]:


plt.figure(figsize=(15,5))
sns.histplot(x="Cabin_Number",data=train_df,hue="Transported",palette="Set2")
plt.title("Cabin_Number Distribution")
plt.xticks(list(range(0,1900,300)))
plt.vlines(300,ymin=0,ymax=550,color="black")
plt.vlines(600,ymin=0,ymax=550,color="black")
plt.vlines(900,ymin=0,ymax=550,color="black")
plt.vlines(1200,ymin=0,ymax=550,color="black")
plt.vlines(1500,ymin=0,ymax=550,color="black")
plt.show()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Insights</b><br>
# 💡 We can observe that Cabin_Number can be divided into different regions with group of 300 passenegrs.<br>
# 💡 So we can create a new features Cabin_Regions which will indicate passenger cabin number region.<br>
# </div>

# ## 3. Creating New Feature "Cabin_Regions" From "Cabin_Number".

# In[27]:


def cabin_regions(df):
    df["Cabin_Region1"] = (df["Cabin_Number"]<300)
    df["Cabin_Region2"] = (df["Cabin_Number"]>=300) & (df["Cabin_Number"]<600)
    df["Cabin_Region3"] = (df["Cabin_Number"]>=600) & (df["Cabin_Number"]<900)
    df["Cabin_Region4"] = (df["Cabin_Number"]>=900) & (df["Cabin_Number"]<1200)
    df["Cabin_Region5"] = (df["Cabin_Number"]>=1200) & (df["Cabin_Number"]<1500)
    df["Cabin_Region6"] = (df["Cabin_Number"]>=1500)


# In[28]:


cabin_regions(train_df)
cabin_regions(test_df)


# **We don't need Cabin_Number Feature anymore so we will drop this feature**

# In[29]:


train_df.drop(columns=["Cabin_Number"],inplace=True)
test_df.drop(columns=["Cabin_Number"],inplace=True)


# ### Visualizing "Cabin_Region" Feature.

# In[30]:


cols = ["Cabin_Region1","Cabin_Region2","Cabin_Region3","Cabin_Region4","Cabin_Region5","Cabin_Region6"]

plt.figure(figsize=(20,25))
for idx,value in enumerate(cols):
    plt.subplot(4,2,idx+1)
    sns.countplot(x=value, hue="Transported", data=train_df, palette="Set2")
    plt.title(f"{value} Distribution")
    plt.tight_layout()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that passengers from Cabin_Region1 are Highly Transported when compared with other cabin regions.<br>
# 💡 we can also observe that as the cabin region number is increasing passengers transport is decreasing.<br>
# </div>

# ## 4. Creating New Feature From "Age"

# In[31]:


train_df["Age"].head().to_frame()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>How we will do feature engineering on Age Feature</b><br>
# 💡 As we have done EDA on Age feature we collected some insights over there that the ages can be splitted into different groups based on Transported.<br>
# 💡 So we will create a new feature name Age Group and will split the Age into different groups on the basics of insights we gainedfrom EDA.<br>
# </div>

# In[32]:


def age_group(df):
    age_group  = []
    for i in df["Age"]:
        if i<=12:
            age_group.append("Age_0-12")
        elif (i>12 and i<=18):
            age_group.append("Age_0-18")
        elif (i>18 and i<=25):
            age_group.append("Age_19-25")
        elif (i>25 and i<=32):
            age_group.append("Age_26-32")
        elif (i>32 and i<=50):
            age_group.append("Age_33_50")
        elif (i>50):
            age_group.append("age_50+")
        else:
            age_group.append(np.nan)
        
    df["Age Group"] = age_group


# In[33]:


age_group(train_df)
age_group(test_df)


# ### Visualizing "Age Group" Feature.

# In[34]:


order = sorted(train_df["Age Group"].value_counts().keys().to_list())

plt.figure(figsize=(14,6))
sns.countplot(x="Age Group",hue="Transported", data=train_df, palette="Set2",order=order)
plt.title("Age Group Distribution");


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Insights</b><br>
# 💡 This new feature looks more relevent to our target data.<br>
# 💡 Age_0-12 & Age_0-18 are more likely to be transported compared to not transported.<br>
# 💡 Age_19-25 , Age_26_32 & Age_33_50 are less likely to be transported compared to not transported.<br>
# 💡 Age_50+ are almost equally transported compared to not transported.<br>
# </div>

# ## 5. Creating New Features Using All Expenditude Features.

# In[35]:


train_df[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]].head()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>How can we do feature engineering on all expenditure featrues</b><br>
# 
# 💡 When we have done EDA on this expenditure features we gained some insights as:-<br>
#   1. We can create a Total Expenditure Feature by combining all the expenditures.<br>
#   2. We can create a No Spending boolean feature from Total Expenditure feature indicating True for those passengers who have spent 0 expense.<br>
#   3. We can split Total Expenditure into different categories indicating whether the person is having no_expense, low_expense, medium_expense or high_expense and can create a new feature Expenditure Category.<br>
# </div>

# In[36]:


exp_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

def new_exp_features(df):
    df["Total Expenditure"] = df[exp_cols].sum(axis=1)
    df["No Spending"] = (df["Total Expenditure"]==0)


# In[37]:


new_exp_features(train_df)
new_exp_features(test_df)


# ### Visualizing "Total Expenditure" Feature.

# In[38]:


plt.figure(figsize=(15,6))
sns.histplot(x="Total Expenditure", hue="Transported", data=train_df, kde=True, palette="Set2",bins=200)
plt.ylim(0,200)
plt.xlim(0,10000)
plt.title("Total Expenditure Distribution");


# **Generating some statistical information from Total Expenditue feature**

# In[39]:


mean = round(train_df["Total Expenditure"].mean())
median = train_df["Total Expenditure"].median()

print("Mean value of Total Expenditure feature is = ",mean)
print("Median value of Total Expenditure feature is = ",median)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Insights</b><br>
# 💡 Using above measure of central tendency values we can split Total Expenditure Features into different expense categories.<br>
# 💡 If Total Expenditure is equal to 0 then No Expense category.<br>
# 💡 If Total Expenditure is between 1-716 then Low Expense category.<br>
# 💡 If Total Expenditure is between 717-1441 then Medium Expense category.<br>
# 💡 If Total Expenditure is greater thean 1441 then High Expense category.<br>
# </div>

# In[40]:


def expenditure_category(df):
    expense_category = []
    
    for i in df["Total Expenditure"]:
        if i==0:
            expense_category.append("No Expense")
        elif (i>0 and i<=716):
            expense_category.append("Low Expense")
        elif (i>716 and i<=1441):
            expense_category.append("Medium Expense")
        elif (i>1441):
            expense_category.append("High Expense")
    
    df["Expenditure Category"] = expense_category


# In[41]:


expenditure_category(train_df)
expenditure_category(test_df)


# ### Visualizing "No Spending" & "Expenditure Category" Features.

# In[42]:


cols = ["No Spending", "Expenditure Category"]

plt.figure(figsize=(18,6))
for idx,column in enumerate(cols):
    plt.subplot(1,2,idx+1)
    sns.countplot(x=column, hue="Transported", data=train_df, palette="Set2")
    plt.title(f"{column} Distribution")
    plt.tight_layout()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
#     
# 💡 In Total Expenditure feature we can observe that passengers having low total expenses are likely to be transported more.<br>
# 💡 In No Spending feature we can observe that passenger having No Spending are highly transported.<br>
# 💡 in Expenditure Category feature we can confirm than passenger having No Expense are highly transported .<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Data Pre-Processing
# </p>
# </div>

# ### 1. Checking Missing Values.

# In[44]:


z = train_df.isnull().sum()[train_df.isnull().sum()>0].to_frame().rename(columns={0:"No. of Missing values"})
z["% of Missing values"] = round(train_df.isnull().sum()[train_df.isnull().sum()>0]*100/len(train_df),2)
z


# ### 2. Visualizing Missing Numbers

# In[45]:


import missingno as msno


# 

# In[46]:


msno.bar(train_df,color="C1",fontsize=22)
plt.show()


# **Another way to visualize missing Values**

# In[47]:


plt.figure(figsize=(14,8))
sns.heatmap(train_df.isnull(),cmap="summer")
plt.show()


# ### 4. Handling Missing Values.

# In[48]:


cat_cols = train_df.select_dtypes(include=["object","bool"]).columns.tolist()
cat_cols.remove("Transported")
num_cols = train_df.select_dtypes(include=["int","float"]).columns.tolist()


# In[49]:


print("Categorical Columns:",cat_cols)
print("\n","-"*120)
print("\nNumerical Columns:",num_cols)


# **Using Simple Imputer Library to Fill Missing Values**

# In[50]:


imputer1 = SimpleImputer(strategy="most_frequent")     ##To fill Categorical Features.
imputer2 = SimpleImputer(strategy="median")            ##To fill numeircal features.


# In[51]:


def fill_missingno(df):
    df[cat_cols] = imputer1.fit_transform(df[cat_cols])
    df[num_cols] = imputer2.fit_transform(df[num_cols])


# In[52]:


fill_missingno(train_df)
fill_missingno(test_df)


# In[53]:


print("Missing numbers left in train_df is:",train_df.isnull().sum().sum())
print("Missing numbers left in test_df is:",test_df.isnull().sum().sum())


# ### 5. Checking Duplicacy in Data.

# In[54]:


print("Duplicate values in training data is: ",train_df.duplicated().sum())
print("Duplicate values in testing data is: ",test_df.duplicated().sum())


# ### 6. Checking Cardinality of Categorical Features.

# In[55]:


print("Cardinality of features in numerical data is: ")
print(train_df.select_dtypes(include=["object"]).nunique())
print("\n","-"*50)
print("\nCardinality of features in categorical data is: ")
print(test_df.select_dtypes(include=["object"]).nunique())


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We have done all feature engineering now we can drop features which have high cardinality.<br>
# 💡 So we can drop passengerId, Cabin , Name , Group and Surname features.<br>
# </div>

# **Dropping Categorical Features with High Cardinality**

# In[56]:


##Extracting passengerId from test data because qe need this for submitting our predictions on kaggle.
pass_df = test_df[["PassengerId"]]


# In[57]:


cols = ["PassengerId","Cabin","Name"]
train_df.drop(columns =cols, inplace=True)
test_df.drop(columns=cols, inplace=True)


# ### 7. Gathering Statistical Information of Numerical Features.

# In[58]:


train_df.describe().T


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Obervation</b><br>
# 💡 We can oberve in RoomService, FoodCourt, ShoppingMall, Spa & VRDeck more than 50 percentile of data are equal to 0.<br>
# 💡 And when we did EDA on this features all of them were having right skewed distribution<br>
# 💡 So we can simply say there is a presence of large amount of outliers in these features.<br>
# 💡 So we can tranform these features to normal distribution using Log Transformation.<br>
# 💡 Since, we are applying log transformation on these expenditure features so we have to apply transformation on Total Expenditure also.<br>
# 💡 So that the model can have better understanding while finding patterns.<br>
# </div>

# ## 8. Applying Log Transformation on Expenditure Features.

# In[59]:


cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Total Expenditure']

for value in cols:
    train_df[value] = np.log(1+train_df[value])
    test_df[value]=np.log(1+test_df[value])


# ### Visualizing these features after Transformation

# In[60]:


x=1

plt.figure(figsize=(20,35))
for i in cols:
    plt.subplot(6,2,x)
    sns.distplot(train_df[i],color="green")
    plt.ylim(0,0.2)
    plt.title(f"{i} Distribution")
    plt.tight_layout()
    x+=1


# ### 9. Checking Data - Types of Features.

# In[61]:


train_df.dtypes


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Note</b><br>
# 💡 CryoSleep , VIP, Travelling_Solo, No Spending, Cabin_Region1, Cabin_Region2, Cabin_Region3, Cabin_Region4, Cabin_Region5, Cabin_Region6 features contains boolean values so we have to change there data-type which will be benefical while encoding our categorical features.<br>
# </div>

# **Changing Data-Type to Boolean**

# In[62]:


cols = ["CryoSleep","VIP","Travelling_Solo","No Spending","Cabin_Region1","Cabin_Region2","Cabin_Region3","Cabin_Region4",
       "Cabin_Region5","Cabin_Region6"]

train_df[cols] = train_df[cols].astype(bool)
test_df[cols] = test_df[cols].astype(bool)


# ### 7. Feature Encoding

# * We will do **One Hot Encoding** for nominal categorical features.
# * We will do **LabelEncoding** for ordinal categorical features.

# In[63]:


nominal_cat_cols = ["HomePlanet","Destination"]
ordinal_cat_cols = ["CryoSleep","VIP","Travelling_Solo","Cabin_Deck","Cabin_Side","Cabin_Region1","Cabin_Region2",
                    "Cabin_Region3","Cabin_Region4","Cabin_Region5","Cabin_Region6","Age Group","No Spending",
                    "Expenditure Category"]


# **Label Encoding**

# In[64]:


enc = LabelEncoder()


# In[65]:


train_df[ordinal_cat_cols] = train_df[ordinal_cat_cols].apply(enc.fit_transform)
test_df[ordinal_cat_cols] = test_df[ordinal_cat_cols].apply(enc.fit_transform)


# **One Hot Encoding**

# In[66]:


train_df = pd.get_dummies(train_df,columns=nominal_cat_cols)
test_df = pd.get_dummies(test_df,columns=nominal_cat_cols)


# 
# **Note**
# * We still have one feature **Transported** left for encoding in training dataset.

# In[67]:


train_df["Transported"].replace({False:0,True:1},inplace=True)


# **Checking all features are encoded or not**

# In[68]:


train_df.head()


# In[69]:


test_df.head()


# ### 8. Selecting Features & Labels For Model Training.

# In[70]:


X = train_df.drop(columns=["Transported"])
y = train_df[["Transported"]]


# ### 9. Feature Scaling

# In[71]:


scaler = StandardScaler()


# In[72]:


X_scaled = scaler.fit_transform(X)
test_df_scaled = scaler.fit_transform(test_df)


# ### 10. Splitting Data For Model Which Don't Need Scaled Data.

# In[73]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[74]:


print(x_train.shape, y_train.shape)


# In[75]:


print(x_test.shape,y_test.shape)


# ### 11. Splitting Data For Model Which Need Scaled Data.

# In[76]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(X_scaled,y,test_size=0.2,random_state=0)


# In[77]:


print(x_train1.shape, y_train1.shape)


# In[78]:


print(x_test1.shape, y_test1.shape)


# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Model Building For Scaled Data
# </p>
# </div>

# In[79]:


training_score = []
testing_score = []


# In[80]:


def model_prediction(model):
    model.fit(x_train1,y_train1)
    x_train_pred1 = model.predict(x_train1)
    x_test_pred1 = model.predict(x_test1)
    a = accuracy_score(y_train1,x_train_pred1)*100
    b = accuracy_score(y_test1,x_test_pred1)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"Accuracy_Score of {model} model on Training Data is:",a)
    print(f"Accuracy_Score of {model} model on Testing Data is:",b)
    print("\n------------------------------------------------------------------------")
    print(f"Precision Score of {model} model is:",precision_score(y_test1,x_test_pred1))
    print(f"Recall Score of {model} model is:",recall_score(y_test1,x_test_pred1))
    print(f"F1 Score of {model} model is:",f1_score(y_test1,x_test_pred1))
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test1,x_test_pred1)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
    plt.show()


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>1. Logistic-Regression </b> Model</h2>

# In[81]:


model_prediction(LogisticRegression())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>2. KNeighborsClassifier </b> Model</h2>

# In[82]:


model_prediction(KNeighborsClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>3. Support-Vector-Classifier </b> Model</h2>

# In[83]:


model_prediction(SVC())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>4. Naive-Bayes </b> Model</h2>

# In[84]:


model_prediction(GaussianNB())


# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Model Building For Un-Scaled Data
# </p>
# </div>

# In[85]:


def model_prediction(model):
    model.fit(x_train,y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    a = accuracy_score(y_train,x_train_pred)*100
    b = accuracy_score(y_test,x_test_pred)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"Accuracy_Score of {model} model on Training Data is:",a)
    print(f"Accuracy_Score of {model} model on Testing Data is:",b)
    print("\n------------------------------------------------------------------------")
    print(f"Precision Score of {model} model is:",precision_score(y_test,x_test_pred))
    print(f"Recall Score of {model} model is:",recall_score(y_test,x_test_pred))
    print(f"F1 Score of {model} model is:",f1_score(y_test,x_test_pred))
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test,x_test_pred)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
    plt.show()


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>5. Decision-Tree-Classifier </b> Model</h2>

# In[86]:


model_prediction(DecisionTreeClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>6. Random-Forest-Classifier </b> Model</h2>

# In[87]:


model_prediction(RandomForestClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>7. Ada-Boost-Classifier </b> Model</h2>

# In[88]:


model_prediction(AdaBoostClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>8. Gradient-Boosting-Classifier </b> Model</h2>

# In[89]:


model_prediction(GradientBoostingClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>9. LGMB Classifier </b> Model</h2>

# In[90]:


model_prediction(LGBMClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>10. XGBClassifier </b> Model</h2>

# In[91]:


model_prediction(XGBClassifier())


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>11. Cat-Boost-Classifier </b> Model</h2>

# In[92]:


model_prediction(CatBoostClassifier(verbose=False))


# ***

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# All Model Performance Comparison
# </p>
# </div>

# In[93]:


models = ["Logistic Regression","KNN","SVM","Naive Bayes","Decision Tree","Random Forest","Ada Boost",
          "Gradient Boost","LGBM","XGBoost","CatBoost"]


# In[94]:


df = pd.DataFrame({"Algorithms":models,
                   "Training Score":training_score,
                   "Testing Score":testing_score})


# In[95]:


df


# ### Plotting above results using column-bar chart.

# In[96]:


df.plot(x="Algorithms",y=["Training Score","Testing Score"], figsize=(16,6),kind="bar",
        title="Performance Visualization of Different Models",colormap="Set1")
plt.show()


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 Highest performance was give by LGBM near to 82%.<br>
# 💡 But RandomForest,XgBoost, & catBoost Model performance was also good.<br>
# 💡 So we will do Hyper-Parameter Tunning on these four Models.<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Hyper-Parameter Tunning of LGBM Model
# </p>
# </div>

# In[97]:


model1 = LGBMClassifier()


# In[98]:


parameters1 = {"n_estimators":[100,300,500,600,650],
              "learning_rate":[0.01,0.02,0.03],
              "random_state":[0,42,48,50],
               "num_leaves":[16,17,18]}


# In[99]:


grid_search1 = GridSearchCV(model1, parameters1, cv=5, n_jobs=-1)


# In[100]:


grid_search1.fit(x_train,y_train.values.ravel())


# In[101]:


grid_search1.best_score_


# In[102]:


best_parameters1 = grid_search1.best_params_
best_parameters1


# ### Creating LGBM Model Using Best Parameters.

# In[103]:


model1 = LGBMClassifier(**best_parameters1)


# In[104]:


model1.fit(x_train,y_train)


# In[105]:


x_test_pred1 = model1.predict(x_test)


# In[106]:


accuracy_score(y_test,x_test_pred1)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 
# 💡 We can clearly observe that our LGBM Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Hyper-Parameter Tunning of CatBoost Model
# </p>
# </div>

# In[107]:


model2 = CatBoostClassifier(verbose=False)


# In[108]:


parameters2 = {"learning_rate":[0.1,0.3,0.5,0.6,0.7],
              "random_state":[0,42,48,50],
               "depth":[8,9,10],
               "iterations":[35,40,50]}


# In[109]:


grid_search2 = GridSearchCV(model2, parameters2, cv=5, n_jobs=-1)


# In[110]:


grid_search2.fit(x_train,y_train)


# In[111]:


grid_search2.best_score_


# In[112]:


best_parameters2 = grid_search2.best_params_
best_parameters2


# ### Creating Cat Boost Model Using Best Parameters

# In[113]:


model2 = CatBoostClassifier(**best_parameters2,verbose=False)


# In[114]:


model2.fit(x_train,y_train)


# In[115]:


x_test_pred2 = model2.predict(x_test)


# In[116]:


accuracy_score(y_test,x_test_pred2)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 
# 💡 We can clearly observe that our CatBoost Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Hyper-Parameter Tunning of XGBoost Model
# </p>
# </div>

# In[117]:


model3 = XGBClassifier()


# In[118]:


parameters3 = {"n_estimators":[50,100,150],
             "random_state":[0,42,50],
             "learning_rate":[0.1,0.3,0.5,1.0]}


# In[119]:


grid_search3 = GridSearchCV(model3, parameters3 , cv=5, n_jobs=-1)


# In[120]:


grid_search3.fit(x_train,y_train)


# In[121]:


grid_search3.best_score_


# In[122]:


best_parameters3 = grid_search3.best_params_
best_parameters3


# ### Creating XGBoost Model Using Best Parameters

# In[123]:


model3 = XGBClassifier(**best_parameters3)


# In[124]:


model3.fit(x_train,y_train)


# In[125]:


x_test_pred3 = model3.predict(x_test)


# In[126]:


accuracy_score(y_test,x_test_pred3)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 
# 💡 We can clearly observe that our XGBoost Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Hyper Parameter Tunning of RandomForest Model
# </p>
# </div>

# In[127]:


model4 = RandomForestClassifier()


# In[128]:


parameters4 = {'n_estimators': [100,300,500,550],
               'min_samples_split':[7,8,9],
               'max_depth': [10,11,12], 
               'min_samples_leaf':[4,5,6]}
    


# In[129]:


grid_search4 = GridSearchCV(model4, parameters4, cv=5, n_jobs=-1)


# In[130]:


grid_search4.fit(x_train,y_train.values.ravel())


# In[131]:


grid_search4.best_score_


# In[132]:


best_parameters4 = grid_search4.best_params_
best_parameters4


# ### Creating Random Forest Model Using Best Parameters

# In[133]:


model4 = RandomForestClassifier(**best_parameters4)


# In[134]:


model4.fit(x_train,y_train)


# In[135]:


x_test_pred4 = model4.predict(x_test)


# In[136]:


accuracy_score(y_test,x_test_pred4)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observations</b><br>
# 
# 💡 We can clearly observe that Random Forest Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Stacking Classifier Model
# </p>
# </div>

# In[137]:


stacking_model = StackingClassifier(estimators=[('LGBM', model1), 
                                                ('CAT Boost', model2),
                                                ("XGBoost", model3),
                                                ('RF', model4)])


# In[138]:


stacking_model.fit(x_train, y_train)


# ### 

# In[139]:


x_train_pred5 = stacking_model.predict(x_train)


# In[140]:


x_test_pred5 = stacking_model.predict(x_test)


# In[141]:


print("Stacking Model accuracy on Training Data is:",accuracy_score(y_train,x_train_pred5)*100)


# In[142]:


print("Stacking Model accuracy on Testing Data is:",accuracy_score(y_test,x_test_pred5)*100)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 💡 We can observe that our Stacking Model is having kind of Best Fitting<br>
# 💡 Stacking Model is not having any kind of over_fitting or under_fitting<br>
# 💡 So, we can use this Stacking Model to predict our test_data and then submit it on kaggle.<br>
# </div>

# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Predicting Test Data
# </p>
# </div>

# In[143]:


pred = stacking_model.predict(test_df)


# In[144]:


pred


# <a id="1.3"></a>
# <h2 style="font-family: Verdana; font-size: 22px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>Submission- </b> Data Format</h2>

# In[145]:


pass_df.head()


# In[146]:


pass_df["Transported"] = pred


# In[147]:


pass_df.head()


# In[148]:


pass_df["Transported"].replace({1:True,0:False},inplace=True)


# In[149]:


pass_df.head()


# In[150]:


pass_df.shape


# **Submission File**

# In[151]:


pass_df.to_csv("spaceship_prediction_project.csv",index=False)


# ---

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:black;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:white;
#           font-size:120%;
#           text-align:center;">
# Conclusion
# </p>
# </div>

# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Titanic Spaceship Project</b><br>
# 
# 💡 The main objective of this project was to predict whether the passengers will be transported to alternate dimensions or not using the independent features given.<br>
# 
# <b>Key-Points</b><br>
# 
# 💡 We were havinng very few usefull independent features in the dataset. <br>
# 💡 So I have done various feature engineering to create some new relevant features for better predictions.<br>
# 💡 The main objective of feature engineering was to avoid data loss.<br>
# 💡 I have used different classifiers machine learning techniques for predictions.<br>
# 💡 Then I  have compared all the preddictions given by different classifier models.<br>
# 💡 Then I have selected the best performing classifier modles.<br>
# 💡 The best performing Models were LGBM, CatBoost, XGboost & RandomForest<br>
# 💡 But this models were having overfiiting.<br>
# 💡 So to reduce overfitting from the model I have done Hyper-Parameter Tunning<br>
# 💡 Then I haved used Stacking Ensemble Technique to boost my predictions.<br>
# 💡 In stacking Model I have used all the Models created after Hyper-Parameter Tunning.<br>
# 💡 In the end I have used Stacking Model to predict our test data.<br>
# </div>

# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Thank You</b><br>
# <b>If you liked this notebook then do vote for me which help my movitvation</b><br>
# <b>If there's any imporvment that I can be done on this notebook just let me know in the comment section</b><br>
# </div>
