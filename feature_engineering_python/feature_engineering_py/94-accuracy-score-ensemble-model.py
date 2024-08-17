#!/usr/bin/env python
# coding: utf-8

# <h3 align="center" style="font-size: 35px; color: #800080; font-family: Georgia;">
#     <span style="color: #008080;"> Author:</span> 
#     <span style="color: black;">Ali Can PAYASLI V2</span>
# </h3>

# <div style="border-radius:10px;border:black solid;padding: 15px;background-color:white;font-size:110%;text-align:left">
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:17px">
# 
#    
# CONTENT:<br><br>
# 1. [Libraries](#1)<br>
# 2. [Data Load and General Overview](#2)<br>
# 3. [Mutual Information](#3)
# 4. [Eda](#4)
#     * [Transported](#5)
#     * [HomePlanet](#6)
#     * [CryoSleep](#7)
#     * [Destination](#8)
#     * [Age](#9)
#     * [VIP](#10)
# 5. [Feature Engineering and Missing Values](#11)
#     * [Transported](#12)
#     * [VIP](#13)
#     * [Spending](#14)
#     * [Name and PassengerId](#15)
#     * [CryoSleep](#16)
#     * [HomePlanet-Destination](#17)
#     * [Cabin](#18)
#     * [Convert Categorical](#19)
#     * [Remaining Missing Value with KNN](#20)
#     * [MEstimate Encoder](#21)
#     * [Outlier Samples and Normalize](#22)
# 6. [Models](#23)
#     * [Train-Test Split](#24)
#     * [KNN](#25)
#     * [Random Forest](#26)
#     * [LightGBM](#27)
#     * [XGB](#28)
#     * [GBM](#29)
# 7. [Model Selection-Ensemble Model](#30)
# </div>
# </div>

# <a id = "1"></a>
# # Libraries

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
get_ipython().system('pip install ycimpute')
from ycimpute.imputer import knnimput
from category_encoders import MEstimateEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "2"></a>
# # Data Load and General Overview

# In[2]:


#sns.reset_orig()


# In[3]:


train_data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")


# In[4]:


train_data.head()


# In[5]:


train_data.info()


# In[6]:


test_data.info()


# In[7]:


train_data.describe(include = "all").T


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **Dataset has 14 features and 8693 samples.Transported of these features is target(dependent) variable.**
# * **Dataset has object, numeric and bool features. Also it has missing values**

# <a id = "3"></a>
# # Mutual Information

# In[8]:


#Copy for protect original of data
df = train_data.copy()


# In[9]:


X = df.copy()
y = X.pop("Transported")


# In[10]:


#Temporary convert for operation of Mutual Information
for col_name in X.select_dtypes(["object"]):
    X[col_name], _ = X[col_name].factorize()


# In[11]:


#Temporary fill missing values for operation of Mutual Information
X = X.apply(lambda x: x.fillna(x.mean())).astype(int)


# In[12]:


features = X.dtypes == int


# In[13]:


def MakeMiScore(X,y,disc_features):
    mi_scores = mutual_info_classif(X,y, discrete_features=disc_features)
    mi_scores = pd.Series(mi_scores, name = "MI Scores", index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores


# In[14]:


print(MakeMiScore(X,y,features))


# In[15]:


def PlotScores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# In[16]:


PlotScores(MakeMiScore(X,y,features))


# <a id = "4"></a>
# # Eda

# In[17]:


train_data.head()


# <a id = "5"></a>
# ## Transported

# In[18]:


print(train_data.Transported.value_counts(normalize = True))
train_data.Transported.value_counts().plot.barh();


# <a id = "6"></a>
# ## HomePlanet

# In[19]:


train_data.HomePlanet.value_counts().plot.barh();


# In[20]:


print(train_data.groupby("HomePlanet")["Transported"].value_counts(normalize=True))
sns.barplot(x = "HomePlanet", y= "Transported", data = df);


# In[21]:


plt.figure(figsize=(10,3))
plt.subplot(121)
train_data["HomePlanet"].value_counts().plot.barh()
plt.subplot(122)
train_data["Destination"].value_counts().plot.barh()
plt.show()


# In[22]:


train_data.groupby("HomePlanet")["Destination"].value_counts(normalize=True)


# In[23]:


print(train_data.groupby("HomePlanet")["VIP"].value_counts(normalize=True))
sns.barplot(train_data, x ="HomePlanet",y = "VIP");


# In[24]:


sns.barplot(x = "HomePlanet", y= "Transported", data = train_data, hue = "Destination")


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **Ratio of Transported passengers and untransported passengers almost same.**
# * **Passengers whose Homeplanet is Earth more after Europa and Mars respectively.**
# * **Homeplanet is effective in Transported. Passengers in Europa are more transported others.**
# * **There aren't VIP passenger in Earth**

# <a id = "7"></a>
# ## CryoSleep

# In[25]:


print(train_data["CryoSleep"].value_counts(normalize = True))
train_data["CryoSleep"].value_counts().plot.barh();


# In[26]:


print(train_data.groupby("CryoSleep")["VIP"].value_counts(normalize = True))


# In[27]:


sns.barplot(x= "CryoSleep", y= "VIP", data = train_data);


# In[28]:


print(train_data.groupby("CryoSleep")["Transported"].value_counts(normalize = True))
sns.barplot(x = "CryoSleep", y= "Transported", data = train_data);


# In[29]:


sns.catplot(x = "CryoSleep",y= "Age", data = train_data)


# In[30]:


train_data.groupby("CryoSleep")[["Spa", "VRDeck", "ShoppingMall", "RoomService", "FoodCourt"]].aggregate(["min", "max"])


# In[31]:


train_data.groupby("CryoSleep")["HomePlanet"].value_counts(normalize = True)


# In[32]:


train_data.groupby("CryoSleep")["Destination"].value_counts(normalize = True)


# In[33]:


train_data.groupby("CryoSleep")[["HomePlanet", "Destination"]].value_counts().plot.barh();


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **35 percent of passengers are sleeping.**
# * **VIP and Age are not effective in CryoSleep but CryoSleep has higher positive corelation with transported.**
# * **Sleeping passengers didn't spend money therefore missing values can fill this situation.**
#     

# <a id = "8"></a>
# ## Destination

# In[34]:


train_data["Destination"].value_counts(normalize = True).plot.barh()


# In[35]:


sns.barplot(x = "Destination", y= "VIP", data = train_data)


# In[36]:


print(train_data.groupby("Destination")["VIP"].value_counts(normalize=True))


# In[37]:


print(train_data.groupby("Destination")["Transported"].value_counts(normalize = True))
sns.barplot(x = "Destination", y= "Transported", data=train_data)


# In[38]:


sns.barplot(x = "Destination", y= "Transported", hue = "CryoSleep",data=train_data)


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **Situation of VIP is not effective Destination**
# * **Those with destination 55 Cancri-e are relatively more likely to be reached than others. This reason for this situation 55 cancri-e passengers may have relatively high sleeping rate.**

# <a id = "9"></a>
# ## Age

# In[39]:


sns.displot(train_data["Age"])


# In[40]:


plt.figure(figsize = (10,10))
plt.subplot(321)
sns.scatterplot(x = "Age", y = "Spa", data = train_data) 
plt.subplot(322)
sns.scatterplot(x = "Age", y = "VRDeck", data = train_data)
plt.subplot(323)
sns.scatterplot(x = "Age", y = "ShoppingMall", data = train_data)
plt.subplot(324)
sns.scatterplot(x = "Age", y = "FoodCourt", data = train_data)
plt.subplot(325)
sns.scatterplot(x = "Age", y = "RoomService", data = train_data)


# In[41]:


print(train_data[train_data["Spa"]>0]["Age"].min())
print(train_data[train_data["VRDeck"]>0]["Age"].min())
print(train_data[train_data["FoodCourt"]>0]["Age"].min())
print(train_data[train_data["RoomService"]>0]["Age"].min())
print(train_data[train_data["ShoppingMall"]>0]["Age"].min())


# In[42]:


(sns
.FacetGrid(train_data, hue = "VIP", height=5,xlim=(0,100))
.map(sns.histplot, "Age")).add_legend()


# In[43]:


train_data.groupby("VIP")["Age"].agg(["min", "max"])


# In[44]:


(sns
.FacetGrid(train_data, hue = "HomePlanet", height=5,xlim=(0,100))
.map(sns.histplot, "Age")).add_legend()


# In[45]:


(sns
.FacetGrid(train_data, hue = "Destination", height=5,xlim=(0,100))
.map(sns.histplot, "Age")).add_legend()


# In[46]:


(sns
.FacetGrid(train_data, hue = "CryoSleep", height=5,xlim=(0,100))
.map(sns.histplot, "Age")).add_legend()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **Passengers under 13 years of age do not have any spend**
# * **There aren't any VIP passengers under 18 years age**

# <a id = "10"></a>
# ## VIP

# In[47]:


print(train_data.VIP.value_counts(normalize=True))
train_data.VIP.value_counts().plot.barh();


# In[48]:


print(train_data.groupby("VIP")["Transported"].value_counts(normalize = True))
sns.barplot(x = "VIP", y= "Transported", data = train_data)


# In[49]:


sns.barplot(x = "VIP", y= "Transported", hue = "CryoSleep",data = train_data);


# In[50]:


sns.catplot(x = "VIP", y = "Spa", data = train_data,height=3) 
plt.show()
sns.catplot(x = "VIP", y = "VRDeck", data = train_data,height=3)
plt.show()
sns.catplot(x = "VIP", y = "ShoppingMall", data = train_data,height=3)
plt.show()
sns.catplot(x = "VIP", y = "FoodCourt", data = train_data,height=3)
plt.show()
sns.catplot(x = "VIP", y = "RoomService", data = train_data,height=3)
plt.show()


# In[51]:


print(train_data.groupby("Destination")["VIP"].value_counts())
sns.barplot(x = "Destination", y= "VIP", data = train_data);


# In[52]:


print(train_data.groupby("HomePlanet")["VIP"].value_counts())
sns.barplot(x = "HomePlanet", y= "VIP", data = train_data);


# In[53]:


(sns
.FacetGrid(train_data, hue = "VIP", height=5,xlim=(0,100))
.map(sns.histplot, "Age")).add_legend()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=blue>Observe</font></h3>
# 
# * **Passengers of VIP percent 2 in all passengers.**
# * **Transported rate of not VIP passengers are higher**
# * **Higher spend for non-VIP travellers, excluding foodcourt**
# * **There aren't VIP passenger in Earth and There aren't any VIP passengers under 18 years age**

# <a id = "11"></a>
# # Feature Engineering and Missing Values

# <a id = "12"></a>
# ## Transported

# In[54]:


# First check missing values
train_data["Transported"].isnull().values.any()


# In[55]:


# Transported convert into 1-0 with check
print(train_data["Transported"].value_counts())
train_data["Transported"] = [1 if i == True else 0 for i in train_data["Transported"]]
print(train_data["Transported"].value_counts())


# In[56]:


# And we can merge train and test data for feature engineering and missing values
AllData = pd.concat([train_data, test_data], ignore_index=True)


# In[57]:


AllData


# <a id = "13"></a>
# ## VIP

# In[58]:


# There aren't VIP passenger in Earth and There aren't any VIP passengers under 18 years age.
# Some of the missing values can be filled according to this determination
AllData["VIP"].isnull().sum()


# In[59]:


AgeVIPIndex = AllData[(AllData["VIP"].isnull() == True) & (AllData["Age"]<18)][["VIP"]].index


# In[60]:


AllData["VIP"][AgeVIPIndex] = False


# In[61]:


EarthVIPIndex = AllData[(AllData["VIP"].isnull() == True) & (AllData["HomePlanet"]=="Earth")][["VIP"]].index


# In[62]:


AllData["VIP"][EarthVIPIndex] = False


# In[63]:


AllData["VIP"].isnull().sum()


# <a id = "14"></a>
# ## Spending

# In[64]:


# Passengers under 13 years of age  and sleeping passengers don't have any spend
#  Some of the missing values can be filled according to this determination
print(AllData["Spa"].isnull().sum())
print(AllData["VRDeck"].isnull().sum())
print(AllData["ShoppingMall"].isnull().sum())
print(AllData["FoodCourt"].isnull().sum())
print(AllData["RoomService"].isnull().sum())


# In[65]:


#Define func for filling of missing values
def FillSpend(dataset, feature):
    spend_index = dataset[
        (dataset[feature].isnull() == True) &
        ((dataset["Age"]<13) | (dataset["CryoSleep"] == True))
    ].index
    dataset[feature][spend_index] = 0
    


# In[66]:


spend_list = ["Spa", "RoomService", "FoodCourt", "VRDeck", "ShoppingMall"]
for i in spend_list:
    FillSpend(AllData, i)


# In[67]:


#check
print(AllData["Spa"].isnull().sum())
print(AllData["VRDeck"].isnull().sum())
print(AllData["ShoppingMall"].isnull().sum())
print(AllData["FoodCourt"].isnull().sum())
print(AllData["RoomService"].isnull().sum())


# <a id = "15"></a>
# ## Name and Passenger Id

# In[68]:


# We can split of name as name and surname after can remove name
# Also missing values of name can be filled group_ıd


# In[69]:


AllData["GroupId"] = AllData.PassengerId.str.split("_", expand = True)[0].astype(int)
AllData["GroupNumber"] = AllData.PassengerId.str.split("_", expand = True)[1].astype(int)


# In[70]:


AllData.drop(["PassengerId"], axis = 1, inplace = True)


# In[71]:


AllData["Surname"] = AllData["Name"].str.split(" ", expand = True)[1]


# In[72]:


AllData.drop(["Name"], axis = 1, inplace = True)


# In[73]:


# Those with group number greater than 1 have the same group id number. 
#Therefore, considering that they are the same family their surnames can be filled in according to the previous sample. 
SurnameIndex = AllData[(AllData["GroupNumber"]>1) & (AllData["Surname"].isnull() == True)].index


# In[74]:


for i in SurnameIndex:
    AllData["Surname"][i] = AllData["Surname"][i-1]


# <a id = "16"></a>
# ## CryoSleep

# In[75]:


AllData[AllData["CryoSleep"].isnull() == True]


# In[76]:


# Those who make any spending are not asleep.Therefore missing values can be filled this situation


# In[77]:


SleepIndex = AllData[
    (AllData["CryoSleep"].isnull() == True) & 
    (
        (AllData["Spa"]>0) | (AllData["RoomService"]>0)
        | (AllData["FoodCourt"]>0) | (AllData["VRDeck"]>0) | (AllData["ShoppingMall"]>0)
    )
].index


# In[78]:


AllData["CryoSleep"][SleepIndex] = False


# <a id = "17"></a>
# ## HomePlanet / Destination

# In[79]:


# Those with group number greater than 1 have the same group id number. 
#Therefore, considering that they are the same family their HomePlanet and Destination can be filled in according to the previous sample.


# In[80]:


HPIndex=AllData[(AllData["GroupNumber"]>1) & (AllData["HomePlanet"].isnull() == True)].index
DestIndex = AllData[(AllData["GroupNumber"]>1) & (AllData["Destination"].isnull() == True)].index


# In[81]:


for i in HPIndex:
    AllData["HomePlanet"][i] = AllData["HomePlanet"][i-1]
for i in DestIndex:
    AllData["Destination"][i] = AllData["Destination"][i-1]


# <a id = "18"></a>
# ## Cabin

# In[82]:


# We can split Cabin Deck and Side
AllData["CabinDeck"] = AllData.Cabin.str.split("/", expand = True)[0]
AllData["CabinSide"] = AllData.Cabin.str.split("/", expand = True)[2]


# In[83]:


AllData.drop(["Cabin"],axis = 1, inplace = True)


# In[84]:


# Those with group number greater than 1 have the same group id number. 
#Therefore, considering that they are the same family their CabinDeck and CabinSide can be filled in according to the previous sample.


# In[85]:


DecIndex=AllData[(AllData["GroupNumber"]>1) & (AllData["CabinDeck"].isnull() == True)].index
SideIndex = AllData[(AllData["GroupNumber"]>1) & (AllData["CabinSide"].isnull() == True)].index


# In[86]:


for i in DecIndex:
    AllData["CabinDeck"][i] = AllData["CabinDeck"][i-1]
for i in SideIndex:
    AllData["CabinSide"][i] = AllData["CabinSide"][i-1]


# <a id = "19"></a>
# ## Convert Categorical

# In[87]:


y = AllData.pop("Transported")


# In[88]:


AllData.info()


# In[89]:


#Label Encoder
EncFeatures = [AllData.select_dtypes(["object"]).columns]

lbe = OrdinalEncoder()
for feature in EncFeatures:
    AllData[feature]= lbe.fit_transform(AllData[feature])
    


# In[90]:


AllData


# <a id = "20"></a>
# ## Remaining Missing Value with KNN

# In[91]:


# Even if some missing samples have been filled in the above operations.
# there are still missing samples. For others, predictive value assignment can be used


# In[92]:


var_names = list(AllData)
n_df = np.array(AllData)
dff = knnimput.KNN(k = 4).complete(n_df)


# In[93]:


AllData = pd.DataFrame(dff, columns=var_names)


# In[94]:


#check
AllData.isnull().values.any()


# <a id = "21"></a>
# ## MEstimate Encoder

# In[95]:


# Some categorical variables are multicategorical, so it's appropriate to use MEstimator


# In[96]:


print(AllData["GroupId"].nunique())
print(AllData["GroupNumber"].nunique())
print(AllData["Surname"].nunique())
print(AllData["CabinDeck"].nunique())


# In[97]:


encoders = MEstimateEncoder(cols = ["GroupId", "GroupNumber", "Surname", "CabinDeck"], m=4.0)
encoders.fit(AllData, y)


# In[98]:


AllData = encoders.transform(AllData)


# In[99]:


print(AllData["GroupId"].nunique())
print(AllData["GroupNumber"].nunique())
print(AllData["Surname"].nunique())
print(AllData["CabinDeck"].nunique())


# <a id = "22"></a>
# ## Outlier Samples and Normalize

# In[100]:


#The variance of variables outlier samples will be selected with LocalOutlierFactor


# In[101]:


clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(AllData)


# In[102]:


clf_scores = clf.negative_outlier_factor_


# In[103]:


np.sort(clf_scores)[0:40]


# In[104]:


thresold_value = np.sort(clf_scores)[18]


# In[105]:


outlier_samples = AllData[clf_scores<thresold_value].to_records(index = False)


# In[106]:


outlier_samples[:]= AllData[clf_scores == thresold_value].to_records(index = False)


# In[107]:


AllData[clf_scores<thresold_value] = pd.DataFrame(outlier_samples, index = AllData[clf_scores<thresold_value].index)


# In[108]:


#check
AllData[clf_scores<thresold_value]


# In[109]:


#Normalize
AllData = AllData.apply(lambda x: (x-np.min(x)) / (np.max(x)-np.min(x)))


# <a id = "23"></a>
# # Models

# <a id = "24"></a>
# ## Train-Test Split

# In[110]:


# The dataset will be restored to premerger state
# It will then be split into train and test for validation


# In[111]:


train_data = AllData.iloc[:8693]


# In[112]:


y.dropna(inplace = True)
train_data["Transported"] = y


# In[113]:


test_data = AllData.iloc[8693:]


# In[114]:


y = train_data.pop("Transported")


# In[115]:


X = train_data


# In[116]:


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=7)


# In[117]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[118]:


print(y_train.value_counts(normalize = True))
print(y_test.value_counts(normalize = True))


# In[119]:


test_data.index = np.arange(0,4277)


# <a id = "25"></a>
# ## KNN

# In[120]:


knn = KNeighborsClassifier()


# In[121]:


knn.fit(X_train, y_train)


# In[122]:


#knn_params = {
    #"n_neighbors": range(1,51)
    #"weights": ["uniform", "distance"]
    #"leaf_size":range(1,51)
#}


# In[123]:


#knn_cv = GridSearchCV(knn, knn_params, cv = 5, n_jobs=-1, verbose =2)
#knn_cv.fit(X_train, y_train)


# In[124]:


#knn_cv.best_params_


# In[125]:


knn_tuned= KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    leaf_size=1
)
knn_tuned.fit(X_train, y_train)


# In[126]:


print("KNN CV Accuracy Score: %.4f"% cross_val_score(knn_tuned, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# <a id = "26"></a>
# ## Random Forest

# In[127]:


rf = RandomForestClassifier(random_state=1)


# In[128]:


rf.fit(X_train, y_train)


# In[129]:


#rf_params = {
    #"n_estimators": [10,20,50,100,200,500,1000]
    #"max_depth": range(1,41)
    #"max_features": range(1,21)
    #"min_samples_split": range(1,11)
#}


# In[130]:


#rf_cv = GridSearchCV(rf, rf_params, cv = 5, n_jobs=-1, verbose = 2)
#rf_cv.fit(X_train, y_train)


# In[131]:


#rf_cv.best_params_


# In[132]:


rf_tuned = RandomForestClassifier(
    random_state=1,
    n_estimators=100,
    max_depth=18,
    max_features=13,
    min_samples_split=3
)
rf_tuned.fit(X_train, y_train)


# In[133]:


print("RF CV Accuracy Score: %.4f"% cross_val_score(rf_tuned, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# <a id = "27"></a>
# ## LightGBM

# In[134]:


lgb = LGBMClassifier(random_state=1,
                    force_col_wise=True)


# In[135]:


lgb.fit(X_train, y_train)


# In[136]:


#lgb_params = {
    #"n_estimators": [20,50,100,200,500]
    #"subsample":np.arange(0,1,0.01)
    #"learning_rate": np.arange(0,1,0.01)
    #"max_depth": range(1,51)
    #"min_child_samples": range(1,20)
    #"num_leaves": range(1,31)
#}


# In[137]:


#lgb_cv = GridSearchCV(lgb, lgb_params, cv = 5, n_jobs=-1, verbose=2)
#lgb_cv.fit(X_train, y_train)


# In[138]:


#lgb_cv.best_params_


# In[139]:


lgb_tuned = LGBMClassifier(random_state=1,
                           n_estimators=50,
                           subsample=0.01,
                           learning_rate=0.05,
                           max_depth = 6,
                           min_child_samples=2,
                           num_leaves = 20,                      
                       
                            force_col_wise=True)
lgb_tuned.fit(X_train, y_train)


# In[140]:


print("LGB CV Accuracy Score: %.4f"% cross_val_score(lgb_tuned, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# <a id = "28"></a>
# ## XGB

# In[141]:


xgb = XGBClassifier(random_state = 1)


# In[142]:


xgb.fit(X_train, y_train)


# In[143]:


#xgb_params = {
    #"n_estimators": [20,50,100,200,500]
    #"max_depth": range(1,51)
    #"subsample": [0.1,0.01,0.2,0.5,0.6,0.8]
    #"learning_rate": [0.1,0.01,0.2,0.02,0.5]
#}


# In[144]:


#xgb_cv = GridSearchCV(xgb, xgb_params, cv = 5, n_jobs=-1, verbose=2)
#xgb_cv.fit(X_train, y_train)


# In[145]:


#xgb_cv.best_params_


# In[146]:


xgb_tuned = XGBClassifier(n_estimators = 20,
                          max_depth = 2,
                          subsample = 0.5,
                          learning_rate = 0.1,                          
                         random_state = 1)
xgb_tuned.fit(X_train, y_train)


# In[147]:


print("XGB CV Accuracy Score: %.4f"% cross_val_score(xgb_tuned, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# <a id = "29"></a>
# ## GBM

# In[148]:


gb = GradientBoostingClassifier(random_state=1)


# In[149]:


gb.fit(X_train, y_train)


# In[150]:


#gb_params = {
        #"n_estimators": [20,50,100,200,500]
       #"learning_rate": [0.1,0.01,0.001,0.2,0.02,0.3,0.5,0.7]
       #"max_depth": range(1,11)
        #"min_samples_split": range(1,11)
    
#}


# In[151]:


#gb_cv = GridSearchCV(gb, gb_params, cv = 5, n_jobs=-1, verbose=2)
#gb_cv.fit(X_train, y_train)


# In[152]:


#gb_cv.best_params_


# In[153]:


gb_tuned = GradientBoostingClassifier(n_estimators=200,
                                      learning_rate=0.2,
                                      max_depth=5,
                                      min_samples_split=3,
                                     random_state=1)
gb_tuned.fit(X_train, y_train)


# In[154]:


print("GB CV Accuracy Score: %.4f"% cross_val_score(gb_tuned, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# <a id = "30"></a>
# # Model Selection (Ensemble Model)

# In[155]:


# We obtained the highest scores in GB,LGB and RF. 
# Since the scores of these models are close to each other, they can be used as ensembles.


# In[156]:


VotingC = VotingClassifier(
    estimators=[("gb",gb_tuned),("lgb",lgb_tuned), ("rf", rf_tuned)],voting="soft", n_jobs=-1
)
VotingC.fit(X_train, y_train)


# In[157]:


print("Voting CV Accuracy Score: %.4f"% cross_val_score(VotingC, X_test, y_test,cv = 5, scoring = "accuracy").mean())


# In[158]:


y_pred = VotingC.predict(X_test)


# In[159]:


disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()

