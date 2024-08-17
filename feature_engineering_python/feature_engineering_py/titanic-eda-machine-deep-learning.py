#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a>
# # <div style="text-align:center; padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>🚢titanic data analysis 📈& Machine & Deep Learning 💡</b></div>

# ![](https://static.timesofisrael.com/atlantajewishtimes/uploads/2022/03/DT6RD9.jpg)

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖Table of Contents</b></div>

# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# * <b> Introduction</b> <br>
#     * <b><mark>0.</mark></b>-------Import Libraries<br>
#     * <b><mark>1.</mark></b>-------Visuallization<br>
#      - <b><mark>1.1</mark></b>------Check Missing Value<br>
#      - <b><mark>1.2</mark></b>------Make Graph - Which is show "feature"- Dead<br>
#      -<b><mark>1.3.1</mark></b>---Name<br>
#      - <b><mark>1.3.2</mark></b>---Sex<br>
#      - <b><mark>1.3.3</mark></b>---Embarked<br>
#      - <b><mark>1.3.4</mark></b>---Fare<br>
#      - <b><mark>1.3.5</mark></b>---Cabin<br>
#      - <b><mark>1.3.6</mark></b>---Family Size<br>
#     * <b><mark>2.</mark></b>------Machine Learning<br>
#      - <b><mark>2.1</mark></b>-----KNN<br>
#      - <b><mark>2.2</mark></b>-----Decision Tree<br>
#      - <b><mark>2.3</mark></b>-----Random Forrest<br>
#      - <b><mark>2.4</mark></b>-----Naive Bayes<br>
#      - <b><mark>2.5</mark></b>-----SVM<br>
#      - <b><mark>2.6</mark></b>-----Machine Learning Result<br>
#     * <b><mark>3.</mark></b>-------Submission

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖Introduction</b></div>

# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# <b> In this notebook, I will do data analysis, EDA and Machine Learning.<br>
# I use Python language in this notebook.<br>
#     <br>In Data Analysis, I will get the most valuable result from each column. For doing this, I will process Data through variable ways. You can check how doing it in this notebook
#     <br>
#     <br>
#     And I'll use kinds of Machine Learning.Based on transformated Data, get KNN, Decision Tree, random forrest, Naive Bayes, SVM accuracy score.
#     compare its scores and submit the highest score among it.
#     <br>
#     <br>
#     And using tensorflow, predict result by deep learning model.

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖Columns</b></div>   

# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# * <b> <mark>Survival</mark></b> - Survival   (0 = No , 1 = Yes) <br>
# * <b> <mark>pclass</mark></b> - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)<br>
# * <b> <mark>sex</mark></b> - sex <br>
# * <b> <mark>Age</mark></b> - Age in years <br>
# * <b> <mark>sibsp</mark></b> - # of siblings / spouses aboard the Titanic <br>
# * <b> <mark>parch</mark></b> - # of parents / children aboard the Titanic<br>
# * <b> <mark>ticket</mark></b> - Ticket number<br>
# * <b> <mark>fare</mark></b>- Passenger fare<br>
# * <b> <mark>cabin</mark></b> - Cabin number<br>
# * <b> <mark>embarked</mark></b> - Port of Embarkation  (C = Cherbourg, Q = Queenstown, S = Southampton)<br>

# <!-- # <div style="text-align: left; font-family:timeroman;"> | Import Libraries </div> -->
# <!-- # <div style="text-align: left; font-family:timeroman;"> <b style='color:blue'>Import</b> Libraries </div> -->
# <!-- > <h1 style="text-align: left; font-family:timeroman;"> <b style='color:blue'>Import</b> Libraries </h1> -->
# <!-- # <b style='color:blue'> 0. Import Libraries </b> -->
# <!-- # <div style=' font-family:timeroman;'> 0. Import Libraries and Data </b> -->

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖0. Import Libraries</b></div>   

# In[1]:


#data
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


# In[2]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()


# In[3]:


test_df.head()


# <b> column explanation

# In[4]:


# survived = 0 - d , 1 - alive
# pclass = Ticket class
# sibsp = Wife, uncle, etc. on Titanic. If it's zero, you'll be alone.
# parch = parent + childer
# embarked  = port of Embarktion


# In[5]:


print(train_df.shape)
print(test_df.shape)


# In[6]:


print("train= ",train_df.info())


# In[7]:


print("test= ",test_df.info())


# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖1. Visuallization </b></div>   

# <a id="1"></a>
# # <div style="padding:10px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>1.1 Check missing value</b></div>

# In[8]:


print(train_df.isnull().sum())
msno.matrix(train_df).set_title("Train set",fontsize=20)


# In[9]:


print(test_df.isnull().sum())
msno.matrix(test_df).set_title("Train set",fontsize=20)


# <a id="1"></a>
# # <div style="padding:10px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>1.2 Make Graph - Which is show "feature"- Dead. </b></div>

# In[10]:


def bar_chart(feature):
    survived = train_df[train_df["Survived"]==1][feature].value_counts()
    dead = train_df[train_df["Survived"]==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ["Survived","Dead"]
    df.plot(kind="bar",stacked=True, figsize=(20,7),title=feature,fontsize=20)
for a in ["Sex","Pclass","SibSp"]:
    bar_chart(a)


# <a id="1"></a>
# # <div style="padding:10px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>1.3 Feature Engineering</b></div>  
# 
# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# <b>the process of turning data measurements into feature bacters
# From a person's point of view, text is easy to understand, but computers are much easier to change to numbers, so it means the process of changing to numbers.

# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.1  </b> Name </h2>

# In[11]:


train_df.head(10)
#The name doesn't reflect much on the result, but Mr. and Mrs in the name imply information about whether or not they are married, so only extract it.


# Remove names other than Mr or Mrs etc.

# In[12]:


train_test_df = [train_df,test_df]

for dataset in train_test_df:
    dataset["Title"] = dataset["Name"].str.extract('([A-Za-z]+)\.',expand=False)


# In[13]:


train_df["Title"]


# In[14]:


train_df["Title"].value_counts()


# In[15]:


title = train_df["Title"].value_counts()
plt.figure(figsize =(20,5))
sns.barplot(x = title.index, y = title.values)


# In[16]:


# use another library to more various visualization
import plotly.express as px
px.bar(x = title.index, y = title.values)


# In[17]:


title_mapping = {
    'Mr': 0, "Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Major":3,"Mile":3,"Countess":3,"Ms":3,"Lady":3,"Johnkheer":3,"Don":3,"Dona":3,"Mme":3,"Capt":3,"Sir":3
}

for dataset in train_test_df:
    dataset["Title"] = dataset["Title"].map(title_mapping)


# In[18]:


train_df.head()


# In[19]:


bar_chart("Title")


# In[20]:


train_df


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.2  </b> Sex </h2>

# In[21]:


sex_mapping = {"male":0,"female":1}
for dataset in train_test_df:
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)


# In[22]:


bar_chart("Sex")


# <b>There are several NaN values for age. I can substitute those values for the total average age. But since we've divided them into categories by sex from Name, we substitute them to obtain the average age of a man and woman, if the gender of a person with NaN value is male, and if it's female, we substitute the average age of a man's average age.

# In[23]:


# use median get the average age
train_df["Age"].fillna(train_df.groupby('Title')['Age'].transform("median"),inplace=True)
test_df["Age"].fillna(test_df.groupby('Title')['Age'].transform("median"),inplace=True)


# In[24]:


# visualization death - age
facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.show()


# <b>Both the most dead age group and the most surviving age group are in their 20s and mid-30s. People in their 0s to 10s and late 30s can see that there are more dead people.

# In[25]:


# visualization death - age, age between 0~20 years old
facet = sns.FacetGrid(train_df, hue="Survived", aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train_df["Age"].max())) # to use xlim, limitation age's range
plt.xlim(0,20) # to limitation X boundary, we can see the graph more specifically


# In[26]:


# visualization death - age, age between 0~20 years old
facet = sns.FacetGrid(train_df, hue="Survived", aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train_df["Age"].max())) # to use xlim, limitation age's range
plt.xlim(20,30)


# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# <b>1.3.2.1 binning:</b><br>
# Technology that weave into categories if it don't give much information.
# I will binning the Age by age. following code is the way how I did it.</div>

# In[27]:


train_df


# In[28]:


df_2 = train_df["Age"]
df_2 = pd.DataFrame(df_2)
df_2.columns=["Age"]
for i in range(891):
    a = df_2['Age'].get(i)
    if int(a) <= 16:
        df_2['Age'][i] = 0
    elif int(a) > 16 and a <=26:
        df_2['Age'][i] = 1
    elif int(a) > 26 and a <= 36:
        df_2['Age'][i] = 2
    elif int(a) >36 and a <= 64:
        df_2["Age"][i] = 3
    else:
        df_2['Age'][i] = 4
        
train_df["Age"] = df_2["Age"]
train_df


# <b>I devided "age" into 5 parts.<br>
# <br>  0-16 years old - 1
# <br>  16-26 years old - 2
# <br>  26-36 years old - 3
# <br>  36-64 years old - 4
# <br>  64- years old - 5

# In[29]:


test_df


# In[30]:


df_2 = test_df["Age"]
df_2 = pd.DataFrame(df_2)
df_2.columns=["Age"]
for i in range(418):
    a = df_2['Age'].get(i)
    if int(a) <= 16:
        df_2['Age'][i] = 0
    elif int(a) > 16 and a <=26:
        df_2['Age'][i] = 1
    elif int(a) > 26 and a <= 36:
        df_2['Age'][i] = 2
    elif int(a) >36 and a < 64:
        df_2["Age"][i] = 3
    else:
        df_2['Age'][i] = 4
test_df["Age"] = df_2["Age"]
test_df


# In[31]:


bar_chart("Age")


# 
# <b>Through the Bar graph, It was found that the age group of 26 to 36 years old was the highest among the dead.

# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.3  </b> Embarked </h2>
# 

# In[32]:


train_df


# In[33]:


pclass1 = train_df[train_df["Pclass"]==1]["Embarked"].value_counts() # the place where a first-class person got off
pclass2 = train_df[train_df["Pclass"]==2]["Embarked"].value_counts() # the place where a second-class person got off
pclass3 = train_df[train_df["Pclass"]==3]["Embarked"].value_counts() # the place where a third-class person got off
df = pd.DataFrame([pclass1,pclass2,pclass3])
df.index = ["1st class","2nd class","3rd class"]
df.plot(kind="bar",stacked = True, figsize=(20,8))


# In[34]:


for dataset in train_test_df:
    dataset["Embarked"] = dataset['Embarked'].fillna("S")
# fill "S" in blanked "Embarked" row
train_df.head()


# In[35]:


embarked_mapping = {"S":0,"C":1,"Q":2} # do mapping(test - number) in embarked row, for more convinence in machine learning
for dataset in train_test_df:
    dataset["Embarked"] = dataset["Embarked"].map(embarked_mapping)


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.4</b>  Fare</h2>
# 

# In[36]:


train_df["Fare"].fillna(train_df.groupby("Pclass")["Fare"].transform("median"),inplace=True)
test_df["Fare"].fillna(test_df.groupby("Pclass")["Fare"].transform("median"),inplace=True)


# In[37]:


facet = sns.FacetGrid(train_df,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Fare",shade=True)
facet.set(xlim=(0,train_df["Fare"].max()))
facet.add_legend()


# In[38]:


facet = sns.FacetGrid(train_df,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Fare",shade=True)
facet.set(xlim=(0,100))
facet.add_legend()


# In[39]:


facet = sns.FacetGrid(train_df,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Fare",shade=True)
facet.set(xlim=(100,200))
facet.add_legend()


# In[40]:


df_2 = train_df["Fare"]
df_2 = pd.DataFrame(df_2)
df_2.columns=["Fare"]
for i in range(891):
    a = df_2['Fare'].get(i)
    if int(a) <= 8:
        df_2['Fare'][i] = 0
    elif int(a) > 8 and a <=17:
        df_2['Fare'][i] = 1
    elif int(a) > 17 and a <= 27:
        df_2['Fare'][i] = 2
    elif int(a) >27 and a < 37:
        df_2["Fare"][i] = 3
    else:
        df_2['Fare'][i] = 4
train_df["Fare"] = df_2["Fare"]
train_df.head()


# In[41]:


df_2 = test_df["Fare"]
df_2 = pd.DataFrame(df_2)
df_2.columns=["Fare"]
for i in range(418):
    a = df_2['Fare'].get(i)
    if int(a) <= 8:
        df_2['Fare'][i] = 0
    elif int(a) > 8 and a <=17:
        df_2['Fare'][i] = 1
    elif int(a) > 17 and a <= 27:
        df_2['Fare'][i] = 2
    elif int(a) >27 and a < 37:
        df_2["Fare"][i] = 3
    else:
        df_2['Fare'][i] = 4
test_df["Fare"] = df_2["Fare"]
test_df.head()


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.5</b>  Cabin</h2>
# 

# In[42]:


for dataset in train_test_df:
    dataset["Cabin"] = dataset["Cabin"].str[:1]

pclass1 = train_df[train_df["Pclass"]==1]["Cabin"].value_counts() #1등급인 사람들의 객실수
pclass2 = train_df[train_df["Pclass"]==2]["Cabin"].value_counts() #2등급인 사람들의 객실수
pclass3 = train_df[train_df["Pclass"]==3]["Cabin"].value_counts() #3등급인 사람들의 객실수
print(pclass1)

df = pd.DataFrame([pclass1,pclass2,pclass3])
df.index = ["1st class","2nd class","3rd class"]
df.plot(kind="bar",stacked = True, figsize=(20,5))


# In[43]:


cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8} # do mapping, 
# If the range of numbers is not similar, the larger range can be considered more important.
# So I'm going to divide it up to a decimal place and give it a similar range.
# it called feature scaling.
for dataset in train_test_df:
    dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)


# In[44]:


train_df["Cabin"].fillna(train_df.groupby("Pclass")["Cabin"].transform("median"),inplace=True)
test_df["Cabin"].fillna(test_df.groupby("Pclass")["Cabin"].transform("median"),inplace=True)


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>1.3.6</b>  Family Size</h2>

# In[45]:


train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df['SibSp'] + test_df["Parch"] + 1


# In[46]:


facet = sns.FacetGrid(train_df,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"FamilySize",shade=True)
facet.set(xlim=(0,train_df["FamilySize"].max()))
facet.add_legend()
plt.xlim(-3)


# In[47]:


family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6 , 6:2 , 7:2.4 , 8:2.8, 9:3.2 , 10:3.6 , 11:4}
for dataset in train_test_df:
    dataset["FamilySize"] = dataset["FamilySize"].map(family_mapping)


# In[48]:


train_df.head()


# In[49]:


test_df.head()


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>Set the training set to match the test set</b>
# 

# In[50]:


drop_list = ["Ticket","SibSp","Parch","Name"]
train_df = train_df.drop(drop_list, axis=1)
test_df = test_df.drop(drop_list, axis=1)
train_df = train_df.drop(["PassengerId"], axis=1)


# In[51]:


train_data = train_df.drop("Survived", axis = 1)
target = train_df["Survived"]

train_data.fillna(0)
pd.DataFrame(train_data.isnull().value_counts())
a = pd.DataFrame(train_data["Title"]).fillna(0)
pd.DataFrame(train_data["Title"]).isnull().value_counts()
a.isnull().value_counts()
train_data["Title"] = a["Title"]
pd.DataFrame(train_data["Title"]).isnull().value_counts()


# In[52]:


train_df.head()


# In[53]:


test_df.head()


# In[54]:


test_data = test_df.drop("PassengerId", axis = 1).copy()


# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖2. Machine Learning</b></div>

# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>Preprocess - scaling</b>
# 

# In[55]:


# import Scaling Model
from sklearn.preprocessing import StandardScaler #All features have a normal distribution of 0 mean and 1 variance.
from sklearn.preprocessing import MinMaxScaler #Makes all features have data values between 0 and 1.
from sklearn.preprocessing import MaxAbsScaler # Makes the absolute value of all features lie between 0 and 1.
from sklearn.preprocessing import RobustScaler
#RobustScaler is similar to StandardScaler.
# However, StandardScaler uses mean and variance, while RobustScaler uses median and quartile.
from sklearn.preprocessing import Normalizer
#The previous four methods use statistics for each feature.

#import Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# import tuing model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


k_fold = KFold(n_splits = 9, shuffle=True, random_state = 0)


# In[56]:


# Scaling Model
ssc = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rsc = RobustScaler()

scl_list = [ssc,mms,mas,rsc]

# Machine Learning Model
knn = KNeighborsClassifier(n_neighbors = 13) #KNN
dtc = DecisionTreeClassifier() # Decision Tree
rfc = RandomForestClassifier(n_estimators=13) #Random Forest
gnb = GaussianNB() # Naive Bayes
svc = SVC() #SVC
gbc = GradientBoostingClassifier()
clf_list = [knn,dtc,rfc,gnb,svc,gbc]

score_list = []
score_name = []

for scl in scl_list:# at first, scale data
    scl.fit(train_data)
    train_data = scl.transform(train_data)
    test_data = scl.transform(test_data)
    for clf in clf_list: # based on scaled data, make machine learning model
        scoring = "accuracy"
        score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring = scoring)
        score_name.append(str(scl)+":"+str(clf))
        score = round(np.mean(score)*100,2)
        score_list.append(score)        


# <a id="1.2"></a>
# # <h2 style="font-family: Serif; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 1px; color: #3162C7; background-color: #DAD9FF;"><b>Check each type of <mark>Scaling method</mark> and <mark>Machine learning</mark> model's Score</b>
# 

# In[57]:


plt.figure(figsize=(20,8))
x = score_name
y = score_list
for i in range(len(x)):
    height = y[i]
    plt.text(x[i], height + 0.25, '%.1f' %height, ha='center', va='bottom', size = 12)
plt.bar(x, y,color='#e35f62')
plt.ylim(60,)


# ### Check out what types of method get the score higher than <mark>80</mark>?

# In[58]:


# make a new dataframe which is filled with model type and its score
ma_result = pd.DataFrame()
ma_result.insert(0,"type",x)
ma_result.insert(0,"score",y)
ma_res_score = []
ma_res_name = []

# seperate score higher than 80.
for i in range(len(ma_result[ma_result["score"] >= 80].value_counts().index)):
    ma_res_score.append(ma_result[ma_result["score"] >= 80].value_counts().index[i][0])
    ma_res_name.append(ma_result[ma_result["score"] >= 80].value_counts().index[i][1])
print(ma_result[ma_result["score"] >= 80].value_counts())
x = ma_res_name
y = ma_res_score

# visualization
plt.figure(figsize=(20,8))
for i in range(len(x)):
    height = y[i]
    plt.text(x[i], height, '%.1f' %height, ha='center', va='bottom', size = 12)

plt.bar(ma_res_name,ma_res_score,color='#e35f62')
plt.ylim(80,83)


# <div style="font-family:Serif;background-color:#DAD9FF; padding:30px; font-size:17px">
# 
# <b>The score of the machine learning model <mark>SVC</mark>, which was scaled using the <mark>Robust scaling model</mark>, recorded the highest score with <mark>82.7</mark>. <br> So I choose this way in machine learning.</b>

# In[59]:


train_data = train_df.drop("Survived", axis = 1)
target = train_df["Survived"]

train_data.fillna(0)
pd.DataFrame(train_data.isnull().value_counts())
a = pd.DataFrame(train_data["Title"]).fillna(0)
a.isnull().value_counts()
train_data["Title"] = a["Title"]
train_data.head()


# In[60]:


scl = RobustScaler()
scl.fit(train_data)
scl.transform(train_data)
scl.transform(test_data)

clf = SVC()
clf.fit(train_data, target)
# drop unnecessary column
test_data = test_df.drop("PassengerId", axis = 1).copy()
prediction = clf.predict(test_data)
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring = scoring)
score = round(np.mean(score)*100,2)
print("score: ",score)
print("prediction: ",prediction)


# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖3. Use Deep learning Predict Result</b></div>   

# In[61]:


# import deep learning model
import tensorflow as tf


# In[62]:


train_data


# In[63]:


target


# In[64]:


import tensorflow as tf

model = tf.keras.models.Sequential(
[
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])# if the layer nuber get too high numb, it can be lead model's overfitting.

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.array(train_data), np.array(target), epochs=150, verbose=0)

# if my deep learning model need to fix or improve, please leave a comment/feedback how should I fix it. 


# In[65]:


submission_dl = []


# In[66]:


# As the sample, Survival must be expressed as either 0 or 1. 
# But the prediction value from deep learning, It expressed percentage(0~1).
# So if its value higher than 0.5, It is more likely to have lived.
# let it transfer to 1. else transfer to 0.
for i in range(len(model.predict(test_data))):
    if model.predict(test_data)[i][0] >= 0.5:
        submission_dl.append(1)
    else:
        submission_dl.append(0)
# print(submission_dl)


# In[67]:


prediction_dl = np.array(submission_dl)
prediction_dl


# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Serif;text-align:left;display:fill;border-radius:5px;background-color:#030066;overflow:hidden"><b>📖4. Submission</b></div>   

# In[68]:


submission = pd.DataFrame({
    "PassengerId":test_df["PassengerId"],
    "Survived":prediction
    # if you want submit MachineLearning Score, use "prediction"
    # if you want submit DeepLearning Score, use "prediction_dl"
    # as my submittion result, Machine Learning Score higher than Deep learning Score. So I choose ML Score.
})

submission.to_csv("submission_ma.csv",index=False)


# In[69]:


submission = pd.read_csv("submission_ma.csv")
submission.head()


# #### <div style='text-align:center; font-family:arial'> 🙇🏻‍♂️Thanks For Watching!<br></div>

# In[ ]:




