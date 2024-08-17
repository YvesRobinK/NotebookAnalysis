#!/usr/bin/env python
# coding: utf-8

# # Titanic data EDA and Prediction

# **Data description**
# 
# Here we have 12 columns-
# 
# **PassengerId** : ID of Passenger
# 
# **Pclass** : Passenger class (1=1st,2=2nd,3=3rd)
# 
# **Survived** : Survival (0=No,1=Yes)
# 
# **Sex** : sex(male & female)
# 
# **Name** : name of passengers
# 
# **Age** : age of passengers
# 
# **Sibsp** : Number of Siblings
# 
# **Parch** : Number of Parents
# 
# **Ticket** : passenger ticket number
# 
# **Fare** : Passenger fare(British pound)
# 
# **Cabin** : cabin
# 
# **Embarked** : Port of Embarkation(C=Cherbourg ,Q=Queenstown, S=Southamption)
# 
# 
# 
# 

# In[1]:


#importing usefull lib

import numpy as np
import pandas as pd
import seaborn as sns
import warnings as wr
wr.filterwarnings("ignore")
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


#reading the data using pandas read_csv. 
#It's ofthen used pandas fuction to read csc file.
import pandas as pd
df1=pd.read_csv("../input/titanic/train.csv")
df1.head()


# In[3]:


#checking shape of data frame
df1.shape


# In[4]:


#here we check mean,std,quantiles value using pandas describe function
df1.describe()


# In[5]:


#extracting all columns from the data frame for forother uses
df1.columns.tolist()


# In[6]:


#count NA values
df1.isnull().sum()


# In[7]:


#dropng unrelated column
#here we going to drop cabin bcz it's have lots of nan vales
#here is nothing use in traing og passenger id so simply we drop it using pandas drop()
df=df1.drop(["PassengerId","Ticket","Cabin"],axis=1)
df.head()


# In[8]:


#filing NA values
#here we filling na values by mean() for numerical values 
#and mode() for categorical 
df["Age"].fillna(df["Age"].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[9]:


df.info()


# # Exploratory Data Analysis

# In[11]:


print(f'Number of people dead as 0 are {df.Survived.value_counts()[0]} and Number of people survived as 1 are {df.Survived.value_counts()[1]}')
sns.countplot(df["Survived"])
plt.show()


# In[12]:


#here wr checking outliers 
f,ax=plt.subplots()
sns.violinplot(data=df.iloc[:,5:7])
sns.despine(offset=10,trim=True)


# In[13]:


f,ax=plt.subplots()
sns.violinplot(data=df.iloc[:,0:2])
#sns.despine(offset=10,trim=True)
sns.swarmplot(data=df.iloc[:,0:2],color="white")


# In[14]:


#here we going to ploat scaterplot to see data distirbution
sns.relplot(x="Pclass",y="Age",hue="Survived",data=df);


# In[15]:


#visualisation how many pasanger survived and how many dead
#here we creat a function for bar_chart 
#for avoiding write same code for defrent columns

def bar_chart(column):
    survived=df[df["Survived"]==1][column].value_counts()
    dead=df[df["Survived"]==0][column].value_counts()
    df1=pd.DataFrame([survived,dead])
    df1.index=["Survived","Dead"]
    df1.plot(kind="bar",figsize=(10,5))


# In[16]:


#here we make a bar chart on sex column
#for checking how many male & female
bar_chart("Sex")


# **In the above chart we can essly analyse that females have more chance to survived.**

# In[17]:


#here we going to make bar char on Pclass
bar_chart("Pclass")


# **by the above chart on Pclass we can say that 1st class passenger have more chance to survived**

# In[18]:


#here we going to make bar chart on sibsp
bar_chart("SibSp")


# **by above chart we can analyes that there is more chance to survivrd those who have 0 or 1 siblings**

# In[19]:


bar_chart("Parch")


# In[20]:


bar_chart("Embarked")


# **by the above chart we can say that there are more chance to survived for those who bord from Southamption**
# 
# **Passenger traveilling from Cherbourg port survived more than other port passenger**

# **Dedacting outliers and removing them**

# In[21]:


#visualisation data on boxplot to see the outliers
def box_plot(column):
    df.boxplot(by="Survived",column=[column],grid=True)



# In[22]:


box_plot("Fare")


# here we can see outliers above the 100 we can considerd them as outlier

# In[23]:


#checking outliers on Sibsp column
box_plot("SibSp")


# by above chart we consider more then 5 siblings as outlierS

# In[24]:


#ploting pair plot
g=sns.PairGrid(df,hue="Survived")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend


# In[25]:


#by value_counts we can see total unique values
df["SibSp"].value_counts()


# In[26]:


#here we chacking largest values row on column Sibsp
df.nlargest(12,["SibSp"])


# In[27]:


#now we gpoing to remove outliers
df=df.drop([159,180,201,324,792,846,863])
df.shape


# In[28]:


#here we going to check outliers on parch
box_plot("Parch")


# In[29]:


df["Parch"].value_counts()


# In[30]:


df.nlargest(12,["Parch"])


# In[31]:


df=df.drop([678])
df.shape


# In[32]:


#here we going to drow heatmap to check co relation between columns 

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# **Frome EDA we got**
# * Cabin column have lots of null values so we drop it,
# * Ticket and paddenger Id is not usefull and does not have impact on survivl so drop it.
# * Passenger travelling in higher class have more chance to survived
# * Females survived more then Males.
# * In the 1st class Females were more then Males it is also a resion that females have more chance to survived.
# * Passenger travelling with siblings ,parents have more chance to survived.
# * Passenger traveilling from Cherbourg port survived more than other port passenger.
# 

# **making title feature using Name**

# In[33]:


df["Title"]=df["Name"].str.split(',',expand=True)[1].str.split('.',expand=True)[0]


# In[34]:


df["Title"].unique()


# **now we can replace many titles with a more comman name as Rare**

# In[35]:


df["Title"]=df["Title"].replace([" Don"," Rev"," Dr"," Major"," Lady"," Sir"," Col"," Capt"," the Countess"," Jonkheer"],"Rare")
df["Title"]=df["Title"].replace([" Mlle", " Ms"]," Miss")
df["Title"]=df["Title"].replace([" Mme"," Mrs"]," Mr")


# In[36]:


df["Title"].unique()


# In[37]:


#droping an relevant columns
#dividing data X(features) and Y(outcome)
X=df.drop(["Fare","Survived","Age","Name"],axis=True)
y=df["Survived"]


# In[38]:


print(X.shape)
print(y.shape)


# In[39]:


X.head()


# **Feature Engineering**

# In[40]:


#Here we encode Embarked in Rank
X.loc[X['Embarked'] == "C", 'Embarked'] = 0
X.loc[X['Embarked'] == "Q", 'Embarked'] = 1
X.loc[X['Embarked'] == "S", 'Embarked'] = 2


# In[41]:


mapping={' Mr':0, ' Miss':1, ' Master':2, 'Rare':3}
X["Title"]=X["Title"].map(mapping)


# In[42]:


#Here we encode Sex in Rank
X.loc[X['Sex'] == "female", 'Sex'] = 0
X.loc[X['Sex'] == "male", 'Sex'] = 1


# In[43]:


X.head()


# In[44]:


X.isnull().sum()


# **Model building**

# In[45]:


#here we going to split data in traing set and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)


# **Training LogisticRegression**

# In[46]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=40)
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))


# **Training DecisionTreeClassifier**

# In[47]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dtc = DecisionTreeClassifier()

parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 32, 1),
    'min_samples_leaf' : range(1, 10, 1),
    'min_samples_split' : range(2, 10, 1),
    'splitter' : ['best', 'random']
}

grid_search_dt = GridSearchCV(dtc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dt.fit(x_train, y_train)


# In[48]:


# best parameters

grid_search_dt.best_params_


# In[49]:


dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, min_samples_leaf = 6,
                             min_samples_split = 8, splitter = 'random')
dtc.fit(x_train, y_train)


# In[50]:


# accuracy score
print(dtc.score(x_test,y_test))


# # Gradient Boosting Classifier

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}

grid_search_gbc = GridSearchCV(gbc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_gbc.fit(x_train, y_train)


# In[52]:


# best parameters 

grid_search_gbc.best_params_


# In[53]:


gbc = GradientBoostingClassifier(learning_rate = 0.1, loss = 'exponential', n_estimators = 100)
gbc.fit(x_train, y_train)


# In[54]:


# accuracy score
print(gbc.score(x_test,y_test))


# **Traing RandomForestClassifier**

# In[55]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=40,min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001)

rfc.fit(x_train,y_train)

#print(rfc.score(x_test,y_test))


# In[56]:


print(rfc.score(x_test,y_test))


# #  Support Vector Classifier (SVC)

# In[57]:


from sklearn.svm import SVC

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(x_train, y_train)


# In[58]:


# best parameters

grid_search.best_params_


# In[59]:


svc = SVC(C = 1, gamma = 0.1)
svc.fit(x_train, y_train)


# In[60]:


print(svc.score(x_test,y_test))


# # K Neighbors Classifier (KNN)

# In[61]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)


# In[62]:


print(knn.score(x_test,y_test))


# In[63]:


key = ['LogisticRegression','DecisionTreeClassifier','GradientBoostingClassifier','RandomForestClassifier','SVC','KNeighborsClassifier']
model=[lr,dtc,gbc,rfc,svc,knn]


# In[64]:


score=[]
for i in model:
    sco = i.score(x_test,y_test)
    score.append(sco)
print(score)


# In[65]:


plt.figure(figsize = (10,5))
sns.barplot(x = score, y = key, palette='pastel')


# ## finaly we chose over best model **GradientBoostingClassifier** for prediction

# # Prediction on Test data
# **clean and feature selection same as training**

# In[66]:


df2=pd.read_csv("../input/titanic/test.csv")

df2.head()


# In[67]:


df2["Title"]=df2["Name"].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
df2["Title"]=df2["Title"].replace([" Don"," Rev"," Dr"," Major"," Lady"," Sir"," Col"," Capt"," the Countess"," Jonkheer"],"Rare")
df2["Title"]=df2["Title"].replace([" Mlle", " Ms"," Dona"]," Miss")
df2["Title"]=df2["Title"].replace([" Mme"," Mrs"]," Mr")
df2["Title"].unique()


# In[68]:


mapping={' Mr':0, ' Miss':1, ' Master':2, 'Rare':3}
df2["Title"]=df2["Title"].map(mapping)


# In[69]:


new_x=df2.drop(["Cabin","PassengerId","Fare","Age","Name","Ticket"],axis=True)
new_x.head()


# In[70]:


new_x.isnull().sum()


# In[71]:


#Here we encode Embarked in Rank
new_x.loc[new_x['Embarked'] == "C", 'Embarked'] = 0
new_x.loc[new_x['Embarked'] == "Q", 'Embarked'] = 1
new_x.loc[new_x['Embarked'] == "S", 'Embarked'] = 2


# In[72]:


new_x.loc[new_x['Sex'] == "female", 'Sex'] = 0
new_x.loc[new_x['Sex'] == "male", 'Sex'] = 1


# In[73]:


new_x.head()


# In[74]:


#here we used over best train model
new_predict=gbc.predict(new_x)
print(new_predict)


# In[75]:


vip=np.array(new_predict).tolist()


# In[76]:


len(vip)


# In[77]:


df2.insert(2,column="Survived",value=vip)
df2.head()


# # macking csv(PassengerId & survived) file to upload

# In[78]:


df3=df2.drop(['Pclass','Title','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
df3.head()


# In[79]:


df3.to_csv('Titanic_modelP_lr.csv',index=False)
df3.head()


# # If you like please Do a up vote

# ### Thanks
