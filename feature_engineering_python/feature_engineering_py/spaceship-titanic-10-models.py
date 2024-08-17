#!/usr/bin/env python
# coding: utf-8

#   #### The Spaceship Titanic challenge on Kaggle is a competition in which the task is to predict which passengers are transported to an alternate dimension, based on a set of variables.

#  Importing libraries

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# In[2]:


train=pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")


# ## EDA

# In[3]:


train.head()


# In[4]:


print("Shape of train dataset : ",train.shape)
print("Shape of test dataset : ",test.shape)


# In[5]:


train.columns


# In[6]:


train.info()


# In[7]:


#analysing Transported column
train['Transported'].value_counts()


# In[8]:


#countplot
sns.countplot(x='Transported',data=train,palette=['g','y'])


# In[9]:


#analysing VIP column
sns.countplot(x='VIP',data=train,palette=['g','y'])


# In[10]:


#create new feature
train['Not Transported']=1-train['Transported']


# In[11]:


#visulazing transported based VIP column
train.groupby('VIP').agg('sum')[['Transported','Not Transported']].plot(kind='bar',figsize=(10,5),stacked=True)


# ## Data Cleaning

# In[12]:


#checking the no of null values in the dataset
train.isnull().sum().sort_values(ascending=False)


# In[13]:


#filling null values with mean in numerical columns

train['Age']=train['Age'].fillna(train['Age'].mean())
train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
train['Spa']=train['Spa'].fillna(train['Spa'].mean())
train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())


# In[14]:


#filling null values with mode in categorical columns

train['HomePlanet']=train['HomePlanet'].fillna(train['HomePlanet'].mode()[0])
train['CryoSleep']=train['CryoSleep'].fillna(train['CryoSleep'].mode()[0])
train['Destination']=train['Destination'].fillna(train['Destination'].mode()[0])
train['VIP']=train['VIP'].fillna(train['VIP'].mode()[0])
train['Cabin']=train['Cabin'].fillna(train['Cabin'].mode()[0])


# In[15]:


train.isnull().sum().sort_values(ascending=False)


# ## Feature Engineering

# In[16]:


#count unique values in cabin column
train.Cabin.value_counts()


# In[17]:


train['Cabin_side'] = train['Cabin'].apply(lambda x: x.split('/')[2])
train['Cabin_side'].unique()


# P - cabin side Port ;
# S - cabin side Starboard

# In[18]:


df1=train
df1.head()


# In[19]:


#converting categorical feature into numerical feature
df1.HomePlanet=df1.HomePlanet.map({'Europa':0,'Earth':1,'Mars':2})
df1.Cabin_side=df1.Cabin_side.map({'P':0,'S':1})


# In[20]:


df1.Destination=df1.Destination.map({'TRAPPIST-1e':0,'PSO J318.5-22':1,'55 Cancri e':2})


# In[21]:


df1["CryoSleep"].replace(False,0,inplace=True)
df1["CryoSleep"].replace(True,1,inplace=True)
df1["VIP"].replace(False,0,inplace=True)
df1["VIP"].replace(True,1,inplace=True)
df1["Transported"].replace(False,0,inplace=True)
df1["Transported"].replace(True,1,inplace=True)


# In[22]:


#droping unwanted columns
df1=train.drop(['Name','Cabin','Not Transported'],axis=1)


# In[23]:


#final_df
df1.head()


# In[24]:


df1.describe()


# In[25]:


#correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df1.corr(),annot=True,linewidth=0.4,cmap='Greens_r')


# In[26]:


#checking null values
df1.isnull().sum()


# ## Model Building

# #### Splitting data

# In[27]:


X=df1.drop(['Transported'],axis=1)
y=df1['Transported']


# In[28]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=0)


# In[29]:


print(f'X_train',X_train.shape)
print(f'y_train',y_train.shape)
print(f'X_val',X_val.shape)
print(f'y_val',y_val.shape)


# ## 1. K Nearest Neighbor

# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[31]:


#to find which value shows the lowest werror
error = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_val)
    error.append(np.mean(pred_i != y_val))


# In[32]:


plt.plot(range(1,30),error,color='green',linestyle='--',marker='o',markersize=10,markerfacecolor='g')


# From this graph, K value of 3 seem to show the lowest mean error.

# In[33]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# #### Prediction and Accuracy

# In[34]:


from sklearn.metrics import accuracy_score

pred1 = knn.predict(X_val)
a1=accuracy_score(y_val,pred1)
print("Accuracy KNN Classifier : ",round(accuracy_score(y_val,pred1),4)*100, '%')


# ## 2. AdaBoost Classifier

# In[35]:


from sklearn.ensemble import AdaBoostClassifier

adbc=AdaBoostClassifier(n_estimators=115,learning_rate=0.412,random_state=42)
adbc.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[36]:


pred2=adbc.predict(X_val)
a2=accuracy_score(y_val,pred2)
print("Accuracy AdaBoost Classifier : ",round(accuracy_score(y_val,pred2),4)*100, '%')


# ## 3. Gaussian Naive Bayes

# In[37]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[38]:


pred3=gnb.predict(X_val)
a3=accuracy_score(y_val,pred3)
print("Accuracy Gaussian Naive Bayes : ",round(accuracy_score(y_val,pred3),4)*100, '%')


# ## 4. Decision Tree Classifier

# In[39]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=42)
dtc.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[40]:


pred4=dtc.predict(X_val)
a4=accuracy_score(y_val,pred4)
print("Accuracy Decision Tree Classifier : ",round(accuracy_score(y_val,pred4),4)*100, '%')


# ## 5. Multinomial Naive Bayes

# In[41]:


from sklearn.naive_bayes import MultinomialNB
mnnb= MultinomialNB()
mnnb.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[42]:


pred5=mnnb.predict(X_val)
a5=accuracy_score(y_val,pred5)
print("Accuracy Multinomial Naive Bayes : ",round(accuracy_score(y_val,pred5),4)*100, '%')


# ## 6. Support Vector Classifier

# In[43]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[44]:


pred6 =svc.predict(X_val)
a6=accuracy_score(y_val,pred6)
print("Accuracy Support Vector Classifier : ",round(accuracy_score(y_val,pred6),1)*100, '%')


# ## 7. Random Forest Classifier

# In[45]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=600,max_depth=18,random_state=42,min_samples_leaf=4)
rfc.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[46]:


pred7=rfc.predict(X_val)
a7=accuracy_score(y_val,pred7)
print("Accuracy Random Forest Classifier : ",round(accuracy_score(y_val,pred7),4)*100, '%')


# ## 8. Multi-layer Perceptron classifier

# In[47]:


from sklearn.neural_network import MLPClassifier
mlp= MLPClassifier()
mlp.fit(X_train, y_train)


# #### Prediction and Accuracy

# In[48]:


pred8=mlp.predict(X_val)
a8=accuracy_score(y_val,pred8)
print("Accuracy Multi-layer Perceptron classifier : ",round(accuracy_score(y_val,pred8),4)*100, '%')


# ## 9. Gradient Boosting

# In[49]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=210,learning_rate=0.01,random_state=42,max_depth=10,subsample=0.7)
gbc.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[50]:


pred9=gbc.predict(X_val)
a9=accuracy_score(y_val,pred9)
print("Accuracy Gradient Boosting : ",round(accuracy_score(y_val,pred9),5)*100, '%')


# ## 10. Logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression
lg= LogisticRegression(random_state=0)
lg.fit(X_train,y_train)


# #### Prediction and Accuracy

# In[52]:


pred10=gbc.predict(X_val)
a10=accuracy_score(y_val,pred10)
print("Accuracy Logistic Regression : ",round(accuracy_score(y_val,pred10),4)*100, '%')


# ## Model  Comparison

# In[53]:


models=['K Nearest Neighbor','AdaBoost Classifier','Gaussian Naive Bayes','Decision Tree Classifier','Multinomial Naive Bayes',
        'Support Vector Classifier','Random Forest Classifier','Multi-layer Perceptron classifier','Gradient Boosting','Logistic Regression']
acc=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
data={'Models':['K Nearest Neighbor','AdaBoost Classifier','Gaussian Naive Bayes','Decision Tree Classifier','Multinomial Naive Bayes',
                'Support Vector Classifier','Random Forest Classifier','Multi-layer Perceptron classifier','Gradient Boosting','Logistic Regression'],
      'Accuracy':[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]}
pd.DataFrame(data).style.background_gradient(cmap='Blues')


# ## Visualizing Accuracy of 10 Models 

# In[54]:


#plotting
plt.title('Comparing Models')
plt.xlabel('Accuracy')
sns.barplot(x=acc,y=models,width=1,palette='rainbow')


# ## Submission

# #### Test Data Exploration

# In[55]:


test.head()


# In[56]:


print("Shape of test dataset : ",test.shape)


# In[57]:


#checking the no of null values in the dataset
test.isnull().sum().sort_values(ascending=False)


# In[58]:


#filling null values with mean in numerical columns

test['Age']=test['Age'].fillna(test['Age'].mean())
test['RoomService']=test['RoomService'].fillna(test['RoomService'].mean())
test['FoodCourt']=test['FoodCourt'].fillna(test['FoodCourt'].mean())
test['ShoppingMall']=test['ShoppingMall'].fillna(test['ShoppingMall'].mean())
test['Spa']=test['Spa'].fillna(test['Spa'].mean())
test['VRDeck']=test['VRDeck'].fillna(test['VRDeck'].mean())


# In[59]:


#filling null values with mode in categorical columns

test['HomePlanet']=test['HomePlanet'].fillna(test['HomePlanet'].mode()[0])
test['CryoSleep']=test['CryoSleep'].fillna(test['CryoSleep'].mode()[0])
test['Destination']=test['Destination'].fillna(test['Destination'].mode()[0])
test['VIP']=test['VIP'].fillna(test['VIP'].mode()[0])
test['Cabin']=test['Cabin'].fillna(test['Cabin'].mode()[0])


# In[60]:


test['Cabin_side'] = test['Cabin'].apply(lambda x: x.split('/')[2])
test['Cabin_side'].unique()


# In[61]:


#converting categorical feature into numerical feature
test.HomePlanet=test.HomePlanet.map({'Europa':0,'Earth':1,'Mars':2})
test.Cabin_side=test.Cabin_side.map({'P':0,'S':1})


# In[62]:


test.Destination=test.Destination.map({'TRAPPIST-1e':0,'PSO J318.5-22':1,'55 Cancri e':2})


# In[63]:


test["CryoSleep"].replace(False,0,inplace=True)
test["CryoSleep"].replace(True,1,inplace=True)
test["VIP"].replace(False,0,inplace=True)
test["VIP"].replace(True,1,inplace=True)


# In[64]:


#droping unwanted columns
test_df=test.drop(['Name','Cabin'],axis=1)


# In[65]:


#final_df
test_df.head()


# In[66]:


#checking null values
test_df.isnull().sum()


# In[67]:


#final prediction we use the Random Forest Classifier with 80% accuracy
pred_final=rfc.predict(test_df)


# In[68]:


a=test_df['PassengerId']
x=pd.DataFrame(a)
b=pred_final
y=pd.DataFrame(b)


# In[69]:


final=pd.concat([x,y],axis=1)
final.replace(0,False,inplace=True)
final.replace(1,True,inplace=True)
final.rename(columns={0:'Transported'},inplace=True)
final


# In[70]:


final.to_csv('spaceship_titanic.csv',index=False)


# In[71]:


#visualizing predicted values
sns.countplot(x='Transported',data=final,palette=['r','g'])

