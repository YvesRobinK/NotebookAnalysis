#!/usr/bin/env python
# coding: utf-8

# <center><h2> 🛳️Welcome to my work! Please check my work and if you have any suggestion or advice, please tell me. That will be very helpful to me!🛳️</h2></center>

# ![k-mitch-hodge-y-9-X5-4-vU-unsplash.jpg](attachment:8e155f67-a0a0-4f63-a7f2-e78f2fea0c1d.jpg)

# # Data Introduction🚢

#     1. survival: Survival	0 = No, 1 = Yes
#     2. pclass: Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#     3. sex: Sex	
#     4. Age: Age in years
#     5. sibsp: number of siblings / spouses aboard the Titanic	
#     6. parch: number of parents / children aboard the Titanic	
#     7. ticket: Ticket number
#     8. fare: Passenger fare
#     9. cabin: Cabin number
#     10. embarked: Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# In[1]:


# import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Let's check train data

# In[2]:


# data check

data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
data.head()

data.describe()


# we can see there are nan values in data. we have to fill in it or delete it.

# In[3]:


data.info()


# In[4]:


data.isna().sum()


# # 1️⃣ㅣ Feature Engineering + EDA📊

# extract only cabin unit without numbers.

# In[5]:


# feature cleaning & feature engineering

data["Cabin"].fillna("Unknown", inplace=True)

Cabin_unit = []
for i in data["Cabin"]:
  Cabin_unit.append(i[0])

data["Cabin_unit"] = Cabin_unit


# In[6]:


plt.figure(figsize=(10,7))
sns.set_palette("pastel")
sns.histplot(x="Cabin_unit", hue="Survived", kde=True, data=data)
plt.show


# In[7]:


plt.figure(figsize=(10,7))
sns.set_palette("pastel")
sns.barplot(x="Cabin_unit", y="Survived", data=data)
plt.show


# In[8]:


data.isna().sum()


# extract features from "Name" value.

# In[9]:


data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().value_counts()

data['Name_re'] = data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Uncommon', 'Don': 'Uncommon', 'Jonkheer': 'Uncommon', 'the Countess': 'Uncommon', 'Sir': 'Uncommon', 'Mme': 'Uncommon', 'Capt': 'Mr', 'Major': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'})

data["Name_re"].value_counts()

data


# In[10]:


plt.figure(figsize=(10,7))
sns.countplot(x="Name_re", hue="Survived", data=data)
plt.title("Name_re Count")
plt.show()


# In[11]:


plt.figure(figsize=(10,7))
sns.barplot(x="Name_re", y="Survived", data=data)
plt.show()


# we don't need "Name" value now. so drop it.

# In[12]:


data.drop("Name", axis=1, inplace=True)


# we still have to fill "Age" values.

# In[13]:


data.isna().sum()


# In[14]:


data["Age"].fillna(data["Age"].mean(), inplace=True)


# In[15]:


data["Embarked"].value_counts()


# most common value is "S"

# In[16]:


data["Embarked"].fillna("S", inplace=True)


# In[17]:


plt.figure(figsize=(10,7))
sns.countplot(x="Embarked", data=data)
plt.title("Embarked Count")
plt.show()


# In[18]:


plt.figure(figsize=(10,7))
sns.barplot(x="Survived", y="Embarked", hue="Pclass", data=data)
plt.show()


# In[19]:


data.isna().sum()


# now finally we finished to fill nan values in data!!!

# In[20]:


plt.figure(figsize=(10,7))
sns.histplot(x="SibSp", hue="Survived", kde=True, data=data)
plt.show()


# In[21]:


plt.figure(figsize=(10,7))
sns.histplot(x="Parch", hue="Survived", kde=True, data=data)
plt.show()


# In[22]:


plt.figure(figsize=(10,7))
sns.lineplot(x="SibSp", y="Survived", data=data)
plt.show()


# In[23]:


plt.figure(figsize=(10,7))
sns.lineplot(x="Parch", y="Survived", data=data)
plt.show()


# In[24]:


data["Family"] = data["SibSp"] + data["Parch"]

data.drop(["SibSp", "Parch"], axis=1, inplace=True)


# In[25]:


plt.figure(figsize=(10,7))
sns.histplot(x="Family", hue="Survived", kde=True, data=data)
plt.show()


# In[26]:


plt.figure(figsize=(10,7))
sns.lineplot(x="Family", y="Survived", data=data)
plt.show()


# In[27]:


Alone = []
for i in data["Family"]:
  if i == 0:
    Alone.append(True)
  else:
    Alone.append(False)

data["Alone"] = Alone

data["Alone"] =data["Alone"].astype("int")

data


# In[28]:


plt.figure(figsize=(10,7))
sns.barplot(x="Alone", y="Survived", hue="Sex", data=data)
plt.show()


# test_data doesn't have Cabin_unit T value.

# In[29]:


data["Cabin_unit"].replace({"T":"U"}, inplace=True)


# In[30]:


data.drop(["Ticket", "Cabin"], axis=1, inplace=True)


# In[31]:


plt.figure(figsize=(25,12))
sns.histplot(x="Fare", hue="Survived", kde=True, data=data)
plt.show()


# In[ ]:





# get dummies!

# In[32]:


data = pd.get_dummies(data, columns=["Sex", "Embarked", "Cabin_unit", "Name_re"], drop_first=True)

data


# define x and y

# In[33]:


x = data.drop(["PassengerId", "Survived"], axis=1)
y = data["Survived"]
x.shape, y.shape


# # 2️⃣ㅣ LigthGBM + GridSearchCV

# i will make model with LightGBM and GridSearchCV! first set parameters.

# In[34]:


# LightGBM + GridSearchCV

import lightgbm as lgb
model = lgb.LGBMClassifier()

params = {
    "n_estimators": [3, 5, 10, 50, 100, 200], # total tree number
    "learning_rate": [0.01, 0.05, 0.1, 0.3], # learning rate
    "max_depth": [1, 3, 5, 10, 15], # max depth of each tree
    "subsample": [0.6, 0.8, 1] # subsample rate of data
}

from sklearn.model_selection import train_test_split, GridSearchCV
model_rs = GridSearchCV(model, params, n_jobs=-1, cv=5)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train

model_rs.fit(x_train, y_train) # LightGBM with GridSearchCV fitting

print(model_rs.best_params_)


# In[36]:


model_rs.best_estimator_.feature_importances_


# In[37]:


pred_1 = model_rs.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_1)


# # 3️⃣ㅣ Feature Importance

# In[38]:


Fimp = pd.DataFrame({"feature": x_train.columns, "importance": model_rs.best_estimator_.feature_importances_}).sort_values("importance", ascending=False)

plt.figure(figsize=(20,30))
sns.barplot(data= Fimp, x="importance", y="feature")
plt.title("Feature Importance", fontsize=20)
plt.show()


# # 4️⃣ㅣ Test data Feature Cleaning & Engineering

# In[39]:


# test data cleaning & engineering

test_data


# In[40]:


test_data.isna().sum()


# In[41]:


test_data["Age"].fillna(test_data["Age"].median(), inplace=True)

test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

test_data["Cabin"].fillna("Unknown", inplace=True)

Cabin_unit = []
for i in test_data["Cabin"]:
  Cabin_unit.append(i[0])

test_data["Cabin_unit"] = Cabin_unit

test_data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().value_counts()

test_data['Name_re'] = test_data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Uncommon', 'Don': 'Uncommon', 'Jonkheer': 'Uncommon', 'the Countess': 'Uncommon', 'Sir': 'Uncommon', 'Mme': 'Uncommon', 'Capt': 'Mr', 'Major': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr', "Dona": "Uncommon"})

test_data.drop(["Name", "Cabin", "Ticket"], axis=1, inplace=True)


# In[42]:


test_data.isna().sum()

test_data["Family"] = test_data["SibSp"] + test_data["Parch"]

test_data.drop(["SibSp", "Parch"], axis=1, inplace=True)

Alone = []
for i in test_data["Family"]:
  if i == 0:
    Alone.append(True)
  else:
    Alone.append(False)

test_data["Alone"] = Alone

test_data["Alone"] =test_data["Alone"].astype("int")

test_data


# In[43]:


test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked", "Cabin_unit", "Name_re"], drop_first=True)

test_data


# # 5️⃣ㅣ Submmision

# In[44]:


test = test_data.drop("PassengerId", axis=1)

pred = model_rs.predict(test)

test_data["Survived"] = pred

submission = test_data[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)


# # 6️⃣ㅣThank you!👍

# <center><h1> I hope you liked my work! If it was helpful, please UPVOTE!!!👍👍👍 </h1></center>
