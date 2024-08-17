#!/usr/bin/env python
# coding: utf-8

# # ⛴️Table of contents🚢
# * [1ㅣ Import Libraries](#Import-Libraries)
# * [2ㅣ Read Data](#Read-Data)
# * [3ㅣ EDA](#EDA)
# * [4ㅣ Feature Cleaning & Engineering](#Feature)
# * [5ㅣ Modeling](#Modeling)
#     - [5-1ㅣ XGBoost](#xgb)
#     - [5-2ㅣ Parameters](#params)
#     - [5-3ㅣ GridSearchCV](#grid)
#     - [5-4ㅣ Best Parameters](#best)
#     - [5-5ㅣ Feature Importance](#feature)
# * [6ㅣ Test Data Cleaning & Engineering](#Test)
# * [7ㅣ Submit](#Submit)

# <a id="Import-Libraries"></a>
# # 1ㅣ Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <a id="Read-Data"></a>
# # 2ㅣ Read Data

# ### Read train data and test data.

# In[2]:


data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[3]:


train_data_rows = data.shape[0]
test_data_rows = test_data.shape[0]


# ### train data & test data

# In[4]:


plt.figure(figsize=(10,10))
plt.bar(0, train_data_rows, label="train data rows")
plt.bar(1, test_data_rows, label="test data rows")
plt.xlabel("train data / test data", fontsize=15)
plt.ylabel("rows count", fontsize=15)
plt.legend()
plt.title("data rows count", fontsize=20)
plt.show()


# ### Check train data.

# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe(include="all")


# In[8]:


data.isna().sum()


# There are NaN value in train data. We don't have to deal with NaN data because we are going to use XGBoost model.

# ### Check Test Data

# In[9]:


test_data.head()


# In[10]:


test_data.info


# In[11]:


test_data.describe()


# In[12]:


test_data.isna().sum()


# <a id="EDA"></a>
# # 3ㅣ EDA

# In[13]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Sex & Survived")
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=data)
plt.show()


# #### What we can know by Sex and Survived
# - female survived much more than male
# - Pclass affects survive rate
# 

# #### Let's watch whether sex value is really important.

# In[14]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Embarked & Survived")
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data)
plt.show()


# #### What we can know by Embarked and Survived
# - Embarked affects number of survive people.
# - number of survive number: C > S > Q

# In[15]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("SibSp & Survived")
sns.lineplot(x="SibSp", y="Survived", hue="Sex", data=data)
plt.show()


# #### What we can know by SibSp and Survived
# - If SibSp value is high survive probability is low.
# - Female survived more than male but when SibSp value is more than 5, no one survived.

# In[16]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Parch & Survived")
sns.lineplot(x="Parch", y="Survived", hue="Sex", data=data)
plt.show()


# #### What we can know by Parch and Survived
# - Parch and Survived graph looks different from SibSp and Survived graph.
# - But it's same that when value is high survive probability is low.

# In[17]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Pclass / Survived")
sns.violinplot(x="Pclass", y="Survived", hue="Sex", data=data, split=True)
plt.show()


# In[18]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Pclass / Fare")
sns.barplot(x="Pclass", y="Fare", data=data)
plt.show()


# In[19]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
sns.displot(x="Age", data=data, kde=True)
plt.show()


# In[20]:


AgeGroup = []
for i in data["Age"]:
    if i <= 4:
        AgeGroup.append("Baby")
    elif 4 < i <= 15:
        AgeGroup.append("Child")
    elif 15< i < 60:
        AgeGroup.append("Adult")
    elif 60 <= i:
        AgeGroup.append("Elder")
    else:
        AgeGroup.append(np.NaN)


# In[21]:


data["AgeGroup"] = AgeGroup


# In[22]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Age Group / Survived")
sns.barplot(x="AgeGroup", y="Survived", data=data)
plt.show()


# #### What we can know by AgeGroup and Survived
# - number of survived people: baby > child > adult > elder

# In[23]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
sns.heatmap(data.corr(), annot=True)
plt.show()


# #### Correspondence between SisSp and Parch is high so let's group into one.

# <a id="Feature"></a>
# # 4ㅣ Feature Cleaning & Engineering

# #### We don't need to deal with NaN value in data. But let's do it for practice.

# In[24]:


data["Cabin"].fillna("Unknown", inplace=True)

Cabin_unit = []
for i in data["Cabin"]:
  Cabin_unit.append(i[0])

data["Cabin_unit"] = Cabin_unit


# In[25]:


plt.figure(figsize=(10,7))
sns.set(font_scale=1.5)
plt.title("Cabin_unit / Survived")
sns.barplot(x="Cabin_unit", y="Survived", data=data)
plt.show()


# #### We can't use raw name value. Make it useful.

# In[26]:


data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().value_counts()

data['Name_re'] = data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip().replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Uncommon', 'Don': 'Uncommon', 'Jonkheer': 'Uncommon', 'the Countess': 'Uncommon', 'Sir': 'Uncommon', 'Mme': 'Uncommon', 'Capt': 'Mr', 'Major': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'})

data["Name_re"].value_counts()

data


# In[27]:


data.drop("Name", axis=1, inplace=True)


# In[28]:


data["Age"].fillna(data["Age"].mean(), inplace=True)


# In[29]:


data["Embarked"].value_counts()


# In[30]:


data["Embarked"].fillna("S", inplace=True)


# #### As I said, I grouped SibSp and Parch into one.

# In[31]:


data["Family"] = data["SibSp"] + data["Parch"]

data.drop(["SibSp", "Parch"], axis=1, inplace=True)


# In[32]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
plt.title("Family size / Survived")
sns.lineplot(x="Family", y="Survived", data=data)
plt.show()


# In[33]:


Alone = []
for i in data["Family"]:
  if i == 0:
    Alone.append(True)
  else:
    Alone.append(False)

data["Alone"] = Alone

data["Alone"] =data["Alone"].astype("int")

data


# In[34]:


plt.figure(figsize=(10,7))
sns.set(font_scale=1.5)
plt.title("Alone / Survived")
sns.barplot(x="Alone", y="Survived", hue="Sex", data=data)
plt.show()


# In[35]:


data["Cabin_unit"].replace({"T":"U"}, inplace=True)


# In[36]:


data.drop(["Ticket", "Cabin"], axis=1, inplace=True)


# In[37]:


data = pd.get_dummies(data, columns=["Sex", "Embarked", "Cabin_unit", "Name_re"], drop_first=True)

data


# In[38]:


x = data.drop(["PassengerId", "Survived", "AgeGroup"], axis=1)
y = data["Survived"]
x.shape, y.shape


# <a id="Modeling"></a>
# # 5ㅣModeling

# <a id="xgb"></a>
# ### XGBoost

# In[39]:


import xgboost as xgb
pre_model = xgb.XGBClassifier()


# <a id="params"></a>
# ### Parameters

# In[40]:


params = {
    "n_estimators": [3, 5, 10, 50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "max_depth": [1, 3, 5, 10, 15],
    "subsample": [0.6, 0.8, 1]
}


# <a id="grid"></a>
# ### GridSearchCV

# In[41]:


from sklearn.model_selection import train_test_split, GridSearchCV
model = GridSearchCV(pre_model, params, n_jobs=-1, cv=5)


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[43]:


model.fit(x_train, y_train)


# <a id="best"></a>
# ### Best Parameters

# In[44]:


print(model.best_params_)


# <a id="feature"></a>
# ### Feature Importance

# In[45]:


model.best_estimator_.feature_importances_


# In[46]:


F_imp = pd.DataFrame({"Feature": x_train.columns, "importance": model.best_estimator_.feature_importances_}).sort_values("importance", ascending=False)

plt.figure(figsize=(15,10))
sns.set(font_scale=1.5)
plt.title("Feature Importance")
sns.barplot(x="importance", y="Feature", data=F_imp)
plt.show()


# <a id="Test"></a>
# # 6ㅣTest Data Cleaning & Engineering

# In[47]:


test_data.isna().sum()


# In[48]:


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


# In[49]:


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


# In[50]:


test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked", "Cabin_unit", "Name_re"], drop_first=True)


# <a id="Submit"></a>
# # 7ㅣSubmit

# In[51]:


test = test_data.drop("PassengerId", axis=1)

pred = model.predict(test)

test_data["Survived"] = pred

submission = test_data[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

