#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import os


# In[2]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[3]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[4]:


train_data.shape, test_data.shape


# # Handling NULL values

# In[5]:


train_data.info(), test_data.info()


# In[6]:


pd.concat([
        train_data.isnull().sum().rename("train"),
        (train_data.isnull().sum() / train_data.isnull().count()).round(4).rename("train (%)"),
        test_data.isnull().sum().rename("test"),
        (test_data.isnull().sum() / test_data.isnull().count()).round(4).rename("test (%)")
    ],
    axis=1
)


# In[7]:


# Only 2 missing => fill most common city
train_data['Embarked'].fillna((train_data['Embarked'].mode()[0]), inplace=True)


# In[8]:


#  Median age of the passenger's honorific & pclass (instead of simply mean or median of the 'Age')

for data in [train_data, test_data]:
    data["Honorific"] = data["Name"].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    
train_data.head()


# In[9]:


for data in [train_data, test_data]:
    median_ages = pd.Series(data.groupby(["Honorific"])["Age"].median()).sort_values(ascending=False)
    honorific_count = data["Honorific"].value_counts().rename("Cnt")

    print(pd.concat([median_ages, honorific_count], axis=1))


# In[10]:


for data in [train_data, test_data]:
    data["Age"] = data.groupby(["Honorific", "Pclass"], sort=False)["Age"] \
                      .apply(lambda x: x.fillna(x.mean()))
    
# There's only 1 Ms. in the test_data and it's missing "Age" => fill the gap w/ Ms' median age from the train_data
age = pd.Series(train_data.groupby(by="Honorific")["Age"].median())["Ms."]
test_data["Age"].fillna(age, inplace=True)


# In[11]:


_dfname_mapping = { 0: "Train", 1: "Test" }

fig, axs = plt.subplots(1, 2, figsize=(20, 4))
for i, data in enumerate([train_data, test_data]):
    sns.boxplot(x=data["Fare"], y=data["Pclass"], orient ='h', showfliers=False, ax=axs[i])
    axs[i].set_title(f"{_dfname_mapping[i]}")
    axs[i].set(ylabel="Passenger Class")
    axs[i].set_yticklabels(["First Class", "Second Class", "Third Class"])
    
plt.show()


# In[12]:


# Checking the pclass-wise median fare:

fig, axs = plt.subplots(1, 2, figsize=(12, 10))
for i, data in enumerate([train_data, test_data]):
    median_fares = pd.Series(data.groupby("Pclass")["Fare"].median())
    median_fares.plot(kind="bar", ax=axs[i])
    for idx in range(3):
        axs[i].text(x=-0.1 + idx, y=median_fares.loc[idx + 1] + 1, s=f"${median_fares.loc[idx + 1].round(2)}")
    
    axs[i].set_title(f"Pclass-wise Median Fare ({_dfname_mapping[i]} dataset)")
    axs[i].set(xlabel="Passenger Class", ylabel="Fare")
    axs[i].set_xticklabels(["First Class", "Second Class", "Third Class"], rotation="horizontal")
    
plt.show()


# In[13]:


# Fill in the missing fare values with median fares of their respective pclass

test_data["Fare"] = test_data.groupby("Pclass", sort=False)["Fare"] \
                             .apply(lambda x: x.fillna(x.mean()))


# In[14]:


# Replace missing cabin with 'n/a' - handle it later

for data in [train_data, test_data]:
    data['Cabin'].fillna('n/a', inplace=True)


# In[15]:


pd.concat([
        train_data.isnull().sum().rename("train"),
        test_data.isnull().sum().rename("test")
    ],
    axis=1
)


# # Feature engineering

# * ❌ ___PassengerID___
# 
# 
# * ✔️ ___Name___
# * ✔️ ___Ticket___
# * ✔️ ___Age___
# * ✔️ ___Sex___
# * ✔️ ___Pclass___
# * ✔️ ___City of Embarkment___
# * ✔️ ___Cabin___
# * ✔️ ___Fare___
# * ✔️ ___Number of Parents or Children (Parch) and Number of Siblings or Spouses (SibSp)___
# * ✔️ ___Honorifics___

# ### ___Name___

# In[16]:


# Use length of the name

train_data["Name_Length"] = train_data["Name"].apply(lambda x: len(x))


# In[17]:


# Visualising names' length w.r.t. Survival:

plt.figure(figsize=(30, 10))
sns.countplot(data=train_data, x="Name_Length", hue="Survived")

plt.title("Survival Distribution among Names' length")
plt.xlabel("Letters amount")
plt.legend(["Deceased", "Survived"])

plt.show()


# In[18]:


nlen_groups = pd.qcut(train_data["Name_Length"], 5)
nlen_groups.value_counts()


# In[19]:


train_data["Survived"].groupby(nlen_groups).mean()


# People with __longer names__ are more likely to __survive__.
# 
# Perhaps people with longer names are more important, and thus more likely to be prioritized for a seat in lifeboat.

# In[20]:


test_data["Name_Length"] = test_data["Name"].apply(lambda x: len(x))


# ### ___Ticket___

# In[21]:


# Use first letter of the ticket
# Perhaps it's a certain attribute of the cabin/passenger

train_data["Ticket_FLetter"] = train_data["Ticket"].apply(lambda x: str(x)[0])
train_data.groupby(["Ticket_FLetter"])["Survived"].mean()


# In[22]:


# Visualising Ticket's first letter w.r.t. Survival:

# fletter_survival_cnt = train_data.groupby("Ticket_FLetter")["Survived"].value_counts()

plt.figure(figsize=(30, 10))
sns.countplot(data=train_data, x="Ticket_FLetter", hue="Survived")

plt.title("Survival Distribution among Tickets (first letter)")
plt.xlabel("Ticket's first letter")
plt.legend(["Deceased", "Survived"])

plt.show()


# The survival rates seem too irregular (perhaps it depends on cabin class and location)

# In[23]:


test_data["Ticket_FLetter"] = test_data["Ticket"].apply(lambda x: str(x)[0])


# ### ___Age___

# In[24]:


fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Visualising the age distribution w.r.t survival:
sns.swarmplot(data=train_data, x="Survived", y="Age", ax=axs[0], size=3)
axs[0].set_title("Age distr w.r.t Survival")
axs[0].set_xticklabels(["Deceased", "Survived"], rotation="horizontal")

# Visualising ages w.r.t survival:
sns.histplot(data=train_data, x='Age', hue="Survived", multiple = "stack", bins = 80, ax=axs[1])
axs[1].set_title("Survival distributation based on passengers' age")
axs[1].legend(["Survived", "Deceased"])

plt.show()


# The survival rates seem too irregular at different ages => it's hard to make meaningful age groups out of them.

# ### ___Sex___

# In[25]:


# Visualising Sex w.r.t. Survival:

sex_survival_cnt = train_data.groupby("Sex")["Survived"].value_counts()

plt.figure(figsize=(12, 8))
sns.countplot(data=train_data, x="Sex", hue="Survived")

for i, sex in enumerate(["male", "female"]):
    deceased_percent = (sex_survival_cnt[sex][0] / sex_survival_cnt[sex].sum() * 100).round(2)
    survival_percent = (sex_survival_cnt[sex][1] / sex_survival_cnt[sex].sum() * 100).round(2)
    
    plt.text(x=-0.25 + i, y=sex_survival_cnt[sex][0] + 3, s =f"{deceased_percent}%")
    plt.text(x=0.13 + i, y=sex_survival_cnt[sex][1] + 3, s=f"{survival_percent}%")

plt.title("Survival Distribution among men and women")
plt.legend(["Deceased", "Survived"])

plt.show()


# More than __80% of men died__ whereas almost __75% of the women survived__.
# 
# Therefore, __sex__ is a distinct predictor of survival and no feature engineering is required.

# ### ___Passenger class (Pclass)___

# In[26]:


# Visualising Pclass w.r.t. Survival:

pclass_survival_cnt = train_data.groupby("Pclass")["Survived"].value_counts()

plt.figure(figsize=(12, 10))
sns.countplot(data=train_data, x="Pclass", hue="Survived")

for i in range(3):
    deceased_percent = (pclass_survival_cnt[i + 1][0] / pclass_survival_cnt[i + 1].sum() * 100).round(2)
    survival_percent = (pclass_survival_cnt[i + 1][1] / pclass_survival_cnt[i + 1].sum() * 100).round(2)
    
    plt.text(x = -0.27 + i, y=pclass_survival_cnt[i + 1][0] + 3, s=f"{deceased_percent}%")
    plt.text(x = 0.12 + i, y=pclass_survival_cnt[i + 1][1] + 3, s=f"{survival_percent}%")
    
plt.title("Survival Distribution among different Pclasses")
plt.xticks([0, 1, 2], ["First-Class", "Second-Class", "Third-Class"])
plt.legend(["Deceased", "Survived"])

plt.show()


# More than __75% of third-class passengers died__ whereas more than __60% of first-class passengers survived__.
# 
# The higher the passenger class, the higher is the survival rate and vice versa. Therefore, __passenger class__ is a distinct predictor of survival and no feature engineering is required.

# ### ___City of Embarkment___

# In[27]:


# Visualising Embarked w.r.t Survival:

emb_survival_cnt = train_data.groupby("Embarked")["Survived"].value_counts()

plt.figure(figsize=(12, 10))
sns.countplot(data=train_data, x="Embarked", hue="Survived")

for i, city_code in enumerate(['S', 'C', 'Q']):
    deceased_percent = (emb_survival_cnt[city_code][0] / emb_survival_cnt[city_code].sum() * 100).round(2)
    survival_percent = (emb_survival_cnt[city_code][1] / emb_survival_cnt[city_code].sum() * 100).round(2)
    
    plt.text(x = -0.27 + i, y=emb_survival_cnt[city_code][0] + 3, s=f"{deceased_percent}%")
    plt.text(x = 0.12 + i, y=emb_survival_cnt[city_code][1] + 3, s=f"{survival_percent}%")

plt.title("Survival Distribution based on City of Embarkment")
plt.xticks([0, 1, 2], ["Southampton", "Cherbourg", "Queenstown"])
plt.xlabel("City of Embarkment")
plt.legend(["Deceased", "Survived"])

plt.show()


# Almost __twice as many passengers from Southampton died as compared to those who survived__. Passengers from __Queenstown__ also have a __poor survival rate__ at around __39%__. Passengers from __Cherbourg__ have the highest __survival rate at 55%__.
# 
# Embarkment also seems like a distinct predictor of survival and no feature engineering is required.

# ### ___Cabin___

# In[28]:


pd.concat([
        train_data["Cabin"].value_counts().rename("train"),
        test_data["Cabin"].value_counts().rename("test")
    ],
    axis=1
)


# In[29]:


# There are too many unique cabin values =>
# Let's create a feature based on the presence of a cabin value

for data in [train_data, test_data]:
    data["Is_Cabin_Present"] = [True if cabin != 'n/a' else False for cabin in data["Cabin"]]

train_data.head()


# In[30]:


# Visualising Cabin presence w.r.t. Survival

cab_survival_cnt = train_data.groupby("Is_Cabin_Present")["Survived"].value_counts()

plt.figure(figsize=(12, 10))
sns.countplot(data=train_data, x="Is_Cabin_Present", hue="Survived")

for i, presence in enumerate([False, True]):
    deceased_percent = (cab_survival_cnt[presence][0] / cab_survival_cnt[presence].sum() * 100).round(2)
    survival_percent = (cab_survival_cnt[presence][1] / cab_survival_cnt[presence].sum() * 100).round(2)
    
    plt.text(x=-0.27 + i, y=cab_survival_cnt[presence][0] + 3, s=f"{deceased_percent}%")
    plt.text(x=0.12 + i, y=cab_survival_cnt[presence][1] + 3, s=f"{survival_percent}%")

plt.title("Survival Distribution based on Presence of the Cabin Values")
plt.xticks([0, 1], ["No Cabin Value", "Cabin Value Presents"])
plt.xlabel("Cabin value presence")
plt.legend(["Deceased", "Survived"])
                                                
plt.show()


# __70%__ of the passengers with no cabin value __died__, whereas only __one-third__ of the passengers __with a cabin__ value __died__. 
# 
# Therefore, presence of a cabin value is a distinct predictor of survival and no futher feature engineering is required.

# ### ___Fare___

# In[31]:


# Visualising the Fare distribution w.r.t Survival

plt.figure(figsize=(20, 10))
sns.histplot(data=train_data, x="Fare", hue="Survived", multiple="stack")
plt.legend(["Survived", "Deceased"])

plt.show()


# In[32]:


# Group fares to find the relationship between Fare and Survival

quantile = train_data["Fare"].quantile(0.5)
a = pd.qcut(train_data['Fare'], [0., 0.5, 1.]).value_counts()

print(f"Low fares:\t0 - {quantile.round(2)}, cnt = {a}")
print(f"High fares:\t{quantile.round(2)} - {train_data['Fare'].max().round(2)}")


# In[33]:


train_data["Fare_Class"] = ["high" if fare > quantile else "low" for fare in train_data["Fare"]]
train_data.head()


# In[34]:


# Visualising the fare class distribution w.r.t Survival

fclass_survival_cnt = train_data.groupby("Fare_Class")["Survived"].value_counts()

plt.figure(figsize=(12, 10))
sns.countplot(data=train_data, x="Fare_Class", hue="Survived", order=["low", "high"])

for i, fclass in enumerate(["low", "high"]):
    deceased_percent = (fclass_survival_cnt[fclass][0] / fclass_survival_cnt[fclass].sum() * 100).round(2)
    survival_percent = (fclass_survival_cnt[fclass][1] / fclass_survival_cnt[fclass].sum() * 100).round(2)
    
    plt.text(x=-0.27 + i, y=fclass_survival_cnt[fclass][0] + 3, s=f"{deceased_percent}%")
    plt.text(x=0.12 + i, y=fclass_survival_cnt[fclass][1] + 3, s=f"{survival_percent}%")

plt.title("Survival Distribution based on Fare Classes")
plt.xticks([0, 1], ["Low Fares", "High Fares"])
plt.xlabel("Fare Class")
plt.legend(["Deceased", "Survived"])
                                                
plt.show()


# The __High fare class__ has __higher survival__ rate and vice versa.
# 
# Therefore, fare groups are a distinct predictor of surival and no further feature engineering is required.

# In[35]:


test_data["Fare_Class"] = ["high" if fare > quantile else "low" for fare in test_data["Fare"]]


# ### ___Number of Parents or Children (Parch) and Number of Siblings or Spouses (SibSp)___

# In[36]:


# Visualising number of parents or children (Parch) and number of siblings or spouses (SibSp) w.r.t Survival

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
for i, col in enumerate(["Parch", "SibSp"]):
    sns.countplot(data=train_data, x=col, hue ="Survived", ax=axs[i])
    
    axs[i].set_title(f"Survival Distribution based on {col}")
    axs[i].legend(["Deceased", "Survived"], loc="upper right")

plt.show()


# In[37]:


# Visualising the SibSp + Parch distribution w.r.t Survival

plt.figure(figsize = (12, 10))
sns.countplot(x=train_data["SibSp"] + train_data["Parch"] + 1, hue=train_data["Survived"])

plt.title(f"Survival Distribution based on Family count")
plt.xlabel("Family count")
plt.legend(["Deceased", "Survived"], loc="upper right")

plt.show()


# In[38]:


# Creating another feature Family_Size based on SibSp and Parch
    
train_data["Family_Size"] = (train_data["SibSp"] + train_data["Parch"] + 1) \
                                .apply(lambda x: "alone" if x == 1 else "medium" if x < 5 else "large")
train_data.head()


# In[39]:


# Visualising Family_Sizes w.r.t. Survival

fsize_survival_cnt = train_data.groupby("Family_Size")["Survived"].value_counts()

plt.figure(figsize=(12, 10))
sns.countplot(data=train_data, x="Family_Size", hue="Survived", order=["alone", "medium", "large"])

for i, fsize in enumerate(["alone", "medium", "large"]):
    deceased_percent = (fsize_survival_cnt[fsize][0] / fsize_survival_cnt[fsize].sum() * 100).round(2)
    survival_percent = (fsize_survival_cnt[fsize][1] / fsize_survival_cnt[fsize].sum() * 100).round(2)
    
    plt.text(x=-0.27 + i, y=fsize_survival_cnt[fsize][0] + 3, s=f"{deceased_percent}%")
    plt.text(x=0.12 + i, y=fsize_survival_cnt[fsize][1] + 3, s=f"{survival_percent}%")

plt.title("Survival Distribution based on Family size")
plt.xticks([0, 1, 2], ['No Family (Alone)', 'Medium Sized Family', 'Large Family'])
plt.xlabel("Family Size")
plt.legend(["Deceased", "Survived"])
                                                
plt.show()


# The relationship between __family sizes__ and survival becomes much __more distinct__ using family size groups (as opposed to using Parch and SibSp).
# 
# A passenger from a __large family__ has more than __80% chance of dying__ and passenger with __no family__ aboard has almost __70% chance of dying__. __Medium__ sized families have the highest __survival rate at 58%__.
# 
# Therefore, __family sizes__ (independently) are distinct predictors of survival and no further feature engineering is required.

# In[40]:


test_data["Family_Size"] = (test_data["SibSp"] + test_data["Parch"] + 1) \
                                .apply(lambda x: "alone" if x == 1 else "medium" if x < 5 else "large")


# In[41]:


# Drop columns that are no longer required

for data in [train_data, test_data]:
    data.drop(["PassengerId", "Name", "Ticket", "Parch", "SibSp", "Cabin"], axis=1, inplace=True)
    
train_data.head()


# In[42]:


test_data.dtypes


# ### ___Scaling & Encoding___

# In[43]:


for data in [train_data, test_data]:
    data["Pclass"] = data["Pclass"].astype("object")
    
numericals = train_data[["Age", "Fare", "Name_Length"]]
categoricals = train_data.select_dtypes(exclude=["int64", "float64"])


# In[44]:


numerical_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)
categorical_transformer = Pipeline(
    steps=[(
        'onehotenc', OneHotEncoder(handle_unknown='ignore', sparse=False)
    )]
)

ct = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numericals.columns),
        ("cat", categorical_transformer, categoricals.columns)
    ],
    remainder='passthrough'
)


# In[45]:


train_data, y_train = train_data.drop("Survived", axis=1), train_data["Survived"]

train_data = pd.DataFrame(ct.fit_transform(train_data), columns=ct.get_feature_names_out())
test_data = pd.DataFrame(ct.transform(test_data), columns=ct.get_feature_names_out())

train_data.head()


# ### ___Honorific___

# In[46]:


# Combine honorific features

for data in [train_data, test_data]:
    #Combine married women into a single feature
    data["cat__Married_Women"] = \
        data["cat__Honorific_Mrs."] \
        + data["cat__Honorific_Mme."] \
        + data["cat__Honorific_Ms."]

    # Combine unmarried women into a single feature
    data["cat__Unmarried_Women"] = \
        data["cat__Honorific_Miss."] \
        + data["cat__Honorific_Mlle."]

    # Combine the rarer honorifics into a single feature
    data["cat__Uncommon_Honorific"] = \
        data["cat__Honorific_Capt."] \
        + data["cat__Honorific_Col."] \
        + data["cat__Honorific_Don."] \
        + data["cat__Honorific_Dr."] \
        + data["cat__Honorific_Jonkheer."] \
        + data["cat__Honorific_Lady."] \
        + data["cat__Honorific_Major."] \
        + data["cat__Honorific_Sir."] \
        + data["cat__Honorific_the"] \
        + data["cat__Honorific_Rev."]


# In[47]:


# Drop all honorific features that have been combined into a new feature

for data in [train_data, test_data]:
    data.drop([
            "cat__Honorific_Miss.", "cat__Honorific_Mlle.", "cat__Honorific_Mrs.",
            "cat__Honorific_Mme.", "cat__Honorific_Ms.", "cat__Honorific_Lady.", "cat__Honorific_the",
            "cat__Honorific_Rev.", "cat__Honorific_Jonkheer.", "cat__Honorific_Capt.",
            "cat__Honorific_Col.", "cat__Honorific_Major.", "cat__Honorific_Don.", "cat__Honorific_Sir.",
            "cat__Honorific_Dr."
        ], 
        axis=1,
        inplace=True
)

train_data.head()


# In[48]:


# Check correlation

corr = train_data.corr()

plt.figure(figsize=(25, 25))
sns.heatmap(corr, annot=True, linewidths=0.5)

plt.show()


# There's some multicollinearity in the data. However in this notebook, I won't take it into account.
# 
# Maybe later I will try to eliminate multicollinearity in the data...

# # Model (Random Forest)

# In[49]:


X_train, X_test = train_data, test_data


# In[50]:


# Search parameters for the optimization

search_parameters = {
    "n_estimators": [50, 100, 400, 700, 1000],
    "criterion": ["entropy", "gini"],
    "min_samples_leaf": [1, 5, 10],
    "min_samples_split" : [2, 4, 10, 12, 16]
}


# In[51]:


rf_estimator = RandomForestClassifier(oob_score=True, random_state=0xDEADBEEF)
grid = GridSearchCV(
    estimator=rf_estimator,
    param_grid=search_parameters,
    scoring="accuracy",
    cv=3,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best parameters for random forest classifier:\n{grid.best_params_}")


# In[52]:


# Model based on the optimal paramters given by GS

rf_model = RandomForestClassifier(
    **grid.best_params_,
    oob_score=True,
    random_state=0xDEADBEEF, 
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_model.oob_score_


# In[53]:


y_pred = rf_model.predict(X_test)


# In[54]:


# Submission

def create_submission(pred):
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
    
    passenger_id = pd.read_csv("/kaggle/input/titanic/test.csv")["PassengerId"]
    survived = pd.Series(pred).rename("Survived")

    output = pd.concat([passenger_id, survived], axis=1)
    output.to_csv("submission.csv", index=False)
    print("Submission was successfully saved!")

create_submission(y_pred)

