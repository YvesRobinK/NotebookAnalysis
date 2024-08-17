#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px;
#             border : black solid;
#             background-color: ##FFFFFF;
#             font-size:160%;
#             text-align: left">
# 
# <h1 style='; border:0; border-radius: 10px; text-shadow: 1px 1px black; font-weight: bold; color:#10928C'><center> TİTANİC SURVİVAL PREDİCTİON </center></h1>

# **1. Download important modules**
# 
# **2. Read train and test datasets**
# 
# **3. Fill the NAN values with mean and common values**
# 
# **4. Divide the values of the variables of "age" and "fare" into different groups**
# 
# **5. Encode "Sex" and "Embarked" variables**
# 
# **6. Creat new features out of other features**
# 
# **7. Select variables to send ML models**
# 
# **8. Build ML model and get results**

# In[1]:


#Here we download the necessary modules to our program
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Here we read train and test datasets
train_set = pd.read_csv("../input/titanic/train.csv")
test_set = pd.read_csv("../input/titanic/test.csv")


# In[3]:


#Here we fill the NAN values of some variables with the mean and common values  of those variables.
train_set["Age"].fillna(train_set["Age"].mean(), inplace = True)
test_set["Age"].fillna(test_set["Age"].mean(), inplace = True)
train_set["Fare"].fillna(0, inplace = True)
train_set["Fare"] = train_set["Fare"].astype("int64")
test_set["Fare"].fillna(0, inplace = True)
test_set["Fare"] = test_set["Fare"].astype("int64")
common_value = "S"
train_set["Embarked"].fillna(common_value, inplace = True)
test_set["Embarked"].fillna(common_value, inplace = True)


# In[4]:


#Here we divide the values of the variables "age" and "fare" into different groups
age_data = [train_set, test_set]
fare_data = [train_set, test_set]
for dat in age_data:
    dat['Age'] = dat['Age'].astype("int64")
    dat.loc[dat['Age'] <= 12, 'Age'] = 0
    dat.loc[(dat['Age'] > 12) & (dat['Age'] <= 19), 'Age'] = 1
    dat.loc[(dat['Age'] > 19) & (dat['Age'] <= 23), 'Age'] = 2
    dat.loc[(dat['Age'] > 23) & (dat['Age'] <= 28), 'Age'] = 3
    dat.loc[(dat['Age'] > 28) & (dat['Age'] <= 34), 'Age'] = 4
    dat.loc[(dat['Age'] > 34) & (dat['Age'] <= 41), 'Age'] = 5
    dat.loc[(dat['Age'] > 41) & (dat['Age'] <= 67), 'Age'] = 6
    dat.loc[dat['Age'] > 67, 'Age'] = 6
for dataf in fare_data:
    dataf.loc[dataf['Fare'] <= 7.91, 'Fare'] = 0
    dataf.loc[(dataf['Fare'] > 7.91) & (dataf['Fare'] <= 14.454), 'Fare'] = 1
    dataf.loc[(dataf['Fare'] > 14.454) & (dataf['Fare'] <= 31), 'Fare'] = 2
    dataf.loc[(dataf['Fare'] > 31) & (dataf['Fare'] <= 99), 'Fare'] = 3
    dataf.loc[(dataf['Fare'] > 99) & (dataf['Fare'] <= 250), 'Fare'] = 4
    dataf.loc[dataf['Fare'] > 250, 'Fare'] = 5
    dataf['Fare'] = dataf['Fare'].astype(int)


# In[5]:


#Here we encode our categorical variables "Sex" and "Embarked" with the LabelEncoder function
lbe = LabelEncoder()
train_set["Sex"] = lbe.fit_transform(train_set["Sex"])
test_set["Sex"] = lbe.fit_transform(test_set["Sex"])
train_set["Embarked"] = lbe.fit_transform(train_set["Embarked"])
test_set["Embarked"] = lbe.fit_transform(test_set["Embarked"])


# In[6]:


#I will create new feature out of other features (Age, Pclass)
data = [train_set, test_set]
for datac in data:
    datac['Age_Class']= datac['Age']* datac['Pclass']
for datas in data:
    datas["He_she_not_alone"] = datas["SibSp"]
    datas["He_she_not_alone"] = np.where(datas["He_she_not_alone"] < 1, 1, 0)

for data_rel in data:
    data_rel['relatives'] = data_rel['SibSp'] + data_rel['Parch']
for data_fpp in data:
    data_fpp['Fare_per_person'] = data_fpp['Fare']/(data_fpp['relatives']+1)
    data_fpp['Fare_per_person'] = data_fpp['Fare_per_person'].astype("int64")


# In[7]:


#Here we select the variables y_train and x_train and x_test to be sent to the ML model from the dataset. We exclude some variables when performing selection work.  Because these variables are not important for the performance of the model.
y_train = train_set["Survived"]
x_train = train_set.drop(["Survived", "PassengerId", "Name", "Cabin", "Ticket",], axis = 1)
x_test= test_set.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1)


# In[8]:


#Here we build our model with RandomForestClassifier, the ensebmle learning method, and send the x_test dataset to the model
rf = RandomForestClassifier(n_estimators = 100)
rf_model = rf.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
#Here we place the "PassengerId" variable and predictions of our model in a new dataset
df = pd.DataFrame({"PassengerId": test_set["PassengerId"], "Survived": y_pred})
df.head(n = 10)


# In[9]:


df.to_csv("submission.csv", index = False)


# # Thank you all 🙂
