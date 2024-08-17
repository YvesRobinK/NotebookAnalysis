#!/usr/bin/env python
# coding: utf-8

# In[205]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[206]:


import os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msg
from scipy import stats
from sklearn.model_selection import train_test_split , cross_val_score , RandomizedSearchCV
from sklearn.preprocessing import StandardScaler ,MinMaxScaler ,LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# In[207]:


df = pd.read_csv("/kaggle/input/titanic/train.csv")
df1 = pd.read_csv("/kaggle/input/titanic/test.csv")
sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[208]:


df


# In[209]:


msg.matrix(df)


# In[210]:


msg.matrix(df1)


# # *We will first Analysis Null Values for all the Column .*

# ## Analysis on Age col

# In[211]:


sb.kdeplot(df['Age'])


# In[212]:


sb.distplot(df['Age'])


# In[213]:


print("Skewness of Age is " , stats.skew(df['Age'].dropna()))
print("Kurtosis of Age is " , stats.kurtosis(df['Age'].dropna()))


# In[214]:


df.corr()['Age'].sort_values(ascending = False)


# In[215]:


print(df['Age'].isna().sum())
print(df1['Age'].isna().sum())


# In[216]:


sb.boxplot(df['Age'])


# In[217]:


sb.boxplot(df1['Age'])


# In[218]:


df['Age'].fillna(df['Age'].median() , inplace = True)
df1['Age'].fillna(df1['Age'].median() , inplace = True)


# In[219]:


sb.distplot(df['Age'] , label = "train_age" , color = "red")
sb.distplot(df1['Age'] , label = "test_age" , color = "blue")


# ## Analysis on Fare col

# In[220]:


sb.distplot(df['Fare'])


# In[221]:


sb.boxplot(df['Fare'])


# In[222]:


sb.distplot(df1['Fare'])


# In[223]:


sb.boxplot(df1['Fare'])


# In[224]:


print("Skewness of Fare is " , stats.skew(df1['Fare'].dropna()))
print("Kurtosis of Fare is " , stats.kurtosis(df1['Fare'].dropna()))


# In[225]:


print(df['Fare'].value_counts().sort_values(ascending = True))


# In[226]:


print(df1['Fare'].value_counts().sort_values(ascending = True))


# In[227]:


df = df[df['Fare'] < 100]    #remove outlier


# In[228]:


sb.distplot(df['Fare'])


# In[229]:


print("Skewness of Fare is " , stats.skew(df['Fare'].dropna()))
print("Kurtosis of Fare is " , stats.kurtosis(df['Fare'].dropna()))


# In[230]:


print(df['Fare'].isna().sum())


# In[231]:


df1['Fare'].fillna(df1['Fare'].median() , inplace = True)


# ## Analysis on Cabin col

# In[232]:


df['Cabin']


# In[233]:


df['Cabin'].unique()


# In[234]:


print((df['Cabin'].isna().sum() / len(df))*100)


# In[235]:


# drop
df.drop(['Cabin'],axis =1 , inplace = True)
df1.drop(['Cabin'],axis =1 , inplace = True)


# In[236]:


df.head()


# ## Analysis on Embarked col

# In[237]:


df['Embarked']


# In[238]:


print(df['Embarked'].unique())
print(df['Embarked'].nunique())


# In[239]:


df['Embarked'].fillna(method = 'ffill' , inplace = True)


# In[240]:


df['Embarked'].isna().sum()


# In[241]:


df1['Embarked'].fillna(method = 'ffill' , inplace = True)


# # *Let's do Feature Engineering (EDA)*

# In[242]:


df.head()


# In[243]:


df.drop(['PassengerId'],axis =1 , inplace =True)
df1.drop(['PassengerId'],axis =1 , inplace =True)


# ## Analysis on Name col

# In[244]:


df['Name']


# In[245]:


df = df.reset_index(drop = True)
for i in range(len(df)):
    if ("Mr" in df.loc[i , 'Name']) or ("Master" in df.loc[i , 'Name']):
        df.loc[i , "gender_category"] = 0
    else:
        df.loc[i , 'gender_category'] = 1


# In[246]:


#df = df.reset_index(drop = True)
for i in range(len(df1)):
    if ("Mr" in df1.loc[i , 'Name']) or ("Master" in df1.loc[i , 'Name']):
        df1.loc[i , "gender_category"] = 0
    else:
        df1.loc[i , 'gender_category'] = 1


# In[247]:


df.drop(['Name'],axis =1 , inplace = True)
df1.drop(['Name'],axis =1 , inplace = True)


# ## Analysis on Sex

# In[248]:


df['Sex']


# In[249]:


sb.countplot(df['Sex'])


# In[250]:


sb.countplot(df1['Sex'])


# In[251]:


le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df1['Sex'] = le.fit_transform(df1['Sex'])


# In[252]:


df['Sex'].unique()


# ## Analysis on Ticket

# In[253]:


df['Ticket']


# In[254]:


df.drop(['Ticket'],axis =1 , inplace = True)
df1.drop(['Ticket'],axis =1 , inplace = True)


# ## Analysis on Embarked

# In[255]:


sb.countplot(df['Embarked'])


# In[256]:


sb.countplot(df1['Embarked'])


# In[257]:


df['Embarked'] = df['Embarked'].replace({"S":3 , "C":2 , "Q":1})
df1['Embarked'] = df1['Embarked'].replace({"S":3 , "C":2 , "Q":1})


# In[258]:


df.head()


# In[259]:


df1.head()


# In[260]:


plt.figure(figsize = (10,8))
sb.heatmap(df.corr() , annot = True)


# In[261]:


df.corr()['Survived'].sort_values(ascending = False)


# # Splitting , Cross Validation and Model Building

# In[262]:


x = df.drop(['Survived'] ,axis =1)
y = df['Survived']


# In[263]:


x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size= 0.27 , random_state =100)


# In[264]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Model - RandomForest

# In[265]:


rf = RandomForestClassifier()


# In[266]:


result = cross_val_score(rf ,x,y, cv =10 ,scoring = 'accuracy',n_jobs =-1 , verbose =1)


# In[267]:


print(result.mean())


# In[268]:


rf.fit(x_train , y_train)


# In[269]:


rf_pred = rf.predict(x_test)


# In[270]:


print("Acc of RandomForest without HyperParameter tuning is " , accuracy_score(rf_pred , y_test))


# ## Model - XGBoostClassifier

# In[271]:


xgb = XGBClassifier()
result = cross_val_score(xgb ,x,y, cv =10 ,scoring = 'accuracy',n_jobs =-1 , verbose =1)


# In[272]:


print(result.mean())


# In[273]:


print(xgb.fit(x_train , y_train))
xgb_pred = xgb.predict(x_test)


# In[274]:


print("Acc of XGBClassifier without HyperParameter tuning is " , accuracy_score(xgb_pred , y_test))


# ## HyperParam tuning for RandomForest

# In[275]:


rf_param = {'n_estimators':list(range(100,500)) , 
         'max_depth':list(range(1,10)) , 
         'criterion':['gini','entropy'] ,
         'max_samples':list(range(1,10))    
}


# In[276]:


rscv = RandomizedSearchCV(rf ,param_distributions=rf_param ,  cv =5 , n_iter=10 , scoring = 'accuracy',n_jobs =-1 , verbose =10)


# In[277]:


rscv.fit(x,y)


# In[278]:


print(rscv.best_score_)
print(rscv.best_estimator_)
print(rscv.best_index_)
print(rscv.best_params_)


# In[279]:


# Fit and predict
rf = RandomForestClassifier(criterion='entropy', max_depth=3, max_samples=9,
                       n_estimators=119)
print(rf.fit(x_train , y_train))
print("Accuracy score of RF after tuning is :", accuracy_score(rf.predict(x_test) , y_test))


# ## HyperParam Tuning for XGBoost classifier

# In[280]:


xgb_param = {'n_estimators':list(range(100,500)) , 
         'max_depth':list(range(1,10)) , 
         'learning_rate':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.05,0.09] ,
         'min_child_weight ':list(range(1,10))    
}

rscv = RandomizedSearchCV(xgb ,param_distributions=xgb_param ,  cv =5 , n_iter=10 , scoring = 'accuracy',n_jobs =-1 , verbose =10)
rscv.fit(x,y)


# In[281]:


print(rscv.best_score_)
print(rscv.best_estimator_)
print(rscv.best_index_)
print(rscv.best_params_)


# In[282]:


# Fit and predict
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.09, max_delta_step=0, max_depth=2,
               min_child_weight =9,
              monotone_constraints='()', n_estimators=448, n_jobs=0,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)

print(xgb.fit(x_train , y_train))
print("Accuracy score of XGB after tuning is :", accuracy_score(xgb.predict(x_test) , y_test))


# ## Acc of RF - 75.33
# ## Acc of XGB - 83.2
# 
# ### Final Model selected - XGB

# In[283]:


df1


# In[284]:


main_pred = xgb.predict(df1)


# In[285]:


main_pred


# ## Make Submission file

# In[286]:


sub


# In[287]:


sub['Survived'] = main_pred


# In[288]:


sub.to_csv("Main_Submission.csv" , index = False)


# ## If you liked this Notebook , please upvote. Gives  Motivation to make new Notebooks :)

# In[ ]:




