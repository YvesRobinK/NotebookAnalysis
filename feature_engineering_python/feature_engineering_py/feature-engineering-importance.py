#!/usr/bin/env python
# coding: utf-8

# # Imports and Defines

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

TRAIN_PATH = "../input/titanic/train.csv"

ID = "PassengerId"
TARGET = "Survived"

train = pd.read_csv(TRAIN_PATH)

def checkNull_fillData(df):
    for col in df.columns:
        if len(df.loc[df[col].isnull() == True]) != 0:
            if df[col].dtype == "float64" or df[col].dtype == "int64":
                df.loc[df[col].isnull() == True,col] = df[col].mean()
            else:
                df.loc[df[col].isnull() == True,col] = df[col].mode()[0]
                
checkNull_fillData(train)
train.head(2)


# # feature engineering

# In[2]:


def feature_cut_engineering(df,COL,DEVIDE_NUM):
    df[COL + "_Cut_" + str(DEVIDE_NUM)] = pd.cut(df[COL].values,DEVIDE_NUM,labels=np.arange(0,DEVIDE_NUM))
    df[COL + "_Cut_" + str(DEVIDE_NUM)] = df[COL +"_Cut_" + str(DEVIDE_NUM)].astype(int)     

def feature_log_engineering(df,COL):
    df[COL + "_Log"] = np.log(df[COL])
    
feature_cut_engineering(train,"Age",50)
feature_cut_engineering(train,"Age",20)
feature_cut_engineering(train,"Age",10)
feature_cut_engineering(train,"Age",5)
feature_log_engineering(train,"Age")

train.head(2)


# # feature importance

# In[3]:


def showFeatureImportance(df):
    FI = np.abs(df.corr()[TARGET]).sort_values()
    df_FI = pd.DataFrame(FI)
    LastFI= df_FI[::-1].T.drop([ID,TARGET],axis=1)
    print(LastFI.T)

    plt.bar(LastFI.T.index, LastFI.T[TARGET])
    plt.title('Feature Importance', fontsize=20)
    plt.xlabel('Feature', fontsize=18)
    plt.ylabel('Importance', fontsize=18)
    plt.xticks(LastFI.T.index, LastFI.T.index, fontsize=10)
    plt.show()

showFeatureImportance(train)

