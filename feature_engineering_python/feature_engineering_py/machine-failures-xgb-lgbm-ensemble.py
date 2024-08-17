#!/usr/bin/env python
# coding: utf-8

# <img src="https://raw.githubusercontent.com/IqmanS/Machine-Learning-Notebooks/main/XGB_HyperParameter_Tuning/banner.jpg">
# 
# # **Binary Classification of Machine Failures**
# 
# ---

# # Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('dark_background')
print("Setup Complete")


# # Importing Datasets

# In[2]:


train_path = "../input/playground-series-s3e17/train.csv"
orig_path = "../input/machine-failure-predictions/machine failure.csv"
test_path = "../input/playground-series-s3e17/test.csv"

train_data = pd.read_csv(train_path,index_col="id")
orig_data = pd.read_csv(orig_path,index_col="UDI")
test_data = pd.read_csv(test_path,index_col="id")

train_data = train_data.append(orig_data)
train_data.head()


# In[3]:


print(train_data.info(),test_data.info())


# In[4]:


# train_data.drop(["Product ID"],inplace=True,axis=1)
# test_data.drop(["Product ID"],inplace=True,axis=1)


# # Data Cleaning

# In[5]:


train_data.columns = ['Product_ID', 'Type', 'Air_Temp_K', 'Process_Temp_K', 'Rot_Speed', 'Torque',
       'Tool_Wear', 'Machine_Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
test_data.columns = ['Product_ID' ,'Type', 'Air_Temp_K', 'Process_Temp_K', 'Rot_Speed', 'Torque',
       'Tool_Wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

catDTypeCols = ['Type']


# In[6]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

train_data["Type"] = encoder.fit_transform(train_data["Type"])
test_data["Type"] = encoder.transform(test_data["Type"])
train_data.head()


# # Feature Engineering

# In[7]:


train_data["Temp_Diff"] = train_data["Process_Temp_K"]-train_data["Air_Temp_K"]
test_data["Temp_Diff"] = test_data["Process_Temp_K"]-test_data["Air_Temp_K"]

train_data["Air_Temp_C"] = train_data["Air_Temp_K"]-273
test_data["Air_Temp_C"] = test_data["Air_Temp_K"]-273

train_data["Process_Temp_C"] = train_data["Process_Temp_K"]-273
test_data["Process_Temp_C"] = test_data["Process_Temp_K"]-273

train_data["Power"] = train_data["Torque"]*train_data["Rot_Speed"]
test_data["Power"] = test_data["Torque"]*test_data["Rot_Speed"]

train_data["Temp_Ratio_K"] = train_data["Air_Temp_K"]/train_data["Process_Temp_K"]
test_data["Temp_Ratio_K"] = test_data["Air_Temp_K"]/test_data["Process_Temp_K"]

train_data["Temp_Ratio_C"] = train_data["Air_Temp_C"]/train_data["Process_Temp_C"]
test_data["Temp_Ratio_C"] = test_data["Air_Temp_C"]/test_data["Process_Temp_C"]

train_data["Product_ID"] = pd.to_numeric(train_data["Product_ID"].str.slice(start=1))
test_data["Product_ID"] = pd.to_numeric(test_data["Product_ID"].str.slice(start=1))

train_data["Failures"] = (train_data["TWF"] + train_data["HDF"] + train_data["PWF"] + train_data["OSF"] + train_data["RNF"])
test_data["Failures"] = (test_data["TWF"] + test_data["HDF"] + test_data["PWF"] + test_data["OSF"] + test_data["RNF"])

train_data["Tool_Wear_Speed"] = train_data["Tool_Wear"] * train_data["Rot_Speed"]
test_data["Tool_Wear_Speed"] = test_data["Tool_Wear"] * test_data["Rot_Speed"]

train_data["TorquexWear"] = train_data["Torque"] * train_data["Tool_Wear"]
test_data["TorquexWear"] = test_data["Torque"] * test_data["Tool_Wear"]


# # Exploratory Data Analysis

# In[8]:


features = [i for i in train_data.columns]
corr = train_data[features].corr(numeric_only=False)
plt.figure(figsize = (18,10))
sns.heatmap(corr, cmap = 'flare', annot = True,vmin=0);
plt.show()


# In[9]:


col=0
plotCols=[]
for i in train_data.columns:
    if (len(train_data[i].unique())>3):
        plotCols.append(i)
        plt.figure(figsize=(8,2))
        plt.title(i)
        sns.boxplot(train_data[i],color=sns.color_palette("rocket")[col],orient="h")
        plt.xticks(rotation=90);
    col = (col+1)%5


# In[10]:


for i in plotCols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_data,x =i,hue="Machine_Failure",bins=40,kde=True,palette="plasma");
    plt.show()


# # Training Model 

# In[11]:


from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


# In[12]:


cols = [i for i in train_data.columns if i!="Machine_Failure"]
seed = np.random.seed(0)

X = train_data[cols]
y = train_data["Machine_Failure"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=seed)


# ## XGB

# In[13]:


# From GridSearchCV
best_params_xgb = {
    'gamma': 0.2, 
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 2000}

#From hyperopt
hypopt_params = {
    'colsample_bylevel': 0.1,
    'colsample_bynode': 0.1,
    'colsample_bytree': 0.1,
    'gamma': 0.8622315538845127,
    'learning_rate': 0.13454749501702748,
    'max_depth': 2,
    'min_child_weight': 3.0,
    'n_estimators': 36,
    'subsample': 0.9104854458851901}


# In[14]:


xgbmodel = XGBClassifier(**best_params_xgb,random_state=seed,eval_metric= "auc",tree_method='gpu_hist', predictor='gpu_predictor')
xgbmodel.fit(X,y)
print("-"*100)
print("ROC Area Under Curve of XGB:",roc_auc_score(y_test, xgbmodel.predict_proba(X_test)[:,1]))


# In[15]:


df = pd.DataFrame()
df["imp"] = xgbmodel.feature_importances_
df["columns"] = X.columns
df.sort_values(by=["imp"], axis=0, ascending=False, inplace=True)
sns.barplot(x=df["imp"],y=df["columns"],palette="flare");


# ## LGBMClassifier

# In[16]:


# From GridSearchCV
best_params_lgbm = {'learning_rate': 0.0045,
 'max_depth': 4,
 'min_child_weight': 3,
 'n_estimators': 3500,
 'subsample':0.6,
 'colsample_bytree':0.7 
}


# In[17]:


lgbmmodel = LGBMClassifier(**best_params_lgbm,random_state=seed,device='gpu')
lgbmmodel.fit(X,y)
print("-"*100)
print("ROC Area Under Curve of LGBM:",roc_auc_score(y_test, lgbmmodel.predict_proba(X_test)[:,1]))


# In[18]:


df = pd.DataFrame()
df["imp"] = lgbmmodel.feature_importances_
df["columns"] = X.columns
df.sort_values(by=["imp"], axis=0, ascending=False, inplace=True)
sns.barplot(x=df["imp"],y=df["columns"],palette="flare");


# ## Voting Classifier Ensemble

# In[19]:


import random
wts = [random.uniform(0.8,1),random.uniform(0.4,0.6)]
wts


# In[20]:


vcmodel = VotingClassifier([("lgbm",lgbmmodel),("xgb",xgbmodel)],voting="soft",weights=wts,verbose=False)
vcmodel.fit(X,y)


# # Creating 'submission.csv' 

# In[21]:


predictions = vcmodel.predict_proba(test_data)[:,1]


# In[22]:


submission = test_data.copy()

colsToDrop = [i for i in submission.columns]
submission.drop(colsToDrop,axis=1,inplace=True)
submission["Machine failure"] = predictions


# In[23]:


submission.head()


# In[24]:


submission.to_csv("submission.csv",index=True,header=True)

