#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# 
# <div align='center'>
#     <h1>Women in Data Science (WiDS Datathon) 2023</h1>
#     
# </div>

# ![image.png](attachment:c393c576-a9ee-4612-aa88-05e10137b6de.png)

# Problem Statement: Extreme weather events are sweeping the globe and range from heat waves, wildfires and drought to hurricanes, extreme rainfall and flooding. These weather events have multiple impacts on agriculture, energy, transportation, as well as low resource communities and disaster planning in countries across the globe.
# 
# Accurate long-term forecasts of temperature and precipitation are crucial to help people prepare and adapt to these extreme weather events. Currently, purely physics-based models dominate short-term weather forecasting. But these models have a limited forecast horizon. The availability of meteorological data offers an opportunity for data scientists to improve sub-seasonal forecasts by blending physics-based forecasts with machine learning. Sub-seasonal forecasts for weather and climate conditions (lead-times ranging from 15 to more than 45 days) would help communities and industries adapt to the challenges brought on by climate change.

# In[2]:


#Import Liberay 


# In[3]:


import numpy as np
import pandas as pd
        
#import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objs as go

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt


# In[4]:


train = pd.read_csv('/kaggle/input/widsdatathon2023/train_data.csv')
test = pd.read_csv('/kaggle/input/widsdatathon2023/test_data.csv')
test_org=test.copy()
train.head()


# In[5]:


test.head()


# In[6]:


#target valriable
train["contest-tmp2m-14d__tmp2m"]


# In[7]:


train.info()


# Dataset has  375734 rows and 246 columns

# In[8]:


#checking null values in train dataset


# In[9]:


round(train.isnull().sum()*100/len(train),2).sort_values(ascending=False)[:10]


# In[10]:


## checkning null values in test dataset
round(test.isnull().sum()*100/len(train),2).sort_values(ascending=False)[:10]


# There is no null values in test dataset

# #Create new columns

# In[11]:


train['year']=pd.DatetimeIndex(train['startdate']).year 
train['month']=pd.DatetimeIndex(train['startdate']).month 
train['day']=pd.DatetimeIndex(train['startdate']).day
test['year']=pd.DatetimeIndex(test['startdate']).year 
test['month']=pd.DatetimeIndex(test['startdate']).month 
test['day']=pd.DatetimeIndex(test['startdate']).day


# #Handle null values 

# In[12]:


train['nmme0-prate-34w__ccsm30'] = train['nmme0-prate-34w__ccsm30'].fillna(train['nmme0-prate-34w__ccsm30'].mean())
train['nmme0-tmp2m-34w__ccsm30'] = train['nmme0-tmp2m-34w__ccsm30'].fillna(train['nmme0-tmp2m-34w__ccsm30'].mean())
train['ccsm30'] = train['ccsm30'].fillna(train['ccsm30'].mean())
train['nmme0-prate-56w__ccsm30'] = train['nmme0-prate-56w__ccsm30'].fillna(train['nmme0-prate-56w__ccsm30'].mean())
train['nmme-tmp2m-56w__ccsm3'] = train['nmme-tmp2m-56w__ccsm3'].fillna(train['nmme-tmp2m-56w__ccsm3'].mean())
train['nmme-prate-56w__ccsm3'] = train['nmme-prate-56w__ccsm3'].fillna(train['nmme-prate-56w__ccsm3'].mean())
train['nmme-tmp2m-34w__ccsm3'] = train['nmme-tmp2m-34w__ccsm3'].fillna(train['nmme-tmp2m-34w__ccsm3'].mean())
train['nmme-prate-34w__ccsm3'] = train['nmme-prate-34w__ccsm3'].fillna(train['nmme-prate-34w__ccsm3'].mean())


# In[13]:


#rechecking null values 
round(train.isnull().sum()*100/len(train),2).sort_values(ascending=False)[:10]


# In[14]:


## remove the irrelevant columns
train=train.drop(['index'],axis=1)
train=train.drop(['startdate'],axis=1)
test=test.drop(['index'],axis=1)
test=test.drop(['startdate'],axis=1)


# In[15]:


#converting object type to category
#train["climateregions__climateregion"]=train["climateregions__climateregion"].astype('category')
train=train.drop(['climateregions__climateregion'],axis=1)
test=test.drop(['climateregions__climateregion'],axis=1)


# In[16]:


train.describe()


# ## Distribution of train and test dataset

# 

# In[17]:


# graphs of longitudinal wind at 250 millibars
t1=train.loc[:, train.columns.str.startswith('nmme0-tmp2m-34w__c')]
t1.plot(subplots=True, figsize=(50,60))


# In[18]:


t2=test.loc[:, test.columns.str.startswith('nmme0-tmp2m-34w__c')]
t2.plot(subplots=True, figsize=(50,60))


# #Checking the distribution of few variables 
#   

# In[19]:


import seaborn as sns
def plot_distribution(var):
    plt.subplots(figsize=(14,7))
    sns.distplot(x=train[var], color='blue', kde=True)
    sns.distplot(x=test[var], color='red', kde=True)
    plt.title(var, weight="bold",fontsize=20, pad=20)
    plt.ylabel("Count", weight="bold", fontsize=15)
    plt.xlabel(var, weight="bold", fontsize=12)
    plt.show()
plot_distribution("contest-rhum-sig995-14d__rhum")


# In[20]:


plot_distribution("wind-vwnd-925-2010-20")


# In[21]:


#Model building using 


# ## Type of Time Series data

# #Types of Time Series Data
# ![image.png](attachment:b68a6fd8-b08e-4f31-87d6-307d6c091f9c.png)
# 
# Full article is here: https://engineering.99x.io/time-series-forecasting-in-machine-learning-3972f7a7a467

# In[22]:


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
target_p=train["contest-tmp2m-14d__tmp2m"]
train["contest-tmp2m-14d__tmp2m"].plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Target')
plt.show()


# In[23]:


target="contest-tmp2m-14d__tmp2m"
y=train[target]
x=train.drop([target],axis=1)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[25]:


import xgboost as xgb
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       tree_method = 'gpu_hist',
                       n_estimators=10000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01, gpu_id=0)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)


# ## Feature Importance

# In[26]:


fi = pd.DataFrame(data=reg.feature_importances_[0:5],
             index=reg.feature_names_in_[0:5],
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()


# ## Prediction

# In[27]:


X_test_prediction = reg.predict(X_test)


# ## Error on test data set

# In[28]:


from sklearn.metrics import mean_squared_error
score = np.sqrt(mean_squared_error(y_test, X_test_prediction))
print(f'RMSE Score on Test set: {score:0.2f}')


# In[29]:


predictions=reg.predict(test)


# ### Tests where error is maximum

# In[30]:


X_test_error = np.abs(y_test - X_test_prediction)
X_test_error.sort_values(ascending=False).head(10)


# In[31]:


test_org[target] = predictions


# In[32]:


test_org[[target,"index"]].to_csv("xgboost.csv",index  = False)


# # Catboost Algorithm 

# In[33]:


get_ipython().system('pip install catboost')


# In[34]:


from catboost import CatBoostRegressor
reg_catb  = CatBoostRegressor(n_estimators=2000,eval_metric='RMSE',learning_rate=0.1, random_seed= 1234)

reg_catb.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],verbose=100)
#reg_catb.fit(X_train, y_train, verbose=100)


# In[35]:


X_test_prediction_cat = reg_catb.predict(X_test)


# In[36]:


#error on test dataset
from sklearn.metrics import mean_squared_error
score = np.sqrt(mean_squared_error(y_test, X_test_prediction_cat))
print(f'RMSE Score on Test set: {score:0.2f}')


# In[37]:


# Top 10 errors
X_test['error_cat'] = np.abs(y_test - X_test_prediction_cat)
X_test['error_cat'].sort_values(ascending=False).head(10)


# In[38]:


test11=test_org.copy()

prediction_catb=reg_catb.predict(test)


# In[39]:


test11[target] = prediction_catb


# In[40]:


test11[[target,"index"]].to_csv("catboost.csv",index  = False)


# In[41]:


test11[[target,"index"]]


# In[42]:


##ensemble the results
prediction_ensemble=test11[target]*0.7+test_org[target]*0.3


# In[43]:


test2=test_org.copy()
test2[target] = prediction_ensemble


# In[44]:


test2[[target,"index"]].to_csv("ensemble.csv",index  = False)


# In[45]:


test2[[target,"index"]]


# ## Conclusion:
#     1) Distribution of the training and test dataset is different. Need to be very careful while building the model. it will easily overfit the test data. 
# ## Next Steps
#     1) More robust cross validation
#      2) handle the data distruibution of train and test dataset
#      3) create more new features 
#      4) Try more ML algorithms

# ![image.png](attachment:098c77d0-6d03-4a4b-8b3a-d4c62497258e.png)
