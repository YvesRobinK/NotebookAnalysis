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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# In[2]:


widsDf_train = pd.read_csv("/kaggle/input/widsdatathon2022/train.csv")
widsDf_train.head()


# In[3]:


widsDf_train.shape


# In[4]:


widsDf_train.info()


# In[5]:


widsDf_train.describe()


# In[6]:


widsDf_train.isna().sum()


# In[7]:


[(column, widsDf_train[column].isna().sum()) for column in widsDf_train.columns if widsDf_train[column].isna().sum() > 0]


# In[8]:


#widsDf.drop(["direction_max_wind_speed","direction_peak_wind_speed","max_wind_speed","days_with_fog"], inplace=True, axis=1)

new_wids_df_train = widsDf_train[['Year_Factor','State_Factor','building_class','facility_type','floor_area','year_built', 'energy_star_rating', 'ELEVATION', 'site_eui']]


# In[9]:


new_wids_df_train.columns


# In[10]:


widsDf_test = pd.read_csv("/kaggle/input/widsdatathon2022/test.csv")
widsDf_test.head()


# In[11]:


widsDf_test.shape


# In[12]:


widsDf_test.info()


# In[13]:


widsDf_test.describe()


# In[14]:


[(column, widsDf_test[column].isna().sum()) for column in widsDf_test.columns if widsDf_test[column].isna().sum() > 0]


# In[15]:


new_wids_df_test = widsDf_test[['Year_Factor','State_Factor','building_class','facility_type','floor_area','year_built', 'energy_star_rating', 'ELEVATION']]


# In[16]:


new_wids_df_test.columns


# In[17]:


new_wids_df_train.info()


# In[18]:


new_wids_df_test.info()


# In[19]:


new_wids_df_train.describe()


# In[20]:


new_wids_df_test.describe()


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(data = new_wids_df_train, x = "site_eui",hue='State_Factor')

plt.figure(figsize=(8,5))
sns.histplot(data = new_wids_df_train, x = "site_eui",hue='building_class')
plt.show()


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(50,3))
plt.subplot(1, 2, 1)
new_wids_df_train['facility_type'].value_counts().plot(kind='bar')
plt.show()


# In[23]:


new_wids_df_train['facility_type'].value_counts()


# In[24]:


new_wids_df_test['facility_type'].value_counts()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,6))
sns.scatterplot(data=new_wids_df_train,x='floor_area',hue='building_class',y='site_eui')
plt.show()


# In[26]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.scatterplot(data=new_wids_df_train,y='floor_area',hue='State_Factor',x='energy_star_rating')

plt.subplot(1,2,2)
sns.scatterplot(data=new_wids_df_test,y='floor_area',hue='State_Factor',x='energy_star_rating')
plt.show()


# In[27]:


plt.figure(figsize=(12,6))

sns.scatterplot(data=new_wids_df_train, x="ELEVATION", y="site_eui",hue='building_class')
plt.show()


# In[28]:


new_wids_df_train['State_Factor'].value_counts()


# In[29]:


new_wids_df_test['State_Factor'].value_counts()


# In[30]:


new_wids_df_train['building_class'].value_counts()


# In[31]:


new_wids_df_test['building_class'].value_counts()


# In[32]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
for col in ['State_Factor','building_class', 'facility_type']:
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'Country'. 
    label_encoder.fit(new_wids_df_train[col])
    new_wids_df_train[col] = label_encoder.transform(new_wids_df_train[col])
    new_wids_df_test[col]= label_encoder.transform(new_wids_df_test[col])

new_wids_df_train.head()


# In[33]:


new_wids_df_test.head()


# In[34]:


new_wids_df_train_corr=new_wids_df_train.corr()
new_wids_df_train_corr.style.background_gradient(cmap="cool")


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
# constructing heap to see correlation
plt.figure(figsize=(20,12))
sns.heatmap(new_wids_df_train_corr, cbar=True, square=True, fmt='.1f', annot=True,annot_kws={'size':8},cmap='Blues')


# In[36]:


features = new_wids_df_train.drop('site_eui',axis=1)
outputLabel = new_wids_df_train['site_eui'] 


# In[ ]:





# In[37]:


from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# Imputation transformer for completing missing values.
imp_mean.fit(features)
features = imp_mean.transform(features)
features = pd.DataFrame(features)
features_test = imp_mean.transform(new_wids_df_test)
features_test = pd.DataFrame(features_test)
features.head


# In[38]:


# define min max scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
scaler.fit(features)
scaled = scaler.transform(features)
scaled_test = scaler.transform(features_test)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled, outputLabel, test_size=0.25, random_state=46)


# In[40]:


#Prepare a Linear Regression Model
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)


# In[41]:


import sklearn
import math
mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
rmse


# In[42]:


import xgboost

xgboost_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.03, gamma=0, subsample=0.75,
                           colsample_bytree=0.4, max_depth=3)
xgboost_model.fit(X_train,y_train)


# In[43]:


predictions = xgboost_model.predict(X_test)
mse_xg = sklearn.metrics.mean_squared_error(y_test,predictions)
rmse_xg = math.sqrt(mse_xg)
rmse_xg


# In[44]:


# from sklearn.model_selection import GridSearchCV
# xgb = xgboost.XGBRegressor()
# parameters = {'objective':['reg:linear'],
#               'learning_rate': [.02, .05, 0.01], #so called `eta` value
#               'max_depth': [3,5],
#               'min_child_weight': [4],
#               'subsample': [0.7],
#               'colsample_bytree': [0.4],
#               'n_estimators': [1000],
#               'reg_alpha': [0.4],
#               'reg_lambda': [2e-08]
# }
# xgb_grid = GridSearchCV(xgb,
#                         parameters,
#                         cv = 5,
#                         verbose=True)

# xgb_grid.fit(X_train,y_train)

# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)


# In[45]:


xgboost_mod = xgboost.XGBRegressor(colsample_bytree=0.4, learning_rate= 0.05, max_depth= 5, min_child_weight= 4,
                          n_estimators= 1000, reg_alpha=0.4, reg_lambda=2e-08, subsample= 0.7)
xgboost_mod.fit(X_train,y_train)


# In[46]:


predictions_mod = xgboost_mod.predict(X_test)
mse_xg = sklearn.metrics.mean_squared_error(y_test,predictions_mod)
rmse_xg = math.sqrt(mse_xg)
rmse_xg


# In[47]:


pred_test = xgboost_mod.predict(scaled_test)
SAMPLE_SUBMISSION_PATH = "../input/widsdatathon2022/sample_solution.csv"
SUBMISSION_PATH = "submission.csv"
sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub['site_eui'] = pred_test
sub.to_csv(SUBMISSION_PATH,index=False)
sub.head()


# In[ ]:





# In[ ]:




