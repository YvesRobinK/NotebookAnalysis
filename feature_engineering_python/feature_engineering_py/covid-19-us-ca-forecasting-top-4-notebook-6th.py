#!/usr/bin/env python
# coding: utf-8

# # COVID-19 USA(California) Confirmed Cases and Fatalities Forecasting

# **In this notebook, the model will be predicting the cumulative number of confirmed COVID19 cases in California, as well as the number of resulting fatalities, for future dates. We understand this is a serious situation, and in no way want to trivialize the human impact this crisis is causing by predicting fatalities. Our goal is to provide better methods for estimates that can assist medical and governmental institutions to prepare and adjust as pandemics unfold.**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Reading the Data**

# In[2]:


train=pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
test=pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")
submission=pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")


# **Exploratory Data Analysis and Visualization**

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train['Date']=pd.to_datetime(train['Date'])
train['Date'] = train['Date'].apply(lambda x:x.date().strftime('%m-%d'))
test['Date']=pd.to_datetime(test['Date'])
test['Date'] = test['Date'].apply(lambda x:x.date().strftime('%m-%d'))


# In[5]:


hor=train['Date']
ver=train['ConfirmedCases']
plt.figure(figsize=(20,10))
plt.plot(hor, ver)
plt.title('Time Series Confirmed Cases')
plt.show()


# In[6]:


hor=train['Date']
ver=train['Fatalities']
plt.figure(figsize=(20,10))
plt.plot(hor, ver)
plt.title('Time Series Fatalities')
plt.show()


# **Analysis and Feature Engineering**

# In[7]:


train1=train[48:]  #excluding first 48 values from train dataset as they are all zero
#train1=train
train1.head()


# In[8]:


X_test1=test[['ForecastId']]+50 #matching the test data Id in line to training ID's


# In[9]:


X1=train1[['Id']]
y_con=train1[['ConfirmedCases']]
y_fat=train1[['Fatalities']]


# In[10]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(7) #Polynomial Feature with degree 7
X=poly.fit_transform(X1)
X_test=poly.fit_transform(X_test1)


# **Building the Model**

# In[11]:


from sklearn.linear_model import Ridge, Lasso, SGDRegressor, LinearRegression
model_con=Lasso()
model_con.fit(X, y_con)


# In[12]:


y_pred_con=model_con.predict(X_test)


# In[13]:


model_fat=Lasso()
model_fat.fit(X, y_fat)


# In[14]:


y_pred_fat=model_fat.predict(X_test)


# In[15]:


y_pred_con1=y_pred_con.ravel()
y_pred_fat1=y_pred_fat.ravel()


# In[16]:


y_pred_con1=y_pred_con1[13:]  #replacing 13 test prediction with training label as they overlap
y_con_t=train1['ConfirmedCases']
y_con_t=y_con_t[2:].ravel()  #getting those 13 labels from training set to put into prediction
#y_con_t=y_con_t[50:].ravel()
y_pred_con_final=np.round(np.append(y_con_t, y_pred_con1))
y_pred_con_final


# In[17]:


y_pred_fat1=y_pred_fat1[13:] #replacing 13 test prediction with training label as they overlap
y_fat_t=train1['Fatalities']
y_fat_t=y_fat_t[2:].ravel() #getting those 13 labels from training set to put into prediction
#y_fat_t=y_fat_t[50:].ravel()
y_pred_fat_final=np.round(np.append(y_fat_t, y_pred_fat1))
y_pred_fat_final


# **Preparing the Submission File**

# In[18]:


data={'ForecastId':submission.ForecastId,'ConfirmedCases':y_pred_con_final, 'Fatalities':y_pred_fat_final}
result=pd.DataFrame(data, index=submission.index)
result.to_csv('/kaggle/working/submission.csv', index=False)
m1=pd.read_csv('/kaggle/working/submission.csv')
m1.head()


# **Predicted Result Visualization**

# In[19]:


hor=test.Date
ver=y_pred_con_final
plt.figure(figsize=(20,10))
plt.plot(hor, ver)
plt.title('Confirmed Cases Prediction')
plt.show()


# In[20]:


hor=test.Date
ver=y_pred_fat_final
plt.figure(figsize=(20,10))
plt.plot(hor, ver)
plt.title('Fatalities Prediction')
plt.show()


# **#StayHome #StaySafe #May Almighty bless us all.**

# **Please upvote if you like this or find this notebook useful, thanks.**
