#!/usr/bin/env python
# coding: utf-8

# # **TPS - July 2021**

# ![](https://www.navy.ac.kr:10001/intro/images/sang_01.jpg)

# ## [Click Here](https://www.kaggle.com/junhyeok99/pycaret-automl-baseline) to check pycaret baseline!

# ## **Library Import**

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ## **DATA LOAD**

# In[29]:


train = pd.read_csv('../input/tabular-playground-series-jul-2021/train.csv')
train


# In[30]:


test = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv')
test


# In[31]:


all_data = pd.concat([train, test])
all_data


# ## **Data Preprocessing**
# 
# *   There are only numeric columns
# *   Maybe need to use linear regression!!
# 
# 

# ### **Datetime Preprocessing**

# In[32]:


all_data.info()


# In[33]:


all_data['date_time'] = pd.to_datetime(all_data['date_time'])
all_data['year'] = all_data['date_time'].dt.year
all_data['month'] = all_data['date_time'].dt.month
all_data['week'] = all_data['date_time'].dt.week
all_data['day'] = all_data['date_time'].dt.day
all_data['dayofweek'] = all_data['date_time'].dt.dayofweek
all_data['time'] = all_data['date_time'].dt.date - all_data['date_time'].dt.date.min()
all_data['hour'] = all_data['date_time'].dt.hour
all_data['time'] = all_data['time'].apply(lambda x : x.days)
all_data.drop(columns = 'date_time', inplace = True)
all_data


# ## One-Hot Encoding - Day of Week column
# 
# ### **This column seems categorical, not numeric**

# In[34]:


all_data['dayofweek'] = all_data['dayofweek'].astype(object)
all_data = pd.get_dummies(all_data)


# ## **Feature Generation**
# 
# *  **SMC**
# 
# ![SMC](https://t1.daumcdn.net/cfile/tistory/992B21365C99ADDC18)
# 
# *  **Dew Point**
# 
# ![Dew Point Equation](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F999094495C930FC311)

# In[35]:


# all_data['SMC'] = (all_data['absolute_humidity'] * 100) / all_data['relative_humidity']
# all_data['Dew_Point'] = 243.12*(np.log(all_data['relative_humidity'] * 0.01) + (17.62 * all_data['deg_C'])/(243.12+all_data['deg_C']))/(17.62-(np.log(all_data['relative_humidity'] * 0.01)+17.62*all_data['deg_C']/(243.12+all_data['deg_C'])))
# all_data['relative_humidity'] = all_data['relative_humidity']/100
# all_data['deg_F'] = all_data['deg_C'] * 1.8 + 32


# In[36]:


train2 = all_data[:len(train)]
test2 = all_data[len(train):]
# train['SMC'] = train2['SMC']


# ### **Scaling**
# 
# #### **Log Scaling - Target values are skewed**

# #### **Scaler**

# In[37]:


def log_scaling(col):
  col = np.log1p(col)
  return col


# In[38]:


cols = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']
for col in cols:
  train2[col] = log_scaling(train2[col])


# #### **Compare with Visualization**

# In[39]:


fig, ax = plt.subplots(len(cols), 2, figsize=(12,12))
n = 0
for i in cols:
  sns.histplot(train[i], ax=ax[n, 0]);
  sns.histplot(train2[i], ax = ax[n, 1]);
  n += 1

fig.tight_layout()
plt.show()


# ### **Split DataSets**

# In[40]:


train_3 = train2.drop(columns = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'])
test_3 = test2.drop(columns = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'])

train_co = train2.drop(columns = ['target_benzene', 'target_nitrogen_oxides'])
train_be = train2.drop(columns = ['target_carbon_monoxide', 'target_nitrogen_oxides'])
train_no = train2.drop(columns = ['target_carbon_monoxide', 'target_benzene'])

test_co = test2.drop(columns = ['target_benzene', 'target_nitrogen_oxides'])
test_be = test2.drop(columns = ['target_carbon_monoxide', 'target_nitrogen_oxides'])
test_no = test2.drop(columns = ['target_carbon_monoxide', 'target_benzene'])


# ## **EDA**

# ### **Groupby Plot**

# #### **Target & Time Relevance (Visualizaition)**
# 
# *   Year
# *   Month
# *   Time
# *   Hour
# 
# ##### **Three targets have similar tendency with above features**

# In[41]:


fig, ax = plt.subplots(4, 3, figsize = (12,10))

ax[0,0].plot(train2.groupby(train2['year'])['target_carbon_monoxide'].mean(), 'r');
ax[0,1].plot(train2.groupby(train2['year'])['target_benzene'].mean(), 'r');
ax[0,2].plot(train2.groupby(train2['year'])['target_nitrogen_oxides'].mean(), 'r');

ax[1,0].plot(train2.groupby(train2['month'])['target_carbon_monoxide'].mean(), 'b');
ax[1,1].plot(train2.groupby(train2['month'])['target_benzene'].mean(), 'b');
ax[1,2].plot(train2.groupby(train2['month'])['target_nitrogen_oxides'].mean(), 'b');

ax[2,0].plot(train2.groupby(train2['time'])['target_carbon_monoxide'].mean(), 'y');
ax[2,1].plot(train2.groupby(train2['time'])['target_benzene'].mean(), 'y');
ax[2,2].plot(train2.groupby(train2['time'])['target_nitrogen_oxides'].mean(), 'y');

ax[3,0].plot(train2.groupby(train2['hour'])['target_carbon_monoxide'].mean(), 'black');
ax[3,1].plot(train2.groupby(train2['hour'])['target_benzene'].mean(), 'black');
ax[3,2].plot(train2.groupby(train2['hour'])['target_nitrogen_oxides'].mean(), 'black');

ax[0,0].set_title('Year-CO')
ax[0,1].set_title('Year-Benzene')
ax[0,2].set_title('Year-NOx')

ax[1,0].set_title('month-CO')
ax[1,1].set_title('month-Benzene')
ax[1,2].set_title('month-NOx')

ax[2,0].set_title('time-CO')
ax[2,1].set_title('time-Benzene')
ax[2,2].set_title('time-NOx')

ax[3,0].set_title('hour-CO')
ax[3,1].set_title('hour-Benzene')
ax[3,2].set_title('hour-NOx')

fig.tight_layout()
plt.show()


# #### **Target & Temp, Humid Relevance (Visualizaition)**
# 
# *   deg_C
# *   Relative_Humidity
# *   Absolute_Humidity
# 
# ##### **Three targets have similar tendency with above features except NOx & Deg_c**

# In[42]:


fig, ax = plt.subplots(3, 3, figsize = (10,10))

ax[0,0].plot(train2.groupby(train2['deg_C'])['target_carbon_monoxide'].mean(), 'r');
ax[0,1].plot(train2.groupby(train2['deg_C'])['target_benzene'].mean(), 'r');
ax[0,2].plot(train2.groupby(train2['deg_C'])['target_nitrogen_oxides'].mean(), 'r');

ax[1,0].plot(train2.groupby(train2['relative_humidity'])['target_carbon_monoxide'].mean(), 'b');
ax[1,1].plot(train2.groupby(train2['relative_humidity'])['target_benzene'].mean(), 'b');
ax[1,2].plot(train2.groupby(train2['relative_humidity'])['target_nitrogen_oxides'].mean(), 'b');

ax[2,0].plot(train2.groupby(train2['absolute_humidity'])['target_carbon_monoxide'].mean(), 'y');
ax[2,1].plot(train2.groupby(train2['absolute_humidity'])['target_benzene'].mean(), 'y');
ax[2,2].plot(train2.groupby(train2['absolute_humidity'])['target_nitrogen_oxides'].mean(), 'y');

ax[0,0].set_title('deg-CO')
ax[0,1].set_title('deg-Benzene')
ax[0,2].set_title('deg-NOx')

ax[1,0].set_title('rel_humid-CO')
ax[1,1].set_title('rel_humid-Benzene')
ax[1,2].set_title('rel_humid-NOx')

ax[2,0].set_title('ab_humid-CO')
ax[2,1].set_title('ab_humid-Benzene')
ax[2,2].set_title('ab_humid-NOx')


fig.tight_layout()
plt.show()


# #### **Target & Sensors Relevance (Visualizaition)**
# 
# *   Sensor 1
# *   Sensor 2
# *   Sensor 3
# *   Sensor 4
# *   Sensor 5
# 
# ##### **Three targets have similar tendency with above features**

# In[43]:


fig, ax = plt.subplots(5, 3, figsize = (10,13))

ax[0,0].plot(train2.groupby(train2['sensor_1'])['target_carbon_monoxide'].mean(), 'r');
ax[0,1].plot(train2.groupby(train2['sensor_1'])['target_benzene'].mean(), 'r');
ax[0,2].plot(train2.groupby(train2['sensor_1'])['target_nitrogen_oxides'].mean(), 'r');

ax[1,0].plot(train2.groupby(train2['sensor_2'])['target_carbon_monoxide'].mean(), 'b');
ax[1,1].plot(train2.groupby(train2['sensor_2'])['target_benzene'].mean(), 'b');
ax[1,2].plot(train2.groupby(train2['sensor_2'])['target_nitrogen_oxides'].mean(), 'b');

ax[2,0].plot(train2.groupby(train2['sensor_3'])['target_carbon_monoxide'].mean(), 'y');
ax[2,1].plot(train2.groupby(train2['sensor_3'])['target_benzene'].mean(), 'y');
ax[2,2].plot(train2.groupby(train2['sensor_3'])['target_nitrogen_oxides'].mean(), 'y');

ax[3,0].plot(train2.groupby(train2['sensor_4'])['target_carbon_monoxide'].mean(), 'black');
ax[3,1].plot(train2.groupby(train2['sensor_4'])['target_benzene'].mean(), 'black');
ax[3,2].plot(train2.groupby(train2['sensor_4'])['target_nitrogen_oxides'].mean(), 'black');

ax[4,0].plot(train2.groupby(train2['sensor_5'])['target_carbon_monoxide'].mean(), 'violet');
ax[4,1].plot(train2.groupby(train2['sensor_5'])['target_benzene'].mean(), 'violet');
ax[4,2].plot(train2.groupby(train2['sensor_5'])['target_nitrogen_oxides'].mean(), 'violet');

ax[0,0].set_title('sensor_1-CO')
ax[0,1].set_title('sensor_1-Benzene')
ax[0,2].set_title('sensor_1-NOx')

ax[1,0].set_title('sensor_2-CO')
ax[1,1].set_title('sensor_2-Benzene')
ax[1,2].set_title('sensor_2-NOx')

ax[2,0].set_title('sensor_3-CO')
ax[2,1].set_title('sensor_3-Benzene')
ax[2,2].set_title('sensor_3-NOx')

ax[3,0].set_title('sensor_4-CO')
ax[3,1].set_title('sensor_4-Benzene')
ax[3,2].set_title('sensor_4-NOx')

ax[4,0].set_title('sensor_5-CO')
ax[4,1].set_title('sensor_5-Benzene')
ax[4,2].set_title('sensor_5-NOx')

fig.tight_layout()
plt.show()


# ### **HeatMap**
# 
# *   **Heatmap shows us that sensor 1~5 are influential feature**
# *   **But 'sensor_3' looks different from others**
# 
# #### **Need to check sensor_3 feature_importance later!!**

# In[44]:


plt.figure(figsize=(12,12))
sns.heatmap(train2.corr());


# ### **BoxPlot**
# 
# #### **Shows that Month data, Hour data are influential because of the temp!**

# In[45]:


fig, ax = plt.subplots(3, 3, figsize = (20,15))
sns.boxplot(train2['year'], train2['target_carbon_monoxide'], ax = ax[0, 0]);
sns.boxplot(train2['year'], train2['target_benzene'], ax= ax[0, 1]);
sns.boxplot(train2['year'], train2['target_nitrogen_oxides'], ax = ax[0, 2]);

sns.boxplot(train2['month'], train2['target_carbon_monoxide'], ax = ax[1, 0]);
sns.boxplot(train2['month'], train2['target_benzene'], ax= ax[1, 1]);
sns.boxplot(train2['month'], train2['target_nitrogen_oxides'], ax = ax[1, 2]);

sns.boxplot(train2['hour'], train2['target_carbon_monoxide'], ax = ax[2,0]);
sns.boxplot(train2['hour'], train2['target_benzene'], ax= ax[2,1]);
sns.boxplot(train2['hour'], train2['target_nitrogen_oxides'], ax = ax[2,2]);

plt.show();


# ## **Modeling**

# ### **Pycaret**

# In[46]:


# !pip install pycaret


# In[47]:


from pycaret.regression import setup, compare_models, blend_models, finalize_model, predict_model, plot_model


# #### **Model**

# In[48]:


def pycaret_model(train, target, test, n_select, fold, opt):
  print('Setup Your Data....')
  setup(data=train,
              target=target,
              numeric_imputation = 'mean',
              silent= True)
  
  print('Comparing Models....')
  best = compare_models(sort=opt, n_select=n_select, fold = fold, exclude = ['xgboost'])

  print('Here is Best Model Feature Importances!')
  plot_model(estimator = best[0], plot = 'feature')
  time.sleep(5)
  
  print('Blending Models....')
  blended = blend_models(estimator_list= best, fold=fold, optimize=opt)
  pred_holdout = predict_model(blended)
    
  print('Finallizing Models....')
  final_model = finalize_model(blended)
  print('Done...!!!')

  pred_esb = predict_model(final_model, test)
  re = pred_esb['Label']

  return re


# #### **Predict Result**

# In[49]:


sub = pd.read_csv('../input/tabular-playground-series-jul-2021/sample_submission.csv')
sub['target_carbon_monoxide'] = np.exp(pycaret_model(train_co, 'target_carbon_monoxide', test_co, 5, 3, 'RMSLE'))-1


# In[50]:


sub['target_benzene'] = np.exp(pycaret_model(train_be, 'target_benzene', test_be, 5, 3, 'RMSLE'))-1


# In[51]:


sub['target_nitrogen_oxides'] = np.exp(pycaret_model(train_no, 'target_nitrogen_oxides', test_no, 4, 3, 'RMSLE')) - 1


# In[52]:


sub


# In[55]:


sub.to_csv('sub.csv', index=False)

