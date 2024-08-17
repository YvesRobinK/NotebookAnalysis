#!/usr/bin/env python
# coding: utf-8

# ## Feature engineering + Random Forest

# In[1]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from functools import reduce
import random

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('../input/X_train.csv')
target = pd.read_csv('../input/y_train.csv')
test = pd.read_csv('../input/X_test.csv')


# Not missed values - good!

# In[3]:


data.info()


# Consider features in this dataset:

# In[4]:


print(data.shape)
data.head()


# In[5]:


data = data[['series_id', 'measurement_number', 'orientation_X',
       'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X',
       'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',
       'linear_acceleration_Y', 'linear_acceleration_Z']]


# In[6]:


f_data = pd.merge(data, target, how='inner')
full_data = f_data[['series_id', 'measurement_number', 'orientation_X',
       'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X',
       'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',
       'linear_acceleration_Y', 'linear_acceleration_Z', 'surface']]


# See on the target distribution. Every class counts are very different, so we need use weights for our classifiers. <br>
# All - 9 classes.

# In[7]:


pd.value_counts(full_data['surface']).plot(kind='barh');


# In[8]:


plt.figure(figsize=(25,20))
sns.set(font_scale=1.5)
sns.heatmap(full_data.corr(),annot=True);


# In[9]:


full_data.columns


# In[10]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


# Some features engineering:

# In[11]:


colum = ['series_id','orientation_X', 'orientation_Y',
       'orientation_Z', 'orientation_W', 'angular_velocity_X',
       'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',
       'linear_acceleration_Y', 'linear_acceleration_Z']
tr_dat = full_data[colum+['surface']].groupby(['series_id']).max()
tr_dat_1 = full_data[colum].groupby(['series_id']).min()
tr_dat_2 = full_data[colum].groupby(['series_id']).median()
tr_dat_3 = full_data[colum].groupby(['series_id']).mean()
tr_dat_4 = full_data[colum].groupby(['series_id']).std()
tr_dat_5 = full_data[colum].groupby(['series_id']).quantile(0.25)
tr_dat_6 = full_data[colum].groupby(['series_id']).quantile(0.5)
tr_dat_7 = full_data[colum].groupby(['series_id']).quantile(0.75)
tr_dat_8 = full_data[colum].groupby(['series_id']).max()/full_data[colum].groupby(['series_id']).min()


# In[12]:


tr_dat_5 = full_data[colum].groupby(['series_id']).quantile(0.25)
tr_dat_5.head()


# In[13]:


ss = [tr_dat,tr_dat_1,tr_dat_2,tr_dat_3,tr_dat_4,tr_dat_5,tr_dat_6,tr_dat_7,tr_dat_8]
df_final = reduce(lambda left,right: pd.merge(left,right,on='series_id'), ss)
df_final.head()


# In[14]:


ts_dat = test[colum].groupby(['series_id']).max()
ts_dat_1 = test[colum].groupby(['series_id']).min()
ts_dat_2 = test[colum].groupby(['series_id']).median()
ts_dat_3 = test[colum].groupby(['series_id']).mean()
ts_dat_4 = test[colum].groupby(['series_id']).std()
ts_dat_5 = test[colum].groupby(['series_id']).quantile(0.25)
ts_dat_6 = test[colum].groupby(['series_id']).quantile(0.5)
ts_dat_7 = test[colum].groupby(['series_id']).quantile(0.75)
ts_dat_8 = test[colum].groupby(['series_id']).max()/test[colum].groupby(['series_id']).min()
ts_ss = [ts_dat,ts_dat_1,ts_dat_2,ts_dat_3,ts_dat_4,ts_dat_5,ts_dat_6,ts_dat_7,ts_dat_8]
ts_final = reduce(lambda left,right: pd.merge(left,right,on='series_id'), ts_ss)
cols = ['orientation_X_x', 'orientation_Y_x', 'orientation_Z_x',
       'orientation_W_x', 'angular_velocity_X_x', 'angular_velocity_Y_x',
       'angular_velocity_Z_x', 'linear_acceleration_X_x',
       'linear_acceleration_Y_x', 'linear_acceleration_Z_x',
       'orientation_X_y', 'orientation_Y_y', 'orientation_Z_y',
       'orientation_W_y', 'angular_velocity_X_y', 'angular_velocity_Y_y',
       'angular_velocity_Z_y', 'linear_acceleration_X_y',
       'linear_acceleration_Y_y', 'linear_acceleration_Z_y', 'orientation_X_x',
       'orientation_Y_x', 'orientation_Z_x', 'orientation_W_x',
       'angular_velocity_X_x', 'angular_velocity_Y_x', 'angular_velocity_Z_x',
       'linear_acceleration_X_x', 'linear_acceleration_Y_x',
       'linear_acceleration_Z_x', 'orientation_X_y', 'orientation_Y_y',
       'orientation_Z_y', 'orientation_W_y', 'angular_velocity_X_y',
       'angular_velocity_Y_y', 'angular_velocity_Z_y',
       'linear_acceleration_X_y', 'linear_acceleration_Y_y',
       'linear_acceleration_Z_y', 'orientation_X_x', 'orientation_Y_x',
       'orientation_Z_x', 'orientation_W_x', 'angular_velocity_X_x',
       'angular_velocity_Y_x', 'angular_velocity_Z_x',
       'linear_acceleration_X_x', 'linear_acceleration_Y_x',
       'linear_acceleration_Z_x', 'angular_velocity_X_y',
       'angular_velocity_Y_y', 'angular_velocity_Z_y',
       'linear_acceleration_X_y', 'linear_acceleration_Y_y',
       'linear_acceleration_Z_y', 'orientation_W_y', 'orientation_X_y',
       'orientation_Y_y', 'orientation_Z_y', 'angular_velocity_X_x',
       'angular_velocity_Y_x', 'angular_velocity_Z_x',
       'linear_acceleration_X_x', 'linear_acceleration_Y_x',
       'linear_acceleration_Z_x', 'orientation_W_x', 'orientation_X_x',
       'orientation_Y_x', 'orientation_Z_x', 'angular_velocity_X_y',
       'angular_velocity_Y_y', 'angular_velocity_Z_y',
       'linear_acceleration_X_y', 'linear_acceleration_Y_y',
       'linear_acceleration_Z_y', 'orientation_W_y', 'orientation_X_y',
       'orientation_Y_y', 'orientation_Z_y', 'orientation_X', 'orientation_Y',
       'orientation_Z', 'orientation_W', 'angular_velocity_X',
       'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',
       'linear_acceleration_Y', 'linear_acceleration_Z']


# In[15]:


trains = df_final[cols]
targs = df_final['surface']
tests = ts_final[cols]


# See on the example of data - it's nonlinear plot for every feature. We need some characteristics of this plot.

# In[16]:


data[data.series_id==0]['angular_velocity_Z'].plot();
plt.show()
data[data.series_id==0]['orientation_X'].plot();


# Add correlation coefficient and intercept to the data. (from sklearn)

# In[17]:


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
new_feat_x = []
new_feat_y = []
new_feat_z = []
new_feat_w = []

skm = LinearRegression()
for i in range(3810):
        qq = data[data.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_X'].index))
        skm.fit(x, qq['orientation_X'].values)
        new_feat_x.append([skm.intercept_, skm.coef_[0]])
for i in range(3810):
        qq = data[data.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_Y'].index))
        skm.fit(x, qq['orientation_Y'].values)
        new_feat_y.append([skm.intercept_, skm.coef_[0]])
for i in range(3810):
        qq = data[data.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_Z'].index))
        skm.fit(x, qq['orientation_Z'].values)
        new_feat_z.append([skm.intercept_, skm.coef_[0]])
for i in range(3810):
        qq = data[data.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_W'].index))
        skm.fit(x, qq['orientation_W'].values)
        new_feat_w.append([skm.intercept_, skm.coef_[0]])
        
new_feat_x = np.array(new_feat_x)
new_feat_y = np.array(new_feat_y)
new_feat_z = np.array(new_feat_z)
new_feat_w = np.array(new_feat_w)


# In[18]:


trains['intercept_x'] = new_feat_x[:,0]
trains['coef_x'] = new_feat_x[:,1]
trains['intercept_y'] = new_feat_y[:,0]
trains['coef_y'] = new_feat_y[:,1]
trains['intercept_z'] = new_feat_z[:,0]
trains['coef_z'] = new_feat_z[:,1]
trains['intercept_w'] = new_feat_w[:,0]
trains['coef_w'] = new_feat_w[:,1]


# In[19]:


new_feat_x = []
new_feat_y = []
new_feat_z = []
new_feat_w = []

new_feat_ax = []
new_feat_ay = []
new_feat_az = []
for i in range(3816):
        qq = test[test.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_X'].index))
        skm.fit(x, qq['orientation_X'].values)
        new_feat_x.append([skm.intercept_, skm.coef_[0]])
for i in range(3816):
        qq = test[test.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_Y'].index))
        skm.fit(x, qq['orientation_Y'].values)
        new_feat_y.append([skm.intercept_, skm.coef_[0]])
for i in range(3816):
        qq = test[test.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_Z'].index))
        skm.fit(x, qq['orientation_Z'].values)
        new_feat_z.append([skm.intercept_, skm.coef_[0]])
for i in range(3816):
        qq = test[test.series_id==i]
        x = np.transpose(np.atleast_2d(qq['orientation_W'].index))
        skm.fit(x, qq['orientation_Z'].values)
        new_feat_w.append([skm.intercept_, skm.coef_[0]])
        
new_feat_x = np.array(new_feat_x)
new_feat_y = np.array(new_feat_y)
new_feat_z = np.array(new_feat_z)
new_feat_w = np.array(new_feat_w)

tests['intercept_x'] = new_feat_x[:,0]
tests['coef_x'] = new_feat_x[:,1]
tests['intercept_y'] = new_feat_y[:,0]
tests['coef_y'] = new_feat_y[:,1]
tests['intercept_z'] = new_feat_z[:,0]
tests['coef_z'] = new_feat_z[:,1]
tests['intercept_w'] = new_feat_w[:,0]
tests['coef_w'] = new_feat_w[:,1]


# In[20]:


print('X_train.shape = ', trains.shape)
trains.head()


# In[21]:


clf = RandomForestClassifier(n_estimators = 2000,
                               max_depth=20, 
                             min_samples_split=5,
                             class_weight='balanced')
                            
clf.fit(trains, targs)
ress  = clf.predict(tests)


# In[22]:


submission = pd.read_csv("../input/sample_submission.csv")
print(len(submission))
print(test.shape)
submission['surface'] = ress
submission.to_csv("NEW_submission.csv", index = False)
submission.head()


# Most interesting - feature importance

# In[23]:


importance = pd.DataFrame({'value': clf.feature_importances_,
                           'feature': np.array(trains.columns)})
importance.sort_values(by='value', ascending=False, inplace=True)
plt.figure(figsize=(20, 20))
sns.barplot(x='value', y='feature', data=importance);


# Thank you for reading! I hope this kernel was useful for you.
