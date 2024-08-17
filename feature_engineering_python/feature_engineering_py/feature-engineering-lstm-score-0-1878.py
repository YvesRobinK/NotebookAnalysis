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


# ![](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error as mae


# ### As previous versions failed because of my notebook tried to allocate more memory than is available.
# > This function helps in reducing memory usage by changing unnecessary data type.
# 
# > You could learn more about it [here](https://www.kaggle.com/questions-and-answers/282144)

# In[3]:


def reduce_memory_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))    

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[4]:


data = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
data = reduce_memory_usage(data)
data.head()


# ## Columns
# **id** - globally-unique time step identifier across an entire file
# 
# **breath_id** - globally-unique time step for breaths
# 
# **R** - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.
# 
# **C** - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.
# time_step - the actual time stamp.
# 
# **u_in** - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
# 
# **u_out** - the control input for the exploratory solenoid valve. Either 0 or 1.
# 
# **pressure** - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# > There is no null value

# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), cmap='cool')


# We can see a strong correaltion between :
# > 'pressure' and 'time_step'
# 
# > 'pressure' and 'u_out'

# ## Splitting of categorical and numerical data.

# ![](https://bookdown.org/ejvanholm/Text-Quant/images/DataTypes.png)

# In[10]:


cat_col = []
num_col = []
for i in data.columns:
    if data[i].value_counts().count() > 10:
        num_col.append(i)
    else:
        cat_col.append(i)
print(f'categorical columns: {cat_col}')
print(f'numerical columns: {num_col}')


# #### Categorical Data

# In[11]:


fig, ax = plt.subplots(1,3,figsize=(12,5))
j=0
for i in cat_col:
    sns.countplot(data[i], palette='cool', ax=ax[j])
    j+=1
fig.suptitle('Countplot of Categorical Data')


# #### Numerical Data

# Removing 'id' and 'breath_id' from numerical column list.

# In[12]:


num_col = num_col[2:]
num_col


# In[13]:


fig, ax = plt.subplots(1,3, figsize=(18,5))
j=0
for i in num_col:
    sns.histplot(data[i], ax=ax[j])
    j+=1
fig.suptitle('Histplot of Numerical Data')


# > I can see a skewness in 'pressure', which is our target variable.
# 
# > And, a great outliers in 'u_in' and 'pressure'.

# ## Outliers
# > Let us have a look at outliers now, by using boxplot.

# In[14]:


fig, ax = plt.subplots(1,3, figsize=(18,5))
j=0
for i in num_col:
    sns.boxplot(data[i], ax=ax[j], palette='cool')
    j+=1
fig.suptitle('Boxplot of Numerical Data')


# > It is always good practice to work with copied dataset. ✓✓

# In[15]:


train = data.copy()


# ## Removing Skewness of target variable.
# #### Methods tried :-
# 1. Log Transformation
# 2. Log + 1 Transformation
# 3. Square Root
# 4. Double Square Root --> This works best.

# In[16]:


# fig, ax = plt.subplots(1,2, figsize=(12,5))
# sns.distplot(train['pressure'], ax=ax[0])

# train['pressure'] = np.where(train['pressure'] < 0, 0, train['pressure'])
# train['pressure'] = np.sqrt(np.sqrt(train['pressure']))

# sns.distplot(train['pressure'], ax=ax[1])


# ## Creating New Features

# In[17]:


def create_new_feat(df):
    df["u_in_sum"]         = df.groupby("breath_id")["u_in"].transform("sum")
    df["u_in_std"]         = df.groupby("breath_id")["u_in"].transform("std")
    df["u_in_min"]         = df.groupby("breath_id")["u_in"].transform("min")
    df["u_in_first"]       = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_last"]        = df.groupby("breath_id")["u_in"].transform("last")
    df["time_passed"]      = df.groupby("breath_id")["time_step"].diff()
    df['area']             = df['time_step'] * df['u_in']
    df['area_2']           = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum']      = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag1']        = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1']       = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1']   = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1']  = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2']        = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2']       = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2']   = df.groupby('breath_id')['u_in'].shift(-2) 
    df['u_out_lag_back2']  = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3']        = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3']       = df.groupby('breath_id')['u_out'].shift(3) 
    df['u_in_lag_back3']   = df.groupby('breath_id')['u_in'].shift(-3) 
    df['u_out_lag_back3']  = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4']        = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4']       = df.groupby('breath_id')['u_out'].shift(4) 
    df['u_in_lag_back4']   = df.groupby('breath_id')['u_in'].shift(-4) 
    df['u_out_lag_back4']  = df.groupby('breath_id')['u_out'].shift(-4) 
    
    df = df.fillna(0)
    
    df['breath_id__u_in__diffmax']  = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['cross']                     = df['u_in']*df['u_out']
    df['cross2']                    = df['time_step']*df['u_out']
    df['R']                         = df['R'].astype(str)
    df['C']                         = df['C'].astype(str)
    df['R__C']                      = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
#     df['u_in_lag5']  = df.groupby('breath_id')['u_in'].shift(5)  #
#     df['u_in_lag6']  = df.groupby('breath_id')['u_in'].shift(6)  #
#     df['u_in_lag7']  = df.groupby('breath_id')['u_in'].shift(7)  #
#     df['u_in_lag8']  = df.groupby('breath_id')['u_in'].shift(8)  #
#     df['u_in_lag9']  = df.groupby('breath_id')['u_in'].shift(9)  #
#     df['u_in_lag10'] = df.groupby('breath_id')['u_in'].shift(10) #
#     df['u_in_lag11'] = df.groupby('breath_id')['u_in'].shift(11) #
#     df['u_in_lag12'] = df.groupby('breath_id')['u_in'].shift(12) #
#     df['u_in_lag13'] = df.groupby('breath_id')['u_in'].shift(13) #
#     df['u_in_lag14'] = df.groupby('breath_id')['u_in'].shift(14) #
#     df['u_in_lag15'] = df.groupby('breath_id')['u_in'].shift(15) #
#     df['u_in_lag16'] = df.groupby('breath_id')['u_in'].shift(16) #
#     df['u_in_lag17'] = df.groupby('breath_id')['u_in'].shift(17) #
#     df['u_in_lag18'] = df.groupby('breath_id')['u_in'].shift(18) #
#     df['u_in_lag19'] = df.groupby('breath_id')['u_in'].shift(19) #
#     df['u_in_lag20'] = df.groupby('breath_id')['u_in'].shift(20) #
    df['time_diff']  = (df['time_step']).groupby(df['breath_id']).diff(1)
    df['time_diff2'] = (df['time_step']).groupby(df['breath_id']).diff(2)
    df['time_diff3'] = (df['time_step']).groupby(df['breath_id']).diff(3)
    df['time_diff4'] = (df['time_step']).groupby(df['breath_id']).diff(4)
    df['time_diff5'] = (df['time_step']).groupby(df['breath_id']).diff(5)
    df['time_diff6'] = (df['time_step']).groupby(df['breath_id']).diff(6)
    df['time_diff7'] = (df['time_step']).groupby(df['breath_id']).diff(7)
    df['time_diff8'] = (df['time_step']).groupby(df['breath_id']).diff(8)
    df['u_in_diff1']                = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1']               = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2']                = df['u_in'] - df['u_in_lag2'] 
    df['u_out_diff2']               = df['u_out'] - df['u_out_lag2'] 
    df['u_in_diff3']                = df['u_in'] - df['u_in_lag3'] 
    df['u_out_diff3']               = df['u_out'] - df['u_out_lag3'] 
    df['u_in_diff4']                = df['u_in'] - df['u_in_lag4'] 
    df['u_out_diff4']               = df['u_out'] - df['u_out_lag4'] 
    return df


# In[18]:


train = create_new_feat(train)
train = train.fillna(train.min())
train.head()


# In[19]:


train = reduce_memory_usage(train)


# ## Working with Test Data

# In[20]:


test_data = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
test_data.head()


# In[21]:


fig, ax = plt.subplots(1,2, figsize=(18,5))
j=0
for i in num_col[:2]:
    sns.histplot(test_data[i], ax=ax[j])
    j+=1
fig.suptitle('Histplot of Numerical Data')


# #### Creating New Features

# In[22]:


test_data = create_new_feat(test_data)
test_data = test_data.fillna(test_data.min())
test_data.head()


# In[23]:


test_data = reduce_memory_usage(test_data)


# In[24]:


from sklearn.preprocessing import RobustScaler
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test_data = test_data.drop(['id', 'breath_id'], axis=1)

RS = RobustScaler()
train = RS.fit_transform(train)
test_data = RS.transform(test_data)

train = train.reshape(-1, 80, train.shape[-1])
test_data = test_data.reshape(-1, 80, train.shape[-1])


# ## Splitting of dependent and independent variable.

# In[25]:


idx_len = round(0.90*len(train))
X_train, X_valid = train[0:idx_len], train[idx_len:]
y_train, y_valid = targets[0:idx_len], targets[idx_len:]


# ## Model

# In[26]:


EPOCH = 500
BATCH_SIZE = 256

lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
es = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min", restore_best_weights=True)

model = keras.models.Sequential([
keras.layers.Input(shape=train.shape[-2:]),    
keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
keras.layers.Dense(64, activation='selu'),
keras.layers.Dense(1),
])

model.compile(optimizer="adam", loss="mae")

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es])


# In[27]:


plt.figure(figsize=(15,3))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# ## Prediction

# In[28]:


pred = model.predict(test_data, batch_size=BATCH_SIZE)


# In[29]:


# sample = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
# sample['id'] = test_data['id']
# sample['pressure'] = pd.DataFrame({'pressure': pred[:]})
# sample.head()


# In[30]:


sample = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
sample['pressure'] = pred.squeeze().reshape(-1, 1).squeeze()

q1 = sample['pressure'].quantile(0.001)
q2 = sample['pressure'].quantile(0.999)
sample['pressure'] = sample['pressure'].apply(lambda x: x if x>q1 else x*0.77)
sample['pressure'] = sample['pressure'].apply(lambda x: x if x<q2 else x*1.1)
sample.to_csv('submission_LSTM.csv', index=False)


# ## Thank you!!!
# ## Hope you enjoyed this notebook. 😊

# In[ ]:




