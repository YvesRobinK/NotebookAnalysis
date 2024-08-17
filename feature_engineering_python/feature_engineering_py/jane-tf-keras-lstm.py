#!/usr/bin/env python
# coding: utf-8

# ### Hi kagglers,
# ### I wish to learn more since this is my first competition on kaggle.
# 
# ### I decide to develop an LSTM model using Tensorflow for this time series data.

# # 1. IMPORT PACKAGES AND LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import datatable
import missingno as msno
import gc
import warnings
warnings.filterwarnings('ignore')
SEED = 11
tf.random.set_seed(SEED)
np.random.seed(SEED)


# # 2. LOAD DATA

# In[ ]:


example_submission_path = '../input/jane-street-market-prediction/example_sample_submission.csv'
example_test_path = '../input/jane-street-market-prediction/example_test.csv'
features_path = '../input/jane-street-market-prediction/features.csv'
train_path = '../input/jane-street-market-prediction/train.csv'

# use pandas to load small data files
example_submission_file = pd.read_csv(example_submission_path)
example_test_file = pd.read_csv(example_test_path)
features_file = pd.read_csv(features_path)

# use datatable to load big data file
train_file = datatable.fread(train_path).to_pandas()
train_file.info()


# ## Reduce memory usage by adopting optimal datatype

# In[ ]:


# It is found from info() that there are only two datatypes - float64 and int32
for c in train_file.columns:
    min_val, max_val = train_file[c].min(), train_file[c].max()
    if train_file[c].dtype == 'float64':
        if min_val>np.finfo(np.float16).min and max_val<np.finfo(np.float16).max:
            train_file[c] = train_file[c].astype(np.float16)
        elif min_val>np.finfo(np.float32).min and max_val<np.finfo(np.float32).max:
            train_file[c] = train_file[c].astype(np.float32)
    elif train_file[c].dtype == 'int32':
        if min_val>np.iinfo(np.int8).min and max_val<np.iinfo(np.int8).max:
            train_file[c] = train_file[c].astype(np.int8)
        elif min_val>np.iinfo(np.int16).min and max_val<np.iinfo(np.int16).max:
            train_file[c] = train_file[c].astype(np.int16)
train_file.info()


# ### That's a great reduction in memory usage (around 73% reduction)! It will help us go further efficiently!

# In[ ]:


# Let's have a look at train data
print(train_file.columns)
train_file.sample(10)


# ### Is there any NAN values in the data?

# In[ ]:


print('There are %s NAN values in the train data'%train_file.isnull().sum().sum())


# # 3. DEFINE PROBLEM
# 
# ### The data provided is a time-series one without explicit target. The problem being binary classification, to decide whether to do trade or not. A specific target "***action***" (named as required in final output) can be introduced based on the value of variable "***resp***" as of now for handling missing values. Values of resp above 0 can be considered a positive signal to trade (action = 1) otherwise say no to trade (action = 0) 
# 
# ### While modeling neural network, we can consider all of the resp, resp_1, resp_2, resp_3 and resp_4 as contributors of our target ***action***.

# In[ ]:


train_file['action'] = (train_file['resp']>0)*1
print('There are %s transactions doing trade and %s transactions not doing trade'%((train_file.action==1).sum(), (train_file.action==0).sum()))


# ### Now the target column has been generated. Trade transactions and No-trade transactions are almost balanced.

# # 4. EXPLORE DATA
# 
# ### It is time to find correlations among features. Since we are using neural network, feature engineering is not mandatory. However for handling missing values data exploration is necessary. 

# In[ ]:


# Let's have a look at number of missing values in each of the features
plt.figure(figsize=(16,5))
null = train_file.isnull().sum()
null = null[null>0]
null.plot()
plt.xticks(np.arange(len(null)), null.index, rotation=90)
plt.xlabel('Features with NAN values', size=14)
plt.title('Number of NAN values in the data', size=16, color='orange')
plt.show()


# ### A very interesting dataset! Even missing values do follow a distributed pattern among features. As said, features are anonymized but not shuffled. They may tell great stories if explored properly. So more care is needed on handling those missing values

# In[ ]:


null_100 = null[null>=100000].index.to_list()
null_50 =  null[null>=50000].index.to_list()
null_10 =  null[null>=10000].index.to_list()
null_5 =  null[null>=5000].index.to_list()
null_0 =  null[null>0].index.to_list()
null_0 = [c for c in null_0 if c not in null_5]
null_5 = [c for c in null_5 if c not in null_10]
null_10 = [c for c in null_10 if c not in null_50]
null_50 = [c for c in null_50 if c not in null_100]


# In[ ]:


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,10))
ax1.plot(null[null_100])
ax2.plot(null[null_50])
ax3.plot(null[null_10])
ax4.plot(null[null_5])
ax1.set_title('Features with more than 100,000 missing values',size=14,color='r')
ax2.set_title('Features with more than 50,000 missing values',size=14,color='r')
ax3.set_title('Features with more than 10,000 missing values',size=14,color='r')
ax4.set_title('Features with more than 5,000 missing values',size=14,color='r')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
null[null_0].plot()
plt.title('Features with less than 5,000 missing values', size=14, c='r')
plt.xticks(range(len(null_0)), null_0, rotation=90)
plt.show()


# In[ ]:


features = train_file.columns[train_file.columns.str.contains('feature')]
print("Features without any missing values are: ", [c for c in features if c not in null.index])


# ### We have some basic views on missing values. Let's try to explore other insights of features 

# In[ ]:


# Let's find number of days the data has
pd.Series(train_file.date.value_counts())


# ### It is observed that 2390491 transactions belong to just 500 days; Transactions-per-day varies from day to day. Day 294 has the minimum of only 29 transactions whereas Day 44 has the maximum of 18884 transactions in total.

# In[ ]:


# Let's have a distribution plot of transactions occur per day to gain better insights
plt.figure(figsize=(16,5))
ax = sns.distplot(train_file.date.value_counts(), bins=250, kde=False)
heights = np.array([rec.get_height() for rec in ax.patches])
normalizer = plt.Normalize(heights.min(), heights.max())
cmap = plt.cm.jet(normalizer(heights))
for rec, color in zip(ax.patches, cmap):
    rec.set_color(color)
plt.xlabel("Distribution of per-day-transactions", size=14)
plt.ylim([-2,22])
plt.show()


# ### Mostly the number of transactions per day ranges from 2500 to 7500. It will be good to have sample days one from lesser-transactions-day and another from more-transaction-day to have better understanding on handling missing values

# In[ ]:


dict(train_file.date.value_counts())


# ### We randomly select day 96, day 242 and day 454 as three representatives of lower number of transactions (2785), most probable number of transactions (4025) and higher number of transactions (7300). Let's have a look at those days

# In[ ]:


plt.figure(figsize=(15,6))
(train_file.groupby('date').apply(lambda x: x.isnull().sum()).sum(axis=1)).plot(color='red')
plt.title('Number of Missing values as per date', size=16, color='g')
plt.ylabel('Number of missing values', size=14)
plt.xlabel('Date',size=14)
plt.show()


# In[ ]:


day_96 = train_file.query('date==96')
day_242 = train_file.query('date==242')
day_454 = train_file.query('date==454')


# ### We can visualize missing values in these representative dates

# In[ ]:


msno.matrix(day_96,color=(0.2,0.2,0.7))
msno.matrix(day_242,color=(0.5,0.2,0.5))
msno.matrix(day_454,color=(0.7,0.5,0.2))
plt.show()


# ### Awesome. Irrespective of number of transactions, missing values do have the same pattern based on time. (It is true to even highest number of transactions day; I tested with date=44). 
# 
# ### So missing values should not be 
# * ### filled with mean values or nanmean values
# * ### filled with any kind of ffill method or something
# * ### dropped
# 
# ### Rather, they should be filled with special distinguishable values so that our model can recognize them! We can make our data to communicate our model regarding start/interval timings via those special values.

# # 5. FILLING MISSING VALUES

# ### We can start by learning about the values in the missing value features. We first consider the features which have missing values lesser than 5,000 in their columns.

# In[ ]:


# Have a look at data and missing values
pd.options.display.max_columns = None
train_file[null_0][train_file[null_0].isnull().any(axis=1)]


# In[ ]:


train_file[null_0].describe().T


# In[ ]:


f, ax = plt.subplots(4,7, figsize=(18,30))
for k,feature in enumerate(null_0):
    i = k//7
    j = k%7
    sns.boxplot(y=feature, x='action', data=train_file, ax = ax[i][j])
    ax[i][j].set_title(str(feature), size=14, color='r')
    ax[i][j].set_ylabel('')
    ax[i][j].set_xlabel('')
plt.suptitle('Features having less than 5,000 missing values', size=16, color='blue')
#plt.tight_layout()
plt.show()


# In[ ]:


f, ax = plt.subplots(2,8, figsize=(18,20))
for k,feature in enumerate(null_5):
    i = k//8
    j = k%8
    sns.boxplot(y=feature, x='action', data=train_file, ax = ax[i][j])
    ax[i][j].set_title(str(feature), size=14, color='g')
    ax[i][j].set_ylabel('')
    ax[i][j].set_xlabel('')
plt.suptitle('Features having more than 5,000 missing values', size=16, color='blue')
#plt.tight_layout()
plt.show()


# In[ ]:


f, ax = plt.subplots(2,8, figsize=(18,20))
for k,feature in enumerate(null_10):
    i = k//8
    j = k%8
    sns.boxplot(y=feature, x='action', data=train_file, ax = ax[i][j])
    ax[i][j].set_title(str(feature), size=14, color='m')
    ax[i][j].set_ylabel('')
    ax[i][j].set_xlabel('')
plt.suptitle('Features having more than 10,000 missing values', size=16, color='blue')
#plt.tight_layout()
plt.show()


# In[ ]:


f, ax = plt.subplots(2,9, figsize=(18,20))
plt.suptitle('Features having more than 50,000 missing values', size=16, color='blue')
for k,feature in enumerate(null_50):
    i = k//9
    j = k%9
    sns.boxplot(y=feature, x='action', data=train_file, ax = ax[i][j])
    ax[i][j].set_title(str(feature), size=14, color='tab:brown')
    ax[i][j].set_ylabel('')
    ax[i][j].set_xlabel('')
#plt.tight_layout()
plt.show()


# In[ ]:


f, ax = plt.subplots(2,7, figsize=(16,20))
for k,feature in enumerate(null_100):
    i = k//7
    j = k%7
    sns.boxplot(y=feature, x='action', data=train_file, ax = ax[i][j])
    ax[i][j].set_title(str(feature), size=14, color='orange')
    ax[i][j].set_ylabel('')
    ax[i][j].set_xlabel('')
plt.suptitle('Features having more than 100,000 missing values', size=16, color='blue')
#plt.tight_layout()
plt.show()


# ### Outliers in each feature compelled me to make taller plots to have a better view. 
# 
# ### Features with different missing value counts do have some similarity among them. Mean of almost every feature lies around zero. Every feature necessarily has great positive outliers. But around half of the features do not have remarkable negative outliers.
# 
# ### So I decide to communicate about MISSING VALUES to my model by assigning them with newer and relatively bigger NEGATIVE OUTLIERS for all those features which have missing values. (And I strongly believe it will work)

# ## A function to fill missing values in train data, test data and future data

# In[ ]:


val_range = train_file[features].max()-train_file[features].min()
filler = pd.Series(train_file[features].min()-0.01*val_range, index=features)
# This filler value will be used as a constant replacement of missing values 

# A function to maintain data type consistency of dataframe
dtype_dict = dict(train_file[features].dtypes)
def consistent_dtype(df):
    return df.astype(dtype_dict)


# In[ ]:


def fill_missing(df):
    df[features] = np.nan_to_num(df[features]) + filler*np.isnan(df[features])
    return df        


# In[ ]:


train = fill_missing(train_file)
train = consistent_dtype(train)


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum().sum()


# ### We have successfully filled all of the missing values with distinguishable values

# # 6. LSTM MODELING
# 

# ### Let's split data into features (X) and target (y)

# In[ ]:


train = train.loc[train.weight > 0]
X = train[features]
y = train[[c for c in train.columns if 'resp' in c]]
y = (y>0)*1
y = (y.mean(axis=1)>0.5).astype(np.int8)


# ## Normalize data for LSTM model

# In[ ]:


X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
def Normalize(df):
    return (df - X_mean)/X_std
X = Normalize(X)


# ## Let's split data into train and validation set in 80-20 ratio

# In[ ]:


X_train = X[train.date<=400]
X_val = X[train.date>400]
y_train = y[train.date<=400]
y_val = y[train.date>400]


# In[ ]:


# LSTM expects 3D input (examples, timestep, features)
print(X_train.shape, X_val.shape)
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
print(X_train.shape, X_val.shape)


# In[ ]:


# A Sequential model
batch_size = 256
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, batch_size=batch_size, input_shape=(1,130), return_sequences=True ),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')


# In[ ]:


train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
train = train.cache().batch(batch_size).repeat()


# In[ ]:


del train_file
del null_0
del null_5
del null_10
del null_50
del null_100
gc.collect()


# In[ ]:


del X_train
del X_val
del y_train
del y_val
gc.collect()


# In[ ]:


model.summary()


# ## Training the model

# In[ ]:


hist = model.fit(train, epochs=20, steps_per_epoch=200, validation_data=val, validation_steps=50)


# In[ ]:


# Let's clear some memory
del train
del val
del X
del y
gc.collect()


# # 7. PREDICTION AND SUBMISSION

# In[ ]:


example_submission_file.info()


# In[ ]:


from tqdm import tqdm
import janestreet
env = janestreet.make_env()
for test_file,pred in tqdm(env.iter_test()):
    if test_file.weight.item()==0:
        pred.action = np.int64(0)
    else:
        test = test_file[features]
        test = fill_missing(test)
        test = consistent_dtype(test)
        test = Normalize(test)
        test = test.values
        test = np.repeat(test, batch_size).reshape(-1, batch_size).T.reshape(batch_size,1,130)
        # LSTM requires data in the right shape only
        action = model(test)
        a = 1 if action[0][0].item()>0.5 else 0
        pred.action = np.int64(a)
        test = test[0]
        test = np.squeeze(test)
    env.predict(pred)


# ### Thank you all for your patience! I am here to learn; Kindly express your suggestions, if any.
