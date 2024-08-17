#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# #import my kaggle_utiles file that has all the custom funcitons i want.
# import sys
# sys.path.append("/home/pavithra/Pictures/learning/ML/kaggle/")
# sys.path
import kaggle_utils_py as kaggle_utils

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import warnings 

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Concatenate, LSTM, GRU
from tensorflow.keras.layers import Bidirectional, Multiply


from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, GroupKFold

warnings.simplefilter("ignore")


# In[2]:


train = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
train_labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
sub = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")


# In[3]:


print("Shape of the data --->",train.shape)
print("Shape of the test data --->",test.shape)


# In[4]:


train.head(10)


# In[5]:


display(train_labels.head())
print("Shape of the label --->", train_labels.shape)


# ## features
# ### train.csv - the training set, comprising ~26,000 60-second recordings of thirteen biological sensors for almost one thousand experimental participants
# - sequence - a unique id for each sequence
# - subject - a unique id for the subject in the experiment
# - step - time step of the recording, in one second intervals
# - sensor_00 - sensor_12 - the value for each of the thirteen sensors at that time step
# ### train_labels.csv - the class label for each sequence.
# - sequence - the unique id for each sequence.
# - state - the state associated to each sequence. This is the target which you are trying to predict.

# In[6]:


# merge the dataset 
# adding labels to the train data
data = pd.merge(train, train_labels,how='left', on="sequence")


# ### find the window size 
# we have 25968 labels and 1558080 (60 * 25968) --> train data samples (there is no null values). each sequence has 60 steps(1 min) marked as 0 -59. So that could be the window size
# 

# In[7]:


data[data["sequence"] == 0]


# <a id="1.1"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.1 ) Common data Analysis</p>

# In[8]:


columns, categorical_col, numerical_col,missing_value_df = kaggle_utils.Common_data_analysis(data, missing_value_highlight_threshold=5.0, display_df = True,
                                                                                         only_show_missing=False)


# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.2 ) Numerical Data -- descriptive, distribution, Quantitative</p>

# In[9]:


kaggle_utils.numerical_data_analysis(data[numerical_col], numerical_col)


# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.3 ) Distribution Analysis</p>

# In[10]:


def plot_kde(data, columns, nrow, ncol, figsize, hue_value=None):
    # find the distubution of the data. ( visualization would be so good)
    fig, ax = plt.subplots(nrow,ncol, figsize=figsize)
    # we have 9 numerical values.
    col, row = ncol,nrow
    col_count = 0
    if row<=1:
        for c in range(col):
            if hue_value:
                sns.kdeplot(data=data, x=columns[col_count],hue=hue_value, ax=ax[c])
            else:
                sns.kdeplot(data=data, x=columns[col_count], ax=ax[c])
    else:
        for r in range(row):
            for c in range(col):
                if col_count >= len(columns):
                    ax[r,c].text(0.5, 0.5, "no data")
                else:
                    if hue_value:
                        sns.kdeplot(data=data, x=columns[col_count],hue=hue_value, ax=ax[r,c])
                    else:
                        sns.kdeplot(data=data, x=columns[col_count], ax=ax[r,c])
                col_count +=1


# <a id="1.3.2"></a>
# # <p style="font-family:newtimeroman;color:#5811D3;font-size:100%;text-align:left;border-radius:10px 10px;">2.1 | Sequence/subject distribution</p>

# In[11]:


# some visualization
col_name = ['sequence', 'subject', 'step']
plot_kde(data, col_name, 1, 3, figsize=(18,8), hue_value="state")


# In[12]:


# some visualization
col_name = ['sequence', 'subject', 'step']
fig, ax = plt.subplots(1, 3, figsize=(18,8))
for col in range(3):    
    sns.histplot(data=data, x=col_name[col], hue="state", ax=ax[col])


# In[13]:


# some visualization
col_name = ['sequence', 'subject', 'step']
plot_kde(data, col_name, 1, 3, figsize=(18,8))


# ## observation
# - step - has a uniform distribution accross all the data as well as based on state
# - sequence - has  a uniform distribution accross all the data

# <a id="1.3.2"></a>
# # <p style="font-family:newtimeroman;color:#5811D3;font-size:100%;text-align:left;border-radius:10px 10px;">2.2 | Sensor distribution</p>

# In[14]:


sensor_cols = ['sensor_'+'%02d'%i for i in range(1, 13)]
plot_kde(data, sensor_cols, 3, 4, figsize=(18,10))


# In[15]:


def plot_hist(data, columns, nrow, ncol, figsize, hue_value=None):
    # find the distubution of the data. ( visualization would be so good)
    fig, ax = plt.subplots(nrow,ncol, figsize=figsize)
    # we have 9 numerical values.
    col, row = ncol,nrow
    col_count = 0

    for r in range(row):
        for c in range(col):
            if col_count >= len(columns):
                ax[r,c].text(0.5, 0.5, "no data")
            else:
                if hue_value:
                    sns.boxplot(data=data, x=columns[col_count],hue=hue_value, ax=ax[r,c])
                else:
                    sns.boxplot(data=data, x=columns[col_count], ax=ax[r,c])
            col_count +=1


sensor_cols = ['sensor_'+'%02d'%i for i in range(1, 13)]
plot_hist(data, sensor_cols, 3, 4, figsize=(18,10))


# ## observation
# - Most of sensor datas are  normally distributed with outliers.
# - All sensors have large set of zero values :(
# - all sensors except 'sensor_02' has outliers at both side -- 'sensor_02' has outliers left side

# In[16]:


def plot_scatter(data, columns, nrow, ncol, figsize, hue_value=None):
    # find the distubution of the data. ( visualization would be so good)
    fig, ax = plt.subplots(nrow,ncol, figsize=figsize)
    # we have 9 numerical values.
    col, row = ncol,nrow
    col_count = 0

    for r in range(row):
        for c in range(col):
            if col_count >= len(columns):
                ax[r,c].text(0.5, 0.5, "no data")
            else:
                if hue_value:
                    sns.scatterplot(data=data, x=columns[col_count],y=data.index, hue=hue_value, ax=ax[r,c])
                else:
                    sns.scatterplot(data=data, x=columns[col_count],y = data.index, ax=ax[r,c])
            col_count +=1


sensor_cols = ['sensor_'+'%02d'%i for i in range(1, 13)]
plot_scatter(data, sensor_cols, 3, 4, figsize=(18,10), hue_value='state')


# ## time series 

# In[17]:


sequences = [0, 1, 2, 8364, 15404]
figure, axes = plt.subplots(13, len(sequences), sharex=True, figsize=(16, 16))
for i, sequence in enumerate(sequences):
    for sensor in range(13):
        sensor_name = f"sensor_{sensor:02d}"
        plt.subplot(13, len(sequences), sensor * len(sequences) + i + 1)
        plt.plot(range(60), train[train.sequence == sequence][sensor_name],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        if sensor == 0: plt.title(f"Sequence {sequence}")
        if sequence == sequences[0]: plt.ylabel(sensor_name)
figure.tight_layout(w_pad=0.1)
plt.suptitle('Selected Time Series', y=1.02)
plt.show()
# This part of code take from Ambros EDA Section. :0


# <a id="2.3"></a>
# # <p style="font-family:newtimeroman;color:#5811D3;font-size:100%;text-align:left;border-radius:10px 10px;"> 2.3 | Target Class balance check(only for classification)</p>

# In[18]:


# mostly has equal number of samples in both classes
sns.countplot(data=data, y="state")


# In[19]:


count = data["state"].value_counts()
print(count)
print("percentage of first class --- >",count[0]/data.shape[0])
print("percentage of second class --->", count[1]/data.shape[0])


# In[20]:


# data for the first 60 seconds
data[data['sequence']==0]


# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.3 ) Outlier Detection</p>

# - There are lots of hypothesis tests to find the presents of outlier. Since our sensor data follows almost normal distribution we can go with **Grubbs Test**

# In[21]:


def grubbs_test(feature_value, col_name):
    print("{:=^40}".format(f" Test starts for {col_name}"))
    n = len(feature_value)
    mean_feature = np.mean(feature_value)
    st_dev_feature = np.std(feature_value)
    g = (max(abs(feature_value-mean_feature))) / st_dev_feature
    print("Grubbs test statistic value:",g)

    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    print("Grubbs Critical Value:",g_critical)
    if g > g_critical:
        print("So our G value is greater than G critical value --> so reject the null hypothesis -- variable has atleast one outlier :(")
    else:
        print("So our G value is lesser than G critical value --> so accept the null hypothesis -- variable has no outlier :) ")
    


# In[22]:


for col in sensor_cols:
    grubbs_test(data[col], col)


# ## observation
# - All the variables has outliers :( -- will get the number of outliers to come to an solution

# In[23]:


df, outlier_df, lower_limit_df, upper_limit_df = kaggle_utils.find_outlier_z_score_method(data,new_feature=True, return_limits=True)
outlier_df["percentage of outlier"] = outlier_df["Number of outliers"] / data.shape[0]
outlier_df


# ## observation
# - We have some amount of outliers -- we can build a deep nueral network / we need some create feature engineering to deal with this outliers. I am going to use deep nueral network so now we can leave this outliers as it is.

# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.4 | Correlation</p>

# In[24]:


plt.figure(figsize=(25,8))
sns.heatmap(data.corr(), annot=True, cbar=True, cmap="YlGnBu")


# ## observation
# - **No null** values :)
# - **No categorical** values :)
# - **Not so much correlated** features :)
# - All features are **almost normally** distributed :)
# - **Balanced target** :) :)
# - All features has **outliers** :( --> but we are going to use deep nueral networks , so no need to worry about this :)
# 
# - [action] Only thing we have limited features -- need most powerful feature engineering 
# - [action] **Need to scale** the data. 

# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.4 | Feature Engineering </p>

# In[25]:


for sensor in sensor_cols: 
    data[f"{sensor}" + '_lag1'] = data.groupby('sequence')[f"{sensor}"].shift(1)  
    data[f"{sensor}" + '_lag1'].fillna(0, inplace=True)
    data[f"{sensor}" + '_diff1'] = data[f"{sensor}"] - data[f"{sensor}" + '_lag1']

    # do the same for test data
    test[f"{sensor}" + '_lag1'] = test.groupby('sequence')[f"{sensor}"].shift(1)  
    test[f"{sensor}" + '_lag1'].fillna(0, inplace=True)
    test[f"{sensor}" + '_diff1'] = test[f"{sensor}"] - test[f"{sensor}" + '_lag1']


# In[26]:


data.head()


# In[27]:


# scale the data 
stand_scale = StandardScaler() # since our data almost look like normal distribution Standscalar is the best option here.

train_X = data[data["state"].isnull() == False]
test_X = test.copy()

col = data.columns.tolist()[3:]
col.remove("state")
train_X[col] = stand_scale.fit_transform(train_X[col])
test[col] = stand_scale.transform(test[col])


# In[28]:


# gte the label 
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
labels = train_labels["state"]

train_X = train_X.drop(["sequence", "subject", "step",'state'], axis=1).values
train_X = train_X.reshape(-1, 60, train_X.shape[-1])

test = test.drop(["sequence", "subject", "step"], axis=1).values
test = test.reshape(-1, 60, test.shape[-1])


# In[29]:


groups = data["sequence"]


# In[30]:


train_X.shape


# <a id="1.2"></a>
# # <p style="background-color:#5811D3;font-family:newtimeroman;color:#FEDFA0;font-size:100%;text-align:left;border-radius:10px 10px;">1.4 | Modelling </p>

# In[31]:


# doing it ... new to LSTM and RNN learning it..................

