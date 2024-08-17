#!/usr/bin/env python
# coding: utf-8

# <h1><center>Mechanisms of action (MoA)
# : Data analysis, visualization, and modeling</center></h1>
# <center><img src="https://thumbs.dreamstime.com/z/gene-therapy-tablets-genetic-code-inside-concept-advancement-medicine-treatment-diseases-57708501.jpg" width="60%"></center>
# 

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center>Quick navigation</center></h2>
# 
#     
#     
# * [Problem Description](#1)
# * [Explanatory Data Analysis (EDA)](#2)
#     - [Example data](#21)
#     - [Missing values](#22)
#     - [Features](#23)
#         - [Gene expression features](#231)
#         - [Cell viability features](#232)
#         - [Cp_time and cp_dose](#233)
#     - [Exploring some realationships](#24)
#     - [Targets](#25)
#     - [Preprocessing and feature engineering](#26)
# * [Training](#3)
#     - [Model definition](#31)
#     - [Training and validation](#32)
#     - [Blending](#33)
# * [Evaluation and Summary](#4)
# * [References](#5)
# 

# <a id="1"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center>Problem Description</center></h2>
# 
# It is important to ask the right question before trying to solve it! Can we predict mechanism of action (MoA) of a drug based on gene expression and cell viability data? Or better to ask first, what is mechansim of action? The term mechanism of action means the biochemical interactions through which a drug generates its pharmacological effect. Scientists know many MoAs of drugs, for example, an antidepressant may have a selective serotonin reuptake inhibitor (SSRI), which affects the brain serotonin level. In this project we are going to train a model that classifies drugs based on their biological activity. The dataset consists of different features of gene expression data, and cell viability data as well as multiple targets of mechansim of action (MoA). This problem is a multilabel classification, which means we have multiple targets (not multiple classes). In this project, we will first perform explanatory data analysis and then train a model using deep neural networks with Keras. We will do a bit model evaluation at the end.

# In[ ]:


# Importing useful libraries
import warnings
warnings.filterwarnings("ignore")

# Adding iterative-stratification 
# Select add data from the right menu and search for iterative-stratification, then add it to your kernel.
import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from time import time
import datetime
import gc

import numpy as np
import pandas as pd 

# ML tools 
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf 
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn.metrics import log_loss
from tensorflow_addons.layers import WeightNormalization
# Setting random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
sns.set(font_scale=1.2)



# <a id="2"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center>Explanatory Data Analysis (EDA)</center></h2>
# 

# <a id="21"></a>
# ## Example data
# First, we are going to see the train and test data size and some of their examples. Please note there are two different target dataframes, non-scored and scored. The non-scored ones are not used for scoring, but we can make use of them to pretrain our network
# <a href="https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing"> [1]</a>
#  

# In[ ]:


df_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
display(df_train.head(3))
print('train data size', df_train.shape)

df_target_ns = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
display(df_target_ns.head(3))
print('train target nonscored size', df_target_ns.shape)


df_target_s = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
display(df_target_s.head(3))
print('train target scored size', df_target_s.shape)


df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
display(df_test.head(3))
print('test data size', df_test.shape)

df_sample = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
display(df_sample.head(3))
print('sample submission size', df_sample.shape)


# <a id="22"></a>
# ## Missing values
# Let's see if there are any missing values, and see some information about our data types. 

# In[ ]:


print(df_train.isnull().sum().any()) # True if there are missing values
print(df_train.info())


# There are no missing values; there are 872 float dtypes, 1 integer and 3 objects. Let's print the latter ones.

# In[ ]:


display(df_train.select_dtypes('int64').head(3))
display(df_train.select_dtypes('object').head(3))


# <a id="23"></a>
# ## Features
# Let's visualize some features randomly:
# <a id="231"></a>
# ### Gene expression features

# In[ ]:


g_features = [cols for cols in df_train.columns if cols.startswith('g-')]


# In[ ]:


color = ['dimgray','navy','purple','orangered', 'red', 'green' ,'mediumorchid', 'khaki', 'salmon', 'blue','cornflowerblue','mediumseagreen']
 
color_ind=0
n_row = 6
n_col = 3
n_sub = 1 
plt.rcParams["legend.loc"] = 'upper right'
fig = plt.figure(figsize=(8,14))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in (np.arange(0,6,1)):
    plt.subplot(n_row, n_col, n_sub)
    sns.kdeplot(df_train.loc[:,g_features[i]],color=color[color_ind],shade=True,
                 label=['mean:'+str('{:.2f}'.format(df_train.loc[:,g_features[i]].mean()))
                        +'  ''std: '+str('{:.2f}'.format(df_train.loc[:,g_features[i]].std()))])
    
    plt.xlabel(g_features[i])
    plt.legend()                    
    n_sub+=1
    color_ind+=1
plt.show()


# <a id="232"></a>
# ### Cell viability features

# In[ ]:


c_features = [cols for cols in df_train.columns if cols.startswith('c-')]


# In[ ]:


n_row = 6
n_col = 3
n_sub = 1 
fig = plt.figure(figsize=(8,14))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
plt.rcParams["legend.loc"] = 'upper left'
for i in (np.arange(0,6,1)):
    plt.subplot(n_row, n_col, n_sub)
    sns.kdeplot(df_train.loc[:,c_features[i]],color=color[color_ind],shade=True,
                 label=['mean:'+str('{:.2f}'.format(df_train.loc[:,c_features[i]].mean()))
                        +'  ''std: '+str('{:.2f}'.format(df_train.loc[:,c_features[i]].std()))])
    
    plt.xlabel(c_features[i])
    plt.legend()                    
    n_sub+=1
    color_ind+=1
plt.show()


# It seems data are somehow normalized and also clipped at -10, 10. Please see this great discussion here: <a href="https://www.kaggle.com/c/lish-moa/discussion/184005"> [2] </a>

# <a id="233"></a>
# ### Cp_time and cp_dose
# 
# cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low which are D1 and D2).

# In[ ]:


fig = plt.figure(figsize=(10,4))
plt.subplots_adjust(right=1.3)
plt.subplot(1, 2, 1)
sns.countplot(df_train['cp_time'],palette='nipy_spectral')
plt.subplot(1, 2, 2)
sns.countplot(df_train['cp_dose'],palette='nipy_spectral')
plt.show()


# We can see there are almost the same number of examples in each treatment duration and dosage features.

# <a id="24"></a>
# 
# ## Exploring some relationships

# Next, we can use stripplot to show the relationship of a feature and a target with respect to dosage and time. Since, this is a multilabel probelm, we only show one label here, which is target 71. We will see later this target is contributing the most to the loss. For the feature, we chose two random g and c features. You may wanna do this with other features and labels to get more insight. 

# In[ ]:


train_copy= df_train.copy()
train_copy['target_71'] = df_target_s.iloc[:,72] # sig_id is included
fig = plt.figure(figsize=(16,8))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy , x='cp_time', y= 'g-3',color='red', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'g-3',color='red', hue='target_71',ax=ax2)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(16,8))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy, x='cp_time', y= 'c-1',color='yellow', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'c-1',color='yellow', hue='target_71',ax=ax2)
plt.show()


# Or we can do the same process with the mean of g and c features. For example, here we plot the mean of g and c features with respect to a target, dosage and time. 

# In[ ]:


train_copy['g_mean'] = train_copy.loc[:, g_features].mean(axis=1) 
fig = plt.figure(figsize=(16,10))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy , x='cp_time', y= 'g_mean',color='red', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'g_mean', color='red', hue='target_71',ax=ax2)
plt.show()


# In[ ]:


train_copy['c_mean'] = train_copy.loc[:, c_features].mean(axis=1) 
fig = plt.figure(figsize=(16,10))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy, x='cp_time', y= 'c_mean',color='yellow', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'c_mean', color='yellow', hue='target_71',ax=ax2)
plt.show()


# We can get some insights from the figures above and apply it in our [Preprocessing](#26) step 

# <a id="25"></a>
# ## Targets
# Below are some scored targets which are used to train the main model. As we can see, the targets are very imbalanced and there are only a few positive examples in some labels. 

# In[ ]:


target_s_copy = df_target_s.copy()
target_s_copy.drop('sig_id', axis=1, inplace=True)
n_row = 20
n_col = 4 
n_sub = 1   
fig = plt.figure(figsize=(20,50))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in np.random.choice(np.arange(0,target_s_copy.shape[1],1),n_row):
    plt.subplot(n_row, n_col, n_sub)
    sns.countplot(y=target_s_copy.iloc[:, i],palette='nipy_spectral',orient='h')
    
    plt.legend()                    
    n_sub+=1
plt.show()


# Let's see the 20 largest positive number of labels in the scored targets. 

# In[ ]:


plt.figure(figsize=(10,10))
target_s_copy.sum().sort_values()[-20:].plot(kind='barh',color='mediumseagreen')
plt.show()


# And here are some non-scored targets. We can see that some labels do no have positive examples at all.

# In[ ]:


target_ns_copy = df_target_ns.copy()
target_ns_copy.drop('sig_id', axis=1, inplace=True)
n_row = 20
n_col = 4 
n_sub = 1   
fig = plt.figure(figsize=(20,50))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in np.random.choice(np.arange(0,target_ns_copy.shape[1],1),n_row):
    plt.subplot(n_row, n_col, n_sub)
    sns.countplot(y=target_ns_copy.iloc[:, i],palette='magma',orient='h')
    
    plt.legend()                    
    n_sub+=1
plt.show()


# And here is the 20 largest positive number of labels in the non-scored targets. 

# In[ ]:


plt.figure(figsize=(10,10))
target_ns_copy.sum().sort_values()[-20:].plot(kind='barh',color='purple')
plt.show()


# As we can see, there are fewer positive examples in non-scored dataset.

# [](http://)<a id="26"></a>
# ## Preprocessing and feature engineering

# The control group is defined as the group in an experiment or study that does not have the desired effect or MoAs here; which means the target labels are zero for them. I will drop the data for this group, and we will later set all predictions of this group to zero.
# 
# We will keep track of the control group (ctl_vehicle) indexes. 
# I dropped cp_type column and mapped the values of time and dose features. I performed some feature engineering based on the insights I got from the [Exploring some relationships](#24) part. <br>
# Update : I added the methods of Rankgauss scaler and PCA from this great kernel : <a href='https://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn?scriptVersionId=44558776' >[3] 

# In[ ]:


ind_tr = df_train[df_train['cp_type']=='ctl_vehicle'].index
ind_te = df_test[df_test['cp_type']=='ctl_vehicle'].index


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(n_quantiles=100,random_state=42, output_distribution="normal")

def preprocess(df):
    df['cp_time'] = df['cp_time'].map({24:1, 48:2, 72:3})
    df['cp_dose'] = df['cp_dose'].map({'D1':0, 'D2':1})
    g_features = [cols for cols in df.columns if cols.startswith('g-')]
    c_features = [cols for cols in df.columns if cols.startswith('c-')]
    for col in (g_features + c_features):
        vec_len = len(df[col].values)
        raw_vec = df[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    return df

X = preprocess(df_train)
X_test = preprocess(df_test)

display(X.head(5))
print('Train data size', X.shape)
display(X_test.head(3))
print('Test data size', X_test.shape)
y = df_target_s.drop('sig_id', axis=1)
display(y.head(3))
print('target size', y.shape)
y0 =  df_target_ns.drop('sig_id', axis=1)


# In[ ]:


# Please see reference 3 for this part
g_features = [cols for cols in X.columns if cols.startswith('g-')]
n_comp = 0.95

data = pd.concat([pd.DataFrame(X[g_features]), pd.DataFrame(X_test[g_features])])
data2 = (PCA(0.95, random_state=42).fit_transform(data[g_features]))
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_g-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns=[f'pca_g-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis=1)
X_test = pd.concat((X_test, test2), axis=1)

c_features = [cols for cols in X.columns if cols.startswith('c-')]
n_comp = 0.95

data = pd.concat([pd.DataFrame(X[c_features]), pd.DataFrame(X_test[c_features])])
data2 = (PCA(0.95, random_state=42).fit_transform(data[c_features]))
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_c-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns=[f'pca_c-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis=1)
X_test = pd.concat((X_test, test2), axis=1)

display(X.head(2))
display(X_test.head(2))


# In[ ]:


def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

X,X_test=fe_stats(X,X_test)
display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)


# In[ ]:


from sklearn.cluster import KMeans
def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 239):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

X ,X_test=fe_cluster(X,X_test)
display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)


# In[ ]:


from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(0.8)  
data = X.append(X_test)
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : X.shape[0]]
test_features_transformed = data_transformed[-X_test.shape[0] : ]


X = pd.DataFrame(X[['sig_id','cp_type', 'cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

X = pd.concat([X, pd.DataFrame(train_features_transformed)], axis=1)


X_test = pd.DataFrame(X_test[['sig_id','cp_type', 'cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

X_test = pd.concat([X_test, pd.DataFrame(test_features_transformed)], axis=1)

display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)


# In[ ]:


y0 = y0[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
y = y[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
X = X[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
X.drop(['cp_type','sig_id'], axis=1, inplace=True)
X_test.drop(['cp_type','sig_id'], axis=1, inplace=True)

print('New data shape', X.shape)


# <a id="3"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center> Training</center></h2>

# <a id="31"></a>
# ## Model definition
# Here we define our neural network model which consists of several dense, dropout and batchnorm layers. I used different activations after my dense layers. We first train the network on non-scored targets and then transfer the weights to train another model on the scored targets. Smoothing the labels may prevent the network from becoming over-confident and has some sort of regularization effect <a href="https://www.kaggle.com/rahulsd91/moa-label-smoothing">[4] </a>. It seems this method works well here. I used Keras Tuner to tune the hyperparameters. The details are in this notebook <a href="https://www.kaggle.com/sinamhd9/hyperparameter-tuning-with-keras-tuner">[5] </a>

# In[ ]:


p_min = 0.001
p_max = 0.999
from tensorflow.keras import regularizers

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -K.mean(y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred))

def create_model(num_cols, hid_layers, activations, dropout_rate, lr, num_cols_y):
    
    inp1 = tf.keras.layers.Input(shape = (num_cols, ))
    x1 = tf.keras.layers.BatchNormalization()(inp1)

    for i, units in enumerate(hid_layers):
        x1 = tf.keras.layers.Dense(units, activation=activations[i])(x1)
        x1 = tf.keras.layers.Dropout(dropout_rate[i])(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
    
    x1 = tf.keras.layers.Dense(num_cols_y,activation='sigmoid')(x1)
    model = tf.keras.models.Model(inputs= inp1, outputs= x1)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                 loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001), metrics=logloss)
    
    return model 
    


# In[ ]:


hid_layers = [[[512, 768, 896],[384, 640, 1024],[768,768,896],[512, 384, 1024],
              [512, 640, 640], [640, 896, 1024], [256,640,896],[512,512,768],
              [512, 384, 896],[512,768,768]],
             [[512, 768, 896],[384, 640, 1024],[768,768,896],[512, 384, 1024],
              [512, 640, 640], [640, 896, 1024], [256,640,896],[512,512,768],
              [512, 384, 896],[512,768,768]],
              [[1152, 640, 3072],[896, 1024, 3584],[1920,1024,3712],[896, 1024, 3456],
              [1408, 896, 3456], [1408, 768, 3456], [2176,640,3840],[1664,1280,2688],
              [1792, 768, 2432],[1280,1664,4096]],
            [[896, 1792, 3712],[2048, 1024, 1664],[1664,1408,1920],[896, 1408, 3072],
              [1152, 1152, 3072], [2816, 3072, 3328], [2304,2432,2176],[3968,4096,2816],
              [1920, 1536, 3072],[128,1920,1664]],
         [[896, 1152, 1408],[1920, 768, 2176],[2048,2048,1792],[2304, 1664, 512],
              [768, 384, 512], [640, 1664, 512], [1664,1920,2688],[2432,1664,1536],
              [640, 896, 2432],[1536,2176,2176]]]

dropout_rate = [[[0.65,0.35,0.35],[0.65,0.35,0.45],[0.7,0.4,0.4],[0.65,0.35,0.45],
                [0.65,0.35,0.45],[0.7,0.3,0.45],[0.7,0.35,0.4],[0.7,0.4,0.4],
               [0.7, 0.3, 0.4],[0.65, 0.3, 0.4]],
               [[0.65,0.35,0.35],[0.65,0.35,0.45],[0.7,0.4,0.4],[0.65,0.35,0.45],
                [0.65,0.35,0.45],[0.7,0.3,0.45],[0.7,0.35,0.4],[0.7,0.4,0.4],
               [0.7, 0.3, 0.4],[0.65, 0.3, 0.4]],
                [[0.7,0.55,0.7],[0.7,0.4,0.6],[0.7,0.5,0.55],[0.7,0.55,0.6],
                [0.7,0.5,0.65],[0.7,0.5,0.65],[0.7,0.55,0.7],[0.7,0.5,0.65],
               [0.7, 0.35, 0.6],[0.7, 0.5, 0.7]],
                [[0.7,0.4,0.7],[0.7,0.7,0.7],[0.7,0.7,0.7],[0.7,0.25,0.7],
                [0.7,0.4,0.6],[0.7,0.6,0.7],[0.7,0.5,0.7],[0.7,0.7,0.7],
               [0.7, 0.45, 0.7],[0.7, 0.3, 0.6]],
             [[0.7,0.7,0.45],[0.7,0.7,0.6],[0.7,0.7,0.4],[0.7,0.7,0.55],
                [0.65,0.45,0.4],[0.7,0.35,0.5],[0.7,0.7,0.45],[0.7,0.6,0.6],
               [0.7, 0.7, 0.25],[0.7, 0.7, 0.25]]]

activations = [[['elu', 'swish', 'selu'], ['selu','swish','selu'], ['selu','swish','selu'],['selu','swish','elu'],
                ['selu','swish','elu'],['elu','swish','selu'],
               ['selu','swish','elu'],['selu','elu','selu'],['selu','swish','selu'],
               ['selu','swish','elu']],
               [['elu', 'swish', 'selu'], ['selu','swish','selu'], ['selu','swish','selu'],['selu','swish','elu'],
                ['selu','swish','elu'],['elu','swish','selu'],
               ['selu','swish','elu'],['selu','elu','selu'],['selu','swish','selu'],
               ['selu','swish','elu']],
                [['selu', 'relu', 'swish'], ['selu','relu','swish'], ['selu','relu','swish'],['selu','relu','swish'],
               ['selu','relu','swish'],['selu','relu','swish'],['selu','relu','swish'],['selu','relu','swish'],
               ['selu','elu','swish'],['selu','relu','swish']],
                [['selu', 'elu', 'swish'], ['elu','swish','relu'], ['elu','swish','selu'],['selu','elu','swish'],
               ['selu','elu','swish'],['selu','elu','swish'],['selu','elu','swish'],['selu','selu','swish'],
               ['selu','relu','swish'],['elu','elu','swish']],
            [['selu', 'swish', 'selu'], ['selu','swish','selu'], ['elu','swish','selu'],['selu','swish','selu'],
               ['selu','relu','relu'],['selu','relu','relu'],['selu','swish','elu'],['selu','swish','relu'],
               ['elu','swish','selu'],['elu','swish','swish']]]

lr = 5e-4

feats = np.arange(0,X.shape[1],1)
inp_size = int(np.ceil(1* len(feats)))
res = y.copy()
df_sample.loc[:, y.columns] = 0
res.loc[:, y.columns] = 0


# In[ ]:


# Defining callbacks

def callbacks():
    rlr = ReduceLROnPlateau(monitor = 'val_logloss', factor = 0.2, patience = 3, verbose = 0, 
                                min_delta = 1e-4, min_lr = 1e-6, mode = 'min')
        
    ckp = ModelCheckpoint("model.h5", monitor = 'val_logloss', verbose = 0, 
                              save_best_only = True, mode = 'min')
        
    es = EarlyStopping(monitor = 'val_logloss', min_delta = 1e-5, patience = 10, mode = 'min', 
                           baseline = None, restore_best_weights = True, verbose = 0)
    return rlr, ckp, es


# In[ ]:


def log_loss_metric(y_true, y_pred):
    metrics = []
    for _target in y.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels = [0,1]))
    return np.mean(metrics)


# <a id="32"></a>
# 
# ## Training and validation
# We use Multilabel Stratified KFold with 5 splits which is added in the beginning to the notebook.<br>

# In[ ]:


test_preds = []
res_preds = []
np.random.seed(seed=42)
n_split = 5
n_top = 10
n_round = 1

for seed in range(n_round):

    split_cols = np.random.choice(feats, inp_size, replace=False)
    res.loc[:, y.columns] = 0
    df_sample.loc[:, y.columns] = 0
    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits = n_split, random_state = seed, shuffle = True).split(X, y)):
        
        start_time = time()
        x_tr = X.astype('float64').values[tr][:, split_cols]
        x_val = X.astype('float64').values[te][:, split_cols]
        y0_tr, y0_val = y0.astype(float).values[tr], y0.astype(float).values[te]
        y_tr, y_val = y.astype(float).values[tr], y.astype(float).values[te]
        x_tt = X_test.astype('float64').values[:, split_cols]
        
        for num in range(n_top):
            model = create_model(inp_size, hid_layers[n][num], activations[n][num], dropout_rate[n][num], lr, y0.shape[1])
            model.fit(x_tr, y0_tr,validation_data=(x_val, y0_val), epochs = 150, batch_size = 128,
                      callbacks = callbacks(), verbose = 0)
            model.load_weights("model.h5")
            model2 = create_model(inp_size, hid_layers[n][num], activations[n][num], dropout_rate[n][num], lr, y.shape[1])
            for i in range(len(model2.layers)-1):
                model2.layers[i].set_weights(model.layers[i].get_weights())

            model2.fit(x_tr, y_tr,validation_data=(x_val, y_val),
                            epochs = 150, batch_size = 128,
                            callbacks = callbacks(), verbose = 0)
                       
            model2.load_weights('model.h5')
        
            df_sample.loc[:, y.columns] += model2.predict(x_tt, batch_size = 128)/(n_split*n_top)
        
            res.loc[te, y.columns] += model2.predict(x_val, batch_size = 128)/(n_top)
        
        oof = log_loss_metric(y.loc[te,y.columns], res.loc[te, y.columns])
        print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}], Seed {seed}, Fold {n}:', oof)

        K.clear_session()
        del model2
        x = gc.collect()

    df_sample.loc[ind_te, y.columns] = 0
    
    test_preds.append(df_sample.copy())
    
    res_preds.append(res.copy())


# <a id="33"></a>
# 
# ## Blending

# We blend the results of all models using averaging. In previous versions we used optimization suggested by this notebook <a href='https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0'>[6] </a>. It is also good to read this notebook to understand the neural split method. <a href='https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras'>[7] </a> <br>Blending may result in score improvement taking into the effect of all the models with slight differences. 

# In[ ]:


aa = [1.0]
res2= res.copy()
res2.loc[:, y.columns] = 0
for i in range(n_round):
    res2.loc[:, y.columns] += aa[i] * res_preds[i].loc[:, y.columns]
print(log_loss_metric(y, res2))


# In[ ]:


df_sample.loc[:, y.columns] = 0
for i in range(n_round):
    df_sample.loc[:, y.columns] += aa[i] * test_preds[i].loc[:, y.columns]
df_sample.loc[ind_te, y.columns] = 0


# In[ ]:


display(df_sample.head())
df_sample.to_csv('submission.csv', index=False)


# <a id="4"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center>Evaluation and Summary</center></h2>

# In this project, we first examined the data and performed some explanatory data analysis. We then trained a model using deep neural networks on the non-scored targets, transfered the weights and trained the model on scored targets. We then blended the results of different network architectures and initializations. Let's dive deep more into the data and see which label is contributing the most to the overall loss. 

# In[ ]:


y_true = y
y_preds = res2


# In[ ]:


losses = []
for i in range(y.shape[1]):
    losses.append(log_loss(y.iloc[:,i], res2.iloc[:,i]))


# In[ ]:


max_loss_ind= np.argmax(losses)
max_loss = np.max(losses)
print("Max loss is", max_loss,'For index', max_loss_ind,'which is',y.iloc[:,max_loss_ind].name)


# In[ ]:


y_max_loss = y.iloc[:,max_loss_ind]
y_max_loss.value_counts()

sns.countplot(y=y_max_loss,palette='nipy_spectral',orient='h')
plt.show()


# As we can see label 71 is contributing the most to the loss. As we saw earlier, this target was also the third top in having the most positive labels. Some may think of using imblearn library to address the imbalance problem. <a href="https://www.kaggle.com/sinamhd9/safe-driver-prediction-a-comprehensive-project">[8] </a> However, this may get complicated for a multilabel problem . <font size="4"> <b>Please leave a comment and share your ideas and let me know if this notebook was useful. Your upvote is appreciated! </b>

# <a id="5"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:red; border:0; color:white' role="tab" aria-controls="home"><center>References</center></h2>
# Almost all the things I shared was the ideas I learned from other kernels which are listed below. I would like to express my gratitude to the authors of these kernels who shared their work. Also, the outline for this notebook was inspired by this notebook <a href='https://www.kaggle.com/isaienkov/mechanisms-of-action-moa-prediction-eda'> [9]</a> <br>
#     
# <a href="https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing"> [1] MOA: Transfer Recipe with Smoothing</a> <br>
# <a href="https://www.kaggle.com/c/lish-moa/discussion/184005"> [2] Competition Insights </a> <br>
# <a href='https://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn?scriptVersionId=44558776' >[3] MoA | Pytorch | 0.01859 | RankGauss | PCA | NN </a> <br> 
# <a href="https://www.kaggle.com/rahulsd91/moa-label-smoothing">[4] MoA Label Smoothing  <a/> <br>
# <a href="https://www.kaggle.com/sinamhd9/hyperparameter-tuning-with-keras-tuner">[5] Hyperparameter tuning with Keras Tuner </a> <br>
# <a href='https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0'>[6] Model Blending Weights Optimisation </a> <br>
# <a href='https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras'>[7] Split Neural Network Approach (TF Keras) </a> <br>
# <a href='https://www.kaggle.com/sinamhd9/safe-driver-prediction-a-comprehensive-project'>[8] Safe driver prediction: A comprehensive project</a> <br>
# <a href='https://www.kaggle.com/isaienkov/mechanisms-of-action-moa-prediction-eda'> [9] Mechanisms of Action (MoA) Prediction. EDA </a>
# 

# 
# 

# In[ ]:





# In[ ]:




