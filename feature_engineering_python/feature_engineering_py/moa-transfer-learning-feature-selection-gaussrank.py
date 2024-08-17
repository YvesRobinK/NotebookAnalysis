#!/usr/bin/env python
# coding: utf-8

# # The Idea
#  The Basic idea behind this notebook is to train a model on the non_scored_targets and then train a model on the scored targets using these weights. The models are trained with features selected using `permutation importance` algorithm. `The implementation of this algorithm is included in this notebook in the form of a class.`<hr>
#  <a style="color:green">The feature engineering ideas are adopted from</a> [this](https://www.kaggle.com/kushal1506/moa-pytorch-feature-engineering-0-01846) <a style="color:green"> notebook by </a>[@kushal1506](https://www.kaggle.com/kushal1506)

# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Table of Contents</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#gauss-rank" role="tab" aria-controls="profile">Applying GaussRankScaler<span class="badge badge-primary badge-pill">1</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#model-data" role="tab" aria-controls="profile">Model Architecture and Data Preparation<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#feat_selection" role="tab" aria-controls="messages">Feature selection<span class="badge badge-primary badge-pill">3</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#feat-eng" role="tab" aria-controls="messages">Feature Engineering<span class="badge badge-primary badge-pill">4</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#transfer-model" role="tab" aria-controls="settings">Training Model on non_scored_targets for transfer<span class="badge badge-primary badge-pill">5</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#feature-model" role="tab" aria-controls="settings">Training Model with Transfered weights<span class="badge badge-primary badge-pill">6</span></a> 

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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt 
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf 
import tensorflow_addons as tfa
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import os
import sys

sys.path.append("../input/iterstrat/")
sys.path.append("../input/rank-gauss/")

from sklearn.metrics import log_loss


# In[3]:


get_ipython().system('pip install ../input/iterstrat/iterative_stratification-0.1.6-py3-none-any.whl')


# In[4]:


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[5]:


def seedAll(seed_value = 42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

seedAll(seed_value=42)


# In[6]:


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')


# In[7]:


GENES = [col for col in train_features if col.startswith('g-')]
CELLS = [col for col in train_features if col.startswith('c-')]
TARGETS = train_targets_scored.columns[1:]


# ## <a id="gauss-rank"> Applying RankGauss</a>

# In[8]:


for col in (GENES + CELLS):

    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# ## <a id="model-data"> Model Architecture and Data Preparation</a>

# In[9]:


def build(shape=None,out_cols=206):
    model = tf.keras.models.Sequential([
                L.InputLayer(input_shape=shape),
                L.BatchNormalization(),
                L.Dropout(0.3),
                tfa.layers.WeightNormalization(L.Dense(480,kernel_initializer="he_normal")),
                L.BatchNormalization(),
                L.Activation(tf.nn.leaky_relu),
                L.Dropout(0.4),
                tfa.layers.WeightNormalization(L.Dense(256,kernel_initializer="he_normal")),
                L.BatchNormalization(),
                L.Activation(tf.nn.leaky_relu),
                L.Dropout(0.2),
                tfa.layers.WeightNormalization(L.Dense(out_cols,activation="sigmoid",kernel_initializer="he_normal"))
            ])
    model.compile(loss="binary_crossentropy",optimizer = tfa.optimizers.AdamW(lr=0.001,weight_decay=1e-4),metrics = ["binary_crossentropy"])
    
    return model

def metric(y_true,y_predicted):

    metrics=[]
    for col in range(y_true.shape[1]):
        metrics.append(log_loss(y_true[:,col],y_predicted[:,col],labels=[0,1]))

    return np.mean(metrics)

def transfer_weight(model_source,model_dest):
    for i in range(len(model_source.layers[:-1])):
        model_dest.layers[i].set_weights(model_source.layers[i].get_weights())
    return model_dest


# In[10]:


def prepare_data(df,targets=train_targets_scored,test=False):
    df = df.drop('sig_id',axis=1)
    if(not test):
        targets = targets.iloc[df[df.cp_type=="trt_cp"].index,:]
        targets = targets.drop('sig_id',axis=1)
        df = df[df.cp_type=="trt_cp"]
        
    df = df.drop('cp_type',axis=1)
    df = pd.concat([pd.get_dummies(df["cp_time"],drop_first=True),df.drop('cp_time',axis=1)],axis=1)
    df["cp_dose"] = df["cp_dose"].map({'D1':0,'D2':1})
        
    if(not test):
        return df,targets
    return df


# In[11]:


train_data,train_targets = prepare_data(train_features,test=False)

test_data = prepare_data(test_features,test=True)


# In[12]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


# # <a id="feat_selection"> Feature Selection</a>
# In the following block of code, A model is defined and trained on the training set. After training a a feature selection algorithm, PermutationImportance  is run on the validation set using that model.<br><a style="color:green;"> PermutationImportance has been implemented from scratch in the class written here.</a><br>
# <br>
# <pre>
# 
# X_train,X_val,Y_train,Y_val = train_test_split(train_data,train_targets,test_size=0.2,random_state=101)
# 
# model = build((X_train.shape[1],))
# model.summary()
# 
# save_weight = ModelCheckpoint('model.learned.hdf5',save_best_only=True,save_weights_only=True,monitor = 'val_loss',mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
# early = EarlyStopping(monitor='val_loss',patience=5,mode='min')
# 
# model.fit(X_train,Y_train,
#          epochs=30,
#          validation_data = (X_val,Y_val),
#          batch_size=128,
#          callbacks = [early,reduce_lr_loss,save_weight])
# 
# model.load_weights('model.learned.hdf5')
# </pre>
# <pre>
# class PermutationImportance:
# 
#     def __init__(self,estimator,threshold):
# 
#         self.model = estimator
#         self.threshold = threshold
#         pd.options.mode.chained_assignment = None
#        
# 
#     def metric(self,y_true,y_predicted):
# 
#         metrics=[]
#         for col in range(y_true.shape[1]):
#             metrics.append(log_loss(y_true[:,col],y_predicted[:,col],labels=[0,1]))
# 
#         return np.mean(metrics)
# 
#         
# 
#     def fit(self,data_i,targets):
# 
#         data = data_i.copy()
#         l = len(data.columns)
# 
#         features = []
# 
#         base_predictions = self.model.predict(data.values)
#         base_loss = self.metric(targets.values,base_predictions)
# 
#         for idx,col in enumerate(data.columns):
#     
#             original = data.iloc[:,idx].copy()
#             suffle = np.copy(original.values)
#             np.random.shuffle(suffle)
#             
#             data.iloc[:,idx] = suffle            
# 
#             predictions = self.model.predict_proba(data.values)
# 
#             loss = self.metric(targets.values,predictions)
# 
#             if loss>base_loss+self.threshold:
# 
#                 features.append(col)
# 
# 
# 
#             data.iloc[:,idx] = original.values
#             
#             print(f'{(idx+1)*100/l}  %\r',end='',flush=True)
# 
#         return features
# 
# 
# pi = PermutationImportance(model,0.00000001)
# features = pi.fit(X_val,Y_val)
# </pre>

# In[13]:


features = [48, 72, 'cp_dose', 'g-0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-8', 'g-10', 'g-11', 'g-12', 'g-13', 'g-14', 'g-15', 'g-16', 'g-17', 'g-18', 'g-20', 'g-21', 'g-22', 'g-24', 'g-25', 'g-27', 'g-28', 'g-29', 'g-30', 'g-32', 'g-33', 'g-34', 'g-35', 'g-36', 'g-37', 'g-38', 'g-40', 'g-41', 'g-42', 'g-44', 'g-45', 'g-46', 'g-47', 'g-48', 'g-49', 'g-50', 'g-51', 'g-52', 'g-53', 'g-55', 'g-56', 'g-57', 'g-59', 'g-60', 'g-61', 'g-62', 'g-63', 'g-64', 'g-65', 'g-66', 'g-68', 'g-69', 'g-73', 'g-75', 'g-76', 'g-77', 'g-78', 'g-79', 'g-81', 'g-82', 'g-84', 'g-89', 'g-90', 'g-91', 'g-93', 'g-94', 'g-95', 'g-96', 'g-97', 'g-98', 'g-99', 'g-100', 'g-101', 'g-102', 'g-103', 'g-105', 'g-107', 'g-108', 'g-110', 'g-111', 'g-113', 'g-114', 'g-116', 'g-117', 'g-119', 'g-120', 'g-122', 'g-123', 'g-124', 'g-125', 'g-126', 'g-127', 'g-128', 'g-130', 'g-131', 'g-133', 'g-134', 'g-136', 'g-137', 'g-140', 'g-141', 'g-142', 'g-143', 'g-144', 'g-145', 'g-146', 'g-148', 'g-149', 'g-150', 'g-151', 'g-153', 'g-156', 'g-157', 'g-158', 'g-159', 'g-160', 'g-163', 'g-164', 'g-166', 'g-167', 'g-169', 'g-172', 'g-173', 'g-174', 'g-175', 'g-176', 'g-177', 'g-178', 'g-180', 'g-183', 'g-184', 'g-185', 'g-186', 'g-187', 'g-188', 'g-190', 'g-191', 'g-192', 'g-194', 'g-195', 'g-198', 'g-200', 'g-201', 'g-202', 'g-203', 'g-204', 'g-205', 'g-206', 'g-207', 'g-208', 'g-209', 'g-212', 'g-213', 'g-214', 'g-216', 'g-217', 'g-218', 'g-221', 'g-222', 'g-223', 'g-224', 'g-226', 'g-228', 'g-229', 'g-230', 'g-231', 'g-232', 'g-233', 'g-234', 'g-235', 'g-240', 'g-241', 'g-242', 'g-243', 'g-244', 'g-245', 'g-246', 'g-248', 'g-250', 'g-251', 'g-252', 'g-253', 'g-254', 'g-256', 'g-258', 'g-260', 'g-262', 'g-263', 'g-264', 'g-265', 'g-267', 'g-268', 'g-269', 'g-271', 'g-272', 'g-273', 'g-274', 'g-275', 'g-277', 'g-278', 'g-279', 'g-280', 'g-281', 'g-282', 'g-283', 'g-284', 'g-287', 'g-288', 'g-289', 'g-291', 'g-292', 'g-293', 'g-294', 'g-296', 'g-297', 'g-298', 'g-299', 'g-300', 'g-301', 'g-302', 'g-303', 'g-304', 'g-305', 'g-306', 'g-309', 'g-311', 'g-312', 'g-313', 'g-314', 'g-315', 'g-317', 'g-318', 'g-321', 'g-322', 'g-323', 'g-324', 'g-325', 'g-329', 'g-330', 'g-331', 'g-332', 'g-333', 'g-334', 'g-336', 'g-337', 'g-340', 'g-341', 'g-342', 'g-343', 'g-344', 'g-345', 'g-347', 'g-348', 'g-350', 'g-351', 'g-352', 'g-353', 'g-354', 'g-355', 'g-356', 'g-357', 'g-358', 'g-359', 'g-360', 'g-362', 'g-363', 'g-365', 'g-369', 'g-370', 'g-371', 'g-372', 'g-373', 'g-375', 'g-376', 'g-377', 'g-378', 'g-379', 'g-380', 'g-381', 'g-382', 'g-383', 'g-384', 'g-386', 'g-387', 'g-389', 'g-390', 'g-391', 'g-392', 'g-394', 'g-395', 'g-396', 'g-397', 'g-398', 'g-399', 'g-400', 'g-401', 'g-402', 'g-403', 'g-404', 'g-405', 'g-406', 'g-407', 'g-408', 'g-409', 'g-410', 'g-411', 'g-414', 'g-415', 'g-418', 'g-421', 'g-422', 'g-423', 'g-424', 'g-425', 'g-427', 'g-429', 'g-430', 'g-431', 'g-432', 'g-433', 'g-434', 'g-435', 'g-436', 'g-437', 'g-438', 'g-439', 'g-440', 'g-441', 'g-442', 'g-443', 'g-444', 'g-445', 'g-446', 'g-447', 'g-449', 'g-450', 'g-451', 'g-453', 'g-454', 'g-455', 'g-456', 'g-457', 'g-458', 'g-459', 'g-460', 'g-461', 'g-462', 'g-463', 'g-464', 'g-465', 'g-466', 'g-468', 'g-469', 'g-471', 'g-472', 'g-473', 'g-476', 'g-478', 'g-479', 'g-480', 'g-481', 'g-482', 'g-483', 'g-485', 'g-486', 'g-487', 'g-489', 'g-490', 'g-491', 'g-492', 'g-493', 'g-494', 'g-495', 'g-497', 'g-499', 'g-500', 'g-503', 'g-505', 'g-506', 'g-507', 'g-509', 'g-511', 'g-512', 'g-513', 'g-515', 'g-516', 'g-518', 'g-519', 'g-520', 'g-521', 'g-522', 'g-524', 'g-525', 'g-526', 'g-527', 'g-528', 'g-530', 'g-533', 'g-534', 'g-536', 'g-537', 'g-538', 'g-539', 'g-540', 'g-541', 'g-543', 'g-544', 'g-547', 'g-550', 'g-551', 'g-552', 'g-553', 'g-554', 'g-555', 'g-556', 'g-557', 'g-558', 'g-559', 'g-560', 'g-561', 'g-562', 'g-563', 'g-564', 'g-565', 'g-566', 'g-567', 'g-569', 'g-570', 'g-571', 'g-572', 'g-573', 'g-575', 'g-578', 'g-581', 'g-583', 'g-584', 'g-585', 'g-586', 'g-587', 'g-588', 'g-589', 'g-590', 'g-594', 'g-596', 'g-597', 'g-598', 'g-599', 'g-600', 'g-601', 'g-602', 'g-603', 'g-604', 'g-605', 'g-606', 'g-608', 'g-610', 'g-611', 'g-612', 'g-613', 'g-614', 'g-615', 'g-616', 'g-617', 'g-618', 'g-619', 'g-620', 'g-621', 'g-623', 'g-624', 'g-625', 'g-626', 'g-627', 'g-628', 'g-630', 'g-631', 'g-633', 'g-634', 'g-635', 'g-636', 'g-638', 'g-639', 'g-640', 'g-643', 'g-645', 'g-648', 'g-649', 'g-651', 'g-655', 'g-656', 'g-657', 'g-658', 'g-659', 'g-661', 'g-663', 'g-664', 'g-665', 'g-666', 'g-667', 'g-668', 'g-669', 'g-670', 'g-671', 'g-672', 'g-674', 'g-676', 'g-677', 'g-678', 'g-679', 'g-680', 'g-681', 'g-682', 'g-683', 'g-688', 'g-689', 'g-690', 'g-691', 'g-692', 'g-694', 'g-696', 'g-697', 'g-698', 'g-699', 'g-701', 'g-703', 'g-704', 'g-705', 'g-707', 'g-709', 'g-710', 'g-711', 'g-712', 'g-713', 'g-714', 'g-715', 'g-716', 'g-718', 'g-719', 'g-720', 'g-721', 'g-722', 'g-723', 'g-724', 'g-725', 'g-726', 'g-727', 'g-729', 'g-730', 'g-731', 'g-732', 'g-733', 'g-734', 'g-735', 'g-736', 'g-737', 'g-741', 'g-742', 'g-745', 'g-747', 'g-752', 'g-753', 'g-754', 'g-755', 'g-757', 'g-758', 'g-759', 'g-761', 'g-763', 'g-764', 'g-765', 'g-766', 'g-767', 'g-768', 'g-769', 'g-770', 'g-771', 'c-1', 'c-3', 'c-4', 'c-7', 'c-8', 'c-9', 'c-12', 'c-13', 'c-14', 'c-15', 'c-16', 'c-17', 'c-18', 'c-21', 'c-22', 'c-23', 'c-24', 'c-25', 'c-26', 'c-27', 'c-29', 'c-30', 'c-31', 'c-34', 'c-35', 'c-37', 'c-38', 'c-39', 'c-40', 'c-41', 'c-42', 'c-44', 'c-46', 'c-47', 'c-48', 'c-49', 'c-50', 'c-51', 'c-54', 'c-56', 'c-57', 'c-58', 'c-59', 'c-61', 'c-62', 'c-64', 'c-65', 'c-66', 'c-67', 'c-69', 'c-73', 'c-76', 'c-77', 'c-79', 'c-81', 'c-82', 'c-86', 'c-87', 'c-89', 'c-91', 'c-93', 'c-94', 'c-95', 'c-98', 'c-99']


# In[14]:


train = train_data.loc[:,features]
test = test_data.loc[:,features]


# # <a id="feat-eng">Feature Engineering</a>
# ## Ideas
# <ul>
# <li>The gene features and cell features in the original set( without selection) are clustered in groups of 35 and 5 respectively. The cluster label is taken as a new feature</li>
# <li> Statistical information about the gene and cell features are included as new feautures. This includes:<hr>
#     <ol>
#         <li>sum of the gene features, sum of the cell features, and sum of gene and cell features combined.</li>
#         <li>mean of the gene features, mean of the cell features, and mean of gene and cell features combined.</li>
#         <li>std of the gene features, std of the cell features, and std of gene and cell features combined.</li>
#         <li>skewness of the gene features, skewness of the cell features, and skewness of gene and cell features combined.</li>
#         <li>kurtosis of the gene features, kurtosis of the cell features, and kurtosis of gene and cell features combined.</li>
#     </ol>
# </li>
# </ul>

# In[15]:


from sklearn.cluster import KMeans
def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 123):
    
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

a ,b=fe_cluster(train_features,test_features)
cols = [col for col in a.columns if col.startswith('clusters')]

train = pd.concat([train,a.loc[train_features[train_features.cp_type=="trt_cp"].index,cols]],axis=1)
test = pd.concat([test,b.loc[:,cols]],axis=1)


# In[16]:


def fe_stats(train, test):
    train_ = train.copy()
    test_ = test.copy()
    features_g = list(train_features.columns[4:776])
    features_c = list(train_features.columns[776:876])
    
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
        
    return train.iloc[:,-15:], test.iloc[:,-15:]

a,b = fe_stats(train_features,test_features)


# In[17]:


train = pd.concat((train,a.loc[train_features[train_features.cp_type=="trt_cp"].index,:]),axis=1)
test = pd.concat((test,b),axis=1)


# # <a>PCA</a>

# ## Genes

# In[18]:


n_comps = 600
pca = PCA(n_components = n_comps)

total_data = train_features.append(test_features)
total_data = total_data.loc[:,GENES]

pca.fit(total_data)
train2 = pca.transform(train_features.loc[train_features[train_features.cp_type=="trt_cp"].index,GENES])
test2 = pca.transform(test_features.loc[:,GENES])

train2 = pd.DataFrame(train2,columns=[f'pca-g_{i}' for i in range(train2.shape[1])])
test2 = pd.DataFrame(test2,columns=[f'pca-g_{i}' for i in range(test2.shape[1])])

total = train2.append(test2)
vt = VarianceThreshold(0.8)
vt.fit(total)

train2 = vt.transform(train2)
test2 = vt.transform(test2)

train2 = pd.DataFrame(train2,columns=[f'pca-g_{i}' for i in range(train2.shape[1])])
test2 = pd.DataFrame(test2,columns=[f'pca-g_{i}' for i in range(test2.shape[1])])

train = pd.concat((train.reset_index(drop=True),train2),axis=1)
test = pd.concat((test,test2),axis=1)


# In[19]:


n_comps = 50
pca = PCA(n_components = n_comps)

total_data = train_features.append(test_features)
total_data = total_data.loc[:,CELLS]

pca.fit(total_data)
train2 = pca.transform(train_features.loc[train_features[train_features.cp_type=="trt_cp"].index,CELLS])
test2 = pca.transform(test_features.loc[:,CELLS])

train2 = pd.DataFrame(train2,columns=[f'pca-c_{i}' for i in range(train2.shape[1])])
test2 = pd.DataFrame(test2,columns=[f'pca-c_{i}' for i in range(test2.shape[1])])

total = train2.append(test2)
vt = VarianceThreshold(0.8)
vt.fit(total)

train2 = vt.transform(train2)
test2 = vt.transform(test2)

train2 = pd.DataFrame(train2,columns=[f'pca-c_{i}' for i in range(train2.shape[1])])
test2 = pd.DataFrame(test2,columns=[f'pca-c_{i}' for i in range(test2.shape[1])])

train = pd.concat((train.reset_index(drop=True),train2),axis=1)
test = pd.concat((test,test2),axis=1)


# # <a id="transfer_model"> Training Model on non_scored targets for Transfer</a>

# In[20]:


Y_transfer = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
Y_transfer = Y_transfer.loc[train_features[train_features.cp_type=="trt_cp"].index,:].drop('sig_id',axis=1)


# In[21]:


X_train, X_val,Y_train,Y_val = train_test_split(train,Y_transfer,test_size=0.2,random_state=101)


# In[22]:


model = build((X_train.shape[1],),402)


# In[23]:


save_weight = ModelCheckpoint('model.learned.hdf5',save_best_only=True,save_weights_only=False,monitor = 'val_loss',mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
early = EarlyStopping(monitor='val_loss',patience=5,mode='min')

model.fit(X_train,Y_train,
         epochs=50,
         batch_size=128,
         validation_data = (X_val,Y_val),
         callbacks=[save_weight,reduce_lr_loss,early])

model.load_weights('model.learned.hdf5')


# In[24]:


ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
ss.loc[:,'sig_id'] = test_features['sig_id'].values
ss.iloc[:,1:]=0


# # <a id="feature-model"> Training Model with Transfered Weights</a>

# In[25]:


mskf = MultilabelStratifiedKFold(n_splits = 7,shuffle=True)
seeds = [42,58,132,456,789]
histories = []
scores = []

for seed in seeds:
    seedAll(seed_value=seed)
    print(f"Training seed {seed}")
    print('='*50)
    
    for idx,(tr_,val_) in enumerate(mskf.split(train,train_targets)):
        print(f'\nFold {idx}')
        print('-'*50)
        
        K.clear_session()
        X_train,X_val,Y_train,Y_val = train.iloc[tr_,:],train.iloc[val_,:],train_targets.iloc[tr_,:],train_targets.iloc[val_,:]
        
        path = f'model.{seed}_{idx}.hdf5'
        save_weight = ModelCheckpoint(path,save_best_only=True,save_weights_only=False,monitor = 'val_loss',mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
        early = EarlyStopping(monitor='val_loss',patience=5,mode='min')
        
        model_fin = build(shape=(X_train.shape[1],))
        model_fin = transfer_weight(model,model_fin)
        
        for layer in model_fin.layers:
            layer.trainable=True
            
        history = model_fin.fit(X_train.values,Y_train,
                 batch_size=128,
                 epochs=50,
                 validation_data=(X_val,Y_val),
                 callbacks=[early,save_weight,reduce_lr_loss]
                 )
        histories.append(history)
        model_fin= tf.keras.models.load_model(path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        
        val_pred = model_fin.predict(X_val)
        score = metric(Y_val.values,val_pred)
        scores.append(score)
        
        print(f"Validation Score: {score}")
        pred = model_fin.predict(test.values)
        
        ss.loc[:,train_targets.columns]+= pred


# In[26]:


ss.loc[:,train_targets.columns] /= 7*len(seeds)
ss.loc[test_features[test_features.cp_type=='ctl_vehicle'].index,train_targets.columns] = 0


# In[27]:


print(f'validation score : {np.mean(scores)}')
plt.figure()

for history in histories:
    plt.plot(history.history["val_loss"],color='red')
    plt.plot(history.history["loss"],color="green")
plt.title("Training Curve")


# In[28]:


ss.to_csv('./submission.csv',index=False)

