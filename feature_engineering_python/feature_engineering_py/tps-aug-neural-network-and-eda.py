#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(2)
from sklearn.model_selection import GroupKFold
from tensorflow import keras
from tensorflow.keras import layers, callbacks
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')
import sys; sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# # Importing Data

# In[2]:


df=pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test=pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')
sub=pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')


# # Checking Missing values
# 1. The `train data` consists `20273` missing values while `test data` has `15709` missing values.
# 2. Feature **measurement_17** has the highest number of missing values.
# 3. Overall both train and test dataset has same ratio of missing values as can be observed in plot below.

# In[3]:


train=df.copy()
ncounts = pd.DataFrame([train.isna().sum(), test.isna().sum()]).T
ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

ncounts.query("train_missing > 0").plot(
    kind="barh", figsize=(20, 8), title="# Missing Values"
)
plt.show()


# # Is there any relationship between missing data🤔🤔?
# > I have plotted this `heatmap` to identify whether value missing in some feature is realted to other or not.

# In[4]:


plt.figure(figsize=(25,10))
sns.heatmap(df[df.columns[df.isna().any()]].isna().transpose(),
            cmap="copper")
plt.title('Heatmap to check relation between Missing Values', weight = 'bold', size = 20, color = 'brown')
plt.xticks(size = 12, color = 'maroon')
plt.yticks(size = 12, color = 'maroon')
plt.show();


# # Feature Distributions Of Training Data
# > Features `product_code` and  all 3 `attribute_[0,..,3]` features  are categorical.
# 

# In[5]:


plt.style.use("fivethirtyeight")
useful_cols=[col for col in df.columns if col not in ["id","failure"]]
cols_dist = [col for col in useful_cols if df[col].dtypes not in ['object']]
color_ = [ '#9D2417', '#AF41B4', '#003389' ,'#3C5F41',  '#967032', '#2734DE'] 
cmap_ = ['mako', 'rainbow', 'crest']

plt.figure(figsize= (20,22))
for i,col in enumerate(train[useful_cols].columns):
    rand_col = color_[random.sample(range(6), 1)[0]]
    plt.subplot(8,3, i+1)
    if col in cols_dist:
        
        sns.histplot(data=train,x=train[col],hue=train['failure'], color = rand_col, fill = rand_col)
        plt.title(col, color = 'black')
        plt.ylabel(" ")
        plt.xlabel(" ")
        plt.tight_layout()
    else:
        sns.countplot(data = train , x = col, hue=train['failure'], palette = cmap_[random.sample(range(3), 1)[0]] )
        plt.title(col, color = 'black')
        plt.ylabel(" ")
        plt.xlabel(" ")
        plt.legend(loc='upper right', borderaxespad=0)
        plt.tight_layout()


# # Loading Feature
# > The loading feature seems to have `right skewed distribution`.
# 
# > Applying `log transformation` seems to make the distribution more normal.

# In[6]:


plt.figure(figsize= (10,5))
plt.style.use('ggplot')
plt.subplot(1,2,1)
sns.histplot(train['loading'],kde=True,color='coral')
plt.title("Orignal loading")
plt.subplot(1,2,2)
sns.histplot(np.log(train["loading"]),kde=True)
plt.title("Log transformed loading")


# # Product Code
# 
# > The data seems to be pivoted/revolving around the `product_code` feature.
# 
# > We can observe from the plot below that the data belonging to product_code `"C"` seems to have more number of missing values for most of the features.
# 
# > For imputing missing values in data we can apply an imputer by `grouping` data on **product code**.

# In[7]:


pc=df.product_code.unique()
comp=pd.DataFrame([df[df.product_code==i].isna().sum() for i in pc]).T
comp.columns=pc
comp.query('A>0').plot(
    kind="barh", figsize=(25, 18), title="Missing Values by product Code"
)
plt.show()


# In[8]:


y=df.failure
X=df.drop('failure',axis=1)


# # Imputing missing values by LGBM imputer
# * I will impute missing values for each product of `product_code` from both *train* and *test* data.

# In[9]:


null_cols=[col for col in train.columns if train[col].isnull().sum()!=0]
non_null=[col for col in X.columns if col not in null_cols]
cat=[col for col in X.columns if X[col].dtypes == 'object']
prd_code=df.product_code.unique() # unique product_code of train data
tst_prd_code=test.product_code.unique() # unique product_code of test data


# In[10]:


lgbm_imtr = LGBMImputer(cat_features=cat, n_iter=50) 
tr=pd.DataFrame()
for pc in prd_code: #looping through product_code
    tr=pd.concat([tr,lgbm_imtr.fit_transform(X[X['product_code']==pc][null_cols])],axis=0) #imputing values for each product_code and appending them to dataframe
tr=pd.concat([tr,X[non_null]],axis=1) 


# In[11]:


ts=pd.DataFrame()
for pc in tst_prd_code:
    ts=pd.concat([ts,lgbm_imtr.fit_transform(test[test['product_code']==pc][null_cols])],axis=0)#imputing values for each product_code and appending them to dataframe
ts=pd.concat([ts,test[non_null]],axis=1)  


# In[12]:


tr=tr.drop('id',axis=1)
ts=ts.drop('id',axis=1)


# # Feature Engineering
# ambrosm suggested in his [notebook](https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense) that feature `measurement_3` has lower failure rate than average failure rate and `measurement_5` has higher failure rate than average failure rate so it can be inferred that availability or unavailability of these features is directly correlated with the failure so we can add availablity of these two features as new features.
# 
# ![](https://www.kaggleusercontent.com/kf/102683452/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..YiD1OMlMokzKqNIiRbiKUw.pchHILuMqwUB9y397qLCzp6n1_Du8-5RruMPF7G1uT69g9yEWQQr0xxIwpBBTZbNg8RFL6qDvEFQmgNJusXCJW8MmYryJSpVyuOVTlKLcGot-RM6AhC2wOLFOljmgK13QtEFAFbryvD70-KPSdn_zEof2A_UZDN8uRiupA1YlOIWlYvAofh4lZz8pT-OTszTCEczK6aaDuOZK0XN2loMjmmyWbsYZ8dBlud-VRP1Njlap99yp0KS4yyJTdDQCt91JBpOAoVUUyQy_JKl6_fLY20DeNtiAeSbm8Ho2dLFxqOJUMHL-tROz8x5-DrTPZmwrkHyZU7mK3pBf_s11uwINLkNKZLzDAu4S1TR8Jx4VdsGD2-LVU4rcuMKzc7SFSQBYO8i4uZ-IQrE9D2mbUPjgm6NYnvYssjUOYsoJkTkVzwnQosanQQWRyDar_bnyNFaVxtSR-SOa5wHGctouzSxEcAHrH2Lpm-EbB_6SoTMaTB1QtvajLzOS7TZrjxBDGTscuni5r7p4TQfAd3lf9VVKNqSIyXM8ztw8B95_KWyFWcDBP9QKkLL3pX5sVJtnhSc63pXmthRRX_Oile46FMp9cJCS17PqN0TvEfdNgxiO7KYGedI7ifl2_ZzkXkFgxHPKxZ9CZYkwntPvE20xClo-ohq60csdYJ7O6jbWkP20cU.h6djV_PvgU3BXVz6sM47HQ/__results___files/__results___14_1.png)

# In[13]:


tr['m_3_missing']=X.measurement_3.isna().astype(int)
tr['m_5_missing']=X.measurement_5.isna().astype(int)

ts['m_3_missing']=test.measurement_3.isna().astype(int)
ts['m_5_missing']=test.measurement_5.isna().astype(int)


# # Label Encoding categorical features
# > All `attribute` features needs to be label encoded.

# In[14]:


cat_pr=[f'attribute_{i}' for i in range(4)]
for column in cat_pr:
    label_encoder = LabelEncoder()
    label_encoder.fit(tr[column].append(ts[column]))
    tr[column] = label_encoder.transform(tr[column])
    ts[column] = label_encoder.transform(ts[column])


# In[15]:


cols_to_use=[col for col in tr.columns if col!='product_code']


# # Scaling Data
# > Scaling entire data by using sklearn's `Standard Scaler`

# In[16]:


scaler = StandardScaler()
training=scaler.fit_transform(tr[cols_to_use])
testing=scaler.fit_transform(ts[cols_to_use])


# In[17]:


X_train, X_test, y_train, y_test=train_test_split(training,y,random_state=28,shuffle=True,test_size=0.2)


# # Neural Network Baseline Model

# In[18]:


tf.random.set_seed(42)
model_14 = tf.keras.Sequential([
  tf.keras.layers.Dense(1024, activation="relu"),
  tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid") 
])
early_stopping = callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3 ,patience=4)
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/10))

model_14.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(lr=0.0007),
                 metrics=["AUC"])

history = model_14.fit(X_train,
                       y_train,
                       epochs=200,
                       callbacks=[early_stopping],
                       validation_data=(X_test, y_test))


# In[19]:


pd.DataFrame(history.history).plot()


# In[20]:


preds=model_14.predict(testing)
preds=preds.reshape(20775,)


# In[21]:


submission = pd.DataFrame({'id': sub.id,'failure': preds})
submission.to_csv('submission.csv', index=False)
submission

