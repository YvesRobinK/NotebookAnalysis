#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


get_ipython().system('pip install pyspark')


# In[3]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.sql import Window, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, StringType, ArrayType, LongType, FloatType, DateType
from sklearn.model_selection import train_test_split
import sys
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import lightgbm as lgb


# In[4]:


spark = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "10G")
                  .config("spark.driver.memory","5G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","4G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .config("spark.kryoserializer.buffer.max.mb", "2047").getOrCreate()
        )


# In[5]:


df = spark.read.csv('/kaggle/input/liverpool-ion-switching/train.csv', header=True)


# In[6]:


df.show()


# Let us describe the dataset.

# In[7]:


description = df.describe().collect()
description


# In[8]:


signal_stddev = float(description[2]['signal'])
open_channels_stddev = float(description[2]['open_channels'])

signal_mean = float(description[1]['signal'])
open_channels_mean = float(description[1]['open_channels'])


# In[9]:


df1 = df.filter(col('time') <= 50)
df2 = df.filter(col('time') > 50)


# In[10]:


last_2 = Window().orderBy('time').rowsBetween(-2, -1)
last_4 = Window().orderBy('time').rowsBetween(-4, -1)
last_8 = Window().orderBy('time').rowsBetween(-8, -1)
last_16 = Window().orderBy('time').rowsBetween(-16, -1)
last_32 = Window().orderBy('time').rowsBetween(-32, -1)

lead_2 = Window().orderBy('time').rowsBetween(1, 2)
lead_4 = Window().orderBy('time').rowsBetween(1, 4)
lead_8 = Window().orderBy('time').rowsBetween(1, 8)
lead_16 = Window().orderBy('time').rowsBetween(1, 16)
lead_32 = Window().orderBy('time').rowsBetween(1, 32)


# In[11]:


df1 = df1.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))
df1 = df1.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))
df1 = df1.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))
df1 = df1.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))
df1 = df1.withColumn('last_32_sig_mean', F.mean('signal').over(last_32))
df1 = df1.withColumn('last_64_sig_mean', F.mean('signal').over(last_64))
df1 = df1.withColumn('last_128_sig_mean', F.mean('signal').over(last_128))

df1 = df1.withColumn('lead_2_sig_mean', F.mean('signal').over(lead_2))
df1 = df1.withColumn('lead_4_sig_mean', F.mean('signal').over(lead_4))
df1 = df1.withColumn('lead_8_sig_mean', F.mean('signal').over(lead_8))
df1 = df1.withColumn('lead_16_sig_mean', F.mean('signal').over(lead_16))
df1 = df1.withColumn('lead_32_sig_mean', F.mean('signal').over(lead_32))
df1 = df1.withColumn('lead_64_sig_mean', F.mean('signal').over(lead_64))
df1 = df1.withColumn('lead_128_sig_mean', F.mean('signal').over(lead_128))

df2 = df2.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))
df2 = df2.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))
df2 = df2.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))
df2 = df2.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))
df2 = df2.withColumn('last_32_sig_mean', F.mean('signal').over(last_32))
df2 = df2.withColumn('last_64_sig_mean', F.mean('signal').over(last_64))
df2 = df2.withColumn('last_128_sig_mean', F.mean('signal').over(last_128))

df2 = df2.withColumn('lead_2_sig_mean', F.mean('signal').over(lead_2))
df2 = df2.withColumn('lead_4_sig_mean', F.mean('signal').over(lead_4))
df2 = df2.withColumn('lead_8_sig_mean', F.mean('signal').over(lead_8))
df2 = df2.withColumn('lead_16_sig_mean', F.mean('signal').over(lead_16))
df2 = df2.withColumn('lead_32_sig_mean', F.mean('signal').over(lead_32))
df2 = df2.withColumn('lead_64_sig_mean', F.mean('signal').over(lead_64))
df2 = df2.withColumn('lead_128_sig_mean', F.mean('signal').over(lead_128))




# In[12]:


df1 = df1.toPandas()


# In[13]:


df2 = df2.toPandas()


# In[14]:


df = pd.concat([df1, df2], sort=False)


# In[15]:


df['signal'] = df['signal'].astype(np.float64)


# In[16]:


df = df.fillna(0)


# In[17]:


df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)


# In[18]:


clf = xgb.XGBClassifier(
    n_estimators=5,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2000
)


# In[19]:


all_but_open_channels = list(df.columns)
all_but_open_channels.remove('open_channels')
all_but_open_channels.remove('time')


# In[20]:


get_ipython().run_line_magic('time', "clf.fit(df_train[all_but_open_channels], df_train['open_channels'])")


# In[21]:


probs = clf.predict_proba(df_test[all_but_open_channels])


# In[22]:


df_test['open_channels']


# In[23]:


y_test = label_binarize(df_test['open_channels'], classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
y_score = probs

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(11):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

auc = roc_auc['micro']

print('AUC performance', auc)


# In[24]:


auc


# In[25]:


y_score.shape


# In[26]:


df = spark.read.csv('/kaggle/input/liverpool-ion-switching/test.csv', header=True)


# In[27]:


df = df.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))
df = df.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))
df = df.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))
df = df.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))
df = df.withColumn('history_mean', F.mean('signal').over(history))


# In[28]:


df = df.toPandas()


# In[29]:


df['signal'] = df['signal'].astype(np.float64)


# In[30]:


df = df.fillna(0)


# In[31]:


preds = clf.predict(df[all_but_open_channels])


# In[32]:


submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[33]:


submission


# In[34]:


submission['open_channels'] = preds


# In[35]:


submission['time'] = submission['time'].map(lambda x: '%.4f' % x)


# In[36]:


submission.to_csv('submission.csv', index=False)


# In[37]:


submission[submission['time'] == 500.0010]


# In[ ]:




