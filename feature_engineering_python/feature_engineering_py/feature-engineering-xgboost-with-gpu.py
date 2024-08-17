#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)


# Standard plotly imports
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff
import os


import warnings
warnings.filterwarnings("ignore")


# ## Create Environment

# In[2]:


import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set


# In[3]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


# In[4]:


print("XGBoost version:", xgb.__version__)


# In[5]:


# print('# File sizes')
# total_size = 0
# start_path = '../input/jane-street-market-prediction'  # To get size of current directory
# for path, dirs, files in os.walk(start_path):
#     for f in files:
#         fp = os.path.join(path, f)
#         total_size += os.path.getsize(fp)
# print("Directory size: " + str(round(total_size/ 1000000, 2)) + 'MB')


# In[6]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'/kaggle/input/jane-street-market-prediction/train.csv\')\nfeatures = pd.read_csv(\'../input/jane-street-market-prediction/features.csv\')\nexample_test = pd.read_csv(\'../input/jane-street-market-prediction/example_test.csv\')\nsample_prediction_df = pd.read_csv(\'../input/jane-street-market-prediction/example_sample_submission.csv\')\nprint ("Data is loaded!")\n')


# In[7]:


print('train shape is {}'.format(train.shape))
print('features shape is {}'.format(features.shape))
print('example_test shape is {}'.format(example_test.shape))
print('sample_prediction_df shape is {}'.format(sample_prediction_df.shape))


# In[8]:


train.head()


# ### Missing Values Count

# In[9]:


missing_values_count = train.isnull().sum()
print (missing_values_count)
total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


# # Is the data balanced or not?

# In[10]:


# I have taked this cell from https://www.kaggle.com/jazivxt/the-market-is-reactive
# And https://www.kaggle.com/drcapa/jane-street-market-prediction-starter-xgb

train = train[train['weight'] != 0]

train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')




# In[11]:


nulls = train.isnull().sum()
nulls_list = list(nulls[(nulls >239049)].index)
nulls_list


# In[12]:


train[nulls_list].corr().style.background_gradient(cmap='viridis')


# In[13]:


import gc
gc.collect()


# In[14]:


train.drop(columns=nulls_list,inplace=True)


# In[15]:


train.fillna(train.mean(axis=0),inplace=True)


# In[16]:


gc.collect()


# In[17]:


corr = train.iloc[: ,7:-2].corr()


# In[18]:


corr.style.background_gradient('coolwarm')


# In[19]:


gc.collect()


# In[20]:


featstr = [i for i in train.columns[7:-2]]


# In[21]:


for i in featstr[1:]:
    print('{}\n0.1%:99.9% are between: {}\nmax: {}\nmin: {}\n75% are under: {}'.format(i,
        np.percentile(train[i],(.1,99.9)), 
            train[i].max(),
                train[i].min(),
                    np.percentile(train[i],75)),
                        '\n===============================')


# In[22]:


gc.collect()


# In[23]:


# To avoid removing more data while looping through the data set we will 
# make a list of 99.9% mark for each and every single feature
# We will also create a list for negative outliers values "using .1 % mark" to be explored later¶
n999 = [ np.percentile(train[i],99.9) for i in featstr[1:]]
n001 = [ np.percentile(train[i],.1) for i in featstr[1:]]


# In[24]:


for i, j in enumerate(featstr[1:]):
    train = train[train[j] < n999[i]]
    gc.collect()


# In[25]:


gc.collect()


# In[26]:


for i,j in zip(featstr[1:][2:34],n001[2:34]):
    train = train[train[i] > j]
    gc.collect();


# In[27]:


tr_c = train.copy()


# In[28]:


gc.collect()


# In[29]:


import os, gc
import cudf
import numpy as np
import cupy as cp
import pandas as pd
import janestreet
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# In[30]:


gc.collect()


# In[31]:


train['action'] = (train['resp'] > 0).astype('int')


# In[32]:


gc.collect()


# In[33]:


train = train.query('weight > 0').reset_index(drop = True)


# In[34]:


gc.collect()


# In[35]:


gc.collect()


# In[ ]:





# In[36]:


X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']


# In[37]:


x = train['action'].value_counts().index
y = train['action'].value_counts().values

trace2 = go.Bar(
     x=x ,
     y=y,
     marker=dict(
         color=y,
         colorscale = 'Viridis',
         reversescale = True
     ),
     name="Imbalance",    
 )
layout = dict(
     title="Data imbalance - action",
     #width = 900, height = 500,
     xaxis=go.layout.XAxis(
     automargin=True),
     yaxis=dict(
         showgrid=False,
         showline=False,
         showticklabels=True,
 #         domain=[0, 0.85],
     ), 
)
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)


# In[38]:


features2 = [col for col in list(train.columns) if 'feature' in col]


# In[39]:


del x, y, train, tr_c


# ## Training
# ##### To activate GPU usage, simply use tree_method='gpu_hist' (took me an hour to figure out, I wish XGBoost documentation was clearer about that).

# In[40]:


clf = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=11,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=-999,
    random_state=2020,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)


# In[41]:


gc.collect()


# In[42]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[ ]:





# In[43]:


# for (test_df, sample_prediction_df) in iter_test:
#     X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
#     #X_test = feature_sign(X_test
#     X_test = X_test.fillna(-999)
#     y_preds = clf.predict(X_test)
#     sample_prediction_df.action = y_preds
#     env.predict(sample_prediction_df)


# In[44]:


print('Creating submissions file...', end='')
rcount = 0
for (test_df, prediction_df) in env.iter_test():
    X_test = test_df.loc[:, featstr]
    y_preds = clf.predict(X_test)
    prediction_df.action = y_preds
    env.predict(prediction_df)
    rcount += len(test_df.index)
print(f'Finished processing {rcount} rows.')


# In[ ]:




