#!/usr/bin/env python
# coding: utf-8

# # Feature engineering
# One soon realize that in these kind of competitions where many different models are trained on the same data and featureset and compared against each other that is is very usefult to keep feature engineering in a separate notebook.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt
from pickle import dump


# Feature engineering in this notebook is modified and copied and from [Improvement base on Tensor Bidirect LSTM](https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173/notebook) by [Ken Sit](https://www.kaggle.com/kensit). Which is further improved by [Chris Deotte](https://www.kaggle.com/cdeotte) in [Ensemble Folds with MEDIAN - [0.153]](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153). It is saved to a python file for use in other notebooks.

# In[2]:


train_ori = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')


# In[3]:


get_ipython().run_cell_magic('writefile', 'VFE.py', '\nimport numpy as np\nimport pandas as pd\n\n# feature engineering\n# from: https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153\ndef add_features(df):\n    df[\'area\'] = df[\'time_step\'] * df[\'u_in\']\n    df[\'area\'] = df.groupby(\'breath_id\')[\'area\'].cumsum()\n    \n    df[\'u_in_cumsum\'] = (df[\'u_in\']).groupby(df[\'breath_id\']).cumsum()\n    \n    df[\'u_in_lag1\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(1)\n    df[\'u_out_lag1\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(1)\n    df[\'u_in_lag_back1\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(-1)\n    df[\'u_out_lag_back1\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(-1)\n    df[\'u_in_lag2\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(2)\n    df[\'u_out_lag2\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(2)\n    df[\'u_in_lag_back2\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(-2)\n    df[\'u_out_lag_back2\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(-2)\n    df[\'u_in_lag3\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(3)\n    df[\'u_out_lag3\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(3)\n    df[\'u_in_lag_back3\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(-3)\n    df[\'u_out_lag_back3\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(-3)\n    df[\'u_in_lag4\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(4)\n    df[\'u_out_lag4\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(4)\n    df[\'u_in_lag_back4\'] = df.groupby(\'breath_id\')[\'u_in\'].shift(-4)\n    df[\'u_out_lag_back4\'] = df.groupby(\'breath_id\')[\'u_out\'].shift(-4)\n    df = df.fillna(0)\n    \n    df[\'breath_id__u_in__max\'] = df.groupby([\'breath_id\'])[\'u_in\'].transform(\'max\')\n    df[\'breath_id__u_out__max\'] = df.groupby([\'breath_id\'])[\'u_out\'].transform(\'max\')\n    \n    df[\'u_in_diff1\'] = df[\'u_in\'] - df[\'u_in_lag1\']\n    df[\'u_out_diff1\'] = df[\'u_out\'] - df[\'u_out_lag1\']\n    df[\'u_in_diff2\'] = df[\'u_in\'] - df[\'u_in_lag2\']\n    df[\'u_out_diff2\'] = df[\'u_out\'] - df[\'u_out_lag2\']\n    \n    df[\'breath_id__u_in__diffmax\'] = df.groupby([\'breath_id\'])[\'u_in\'].transform(\'max\') - df[\'u_in\']\n    df[\'breath_id__u_in__diffmean\'] = df.groupby([\'breath_id\'])[\'u_in\'].transform(\'mean\') - df[\'u_in\']\n    \n    df[\'breath_id__u_in__diffmax\'] = df.groupby([\'breath_id\'])[\'u_in\'].transform(\'max\') - df[\'u_in\']\n    df[\'breath_id__u_in__diffmean\'] = df.groupby([\'breath_id\'])[\'u_in\'].transform(\'mean\') - df[\'u_in\']\n    \n    df[\'u_in_diff3\'] = df[\'u_in\'] - df[\'u_in_lag3\']\n    df[\'u_out_diff3\'] = df[\'u_out\'] - df[\'u_out_lag3\']\n    df[\'u_in_diff4\'] = df[\'u_in\'] - df[\'u_in_lag4\']\n    df[\'u_out_diff4\'] = df[\'u_out\'] - df[\'u_out_lag4\']\n    df[\'cross\']= df[\'u_in\']*df[\'u_out\']\n    df[\'cross2\']= df[\'time_step\']*df[\'u_out\']\n    \n    df[\'R\'] = df[\'R\'].astype(str)\n    df[\'C\'] = df[\'C\'].astype(str)\n    df[\'R__C\'] = df["R"].astype(str) + \'__\' + df["C"].astype(str)\n    df = pd.get_dummies(df)\n    return df\n')


# In[4]:


from VFE import add_features

train = add_features(train_ori)
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)


# In[5]:


# normalise the dataset
RS = RobustScaler()
train = RS.fit_transform(train)

# Reshape to group 80 timesteps for each breath ID
train = train.reshape(-1, 80, train.shape[-1])


# The scaler is saved here to pickle, for use in other notebooks.

# In[6]:


dump(RS, open('RS.pkl', 'wb'))


# Save to Numpy.

# In[7]:


np.save('x_train.npy', train)
np.save('y_train.npy', targets)


# In[ ]:




