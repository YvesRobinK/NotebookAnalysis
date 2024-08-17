#!/usr/bin/env python
# coding: utf-8

# ## AMEX Default Competition - Feature Engineering and Feature Selection by Pearson Correlation   
#     
# **this notebook does the following:**
# 
# - Data processing: 
#     - impute missing values
# - Feature engineering:
#     - creating dummy features for categrical features
#     - creating log features for some numeric features
# - Feature selection:
#     - select features by pearson correlation
#     
#     
# **addtional notes**
# - The input data for this file 
#     - notebook that generates the input data: https://www.kaggle.com/code/xxxxyyyy80008/amex-feature-engineering-agg-by-cust-id
#     - data files: https://www.kaggle.com/datasets/xxxxyyyy80008/amex-agg-data-rev2
#     
# - The output from this file:
#     - data files: https://www.kaggle.com/datasets/xxxxyyyy80008/amex-agg-data-rev2

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


# In[2]:


import pandas as pd
import numpy as np
import os 
from pathlib import Path

from datetime import datetime, timedelta
import time 
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pyarrow.parquet as pq
import pyarrow as pa


# In[3]:


import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows=999
pd.options.display.max_columns=999


# In[4]:


get_ipython().run_cell_magic('time', '', "df=pd.read_parquet('/kaggle/input/amex-agg-data-rev2/agg_train_all_rev2.parquet', engine='pyarrow')\n")


# In[5]:


all_cols = df.columns.tolist()
all_cols.sort()
print(all_cols)
len(all_cols)


# In[6]:


id_feats = ['customer_ID']
date_col =  'S_2'
cat_feats = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68', 'B_31']
missing20 = ['D_87', 'D_88', 'D_108', 'D_111', 'D_110', 'B_39', 'D_73', 'B_42', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'R_9', 'B_29', 'D_106', 'D_132', 'D_49', 'R_26', 'D_76', 'D_66', 'D_42', 'D_142', 'D_53', 'D_82', 'D_50', 'B_17', 'D_105', 'D_56', 'S_9', 'D_77', 'D_43', 'S_27', 'D_46']
float_feats = ['R_17', 'B_40', 'R_27', 'S_18', 'B_13', 'B_33', 'R_20', 'S_6', 'R_23', 'R_16', 'B_24', 'D_125', 'D_44', 'D_91', 'D_71', 'P_2', 'B_15', 'D_103', 'S_12', 'D_144', 'D_123', 'D_94', 'D_70', 'D_39', 'P_4', 'S_23', 'R_12', 'S_5', 'D_72', 'B_27', 'B_6', 'D_89', 'D_143', 'D_80', 'B_3', 'B_28', 'R_11', 'B_14', 'B_1', 'D_124', 'D_109', 'B_25', 'B_36', 'B_5', 'B_18', 'D_61', 'R_13', 'B_37', 'S_7', 'D_104', 'B_26', 'B_4', 'R_6', 'D_133', 'B_21', 'S_19', 'D_115', 'R_18', 'D_45', 'D_69', 'S_24', 'D_84', 'S_17', 'B_12', 'D_52', 'R_24', 'D_127', 'R_14', 'D_113', 'D_83', 'D_141', 'B_10', 'S_22', 'D_96', 'R_15', 'S_25', 'D_54', 'D_60', 'D_59', 'S_11', 'R_8', 'D_74', 'R_4', 'D_118', 'D_62', 'B_7', 'S_15', 'B_2', 'R_28', 'S_26', 'D_119', 'D_86', 'D_81', 'D_93', 'R_3', 'B_16', 'B_9', 'D_107', 'D_78', 'D_140', 'S_13', 'B_11', 'D_47', 'R_1', 'D_55', 'R_22', 'D_102', 'D_112', 'D_131', 'B_32', 'R_10', 'R_7', 'R_19', 'D_41', 'D_130', 'B_23', 'R_5', 'D_121', 'B_19', 'P_3', 'B_8', 'D_79', 'D_122', 'S_3', 'R_25', 'D_92', 'D_58', 'D_51', 'B_41', 'S_8', 'B_22', 'D_139', 'R_2', 'D_48', 'D_145', 'D_129', 'B_20', 'S_16', 'S_20', 'D_128', 'D_75', 'D_65', 'R_21']
to_log_feats= ['B_11', 'D_102', 'D_107', 'R_28', 'D_137', 'D_108', 'D_115', 'D_138', 'B_9', 'B_40', 'D_39', 'D_113', 'D_119', 'D_60', 'B_5', 'B_4', 'D_136', 'B_18', 'D_51', 'D_44', 'S_26', 'D_135', 'D_131', 'B_13', 'B_36', 'D_106', 'D_118', 'B_32', 'B_12', 'B_27', 'B_28', 'B_41', 'D_125', 'B_22', 'D_41', 'B_29', 'D_49', 'B_24', 'D_133', 'B_42', 'D_45', 'B_3', 'S_5', 'D_140', 'B_23', 'B_21', 'D_123', 'D_109', 'B_26', 'D_43']



# In[7]:


eps =  1e-8
log_feats = []

for c in all_cols:
    
    if c in ['customer_ID', 'target']:
        continue
    
    if df[c].dtype not in ['int64', 'int32', 'float64', 'float32']:
        continue
        
    if '|' in c:
        c0, c1 = c.split('|')
        if (c0 in cat_feats):
            if (c1 == 'last'):
                if c0 in ['D_68', 'D_120', 'D_126', 'B_38', 'D_116', 'D_117', 'B_31', 'B_30', 'D_114']:
                    df[c] = df[c].fillna(value=999)
                if c0 in [ 'D_63',  'D_64']: 
                    df[c] = df[c].fillna(value='NA') 
                    
                    dummies_ = pd.get_dummies(df[c])
                    dummy_feats_ =  [f'{c}={cc}' for cc in dummies_.columns]
                    dummies_.columns = dummy_feats_
                    df[dummy_feats_] = dummies_.values
        
            else:
                df[c] = df[c].fillna(value=0)
        else:
            df[c] = df[c].fillna(value=df[c].mean())
            if (c0 in to_log_feats) & (df[c].min()>0):
                df[f'log_{c}'] = np.log(df[c].values + eps)
                log_feats.append(f'log_{c}')
    elif '=' in c:
        df[c] = df[c].fillna(value=0)
        



# In[18]:


all_cols = df.columns.tolist()
all_cols.sort()
print(all_cols)
len(all_cols)


# In[14]:


float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
df[float64_cols] = np.float32(df[float64_cols].values)
#---convert int64 to int32
int64_cols = df.select_dtypes(include=['int64']).columns.tolist()
df[int64_cols] = np.int32(df[int64_cols].values)


# In[15]:


float_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
#---convert int64 to int32
int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()


# In[19]:


set(all_cols)-set(float_cols+int_cols)


# In[20]:


len(float_cols), len(int_cols), len(all_cols), len(float_cols)+ len(int_cols)


# In[21]:


get_ipython().run_cell_magic('time', '', "pq.write_table(pa.Table.from_pandas(df), f'agg_train_all_rev2_rev.parquet', compression = 'GZIP')\n")


# ## calculate the correlation

# In[22]:


get_ipython().run_cell_magic('time', '', "corr_list = []\nall_cols = list(set(all_cols) - set(['customer ID', 'target']))\ny = df['target'].values\nall_num_feats = float_cols + int_cols\nfor c in all_num_feats:\n    x_ = df[c].values\n    corr_list.append([c, np.corrcoef(x_, y)[0, 1]])\n")


# In[23]:


corr_target = pd.DataFrame(data=corr_list, columns=['feat', 'corr'])
corr_target.sort_values(by='corr', ascending=False, inplace=True)
corr_target.to_csv('corr_w_target.csv', sep='|', index=False)


# In[26]:


corr_target.head(20)


# In[30]:


get_ipython().run_cell_magic('time', '', "corr_list = []\nall_num_feats = float_cols + int_cols\nall_num_feats = list(set(all_num_feats) - set(['target']))\n\nfor i in range(len(all_num_feats)-1):\n    c1 = all_num_feats[i]\n    x1_ = df[c1].values\n    for j in range(i+1, len(all_num_feats)):\n        c2 = all_num_feats[j]\n        x2_ = df[c2].values\n        corr_list.append([c1, c2,  np.corrcoef(x1_, x2_)[0, 1]])\n\n")


# In[31]:


corr_feats = pd.DataFrame(data=corr_list, columns=['feat1', 'feat2', 'corr'])
corr_feats.sort_values(by='corr', ascending=False, inplace=True)
corr_feats.to_csv('corr_feats.csv', sep='|', index=False)


# In[42]:


corr_feats.head(10)


# In[58]:


corr_target.sort_values(by='corr', ascending=False, inplace=True)
min_corr = 0.75
max_feats = 150

selected = [corr_target.iloc[0]['feat']]
for _, row in corr_target.iloc[1:].iterrows():
    feat1 = row['feat']
    corr_pass = True
    if c=='target':
        continue
    for feat0 in selected:
        sel_by = ((corr_feats['feat1']==feat0) & (corr_feats['feat2']==feat1))
        sel_by = sel_by | ((corr_feats['feat2']==feat0) & (corr_feats['feat1']==feat1))
        corr_ = corr_feats[sel_by]['corr'].iloc[0]
        if corr_>min_corr:
            corr_pass = False
            break
    if corr_pass:
        selected.append(feat1)
    
    if len(selected)>=max_feats:
        break
    


# In[59]:


print(selected)


# In[67]:


selected = ['D_48|last', 'R_1|max', 'B_9|last', 'log_B_3|last', 'D_44|last', 'log_B_23|last', 'log_B_11|last', 'D_58|last', 'R_1|last', 'log_B_9|mean', 'D_61|min', 'B_4|last', 'R_10|max', 'B_22|max', 'B_19|last', 'B_7|max', 'R_2|max', 'log_D_44|min', 'B_11|last', 'B_9|min', 'B_30|nunique', 'R_2|last', 'log_B_22|last', 'D_39|max', 'R_2|mean', 'log_B_40|max', 'D_41|last', 'B_20|mean', 'log_D_41|max', 'B_23|min', 'S_7|max', 'log_B_4|min', 'D_70|max', 'R_3|mean', 'D_78|last', 'B_8|min', 'D_55|min', 'D_39|last', 'B_17|mean', 'S_15|max', 'log_B_40|min', 'B_38=4.0', 'B_1|min', 'R_8|max', 'R_15|max', 'S_3|last', 'B_38|nunique', 'D_84|max', 'D_43|mean', 'B_38=5.0', 'P_4|max', 'R_10|last', 'S_3|min', 'log_B_26|mean', 'B_40|min', 'R_24|max', 'log_D_39|last', 'B_28|last', 'log_D_43|mean', 'R_16|max', 'log_B_28|max', 'D_131|max', 'R_26_mean2std', 'log_B_26|last', 'S_15|mean', 'D_72|max', 'R_5|last', 'B_30=2.0', 'B_17_mean2std', 'log_B_22|min', 'log_B_21|mean', 'R_3|min', 'D_130|max', 'D_120=1.0', 'R_11|mean', 'D_46|mean', 'B_25|last', 'D_53_mean2std', 'D_59|last', 'R_24|last', 'R_6|max', 'S_15|last', 'R_9_mean2std', 'D_81|mean', 'D_70|min', 'D_135_mean2std', 'D_89|mean', 'R_15|last', 'log_S_5|mean', 'D_131|min', 'D_53|last', 'R_13|mean', 'D_42|max', 'R_17|max', 'B_38=6.0', 'log_B_24|mean', 'B_32|max', 'B_22|min', 'R_20|mean', 'P_4|min', 'S_22|max', 'R_6|last', 'log_D_133|last', 'D_89|last', 'log_D_113|min', 'R_21|last', 'log_D_41|min', 'B_17|max', 'D_133|max', 'D_113|max', 'D_120|nunique', 'log_D_60|min', 'D_114=0.0', 'R_20|last', 'D_133|last', 'B_31|nunique', 'R_25|max', 'R_22|max', 'D_46|last', 'S_22|mean', 'R_1|min', 'B_38=7.0', 'B_24|mean', 'log_B_24|last', 'B_24|last', 'log_D_39|min', 'S_27_mean2std', 'log_D_133|min', 'D_41|min', 'D_140|max', 'B_32|last', 'R_7|max', 'B_21|max', 'R_19|last', 'D_143|last', 'log_D_107|min', 'D_64=U', 'log_B_41|max', 'D_53|min', 'B_14|min', 'R_17|last', 'B_41|last', 'R_16|last', 'D_61|max', 'D_65|max', 'D_107|max', 'S_23|mean', 'D_145|last', 'D_46|min', 'log_B_21|last']

corr_target[corr_target['feat'].isin(selected)]


# In[60]:


corr_target.sort_values(by='corr', ascending=False, inplace=True)
min_corr = 0.85
max_feats = 150

selected = [corr_target.iloc[0]['feat']]
for _, row in corr_target.iloc[1:].iterrows():
    feat1 = row['feat']
    corr_pass = True
    if c=='target':
        continue
    for feat0 in selected:
        sel_by = ((corr_feats['feat1']==feat0) & (corr_feats['feat2']==feat1))
        sel_by = sel_by | ((corr_feats['feat2']==feat0) & (corr_feats['feat1']==feat1))
        corr_ = corr_feats[sel_by]['corr'].iloc[0]
        if corr_>min_corr:
            corr_pass = False
            break
    if corr_pass:
        selected.append(feat1)
    
    if len(selected)>=max_feats:
        break
    


# In[61]:


print(selected)


# In[68]:


selected = ['D_48|last', 'R_1|max', 'log_D_44|mean', 'B_9|last', 'log_B_3|last', 'D_61|last', 'D_55|last', 'D_44|last', 'log_B_23|last', 'B_3|last', 'D_48|min', 'B_7|last', 'log_B_11|last', 'D_75|last', 'D_61|mean', 'log_B_22|mean', 'log_B_9|last', 'log_B_11|mean', 'R_1|last', 'log_B_23|max', 'B_4|last', 'B_20|last', 'R_10|max', 'B_19|last', 'B_1|last', 'log_B_4|last', 'R_2|max', 'log_B_40|last', 'log_D_44|min', 'B_9|max', 'log_B_40|mean', 'B_9|min', 'log_B_9|max', 'B_30|nunique', 'log_B_23|min', 'B_16|max', 'R_2|last', 'D_78|max', 'log_B_11|min', 'log_B_9|min', 'R_4|max', 'B_22|last', 'log_B_4|max', 'log_B_3|min', 'D_39|max', 'R_2|mean', 'D_58|min', 'B_7|min', 'D_41|last', 'log_D_41|max', 'S_7|max', 'log_D_41|last', 'D_70|max', 'D_44|min', 'R_4|last', 'R_3|mean', 'D_78|last', 'B_8|min', 'D_55|min', 'D_39|last', 'B_17|mean', 'S_15|max', 'log_B_40|min', 'B_17|last', 'B_38=4.0', 'B_1|min', 'S_7|last', 'R_8|max', 'D_70|last', 'R_15|max', 'B_38|nunique', 'B_3|min', 'D_84|max', 'B_16|min', 'D_43|mean', 'B_38=5.0', 'log_B_26|max', 'D_39|mean', 'P_4|max', 'R_15|mean', 'R_10|last', 'log_D_39|max', 'B_17|min', 'S_3|min', 'D_41|mean', 'B_19|min', 'R_8|mean', 'B_40|min', 'B_20|min', 'R_24|max', 'R_16|mean', 'log_D_39|last', 'R_3|last', 'D_43|last', 'B_28|last', 'log_D_43|mean', 'D_84|last', 'R_16|max', 'D_131|max', 'R_26_mean2std', 'D_81|max', 'log_B_26|last', 'S_15|mean', 'D_72|max', 'R_5|last', 'B_30=2.0', 'B_17_mean2std', 'D_79|max', 'log_B_22|min', 'R_13|max', 'log_B_21|mean', 'R_8|last', 'D_42_mean2std', 'D_130|last', 'R_3|min', 'D_49_mean2std', 'R_24|mean', 'D_120=1.0', 'R_11|mean', 'D_46|mean', 'B_25|last', 'D_53_mean2std', 'log_D_131|last', 'D_59|last', 'R_24|last', 'R_6|max', 'S_15|last', 'R_9_mean2std', 'D_70|min', 'R_20|max', 'D_135_mean2std', 'B_25|min', 'D_89|mean', 'D_132_mean2std', 'R_15|last', 'log_S_5|mean', 'log_D_43|min', 'D_131|min', 'S_20|max', 'D_53|last', 'R_13|mean', 'R_11|max', 'log_D_131|min', 'D_42|max', 'R_17|max', 'B_38=6.0', 'log_B_24|mean', 'B_32|max', 'B_22|min', 'P_4|min']

corr_target[corr_target['feat'].isin(selected)]


# In[62]:


len(selected)


# In[ ]:




