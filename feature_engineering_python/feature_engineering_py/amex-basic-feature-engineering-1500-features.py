#!/usr/bin/env python
# coding: utf-8

# # Notes

# Feature Engineering is an important part of the modeling pipeline to help expose different patterns in the data to the model. This notebook does some basic feature engineering and I hope it can serve as a decent starting point. With some experimentation, you can quickly get to the headache of feature selection. Runs in under a minute, thanks to the RAPIDS library, and produces 1486 features (not quite 1500)
# 
# I have saved the result of this notebook as a dataset
# 
# https://www.kaggle.com/datasets/illidan7/amexfeatureeng
# 
# There have been a lot of good feature engineering notebooks published for this competition so far. A lot of the features below are based on what others have shared and I have tried to add in some of my own as well
# 
# - Date based features (Column S_2)
# - "After pay" features (https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb)
# - Null columns handling 
#     - \>30% null; Count num nulls
#     - \>90% null; keep only last
# 
# - Categorical features
#     - cat1 features (The categorical features mentioned in the competition data page) https://www.kaggle.com/competitions/amex-default-prediction/data
#     - cat2 features (Low cardinality features; <=4 unique values)
#     - cat3 features (Low cardinality features; >=8 and <=21 unique values)
# 
# 
# - Last - First (https://www.kaggle.com/code/thedevastator/lag-features-are-all-you-need)
# - Last - mean features (https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977)
# 
# 
# Credits:
# - https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
# - https://www.kaggle.com/competitions/amex-default-prediction/discussion/333940
# - https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
# 
# Check out their work! Really appreciate the chance to learn from the Kaggle community

# # Load libraries

# In[1]:


# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
import cupy, cudf # GPU libraries
import matplotlib.pyplot as plt, gc, os
import seaborn as sns

print('RAPIDS version',cudf.__version__)


# In[2]:


# # VERSION NAME FOR SAVED MODEL FILES
# VER = 1

# # TRAIN RANDOM SEED
# SEED = 42

# # FILL NAN VALUE
# NAN_VALUE = -127 # will fit in int8

# # FOLDS PER MODEL
# FOLDS = 5


# # Read train data

# In[3]:


get_ipython().run_cell_magic('time', '', '\ndef read_file(path = \'\', usecols = None):\n    \n    # LOAD DATAFRAME\n    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)\n    else: df = cudf.read_parquet(path)\n    \n    # REDUCE DTYPE FOR CUSTOMER AND DATE\n    df[\'customer_ID\'] = df[\'customer_ID\'].str[-16:].str.hex_to_int().astype(\'int64\')\n    df.S_2 = cudf.to_datetime( df.S_2 )\n    df = df.sort_values([\'customer_ID\',\'S_2\'])\n                    \n    #################################\n    # Compute date based features\n    #################################\n    \n    df[\'S_2_dayofweek\'] = df[\'S_2\'].dt.weekday\n    df[\'S_2_dayofmonth\'] = df[\'S_2\'].dt.day\n    \n    df[\'days_since_1970\'] = df.S_2.astype(\'int64\')/1e9/(60*60*24)\n    df[\'S_2_diff\'] = df.days_since_1970.diff()\n    df[\'x\'] = df.groupby(\'customer_ID\').S_2.agg(\'cumcount\')\n    df.loc[df.x==0,\'S_2_diff\'] = 0\n    df = df.drop([\'days_since_1970\',\'x\'], axis=1)\n    \n    #################################\n    # Compute "after pay" features\n    #################################\n    \n    for bcol in [f\'B_{i}\' for i in [1,2,3,4,5,9,11,14,17,24]]+[\'D_39\',\'D_131\']+[f\'S_{i}\' for i in [16,23]]:\n        for pcol in [\'P_2\',\'P_3\']:\n            if bcol in df.columns:\n                df[f\'{bcol}-{pcol}\'] = df[bcol] - df[pcol]\n    \n    ###########################\n    # Null columns handling\n    ###########################\n    \n    nullvals = df.isnull().sum() / df.shape[0]\n    nullCols = nullvals[nullvals>0.3].index.to_arrow().to_pylist()\n    \n    for col in nullCols:\n        df[col+\'_null\'] = df[col].isnull().astype(int)\n    \n    # Drop raw date column\n    df = df.drop(columns=[\'S_2\'])\n    \n    print(\'shape of data:\', df.shape)\n    \n    return df\n\nprint(\'Reading train data...\')\nTRAIN_PATH = \'../input/amex-data-integer-dtypes-parquet-format/train.parquet\'\ntrain = read_file(path = TRAIN_PATH)\n\n\nprint(train.shape)\n')


# # Build features

# In[4]:


get_ipython().run_cell_magic('time', '', '\ndef process_and_feature_engineer(df):\n    \n    all_cols = [c for c in list(df.columns) if c not in [\'customer_ID\',\'S_2\']]\n    cat1_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]\n    cat2_features = [\n                    \'B_31\',\'B_32\',\'B_33\',\'D_103\',\'D_109\',\'D_111\',\'D_127\',\n                    \'D_129\',\'D_135\',\'D_137\',\'D_139\',\'D_140\',\'D_143\',\'D_86\',\n                    \'D_87\',\'D_92\',\'D_93\',\'D_94\',\'D_96\',\'R_15\',\'R_19\',\'R_2\',\'R_21\',\n                    \'R_22\',\'R_23\',\'R_24\',\'R_25\',\'R_28\',\'R_4\',\'S_18\',\'S_20\',\'S_6\'\n                       ]\n    cat3_features = [\n                    \'R_9\',\'R_18\',\'R_10\',\'R_11\',\'D_89\',\'D_91\',\'D_81\',\'D_82\',\'D_136\',\n                    \'D_138\',\'D_51\',\'D_123\',\'D_125\',\'D_108\',\'B_41\',\'B_22\',\n                       ]\n    \n    nullvals = df.isnull().sum() / df.shape[0]\n    exclnullCols = nullvals[nullvals>0.9].index.to_arrow().to_pylist()\n    nullCols = nullvals[nullvals>0.3].index.to_arrow().to_pylist()\n    nullAggCols = [col + "_null" for col in nullCols]\n    \n    cat_features = cat1_features + cat2_features + cat3_features + exclnullCols + nullAggCols\n    \n    num_features = [col for col in all_cols if col not in cat_features]\n\n    test_num_agg = df.groupby("customer_ID")[num_features].agg([\'first\',\'mean\', \'std\', \'min\', \'max\', \'last\'])\n    test_num_agg.columns = [\'_\'.join(x) for x in test_num_agg.columns]\n        \n    # Diff/Div columns\n    for col in test_num_agg.columns:\n        \n        # Last/First\n        if \'last\' in col and col.replace(\'last\', \'first\') in test_num_agg.columns:\n            test_num_agg[col + \'_life_sub\'] = test_num_agg[col] - test_num_agg[col.replace(\'last\', \'first\')]\n  #             test_num_agg[col + \'_life_div\'] = cupy.where((test_num_agg[col.replace(\'last\', \'first\')].isnull()), 0, \n#                                                          cupy.where((test_num_agg[col.replace(\'last\', \'first\')]==0), 0, test_num_agg[col] / test_num_agg[col.replace(\'last\', \'first\')]))\n        # Last/Mean\n        if \'last\' in col and col.replace(\'last\', \'mean\') in test_num_agg.columns:\n            test_num_agg[col + \'_lmean_sub\'] = test_num_agg[col] - test_num_agg[col.replace(\'last\', \'mean\')]\n#             test_num_agg[col + \'_lmean_div\'] = cupy.where((test_num_agg[col.replace(\'last\', \'first\')].isnull()) | (test_num_agg[col.replace(\'last\', \'first\')]==0), 0, test_num_agg[col] / test_num_agg[col.replace(\'last\', \'first\')])\n    \n    test_cat1_agg = df.groupby("customer_ID")[cat1_features].agg([\'first\', \'last\', \'nunique\'])\n    test_cat1_agg.columns = [\'_\'.join(x) for x in test_cat1_agg.columns]\n    \n    test_cat2_agg = df.groupby("customer_ID")[cat2_features].agg([\'first\', \'last\', \'nunique\'])\n    test_cat2_agg.columns = [\'_\'.join(x) for x in test_cat2_agg.columns]\n    \n    test_cat3_agg = df.groupby("customer_ID")[cat3_features].agg([\'first\', \'last\', \'nunique\',\'min\', \'max\',\'mean\', \'std\'])\n    test_cat3_agg.columns = [\'_\'.join(x) for x in test_cat3_agg.columns]\n    \n    test_null_agg = df.groupby("customer_ID")[nullAggCols].agg([\'count\'])\n    test_null_agg.columns = [\'_\'.join(x) for x in test_null_agg.columns]\n    \n    test_exclnull_agg = df.groupby("customer_ID")[exclnullCols].agg([\'last\'])\n    test_exclnull_agg.columns = [\'_\'.join(x) for x in test_exclnull_agg.columns]\n         \n    temp1 = df.groupby([\'customer_ID\'])[\'P_2\'].count()\n    temp1 = temp1.reset_index()\n    temp1.columns = [\'customer_ID\',\'num_statements\']\n    temp1 = temp1.set_index(\'customer_ID\')\n \n    df = cudf.concat([test_num_agg, test_cat1_agg, test_cat2_agg, test_cat3_agg, temp1, test_null_agg, test_exclnull_agg], axis=1) #test_bal_agg\n    del test_num_agg, test_cat1_agg, test_cat2_agg, test_cat3_agg, temp1, test_null_agg, test_exclnull_agg\n    _ = gc.collect()\n     \n    print(\'shape after engineering\', df.shape )\n    \n    return df\n\ntrain = process_and_feature_engineer(train)\n\nprint(train.shape)\n')


# In[5]:


get_ipython().run_cell_magic('time', '', "\n# ADD TARGETS\ntargets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')\ntargets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')\ntargets = targets.set_index('customer_ID')\n\ntrain = train.merge(targets, left_index=True, right_index=True, how='left')\ntrain.target = train.target.astype('int8')\ndel targets\n_ = gc.collect()\n\n# NEEDED TO MAKE CV DETERMINISTIC (cudf merge above randomly shuffles rows)\ntrain = train.sort_index().reset_index()\n\n# FEATURES\nprint(f'There are {len(train.columns[1:-1])} features!')\n")


# # Save dataset

# In[6]:


get_ipython().run_cell_magic('time', '', '\ntrain.to_parquet("train_features.parquet")\n')

