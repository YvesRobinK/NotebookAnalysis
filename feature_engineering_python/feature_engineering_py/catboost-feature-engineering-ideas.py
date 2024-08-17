#!/usr/bin/env python
# coding: utf-8

# # American Express - Default Prediction
# # By Mohamed Eltayeb

# ## Note: Please Enable the GPU and upvote if you find this notebook useful.

# # Import libraries

# In[ ]:


import numpy as np
import cudf 
import cupy
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm


from sklearn.model_selection import KFold
from sklearn import base

from catboost import CatBoostClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams["figure.figsize"] = (12, 8)


# # Define Functions

# In[ ]:


#Competition Metric
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return [0.5 * (gini[1]/gini[0] + top_four),top_four, gini[1]/gini[0]]


# In[ ]:


#Plot the LGBM Features Importances
def ShowImp(features, importances, num = 20, fig_size = (2*num, num)):
    feature_imp = cudf.DataFrame({'Value':importances,'Feature':features})
    feature_imp = feature_imp.sort_values(by="Value",ascending=False)[:num]
    print(feature_imp)


# In[ ]:


#Reduce Memory Usage
def reduce_memory_usage(df):
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:

                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
    
    return df


# In[ ]:


#READING DATA
def read_file(path = '', usecols = None):
    
    # LOAD DATAFRAME
    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)
    else: df = cudf.read_parquet(path)
        
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    
    # SORT BY CUSTOMER AND DATE
    df = df.sort_values(['customer_ID','S_2'])
    df = df.reset_index(drop=True)
    
    #REDUCE MEMORY USAGE
    df = reduce_memory_usage(df)
    print('shape of data:', df.shape)
    
    return df


# # Training Data Preparation

# In[ ]:


print('Reading train data...')
TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
train_df = read_file(path = TRAIN_PATH)


# ## Temporary Dataset

# In[ ]:


train_df_new = cudf.DataFrame(train_df['customer_ID'].drop_duplicates(),columns=['customer_ID'])


# ## Polar Coordinates of P_2 and The Other Features

# In[ ]:


def Polar(X,y, a = 9.9, b = 9.9): # a and b represnt the center
    r = np.sqrt((X-a)**2 + (y-b)**2)
    phi = np.arctan2((y-a), (X-b))
    return r, phi

feats = ['B_18','B_2','B_33','D_62','D_77','D_47','P_3','D_45','D_51','R_27','S_25','D_112',
         'D_121','D_128','D_52','D_115','D_114','D_127','D_68','D_118','D_119','D_122','D_54',
         'D_129','D_134','S_8','D_92','R_12','D_91','D_76','D_56','D_117','B_42','S_6','D_73',
         'S_13','D_142','D_126','D_140','R_16','B_41','R_21','B_32','B_24','D_133','R_17','D_120',
         'D_46','D_145','D_139','D_143','D_138','D_136','R_6','D_135','B_25','D_137','R_26','D_141',
         'R_13','D_79','D_39','D_89','D_113','R_20','R_15','S_15','R_8','D_130','D_131','D_64','D_59',
         'R_24','R_5','B_28','D_88','R_10','P_4','D_78','D_43','R_3','R_9','D_41','D_70','R_4','D_81',
         'D_84','S_3','B_17','D_72','B_11','B_30','S_7','B_37','B_1','B_22','B_8','R_2','B_19','D_53',
         'B_4','D_61','B_38','B_3','B_20','R_1','D_42','B_16','B_23','B_7','D_74','D_44','D_75','D_58',
         'B_9','D_55','D_48']

for feat in tqdm(feats):
    
    train_df[f'P_2_{feat}_R'], train_df[f'P_2_{feat}_Phi'] = Polar(train_df["P_2"].fillna(127).to_array(),train_df[feat].fillna(127).to_array())
    
    train_df_new[f'P_2_{feat}_R_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'P_2_{feat}_R'].mean())
    train_df_new[f'P_2_{feat}_R_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'P_2_{feat}_R'].last())
    train_df_new[f'P_2_{feat}_Phi_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'P_2_{feat}_Phi'].mean())
    train_df_new[f'P_2_{feat}_Phi_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'P_2_{feat}_Phi'].last())
    
    train_df.drop([f'P_2_{feat}_R',f'P_2_{feat}_Phi'],inplace=True,axis=1)
    train_df_new = reduce_memory_usage(train_df_new)


# ## Adding Bins for P_2, D_39, B_1

# In[ ]:


for feat in ['P_2','B_1']:
    percentile = {}
    for i, per in enumerate([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        percentile[i] = train_df[feat].quantile(per)
    train_df[f'{feat}_bins'] = cudf.cut(train_df[feat],[percentile[x] for x in range(0,11)], labels=False).fillna(-127).astype(int)
    
train_df['D_39_bins'] = cudf.cut(train_df['D_39'],[0,31,61,91,121,151,181,np.inf], labels=False).fillna(-127).astype(int)


# ## Features Groups (Have similar mean or high correlation with each other)

# In[ ]:


feats_groups = []
feats_groups.append(['D_111','D_108','D_87','D_136','D_138','D_135','D_137'])
feats_groups.append(['D_136','D_138','D_135','D_137'])
feats_groups.append(['R_28','R_23','R_18'])
feats_groups.append(['R_25','R_22'])
feats_groups.append(['R_17','R_4','R_19','R_21','R_15','R_13','R_24'])
feats_groups.append(['R_10','R_5','R_6'])
feats_groups.append(['B_14','B_13'])
feats_groups.append(['B_11','B_42'])
feats_groups.append(['B_11','B_25'])
feats_groups.append(['B_37','B_1'])
feats_groups.append(['B_37','B_1','B_11'])
feats_groups.append(['D_139','D_143'])
feats_groups.append(['D_43','D_69'])
feats_groups.append(['D_102','B_9','D_62'])
feats_groups.append(['D_56','B_40'])
feats_groups.append(['S_3','S_7'])
feats_groups.append(['S_12','S_6'])
feats_groups.append(['D_115','D_119','D_118'])
feats_groups.append(['D_47','D_129','D_49'])
feats_groups.append(['P_3','P_2'])
feats_groups.append(['D_63','D_75'])
feats_groups.append(['S_8','S_13'])
feats_groups.append(['B_4','B_19'])
feats_groups.append(['B_2','B_18'])
feats_groups.append(['B_2','B_33'])
feats_groups.append(['D_75','D_58','D_74'])
feats_groups.append(['D_48','P_2','D_55','D_61'])
feats_groups.append(['D_75','D_55','D_58'])
feats_groups.append(['D_48','P_2'])
feats_groups.append(['D_48','D_61','D_55'])
feats_groups.append(['D_48','D_58'])
feats_groups.append(['B_9','D_75'])
feats_groups.append(['D_69','D_65'])
feats_groups.append(['D_106','D_39'])
feats_groups.append(['B_10','B_16'])


# In[ ]:


for i,k in enumerate(feats_groups):
    
    train_df[f'Feats_Group{i}_Mean'] = train_df[k].mean(axis=1)
    train_df_new[f'Feats_Group{i}_Mean_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'Feats_Group{i}_Mean'].mean())
    train_df_new[f'Feats_Group{i}_Mean_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'Feats_Group{i}_Mean'].last())
    
    train_df[f'Feats_Group{i}_Sum'] = train_df[k].sum(axis=1)
    train_df_new[f'Feats_Group{i}_Sum_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'Feats_Group{i}_Sum'].mean())
    train_df_new[f'Feats_Group{i}_Sum_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'Feats_Group{i}_Sum'].last())
    
    train_df.drop([f'Feats_Group{i}_Mean',f'Feats_Group{i}_Sum'],inplace=True,axis=1)
    train_df_new = reduce_memory_usage(train_df_new)


# ## Add Some Interactions Between D_106 and D_39

# In[ ]:


train_df['D_106_D_39_sum'] = train_df['D_106']+ train_df['D_39']
train_df['D_106_D_39_mean'] = train_df['D_106']+ train_df['D_39']
train_df['D_106_D_39_ratio'] = train_df['D_106'] / train_df['D_39']


# ## Target Encoding

# In[ ]:


#Target Encoding For Training Data
class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self,colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col


    def fit(self, X, y=None):
        return self


    def transform(self,X,test_df):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X,X[self.targetName]):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X[col_mean_name][val_ind] = X_val[self.colnames].fillna(-127).map(X_tr.groupby(self.colnames)[self.targetName].mean()).to_array()

        X[col_mean_name].fillna(mean_of_target, inplace = True)
        
        if self.verbosity:

            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            

        return X, test_df


# In[ ]:


# Add the training labels to add the target encoded features then drop the target
labels = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
labels['customer_ID'] = labels['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
train_df['Target'] = train_df['customer_ID'].map(labels.groupby('customer_ID')['target'].last()).astype(np.int8)
train_df = reduce_memory_usage(train_df)
del labels


# In[ ]:


cat_feats = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",'D_39_bins','P_2_bins']
for feat in tqdm(cat_feats):
    tar_enc = KFoldTargetEncoderTrain(feat,'Target',n_fold=5)
    tar_enc.fit(train_df)
    train_df, _ = tar_enc.transform(train_df,'')
    
train_df.drop('Target',inplace=True,axis=1)   #Drop the target (will be added again after making all the features)
train_df = reduce_memory_usage(train_df)


# ## Define The Features Lists

# In[ ]:


all_cols = [c for c in list(train_df.columns) if c not in ['customer_ID','S_2']]
cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",
               'D_39_bins','P_2_bins','B_1_bins']
num_features = [col for col in all_cols if col not in cat_features]
cat_features = ["B_38","D_114","D_117","D_120","D_63","D_64","D_66",      # Dropped some useless features
                'D_39_bins','P_2_bins','B_1_bins']


# ## Adding Aggergations for Some Lagged Features

# In[ ]:


feats = ['B_2','D_39','B_1','P_2','D_106_D_39_sum','D_106_D_39_mean','D_106_D_39_ratio']

for Feature in tqdm(feats):
    for window in [1,2,3]:
        train_df[f'{Feature}_lag_{window}'] = train_df[Feature].shift(window)
        
        train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'{Feature}_lag_{window}'].last())
        train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_current'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'{Feature}'].last())
        train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_CurrentDiffLast'] = train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_current'] - train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last']
        train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[f'{Feature}_lag_{window}'].mean())
        train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_LastDiffMean'] = train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last'] - train_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_mean']
        
        train_df_new = reduce_memory_usage(train_df_new)
        train_df.drop(f'{Feature}_lag_{window}',inplace=True,axis=1)
        train_df_new.drop([f'{Feature}_lag_{window}_Agg_customer_ID_mean',f'{Feature}_lag_{window}_Agg_customer_ID_current'],inplace=True,axis=1)


# ## Adding Aggregations for All The Other Features (We will use the aggregations for training instead of the original values to make the data fit into the memory)

# In[ ]:


#Numerical Features
for Feature in tqdm(num_features):
                train_df_new[f'{Feature}_Agg_customer_ID_mean'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].mean())
                train_df_new[f'{Feature}_Agg_customer_ID_std'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].std())  
                train_df_new[f'{Feature}_Agg_customer_ID_min'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].min())  
                train_df_new[f'{Feature}_Agg_customer_ID_max'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].max())  
                train_df_new[f'{Feature}_Agg_customer_ID_first'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].first())
                train_df_new[f'{Feature}_Agg_customer_ID_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].last())
                train_df_new[f'{Feature}_Agg_customer_ID_LastDiffMean'] = train_df_new[f'{Feature}_Agg_customer_ID_last'] - train_df_new[f'{Feature}_Agg_customer_ID_mean']
                train_df_new[f'{Feature}_Agg_customer_ID_LastDiffFirst'] = train_df_new[f'{Feature}_Agg_customer_ID_last'] - train_df_new[f'{Feature}_Agg_customer_ID_first']
                
                train_df.drop(Feature,inplace=True,axis=1)
                train_df_new = reduce_memory_usage(train_df_new)
#-------------------------------------------------------------------------------------------------------------------------------------------------
#Categorical Features
train_df_new['D_39_bins_Agg_customer_ID_nunique'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')['D_39_bins'].nunique())  
for Feature in tqdm(cat_features):
                train_df_new[f'{Feature}_Agg_customer_ID_last'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].last())
                train_df_new[f'{Feature}_Agg_customer_ID_first'] = train_df_new['customer_ID'].map(train_df.groupby('customer_ID')[Feature].first())
                
                train_df.drop(Feature,inplace=True,axis=1)
                train_df_new = reduce_memory_usage(train_df_new)

print('Finished!!!')
del train_df


# ## Add The Target to The Dataset

# In[ ]:


labels = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
train_df_new = train_df_new.sort_index().reset_index(drop=True)
labels['customer_ID'] = labels['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
train_df_new['Target'] = train_df_new['customer_ID'].map(labels.groupby('customer_ID')['target'].last()).astype(np.int8)
train_df_new = reduce_memory_usage(train_df_new)
del labels


# ## Drop Some Useless Features (They increase the memory usage without any positive impact on the results)

# In[ ]:


# 1
print('Training Data Shape Before: ',train_df_new.shape)
feat_df = pd.read_csv('../input/features-importance/Features.csv')
feats = feat_df[feat_df['importance'] == 0.0].feature.to_array()
train_df_new.drop(feats,inplace=True,axis=1)
print('Training Data Shape After: ',train_df_new.shape)
del feat_df, feats


# In[ ]:


# 2
feats = ['S_20_Agg_customer_ID_min','R_20_Agg_customer_ID_first','D_139_Agg_customer_ID_mean','D_138_Agg_customer_ID_min',
         'D_137_Agg_customer_ID_LastDiffFirst','R_24_Agg_customer_ID_LastDiffFirst','D_123_Agg_customer_ID_last',
         'D_87_Agg_customer_ID_mean','R_17_Agg_customer_ID_first','R_25_Agg_customer_ID_mean','D_86_Agg_customer_ID_LastDiffFirst',
         'B_33_Agg_customer_ID_max','D_137_Agg_customer_ID_max','D_81_Agg_customer_ID_min','D_108_Agg_customer_ID_last',
         'S_18_Agg_customer_ID_LastDiffFirst','D_93_Agg_customer_ID_LastDiffFirst','D_96_Agg_customer_ID_LastDiffFirst',
         'R_28_Agg_customer_ID_LastDiffFirst','R_13_Agg_customer_ID_std','D_139_Agg_customer_ID_first','D_88_Agg_customer_ID_first',
         'D_123_Agg_customer_ID_min','B_32_Agg_customer_ID_max','B_31_Agg_customer_ID_std','R_24_Agg_customer_ID_last',
         'R_13_Agg_customer_ID_mean','R_15_Agg_customer_ID_LastDiffFirst','R_2_Agg_customer_ID_first','R_23_Agg_customer_ID_mean',
         'S_18_Agg_customer_ID_first','R_17_Agg_customer_ID_last','R_17_Agg_customer_ID_max','R_18_Agg_customer_ID_max',
         'D_89_Agg_customer_ID_min','S_18_Agg_customer_ID_max','R_17_Agg_customer_ID_std','D_89_Agg_customer_ID_max',
         'R_20_Agg_customer_ID_min','R_8_Agg_customer_ID_max','D_88_Agg_customer_ID_max','B_31_Agg_customer_ID_LastDiffFirst',
         'D_135_Agg_customer_ID_first','R_15_Agg_customer_ID_first','R_13_Agg_customer_ID_max','D_96_Agg_customer_ID_first',
         'R_7_Agg_customer_ID_min','R_23_Agg_customer_ID_LastDiffMean','R_8_Agg_customer_ID_first','B_41_Agg_customer_ID_min',
         'B_31_Agg_customer_ID_mean','D_127_Agg_customer_ID_LastDiffFirst','R_22_Agg_customer_ID_max','R_21_Agg_customer_ID_first',
         'D_111_Agg_customer_ID_min','D_92_Agg_customer_ID_first','D_109_Agg_customer_ID_LastDiffMean','D_96_Agg_customer_ID_min',
         'R_10_Agg_customer_ID_min','D_140_Agg_customer_ID_max','R_8_Agg_customer_ID_min','D_135_Agg_customer_ID_min',
         'B_31_Agg_customer_ID_first','D_137_Agg_customer_ID_min','R_20_Agg_customer_ID_max','D_103_Agg_customer_ID_min',
         'R_23_Agg_customer_ID_max','S_18_Agg_customer_ID_min','R_13_Agg_customer_ID_first','D_86_Agg_customer_ID_first',
         'D_137_Agg_customer_ID_last','D_93_Agg_customer_ID_first','R_25_Agg_customer_ID_first','D_93_Agg_customer_ID_min',
         'R_18_Agg_customer_ID_mean','R_22_Agg_customer_ID_last','D_137_Agg_customer_ID_first','S_18_Agg_customer_ID_last',
         'D_93_Agg_customer_ID_max','S_6_Agg_customer_ID_first','R_23_Agg_customer_ID_std','D_92_Agg_customer_ID_max',
         'D_109_Agg_customer_ID_LastDiffFirst','B_33_Agg_customer_ID_first','D_111_Agg_customer_ID_first','D_108_Agg_customer_ID_first',
         'R_28_Agg_customer_ID_min','D_109_Agg_customer_ID_last','R_28_Agg_customer_ID_max','D_127_Agg_customer_ID_first',
         'D_86_Agg_customer_ID_min','R_5_Agg_customer_ID_first','D_94_Agg_customer_ID_first','D_93_Agg_customer_ID_last',
         'D_89_Agg_customer_ID_first','D_94_Agg_customer_ID_last','R_15_Agg_customer_ID_min','R_18_Agg_customer_ID_LastDiffFirst',
         'D_108_Agg_customer_ID_min','R_28_Agg_customer_ID_last','R_23_Agg_customer_ID_first','D_87_Agg_customer_ID_first',
         'R_21_Agg_customer_ID_min','R_23_Agg_customer_ID_LastDiffFirst','R_25_Agg_customer_ID_min','R_17_Agg_customer_ID_mean',
         'B_31_Agg_customer_ID_max','R_13_Agg_customer_ID_min','D_109_Agg_customer_ID_min','D_109_Agg_customer_ID_max',
         'D_109_Agg_customer_ID_first','R_4_Agg_customer_ID_min','D_127_Agg_customer_ID_max','D_127_Agg_customer_ID_min',
         'R_15_Agg_customer_ID_last','R_2_Agg_customer_ID_min','R_22_Agg_customer_ID_min','R_23_Agg_customer_ID_last',
         'D_92_Agg_customer_ID_last','R_5_Agg_customer_ID_min','D_87_Agg_customer_ID_min','D_87_Agg_customer_ID_max',
         'D_87_Agg_customer_ID_last','D_87_Agg_customer_ID_LastDiffFirst','D_94_Agg_customer_ID_min','R_24_Agg_customer_ID_min',
         'R_24_Agg_customer_ID_first','D_94_Agg_customer_ID_LastDiffFirst','D_139_Agg_customer_ID_min','R_4_Agg_customer_ID_first',
         'R_17_Agg_customer_ID_min','R_18_Agg_customer_ID_min','R_18_Agg_customer_ID_first','R_18_Agg_customer_ID_last']

#--------------------------------------------------------------------------------------------------------------------------------

print('Training Data Shape Before: ',train_df_new.shape)
train_df_new.drop(feats,inplace=True,axis=1)
print('Training Data Shape After: ',train_df_new.shape)
train_df = train_df_new
del feats


# # Validation

# In[ ]:


# Use the aggregated categorical features instead of the original ones
cat_feats = ["B_38","D_114","D_117","D_120","D_63","D_64","D_66",'D_39_bins','P_2_bins','B_1_bins']
cat_feats0 = [f'{Feature}_Agg_customer_ID_last' for Feature in cat_feats]
cat_feats1 = [f'{Feature}_Agg_customer_ID_first' for Feature in cat_feats]
cat_feats = cat_feats0 + cat_feats1
del cat_feats0, cat_feats1


# In[ ]:


importances = []
oof = []
TRAIN_SUBSAMPLE = 1.0  #Reduce if the notebook crashed due to low memory space
FOLDS = 5
gc.collect()

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
for fold,(train_idx, valid_idx) in enumerate(skf.split( train_df, train_df.Target )):
    
    # Train with subsample of train fold data
    if TRAIN_SUBSAMPLE<1.0:
        np.random.seed(42)
        train_idx = np.random.choice(train_idx,int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)
    
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
    print('#'*25)
    
    # TRAIN, VALID, TEST FOR FOLD K
    X_train = train_df.loc[train_idx, train_df.drop('Target',axis=1).columns]
    y_train = train_df.loc[train_idx, 'Target']
    X_valid = train_df.loc[valid_idx, train_df.drop('Target',axis=1).columns]
    y_valid = train_df.loc[valid_idx, 'Target']

    #Define the model with early stopping
    cb_params = {'loss_function': 'Logloss','iterations': 50000,
                 'task_type':'GPU', 'learning_rate': 0.01, 'depth': 4,
                 'verbose': 0, 'od_type': 'Iter', 'od_wait': 500}
    model = CatBoostClassifier(**cb_params, random_state=42)
    model.fit(X_train.to_pandas(), y_train.to_pandas(), eval_set=[(X_valid.to_pandas(), y_valid.to_pandas())], verbose=100)
    model.save_model(f'CB_fold{fold}')
    del X_train, y_train
    gc.collect()
    
    # Infer OOF fold K 
    oof_preds = cudf.DataFrame(model.predict_proba(X_valid.to_pandas()))
    acc = amex_metric_mod(y_valid.to_pandas().values, oof_preds[1].to_pandas().values)
    print('Kaggle Metric =',acc[0],'\n')    
    del X_valid, y_valid
    gc.collect()
    
    #Store the features importance of fold K
    df = cudf.DataFrame({'feature':model.feature_names_,f'importance_{fold}':model.feature_importances_})
    importances.append(df)
    del df, model
    gc.collect()
    
    # Save OOF to calculate the overall score
    df = train_df.loc[valid_idx, ['customer_ID','Target'] ].copy()
    df['oof_pred'] = oof_preds[1].to_array()
    oof.append(df)
    del df
    gc.collect()
    
#--------------------------------------------------------------------------------------------------------------------------------
print('#'*25)
oof = cudf.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
acc = amex_metric_mod(oof.Target.values, oof.oof_pred.values)
print('OVERALL CV Kaggle Metric =',acc[0],'\n')

#CV Results, 100% Training Sample, about 75 mins required to finish with GPU:

# Fold 1:
# Kaggle Metric = 0.7974942621446841  
# Fold 2:
# Kaggle Metric = 0.796183175665393
# Fold 3:
# Kaggle Metric = 0.792862926329444 
# Fold 4:
# Kaggle Metric = 0.7902024309752875  
# Fold 5:
# Kaggle Metric = 0.7990614504769618 

# OVERALL CV Kaggle Metric = 0.7952669259876113


# ## Calculate The Features Importance

# In[ ]:


df = importances[0].copy()
for k in range(1,FOLDS): df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)
df['Index'] = range(0,len(df.index))
NUM_FEATURES = 30
ShowImp(df.feature,df.importance,NUM_FEATURES)
df.to_csv('Features_Importance.csv')
del df


# # --------------------------------------------------------------------------------------------------------------

# # Test Data Preparation

# In[ ]:


print('Reading test data...')
TEST_PATH = '../input/amex-data-integer-dtypes-parquet-format/test.parquet'
test_df = read_file(path = TEST_PATH)


# ## Temporary Dataset

# In[ ]:


test_df_new = cudf.DataFrame(test_df['customer_ID'].drop_duplicates(),columns=['customer_ID'])


# ## Polar Coordinates of P_2 and The Other Features

# In[ ]:


def Polar(X,y, a = 9.9, b = 9.9): # a and b represnt the center
    r = np.sqrt((X-a)**2 + (y-b)**2)
    phi = np.arctan2((y-a), (X-b))
    return r, phi

feats = ['B_18','B_2','B_33','D_62','D_77','D_47','P_3','D_45','D_51','R_27','S_25','D_112',
         'D_121','D_128','D_52','D_115','D_114','D_127','D_68','D_118','D_119','D_122','D_54',
         'D_129','D_134','S_8','D_92','R_12','D_91','D_76','D_56','D_117','B_42','S_6','D_73',
         'S_13','D_142','D_126','D_140','R_16','B_41','R_21','B_32','B_24','D_133','R_17','D_120',
         'D_46','D_145','D_139','D_143','D_138','D_136','R_6','D_135','B_25','D_137','R_26','D_141',
         'R_13','D_79','D_39','D_89','D_113','R_20','R_15','S_15','R_8','D_130','D_131','D_64','D_59',
         'R_24','R_5','B_28','D_88','R_10','P_4','D_78','D_43','R_3','R_9','D_41','D_70','R_4','D_81',
         'D_84','S_3','B_17','D_72','B_11','B_30','S_7','B_37','B_1','B_22','B_8','R_2','B_19','D_53',
         'B_4','D_61','B_38','B_3','B_20','R_1','D_42','B_16','B_23','B_7','D_74','D_44','D_75','D_58',
         'B_9','D_55','D_48']

for feat in tqdm(feats):
    test_df[f'P_2_{feat}_R'], test_df[f'P_2_{feat}_Phi'] = Polar(test_df["P_2"].fillna(127).to_array(),test_df[feat].fillna(127).to_array())
    
    test_df_new[f'P_2_{feat}_R_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'P_2_{feat}_R'].mean())
    test_df_new[f'P_2_{feat}_R_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'P_2_{feat}_R'].last())
    test_df_new[f'P_2_{feat}_Phi_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'P_2_{feat}_Phi'].mean())
    test_df_new[f'P_2_{feat}_Phi_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'P_2_{feat}_Phi'].last())
    
    test_df.drop([f'P_2_{feat}_R',f'P_2_{feat}_Phi'],inplace=True,axis=1)
    test_df_new = reduce_memory_usage(test_df_new)


# ## Adding Bins for P_2, D_39, B_1

# In[ ]:


TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'   # Use the same boundaries used in training set
train_df = read_file(path = TRAIN_PATH, usecols=['customer_ID','S_2','P_2','B_1','D_39']) # Needed later for Target encoding for test set
for feat in ['P_2','B_1']:
    perc = {}
    for i, per in enumerate([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        percentile[i] = train_df[feat].quantile(per)
    test_df[f'{feat}_bins'] = pd.cut(test_df[feat],[percentile[x] for x in range(0,11)], labels=False).fillna(-127).astype(int)
    train_df[f'{feat}_bins'] = pd.cut(train_df[feat],[percentile[x] for x in range(0,11)], labels=False).fillna(-127).astype(int)

train_df['D_39_bins'] = pd.cut(train_df['D_39'],[0,31,61,91,121,151,181,np.inf], labels=False).fillna(-127).astype(int)   
test_df['D_39_bins'] = pd.cut(test_df['D_39'],[0,31,61,91,121,151,181,np.inf], labels=False).fillna(-127).astype(int)
test_df = reduce_memory_usage(test_df)
del perc


# ## Features Groups (Have similar mean or high correlation with each other)

# In[ ]:


feats_groups = []
feats_groups.append(['D_111','D_108','D_87','D_136','D_138','D_135','D_137'])
feats_groups.append(['D_136','D_138','D_135','D_137'])
feats_groups.append(['R_28','R_23','R_18'])
feats_groups.append(['R_25','R_22'])
feats_groups.append(['R_17','R_4','R_19','R_21','R_15','R_13','R_24'])
feats_groups.append(['R_10','R_5','R_6'])
feats_groups.append(['B_14','B_13'])
feats_groups.append(['B_11','B_42'])
feats_groups.append(['B_11','B_25'])
feats_groups.append(['B_37','B_1'])
feats_groups.append(['B_37','B_1','B_11'])
feats_groups.append(['D_139','D_143'])
feats_groups.append(['D_43','D_69'])
feats_groups.append(['D_102','B_9','D_62'])
feats_groups.append(['D_56','B_40'])
feats_groups.append(['S_3','S_7'])
feats_groups.append(['S_12','S_6'])
feats_groups.append(['D_115','D_119','D_118'])
feats_groups.append(['D_47','D_129','D_49'])
feats_groups.append(['P_3','P_2'])
feats_groups.append(['D_63','D_75'])
feats_groups.append(['S_8','S_13'])
feats_groups.append(['B_4','B_19'])
feats_groups.append(['B_2','B_18'])
feats_groups.append(['B_2','B_33'])
feats_groups.append(['D_75','D_58','D_74'])
feats_groups.append(['D_48','P_2','D_55','D_61'])
feats_groups.append(['D_75','D_55','D_58'])
feats_groups.append(['D_48','P_2'])
feats_groups.append(['D_48','D_61','D_55'])
feats_groups.append(['D_48','D_58'])
feats_groups.append(['B_9','D_75'])
feats_groups.append(['D_69','D_65'])
feats_groups.append(['D_106','D_39'])
feats_groups.append(['B_10','B_16'])


# In[ ]:


for i,k in enumerate(feats_groups):
    test_df[f'Feats_Group{i}_Mean'] = test_df[k].mean(axis=1)
    test_df_new[f'Feats_Group{i}_Mean_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'Feats_Group{i}_Mean'].mean())
    test_df_new[f'Feats_Group{i}_Mean_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'Feats_Group{i}_Mean'].last())
    test_df[f'Feats_Group{i}_Sum'] = test_df[k].sum(axis=1)
    test_df_new[f'Feats_Group{i}_Sum_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'Feats_Group{i}_Sum'].mean())
    test_df_new[f'Feats_Group{i}_Sum_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'Feats_Group{i}_Sum'].last())
    test_df.drop([f'Feats_Group{i}_Mean',f'Feats_Group{i}_Sum'],inplace=True,axis=1)
    test_df_new = reduce_memory_usage(test_df_new)


# ## Add Some Interactions Between D_106 and D_39

# In[ ]:


test_df['D_106_D_39_sum'] = test_df['D_106'] + test_df['D_39']
test_df['D_106_D_39_mean'] = test_df['D_106'] + test_df['D_39']
test_df['D_106_D_39_ratio'] = test_df['D_106'] / test_df['D_39']


# ## Target Encoding

# In[ ]:


#Target Encoding for Testing Data
class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self,colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col


    def fit(self, X, y=None):
        return self


    def transform(self,X,test_df):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold)



        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X,X[self.targetName]):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X[col_mean_name][val_ind] = X_val[self.colnames].fillna(-127).map(X_tr.groupby(self.colnames)[self.targetName].mean()).to_array()

        X[col_mean_name].fillna(mean_of_target, inplace = True)
        
        test_df[col_mean_name] = test_df[self.colnames].fillna(-127).map(X.groupby(self.colnames)[col_mean_name].mean())
        
        if self.verbosity:

            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            

        return X, test_df


# In[ ]:


# Need Training Data to add Target encoding for Testing data
TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
cat_feats = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",'customer_ID','S_2']
train_df1 = read_file(path = TRAIN_PATH, usecols = cat_feats)
# Add Labels
labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
labels['customer_ID'] = labels['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
train_df1['Target'] = train_df1['customer_ID'].map(labels.groupby('customer_ID')['target'].last()).astype(np.int8)
# Reduce Memory Usage
train_df1 = reduce_memory_usage(train_df1)
train_df1.drop(['customer_ID','S_2'],inplace=True,axis=1)
train_df = pd.concat([train_df,train_df1],axis=1)
del labels


# In[ ]:


# Add the Target encoding for testing data
feats = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",'D_39_bins','P_2_bins']
for feat in tqdm(feats):
    tar_enc = KFoldTargetEncoderTrain(feat,'Target',n_fold=5)
    tar_enc.fit(train_df)
    _, test_df = tar_enc.transform(train_df,test_df)
test_df = reduce_memory_usage(test_df)
del train_df


# ## Define The Features Lists

# In[ ]:


all_cols = [c for c in list(test_df.columns) if c not in ['customer_ID','S_2']]
cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",
               'D_39_bins','P_2_bins','B_1_bins']
num_features = [col for col in all_cols if col not in cat_features]
cat_features = ["B_38","D_114","D_117","D_120","D_63","D_64","D_66",
                'D_39_bins','P_2_bins','B_1_bins']


# ## Adding Aggergations for Some Lagged Features

# In[ ]:


feats = ['B_2','D_39','B_1','P_2','D_106_D_39_sum','D_106_D_39_mean','D_106_D_39_ratio']

for Feature in tqdm(feats):
    for window in [1,2,3]:
        test_df[f'{Feature}_lag_{window}'] = test_df[Feature].shift(window)
        
        test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'{Feature}_lag_{window}'].last())
        test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_current'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'{Feature}'].last())
        test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_CurrentDiffLast'] = test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_current'] - test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last']
        test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[f'{Feature}_lag_{window}'].mean())
        test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_LastDiffMean'] = test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_last'] - test_df_new[f'{Feature}_lag_{window}_Agg_customer_ID_mean']
        
        test_df_new = reduce_memory_usage(test_df_new)
        test_df.drop(f'{Feature}_lag_{window}',inplace=True,axis=1)
        test_df_new.drop([f'{Feature}_lag_{window}_Agg_customer_ID_mean',f'{Feature}_lag_{window}_Agg_customer_ID_current'],inplace=True,axis=1)


# ## Adding Aggregations for All The Other Features

# In[ ]:


#Numerical Features
for Feature in tqdm(num_features):
                test_df_new[f'{Feature}_Agg_customer_ID_mean'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].mean())
                test_df_new[f'{Feature}_Agg_customer_ID_std'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].std())  
                test_df_new[f'{Feature}_Agg_customer_ID_min'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].min())  
                test_df_new[f'{Feature}_Agg_customer_ID_max'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].max())  
                test_df_new[f'{Feature}_Agg_customer_ID_first'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].first())
                test_df_new[f'{Feature}_Agg_customer_ID_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].last())
                test_df_new[f'{Feature}_Agg_customer_ID_LastDiffMean'] = test_df_new[f'{Feature}_Agg_customer_ID_last'] - test_df_new[f'{Feature}_Agg_customer_ID_mean']
                test_df_new[f'{Feature}_Agg_customer_ID_LastDiffFirst'] = test_df_new[f'{Feature}_Agg_customer_ID_last'] - test_df_new[f'{Feature}_Agg_customer_ID_first']
                
                test_df.drop(Feature,inplace=True,axis=1)
                test_df_new = reduce_memory_usage(test_df_new)
#-------------------------------------------------------------------------------------------------------------------------------------------------
#Categorical Features
test_df_new['D_39_bins_Agg_customer_ID_nunique'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')['D_39_bins'].nunique())  
for Feature in tqdm(cat_features):
                test_df_new[f'{Feature}_Agg_customer_ID_last'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].last())
                test_df_new[f'{Feature}_Agg_customer_ID_first'] = test_df_new['customer_ID'].map(test_df.groupby('customer_ID')[Feature].first())
                
                test_df.drop(Feature,inplace=True,axis=1)
                test_df_new = reduce_memory_usage(test_df_new)

print('Finished!!!')
del test_df


# ## Drop Some Useless Features

# In[ ]:


# 1
print('Testing Data Shape Before: ',test_df_new.shape)
feat_df = pd.read_csv('../input/features-importance/Features.csv')
feats = feat_df[feat_df['importance'] == 0.0].feature.to_array()
test_df_new.drop(feats,inplace=True,axis=1)
print('Testing Data Shape After: ',test_df_new.shape)
del feat_df, feats


# In[ ]:


# 2
feats = ['S_20_Agg_customer_ID_min','R_20_Agg_customer_ID_first','D_139_Agg_customer_ID_mean','D_138_Agg_customer_ID_min',
         'D_137_Agg_customer_ID_LastDiffFirst','R_24_Agg_customer_ID_LastDiffFirst','D_123_Agg_customer_ID_last',
         'D_87_Agg_customer_ID_mean','R_17_Agg_customer_ID_first','R_25_Agg_customer_ID_mean','D_86_Agg_customer_ID_LastDiffFirst',
         'B_33_Agg_customer_ID_max','D_137_Agg_customer_ID_max','D_81_Agg_customer_ID_min','D_108_Agg_customer_ID_last',
         'S_18_Agg_customer_ID_LastDiffFirst','D_93_Agg_customer_ID_LastDiffFirst','D_96_Agg_customer_ID_LastDiffFirst',
         'R_28_Agg_customer_ID_LastDiffFirst','R_13_Agg_customer_ID_std','D_139_Agg_customer_ID_first','D_88_Agg_customer_ID_first',
         'D_123_Agg_customer_ID_min','B_32_Agg_customer_ID_max','B_31_Agg_customer_ID_std','R_24_Agg_customer_ID_last',
         'R_13_Agg_customer_ID_mean','R_15_Agg_customer_ID_LastDiffFirst','R_2_Agg_customer_ID_first','R_23_Agg_customer_ID_mean',
         'S_18_Agg_customer_ID_first','R_17_Agg_customer_ID_last','R_17_Agg_customer_ID_max','R_18_Agg_customer_ID_max',
         'D_89_Agg_customer_ID_min','S_18_Agg_customer_ID_max','R_17_Agg_customer_ID_std','D_89_Agg_customer_ID_max',
         'R_20_Agg_customer_ID_min','R_8_Agg_customer_ID_max','D_88_Agg_customer_ID_max','B_31_Agg_customer_ID_LastDiffFirst',
         'D_135_Agg_customer_ID_first','R_15_Agg_customer_ID_first','R_13_Agg_customer_ID_max','D_96_Agg_customer_ID_first',
         'R_7_Agg_customer_ID_min','R_23_Agg_customer_ID_LastDiffMean','R_8_Agg_customer_ID_first','B_41_Agg_customer_ID_min',
         'B_31_Agg_customer_ID_mean','D_127_Agg_customer_ID_LastDiffFirst','R_22_Agg_customer_ID_max','R_21_Agg_customer_ID_first',
         'D_111_Agg_customer_ID_min','D_92_Agg_customer_ID_first','D_109_Agg_customer_ID_LastDiffMean','D_96_Agg_customer_ID_min',
         'R_10_Agg_customer_ID_min','D_140_Agg_customer_ID_max','R_8_Agg_customer_ID_min','D_135_Agg_customer_ID_min',
         'B_31_Agg_customer_ID_first','D_137_Agg_customer_ID_min','R_20_Agg_customer_ID_max','D_103_Agg_customer_ID_min',
         'R_23_Agg_customer_ID_max','S_18_Agg_customer_ID_min','R_13_Agg_customer_ID_first','D_86_Agg_customer_ID_first',
         'D_137_Agg_customer_ID_last','D_93_Agg_customer_ID_first','R_25_Agg_customer_ID_first','D_93_Agg_customer_ID_min',
         'R_18_Agg_customer_ID_mean','R_22_Agg_customer_ID_last','D_137_Agg_customer_ID_first','S_18_Agg_customer_ID_last',
         'D_93_Agg_customer_ID_max','S_6_Agg_customer_ID_first','R_23_Agg_customer_ID_std','D_92_Agg_customer_ID_max',
         'D_109_Agg_customer_ID_LastDiffFirst','B_33_Agg_customer_ID_first','D_111_Agg_customer_ID_first','D_108_Agg_customer_ID_first',
         'R_28_Agg_customer_ID_min','D_109_Agg_customer_ID_last','R_28_Agg_customer_ID_max','D_127_Agg_customer_ID_first',
         'D_86_Agg_customer_ID_min','R_5_Agg_customer_ID_first','D_94_Agg_customer_ID_first','D_93_Agg_customer_ID_last',
         'D_89_Agg_customer_ID_first','D_94_Agg_customer_ID_last','R_15_Agg_customer_ID_min','R_18_Agg_customer_ID_LastDiffFirst',
         'D_108_Agg_customer_ID_min','R_28_Agg_customer_ID_last','R_23_Agg_customer_ID_first','D_87_Agg_customer_ID_first',
         'R_21_Agg_customer_ID_min','R_23_Agg_customer_ID_LastDiffFirst','R_25_Agg_customer_ID_min','R_17_Agg_customer_ID_mean',
         'B_31_Agg_customer_ID_max','R_13_Agg_customer_ID_min','D_109_Agg_customer_ID_min','D_109_Agg_customer_ID_max',
         'D_109_Agg_customer_ID_first','R_4_Agg_customer_ID_min','D_127_Agg_customer_ID_max','D_127_Agg_customer_ID_min',
         'R_15_Agg_customer_ID_last','R_2_Agg_customer_ID_min','R_22_Agg_customer_ID_min','R_23_Agg_customer_ID_last',
         'D_92_Agg_customer_ID_last','R_5_Agg_customer_ID_min','D_87_Agg_customer_ID_min','D_87_Agg_customer_ID_max',
         'D_87_Agg_customer_ID_last','D_87_Agg_customer_ID_LastDiffFirst','D_94_Agg_customer_ID_min','R_24_Agg_customer_ID_min',
         'R_24_Agg_customer_ID_first','D_94_Agg_customer_ID_LastDiffFirst','D_139_Agg_customer_ID_min','R_4_Agg_customer_ID_first',
         'R_17_Agg_customer_ID_min','R_18_Agg_customer_ID_min','R_18_Agg_customer_ID_first','R_18_Agg_customer_ID_last']

#--------------------------------------------------------------------------------------------------------------------------------

print('Testing Data Shape Before: ',test_df_new.shape)
test_df_new.drop(feats,inplace=True,axis=1)
print('Testing Data Shape After: ',test_df_new.shape)
test_df = test_df_new
del feats


# # Inference

# In[ ]:


customers = test_df[['customer_ID']].drop_duplicates().sort_index().to_array().flatten()
FOLDS = 5
gc.collect()


# ## Predicting The Test Set in Two Parts Due to Memory Limits

# In[ ]:


print('Predicting Part 1...')
model = CatBoostClassifier()
print(f'Fold: 0')
model.load_model(f'./CB_fold0')
preds1 = model.predict_proba(test_df[:int(len(test_df.index)/2)].to_pandas())[:,1]
for f in range(1,FOLDS):
    print(f'Fold: {f}')
    model.load_model(f'./CB_fold{f}')
    preds1 += model.predict_proba(test_df[:int(len(test_df.index)/2)].to_pandas())[:,1]
preds1 /= FOLDS
print(f'Done Part 1')
    
# CLEAN MEMORY
del model
gc.collect()


# In[ ]:


# Drop the First part of The Test Set
test_df = test_df[int(len(test_df.index)/2):]


# In[ ]:


print('Predicting Part 2...')
model = CatBoostClassifier()
print(f'Fold: 0')
model.load_model(f'./CB_fold0')
preds2 = model.predict_proba(test_df.to_pandas())[:,1]
for f in range(1,FOLDS):
    print(f'Fold: {f}')
    model.load_model(f'./CB_fold{f}')
    preds2 += model.predict_proba(test_df.to_pandas())[:,1]
preds2 /= FOLDS
print(f'Done Part 2')
    
# CLEAN MEMORY
del model, test_df
gc.collect()


# ## Make The Submission File
# 

# In[ ]:


test_preds = np.concatenate([preds1,preds2])
Predictions = pd.DataFrame(index=customers,data={'prediction':test_preds})
sub = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')[['customer_ID']]
sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.set_index('customer_ID_hash')
sub = sub.merge(Predictions[['prediction']], left_index=True, right_index=True, how='left')
sub = sub.reset_index(drop=True)
del Predictions

# DISPLAY PREDICTIONS
sub.to_csv('AMEX_FinalSubmission.csv',index=False)
print('Submission file shape is', sub.shape)
plt.hist(sub['prediction'].to_pandas(),bins=150)
sub.head()

