#!/usr/bin/env python
# coding: utf-8

# ## In this noteebook I have created bio-informative features based on the protein sequence given in the train dataset.

# In[1]:


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder

COMMON_PATH = '/kaggle/input/novozymes-enzyme-stability-prediction'


# ### Loading Data Files

# In[2]:


train = pd.read_csv(f'{COMMON_PATH}/train.csv')
test = pd.read_csv(f'{COMMON_PATH}/test.csv')


# ### Reading PDB File using BioPython package
# 
# The structure object obtained from the parser can be used to get the secondary structure, b-factor and other details regarding the atoms.

# In[3]:


from Bio.PDB.PDBParser import PDBParser

pdb_parser = PDBParser()
structure = pdb_parser.get_structure("PHA-L", "/kaggle/input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb")


# ## Feature Engineering using BioPython

# In[4]:


train['protein_seq_len'] = train['protein_sequence'].str.len()
test['protein_seq_len'] = test['protein_sequence'].str.len()


# In[5]:


def get_protein_analysis_obj(string_seq):
    return ProteinAnalysis(string_seq)


# In[6]:


train['protein_analysis_obj'] = train['protein_sequence'].apply(lambda x:get_protein_analysis_obj(x))
train['protein_sequence_obj'] = train['protein_analysis_obj'].apply(lambda x: x.sequence)

test['protein_analysis_obj'] = test['protein_sequence'].apply(lambda x:get_protein_analysis_obj(x))
test['protein_sequence_obj'] = test['protein_analysis_obj'].apply(lambda x: x.sequence)


# In[7]:


train['protein_molecular_weight'] = train['protein_analysis_obj'].apply(lambda x:x.molecular_weight())
test['protein_molecular_weight'] = test['protein_analysis_obj'].apply(lambda x:x.molecular_weight())


# ### Aromaticity is the relative frequency of (phenylalanine, tyrosine, tryptophan) amino acids in the protein structure. It is a crucial component as it helps in telling the stability of a protein and how well it binds with other proteins. These three amino acids are also known as aromatic amino acids.

# In[8]:


train['protein_aromaticity'] = train['protein_analysis_obj'].apply(lambda x:x.aromaticity())
test['protein_aromaticity'] = test['protein_analysis_obj'].apply(lambda x:x.aromaticity())


# In[9]:


train['protein_charge_at_ph7'] = train['protein_analysis_obj'].apply(lambda x: round(x.charge_at_pH(pH=7),2))
test['protein_charge_at_ph7'] = test['protein_analysis_obj'].apply(lambda x: round(x.charge_at_pH(pH=7),2))


# ### There are a total of 22 amino acids in the genetic code. Out of which 20 are standard and remaining 2 can be incorporated by special translation mechanisms. 

# In[10]:


amino_acids_codes = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'] + ['B','Z']

for code in amino_acids_codes:
    train[f'amino_count_{code}'] = train['protein_sequence_obj'].apply(lambda x:x.count(code))
    test[f'amino_count_{code}'] = test['protein_sequence_obj'].apply(lambda x:x.count(code))


# #### GRAVY (grand average of hydropathy) value for the protein sequences you enter. The GRAVY value is calculated by adding the hydropathy value for each residue and dividing by the length of the sequence. A higher value is more hydrophobic. A lower value is more hydrophilic.

# In[11]:


train['protein_gravy_val'] = train['protein_analysis_obj'].apply(lambda x: x.gravy())
test['protein_gravy_val'] = test['protein_analysis_obj'].apply(lambda x: x.gravy())


# #### Isoelectric point is the PH at which charge on the protein becomes zero. It has relation with the melting point of the protein which affects the thermal stability.

# In[12]:


train['protein_isoelectric_point'] = train['protein_analysis_obj'].apply(lambda x: x.isoelectric_point())
test['protein_isoelectric_point'] = test['protein_analysis_obj'].apply(lambda x: x.isoelectric_point())


# In[13]:


# Limiting data upto sequence length 221
train = train.loc[train['protein_seq_len'] <= 221]
train = train.reset_index(drop=True)


# In[14]:


def gen_sequence_features(data):
    protein_seq = [list(seq) for seq in data['protein_sequence'].values.tolist()]
    seq_df = pd.DataFrame(protein_seq)
    seq_df.columns = [f'protein_seq_idx_char_{i+1}' for i in range(221)]
    return pd.concat([data, seq_df], axis=1)


# In[15]:


train = gen_sequence_features(train)
test = gen_sequence_features(test)


# In[16]:


ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=23)
ordinal_enc_cols = [col for col in train.columns if 'protein_seq_idx_char_' in col]
train[ordinal_enc_cols] = ord_enc.fit_transform(train[ordinal_enc_cols])
test[ordinal_enc_cols] = ord_enc.transform(test[ordinal_enc_cols])


# In[17]:


cols_to_ignore = ['seq_id','protein_sequence','data_source','protein_analysis_obj','protein_sequence_obj']
X = train.loc[:,~train.columns.isin(cols_to_ignore +['tm'])]
y = train['tm']

test = test.loc[:,~test.columns.isin(cols_to_ignore)]


# In[18]:


kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
score_list_lgb = []
test_preds_lgb = []
fold = 1

for train_index, test_index in kf.split(X, y):
    
    ## Splitting the data
    X_train , X_val = X.iloc[train_index], X.iloc[test_index]  
    Y_train, Y_val = y.iloc[train_index], y.iloc[test_index]    
    
    print("X_train shape is :", X_train.shape, "X_val shape is", X_val.shape)
    y_pred_list = []
    
    model_lgb = LGBMRegressor(n_estimators = 100, max_depth=7, metric='rmse', num_leaves=80)

    model = model_lgb.fit(X_train, Y_train)
    model_pred = model_lgb.predict(X_val)
    
    score = spearmanr(Y_val, model_pred)[0]
    print('Fold ', str(fold), ' result is:', score, '\n')
    score_list_lgb.append(score)

    test_preds_lgb.append(model_lgb.predict(test))
    fold +=1


# In[19]:


test_preds_lgb = pd.DataFrame(test_preds_lgb).mean(axis = 0)
submission = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/sample_submission.csv')
submission['tm'] = test_preds_lgb
submission.to_csv('submission.csv', index=False)


# In[ ]:




