#!/usr/bin/env python
# coding: utf-8

# # Amino Acid Properties For Feature Engineering

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from itertools import product

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

Aminoacids = ['A', 'R', 'N','D','C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
Blocks = []

maxSize = 4
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Aminoacids, repeat = k)])


# In[2]:


data = pd.read_csv('../input/novozymes-enzyme-stability-prediction/train.csv',index_col="seq_id")
dataMod = pd.read_csv('../input/novozymes-enzyme-stability-prediction/train_updates_20220929.csv',index_col="seq_id")

#from https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356251 Petr1zi0's answer

all_features_nan = dataMod.isnull().all("columns")
data = data.drop(index=dataMod[all_features_nan].index)

swap_ph_tm_indices = dataMod[~all_features_nan].index
data.loc[swap_ph_tm_indices, ["pH", "tm"]] = dataMod.loc[swap_ph_tm_indices, ["pH", "tm"]]

data['seqsize'] = [len(val) for val in data['protein_sequence']]

data = data.fillna(data['pH'].max())


# # Feature engineering 
# 
# Mean properties are calculated by the dot product between the properties matrix and the normalized frequency of each amino acid. This results in a matrix of size (samples, properties). Aminoacid properties were retrieved from the AA index database

# In[3]:


def SplitString(String,ChunkSize):
    '''
    Split a string ChunkSize fragments using a sliding windiow

    Parameters
    ----------
    String : string
        String to be splitted.
    ChunkSize : int
        Size of the fragment taken from the string .

    Returns
    -------
    Splitted : list
        Fragments of the string.

    '''
    try:
        localString=str(String.seq)
    except AttributeError:
        localString=str(String)
      
    if ChunkSize==1:
        Splitted=[val for val in localString]
    
    else:
        nCharacters=len(String)
        Splitted=[localString[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def CountUniqueElements(UniqueElements,String,Processed=False):
    '''
    Calculates the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    Parameters
    ----------
    UniqueElements : array,list
        Elements to be analized.
    String : strting
        Sequence data.
    Processed : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    localCounter : array
        Normalized frequency of each unique fragment.
    '''
    
    nUnique = len(UniqueElements)
    localCounter = [0 for k in range(nUnique)]
    
    if Processed:
        ProcessedString = String
    else:
        ProcessedString = SplitString(String,len(UniqueElements[0]))
        
    nSeq = len(ProcessedString)
    UniqueDictionary = {}
    
    for k,val in enumerate(UniqueElements):
        UniqueDictionary[val] = k
    
    for val in ProcessedString:
        
        if val in UniqueElements:
            
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
            
    localCounter=[val/nSeq for val in localCounter]
    
    return localCounter


# In[4]:


dta0 = np.array([CountUniqueElements(Aminoacids,val) for val in data['protein_sequence']])
props = pd.read_csv('/kaggle/input/amino-acid-properties/aaindex1.csv')
feats = np.array(props[Aminoacids])
finalfeats = dta0.dot(feats.T)
seqFeatures = pd.DataFrame(finalfeats,columns=props['description'])
seqFeatures['seq_id'] = data.index
seqFeatures = seqFeatures.set_index('seq_id')

data = pd.concat([data,seqFeatures],axis=1)
data = data.fillna(0)


# In[5]:


Xtrain, Xtest, _, _ = train_test_split(data.index, np.arange(len(data.index)), test_size=0.1, random_state=42)
cols = list(seqFeatures)+['pH','seqsize']


# # Kfold cross validation

# In[6]:


kf = KFold(n_splits=10)
perf = []
Models = []
Scalers = []

for train_index, test_index in kf.split(Xtrain):
    
    localXtrain, localXtest = Xtrain[train_index], Xtrain[test_index]
    scaler = MinMaxScaler()
    scaler.fit(np.array(data[cols].loc[localXtrain]))
    Scalers.append(scaler)
    
    trainData = scaler.transform(np.array(data[cols].loc[localXtrain]))
    testData = scaler.transform(np.array(data[cols].loc[localXtest]))
    
    reg = HistGradientBoostingRegressor(max_iter=2500,learning_rate=0.01,random_state=0)
    reg.fit(trainData, data['tm'].loc[localXtrain])
    Models.append(reg)
    
    trainPreds = reg.predict(testData)
    rho, pval = stats.spearmanr(np.array(data['tm'].loc[localXtest]),trainPreds)
    perf.append(rho)
    print(rho)


# In[7]:


bestLoc = np.argmax(perf)
bestModel = Models[bestLoc]
bestScaler = Scalers[bestLoc]


# # Predictions

# In[8]:


subdata = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/test.csv',index_col="seq_id")
subdata['seqsize'] = [len(val) for val in subdata['protein_sequence']]
subdata = subdata.fillna(subdata['pH'].max())

sdta0 = np.array([CountUniqueElements(Aminoacids,val) for val in subdata['protein_sequence']])
sfinalfeats = sdta0.dot(feats.T)
seqFeaturesS = pd.DataFrame(sfinalfeats,columns=props['description'])
seqFeaturesS['seq_id'] = subdata.index
seqFeaturesS = seqFeaturesS.set_index('seq_id')

subdata = pd.concat([subdata,seqFeaturesS],axis=1)
subdata = subdata.fillna(0)


# In[9]:


predData = np.array(subdata[cols])
predData = bestScaler.transform(predData)
preds = bestModel.predict(predData)


# In[10]:


plt.hist(preds)


# In[11]:


submission = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/sample_submission.csv')
submission['tm'] = preds
submission.to_csv('submission.csv',index=False)

