#!/usr/bin/env python
# coding: utf-8
## In this noteebook I have created bio-informative features based on the protein sequence given in the train dataset.
# In[1]:


import os
import time
import string
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

COMMON_PATH = '/kaggle/input/novozymes-enzyme-stability-prediction'
NN_MODEL = False


# In[2]:


def seed_everything(seed=47):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()


# ### Loading Data Files

# In[3]:


train = pd.read_csv(f'{COMMON_PATH}/train.csv')
test = pd.read_csv(f'{COMMON_PATH}/test.csv')
tu = pd.read_csv(f'{COMMON_PATH}/train_updates_20220929.csv')


# #### Removing bad sequences from train data based on updated file

# In[4]:


bad_seq_ids = tu['seq_id'].values.tolist()
train = train[~train['seq_id'].isin(bad_seq_ids)]
train = train.reset_index(drop=True)


# In[5]:


print(f"Shape of train data after removing bad sequences: {train.shape}")
print(f"Shape of test data: {test.shape}")


# ## Feature Engineering using BioPython & Sklearn Pipeline

# In[6]:


def feat_engg(data):
    data['protein_seq_len'] = data['protein_sequence'].str.len()
    data['protein_analysis_obj'] = data['protein_sequence'].apply(lambda x:ProteinAnalysis(x))
    data['protein_sequence_obj'] = data['protein_analysis_obj'].apply(lambda x: x.sequence)
    data['protein_molecular_weight'] = data['protein_analysis_obj'].apply(lambda x:x.molecular_weight())
    data['protein_aromaticity'] = data['protein_analysis_obj'].apply(lambda x:x.aromaticity())
    data['protein_charge_at_ph7'] = data['protein_analysis_obj'].apply(lambda x: round(x.charge_at_pH(pH=7),2))
    data['protein_gravy_val'] = data['protein_analysis_obj'].apply(lambda x: x.gravy())
    data['protein_isoelectric_point'] = data['protein_analysis_obj'].apply(lambda x: x.isoelectric_point())
    
    amino_acids_codes = [alpha for alpha in string.ascii_uppercase if alpha not in ['J', 'O', 'U', 'X']]
    for code in amino_acids_codes:
        data[f'amino_count_{code}'] = data['protein_sequence_obj'].apply(lambda x:x.count(code))
        
    X, y = build_vector(data)
    df = gen_sequence_features(data, X)
    return df, y


# In[7]:


def gen_sequence_features(data, seq):
    seq_df = pd.DataFrame(seq)
    seq_df.columns = [f'protein_seq_idx_char_{i+1}' for i in range(seq.shape[1])]
    return pd.concat([data, seq_df], axis=1)


# In[8]:


## Using built_vector function from @TSCHEUNG

def build_vector(df):
    
    sequence_letters = [alpha for alpha in string.ascii_uppercase if alpha not in ['J', 'O', 'U', 'X', 'B', 'Z']]
    N = len(sequence_letters)
    is_train = True if 'tm' in df.columns else False

    X = []
    y = []
    for index, data in enumerate(df['protein_sequence']):
        
        lists = list(data)
        freq_matrix = np.zeros((N, N))
        
        for j in range(len(lists)):
            if j >0 and j < len(lists)-1:
                index0 = sequence_letters.index(lists[j-1])
                index1 = sequence_letters.index(lists[j])
                index2 = sequence_letters.index(lists[j+1])
                freq_matrix[index1][index0] += 1
                freq_matrix[index1][index2] += 1
        X.append((freq_matrix/(N**2)).reshape(-1))
        
        if is_train:
            y.append(df.loc[index, 'tm'])
            
    return np.array(X), np.array(y)


# In[9]:


# FunctionTransformer custom class
class DFFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self


# In[10]:


pipeline = Pipeline([("feat_eng", DFFunctionTransformer(feat_engg))])
train, y = pipeline.fit_transform(train)
test, _ = pipeline.transform(test)


# In[11]:


cols_to_ignore = ['seq_id','protein_sequence','data_source','protein_analysis_obj','protein_sequence_obj']
X = train.loc[:,~train.columns.isin(cols_to_ignore +['tm'])]
y = train['tm']

test = test.loc[:,~test.columns.isin(cols_to_ignore)]


# In[12]:


kf = KFold(n_splits = 5, shuffle = True, random_state = 47)
test_preds_lgb = []
fold = 1
overfits = []

for train_index, test_index in kf.split(X, y):
    
    ## Splitting the data
    X_train , X_val = X.iloc[train_index], X.iloc[test_index]  
    Y_train, Y_val = y.iloc[train_index], y.iloc[test_index]    
    
    y_pred_list = []
    
    model_lgb = LGBMRegressor(n_estimators = 250, 
                              learning_rate = 0.01,
                              num_leaves = 31,
                              max_depth = 5, 
                              reg_alpha = 1, 
                              reg_lambda = 5, 
                              subsample = 0.75,
                              colsample_bytree = 0.55)
    model = model_lgb.fit(X_train, Y_train)
    train_pred = model_lgb.predict(X_train)
    model_pred = model_lgb.predict(X_val)
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns}).sort_values(by='Value', ascending=False)
    train_score = round(spearmanr(Y_train, train_pred)[0],4)
    valid_score = round(spearmanr(Y_val, model_pred)[0],4)
    overfit = round((train_score - valid_score)*100,2)
    overfits.append(overfit)
    print(f'Fold {fold} - Train Score: {train_score} ; Valid Score: {valid_score} ; Overfit: {overfit} %')

    test_preds_lgb.append(model_lgb.predict(test))
    fold +=1
print(f"Overall average overfitting %: {round(sum(overfits)/len(overfits),2)}")


# In[13]:


if NN_MODEL:
    train_features = torch.tensor(X.values, dtype=torch.float32)

    train_labels = torch.tensor(y.values.reshape(-1,1),
                                dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(train_features,train_labels)

    temp_num = int(len(dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [temp_num, len(dataset) - temp_num])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1024, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1024)


# In[14]:


if NN_MODEL:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class NESPModel(nn.Module):
        def __init__(self, n_features):
            super().__init__()

            self.fc = nn.Linear(n_features, 456)
            self.fc2 = nn.Linear(456, 256)
            self.fc3 = nn.Linear(256,128)
            self.fc4 = nn.Linear(128,64)
            self.fc5 = nn.Linear(64,1)

            self.act = nn.ReLU()
            self.drop = nn.Dropout(0.05)
            self.norm1 = nn.BatchNorm1d(456)
            self.norm2 = nn.BatchNorm1d(256)
            self.norm3 = nn.BatchNorm1d(128)
            self.norm4 = nn.BatchNorm1d(64)

        def forward(self, inputs):

            x = self.act(self.norm1(self.fc(inputs)))
            x = self.act(self.norm2(self.fc2(x)))
            x = self.act(self.norm3(self.fc3(x)))
            x = self.act(self.norm4(self.fc4(x)))
            x = self.act(self.fc5(x))
            return x



    model = NESPModel(X.shape[1])
    model = model.to(device)
    print(model)


# In[15]:


get_ipython().run_cell_magic('time', '', '\nif NN_MODEL:\n    loss_fn = nn.L1Loss()\n\n    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)\n    train_loss_all = []\n\n    for epoch in range(75):\n        print(f"Training Epoch: {epoch+1}")\n        train_loss = 0\n        train_num = 0\n        for step,(X, y) in enumerate(train_dataloader):\n            X, y = X.to(device), y.to(device)\n            output = model(X)\n            loss = loss_fn(output, y)\n            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()\n\n            train_loss += loss.item() * X.size(0)\n            train_num += X.size(0)\n        print(f"Epoch loss: {train_loss/train_num}")\n        train_loss_all.append(train_loss / train_num)\n    \n    # Plotting Training curve\n    plt.figure(figsize=(6,3))\n    plt.plot(train_loss_all, "ro-", label="Train loss")\n    plt.legend()\n    plt.grid()\n    plt.xlabel("epoch")\n    plt.ylabel("loss")\n\n    plt.show()\n')


# In[16]:


if NN_MODEL:
    val_losses = []

    val_loss = 0
    val_num = 0
    for step,(X, y) in enumerate(val_dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_fn(output, y)

        val_loss += loss.item() * X.size(0)
        val_num += X.size(0)

    val_losses.append(val_loss / val_num)
    print(f"Val Loss:{sum(val_losses)/len(val_losses)}")
    
    test_dataset = torch.tensor((test.values), dtype=torch.float32)
    output = model(test_dataset.to(device)).to(device)
    nn_preds = output.cpu().data.numpy()
    nn_preds_df = pd.DataFrame(nn_preds, columns=['tm'])


# In[17]:


test_preds = pd.DataFrame(test_preds_lgb).sum(axis=0)
test_preds = test_preds / 5
submission = pd.read_csv(f'{COMMON_PATH}/sample_submission.csv')
submission['tm'] = test_preds
submission.to_csv('submission.csv', index=False)


# In[18]:


submission.head(10)


# In[ ]:




