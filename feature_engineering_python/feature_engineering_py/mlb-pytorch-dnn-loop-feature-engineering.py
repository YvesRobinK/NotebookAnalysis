#!/usr/bin/env python
# coding: utf-8

# This is my first baseline linear NN for this competition implemented in Pytorch. Since there are no Pytorch pipelines publicly available yet, I thought I would share mine. Inference runs without problems and the score seems reasonable to me. Still, I am not 100 % sure that the pipeline is bug-free. So, please do check yourself if you want to build off of this baseline.
# 
# **Preprocessing:** You can find the code to preprocess the train files in my other notebook: https://www.kaggle.com/nicohrubec/unnest-train?scriptVersionId=66371803
# 
# **Model:** Simple 3-layer DNN with 100 neurons each.
# 
# **Validation:** 5-Fold TimeSeriesSplit.
# 
# **Feature Engineering:** For now very basic, the model gets fed with some time features (weekday, week, month) + some target means (overall player target mean and player target mean for each year). I do not use any pandas for the target means but use a loop in which I build up dictionaries row-wise that store stats for each player. I then use these dictionaries to calculate the features. That way there is no leakage and it's quite straightforward to update the stats with new incoming test data.
# 
# I'd definitely appreciate any feedback. If I did miss to give any credits, please write in the comments below. Feel free to ask any questions, if you do not understand any part of the code. Also pls upvote, if you find it useful.
# 
# 
# **Changelog:**
# 
# **v2:**
# - Fix the validation setup. Feature dictionary now get computed within the fold loop to prevent leakage. 
# - Clip predictions.
# 
# **v3:**
# - Remove year mean features.
# 
# **v4:**
# - Fix inference so that test is predicted with all 5 fold models.
# 
# **v6:**
# - Fix another mild leak in the validation procedure. Before the update feature dicts were updated with the current row before obtaining the feature values for the current row. This is not a real issue with the current feature sets, since only very broad means are used, but would become a more problematic source of leakage as soon as more granular features are introduced.
# 
# **v8:**
# - Add targets stats as shown in this notebook: https://www.kaggle.com/mlconsult/1-38-lb-lightgbm-with-target-statistics/notebook. Note that the CV score gets slightly inflated by that.
# - Cleaning up some vars that are no longer needed.

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('../input/unnest-train/train_nextDayPlayerEngagement.csv')
target_stats = pd.read_csv('../input/player-target-stats/player_target_stats.csv')


# In[3]:


class CFG:
    nfolds = 5
    batch_size = 1024
    val_batch_size = batch_size * 8
    nepochs = 10
    lr = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    targets = ['target1', 'target2', 'target3', 'target4']
    engineered_cols = ['player_t1_mean', 'player_t2_mean', 'player_t3_mean', 'player_t4_mean']
    cols = ['month', 'week', 'weekday'] + engineered_cols + [col for col in target_stats.columns if col not in ['playerId']]
    debug = False


# In[4]:


train


# In[5]:


if CFG.debug:
    train = train[:10000]


# In[6]:


class MLBDataset(Dataset):
    def __init__(self, df, targets=None, mode='train'):
        self.mode = mode
        self.data = df
        if mode == 'train':
            self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x = self.data[idx]
            y = np.array(self.targets[idx])
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        elif self.mode == 'test':
            return torch.from_numpy(self.data[idx]).float()


# In[7]:


class MLBModel(nn.Module):
    def __init__(self, num_cols):
        super(MLBModel, self).__init__()
        self.dense1 = nn.Linear(num_cols, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, len(CFG.targets))

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x).squeeze()

        return x


# In[8]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# In[9]:


# takes an unnested df and does a feature lookup for each player from the given feature dictionary
def add_player_features(row, player_dict, n_feats, id_col, nan_value=-1):
    features = np.full((n_feats), nan_value, dtype=np.float32)
    
    pid = row[id_col]
    
    if pid in player_dict:
        # overall player means
        p_count = player_dict[pid]['count']
        features[0] = player_dict[pid]['mean_t1'] / p_count
        features[1] = player_dict[pid]['mean_t2'] / p_count
        features[2] = player_dict[pid]['mean_t3'] / p_count
        features[3] = player_dict[pid]['mean_t4'] / p_count
    
    return features


# In[10]:


# takes a sample and uses it to update the features in the player_dict
# if a dict is given further calculations will be based on it, else an empty one is created
def update_player_features(row, player_dict, nan_value=-1):
    assert player_dict is not None
    
    pid = row[2]
    t1, t2, t3, t4 = row[3], row[4], row[5], row[6]
    
    if pid in player_dict:
        # update target lags
        player_dict[pid]['mean_t1'] += t1
        player_dict[pid]['mean_t2'] += t2
        player_dict[pid]['mean_t3'] += t3
        player_dict[pid]['mean_t4'] += t4
        player_dict[pid]['count'] += 1
    else:
        # init the feature dict for this player
        player_dict[pid] = {'mean_t1': t1, 'mean_t2': t2, 'mean_t3': t3, 'mean_t4': t4, 'count': 1}
    
    return player_dict


# In[11]:


def do_feature_engineering(df, id_col=2):
    player_features = {}
    nfeats = len(CFG.engineered_cols)
    features = np.zeros((df.shape[0], nfeats), dtype=np.float32)
    
    for idx, row in enumerate(tqdm(df.values)):
        row_features = add_player_features(row, player_features, nfeats, id_col=id_col)
        player_features = update_player_features(row, player_features)
        features[idx] = row_features
    
    features = pd.DataFrame(features, columns=CFG.engineered_cols)
    df = pd.concat([df, features], axis=1)
    
    return df, player_features


# In[12]:


def do_feature_engineering_test(df, player_dict, id_col=2):
    features = np.zeros((df.shape[0], len(CFG.engineered_cols)), dtype=np.float32)
    
    for idx, row in enumerate(df.values):
        row_features = add_player_features(row, player_dict, len(CFG.engineered_cols), id_col=id_col)
        features[idx] = row_features
    
    features = pd.DataFrame(features, columns=CFG.engineered_cols, index=df.index)
    df = pd.concat([df, features], axis=1)
    
    return df


# In[13]:


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    train_loss = 0.0

    for i, (sample, target) in enumerate(tqdm(loader)):
        sample, target = sample.to(device), target.to(device)
        optimizer.zero_grad()

        preds = model(sample)

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(loader)

    return model, train_loss


# In[14]:


def validate(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for i, (sample, target) in enumerate(tqdm(loader)):
            sample, target = sample.to(device), target.to(device)

            preds = model(sample)
            loss = criterion(preds, target)
            val_loss += loss.item() / len(loader)

            val_preds.append(preds.cpu())
            val_targets.append(target.cpu())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

    return val_loss, val_preds


# In[15]:


class ColL1Loss(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        self.mae = nn.L1Loss()
        assert num_targets != 0
        self.num_targets = num_targets
    
    def forward(self, preds, targets):
        l1 = 0.0
        
        for i in range(self.num_targets):
            l1 += self.mae(preds[:, i], targets[:, i])
        
        return l1 / self.num_targets


# In[16]:


def train_fold(xtrn, ytrn, xval, yval, fold):
    print(f"Train fold {fold}")
    train_set = MLBDataset(xtrn, ytrn)
    val_set = MLBDataset(xval, yval)
    
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CFG.val_batch_size, shuffle=False)
    
    model = MLBModel(xtrn.shape[1]).to(CFG.device)
    criterion = ColL1Loss(len(CFG.targets))
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    
    best_loss = {'train': np.inf, 'val': np.inf}
    best_val_preds = None
    
    for epoch in range(1, CFG.nepochs + 1):
        print(f"Train epoch {epoch}")
        model, loss = train_epoch(model, train_loader, optimizer, CFG.device, criterion)
        val_loss, val_preds = validate(model, val_loader, CFG.device, criterion)
        
        # save best model
        if val_loss < best_loss['val']:
            best_loss = {'train': loss, 'val': val_loss}
            torch.save(model.state_dict(), f'fold{fold}.pt')
            
            # save best oof predictions
            best_val_preds = val_preds
            np.save(f'oof_fold{fold}.npy', val_preds)
        
        print("Train loss: {:5.5f}".format(loss))
        print("Val loss: {:5.5f}".format(val_loss))
    
    return best_val_preds


# In[17]:


# extract very basic datetime features
def extract_date_feats(df):
    df['year'] = pd.DatetimeIndex(df['engagementMetricsDate']).year
    df['month'] = pd.DatetimeIndex(df['engagementMetricsDate']).month
    df['week'] = pd.DatetimeIndex(df['engagementMetricsDate']).week
    df['weekday'] = pd.DatetimeIndex(df['engagementMetricsDate']).weekday
    
    return df


# In[18]:


def train_kfolds(df):
    set_seed(CFG.seed)
    ts = TimeSeriesSplit(n_splits=CFG.nfolds)
    df = extract_date_feats(df)
    
    oof_preds = []
    oof_targets = []
    
    for fold_idx, (trn_idx, val_idx) in enumerate(ts.split(df)):
        trn_df, val_df = df.iloc[trn_idx], df.iloc[val_idx]
        
        trn_df, player_dict = do_feature_engineering(trn_df)
        val_df = do_feature_engineering_test(val_df, player_dict)
        trn_df = pd.merge(trn_df, target_stats, how='left', on='playerId')
        val_df = pd.merge(val_df, target_stats, how='left', on='playerId')
        
        xtrn, ytrn = trn_df[CFG.cols].values, trn_df[CFG.targets].values
        xval, yval = val_df[CFG.cols].values, val_df[CFG.targets].values
        
        val_preds = train_fold(xtrn, ytrn, xval, yval, fold_idx)
        
        oof_preds.append(val_preds)
        oof_targets.append(yval)
        del trn_df, val_df, player_dict
        del xtrn, ytrn, xval, yval
    
    oof_preds = torch.from_numpy(np.clip(np.concatenate(oof_preds), 0, 100)).float()
    oof_targets = torch.from_numpy(np.concatenate(oof_targets)).float()
    
    eval_metric = ColL1Loss(len(CFG.targets))
    oof_loss = eval_metric(oof_preds, oof_targets)
    df, player_dict = do_feature_engineering(df)
    del oof_preds, oof_targets
    del df
    
    return oof_loss, player_dict


# In[19]:


loss, player_dict = train_kfolds(train)
print("OOF loss: {:5.5f}".format(loss))
del train


# In[20]:


def predict_test(test, model, device):
    model.eval()
    model = model.to(device)
    test_preds = []
    
    test_set = MLBDataset(xtest, mode='test')
    test_loader = DataLoader(test_set, batch_size=CFG.val_batch_size, shuffle=False)
    
    with torch.no_grad():
        for i, (sample) in enumerate(test_loader):
            sample = sample.to(device)
            preds = model(sample)
            test_preds.append(preds.cpu())
        
        test_preds = np.concatenate(test_preds)
        
    return test_preds


# In[21]:


import mlb
env = mlb.make_env()

for (test_df, sub) in env.iter_test():
    sub_df = sub.copy()
    sub_df['playerId'] = sub_df["date_playerId"].apply(lambda x: int( x.split("_")[1] ) )
    sub_df['engagementMetricsDate'] = sub_df["date_playerId"].apply(lambda x: int( x.split("_")[0] ) )
    sub_df['engagementMetricsDate'] = sub_df['engagementMetricsDate'].apply(lambda x: str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:])
    sub_df = extract_date_feats(sub_df)
    sub_df = do_feature_engineering_test(sub_df, player_dict, id_col=5)
    sub_df = pd.merge(sub_df, target_stats, how='left', on='playerId')
    
    for fold_idx in range(CFG.nfolds):
        # init model and load checkpoint
        model = MLBModel(len(CFG.cols))
        path = f'fold{fold_idx}.pt'
        model.load_state_dict(torch.load(path))
        
        # predict
        xtest = sub_df[CFG.cols].values
        sub[CFG.targets] += np.clip(predict_test(xtest, model, CFG.device), 0, 100) / CFG.nfolds
    
    env.predict(sub)


# In[22]:


sub_df


# In[23]:


sub


# In[ ]:




