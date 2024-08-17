#!/usr/bin/env python
# coding: utf-8

# # [Train, infer] Catalyst + PyTorch RNN Baseline
# 
# ![](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)
# 
# This is mainly a RNN modification of MatthewMasters' CNN baseline (do have a look there and upvote it). The RNN model gives considerable improvements in CV over a CNN, and it seems the same can be said for TensorFlow given Xhlulu's brilliant kernel.
# 
# Now the main part of this notebook is to demonstrate how Catalyst simplifies your training loop in a few ways:-
# + Makes it much easier to train with PyTorch
# + Inference too gets simplified drastically.

# # Imports and helpers

# Typical Data science/machine learning stack for Torch with the addition of `catalyst`.

# In[1]:


import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.nn.functional as F
import catalyst.dl as dl
import catalyst.dl.utils as utils

import pandas as pd
import numpy as np


# This involves defining a few basic functions: one-hot encoding, feature engineering and typical preprocessing functions.

# In[2]:


def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)))
    for idx, char in enumerate(string):
        encoding[idx, categories.index(char)] = 1
    return encoding

def featurize(entity):
    sequence = one_hot(list('ACGU'), entity['sequence'])
    structure = one_hot(list('.()'), entity['structure'])
    loop_type = one_hot(list('BEHIMSX'), entity['predicted_loop_type'])
    features = np.hstack([sequence, structure, loop_type])
    return features 

def char_encode(index, features, feature_size):
    half_size = (feature_size - 1) // 2
    
    if index - half_size < 0:
        char_features = features[:index+half_size+1]
        padding = np.zeros((int(half_size - index), char_features.shape[1]))
        char_features = np.vstack([padding, char_features])
    elif index + half_size + 1 > len(features):
        char_features = features[index-half_size:]
        padding = np.zeros((int(half_size - (len(features) - index))+1, char_features.shape[1]))
        char_features = np.vstack([char_features, padding])
    else:
        char_features = features[index-half_size:index+half_size+1]
    
    return char_features


# # Setup Model and Data Processing

# In[3]:


class VaxDataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        self.test = test
        self.features = []
        self.targets = []
        self.ids = []
        self.load_data()
    
    def load_data(self):
        with open(self.path, 'r') as text:
            for line in text:
                records = json.loads(line)
                features = featurize(records)
                
                for char_i in range(records['seq_scored']):
                    char_features = char_encode(char_i, features, 21)
                    self.features.append(char_features)
                    self.ids.append('%s_%d' % (records['id'], char_i))
                        
                if not self.test:
                    targets = np.stack([records['reactivity'], records['deg_Mg_pH10'], records['deg_Mg_50C']], axis=1)
                    self.targets.extend([targets[char_i] for char_i in range(records['seq_scored'])])
                    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        if self.test:
            return self.features[index], self.ids[index]
        else:
            return self.features[index], self.targets[index], self.ids[index]


# In[4]:


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
 


class VaxModel(nn.Module):
    def __init__(self):
        super(VaxModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(14, 32, 1, 1),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, 1, 1),
        )
        self.layers2 = nn.Sequential(
            nn.GRU(42, 32),
        )
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )
    
    def forward(self, features):
        features = self.layers(features)
        features = features.permute(1, 0, 2)
        features = self.layers2(features)
         
        final = self.final(features[0])
        return final[0, :, :]


# In[5]:


model = VaxModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9)
criterion = nn.MSELoss()


# In[6]:


train_dataset = VaxDataset('../input/stanford-covid-vaccine/train.json')
train_dataloader = DataLoader(train_dataset, 16, shuffle=True, num_workers=4, pin_memory=True)


# # Training loop

# In[7]:


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device).permute(0, 2, 1).float()), batch[1]

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch[0], batch[1]
        x = x.cuda().permute(0,2,1).float()
        y = y.cuda().float()
        y_hat = self.model(x)

        loss = criterion(y_hat, y)
        score = mcrmse_loss(y_hat, y)
        self.batch_metrics.update(
            {"loss": loss, 'metric': score}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


# In[8]:


device = utils.get_device()


# In[9]:


def mcrmse_loss(y_true, y_pred, N=3):
    """
    Calculates competition eval metric
    """
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis=0)/n)) / N


# In[10]:


test_dataset = VaxDataset('../input/stanford-covid-vaccine/test.json', test=True)
test_dataloader = DataLoader(test_dataset, 16, num_workers=4, drop_last=False, pin_memory=True)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3,max_lr=1e-2,step_size_up=2000)

loaders = {
    'train': train_dataloader
    
}
runner = CustomRunner(device=device)
# model training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    logdir="../working",
    num_epochs=5,
    scheduler=scheduler,
    verbose=False,
    load_best_on_end=True,
    
)


# Check a few of the results post-training.

# In[11]:


utils.plot_metrics(
    logdir="../working", 
    # specify which metrics we want to plot
    metrics=["loss", "metric"]
)


# # Submission

# In[12]:


sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv', index_col='id_seqpos')

for predictions, ids in runner.predict_loader(loader=test_dataloader):
    sub.loc[ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = predictions.detach().cpu().numpy()


# In[13]:


sub.head()


# In[14]:


sub.to_csv('submission.csv')

