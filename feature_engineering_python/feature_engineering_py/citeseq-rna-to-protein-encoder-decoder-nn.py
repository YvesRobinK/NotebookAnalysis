#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# ### Background
# 
# - CITEseq samples take input X (RNA sequence vector) to predict output Y (Protein sequence vector)
# - Import generated features (n_samples, 240) shape created using PCA and simple feature engineering
# - Feed samples to encoder-decoder NN (see structure below)
# - Train one fold using pytorch (50 epochs, AdamW optimizer, and Cosine scheduler)
# - Use network to make predictions
# 
# ### Improving Upon this Notebook
# 
# **The best way to improve upon this notebook is likely feature engineering**
# - Look for feature importance
# - Improve dimensionality reduction technique
# - Use domain knowledge
# - This is just a baseline
# 
# **Apply similar model for Multiome samples** 
# - Currently, I am just borrowing another submission for multiome and this notebook only predicts CITEseq
# - However, you can expand upon this notebook to make Multiome predictions
# 
# **Change NN Structure**
# - test rnn or cnn
# - try adding attention mechanism
# - change structure to adjust for new features
# 
# ### Neural Network Structure
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6537187%2Fcc192e980bc9f9248d2bcae2c6accd74%2Fencoder_decoder.PNG?generation=1662083041543650&alt=media)
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6537187%2F1a17ab66143625efff11e8a063e1dac1%2Fenc_dec2.PNG?generation=1662083054477703&alt=media)
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6537187%2Fb373fe534194a31dfc3505f90f488b25%2FFCBlock.PNG?generation=1662087114666150&alt=media)

# # Read Data

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DATA_DIR = "/kaggle/input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


# In[2]:


get_ipython().system('pip install --quiet tables')


# In[3]:


cite_train_x = np.load("../input/citeseq-pca-240-preprocessing/cite_train_x.npy")
print(cite_train_x.shape)

cite_test_x = np.load("../input/citeseq-pca-240-preprocessing/cite_test_x.npy")
print(cite_train_x.shape)

cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
print(cite_train_y.shape)


# # Config

# In[4]:


class CFG:
    tr_batch_size = 16 # 16
    va_batch_size = 128 # 32
    
    optimizer = "AdamW"
    lr = 1e-5
    weight_decay = 0.1
    betas = (0.9, 0.999)
    epochs = 50


# # Dataset

# In[5]:


class CtieseqDataset(Dataset):
    """
    Train, Validation or Test dataset for CITEseq samples
    Prepares data for simple vector to vector NN
    """
    def __init__(self, X, y=None):
        self.train = False 
        if y is not None:
            self.train = True
        self.X = X
        self.y = y
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        
        if self.train:
            y = self.y[idx]
            return {
                "X" : torch.tensor(X).to(device),
                "y" : torch.tensor(y).to(device)
            }
        else:
            return {
                "X" : torch.tensor(X).to(device)
            }


# # Utils

# In[6]:


def criterion(outputs, labels):
    """ MSE Loss function"""
    return nn.MSELoss()(outputs, labels)

def correlation_score(y_true, y_pred):
    """
    Scores the predictions according to the competition rules. 
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient
    """
    
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)

def get_optimizer(model, lr, weight_decay, betas):
    """ Gets AdamW optimizer """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr,
                      weight_decay=weight_decay,
                      betas=betas,
                     )
    return optimizer

def get_scheduler(optimizer, T_max=300):
    """ Gets Consine scheduler """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=T_max)
    return scheduler


# # Model

# In[7]:


class FCBlock(nn.Module):
    """
    A Pytorch Block for a fully connected Layer
    Includes Linear, Activation Function, and Dropout
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.selu(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    """
    Encoder module to generate embeddings of a RNA vector
    """
    def __init__(self):
        super().__init__()
        self.l0 = FCBlock(240, 120, 0.05)
        self.l1 = FCBlock(120, 60, 0.05)
        self.l2 = FCBlock(60, 30, 0.05)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class Decoder(nn.Module):
    """
    Decoder module to extract Protein sequences from RNA embeddings
    """
    def __init__(self):
        super().__init__()
        self.l0 = FCBlock(30, 70, 0.05)
        self.l1 = FCBlock(70, 100, 0.05)
        self.l2 = FCBlock(100, 140, 0.05)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class CtieseqModel(nn.Module):
    """
    Wrapper for the Encoder and Decoder modules
    Converts RNA sequence to Protein sequence
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.decoder(embeddings)
        return outputs


# # Train Loop Functions

# In[8]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[9]:


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """ Trains one epoch and returns loss """
    model.train()
    
    losses = AverageMeter()
    corr = AverageMeter()
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        X, y = data["X"], data["y"]
        
        batch_size = X.size(0)

        outputs = model(X)

        n = outputs.size(0)
        loss = criterion(outputs, y)
        losses.update(loss.item(), n)
        loss.backward()
        
        outputs = outputs.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        corr_score = correlation_score(y, outputs)
        corr.update(corr_score, n)
        
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
        
        bar.set_postfix(Epoch=epoch, Train_Loss=losses.avg, Corr=corr.avg,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return losses.avg


# In[10]:


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    """ Evaluates one epoch and returns loss """
    model.eval()
    
    losses = AverageMeter()
    corr = AverageMeter()
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        X, y = data["X"], data["y"]
        
        batch_size = X.size(0)

        outputs = model(X)
        
        n = outputs.size(0)
        loss = criterion(outputs, y)
        losses.update(loss.item(), n)
        
        outputs = outputs.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        corr_score = correlation_score(y, outputs)
        corr.update(corr_score, n)
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=losses.avg, Corr=corr.avg,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return losses.avg


# In[11]:


def train_one_fold(model, 
                   optimizer, 
                   scheduler, 
                   train_loader, 
                   valid_loader, 
                   fold):
    """ Trains and saves a full fold of a pytorch model """
    best_epoch_loss = np.inf
    model.to(device)

    for epoch in range(CFG.epochs):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, 
                                           optimizer, 
                                           scheduler, 
                                           dataloader=train_loader, 
                                           device=device, 
                                           epoch=epoch)

        val_epoch_loss = valid_one_epoch(model,
                                         optimizer, 
                                         valid_loader, 
                                         device=device, epoch=epoch)
        
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            torch.save(model.state_dict(), f"model_f{fold}.bin")
            
    print("Best Loss: {:.4f}".format(best_epoch_loss))


# # Run Training

# In[12]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_list = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(cite_train_x)):
    print(f"\nfold = {fold}")
    X_tr = cite_train_x[idx_tr] 
    y_tr = cite_train_y[idx_tr]
    
    X_va = cite_train_x[idx_va]
    y_va = cite_train_y[idx_va]
    
    ds_tr = CtieseqDataset(X_tr, y_tr)
    ds_va = CtieseqDataset(X_tr, y_tr)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.tr_batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=CFG.va_batch_size, shuffle=False)
    
    model = CtieseqModel()
    optimizer = get_optimizer(model, CFG.lr, CFG.weight_decay, CFG.betas)
    scheduler = get_scheduler(optimizer) 

    train_one_fold(model, optimizer, scheduler, dl_tr, dl_va, fold)
    


# # Predict & Submit

# In[13]:


def predict(fold):
    preds = list()
    ds = CtieseqDataset(cite_test_x)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    model = CtieseqModel()
    model.load_state_dict(torch.load(f"model_f{fold}.bin"))
    model.eval()

    bar = tqdm(enumerate(dl), total=len(dl))
    for step, data in bar:        
        X = data["X"]

        batch_size = X.size(0)

        outputs = model(X)
        preds.append(outputs.detach().cpu().numpy())
    test_pred = np.concatenate(preds)
    return test_pred

test_preds = np.array([predict(0), predict(1), predict(2), predict(3), predict(4)])
test_preds = np.mean(test_preds, axis=0)
print(test_preds.shape)


# In[14]:


# LB Inflation
test_preds[:7476] = cite_train_y[:7476]

# Submission
submission = pd.read_csv('../input/msci-multiome-quickstart-w-sparse-matrices/submission.csv', index_col='row_id', squeeze=True)
submission.iloc[:len(test_preds.ravel())] = test_preds.ravel()
assert not submission.isna().any()
submission = submission.round(6) # reduce the size of the csv
submission.to_csv('submission.csv')
submission


# In[ ]:




