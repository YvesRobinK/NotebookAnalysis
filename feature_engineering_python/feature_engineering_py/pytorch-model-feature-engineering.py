#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, tqdm, warnings
from itertools import chain, combinations
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from category_encoders.cat_boost import CatBoostEncoder


# In[2]:


train_df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
sample_sub = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')


# In[3]:


device =  'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32 
lr = 0.001
epochs = 200
es = 7
lr_reduce = es - 3
path = './model_best'


# In[4]:


train_df


# In[5]:


train_df.describe()


# In[6]:


train_df.info()


# In[7]:


# train_df['Ticket'].fillna('X', inplace=True)
# train_df['Ticket'] = train_df['Ticket'].apply(lambda x: x.split())
# types = set([x[0] for x in train_df['Ticket'].to_numpy() if len(x) > 1])
# len(list(types))


# In[8]:


train_df['Cabin'].fillna('X-1', inplace=True)
# types = set()
numbers = list()
for i in train_df['Cabin']:
    numbers.append(int(i[1:]))
print(np.min(numbers))
print(np.max(numbers))
print(np.std(numbers))


# ### Feature Engineering
# * ['Cabin_letter'] get cabin letter
# * ['Cabin_num'] get ticket type
# * ['Ticket_type'] get cabin number // 1000 , eg C1534 = 2, A.9875 = 10 // didn't work
# * interactions between numerical columns

# In[9]:


to_oh = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_letter', 'Ticket_type', 'Cabin_num']
to_interact = ['Pclass','Age','SibSp','Parch','Fare']

def normalize(col):
    return (col - col.mean()) / col.std()

def feature_engineering(train, test):
    df = pd.concat([test, train], axis=0)
    
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna('X', inplace=True)
    df['Cabin'].fillna('X-1', inplace=True)
    df['Ticket'].fillna('X', inplace=True)
    
    for i in combinations(to_interact, 2):
        new_col = i[0] + '_' + i[1]
        df[new_col] = df[i[0]] * df[i[1]]
        df[new_col] = normalize(df[new_col])
    
    
    df['Cabin_letter'] = df['Cabin'].apply(lambda x: x[0])
    df['Cabin_num'] = df['Cabin'].apply(lambda x: int(x[1:]) // 1000)
    df['Ticket_type'] = df['Ticket'].apply(lambda x: x[0])
    
    df['Age'] = normalize(df['Age'])
    df['Fare'] = normalize(df['Fare'])
#     df['Cabin_num'] = normalize(df['Cabin_num'])
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Ticket'], axis=1)
    df = pd.concat([pd.get_dummies(df[to_oh]), df.drop(to_oh, axis=1)], axis=1)

    new_train = df[len(train):]
    new_test = df[:len(train)]
    
    return new_train, new_test.drop(['Survived'], axis=1)
train_df, test_df = feature_engineering(train_df, test_df)
INPUT_SHAPE = len(test_df.columns)


# In[10]:


class TitanicDataset():
    def __init__(self, data, target = None):
        self.data = data.values
        if target is not None:
            self.target = target.values
        else:
            self.target = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx, :]
        x = torch.tensor(x, dtype=torch.float32)
        
        if self.target is not None:
            y = self.target[idx]
            return x, torch.tensor([y], dtype=torch.float32)
        else:
            return x
        


# In[11]:


class TitanicModel(nn.Module):
    def __init__(self, input_shape=INPUT_SHAPE):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1)
        
    def forward(self, inp):
        x = self.fc1(inp)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        out = self.out(x)
        
        return out


# In[12]:


def train_one_epoch(model, train_dl, optimizer, criterion):
    model.train()
    epoch_loss = []
    for (X, y) in train_dl:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)

def validate(model, valid_dl, criterion):
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for (X, y) in valid_dl:
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            valid_loss.append(loss.item())
    return np.mean(valid_loss)


def train():
    best_loss = 10e10
    es_count = 0
    global lr
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_df.drop(['Survived'], axis=1),
        train_df['Survived'],
        stratify=train_df['Survived']
    )
    
    train_dl = DataLoader(
        TitanicDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    valid_dl = DataLoader(
        TitanicDataset(X_test, y_test),
        batch_size=8*batch_size
    )
    
    criterion = nn.BCEWithLogitsLoss()
    model = TitanicModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion)
        valid_loss = validate(model, valid_dl, criterion)
        
        print(f"{epoch + 1} epoch, train loss = {train_loss:.6f}, valid_loss = {valid_loss:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), path)
            es_count = 0
        else:
            es_count += 1
        
        if es_count == es:
            break
        if es_count == lr_reduce:
            lr *= 0.5
train()


# In[13]:


def predict():
    valid_dl = DataLoader(
        TitanicDataset(test_df),
        batch_size=8*batch_size
    )
    model = TitanicModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    preds = []
    
    with torch.no_grad():
        for X in valid_dl:
            y = F.sigmoid(model(X))
            preds.append(y.detach().numpy())
            
    return preds

preds = predict()


# In[14]:


sample_sub['Survived'] = np.where(np.vstack(preds).squeeze() >= 0.5, 1, 0)


# In[15]:


sample_sub.to_csv('submission.csv', index=False)

