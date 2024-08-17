#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import time


# # My feature engineering function
# Most of my features are taken from public notebooks and are relatively simple

# In[2]:


def add_features(df):
    #df['area'] = df['time_step'] * df['u_in']
    #df['area'] = df.groupby('breath_id')['area'].cumsum()

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()


    # fast area calculation
    df['time_delta'] = df['time_step'].diff()
    df['time_delta'].fillna(0, inplace=True)
    df['time_delta'].mask(df['time_delta'] < 0, 0, inplace=True)
    df['tmp'] = df['time_delta'] * df['u_in']
    df['area_true'] = df.groupby('breath_id')['tmp'].cumsum()
    df['tmp'] = df['u_out']*(-1)+1 # inversion of u_out

    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    #df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    #df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    #df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    #df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    #df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    #df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    #df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    #df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)

    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    #df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    #df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    #df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    #df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    #df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    #df['cross']= df['u_in']*df['u_out']
    #df['cross2']= df['time_step']*df['u_out']

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df


# In[3]:


print("Loading data")
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
masks=np.array(train['u_out']==0).reshape(-1, 80)
targets = train[['pressure']].to_numpy().reshape(-1, 80)

print("Adding features")
train = add_features(train)
test = add_features(test)

from sklearn.preprocessing import RobustScaler, normalize

print("Dropping some features")
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)
columns=train.columns
np.save('columns',np.array(train.columns))

print("Normalizing")
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

print("Reshaping")
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])


# # Split dataset into train/val

# In[4]:


from sklearn.model_selection import KFold

fold=0

kf = KFold(n_splits=10,random_state=2020,shuffle=True)

train_features=[train[i] for i in list(kf.split(train))[fold][0]]
val_features=[train[i] for i in list(kf.split(train))[fold][1]]
train_targets=[targets[i] for i in list(kf.split(targets))[fold][0]]
val_targets=[targets[i] for i in list(kf.split(targets))[fold][1]]
train_masks=[masks[i] for i in list(kf.split(targets))[fold][0]]
val_masks=[masks[i] for i in list(kf.split(targets))[fold][1]]

#exit()

print(f"### in total there are {len(train_features)} in train###")
print(f"### in total there are {len(val_features)} in val###")


# # Create dataloader

# In[5]:


import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


batch_size=128

class SAKTDataset(Dataset):
    def __init__(self, features, targets, masks, train=True): #HDKIM 100
        super(SAKTDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index].astype('float32'),self.targets[index].astype('float32'),self.masks[index].astype('bool')

class TestDataset(Dataset):
    def __init__(self, features): #HDKIM 100
        super(TestDataset, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.features[index].astype('float32')

train_dataset = SAKTDataset(train_features,train_targets,train_masks)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
del train_features

val_dataset = SAKTDataset(val_features,val_targets,val_masks)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
del val_features


# In[6]:


import torch
from torch import nn
from torch.nn import functional as F

class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM=nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1=nn.Linear(d_model*2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)


    def forward(self, x):
        res=x
        x, _ = self.LSTM(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=res+x
        return x
    
class SAKTModel(nn.Module):
    def __init__(self, n_skill, n_cat, nout, max_seq=100, embed_dim=128, pos_encode='LSTM', nlayers=2, rnnlayers=3,
    dropout=0.1, nheads=8):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        #self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        if pos_encode=='LSTM':
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU':
            self.pos_encoder = nn.ModuleList([ResidualGRU(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU2':
            self.pos_encoder = nn.GRU(embed_dim,embed_dim, num_layers=2,dropout=dropout)
        elif pos_encode=='RNN':
            self.pos_encoder = nn.RNN(embed_dim,embed_dim,num_layers=2,dropout=dropout)
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(n_skill, embed_dim)
        self.cat_embedding = nn.Embedding(n_cat, embed_dim, padding_idx=0)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout) for i in range(nlayers)]
        conv_layers = [nn.Conv1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        deconv_layers = [nn.ConvTranspose1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        layer_norm_layers = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        layer_norm_layers2 = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, nout)
        self.downsample = nn.Linear(embed_dim*2,embed_dim)

    def forward(self, numerical_features, categorical_features=None):
        device = numerical_features.device
        numerical_features=self.embedding(numerical_features)
        x = numerical_features#+categorical_features
        x = x.permute(1, 0, 2)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x=lstm(x)

        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)



        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm2(x)
            x=res+x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
    


# In[7]:


model = SAKTModel(train.shape[-1], 10, 1, embed_dim=256, pos_encode='LSTM',
                  max_seq=None, nlayers=3, rnnlayers=3,
                  dropout=0,nheads=16).cuda()


# In[8]:


#install ranger optimizer
#! git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
#! pip install -e Ranger-Deep-Learning-Optimizer
get_ipython().system(' pip install pytorch_ranger')


# In[9]:


#optimizer and criterion
from pytorch_ranger import Ranger
optimizer = Ranger(model.parameters(), lr=8e-4)
criterion = nn.L1Loss(reduction='none')


# # Training loop

# In[10]:


epochs=150
val_metric = 100
best_metric = 100
cos_epoch=int(epochs*0.75)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(epochs-cos_epoch)*len(train_dataloader))
steps_per_epoch=len(train_dataloader)
val_steps=len(val_dataloader)


# In[11]:


for epoch in range(epochs):
    model.train()
    train_loss=0
    t=time.time()
    for step,batch in enumerate(train_dataloader):
        #series=batch.to(device)#.float()
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        #exit()

        optimizer.zero_grad()
        output=model(features,None)
        #exit()
        #exit()

        loss=criterion(output,targets)#*loss_weight_vector
        loss=torch.masked_select(loss,mask)
        loss=loss.mean()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #scheduler.step()
        print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, steps_per_epoch, train_loss/(step+1), time.time()-t),end='\r',flush=True)
        if epoch > cos_epoch:
            scheduler.step()
        #break
    print('')
    train_loss/=(step+1)

    #exit()
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    masks=[]
    for step,batch in enumerate(val_dataloader):
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        with torch.no_grad():
            output=model(features,None)

            loss=criterion(output,targets)
            loss=torch.masked_select(loss,mask)
            loss=loss.mean()
            val_loss+=loss.item()
            #val_metric.append(MCMAE(output.reshape(-1,4),labels.reshape(-1,4),stds[-4:]))
            preds.append(output.cpu())
            truths.append(targets.cpu())
            masks.append(mask.cpu())
        print ("Validation Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, val_steps, val_loss/(step+1), time.time()-t),end='\r',flush=True)

    preds=torch.cat(preds).numpy()
    truths=torch.cat(truths).numpy()
    masks=torch.cat(masks).numpy()
    val_metric=(np.abs(truths-preds)*masks).sum()/masks.sum()#*stds['pressure']
    #exit()
    print('')
    #val_metric=torch.stack(val_metric).mean().cpu().numpy()
    val_loss/=(step+1)


    if val_metric < best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(),f'model{fold}.pth')


# # After training, create test dataset and make predictions on test dataset

# In[12]:


test_dataset = TestDataset(test)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)


# In[13]:


PRESSURE_MIN=-1.895744294564641
PRESSURE_STEP=0.07030214545121005
PRESSURE_MAX = 64.82099173863328

submission=pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

preds=[]
for batch in tqdm(test_dataloader):
    features=batch.cuda()
    #features=features
    with torch.no_grad():
        temp=[]
        #for model in MODELS:
        output=model(features,None)
        preds.append(output.cpu())

preds=torch.cat(preds)#.reshape(-1).numpy()
print(preds.shape)
post_processed=preds.reshape(-1)
post_processed=torch.round( (post_processed - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
submission['pressure']=post_processed.numpy().clip(PRESSURE_MIN,PRESSURE_MAX)
submission.to_csv('submission.csv',index=False)

