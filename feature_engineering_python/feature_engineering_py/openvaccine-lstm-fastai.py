#!/usr/bin/env python
# coding: utf-8

# # Open Vaccine Fastai RNN

# I will use this notebook to experiment with various RNN approaches to Open Vaccine competition using fastai library. To read more about RNN with fastai, read this: https://github.com/fastai/fastbook/blob/master/12_nlp_dive.ipynb

# ## All updates:
# - FIX: predict for 130-long sequences in test
# - Visualize predictions
# - loss function (from xhlulu)
# - FIX: kernel now running on GPU! (thanks to fast.ai forums, especially Satyabrata Pal and Zach Mueller!)
# - some hyperparameter tuning...
# - improved inference time with pandas explode
# - add bpps feature (from tito)
# - k-fold validation and ensemble
# - hyperparam tuning
#     - epochs count
#     - batch size
#     - learning rates
#     - embedding / hidden sizes / n-layers
#     - dropout

# # Imports, installs, reading the data

# In[1]:


get_ipython().system('pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q')
get_ipython().system('pip install fastai==2.0.13 -q')


# In[2]:


from fastai.text.all import *
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from torch import nn
from sklearn.model_selection import KFold


# In[3]:


path = '/kaggle/input/stanford-covid-vaccine'
train = pd.read_json(f'{path}/train.json',lines=True)
test = pd.read_json(f'{path}/test.json', lines=True)
sub = pd.read_csv(f'{path}/sample_submission.csv')


# In[4]:


train.shape, train['id'].nunique(), test.shape, sub.shape


# In[5]:


# BPPS features from: https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)


# In[6]:


train = train.sample(frac=1, random_state=42)


# # Preparing the data for RNN

# In[7]:


all1 = []
all2 = []
all3 = []
for i in range(len(train)):
    all1.extend(train['sequence'].loc[i])
    all2.extend(train['structure'].loc[i])
    all3.extend(train['predicted_loop_type'].loc[i])


# In[8]:


all1 = L(all1)
all2 = L(all2)
all3 = L(all3)


# In[9]:


vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()


# In[10]:


word2idx1 = {w:i for i,w in enumerate(vocab1)}
word2idx2 = {w:i for i,w in enumerate(vocab2)}
word2idx3 = {w:i for i,w in enumerate(vocab3)}


# In[11]:


def joiner(row):
    l1 =  list(row[0])
    l2 =  list(row[1])
    l3 =  list(row[2])
    l4 =  list(row[3])
    l5 =  list(row[4])
    l6 =  list(row[5])
    out = [[word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]], l4[i], l5[i], l6[i]] for i in range(len(l1))]
    return out


# In[12]:


train['seqs'] = train[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)


# In[13]:


train = train[train['SN_filter'] == 1]


# In[14]:


txts = L([x for x in train['seqs'].values])
tgts1 = L([x for x in train['reactivity'].values])
tgts2 = L([x for x in train['deg_Mg_pH10'].values])
tgts3 = L([x for x in train['deg_pH10'].values])
tgts4 = L([x for x in train['deg_Mg_50C'].values])
tgts5 = L([x for x in train['deg_50C'].values])


# In[15]:


seqs = L((tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]])) for i in range(len(txts)))


# # Test Data Preparation

# In[16]:


test['seqs'] = test[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)
test107 = test[test['seq_length'] == 107].reset_index(drop=True)
test130 = test[test['seq_length'] == 130].reset_index(drop=True)
len(test107), len(test130)


# ## 107-length

# In[17]:


test107_ids = pd.DataFrame()
test107_ids['id'] = test107['id']
for i in range(11): # fill up the batch for prediction
    test107_ids.loc[len(test107_ids)] = 'id_dummy'    
test107_ids['seqnum'] = ''    
test107_ids['seqnum'] = test107_ids['seqnum'].astype(object)
sn = np.array(list(range(107)))
for i in range(len(test107_ids)):
    test107_ids['seqnum'].loc[i] = sn
test107_ids = test107_ids.explode('seqnum').reset_index(drop=True)
test107_ids['id_seqpos'] = test107_ids.apply(lambda r: str(r[0]) + '_' + str(r[1]), axis=1)


# In[18]:


test107_seqs = [(tensor(x), torch.zeros(5,68)) for x in test107['seqs'].values]
len(test107_seqs)
#11 empty seqs to fill up the batch :/
test107_seqs_empty = [(torch.zeros((107,6), dtype=torch.long), torch.zeros(5,68)) for i in range(11)]
test107_seqs += test107_seqs_empty
test107_seqs = L(test107_seqs)
len(test107_seqs), len(test107_seqs) % 32


# In[19]:


test107_seqs = [(a.to('cuda'), b.to('cuda')) for (a,b) in test107_seqs]


# ## 130-length

# In[20]:


test130_ids = pd.DataFrame()
test130_ids['id'] = test130['id']
for i in range(3): # fill up the batch for prediction
    test130_ids.loc[len(test130_ids)] = 'id_dummy'    
test130_ids['seqnum'] = ''    
test130_ids['seqnum'] = test130_ids['seqnum'].astype(object)
sn = np.array(list(range(130)))
for i in range(len(test130_ids)):
    test130_ids['seqnum'].loc[i] = sn
test130_ids = test130_ids.explode('seqnum').reset_index(drop=True)
test130_ids['id_seqpos'] = test130_ids.apply(lambda r: str(r[0]) + '_' + str(r[1]), axis=1)


# In[21]:


test130_seqs = [(tensor(x), torch.zeros(5,68)) for x in test130['seqs'].values]
len(test130_seqs)
#3 empty seqs to fill up the batch :/
test130_seqs_empty = [(torch.zeros((130,6), dtype=torch.long), torch.zeros(5,68)) for i in range(3)]
test130_seqs += test130_seqs_empty
test130_seqs = L(test130_seqs)
len(test130_seqs), len(test130_seqs) % 32


# In[22]:


test130_seqs = [(a.to('cuda'), b.to('cuda')) for (a,b) in test130_seqs]


# # Config

# In[23]:


BS = 32 # batch size 
ES = 32 # embedding size
NH = 512 # number hidden units
NL = 3 # number layers
DO = 0.3 # dropout
EP = 20 # epochs
LR = 0.009281670019785143 # learning rate
WD = 0.0 # weight decay


# # Model and loss function

# In[24]:


sl = 107

class OVModel(Module):
    def __init__(self, vocab1_sz, vocab2_sz, vocab3_sz, emb_sz, n_hidden, n_layers, p, y_range=None):
        self.y_range = y_range
        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(emb_sz*3+3, n_hidden, n_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden*2, 5)
        self.h = [torch.zeros(n_layers*2, BS, n_hidden).to('cuda') for _ in range(2)]
        
    def forward(self, x):
        e1 = self.i_h1(x[:,:,0].long())
        e2 = self.i_h2(x[:,:,1].long())
        e3 = self.i_h3(x[:,:,2].long())
        bp = x[:,:,3:]
        e = torch.cat((e1, e2, e3, bp), dim=2)
        raw,h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)
        if self.y_range is None: 
            self.h = [h_.detach() for h_ in h]
            return out, raw, do        
        out = torch.sigmoid(out) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
        self.h = [h_.detach() for h_ in h]
        return out, raw, do
    
    def reset(self): 
        for h in self.h: h.zero_()


# In[25]:


def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:,:68,:]
    l1 = F.mse_loss(inp[:,:,0], targ[:,0,:])
    l2 = F.mse_loss(inp[:,:,1], targ[:,1,:])
    l3 = F.mse_loss(inp[:,:,2], targ[:,2,:])
    l4 = F.mse_loss(inp[:,:,3], targ[:,3,:])
    l5 = F.mse_loss(inp[:,:,4], targ[:,4,:])
    return torch.sqrt((l1 + l2 + l3 + l4 +l5)/5)


# # Data Loaders, Training, Inference

# In[26]:


test_dl107 = DataLoader(dataset=test107_seqs, bs=BS, shuffle=False, drop_last=True)
test_dl130 = DataLoader(dataset=test130_seqs, bs=BS, shuffle=False, drop_last=True)


# In[27]:


spltidx = np.array(range(len(seqs)))
kf = KFold(n_splits=5)
splts = list(kf.split(spltidx))


# In[28]:


all_preds107 = []
all_preds130 = []

for i in range(5):
    dls = DataLoaders.from_dsets(seqs[splts[i][0]], seqs[splts[i][1]], bs=BS, drop_last=True, shuffle=True).cuda()
    net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None)
    learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)
    learn.fit_one_cycle(EP, LR, wd=WD)
    preds107 = learn.get_preds(dl=test_dl107, reorder=False)
    all_preds107.append(preds107[0][0])
    preds130 = learn.get_preds(dl=test_dl130, reorder=False)
    all_preds130.append(preds130[0][0])


# # Submission

# In[29]:


predictions107 = sum(all_preds107) / len(all_preds107)
predictions107.shape

predictions130 = sum(all_preds130) / len(all_preds130)
predictions130.shape


# In[30]:


s107 = pd.DataFrame()
s107['id_seqpos'] = test107_ids['id_seqpos']
s107['reactivity'] = predictions107[:,:,0].flatten().numpy().tolist()
s107['deg_Mg_pH10'] = predictions107[:,:,1].flatten().numpy().tolist()
s107['deg_pH10'] = predictions107[:,:,2].flatten().numpy().tolist()
s107['deg_Mg_50C'] = predictions107[:,:,3].flatten().numpy().tolist()
s107['deg_50C'] = predictions107[:,:,4].flatten().numpy().tolist()
s107 = s107.iloc[:-11*107]


# In[31]:


s130 = pd.DataFrame()
s130['id_seqpos'] = test130_ids['id_seqpos']
s130['reactivity'] = predictions130[:,:,0].flatten().numpy().tolist()
s130['deg_Mg_pH10'] = predictions130[:,:,1].flatten().numpy().tolist()
s130['deg_pH10'] = predictions130[:,:,2].flatten().numpy().tolist()
s130['deg_Mg_50C'] = predictions130[:,:,3].flatten().numpy().tolist()
s130['deg_50C'] = predictions130[:,:,4].flatten().numpy().tolist()
s130 = s130.iloc[:-3*130]


# In[32]:


s = pd.concat([s107, s130], axis=0)


# In[33]:


s.to_csv('submission.csv', index=False)


# In[34]:


s.head()


# # Visualize Predictions

# In[35]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


viz = pd.DataFrame()
viz['reactivity'] = predictions130[:,:,0].numpy().tolist()
viz['deg_Mg_pH10'] = predictions130[:,:,1].numpy().tolist()
viz['deg_Mg_50C'] = predictions130[:,:,3].numpy().tolist()


# In[37]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(21,9), sharex=True, sharey=True)
fig.suptitle('Reactivity', fontsize=24, color='blue')
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz['reactivity'].loc[i])
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[38]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(21,9), sharex=True, sharey=True)
fig.suptitle('deg_Mg_pH10', fontsize=24, color='blue')
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz['deg_Mg_pH10'].loc[i])
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[39]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(21,9), sharex=True, sharey=True)
fig.suptitle('deg_Mg_50C', fontsize=24, color='blue')
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz['deg_Mg_50C'].loc[i])
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[ ]:




