#!/usr/bin/env python
# coding: utf-8

# # Part 1 : Understanding Ion-switching

# ![](https://image.shutterstock.com/z/stock-photo-types-of-ion-channel-classification-by-gating-mechanism-of-action-voltage-gated-ligand-gated-514716034.jpg)

# in this news we can see [Ion channel VRAC enhances immune response against viruses](https://neurosciencenews.com/vrac-ion-channel-virus-16144/)

# ![](https://i2.wp.com/neurosciencenews.com/files/2020/04/ion-channel-viruses-neuroscinews.jpg?w=800&ssl=1)

# > from the image above we can see that Upon infection of cells with a DNA virus (left), viral DNA binds to the enzyme cGAS which then synthesizes the messenger molecule cGAMP. The present work shows that cGAMP can leave the cell through the anion channel VRAC and diffuses to non-infected cells in the vicinity. After entering the cell – again through VRAC – it binds to a receptor called STING and stimulates indirectly the synthesis of interferon, which leaves the cell and suppresses, after binding to a receptor, the propagation of the virus (left cell). This provides a powerful amplification of the innate immune response against DNA viruses. The image is credited to Rosa Planells-Cases.
# 
# for more detail check this : [Ion channel VRAC enhances immune response against viruses](https://neurosciencenews.com/vrac-ion-channel-virus-16144/)
# 
# From human diseases to how climate change affects plants, faster detection of ion channels could greatly accelerate solutions to major world problems.

# # imports

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# **The training data is recordings in time. At each 10,000th of a second, the strength of the signal was recorded and the number of ion channels open was recorded. It is our task to build a model that predicts the number of open channels from signal at each time step. Furthermore we are told that the data was recorded in batches of 50 seconds. Therefore each 500,000 rows is one batch. The training data contains 10 batches and the test data contains 4 batches. Let's display the number of open channels and signal strength together for each training batch.**

# # Electrophysiology: The Art of the Heart's Rhythm
# 
# **[What Happens During an Electrophysiology Study?](https://www.chestercountyhospital.org/news/health-eliving-blog/2019/february/electrophysiology-the-art-of-the-hearts-rhythm)**

# > An electrophysiology study (EPS) is a test that records your heart's electrical activity. It lets the electrophysiologist know if it's beating as it should or if you'll need treatment to get it back in rhythm.
# 
# ![](https://www.chestercountyhospital.org/-/media/images/chestercounty/news%20images/2019/arrhythmia.ashx?h=419&w=800&la=en)

# ![](https://image.slidesharecdn.com/m7s5jm7zqxq4ntnu80nv-signature-5967dd0451739e2af475d635299752bfdaf370c5b5350b71f63bc5acce8dc46d-poli-150110161535-conversion-gate01/95/ionic-equilibria-and-membrane-potential-13-638.jpg?cb=1420906606)
# 
# ![](https://image.slidesharecdn.com/m7s5jm7zqxq4ntnu80nv-signature-5967dd0451739e2af475d635299752bfdaf370c5b5350b71f63bc5acce8dc46d-poli-150110161535-conversion-gate01/95/ionic-equilibria-and-membrane-potential-15-638.jpg?cb=1420906606)

# ![](https://image.slidesharecdn.com/m7s5jm7zqxq4ntnu80nv-signature-5967dd0451739e2af475d635299752bfdaf370c5b5350b71f63bc5acce8dc46d-poli-150110161535-conversion-gate01/95/ionic-equilibria-and-membrane-potential-19-638.jpg?cb=1420906606)

# from the picture above we can see, in state 1 when Ionic Current (nA) is in 0th level means channel is closed then in state 2 we can see the graph moving toward -2 means the voltage gate is open and electric current are flowing through so the channel is open then in state 3 it is inactive as it reaches -2 level then in stage 4 it is inactivated again because it hasn't yet reach 0th level then in state 5 it gets closed as we reach at 0th level position
# 
# **I'm not expert in this field,just trying to understand. so i can be wrong,correct me in the comment box if i am wrong**
# 
# info source : [Ionic Equilibria and Membrane Potential ](https://www.slideshare.net/CsillaEgri/membrane-potential-43389721)

# # what does Membrane Protein do?

# **Membrane proteins are the gatekeepers to the cell and are essential to the function of all cells, controlling the flow of molecules and information across the cell membrane.**
# source : [Membrane Protein](https://www.sciencedirect.com/topics/medicine-and-dentistry/membrane-protein)

# [**Ion Channels and Action Potential Generation**](https://www.sciencedirect.com/topics/neuroscience/voltage-clamp)

# The voltage-clamp technique is an experimental method that allows an experimenter to control (or “command”) the desired membrane voltage of the cell. The experimenter uses a set of electronic equipment (referred to here as a voltage-clamp device) to hold the membrane voltage at a desired level (the command voltage) while measuring the current that flows across the cell membrane at that voltage. The voltage-clamp device uses a negative feedback circuit to control the membrane voltage. To do this, the equipment measures the membrane voltage and compares it with the command voltage set by the experimenter. If the measured voltage is different from the command voltage, an error signal is generated and this tells the voltage-clamp device to pass current through an electrode in the neuron in order to correct the error and set the voltage to the command level. This can be accomplished using two microelectrodes inserted into the cell, one to measure voltage and another to pass current (see Figure Below), or using one large-diameter electrode that performs both functions.
# 
# ![](https://ars.els-cdn.com/content/image/3-s2.0-B9780128153208000041-f04-10-9780128153208.jpg?_)
# 
# > Figure Description : The two-electrode voltage-clamp technique.
# > This diagram depicts the circuit that is used to clamp the voltage of a neuron and measure the current that flows at that membrane voltage.

# For more about Ion channels understanding please check this research paper : [Voltage Clamp](https://www.sciencedirect.com/topics/neuroscience/voltage-clamp) , this nature publication [Deep-Channel uses deep neural networks to detect single-molecule events from patch-clamp data](https://www.nature.com/articles/s42003-019-0729-3) and also this beautiful EDA kernel : [Ion Switching Competition : Signal EDA](https://www.kaggle.com/tarunpaparaju/ion-switching-competition-signal-eda)

# # About Data

# as from this kernel [One Feature Model](https://www.kaggle.com/cdeotte/one-feature-model-0-930) we get to know that The training data is recordings in time. At each 10,000th of a second, the strength of the signal was recorded and the number of ion channels open was recorded. It is our task to build a model that predicts the number of open channels from signal at each time step. Furthermore we are told that the data was recorded in batches of 50 seconds. Therefore each 500,000 rows is one batch. The training data contains 10 batches and the test data contains 4 batches. Let's display the number of open channels and signal strength together for each training batch.

# In[3]:


#https://www.kaggle.com/cdeotte/one-feature-model-0-930
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5)); res = 1000
plt.plot(range(0,train.shape[0],res),train.signal[0::res])
for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)
plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 
plt.title('Training Data Signal - 10 batches',size=20)
plt.show()


# In[4]:


#https://www.kaggle.com/cdeotte/one-feature-model-0-930
plt.figure(figsize=(20,5)); res = 1000
plt.plot(range(0,train.shape[0],res),train.open_channels[0::res])
for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)
plt.xlabel('Row',size=16); plt.ylabel('Channels Open',size=16); 
plt.title('Training Data Open Channels - 10 batches',size=20)
plt.show()


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


print(len(train))
print(len(test))


# In[8]:


train.describe()


# # Part 2 : Modeling
# 

# for experimenting with wavenet pytorch model i will use this kernel [Wavenet pytorch ](https://www.kaggle.com/cswwp347724/wavenet-pytorch) and in each version i will try my level best to update models performance by trying different tactics,check ChangeLog section below....

# # Before Understanding Wavenet we need to understand what's The Kalman Filtering?

# **The Kalman filtering is an amazing tool for estimating predicted values. It is an iterative mathematical process that uses set of equations and consecutive data inputs to quickly estimate the true value,position,velocity etc of the object being measured. The reason for using this is : "Let's say you 50 or 100 data points that come in one at a time,yes we can do something like find the distribution of these data points and find the average value and say we have the average value that might be close to true value but in order to do that we need whole bunch of inputs already and the good thing is the kalman filter doesn't wait for a whole bunch of inputs and it very quickly starts to narrow in to the true value by taking a few of those inputs and by understanding the variations or uncertainty of those inputs  **

# i found the attached video below best for understanding kalman filtering

# In[9]:


from IPython.core.display import HTML


HTML('''<iframe width="560" height="315" src="https://www.youtube.com/embed/CaCcOwJPytQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> ''')


# # ChangeLog
# 
# the goal is to improve the models performance,so in 
# 
# * version 1 : i will try adding Stochastic weight averaging(swa) and adamW (lb 0.939)
# * version 2 : Adam with swa_lr=0.002 (lb 0.94)
# * version 3 : Adding LSTM layer before conv2
# * version 4 : our model was never using LSTM in version 3,i am trying to add LSTM again after wave_block4 (if i am making mistakes again,please help me in the comment box) [failed : waited more than 8 hours]
# 
# * version 5 : 1 epoch for 5 fold takes  4min 23s so i will try 80 epochs instead of 150 (got lb 0.942) 
# * version 6 : trying [Wavenet with SHIFTED-RFC Proba](https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645) as [this kernel ](https://www.kaggle.com/sggpls/wavenet-with-shifted-rfc-proba) for 90 epochs and batch size = 32
# * version 7 : solving SWA issue,trying cyclicLR and  solving model bug
# * version 8 : doing res = torch.add(res, x) instead of res+x and switching back to reducelronplateau scheduler and epoch = 150,swa_lr = 0.0011, added 1 more lstm before first wave block

# [ STOCHASTIC WEIGHT AVERAGING](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
# 
# Stochastic Weight Averaging (SWA) is a simple procedure that improves generalization in deep learning over Stochastic Gradient Descent (SGD) at no additional cost, and can be used as a drop-in replacement for any other optimizer in PyTorc
# 
# ![](https://scontent.fdac6-1.fna.fbcdn.net/v/t1.0-9/59705847_2248977985403173_8149245770332110848_o.png?_nc_cat=107&_nc_sid=8024bb&_nc_ohc=PMOb2aDgLVsAX9dRes8&_nc_ht=scontent.fdac6-1.fna&oh=13176a691e130400ac1229830ffc27cc&oe=5EBDC505)

# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
#from tqdm import tqdm
from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

import os


# Any results you write to the current directory are saved as output.


# The WaveNet proposes an autoregressive learning with the help of convolutional networks with some tricks. Basically, we have a convolution window sliding on the  data, and at each step try to predict the next sample value that it did not see yet. In other words, it builds a network that learns the causal relationships between consequtive timesteps. (see below)
# 
# ![](https://miro.medium.com/max/1400/1*ABrc8jyFQ6xRubDtKE7ZBg.png)
# 
# Here, the receptive size of the network is 5: We try to predict the next sample by using last 5 steps.
# 
# ![](https://miro.medium.com/max/884/1*tlBZS9pSpk90H5jm3O3JhA.png)
# 
# Sample x_t is dependent on the previous n samples [not exactly t=1 to t=T].
# 
# ![](https://miro.medium.com/max/1400/1*ZYts5RoIwHobDqtpC66LuQ.png)
# 
# In order to increase the size of the receptive field, we can apply dilations. At each layer, number of dilation is increased by the factor of 2, hence the receptive size is 16 instead of 5.
# 
# ![](https://miro.medium.com/max/1140/1*www46FWqJCc3OZQKP_QRoQ.gif)

# ![](https://miro.medium.com/max/1400/1*H7ZkZ5Ftd0gutXZylM0Ctg.png)
# 
# **The overall model involves some stacks of dilated conv layers, nonlinear filter and gates, residual and skip connections, and last 1x1 convolutions.**

# **For more check this [WaveNet Implementation and Experiments](https://medium.com/@evinpinar/wavenet-implementation-and-experiments-2d2ee57105d5) from where i took all the informations attached above while learning WaveNet **

# In[11]:


# configurations and main hyperparammeters
EPOCHS = 150
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 5

outdir = 'wavenet_models'
flip = False
noise = False


if not os.path.exists(outdir):
    os.makedirs(outdir)



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# In[12]:


# read data
def read_data():
    train = pd.read_csv('/kaggle/input/clean-kalman/train_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/kaggle/input/clean-kalman/test_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32})
    #from https://www.kaggle.com/sggpls/wavenet-with-shifted-rfc-proba and
    # https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645
    Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")
    Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]
        
    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    #print(df)
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


def split(GROUP_BATCH_SIZE=4000, SPLITS=5):
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size=GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print(train.head())
    print('Feature Engineering Completed...')

    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=SPLITS)
    splits = [x for x in kf.split(train, train[target], group)]
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])
        new_splits.append(new_split)
    target_cols = ['open_channels']
    print(train.head(), train.shape)
    train_tr = np.array(list(train.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
    print(train.shape, test.shape, train_tr.shape)
    return train, test, train_tr, new_splits


# # wavenet 

# In[13]:


#from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


# In[14]:


import torch.nn.functional as F


# In[15]:


# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver        
class Wave_Block(nn.Module):
    
    def __init__(self,in_channels,out_channels,dilation_rates):
        super(Wave_Block,self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(in_channels,out_channels,kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))
            
    def forward(self,x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x))*F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            #x += res
            res = torch.add(res, x)
        return res
    
    

    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 128
        self.LSTM1 = nn.GRU(input_size=19,hidden_size=64,num_layers=2,batch_first=True,bidirectional=True)

        self.LSTM = nn.GRU(input_size=input_size,hidden_size=64,num_layers=2,batch_first=True,bidirectional=True)
        #self.attention = Attention(input_size,4000)
        #self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')
       
        
        self.wave_block1 = Wave_Block(128,16,12)
        self.wave_block2 = Wave_Block(16,32,8)
        self.wave_block3 = Wave_Block(32,64,4)
        self.wave_block4 = Wave_Block(64, 128, 1)
        self.fc = nn.Linear(128, 11)
            
    def forward(self,x):
        x,_ = self.LSTM1(x)
        x = x.permute(0, 2, 1)
      
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        
        #x,_ = self.LSTM(x)
        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x,_ = self.LSTM(x)
        #x = self.conv1(x)
        #print(x.shape)
        #x = self.rnn(x)
        #x = self.attention(x)
        x = self.fc(x)
        return x

   
    
class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize


    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or \
                (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0


# In[16]:


from torch.utils.data import Dataset, DataLoader
class IronDataset(Dataset):
    def __init__(self, data, labels, training=True, transform=None, seq_len=5000, flip=0.5, noise_level=0, class_split=0.0):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.training = training
        self.flip = flip
        self.noise_level = noise_level
        self.class_split = class_split
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        labels = self.labels[idx]

        return [data.astype(np.float32), labels.astype(int)]


# In[17]:


train, test, train_tr, new_splits = split()


# In[18]:


pip install torchcontrib


# In[19]:


from torchcontrib.optim import SWA
import torchcontrib


# In[20]:


model = Classifier()
model


# In[21]:


get_ipython().run_cell_magic('time', '', 'test_y = np.zeros([int(2000000/GROUP_BATCH_SIZE), GROUP_BATCH_SIZE, 1])\ntest_dataset = IronDataset(test, test_y, flip=False)\ntest_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False)\ntest_preds_all = np.zeros((2000000, 11))\n\n\noof_score = []\nfor index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):\n    print("Fold : {}".format(index))\n    train_dataset = IronDataset(train[train_index], train_tr[train_index], seq_len=GROUP_BATCH_SIZE, flip=flip, noise_level=noise)\n    train_dataloader = DataLoader(train_dataset, NNBATCHSIZE, shuffle=True,num_workers = 16)\n\n    valid_dataset = IronDataset(train[val_index], train_tr[val_index], seq_len=GROUP_BATCH_SIZE, flip=False)\n    valid_dataloader = DataLoader(valid_dataset, NNBATCHSIZE, shuffle=False)\n\n    it = 0\n    model = Classifier()\n    model = model.cuda()\n\n    early_stopping = EarlyStopping(patience=40, is_maximize=True,\n                                   checkpoint_path=os.path.join(outdir, "gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index,\n                                                                                                             it)))\n\n    weight = None#cal_weights()\n    criterion = nn.CrossEntropyLoss(weight=weight)\n    optimizer = torch.optim.Adam(model.parameters(), lr=LR)\n    optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=2, swa_lr=0.0011)\n    \n    \n\n\n    #schedular = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=LR, max_lr=0.003, step_size_up=len(train_dataset)/2, cycle_momentum=False)\n    \n    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\'max\', patience=2, factor=0.2)\n    \n    avg_train_losses, avg_valid_losses = [], []\n\n    \n\n    for epoch in range(EPOCHS):\n        \n        train_losses, valid_losses = [], []\n        tr_loss_cls_item, val_loss_cls_item = [], []\n\n        model.train()  # prep model for training\n        train_preds, train_true = torch.Tensor([]).cuda(), torch.LongTensor([]).cuda()#.to(device)\n        \n        print(\'**********************************\')\n        print("Folder : {} Epoch : {}".format(index, epoch))\n        print("Curr learning_rate: {:0.9f}".format(optimizer.param_groups[0][\'lr\']))\n        \n            #loss_fn(model(input), target).backward()\n        for x, y in tqdm(train_dataloader):\n            x = x.cuda()\n            y = y.cuda()\n            #print(x.shape)\n            \n         \n            \n            optimizer.zero_grad()\n            predictions = model(x)\n\n            predictions_ = predictions.view(-1, predictions.shape[-1])\n            y_ = y.view(-1)\n\n            loss = criterion(predictions_, y_)\n\n            # backward pass: compute gradient of the loss with respect to model parameters\n            loss.backward()\n            # perform a single optimization step (parameter update)\n            optimizer.step()\n            \n            schedular.step(loss)\n            # record training lossa\n            train_losses.append(loss.item())\n            train_true = torch.cat([train_true, y_], 0)\n            train_preds = torch.cat([train_preds, predictions_], 0)\n\n        #model.eval()  # prep model for evaluation\n        \n        optimizer.update_swa()\n        optimizer.swap_swa_sgd()\n        val_preds, val_true = torch.Tensor([]).cuda(), torch.LongTensor([]).cuda()\n        print(\'EVALUATION\')\n        with torch.no_grad():\n            for x, y in tqdm(valid_dataloader):\n                x = x.cuda()#.to(device)\n                y = y.cuda()#..to(device)\n\n                predictions = model(x)\n                predictions_ = predictions.view(-1, predictions.shape[-1])\n                y_ = y.view(-1)\n\n                loss = criterion(predictions_, y_)\n\n                valid_losses.append(loss.item())\n\n\n                val_true = torch.cat([val_true, y_], 0)\n                val_preds = torch.cat([val_preds, predictions_], 0)\n \n        \n        # calculate average loss over an epoch\n        train_loss = np.average(train_losses)\n        valid_loss = np.average(valid_losses)\n        avg_train_losses.append(train_loss)\n        avg_valid_losses.append(valid_loss)\n        print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))\n\n        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),\n                               labels=list(range(11)), average=\'macro\')\n\n        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),\n                             labels=list(range(11)), average=\'macro\')\n\n        schedular.step(val_score)\n        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))\n        res = early_stopping(val_score, model)\n        #print(\'fres:\', res)\n        if  res == 2:\n            print("Early Stopping")\n            print(\'folder %d global best val max f1 model score %f\' % (index, early_stopping.best_score))\n            break\n        elif res == 1:\n            print(\'save folder %d global val max f1 model score %f\' % (index, val_score))\n    print(\'Folder {} finally best global max f1 score is {}\'.format(index, early_stopping.best_score))\n    oof_score.append(round(early_stopping.best_score, 6))\n    \n    model.eval()\n    pred_list = []\n    with torch.no_grad():\n        for x, y in tqdm(test_dataloader):\n            \n            x = x.cuda()\n            y = y.cuda()\n\n            predictions = model(x)\n            predictions_ = predictions.view(-1, predictions.shape[-1]) # shape [128, 4000, 11]\n            #print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)\n            pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy()) # shape (512000, 11)\n            #a = input()\n        test_preds = np.vstack(pred_list) # shape [2000000, 11]\n        test_preds_all += test_preds\n   \n')


# In[22]:


print('all folder score is:%s'%str(oof_score))
print('OOF mean score is: %f'% (sum(oof_score)/len(oof_score)))
print('Generate submission.............')
submission_csv_path = '/kaggle/input/liverpool-ion-switching/sample_submission.csv'
ss = pd.read_csv(submission_csv_path, dtype={'time': str})
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds.csv", index=False)
print('over')


# **Lot more to come,i am new in this field, any suggestions in the comment box for improving this model is highly appreciated,thanks**

# In[23]:


'''x = torch.randn((16,4000, 128))
print(x.shape)
#x = x.permute(0, 2, 1)
print(x.shape)
#x = x.permute(0, 2, 1)
attention = Attention(128,4000)
attention(x)'''


# In[24]:


'''x = torch.randn((2,64,300))
print(x.shape)
#x = x.permute(0, 2, 1)
print(x.shape)
#x = x.permute(0, 2, 1)
attention = Attention(300,64)
attention(x)
#attention'''


# In[ ]:




