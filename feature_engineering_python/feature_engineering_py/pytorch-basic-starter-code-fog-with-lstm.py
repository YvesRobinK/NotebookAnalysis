#!/usr/bin/env python
# coding: utf-8

# **Thaks to DANIEL, DATAMANYO**
# ---------------
# - This notebook mainly check variable with Ref video/document and Guide for pytorch model
# - Any comment is welcome!

# # Library and Data load

# In[1]:


# import library
import numpy as np
import pandas as pd
from sklearn import *
import xgboost
import glob
from sklearn.preprocessing import RobustScaler, normalize
import lightgbm as lgb

p = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'

subjects = pd.read_csv(p+'subjects.csv') #	Subject 	Visit 	Age 	Sex 	YearsSinceDx 	UPDRSIII_On 	UPDRSIII_Off 	NFOGQ
events = pd.read_csv(p+'events.csv') #	Id 	Init 	Completion 	Type 	Kinetic
tasks = pd.read_csv(p+'tasks.csv') #Id 	Begin 	End 	Task
daily = pd.read_csv(p+'daily_metadata.csv') # 	Id 	Subject 	Visit 	Beginning of recording [00:00-23:59]
meta = pd.read_csv(p+'tdcsfog_metadata.csv') #Id 	Subject 	Visit 	Test 	Medication
defog = pd.read_csv(p+'defog_metadata.csv') #Id 	Subject 	Visit 	Medication
sub = pd.read_csv(p+'sample_submission.csv') #	Id 	StartHesitation 	Turn 	Walking
#unlabeled = glob.glob(p+'unlabeled/**')
train = glob.glob(p+'train/**/**') #Time 	AccV 	AccML 	AccAP 	StartHesitation 	Turn 	Walking 	Valid 	Task
test = glob.glob(p+'test/**/**') #Time 	AccV 	AccML 	AccAP


# In[2]:


# func read data 
def reader(f):
    df = pd.read_csv(f)
    df['Id'] = f.split('/')[-1].split('.')[0]
    return df
# read train data
train = pd.concat([reader(f) for f in train]).fillna(0); print(train.shape)
cols = [c for c in train.columns if c not in ['Id', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']] # except categorical and target fetaure


# In[3]:


train.head()


# In[4]:


# read test data
test = pd.concat([reader(f) for f in test]).fillna(0); print(test.shape)


# In[5]:


# Reduce Memory Usage
# reference : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 @ARJANGROEN

def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df


# In[6]:


train = reduce_memory_usage(train)
test = reduce_memory_usage(test)


# # Feature engineering

# In[7]:


def sqrt_df(x):
    return np.sqrt(abs(x))


def feature_engineering(x):
    # moving average
    # x[["AccV_ma","AccML_ma","AccAP_ma"]] = x[["AccV","AccML","AccAP"]].rolling(window=2).mean()
    
    # delta with time
    # x["AccV_delta"] = (x.AccV - x.AccV.shift()).fillna(0)
    # x["AccML_delta"] = (x.AccML - x.AccML.shift()).fillna(0)
    # x["AccAP_delta"] = (x.AccAP - x.AccAP.shift()).fillna(0)
    
    # stride
    x["Stride"] = x["AccV"] + x["AccML"] + x["AccAP"]
    
    # step
    x["Step"] = x["Stride"].apply(sqrt_df)
    
    # fillna    
    cols = [c for c in x.columns if c not in ['Id', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']] # renew cols for new data from feature engineering
    x[cols] = x[cols].fillna(0)
    return x


# In[8]:


# add feature to train dataset
train = feature_engineering(train)
train.head()


# In[9]:


# add feature to test dataset
test = feature_engineering(test)


# *Through the Viedo(The video from description of this competiton) of walking patterns commonly observed in patients with Parkinson's disease, the patient's body was bent forward, the stride was short, and the soles of the feet were attracted to the ground.
# At the same time, it was confirmed that people walk a lot with short strides in situations such as start point and turn.*

# ![image.png](attachment:5b22e00c-66d2-4e04-b541-9d5b2f0beb89.png)  
# - sample of gait cycle(https://www.orthobullets.com/foot-and-ankle/7001/gait-cycle)  
# 
# I guess step is key feature of freezing of gait

# ### Check the event with moving average of step

# In[10]:


train[train["Turn"]==1]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# In[11]:


train[train["Turn"]==0]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# *When a Parkinson's disease patient turns, Patient moves a little and repeatedly  
# So, I guesses that the value measured by the sensor system was small on the protrusion*

# In[12]:


train[train["Walking"]==1]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# In[13]:


train[train["Walking"]==0]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# *If a patient with Parkinson's disease has a walking disorder, it is difficult to take a quick step, so it is assumed that the value measured by the sensor system is small.*

# In[14]:


train[train["StartHesitation"]==1]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# In[15]:


train[train["StartHesitation"]==0]["Step"].rolling(window=10,min_periods=1).mean().head(10)


# *The sensor system estimates that the value measured by Parkinson's disease patients is larger because their upper body moves more than their legs when they start walking.*

# In[16]:


# add feature
train["Step_ma10"] = train["Step"].rolling(window=10,min_periods=1).mean()
test["Step_ma10"] = test["Step"].rolling(window=10,min_periods=1).mean()


# # Model train 

# In[17]:


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # for using gpu


# In[18]:


cols = [c for c in train.columns if c not in ['Id', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']] # for drop the target!


# In[19]:


# Data split
x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train[['StartHesitation', 'Turn' , 'Walking']],shuffle=False, test_size=.3, random_state=3)


# In[20]:


#dataset
from torch.utils.data import Dataset

class dataset_build(Dataset):
    def __init__(self,x,y):
        # make pytorch tesnsor
        self.x = torch.tensor(x.to_numpy(),dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(),dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

train_dataset = dataset_build(x1,y1)
valid_dataset = dataset_build(x2,y2)


# In[21]:


class dataset_build_test(Dataset):
    def __init__(self,x):
        # make pytorch tesnsor
        self.id_num = x["Id"].to_numpy()
        self.x = torch.tensor(x[cols].to_numpy(),dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.id_num[idx]
  
    def __len__(self):
        return self.len


# In[22]:


cols_ = [c for c in train.columns if c not in ['StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']] # for drop the target!
test_dataset = dataset_build_test(test[cols_])


# In[33]:


#dataloader
from torch.utils.data import DataLoader 

train_loader = DataLoader(train_dataset,shuffle=False,drop_last=True, batch_size=256)
test_loader = DataLoader(test_dataset,shuffle=False, batch_size=32)


# In[34]:


#neural network
from torch import nn

class neural_network(nn.Module):
    def __init__(self):
        super(neural_network,self).__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(in_features=5,out_features=3)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output

model = neural_network().to(device)


# In[37]:


# optimizer , loss
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
epochs = 7


# In[39]:


#training loop
for i in range(epochs):
    loss_sum = 0
    for j,data in enumerate(train_loader):
        train_x,train_y = data[:][0],data[:][1]
        y_pred = model(train_x.reshape(-1,7,1).to(device))
        y_pred = y_pred
        loss = criterion(y_pred,train_y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        loss_sum += loss.item()
        
    if i%2 == 0:
        print(i,"th iteration : ",loss_sum/len(train_loader))

## Submit


# In[44]:


# 0 th iteration :  0.7242311756713296
# 2 th iteration :  0.5415780932241675
# 4 th iteration :  0.36264049582903135
# 6 th iteration :  0.2817067605714282


# In[40]:


# predict test dataset
submission = []
id_num = []
for test_data in test_loader:
    y_test_pred = model(test_data[:][0].reshape(-1,7,1).to(device))
    y_test_pred = y_test_pred.reshape(-1,3)
    y_test_pred = torch.clip(y_test_pred,0,1)
    submission.extend(y_test_pred.detach().cpu().numpy())
    id_num.extend(test_data[:][1])


# In[42]:


preds = np.array(submission)
sub[['StartHesitation', 'Turn' , 'Walking']]=preds
sub["Id"] = np.array(id_num)
sub.to_csv("submission.csv", index=False)


# In[43]:


sub[['Id','StartHesitation', 'Turn' , 'Walking']].head()


# **The END**
# ---------------
# - This notebook shows the process of implementing the model using the basic of Pytorch.
# - I hope that you like any point in this notebook
