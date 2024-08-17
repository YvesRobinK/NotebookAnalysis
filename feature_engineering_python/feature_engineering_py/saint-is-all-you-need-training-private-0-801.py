#!/usr/bin/env python
# coding: utf-8

# Thanks for very great notebook. The notebook got many idea from the following.
# 1. https://www.kaggle.com/manikanthr5/riiid-sakt-model-training-public
# 2. https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing
# 3. https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates
# 
# The notebook will raise out of memory error. I trained the model in local machine.
# It needs around 20~30GB ram.

# In[ ]:


import psutil
import joblib
import random
import logging
from tqdm import tqdm

import numpy as np
import gc
import pandas as pd
import time

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[ ]:


# logging.basicConfig(level=logging.DEBUG, filename="logfile20.txt", filemode="a+",
#                         format="%(asctime)-15s %(levelname)-8s %(message)s")


# * feature_time_lag function is to calculate the time delta between current time and previous time. The previous time is not allowed with same **task_container_id**.

# In[ ]:


def feature_time_lag(df, time_dict):

    tt = np.zeros(len(df), dtype=np.int64)

    for ind, row in enumerate(df[['user_id','timestamp','task_container_id']].values):

        if row[0] in time_dict.keys():
            if row[2]-time_dict[row[0]][1] == 0:

                tt[ind] = time_dict[row[0]][2]

            else:
                t_last = time_dict[row[0]][0]
                task_ind_last = time_dict[row[0]][1]
                tt[ind] = row[1]-t_last
                time_dict[row[0]] = (row[1], row[2], tt[ind])
        else:
            # time_dict : timestamp, task_container_id, lag_time
            time_dict[row[0]] = (row[1], row[2], -1)
            tt[ind] =  0

    df["time_lag"] = tt
    return df


# In[ ]:


get_ipython().system('nvidia-smi')


# # Parameters

# In[ ]:


MAX_SEQ = 100
D_MODEL = 256 
N_LAYER = 2
BATCH_SIZE = 256
DROPOUT = 0.1
TIME_CAT_FLAG = True


# # Data preprocessing

# * train_df.pkl is made from tito's [CV Strategy](https://www.kaggle.com/its7171/cv-strategy) and sorted by **viretual_time_stamp**.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import pickle\nwith open("../input/saint-plus-data-new/train_df.pkl","rb") as f:\n    train_df = pickle.load(f)\n')


# In[ ]:


question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'time_dict = dict()\ntrain_df = feature_time_lag(train_df, time_dict)\n# del time_dict\n')


# In[ ]:


train_df.head()


# * Save the time_dict.pkl for inference.

# In[ ]:


# joblib.dump(time_dict,"/content/drive/MyDrive/riiid/SAINT_data/time_dict.pkl.zip")
# joblib.dump(time_dict, os.path.join(path, "time_dict.pkl.zip"))


# In[ ]:


train_df = train_df[["timestamp","user_id","content_id","content_type_id","answered_correctly","prior_question_elapsed_time","prior_question_had_explanation","time_lag"]]


# In[ ]:


train_df = train_df[train_df.content_type_id == 0]
# train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop=True)


# In[ ]:


train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.fillna(0)
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value = False).astype(int)


# In[ ]:


train_df = train_df.merge(question[["question_id","part"]], how = "left", left_on = 'content_id', right_on = 'question_id')


# * **TIME_CAT_FLAG** is flag for continuous or categorical embedding. If we choose continuous embedding, it use **rank gauss** to normalize **lag time** and **prior question elapsed time**.

# In[ ]:


if TIME_CAT_FLAG == False:
    scaler_elapsed_time = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    scaler_lag_time = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    
    raw_vec_elapsed_time = train_df.prior_question_elapsed_time.values.reshape(-1, 1)
    raw_vec_lag_time = train_df.time_lag.values.reshape(-1, 1)
    
    scaler_elapsed_time.fit(raw_vec_elapsed_time)
    scaler_lag_time.fit(raw_vec_lag_time)
    
    train_df.prior_question_elapsed_time = scaler_elapsed_time.transform(raw_vec_elapsed_time).reshape(1, -1)[0]
    train_df.time_lag = scaler_lag_time.transform(raw_vec_lag_time).reshape(1, -1)[0]
    


# In[ ]:


train = train_df.iloc[:int(95/100 * len(train_df))]
val = train_df.iloc[int(95/100 * len(train_df)):]


# In[ ]:


train.shape


# In[ ]:


val.shape


# In[ ]:


## drop outlier
# temp_df = train_df.groupby("user_id").count().reset_index()
# outlier = temp_df[(temp_df.timestamp>9000)].user_id.tolist()
# train_df = train_df[~train_df.user_id.isin(outlier)]


# In[ ]:


skills = train["content_id"].unique()
n_skill = len(skills)
print("number skills", len(skills))


# In[ ]:


n_part = len(train["part"].unique())


# In[ ]:


del train_df 
gc.collect()


# * Transform training and validation data to sequential format as input as Transformer

# In[ ]:


train_group = train[['user_id', 'content_id', 'answered_correctly', 'part', 'prior_question_elapsed_time', 'time_lag', 'prior_question_had_explanation']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values,
            r['time_lag'].values,
            r['prior_question_had_explanation'].values))


# In[ ]:


val_group = val[['user_id', 'content_id', 'answered_correctly', 'part', 'prior_question_elapsed_time', 'time_lag', 'prior_question_had_explanation']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values,
            r['time_lag'].values,
            r['prior_question_had_explanation'].values))


# In[ ]:


del train_df
gc.collect()


# * We need to save the group data for inference. In training phase, I split the data to training/validation set. However, we have to use full data to make group data for inference.
# * [Transformer group data for inference](https://www.kaggle.com/m10515009/transformer-group-data-for-inference/)

# # SAINT Dataset

# In[ ]:


class SAINTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q, qa, part, pri_elap, lag, pri_exp = group[user_id]
            if len(q) < 2:
                continue
            
            # Credit to https://www.kaggle.com/manikanthr5/riiid-sakt-model-training-public
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial], part[:initial], pri_elap[:initial], lag[:initial], pri_exp[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end], part[start:end], pri_elap[start:end], lag[start:end], pri_exp[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa, part, pri_elap, lag, pri_exp)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, part_, pri_elap_, lag_, pri_exp_ = self.samples[user_id]
        seq_len = len(q_)

        ## for zero padding
        q_ = q_+1
        pri_exp_ = pri_exp_ + 1
        res_ = qa_ + 1
        
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        res = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        pri_elap = np.zeros(self.max_seq, dtype=float)
        lag = np.zeros(self.max_seq, dtype=float)
        pri_exp = np.zeros(self.max_seq, dtype=int)

        if seq_len == self.max_seq:

            q[:] = q_
            qa[:] = qa_
            res[:] = res_
            part[:] = part_
            pri_elap[:] = pri_elap_
            lag[:] = lag_
            pri_exp[:] = pri_exp_
            
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            res[-seq_len:] = res_
            part[-seq_len:] = part_
            pri_elap[-seq_len:] = pri_elap_
            lag[-seq_len:] = lag_
            pri_exp[-seq_len:] = pri_exp_
        
        exercise = q[1:]
        part = part[1:]
        response = res[:-1]
        label = qa[1:]
        elap = pri_elap[1:]

        ## It's different from paper. The lag time including present lag time have more information. 
        lag = lag[1:]
        pri_exp = pri_exp[1:]


        return exercise, part, response, elap, lag, pri_exp, label


# In[ ]:


train_dataset = SAINTDataset(train_group, n_skill)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


val_dataset = SAINTDataset(val_group, n_skill)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


# In[ ]:


item = val_dataset.__getitem__(3)


# In[ ]:


item


# # Transformer model(SAINT+ like)

# In[ ]:


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAINTModel(nn.Module):
    def __init__(self, n_skill, n_part, max_seq=MAX_SEQ, embed_dim= 128, time_cat_flag = True):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part
        self.time_cat_flag = time_cat_flag

        self.e_embedding = nn.Embedding(self.n_skill+1, embed_dim) ## exercise
        self.c_embedding = nn.Embedding(self.n_cat+1, embed_dim) ## category
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim) ## position
        self.res_embedding = nn.Embedding(2+1, embed_dim) ## response


        if self.time_cat_flag == True:
            self.elapsed_time_embedding = nn.Embedding(300+1, embed_dim) ## elapsed time (the maximum elasped time is 300)
            self.lag_embedding1 = nn.Embedding(300+1, embed_dim) ## lag time1 for 300 seconds
            self.lag_embedding2 = nn.Embedding(1440+1, embed_dim) ## lag time2 for 1440 minutes
            self.lag_embedding3 = nn.Embedding(365+1, embed_dim) ## lag time3 for 365 days

        else:
            self.elapsed_time_embedding = nn.Linear(1, embed_dim, bias=False) ## elapsed time
            self.lag_embedding = nn.Linear(1, embed_dim, bias=False) ## lag time


        self.exp_embedding = nn.Embedding(2+1, embed_dim) ## user had explain

        self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = DROPOUT)

        self.dropout = nn.Dropout(DROPOUT)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, question, part, response, elapsed_time, lag_time, exp):

        device = question.device  

        ## embedding layer
        question = self.e_embedding(question)
        part = self.c_embedding(part)
        pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        res = self.res_embedding(response)
        exp = self.exp_embedding(exp)

        if self.time_cat_flag == True:

            ## feature engineering
            ## elasped time
            elapsed_time = torch.true_divide(elapsed_time, 1000)
            elapsed_time = torch.round(elapsed_time)
            elapsed_time = torch.where(elapsed_time.float() <= 300, elapsed_time, torch.tensor(300.0).to(device)).long()
            elapsed_time = self.elapsed_time_embedding(elapsed_time)

            ## lag_time1
            lag_time = torch.true_divide(lag_time, 1000)
            lag_time = torch.round(lag_time)
            lag_time1 = torch.where(lag_time.float() <= 300, lag_time, torch.tensor(300.0).to(device)).long()

            ## lag_time2
            lag_time = torch.true_divide(lag_time, 60)
            lag_time = torch.round(lag_time)
            lag_time2 = torch.where(lag_time.float() <= 1440, lag_time, torch.tensor(1440.0).to(device)).long()

            ## lag_time3
            lag_time = torch.true_divide(lag_time, 1440)
            lag_time = torch.round(lag_time)
            lag_time3 = torch.where(lag_time.float() <= 365, lag_time, torch.tensor(365.0).to(device)).long()

            ## lag time
            lag_time1 = self.lag_embedding1(lag_time1) 
            lag_time2 = self.lag_embedding2(lag_time2) 
            lag_time3 = self.lag_embedding3(lag_time3)
            
            enc = question + part + pos_id + exp
            dec = pos_id + res + elapsed_time + lag_time1 + lag_time2 + lag_time3
  

        else:

            elapsed_time = elapsed_time.view(-1,1)
            elapsed_time = self.elapsed_time_embedding(elapsed_time)
            elapsed_time = elapsed_time.view(-1, MAX_SEQ-1, self.embed_dim)

            lag_time = lag_time.view(-1,1)
            lag_time = self.lag_embedding(lag_time)
            lag_time = lag_time.view(-1, MAX_SEQ-1, self.embed_dim)

            # elapsed_time = elapsed_time.view(-1, MAX_SEQ-1, 1)  ## [batch, s_len] => [batch, s_len, 1]
            # elapsed_time = self.elapsed_time_embedding(elapsed_time)

            enc = question + part + pos_id + exp
            dec = pos_id + res + elapsed_time + lag_time
        

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)


# # New model or pretrained model

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = SAINTModel(n_skill, n_part, embed_dim= D_MODEL, time_cat_flag = TIME_CAT_FLAG)

# try:
#     model.load_state_dict(torch.load("./SAINT_model/saint_plus_model_20210102_padding_v2.pt"))
# except:
#     model.load_state_dict(torch.load("./SAINT_model/saint_plus_model_20210102_padding_v2.pt", map_location='cpu'))

## AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)


# In[ ]:


print(model)


# # Model training

# In[ ]:


def train_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device="cpu", time_cat_flag = True):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    start_time = time.time()

    ## training
    for item in train_dataloader:
        exercise = item[0].to(device).long()
        part = item[1].to(device).long()
        response = item[2].to(device).long()

        if time_cat_flag == True:
            elapsed_time = item[3].to(device).long()
            lag_time = item[4].to(device).long()
        else :
            elapsed_time = item[3].to(device).float()
            lag_time = item[4].to(device).float()

        exp = item[5].to(device).long()
        label = item[6].to(device).float()
        target_mask = (exercise != 0)

        optimizer.zero_grad()
        output = model(exercise, part, response, elapsed_time, lag_time, exp)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        # mask the output
        output_mask = torch.masked_select(output, target_mask)
        label_mask = torch.masked_select(label, target_mask)

        labels.extend(label_mask.view(-1).data.cpu().numpy())
        outs.extend(output_mask.view(-1).data.cpu().numpy())

    train_auc = roc_auc_score(labels, outs)
    train_loss = np.mean(train_loss)

    labels = []
    outs = []
    val_loss = []

    # validation
    model.eval()
    for item in val_dataloader:
        exercise = item[0].to(device).long()
        part = item[1].to(device).long()
        response = item[2].to(device).long()

        if time_cat_flag == True:
            elapsed_time = item[3].to(device).long()
            lag_time = item[4].to(device).long()
        else :
            elapsed_time = item[3].to(device).float()
            lag_time = item[4].to(device).float()

        exp = item[5].to(device).long()
        label = item[6].to(device).float()
        target_mask = (exercise != 0)
        
        output = model(exercise, part, response, elapsed_time, lag_time, exp)
        
        ## mask the output
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        
        loss = criterion(output, label)
        val_loss.append(loss.item())

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    val_auc = roc_auc_score(labels, outs)
    val_loss = np.mean(val_loss)

    elapsed_time = time.time() - start_time 

    return train_loss, train_auc, val_loss, val_auc, elapsed_time


# In[ ]:


epochs = 10
for epoch in range(epochs):
    train_loss, train_auc, val_loss, val_auc, elapsed_time = train_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device, time_cat_flag = TIME_CAT_FLAG)
    print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, elapsed_time))
#     logging.info("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, elapsed_time))


# In[ ]:


torch.save(model.state_dict(), "./saint_plus_model.pt")


# In[ ]:




