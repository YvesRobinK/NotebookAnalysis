#!/usr/bin/env python
# coding: utf-8

# # RIIID - LightGBM and SAKT Ensemble Inference
# 
# ### If you like this kernel or forking this kernel, please consider upvoting this and the kernels I copied (acknowledgements) from. It helps them reach more people. 
# 
# 
# ## SAKT Model
# - Public Leaderboard Score: 0.773
# - **Pretrained Dataset**: https://www.kaggle.com/manikanthr5/riiid-sakt-model-dataset-public/
# - **Acknowledgement**: All the credits go to this popular notebook https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates which is a modification of https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing. Please show some support to these original work kernels.
# - **Possible Improvements**:
#   - All the data in this notebook is used for training, so create a train and valid dataset for cross validation. Note: For me this degraded my LB score.
# - Some other text book ideas you could try:
#   - Using Label Smoothing
#   - Using Learning Rate Schedulers
#   - Increase the max sequence length and/or embedding dimension
#   - Add more attention layers 
#   
# ## LightGBM Model
# - Public Leaderboard Score: 0.760
# - **Training Notebook**: https://www.kaggle.com/manikanthr5/riiid-lgbm-single-model-ensembling-training
# - **Inference Notebook**: https://www.kaggle.com/manikanthr5/riiid-lgbm-single-model-ensembling-scoring
# - **Pretrained Dataset**: https://www.kaggle.com/manikanthr5/lgbm-with-loop-feature-engineering-dataset
# - **Acknowledgement**: I have modified this popular notebook https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering/
# - **Improvement Chances**: 
#   - This notebook is pretty simple. Try to add features. Feature Engineering is the key to improving LightGBM Models. By adding good features to the training notebook it is possible to get LB 0.771+.
#   - Beaware of target leakage

# In[1]:


import gc
import psutil
import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")


# ## LigthGBM Feature Update

# In[2]:


# funcs for user stats with loop
def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt, row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1


# ## Import RIIID API

# In[3]:


import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


# ## SAKT Model

# In[4]:


MAX_SEQ = 100

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=128): #HDKIM 100
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x, question_ids):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight
    
class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ): #HDKIM 100
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_          
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])
        
        return x, questions


# ## LightGBM Pretrained Imports
# 
# The pretrained weights are available in this dataset: https://www.kaggle.com/manikanthr5/lgbm-with-loop-feature-engineering-dataset

# In[5]:


answered_correctly_sum_u_dict = joblib.load("../input/lgbm-with-loop-feature-engineering-dataset/answered_correctly_sum_u_dict.pkl.zip")
count_u_dict = joblib.load("../input/lgbm-with-loop-feature-engineering-dataset/count_u_dict.pkl.zip")

questions_df = pd.read_feather('../input/lgbm-with-loop-feature-engineering-dataset/questions_df.feather')
content_df = pd.read_feather('../input/lgbm-with-loop-feature-engineering-dataset/content_df.feather')

prior_question_elapsed_time_mean = joblib.load("../input/lgbm-with-loop-feature-engineering-dataset/prior_question_elapsed_time_mean.pkl.zip")

TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_u', 'answered_correctly_sum_u', 'count_u', 
         'answered_correctly_avg_c', 'part', 'prior_question_had_explanation', 
         'prior_question_elapsed_time'
        ]
lgbm_model = lgb.Booster(model_file="../input/lgbm-with-loop-feature-engineering-dataset/fold0_lgb_model.txt")
lgbm_model.best_iteration = joblib.load("../input/lgbm-with-loop-feature-engineering-dataset/fold0_lgb_model_best_iteration.pkl.zip")
optimized_weights = joblib.load("../input/lgbm-with-loop-feature-engineering-dataset/optimized_weights.pkl.zip")


# In[6]:


feature_importance = lgbm_model.feature_importance()
feature_importance = pd.DataFrame(
    {'Features': FEATS, 'Importance': feature_importance}
).sort_values('Importance', ascending = False)

fig = plt.figure(figsize = (8, 6))
fig.suptitle('Feature Importance', fontsize = 20)
plt.tick_params(axis = 'x', labelsize = 12)
plt.tick_params(axis = 'y', labelsize = 12)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)
sns.barplot(
    x=feature_importance['Importance'], 
    y=feature_importance['Features'], 
    orient='h'
)
plt.show()


# ## SAKT Pretrained Model Imports

# The pretrained weights are from this dataset: https://www.kaggle.com/manikanthr5/riiid-sakt-model-dataset-public/

# In[7]:


skills = joblib.load("../input/riiid-sakt-model-dataset-public/skills.pkl.zip")
n_skill = len(skills)
print("number skills", len(skills))
group = joblib.load("../input/riiid-sakt-model-dataset-public/group.pkl.zip")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sakt_model = SAKTModel(n_skill, embed_dim=128)
try:
    sakt_model.load_state_dict(torch.load("../input/riiid-sakt-model-dataset-public/sakt_model.pt"))
except:
    sakt_model.load_state_dict(torch.load("../input/riiid-sakt-model-dataset-public/sakt_model.pt", map_location='cpu'))

sakt_model.to(device)
sakt_model.eval()


# ## Inference

# In[8]:


prev_test_df = None

for (test_df, sample_prediction_df) in iter_test:
    if (prev_test_df is not None) & (psutil.virtual_memory().percent < 90):
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
        
        # This is for SAKT
        prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))
        for prev_user_id in prev_group.index:
            if prev_user_id in group.index:
                group[prev_user_id] = (
                    np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:], 
                    np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:]
                )
 
            else:
                group[prev_user_id] = (
                    prev_group[prev_user_id][0], 
                    prev_group[prev_user_id][1]
                )
                   
        # This is for LGBM
        update_user_feats(prev_test_df, answered_correctly_sum_u_dict, count_u_dict)

    prev_test_df = test_df.copy()
    test_df = test_df[test_df.content_type_id == False]
    test_df.reset_index(drop=True)
    
    # This is for SAKT
    test_dataset = TestDataset(group, test_df, skills)
    test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)
    
    sakt_predictions = []

    for item in test_dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()

        with torch.no_grad():
            output, att_weight = sakt_model(x, target_id)
        sakt_predictions.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())
    
    # This is for LGBM
    test_df.reset_index(drop=True, inplace=True)
    test_df = add_user_feats_without_update(test_df, answered_correctly_sum_u_dict, count_u_dict)
    test_df = pd.merge(test_df, content_df, on='content_id',  how="left")
    test_df = pd.merge(test_df, questions_df, left_on='content_id', right_on='question_id', how='left')
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time_mean'] = test_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
    lgbm_predictions = lgbm_model.predict(test_df[FEATS], num_iteration=lgbm_model.best_iteration)
    
    # Simple Model Score Averaging
    test_df['answered_correctly'] =  0.5 * np.array(sakt_predictions) + 0.5 * lgbm_predictions
    env.predict(test_df[['row_id', 'answered_correctly']])


# ## Attention Weight Visualization

# In[9]:


att_weight = att_weight.detach().cpu().numpy()


# In[10]:


attention_weights = att_weight[0]

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(attention_weights, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(attention_weights, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Attention Weight")

plt.show()


# In[11]:


attention_weights = att_weight[1]

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(attention_weights, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(attention_weights, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Attention Weight")

plt.show()


# In[12]:


attention_weights = att_weight[10]

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(attention_weights, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(attention_weights, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Attention Weight")

plt.show()


# In[13]:


attention_weights = att_weight[16]

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(attention_weights, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(attention_weights, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Attention Weight")

plt.show()


# In[14]:


attention_weights = att_weight[32]

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(attention_weights, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(attention_weights, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Attention Weight")

plt.show()


# ## Observations
# 
# - Need to use mask and remove the padded cases
# - Attention has some pattern that is repeating every 10 timestamps

# In[ ]:




