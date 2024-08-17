#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">U.S. PATENT PHRASE TO PHRASE MATCHING</p>

# <h1 align='center'>Introduction 📝</h1>
# The aim of this competition is to predict the similarity of the pairs of phrases which is an anchor and a target phrase on a scale from 0 (not at all similar) to 1 (identical in meaning) within a patent's context.
# 
# ##  <font color="red"> Please consider upvoting the kernel if you find it useful.</font>

# <h1 align='center'>Dataset Info 📈</h1>
# <b>Columns of the train data-</b> 
# 
# * ```id``` -  a unique identifier for a pair of phrases
# * ```anchor``` - the first phrase
# * ```target``` - the second phrase
# * ```context``` - the CPC classification (version 2021.05), which indicates the subject within which the similarity is to be scored
# * ```score``` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# <h1 align='center'>Evaluation Metric 📐</h1>
# Submissions are evaluated on the Pearson correlation coefficient between the predicted and actual similarity scores.
# 
# <img src='https://user-images.githubusercontent.com/55939250/151697692-562f6439-170a-4869-856d-eaa11b2da5f5.jpg' width=500px>
# 
# where,<br> 
# * r = Pearson Correlation Coefficient
# * n = Number of samples
# * x = First variable samples
# * y = Second variable samples

# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">TABLE OF CONTENTS</p>
# <ul style="list-style-type:square">
#     <li><a href="#1">Importing Libraries</a></li>
#     <li><a href="#2">Reading the data</a></li>
#     <li><a href="#3">Exploratory Data Analysis</a></li>
#     <ul style="list-style-type:disc">
#         <li><a href="#3.1">Score</a></li>
#         <li><a href="#3.2">Anchor</a></li>
#         <li><a href="#3.3">Target</a></li>
#         <li><a href="#3.4">Context</a></li>
#         <li><a href="#3.5">CPC Code Description</a></li>
#     </ul>
#     <li><a href="#4">Create Folds</a></li>
#     <li><a href="#5">Dataset Class</a></li>
#     <li><a href="#6">Baseline Model</a></li>
#     <li><a href="#7">Utility Functions</a></li>
#     <li><a href="#8">Training</a></li>
# </ul>

# <a id='1'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">IMPORTING LIBRARIES</p>

# In[1]:


import os
import gc
import random
import requests
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
from PIL import Image
from tqdm import tqdm
from scipy import stats
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup

import warnings
warnings.simplefilter('ignore')


# In[2]:


class CONFIG:
    seed=2022
    num_fold = 5
    model = 'anferico/bert-for-patents'
    max_len = 16
    train_batch_size = 16
    valid_batch_size = 32
    epochs = 2
    learning_rate = 1e-5
    scheduler = 'linear'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG.tokenizer = AutoTokenizer.from_pretrained(CONFIG.model)


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(CONFIG.seed)


# <a id='2'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">READING THE DATA</p>

# In[4]:


train = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
test = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
sub = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')


# In[5]:


train.head()


# In[6]:


train.info()


# <a id='3'></a>
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">EXPLORATORY DATA ANALYSIS</p>

# <a id='3.1'></a>
#     
# ## 1. Score

# In[7]:


fig, ax = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Distribution of Score', size=20)

sns.countplot(x='score', data=train, palette='hls', ax=ax[0])

sizes = []
no_annotations = len(train[train['score']==0])
sizes.append(no_annotations)
annotated = len(train[train['score']!=0])
sizes.append(annotated)

print('Number of Rows having a score of 0 -', no_annotations)
print('Number of Rows having score greater than 0 -', annotated)

labels = ['Score = 0', 'Score > 0']
colors = ['#db0400', '#f7b211']
ax[1].pie(sizes, colors=colors, startangle=90, labels=labels,
        autopct='%.2f%%', pctdistance=0.7,textprops={'fontsize':20}, counterclock=False)

plt.show()


# <a id='3.2'></a>
# ## 2. Anchor

# In[8]:


top = Counter([anc for anc in train['anchor']])

top = dict(top.most_common(50))

plt.figure(figsize=(20, 6))

sns.barplot(x=list(top.keys()), y=list(top.values()), palette='hls')
plt.xticks(rotation=90)
plt.title("Top 50 First Phrases (Anchor)", fontsize=20)

plt.show()


# In[9]:


fig, ax = plt.subplots(2, 1, figsize=(20, 12))

text_len = train['anchor'].str.split().map(lambda x : len(x))

sns.countplot(text_len, ax=ax[0])
ax[0].set_title("Word Count Distribution", size=20)

avg_word_len = train['anchor'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x : np.mean(x))
sns.histplot(avg_word_len, ax=ax[1], kde=True, color='#ffa408')
ax[1].set_title('Average Word Length Distribution', size=20)

plt.show()


# In[10]:


plt.figure(figsize=(10, 10))
text = train['anchor'].values
url = 'https://static.vecteezy.com/system/resources/previews/000/263/280/non_2x/vector-open-book.jpg'
im = np.array(Image.open(requests.get(url, stream=True).raw))
cloud = WordCloud(stopwords = STOPWORDS,
                  background_color='black',
                  mask = im,
                  max_words = 200,
                  ).generate(" ".join(text))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# <a id='3.3'></a>
# ## 3. Target

# In[11]:


fig, ax = plt.subplots(2, 1, figsize=(20, 12))

text_len = train['target'].str.split().map(lambda x : len(x))

sns.countplot(text_len, ax=ax[0])
ax[0].set_title("Word Count Distribution", size=20)

avg_word_len = train['target'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x : np.mean(x))
sns.histplot(avg_word_len, ax=ax[1], kde=True, color='#ffa408')
ax[1].set_title('Average Word Length Distribution', size=20)

plt.show()


# In[12]:


plt.figure(figsize=(10, 10))
text = train['target'].values
url = 'https://static.vecteezy.com/system/resources/previews/000/263/280/non_2x/vector-open-book.jpg'
im = np.array(Image.open(requests.get(url, stream=True).raw))
cloud = WordCloud(stopwords = STOPWORDS,
                  background_color='black',
                  mask = im,
                  max_words = 300,
                  ).generate(" ".join(text))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# <a id='3.4'></a>
# ## 4. Context

# In[13]:


plt.figure(figsize=(20, 6))

sns.countplot(x='context', data=train, palette='hls')
plt.xticks(rotation=90)
plt.title("Distribution of Context", fontsize=20)

plt.show()


# The first letter is the "section symbol" consisting of a letter from "A" ("Human Necessities") to "H" ("Electricity") or "Y" for emerging cross-sectional technologies. This is followed by a two-digit number to give a "class symbol" ("A01" represents "Agriculture; forestry; animal husbandry; trapping; fishing").
# 
# * A: Human Necessities
# * B: Operations and Transport
# * C: Chemistry and Metallurgy
# * D: Textiles
# * E: Fixed Constructions
# * F: Mechanical Engineering
# * G: Physics
# * H: Electricity
# * Y: Emerging Cross-Sectional Technologies

# In[14]:


train['section'] = train['context'].astype(str).str[0]
train['classes'] = train['context'].astype(str).str[1:]


# In[15]:


sections = {"A" : "A - Human Necessities", 
            "B" : "B - Operations and Transport",
            "C" : "C - Chemistry and Metallurgy",
            "D" : "D - Textiles",
            "E" : "E - Fixed Constructions",
            "F" : "F- Mechanical Engineering",
            "G" : "G - Physics",
            "H" : "H - Electricity",
            "Y" : "Y - Emerging Cross-Sectional Technologies"}


# In[16]:


plt.figure(figsize=(15, 5))

sns.countplot(x='section', data=train, palette='rainbow', order = list(sections.keys())[:-1])
plt.xticks([0, 1,2, 3, 4, 5, 6, 7], list(sections.values())[:-1], rotation='vertical')
plt.title("Distribution of Section", fontsize=20)

plt.show()


# In[17]:


plt.figure(figsize=(20, 6))

sns.histplot(x='score', hue='section', data=train, bins=10, multiple="stack")
plt.title("Distribution of Score with respect to Sections", fontsize=20)

plt.show()


# In[18]:


plt.figure(figsize=(20, 6))

sns.countplot(x='classes', data=train, palette='Spectral')
plt.xticks(rotation=90)
plt.title("Distribution of Classes", fontsize=20)

plt.show()


# <a id='3.5'></a>
# ## 5. CPC-Code Description
# Additional Data - https://www.kaggle.com/datasets/xhlulu/cpc-codes

# In[19]:


df_cpc = pd.read_csv('../input/cpc-codes/titles.csv')
df_cpc.head(5)


# In[20]:


train['title'] = train['context'].map(df_cpc.set_index('code')['title']).str.lower()


# In[21]:


plt.figure(figsize=(10, 10))
text = train['title'].values
url = 'https://static.vecteezy.com/system/resources/previews/000/263/280/non_2x/vector-open-book.jpg'
im = np.array(Image.open(requests.get(url, stream=True).raw))
cloud = WordCloud(stopwords = STOPWORDS,
                  background_color='black',
                  mask = im,
                  max_words = 200,
                  ).generate(" ".join(text))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# <a id='4'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">CREATE FOLDS</p>

# In[22]:


encoder = LabelEncoder()
train['score_encoded'] = encoder.fit_transform(train['score'])

skf = StratifiedKFold(n_splits=CONFIG.num_fold, shuffle=True, random_state=CONFIG.seed)

for k, (_, val_ind) in enumerate(skf.split(X=train, y=train['score_encoded'])):
    train.loc[val_ind, 'fold'] = k


# <a id='5'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">DATASET CLASS</p>

# In[23]:


class USPatentDataset(Dataset):
    def __init__(self, df):
        self.anchor = df['anchor']
        self.target = df['target']
        self.score = df['score']
        
    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self, index):
        anchor = self.anchor[index]
        target = self.target[index]
        score = self.score[index]
        
        inputs = CONFIG.tokenizer(anchor, target, padding='max_length', max_length=CONFIG.max_len, truncation=True)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(score, dtype=torch.float),
        }


# <a id='6'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">BASELINE MODEL</p>

# In[24]:


class USPatentModel(nn.Module):
    def __init__(self):
        super(USPatentModel, self).__init__()
        
        config = AutoConfig.from_pretrained(CONFIG.model)
        
        self.bert = AutoModel.from_pretrained(CONFIG.model, config=config)
        self.drop = nn.Dropout(0.2)
        self.output = nn.Linear(config.hidden_size, 1)
        
    def forward(self, *inputs):
        x = self.bert(*inputs)[1]
        x = self.drop(x)
        x = self.output(x)
        
        return x


# <a id='7'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">UTILITY FUNCTIONS</p>

# In[25]:


# Function to get data according to the folds
def get_data(fold):
    train_df = train[train['fold'] != fold].reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)
    
    train_dataset = USPatentDataset(train_df)
    valid_dataset = USPatentDataset(valid_df)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size, shuffle=False)
    
    return train_loader, valid_loader

def loss_fn(outputs, labels):
    return nn.MSELoss()(outputs, labels)

def compute_pearson(outputs, labels):
    # Squash values between 0 to 1
    outputs[outputs < 0] = 0
    outputs[outputs > 1] = 1
    
    # Round off to nearest 0.25 factor
    outputs = 0.25 * np.round(outputs/0.25) 
    
    pearsonr = stats.pearsonr(outputs, labels)[0]
    return pearsonr

def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=CONFIG.learning_rate)

    return optimizer

def get_scheduler(cfg, optimizer, train_loader):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int((len(train_loader)*CONFIG.epochs*6)/100),
            num_training_steps=CONFIG.epochs*len(train_loader),
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int((len(train_loader)*CONFIG.epochs*6)/100),
            num_training_steps=CONFIG.epochs*len(train_loader),
        )
    return scheduler


# In[26]:


def train_fn(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    
    running_loss = 0
    preds = []
    label=[]
    progress_bar = tqdm(data_loader, position=0)
    
    for step, data in enumerate(progress_bar):
        ids = data['ids'].to(device)
        masks = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['label'].to(device)
        
        outputs = model(ids, masks, token_type_ids)
        loss = loss_fn(outputs.view(-1, 1), labels.view(-1, 1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        
        preds.extend(outputs.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())
        
        progress_bar.set_description(f"Epoch [{epoch+1}/{CONFIG.epochs}]")
        progress_bar.set_postfix(loss=running_loss/(step+1))
    
    train_pearson = compute_pearson(np.array(preds), np.array(label))
    
    return train_pearson

def valid_fn(model, data_loader, device, epoch):
    model.eval()
    
    running_loss = 0
    preds = []
    label = []
    progress_bar = tqdm(data_loader, position=0)
    
    for step, data in enumerate(progress_bar):
        ids = data['ids'].to(device)
        masks = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['label'].to(device)

        outputs = model(ids, masks, token_type_ids)
        loss = loss_fn(outputs.view(-1, 1), labels.view(-1, 1))

        running_loss += loss.item()

        preds.extend(outputs.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())

        progress_bar.set_description(f"Epoch [{epoch+1}/{CONFIG.epochs}]")
        progress_bar.set_postfix(loss=running_loss/(step+1))
    
    valid_pearsonr = compute_pearson(np.array(preds), np.array(label))
    
    return valid_pearsonr


# In[27]:


def run(fold):
    train_loader, valid_loader = get_data(fold)
    
    model = USPatentModel().to(CONFIG.device)
    
    optimizer = get_optimizer(model)
    
    scheduler = get_scheduler(CONFIG, optimizer, train_loader)
    
    best_valid_pearson = 0
    for epoch in range(CONFIG.epochs):
        train_pearson = train_fn(model, train_loader, optimizer, scheduler, CONFIG.device, epoch)
        valid_pearson = valid_fn(model, valid_loader, CONFIG.device, epoch)
        print(f"Train Pearson Coeff - {train_pearson}, Valid Pearson Coeff - {valid_pearson}")
        if valid_pearson > best_valid_pearson:
            print(f"Validation F1 Improved - {best_valid_pearson} ---> {valid_pearson}")
            torch.save(model.state_dict(), f'./model_{fold}.bin')
            print(f"Saved model checkpoint at ./model_{fold}.bin")
            best_valid_pearson = valid_pearson
    
    return best_valid_pearson


# <a id='8'></a>
# 
# # <p style="background-color:#FFE67C;font-family:fantasy;color:#295F2D;font-size:150%;text-align:center;border-radius:20px 40px;">TRAINING</p>

# In[28]:


for fold in range(CONFIG.num_fold):
    print("=" * 30)
    print("Training Fold - ", fold)
    print("=" * 30)
    best_valid_pearson = run(fold)
    print(f'Best Pearson Correlation Coefficient: {best_valid_pearson:.5f}')
    
    gc.collect()
    torch.cuda.empty_cache()    
    break # To run for all the folds, just remove this break


# ## References
# * [EDA and feature engineering](https://www.kaggle.com/code/remekkinas/eda-and-feature-engineering)
# * [tez training phrase matching](https://www.kaggle.com/code/abhishek/tez-training-phrase-matching)

# <div class="alert alert-block alert-info">
# If you are a beginner to NLP then I would refer my another notebook and it will definitely help you to start in NLP:-
# </div>
# <div class="row" align="center">
#     <div class = "card">
#       <div class = "card-body" style = "width: 20rem; ">
#         <h5 class = "card-title" style = "font-size: 1.2em;"align="center">Natural Language Processing</h5>
#           <img src="https://www.asksid.ai/wp-content/uploads/2021/02/an-introduction-to-natural-language-processing-with-python-for-seos-5f3519eeb8368.png" class = "card_img-top" style = "padding: 2% 0;width:19rem;height:10rem;border-radius:30%">
#         <p class="card-text" style = "font-size: 1.0em;text-align: center "><b>(Most) NLP Techniques📚</b></p>
#         <a href = "https://www.kaggle.com/utcarshagrawal/commonlit-eda-most-nlp-techniques" class = "btn btn-info btn-lg active"  role = "button" style = "color: white; margin: 0 15% 0 25%" data-toggle = "popover" title = "Click">Click here</a>
#       </div>
#     </div>
#   </div>

# <div class="alert alert-block alert-info">
#     <h2 align='center'>Please consider upvoting the kernel if you found it useful.</h2>
# </div>
