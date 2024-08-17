#!/usr/bin/env python
# coding: utf-8

# This notebook is the Inference code of this [train code](https://www.kaggle.com/code/mujrush/train-deberta-v3-base-lgbm).
# The overall code is based on the greate code [here](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference).
# 
# I am not an expert in machine learning, Please let me know if there are any mistakes.
# 
# I hope this will be useful to someone!

# # Directory settings

# In[1]:


# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
MODEL_DIR = '../input/fb3-deberta-v3-base-baseline-train/'
LGB_MODEL_DIR = '../input/train-deberta-v3-base-lgbm/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# # CFG

# In[2]:


class CFG:
    num_workers=4
    path="../input/fb3-deberta-v3-base-baseline-train/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-base"
    gradient_checkpointing=False
    batch_size=24
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    max_len = 512


# # Import

# In[3]:


import os
import random
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import warnings
warnings.filterwarnings('ignore')


import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

#os.system('pip install iterative-stratification==0.1.7')
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

os.system('pip uninstall -y transformers')
os.system('pip uninstall -y tokenizers')
os.system('python -m pip install --no-index --find-links=../input/fb3-pip-wheels transformers')
os.system('python -m pip install --no-index --find-links=../input/fb3-pip-wheels tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')

from textblob import TextBlob
import nltk
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import lightgbm as lgb

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


# In[4]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# # Data Loading

# In[5]:


# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

print(f"test.shape: {test.shape}")
display(test.head())
print(f"submission.shape: {submission.shape}")
display(submission.head())


# # Text feature engineering

# In[6]:


def pos_count(sent):
    nn_count = 0
    pr_count = 0
    vb_count = 0
    jj_count = 0
    uh_count = 0
    cd_count = 0
    sent = nltk.word_tokenize(sent) 
    sent = nltk.pos_tag(sent)
    for token in sent:
        if token[1] in ['NN', 'NNP', 'NNS']: 
            nn_count += 1
        if token[1] in ['PRP','PRP$']:
            pr_count += 1
        if token[1] in ['VB','VBD', 'VBG', 'VBN']: 
            vb_count += 1
        if token[1] in ['JJ','JJR','JJS']: 
            jj_count += 1
        if token[1] in ['UH']:
            uh_count += 1
        if token[1] in ['CD']:
            cd_count += 1
    return pd.Series([nn_count, pr_count, vb_count, jj_count, uh_count, cd_count])

def contraction_count(sent):
    count=0
    count += re.subn(r"won\'t",'', sent)[1] 
    count += re.subn(r"can\'t",'', sent)[1]
    count += re.subn(r"\'re",'',sent)[1]
    count += re.subn(r"\'s", '', sent)[1]
    count += re.subn(r"\'d", '', sent)[1]
    count += re.subn(r"\'ll", '', sent)[1]
    count += re.subn(r"\'t", '', sent)[1]
    count += re.subn(r"\'ve", '', sent)[1]
    count += re.subn(r"\'m", '', sent)[1]
    return count


# In[7]:


#text feature

def text_features(df, col):
    df[f'{col}_num_words'] = df[col].apply(lambda x: len(str(x).split())) #num_words count　
    df[f'{col}_num_unique_words'] = df[col].apply(lambda x: len(set(str(x).split()))) #num_unique_words count　　
    df[f'{col}_num_chars'] = df[col].apply(lambda x: len(str(x))) #num_chars　 count
    df[f'{col}_num_stopwords'] = df[col].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords.words('english')])) #stopword count
    df[f'{col}_num_punctuations'] = df[col].apply(lambda x: len([c for c in str(x) if c in list(string.punctuation)])) #num_punctuations count
    df[f'{col}_num_words_upper'] = df[col].apply(lambda x: len([w for w in str(x) if w.isupper()])) #num_words_upper count
    df[f'{col}_num_words_title'] = df[col].apply(lambda x: len([w for w in str(x) if w.istitle()])) #num_words_title count
    df[f'{col}_mean_word_len'] = df[col].apply(lambda x: np.mean([len(w) for w in x.split()])) #mean_word_len
    df[f"{col}_num_paragraphs"] = df[col].apply(lambda x: len(x.split('\n'))) #num_paragraphs count
    df[f"{col}_num_contractions"] = df[col].apply(contraction_count) #num_contractions count
    df[f"{col}_polarity"] = df[col].apply(lambda x: TextBlob(x).sentiment[0]) #TextBlob
    df[f"{col}_subjectivity"] = df[col].apply(lambda x: TextBlob(x).sentiment[1]) #TextBlob
    df[[f'{col}_nn_count',f'{col}_pr_count',f'{col}_vb_count',f'{col}_jj_count',f'{col}_uh_count',f'{col}_cd_count']] = df[col].apply(pos_count) #pos count
    return df


# In[8]:


test = text_features(test,'full_text')
test.head()


# In[9]:


feature_cols = list(test.iloc[:,2:].columns)
feature_cols[:10]


# In[10]:


CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')


# # Dataset

# In[11]:


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['full_text'].values
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens = True,
            max_length=self.cfg.max_len,
            pad_to_max_length=True,
            truncation = True
            )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


# # model

# In[12]:


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class FeedBack3(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# # inference

# In[13]:


# ====================================================
# Helper functions
# ====================================================
def get_features(test_loader, model, device):
    model.eval()
    features = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, inputs in tk0:
        for k, v in inputs.items():
            inputs = collate(inputs)
            inputs[k] = v.to(device)
        with torch.no_grad():
            feature = model.feature(inputs)
        features.append(feature.to('cpu').numpy())
    features = np.concatenate(features)
    return features


# In[14]:


TEXT_FEATURES = [] 

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset, 
                         batch_size=CFG.batch_size, 
                         shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
for fold in range(CFG.n_fold):
    model = FeedBack3(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(MODEL_DIR + f"{CFG.model.replace('/','-')}_fold{fold}_best.pth", 
                       map_location=torch.device('cpu'))['model']
    model.load_state_dict(state)
    model.to(device)
    features = get_features(test_loader, model, device)
    TEXT_FEATURES.append(features)
    del state; gc.collect()
    torch.cuda.empty_cache()


# # LGB

# In[15]:


features = feature_cols + [f"text_{i}" for i in np.arange(768)] #deberta emb : 768


# In[16]:


len(features)


# In[17]:


def inference_single_lightgbm(test, features, model_path, fold):
    test[[f"text_{i}" for i in np.arange(768)]] = TEXT_FEATURES[fold]
    with open(model_path, 'rb') as fin:
        clf = pickle.load(fin)
    prediction = clf.predict(test[features])
    return prediction


# In[18]:


model_paths = [(fold, LGB_MODEL_DIR+f'lightgbm_fold{fold}.pkl') for fold in range(4)]
predictions = [inference_single_lightgbm(test, features, model_path, fold) for fold, model_path in model_paths]
predictions = np.mean(predictions, 0)


# # Submission

# In[19]:


test[CFG.target_cols] = predictions
submission = submission.drop(columns=CFG.target_cols).merge(test[['text_id'] + CFG.target_cols], on='text_id', how='left')
display(submission.head())
submission[['text_id'] + CFG.target_cols].to_csv('submission.csv', index=False)

