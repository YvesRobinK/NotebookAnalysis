#!/usr/bin/env python
# coding: utf-8

# ## Analysis and using models from three notebooks
# 
# **1.** Deberta v3 large (0.8392)
# > [Inference BERT for usPatents](https://www.kaggle.com/code/leehann/inference-bert-for-uspatents)
# 
# **2.** Deberta v3 large (0.8338)
# > [PPPM / Deberta-v3-large baseline [inference]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference)
# 
# **3.** Roberta-large (0.8143)
# > [PatentPhrase RoBERTa Inference](https://www.kaggle.com/code/santhoshkumarv/patentphrase-roberta-inference-lb-0-814)
# 
# #### Please upvote the original notebooks!
# 
# ## UPD: I have an error in my code (Version 1)!
# 
# Method merge in model 1 shuffled the dataframe.
# 
# ```
# test = test.merge(titles, left_on='context', right_on='code')
# ```
# 
# So I reseted index, merged, sorted and drop index.
# 
# ```
# test.reset_index(inplace=True)
# test = test.merge(titles, left_on='context', right_on='code')
# test.sort_values(by='index', inplace=True)
# test.drop(columns='index', inplace=True)
# ```

# # 1. Import & Def & Set & Load

# In[1]:


import os
import gc
import random

import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False

    
def inference_fn(test_loader, model, device, is_sigmoid=True):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
            
        with torch.no_grad():
            output = model(inputs)
        
        if is_sigmoid == True:
            preds.append(output.sigmoid().to('cpu').numpy())
        else:
            preds.append(output.to('cpu').numpy())

    return np.concatenate(preds)    
    

def upd_outputs(data, is_trim=False, is_minmax=False, is_reshape=False):
    min_max_scaler = MinMaxScaler()
    
    if is_trim == True:
        data = np.where(data <=0, 0, data)
        data = np.where(data >=1, 1, data)

    if is_minmax ==True:
        data = min_max_scaler.fit_transform(data)
    
    if is_reshape == True:
        data = data.reshape(-1)
        
    return data


# In[3]:


pd.set_option('display.precision', 4)
cm = sns.light_palette('green', as_cmap=True)
props_param = "color:white; font-weight:bold; background-color:green;"

CUSTOM_SEED = 42
CUSTOM_BATCH = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


competition_dir = "../input/us-patent-phrase-to-phrase-matching/"

submission = pd.read_csv(competition_dir+'sample_submission.csv')
test_origin = pd.read_csv(competition_dir+'test.csv')
test_origin.head()


# # 2. Extract predictions
# 
# ## 2.1 Deberta v3 large - 1

# In[5]:


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           max_length=cfg.max_len,
                           padding="max_length",
                           truncation=True)
    
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
        
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg        
        self.text = df['text'].values
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.text[item])
        
        return inputs
   
    
class CustomModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        self.base = AutoModelForSequenceClassification.from_config(config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim,1)
        
    def forward(self, inputs):
        output = self.base(**inputs)

        return output[0]


# In[6]:


seed_everything(CUSTOM_SEED)


# In[7]:


class CFG:
    model_path='../input/deberta-v3-large/deberta-v3-large'
    batch_size=CUSTOM_BATCH
    num_workers=2
    max_len=130
    trn_fold=[0, 1, 2, 3]

CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

context_mapping = torch.load("../input/folds-dump-the-two-paths-fix/cpc_texts.pth")


# In[8]:


test = test_origin.copy()
titles = pd.read_csv('../input/cpc-codes/titles.csv')

test.reset_index(inplace=True)
test = test.merge(titles, left_on='context', right_on='code')
test.sort_values(by='index', inplace=True)
test.drop(columns='index', inplace=True)

test['context_text'] = test['context'].map(context_mapping)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
test['text'] = test['text'].apply(str.lower)

test.head()


# In[9]:


deberta_predicts_1 = []

test_dataset = TestDataset(CFG, test)
test_dataloader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size, shuffle=False,
                             num_workers=CFG.num_workers,
                             pin_memory=True, drop_last=False)

deberta_simple_path = "../input/us-patent-deberta-simple/microsoft_deberta-v3-large"

for fold in CFG.trn_fold:
    fold_path = f"{deberta_simple_path}_best{fold}.pth"
    
    model = CustomModel(CFG.model_path)    
    state = torch.load(fold_path, map_location=torch.device('cpu'))  # DEVICE
    model.load_state_dict(state['model'])
    
    prediction = inference_fn(test_dataloader, model, DEVICE, is_sigmoid=False)
    
    deberta_predicts_1.append(prediction)
    
    del model, state, prediction
    torch.cuda.empty_cache()
    gc.collect()


# In[10]:


# -------------- inference_fn([...], is_sigmoid=False)
deberta_predicts_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in deberta_predicts_1]
deberta_predicts_1 = pd.DataFrame(deberta_predicts_1).T

deberta_predicts_1.head(10).style.background_gradient(cmap=cm, axis=1)


# In[11]:


del test, test_dataset
gc.collect()


# ## 2.2 Deberta v3 large - 2

# In[12]:


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
        
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs

    
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
            
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
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
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        
        return output


# In[13]:


seed_everything(CUSTOM_SEED)


# In[14]:


class CFG:
    num_workers=2
    path="../input/pppm-deberta-v3-large-baseline-w-w-b-train/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-large"
    batch_size=CUSTOM_BATCH
    fc_dropout=0.2
    target_size=1
    max_len=133
    trn_fold=[0, 1, 2, 3]
    
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')

context_mapping = torch.load(CFG.path+"cpc_texts.pth")


# In[15]:


test = test_origin.copy()

test['context_text'] = test['context'].map(context_mapping)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']

test.head()


# In[16]:


deberta_predicts_2 = []

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers,
                         pin_memory=True, drop_last=False)

folds_path = CFG.path + f"{CFG.model.replace('/', '-')}"

for fold in CFG.trn_fold:
    fold_path = f"{folds_path}_fold{fold}_best.pth"
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(fold_path, map_location=torch.device('cpu'))  # DEVICE
    model.load_state_dict(state['model'])
    
    prediction = inference_fn(test_loader, model, DEVICE)
    deberta_predicts_2.append(prediction)
    
    del model, state, prediction
    torch.cuda.empty_cache()
    gc.collect()


# In[17]:


deberta_predicts_2 = [upd_outputs(x, is_reshape=True) for x in deberta_predicts_2]
deberta_predicts_2 = pd.DataFrame(deberta_predicts_2).T

deberta_predicts_2.head(10).style.background_gradient(cmap=cm, axis=1)


# In[18]:


del test, test_dataset
gc.collect()


# ## 2.3. Roberta-large

# In[19]:


def prepare_input(cfg, text, target):
    inputs = cfg.tokenizer(text, target,
                           padding="max_length",
                           max_length=cfg.max_len,
                           truncation=True)

    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
        
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values
        self.target = df['target'].values
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        target = self.target[item]
        
        inputs = prepare_input(self.cfg, text, target)
        
        return inputs

    
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(CFG.config_path)

        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": hidden_dropout_prob,
                       "layer_norm_eps": layer_norm_eps,
                       "add_pooling_layer": False})
        
        self.transformer = AutoModel.from_pretrained(CFG.config_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, CFG.num_targets)
        
    def forward(self, inputs):
        transformer_out = self.transformer(**inputs)
        last_hidden_states = transformer_out[0]
        last_hidden_states = self.dropout(torch.mean(last_hidden_states, 1))
        logits1 = self.output(self.dropout1(last_hidden_states))
        logits2 = self.output(self.dropout2(last_hidden_states))
        logits3 = self.output(self.dropout3(last_hidden_states))
        logits4 = self.output(self.dropout4(last_hidden_states))
        logits5 = self.output(self.dropout5(last_hidden_states))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        return logits


# In[20]:


seed_everything(CUSTOM_SEED)


# In[21]:


@dataclass(frozen=True)
class CFG:
    num_workers=2
    config_path='../input/robertalarge'
    model_path='../input/phrase-matching-roberta-training-pytorch-wandb'
    model_name='roberta-large'
    batch_size=CUSTOM_BATCH
    max_len=128
    num_targets=1
    trn_fold=[0, 1, 2, 3, 4]
    tokenizer=AutoTokenizer.from_pretrained('../input/robertalarge')

context_mapping = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies",
}


# In[22]:


test = test_origin.copy()

test['context_text'] = test['context'].str.slice(stop=1).map(context_mapping)
test['text'] = test['context_text'] + ' ' + test['anchor']


# In[23]:


test.head()


# In[24]:


roberta_predicts = []

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers,
                         pin_memory=True, drop_last=False)

folds_path = CFG.model_path + f"/{CFG.model_name.replace('-','_')}"

for fold in CFG.trn_fold:
    fold_path = f"{folds_path}_patent_model_{fold}.pth"
    
    model = CustomModel()
    state = torch.load(fold_path, map_location=torch.device('cpu'))  # DEVICE
    model.load_state_dict(state)

    prediction = inference_fn(test_loader, model, DEVICE)
    roberta_predicts.append(prediction)
    
    del model, state, prediction
    torch.cuda.empty_cache()    
    gc.collect()


# In[25]:


roberta_predicts = [upd_outputs(x, is_reshape=True) for x in roberta_predicts]
roberta_predicts = pd.DataFrame(roberta_predicts).T

roberta_predicts.head(10).style.background_gradient(cmap=cm, axis=1)


# In[26]:


del test, test_dataset
gc.collect()


# # 3. Comparison / Ensemble

# In[27]:


all_predictions = pd.concat(
    [deberta_predicts_1, deberta_predicts_2, roberta_predicts],
    keys=['deberta 1', 'deberta 2', 'roberta'],
    axis=1
)

all_predictions.head(10) \
    .assign(mean=lambda x: x.mean(axis=1)) \
        .style.background_gradient(cmap=cm, axis=1)


# In[28]:


all_mean = pd.DataFrame({
    'deberta 1': deberta_predicts_1.mean(axis=1),
    'deberta 2': deberta_predicts_2.mean(axis=1),
    'roberta': roberta_predicts.mean(axis=1)
})

all_mean.head(10) \
    .assign(mean=lambda x: x.mean(axis=1)) \
        .style.highlight_max(axis=1, props=props_param)


# In[29]:


# === N1 ===
# weights_ = [0.33, 0.33, 0.33]
# final_predictions = all_mean.mul(weights_).sum(axis=1)

# === N2 ===
# final_predictions = all_mean.median(axis=1)
final_predictions = all_mean.mean(axis=1)

# === N3 ===
# final_predictions = all_predictions.mean(axis=1)

# === N4 ===
# combs = pd.DataFrame({
#     'deberta_1': deberta_predicts_1.mean(axis=1),
#     'deb_2+rob': (deberta_predicts_2.mean(axis=1) * 0.666) \
#                     + (roberta_predicts.mean(axis=1) * 0.333)
# })
# display(combs.head())
# final_predictions = combs.median(axis=1)
# final_predictions = combs.mean(axis=1)

final_predictions.head()


# # 4. Submission

# In[30]:


submission = pd.DataFrame({
    'id': test_origin['id'],
    'score': final_predictions,
})

submission.head(14)


# In[31]:


# ===================  Baseline
# 0  4112d61851461f60  0.56127
# 1  09e418c93a776564  0.72176
# 2  36baf228038e314b  0.47086
# 3  1f37ead645e7f0c8  0.25826
# 4  71a5b6ad068d531f  0.00908
# 5  474c874d0c07bd21  0.48173


# In[32]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




