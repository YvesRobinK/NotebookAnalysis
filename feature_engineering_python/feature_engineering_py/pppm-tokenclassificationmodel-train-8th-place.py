#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# Thie notebook is a code for training a token classification model that I wrote about in <a href="https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332492" target="_blank">this post</a>.
# 
# As you will see,  I referred a lot from @yasufuminakama's notebook. If you think this notebook helpful, please upvote <a href="https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-w-w-b-train" target="_blank">his notebook</a>.
# 

# ## Initialization

# In[1]:


from __future__ import annotations

param = {
    'apex': True,
    'awp_eps': 1e-2,
    'awp_lr': 1e-4,
    'batch_size': 1, # 2
    'betas': (0.9, 0.999),
    'ckpt_name': 'deberta_v3_large',
    'debug': True, # False
    'decoder_lr': 1e-5,
    'encoder_lr': 1e-5,
    'eps': 1e-6,
    'max_grad_norm': 1000,
    'max_len': 400, # 512
    'min_lr': 1e-7,
    'model_name': 'microsoft/deberta-v3-large',
    'n_cycles': 0.5,
    'n_epochs': 2, # 12
    'n_eval_steps': 100,
    'n_folds': 2, # 4
    'n_gradient_accumulation_steps': 1,
    'n_warmup_steps': 0,
    'n_workers': 0,
    'nth_awp_start_epoch': 1, # 4
    'output_dir': './output/',
    'print_freq': 100,
    'scheduler_name': 'cosine',
    'seed': 42,
    'tar_token': '[TAR]',
    'weight_decay': 0.01,
}


# In[2]:


class Config:
    def __init__(self, d: dict) -> None:
        for k,v in d.items():
            setattr(self, k, v)

cfg = Config(d=param)


# In[3]:


import os

if not os.path.exists(cfg.output_dir):
    os.makedirs(cfg.output_dir)


# In[4]:


import os
import random
import numpy as np
import torch

def seed_everything(seed:int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=cfg.seed)


# In[5]:


from log import _Logger
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename: str) -> _Logger:
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger(filename=cfg.output_dir+'train')


# ## CV Split

# In[6]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.model_selection import StratifiedGroupKFold

train_df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
if cfg.debug: 
    train_df = train_df.sample(n=1000, random_state=cfg.seed).reset_index(drop=True)

kf = StratifiedGroupKFold(
    n_splits=cfg.n_folds, 
    shuffle=True, 
    random_state=cfg.seed
)
train_df["score_map"] = train_df["score"].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
train_df['fold'] = -1
for f, (tx, vx) in enumerate(kf.split(train_df, train_df["score_map"], train_df["anchor"])):
    train_df.loc[vx, "fold"] = f
display(train_df.groupby("fold").size())


# ## Feature Engineering and Data Transformation

# In[7]:


import re
from numpy import ndarray
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def create_word_normalizer() -> function:
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    def normalize(word):
        w = word.lower()
        w = lemmatizer.lemmatize(w)
        w = ps.stem(w)
        return w
    return normalize

def __normalize_words(titles: list) -> list:
    stop_words = set(stopwords.words('english'))
    normalizer = create_word_normalizer()
    titles = [normalizer(t) for t in titles if t not in stop_words]
    return titles

def normalize_words(words: ndarray, unique=True):
    if type(words) is str:
        words = [words]
    sep_re = r'[\s\(\){}\[\];,\.]+'
    num_re = r'\d'
    words = re.split(sep_re, ' '.join(words).lower())
    words = [w for w in words if len(w) >= 3 and not re.match(num_re, w)]
    if unique:
        words = list(set(words))
        words = set(__normalize_words(words))
    else:
        words = __normalize_words(words)
    return words

def filter_title(title: str) -> str:
    titles = normalize_words(title, unique=False)
    return ','.join([t for t in titles if t in include_words])

cpc_codes = pd.read_csv("../input/cpc-codes/titles.csv", engine='python')

norm_titles = normalize_words(cpc_codes['title'].to_numpy(), unique=False)
anchor_targets = train_df['target'].unique().tolist() + train_df['anchor'].unique().tolist()
norm_anchor_targes = normalize_words(anchor_targets)

include_words = set(norm_titles) & norm_anchor_targes

tmp_cpc_codes = cpc_codes.copy()
tmp_cpc_codes = tmp_cpc_codes[cpc_codes['code'].str.len() >= 4]

tmp_cpc_codes['section_class'] = tmp_cpc_codes['code'].apply(lambda x: x[:3])
title_group_df = tmp_cpc_codes.groupby('section_class', as_index=False)[['title']].agg(list)
title_group_df = title_group_df[title_group_df['section_class'].str.len() == 3]
title_group_df['title'] = title_group_df['title'].apply(lambda lst: ' '.join(lst))

title_group_df['norm_title'] = title_group_df['title'].agg(filter_title)

vectorizer = CountVectorizer()
c_vect = vectorizer.fit_transform(title_group_df['norm_title'])
r = np.argsort(c_vect.toarray(), axis=1)[:, ::-1][::, :400]
vect_words = vectorizer.get_feature_names_out()
t_words = np.vectorize(lambda v: vect_words[v])(r)

norm_title = title_group_df['norm_title'].str.split(',').to_numpy().tolist()
res = []
for (n, t) in zip(norm_title, t_words):
    res.append(','.join(set(n) & set(t)))

title_group_df['norm_title'] = res
title_group_df['section'] = title_group_df.section_class.str[0:1]
title_group_df['section_title'] = title_group_df['section'].map(cpc_codes.set_index('code')['title']).str.lower() + ';' + title_group_df['section_class'].map(cpc_codes.set_index('code')['title']).str.lower()
title_group_df['context_text'] = title_group_df['section_title'] + '[SEP]' + title_group_df['norm_title']
cpc_texts = dict(title_group_df[['section_class', 'context_text']].to_numpy().tolist())


# In[8]:


# aggregate by anchor and context
af_dict = {}
for i,r in train_df[['anchor', 'fold']].iterrows():
    af_dict[r.anchor] = r.fold
anchor_context_grouped_target = train_df.groupby(['anchor', 'context'])['target'].apply(list)
anchor_context_grouped_score = train_df.groupby(['anchor', 'context'])['score'].apply(list)
anchor_context_grouped_id = train_df.groupby(['anchor', 'context'])['id'].apply(list)
i = pd.DataFrame(anchor_context_grouped_id).reset_index()
s = pd.DataFrame(anchor_context_grouped_score).reset_index()
t = pd.DataFrame(anchor_context_grouped_target).reset_index()
train_df = s.merge(t, on=['anchor', 'context'])
train_df = train_df.merge(i, on=['anchor', 'context'])
train_df['context_text'] = train_df['context'].map(cpc_texts)
train_df = train_df.rename(columns={'target': 'targets', 'score': 'scores', 'id': 'ids'})
train_df['fold'] = train_df['anchor'].map(af_dict)
display(train_df.head())
display(train_df.groupby('fold').size())


# ## Tokenizer

# In[9]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
special_tokens_dict = {'additional_special_tokens': [f'[{cfg.tar_token}]']}
tokenizer.add_special_tokens(special_tokens_dict)
tar_token_id = tokenizer(f'[{cfg.tar_token}]', add_special_tokens=False)['input_ids'][0]
logger.info(f'tar_token_id: {tar_token_id}')
setattr(tokenizer, 'tar_token', f'[{cfg.tar_token}]')
setattr(tokenizer, 'tar_token_id', tar_token_id)
tokenizer.save_pretrained(f'{cfg.output_dir}tokenizer/')


# ## Dataset

# In[10]:


import random
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

class TrainDataset(Dataset):
    def __init__(self, df: DataFrame, is_valid: bool, tokenizer: PreTrainedTokenizer, max_len: int):
        self.anchors = df['anchor'].to_numpy()
        self.target_lists = df['targets'].to_numpy()
        self.id_lists = df['ids'].to_numpy()
        self.context_texts = df['context_text'].to_numpy()
        self.score_lists = df['scores'].to_numpy()
        self.is_valid = is_valid
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.id_lists)

    def __getitem__(self, item: int) -> "tuple[dict, Tensor, Tensor]":
        
        scores = np.array(self.score_lists[item])
        target_mask = np.zeros(self.max_len)
        targets = np.array(self.target_lists[item])

        if not self.is_valid:
            indices = list(range(len(scores)))
            random.shuffle(indices)
            scores = scores[indices]
            targets = targets[indices]

        text = ''
        text += self.tokenizer.cls_token
        text += self.anchors[item]
        text += self.tokenizer.sep_token
        for target in targets:
            text += target + self.tokenizer.tar_token
        text += self.context_texts[item] + self.tokenizer.sep_token
        
        encoded = self.tokenizer(
            text,
            max_length = self.max_len,
            padding='max_length',
            add_special_tokens=False,
            truncation=True
        )

        # [cls]+[anchor]+[sep]+[target]+[tar]+[target]+[tar]...+[tar]+[cpc_text]+[sep]
        label = torch.full([self.max_len], -1, dtype=torch.float)
        
        cnt_tar = 0
        cnt_sep = 0
        nth_target = -1
        prev_i = -1

        for i, input_id in enumerate(encoded['input_ids']):
            if input_id == self.tokenizer.tar_token_id:
                cnt_tar += 1
                if cnt_tar == len(targets):
                    break
            if input_id == self.tokenizer.sep_token_id:
                cnt_sep += 1
            
            if cnt_sep == 1 and input_id not in [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.tar_token_id]:
                if (i-prev_i) > 1:
                    nth_target += 1
                label[i] = scores[nth_target]
                target_mask[i] = 1
                prev_i = i

        for k,v in encoded.items():
            encoded[k] = torch.tensor(v, dtype=torch.long)

        return encoded, target_mask, label


# ## Model and AWP

# In[11]:


from torch import Tensor
from torch.nn import Module
from transformers import AutoModel, AutoConfig

class CustomModel(Module):
    def __init__(self, model_name: str, n_vocabs: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(
            model_name, config=self.model_config)
        self.model.resize_token_embeddings(n_vocabs)
        self.fc = nn.Linear(self.model_config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module: Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs: dict) -> Tensor:
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs: dict) -> Tensor:
        feature = self.feature(inputs)
        output = self.fc(feature).squeeze(-1)
        return output


# In[12]:


from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

class AWP:
    def __init__(
        self,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        apex: bool,
        adv_param: str="weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: dict, label: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step() # モデルを近傍の悪い方へ改変
            y_preds = self.model(inputs)
            adv_loss = self.criterion(
                y_preds.view(-1, 1), label.view(-1, 1))
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# ## Trainer

# In[13]:


import os
import gc
from log import _Logger
import random
import warnings
from functools import reduce
warnings.filterwarnings("ignore")
import numpy as np
from numpy import ndarray
import scipy as sp
import torch
from torch import inference_mode
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from IPython.display import display
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_score(y_true: ndarray, y_pred: ndarray) -> float:
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

class Trainer:

    def __init__(self, cfg:Config, logger: _Logger, tokenizer: PreTrainedTokenizer) -> None:
        self.cfg = cfg
        self.logger = logger
        self.tokenizer = tokenizer

    def save_ckpt(self, fold: int, model: Module, predictions: ndarray) -> None:
        torch.save(
            {'model': model.state_dict(), 'predictions': predictions},
            f'{self.cfg.output_dir}{self.cfg.ckpt_name}_fold{fold}_best.pth'
        )
        self.logger.info('model has been saved.')

    @inference_mode()
    def valid_fn(self, dl: DataLoader, model: Module, criterion: _Loss) -> "tuple[float, list[list[float]]]":
        model.eval()
        preds = []
        tot_loss = 0
        for step, (inputs, target_masks, labels) in enumerate(dl):

            for k, v in inputs.items():
                inputs[k] = v.cuda()
            labels = labels.cuda()
            
            y_preds = model(inputs) # (batch_size, max_len)

            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            mask = (labels.view(-1, 1) != -1)
            loss = torch.masked_select(loss, mask)
            loss = loss.mean()
            
            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            tot_loss += loss.item()

            y_preds = y_preds.sigmoid().to('cpu').numpy()
            labels = labels.to('cpu').numpy() # (batch_size, max_len)

            anchorwise_preds = []
            for pred, target_mask, in zip(y_preds, target_masks):
                prev_i = -1
                targetwise_pred_scores = []
                for i, (p, tm) in enumerate(zip(pred, target_mask)):
                    if tm != 0:
                        if i-1 == prev_i:
                            targetwise_pred_scores[-1].append(p)
                        else:
                            targetwise_pred_scores.append([p])
                        prev_i = i
                for targetwise_pred_score in targetwise_pred_scores:
                    anchorwise_preds.append(np.mean(targetwise_pred_score))
            preds.append(anchorwise_preds)

            if step % cfg.print_freq == 0 or step == (len(dl) - 1):
                print('EVAL: [{0}/{1}] '
                    'Loss: {loss:.4f}({avg_loss:.4f}) '
                    .format(step, len(dl),
                            loss=loss.item(),
                           avg_loss=tot_loss/(step+1))
                )
        
        return tot_loss/(step+1), preds

    def train_with_eval(self,
                        fold: int,
                        train_loader: DataLoader,
                        valid_loader: DataLoader,
                        valid_labels: ndarray,
                        model: Module,
                        criterion: _Loss,
                        optimizer: Optimizer,
                        epoch: int,
                        scheduler: _LRScheduler,
                        best_score: float) -> "tuple[float, float]":
        
        if not epoch < self.cfg.nth_awp_start_epoch:
            self.logger.info(f'AWP training with epoch {epoch+1}')

        model.train()
        awp = AWP(
            model, 
            criterion, 
            optimizer,
            self.cfg.apex,
            adv_lr=self.cfg.awp_lr, 
            adv_eps=self.cfg.awp_eps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.apex)
        global_step = 0
        tot_loss = 0
        for step, (inputs, _, labels) in enumerate(train_loader):
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            labels = labels.cuda()
            with torch.cuda.amp.autocast(enabled=self.cfg.apex):
                y_preds = model(inputs)

            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            mask = (labels.view(-1, 1) != -1)
            loss = torch.masked_select(loss, mask).mean()
            
            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.cfg.max_grad_norm)

            if self.cfg.nth_awp_start_epoch <= epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()
            tot_loss += loss.item()

            if (step + 1) % self.cfg.n_gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()

            if step % self.cfg.print_freq == 0 or step == (len(train_loader) - 1):
                self.logger.info('Epoch: [{0}][{1}/{2}] '
                    'Loss: {loss:.4f}({avg_loss:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  '
                    .format(epoch + 1, step, len(train_loader),
                            loss=loss.item(),
                            avg_loss=tot_loss/(step+1),
                            grad_norm=grad_norm,
                            lr=scheduler.get_lr()[0]))

            if (step + 1) % self.cfg.n_eval_steps == 0:
                
                val_loss, predictions = self.valid_fn(
                    valid_loader, model, criterion)
                score = get_score(
                    valid_labels, 
                    np.array(reduce(lambda a,b: a+b, predictions)))
                logger.info(
                    f'Epoch {epoch+1} - avg_train_loss: {tot_loss/(step+1):.4f}  avg_val_loss: {val_loss:.4f}')
                logger.info(f'Epoch {epoch+1} Step  Score: {score:.4f}')
                if best_score < score:
                    best_score = score
                    logger.info({f"[fold{fold}] best score": score})
                    self.save_ckpt(fold, model, np.array(reduce(lambda a,b: a+b, predictions)))
                model.train()
        return tot_loss/(step+1), best_score


    def train_loop(self, folds: DataFrame, fold: int) -> DataFrame:

        self.logger.info(f"========== fold: {fold} training ==========")

        train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
        valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
        valid_labels = valid_folds['scores'].explode().to_numpy()

        train_dataset = TrainDataset(
            df=train_folds, is_valid=False, tokenizer=self.tokenizer, max_len=self.cfg.max_len)
        valid_dataset = TrainDataset(
            df=valid_folds, is_valid=True, tokenizer=self.tokenizer, max_len=self.cfg.max_len)

        train_loader = DataLoader(train_dataset,
                                batch_size=self.cfg.batch_size,
                                shuffle=True,
                                num_workers=self.cfg.n_workers, 
                                pin_memory=True, 
                                drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                batch_size=self.cfg.batch_size*2,
                                shuffle=False,
                                num_workers=self.cfg.n_workers, 
                                pin_memory=True, 
                                drop_last=False)

        model = CustomModel(
            self.cfg.model_name, n_vocabs=len(self.tokenizer))
        torch.save(model.model_config, f'{self.cfg.output_dir}config.pth')
        model.cuda()

        def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],
                'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=self.cfg.encoder_lr,
                                                    decoder_lr=self.cfg.decoder_lr,
                                                    weight_decay=self.cfg.weight_decay)
        optimizer = AdamW(
            optimizer_parameters, 
            lr=self.cfg.encoder_lr,
            eps=self.cfg.eps, 
            betas=self.cfg.betas)


        def get_scheduler(scheduler_name: str, optimizer: Optimizer, num_train_steps: int, n_cycles: int) -> _LRScheduler:
            if scheduler_name == 'linear':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=cfg.n_warmup_steps, num_training_steps=num_train_steps
                )
            elif scheduler_name == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=cfg.n_warmup_steps, num_training_steps=num_train_steps, num_cycles=n_cycles
                )
            return scheduler

        num_train_steps = int(len(train_folds) / self.cfg.batch_size * self.cfg.n_epochs)
        scheduler = get_scheduler(
            cfg.scheduler_name, optimizer, num_train_steps, cfg.n_cycles)

        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_score = -1000.0

        for epoch in range(self.cfg.n_epochs):

            avg_loss, best_score = self.train_with_eval(
                fold, 
                train_loader, 
                valid_loader, 
                valid_labels,
                model, 
                criterion, 
                optimizer, 
                epoch, 
                scheduler, 
                best_score)

            avg_val_loss, predictions = self.valid_fn(
                valid_loader, model, criterion)

            # scoring
            score = get_score(valid_labels, np.array(reduce(lambda a,b: a+b, predictions)))

            logger.info(
                f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}')
            logger.info(f'Epoch {epoch+1} - Score: {score:.4f}')

            if best_score < score:
                best_score = score
                logger.info(
                    f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                self.save_ckpt(fold, model, np.array(reduce(lambda a,b: a+b, predictions)))

        predictions = torch.load(f"{self.cfg.output_dir}{self.cfg.ckpt_name}_fold{fold}_best.pth",
                                map_location=torch.device('cpu'))['predictions']
        # to no-aggregated df
        valid_folds = valid_folds.explode(['scores', 'targets', 'ids']).rename(columns={
            'scores': 'score',
            'targets': 'target',
            'ids': 'id',
        }).reset_index(drop=True)
        valid_folds['pred'] = predictions

        torch.cuda.empty_cache(); gc.collect()

        return valid_folds


# ## Run!

# In[14]:


trainer = Trainer(
    cfg=cfg,
    logger=logger,
    tokenizer=tokenizer
)
oof_df = pd.DataFrame()
for fold in range(cfg.n_folds):
    _oof_df = trainer.train_loop(train_df, fold)
    _oof_df.to_pickle(f"{cfg.output_dir}oof_fold{fold}.pkl")
    oof_df = pd.concat([oof_df, _oof_df])
    logger.info(f"========== fold: {fold} result ==========")
    _score = get_score(
        _oof_df['score'].to_numpy(), 
        _oof_df['pred'].to_numpy()
    )
    logger.info({f"[fold{fold}] best score": _score})
oof_df = oof_df.reset_index(drop=True)

logger.info(f"========== CV ==========")
score = get_score(
    oof_df['score'].to_numpy(), 
    oof_df['pred'].to_numpy()
)
logger.info({f"overall score": score})
oof_df.to_pickle(f'{cfg.output_dir}oof_df.pkl')


# In[ ]:




