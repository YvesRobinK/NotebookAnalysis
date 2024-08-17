#!/usr/bin/env python
# coding: utf-8

# # Temporal Convolution with Only 4 Features (Inference Part)
# 
# ## Introduction
# Before diving into the manual feature engineering, I try to see how far a DL-based model can go with a small set of features, including only:
# 1. Difference of `elapsed_time`
# 2. `event_name`
# 3. `name`
# 4. `room_fqid`
# 
# First, `event_name` is combined with `name` to form a new feature `event_comb` (*i.e.*, event combination). Then, I feed these three processed features into the model to learn **event-aware temporal patterns**. Finally, this model can achieve the performance summarized as follows:
# 
# | CV (GroupKFold with k=5) | Holdout (Released Old Test Set) | LB    |
# | ------------------------ | ------------------------------- | ----- |
# | 0.6914                   | 0.6911                          | 0.694 |
# 
# 
# ## About this Notebook
# In this kernel, I run the inference process with 5-fold model blending (equally-weighted). The training part is still a work in progress and will be published soon.
# 
# 
# ## Acknowledgements
# Special thanks to [@chrisqiu](https://www.kaggle.com/chrisqiu)'s sharing [here](https://www.kaggle.com/code/chrisqiu/pytorch-using-only-1-column-submission), which provides the inspiration for model building.
# 
# ## Import Packages

# In[1]:


import gc
import os
from typing import Any, Dict, List, Optional, Tuple
import pickle
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ## Define Data Path

# In[2]:


exp_dump_path = "/kaggle/input/0330-16-53-13/0330-16_53_13"
cat2code_path = "/kaggle/input/cat2code-v2/cat2code"


# ## Evoke Experiment Configuration

# In[3]:


CAT_FEATS = ["event_comb_code", "room_fqid_code"]
CAT_FEAT_SIZE = {
    "event_comb_code": 19,
    "room_fqid_code": 19,
}
DEVICVE = torch.device("cpu")

# Load model configuration
with open(os.path.join(exp_dump_path, "config/cfg.yaml"), "r") as f:
    cfg = yaml.full_load(f)
model_cfg = cfg["model"]

best_thres = 0.63
t_window = 1000


# ## Load Model

# In[4]:


class TConvLayer(nn.Module):
    """Dilated temporal convolution layer."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
        act: str = "relu",
        dropout: float = 0.1,
    ):
        super(TConvLayer, self).__init__()
        
        # Network parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        
        # Model blocks
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation, bias=bias),
            dim=None
        )
        self.bn = nn.BatchNorm1d(out_dim)
        if act == "relu":
            self.act = nn.ReLU()
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Shape:
            x: (B, C, P)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x
    

class EventAwareEncoder(nn.Module):
    
    def __init__(
        self,
        h_dim: int = 128,
        out_dim: int = 128,
        readout: bool = True,
        cat_feats: List[str] = ["event_comb_code", "room_fqid_code"]
    ):
        super(EventAwareEncoder, self).__init__()

        # Network parameters
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.cat_feats = cat_feats

        # Model blocks
        # Categorical embeddings
        self.embs = nn.ModuleList()
        for cat_feat in cat_feats:
            self.embs.append(nn.Embedding(CAT_FEAT_SIZE[cat_feat] + 1, 32, padding_idx=0))
        self.emb_enc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        # Feature extractor
        self.convs = nn.ModuleList()
        for l, (dilation, kernel_size) in enumerate(zip([2**i for i in range(3)], [7, 7, 5])):
            self.convs.append(TConvLayer(64, h_dim, kernel_size, dilation=1))   # No dilation
        # Readout layer
        if readout:
            self.readout = nn.Sequential(
                nn.Linear(2 * (h_dim // 2), out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
        else:
            self.readout = None

    def forward(self, x: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, P, C)
            x_cat: (B, P, #cat_feats)
        """

        # Categorical embeddings
        x_cat = x_cat + 1
        x_emb = []
        for i in range(len(self.cat_feats)):
            x_emb.append(self.embs[i](x_cat[..., i]))  # (B, P, emb_dim)
        x_emb = torch.cat(x_emb, dim=-1)  # (B, P, emb_dim_cat)
        x_emb = self.emb_enc(x_emb) + x_emb  # (B, P, emb_dim_cat)
        x = x * x_emb  # (B, P, emb_dim_cat)
        x = self.dropout(x)
        
        # Feature extractor
        x = x.transpose(1, 2)  # (B, C, P)
        x_skip = []
        for l in range(3):
            x_conv = self.convs[l](x)   # (B, C * 2, P')
            x_filter, x_gate = torch.split(x_conv, x_conv.size(1) // 2, dim=1)
            x_conv = F.tanh(x_filter) * F.sigmoid(x_gate)   # (B, C, P')
            
            x_conv = self.dropout(x_conv)
            
            # Skip connection
            x_skip.append(x_conv.unsqueeze(dim=1))  # (B, L (1), C, P')
            
            # Residual
            x = x_conv
            
        # Process skipped latent representation
        for l in range(3-1):
            x_skip[l] = x_skip[l][..., -x_skip[-1].size(3) : ]
        x_skip = torch.cat(x_skip, dim=1)   # (B, L, C, P)
        x_skip = torch.sum(x_skip, dim=1)   # (B, C, P')

        # Readout layer
        if self.readout is not None:
            x_std = torch.std(x_skip, dim=-1)  # Std pooling
            x_mean = torch.mean(x_skip, dim=-1)  # Mean pooling
            x = torch.cat([x_std, x_mean], dim=1)
            x = self.readout(x)  # (B, out_dim)

        return x


class EventConvSimple(nn.Module):

    def __init__(self, n_lvs: int, out_dim: int, **model_cfg: Any):
        self.name = self.__class__.__name__
        super(EventConvSimple, self).__init__()

        enc_out_dim = 128

        # Network parameters
        self.n_lvs = n_lvs
        self.out_dim = out_dim
        self.cat_feats = model_cfg["cat_feats"]
        
        self.encoder = EventAwareEncoder(h_dim=128, out_dim=enc_out_dim, cat_feats=self.cat_feats)
        self.clf = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_out_dim // 2, out_dim),
        )

    def forward(self, x: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, P, C)
            x_cat: (B, P, 3)
        """
        x = self.encoder(x, x_cat)
        x = self.clf(x)
        
        x = F.sigmoid(x)

        return x


# In[5]:


model_path = os.path.join(exp_dump_path, "models")

models = {"0-4": [], "5-12": [], "13-22": []}
for model_file in os.listdir(model_path):
    if "encoder" in model_file: continue
    if "0-4" in model_file:
        model = EventConvSimple(5, 3, **model_cfg)
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, model_file),
                map_location=torch.device('cpu')
            )
        )
        models["0-4"].append(model)
    elif "5-12" in model_file:
        model = EventConvSimple(8, 10, **model_cfg)
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, model_file),
                map_location=torch.device('cpu')
            )
        )
        models["5-12"].append(model)
    elif "13-22" in model_file:
        model = EventConvSimple(10, 5, **model_cfg)
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, model_file),
                map_location=torch.device('cpu')
            )
        )
        models["13-22"].append(model)


# ## Run Inference

# In[6]:


import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[7]:


cat2code_ = {}
for cat_feat in CAT_FEATS:
    with open(os.path.join(cat2code_path, f"{cat_feat[:-5]}2code.pkl"), "rb") as f:
        cat2code_[cat_feat] = pickle.load(f)
        
et_diff_upper_bound = 3.6e6


# In[8]:


def drop_multi_game_naive(df: pd.DataFrame, local: bool = True) -> pd.DataFrame:
    """Drop events not occurring at the first game play.
    
    Parameters:
        df: input DataFrame
    
    Return:
        df: DataFrame with events occurring at the first game play only
    """
    if local:
        df["lv_diff"] = df.groupby("session_id").apply(lambda x: x["level"].diff().fillna(0)).values
    else:
        df["lv_diff"] = df["level"].diff().fillna(0)
    reversed_lv_pts = df["lv_diff"] < 0
    df.loc[~reversed_lv_pts, "lv_diff"] = 0
    if local:
        df["multi_game_flag"] = df.groupby("session_id")["lv_diff"].cumsum()
    else:
        df["multi_game_flag"] = df["lv_diff"].cumsum()
    multi_game_mask = df["multi_game_flag"] < 0
    multi_game_rows = df[multi_game_mask].index
    df = df.drop(multi_game_rows).reset_index(drop=True)
    
    return df


# In[9]:


@torch.no_grad()
def quick_infer(X: Tuple[Tensor, Optional[Tensor]], models: List[nn.Module]) -> Tensor:
    
    x, x_cat = X
    n_models = len(models)
    
    for i, model in enumerate(models):
        model.eval()
        if i == 0:
            y_pred = model(x, x_cat) / n_models   # (1, n_qns (out_dim))
        else:
            y_pred += model(x, x_cat) / n_models   # (1, n_qns (out_dim))
    
    y_pred = y_pred.reshape((-1, 1)).detach().numpy()
    
    
    return y_pred


# In[10]:


for (test, sample_submission) in iter_test:
    cur_lv_gp = test["level_group"].unique()[0]
    test = drop_multi_game_naive(test, local=False)
    test["et_diff"] = test["elapsed_time"].diff().fillna(0).clip(0, et_diff_upper_bound)
    test["event_comb"] = test["event_name"] + "_" + test["name"]
    for cat_feat in CAT_FEATS:
        test[cat_feat] = test[cat_feat[:-5]].map(cat2code_[cat_feat]).astype(np.int32)
    
    x = test["et_diff"].values[-t_window:]
    x_cat = test[CAT_FEATS].values[-t_window:]
    if len(x) < t_window:
        pad_len = t_window - len(x)
        x = np.pad(x, (pad_len, 0), "constant")
        x_cat = np.pad(x_cat, ((pad_len, 0), (0, 0)), "constant", constant_values=-1)  # (P, #cat_feats)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=-1)   # Add B, C dim, (1, P, 1)
    x_cat = torch.tensor(x_cat, dtype=torch.int32).unsqueeze(dim=0)   # (1, P, 3)
    
    try:
        # Run quick inference
        y_pred = quick_infer((x, x_cat), models[cur_lv_gp])   # (1, n_qns)
        y_pred = (y_pred > best_thres).astype(np.int)

        sample_submission.loc[:, "correct"] = y_pred
    except:
        sample_submission.loc[:, "correct"] = 0
    env.predict(sample_submission)


# In[ ]:




