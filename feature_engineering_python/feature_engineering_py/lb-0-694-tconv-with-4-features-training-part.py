#!/usr/bin/env python
# coding: utf-8

# # [LB 0.694] Event-Aware TConv with Only 4 Features - Training Part
# 
# ## Introduction
# Before diving into the manual feature engineering, I'm interested in seeing how far a DL-based model can go with a small set of features. After running experiments day and night, I temporarily can achieve **CV 0.6914** and LB 0.694 with only four features, including:
# 
# 1. Difference of `elapsed_time`
# 2. `event_name`
# 3. `name`
# 4. `room_fqid`
# 
# First, `event_name` is combined with `name` to form a new feature `event_comb` (*i.e.*, event combination), which represents the **status** of the corresponding event (*e.g.*, `notebook_click_open`, opening the notebook). As many have pointed out that the difference of `elapsed_time` is a key feature to use, I also treat it as a (and the only) numeric feature, which interacts with the other categorical features. The motivation behind the scene is that I hope the model can capture **event-aware temporal patterns**; that is, each time difference value is **enriched by the event information and where the event take places**. Finally, the model achieves the performance summarized as follows:
# 
# | CV (GroupKFold with k=5) | Holdout (Released Old Test Set) | LB    |
# | ------------------------ | ------------------------------- | ----- |
# | 0.6914                   | 0.6911                          | 0.694 |
# 
# ## About this Notebook
# In this kernel, I implement a complete pipeline, including data cleaning and processing, simple feature engineering, and model training with the specified CV scheme (*i.e.,* `GroupKFold` with `k=5`). After the process is done, output objects (*e.g.,* best model checkpoints) can be downloaded and used in the inference part.
# 
# ## Acknowledgements
# * Special thanks to [@chrisqiu](https://www.kaggle.com/chrisqiu)'s sharing [here](https://www.kaggle.com/code/chrisqiu/0-681-pytorch-using-only-1-column), which provides the inspiration for model building.
# * For section 7, all the credits should go to [@cdeotte](https://www.kaggle.com/cdeotte). I use and modify the code snippets in [his amazing starter notebook](https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-680).
# 
# ## Other Works
# * To explore dangers in data, please see [EDA on Game Progress](https://www.kaggle.com/code/abaojiang/eda-on-game-progress)
# * To run the inference counterpart of this notebook, please see [[LB 0.694] Event-Aware TConv with Only 4 Features](https://www.kaggle.com/code/abaojiang/lb-0-694-event-aware-tconv-with-only-4-features)
# 
# <a id="toc"></a>
# ## Table of Contents
# * [1. Prepare Data](#prep_data)
# * [2. Define Dataset](#dataset)
# * [3. Define Model Architecture](#model)
# * [4. Define Evaluator](#evaluator)
# * [5. Define Trainer](#trainer)
# * [6. Run Cross-Validation](#cv)
# * [7. Derive Final CV Score](#cv_score)
# 
# ## Import Packages

# In[1]:


import os
import gc
import pickle
import random
import warnings
warnings.simplefilter("ignore")
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import optim, Tensor
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss
from torchmetrics import AUROC

N_QNS = 18
LEVEL = list(range(23))
LEVEL_GROUP = ["0-4", "5-12", "13-22"]
LVGP_ORDER = {"0-4": 0, "5-12": 1, "13-22": 2}
QNS_PER_LV_GP = {"0-4": list(range(1, 4)), "5-12": list(range(4, 14)), "13-22": list(range(14, 19))}
LV_PER_LV_GP = {"0-4": list(range(0, 5)), "5-12": list(range(5, 13)), "13-22": list(range(13, 23))}
CAT_FEAT_SIZE = {
    "event_comb_code": 19,
    "room_fqid_code": 19,
}


# ## Define Experiment Configuration 

# In[2]:


def seed_all(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Parameters:
        seed: manually specified seed number

    Return:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# In[3]:


class CFG:
    # ==Mode==
    # Specify True to enable model training
    train = False
    
    # ==Data===
    FEATS = ["et_diff", "event_comb_code", "room_fqid_code"]
    CAT_FEATS = ["event_comb_code", "room_fqid_code"]
    COLS_TO_USE = ["session_id", "level", "level_group", "elapsed_time",
                   "event_name", "name", "room_fqid"]
    T_WINDOW = 1000
    
    # ==Training==
    SEED = 42
    DEVICE = "cuda:0"
    EPOCH = 100
    CKPT_METRIC = "f1@0.63"

    # ==DataLoader==
    BATCH_SIZE = 128
    NUM_WORKERS: 4

    # ==Solver==
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # ==Early Stopping==
    ES_PATIENCE = 0

    # ==Evaluator==
    EVAL_METRICS = ["auroc", "f1"]

INPUT_PATH = "/kaggle/input/jo-old-train/upload/train.parquet"
TARGET_PATH = "/kaggle/input/jo-old-train/upload/train_labels.csv"
cfg = CFG()
seed_all(cfg.SEED)


# <a id="prep_data"></a>
# ## 1. Prepare Data
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)
# 
# ### *1.1 Load and Clean Data*

# In[4]:


def summarize(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    n_rows_to_display: Optional[int] = 5,
) -> None:
    """Summarize DataFrame.

    Parameters:
        df: input data
        file_name: name of the input file
        n_rows_to_display: number of rows to display

    Return:
        None
    """
    file_name = "Data" if file_name is None else file_name

    # Derive NaN ratio for each column
    nan_ratio = pd.isna(df).sum() / len(df) * 100
    nan_ratio.sort_values(ascending=False, inplace=True)
    nan_ratio = nan_ratio.to_frame(name="NaN Ratio").T

    # Derive zero ratio for each column
    zero_ratio = (df == 0).sum() / len(df) * 100
    zero_ratio.sort_values(ascending=False, inplace=True)
    zero_ratio = zero_ratio.to_frame(name="Zero Ratio").T

    # Print out summarized information
    print(f"=====Summary of {file_name}=====")
    display(df.head(n_rows_to_display))
    print(f"Shape: {df.shape}")
    print("NaN ratio:")
    display(nan_ratio)
    print("Zero ratio:")
    display(zero_ratio)
    
def drop_multi_game_naive(
    df: pd.DataFrame,
    local: bool = True
) -> pd.DataFrame:
    """Drop events not occurring at the first game play.
    
    Note: `groupby` should be done with `sort=False` for the new
    training set, because `session_id` isn't sorted by default.
    
    Parameters:
        df: input DataFrame
        local: if False, the processing step is simplified based on the
            properties of returned test DataFrame by time series API
    
    Return:
        df: DataFrame with events occurring at the first game play only
    """
    df = df.copy()
    if local:
        df["lv_diff"] = df.groupby("session_id", sort=False).apply(lambda x: x["level"].diff().fillna(0)).values
    else:
        df["lv_diff"] = df["level"].diff().fillna(0)
    reversed_lv_pts = df["lv_diff"] < 0
    df.loc[~reversed_lv_pts, "lv_diff"] = 0
    if local:
        df["multi_game_flag"] = df.groupby("session_id", sort=False)["lv_diff"].cumsum().values
    else:
        df["multi_game_flag"] = df["lv_diff"].cumsum()
    multi_game_mask = df["multi_game_flag"] < 0
    multi_game_rows = df[multi_game_mask].index
    df = df.drop(multi_game_rows).reset_index(drop=True)
    
    # Drop redundant columns
    df.drop(["lv_diff", "multi_game_flag"], axis=1, inplace=True)
    
    return df

def map_lvgp_order(lvgp_seq: pd.Series) -> pd.Series:
    """Map level_group sequence to level_group order sequence.
    
    Parameters:
        lvgp_seq: level_group sequence
    
    Return:
        lvgp_order_seq: level_group order sequence
    """
    lvgp_order_seq = lvgp_seq.map(LVGP_ORDER)
    
    return lvgp_order_seq

def check_multi_game(df: pd.DataFrame) -> bool:
    """Check if multiple game plays exist in any session.
    
    Parameters:
        df: input DataFrame
    
    Return:
        multi_game_exist: if True, multiple game plays exist in at
            least one of the session
    """
    multi_game_exist = False
    for i, (sess_id, gp) in enumerate(df.groupby("session_id", sort=False)):
        if ((not gp["level"].is_monotonic_increasing)
            or (not gp["lvgp_order"].is_monotonic_increasing)):
            multi_game_exist = True
            break
            
    return multi_game_exist


# In[5]:


df = pd.read_parquet(INPUT_PATH, columns=cfg.COLS_TO_USE)
y = pd.read_csv(TARGET_PATH)
summarize(df, "X", 2); summarize(y, "y", 2)


# In[6]:


df = drop_multi_game_naive(df)
df["lvgp_order"] = map_lvgp_order(df["level_group"])
if check_multi_game(df):
    print("There exist multiple game plays in at least one session.")


# ### *1.2 Do Simple Feature Engineering*

# In[7]:


def get_factorize_map(
    values: Union[np.ndarray, pd.Series, pd.Index],
    sort: bool = True
) -> Dict[str, int]:
    """Factorize array and return numeric representation map.

    Parameters:
        values: 1-D sequence to factorize
        sort: whether to sort unique values

    Return:
        val2code: mapping from value to numeric code
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    vals_uniq = values.unique()
    if sort:
        vals_uniq = sorted(vals_uniq)
    val2code = {val: code for code, val in enumerate(vals_uniq)}

    return val2code


# In[8]:


# Generate the only numeric feature
df["et_diff"] = df.groupby("session_id", sort=False).apply(lambda x: x["elapsed_time"].diff().fillna(0)).values
df["et_diff"] = df["et_diff"].clip(0, 3.6e6)

# Factorize categorical features
df["event_comb"] = df["event_name"] + "_" + df["name"]
for cat_feat in cfg.CAT_FEATS:
    orig_col = cat_feat[:-5]
    cat2code = get_factorize_map(df[orig_col])
    df[cat_feat] = df[orig_col].map(cat2code)
    
    with open(f"./{orig_col}2code.pkl", "wb") as f:
        pickle.dump(cat2code, f)
        
X = df[["session_id", "level_group", "level"] + cfg.FEATS]
del df; _ = gc.collect()


# ### *1.3 Process Labels*

# In[9]:


y["q"] = y["session_id"].apply(lambda x: x.split("_q")[1]).astype(int)
y["session_id"] = y["session_id"].apply(lambda x: x.split("_q")[0]).astype(int)
y = y.sort_values(["session_id", "q"]).reset_index(drop=True)
qn2lvgp = {qn: lv_gp for lv_gp, qns in QNS_PER_LV_GP.items() for qn in qns}
y["level_group"] = y["q"].map(qn2lvgp)
y_lvgp = y.groupby(["session_id", "level_group"]).apply(lambda x: list(x["correct"])).reset_index()
y_lvgp["lvgp_order"] = y_lvgp["level_group"].map(LVGP_ORDER)
y_lvgp = y_lvgp.sort_values(["session_id", "lvgp_order"]).reset_index(drop=True)
y_lvgp.rename({0: "correct"}, axis=1, inplace=True)
y_lvgp.drop(["lvgp_order"], axis=1, inplace=True)

print("=====Labels Aggregated by `level_group`=====")
y_lvgp.head(3)


# In[10]:


y_all = y_lvgp.groupby("session_id").apply(lambda x: list(x["correct"])).reset_index()
ans_tmp = np.array(list(y_all.apply(lambda x: np.concatenate(x[0]), axis=1).values))
y_all.drop(0, axis=1, inplace=True)
y_all[list(range(18))] = ans_tmp
y_all = y_all.set_index("session_id").sort_index()

print("=====Flattened Labels=====")
y_all.head(3)


# In[11]:


del y; _ = gc.collect()


# <a id="dataset"></a>
# ## 2. Define Dataset
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[12]:


class LvGpDataset(Dataset):
    """Dataset for `level_group`-wise modeling.

    Input data are all from the same `level_group`. Also, if `t_window`
    isn't long enough, only raw features from the last level of a
    `level_group` is retrieved.

    Commonly used notations are summarized as follows:
    * `N` denotes the number of samples
    * `P` denotes the length of lookback time window
    * `Q` denotes the number of questions in the current level group
    * `C` denotes the number of channels (numeric features)
        *Note: Currently, only `et_diff` is used as the numeric feature
    * `M` denotes the number of categorical features

    Each sample is session-specific, which is illustrated as follows:

    Let C=1 (i.e., only `et_diff` is used as the feature) and P=60, we
    have data appearance like:

    idx  t-59  t-58  ...  t-1  t
    0     1     2          0   2   -> session_id == 20090312431273200
    1     2     3          4   1   -> session_id == 20090312433251036
    .
    .
    .
    N-2
    N-1

    Parameters:
        data: input data
        t_window: length of lookback time window
    """

    n_samples: int

    def __init__(
        self,
        data: Tuple[pd.DataFrame, pd.DataFrame],
        t_window: int,
        **kwargs: Any,
    ) -> None:
        self.X_base, self.y_base = data
        #         self.y_base["correct"] = self.y_base["correct"].apply(lambda x: ast.literal_eval(x))
        self.t_window = t_window
        
        # Specify features to use
        self.feats = [feat for feat in cfg.FEATS if feat not in cfg.CAT_FEATS]
        self.cat_feats = cfg.CAT_FEATS

        # Setup level-related metadata
        self.lv_gp = self.X_base["level_group"].unique()[0]
        self.levels = LV_PER_LV_GP[self.lv_gp]
        self.n_lvs = len(self.levels)

        # Generate data samples
        self._chunk_X_y()
        self._set_n_samples()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        X = self.X[idx, ...]
        y = self.y[idx, ...]
        X_cat = self.X_cat[idx, ...]
        
        data_sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "X_cat": torch.tensor(X_cat, dtype=torch.int32),
            "y": torch.tensor(y, dtype=torch.float32),
        }

        return data_sample

    def _set_n_samples(self) -> None:
        """Derive the number of samples."""
        self.n_samples = len(self.X)

    def _chunk_X_y(self) -> None:
        """Chunk data samples."""
        X, y = [], []
        X_cat = []

        for sess_id in self.y_base["session_id"].values:
            # Target columns hard-coded temporarily
            X_sess = self.X_base[self.X_base["session_id"] == sess_id]
            
            pad_len = 0
            x_num = []
            for i, feat in enumerate(self.feats):
                x_num_i = X_sess[feat].values[-self.t_window :]
                if i == 0 and len(x_num_i) < self.t_window:
                    pad_len = self.t_window - len(x_num_i)
                
                if pad_len != 0:
                    x_num_i = np.pad(x_num_i, (pad_len, 0), "constant")
                    
                x_num.append(x_num_i)
            
            x_cat = X_sess[self.cat_feats].values[-self.t_window :]
            if pad_len != 0:
                x_cat = np.pad(x_cat, ((pad_len, 0), (0, 0)), "constant", constant_values=-1)  # (P, M)

            x_num = np.stack(x_num, axis=1)   # (P, C)
            X.append(x_num)
            X_cat.append(x_cat)
            y.append(self.y_base[self.y_base["session_id"] == sess_id]["correct"].values[0])

        self.X = np.stack(X)  # (N, P, C)
        self.X_cat = np.stack(X_cat)  # (N, P, M)
        self.y = np.vstack(y)  # (N, Q)


# In[13]:


def build_dataloaders(
    data_tr: Tuple[pd.DataFrame, pd.DataFrame],
    data_val: Tuple[pd.DataFrame, pd.DataFrame],
    batch_size: int,
    **dataset_cfg: Any,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation dataloaders.

    Parameters:
        data_tr: training data
        data_val: validation data
        dataloader_cfg: hyperparameters of dataloader
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training dataloader
        val_loader: validation dataloader
    """
    if data_tr is not None:
        train_loader = DataLoader(
            LvGpDataset(data_tr, **dataset_cfg),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=None,
        )
    else:
        train_loader = None
    if data_val is not None:
        val_loader = DataLoader(
            LvGpDataset(data_val, **dataset_cfg),
            batch_size=batch_size,
            shuffle=False,  # Hard-coded
            num_workers=2,
            collate_fn=None,
        )
    else:
        val_loader = None

    return train_loader, val_loader


# <a id="model"></a>
# ## 3. Define Model Architecture
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)
# 
# Following is the overview of the model architecture,
# 
# [![2023-04-02-10-39-43.png](https://i.postimg.cc/2y9chX1n/2023-04-02-10-39-43.png)](https://postimg.cc/wRJQd2cB)

# In[14]:


class TConvLayer(nn.Module):
    """Dilated temporal convolution layer.
    
    Considering the time cost, I currently disable dilation.
    """
    
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
            x: (B, C', P)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x
    

class EventAwareEncoder(nn.Module):
    """Event-aware encoder based on 1D-Conv."""
    
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
            x_cat: (B, P, M)
        """

        # Categorical embeddings
        x_cat = x_cat + 1
        x_emb = []
        for i in range(len(self.cat_feats)):
            x_emb.append(self.embs[i](x_cat[..., i]))  # (B, P, emb_dim)
        x_emb = torch.cat(x_emb, dim=-1)  # (B, P, C')
        x_emb = self.emb_enc(x_emb) + x_emb  # (B, P, C')
        x = x * x_emb  # (B, P, C')
        x = self.dropout(x)
        
        # Feature extractor
        x = x.transpose(1, 2)  # (B, C', P)
        x_skip = []
        for l in range(3):
            x_conv = self.convs[l](x)   # (B, C' * 2, P')
            x_filter, x_gate = torch.split(x_conv, x_conv.size(1) // 2, dim=1)
            x_conv = F.tanh(x_filter) * F.sigmoid(x_gate)   # (B, C', P')
            
            x_conv = self.dropout(x_conv)
            
            # Skip connection
            x_skip.append(x_conv.unsqueeze(dim=1))  # (B, L (1), C', P')
            
            x = x_conv
            
        # Process skipped latent representation
        for l in range(3-1):
            x_skip[l] = x_skip[l][..., -x_skip[-1].size(3) : ]
        x_skip = torch.cat(x_skip, dim=1)   # (B, L, C', P_truc)
        x_skip = torch.sum(x_skip, dim=1)   # (B, C', P_truc)

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
            x_cat: (B, P, M)
        """
        x = self.encoder(x, x_cat)
        x = self.clf(x)

        return x


# <a id="evaluator"></a>
# ## 4. Define Evaluator
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[15]:


class Evaluator(object):
    """Evaluator.

    Parameters:
        metric_names: evaluation metrics
        n_qns: number of questions
    """

    eval_metrics: Dict[str, Callable[..., Union[float]]]

    def __init__(self, metric_names: List[str], n_qns: int):
        self.metric_names = metric_names
        self.n_qns = n_qns
        self.eval_metrics = {}
        self._build()

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Parameters:
            y_true: groundtruths
            y_pred: predicting results

        Return:
            eval_result: evaluation performance report
        """
        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            if metric_name == "f1":
                for thres in [0.63]:
                    eval_result[f"{metric_name}@{thres}"] = metric(y_pred, y_true, thres)
            else:
                eval_result[metric_name] = metric(y_pred, y_true)

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "auroc":
                self.eval_metrics[metric_name] = self._AUROC
            elif metric_name == "f1":
                self.eval_metrics[metric_name] = self._F1

    def _AUROC(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Area Under the Receiver Operating Characteristic curve.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths

        Return:
            auroc: area under the receiver operating characteristic
                curve
        """
        metric = AUROC(task="multilabel", num_labels=self.n_qns)
        _ = metric(y_pred, y_true.int())
        auroc = metric.compute().item()
        metric.reset()

        return auroc

    def _F1(self, y_pred: Tensor, y_true: Tensor, thres: float) -> float:
        """F1 score.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths
            thres: threshold to convert probability to bool

        Return:
            f1: F1 score
        """
        y_pred = (y_pred.numpy().reshape(-1, ) > thres).astype("int")
        y_true = y_true.numpy().reshape(-1, )
        f1 = f1_score(y_true, y_pred, average="macro")

        return f1


# <a id="trainer"></a>
# ## 5. Define Trainer
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[16]:


class BaseTrainer:
    """Base class for all customized trainers.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        evaluator: task-specific evaluator
    """

    def __init__(
        self,
        cfg: Type[CFG],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: _LRScheduler,
        evaluator: Evaluator,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.evaluator = evaluator

        self.device = cfg.DEVICE
        self.epochs = cfg.EPOCH

        # Model checkpoint
        self.ckpt_metric = cfg.CKPT_METRIC

        self._iter = 0
        self._track_best_model = True

    def train_eval(self, proc_id: Union[str, int]) -> Tuple[Module, Dict[str, Tensor]]:
        """Run training and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Return:
            best_model: model instance with the best monitored
                objective (e.g., the lowest validation loss)
            y_preds: inference results on different datasets
        """
        best_val_loss = 1e18
        best_epoch = 0
        try:
            best_model = deepcopy(self.model)
        except RuntimeError as e:
            best_model = None
            self._track_best_model = False
            print("In-memoey best model tracking is disabled.")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            val_metric_msg = ""
            for metric, score in val_result.items():
                val_metric_msg += f"{metric.upper()} {round(score, 4)} | "
            print(f"Epoch{epoch} | Training loss {train_loss:.4f} | Validation loss {val_loss:.4f} | {val_metric_msg}")

            # Record the best checkpoint
            ckpt_metric_val = val_result[self.ckpt_metric]
            ckpt_metric_val = -ckpt_metric_val
            if ckpt_metric_val < best_val_loss:
                print(f"Validation performance improves at epoch {epoch}!!")
                best_val_loss = ckpt_metric_val
                if self._track_best_model:
                    best_model = deepcopy(self.model)
                else:
                    self._save_ckpt()
                best_epoch = epoch

        # Run final evaluation
        if not self._track_best_model:
            self._load_best_ckpt()
            best_model = self.model
        else:
            self.model = best_model
        final_prf_report, y_preds = self._eval_with_best()
        self._log_best_prf(final_prf_report)

        return best_model, y_preds

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
                *Note: If multitask is used, returned object will be
                    a dictionary containing losses of subtasks and the
                    total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(
        self,
        return_output: bool = False,
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: if evaluation is run on test set, set it to True
                *Note: The setting is mainly used to disable DAE doping
                    during test phase.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        raise NotImplementedError

    def _eval_with_best(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tensor]]:
        """Run final evaluation process with the best checkpoint.

        Return:
            final_prf_report: performance report of final evaluation
            y_preds: inference results on different datasets
        """
        final_prf_report = {}
        y_preds = {}

        self._disable_shuffle()
        dataloaders = {"train": self.train_loader}
        if self.eval_loader is not None:
            dataloaders["val"] = self.eval_loader

        for datatype, dataloader in dataloaders.items():
            self.eval_loader = dataloader
            eval_loss, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[datatype] = eval_result
            y_preds[datatype] = y_pred

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _save_ckpt(self, proc_id: int = 0, save_best_only: bool = True) -> None:
        """Save checkpoints.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed
            save_best_only: only checkpoint of the best epoch is saved

        Return:
            None
        """
        torch.save(self.model.state_dict(), "model_tmp.pt")

    def _load_best_ckpt(self, proc_id: int = 0) -> None:
        """Load the best model checkpoint for final evaluation.

        The best checkpoint is loaded and assigned to `self.model`.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed

        Return:
            None
        """
        device = torch.device(self.device)
        self.model.load_state_dict(
            torch.load("model_tmp.pt", map_location=device)
        )
        self.model = self.model.to(device)

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Parameters:
            prf_report: performance report

        Return:
            None
        """
        import json

        print(">>>>> Performance Report - Best Ckpt <<<<<")
        print(json.dumps(prf_report, indent=4))


# In[17]:


class MainTrainer(BaseTrainer):
    """Main trainer.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        train_loader: training data loader
        eval_loader: validation data loader
    """

    def __init__(
        self,
        cfg: Type[CFG],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: _LRScheduler,
        evaluator: Evaluator,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ):
        super(MainTrainer, self).__init__(
            cfg, model, loss_fn, optimizer, lr_skd, evaluator
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            x_cat = batch_data["X_cat"].to(self.device)
            y = batch_data["y"].to(self.device)

            # Forward pass
            output = self.model(x, x_cat)
            self._iter += 1

            # Derive loss
            loss = self.loss_fn(output, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            train_loss_total += loss.item()

            # Free mem.
            del x, y, output
            _ = gc.collect()

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: always ignored, exists for compatibility

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        eval_loss_total = 0
        y_true = None
        y_pred = None

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            x_cat = batch_data["X_cat"].to(self.device)
            y = batch_data["y"].to(self.device)

            # Forward pass
            output = self.model(x, x_cat)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            if i == 0:
                y_true = y.detach().cpu()
                y_pred = output.detach().cpu()
            else:
                # Avoid situ ation that batch_size is just equal to 1
                y_true = torch.cat((y_true, y.detach().cpu()))
                y_pred = torch.cat((y_pred, output.detach().cpu()))

            del x, y, output
            _ = gc.collect()

        y_pred = F.sigmoid(y_pred)  # Tmp. workaround (because loss has built-in sigmoid)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)
        eval_result = self.evaluator.evaluate(y_true, y_pred)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None


# <a id="cv"></a>
# ## 6. Run Cross-Validation
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[18]:


if cfg.train:
    sess_id = X["session_id"].unique()

    oof_pred = pd.DataFrame(np.zeros((len(sess_id), N_QNS)), index=sess_id)
    cv = GroupKFold(n_splits=5)
    for i, (tr_idx, val_idx) in enumerate(cv.split(X=X, groups=X["session_id"])):
        print(f"Training and evaluation process of fold{i} starts...")

        # Prepare data
        X_tr, X_val = X.iloc[tr_idx, :], X.iloc[val_idx, :]
        sess_tr, sess_val = X_tr["session_id"].unique(), X_val["session_id"].unique()
        y_tr, y_val = y_lvgp[y_lvgp["session_id"].isin(sess_tr)], y_lvgp[y_lvgp["session_id"].isin(sess_val)]

        # Run level_group-wise modeling
        oof_pred_fold = []
        for lv_gp in LEVEL_GROUP:
            print(f"=====LEVEL GROUP {lv_gp}=====")
            qn_idx = QNS_PER_LV_GP[lv_gp]  # Question index
            lvs = LV_PER_LV_GP[lv_gp]
            X_tr_, X_val_ = X_tr[X_tr["level_group"] == lv_gp], X_val[X_val["level_group"] == lv_gp]
            y_tr_, y_val_ = y_tr[y_tr["level_group"] == lv_gp], y_val[y_val["level_group"] == lv_gp]

            # Build dataloader
            train_loader, val_loader = build_dataloaders((X_tr_, y_tr_), (X_val_, y_val_), cfg.BATCH_SIZE, **{"t_window": cfg.T_WINDOW})

            # Build model
            model = EventConvSimple(len(lvs), len(qn_idx), **{"cat_feats": cfg.CAT_FEATS})
            model.to(cfg.DEVICE)

            # Build criterion
            loss_fn = nn.BCEWithLogitsLoss()

            # Build solvers
            optimizer = optim.Adam(list(model.parameters()), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            lr_skd = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, eta_min=1e-5, T_mult=1)

            # Build evaluator
            evaluator = Evaluator(cfg.EVAL_METRICS, len(qn_idx))

            # Build trainer
            trainer_cfg = {
                "cfg": cfg,
                "model": model,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
                "lr_skd": lr_skd,
                "evaluator": evaluator,
                "train_loader": train_loader,
                "eval_loader": val_loader,
            }
            trainer = MainTrainer(**trainer_cfg)

            # Run training and evaluation processes for one fold
            best_model, best_preds = trainer.train_eval(lv_gp)
            oof_pred_fold.append(best_preds["val"])

            # Dump output objects of the current fold
            torch.save(best_model.state_dict(), f"fold{i}_{lv_gp}")

            # Free mem.
            del (X_tr_, X_val_, y_tr_, y_val_, train_loader, val_loader,
                 model, loss_fn, optimizer, lr_skd, evaluator, trainer)
            _ = gc.collect()

        # Record oof prediction of the current fold
        oof_pred.loc[sess_val, :] = torch.cat(oof_pred_fold, dim=1).numpy()
else:
    oof_pred = pd.read_csv("/kaggle/input/0330-16-53-13/0330-16_53_13/preds/oof.csv")
    oof_pred.set_index("session", inplace=True)
    oof_pred.rename({"session": "session_id"}, axis=1, inplace=True)


# <a id="cv_score"></a>
# ## 7. Derive Final CV Score
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[19]:


def derive_cv_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    return_traces: bool = False
) -> Tuple[float, float, Optional[Tuple[List[float], List[float]]]]:
       """Compute the fianl CV score (i.e., f1 score).

       Parameters:
           y_true: groundtruths
           y_pred: predicting results
           return_traces: if True, f1 scores at different thresholds
               are returned

       Return:
           best_f1: final CV score at the best prob threshold
           best_thres: threshold with the best CV score
           f1s: f1 scores at different thresholds
           thresholds: thresholds to convert probability to bool
       """
       y_pred = y_pred.sort_index()
       thres_range = np.arange(0.2, 0.81, 0.01)
       
       f1s, thresholds = [], []
       best_f1 = 0.0
       best_thres = 0.0
       for thres in thres_range:
           y_pred_bool = (y_pred.values.reshape(-1, ) > thres).astype("int")
           f1 = f1_score(y_true.values.reshape(-1, ), y_pred_bool, average="macro")
           f1s.append(f1)
           thresholds.append(thres)

           if f1 > best_f1:
               best_f1 = f1
               best_thres = thres

       traces = (f1s, thresholds)

       if return_traces:
           return best_f1, best_thres, traces
       else:
           return best_f1, best_thres, None


# In[20]:


best_f1, best_thres, (f1s, thresholds) = derive_cv_score(y_all, oof_pred, return_traces=True)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(thresholds, f1s, "-o", color="blue")
ax.scatter([best_thres], [best_f1], color="red", s=300, alpha=1)
ax.set_title(f"Threshold vs. F1 with Best F1 = {best_f1:.4f} at Best Threshold = {best_thres:.3}")
ax.set_xlabel("Threshold", size=14)
ax.set_ylabel("CV Score (F1)", size=14)
plt.show()


# In[ ]:




