#!/usr/bin/env python
# coding: utf-8

# ## Using Polars
# 
# I learnt `polars` from last OTTO competition, which helped our team to do fast feature engineering when we faced `cudf`'s GPU memory errors.
# 
# In this competition, although the data size is not as large as OTTO's, it is still convenient to use `polars` with cpu for easy train-inference adaption. Moreover, from my own experiment, the `polars` is faster than `pandas` in Kaggle's notebook even there is only 2-core CPU avalaible, which restricts the full strength of `polars` parallelism.
# 
# And last, the current Kaggle environment (2023-02-17) has `polars` supported! You can save about 40s from installing it.

# In[1]:


import os
import gc
import sys

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from catboost import CatBoostClassifier, Pool

level_groups = ["0-4", "5-12", "13-22"]
level_groups_reverse = {'0-4': 0, '5-12': 1, '13-22': 2}
levels = {'0-4': (0, 5), '5-12': (5, 13), '13-22': (13, 23)}
questions = {'0-4': (1, 4), '5-12': (4, 14), '13-22': (14, 19)}
EVENTS = ['checkpoint', 'cutscene_click', 'map_click', 'map_hover', 'navigate_click', 'notebook_click', 'notification_click', 'object_click', 'object_hover', 'observation_click', 'person_click']
CATS = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMS = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration', "time_past"]


# In[2]:


columns = [
    pl.col("page").cast(pl.Float32),
    (
        (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
         .fill_null(0)
         .clip(0, 1e9)
         .over(["session_id", "level_group"])
         .alias("time_past")
    ),
]
aggs = [
    *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique") for c in CATS],
    *[pl.col(c).mean().alias(f"{c}_mean") for c in NUMS],
    *[pl.col(c).std().alias(f"{c}_std") for c in NUMS],
    *[(pl.col("event_name") == c).sum().alias(f"{c}_sum") for c in EVENTS],
]

label = (pl.read_csv("/kaggle/input/predict-student-performance-from-game-play/train_labels.csv")
    .select([
        pl.col("session_id").str.split("_q").arr.get(0).cast(pl.Int64),
        pl.col("session_id").str.split("_q").arr.get(1).cast(pl.Int32).alias("qid"),
        pl.col("correct").cast(pl.UInt8)
    ])
    .sort(["session_id", "qid"])
    .groupby("session_id")
    .agg(pl.col("correct"))
    .select([
        pl.col("session_id"),
        *[pl.col("correct").arr.get(i).alias(f"correct_{i + 1}") for i in range(18)]
    ])
)

df = (pl.read_csv("/kaggle/input/predict-student-performance-from-game-play/train.csv")
    .drop(["fullscreen", "hq", "music"])
    .with_columns(columns)
    .groupby(["session_id", "level_group"], maintain_order = True)
    .agg(aggs)
    .sort(["session_id", "level_group"])
    .join(label, on = "session_id", how = "left")
    .fill_null(-1)
    .to_pandas()
)


gc.collect()


# In[3]:


models = {}
results = [[[], []] for _ in range(18)]
split = list(GroupKFold(5).split(df["session_id"].unique(), groups = df["session_id"].unique()))
for q in tqdm(range(1, 19)):
    if q <= 3: grp = '0-4'
    elif q <= 13: grp = '5-12'
    elif q <= 22: grp = '13-22'
    for fold, (train_idx, valid_idx) in enumerate(split):
        df_train = df.query("level_group==@grp").iloc[train_idx]
        df_valid = df.query("level_group==@grp").iloc[valid_idx]
        feature_cols = sorted([_ for _ in df_train.columns if _ not in ["session_id", "level_group"] and not _.startswith("correct_")])
        train_pool = Pool(df_train[feature_cols].astype(np.float32), 
                          df_train[f"correct_{q}"])
        valid_pool = Pool(df_valid[feature_cols].astype(np.float32), 
                          df_valid[f"correct_{q}"])
        
        model = CatBoostClassifier(
            iterations = 1000,
            early_stopping_rounds = 50,
            depth = 4,
            learning_rate = 0.05,
            loss_function = "Logloss",
            random_seed = 0,
            metric_period = 1,
            subsample = 0.8,
            colsample_bylevel = 0.4,
            verbose = 0,
        )
        
        model = model.fit(train_pool, eval_set = valid_pool)
        
        y = valid_pool.get_label()
        yhat = model.predict_proba(valid_pool)[:,1]

        models[(fold, q)] = model
        
        results[q - 1][0].append(y)
        results[q - 1][1].append(yhat)
results = [[np.concatenate(_) for _ in _] for _ in results]


# In[4]:


for (fold, q), model in models.items():
    model.save_model(f"fold{fold}_q{q}.cbm")

true = pd.DataFrame(np.stack([_[0] for _ in results]).T)
oof = pd.DataFrame(np.stack([_[1] for _ in results]).T)

scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4, 0.81, 0.01):
    preds = (oof.values.reshape(-1) > threshold).astype('int')
    m = f1_score(true.values.reshape(-1), preds, average = 'macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m > best_score:
        best_score = m
        best_threshold = threshold
        
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()

print(f'When using optimal threshold = {best_threshold:.2f}...')
for k in range(18):
    m = f1_score(true[k].values, (oof[k].values > best_threshold).astype('int'), average = 'macro')
    print(f'Q{k}: F1 =',m)
m = f1_score(true.values.reshape(-1), (oof.values > best_threshold).reshape(-1).astype('int'), average = 'macro')
print('==> Overall F1 =', m)

