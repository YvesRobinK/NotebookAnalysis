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
#     pl.col("page").cast(pl.Float32),
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
models_list = [[CatBoostClassifier().load_model(
    f"/kaggle/input/cpu-catboost-baseline-using-polars-train/fold{fold}_q{q}.cbm"
) for fold in range(5)] for q in range(1, 19)]


# In[3]:


def trans(test):
    return (pl.from_pandas(test)    
            .with_columns(columns)
            .select(aggs)
            .fill_null(-1)
            .to_pandas()
            )


# In[4]:


import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[5]:


for (sample_submission, test) in iter_test:
    target_level_group = level_groups_reverse[test.level_group.iloc[0]]
    df = trans(test)
    
    fold = 0
    preds = []
    for q in range(*questions[level_groups[target_level_group]]):
        model = models_list[q - 1][fold]
        feature_cols = model.feature_names_
        pred = model.predict_proba(df[feature_cols].astype(np.float32))[0,1]
        preds.append(int(pred > 0.63))

    sample_submission["correct"] = preds

    env.predict(sample_submission)


# In[6]:


sub = pd.read_csv('submission.csv')
print(sub.shape, sub.correct.mean())
sub.head()

