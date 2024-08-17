#!/usr/bin/env python
# coding: utf-8

# # LightGBM Quickstart for the *American Express - Default Prediction* competition
# 
# This notebook shows how to apply LightGBM to the competition data, and it introduces a space-efficient way of feature engineering.
# 
# It is based on the [EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/amex-eda-which-makes-sense).

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats
import warnings
from colorama import Fore, Back, Style
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay
from lightgbm import LGBMClassifier, log_evaluation

plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
plt.rcParams['text.color'] = 'w'

INFERENCE = True # set to False if you only want to cross-validate


# In[2]:


# @yunchonggan's fast metric implementation
# From https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)


# # Reading and preprocessing the data
# 
# We read the data from @raddar's [dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format). @raddar has [denoised the data](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514) so that we can achieve better results with his dataset than with the original competition csv files.
# 
# Then we create three groups of features:
# - Selected features averaged over all statements of a customer
# - The minimum or maximum of selected features over all statements of a customer
# - Selected features taken from the last statement of a customer
# 
# The code has been optimized for memory efficiency rather than readability. In particular, `.iloc[mask_array, columns]` needs much less RAM than the groupby construction used in a previous version of the notebook.
# 
# Preprocessing for LightGBM is much simpler than for neural networks:
# 1. Neural networks can't process missing values; LightGBM handles them automatically.
# 1. Categorical features need to be one-hot encoded for neural networks; LightGBM handles them automatically.
# 1. With neural networks, you need to think about outliers; tree-based algorithms deal with outliers easily.
# 1. Neural networks need scaled inputs; tree-based algorithms don't depend on scaling.

# In[3]:


get_ipython().run_cell_magic('time', '', 'features_avg = [\'B_1\', \'B_2\', \'B_3\', \'B_4\', \'B_5\', \'B_6\', \'B_8\', \'B_9\', \'B_10\', \'B_11\', \'B_12\', \'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_18\', \'B_19\', \'B_20\', \'B_21\', \'B_22\', \'B_23\', \'B_24\', \'B_25\', \'B_28\', \'B_29\', \'B_30\', \'B_32\', \'B_33\', \'B_37\', \'B_38\', \'B_39\', \'B_40\', \'B_41\', \'B_42\', \'D_39\', \'D_41\', \'D_42\', \'D_43\', \'D_44\', \'D_45\', \'D_46\', \'D_47\', \'D_48\', \'D_50\', \'D_51\', \'D_53\', \'D_54\', \'D_55\', \'D_58\', \'D_59\', \'D_60\', \'D_61\', \'D_62\', \'D_65\', \'D_66\', \'D_69\', \'D_70\', \'D_71\', \'D_72\', \'D_73\', \'D_74\', \'D_75\', \'D_76\', \'D_77\', \'D_78\', \'D_80\', \'D_82\', \'D_84\', \'D_86\', \'D_91\', \'D_92\', \'D_94\', \'D_96\', \'D_103\', \'D_104\', \'D_108\', \'D_112\', \'D_113\', \'D_114\', \'D_115\', \'D_117\', \'D_118\', \'D_119\', \'D_120\', \'D_121\', \'D_122\', \'D_123\', \'D_124\', \'D_125\', \'D_126\', \'D_128\', \'D_129\', \'D_131\', \'D_132\', \'D_133\', \'D_134\', \'D_135\', \'D_136\', \'D_140\', \'D_141\', \'D_142\', \'D_144\', \'D_145\', \'P_2\', \'P_3\', \'P_4\', \'R_1\', \'R_2\', \'R_3\', \'R_7\', \'R_8\', \'R_9\', \'R_10\', \'R_11\', \'R_14\', \'R_15\', \'R_16\', \'R_17\', \'R_20\', \'R_21\', \'R_22\', \'R_24\', \'R_26\', \'R_27\', \'S_3\', \'S_5\', \'S_6\', \'S_7\', \'S_9\', \'S_11\', \'S_12\', \'S_13\', \'S_15\', \'S_16\', \'S_18\', \'S_22\', \'S_23\', \'S_25\', \'S_26\']\nfeatures_min = [\'B_2\', \'B_4\', \'B_5\', \'B_9\', \'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_19\', \'B_20\', \'B_28\', \'B_29\', \'B_33\', \'B_36\', \'B_42\', \'D_39\', \'D_41\', \'D_42\', \'D_45\', \'D_46\', \'D_48\', \'D_50\', \'D_51\', \'D_53\', \'D_55\', \'D_56\', \'D_58\', \'D_59\', \'D_60\', \'D_62\', \'D_70\', \'D_71\', \'D_74\', \'D_75\', \'D_78\', \'D_83\', \'D_102\', \'D_112\', \'D_113\', \'D_115\', \'D_118\', \'D_119\', \'D_121\', \'D_122\', \'D_128\', \'D_132\', \'D_140\', \'D_141\', \'D_144\', \'D_145\', \'P_2\', \'P_3\', \'R_1\', \'R_27\', \'S_3\', \'S_5\', \'S_7\', \'S_9\', \'S_11\', \'S_12\', \'S_23\', \'S_25\']\nfeatures_max = [\'B_1\', \'B_2\', \'B_3\', \'B_4\', \'B_5\', \'B_6\', \'B_7\', \'B_8\', \'B_9\', \'B_10\', \'B_12\', \'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_18\', \'B_19\', \'B_21\', \'B_23\', \'B_24\', \'B_25\', \'B_29\', \'B_30\', \'B_33\', \'B_37\', \'B_38\', \'B_39\', \'B_40\', \'B_42\', \'D_39\', \'D_41\', \'D_42\', \'D_43\', \'D_44\', \'D_45\', \'D_46\', \'D_47\', \'D_48\', \'D_49\', \'D_50\', \'D_52\', \'D_55\', \'D_56\', \'D_58\', \'D_59\', \'D_60\', \'D_61\', \'D_63\', \'D_64\', \'D_65\', \'D_70\', \'D_71\', \'D_72\', \'D_73\', \'D_74\', \'D_76\', \'D_77\', \'D_78\', \'D_80\', \'D_82\', \'D_84\', \'D_91\', \'D_102\', \'D_105\', \'D_107\', \'D_110\', \'D_111\', \'D_112\', \'D_115\', \'D_116\', \'D_117\', \'D_118\', \'D_119\', \'D_121\', \'D_122\', \'D_123\', \'D_124\', \'D_125\', \'D_126\', \'D_128\', \'D_131\', \'D_132\', \'D_133\', \'D_134\', \'D_135\', \'D_136\', \'D_138\', \'D_140\', \'D_141\', \'D_142\', \'D_144\', \'D_145\', \'P_2\', \'P_3\', \'P_4\', \'R_1\', \'R_3\', \'R_5\', \'R_6\', \'R_7\', \'R_8\', \'R_10\', \'R_11\', \'R_14\', \'R_17\', \'R_20\', \'R_26\', \'R_27\', \'S_3\', \'S_5\', \'S_7\', \'S_8\', \'S_11\', \'S_12\', \'S_13\', \'S_15\', \'S_16\', \'S_22\', \'S_23\', \'S_24\', \'S_25\', \'S_26\', \'S_27\']\nfeatures_last = [\'B_1\', \'B_2\', \'B_3\', \'B_4\', \'B_5\', \'B_6\', \'B_7\', \'B_8\', \'B_9\', \'B_10\', \'B_11\', \'B_12\', \'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_18\', \'B_19\', \'B_20\', \'B_21\', \'B_22\', \'B_23\', \'B_24\', \'B_25\', \'B_26\', \'B_28\', \'B_29\', \'B_30\', \'B_32\', \'B_33\', \'B_36\', \'B_37\', \'B_38\', \'B_39\', \'B_40\', \'B_41\', \'B_42\', \'D_39\', \'D_41\', \'D_42\', \'D_43\', \'D_44\', \'D_45\', \'D_46\', \'D_47\', \'D_48\', \'D_49\', \'D_50\', \'D_51\', \'D_52\', \'D_53\', \'D_54\', \'D_55\', \'D_56\', \'D_58\', \'D_59\', \'D_60\', \'D_61\', \'D_62\', \'D_63\', \'D_64\', \'D_65\', \'D_69\', \'D_70\', \'D_71\', \'D_72\', \'D_73\', \'D_75\', \'D_76\', \'D_77\', \'D_78\', \'D_79\', \'D_80\', \'D_81\', \'D_82\', \'D_83\', \'D_86\', \'D_91\', \'D_96\', \'D_105\', \'D_106\', \'D_112\', \'D_114\', \'D_119\', \'D_120\', \'D_121\', \'D_122\', \'D_124\', \'D_125\', \'D_126\', \'D_127\', \'D_130\', \'D_131\', \'D_132\', \'D_133\', \'D_134\', \'D_138\', \'D_140\', \'D_141\', \'D_142\', \'D_145\', \'P_2\', \'P_3\', \'P_4\', \'R_1\', \'R_2\', \'R_3\', \'R_4\', \'R_5\', \'R_6\', \'R_7\', \'R_8\', \'R_9\', \'R_10\', \'R_11\', \'R_12\', \'R_13\', \'R_14\', \'R_15\', \'R_19\', \'R_20\', \'R_26\', \'R_27\', \'S_3\', \'S_5\', \'S_6\', \'S_7\', \'S_8\', \'S_9\', \'S_11\', \'S_12\', \'S_13\', \'S_16\', \'S_19\', \'S_20\', \'S_22\', \'S_23\', \'S_24\', \'S_25\', \'S_26\', \'S_27\']\nfor i in [\'test\', \'train\'] if INFERENCE else [\'train\']:\n    df = pd.read_parquet(f\'../input/amex-data-integer-dtypes-parquet-format/{i}.parquet\')\n    cid = pd.Categorical(df.pop(\'customer_ID\'), ordered=True)\n    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer\n    if \'target\' in df.columns:\n        df.drop(columns=[\'target\'], inplace=True)\n    gc.collect()\n    print(\'Read\', i)\n    df_avg = (df\n              .groupby(cid)\n              .mean()[features_avg]\n              .rename(columns={f: f"{f}_avg" for f in features_avg})\n             )\n    gc.collect()\n    print(\'Computed avg\', i)\n    df_min = (df\n              .groupby(cid)\n              .min()[features_min]\n              .rename(columns={f: f"{f}_min" for f in features_min})\n             )\n    gc.collect()\n    print(\'Computed min\', i)\n    df_max = (df\n              .groupby(cid)\n              .max()[features_max]\n              .rename(columns={f: f"{f}_max" for f in features_max})\n             )\n    gc.collect()\n    print(\'Computed max\', i)\n    df = (df.loc[last, features_last]\n          .rename(columns={f: f"{f}_last" for f in features_last})\n          .set_index(np.asarray(cid[last]))\n         )\n    gc.collect()\n    print(\'Computed last\', i)\n    df = pd.concat([df, df_min, df_max, df_avg], axis=1)\n    if i == \'train\': train = df\n    else: test = df\n    print(f"{i} shape: {df.shape}")\n    del df, df_avg, df_min, df_max, cid, last\n\ntarget = pd.read_csv(\'../input/amex-default-prediction/train_labels.csv\').target.values\nprint(f"target shape: {target.shape}")\n')


# # Cross-validation
# 
# We cross-validate with a five-fold StratifiedKFold because the classes are imbalanced.
# 
# Notice that lightgbm logs the validation score with the competition's scoring function every hundred iterations.

# In[4]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\nONLY_FIRST_FOLD = False\n\nfeatures = [f for f in train.columns if f != \'customer_ID\' and f != \'target\']\n\ndef my_booster(random_state=1, n_estimators=1200):\n    return LGBMClassifier(n_estimators=n_estimators,\n                          learning_rate=0.03, reg_lambda=50,\n                          min_child_samples=2400,\n                          num_leaves=95,\n                          colsample_bytree=0.19,\n                          max_bins=511, random_state=random_state)\n      \nprint(f"{len(features)} features")\nscore_list = []\ny_pred_list = []\nkf = StratifiedKFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train, target)):\n    X_tr, X_va, y_tr, y_va, model = None, None, None, None, None\n    start_time = datetime.datetime.now()\n    X_tr = train.iloc[idx_tr][features]\n    X_va = train.iloc[idx_va][features]\n    y_tr = target[idx_tr]\n    y_va = target[idx_va]\n    \n    model = my_booster()\n    with warnings.catch_warnings():\n        warnings.filterwarnings(\'ignore\', category=UserWarning)\n        model.fit(X_tr, y_tr,\n                  eval_set = [(X_va, y_va)], \n                  eval_metric=[lgb_amex_metric],\n                  callbacks=[log_evaluation(100)])\n    X_tr, y_tr = None, None\n    y_va_pred = model.predict_proba(X_va, raw_score=True)\n    score = amex_metric(y_va, y_va_pred)\n    n_trees = model.best_iteration_\n    if n_trees is None: n_trees = model.n_estimators\n    print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now() - start_time)[-12:-7]} |"\n          f" {n_trees:5} trees |"\n          f"                Score = {score:.5f}{Style.RESET_ALL}")\n    score_list.append(score)\n    \n    if INFERENCE:\n        y_pred_list.append(model.predict_proba(test[features], raw_score=True))\n        \n    if ONLY_FIRST_FOLD: break # we only want the first fold\n    \nprint(f"{Fore.GREEN}{Style.BRIGHT}OOF Score:                       {np.mean(score_list):.5f}{Style.RESET_ALL}")\n')


# # Prediction histogram

# In[5]:


def sigmoid(log_odds):
    return 1 / (1 + np.exp(-log_odds))

plt.figure(figsize=(10, 4))
plt.hist(sigmoid(y_va_pred[y_va == 0]), bins=np.linspace(0, 1, 101),
         alpha=0.5, density=True, label='0')
plt.hist(sigmoid(y_va_pred[y_va == 1]), bins=np.linspace(0, 1, 101),
         alpha=0.5, density=True, label='1')
plt.xlabel('y_pred')
plt.ylabel('density')
plt.title('OOF Prediction histogram', color='k')
plt.legend()
plt.show()


# # Calibration diagram
# 
# The calibration diagram shows how the model predicts the default probability of customers:

# In[6]:


plt.figure(figsize=(12, 4))
CalibrationDisplay.from_predictions(y_va, sigmoid(y_va_pred), n_bins=50,
                                    strategy='quantile', ax=plt.gca())
plt.title('Probability calibration')
plt.show()


# # Submission
# 
# We submit the mean of the five predictions. As proposed by @lucasmorin, we [take the mean of the log odds](https://www.kaggle.com/competitions/amex-default-prediction/discussion/329103) rather than of the probabilities.

# In[7]:


if INFERENCE:
    sub = pd.DataFrame({'customer_ID': test.index,
                        'prediction': np.mean(y_pred_list, axis=0)})
    sub.to_csv('submission.csv', index=False)
    display(sub)


# As a final check, we verify that the test prediction distribution equals the validation prediction distribution. 

# In[8]:


plt.figure(figsize=(12, 4))
plt.hist(sigmoid(sub.prediction), bins=np.linspace(0, 1, 101), density=True)
plt.hist(sigmoid(y_va_pred), bins=np.linspace(0, 1, 101), rwidth=0.5, color='orange', density=True)
plt.show()


# In[ ]:




