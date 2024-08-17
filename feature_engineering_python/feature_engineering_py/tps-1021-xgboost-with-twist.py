#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import gc
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


# ## Load source datasets

# In[2]:


train_df = pd.read_csv("../input/tabular-playground-series-oct-2021/train.csv")
train_df.set_index('id', inplace=True)
print(f"train_df: {train_df.shape}")
train_df.head()


# In[3]:


test_df = pd.read_csv("../input/tabular-playground-series-oct-2021/test.csv")
test_df.set_index('id', inplace=True)
print(f"test_df: {test_df.shape}")
test_df.head()


# ## Feature Engineering

# In[4]:


cat_cols = [col for col in test_df.columns if train_df[col].nunique() < 5]
num_cols = [col for col in test_df.columns if col not in cat_cols]
print(f"cat_cols: {len(cat_cols)} \nnum_cols: {len(num_cols)}")


# In[5]:


train_df[num_cols] = train_df[num_cols].astype('float32')
train_df[cat_cols] = train_df[cat_cols].astype('uint8')

test_df[num_cols] = test_df[num_cols].astype('float32')
test_df[cat_cols] = test_df[cat_cols].astype('uint8')

print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")

features = test_df.columns.tolist()
print(f"Num features: {len(features)}")

cat_cols_indices = [train_df.columns.get_loc(col) for col in cat_cols]
print(f"cat_cols_indices: {cat_cols_indices}")


# ## Helper Function

# In[6]:


def plot_confusion_matrix(cm, classes):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()


# ## Model Hyperparameters

# In[7]:


FOLD = 10
SEEDS = [791, 225, 508]

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'use_label_encoder': False,
    'n_estimators': 10000,
    'max_depth': 6,
    'gamma': 0.6408,
    'subsample': 0.7,
    'colsample_bytree': 0.3,
    'colsample_bylevel': 0.6,
    'min_child_weight': 56.42,
    'reg_lambda': 75.567,
    'reg_alpha': 0.1177,
    'verbosity': 0,
    'random_state': 2021
}


# ## XGBoost Model

# In[8]:


counter = 0
oof_score = 0
y_pred_final_xgb = np.zeros((test_df.shape[0], 1))
y_pred_meta_xgb = np.zeros((train_df.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(train_df[features], train_df['target'])):
        counter += 1

        train_x, train_y = train_df[features].iloc[train], train_df['target'].iloc[train]
        val_x, val_y = train_df[features].iloc[val], train_df['target'].iloc[val]

        params['learning_rate']=0.03
        init_model = XGBClassifier(**params)

        init_model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                       early_stopping_rounds=200, verbose=500)

        params['learning_rate']=0.01
        model = XGBClassifier(**params)

        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                  early_stopping_rounds=200, verbose=300, xgb_model=init_model)
        
        y_pred = model.predict_proba(val_x, iteration_range=(0, model.best_iteration))[:,-1]
        y_pred_meta_xgb[val] += np.array([y_pred]).T
        y_pred_final_xgb += np.array([model.predict_proba(test_df, iteration_range=(0, model.best_iteration))[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nXGBoost | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nXGBoost | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_xgb = y_pred_meta_xgb / float(len(SEEDS))
y_pred_final_xgb = y_pred_final_xgb / float(counter)
oof_score /= float(counter)
print("XGBoost | Aggregate OOF Score: {}".format(oof_score))


# In[9]:


y_pred_meta = np.mean(y_pred_meta_xgb, axis=1)
y_pred = (y_pred_meta>0.5).astype(int)
print(classification_report(train_df['target'], y_pred))


# In[10]:


cnf_matrix = confusion_matrix(train_df['target'], y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# ## Save meta features

# In[11]:


np.savez_compressed('./TPS_1021_XGB_Meta_Features.npz',
                    y_pred_meta_xgb=y_pred_meta_xgb,  
                    y_pred_final_xgb=y_pred_final_xgb)


# ## Create submission files

# In[12]:


submit_df = pd.read_csv("../input/tabular-playground-series-oct-2021/sample_submission.csv")
submit_df['target'] = y_pred_final_xgb.ravel()
submit_df.to_csv("XGB_Submission.csv", index=False)
submit_df.head()


# In[ ]:




