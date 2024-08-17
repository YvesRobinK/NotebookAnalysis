#!/usr/bin/env python
# coding: utf-8

# # Third Notebook for my Blending Notebook
# 
# - [TPS Oct 2021 - The Melling Blend](https://www.kaggle.com/mmellinger66/tps-oct-2021-the-melling-blend)
# 
# ### Files needed for the blend
# 
# - train_pred_3.csv
# - test_pred_3.csv
# 
# # References
# 
# - https://www.kaggle.com/mohammadkashifunique/single-xgboost-model-featureengineering
# - https://www.kaggle.com/joecooper/tsp-single-xgboost-model
# 

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from pathlib import Path


# # Configuration

# In[ ]:


class Config:
    debug = False
    competition = "TPS_202110"
    seed = 42
    NFOLDS = 5
    EPOCHS = 10


# In[ ]:


data_dir = Path('../input/tabular-playground-series-oct-2021') # Change me every month


# # Load Data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = pd.read_csv(data_dir / "train.csv",\n#                       nrows=100000\n                      )\n\ntest_df = pd.read_csv(data_dir / "test.csv")\nsample_submission = pd.read_csv(data_dir / "sample_submission.csv")\n\nprint(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")\nprint(f"test data : Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")\n')


# # Reduce Memory
# 
# Too many memory issues. Got a function to reduce the float and int types by checking the max column value and setting column to minimum necessary type.
# 
# - https://www.kaggle.com/hrshuvo/tps-oct-21-xgb-kfold
# - https://www.kaggle.com/rinnqd/reduce-memory-usage

# In[ ]:


# this function will help to reduce momory 
# data will be smaller with the same value

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = reduce_mem_usage(train_df)\ntest_df  = reduce_mem_usage(test_df)\n')


# # Feature Engineering

# In[ ]:


train_df['std'] = train_df.std(axis=1)
train_df['min'] = train_df.min(axis=1)
train_df['max'] = train_df.max(axis=1)

test_df['std'] = test_df.std(axis=1)
test_df['min'] = test_df.min(axis=1)
test_df['max'] = test_df.max(axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# train_df = pd.DataFrame (data=scaler.fit_transform(train_df), columns=train_df.columns)
# test_df = pd.DataFrame (data=scaler.transform(test_df), columns=test_df.columns)


# In[ ]:


features = [col for col in train_df.columns if col not in ('id', 'target')]


# # Extract Target and Drop Unused Columns

# In[ ]:


y = train_df.target

test_df = test_df.drop(["id"], axis=1)
X = train_df.drop(["id", "target"], axis=1)


# In[ ]:


# X['std'] = X.std(axis=1)
# X['min'] = X.min(axis=1)
# X['max'] = X.max(axis=1)

# test_df['std'] = test_df.std(axis=1)
# test_df['min'] = test_df.min(axis=1)
# test_df['max'] = test_df.max(axis=1)


# In[ ]:


X.head()


# # Model

# In[ ]:


# Run Quickly

xgb_params000 = {
    'device_type':'gpu',  # Use cpu/gpu
    'gpu_id':0,
    'gpu_platform_id':0,
#     'objective':'binary:logistic',
    'use_label_encoder': False,
    'tree_method': 'gpu_hist',

    'metric': 'auc',
#     'num_leaves': 150,
    'learning_rate': 0.05,
    'max_depth': 3,

#     'n_estimators': 10000,
    }


# In[ ]:


xgb_params = {
    'max_depth': 6,
    'n_estimators': 9500,
    'subsample': 0.7,
    'colsample_bytree': 0.2,
    'colsample_bylevel': 0.6000000000000001,
    'min_child_weight': 56.41980735551558,
    'reg_lambda': 75.56651890088857,
    'reg_alpha': 0.11766857055687065,
    'gamma': 0.6407823221122686,
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'use_label_encoder': False
    }


# ## XGBoost
# 
# 3 Chained Models, slowly relaxing the learning rate

# In[ ]:


final_test_predictions = []
final_valid_predictions = {}
scores = []

kf = StratifiedKFold(n_splits=Config.NFOLDS, shuffle=True, random_state=Config.seed)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X = X, y = y)):

    print(10*"=", f"Fold={fold}", 10*"=")

    x_train = X.loc[train_idx, :]
    x_valid = X.loc[valid_idx, :]
    
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    xgb_params['learning_rate']=0.007
    model1 = XGBClassifier(**xgb_params)

    model1.fit(x_train, y_train,
          early_stopping_rounds=100,
          eval_set=[(x_valid, y_valid)],
          verbose=0)

    xgb_params['learning_rate']=0.01
    model2 = XGBClassifier(**xgb_params)
    
    model2.fit(x_train, y_train,
          early_stopping_rounds=100,
          eval_set=[(x_valid, y_valid)],
          verbose=0,
          xgb_model=model1)

    xgb_params['learning_rate']=0.05
    model3 = XGBClassifier(**xgb_params)
    
    model3.fit(x_train, y_train,
          early_stopping_rounds=100,
          eval_set=[(x_valid, y_valid)],
          verbose=0,
          xgb_model=model2)

    
    preds_valid = model3.predict_proba(x_valid)[:, -1]
    # Want probability or classification?
    final_valid_predictions.update(dict(zip(valid_idx, preds_valid)))

    auc = roc_auc_score(y_valid,  preds_valid)
    print('auc: ', auc)
    scores.append(auc)
    
#     test_preds = model.predict_proba(test_df[features])[:, -1]
    test_preds = model3.predict_proba(test_df)[:, -1]

    final_test_predictions.append(test_preds)


# In[ ]:


print(f"scores -> mean: {np.mean(scores)}, std: {np.std(scores)}")


# # Save OOF Predictions
# 
# Save the dictionary that we created for all the training predictions that were made when each fold was used for validation

# In[ ]:


final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_3"]
final_valid_predictions.to_csv("train_pred_3.csv", index=False)


# In[ ]:


# print(model.objective)


# # Predict on Test Data

# In[ ]:


df = pd.DataFrame(np.column_stack(final_test_predictions))
df['mean'] = df.mean(axis=1)
df


# # Submission File

# In[ ]:


sample_submission['target'] = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.to_csv("test_pred_3.csv",index=None)
sample_submission.to_csv("basic_xgb_cv_fe.csv",index=None)
sample_submission


# In[ ]:




