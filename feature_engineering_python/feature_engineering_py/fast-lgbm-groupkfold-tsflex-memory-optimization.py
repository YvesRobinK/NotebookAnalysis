#!/usr/bin/env python
# coding: utf-8

# # time-series 🤝 tsflex 🚀

# ### **Thanks to great works from JEROENVDD and DATAMANYO.**
# 
# ### new improvements based on groupkfold-cross-validation-tsflex
# 
# 1. use LGBM model with early stop
# 2. memory optimization by dumpping middle fold train data on disk
# 
# ### past improvements 
# 
# 1. add metadata infomation and subject infomation, fix a little bug of subject feature in JEROENVDD‘s origin notebook
# 2. make GroupKfold Cross Validation
# 
# #### reference
# 
# * @JEROENVDD
#     * https://www.kaggle.com/code/jeroenvdd/time-series-tsflex
# * @DATAMANYO
#     * https://www.kaggle.com/code/kimtaehun/simple-lgbm-multi-class-classification-baseline
# * @此般浅薄
#     * https://www.kaggle.com/code/xzj19013742/groupkfold-cross-validation-tsflex
# 
# <br>

# <div style="background-color:#f2f2f2; padding:20px; border-radius: 10px;">
#     <h2 style="color:#595959;">Check out <a href="https://github.com/predict-idlab/tsflex" target="_blank" style="color:#0099cc;">tsflex</a>!</h2>
#     <h4 style="color:#737373;">tsflex is a Python package for flexible and efficient time series feature extraction. It's great for data preprocessing and feature engineering for time series data. Check it out on <a href="https://github.com/predict-idlab/tsflex" target="_blank" style="color:#0099cc;">GitHub</a> today!</h4>
#     
# <p style="color:#737373;">This notebook is a fork of the <a href="https://www.kaggle.com/code/jazivxt/familiar-solvs" target="_blank" style="color:#0099cc;">familiar-solvs notebook of jazivxt</a> and adds tsflex to extract some basic <a href="https://github.com/dmbee/seglearn" target="_blank" style="color:#0099cc;">seglearn</a> features.</p>
#     
# </div>

# In[1]:


# Install tsflex and seglearn
get_ipython().system('pip install tsflex --no-index --find-links=file:///kaggle/input/time-series-tools')
get_ipython().system('pip install seglearn --no-index --find-links=file:///kaggle/input/time-series-tools')


# In[2]:


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn import *
import glob

p = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'

train = glob.glob(p+'train/**/**')
test = glob.glob(p+'test/**/**')
subjects = pd.read_csv(p+'subjects.csv')
tasks = pd.read_csv(p+'tasks.csv')
sub = pd.read_csv(p+'sample_submission.csv')

tdcsfog_metadata=pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv')
defog_metadata=pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv')
# daily_metadata=pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/daily_metadata.csv')
tdcsfog_metadata['Module']='tdcsfog'
defog_metadata['Module']='defog'
# daily_metadata['Module']='daily'
metadata=pd.concat([tdcsfog_metadata,defog_metadata])
metadata


# In[3]:


# https://www.kaggle.com/code/jazivxt/familiar-solvs
tasks['Duration'] = tasks['End'] - tasks['Begin']
tasks = pd.pivot_table(tasks, values=['Duration'], index=['Id'], columns=['Task'], aggfunc='sum', fill_value=0)
tasks.columns = [c[-1] for c in tasks.columns]
tasks = tasks.reset_index()
tasks['t_kmeans'] = cluster.KMeans(n_clusters=10, random_state=3).fit_predict(tasks[tasks.columns[1:]])

subjects = subjects.fillna(0).groupby('Subject').median()
subjects = subjects.reset_index()
# subjects.rename(columns={'Subject':'Id'}, inplace=True)
subjects['s_kmeans'] = cluster.KMeans(n_clusters=10, random_state=3).fit_predict(subjects[subjects.columns[1:]])
subjects=subjects.rename(columns={'Visit':'s_Visit','Age':'s_Age','YearsSinceDx':'s_YearsSinceDx','UPDRSIII_On':'s_UPDRSIII_On','UPDRSIII_Off':'s_UPDRSIII_Off','NFOGQ':'s_NFOGQ'})

display(tasks)
display(subjects)


# ## merge metadata and subject info

# In[4]:


import pathlib


complex_featlist=['Visit','Test','Medication','s_Visit','s_Age','s_YearsSinceDx','s_UPDRSIII_On','s_UPDRSIII_Off','s_NFOGQ','s_kmeans']
metadata_complex=metadata.merge(subjects,how='left',on='Subject').copy()
metadata_complex['Medication']=metadata_complex['Medication'].factorize()[0]
train_ids=[pathlib.Path(i).parts[-1].split('.')[0] for i in glob.glob(p+'train/**/**')]
metadata_complex['is_train']=metadata_complex['Id'].isin(train_ids)

display(metadata_complex)


# In[ ]:





# ## Create a tsflex feature collection

# In[5]:


from seglearn.feature_functions import base_features, emg_features

from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
from tsflex.features.integrations import seglearn_feature_dict_wrapper


basic_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(base_features()),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[5_000],
    strides=[5_000],
)

emg_feats = emg_features()
del emg_feats['simple square integral'] # is same as abs_energy (which is in base_features)

emg_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(emg_feats),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[5_000],
    strides=[5_000],
)

fc = FeatureCollection([basic_feats, emg_feats])


# ## Extract the features for 5 train fold

# In[6]:


import pathlib
def reader(f):
    try:
        df = pd.read_csv(f, index_col="Time", usecols=['Time', 'AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn' , 'Walking'])
        df['Id'] = f.split('/')[-1].split('.')[0]
        df['Module'] = pathlib.Path(f).parts[-2]
        df = pd.merge(df, tasks[['Id','t_kmeans']], how='left', on='Id').fillna(-1)
#         df = pd.merge(df, subjects[['Id','s_kmeans']], how='left', on='Id').fillna(-1)
        df = pd.merge(df, metadata_complex[['Id','Subject']+['Visit','Test','Medication','s_kmeans']], how='left', on='Id').fillna(-1)
        df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin").astype(np.float32)
        df = df.merge(df_feats, how="left", left_index=True, right_index=True)
        df.fillna(method="ffill", inplace=True)
        return df
    except: pass
# train = pd.concat([reader(f) for f in tqdm(train)]).fillna(0); print(train.shape)
# cols = [c for c in train.columns if c not in ['Id','Subject','Module', 'Time', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']]
# pcols = ['StartHesitation', 'Turn' , 'Walking']
# scols = ['Id', 'StartHesitation', 'Turn' , 'Walking']


# In[7]:


from sklearn.model_selection import GroupKFold
N_FOLDS=5
n_use_fold=5
choices = [1, 2, 3]

train_metadata_complex=metadata_complex[metadata_complex['is_train']==True].reset_index(drop=True)


kfold = GroupKFold(N_FOLDS)
groups=kfold.split(train_metadata_complex, groups=train_metadata_complex.Subject)
groups=list(groups)

#### get Id of each fold
fold_idss=[train_metadata_complex.loc[i[1],['Module','Id']].drop_duplicates().apply(
        lambda x:f"/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/{x['Module']}/{x['Id']}.csv",axis=1
    ).tolist()
           for i in groups]


# In[8]:


for fold,fold_ids in enumerate(fold_idss):
    fold_train=pd.concat([reader(f) for f in tqdm(fold_ids)]).fillna(0); print(fold_train.shape)
    fold_train['fold']=fold
    conditions = [
        (fold_train['StartHesitation'] == 1),
        (fold_train['Turn'] == 1),
        (fold_train['Walking'] == 1)]

    fold_train['event'] = np.select(conditions, choices, default=0)
    fold_train.to_parquet('fold_train.pq',partition_cols=['fold'])

cols = [c for c in fold_train.columns if c not in ['Id','Subject','Module', 'Time', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event','event','fold']]
pcols = ['StartHesitation', 'Turn' , 'Walking']
scols = ['Id', 'StartHesitation', 'Turn' , 'Walking']


# ## Train the model

# In[9]:


import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')



#setting up the parameters
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=7
params['num_class']=4 #no.of unique values in the target class not inclusive of the end value
params['verbose']=-1


regs=[]
cvs=[]

for fold in tqdm(range(N_FOLDS), total=N_FOLDS, desc="Folds"):
    if fold>=n_use_fold:
        break
    train_filter=[[('fold','=',fld)] for fld in range(N_FOLDS) if fld !=fold]
    train=pd.read_parquet('fold_train.pq',filters=train_filter).sample(n=2000000,random_state=100).reset_index(drop=True)
    valid=pd.read_parquet('fold_train.pq',filters=[('fold','=',fold)])
    x_tr,y_tr,e_tr=train[cols],train[pcols],train['event']
    x_te,y_te,e_te=valid[cols],valid[pcols],valid['event']    
    
    lgb_train = lgb.Dataset(x_tr, e_tr)
    lgb_valid = lgb.Dataset(x_te, e_te)
    
    #training the model
    reg = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 10500,
        valid_sets = [lgb_train, lgb_valid],
        early_stopping_rounds = 50,
        verbose_eval = 500,
        )
    
    
    regs.append(reg)
    cv=metrics.average_precision_score(y_te, reg.predict(x_te)[:,1:].clip(0.0,1.0))
    cvs.append(cv)

display(pd.Series(cvs).to_frame().T)
# print(cvs)
print(sum(cvs)/n_use_fold)


# In[10]:


valid_thre=0
valid_modelids=[ind for ind,cv in enumerate(cvs) if cv>valid_thre]
valid_modelids


# ## Predict for test

# In[11]:


sub['t'] = 0
submission = []
for f in test:
    df = pd.read_csv(f)
    df.set_index('Time', drop=True, inplace=True)
    df['Id'] = f.split('/')[-1].split('.')[0]
#     df = df.fillna(0).reset_index(drop=True)
    df = pd.merge(df, tasks[['Id','t_kmeans']], how='left', on='Id').fillna(-1)
    df = pd.merge(df, metadata_complex[['Id','Subject']+['Visit','Test','Medication','s_kmeans']], how='left', on='Id').fillna(-1)
    df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin")
    df = df.merge(df_feats, how="left", left_index=True, right_index=True)
    df.fillna(method="ffill", inplace=True)
    
    res_vals=[]
    for i_fold in range(n_use_fold):
        if i_fold in valid_modelids:
            res_val=np.round(regs[i_fold].predict(df[cols])[:,1:].clip(0.0,1.0),5)
            res_vals.append(np.expand_dims(res_val,axis=2))
    res_vals=np.mean(np.concatenate(res_vals,axis=2),axis=2)
    res = pd.DataFrame(res_vals, columns=pcols)
    
    df = pd.concat([df,res], axis=1)
    df['Id'] = df['Id'].astype(str) + '_' + df.index.astype(str)
    submission.append(df[scols])
submission = pd.concat(submission)
submission = pd.merge(sub[['Id','t']], submission, how='left', on='Id').fillna(0.0)
submission[scols].to_csv('submission.csv', index=False)


# In[ ]:




