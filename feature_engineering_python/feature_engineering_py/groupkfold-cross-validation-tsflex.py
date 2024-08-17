#!/usr/bin/env python
# coding: utf-8

# # time-series 🤝 tsflex 🚀

# ### **Thanks to great works from JEROENVDD.**
# 
# ### improvements
# 
# 1. add metadata infomation and subject infomation, fix a little bug of subject feature in JEROENVDD‘s origin notebook
# 2. make GroupKfold Cross Validation
# 
# #### reference
# 
# * @JEROENVDD
#     * Origin notebook link https://www.kaggle.com/code/jeroenvdd/time-series-tsflex
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


# In[4]:


complex_featlist=['Visit','Test','Medication','s_Visit','s_Age','s_YearsSinceDx','s_UPDRSIII_On','s_UPDRSIII_Off','s_NFOGQ','s_kmeans']
metadata_complex=metadata.merge(subjects,how='left',on='Subject').copy()
metadata_complex['Medication']=metadata_complex['Medication'].factorize()[0]

display(metadata_complex)


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


# ## Extract the features

# In[ ]:





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
train = pd.concat([reader(f) for f in tqdm(train)]).fillna(0); print(train.shape)
cols = [c for c in train.columns if c not in ['Id','Subject','Module', 'Time', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']]
pcols = ['StartHesitation', 'Turn' , 'Walking']
scols = ['Id', 'StartHesitation', 'Turn' , 'Walking']


# In[7]:


train=train.reset_index(drop=True)


# ## Train the model

# In[8]:


from sklearn.model_selection import GroupKFold

N_FOLDS=5
kfold = GroupKFold(N_FOLDS)
group_var = train.Subject
groups=kfold.split(train, groups=group_var)
regs=[]
cvs=[]
for fold, (tr_idx,te_idx ) in enumerate(tqdm(groups, total=N_FOLDS, desc="Folds")):
    tr_idx=pd.Series(tr_idx).sample(n=2000000,random_state=100).values
    reg = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=7, n_jobs=-1, random_state=3)
    x_tr,y_tr=train.loc[tr_idx,cols],train.loc[tr_idx,pcols]
    x_te,y_te=train.loc[te_idx,cols],train.loc[te_idx,pcols]
    reg.fit(x_tr,y_tr)
    regs.append(reg)
    cv=metrics.average_precision_score(y_te, reg.predict(x_te).clip(0.0,1.0))
    cvs.append(cv)
print(cvs)

# This should be some proper cross validation..
x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train[pcols], test_size=.10, random_state=3, stratify=train[pcols])
reg = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=7, n_jobs=-1, random_state=3)
reg.fit(x2,y2)
print(metrics.average_precision_score(y1[:1_000_000], reg.predict(x1[:1_000_000]).clip(0.0,1.0)))
# ## Predict for test

# In[9]:


sub['t'] = 0
submission = []
for f in test:
    df = pd.read_csv(f)
    df.set_index('Time', drop=True, inplace=True)
    df['Id'] = f.split('/')[-1].split('.')[0]
#     df = df.fillna(0).reset_index(drop=True)
    df = pd.merge(df, tasks[['Id','t_kmeans']], how='left', on='Id').fillna(-1)
#     df = pd.merge(df, subjects[['Id','s_kmeans']], how='left', on='Id').fillna(-1)
    df = pd.merge(df, metadata_complex[['Id','Subject']+['Visit','Test','Medication','s_kmeans']], how='left', on='Id').fillna(-1)
    df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin")
    df = df.merge(df_feats, how="left", left_index=True, right_index=True)
    df.fillna(method="ffill", inplace=True)
#     res = pd.DataFrame(np.round(reg.predict(df[cols]).clip(0.0,1.0),3), columns=pcols)
    
    res_vals=[]
    for i_fold in range(N_FOLDS):
        res_val=np.round(regs[i_fold].predict(df[cols]).clip(0.0,1.0),3)
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




