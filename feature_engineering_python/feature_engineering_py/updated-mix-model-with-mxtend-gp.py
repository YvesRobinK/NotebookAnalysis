#!/usr/bin/env python
# coding: utf-8

# # <span style="color:red;">Machine learning-detected signal predicts time to earthquake</span>

# In[ ]:


get_ipython().run_cell_magic('html', '', '<h1><b>LIVE EARTHQUACK MAP</b></h1>\n<iframe width="800" height="600" src="https://ds.iris.edu/seismon/" allowfullscreen style="align:center;"></iframe>\n')


# <span style="color:blue;"><strong>ABOUT COMPETITION</strong></span>
# -------------------------------------
# <div class="competition-overview__content"><div><div class="markdown-converter__text--rendered"><img src="https://storage.googleapis.com/kaggle-media/competitions/LANL/nik-shuliahin-585307-unsplash.jpg" alt="map" width="300" style="float:right;" class="hoverZoomLink">
# > <p>Forecasting earthquakes is one of the most important problems in Earth science because of their devastating consequences. Current scientific studies related to earthquake forecasting focus on three key points: <b>when</b> the event will occur, <b>where</b> it will occur, and <b>how large</b> it will be.</p>  
# 
# <span style="color:blue;"><strong>WHAT THEY WANT:</strong></span>
# > <p>In this competition, you will address <b>when</b> the earthquake will take place. Specifically, you’ll predict the time remaining before laboratory earthquakes occur from real-time seismic data. </p>
# 
# <span style="color:blue;"><strong>CHALLANGE:</strong></span>
# > <p>If this challenge is solved and the **physics are ultimately shown to scale from the laboratory to the field**, researchers will have the potential to **improve earthquake hazard assessments** that could **save lives and billions of dollars in infrastructure.**This challenge is hosted by  <a href="https://www.lanl.gov/" rel="nofollow">Los Alamos National Laboratory</a> which enhances national security by ensuring the safety of the U.S. nuclear stockpile, developing technologies to reduce threats from weapons of mass destruction, and solving problems related to energy, environment, infrastructure, health, and global security concerns.</p>
# 
# ### <span style="color:red;"><strong>SUBMISSION FORMAT</strong></span>
# > * Submissions are evaluated using the [**mean absolute error**](https://en.wikipedia.org/wiki/Mean_absolute_error) between the predicted time remaining before the next lab earthquake and the act remaining time.
# 
# ### <span style="color:red;"><strong>Submission File</strong></span>
# 
# For each `seg_id` in the test set folder, you must predict `time_to_failure`, which is the remaining time before the next lab earthquake. The file should contain a header and have the following format:
# 
#     seg_id,time_to_failure
#     seg_00030f,0
#     seg_0012b5,0
#     seg_00184e,0
#     ...
# 
# 
# <span style="color:red;">**GOAL OF COMPETITION**</span>
# -----------------
# 
# * ***The goal of this competition is to use seismic signals to predict the timing of laboratory earthquakes.*** The *data comes from a well-known experimental set-up used to study earthquake physics.* The` acoustic_data` input signal is used to **predict the time remaining before the next laboratory earthquake (time_to_failure).**
# * The ***training data** is a **single, continuous segment of experimental data.** The ***test data*** consists of a folder containing many **small segments.** The data within each **test file is continuous, but the test files do not represent a continuous segment of the experiment**; thus, the **predictions cannot be assumed to follow the same regular pattern seen in the training file.**
# * For each `seg_id` in the test folder, you should predict a single `time_to_failure` corresponding to the time between the **last row of the segment and the next laboratory earthquake.**
# 
# <span style="color:Red;">**DATA DESCRIPTION**</span>
# ----
# ### <span style="color:blue;">**File descriptions**: </span>
# * **train.csv** - A single, continuous training segment of experimental data.
# * **test** - A folder containing many small segments of test data.
# * **sample_sumbission.csv** - A sample submission file in the correct format.
# 
# ### <span style="color:blue;">**Data fields**:</span>
# * **acoustic_data** - the seismic signal [int16]
# * **time_to_failure** - the time (in seconds) until the next laboratory earthquake [float64]
# * **seg_id** - the test segment ids for which predictions should be made (one prediction per segment)

# <strong><span style="color:Red;">*Article By Los Alamos National Laboratory*</span></strong>
# <strong><span style="color:black;">‘Fingerprint’ of fault displacement also forecasts magnitude of rupture</span></strong>
# -----------------------------------------------------------------------
# 
# LOS ALAMOS, N.M., Dec. 17, 2018—Machine-learning research published in two related papers today in _Nature Geosciences_ reports the detection of seismic signals accurately predicting the Cascadia fault’s slow slippage, a type of failure observed to precede large earthquakes in other subduction zones.
# 
# * **Los Alamos National Laboratory researchers applied machine learning to analyze Cascadia data and discovered the megathrust broadcasts a constant tremor, a fingerprint of the fault’s displacement.** More importantly, they **found a direct parallel between the loudness of the fault’s acoustic signal and its physical changes**. Cascadia’s groans, previously discounted as meaningless noise, foretold its fragility.
# * **“Cascadia’s behavior was buried in the data**. Until **machine learning revealed precise patterns**, we all discarded the **continuous signal as noise, but it was full of rich information.** We discovered a **highly predictable sound pattern that indicates slippage and fault failure**,” said Los Alamos scientist Paul Johnson. “We also found a precise link between the fragility of the fault and the signal’s strength, which can help us more accurately predict a megaquake.”  
# * The **new papers** were authored by ***Johnson, Bertrand Rouet-Leduc and Claudia Hulbert*** from the ***Laboratory’s Earth and Environmental Sciences Division, Christopher Ren from the Laboratory’s Intelligence and Space Research Division and collaborators at Pennsylvania State University.***
# * **Machine learning crunches massive seismic data sets to find distinct patterns by learning from self-adjusting algorithms to create decision trees that select and retest a series of questions and answers.** Last year, the team simulated an earthquake in a laboratory, using steel blocks interacting with rocks and pistons, and recorded sounds that they analyzed by machine learning. They discovered that the numerous seismic signals, previously discounted as meaningless noise, pinpointed when the simulated fault would slip, a major advance towards earthquake prediction. Faster, more powerful quakes had louder signals.
# * The team decided to apply their new paradigm to the real world: Cascadia. Recent research reveals that Cascadia has been active, but noted activity has been seemingly random. This team analyzed 12 years of real data from seismic stations in the region and found similar signals and results: Cascadia’s constant tremors quantify the displacement of the slowly slipping portion of the subduction zone. In the laboratory, the authors identified a similar signal that accurately predicted a broad range of fault failure. Careful monitoring in Cascadia may provide new information on the locked zone to provide an early warning system.
# 
# The papers:
# -------------------
# 
# *   [Similarity of fast and slow earthquakes illuminated by machine learning](https://www.nature.com/articles/s41561-018-0272-8 "Machine learning"), Nature Geoscience, Dec. 17, 2018
# *   [Continuous chatter of the Cascadia subduction zone revealed by machine learning,](https://www.nature.com/articles/s41561-018-0274-6) Nature Geoscience, Dec. 17, 2018
# 
# **REFERENCES:**  
# https://www.lanl.gov/discover/news-release-archive/2018/December/1217-machine-learning.php

# In[ ]:


get_ipython().run_cell_magic('html', '', '\n<h1> <strong><span style="color:blue;">Can We Predict Earthquakes?</span></strong></h1>\n<iframe width="800" height="400" src="https://www.youtube.com/embed/uUEzGcRJIZE" style="align:center;" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n')


# ## Outline
# ---
# * [**1.Read Data**](#1.Read-Data)
# * [**2.Simple Exploration**](#2.Simple-Exploration)
# * [**3.Feature Engineering**](#3.Feature-Engineering)
# * [**4.Sanity Check**](#4.Sanity-Check)
# * [**5.Data Transformation**](#5.Data-Transformation)
# * [**6.Model Training**](#6.Model-Training)
# 	* [**1.nuSVR**](#1.nuSVR)
# 	* [**2.SVR**](#2.SVR)
# 	* [**3.BayesianRidge**](#3.BayesianRidge)
# 	* [**4.LightGBM Regression**](#4.LightGBM-Regression)
# 	* [**5.CatBoost Regression**](#5.CatBoost-Regression)
# * [**7.Prediction**](#7.Prediction)
# * [**8.Stacking using lightgbm**](#8.Stacking-using-lightgbm)
# * [**9.Total Analysis of all model**](#9.Total-Analysis-of-all-model)
# * [**10.Best Model**](#10.Best-Model)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
print(os.listdir("../input"))
from mlxtend.regressor import stacking_regression
# pandas doesn't show us all the decimals
pd.options.display.precision = 15

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR,LinearSVR, SVR
from sklearn.metrics import mean_absolute_error,r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, BayesianRidge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score


# ## <span style="color:blue;"><strong>1.Read Data</strong></span>

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})\ngc.collect()\n")


# ## <span style="color:blue;"><strong>2.Simple Exploration</strong></span>

# In[ ]:


plt.figure(figsize=(20, 5))
plt.plot(train['acoustic_data'].values[::100], color='blue', label='Acoustic Data')
plt.legend()
plt.ylabel("Acoustic Data value")
plt.xlabel("Total Value Count")
plt.title('Acoustic Data')
plt.show()
plt.figure(figsize=(20, 5))
plt.plot(train['time_to_failure'].values[::100], color='red', label='Time_to_failure')
plt.legend()
plt.ylabel("Time Data value")
plt.xlabel("Time Value Count")
plt.title('Time to failure')
plt.show()


# In[ ]:


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[ ]:


gc.collect()


# ## <span style="color:blue;"><strong>3.Feature Engineering</strong></span>

# In[ ]:


# Create a training file with simple derived features
# Feature Engineering : https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction?scriptVersionId=9550007

rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01',
                                'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend'])
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'q95'] = np.quantile(x,0.95)
    X_train.loc[segment, 'q99'] = np.quantile(x,0.99)
    X_train.loc[segment, 'q05'] = np.quantile(x,0.05)
    X_train.loc[segment, 'q01'] = np.quantile(x,0.01)
    
    X_train.loc[segment, 'abs_max'] = np.abs(x).max()
    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X_train.loc[segment, 'abs_std'] = np.abs(x).std()
    X_train.loc[segment, 'trend'] = add_trend_feature(x)
    X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    
    
display(X_train.head())
gc.collect()


# # <span style="color:blue;"><strong>4.Sanity Check</strong></span>

# In[ ]:


get_ipython().run_cell_magic('time', '', "axs = pd.scatter_matrix(X_train[::100], figsize=(20,12), diagonal='kde')\ndisplay(X_train[::100].corr())\ngc.collect()\n")


# # <span style="color:blue;"><strong>5.Data Transformation</strong></span>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'scaler = StandardScaler()\nscaler.fit(X_train)\nX_train_scaled = scaler.transform(X_train)\ngc.collect()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "axs = pd.scatter_matrix(X_train[::100], figsize=(20,12), diagonal='kde')\ndisplay(X_train[::100].corr())\ngc.collect()\n")


# # <span style="color:blue;"><strong>6.Model Training</strong></span>

# ## 1.nuSVR

# In[ ]:


def nusvr_code(NuSVR,X_train_scaled,y_train):
    svm1 = NuSVR(nu=0.95, gamma=0.62,C=2.45)
    svm1.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm1.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('NuSVR')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score,svm1)
    
y_pred_nusvr, score, svm1 = nusvr_code(NuSVR,X_train_scaled,y_train)


# ## 2.SVR

# In[ ]:


def svr_code(SVR,X_train_scaled,y_train):
    svm3 = SVR(C=1000, verbose = 1)
    svm3.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm3.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('SVR')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm3)
    
y_pred_SVR, score, svm3 = svr_code(SVR,X_train_scaled,y_train)


# ## 3.Kernel Ridge

# In[ ]:


def br_code(BayesianRidge,X_train_scaled,y_train):
    svm2 = KernelRidge(kernel='rbf',alpha = 0.05, gamma = 0.06)
    svm2.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm2.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('Kernel Ridge Regression')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm2)
    
y_pred_Bayesian, score, svm2 = br_code(BayesianRidge,X_train_scaled,y_train)


# ## 4.LightGBM Regression

# In[ ]:


svm5 = LGBMRegressor(num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=1000)
svm5.fit(X_train_scaled, y_train.values.flatten())
y_pred_lgb = svm5.predict(X_train_scaled)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_lgb)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM')
score = rmse(y_train.values.flatten(), y_pred_lgb)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_lgb)
print(f'Score: {score:0.3f}')


# ## 5.CatBoost Regression

# In[ ]:


def cat_code(CatBoostRegressor,X_train_scaled,y_train):
    svm4 = CatBoostRegressor(depth=8)
    svm4.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm4.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('CATBOOST')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm4)
    
y_pred_cat, score, svm4 = cat_code(CatBoostRegressor,X_train_scaled,y_train)


# # <span style="color:blue;"><strong>7.Prediction</strong></span>

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)
    
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()
    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)
    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)    


# # <span style="color:blue;"><strong>8.Stacking using lightgbm</strong></span>

# In[ ]:


f = [y_pred_nusvr,y_pred_SVR,y_pred_cat,y_pred_lgb,y_pred_Bayesian]


# In[ ]:


f = np.transpose(f)


# In[ ]:


f.shape


# In[ ]:


svm6 = LGBMRegressor()
svm6.fit(f, y_train.values.flatten())
y_pred_stack = svm6.predict(f)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_lgb)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM Stacking')
score = rmse(y_train.values.flatten(), y_pred_stack)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_lgb)
print(f'Score: {score:0.3f}')


# # <span style="color:blue;"><strong>9.Total Analysis of all model</strong></span>
# 
# | Model | RMSE | F1SCORE |
# |--|--|--|
# |**nuSVR**|**2.647**| **0.48** |
# |**SVR**|**2.635**|**0.485**|
# |**Kernel Ridge**|**2.653**|**0.478**|
# |**Lightgbm**|**2.029**|**0.695**|
# |**Catboost**|**2.491**|**0.54**|
# |**Stacking**|**1.200**|**0.695**|

# In[ ]:


d = {'Model': ['nuSVR', 'SVR','Kernel Ridge','Lightgbm','Catboost','Stacking'], 'RMSE': [2.647,2.635,2.653,2.029,2.491,1.20],'F1_Score': [0.48,0.485,0.478,0.695,0.54,0.695]}
analysis_df = pd.DataFrame(d)
# display(analysis_df)

analysis_df.index = analysis_df.Model
del analysis_df['Model']
display(analysis_df)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(analysis_df.index,analysis_df.RMSE,'mD-',animated=True)
plt.scatter(analysis_df.index, analysis_df.RMSE,s=y*50, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title("RMSE by Model")
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(analysis_df.index,analysis_df.F1_Score,'rD-',animated=True)
plt.scatter(analysis_df.index, analysis_df.F1_Score,s=y*50, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
plt.xlabel('Model')
plt.ylabel('F1_Score')
plt.title("F1_Score by Model")
plt.show()


# # <span style="color:blue;"><strong>8.Blending</strong></span>

# In[ ]:


svm5


# In[ ]:


from mlxtend.regressor import StackingRegressor
sclf = StackingRegressor(regressors=[svm1,svm2,svm3,svm4,svm5,svm6], 
                          meta_regressor=SVR())

sclf.fit(X_train_scaled, y_train.values.flatten())


# In[ ]:


y_pred_final = sclf.predict(X_train_scaled)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_final)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM')
score = rmse(y_train.values.flatten(), y_pred_final)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_final)
print(f'Score: {score:0.3f}')


# In[ ]:


X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = sclf.predict(X_test_scaled)
# submission['time_to_failure1'] = svm1.predict(X_test_scaled)
# submission['time_to_failure2'] = svm2.predict(X_test_scaled)
# submission['time_to_failure3'] = svm3.predict(X_test_scaled)
# submission['time_to_failure4'] = svm4.predict(X_test_scaled)
# submission['time_to_failure5'] = svm5.predict(X_test_scaled)
# submission['time_to_failure'] = (submission['time_to_failure1']+submission['time_to_failure2']+submission['time_to_failure3']+submission['time_to_failure4']+submission['time_to_failure5'])/5

# del submission['time_to_failure1'],submission['time_to_failure2'],submission['time_to_failure3'],submission['time_to_failure4'],submission['time_to_failure5']


# In[ ]:


submission.to_csv("Advance_stack.csv")


# # <span style="color:blue;"><strong>10.Best Model</strong></span>

# In[ ]:


submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm5.predict(X_test_scaled)
submission1.to_csv("submission_lgbbestmodel.csv")


# In[ ]:


submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm3.predict(X_test_scaled)
submission1.to_csv("submission_svrbestmodel.csv")


# In[ ]:


submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm1.predict(X_test_scaled)
submission1.to_csv("submission_nusvrbestmodel.csv")

