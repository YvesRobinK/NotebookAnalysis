#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Notebooks used:**
# 
# The submission.csv files from these notebooks are stored in a public [Dataset](https://www.kaggle.com/datasets/jbomitchell/spaceship-titanic-models)
# 
# [[Pycaret] Visualization + Optimization (0.81)](https://www.kaggle.com/arootda/pycaret-visualization-optimization-0-81) by @arootda (v10 0.80967)
# 
# [Spaceship Titanic: autogluon](https://www.kaggle.com/smakoto/spaceship-titanic-autogluon) by @smakoto (v2 0.80921)
# 
# [Spaceship Titanic : FLAML : ap](https://www.kaggle.com/gauravduttakiit/spaceship-titanic-flaml-ap) by @gauravduttakiit (v1 0.80921)
# 
# [[Pycaret] Spaceship - FE + Catboost](https://www.kaggle.com/edwintyh/pycaret-spaceship-fe-catboost) by @edwintyh (v8 0.80804)
# 
# [Spaceship Titanic: EDA + FE + CatBoost](https://www.kaggle.com/dmitrylessy/spaceship-titanic-eda-fe-catboost) by @dmitrylessy (v3 0.80710) 
# 
# [Spaceship Titanic](https://www.kaggle.com/jasoninzana/spaceship-titanic) by @jasoninzana (v11 0.80687)
# 
# [Feature Engineering, EDA and LightGBM](https://www.kaggle.com/taranmarley/feature-engineering-eda-and-lightgbm) by @taranmarley (v16 0.80640)
# 
# [Spaceship Titanic, Pseudo Labels + Blending](https://www.kaggle.com/hasanbasriakcay/spaceship-titanic-pseudo-labels-blending) by @hasanbasriakcay (v55 0.80617)
# 
# [Blasted Wormholes! Detailed EDA and models](https://www.kaggle.com/max1mum/blasted-wormholes-detailed-eda-and-models) by @max1mum (v24 0.80593)
# 
# [Spaceship_Titanic_Starter](https://www.kaggle.com/drcapa/spaceship-titanic-starter) by @drcapa (v9 0.80570)
# 
# [My Spaceship Titanic 80.5%](https://www.kaggle.com/adamml/my-spaceship-titanic-80-5) by @adamml (v4 0.80500)
# 
# [Detailed Classification Notebook | Spaceship](https://www.kaggle.com/dylanyves/detailed-classification-notebook-spaceship) by @dylanyves (v9 0.78419; plus a kNN model based on this 0.76689)
# 
# [Imputing Missing Values by Group + Custom Thresh](https://www.kaggle.com/code/rizkykiky/imputing-missing-values-by-group-custom-thresh) by @rizkykiky (v4 0.80991)
# 
# [SPACESHIP PREDICT VOTING CLASSIFIER](https://www.kaggle.com/code/flaviocavalcante/spaceship-predict-voting-classifier) by @flaviocavalcante (v5 0.80991)
# 
# [Top 10 notebook](https://www.kaggle.com/code/opamusora/top-10-notebook) by @opamusora (v5 0.81131)
# 
# [LB Probing!?!1?11!1?!](https://www.kaggle.com/code/rizkykiky/lb-probing-1-11-1) by @rizkykiky (v1 0.80967)

# In[2]:


sub = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sub.sort_values(by=['PassengerId'], inplace=True)
sub['Transported'] = sub['Transported'].astype('float')
sub.head(10)


# In[3]:


sub1 = pd.read_csv('../input/spaceship-titanic-models/80967_submission_v10.csv')
sub1.sort_values(by=['PassengerId'], inplace=True)
sub1['Transported'] = sub1['Transported'].astype('float')
sub2 = pd.read_csv('../input/spaceship-titanic-models/80921_submission_v1.csv')
sub2.sort_values(by=['PassengerId'], inplace=True)
sub2['Transported'] = sub2['Transported'].astype('float')
sub3 = pd.read_csv('../input/spaceship-titanic-models/80921a_submission_smakoto_v2.csv')
sub3.sort_values(by=['PassengerId'], inplace=True)
sub3['Transported'] = sub3['Transported'].astype('float')
sub4 = pd.read_csv('../input/spaceship-titanic-models/80804_submission_v8.csv')
sub4.sort_values(by=['PassengerId'], inplace=True)
sub4['Transported'] = sub4['Transported'].astype('float')
sub5 = pd.read_csv('../input/spaceship-titanic-models/80710_submission_v3.csv')
sub5.sort_values(by=['PassengerId'], inplace=True)
sub5['Transported'] = sub5['Transported'].astype('float')
sub6 = pd.read_csv('../input/spaceship-titanic-models/80687_submission_v11.csv')
sub6.sort_values(by=['PassengerId'], inplace=True)
sub6['Transported'] = sub6['Transported'].astype('float')
sub7 = pd.read_csv('../input/spaceship-titanic-models/80640_submission_v16.csv')
sub7.sort_values(by=['PassengerId'], inplace=True)
sub7['Transported'] = sub7['Transported'].astype('float')
sub8 = pd.read_csv('../input/spaceship-titanic-models/80617_submission_stack_v55.csv')
sub8.sort_values(by=['PassengerId'], inplace=True)
sub8['Transported'] = sub8['Transported'].astype('float')
sub9 = pd.read_csv('../input/spaceship-titanic-models/80593_submission_v24.csv')
sub9.sort_values(by=['PassengerId'], inplace=True)
sub9['Transported'] = sub9['Transported'].astype('float')
sub10 = pd.read_csv('../input/spaceship-titanic-models/80570_submission_v9.csv')
sub10.sort_values(by=['PassengerId'], inplace=True)
sub10['Transported'] = sub10['Transported'].astype('float')
sub11 = pd.read_csv('../input/spaceship-titanic-models/80500_submission_v4.csv')
sub11.sort_values(by=['PassengerId'], inplace=True)
sub11['Transported'] = sub11['Transported'].astype('float')
sub12 = pd.read_csv('../input/spaceship-titanic-models/78419_submission_v9.csv')
sub12.sort_values(by=['PassengerId'], inplace=True)
sub12['Transported'] = sub12['Transported'].astype('float')
sub13 = pd.read_csv('../input/spaceship-titanic-models/76689_submission_knn_v3.csv')
sub13.sort_values(by=['PassengerId'], inplace=True)
sub13['Transported'] = sub13['Transported'].astype('float')
sub14 = pd.read_csv('../input/spaceship-titanic-models/80991_submission_v4.csv')
sub14.sort_values(by=['PassengerId'], inplace=True)
sub14['Transported'] = sub14['Transported'].astype('float')
sub15 = pd.read_csv('../input/spaceship-titanic-models/80991a_submission_v5.csv')
sub15.sort_values(by=['PassengerId'], inplace=True)
sub15['Transported'] = sub15['Transported'].astype('float')
sub16 = pd.read_csv('../input/spaceship-titanic-models/81131a_submission_v5.csv')
sub16.sort_values(by=['PassengerId'], inplace=True)
sub16['Transported'] = sub16['Transported'].astype('float')
sub17 = pd.read_csv('../input/spaceship-titanic-models/80967a_submission_v1.csv')
sub17.sort_values(by=['PassengerId'], inplace=True)
sub17['Transported'] = sub17['Transported'].astype('float')
sub18 = pd.read_csv('../input/notebooke9a60f1c3c/submission.csv')
sub18.sort_values(by=['PassengerId'], inplace=True)
sub18['Transported'] = sub18['Transported'].astype('float')
sub19 = pd.read_csv('../input/spaceship-predictions/my_submission_051922.csv')
sub19.sort_values(by=['PassengerId'], inplace=True)
sub19['Transported'] = sub19['Transported'].astype('float')


# This model is an ensemble consisting of a weighted average of models from public notebooks.
# 
# Each model represented in the ensemble is weighted by a factor that depends in an exponentially decaying manner on its public LB score. Specifically, each model is given an exponential weight according to
# 
# exp(b*(x-S))
# 
# where x is the LB score of that model. Larger scores are better, hence the weights get larger as x increases and get smaller as x decreases. Thus, the best models have the largest weights and make the largest contributions to the ensemble.
# 
# The parameter b is the one meaningfully adjustable parameter of the model, the larger b is then the faster the weights decay as the score gets worse. S is a calibration parameter defined such that if S is set to the best single model score then the highest unnormalised weight exp(b*(x-S)) is 1.0, which is convenient, but not essential.
# 
# The sum of the unnormalised weights is called q. Once all weights have been calculated, these weights are normalised by dividing them all by q.

# In[4]:


b = 3110.0
S = 0.81482
q = 0.0


# In[5]:


sub['Transported'] = sub1['Transported']*np.exp(b*(0.80967-S))
q = q + np.exp(b*(0.80967-S))
sub['Transported'] = sub['Transported'] + sub2['Transported']*np.exp(b*(0.80921-S))
q = q + np.exp(b*(0.80921-S))
sub['Transported'] = sub['Transported'] + sub3['Transported']*np.exp(b*(0.80921-S))
q = q + np.exp(b*(0.80921-S))
sub['Transported'] = sub['Transported'] + sub4['Transported']*np.exp(b*(0.80804-S))
q = q + np.exp(b*(0.80804-S))
sub['Transported'] = sub['Transported'] + sub5['Transported']*np.exp(b*(0.80710-S))
q = q + np.exp(b*(0.80710-S))
sub['Transported'] = sub['Transported'] + sub6['Transported']*np.exp(b*(0.80687-S))
q = q + np.exp(b*(0.80687-S))
sub['Transported'] = sub['Transported'] + sub7['Transported']*np.exp(b*(0.80640-S))
q = q + np.exp(b*(0.80640-S))
sub['Transported'] = sub['Transported'] + sub8['Transported']*np.exp(b*(0.80617-S))
q = q + np.exp(b*(0.80617-S))
sub['Transported'] = sub['Transported'] + sub9['Transported']*np.exp(b*(0.80593-S))
q = q + np.exp(b*(0.80593-S))
sub['Transported'] = sub['Transported'] + sub10['Transported']*np.exp(b*(0.80570-S))
q = q + np.exp(b*(0.80570-S))
sub['Transported'] = sub['Transported'] + sub11['Transported']*np.exp(b*(0.80500-S))
q = q + np.exp(b*(0.80500-S))
sub['Transported'] = sub['Transported'] + sub12['Transported']*np.exp(b*(0.78419-S))
q = q + np.exp(b*(0.78419-S))
sub['Transported'] = sub['Transported'] + sub13['Transported']*np.exp(b*(0.76689-S))
q = q + np.exp(b*(0.76689-S))
sub['Transported'] = sub['Transported'] + sub14['Transported']*np.exp(b*(0.80991-S))
q = q + np.exp(b*(0.80991-S))
sub['Transported'] = sub['Transported'] + sub15['Transported']*np.exp(b*(0.80991-S))
q = q + np.exp(b*(0.80991-S))
sub['Transported'] = sub['Transported'] + sub16['Transported']*np.exp(b*(0.81131-S))
q = q + np.exp(b*(0.81131-S))
sub['Transported'] = sub['Transported'] + sub17['Transported']*np.exp(b*(0.80967-S))
q = q + np.exp(b*(0.80967-S))
sub['Transported'] = sub['Transported'] + sub18['Transported']*np.exp(b*(0.81482-S))
q = q + np.exp(b*(0.81482-S))
sub['Transported'] = sub['Transported'] + sub19['Transported']*np.exp(b*(0.81178-S))
q = q + np.exp(b*(0.81178-S))
sub['Transported'] = sub['Transported']/q
print(q)
sub.head(10)


# So we now have a real number between 0.0 and 1.0 for each passenger, reflecting our estimate of the likelihood or probability of that passenger having been transported. For our prediction, we will round these to the nearest integer, either 0 or 1.

# In[6]:


sub['Transported'] = np.rint(sub['Transported'])
sub.head(10)


# Finally, we convert these to Boolean form and submit.

# In[7]:


sub['Transported'] = sub['Transported'].astype('bool')
sub.to_csv('submission.csv', index=False)
sub.head(10)

