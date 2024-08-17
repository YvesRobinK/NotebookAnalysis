#!/usr/bin/env python
# coding: utf-8

# <div>
#     <h1 align="center">AutoGluon & Missing Values</h1>    
#     <h1 align="center">Tabular Playground Series - Sep 2021</h1> 
#     <h4 align="center">By: Somayyeh Gholami & Mehran Kazeminia</h4>
# </div>

# <div class="alert alert-success">  
# </div>

# ## Description:
# 
# ### The capabilities of "AutoGluon" are enormous. With just a few lines of coding, you can get a good result. But is "Missing Values" handling successful in "AutoGluon"? Or is it better to do this step by ourselves.
# 
# ### In this notebook, "Model-1" was initially created using only "AutoGluon". But we then handled the "Missing Values" ourselves and then provided these results to "AutoGluon" to create the "Model-2".
# 
# ### The "Model-2" score at the same time was much better than the "Model-1" score. We checked this several times. However, if we did not make a mistake, we can conclude that handling "Missing Values" manually is still better.
# 
# ### Good Luck.
# 

# <div class="alert alert-success">
#     <h3 align="center">If you find this work useful, please don't forget upvoting :)</h3>
# </div>

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <div class="alert alert-success">  
# </div>

# ## Competition Evaluation
# 
# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
# 
# Thanks to: @ihelon

# In[4]:


def roc_auc(true_list, pred_list):
    
    fpr, tpr, _ = roc_curve(true_list, pred_list)    
    roc_auc = auc(fpr, tpr)

    print(f'FPR: {fpr}')
    print(f'TPR: {tpr}')
    print(f'{list(zip(fpr,tpr))}')
    print(f'\nROC_AUC: %0.2f\n' %roc_auc)
    
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6, 6), facecolor='lightgray')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'\nThe area under the ROC curve\n')
    plt.legend(loc="lower right")
    plt.show()
       


# In[5]:


true_list  = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

pred_list1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

pred_list2 = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8])

pred_list3 = np.array([0.8, 0.8, 0.8, 0.8, 0.5, 0.2, 0.2, 0.2, 0.2, 0.8])


# In[6]:


roc_auc(true_list , pred_list1)


# In[7]:


roc_auc(true_list , pred_list2)


# In[8]:


roc_auc(true_list , pred_list3)


# <div class="alert alert-success">  
# </div>

# ## Data Set of Challenge

# In[9]:


DF1 = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')

DF2 = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')

SAM = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')

display(DF1.shape, DF2.shape, SAM.shape)


# In[10]:


MV1 = DF1.isnull().sum()
MV2 = DF2.isnull().sum()

print(f'Missing Value DF1:\n{MV1[MV1 > 0]}\n')
print(f'Missing Value DF2:\n{MV2[MV2 > 0]}\n')


# In[11]:


display(DF1, DF2)

# display(DF1.describe().transpose())
# display(DF2.describe().transpose())


# In[12]:


print('=' * 40)
DF1.info(memory_usage='deep')
print('=' * 40)
DF2.info(memory_usage='deep')
print('=' * 40)


# In[13]:


columns = DF2.columns[1:]
display(columns)


# In[14]:


DF1['claim'].value_counts().plot(figsize=(4, 4), kind='bar')


# In[15]:


DF1['claim'].value_counts().plot(figsize=(6, 6), kind='pie')

DF1['claim'].value_counts(normalize=True)


# In[16]:


X = DF1.drop(columns = ['id','claim'])

XX = DF2.drop(columns = ['id'])

y = DF1.claim

#display(X, XX, y)
#display(y.min(), y.max())


# In[17]:


hist_data = [ y ]  

group_labels = ['y']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 

fig.show()


# ## Split

# In[18]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.50, random_state=123) 

val_X.to_csv("val_X.csv",index=False)
val_y.to_csv("val_y.csv",index=False)


# ## Scaling

# In[19]:


X_scaled = minmax_scaling(X, columns=X.columns)

XX_scaled = minmax_scaling(XX, columns=XX.columns)

#display(X_scaled, XX_scaled)


# <div class="alert alert-success">  
# </div>

# ## Model - 1 
# 
# ## AutoGluon
# 
# Thanks to: @antonellomartiello

# In[20]:


get_ipython().system('pip install autogluon')

from autogluon.tabular import TabularDataset, TabularPredictor


# In[21]:


#data1 = TabularDataset('/kaggle/input/tabular-playground-series-sep-2021/train.csv').drop('id', axis=1)

#data2 = TabularDataset('/kaggle/input/tabular-playground-series-sep-2021/test.csv').drop('id', axis=1)

#display(data1.shape,data2.shape)


# In[22]:


#model1 = TabularPredictor(label= 'claim',
#                          eval_metric= 'roc_auc',
#                          verbosity= 3)

#model1.fit(train_data= data1,
#           time_limit= 3* 3600,
#           presets='best_quality',
#           verbosity= 3)

#model1.leaderboard(data1, silent=True)


# In[23]:


#results = model1.fit_summary()


# In[24]:


#pred1 = model1.predict_proba(data2)
#display(pred1)


# In[25]:


#sub1 = SAM.copy()

#sub1.iloc[:, 1] = pred1[1]
#display(sub1)


# In[26]:


#hist_data = [sub1.claim]  

#group_labels = ['AutoGluon - 1']
    
#fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 

#fig.show()


# In[27]:


#sub1.to_csv("submission_AutoGluon1.csv",index=False)
#Public Score: 


# <div class="alert alert-success">  
# </div>

# ## Missing Values 
# 
# ## Feature Engineering
# 
# Thanks to: @mlanhenke

# In[28]:


df1 = DF1.drop(columns = ['id','claim'])

df2 = DF2.drop(columns = ['id'])

display(df1.shape,df2.shape)


# In[29]:


df1['mvl_row'] = df1.isna().sum(axis=1)
df1['min_row'] = df1.min(axis=1)
df1['std_row'] = df1.std(axis=1)

pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])

df1 = pd.DataFrame(columns=df1.columns, data=pipeline.fit_transform(df1))
df1['claim'] = DF1['claim']
display(df1)


# In[30]:


df2['mvl_row'] = df2.isna().sum(axis=1)
df2['min_row'] = df2.min(axis=1)
df2['std_row'] = df2.std(axis=1)

pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])

df2 = pd.DataFrame(columns=df2.columns, data=pipeline.fit_transform(df2))
display(df2)


# <div class="alert alert-success">  
# </div>

# ## Model - 2 
# 
# ## AutoGluon & Feature Engineering

# In[31]:


model2 = TabularPredictor(label= 'claim',
                          eval_metric= 'roc_auc',
                          verbosity= 3)

model2.fit(train_data= df1,
           time_limit= 3* 3600,
           presets='best_quality',
           verbosity= 3)

model2.leaderboard(df1, silent=True)


# In[32]:


results = model2.fit_summary()


# In[33]:


pred2 = model2.predict_proba(df2)
display(pred2)


# In[34]:


sub2 = SAM.copy()

sub2.iloc[:, 1] = pred2[1]
display(sub2)


# In[35]:


hist_data = [sub2.claim]  

group_labels = ['AutoGluon - 2']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 

fig.show()


# In[36]:


sub2.to_csv("submission_AutoGluon2.csv",index=False)
# Public Score:


# <div class="alert alert-success">  
# </div>

# ## Ensembling

# In[37]:


def ensembling(main, support, coeff): 
    
    suba  = main.copy() 
    subav = suba.values
       
    subb  = support.copy()
    subbv = subb.values    
           
    ense  = main.copy()    
    ensev = ense.values  
 
    for i in range (len(main)):
        
        pera = subav[i, 1]
        perb = subbv[i, 1]
        per = (pera * coeff) + (perb * (1.0 - coeff))   
        ensev[i, 1] = per
        
    ense.iloc[:, 1] = ensev[:, 1]  
    
    ###############################    
    X  = suba.iloc[:, 1]
    Y1 = subb.iloc[:, 1]
    Y2 = ense.iloc[:, 1]
    
    plt.style.use('seaborn-whitegrid') 
    plt.figure(figsize=(9, 9), facecolor='lightgray')
    plt.title(f'\nE N S E M B L I N G\n')   
      
    plt.scatter(X, Y1, s=1.5, label='Support')    
    plt.scatter(X, Y2, s=1.5, label='Generated')
    plt.scatter(X, X , s=0.1, label='Main(X=Y)')
    
    plt.legend(fontsize=12, loc=2)
    #plt.savefig('Ensembling_1.png')
    plt.show()     
    ###############################   
    ense.iloc[:, 1] = ense.iloc[:, 1].astype(float)
    hist_data = [subb.iloc[:, 1], ense.iloc[:, 1], suba.iloc[:, 1]] 
    group_labels = ['Support', 'Ensembling', 'Main']
    
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False)
    fig.show()   
    ###############################       
    
    return ense      


# Thanks to: @mlanhenke https://www.kaggle.com/mlanhenke/tps-09-single-catboostclassifier

# In[38]:


path0 = '../input/tps9-81783/TPS9_81783.csv' 

sub81783 = pd.read_csv(path0)


# In[39]:


hist_data = [sub81783.claim]  

group_labels = ['Public Score: 0.81783']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 

fig.show()


# Thanks to: @maximkazantsev https://www.kaggle.com/maximkazantsev/tps-09-21-eda-lightgbm-with-folds

# In[40]:


path1 = '../input/tps9-81800/TPS9_81800.csv' 

sub81800 = pd.read_csv(path1)


# In[41]:


hist_data = [sub81800.claim]  

group_labels = ['Public Score: 0.81800']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 

fig.show()


# In[42]:


sub3 = ensembling(sub81783, sub2, 0.80)

sub4 = ensembling(sub81800, sub3, 0.70)


# <div class="alert alert-success">  
# </div>

# ## Submission

# In[43]:


sub3.to_csv("submission3.csv",index=False)
sub4.to_csv("submission_Final.csv",index=False)
get_ipython().system('ls')


# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>
