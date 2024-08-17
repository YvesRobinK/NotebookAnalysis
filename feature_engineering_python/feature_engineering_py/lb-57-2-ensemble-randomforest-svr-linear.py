#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # 1. Define Metric SMAPE

# In[2]:


def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)


# # 2. Load Train and Sample Test Data

# ## 2.1 Protein data

# In[3]:


proteins = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv')
print('Proteins shape:',proteins.shape)
proteins.head()


# In[4]:


proteins_test = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/example_test_files/test_proteins.csv')
print('Proteins test shape:',proteins_test.shape)
proteins_test.head()


# ## 2.2 Peptides data

# In[5]:


peptides = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv')
print('Peptides shape:', peptides.shape)
peptides.head()


# In[6]:


peptides_test = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/example_test_files/test_peptides.csv')
print('Peptides test shape:', peptides_test.shape)
peptides_test.head()


# ## 2.3 Clinical data

# In[7]:


clinical = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
print('Clinical shape:', clinical.shape)
clinical.head()


# In[8]:


test = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/example_test_files/test.csv')
print('Test shape:', test.shape)
test.head()


# # 3. Some EDA

# In[9]:


proteins.groupby('visit_id').agg({'UniProt':'nunique','patient_id':'count','NPX':['min','max','mean','std']}).reset_index()


# In[10]:


peptides.groupby('visit_id').agg({'UniProt':'nunique','patient_id':'count','Peptide':'nunique','PeptideAbundance': ['min','max','mean','std']}).reset_index()


# # 4. Make dataset for training

# ## 4.1 Training only first month (0's visit_month)

# In[11]:


df_0 = clinical[(clinical.visit_month == 0)][['visit_id','updrs_1']]
print('Train shape:', df_0.shape)
df_0.head()


# ## 4.2 Feature Engineering

# ### 4.2.1 Proteins features

# In[12]:


proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std'))\
                .reset_index()
proteins_npx_ft.head()


# In[13]:


df_proteins = pd.merge(proteins, df_0, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()
proteins_Uniprot_updrs.head()


# In[14]:


df_proteins = pd.merge(proteins, proteins_Uniprot_updrs, on = 'UniProt', how = 'left')
proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),\
                                                          proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std'))\
                .reset_index()
proteins_UniProt_ft.head()


# ### 4.2.2 Peptides features

# In[15]:


peptides.head()


# In[16]:


peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),\
                                                                Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std'))\
                .reset_index()
peptides_PeptideAbundance_ft.head()


# In[17]:


df_peptides = pd.merge(peptides, df_0, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()
peptides_PeptideAbundance_updrs.head()


# In[18]:


df_peptides = pd.merge(peptides, peptides_PeptideAbundance_updrs, on = 'Peptide', how = 'left')
peptides_ft = df_peptides.groupby('visit_id').agg(peptides_updrs_1_min=('updrs_1_sum','min'), peptides_updrs_1_max=('updrs_1_sum','max'),\
                                                          peptides_updrs_1_mean=('updrs_1_sum','mean'), peptides_updrs_1_std=('updrs_1_sum','std'))\
                .reset_index()
peptides_ft


# ### 4.2.3 Put it all together

# In[19]:


df_0_1 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_1']]
df_0_2 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_2']]
df_0_3 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_3']]
df_0_4 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_4']]

df_proteins = pd.merge(proteins, df_0_1, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs1 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_2, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs2 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_3, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs3 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_4, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs4 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_1, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs1 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_2, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs2 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_3, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs3 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_4, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs4 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_proteins_fts = [proteins_Uniprot_updrs1, proteins_Uniprot_updrs2, proteins_Uniprot_updrs3, proteins_Uniprot_updrs4]
df_peptides_fts = [peptides_PeptideAbundance_updrs1, peptides_PeptideAbundance_updrs2, peptides_PeptideAbundance_updrs3, peptides_PeptideAbundance_updrs4]
df_lst = [df_0_1, df_0_2, df_0_3, df_0_4]


# In[20]:


def features(df, proteins, peptides, classes):
    proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std'))\
                    .reset_index()
    peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),\
                                                                    Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std'))\
                    .reset_index()

    df_proteins = pd.merge(proteins, df_proteins_fts[classes], on = 'UniProt', how = 'left')
    proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),\
                                                              proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std'))\
                    .reset_index()
    df_peptides = pd.merge(peptides, df_peptides_fts[classes], on = 'Peptide', how = 'left')
    peptides_ft = df_peptides.groupby('visit_id').agg(peptides_updrs_1_min=('updrs_1_sum','min'), peptides_updrs_1_max=('updrs_1_sum','max'),\
                                                              peptides_updrs_1_mean=('updrs_1_sum','mean'), peptides_updrs_1_std=('updrs_1_sum','std'))\
                    .reset_index()

    df = pd.merge(df, proteins_npx_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_PeptideAbundance_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, proteins_UniProt_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_ft, on = 'visit_id', how = 'left')
    df = df.fillna(df.mean())
    return df


# In[21]:


train_0 = features(df_0_1, proteins, peptides, 0)
train_0


# # 5. Training

# In[22]:


import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  

import warnings

warnings.filterwarnings('ignore')


# In[23]:


model = {}
mms = MinMaxScaler()
n_estimators = list(range(5,100)) # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = list(range(2,11)) # minimum sample number to split a node
min_samples_leaf = list(range(1,10)) # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points


for i in range(3):
    print('--------------------------------------------------------')
    print('Model {0}'.format(i + 1))
    train_0 = features(df_lst[i], proteins, peptides, i)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    train_0[scale_col] = mms.fit_transform(train_0[scale_col])
    
    rfc = RandomForestRegressor()
    sv = SVR()
    ln = Lasso()
    dt = DecisionTreeRegressor()
    
    svc_params = [{'kernel':['poly','rbf','sigmoid','linear']}]
    forest_params = [{'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}]
    linear_params = [{'alpha':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]}]
    tree_params = [{

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf}]
    
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = RandomizedSearchCV(rfc, forest_params, cv = cv, scoring=make_scorer(smape), verbose = -1)
    svc = RandomizedSearchCV(sv, svc_params, cv = cv, scoring=make_scorer(smape), verbose = -1)
    linear = RandomizedSearchCV(ln, linear_params, cv = cv, scoring=make_scorer(smape), verbose = -1)
    dec = RandomizedSearchCV(dt, tree_params, cv = cv, scoring=make_scorer(smape), verbose = -1)
    
    X = train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1)
    y = train_0['updrs_{0}'.format(i + 1)].astype(np.float32)
    clf.fit(X, y)
    svc.fit(X,y)
    linear.fit(X,y)
    dec.fit(X,y)

    print('rfc best params:',clf.best_params_)

    print('rfc best score',clf.best_score_)
    
    print('svc best param',svc.best_params_)

    print('svc best score',svc.best_score_)
    
    print('linear best param',linear.best_params_)

    print('linear best score',linear.best_score_)
    
    print('decision tree best param',dec.best_params_)

    print('decision tree best score',dec.best_score_)
    
    print('Train smape rfc:',smape(train_0['updrs_{0}'.format(i + 1)], clf.predict(train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1))))
    print('Train smape svc:',smape(train_0['updrs_{0}'.format(i + 1)], svc.predict(train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1))))
    print('Train smape linear:',smape(train_0['updrs_{0}'.format(i + 1)], linear.predict(train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1))))
    print('Train smape decision tree:',smape(train_0['updrs_{0}'.format(i + 1)], dec.predict(train_0.drop(columns = ['visit_id','updrs_{0}'.format(i + 1)], axis = 1))))
    
    
    model['rfc_' + str(i)] = clf
    model['svc_' + str(i)] = svc
    model['linear_' + str(i)] = linear
    model['dec_' + str(i)] = dec


# # 6. Inference

# In[24]:


import amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()


# In[25]:


def map_test(x):
    updrs = x.split('_')[2] + '_' + x.split('_')[3]
    month = int(x.split('_plus_')[1].split('_')[0])
    visit_id = x.split('_')[0] + '_' + x.split('_')[1]
    # set all predictions 0 where updrs equals 'updrs_4'
    if updrs=='updrs_3':
#         rating = updrs_3_pred[month]
        rating = df[df.visit_id == visit_id]['pred2'].values[0]
    elif updrs=='updrs_4':
        rating = 0
    elif updrs =='updrs_1':
        rating = df[df.visit_id == visit_id]['pred0'].values[0]
    else:
        rating = df[df.visit_id == visit_id]['pred1'].values[0]
    return rating

counter = 0
# The API will deliver four dataframes in this specific order:
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    df = test[['visit_id']].drop_duplicates('visit_id')
    pred_0 = features(df[['visit_id']], test_proteins, test_peptides, 0)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_0[scale_col] = mms.fit_transform(pred_0[scale_col])
    pred_0 = (model['rfc_0'].predict(pred_0.drop(columns = ['visit_id'], axis = 1)) \
              + model['svc_0'].predict(pred_0.drop(columns = ['visit_id'], axis = 1))\
                 + model['linear_0'].predict(pred_0.drop(columns = ['visit_id'], axis = 1))\
                     + model['dec_0'].predict(pred_0.drop(columns = ['visit_id'], axis = 1)))/4
    df['pred0'] = np.ceil(pred_0 + 0)
    
    pred_1 = features(df[['visit_id']], test_proteins, test_peptides, 1)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_1[scale_col] = mms.fit_transform(pred_1[scale_col])
    pred_1 = (model['rfc_1'].predict(pred_1.drop(columns = ['visit_id'], axis = 1)) \
              + model['svc_1'].predict(pred_1.drop(columns = ['visit_id'], axis = 1))\
                 + model['linear_1'].predict(pred_1.drop(columns = ['visit_id'], axis = 1))\
                     + model['dec_1'].predict(pred_1.drop(columns = ['visit_id'], axis = 1)))/4
    df['pred1'] = np.ceil(pred_1 + 0.5)
    
    pred_2 = features(df[['visit_id']], test_proteins, test_peptides, 2)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_2[scale_col] = mms.fit_transform(pred_2[scale_col])
    pred_2 = (model['rfc_2'].predict(pred_2.drop(columns = ['visit_id'], axis = 1)) \
                + model['svc_2'].predict(pred_2.drop(columns = ['visit_id'], axis = 1))\
                     + model['linear_2'].predict(pred_2.drop(columns = ['visit_id'], axis = 1))\
                         + model['dec_2'].predict(pred_2.drop(columns = ['visit_id'], axis = 1)))/4
    df['pred2'] = np.ceil(pred_2 + 1.5)
    
    sample_submission['rating'] = sample_submission['prediction_id'].apply(map_test)
    env.predict(sample_submission)
    
    if counter == 0:
        display(test)
        display(sample_submission)
        
    counter += 1


# In[ ]:




