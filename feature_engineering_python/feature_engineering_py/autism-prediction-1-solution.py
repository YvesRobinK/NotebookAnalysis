#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import preprocessing    
le = preprocessing.LabelEncoder()

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from imblearn.over_sampling import RandomOverSampler

pd.set_option('max_columns',None)


# # Tried almost everything:
# 
# * LAMA
# * LR
# * EXTRA TREE
# * Ridge Classifier
# 
# > Features used: 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
#        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity'
# 
# > Result is same with or without ethnicity
# 
# > Even simple LR also gives the same result
# 

# In[2]:


train_df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/train.csv')
test_df  = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/test.csv')


# In[3]:


train_df


# In[4]:


train_df["ethnicity"][train_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish"})] = "Others"

test_df["ethnicity"][test_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish"})] = "Others"


# In[5]:


train_df['austim'] = le.fit_transform(train_df['austim'])
test_df['austim'] = le.fit_transform(test_df['austim'])


train_df['ethnicity'] = le.fit_transform(train_df['ethnicity'])
test_df['ethnicity'] = le.fit_transform(test_df['ethnicity'])


# In[6]:


X = train_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
       'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
y = train_df['Class/ASD']
X_test = test_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
       'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
X_test


# # Feature Engineering

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18,8))

df = train_df.corr()

mask = np.triu(np.ones_like(df))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df, annot=True, cbar=False, cmap="Blues",mask=mask)
plt.show()


# # Extra Tree Classifier

# In[8]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'n_estimators': [50, 150, 200, 250, 300, 500, 1000],
        'max_depth': [2, 4, 6, 8, 10]}

model_xt = ExtraTreesClassifier(

    random_state=0,
)

grid_model = GridSearchCV(model_xt,param_grid,cv=kf)

grid_model.fit(X, y)


# In[9]:


preds_xt = grid_model.predict_proba(X_test)


# In[10]:


preds_xt[:,1]


# # Logistic Regression

# In[11]:


from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

param_grid={"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}

model_lr = LogisticRegression(solver='saga', 
                              tol=1e-5, max_iter=10000,
                              random_state=0,
#                               C=0.22685190926977272,
#                               penalty='l2',
                             )

grid_model_lr = GridSearchCV(model_lr,param_grid,cv=kf)

grid_model_lr.fit(X, y)


# In[12]:


preds_lr = grid_model_lr.predict_proba(X_test)


# # Ensembling LR and XTRA

# In[13]:


sub = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/sample_submission.csv')


# In[14]:


new_preds = preds_lr[:,1]*0.58 + preds_xt[:,1]*0.42 


# In[15]:


submission_rc = pd.DataFrame({'ID':sub.ID,'Class/ASD':new_preds})
submission_rc.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




