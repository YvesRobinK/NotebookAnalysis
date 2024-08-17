#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
train.head()


# In[2]:


test = pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')
test.head()


# In[3]:


train.info()


# In[4]:


train.nunique()


# In[5]:


train=train.drop(['id'],axis=1)
train.head()


# In[6]:


test=test.drop(['id'],axis=1)
test.head()


# In[7]:


get_ipython().system('pip install openfe')


# In[8]:


X= train.drop(columns=['Hardness'])
y = train['Hardness']


# In[9]:


from openfe import OpenFE, transform
ofe = OpenFE()
features = ofe.fit(data=X, label=y)


# In[10]:


X, test = transform(X, test, features,n_jobs=4)
X.shape,test.shape


# In[11]:


X['Hardness']=y
train=X.copy()
train.head()


# In[12]:


train.reset_index(inplace=True,drop=True)
train.head()


# In[13]:


test.reset_index(inplace=True,drop=True)
test.head()


# In[14]:


round(train.isnull().sum()*100/len(train),2).sort_values(ascending=False)


# In[15]:


get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')


# In[16]:


import sys
sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer


# In[17]:


col=train.columns.tolist()
col.remove('Hardness')
col[:5]


# In[18]:


get_ipython().run_cell_magic('time', '', 'lgbm_imtr = LGBMImputer(n_iter=500)\n\ntrain_iterimp = lgbm_imtr.fit_transform(train[col])\ntest_iterimp = lgbm_imtr.transform(test[col])\n\n# Create train test imputed dataframe\ntrain_ = pd.DataFrame(train_iterimp, columns=col)\ntest = pd.DataFrame(test_iterimp, columns=col)\n')


# In[19]:


train_['Hardness'] = train['Hardness']
train=train_.copy()
train.head()


# In[20]:


traino=train.copy()
traino.head()


# In[21]:


get_ipython().system('pip install featurewiz')
get_ipython().system('pip install Pillow==9.0.0')
get_ipython().system('pip install xlrd — ignore-installed — no-deps')


# In[22]:


import featurewiz as gwiz
wiz =gwiz.FeatureWiz(verbose=2)


# In[23]:


y = train.pop('Hardness')
X = train


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42,shuffle=True)


# In[25]:


X_train = wiz.fit_transform(X_train, y_train)
X_train.head()


# In[26]:


X_test= wiz.transform(X_test)
X_test.head()


# In[27]:


wiz.features[:5]


# In[28]:


test=test[wiz.features]
test.head()


# In[29]:


col=wiz.features
col.append('Hardness')
col[-5:]


# In[30]:


train=traino[col]
train.head()


# In[31]:


sol = pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')
sol.head()


# In[32]:


train.to_csv('./train.csv',index=False)
test.to_csv('./test.csv',index=False)
sol.to_csv('./sol.csv',index=False)

