#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import category_encoders as ce
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, metrics


# In[2]:


# reading the dataset from raw csv file
get_ipython().system('pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl')
import datatable as dt
train = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas() #, max_nrows=2000555


# In[3]:


#train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',nrows=10**6)#, nrows=10**


# In[4]:


get_ipython().run_cell_magic('time', '', "data_types_dict = {\n    'row_id': 'int64',\n    'timestamp': 'int64',\n    'user_id': 'int32',\n    'content_id': 'int16',\n    'content_type_id': 'int8',\n    'answered_correctly': 'int8',\n    'prior_question_elapsed_time': 'float32',\n    'prior_question_had_explanation': 'boolean'\n}\ntrain = train.astype(data_types_dict)\ntrain['prior_question_had_explanation'].fillna(False, inplace=True)\n")


# In[5]:


gc.collect()


# In[6]:


train= train.drop(columns=['row_id','task_container_id','user_answer'])


# In[7]:


gc.collect()


# In[8]:


get_ipython().run_cell_magic('time', '', 'train = train[train.content_type_id == 0]\ntrain[\'attempt\'] = train.assign(dif = ((train.content_id.diff() != 0) | (train.user_id.diff() != 0))).groupby(["user_id","content_id"]).dif.cumsum()\ntrain[\'attempt\']=train[\'attempt\'] - 1\ntrain[\'attempt\'] = train[\'attempt\'].apply(lambda x: 4 if x >= 4 else x)\n#train = train[(train[\'content_id\']==405) & (train[\'user_id\']==1108148)]\ntrain= train.drop(columns=[\'content_type_id\'])\n')


# In[9]:


gc.collect()


# In[10]:


#1- UUUUUUUUUUUUUUUUUUUUUUUUUUUUU
grouped_by_df = train[['user_id','answered_correctly']].groupby('user_id')
train_user = grouped_by_df.agg({'answered_correctly': ['mean', 'sum','count']}).copy()
train_user.columns=['user_mean_accuracy','user_count_correct','user_count']


# In[11]:


#QQQQQQQQQQQQQQQQ
grouped_by_df = train[['content_id','answered_correctly']].groupby('content_id')
train_Q = grouped_by_df.agg({'answered_correctly': ['mean']}).copy()
train_Q.columns=['Q_mean_accuracy']


# In[12]:


#Questions
questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')


# In[13]:


tag = questions["tags"].str.split(" ", expand = True)
tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']

questions =  pd.concat([questions,tag],axis=1)
questions['tags1'] = pd.to_numeric(questions['tags1'], errors='coerce')
questions['tags2'] = pd.to_numeric(questions['tags2'], errors='coerce')
questions['tags3'] = pd.to_numeric(questions['tags3'], errors='coerce')
questions['tags4'] = pd.to_numeric(questions['tags4'], errors='coerce')
questions['tags5'] = pd.to_numeric(questions['tags5'], errors='coerce')
questions['tags6'] = pd.to_numeric(questions['tags6'], errors='coerce')



questions['tags'] = questions['tags'].astype(str)
cat_features = ['tags']
encoder = LabelEncoder()
label_encoder = preprocessing.LabelEncoder()
for feature in cat_features:
    encoded = label_encoder.fit_transform(questions[cat_features])
    questions[feature + '_labels'] = encoded
    

questions= questions.drop(columns=['tags1','tags2','tags3','tags4','tags5','tags6','bundle_id','correct_answer'])
questions.fillna(0, inplace=True)
questions


# In[14]:


train_Q=train_Q.merge(
    questions,
    how='left',
    left_on='content_id',
    right_on='question_id')
train_Q.info()


# In[15]:


questions = None
del(questions)
gc.collect()


# In[16]:


train_Q


# In[17]:


def convertBoolean(x):
    if str(x) == "False":
        return 0
    elif str(x) == "True":
        return 1
    else:
        return 0


# In[18]:


def handleM(_dt_train, _train_user, _train_Q):#
    _dt_train['prior_question_had_explanation'].fillna(False, inplace=True)
    _dt_train["prior_question_had_explanation_enc"] = _dt_train['prior_question_had_explanation'].apply(convertBoolean)
    
    
    _dt_train = _dt_train.merge(_train_user, how = 'left', on = 'user_id')
    del _train_user
    gc.collect()
    
    _dt_train = _dt_train.merge(_train_Q, how = 'left', left_on = 'content_id',right_on = 'question_id')
    del _train_Q
    gc.collect()
    
    _dt_train['mean_user_content_accuracy'] =2 * (_dt_train['user_mean_accuracy'] *  _dt_train['Q_mean_accuracy']) / (_dt_train['user_mean_accuracy'] + _dt_train['Q_mean_accuracy'])
  
    _dt_train = _dt_train.drop(columns=['user_id', 'content_id','prior_question_had_explanation','question_id'])
       
    _dt_train.fillna(0.5, inplace=True)
    
    return _dt_train
    


# In[19]:


#train=train.sample(frac=0.4,random_state=10)
train = train.sort_values(['timestamp'], ascending=True)
train_part_len=int(len(train)*0.66)
train = train.iloc[train_part_len:,:]


# In[20]:


gc.collect()


# In[21]:


dt_train=handleM(train, train_user, train_Q)#


# In[22]:


train = None
del(train)
gc.collect()


# In[23]:


dt_train.info()


# In[24]:


features = [
    # user features
    'user_mean_accuracy',
    'user_count_correct',
    'user_count',
    # content features
    'Q_mean_accuracy',
    # part features
    'part',
    # other features
    'Q_mean_accuracy',
    'attempt',
    'prior_question_elapsed_time'
]



target = 'answered_correctly'

# add categorical features indices
categorical_feature = ['part', 'tags', 'tags_label', 'prior_question_had_explanation']
categorical_feature_idxs = []
for v in categorical_feature:
    try:
        categorical_feature_idxs.append(features.index(v))
    except:
        pass
    
dt_y = dt_train[target]
dt_x = dt_train[features]


# In[25]:


dt_x.info()


# In[26]:


dt_train = None
del(dt_train)
gc.collect()


# In[27]:


valid_fraction = 0.05
valid_size = int(len(dt_x) * valid_fraction)

train_x = dt_x[:-1 * valid_size]
valid_x = dt_x[-valid_size:]


train_y = dt_y[:-1 * valid_size]
valid_y = dt_y[-valid_size:]


# In[28]:


print("dt_x shape" + str(dt_x.shape))
print("train_x shape" + str(train_x.shape))
print("valid_x shape" + str(valid_x.shape))


# In[29]:


features=train_x.columns.tolist()


# In[30]:


train_x


# In[31]:


gc.collect()


# In[ ]:





# In[32]:


import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

dtrain = lgb.Dataset(train_x, label=train_y)
dvalid = lgb.Dataset(valid_x, label=valid_y)

params= {
    'objective': 'binary',
    'seed': 42,
    'metric': 'auc',
    'learning_rate': 0.05,
    'max_bin': 1500,
    'num_leaves': 70 
    }
    
model = lgb.train(
        params, 
        dtrain, 
        num_boost_round=2500, 
        valid_sets=[dtrain,dvalid], 
        early_stopping_rounds=20, 
        verbose_eval=50,
        categorical_feature = categorical_feature_idxs,
        feature_name = features,
        )
#print('auc:', roc_auc_score(test_y,model.predict(test_x)))
#print('auc:', roc_auc_score(valid_y,model.predict(valid_x)))   


# In[33]:


#displaying the most important features
lgb.plot_importance(model)
plt.show()


# In[34]:


import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


# In[35]:


for (test_df, sample_prediction_df) in iter_test:
    test_df['prior_question_had_explanation'].fillna(False, inplace=True)
    test_df["prior_question_had_explanation_enc"] = test_df['prior_question_had_explanation'].apply(convertBoolean)

    
    test_df = test_df[test_df.content_type_id == 0]
    test_df['attempt'] = test_df.assign(dif = ((test_df.content_id.diff() != 0) | (test_df.user_id.diff() != 0))).groupby(["user_id","content_id"]).dif.cumsum()
    test_df['attempt']=test_df['attempt'] - 1
    test_df['attempt'] = test_df['attempt'].apply(lambda x: 4 if x >= 4 else x)
    
    
    test_df = test_df.merge(train_user, how = 'left', on = 'user_id')
    test_df = test_df.merge(train_Q, how = 'left', left_on = 'content_id',right_on = 'question_id')
    
    
    test_df['mean_user_content_accuracy'] =2 * (test_df['user_mean_accuracy'] *  test_df['Q_mean_accuracy']) / (test_df['user_mean_accuracy'] + test_df['Q_mean_accuracy'])


    test_df.fillna(0.5, inplace=True)
   

    test_df['answered_correctly'] =  model.predict(test_df[features])
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

