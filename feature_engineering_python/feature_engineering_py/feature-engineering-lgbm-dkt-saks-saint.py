#!/usr/bin/env python
# coding: utf-8

# ## References
# 1. Loop Feature Engineering: https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering
# 2. Cross Validation: https://www.kaggle.com/its7171/cv-strategy
# 3. Deep Knowledge Tracing (DKT): https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
# 4. Self-Attentive model for Knowledge Tracing (SAKT): https://arxiv.org/pdf/1907.06837.pdf
# 5. SAINT: https://arxiv.org/pdf/2010.12042.pdf

# In[1]:


import pickle
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict, deque
from tqdm.notebook import tqdm
import lightgbm as lgb


# In[2]:


train_pickle = '../input/riiid-cross-validation-files/cv1_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv1_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'


# # Feature Engineering

# For LGBM, I use 4 additional features: 
# 1. the number of questions that a student solved
# 2. the number of questions that a student answered correctly
# 3. the correction rate of a student = (2)/(1)
# 4. time difference between two interactive sessions: timestamp(t) - timestamp(t-1). When t = 0, I set the difference equals zero

# In[3]:


last_time_u_dict = defaultdict(int)
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)

past_question = defaultdict(list)
past_time_diff = defaultdict(list)
past_prior_elaps = defaultdict(list)
past_answer = defaultdict(list)


# In[4]:


# funcs for user stats with loop
def add_user_feats(df,last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    td = np.zeros(len(df), dtype=np.int64)

    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly','timestamp']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        td[cnt] = row[2] - last_time_u_dict[row[0]] #time difference can be 0 when there are more than 1 questions in an interactive session
        
        if row[2] == last_time_u_dict[row[0]]: #This fixes the problem. 
            td[cnt] = td[cnt-1]
            
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
        last_time_u_dict[row[0]] = row[2]
                  
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu, 'time_diff':td})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df


# In[5]:


def add_user_feats2(df, past_question, past_answer, past_prior_elaps, past_time_diff):
    
    for _, row in enumerate(tqdm(df[['user_id','answered_correctly','content_id','prior_question_elapsed_time','time_diff']].values)):
        
        if row[0] not in past_question:
            new_list = deque([0]*30, maxlen = 30)
            new_list.append(row[2])
            past_question[row[0]] = new_list
        else:
            past_question[row[0]].append(row[2])
            

        if row[0] not in past_answer:
            new_list = deque([0]*30, maxlen = 30)
            new_list.append(row[1])
            past_answer[row[0]] = new_list
        else:
            past_answer[row[0]].append(row[1])     

            
        if row[0] not in past_prior_elaps:
            new_list = deque([0]*30, maxlen = 30)
            new_list.append(row[3])
            past_prior_elaps[row[0]] = new_list
        else:
            past_prior_elaps[row[0]].append(row[3])   
            
            
        if row[0] not in past_time_diff:
            new_list = deque([0]*30, maxlen = 30)
            new_list.append(row[4])
            past_time_diff[row[0]] = new_list
        else:
            past_time_diff[row[0]].append(row[4])   


# In[6]:


def add_user_feats_without_update(df ,last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict):

    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    td = np.zeros(len(df), dtype=np.int64)
    
    for cnt, row in enumerate(df[['user_id','part','timestamp']].values):
        td[cnt] = row[2] - last_time_u_dict[row[0]]
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        
                
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu, 'time_diff':td})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df


# In[7]:


def get_user_feats_for_nn_without_update(df, past_question, past_answer, past_prior_elaps, past_time_diff):
    current_question = []
    past_question_answer = []
    past_answer_correctly = []
    past_time = []
    past_prior = []
    #past_other_feats = []
    
    
    for cnt,row in enumerate(tqdm(df[['user_id','content_id','prior_question_elapsed_time','time_diff']].values)):
        
        if row[0] not in past_answer:
            temp_answer = [0]*30
            temp_past_answer = [0]*30
        else:
            temp_answer = past_answer[row[0]].copy()
            temp_past_answer = past_answer[row[0]].copy()
        
        
        if row[0] not in past_question:
            temp_question = [0]*30
        else:
            temp_question = past_question[row[0]].copy()
        
        
        temp_past_answer= [x+1 if y > 0 else 0 for x , y in zip(temp_past_answer, temp_question)]
        past_answer_correctly.append(temp_past_answer)
        
        temp_past_question_answer = [x+y*13523 for x,y in zip(temp_question, temp_answer)]
        past_question_answer.append(temp_past_question_answer)
        
        temp_question.append(row[1]+1)
        current_question.append([temp_question[i] for i in range(30)])
        
        
        if row[0] not in past_prior_elaps:
            temp_elaps = [0]*29 + [row[2]/3e5]
        else:
            temp_elaps = past_prior_elaps[row[0]].copy()
            temp_elaps.append(row[2]/3e5)
        past_prior.append(temp_elaps)
        
        
        if row[0] not in past_time_diff:
            temp_time_diff = [0]*29 + [row[3]/1e6]
        else:
            temp_time_diff = past_time_diff[row[0]].copy()
            temp_time_diff.append(row[3]/1e6)
            
        past_time.append(temp_time_diff)
    
    current_question = np.array(current_question)
    past_question_answer = np.array(past_question_answer)    
    past_other_feats = np.dstack((past_prior,past_time))
    past_answer_correctly = np.array(past_answer_correctly)
    
    return current_question, past_question_answer, past_other_feats, past_answer_correctly


# In[8]:


def update_user_feats(df, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id', 'answered_correctly', 'content_type_id', 'timestamp', 'content_id','prior_question_elapsed_time','time_diff']].values:
        if row[2] == 0:    
            
            if row[0] not in past_question:
                new_list = deque([0]*30, maxlen = 30)
                new_list.append(row[4]+1)
                past_question[row[0]] = new_list
            else:
                past_question[row[0]].append(row[4]+1)
                
            
            if row[0] not in past_answer:
                new_list = deque([0]*30, maxlen = 30)
                new_list.append(row[1])
                past_answer[row[0]] = new_list
            else:
                past_answer[row[0]].append(row[1])     

            
            if row[0] not in past_prior_elaps:
                new_list = deque([0]*30, maxlen = 30)
                new_list.append(row[5]/3e5)
                past_prior_elaps[row[0]] = new_list
            else:
                past_prior_elaps[row[0]].append(row[5]/3e5) 
                
                
            if row[0] not in past_time_diff:
                new_list = deque([0]*30, maxlen = 30)
                if row[6] >= 1e6:
                    new_list.append(0)
                else:
                    new_list.append(row[6]/1e6)
                past_time_diff[row[0]] = new_list
            else:
                if row[6] >= 1e6:
                    past_time_diff[row[0]].append(0)   
                else:
                    past_time_diff[row[0]].append(row[6]/1e6)               
            
                
            last_time_u_dict[row[0]] = row[3]
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1


# In[9]:


# read data
feld_needed = ['user_id','content_id','answered_correctly','timestamp','prior_question_elapsed_time','prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]


# In[10]:


train = train.loc[train.answered_correctly != -1].reset_index(drop=True)
valid = valid.loc[valid.answered_correctly != -1].reset_index(drop=True)
_=gc.collect()


# In[11]:


prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()


# In[12]:


content_df = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','std']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c','answered_correctly_std_c']


# In[13]:


train1 = train.groupby('user_id').tail(31).reset_index(drop=True)
valid1 = valid.groupby('user_id').tail(31).reset_index(drop=True)

train = train[-5000000:].reset_index(drop=True) #this notebook is just an illustration, thus, I will use 5M trainig.

_=gc.collect()


# In[14]:


# extract features for training purpose

train = add_user_feats(train, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)
valid = add_user_feats(valid, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)


# In[15]:


questions_df = pd.read_csv(question_file)
questions_df.tags.fillna('-1-1', inplace = True)
questions_df['tag'] = pd.factorize(questions_df.tags)[0]


# In[16]:


# I repeat the step so that I extract features from all students for prediction purpose.

last_time_u_dict = defaultdict(int)
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)


train1 = add_user_feats(train1, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)
valid1 = add_user_feats(valid1, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)


train1 = pd.merge(train1, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid1 = pd.merge(valid1, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')

train1.time_diff[train1.time_diff >= 1e6] = 1e6
valid1.time_diff[valid1.time_diff >= 1e6] = 1e6


train1.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean, inplace = True)
valid1.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean, inplace = True)


add_user_feats2(train1, past_question, past_answer, past_prior_elaps, past_time_diff)
add_user_feats2(valid1, past_question, past_answer, past_prior_elaps, past_time_diff)

del(train1)
del(valid1)
_=gc.collect()


# In[17]:


train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")


# In[18]:


train = pd.merge(train, questions_df[['question_id', 'part','tag']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part','tag']], left_on = 'content_id', right_on = 'question_id', how = 'left')


# In[19]:


train['prior_question_elapsed_time'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')


# ## LGBM

# In[20]:


TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_u', 'answered_correctly_avg_c','answered_correctly_std_c','answered_correctly_sum_u','count_u', 'part','tag', 'prior_question_elapsed_time','time_diff', 'prior_question_had_explanation']
dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
#train.drop(dro_cols, axis=1, inplace=True)
#valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()


# In[21]:


lgb_train = lgb.Dataset(train[FEATS], y_tr, categorical_feature = ['part', 'prior_question_had_explanation', 'tag'],free_raw_data=False)
lgb_valid = lgb.Dataset(valid[FEATS], y_va, categorical_feature = ['part', 'prior_question_had_explanation', 'tag'], reference=lgb_train,free_raw_data=False)
#del train, y_tr
_=gc.collect()


# # Hyperparameter Tuning

# In[22]:


import optuna


# In[23]:


def objective(trial):    
    params = {
            'num_leaves': trial.suggest_int('num_leaves', 32, 512),
            'max_bin': trial.suggest_int('max_bin', 700, 900),
            'boosting' : 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 4, 16),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 16),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 4, 80),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
            'early_stopping_rounds': 5
            }
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=1000, num_boost_round=10000)
    val_pred = model.predict(valid[FEATS])
    score = roc_auc_score(y_va, val_pred)
    print(f"AUC = {score}")
    return score


# In[24]:


# Bayesian optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1) #you can increase trial numbers to get better parameters


# In[25]:


print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


# In[26]:


parameters = trial.params
parameters['objective'] = 'binary'
parameters['metric'] = 'auc'
parameters['early_stopping_rounds'] = 5
parameters['boosting'] = 'gbdt'


# In[27]:


lgbm_model = lgb.train(
                    parameters, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                )

print('auc:', roc_auc_score(y_va, lgbm_model.predict(valid[FEATS])))
_ = lgb.plot_importance(lgbm_model)


# In[28]:


import matplotlib.pyplot as plt
lgb.plot_importance(lgbm_model, importance_type = 'gain')
plt.show()


# # DEEP LEARNING MODELs

# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, Embedding, Concatenate, Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import multiply, Reshape
from tensorflow.keras.utils import Sequence


# ###### Feature Engineering for Deep learning Models:
# 
# I need to modify some features so that nueral network performs as I intend. So I rescale some variables:
# 1. If time difference is larger than 10^6 = around 16 minutes, then I set time difference equals 10^6.
# 2. I rescale time difference and prior question elapsed time varialbes by 1e6 and 3e5, respectively so that they take value between 0 and 1. (NN performs funny when the values are too large)

# In[30]:


#Feature Engineering for Deep learning Models:
train.time_diff[train.time_diff >= 1e6] = 1e6
valid.time_diff[valid.time_diff >= 1e6] = 1e6


train.prior_question_elapsed_time = train.prior_question_elapsed_time/3e5
train.time_diff = train.time_diff/1e6


valid.prior_question_elapsed_time = valid.prior_question_elapsed_time/3e5 
valid.time_diff = valid.time_diff/1e6


# Now, I am going to reshape the dataframe into 3d matrix: sample_size,window_size,features
# 1. I will throw away new students (who solved less than 30 questions) for the fast computation purpose only.
# 2. I choose window size = 30. In SAINT paper, they choose 100. I believe this is one of hyperparameter that you need to tune yourself.
# 3. I pad on the left.
# 3. I add +1 on tag, question_id, lagged_answered_correctly (decoder input for SAINT) as I will use zeros to mask
# 
# There are 5 features to work on for Deep learning architectures:
# 1. answered_correctly (output)
# 2. lagged_answered_correctly (decoder input for SAINT)
# 3. current_question_id (encoder input for SAINT, query for SAKT, input for DKT)
# 4. prior_time_elapsed (decoder input for SAINT, SAKT, and DKT)
# 5. time difference (decoder input for SAINT, SAKT, and DKT)

# In[31]:


#train_user_count = train.user_id.value_counts()
#train_del_user = train_user_count[train_user_count<30]
#train = train[~train.user_id.isin(train_del_user.index)]


#valid_user_count = valid.user_id.value_counts()
#valid_del_user = valid_user_count[valid_user_count<30]
#valid = valid[~valid.user_id.isin(valid_del_user.index)]


# In[32]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 31


# In[33]:


train_group = train.groupby('user_id')
train_y = [frame['answered_correctly'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in train_group
           for i in range(0, len(frame['answered_correctly'].to_numpy()[:, None]), max_len)]
train_y = np.reshape(pad_sequences(train_y, padding="pre"),(-1,max_len))


train_past_answer = [(frame['answered_correctly']+1).to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in train_group
           for i in range(0, len(frame['answered_correctly'].to_numpy()[:, None]), max_len) ]
train_past_answer = np.reshape(pad_sequences(train_past_answer, padding="pre"),(-1,max_len))


train_current_question = [(frame['content_id']+1).to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in train_group
                         for i in range(0, len(frame['content_id'].to_numpy()[:,None]), max_len )]
train_current_question = np.reshape(pad_sequences(train_current_question, padding="pre"),(-1,max_len))


train_prior_elaps_time = [frame['prior_question_elapsed_time'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in train_group
                         for i in range(0, len(frame['prior_question_elapsed_time'].to_numpy()[:,None]), max_len )]
train_prior_elaps_time = np.reshape(pad_sequences(train_prior_elaps_time, padding="pre", dtype = 'float32'),(-1,max_len))


train_time_diff = [frame['time_diff'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in train_group
                         for i in range(0, len(frame['time_diff'].to_numpy()[:,None]), max_len )]
train_time_diff = np.reshape(pad_sequences(train_time_diff, padding="pre", dtype = 'float32'),(-1,max_len))


# In[34]:


train_answered_correcly = train_y[:,1:]
train_past_answered_correctly = train_past_answer[:,:-1]


# Following line will create tuple (question_id, answered_correctly)
train_past_question_answer = train_current_question[:,:-1] + train_y[:,:-1]*(train.content_id.max()+1)


train_current_question = train_current_question[:,1:]


train_prior_elaps_time = train_prior_elaps_time[:,1:]
train_time_diff = train_time_diff[:,1:]


# In[35]:


valid_group = valid.groupby('user_id')
valid_y = [frame['answered_correctly'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in valid_group
           for i in range(0, len(frame['answered_correctly'].to_numpy()[:, None]), max_len) ]
valid_y = np.reshape(pad_sequences(valid_y, padding="pre"),(-1,max_len))


valid_past_answer = [(frame['answered_correctly']+1).to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in valid_group
           for i in range(0, len(frame['answered_correctly'].to_numpy()[:, None]), max_len) ]
valid_past_answer = np.reshape(pad_sequences(valid_past_answer, padding="pre"),(-1,max_len))


valid_current_question = [(frame['content_id']+1).to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in valid_group
                         for i in range(0, len(frame['content_id'].to_numpy()[:,None]), max_len )]
valid_current_question = np.reshape(pad_sequences(valid_current_question, padding="pre"),(-1,max_len))


valid_prior_elaps_time = [frame['prior_question_elapsed_time'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in valid_group
                         for i in range(0, len(frame['prior_question_elapsed_time'].to_numpy()[:,None]), max_len )]
valid_prior_elaps_time = np.reshape(pad_sequences(valid_prior_elaps_time, padding="pre", dtype = 'float32'),(-1,max_len,1))


valid_time_diff = [frame['time_diff'].to_numpy()[:, None].tolist()[i:i+max_len] for _, frame in valid_group
                         for i in range(0, len(frame['time_diff'].to_numpy()[:,None]), max_len )]
valid_time_diff = np.reshape(pad_sequences(valid_time_diff, padding="pre", dtype = 'float32'),(-1,max_len,1))


# In[36]:


valid_answered_correcly = valid_y[:,1:]
valid_past_answered_correctly = valid_past_answer[:,:-1]
valid_past_question_answer = valid_current_question[:,:-1] + valid_y[:,:-1]*(valid.content_id.max()+1)


valid_current_question = valid_current_question[:,1:]


valid_prior_elaps_time = valid_prior_elaps_time[:,1:]
valid_time_diff = valid_time_diff[:,1:]


# # SAKT

# I motify the code from https://www.tensorflow.org/tutorials/text/transformer. 
# You will have to look closely at encoder and encoder layer which I modified.

# In[37]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_output_mask(seq):
    seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
    return seq[:, :, tf.newaxis]  # (batch_size, 1, 1, seq_len)



def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)




def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)




def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                       (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights





def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


    
    
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.2):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, e, training, mask):

        attn_output, _ = self.mha(x, x, e, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(e + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                   maximum_position_encoding, rate=0.2):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, e, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, e, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, en_d_model, en_num_heads, dff, pe_input, rate=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, en_d_model, en_num_heads, dff, 
                           pe_input, rate)


        self.second_final_layer = tf.keras.layers.Dense(dff)
        self.final_layer = Dense(1,activation = 'sigmoid')
    
    def call(self, inp1, inp2, training, mask):

        enc_output = self.encoder(inp1, inp2, training, mask)  # (batch_size, inp_seq_len, d_model)
            
        second_final_output = self.second_final_layer(enc_output)  # (batch_size, tar_seq_len, question_answer_pair_size)
        final_output = self.final_layer(second_final_output)
        return final_output


# In[38]:


num_layers = 2
d_model = 64
num_heads = 2
dff = 256 
question_answer_pair_size = train_past_question_answer.max()+1

n_question = train_current_question.max()+1


pe_input = 30


def build(num_layers, d_model, num_heads, dff, question_answer_pair_size, n_question, pe_input):
    
    masking_func = lambda inputs, previous_mask: previous_mask
    en_input1 = Input(batch_shape = (None, None), name = 'question_answer_pair')
    en_input1_embed = Embedding(question_answer_pair_size, d_model)(en_input1)
    en_input2 = Input(batch_shape = (None, None, 1), name = 'other_feature1')
    en_input2_embed = Dense(d_model, input_shape = (None, None, 1))(en_input2)
    en_input3 = Input(batch_shape = (None, None, 1), name = 'other_feature2')
    en_input3_embed = Dense(d_model, input_shape = (None, None, 1))(en_input3)
    
    en_input_embed_sum = tf.math.add_n([en_input1_embed, en_input2_embed, en_input3_embed])
    
    
    en_input4 = Input(batch_shape = (None, None), name = 'current_question')
    en_input4_embed = Embedding(n_question, d_model)(en_input4)
    
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(en_input_embed_sum)[1])
    padding_mask = create_padding_mask(en_input1)
    combined_mask = tf.maximum(look_ahead_mask, padding_mask)
    
    
    transformer = Transformer(num_layers, d_model, num_heads, dff, pe_input)
    
    final_output = transformer(en_input_embed_sum, en_input4_embed, True, combined_mask)
    output_mask = create_padding_output_mask(en_input1)
    output = multiply([final_output, output_mask])
    
    model = Model(inputs=[en_input1, en_input2, en_input3, en_input4], outputs=output)
    model.compile( optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics=['accuracy',AUC()])
    
    return model

SAKT_model = build(num_layers, d_model, num_heads, dff, question_answer_pair_size, n_question, pe_input)


# In[39]:


SAKT_model.fit([train_past_question_answer, train_prior_elaps_time, train_time_diff, train_current_question],train_answered_correcly, 
             validation_data=([valid_past_question_answer, valid_prior_elaps_time, valid_time_diff, valid_current_question], valid_answered_correcly), batch_size = 200,
             epochs = 2, verbose = 1)


# # SAINT

# In[40]:


class EncoderLayer2(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.2):
        super(EncoderLayer2, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



class Encoder2(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                   maximum_position_encoding, rate=0.2):
        super(Encoder2, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer2(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    
class DecoderLayer2(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.2):
        super(DecoderLayer2, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
                look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    
    
class Decoder2(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                    maximum_position_encoding, rate=0.2):
        super(Decoder2, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer2(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
               look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
              x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights    

class Transformer2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, padding_length, rate=0.2):
        super(Transformer2, self).__init__()

        self.encoder = Encoder2(num_layers, d_model, num_heads, dff, padding_length)
        
        self.decoder = Decoder2(num_layers, d_model, num_heads, dff, padding_length)

        self.second_final_layer = tf.keras.layers.Dense(dff)
        self.final_layer = Dense(1,activation = 'sigmoid')
    
    def call(self, inp1, inp2, training, en_combined_mask, de_look_ahead_mask, de_padding_mask):

        enc_output = self.encoder(inp1, training, en_combined_mask)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                inp2, enc_output, training, de_look_ahead_mask, de_padding_mask)
            
        second_final_output = self.second_final_layer(dec_output)  # (batch_size, tar_seq_len, question_answer_pair_size)
        final_output = self.final_layer(second_final_output)
        return final_output


# In[41]:


num_layers = 1
d_model = 64
num_heads = 2
dff = 256


n_question = train_current_question.max()+1
n_answer = 3

pe_input = 30

def build(num_layers, d_model, num_heads, dff, n_question, n_answer, pe_input):

    en_input1 = Input(batch_shape = (None, None), name = 'current_question')
    en_input1_embed = Embedding(n_question, d_model)(en_input1)

    
    en_look_ahead_mask = create_look_ahead_mask(tf.shape(en_input1)[1])
    en_padding_mask = create_padding_mask(en_input1)
    en_combined_mask = tf.maximum(en_look_ahead_mask, en_padding_mask)
    
    
    
    #en_input1_embed = K.sum(en_input1_embed, axis = -2)
    de_input2 = Input(batch_shape = (None, None), name = 'past_answer')
    de_input2_embed = Embedding(n_answer, d_model)(de_input2)
    de_input3 = Input(batch_shape = (None, None, 1), name = 'other_feature1')
    de_input3_embed = Dense(d_model, input_shape = (None, None, 1))(de_input3)
    de_input4 = Input(batch_shape = (None, None, 1), name = 'other_feature2')
    de_input4_embed = Dense(d_model, input_shape = (None, None, 1))(de_input4)   
    de_input = tf.math.add_n([de_input2_embed, de_input3_embed, de_input4_embed])
    
    #de_look_ahead_mask = create_look_ahead_mask(tf.shape(de_input4)[1])
    #de_padding_mask = create_padding_mask(de_input4)
    #de_combined_mask = tf.maximum(de_look_ahead_mask, de_padding_mask)
    
    
    transformer = Transformer2(num_layers, d_model, num_heads, dff, pe_input)
    
    final_output = transformer(en_input1_embed, de_input, True, en_combined_mask, en_combined_mask, en_combined_mask)
    output_mask = create_padding_output_mask(en_input1)
    
    model = Model(inputs=[en_input1, de_input2, de_input3, de_input4], outputs=final_output)
    model.compile( optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics=['accuracy',AUC()])
    
    return model

SAINT_model = build(num_layers, d_model, num_heads, dff, n_question, n_answer, pe_input)


# In[42]:


SAINT_model.fit([train_current_question, train_past_answered_correctly, train_prior_elaps_time, train_time_diff],train_answered_correcly, 
             validation_data=([valid_current_question, valid_past_answered_correctly, valid_prior_elaps_time, valid_time_diff], valid_answered_correcly), 
             batch_size = 200,
             epochs = 2, verbose = 1)


# # DKT

# For DKT, I subtract 1 from current_question, (tag, answered_correctly) tuple, (part, answered_correctly) tuple. This is because I will use tensorflow.one_hot to create one hot encoding. tensorflow.one_hot works in a following way: suppose I have a vector 
# 
# > y = [0, 2, -1, 1]
# 
# and I have 4 cateogries and each are indexed by 0-3. Then,
# 
# > tensorflow.one_hot(y,4) =
# 
# > [[1, 0, 0, 0],  # one_hot(0)
# 
# >  [0, 0, 1, 0],  # one_hot(2)
# 
# >  [0, 0, 0, 0],  # one_hot(-1)
# 
# >  [0, 1, 0, 0]]  # one_hot(1)
# 
# Thus, I need to change padding with 0 with -1. Then, once I create one hot encoding, I can use zeros to mask.

# In[43]:


train_answered_correcly = np.reshape(train_y[:,1:],(-1,30,1))
train_past_question_answer -= 1
train_current_question -= 1
train_other_feats = np.dstack((train_prior_elaps_time,train_time_diff))

valid_answered_correcly = np.reshape(valid_y[:,1:],(-1,30,1))
valid_past_question_answer -= 1
valid_current_question -= 1
valid_other_feats = np.dstack((valid_prior_elaps_time,valid_time_diff))


# In[44]:


# Parameter setting
other_input_dim = 2
hidden_layer_size = 50
input_dim_order = train_current_question.max() + 1
prev_q_perform_dim = train_past_question_answer.max() + 1


# In[45]:


def dkt_build(hidden_layer_size, input_dim_order, prev_q_perform_dim, other_input_dim):    
    # Inputs of DKT: tuples of question and answers. As we have 13k+ questions, the total number of tuples will be 23k+.
    # One hot encoding for 23k+ categorical variables will consume too much memory.
    # Thus, I replace questions with tags and parts.
    # Plus, I added two additional features: prior_elaps_time and time_diff as used in SAINT
    
    masking_func = lambda inputs, previous_mask: previous_mask #masking_func is needed as lambda function I used below will ignore a masking layer
    
    prev_question_ans = Input(batch_shape = (None, None), dtype = 'int32', name = 'prev_qn_ans') # input1 (questions, answer correctly) tuple
    one_hot_prev_question_ans = tf.one_hot(prev_question_ans, prev_q_perform_dim, axis = -1) # one hot encoding input1
    
    other_input = Input(batch_shape = (None, None, other_input_dim), name= 'other_input') # other features
    
    one_hot = Concatenate()([one_hot_prev_question_ans, other_input]) # concatenate all input features
    
    masked_one_hot = (Masking(mask_value= 0, input_shape = (None, None, prev_q_perform_dim + other_input_dim)))(one_hot) # mask
    
    
    lstm_out = LSTM(hidden_layer_size, input_shape = (None, None, prev_q_perform_dim + other_input_dim),
                    dropout=0.2, recurrent_dropout =0.2, return_sequences = True)(masked_one_hot)
    
    
    dense_out = Dense(input_dim_order, input_shape = (None, None, hidden_layer_size), activation='sigmoid')(lstm_out)
    
    order = Input(batch_shape = (None, None), dtype = 'int32', name = 'order') # input3 (questions)
    one_hot_order = tf.one_hot(order, input_dim_order, axis = -1) # one hot encoding questions
    
    # vector multiplication
    merged = multiply([dense_out, one_hot_order])
    
    def reduce_dim(x):
        x = K.max(x, axis = 2, keepdims = True)
        return x

    def reduce_dim_shape(input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        print ("reduced_shape", shape)
        return tuple(shape)
    
    reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape, mask = masking_func)(merged)
    
    
    model = Model(inputs=[prev_question_ans, other_input, order], outputs=reduced)
    model.compile( optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics=['accuracy',AUC()])

    return model


# In[46]:


dkt_model = dkt_build(hidden_layer_size, input_dim_order, prev_q_perform_dim, other_input_dim)


# In[47]:


dkt_model.fit([train_past_question_answer, train_other_feats, train_current_question], train_answered_correcly,
                    validation_data=([valid_past_question_answer, valid_other_feats, valid_current_question], valid_answered_correcly),
                          epochs = 2, verbose = 1, batch_size = 200)


# # Prediction

# In[48]:


import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()
set_predict = env.predict


# In[49]:


previous_test_df = None
for (test_df, sample_prediction_df) in iter_test:
    if previous_test_df is not None:
        previous_test_df[TARGET] = np.array(eval(test_df["prior_group_answers_correct"].iloc[0]))[mask]
        update_user_feats(previous_test_df, last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)
    
    test_df = pd.merge(test_df, questions_df[['question_id', 'part','tag']], left_on='content_id', right_on = 'question_id', how='left')
    mask = (test_df['content_type_id'] == 0).values.tolist()
    test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
    test_df = add_user_feats_without_update(test_df , last_time_u_dict, answered_correctly_sum_u_dict, count_u_dict)
    test_df = pd.merge(test_df, content_df, on='content_id',  how="left")
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time'] = test_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
    previous_test_df = test_df.copy()
    lgbm_predict =  lgbm_model.predict(test_df[FEATS])
    
    current_question, past_question_answer, past_other_feats, past_answer_correctly = get_user_feats_for_nn_without_update(test_df, past_question, past_answer, past_prior_elaps, past_time_diff)
    prior_elaps_time = np.reshape(past_other_feats[:,:,0],(-1,30,1))
    time_diff = np.reshape(past_other_feats[:,:,1],(-1,30,1))    
    
    sakt_predict = SAKT_model.predict([past_question_answer, prior_elaps_time, time_diff, current_question], batch_size = 500)    
    
    saint_predict = SAINT_model.predict([current_question, past_answer_correctly, prior_elaps_time, time_diff], batch_size = 500)
    
    past_question_answer = past_question_answer - 1
    current_question = current_question - 1
    
    dkt_predict = dkt_model.predict([past_question_answer, past_other_feats, current_question], batch_size = 200)    
    
    test_df[TARGET] = dkt_predict[:,-1,0]*(0.1)  + sakt_predict[:,-1,0]*(0.15) + lgbm_predict*0.6 + saint_predict[:,-1,0]*(0.15)
    
    set_predict(test_df[['row_id', TARGET]])
    #---
    #print(sample_prediction_df)
    #print(test_df[['row_id', TARGET]])
    #print(test_df.shape, sample_prediction_df.shape, test_df[TARGET].shape)
    #---

