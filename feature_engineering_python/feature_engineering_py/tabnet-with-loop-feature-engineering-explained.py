#!/usr/bin/env python
# coding: utf-8

# ## Credits
# 
# This code is almost entirely coming from this one : https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering
# 
# Please give credit to the original version and upvote @its7171 work.
# 
# ## What about this notebook?
# 
# ### How to simply use TabNet
# - It shows that TabNet is as easy to use as LGBM or XGBoost.
# - When using GPU and OneCycleLearningRate you can get decent results in a decent amount of time. Note that this is the worst case scenario to compare training time between boosting algorithm and TabNet as boosting algorithm get very slow with high number of classes while TabNet stays almost as fast. Also the number of features is small, this lower the advantage of using a GPU as even large batches won't fill the GPU.
# - Also this shows how to use embeddings properly as I have seen some mistakes shared on other notebooks.
# 
# ### How to take advantage of interpretability?
# - I think I haven't emphasize enough the power of explanability given by attention mechanism in my previous posts. I tried here to give a few hits on how explanability could be used to understand the model better.
# - Interpretability is always underestimated in Kaggle Competitions as it won't help for the final score. But in practise, it's always good to be able to give some explanations either in production (to the final user) or before getting to production (to convince the C-level board members). 
# 
# ### Disclaimer
# - I haven't spend much time on this competition so this is a very shallow analysis but I hope it would inspire some people to push this further.
# - I guess the final score could be better with parameter tuning and more features. Not sure that TabNet will outperform LGBM here, could still be a powerful addition in a blend.
# - A new release is coming soon, with a few bugfixes and TabNetPretrainer. Stay tuned!

# In[1]:


get_ipython().system(' pip install /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl')


# In[2]:


import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm.notebook import tqdm
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## setting
# CV files are generated by [this notebook](https://www.kaggle.com/its7171/cv-strategy)

# In[3]:


train_pickle = '../input/riiid-cross-validation-files/cv1_train.pickle'
valid_pickle = '../input/riiid-cross-validation-files/cv1_valid.pickle'
question_file = '../input/riiid-test-answer-prediction/questions.csv'
debug = False
validaten_flg = False


# ## feature engineering

# In[4]:


# funcs for user stats with loop
def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1


# In[5]:


# read data
feld_needed = ['row_id', 'user_id', 'content_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]


# In[6]:


# read data
feld_needed = ['row_id', 'user_id', 'content_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]
if debug:
    train = train[:1000000]
    valid = valid[:100000]
else:
    # Not using all training data as I came across memory issues
    train = train[:10000000]
    valid = valid[:100000]
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# answered correctly average for each content
content_df = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c']
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

# user stats features with loops
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)
train = add_user_feats(train, answered_correctly_sum_u_dict, count_u_dict)
valid = add_user_feats(valid, answered_correctly_sum_u_dict, count_u_dict)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# use only last 30M training data for limited memory on kaggle env.
#train = train[-30000000:]

# part
questions_df = pd.read_csv(question_file)
train = pd.merge(train, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')


# # Defining features and categorical features

# In[7]:


TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_u',
         'answered_correctly_sum_u',
         'count_u',
         'answered_correctly_avg_c',
         'part',
         'prior_question_had_explanation',
         'prior_question_elapsed_time']

# Define categorical features
CAT_FEATS = ['part']

cat_idxs = []
cat_dims = []
for cat_feat in CAT_FEATS:
    cat_idx = FEATS.index(cat_feat)
    cat_dim = train[cat_feat].nunique()
    cat_idxs.append(cat_idx)
    cat_dims.append(cat_dim)
    
# Label encode categorical features
label_encoders = {}
for cat_feat in CAT_FEATS:
    l_enc = LabelEncoder()
    train.loc[:, cat_feat] = l_enc.fit_transform(train.loc[:, cat_feat].values.reshape(-1, 1))
    valid.loc[:, cat_feat] = l_enc.transform(valid.loc[:, cat_feat].values.reshape(-1, 1))
    label_encoders[cat_feat] = l_enc

dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
train.drop(dro_cols, axis=1, inplace=True)
valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()


# In[8]:


X_train, y_train = train[FEATS].values, y_tr.values
X_valid, y_valid = valid[FEATS].values, y_va.values

# TabNet does not allow Nan values
# A better fillna method might improve scores
X_train = np.nan_to_num(X_train, nan=-1)
X_valid = np.nan_to_num(X_valid, nan=-1)

del train, y_tr
_=gc.collect()


# ## modeling

# In[9]:


BS = 2**12

# Training for more epoch might improve the model performance
# at the cost of longer training time
MAX_EPOCH = 10

# Defining TabNet model
model = TabNetClassifier(n_d=32, n_a=32, n_steps=3, gamma=1.2,
                         n_independent=2, n_shared=2,
                         lambda_sparse=0., seed=0,
                         clip_value=1,
                         cat_idxs=cat_idxs,
                         cat_dims=cat_dims,
                         cat_emb_dim=1,
                         mask_type='entmax',
                         device_name='auto',
                         optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=2e-2),
                         scheduler_params=dict(max_lr=0.05,
                                               steps_per_epoch=int(X_train.shape[0] / BS),
                                               epochs=MAX_EPOCH,
                                               #final_div_factor=100,
                                               is_batch_level=True),
                         scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                         verbose=1,)

model.fit(X_train=X_train, y_train=y_train,
          eval_set=[(X_valid, y_valid)],
          eval_name=["valid"],
          eval_metric=["auc"],
          batch_size=BS,
          virtual_batch_size=256,
          max_epochs=MAX_EPOCH,
          drop_last=True,
          pin_memory=True
         )


# In[10]:


val_preds = model.predict_proba(X_valid)[:, -1]
print('auc:', roc_auc_score(y_va, val_preds))


# # Basic Features importance
# 
# This is what you find with almost all machine learning models.
# 
# It's useful to discard features and also to understand in a very high level point of view what's important for the model.
# 
# Since TabNet is selecting features on the fly, I'm not sure it's very useful to discard features though.

# In[11]:


feat_importances = model.feature_importances_
indices = np.argsort(feat_importances)


# In[12]:


plt.figure()
plt.title("Feature importances")
plt.barh(range(len(feat_importances)), feat_importances[indices],
       color="r", align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(len(feat_importances)), [FEATS[idx] for idx in indices])
plt.ylim([-1, len(feat_importances)])
plt.show()


# # Understanding the model better

# In[13]:


LIMIT_EXPLAIN = 100000
explain_mat, masks = model.explain(X_valid[:LIMIT_EXPLAIN, :])
# Normalize the importance by sample
normalized_explain_mat = np.divide(explain_mat, explain_mat.sum(axis=1).reshape(-1, 1))

# Add prediction to better understand correlation between features and predictions
explain_and_preds = np.hstack([normalized_explain_mat, val_preds[:LIMIT_EXPLAIN].reshape(-1, 1)])


# In[14]:


import plotly.express as px

px.imshow(explain_and_preds[:200, :],
          labels=dict(x="Features", y="Samples", color="Importance"),
          x=FEATS+["prediction"],
          title="Sample wise feature importance (reality is more complex than global feature importance)")


# In[15]:


correlation_importance = np.corrcoef(explain_and_preds.T)

px.imshow(correlation_importance,
          labels=dict(x="Features", y="Features", color="Correlation"),
          x=FEATS+["prediction"], y=FEATS+["prediction"],
          title="Correlation between attention mechanism for each feature and predictions")


# # How to read this?
# 
# Understanding a complex Machine Learning algorithm is a hard task. Here I'm just giving some hints on what could be done to understand the results of TabNet, taking advantage of the attention mechanism.
# 
# ### What can we say here ?
# 
# It seems that `prior_question_had_explanation` is totally useless for the model, previous question does not seem to be relevant to predict success on the current question. It also seems that the model does not care about `answered_correctly_avg_u` this might come from the fact that using `answered_correctly_avg_u` is enough to take a decision.
# 
# For example, you can see a high negative correlation between `answered_correctly_avg_c` and `answered_correctly_avg_u`. In the meantime you see a positive correlation between `prediction` and `answered_correctly_avg_c` while a (small) negative correlation between `prediction` and `answered_correctly_avg_u`.
# 
# So maybe one way to say this in English : When the model sees an easy question (high `answered_correctly_avg_c` ) it does not need to look at how good the student is. When it needs to look at harder questions, then it matters to see if the student was doing good previously or not.
# 
# Those sentences are partially false of course, but can give a general statement about what the model is looking at. This can help users trust the ML model and also select input features so that the reasonning align more with something accpetable. A more in-depth analysis would give better explanations at a question/student level.
# 
# ### Disclaimer : results changes at each epoch
# 
# It's actually funny to see how those results can change even for the same model at different epochs. The model changes its mind at each epoch, for example after one epoch it still looks sometimes at `prior_question_had_explanation`, but after a few epochs it stops completely and starts disregarding this feature.

# ## inference

# In[16]:


class Iter_Valid(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self
    
    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df= self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1
        pre_content_type_id = -1
        user_answer_list = []
        answered_correctly_list = []
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]
            if crr_user_id in added_user and (crr_user_id != pre_added_user or (crr_task_container_id != pre_task_container_id and crr_content_type_id == 0 and pre_content_type_id == 0)):
                # known user(not prev user or (differnt task container and both question))
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and (crr_task_container_id == pre_task_container_id or crr_content_type_id == 1):
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            pre_content_type_id = crr_content_type_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()


# In[17]:


# You can debug your inference code to reduce "Submission Scoring Error" with `validaten_flg = True`.
# Please refer https://www.kaggle.com/its7171/time-series-api-iter-test-emulator about Time-series API (iter_test) Emulator.

if validaten_flg:
    target_df = pd.read_pickle(valid_pickle)
    if debug:
        target_df = target_df[:10000]
    iter_test = Iter_Valid(target_df,max_user=1000)
    predicted = []
    def set_predict(df):
        predicted.append(df)
    # reset answered_correctly_sum_u_dict and count_u_dict
    answered_correctly_sum_u_dict = defaultdict(int)
    count_u_dict = defaultdict(int)
    train = pd.read_pickle(train_pickle)[['user_id','answered_correctly','content_type_id']]
    if debug:
        train = train[:1000000]
    train = train[train.content_type_id == False].reset_index(drop=True)
    update_user_feats(train, answered_correctly_sum_u_dict, count_u_dict)
    del train
else:
    import riiideducation
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict


# In[18]:


previous_test_df = None
for (test_df, sample_prediction_df) in iter_test:
    if previous_test_df is not None:
        previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
        update_user_feats(previous_test_df, answered_correctly_sum_u_dict, count_u_dict)
    previous_test_df = test_df.copy()
    test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
    test_df = add_user_feats_without_update(test_df, answered_correctly_sum_u_dict, count_u_dict)
    test_df = pd.merge(test_df, content_df, on='content_id',  how="left")
    test_df = pd.merge(test_df, questions_df, left_on='content_id', right_on='question_id', how='left')
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time_mean'] = test_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
    
    for cat_feat in CAT_FEATS:
        l_enc = label_encoders[cat_feat]
        test_df.loc[:, cat_feat] = l_enc.fit_transform(test_df.loc[:, cat_feat].values.reshape(-1, 1))
    
    X_test = np.nan_to_num(test_df[FEATS].values, nan=-1)
    test_df[TARGET] =  model.predict_proba(X_test)[:, -1]
    set_predict(test_df[['row_id', TARGET]])


# In[19]:


if validaten_flg:
    y_true = target_df[target_df.content_type_id == 0].answered_correctly
    y_pred = pd.concat(predicted).answered_correctly
    print(roc_auc_score(y_true, y_pred))


# ### You've reach the end! Congrats!
