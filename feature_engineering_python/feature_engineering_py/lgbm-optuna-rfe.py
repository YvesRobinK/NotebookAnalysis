#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gc

import riiideducation
from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


env = riiideducation.make_env()


# In[3]:


train = pd.read_csv(
    '/kaggle/input/riiid-test-answer-prediction/train.csv',
    usecols=[1, 2, 3, 4, 5, 7, 8, 9],
    dtype={
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'answered_correctly':'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'boolean'
    }
)


# In[4]:


questions_df = pd.read_csv(
    '/kaggle/input/riiid-test-answer-prediction/questions.csv',                         
    usecols=[0, 3],
    dtype={
        'question_id': 'int16',
        'part': 'int8'}
)


# In[5]:


lectures_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')


# ### Thanks https://www.kaggle.com/takamotoki/lgbm-iii-part3-adding-lecture-features for feature creation part

# In[6]:


lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')
lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])
part_lectures_columns = [column for column in lectures_df.columns if column.startswith('part')]
types_of_lectures_columns = [column for column in lectures_df.columns if column.startswith('type_of_')]


# In[7]:


train_lectures = train[train.content_type_id == True].merge(lectures_df, left_on='content_id', right_on='lecture_id', how='left')


# In[8]:


user_lecture_stats_part = train_lectures.groupby('user_id')[part_lectures_columns + types_of_lectures_columns].sum()


# In[9]:


for column in user_lecture_stats_part.columns:
    bool_column = column + '_boolean'
    user_lecture_stats_part[bool_column] = (user_lecture_stats_part[column] > 0).astype(int)


# In[10]:


del train_lectures
gc.collect()


# In[11]:


train = train[train.content_type_id == False].sort_values('timestamp').reset_index(drop = True)


# In[12]:


elapsed_mean = train.prior_question_elapsed_time.mean()


# In[13]:


group1 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['count'])
group1.columns = ['avg_questions']
group2 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['nunique'])
group2.columns = ['avg_questions']
group3 = group1 / group2


# In[14]:


group3['avg_questions_seen'] = group3.avg_questions.cumsum()


# In[15]:


results_u_final = train.loc[train.content_type_id == False, ['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_final.columns = ['answered_correctly_user']

results_u2_final = train.loc[train.content_type_id == False, ['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_final.columns = ['explanation_mean_user']


# In[16]:


prior_mean_user = results_u2_final.explanation_mean_user.mean()


# In[17]:


train = pd.merge(train, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')


# In[18]:


results_q_final = train.loc[train.content_type_id == False, ['question_id','answered_correctly']].groupby(['question_id']).agg(['mean'])
results_q_final.columns = ['quest_pct']


# In[19]:


results_q2_final = train.loc[train.content_type_id == False, ['question_id','part']].groupby(['question_id']).agg(['count'])
results_q2_final.columns = ['count']


# In[20]:


question2 = pd.merge(questions_df, results_q_final, left_on = 'question_id', right_on = 'question_id', how = 'left')


# In[21]:


question2 = pd.merge(question2, results_q2_final, left_on = 'question_id', right_on = 'question_id', how = 'left')


# In[22]:


question2.quest_pct = round(question2.quest_pct, 5)


# In[23]:


train.drop(['timestamp', 'content_type_id', 'question_id', 'part'], axis=1, inplace=True)


# In[24]:


validation = train.groupby('user_id').tail(5)
train = train[~train.index.isin(validation.index)]


# In[25]:


results_u_val = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_val.columns = ['answered_correctly_user']

results_u2_val = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_val.columns = ['explanation_mean_user']


# In[26]:


X = train.groupby('user_id').tail(18)
train = train[~train.index.isin(X.index)]
len(X) + len(train) + len(validation)


# In[27]:


results_u_X = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_X.columns = ['answered_correctly_user']

results_u2_X = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_X.columns = ['explanation_mean_user']


# In[28]:


del(train)
gc.collect()


# In[29]:


X = pd.merge(X, group3, left_on=['task_container_id'], right_index= True, how="left")
X = pd.merge(X, results_u_X, on=['user_id'], how="left")
X = pd.merge(X, results_u2_X, on=['user_id'], how="left")

X = pd.merge(X, user_lecture_stats_part, on=['user_id'], how="left")


# In[30]:


validation = pd.merge(validation, group3, left_on=['task_container_id'], right_index= True, how="left")
validation = pd.merge(validation, results_u_val, on=['user_id'], how="left")
validation = pd.merge(validation, results_u2_val, on=['user_id'], how="left")

validation = pd.merge(validation, user_lecture_stats_part, on=['user_id'], how="left")


# In[31]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

X.prior_question_had_explanation.fillna(False, inplace = True)
validation.prior_question_had_explanation.fillna(False, inplace = True)

validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])
X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])


# In[32]:


content_mean = question2.quest_pct.mean()


# In[33]:


question2.quest_pct = question2.quest_pct.mask((question2['count'] < 3), .65)

question2.quest_pct = question2.quest_pct.mask((question2.quest_pct < .2) & (question2['count'] < 21), .2)

question2.quest_pct = question2.quest_pct.mask((question2.quest_pct > .95) & (question2['count'] < 21), .95)


# In[34]:


X = pd.merge(X, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
validation = pd.merge(validation, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
X.part = X.part - 1
validation.part = validation.part - 1


# In[35]:


y = X['answered_correctly']
X = X.drop(['answered_correctly'], axis=1)

y_val = validation['answered_correctly']
X_val = validation.drop(['answered_correctly'], axis=1)


# In[36]:


X = X[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
       'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
       'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
       'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
       'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']]

X_val = X_val[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
               'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
               'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
               'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
               'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
               'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']]


# In[37]:


X['answered_correctly_user'].fillna(0.65,  inplace=True)
X['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X['quest_pct'].fillna(content_mean, inplace=True)

X['part'].fillna(4, inplace = True)
X['avg_questions_seen'].fillna(1, inplace = True)
X['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X['prior_question_had_explanation_enc'].fillna(0, inplace = True)

X['part_1'].fillna(0, inplace = True)
X['part_2'].fillna(0, inplace = True)
X['part_3'].fillna(0, inplace = True)
X['part_4'].fillna(0, inplace = True)
X['part_5'].fillna(0, inplace = True)
X['part_6'].fillna(0, inplace = True)
X['part_7'].fillna(0, inplace = True)
X['type_of_concept'].fillna(0, inplace = True)
X['type_of_intention'].fillna(0, inplace = True)
X['type_of_solving_question'].fillna(0, inplace = True)
X['type_of_starter'].fillna(0, inplace = True)
X['part_1_boolean'].fillna(0, inplace = True)
X['part_2_boolean'].fillna(0, inplace = True)
X['part_3_boolean'].fillna(0, inplace = True)
X['part_4_boolean'].fillna(0, inplace = True)
X['part_5_boolean'].fillna(0, inplace = True)
X['part_6_boolean'].fillna(0, inplace = True)
X['part_7_boolean'].fillna(0, inplace = True)
X['type_of_concept_boolean'].fillna(0, inplace = True)
X['type_of_intention_boolean'].fillna(0, inplace = True)
X['type_of_solving_question_boolean'].fillna(0, inplace = True)
X['type_of_starter_boolean'].fillna(0, inplace = True)


# In[38]:


X_val['answered_correctly_user'].fillna(0.65,  inplace=True)
X_val['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X_val['quest_pct'].fillna(content_mean,  inplace=True)

X_val['part'].fillna(4, inplace = True)
X_val['avg_questions_seen'].fillna(1, inplace = True)
X_val['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X_val['prior_question_had_explanation_enc'].fillna(0, inplace = True)

X_val['part_1'].fillna(0, inplace = True)
X_val['part_2'].fillna(0, inplace = True)
X_val['part_3'].fillna(0, inplace = True)
X_val['part_4'].fillna(0, inplace = True)
X_val['part_5'].fillna(0, inplace = True)
X_val['part_6'].fillna(0, inplace = True)
X_val['part_7'].fillna(0, inplace = True)
X_val['type_of_concept'].fillna(0, inplace = True)
X_val['type_of_intention'].fillna(0, inplace = True)
X_val['type_of_solving_question'].fillna(0, inplace = True)
X_val['type_of_starter'].fillna(0, inplace = True)
X_val['part_1_boolean'].fillna(0, inplace = True)
X_val['part_2_boolean'].fillna(0, inplace = True)
X_val['part_3_boolean'].fillna(0, inplace = True)
X_val['part_4_boolean'].fillna(0, inplace = True)
X_val['part_5_boolean'].fillna(0, inplace = True)
X_val['part_6_boolean'].fillna(0, inplace = True)
X_val['part_7_boolean'].fillna(0, inplace = True)
X_val['type_of_concept_boolean'].fillna(0, inplace = True)
X_val['type_of_intention_boolean'].fillna(0, inplace = True)
X_val['type_of_solving_question_boolean'].fillna(0, inplace = True)
X_val['type_of_starter_boolean'].fillna(0, inplace = True)


# In[39]:


params = {
    'num_leaves': 31, 
    'n_estimators': 200, 
    'max_depth': 8, 
    'min_child_samples': 356, 
    'learning_rate': 0.2982483634778906, 
    'min_data_in_leaf': 82, 
    'bagging_fraction': 0.6545628633239445, 
    'feature_fraction': 0.9164482379289846,
    'random_state': 666
}

full_model = LGBMClassifier(**params)
full_model.fit(X, y)

preds = full_model.predict_proba(X_val)[:,1]
print('LGB roc auc', roc_auc_score(y_val, preds))

full_xgb = XGBClassifier(random_state=666)
full_xgb.fit(X, y)

preds = full_xgb.predict_proba(X_val)[:,1]
print('XGB roc auc', roc_auc_score(y_val, preds))

full_lr = LogisticRegression(random_state=666)
full_lr.fit(X, y)

preds = full_lr.predict_proba(X_val)[:,1]
print('LR roc auc', roc_auc_score(y_val, preds))


# In[40]:


import optuna
from optuna.samplers import TPESampler


# In[41]:


rfe = RFE(estimator=DecisionTreeClassifier(random_state=666), n_features_to_select=14)
rfe.fit(X, y)
X = rfe.transform(X)
X_val = rfe.transform(X_val)


# In[42]:


sampler = TPESampler(seed=666)

def create_model(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 31)
    n_estimators = trial.suggest_int("n_estimators", 20, 300)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)
    model = LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_child_samples=min_child_samples, 
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        random_state=666
)
    return model

def objective(trial):
    model = create_model(trial)
    model.fit(X, y)
    preds = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, preds)
    return score

# run optuna
# study = optuna.create_study(direction="maximize", sampler=sampler)
# study.optimize(objective, n_trials=350)
# params = study.best_params
# params['random_state'] = 666

params = {
    'num_leaves': 28, 
    'n_estimators': 295, 
    'max_depth': 8, 
    'min_child_samples': 1178, 
    'learning_rate': 0.2379173491475032, 
    'min_data_in_leaf': 35, 
    'bagging_fraction': 0.8389723511600549, 
    'feature_fraction': 0.9606189400533491,
    'random_state': 666
}

model = LGBMClassifier(**params)
model.fit(X, y)

preds = model.predict_proba(X_val)[:,1]
roc_auc_score(y_val, preds)


# In[43]:


X = pd.DataFrame(X)
X_val = pd.DataFrame(X_val)

y = pd.DataFrame(y)
y_val = pd.DataFrame(y_val)


# In[44]:


models = []
preds = []
for n, (tr, te) in enumerate(KFold(n_splits=5, random_state=666, shuffle=True).split(y)):
    print(f'Fold {n}')
    model = LGBMClassifier(**params)
    model.fit(X.values[tr], y.values[tr])
    
    pred = model.predict_proba(X_val)[:, 1]
    preds.append(pred)
    print('Fold roc auc:', roc_auc_score(y.values[te], model.predict_proba(X.values[te])[:, 1])) 
    models.append(model)


# In[45]:


predictions = preds[0]
for i in range(1, 5):
    predictions += preds[i]
predictions /= 5

print('ROC AUC', roc_auc_score(y_val, predictions))


# In[46]:


iter_test = env.iter_test()


# In[47]:


for (test_df, sample_prediction_df) in iter_test:
    test_df['task_container_id'] = test_df.task_container_id.mask(test_df.task_container_id > 9999, 9999)
    test_df = pd.merge(test_df, group3, left_on=['task_container_id'], right_index= True, how="left")
    test_df = pd.merge(test_df, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
    test_df = pd.merge(test_df, results_u_final, on=['user_id'],  how="left")
    test_df = pd.merge(test_df, results_u2_final, on=['user_id'],  how="left")
    
    test_df = pd.merge(test_df, user_lecture_stats_part, on=['user_id'], how="left")
    test_df['part_1'].fillna(0, inplace = True)
    test_df['part_2'].fillna(0, inplace = True)
    test_df['part_3'].fillna(0, inplace = True)
    test_df['part_4'].fillna(0, inplace = True)
    test_df['part_5'].fillna(0, inplace = True)
    test_df['part_6'].fillna(0, inplace = True)
    test_df['part_7'].fillna(0, inplace = True)
    test_df['type_of_concept'].fillna(0, inplace = True)
    test_df['type_of_intention'].fillna(0, inplace = True)
    test_df['type_of_solving_question'].fillna(0, inplace = True)
    test_df['type_of_starter'].fillna(0, inplace = True)
    test_df['part_1_boolean'].fillna(0, inplace = True)
    test_df['part_2_boolean'].fillna(0, inplace = True)
    test_df['part_3_boolean'].fillna(0, inplace = True)
    test_df['part_4_boolean'].fillna(0, inplace = True)
    test_df['part_5_boolean'].fillna(0, inplace = True)
    test_df['part_6_boolean'].fillna(0, inplace = True)
    test_df['part_7_boolean'].fillna(0, inplace = True)
    test_df['type_of_concept_boolean'].fillna(0, inplace = True)
    test_df['type_of_intention_boolean'].fillna(0, inplace = True)
    test_df['type_of_solving_question_boolean'].fillna(0, inplace = True)
    test_df['type_of_starter_boolean'].fillna(0, inplace = True)
    
    test_df['answered_correctly_user'].fillna(0.65,  inplace=True)
    test_df['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
    test_df['quest_pct'].fillna(content_mean,  inplace=True)
    test_df['part'] = test_df.part - 1

    test_df['part'].fillna(4, inplace = True)
    test_df['avg_questions_seen'].fillna(1, inplace = True)
    test_df['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
    test_df['prior_question_had_explanation'].fillna(False, inplace=True)
    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])
    
    full_preds = full_model.predict_proba(test_df[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
                                                            'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
                                                            'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
                                                            'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
                                                            'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
                                                            'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']])[:, 1]
    
    full_preds_xgb = full_xgb.predict_proba(test_df[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
                                                            'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
                                                            'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
                                                            'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
                                                            'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
                                                            'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']])[:, 1]
    
    full_preds_lr = full_lr.predict_proba(test_df[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
                                                            'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
                                                            'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
                                                            'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
                                                            'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
                                                            'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']])[:, 1]
    


    
    X_test = rfe.transform(test_df[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
                                                            'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
                                                            'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
                                                            'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
                                                            'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
                                                            'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']])
    
    preds = [model.predict_proba(X_test)[:,1] for model in models]
    
    predictions = preds[0]
    for i in range(1, 5):
        predictions += preds[i]
    predictions /= 5
    
    test_df['answered_correctly'] =  predictions * 0.65 + full_preds * 0.15 + full_preds_xgb * 0.15 + full_preds_lr * 0.05
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])


# In[ ]:




