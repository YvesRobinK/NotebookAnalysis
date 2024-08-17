#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install dabl')


# In[2]:


import pandas as pd
import numpy as np
import string
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
get_ipython().run_line_magic('matplotlib', 'inline')
import dabl

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)


from IPython.display import Markdown
def bold(string):
    display(Markdown(string))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 


# ![](https://www.koreatechtoday.com/wp-content/uploads/2020/04/riiid-logo-background-scaled.jpg)
# 
# **In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data. [source](https://www.kaggle.com/c/riiid-test-answer-prediction)**

# In[3]:


path = '/kaggle/input'

train = pd.read_csv(f'{path}/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=9 * (10**5), 
                       dtype={'row_id': 'int64',
                              'timestamp': 'int64',
                              'user_id': 'int32',
                              'content_id': 'int16',
                              'content_type_id': 'int8',
                              'task_container_id': 'int16',
                              'user_answer': 'int8',
                              'answered_correctly': 'int8',
                              'prior_question_elapsed_time': 'float32', 
                              'prior_question_had_explanation': 'boolean',
                             }
                      )

test = pd.read_csv(f'{path}/riiid-test-answer-prediction/example_test.csv')
submit = pd.read_csv(f'{path}/riiid-test-answer-prediction/example_sample_submission.csv')
questions = pd.read_csv(f'{path}/riiid-test-answer-prediction/questions.csv')
lectures = pd.read_csv(f'{path}/riiid-test-answer-prediction/lectures.csv')
print('Train shapes: ', train.shape)
print('Test shapes: ', test.shape)


# In[4]:


def description(df):
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.iloc[0].values
    summary['Second Value'] = df.iloc[1].values
    summary['Third Value'] = df.iloc[2].values
    return summary


# In[5]:


description(train)


# In[6]:


## pandas describe

train.describe()


# #### Files
# **train.csv**
# 
# * `row_id:` (int64) ID code for the row.
# 
# * `timestamp:` (int64) the time between this user interaction and the first event from that user.
# 
# * `user_id:` (int32) ID code for the user.
# 
# * `content_id:` (int16) ID code for the user interaction
# 
# * `content_type_id:` (int8) 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.
# 
# * `task_container_id:` (int16) Id code for the batch of questions or lectures. For example, a user might see three questions in a row before seeing the explanations for any of them. Those three would all share a task_container_id. Monotonically increasing for each user.
# 
# * `user_answer:` (int8) the user's answer to the question, if any. Read -1 as null, for lectures.
# 
# * `answered_correctly:` (int8) if the user responded correctly. Read -1 as null, for lectures.
# 
# * `prior_question_elapsed_time:` (float32) How long it took a user to answer their previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Note that the time is the total time a user took to solve all the questions in the previous bundle.
# 
# * `prior_question_had_explanation:` (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback.
# 
# **questions.csv: metadata for the questions posed to users.**
# 
# * `question_id:` foreign key for the train/test content_id column, when the content type is question (0).
# 
# * `bundle_id:` code for which questions are served together.
# 
# * `correct_answer:` the answer to the question. Can be compared with the train user_answer column to check if the user was right.
# 
# * `part:` top level category code for the question.
# 
# * `tags:` one or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions together.
# 
# **lectures.csv: metadata for the lectures watched by users as they progress in their education.**
# 
# * `lecture_id:` foreign key for the train/test content_id column, when the content type is lecture (1).
# 
# * `part:` top level category code for the lecture.
# 
# * `tag:` one tag codes for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.
# 
# * `type_of:` brief description of the core purpose of the lecture
# 
# 

# # Train.csv
# 
# ## Target Distribution- Answered correctly

# In[7]:


total = len(train)
plt.figure(figsize=(10,6))

g = sns.countplot(x='answered_correctly', data=train, palette='viridis')
g.set_title("TARGET DISTRIBUTION", fontsize = 20)
g.set_xlabel("Target Vaues", fontsize = 15)
g.set_ylabel("Count", fontsize = 15)
sizes=[] # Get highest values in y
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

plt.show()


# **On average users answer 64.33% questions correctly. -1 as null, for lectures, we should exclude them for answers analysis.**

# In[8]:


id_col = ['user_id', 'content_id', 'content_type_id', 'task_container_id']
plt.figure(figsize=(10,6))
for i, col in enumerate(id_col):
    plt.subplot(2, 2, i + 1)
    sns.distplot(train[col], color='green', 
                 hist_kws={'alpha':1,"linewidth": 2},
                 kde_kws={"color": "red", "lw": 2, 'bw':0.01})
    plt.title(col)
    plt.tight_layout()


# In[9]:


time_col = ['timestamp', 'prior_question_elapsed_time',]
plt.figure(figsize=(10,6))
for i, col in enumerate(time_col):
    plt.subplot(1, 2, i + 1)
    train[col].hist(bins = 50,color='red')
    plt.title(col)
    plt.tight_layout()


# **Timestamp represents the time from the first user interaction to the current one and Prior question elapsed time represents how long it took a user to answer their previous question bundle.**

# In[10]:


col = ['prior_question_had_explanation', 'user_answer',]

total = len(train)
plt.figure(figsize=(12,5), dpi=60)

for i, col in enumerate(col):
    plt.subplot(1, 2, i + 1)
    g=sns.countplot(train[col], palette='cividis')
    sizes=[] # Get highest values in y
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 
    g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights
    plt.title(col)
    plt.tight_layout()


# In[11]:


ans_col = ['answered_correctly', 'user_answer',]

total = len(train)
plt.figure(figsize=(12,5), dpi=100)

for i, col in enumerate(ans_col):
    plt.subplot(1, 2, i + 1)
    sns.countplot(train[col], palette='cividis', hue = train['prior_question_had_explanation'])
    plt.title(col)
    plt.tight_layout()


# **Majority user saw an explanation and the correct responses after answering the previous question bundle.**

# ### Correct Answers by users
# 
# some code is taken from https://www.kaggle.com/ilialar/simple-eda-and-baseline https://www.kaggle.com/lgreig/simple-lgbm-baseline

# In[12]:


train_only_df = train[train['answered_correctly']!=-1]
grouped_by_user_df = train_only_df.groupby('user_id')
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'sum', 'std', 'median', 'skew']})

fig,ax=plt.subplots(figsize=(15,8), dpi=100)

plt.subplot(2, 2, 1)
g1=user_answers_df[('answered_correctly','mean')].hist(bins=100, color='teal')
g1.set_title("users correct answer mean dist.", fontweight='bold')

plt.subplot(2, 2, 2)
g2=user_answers_df[('answered_correctly','count')].hist(bins=100, color='teal')
g2.set_title('users correct answer count dist.', fontweight='bold')

plt.subplot(2, 2, 3)
g3=user_answers_df[user_answers_df[('answered_correctly','count')]<= 100][('answered_correctly','mean')].hist(bins=100, color='teal')
g3.set_title('users correct answer mean dist. less than 100 question', fontweight='bold')

plt.subplot(2, 2, 4)
g4=user_answers_df[user_answers_df[('answered_correctly','count')]>=100][('answered_correctly','mean')].hist(bins=100, color='teal')
g4.set_title('users correct answer count dist. more than 100 question', fontweight='bold')
plt.tight_layout()
plt.show()


# **Average user score is lower than the overall % of correct answers(bottom left graph). It means heavy users have even better scores(bottom right graph).**

# In[13]:


user_time_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'],
                                        'timestamp': ['mean', 'count']})

fig,ax=plt.subplots(figsize=(15,10), dpi=100)

plt.subplot(3, 2, 1)
plt.scatter(x = user_answers_df[('answered_correctly','count')], 
            y = user_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w correct answer mean and count',fontweight='bold')

plt.subplot(3, 2, 2)
plt.scatter(x = user_answers_df[('answered_correctly','std')], 
            y = user_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w correct answer mean and std',fontweight='bold')

plt.subplot(3, 2, 3)
plt.scatter(x = user_answers_df[('answered_correctly','median')], 
            y = user_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w correct answer mean and mediam',fontweight='bold')

plt.subplot(3, 2, 4)
plt.scatter(x = user_answers_df[('answered_correctly','skew')], 
            y = user_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w correct answer mean and skew',fontweight='bold')

plt.subplot(3, 2, 5)
plt.scatter(x = user_answers_df[('answered_correctly','sum')], 
            y = user_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w correct answer mean and sum',fontweight='bold')

plt.subplot(3, 2, 6)
plt.scatter(x = user_time_answers_df[ ('timestamp','mean')], 
            y = user_time_answers_df[ ('answered_correctly','mean')], color='teal')
plt.title('relation b/w  timestamp mean and correct answer mean',fontweight='bold')
plt.tight_layout()
plt.show()


# **There is relationship between the average score for the active user, and the number of questions answered;  there is relation average timestamp and average correct answer can be useful for baseline.**

# ### Correct Answers by Content

# In[14]:


grouped_by_content_df = train_only_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'sum','std', 'median', 'skew']})

fig,ax=plt.subplots(figsize=(15,8), dpi=100)

plt.subplot(2, 2, 1)
g1=content_answers_df[('answered_correctly','mean')].hist(bins=100, color='darkred')
g1.set_title("content correct answer mean dist.", fontweight='bold')

plt.subplot(2, 2, 2)
g2=content_answers_df[('answered_correctly','count')].hist(bins=100, color='darkred')
g2.set_title('content answer count dist.', fontweight='bold')

plt.subplot(2, 2, 3)
g3=content_answers_df[content_answers_df[('answered_correctly','count')]<= 100][('answered_correctly','mean')].hist(bins=100, color='darkred')
g3.set_title('content correct answer mean dist. less than 100 question', fontweight='bold')

plt.subplot(2, 2, 4)
g4=content_answers_df[content_answers_df[('answered_correctly','count')]>=100][('answered_correctly','mean')].hist(bins=100, color='darkred')
g4.set_title('content correct answer count dist. more than 100 question', fontweight='bold')
plt.tight_layout()
plt.show()


# In[15]:


user_time_content_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'],
                                        'timestamp': ['mean', 'count']})

fig,ax=plt.subplots(figsize=(15,10), dpi=100)

plt.subplot(3, 2, 1)
plt.scatter(x = content_answers_df[('answered_correctly','count')], 
            y=content_answers_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w correct answer mean and count', fontweight='bold')

plt.subplot(3, 2, 2)
plt.scatter(x = content_answers_df[('answered_correctly','std')], 
            y=content_answers_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w correct answer mean and std', fontweight='bold')

plt.subplot(3, 2, 3)
plt.scatter(x = content_answers_df[('answered_correctly','median')], 
            y=content_answers_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w correct answer mean and median', fontweight='bold')

plt.subplot(3, 2, 4)
plt.scatter(x = content_answers_df[('answered_correctly','skew')], 
            y=content_answers_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w correct answer mean and skew', fontweight='bold')

plt.subplot(3, 2, 5)
plt.scatter(x = content_answers_df[('answered_correctly','sum')], 
            y=content_answers_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w correct answer mean and sum', fontweight='bold')

plt.subplot(3, 2, 6)
plt.scatter(x = user_time_content_df[ ('timestamp','mean')], 
            y = user_time_content_df[ ('answered_correctly','mean')], color='darkred')
plt.title('relation b/w  timestamp mean and correct answer count', fontweight='bold')
plt.tight_layout()
plt.show()


# # Question.csv
# 

# In[16]:


description(questions)


# In[17]:


id_col = ['question_id', 'bundle_id']
plt.figure(figsize=(10,6))
for i, col in enumerate(id_col):
    plt.subplot(1, 2, i + 1)
    sns.distplot(questions[col], color='green',bins=100, 
                 hist_kws={'alpha':1,"linewidth": 1},
                 kde_kws={"color": "red", "lw": 2, 'bw':0.01})
    plt.title(col)
    plt.tight_layout()


# In[18]:


questions_df = questions.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')
bundle_dict = questions_df['bundle_id'].value_counts().to_dict()

questions_df['right_answers'] = questions_df[('answered_correctly', 'mean')] * questions_df[('answered_correctly', 'count')]
questions_df['bundle_size'] = questions_df['bundle_id'].apply(lambda x: bundle_dict[x])


# In[19]:


questions_df.head()


# In[20]:


col = ['correct_answer', 'part', 'bundle_size']

total = len(questions_df)
plt.figure(figsize=(15,8), dpi=100)

for i, col in enumerate(col):
    plt.subplot(2, 2, i + 1)
    g=sns.countplot(questions_df[col], palette='gist_yarg')
    sizes=[] # Get highest values in y
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 
    g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights
    plt.title(col)
    plt.tight_layout()


# In[21]:


fig,ax=plt.subplots(figsize=(15,6), dpi=50)
plt.subplot(1, 2, 1)
plt.scatter(x = questions_df[('answered_correctly','count')], 
            y=questions_df['right_answers'], color='royalblue')
plt.title('relation b/w right answer and question asked (count)', fontweight='bold')

plt.subplot(1, 2, 2)
plt.scatter(x = questions_df['right_answers'], 
            y = questions_df[ ('answered_correctly','mean')], color='royalblue')
plt.title('relation b/w  right_answers and correct answer mean', fontweight='bold')
plt.tight_layout()
plt.show()


# In[22]:


grouped_by_bundle_df = questions_df.groupby('bundle_id')
bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', ('answered_correctly', 'count'): 'sum'}).copy()
bundle_answers_df.columns = ['bundle_rignt_answers', 'bundle_questions_asked']
bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_rignt_answers'] / bundle_answers_df['bundle_questions_asked']


# In[23]:


bundle_answers_df.head()


# In[24]:


fig,ax=plt.subplots(figsize=(15,6), dpi=50)
plt.subplot(1, 2, 1)
plt.scatter(x = bundle_answers_df['bundle_questions_asked'], 
            y=bundle_answers_df['bundle_accuracy'], color='dodgerblue')
plt.title('relation b/w bundle_questions_asked and bundle_accuracy', fontweight='bold')

plt.subplot(1, 2, 2)
plt.scatter(x = bundle_answers_df['bundle_rignt_answers'], 
            y = bundle_answers_df['bundle_accuracy'], color='dodgerblue')
plt.title('relation b/w  bundle_rignt_answers and bundle_accuracy', fontweight='bold')
plt.tight_layout()
plt.show()


# In[25]:


grouped_by_part_df = questions_df.groupby('part')
part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', ('answered_correctly', 'count'): 'sum'}).copy()
part_answers_df.columns = ['part_rignt_answers', 'part_questions_asked']
part_answers_df['part_accuracy'] = part_answers_df['part_rignt_answers'] / part_answers_df['part_questions_asked']
part_answers_df


# # Lectures.csv

# In[26]:


description(lectures)


# In[27]:


col = ['type_of', 'part']

total = len(lectures)
plt.figure(figsize=(15,8), dpi=100)

for i, col in enumerate(col):
    plt.subplot(2, 2, i + 1)
    g=sns.countplot(lectures[col], palette='RdGy')
    sizes=[] # Get highest values in y
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 
    g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights
    plt.title(col)
    plt.tight_layout()


# ## Feature Engineering

# **Let's use this new feature in our baseline model**

# In[28]:


train_df = train[train['answered_correctly']!=-1]
grouped_by_user_df = train_df.groupby('user_id')
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'sum','std', 'median', 'skew']}).copy()
user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'sum_user_accuracy', 'std_user_accuracy', 'median_user_accuracy', 'skew_user_accuracy']


# In[29]:


grouped_by_content_df = train_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'sum', 'std', 'median', 'skew'] }).copy()
content_answers_df.columns = ['mean_accuracy', 'question_asked', 'sum_accuracy', 'std_accuracy', 'median_accuracy', 'skew_accuracy']


# In[30]:


questions_df = questions.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')
bundle_dict = questions_df['bundle_id'].value_counts().to_dict()

questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']
questions_df['bundle_size'] =questions_df['bundle_id'].apply(lambda x: bundle_dict[x])


# In[31]:


grouped_by_bundle_df = questions_df.groupby('bundle_id')
bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()
bundle_answers_df.columns = ['bundle_rignt_answers', 'bundle_questions_asked']
bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_rignt_answers'] / bundle_answers_df['bundle_questions_asked']


# In[32]:


grouped_by_part_df = questions_df.groupby('part')
part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()
part_answers_df.columns = ['part_rignt_answers', 'part_questions_asked']
part_answers_df['part_accuracy'] = part_answers_df['part_rignt_answers'] / part_answers_df['part_questions_asked']


# In[33]:


lectures_df = lectures.groupby('part')
lectures_df = lectures_df.agg({'type_of': 'count'})
lectures_df.columns = ['type_of_count']


# In[34]:


new_train_df = train_df.merge(user_answers_df, how = 'left', on = 'user_id')\
                        .merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')\
                        .merge(bundle_answers_df, how = 'left', on = 'bundle_id')\
                        .merge(part_answers_df, how = 'left', on = 'part')\
                        .merge(lectures_df, how='left',on='part')
del train
del questions
del lectures
del train_only_df
del grouped_by_user_df
del grouped_by_content_df
del grouped_by_bundle_df
del grouped_by_part_df

import gc
gc.collect()


# In[35]:


new_train_df


# In[36]:


le = LabelEncoder()
new_train_df['prior_question_had_explanation'] = new_train_df['prior_question_had_explanation'].fillna(value = False).astype(bool)
new_train_df["prior_question_had_explanation"] = le.fit_transform(new_train_df["prior_question_had_explanation"])
new_train_df.fillna(0, inplace = True)


# In[37]:


#https://github.com/dabl/dabl
plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('fivethirtyeight')
dabl.plot(new_train_df, target_col = 'answered_correctly')


# In[38]:


corr = new_train_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(25, 25))
    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.5,cmap="coolwarm",annot=True)


# ## Baseline Model

# In[39]:


features = ['timestamp', 'sum_user_accuracy',
       'prior_question_elapsed_time', 'prior_question_had_explanation',
       'mean_user_accuracy', 'questions_answered', 'std_user_accuracy',
       'median_user_accuracy', 'skew_user_accuracy', 'correct_answer', 'mean_accuracy',
       'question_asked', 'sum_accuracy','std_accuracy', 'median_accuracy', 'skew_accuracy',
       'right_answers', 'bundle_size', 'bundle_rignt_answers',  
       'bundle_questions_asked', 'bundle_accuracy', 'part_rignt_answers',
       'part_questions_asked', 'part_accuracy', 'type_of_count']

target = 'answered_correctly'


# In[40]:


new_train_df = new_train_df.sort_values(['user_id'])

y = new_train_df[target]

X = new_train_df[features]

del new_train_df


# In[41]:


scores = []
feature_importance = pd.DataFrame()
models = []


# In[42]:


params = {'num_leaves': 40,
          'max_depth': 4,
          'subsample':0.8,
          'objective': 'binary',
          'learning_rate': 0.001,
          "boosting_type": "gbdt",
          "metric": 'auc',
          'n_estimators': 100,
          'min_child_samples':30,
          'num_parallel_tree': 1000,
          'subsample_freq':15,
          'n_jobs':-1,
          'is_higher_better': True,
          'first_metric_only': True
         }


# In[43]:


#https://www.kaggle.com/artgor/riiid-eda-feature-engineering-and-models
folds = StratifiedKFold(n_splits=5, shuffle=False)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print(f'Fold {fold_n} started at {time.ctime()}')
    X_train, X_valid = X[features].iloc[train_index], X[features].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',
            verbose=50, early_stopping_rounds=10)
    score = max(model.evals_result_['valid_1']['auc'])
    
    models.append(model)
    scores.append(score)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    break


# In[44]:


print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


# In[45]:


feature_importance["importance"] /= 1
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');


# In[46]:


import riiideducation

env = riiideducation.make_env()


# In[47]:


iter_test = env.iter_test()


# In[48]:


for (test_df, sample_prediction_df) in iter_test:
    y_preds = []
    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')\
                        .merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')\
                        .merge(bundle_answers_df, how = 'left', on = 'bundle_id')\
                        .merge(part_answers_df, how = 'left', on = 'part')\
                        .merge(lectures_df, how='left',on='part')
    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)
    
    test_df.fillna(-1, inplace = True)
    test_df["prior_question_had_explanation_enc"] = le.fit_transform(test_df["prior_question_had_explanation"])

    for model in models:
        y_pred = model.predict_proba(test_df[features], num_iteration=model.best_iteration_)[:, 1]
        y_preds.append(y_pred)

    y_preds = sum(y_preds) / len(y_preds)
    test_df['answered_correctly'] = y_preds
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])


# In[49]:


# %%time
# params = {              'n_estimators': 100,
#                         'seed': 44,
#                         'colsample_bytree': 0.8,
#                         'subsample': 0.7,
#                         'learning_rate': 0.01,
#                         'objective': 'binary:logistic',
#                         'max_depth': 5,
#                         'num_parallel_tree': 1000,
#                         'min_child_weight': 20,
#                         'eval_metric':'auc',
#                         'gamma':0.1,
#                         'tree_method':'gpu_hist'}

# model = XGBClassifier(**params)
# eval_set = [(X_train, y_train), (X_valid, y_valid)]
# model.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=10)


# In[50]:


# # plot AUC
# results = model.evals_result()
# epochs = len(results['validation_0']['auc'])
# x_axis = range(0, epochs)
# fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(x_axis, results['validation_0']['auc'], label='Train')
# ax.plot(x_axis, results['validation_1']['auc'], label='Test')
# ax.legend()
# plt.ylabel('AUC')
# plt.title('XGBoost AUC')
# plt.show()


# In[51]:


# from xgboost import plot_importance
# fig,ax=plt.subplots(figsize=(8,5))
# plot_importance(model, color='red', height=0.5,ax=ax, importance_type='weight')
# plt.show()


# In[52]:


# from xgboost import plot_importance
# fig,ax=plt.subplots(figsize=(8,5))
# plot_importance(model, color='red', height=0.5,ax=ax, importance_type='gain')
# plt.show()


# In[ ]:




