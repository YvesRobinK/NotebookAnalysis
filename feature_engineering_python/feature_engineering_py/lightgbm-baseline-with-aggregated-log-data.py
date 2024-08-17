#!/usr/bin/env python
# coding: utf-8

# ### **<span style="color: #009933;">💳 Notebook Workflow At a Glance</span>**

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Notebook Overview:</u></b><br>
#     
# * <i> This is log data, which means each session has multiple events. For modeling, it's important to decide analysis unit.For this, please refer to basic submission demo. https://www.kaggle.com/code/philculliton/basic-submission-demo </i><br>
#     
# * <i> For each (session_id) _(question #), you are predicting the correct column, identifying whether you believe the user for this particular session will answer this question correctly, using only the previous information for the session.</i><br>
#     
# * <i> In this case, you can treat each session_id as one user. Simply speaking, I will look at all the log data of each session and predict whether user could answer each questions correctly.</i><br>
#     
# * <i> I assumed that user's behavior on each game level is important. Therefore, I will aggregate the data by each sessions' (users) game level. Also, I will make dummies of event_name variables so that model could learn more about each event. </i><br>
#     
# * <i> In sum, we build three models. 
#     - learn log behavior of level group 0-4, predict questions 1-3.
#     - learn log behavior of level group 5-12, predict questions 4~13
#     - learn log behavior of level group 13-22, predict questions 14~18.
#     </i><br>
#     
# * <i> In terms of modeling code, original work is done by https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664 ; Also, When ti comes to feature engineering, I modified part of this notebook. </i><br>
#    
# </div>

# ![image.png](attachment:35b0fc1f-246c-4790-ba55-317c1bd850d9.png)

# #### **<span style="color: #009933;">Exploratory Data Analysis</span>**

# - session_id - the ID of the session the event took place in
# - index - the index of the event for the session
# - elapsed_time - how much time has passed (in milliseconds) between the start of the session and when the event was recorded
# - event_name - the name of the event type
# - name - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)
# - level - what level of the game the event occurred in (0 to 22)
# - page - the page number of the event (only for notebook-related events)
# - room_coor_x - the coordinates of the click in reference to the in-game room (only for click events)
# - room_coor_y - the coordinates of the click in reference to the in-game room (only for click events)
# - screen_coor_x - the coordinates of the click in reference to the player’s screen (only for click events)
# - screen_coor_y - the coordinates of the click in reference to the player’s screen (only for click events)
# - hover_duration - how long (in milliseconds) the hover happened for (only for hover events)
# - text - the text the player sees during this event
# - fqid - the fully qualified ID of the event
# - room_fqid - the fully qualified ID of the room the event took place in
# - text_fqid - the fully qualified ID of the
# - fullscreen - whether the player is in fullscreen mode
# - hq - whether the game is in high-quality
# - music - whether the game music is on or off
# - level_group - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>EDA summary:</u></b><br>
#     
# * <i> There are <b><u>21</u></b> columns in total - <b><u>20</u></b> X variables and <b><u>1</u></b> Y variable(correct) /1 extra variables (session_id) </i><br>
# * <i> Some variables have missing data. While variables has missing data only.</i><br>
# * <i> In terms of data type,there are 7 object type ,3 int type and 9 float64 type. </i><br>
# * <i> There are only 3 unique session_id on test dataset. Therefore, we need to predict 3 x 18(questions) = 54 rows.</i><br>
# </div>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


from matplotlib import ticker
import time
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# In[2]:


# Reduce Memory Usage
# reference : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 @ARJANGROEN

def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df


# In[3]:


train_df = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train.csv')
train_df.info()


# In[4]:


train_df = reduce_memory_usage(train_df)
train_df.info()


# In[5]:


import gc
gc.collect()


# In[6]:


train_label = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')
train_label = reduce_memory_usage(train_label)
train_label['session'] = train_label.session_id.apply(lambda x: int(x.split('_')[0]) )
train_label['q'] = train_label.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
print( 'shape of label dataset is:',train_label.shape )


# In[7]:


train_label.head()


# In[8]:


gc.collect()


# In[9]:


def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# In[10]:


summary_table = summary(train_df)
summary_table


# #### <span style="color:#339966;"> I will skip detailed EDA process.
#     I assumes:
#     - text matters
#     - level of game matters
#     - event type matters
#     - elapsed time matters\
#     
#     Also assumes:
#     - 'page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
#        'hover_duration', 'text_fqid', 'fullscreen', 'hq',
#        'music', 'level_group'
#     these data is not useful. (As there are too many missing values....)
#     For coordinates variables, I am not sure how to leverage it due to lack of domain knowledge.
#     
# </span>
# 
# #### <span style="color:#339966;"> I will try to aggregate data by user level (session level).</span>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Feature engineering:</u></b><br>
#     
# * <i> Very smart and useful function from https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664   </i><br>
# * <i> I assumed event is important factor for prediction. Therefore, I made dummies of event_name.</i><br>
# * <i> I only added sum, count, mean values. You can create more variables thru EDA or domain knowledge. </i><br>
# * <i> We will train with 16 features and train with 11779 users info</i><br>
# </div>

# In[11]:


#create dummies
just_dummies = pd.get_dummies(train_df['event_name'])

train_df = pd.concat([train_df, just_dummies], axis=1)


# In[14]:


train_df.head()


# In[15]:


train_df['event_name'].value_counts()


# In[16]:


count_var = ['event_name', 'fqid','room_fqid', 'text']
mean_var = ['elapsed_time','level']
event_var = ['navigate_click','person_click','cutscene_click','object_click','map_hover','notification_click',
            'map_click','observation_click','checkpoint','elapsed_time']


# In[17]:


# reference: https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook
def feature_engineer(train):
    dfs = []
    for c in count_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in mean_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in event_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('sum')
        tmp.name = tmp.name + '_sum'
        dfs.append(tmp)
    df = pd.concat(dfs,axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    df = df.set_index('session_id')
    return df


# In[18]:


df_tr = feature_engineer(train_df)
print( df_tr.shape )


# In[22]:


df_tr.head()


# In[20]:


gc.collect()


# In[21]:


#check data type
df_tr.dtypes


# In[23]:


FEATURES = [c for c in df_tr.columns if c != 'level_group']
print('We will train with', len(FEATURES) ,'features')
ALL_USERS = df_tr.index.unique()
print('We will train with', len(ALL_USERS) ,'users info')


# 

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Modeling:</u></b><br>
#     
# * <i> Very smart and useful function from https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664   </i><br>
# * <i> I used LGBM classifier instead of RF from original code.</i><br>
# * <i> Choose best threhold based on F1 Score, which is same with the original code. </i><br>
# * <i> Do not forget that you need to transform test dataset from API.</i><br>
# </div>

# In[24]:


gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)
models = {}

# COMPUTE CV SCORE WITH 5 GROUP K FOLD
for i, (train_index, test_index) in enumerate(gkf.split(X=df_tr, groups=df_tr.index)):
    print('#'*25)
    print('### Fold',i+1)
    print('#'*25)
    
    
    lgb_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate': 0.002,
    'max_depth': 6,
    'num_iterations': 1000}
    
    
    # ITERATE THRU QUESTIONS 1 THRU 18
    for t in range(1,19):
        print(t,', ',end='')
        
        # USE THIS TRAIN DATA WITH THESE QUESTIONS
        if t<=3: grp = '0-4'
        elif t<=13: grp = '5-12'
        elif t<=22: grp = '13-22'
            
        # TRAIN DATA
        train_x = df_tr.iloc[train_index]
        train_x = train_x.loc[train_x.level_group == grp]
        train_users = train_x.index.values
        train_y = train_label.loc[train_label.q==t].set_index('session').loc[train_users]
        
        # VALID DATA
        valid_x = df_tr.iloc[test_index]
        valid_x = valid_x.loc[valid_x.level_group == grp]
        valid_users = valid_x.index.values
        valid_y = train_label.loc[train_label.q==t].set_index('session').loc[valid_users]
        
        # TRAIN MODEL
        clf =  LGBMClassifier(**lgb_params)
        clf.fit(train_x[FEATURES].astype('float32'), train_y['correct'])
        
        # SAVE MODEL, PREDICT VALID OOF
        models[f'{grp}_{t}'] = clf
        oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES].astype('float32'))[:,1]
        
    print()


# In[25]:


# PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
true = oof.copy()
for k in range(18):
    # GET TRUE LABELS
    tmp = train_label.loc[train_label.q == k+1].set_index('session').loc[ALL_USERS]
    true[k] = tmp.correct.values


# In[26]:


# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (oof.values.reshape((-1))>threshold).astype('int')
    m = f1_score(true.values.reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold


# In[27]:


import matplotlib.pyplot as plt

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()


# In[28]:


print('When using optimal threshold...')
for k in range(18):
        
    # COMPUTE F1 SCORE PER QUESTION
    m = f1_score(true[k].values, (oof[k].values>best_threshold).astype('int'), average='macro')
    print(f'Q{k}: F1 =',m)
    
# COMPUTE F1 SCORE OVERALL
m = f1_score(true.values.reshape((-1)), (oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')
print('==> Overall F1 =',m)


# In[29]:


import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[30]:


limits = {'0-4':(1,4), '5-12':(4,14), '13-22':(14,19)}

for (sample_submission, test) in iter_test:
    
    dummies = pd.get_dummies(test['event_name'])
    test = pd.concat([test, dummies], axis=1)
    df = feature_engineer(test)
    grp = test.level_group.values[0]
    a,b = limits[grp]
    for t in range(a,b):
        clf = models[f'{grp}_{t}']
        p = clf.predict_proba(df[FEATURES].astype('float32'))[:,1]
        mask = sample_submission.session_id.str.contains(f'q{t}')
        sample_submission.loc[mask,'correct'] = int(p.item()>best_threshold)
    
    env.predict(sample_submission)


# In[33]:


submit = pd.read_csv('submission.csv')


# In[40]:


submit #my daily submission is over 5...so couldn't see the result yet;


# In[ ]:





# In[ ]:





# 
