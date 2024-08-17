#!/usr/bin/env python
# coding: utf-8

# ### Feature Engineering with part 1
# 
# My intention in this notebook is to do feature engineering without looking too much at other public kernels to avoid potential biases and risk of limiting  new creative and hopefully helpful feature ideas. This is the part 1 of the feature engineering notebooks that will be implemented.
# 
# This notebook implements overall or grouped user specific statistical features based on user activity history.
# 
# 
# **You can also skip and go to modeling kernel:** [Part 1 Modeling Notebook](https://www.kaggle.com/keremt/fastai-model-part1-regression/)

# ### Imports
# 
# We will use fastai v1

# In[1]:


from fastai.core import *
Path.read_csv = lambda o: pd.read_csv(o)
input_path = Path("/kaggle/input/data-science-bowl-2019")
pd.options.display.max_columns=200
pd.options.display.max_rows=200
input_path.ls()


# ### Read data

# In[2]:


sample_subdf = (input_path/'sample_submission.csv').read_csv()
specs_df = (input_path/"specs.csv").read_csv()
train_df = (input_path/"train.csv").read_csv()
train_labels_df = (input_path/"train_labels.csv").read_csv()
test_df = (input_path/"test.csv").read_csv()


# ### Inspect data

# In[3]:


train_labels_df.shape, train_df.shape, test_df.shape, specs_df.shape, sample_subdf.shape


# In[4]:


# example submission
sample_subdf.head(2)


# In[5]:


# training labels - how target: accuracy_group is created
train_labels_df.head(2)


# ### Train labels
# 
# Main target **accuracy_group** is highly positively correlated with **accuracy**, so we can potentially use this as a pseudo target to convert this problem into a regression problem which will allow us to have ability to feed more granular signal during supervised learning and also it will give us the ability to post process by setting thresholds to create groups from soft prediction.

# In[6]:


train_labels_df[['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].corr()


# We can actually see there is a direct rule for converting **accuracy** to **accuracy_group**.
# 
# - `y==0 -> 0`,  `y>0 & y<0.5 -> 1`, `y==0.5 -> 2`, `y==1 -> 3`
# 
# We can tune our thresholds to based on validation set to better align conversion from **predicted accuracy** to **accuracy_group**.

# In[7]:


train_labels_df.pivot_table(values= "installation_id",index="accuracy_group", columns="accuracy", aggfunc=np.count_nonzero)


# In[8]:


train_df.head(2)


# In[9]:


test_df.head(2)


# In[10]:


specs_df.head(2)


# ### What to predict?
# 
# **What is predicted:** We predict last assessment in test set that which has start code `event_code==2000`.
# 
# **Note:** During feature engineering we need to keep in mind that we can only use historical data to generate features because any data after `event_code=2000` in test set is truncated. Using global data for features won't be a problem and it will be time independent, e.g. priors, but in general for user centric features there shouldn't be any future information.
# 
# For example, we can calculate mean accuracy of a particular group or combination of groups globally without caring about time leakage since that information will also be available in test time. These type of features can be considered as priors that we feed into our model. This is sort of target encoding and even though we are allowed to use all data it should be done with caution, e.g. using out of folds, CV, etc.., to make sure that we don't overfit. See `Holdout Type` section from [h2o.ai](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html#holdout-type) for more information.

# In[11]:


# Get last assessment start for each installation_id - it should have 'event_code' == 2000, we have exactly 1000 test samples that we need predictions of
test_assessments_df = test_df.sort_values("timestamp").query("type == 'Assessment' & event_code == 2000").groupby("installation_id").tail(1).reset_index(drop=True)


# In[12]:


# event_data, installation_id, event_count, event_code, game_time is constant for any assessment start
# for extarcting similar rows we can look at event_code==2000 and type==Assessment combination for each installation_id
test_assessments_df = test_assessments_df.drop(['event_data', 'installation_id', 'event_count', 'event_code', 'game_time'],1); test_assessments_df


# In[13]:


# there is unique event_id for each assesment
test_assessments_df.pivot_table(values=None, index="event_id", columns="title", aggfunc=np.count_nonzero)['game_session']


# In[14]:


# there are common worlds among different assessments
test_assessments_df.pivot_table(values=None, index="world", columns="title", aggfunc=np.count_nonzero)['game_session']


# In[15]:


test_assessments_df.describe(include='all')


# ### Training data with labels
# 
# Let's merge train_labels to `train_df`

# In[16]:


def get_assessment_start_idxs(df): return listify(df.query("type == 'Assessment' & event_code == 2000").index)


# In[17]:


# drop installation ids without at least 1 completed assessment
_train_df = train_df[train_df.installation_id.isin((train_labels_df.installation_id).unique())].reset_index(drop=True)


# In[18]:


# join training labels to game starts by game sessions  
_trn_str_idxs = get_assessment_start_idxs(_train_df)
_label_df = _train_df.iloc[_trn_str_idxs]
_label_df = _label_df.merge(train_labels_df[['game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']], "left", on="game_session")
_label_df = _label_df[["event_id", "installation_id", 'game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']]
_label_df.head()


# In[19]:


_label_df['accuracy_group'].value_counts(dropna=False).sort_index()


# In[20]:


# join labels to train by event_id, game_session, installation_id
train_with_labels_df = _train_df.merge(_label_df, "left", on=["event_id", "game_session", "installation_id"])
train_with_labels_df['accuracy_group'].value_counts(dropna=False).sort_index()


# In[21]:


train_with_labels_df.shape


# In[22]:


# success statistics per game
(train_with_labels_df.query("type == 'Assessment'")
                     .groupby(["title", "world"])['accuracy']
                     .agg({np.mean, np.median, np.max, np.min})
                     .sort_values("mean"))


# In[23]:


def count_nonnan(l): return np.sum([0 if np.isnan(o) else 1 for o in l])


# In[24]:


# verify that all training installation ids have at least one assesment with non NaN label
assert not any(train_with_labels_df.groupby("installation_id")['accuracy'].agg(count_nonnan) == 0) 


# In[25]:


# save dataframe train with labels
train_with_labels_df.to_csv("train_with_labels.csv", index=False)


# In[26]:


# save MEM space
del _label_df
del _train_df
gc.collect()


# ### Validation: Split by `installation_id`
# 
# During modeling it's better to craete validation sets by splitting by `installation_id` rather than random. 

# ### History of a user
# 
# Let's check history of a single user to get a better understanding of events and possible feature engineering ideas. 
# 
# **Assessment Start ID:** <`installation_id`, `game_session`, `event_id`>

# In[27]:


from fastai.tabular.transform import add_datepart


# In[28]:


# set filtered and labels added df
train_df = train_with_labels_df


# In[29]:


def get_assessment_start_idxs_with_labels(df):
    "return indexes that will be used for supervised learning"
    df = df[~df.accuracy.isna()]
    return listify(df.query("type == 'Assessment' & event_code == 2000").index)


# In[30]:


def get_sorted_user_df(df, ins_id):
    "extract sorted data for a given installation id and add datetime features"
    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)
    add_datepart(_df, "timestamp", time=True)
    return _df


# In[31]:


# pick installation_id and get data until an assessment_start
rand_id = np.random.choice(train_df.installation_id)
user_df = get_sorted_user_df(train_df, rand_id)
start_idxs = get_assessment_start_idxs_with_labels(user_df)
print(f"Assessment start idxs in user df: {start_idxs}")


# In[32]:


# we would like to get and create features for each assessment start for supervised learning
str_idx = start_idxs[1]
user_assessment_df = user_df[:str_idx+1]; user_assessment_df


# ### Feature Engineering Ideas
# 
# - Generate individual features using history until an assessment start (part 1)
# - Target encoding (TODO)
# - Generate global aggregate features by groupby, independent of time (TODO)
# - Encode sequential events from `event_data` using RNNs, e.g. embeddings for different groups of `event_data` can be generated (TODO)
# - Graph based features (TODO)

# ### 1)  Static Until Assessment Start Features
# 
# Below we implement functions that will take `user_assessment_df` and spit out some feature vector for a unique assessment start which has unique id by combination of  <`installation_id`, `game_session`, `event_id`>.
# 
# ```
# Value: Time Elapsed x Count/Freq x Event Count x Game Time 
# By: Media Type x Title x Event Id x World x Event Code
# 
# ```

# In[33]:


from fastai.tabular import *
import types

stats = ["median","mean","sum","min","max"]

UNIQUE_COL_VALS = types.SimpleNamespace(
    event_ids = np.unique(train_df.event_id),
    media_types = np.unique(train_df.type),
    titles = np.unique(train_df.title),
    worlds = np.unique(train_df.world),
    event_codes = np.unique(train_df.event_code),
)


# In[34]:


def array_output(f):
    def inner(*args, **kwargs): return array(listify(f(*args, **kwargs))).flatten()
    return inner


# In[35]:


feature_funcs = []


# > #### Time Elapsed

# In[36]:


@array_output
def time_elapsed_since_hist_begin(df):
    "total time passed until assessment begin"
    return df['timestampElapsed'].max() - df['timestampElapsed'].min()

feature_funcs.append(time_elapsed_since_hist_begin)
time_elapsed_since_hist_begin(user_assessment_df)


# In[37]:


@array_output
def time_elapsed_since_each(df, types, dfcol):
    "time since last occurence of each types, if type not seen then time since history begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    last_elapsed = df['timestampElapsed'].max()
    _d = dict(df.iloc[:-1].groupby(dfcol)['timestampElapsed'].max())
    return [last_elapsed - _d[t] if t in _d else time_elapsed_since_hist_begin(df)[0] for t in types]


# In[38]:


feature_funcs.append(partial(time_elapsed_since_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(time_elapsed_since_each, types="titles", dfcol="title"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(time_elapsed_since_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_codes", dfcol="event_code"))


# > #### Count/Freq

# In[39]:


def countfreqhist(df, types, dfcol, freq=False):
    "count or freq of types until assessment begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    _d = dict(df[dfcol].value_counts(normalize=(True if freq else False)))
    return [_d[t] if t in _d else 0 for t in types]


# In[40]:


feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=False))
feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=True))

feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=False))
feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=True))

feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=False))
feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=True))


# > #### Event Count

# In[41]:


@array_output
def overall_event_count_stats(df):
    "overall event count stats until assessment begin"
    return df['event_count'].agg(stats)

feature_funcs.append(overall_event_count_stats)
overall_event_count_stats(user_assessment_df)


# In[42]:


@array_output
def event_count_stats_each(df, types, dfcol):
    "event count stats per media types until assessment begin, all zeros if media type missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['event_count'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]


# In[43]:


feature_funcs.append(partial(event_count_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(event_count_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(event_count_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(event_count_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(event_count_stats_each, types="event_codes", dfcol="event_code"))


# > #### Game Time

# In[44]:


@array_output
def overall_session_game_time_stats(df):
    "overall session game time stats until assessment begin"
    return df['game_time'].agg(stats)

feature_funcs.append(overall_session_game_time_stats)
overall_session_game_time_stats(user_assessment_df)


# In[45]:


@array_output
def session_game_time_stats_each(df, types, dfcol):
    "session game time stats per media types until assessment begin, all zeros if missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['game_time'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]


# In[46]:


feature_funcs.append(partial(session_game_time_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(session_game_time_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(session_game_time_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_codes", dfcol="event_code"))


# In[47]:


sample_features = np.concatenate([f(user_assessment_df) for f in feature_funcs]); sample_features.shape


# ### Compute all features for train and test
# 
# These are mostly static features per assessment start which we can compute and save for later use

# In[48]:


def get_test_assessment_start_idxs(df): 
    return list(df.sort_values("timestamp")
                  .query("type == 'Assessment' & event_code == 2000")
                  .groupby("installation_id").tail(1).index)


# In[49]:


# trn_str_idxs = get_assessment_start_idxs_with_labels(train_with_labels_df)
# test_str_idxs = get_test_assessment_start_idxs(test_df)


# In[50]:


# Get training features
def get_train_feats_row(ins_id, i):
    "get all assessment start features for an installation id"
    rows = [] # collect rows with features for each assessment start
    user_df = get_sorted_user_df(train_with_labels_df, ins_id)
    start_idxs = get_assessment_start_idxs_with_labels(user_df); start_idxs
    for idx in start_idxs:
        assessment_row = user_df.iloc[idx]
        _df = user_df[:idx+1]
        row_feats = np.concatenate([f(_df) for f in feature_funcs])
        feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])
        row = pd.concat([assessment_row, feat_row])
        rows.append(row)
    return rows


# In[51]:


# # compute static features for train assessment start
# installation_ids = train_with_labels_df.installation_id.unique()
# res = parallel(get_train_feats_row, (installation_ids))
# train_with_features_df = pd.concat([row for rows in res for row in rows],1).T


# In[52]:


# train_with_features_df.head()


# In[53]:


# train_with_features_df.to_csv("train_with_features_part1.csv")


# In[54]:


def get_test_feats_row(idx, i):
    "get all faeatures by an installation start idx"
    ins_id = test_df.loc[idx, "installation_id"]
    _df = get_sorted_user_df(test_df, ins_id)
    assessment_row = _df.iloc[-1]
    row_feats = np.concatenate([f(_df) for f in feature_funcs])
    feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])
    row = pd.concat([assessment_row, feat_row])
    return row


# In[55]:


# # compute static features for test assessment start and save 
# start_idxs = get_test_assessment_start_idxs(test_df)
# res = parallel(get_test_feats_row, start_idxs)
# test_with_features_df = pd.concat(res,1).T


# In[56]:


# test_with_features_df.head()


# In[57]:


# test_with_features_df.to_csv("test_with_features_part1.csv")


# **Note:**
# Kernel doesn't complete within given time although it completes in my local laptop within just 1 hour. So I ran everything locally and registered necessary data as a dataset here: https://www.kaggle.com/keremt/dsbowl2019-feng-part1  

# ### Next steps for part 2

# ### - Target encoding

# ### - Event data features
# 
# - Each `event_id` have a single `event_code` 
# - `event_code` can be shared across games, e.g. `event_code==2000` and there are more eventhough content is different 
# - `event_id` and `event_code` has a single `title` except for `event_code==2000` which indicates start and has 20 `title`
# - There are 386 unique `event_id`
# - Same `event_id` can have different descriptions: 
# 
# ```
# {"description":"Ah! See the pans move? Now put a bowl on the other pan.","identifier":"Cleo_SeePansMove,Cleo_PutBowlOtherPan","media_type":"audio","total_duration":8570,"round":1,"event_count":12,"game_time":19061,"event_code":3010} 8d7e386c Happy Camel
# 
# {"description":"When one side tips down… that bowl is heavier. Now think… which bowl has the toy inside?","identifier":"Cleo_TipsHeavier,Cleo_ThinkWhich","media_type":"audio","total_duration":9192,"round":1,"event_count":16,"game_time":23378,"event_code":3010} 8d7e386c Happy Camel
# ```
# 
# - Same description can be seen in different `event_id`: 
# 
# ```
# {"description":"Epidermis' toy is in one of these bowls of camel food! But which one? We shall find the toys in no time thanks to my pan balance! This amazing device for measuring weight. First, move a bowl to one of the pans.","identifier":"Cleo_EpidermisToy,Cleo_PanBalance,Cleo_MoveBowlToPan","media_type":"audio","duration":7317,"round":1,"event_count":11,"game_time":19061,"event_code":3110} 69fdac0a Happy Camel
# 
# {"description":"Epidermis' toy is in one of these bowls of camel food! But which one? We shall find the toys in no time thanks to my pan balance! This amazing device for measuring weight. First, move a bowl to one of the pans.","identifier":"Cleo_EpidermisToy,Cleo_PanBalance,Cleo_MoveBowlToPan","media_type":"audio","total_duration":20920,"round":1,"event_count":6,"game_time":11744,"event_code":3010} 8d7e386c Happy Camel
# ![](http://)```

# ### end
