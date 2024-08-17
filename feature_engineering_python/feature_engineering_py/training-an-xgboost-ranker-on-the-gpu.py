#!/usr/bin/env python
# coding: utf-8

# In this notebook we will train an `XGBoost Ranker` on the GPU and perform prediction.
# 
# Training with varied architectures and ensembling (please see [💡 [2 methods] How-to ensemble predictions 🏅🏅🏅](https://www.kaggle.com/code/radek1/2-methods-how-to-ensemble-predictions) for a tutorial on ensembling) can offer you a significant jump on the LB!
# 
# Training with `XGBoost` however offers more additional advantages. In comparison to `LGBM`, `XGBoost` allows you to train with the following objectives (`LGBM` gives you access to a single loss only for ranking, training with different objectives is a great way of improving your ensemble!):
# * `rank:pairwise`
# * `rank:ndcg`
# * `rank:map`
# 
# On top of that, we will train on the GPU! 🔥 GPU can offer a significant speed-up. You can train more and bigger models in a shorter amount of time. However, when training on the GPU with large amounts of tabular data, you can easily run into problems (how to load the data onto the GPU for processing in chunks, how to manage memory).
# 
# As we want to focus on feature engineering and training lets offload all the low level, tedious considerations to the `Merlin Framework`!
# 
# In this notebook, we will introduce the entire pipeline. We will preprocess our data on the GPU using a library specifically designed for tabular data preprocessing, `NVTabular`. We will then proceed to train our `XGBoost` model with `Merlin Models`. In the background  we will leverage `dask_cuda` and distributed training to optimize the use of available GPU RAM, but we will let the libraries handle all that! No additional configuration will be required from us.
# 
# Let's get started!
# 
# ## Other resources you might find useful:
# 
# * [💡 [2 methods] How-to ensemble predictions 🏅🏅🏅](https://www.kaggle.com/code/radek1/2-methods-how-to-ensemble-predictions)
# * [co-visitation matrix - simplified, imprvd logic 🔥](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
# * [💡 Word2Vec How-to [training and submission]🚀🚀🚀](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission)
# * [local validation tracks public LB perfecty -- here is the setup](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
# * [💡 For my friends from Twitter and LinkedIn -- here is how to dive into this competition 🐳](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368560)
# * [Full dataset processed to CSV/parquet files with optimized memory footprint](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)

# # Libraries installation
# 
# We will need a couple of libraries that do not come preinstalled on the Kaggle VM. Let's install them here.

# In[1]:


get_ipython().run_cell_magic('capture', '', '\n!pip install nvtabular==1.3.3 merlin-models polars merlin-core==v0.4.0 dask_cuda\n')


# # Data Processing

# We will briefly preprocess our data using polars. After that step, we will hand it over to `NVTabular` to tag our data (so that our model will know where to find the information it needs for training).

# In[2]:


from nvtabular import *
from merlin.schema.tags import Tags
import polars as pl
import xgboost as xgb

from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost
from nvtabular.ops import AddTags


# In[3]:


train = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test.parquet')
train_labels = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test_labels.parquet')

def add_action_num_reverse_chrono(df):
    return df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
    ])

def add_session_length(df):
    return df.select([
        pl.col('*'),
        pl.col('session').count().over('session').alias('session_length')
    ])

def add_log_recency_score(df):
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    return df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)

def add_type_weighted_log_recency_score(df):
    type_weights = {0:1, 1:6, 2:3}
    type_weighted_log_recency_score = pl.Series(df['type'].apply(lambda x: type_weights[x]) * df['log_recency_score'])
    return df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df

pipeline = [add_action_num_reverse_chrono, add_session_length, add_log_recency_score, add_type_weighted_log_recency_score]

train = apply(train, pipeline)

type2id = {"clicks": 0, "carts": 1, "orders": 2}

train_labels = train_labels.explode('ground_truth').with_columns([
    pl.col('ground_truth').alias('aid'),
    pl.col('type').apply(lambda x: type2id[x])
])[['session', 'type', 'aid']]

train_labels = train_labels.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32)
])

train_labels = train_labels.with_column(pl.lit(1).alias('gt'))

train = train.join(train_labels, how='left', on=['session', 'type', 'aid']).with_column(pl.col('gt').fill_null(0))


# Let us now define the preprocessing steps we would like to apply to our data.

# In[4]:


train_ds = Dataset(train.to_pandas())

feature_cols = ['aid', 'type','action_num_reverse_chrono', 'session_length', 'log_recency_score', 'type_weighted_log_recency_score']
target = ['gt'] >> AddTags([Tags.TARGET])
qid_column = ['session'] >>  AddTags([Tags.USER_ID]) # we will use sessions as a query ID column
                                                     # in XGBoost parlance this a way of grouping together for training
                                                     # when training with LGBM we had to calculate session lengths, but here the model does all the work for us!


# Having defined the preprocessing steps, we can now apply them to our data. The preprocessing is going to run on the GPU!

# In[5]:


wf = Workflow(feature_cols + target + qid_column)
train_processed = wf.fit_transform(train_ds)


# # Model training

# In[6]:


ranker = XGBoost(train_processed.schema, objective='rank:pairwise')


# The `Distributed` context manager will start a dask cudf cluster of us. A Dask cluster will be able to better manage memory usage for us. Normally, setting it up would be quite tedious -- here, we get all the benefits with a single line of Python code!

# In[7]:


# version mismatch doesn't result in a loss of functionality here for us
# it stems from the versions of libraries that the Kaggle vm comes preinstalled with

with Distributed():
    ranker.fit(train_processed)


# We have now trained our model! Let's predict on test!

# # Predict on test data

# Let's load our test set, process it and predict on it.

# In[8]:


test = pl.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
test = apply(test, pipeline)
test_ds = Dataset(test.to_pandas())

wf = wf.remove_inputs(['gt']) # we don't have ground truth information in test!

test_ds_transformed = wf.transform(test_ds)


# Let's output the predictions

# In[9]:


test_preds = ranker.booster.predict(xgb.DMatrix(test_ds_transformed.compute()))


# # Create submission

# In[10]:


test = test.with_columns(pl.Series(name='score', values=test_preds))
test_predictions = test.sort(['session', 'score'], reverse=True).groupby('session').agg([
    pl.col('aid').limit(20).list()
])


# In[11]:


session_types = []
labels = []

for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid'].to_numpy()):
    l = ' '.join(str(p) for p in preds)
    for session_type in ['clicks', 'carts', 'orders']:
        labels.append(l)
        session_types.append(f'{session}_{session_type}')


# In[12]:


submission = pl.DataFrame({'session_type': session_types, 'labels': labels})
submission.write_csv('submission.csv')

