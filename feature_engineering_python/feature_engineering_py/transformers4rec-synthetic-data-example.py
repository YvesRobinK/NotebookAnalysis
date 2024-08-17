#!/usr/bin/env python
# coding: utf-8

# **IMPORTANT NOTE: This example demonstrates how to run the synthetic data example from the [transformers4rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) library. Currently, the competition data is not utilized/replaced by the synthetic data.**
# 
# - https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based
# 
# Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation and can work with PyTorch.
# 
# It works as a bridge between NLP and recommender systems by integrating with one the most popular NLP frameworks [HuggingFace Transformers](https://github.com/huggingface/transformers), making state-of-the-art Transformer architectures available for RecSys researchers and industry practitioners.
# 
# <img src="https://raw.githubusercontent.com/NVIDIA-Merlin/Transformers4Rec/main/_images/sequential_rec.png" alt="Sequential and Session-based recommendation with Transformers4Rec" style="width:800px;display:block;margin-left:auto;margin-right:auto;"/><br>
# <div style="text-align: center; margin: 20pt">
#   <figcaption style="font-style: italic;">Sequential and Session-based recommendation with Transformers4Rec</figcaption>
# </div>
# 
# Transformers4Rec supports multiple input features and provides configurable building blocks that can be easily combined for custom architectures.
# 
# You can build a fully GPU-accelerated pipeline for sequential and session-based recommendation with Transformers4Rec and its smooth integration with other components of [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin):  [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) for preprocessing and [Triton Inference Server](https://github.com/triton-inference-server/server).
# 
# And in their examples directory, you can find the following tutorials/examples:
# - [End-to-end session-based recommendation](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/end-to-end-session-based)
# - [Tutorial - End-to-End Session-Based Recommendation on GPU](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial)

# In[1]:


get_ipython().system('pip install -q transformers4rec[pytorch,nvtabular]')
get_ipython().system('pip install -q -U nvtabular==1.3.3')


# # ETL with NVTabular

# ### Import required libraries
# 

# In[2]:


import os
import glob

import numpy as np
import pandas as pd

import cudf
import cupy as cp
import nvtabular as nvt
from nvtabular.ops import *
from merlin.schema.tags import Tags


# ### Define Input/Output Path

# In[3]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "data/")


# ### Create a Synthetic Input Data
# 

# In[4]:


NUM_ROWS = 1000000
long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., NUM_ROWS).astype(np.int32), 1, 50000)

# generate random item interaction features 
df = pd.DataFrame(np.random.randint(70000, 90000, NUM_ROWS), columns=['session_id'])
df['item_id'] = long_tailed_item_distribution

# generate category mapping for each item-id
df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
df['timestamp/age_days'] = np.random.uniform(0, 1, NUM_ROWS)
df['timestamp/weekday/sin']= np.random.uniform(0, 1, NUM_ROWS)

# generate day mapping for each session 
map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
df['day'] =  df.session_id.map(map_day)


# In[5]:


df.head()


# ### Feature Engineering with NVTabular
# 

# In[6]:


# Categorify categorical features
categ_feats = ['session_id', 'item_id', 'category'] >> nvt.ops.Categorify(start_index=1)

# Define Groupby Workflow
groupby_feats = categ_feats + ['day', 'timestamp/age_days', 'timestamp/weekday/sin']

# Groups interaction features by session and sorted by timestamp
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    aggs={
        "item_id": ["list", "count"],
        "category": ["list"],     
        "day": ["first"],
        "timestamp/age_days": ["list"],
        'timestamp/weekday/sin': ["list"],
        },
    name_sep="-")

# Select and truncate the sequential features
sequence_features_truncated = (groupby_features['category-list']) >> nvt.ops.ListSlice(0,20) >> nvt.ops.Rename(postfix = '_trim')

sequence_features_truncated_item = (
    groupby_features['item_id-list']
    >> nvt.ops.ListSlice(0,20) 
    >> nvt.ops.Rename(postfix = '_trim')
    >> TagAsItemID()
)  
sequence_features_truncated_cont = (
    groupby_features['timestamp/age_days-list', 'timestamp/weekday/sin-list'] 
    >> nvt.ops.ListSlice(0,20) 
    >> nvt.ops.Rename(postfix = '_trim')
    >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)

# Filter out sessions with length 1 (not valid for next-item prediction training and evaluation)
MINIMUM_SESSION_LENGTH = 2
selected_features = (
    groupby_features['item_id-count', 'day-first', 'session_id'] + 
    sequence_features_truncated_item +
    sequence_features_truncated + 
    sequence_features_truncated_cont
)
    
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH)


workflow = nvt.Workflow(filtered_sessions)
dataset = nvt.Dataset(df, cpu=False)
# Generating statistics for the features
workflow.fit(dataset)
# Applying the preprocessing and returning an NVTabular dataset
sessions_ds = workflow.transform(dataset)
# Converting the NVTabular dataset to a Dask cuDF dataframe (`to_ddf()`) and then to cuDF dataframe (`.compute()`)
sessions_gdf = sessions_ds.to_ddf().compute()


# In[7]:


sessions_gdf.head(3)


# In[8]:


workflow.save('workflow_etl')


# In[9]:


workflow.fit_transform(dataset).to_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt"))


# ### Export pre-processed data by day
# 

# In[10]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR",os.path.join(INPUT_DATA_DIR, "sessions_by_day"))
get_ipython().system('mkdir -p $OUTPUT_DIR')


# In[11]:


from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_DIR,
                       partition_col='day-first',
                       timestamp_col='session_id', 
                      )


# ### Checking the preprocessed outputs
# 

# In[12]:


TRAIN_PATHS = sorted(glob.glob(os.path.join(OUTPUT_DIR, "1", "train.parquet")))


# In[13]:


gdf = cudf.read_parquet(TRAIN_PATHS[0])
gdf


# # Session-based Recommendation with XLNET
# 

# In[14]:


import os
import rich
import pkg_resources
rich.__version__ = pkg_resources.get_distribution("rich").version

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
import torch 

from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory


# ### Set the schema object
# 

# In[15]:


from merlin_standard_lib import Schema
SCHEMA_PATH = os.environ.get("INPUT_SCHEMA_PATH", "data/processed_nvt/schema.pbtxt")
schema = Schema().from_proto_text(SCHEMA_PATH)


# In[16]:


get_ipython().system('head -20 $SCHEMA_PATH')


# In[17]:


# You can select a subset of features for training
schema = schema.select_by_name(['item_id-list_trim', 
                                'category-list_trim', 
                                'timestamp/weekday/sin-list_trim',
                                'timestamp/age_days-list_trim'])


# ### Define the sequential input module
# 

# In[18]:


inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="mlm",
)


# ### Define the Transformer Block
# 

# In[19]:


# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Defines the evaluation top-N metrics and the cut-offs
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

# Define a head related to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                              metrics=metrics),
    inputs=inputs,
)

# Get the end-to-end Model class 
model = tr.Model(head)


# ### Set Training arguments
# 

# In[20]:


from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
# Set hyperparameters for training 
train_args = T4RecTrainingArguments(data_loader_engine='nvtabular', 
                                    dataloader_drop_last = True,
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = 128, 
                                    per_device_eval_batch_size = 32,
                                    output_dir = "./tmp", 
                                    learning_rate=0.0005,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=5,
                                    max_sequence_length=20, 
                                    report_to = [],
                                    logging_steps=50,
                                    no_cuda=False)


# ### Daily Fine-Tuning: Training over a time window
# 

# In[21]:


# Instantiate the T4Rec Trainer, which manages training and evaluation for the PyTorch API
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)


# In[22]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{INPUT_DATA_DIR}/sessions_by_day")


# In[23]:


get_ipython().run_cell_magic('time', '', 'start_time_window_index = 1\nfinal_time_window_index = 7\n#Iterating over days of one week\nfor time_index in range(start_time_window_index, final_time_window_index):\n    # Set data \n    time_index_train = time_index\n    time_index_eval = time_index + 1\n    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))\n    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))\n    print(train_paths)\n    \n    # Train on day related to time_index \n    print(\'*\'*20)\n    print("Launch training for day %s are:" %time_index)\n    print(\'*\'*20 + \'\\n\')\n    trainer.train_dataset_or_path = train_paths\n    trainer.reset_lr_scheduler()\n    trainer.train()\n    trainer.state.global_step +=1\n    print(\'finished\')\n    \n    # Evaluate on the following day\n    trainer.eval_dataset_or_path = eval_paths\n    train_metrics = trainer.evaluate(metric_key_prefix=\'eval\')\n    print(\'*\'*20)\n    print("Eval results for day %s are:\\t" %time_index_eval)\n    print(\'\\n\' + \'*\'*20 + \'\\n\')\n    for key in sorted(train_metrics.keys()):\n        print(" %s = %s" % (key, str(train_metrics[key]))) \n    wipe_memory()\n')


# ### Saves the model
# 

# In[24]:


trainer._save_model_and_checkpoint(save_model_class=True)


# ### Reloads the model
# 

# In[25]:


trainer.load_model_trainer_states_from_checkpoint('./tmp/checkpoint-%s'%trainer.state.global_step)


# ### Re-compute eval metrics of validation data
# 

# In[26]:


eval_data_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))


# In[27]:


# set new data from day 7
eval_metrics = trainer.evaluate(eval_dataset=eval_data_paths, metric_key_prefix='eval')
for key in sorted(eval_metrics.keys()):
    print("  %s = %s" % (key, str(eval_metrics[key])))


# In[ ]:




