#!/usr/bin/env python
# coding: utf-8

# # CommonLit - Evaluate Student Summaries Dataset with TensorFlow Decision Forests

# This notebook walks you through how to train a baseline Random Forest model using TensorFlow Decision Forests on the **CommonLit - Evaluate Student Summaries** dataset made available for this competition.
# 
# Roughly, the code will look as follows:
# 
# ```
# import tensorflow_decision_forests as tfdf
# import pandas as pd
# 
# dataset = pd.read_csv("project/dataset.csv")
# tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")
# 
# model = tfdf.keras.RandomForestModel()
# model.fit(tf_dataset)
# 
# print(model.summary())
# ```
# 
# Decision Forests are a family of tree-based models including Random Forests and Gradient Boosted Trees. They are the best place to start when working with tabular data, and will often outperform (or provide a strong baseline) before you begin experimenting with neural networks.

# # Import the libraries

# In[1]:


import string
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)


# # Load the Dataset

# ### Load the prompt csv

# In[3]:


df_train_prompt = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv')
print("Full prompt train dataset shape is {}".format(df_train_prompt.shape))


# The data is composed of 4 columns and 4 entries. We can see all 4 dimensions of our dataset by using the following code:

# In[4]:


df_train_prompt.head()


# ### Load the summaries csv

# In[5]:


df_train_summaries = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv')
print("Full summaries train dataset shape is {}".format(df_train_summaries.shape))


# The data is composed of 5 columns and 7165 entries. We can see all 5 dimensions of our dataset by printing out the first 5 entries using the following code:

# In[6]:


df_train_summaries.head()


# ### Combine both summaries and prompt csv's based on prompt id

# In[7]:


df_train = df_train_summaries.merge(df_train_prompt, on='prompt_id')
print("Full summaries train dataset shape is {}".format(df_train.shape))


# The data is composed of 8 columns and 7165 entries. We can see all 8 dimensions of our dataset by printing out the first 5 entries using the following code:

# In[8]:


df_train.head()


# # Quick basic dataset exploration

# In[9]:


df_train.describe()


# In[10]:


df_train.info()


# # Label data distribution

# In[11]:


plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df_train, x='content')
plt.subplot(1, 2, 2)
sns.histplot(data=df_train, x='wording')
plt.show()


# # LLM vs Non-LLM solutions
# 
# In this notebook, we won't be using an LLM model to embedd the text features. Instead we will calculate some numeric features from the text features like token count, length etc. Using these numeric features, we can create tabular data that can be used train our Model. This approach shows that sometimes non-LLM solutions can also give good results.   

# # Preprocess the data

# In[12]:


# Reference: https://www.kaggle.com/code/sercanyesiloz/commonlit-tf-idf-xgb-baseline#4.-Feature-Engineering

# A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, 
# both when indexing entries for searching and when retrieving them as the result of a search query.
# Count the stop words in the text.
def count_stopwords(text: str) -> int:
    stopword_list = set(stopwords.words('english'))
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopword_list)
    return stopwords_count

# Count the punctuations in the text.
# punctuation_set -> !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

# Count the digits in the text.
def count_numbers(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    numbers_count = len(numbers)
    return numbers_count

# This function applies all the above preprocessing functions on a text feature.
def feature_engineer(dataframe: pd.DataFrame, feature: str = 'text') -> pd.DataFrame:
    dataframe[f'{feature}_word_cnt'] = dataframe[feature].apply(lambda x: len(x.split(' ')))
    dataframe[f'{feature}_length'] = dataframe[feature].apply(lambda x: len(x))
    dataframe[f'{feature}_stopword_cnt'] = dataframe[feature].apply(lambda x: count_stopwords(x))
    dataframe[f'{feature}_punct_cnt'] = dataframe[feature].apply(lambda x: count_punctuation(x))
    dataframe[f'{feature}_number_cnt'] = dataframe[feature].apply(lambda x: count_numbers(x))
    return dataframe


# In[13]:


preprocessed_df = feature_engineer(df_train)
print("Full summaries train dataset shape is {}".format(preprocessed_df.shape))


# The data is composed of 13 columns and 7165 entries. We can see all 13 dimensions of our dataset by printing out the first 5 entries using the following code:

# In[14]:


preprocessed_df.head()


# In[15]:


preprocessed_df.describe()


# ## Extract feature columns

# In[16]:


FEATURE_COLUMNS = preprocessed_df.drop(columns = ['student_id', 'prompt_id', 'text', 'prompt_question', 
                                           'prompt_title', 'prompt_text', 'content', 'wording'], axis = 1).columns.to_list()


# In[17]:


FEATURE_COLUMNS


# ## Plot feature columns

# In[18]:


figure, axis = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.25, wspace=0.3)

for i, column_name in enumerate(FEATURE_COLUMNS):
    row = i//2
    col = i % 2
    bp = sns.barplot(ax=axis[row, col], x=preprocessed_df['student_id'], y=preprocessed_df[column_name])
    bp.set(xticklabels=[])
    axis[row, col].set_title(column_name)
axis[2, 1].set_visible(False)
plt.show()


# Now let us split the dataset into training and testing datasets:

# In[19]:


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(preprocessed_df)
train_ds_pd.shape, valid_ds_pd.shape


# In[20]:


train_ds_pd.head()


# There's one more step required before we can train the model. We need to convert the datatset from Pandas format (`pd.DataFrame`) into TensorFlow Datasets format (`tf.data.Dataset`).
# 
# [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) is a high performance data loading library which is helpful when training neural networks with accelerators like GPUs and TPUs.
# 
# By default the Random Forest Model is configured to train classification tasks. Since this is a regression problem, we will specify the type of the task (`tfdf.keras.Task.REGRESSION`) as a parameter here.

# In[21]:


# `content` label datatset columns
FEATURE_CONTENT = FEATURE_COLUMNS + ['content']

# `wording` label datatset columns
FEATURE_WORDING = FEATURE_COLUMNS + ['wording']

# Convert dataframes to corresponding datasets
content_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd[FEATURE_CONTENT], label='content', task = tfdf.keras.Task.REGRESSION)
wording_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd[FEATURE_WORDING], label='wording', task = tfdf.keras.Task.REGRESSION)


# # Select a Model
# 
# There are several tree-based models for you to choose from.
# 
# * RandomForestModel
# * GradientBoostedTreesModel
# * CartModel
# * DistributedGradientBoostedTreesModel
# 
# To start, we'll work with a Random Forest. This is the most well-known of the Decision Forest training algorithms.
# 
# A Random Forest is a collection of decision trees, each trained independently on a random subset of the training dataset (sampled with replacement). The algorithm is unique in that it is robust to overfitting, and easy to use.
# 
# We can list the all the available models in TensorFlow Decision Forests using the following code:

# In[22]:


tfdf.keras.get_all_models()


# # Configure the model
# 
# TensorFlow Decision Forests provides good defaults for you (e.g. the top ranking hyperparameters on our benchmarks, slightly modified to run in reasonable time). If you would like to configure the learning algorithm, you will find many options you can explore to get the highest possible accuracy.
# 
# You can select a template and/or set parameters as follows:
# 
# ```rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")```
# 
# Read more [here](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel).

# # Create a Random Forest
# 
# Today, we will use the defaults to create the Random Forest Model while specifiyng the task type as `tfdf.keras.Task.REGRESSION`.

# In[23]:


# Create RandomForestModel for label content
model_content = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model_content.compile(metrics=["mse"])

# Create RandomForestModel for label wording
model_wording = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model_wording.compile(metrics=["mse"])


# # Train the model
# 
# We will train the model using a one-liner.
# 
# Note: you may see a warning about Autograph. You can safely ignore this, it will be fixed in the next release.

# In[24]:


# Training RandomForestModel for label content
model_content.fit(x=content_train_ds)

# Training RandomForestModel for label wording
model_wording.fit(x=wording_train_ds)


# # Visualize the model
# One benefit of tree-based models is that we can easily visualize them. The default number of trees used in the Random Forests is 300. We can select a tree to display below.

# In[25]:


# Visualize model_content
tfdf.model_plotter.plot_model_in_colab(model_content, tree_idx=0, max_depth=3)


# In[26]:


# Visualize model_content
tfdf.model_plotter.plot_model_in_colab(model_wording, tree_idx=0, max_depth=3)


# # Evaluate the model on the Out of bag (OOB) data and the validation dataset
# 
# Before training the dataset we have manually seperated 20% of the dataset for validation named as `valid_ds`.
# 
# We can also use Out of bag (OOB) score to validate our RandomForestModel.
# To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm and the rest of the samples are used to finetune the model.The subset of data that is not chosen is known as Out of bag data (OOB).
# OOB score is computed on the OOB data.
# 
# Read more about OOB data [here](https://developers.google.com/machine-learning/decision-forests/out-of-bag).
# 
# The training logs show the Root Mean Squared Error (RMSE) evaluated on the out-of-bag dataset according to the number of trees in the model. Let us plot this.
# 
# Note: Smaller values are better for this hyperparameter.

# In[27]:


# Plot log data for model_content
logs = model_content.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()


# In[28]:


# Plot log data for model_wording
logs = model_wording.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()


# We can also see some general stats on the OOB dataset:

# In[29]:


# General stats for model_content
inspector_content = model_content.make_inspector()
inspector_content.evaluation()


# In[30]:


# General stats for model_wording
inspector_wording = model_wording.make_inspector()
inspector_wording.evaluation()


# Now, let us run an evaluation using the validation dataset.

# In[31]:


# Create validation dataset for model_content
valid_ds_content = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd[FEATURE_CONTENT], label="content", task = tfdf.keras.Task.REGRESSION)

# Create validation dataset for model_wording
valid_ds_wording = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd[FEATURE_WORDING], label="wording", task = tfdf.keras.Task.REGRESSION)

# Run evaluation for model_content
evaluation_content = model_content.evaluate(x=valid_ds_content,return_dict=True)
for name, value in evaluation_content.items():
  print(f"{name}: {value:.4f}")

# Run evaluation for model_wording
evaluation_wording = model_wording.evaluate(x=valid_ds_wording,return_dict=True)
for name, value in evaluation_wording.items():
  print(f"{name}: {value:.4f}")


# # Variable importances
# 
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests.
# Let us list the available `Variable Importances` for Decision Trees:

# In[32]:


print(f"Available variable importances for model_content:")
for importance in inspector_content.variable_importances().keys():
  print("\t", importance)


# In[33]:


print(f"Available variable importances for model_wording:")
for importance in inspector_wording.variable_importances().keys():
  print("\t", importance)


# As an example, let us display the important features for the Variable Importance `NUM_AS_ROOT`.
# 
# The larger the importance score for `NUM_AS_ROOT`, the more impact it has on the outcome of the model.
# 
# By default, the list is sorted from the most important to the least. From the output you can infer that the feature at the top of the list is used as the root node in most number of trees in the random forest than any other feature.

# In[34]:


# For model_content.
# Each line is: (feature name, (index of the feature), importance score)
inspector_content.variable_importances()["NUM_AS_ROOT"]


# In[35]:


# For model_wording.
# Each line is: (feature name, (index of the feature), importance score)
inspector_wording.variable_importances()["NUM_AS_ROOT"]


# # Submission

# In[36]:


df_test_prompt = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv')
df_test_summaries = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv')


# In[37]:


df_test = df_test_summaries.merge(df_test_prompt, on='prompt_id')


# In[38]:


df_test.head()


# In[39]:


processed_test_df = feature_engineer(df_test)


# In[40]:


processed_test_df.head()


# In[41]:


test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(processed_test_df[FEATURE_COLUMNS], task = tfdf.keras.Task.REGRESSION)


# In[42]:


processed_test_df['content'] = model_content.predict(test_ds)
processed_test_df['wording'] = model_wording.predict(test_ds)


# In[43]:


processed_test_df.head()


# In[44]:


processed_test_df[['student_id', 'content', 'wording']].to_csv('submission.csv',index=False)
display(pd.read_csv('submission.csv'))

