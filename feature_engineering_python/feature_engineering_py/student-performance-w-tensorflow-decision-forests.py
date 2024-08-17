#!/usr/bin/env python
# coding: utf-8

# # Student Performance from Game Play Using TensorFlow Decision Forests
# 
# ---
# 
# This notebook will take you through the steps needed to train a baseline Gradient Boosted Trees Model using TensorFlow Decision Forests on the `Student Performance from Game Play` dataset made available for this competition, to predict if players will answer questions correctly.
# We will load the data from a CSV file. Roughly, the code will look as follows:
# 
# ```
# import tensorflow_decision_forests as tfdf
# import pandas as pd
#   
# dataset = pd.read_csv("project/dataset.csv")
# tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")
# 
# model = tfdf.keras.GradientBoostedTreesModel()
# model.fit(tf_dataset)
#   
# print(model.summary())
# ```
# 
# We will also learn how to optimize reading of big datasets, do some feature engineering, data visualization and calculate better results using the F1-score
# 
# 
# Decision Forests are a family of tree-based models including Random Forests and Gradient Boosted Trees. They are the best place to start when working with tabular data, and will often outperform (or provide a strong baseline) before you begin experimenting with neural networks.

# One of the key aspects of TensorFlow Decision Forests that makes it even more suitable for this competition, particularly given the runtime limitations, is that it has been extensively tested for training and inference on CPUs, making it possible to train it on lower-end machines.

# # Import the Required Libraries

# In[1]:


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_decision_forests as tfdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


print("TensorFlow Decision Forests v" + tfdf.__version__)
print("TensorFlow Addons v" + tfa.__version__)
print("TensorFlow v" + tf.__version__)


# # Load the Dataset
# 
# Since the dataset is huge, some people may face memory errors while reading the dataset from the csv. To avoid this, we will try to optimize the memory used by Pandas to load and store the dataset.
# 
# 
# When Pandas loads a dataset, by default, it automatically detects the data types of the different columns.
# Irresepective of the maximum value that is stored in these columns, Pandas assigns `int64` for numerical columns, `float64` for float columns, `object` dtype for string columns etc.
# 
# 
# We may be able to reduce the size of these columns in memory by downcasting numerical columns to smaller types (like `int8`, `int32`, `float32` etc.), if their maximum values don't need the larger types for storage, (like `int64`, `float64` etc.).
# 
# 
# Similarly, Pandas automatically detects string columns as `object` datatype. To reduce memory usage of string columns which store categorical data, we specify their datatype as `category`.
# 
# 
# Many of the columns in this dataset can be downcast to smaller types.
# 
# We will provide a dict of `dtypes` for columns to pandas while reading the dataset.

# In[3]:


# Reference: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384359
dtypes={
    'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'room_coor_x':np.float32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
    'text':'category',
    'fqid':'category',
    'room_fqid':'category',
    'text_fqid':'category',
    'fullscreen':'category',
    'hq':'category',
    'music':'category',
    'level_group':'category'}

dataset_df = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train.csv', dtype=dtypes)
print("Full train dataset shape is {}".format(dataset_df.shape))


# The data is composed of 20 columns and 26296946 entries. We can see all 20 dimensions of our dataset by printing out the first 5 entries using the following code:

# In[4]:


# Display the first 5 examples
dataset_df.head(5)


# Please note that `session_id` uniquely identifies a user session.

# # Load the labels
# 
# The labels for the training dataset are stored in the `train_labels.csv`. It consists of the information on whether the user in a particular session answered each question correctly. Load the labels data by running the following code. `

# In[5]:


labels = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')


# Each value in the column, `session_id` is a combination of both the session and the question number. 
# We will split these into individual columns for ease of use.

# In[6]:


labels['session'] = labels.session_id.apply(lambda x: int(x.split('_')[0]) )
labels['q'] = labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )


#  Let us take a look at the first 5 entries of `labels` using the following code:

# In[7]:


# Display the first 5 examples
labels.head(5)


# Our goal is to train models for each question to predict the label `correct` for any input user session. 

# # Bar chart for label column: correct

# First we will plot a bar chart for the values of the label `correct`.

# In[8]:


plt.figure(figsize=(3, 3))
plot_df = labels.correct.value_counts()
plot_df.plot(kind="bar", color=['b', 'c'])


# Now, let us plot the values of the label column `correct` for each question.

# In[9]:


plt.figure(figsize=(10, 20))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.suptitle("\"Correct\" column values for each question", fontsize=14, y=0.94)
for n in range(1,19):
    #print(n, str(n))
    ax = plt.subplot(6, 3, n)

    # filter df and plot ticker on the new subplot axis
    plot_df = labels.loc[labels.q == n]
    plot_df = plot_df.correct.value_counts()
    plot_df.plot(ax=ax, kind="bar", color=['b', 'c'])
    
    # chart formatting
    ax.set_title("Question " + str(n))
    ax.set_xlabel("")


# # Prepare the dataset
# 
# As summarized in the competition overview, the dataset presents the questions and data to us in order of `levels - level segments`(represented by column `level_group`) 0-4, 5-12, and 13-22. We have to predict the correctness of each segment's questions as they are presented. To do this we will create basic aggregate features from the relevant columns. You can create more features to boost your scores. 
# 
# First, we will create two separate lists with names of the Categorical columns and Numerical columns. We will avoid columns `fullscreen`, `hq` and `music` since they don't add any useful value for this problem statement.

# In[10]:


CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 
        'screen_coor_x', 'screen_coor_y', 'hover_duration']


# For each categorical column, we will first group the dataset by `session_id`  and `level_group`. We will then count the number of **distinct elements** in the column for each group and store it temporarily.
# 
# For all numerical columns, we will group the dataset by `session id` and `level_group`. Instead of counting the number of distinct elements, we will calculate the `mean` and `standard deviation` of the numerical column for each group and store it temporarily.
# 
# After this, we will concatenate the temporary data frames we generated in the earlier step for each column to create our new feature engineered dataset.

# In[11]:


# Reference: https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook

def feature_engineer(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')
    return dataset_df


# In[12]:


dataset_df = feature_engineer(dataset_df)
print("Full prepared dataset shape is {}".format(dataset_df.shape))


# Our feature engineered dataset is composed of 22 columns and 70686 entries. 

# # Basic exploration of the prepared dataset

# Let us print out the first 5 entries using the following code:

# In[13]:


# Display the first 5 examples
dataset_df.head(5)


# In[14]:


dataset_df.describe()


# # Numerical data distribution¶
# 
# Let us plot some numerical columns and their value against each level_group:

# In[15]:


figure, axis = plt.subplots(3, 2, figsize=(10, 10))

for name, data in dataset_df.groupby('level_group'):
    axis[0, 0].plot(range(1, len(data['room_coor_x_std'])+1), data['room_coor_x_std'], label=name)
    axis[0, 1].plot(range(1, len(data['room_coor_y_std'])+1), data['room_coor_y_std'], label=name)
    axis[1, 0].plot(range(1, len(data['screen_coor_x_std'])+1), data['screen_coor_x_std'], label=name)
    axis[1, 1].plot(range(1, len(data['screen_coor_y_std'])+1), data['screen_coor_y_std'], label=name)
    axis[2, 0].plot(range(1, len(data['hover_duration'])+1), data['hover_duration_std'], label=name)
    axis[2, 1].plot(range(1, len(data['elapsed_time_std'])+1), data['elapsed_time_std'], label=name)
    

axis[0, 0].set_title('room_coor_x')
axis[0, 1].set_title('room_coor_y')
axis[1, 0].set_title('screen_coor_x')
axis[1, 1].set_title('screen_coor_y')
axis[2, 0].set_title('hover_duration')
axis[2, 1].set_title('elapsed_time_std')

for i in range(3):
    axis[i, 0].legend()
    axis[i, 1].legend()

plt.show()


# Now let us split the dataset into training and testing datasets:

# In[16]:


def split_dataset(dataset, test_ratio=0.20):
    USER_LIST = dataset.index.unique()
    split = int(len(USER_LIST) * (1 - 0.20))
    return dataset.loc[USER_LIST[:split]], dataset.loc[USER_LIST[split:]]

train_x, valid_x = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_x), len(valid_x)))


# # Select a Model
# There are several tree-based models for you to choose from.
# 
# - RandomForestModel
# - GradientBoostedTreesModel
# - CartModel
# - DistributedGradientBoostedTreesModel
# 
# We can list all the available models in TensorFlow Decision Forests using the following code:

# In[17]:


tfdf.keras.get_all_models()


# To get started, we'll work with a Gradient Boosted Trees Model. This is one of the well-known Decision Forest training algorithms.
# 
# A Gradient Boosted Decision Tree is a set of shallow decision trees trained sequentially. Each tree is trained to predict and then "correct" for the errors of the previously trained trees.

# # How can I configure a tree-based model?
# 
# TensorFlow Decision Forests provides good defaults for you (e.g., the top ranking hyperparameters on our benchmarks, slightly modified to run in reasonable time). If you would like to configure the learning algorithm, you will find many options you can explore to get the highest possible accuracy.
# 
# You can select a template and/or set parameters as follows:
# ```
# rf = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
# ```
# 
# You can read more [here](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel).

# # Training

# 
# We will train a model for each question to predict if the question will be answered correctly by a user. 
# There are a total of 18 questions in the dataset. Hence, we will be training 18 models, one for each question.
# 
# We need to provide a few data structures to our training loop to store the trained models, predictions on the validation set and evaluation scores for the trained models.
# 
# We will create these using the following code:
# 

# In[18]:


# Fetch the unique list of user sessions in the validation dataset. We assigned 
# `session_id` as the index of our feature engineered dataset. Hence fetching 
# the unique values in the index column will give us a list of users in the 
# validation set.
VALID_USER_LIST = valid_x.index.unique()

# Create a dataframe for storing the predictions of each question for all users
# in the validation set.
# For this, the required size of the data frame is: 
# (no: of users in validation set  x no of questions).
# We will initialize all the predicted values in the data frame to zero.
# The dataframe's index column is the user `session_id`s. 
prediction_df = pd.DataFrame(data=np.zeros((len(VALID_USER_LIST),18)), index=VALID_USER_LIST)

# Create an empty dictionary to store the models created for each question.
models = {}

# Create an empty dictionary to store the evaluation score for each question.
evaluation_dict ={}


# Before training the data we have to understand how `level_groups` and `questions` are associated to each other.
# 
# In this game the first quiz checkpoint(i.e., questions 1 to 3) comes after finishing levels 0 to 4. So for training questions 1 to 3 we will use data from the `level_group` 0-4. Similarly, we will use data from the `level_group` 5-12 to train questions from 4 to 13 and data from the `level_group` 13-22 to train questions from 14 to 18.
# 
# We will train a model for each question and store the trained model in the `models` dict.

# In[19]:


# Iterate through questions 1 to 18 to train models for each question, evaluate
# the trained model and store the predicted values.
for q_no in range(1,19):

    # Select level group for the question based on the q_no.
    if q_no<=3: grp = '0-4'
    elif q_no<=13: grp = '5-12'
    elif q_no<=22: grp = '13-22'
    print("### q_no", q_no, "grp", grp)
    
        
    # Filter the rows in the datasets based on the selected level group. 
    train_df = train_x.loc[train_x.level_group == grp]
    train_users = train_df.index.values
    valid_df = valid_x.loc[valid_x.level_group == grp]
    valid_users = valid_df.index.values

    # Select the labels for the related q_no.
    train_labels = labels.loc[labels.q==q_no].set_index('session').loc[train_users]
    valid_labels = labels.loc[labels.q==q_no].set_index('session').loc[valid_users]

    # Add the label to the filtered datasets.
    train_df["correct"] = train_labels["correct"]
    valid_df["correct"] = valid_labels["correct"]

    # There's one more step required before we can train the model. 
    # We need to convert the datatset from Pandas format (pd.DataFrame)
    # into TensorFlow Datasets format (tf.data.Dataset).
    # TensorFlow Datasets is a high performance data loading library 
    # which is helpful when training neural networks with accelerators like GPUs and TPUs.
    # We are omitting `level_group`, since it is not needed for training anymore.
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df.loc[:, train_df.columns != 'level_group'], label="correct")
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df.loc[:, valid_df.columns != 'level_group'], label="correct")

    # We will now create the Gradient Boosted Trees Model with default settings. 
    # By default the model is set to train for a classification task.
    gbtm = tfdf.keras.GradientBoostedTreesModel(verbose=0)
    gbtm.compile(metrics=["accuracy"])

    # Train the model.
    gbtm.fit(x=train_ds)

    # Store the model
    models[f'{grp}_{q_no}'] = gbtm

    # Evaluate the trained model on the validation dataset and store the 
    # evaluation accuracy in the `evaluation_dict`.
    inspector = gbtm.make_inspector()
    inspector.evaluation()
    evaluation = gbtm.evaluate(x=valid_ds,return_dict=True)
    evaluation_dict[q_no] = evaluation["accuracy"]         

    # Use the trained model to make predictions on the validation dataset and 
    # store the predicted values in the `prediction_df` dataframe.
    predict = gbtm.predict(x=valid_ds)
    prediction_df.loc[valid_users, q_no-1] = predict.flatten()     


# # Inspect the Accuracy of the models.
# 
# We trained a model for each question. Now let us check the accuracy of each model and overall accuracy for all the models combined. 
# 
# Note: Since the label distribution is imbalanced, we can't make an assumption on the model performance from accuracy score alone. 

# In[20]:


for name, value in evaluation_dict.items():
  print(f"question {name}: accuracy {value:.4f}")

print("\nAverage accuracy", sum(evaluation_dict.values())/18)


# # Visualize the model
# 
# One benefit of tree-based models is that we can easily visualize them. The default number of trees used in the Random Forests is 300. 
# 
# Let us pick one model from `models` dict and select a tree to display below.

# In[21]:


tfdf.model_plotter.plot_model_in_colab(models['0-4_1'], tree_idx=0, max_depth=3)


# # Variable importances
# 
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests. Let us pick one model from models dict and inspect it.
# 
# Let us list the available Variable Importances for Decision Trees:

# In[22]:


inspector = models['0-4_1'].make_inspector()

print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# As an example, let us display the important features for the Variable Importance NUM_AS_ROOT.
# 
# The larger the importance score for NUM_AS_ROOT, the more impact it has on the outcome of the model for Question 1(i.e., model\["0-4_1"\]).
# 
# By default, the list is sorted from the most important to the least. From the output you can infer that the feature at the top of the list is used as the root node in most number of trees in the gradient boosted trees  than any other feature.

# In[23]:


# Each line is: (feature name, (index of the feature), importance score)
inspector.variable_importances()["NUM_AS_ROOT"]


# # Threshold-Moving for Imbalanced Classification
# 
# Since the values of the column `correct` is fairly imbalanced, using the default threshold of `0.5` to map the predictions into classes 0 or 1 can result in poor performance. 
# In such cases, to improve performance we will calculate the `F1 score` for a certain range of thresholds and try to find the best threshold aka, threshold with highest `F1 score`. Then we will use this threshold to map the predicted probabilities to class labels 0 or 1.
# 
# Please note that we are using `F1 score` since it is a better metric than `accuracy` to evaluate problems with class imbalance.

# In[24]:


# Create a dataframe of required size:
# (no: of users in validation set x no: of questions) initialized to zero values
# to store true values of the label `correct`. 
true_df = pd.DataFrame(data=np.zeros((len(VALID_USER_LIST),18)), index=VALID_USER_LIST)
for i in range(18):
    # Get the true labels.
    tmp = labels.loc[labels.q == i+1].set_index('session').loc[VALID_USER_LIST]
    true_df[i] = tmp.correct.values

max_score = 0; best_threshold = 0

# Loop through threshold values from 0.4 to 0.8 and select the threshold with 
# the highest `F1 score`.
for threshold in np.arange(0.4,0.8,0.01):
    metric = tfa.metrics.F1Score(num_classes=2,average="macro",threshold=threshold)
    y_true = tf.one_hot(true_df.values.reshape((-1)), depth=2)
    y_pred = tf.one_hot((prediction_df.values.reshape((-1))>threshold).astype('int'), depth=2)
    metric.update_state(y_true, y_pred)
    f1_score = metric.result().numpy()
    if f1_score > max_score:
        max_score = f1_score
        best_threshold = threshold
        
print("Best threshold ", best_threshold, "\tF1 score ", max_score)


# # Submission
# 
# Here you'll use the `best_threshold` calculate in the previous cell

# In[25]:


# Reference
# https://www.kaggle.com/code/philculliton/basic-submission-demo
# https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook


import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()

limits = {'0-4':(1,4), '5-12':(4,14), '13-22':(14,19)}

for (test, sample_submission) in iter_test:
    test_df = feature_engineer(test)
    grp = test_df.level_group.values[0]
    a,b = limits[grp]
    for t in range(a,b):
        gbtm = models[f'{grp}_{t}']
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df.loc[:, test_df.columns != 'level_group'])
        predictions = gbtm.predict(test_ds)
        mask = sample_submission.session_id.str.contains(f'q{t}')
        n_predictions = (predictions > best_threshold).astype(int)
        sample_submission.loc[mask,'correct'] = n_predictions.flatten()
    
    env.predict(sample_submission)


# In[26]:


get_ipython().system(' head submission.csv')

