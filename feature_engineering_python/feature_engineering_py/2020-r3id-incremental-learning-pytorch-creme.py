#!/usr/bin/env python
# coding: utf-8

# # Introduction / Setup
# 
# This kernel demonstrates online/incremental learning on the Riiid! Answer Correctness Prediction.
# 
# Incremental learning is especially beneficial for timeseries, as the model is adapted each time a datapoint is added. Instead of pre-training on a large batch of data and deploying a static model to perform inference for the rest of its life, we can deploy a untrained model and let it train as data flows through it. This makes it easy to modify and limits computing power needed, plus the model is able to adapt to slowly changing data.
# 
# While this approach will likely result in a lower score than using the entire train set to pre-train a static model (given the same level of feature engineering), it seems like the practically better way to handle this challenge - especially if we consider applicability of the result to the actual data in the competition organizer's system.

# In[1]:


# import some typical packages and fix random seed

import numpy as np
import pandas as pd

import time

from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# We'll need to install creme from its wheel file as internet has to be turned off for submission kernels. Creme is a package explicitly written for online/incremental learning and can be found here:
# 
# https://creme-ml.github.io/
# 
# https://github.com/creme-ml/creme/

# In[2]:


# install creme from wheel
get_ipython().system('pip install ../input/cremewheels/creme-0.6.1-cp37-cp37m-manylinux1_x86_64.whl')


# Defining datatypes for loading train.csv as suggested in the official intro kernel.

# In[3]:


# datatypes for loading train.csv
train_dtypes = {'row_id': 'int64',
                'timestamp': 'int64',
                'user_id': 'int32',
                'content_id': 'int16',
                'content_type_id': 'int8',
                'task_container_id': 'int16',
                'user_answer': 'int8',
                'answered_correctly': 'int8',
                'prior_question_elapsed_time': 'float32',
                'prior_question_had_explanation': 'boolean'}


# # Set up data handling, define features
# 
# We want to store some answering history to help with our predictions, so we'll go with two dataframes - one for question-specific statistics and one for user-specific statistics. On the most basic level we want to at least know for each question how often it was attempted and how often it was answered correctly, and for each user how many questions were attempted and answered correctly.

# In[4]:


# dataframe keeping question-specific statistics and features
questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
questions.drop(columns=['question_id', 'bundle_id', 'correct_answer', 'tags', 'part'], inplace=True)
questions['num_answers'] = 0.
questions['num_correct'] = 0.

#----------------------------------------------------------------------------------------------------

# dataframe keeping user-specific statistics and features
users = pd.DataFrame(columns = ['num_answers', 'num_correct'], dtype=np.float)


# In[5]:


questions.sample()


# In[6]:


users


# This function allows use to add information to these two dataframes. Given one or more rows from the test or train dataset with known answering outcome, it increases the attempt / correct answers counters both in the <code>questions</code> and <code>users</code> dataframes. We can always decide to store other information later on.

# In[7]:


def add_data(df):   
    # ---------------------------------------------------------------------------------------
    gb = df.groupby(by=['user_id', 'answered_correctly']).count().max(axis=1)
    # add users which aren't recorded yet
    for uid in np.setdiff1d(gb.index.levels[0], users.index):
        users.at[uid] = 0
    # add question counts [user]
    for uid in gb.index.levels[0]:
        users.loc[uid, 'num_answers'] += gb.loc[uid].sum()
        if 1 in gb[uid].index.get_level_values(0):
            users.loc[uid, 'num_correct'] += gb.loc[uid].loc[1]

    # ---------------------------------------------------------------------------------------
    # add question counts [question]
    gb = df.groupby(by=['content_id']).count().max(axis=1)
    questions['num_answers'] = questions['num_answers'].add(gb.reindex(questions.index, fill_value=0))
    
    gb = df[df['answered_correctly'] == 1].groupby(by=['content_id']).count().max(axis=1)
    questions['num_correct'] = questions['num_correct'].add(gb.reindex(questions.index, fill_value=0))
    return


# This function generates features using the dataframe given to it (train or test) and the current status of the <code>users</code> and <code>questions</code> tables. It returns a list of the label names and a feature array for the given batch.

# In[8]:


def get_features(df):
    uid_list = df['user_id'].values
    qid_list = df['content_id'].values

    for uid in np.setdiff1d(np.unique(uid_list), users.index):
        users.at[uid] = 0
    q_selec = questions.reindex(df['content_id'].values).copy()
    u_selec = users.reindex(df['user_id'].values).copy()

    # create features
    df['mean_q_correctness'] = q_selec['num_correct'].divide(q_selec['num_answers']).values
    df['mean_u_correctness'] = u_selec['num_correct'].divide(u_selec['num_answers']).values
    
    # fill invalid values
    fill_dict = {
        'mean_q_correctness': 0.5,
        'mean_u_correctness': 0.5,    
    }
    for col in [*fill_dict]:
        df[col].fillna(value=fill_dict[col], inplace=True)
        
    # return only columns we want to use for predicting
    use_cols = ['mean_q_correctness', 'mean_u_correctness']
            
    return use_cols, df[use_cols].values.astype(np.float)


# # Define models
# 
# Now that the data handling is set up, let's have a look at models which we want to use for prediction. Here we'll use a Pytorch multilayer perceptron (MLP) and a model using the creme package. We can quickly add more models without having to do any noticeable modifications in the code further below as long as we stick to a certain format.
# 
# Here I built each model to have three utility functions:
# * <code>predict(X)</code> uses the model in its current state to predict an output <code>y_pred</code> for the input X
# * <code>fit(X, y)</code> fits a new batch of data to the model
# * <code>iterate_batch(X, y)</code> is simply a combination of the two, first obtaining a prediction for <code>X</code> before updating the model using the true <code>y</code> values given and returning the predicted output <code>y_pred</code>.
# 
# The first two functions are used for test data, where we first need to predict on a dataset and submit the prediction before we obtain the true outcome with which we can update our models. For train data we can simply use <code>iterate_batch()</code> to perform both steps in one.

# ## Pytorch MLP
# 
# Simple MLP with three hidden layers and ReLu activations

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.optim import SGD, Adam

torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RiiidNet(nn.Module):
    
    def __init__(self, NFEATURES, NHIDDEN):
        super(RiiidNet, self).__init__()
        self.NFEATURES = NFEATURES
        self.fc1 = nn.Linear(NFEATURES, NHIDDEN)
        self.fc2 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc3 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc4 = nn.Linear(NHIDDEN, NHIDDEN)
        self.fc5 = nn.Linear(NHIDDEN, 2)

    def forward(self, x):
        x = x.view(-1, self.NFEATURES)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = torch.softmax(x, dim=1)
        return x
    
class MLPModel():
    
    def __init__(self, NFEATURES, NHIDDEN, device):
        self.model = RiiidNet(NFEATURES, NHIDDEN).to(device)
        self.optimiser = Adam(self.model.parameters(), lr=0.002)
        self.device = device
        
    def predict(self, X):
        with torch.no_grad():
            # prepare data
            X = torch.tensor(X, dtype=torch.float).to(self.device)
            # forward pass
            output = self.model(X)
        return output[:,1].numpy()
    
    def fit(self, X, y):
        self.iterate_batch(X, y)
        return
    
    def iterate_batch(self, X, y):
        # prepare data
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)
        y_full = torch.zeros(len(y), 2).to(self.device)
        y_full[np.where(y.numpy()),1] = 1

        # zero gradients
        self.model.zero_grad()
        # forward pass
        output = self.model(X)

        # calculate loss
        loss = F.binary_cross_entropy(output, y_full)
        # backward pass
        loss.backward()
        # update parameters
        self.optimiser.step()
        
        return output[:,1].detach().numpy()


# ## Creme
# 
# Here we use the logistic regression model of the creme package. Predictions and model updates can be performed either one by one (<code>predict_proba_one</code> and <code>fit_one</code>) or in batches (<code>predict_proba_many</code> and <code>fit_many</code>). However, when I tried with the <code>_many</code> functions my ROC area under curve scores nosedived quite badly - so I'm not using this for now (might not be working perfectly yet, creme is pretty new) but keep the code for demonstration / completeness.

# In[10]:


from creme import linear_model
from creme import optim

class CremeModel():
    
    def __init__(self):
        self.optimizer = optim.SGD(lr=0.01)
        self.model = linear_model.LogisticRegression(self.optimizer)
             
    def predict(self, X, many=False):
        if many:
            # predict using dataframe
            X_df = pd.DataFrame(data=X)
            y_pred = self.model.predict_proba_many(X_df).loc[:,True].values
        else:
            # allocate prediction list
            y_pred = np.zeros((X.shape[0]))
            # make numeric keys for dict
            tmp_names = [str(i) for i in range(X.shape[1])]
            for i in range(len(y_pred)):
                # create data dict
                X_dict = dict(zip(tmp_names, list(X[i])))
                # predict
                y_pred[i] = self.model.predict_proba_one(X_dict)[True]
        return y_pred
    
    def fit(self, X, y, many=False):
        if many:
            # prepare data
            X_df = pd.DataFrame(data=X)
            y_df = pd.Series(data=y)
            # train / update
            self.model.fit_many(X_df, y_df)
        else:
            # make numeric keys for dict
            tmp_names = [str(i) for i in range(X.shape[1])]
            for i in range(len(y)):
                # create data dict
                X_dict = dict(zip(tmp_names, list(X[i])))
                # train / update
                self.model.fit_one(X_dict, y[i])
        return 
    
    def iterate_batch(self, X, y, many=False):
        y_pred = self.predict(X)
        self.fit(X,y)
        return y_pred


# # Perform incremental training
# 
# Now we just set up these models and feed some batches of train data through it. I'm stopping after 300 batches of 10,000 rows to keep the waiting time down.

# In[11]:


mlp = MLPModel(2, 32, device)
lrg = CremeModel()

models = {
    'MLP': mlp,
    'LRG': lrg,
}

metrics = {}
for tag in [*models] + ['mean']:
    metrics[tag] = {}
    metrics[tag]['accuracy'] = []
    metrics[tag]['roc_auc'] = []

trainfile = '/kaggle/input/riiid-test-answer-prediction/train.csv'

for n, train_part in enumerate(pd.read_csv(trainfile, chunksize=10**4, dtype=train_dtypes, iterator=True)):
    # discard lecture rows
    train_part = train_part[train_part['content_type_id']==0]
    
    # get features
    label_names, X = get_features(train_part.copy())
    # get targets
    y = train_part['answered_correctly'].values.astype(np.float)
    
    # iterate all models
    preds = {}
    for tag in [*models]:
        preds[tag] = models[tag].iterate_batch(X, y)

    # get mean of all predictions
    preds['mean'] = np.mean([preds[tag] for tag in [*models]], axis=0)
    
    # calculate metrics
    for tag in [*models] + ['mean']:
        acc = ((preds[tag] > 0.5).astype(float) == y).sum() / len(y)
        roc_auc = roc_auc_score(y, preds[tag])
        metrics[tag]['accuracy'].append(acc)
        metrics[tag]['roc_auc'].append(roc_auc)
        
        if tag == 'mean':
            print('Chunk: {} \t Accuracy: {:0.3f} \t ROC AUC: {:0.3f}'.format(
                n, acc, roc_auc))
            
    add_data(train_part.copy())
    
    if n > 500:
        break


# Here come plots of accuracy and score...

# In[12]:


from scipy.signal import convolve

def graphfilter(x, N=3):
    filt = np.ones((N)) / N
    x_full = np.concatenate([
        np.full((int((N-1)/2)), x[0]),
        x,
        np.full((int((N-1)/2)), x[-1])])
    return convolve(x, filt, mode='valid')


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(graphfilter(metrics['mean']['accuracy']), label='Accuracy')
plt.plot(graphfilter(metrics['mean']['roc_auc']), label='ROC AUC')
plt.xlabel('Chunk number')
plt.ylabel('Accuracy / ROC AUC')
plt.ylim([0.4,0.9])
plt.legend()
plt.show()


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
for tag in [*models] + ['mean']:
    plt.plot(graphfilter(metrics[tag]['roc_auc']), label=tag)
plt.xlabel('Chunk number')
plt.ylabel('ROC AUC')
plt.ylim([0.4,0.9])
plt.legend()
plt.show()


# And the mean ROC area under curve from the train data. Note that while this is train data, this score is calculated entirely on unseen data - each batch is predicted before the model is updated with it.

# In[15]:


print('Mean ROC AUC:', np.mean(metrics['mean']['roc_auc']))


# # Test evaluation
# 
# For completeness' sake, we feed the test data through as well.

# In[16]:


import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


# In[17]:


prev_test = None
count = 0

for (test_df, sample_prediction_df) in iter_test:
    
    if (not prev_test is None):
        # take answers returned and update data with previous test set
        prev_answers = np.array(eval(test_df.iloc[0]['prior_group_answers_correct']), dtype=np.int)
        prev_test['answered_correctly'] = prev_answers
        prev_test = prev_test[prev_test['content_type_id'] == 0]
        add_data(prev_test.copy())

        # iterate models with previous data
        for tag in [*models]:
            models[tag].fit(X, prev_test['answered_correctly'].values)

        # calculate metrics if possible
        try:
            acc = ((preds['mean'] > 0.5).astype(float) == prev_answers).sum() / len(prev_answers)
            roc_auc = roc_auc_score(prev_answers, preds[tag])
            print('Group: {} \t Accuracy: {:0.3f} \t ROC AUC: {:0.3f}'.format(count, acc, roc_auc))
        except:
            print('Group: {} \t Could not calculate metrics'.format(count))
    
    # copy new test_df into prev_test
    prev_test = test_df.copy()
    
    # remove lecture rows in test data
    test_df = test_df[test_df['content_type_id'] == 0]
    
    # get features
    label_names, X = get_features(test_df.copy())
    
    # predict all models
    preds = {}
    for tag in [*models]:
        preds[tag] = models[tag].predict(X)

    # get mean of all predictions
    preds['mean'] = np.mean([preds[tag] for tag in [*models]], axis=0)
    
    # submit predictions
    test_df['answered_correctly'] = preds['mean']
    env.predict(test_df.loc[:,['row_id', 'answered_correctly']])
    
    count += 1


# Feel free to comment if you have any questions/suggestions and/or leave an upvote!
