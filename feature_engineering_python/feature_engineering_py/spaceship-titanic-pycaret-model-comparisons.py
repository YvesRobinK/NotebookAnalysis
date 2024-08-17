#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! 😊
# 
# 👉 EDA is done in this [notebook](https://www.kaggle.com/hasanbasriakcay/spaceship-titanic-eda-fe-baseline).

# In[1]:


import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("../input/spaceship-titanic/train.csv")
test = pd.read_csv("../input/spaceship-titanic/test.csv")
submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")

display(train.head())
display(test.head())
display(submission.head())


# # Feature Engineering

# In[2]:


_, _, train["Cabin_3"] = train["Cabin"].str.split("/", expand=True)
_, _, test["Cabin_3"] = test["Cabin"].str.split("/", expand=True)

train.drop(["Name", "Cabin"], axis=1, inplace=True)


# # Modelling

# In[3]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret[full]\n')


# In[4]:


from pycaret.classification import *

numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(train.columns) - set(numeric_cols))
object_cols.remove("Transported")
ignore_cols = ["PassengerId"]

clf = setup(data=train,
            target='Transported',
            normalize = True,
            normalize_method = 'robust',
            create_clusters = True,
            #feature_interaction = True,
            numeric_features = numeric_cols,
            categorical_features = object_cols,
            ignore_features = ignore_cols,
            session_id = 42,
            use_gpu = False,
            silent = True,
            fold = 10,
            n_jobs = -1)


# In[5]:


N = 2
top = compare_models(sort = 'Accuracy', n_select = N)


# # Stacking

# In[6]:


stack = stack_models(top, optimize='Accuracy')
predict_model(stack);


# In[7]:


final_stack = finalize_model(stack)


# In[8]:


plot_model(final_stack, plot='error')


# In[9]:


plot_model(final_stack, plot = 'confusion_matrix')


# # Blending

# In[10]:


blend = blend_models(top, optimize='Accuracy')
predict_model(blend);


# In[11]:


final_blend = finalize_model(blend)


# In[12]:


plot_model(final_blend, plot='error')


# In[13]:


plot_model(final_blend, plot = 'confusion_matrix')


# # Ensembling

# In[14]:


ensemble = ensemble_model(top[0], method='Bagging')
predict_model(ensemble);


# In[15]:


final_ensemble = finalize_model(ensemble)


# In[16]:


plot_model(final_ensemble, plot='error')


# In[17]:


plot_model(final_ensemble, plot = 'confusion_matrix')


# # Predictions

# In[18]:


import gc
gc.collect()
unseen_predictions_stack = predict_model(final_stack, data=test)
unseen_predictions_blend = predict_model(final_blend, data=test)
unseen_predictions_ensemble = predict_model(final_ensemble, data=test)
unseen_predictions_stack.head()


# In[19]:


assert(len(test.index)==len(unseen_predictions_stack))
sub = pd.DataFrame(list(zip(submission.PassengerId, unseen_predictions_stack.Label)),columns = ['PassengerId', 'Transported'])
sub.to_csv('submission_stack.csv', index = False)
sub = pd.DataFrame(list(zip(submission.PassengerId, unseen_predictions_blend.Label)),columns = ['PassengerId', 'Transported'])
sub.to_csv('submission_blend.csv', index = False)
sub = pd.DataFrame(list(zip(submission.PassengerId, unseen_predictions_ensemble.Label)),columns = ['PassengerId', 'Transported'])
sub.to_csv('submission_ensemble.csv', index = False)
sub.head()


# In[20]:


def plot_preds_dist(df, preds, target, ax=None, title=''):
    train_test_preds = pd.DataFrame()
    train_test_preds['label'] = list(df[target]) + list(preds)
    train_test_preds['train_test'] = 'Test preds'
    train_test_preds.loc[0:len(df[[target]]), 'train_test'] = 'Training'
    
    if ax==None:
        fig, ax = plt.subplots(figsize=(16,3))
        sns.countplot(data=train_test_preds, x='label', hue='train_test', ax=ax)
        ax.set_title(title);
    else:
        sns.countplot(data=train_test_preds, x='label', hue='train_test', ax=ax)
        ax.set_title(title);


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 1, figsize=(16, 8))
plt.subplots_adjust(hspace=0.5)
plot_preds_dist(train, unseen_predictions_stack.Label, "Transported", ax=axes[0], title="Stack")
plot_preds_dist(train, unseen_predictions_blend.Label, "Transported", ax=axes[1], title="Blend")
plot_preds_dist(train, unseen_predictions_ensemble.Label, "Transported", ax=axes[2], title="Ensemble")

