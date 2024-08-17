#!/usr/bin/env python
# coding: utf-8

# ONE OF THE MOST IMPORTANT TASKS IN A KAGGLE COMPETITION IS HYPER-PARAMETER TUNING.<br>THIS NOTEBOOK DEMONSTRATES HOW TO USE WEIGHTS AND BIASES FOR HYPER-PARAMETER TUNING
# 
# WE USED OUTPUTS OF THIS NOTEBOOK FOR DATA PURPOSES : [AMEX - Data Preprocesing & Feature Engineering](https://www.kaggle.com/code/susnato/amex-data-preprocesing-feature-engineering)
# 

# ## WHAT IS HYPER-PARAMETER SWEEP?
# Hyper-Parameter Sweep is a process where you run the model with different sets of hyper-parameter and see which works the best. It might be tiring to keep track of models logs, thats where Weight & Biases comes in.  
# 
# ## How does Weights & Biases Hyper-Parameter Sweep work?
# 
# There are two parts of a WandB Sweep. The first one is a `SWEEP CONTROLLER` and the other one is `SWEEP AGENTS`.
# 
# <u>`SWEEP CONTROLLER`</u> : This is the main controller which keep tracks of the whole Sweep process. WandB provides a sweep controller by themselves (you can setup your own sweep controller as well on your local machine but here we are using thiers). When a sweep is run the sweep controller automatically computes all the possible sets of hyper-parameters and it gives the SWEEP AGENTS, a particular set of hyper-parameters then after Sweep Agents have evaluated the model on those parameters, they give back those logs(metrics, losses and system information) to the Sweep Contoller. 
# 
# <u>`SWEEP AGENTS`</u> : This can be any computer from our end which is capable of computing the models training logs, when the Sweep Controller asks it to. One of the best things about WandB is that, you can add as many sweep agents as you like. For example, if you are competing in a group of 5 then all of the members can be Sweep Agents at the same time (parallelly running the Sweeps). This drastically reduces the time required for Sweep.
# 
# ![](https://i.postimg.cc/hvDcB0xV/WANDB-controller-and-agents.png)
# 
# 
# ## THERE ARE 3 STEPS YOU NEED TO DO TO RUN A HYPER-PARAMETER SWEEP :-
# 
# 1. **Define the sweep:** we do this by creating a dictionary that specifies the parameters to search through, the search strategy and the optimization metric.
# 
# 2. **Initialize the sweep:** with one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
# `sweep_id = wandb.sweep(sweep_config)`
# 
# 3. **Run the sweep agent:** also accomplished with one line of code, we call `wandb.agent()` and pass the `sweep_id` to run, along with a function that defines your model architecture and trains it:
# `wandb.agent(sweep_id, function=train)`
# 

# First we will install the latest version of WandB 

# In[1]:


get_ipython().system('pip install wandb --upgrade -q')


# In[2]:


import os
import gc
import glob
import tqdm
import numpy as np
import pandas as pd


# Then import and Login with your WANDB Account (If you don't have one go to https://wandb.ai/site and click Sign Up)
# 
# I have my wandb `authorization key` saved in Secrets so I am using it from there. But you can just write `wandb.login()` and it will ask you to go to https://wandb.ai/authorize to get your key and then paste it in the box.

# In[3]:


import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb_api")
wandb.login(key=wandb_key)


# In[4]:


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
train_labels['customer_ID'] = train_labels['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype(np.int64)
train_labels = train_labels.set_axis(train_labels['customer_ID'])
train_labels = train_labels.drop(['customer_ID'], axis=1)

train_pkls = sorted(glob.glob('../input/amex-data-preprocesing-feature-engineering/train_data_*'))
train_y = sorted(glob.glob('../input/amex-data-preprocesing-feature-engineering/train_y_*.npy'))
test_pkls = sorted(glob.glob('../input/amex-data-preprocesing-feature-engineering/test_data_*'))

useful_features = np.load('../input/amexxgboost-usefulfeatures/useful_features_4.npy')


# ## Data Preparation
# 
# One of the most important thing to remember when running WandB Sweep is that, the data need to be consistent throughout the whole sweep and also same for all Sweep Agents. Because even slighest difference in data can lead to different model results so the Hyper Parameters set won't be properly evaluated.
# 
# Here I am using the output from the notebook I created for Feature Engineering : [AMEX - Data Preprocesing & Feature Engineering](https://www.kaggle.com/code/susnato/amex-data-preprocesing-feature-engineering)
# 
# Then I am dividing the data(80%-20%) with keeping the seed same. 

# In[5]:


from sklearn.model_selection import train_test_split

train_df = pd.read_pickle(train_pkls[0])
print(train_pkls[0])
for i in train_pkls[1:]:
    print(i)
    train_df = train_df.append(pd.read_pickle(i))
    gc.collect()
    
y = train_labels.loc[train_df.index.values].values.astype(np.int8)
train_df = train_df.drop(['D_64_1', 'D_66_0', 'D_68_0'], axis=1)
train_df = train_df[useful_features]

X_train, X_val, y_train, y_val = train_test_split(train_df, y,
                                                    stratify=y, 
                                                    test_size=0.20,
                                                    random_state=SEED)
print(train_df.shape, X_train.shape, X_val.shape, y_train.shape, y_val.shape)
del train_df, y
gc.collect()

print(X_train.info(), X_val.info())


# In[6]:


import xgboost as xgb

def amex_metric(y_true, y_pred):
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


# ## Step : 1 (Define the sweep)
# 
# Here we define everything about the Sweep.
# * First we are defining out metric (`model_score`) and setting out goal as `maximize`.
# * Then we are defining the Parameters with the values that we want to evaluate our sweep on.
# * Then at last we are defining out sweep method. There are 3 methods that we can try, those are `random`, `grid` and `bayesian` .  We are sticking with `random` search.

# In[7]:


import pprint


metric = {
    'name': 'model_score',
    'goal': 'maximize'   
    }
parameters_dict = {
    'n_estimators': {
        'values': [1000, 1500]
                    },
    'max_depth': {
        'values': [1, 2, 3, 4]
                 },
    'learning_rate': {
          'values' : [0.05, 0.07, 0.1, 0.2, 0.4, 0.6, 0.8]
                     },
    'subsample':{
          'values':[0.5, 0.7, 0.9]
                },
    'colsample_bytree':{
          'values':[0.4, 0.5 , 0.7 , 1.0]
                       },
    'min_child_weight':{
          'values':[ 3, 5, 7 ]    
                       },
    'reg_alpha': {
          'values':[0.0, 0.5, 1.0, 2.0]
                 },
    'reg_lambda':{
          'values':[0.0, 0.5, 1.0, 2.0]
                 },
    }

sweep_config = {
    'method': 'random'
    }
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)


# ## Step : 2 (Initialize the sweep)
# 
# With one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
# `sweep_id = wandb.sweep(sweep_config)`

# In[8]:


sweep_id = wandb.sweep(sweep_config, project="AMEX-XGBoost-Sweep")
print(sweep_id)


# If there is already a sweep going on and you want to add another Sweep Agent(device) to it then just use `<your_wandb_name>/<project name>/<previous sweep id>` as sweep id.
# 
# For example I have just defined this sweep but I already have a sweep defined which already has ran for 5 runs.
# 
# 
# ![](https://i.postimg.cc/nrsL9kRB/Screenshot-2022-06-13-at-08-36-46-Weights-Biases.png)
# 
# 
# If I want to use that sweep(continue using that) then instead of defining this I will write <br> `sweep_id = "susnato/AMEX-XGBoost-Sweep/jlmmfd10"`

# ## Step : 3 (Run the Sweep Agent)
# 
# 

# ## Now we define the train() function 
# 
# When a sweep is called the Sweep Agent runs this `train()` function. This function must have all the important things like creating a model with the sets of parameters given by the Sweep Controller, training the model, logging the metrics-losses, even sometimes saving some useful files as model, feature_importance for future.
# 
# Here, **`wandb.init()`** is used to Initialize a new W&B Run. Then we defined a default config for our Sweep(This will be replaced by the set of hyper-parameters in each run), then
# **`wandb.config`** is used to get the new set of hyper-parameter for each run. Then after running and evaluating the model we used **`wandb.log()`** to log necessary metrics like `model_score` and save files like `feature_importance` and the model itself. For more details about `wandb.log()` please refer to this [notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb) .

# In[9]:


def train():
    config_defaults = {
        'n_estimators' : 500,
        'max_depth' : 3,
        'learning_rate' : 0.08, 
        'subsample' : 1,
        'colsample_bytree' : 1, 
        'min_child_weight' : 2,
        'reg_alpha' : 1,
        'reg_lambda' : 2,
      }

    wandb.init(config=config_defaults) 
    config = wandb.config
    model = xgb.XGBClassifier(n_estimators = config.n_estimators, max_depth = config.max_depth, 
                          learning_rate = config.learning_rate, subsample = config.subsample,
                          colsample_bytree = config.colsample_bytree, min_child_weight = config.min_child_weight,
                          reg_alpha = config.reg_alpha, reg_lambda = config.reg_lambda,
                          eval_metric = amex_metric, random_state = 42,        
                          tree_method ='gpu_hist', predictor = 'gpu_predictor')
    model.fit(X_train, y_train,
             eval_set=[(X_train, y_train), (X_val, y_val)],
             early_stopping_rounds=None,
             verbose=50,
             )
    
    #wandb.log({"train_amex_metric": model.evals_result()['validation_0']['amex_metric']})
    #wandb.log({"val_amex_metric": model.evals_result()['validation_1']['amex_metric']})
    
    val_score = amex_metric(y_true=y_val.reshape(-1, ), y_pred=model.predict_proba(X_val)[:, 1].reshape(-1, ).astype(np.float32))
    wandb.log({"model_score": val_score})
    
    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    feature_important_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    
    model.save_model("XGB_model_{}.xgb".format(val_score))
    feature_important_df.to_pickle("XGB_model_{}_feature_importance.pkl".format(val_score))
    
    wandb.save("XGB_model_{}.xgb".format(val_score))
    wandb.save("XGB_model_{}_feature_importance.pkl".format(val_score))
    
    del val_score, feature_important_df, keys, values, model
    gc.collect()


# ## Here we launch our sweep
# Since we are using `random` search, the Sweep will keep going forever so we need to restrict the number of times we want out sweep to run, we defined that using `count=5` .

# In[10]:


wandb.agent(sweep_id, train, count=5)


# ## Visualization
# 
# If you click on the link given in the outputs it will redirect you to the sweep page. For example you can go to this sweep we just ran, [link](https://wandb.ai/susnato/AMEX-XGBoost-Sweep/sweeps/194gg9yt?workspace=user-susnato)
# I have made it public so all of us can access it.
# 
# Here are some example plots you will find there,
# 
# ![](https://i.postimg.cc/kgzJbK3d/Screenshot-2022-06-13-at-09-54-05-Weights-Biases.png)
# 
# ![](https://i.postimg.cc/xCynhZk0/Screenshot-2022-06-13-at-09-53-41-Weights-Biases.png)
# 
# ![](https://i.postimg.cc/9f9Wt0Cw/Screenshot-2022-06-13-at-09-54-36-Weights-Biases.png)
# 
# ![](https://i.postimg.cc/fLhMHsR2/Screenshot-2022-06-13-at-09-54-48-Weights-Biases.png)
# 

# ## CONCLUSION
# 
# Weights and Biases is a great tool for saving important logs during training. I bet that most of us at least have a laptop(5 or 6 years old which can barely run Chrome) sitting in a corner eating dust. But with the help of Weights & Biases we can now use that to Tune our model's Hyper-ParaParamameters; even can open multiple google accounts and use multiple colabs at same time. And the best part is that WandB will take care of the Hardest Part.

# In[ ]:




