#!/usr/bin/env python
# coding: utf-8

#  # Introduction to Light GBM
# ## A Women in Kaggle Philly Workshop
# 
# 
# In this notebook, we will go through a kaggle competition together. This tutorial assumes that you have some basic knowlege of Kaggle and its competitions, statistical models, and Python. By the end of this notebook, you should be able to submit a prediction using Light GBM to the Talkdata competition. 
# 
# Please fork this notebook so that you can edit the codes to do the exercises.
# 
# ## What we will do today:
# 1. Brief Introduction 
# 2. Loading large dataset
# 3. Feature Engineering
# 4. Modeling & Evaluation
# 5. Feature Importance
# 6. Create Submission File
# 
# ## Target today:
# Submit your prediction file to the kaggle competition.
# 
# ## References
# This notebook is written for a workshop organized by the Women in Kaggle Philly Meetup group.
# 
# This notebook is built upon ideas from: 
# 
# https://www.kaggle.com/asraful70/talkingdata-added-new-features-in-lightgbm
# 
# https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask

# ## 1. Brief Introduction
# ### 1.1 About the competition
# 
# For this competition, your objective is to predict whether a user will download an app after clicking a mobile app advertisement. The training dataset is 1.21 GB, incuding records of 180 milion clicks with ip, app, device, os, channel, click_time as main features, and is_attributed (whether the app is downloaded or not) as the target variable to be predicted. Please refer to the [competition overview](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) for details.
# 
# This is a very big dataset and kaggle kernel cannot handle training with the full data. If you have a super computer, you can download the full dataset and train locally. However, it is still possible to participate the competition with a laptop. That is why we are going to learn about Light GBM today.
# 
# ### 1.2 A very brief introduction of Light GBM
# 
# To put it simple, Light GBM, or the Light Gradient Boosting Model, grows trees vertically while the other tree_based algorithems (like XGB) grow trees horizontally. Light GBM is designed for handling very large datasets with faster speed and lower memory (hence "light"). I do not recommend using light GBM on small datasets (e.g. fewer than 10,000 rows) because it tends to overfit. 
# 
# For more information on Light GBM, I recommend reading the following posts:
# 
# https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
# 
# https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# 
# https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
# 
# 

# ## 2. Loading Large Dataset
# 
# First we are going to load all libraries needed for this notebook. There are a few simple tricks to consider for loading very large datasets. 

# In[ ]:


# Load libraries
import numpy as np # linear algebra
import pandas as pd # data processing
from pandas import Series, DataFrame # to deal with time data
import gc # To collect RAM garbage
import time # To get current time, used to calculate model training time
from sklearn.model_selection import train_test_split # To split training and validation datasets
import matplotlib.pyplot as plt # For plotting feature importance
import lightgbm as lgb # Light gbm model
import warnings
warnings.filterwarnings('ignore') # Toignore warnings


# ### Trick 1: Set a debug mode
# 
# This is a really really large dataset. When we write our codes, we need to debug from time to time, but it would be extremely slow if we test every line of code with the whole dataset. In this case, we will set a "debugging mode" under which we only import a small part of the dataset for faster performance. Please make sure to set debug=1 throughout the workshop. 

# In[ ]:


# Set debug mode. When debug=1, we'll only be importing a few lines.
# When debug=0, we'll import a much larger dataset to do serious training.
# You'll see how to set this up in a later code block. 
# Make sure to set debug=1 throughout the workshop! 

debug=1


# ### Trick 2: Define data types before importing
# 
# If pandas does not know the data type for each feature, it will need to assign more RAM to handle them. Therefore, assinging data types beforehand will help save a lot of computational power. 

# In[ ]:


# Define data types
# uint32 is an unsigned integer with 32 bit 
# which means that you can represent 2^32 numbers (0-4294967295)
dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint8',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }


# ### Trick 3: Select columns before importing
# 
# 

# In[ ]:


# Only import columns you need: create a list before you actually read the data
train_cols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']


# In[ ]:


# Exercise: create a list called test_cols with the following feature: 
# 'ip','app','device','os', 'channel', 'click_time', 'click_id'




# Now, set up debug mode and import data 

# In[ ]:


# It takes a long time to load a large dataset, so we print a mark here just to keep track of the process
print("Loading training data...")

# Now reading the training and test data.
# Load only a few lines if debug=1; Load a much larger part of the dataset if debug=0
if debug:
    train = pd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time'], 
                        nrows=100000, usecols=train_cols)
    test = pd.read_csv("../input/test.csv", dtype=dtypes, parse_dates=['click_time'], 
                       nrows=100000, usecols=test_cols)
else: 
    train =  pd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time'], 
                         # skiprows=range(1,129903891), this will skip the first n rows
                         nrows=1000000, usecols=train_cols)
    test = pd.read_csv("../input/test.csv", dtype=dtypes, parse_dates=['click_time'], 
                       usecols=test_cols)
print ("Loading finished")


# ## 3. Feature Engineering
# From now on we are going to do some feature engineering. Whenever we add a new feature in the trainining dataset, we have to create the same feature in the test dataset. To avoid duplicate coding, we will combine the training and test datasets before we do anything with the features. 

# In[ ]:


# Exercise: Print a sentence to indicate that we are now "processing data"



# ### 3.1 Combine training and test data before feature engineering

# In[ ]:


# First, we have to get the length of the training data. 
# We'll need this number when we have to split the training and test data again
len_train = len(train)    

# Now append test data to training data and make a new data frame called full
full=train.append(test)   


# ### Trick 4: Delete large objects and use gc.collect()

# In[ ]:


# Now we have stored both training and test data in a new dataframe called full
# We can delete training and test data because we don't need them anymore and they are very large
del test  
del train 

# Collect any other temp garbage in the RAM. 
# It's a good habit to gc.collect() from time to time when you deal with large datasets
gc.collect() 


# In[ ]:


# Assign a list for predictors and target. We'll use these two lists very soon
predictors = ["ip","app","device","os","channel"]
target = "is_attributed"

# Create a list with names of categorical variables. LGBM can handle categorical variables smoothly.
categorical = ["ip", "app", "device","os", "channel"]


# ### 3.2 Extract time features
# 
# The first thing we can do with this dataset is to extract the time features. Remember "click_time"? It marked the day, hour, minute, and second when each click happened. We can extract those time features seperately. 

# In[ ]:


# Getting time features
# Get "day" from "click_time" and add it to "full" as a new feature
full['day'] = pd.to_datetime(full.click_time).dt.day.astype('int8')
# Append "day" to the predictor list
predictors.append('day')

# Get "hour" from "click_time" and add it to "full" as a new feature
full['hour'] = pd.to_datetime(full.click_time).dt.hour.astype('int8')
# Append "hour" to the predictor list
predictors.append('hour')


# In[ ]:


# Exercise: get "minute" and "second" from "click_time" and add them to "full" as new features


# Exercise: Append "minute" and "second" to the predictor list


# Note: sometimes it doesn't make sense to include all time features. For example, in this dataset, the "day" feature in the training and test datasets do not overlap at all (Monday to Thursday in train, Friday in test). Also, it is hard to image how "minute" and "second" would influence the probability of downloading. "Hour", on the other hand, may play a role (e.g. people may be more likely to download an app after working hours). This is just for practice. You can try to drop some of the time features later to see how the final score changes. 

# ### 3.3 Extracting time difference between two clicks
# 
# The next step is a little bit tricky. We're extracting one more feature called "next click", which calculates the time difference between next click and the current click from the same ip, same os, same device, and same channel (therefore very likely to be the same person). We doubt that if this time difference is too short (two clicks happen very fast), then the current click is more likely to be a fake one. 

# In[ ]:


# shift means we are shifting the whole column up by one row, which basically means we're getting the next value in line
# We subtract the time of current click from the time of next click, so we get the time difference between two clicks
# We convert this difference to seconds, and claim that its data type is "float32", which means real number with 32 bit
same=["channel", "app", "os", "device","ip"]
full['next_click'] = (full.groupby(same).click_time.shift(-1) - full.click_time).dt.seconds.astype('float32')

# Append "next_click" to the predictor list
predictors.append('next_click')


# In[ ]:


# Exercise: using very similar method, create a new feature called "prev_click", 
# indicating the time difference between the current click and the previuos click
# Hint: just change -1 to +1, and switch the two click_time


# ### 3.4 Creating grouped count features
# 
# Next we will create a feature calculating the number of clicks for each ip in the same hour on the same day. We assume that if there are too many clicks from the same ip, then these clicks might be fake. 

# In[ ]:


# Set the group. You can have other combinations
group = ['ip','day','hour']

# group by ip+day+hour, choose one column (click_time) to count the number of rows, 
# fill this number into the click_time variable, then change the column name to ip_day_hour
gp = full.groupby(group)["click_time"].count().reset_index().rename(index=str, columns={'click_time':"ip_day_hour"})

# merge back with full data
full = full.merge(gp, on=group, how='left')

# Append new variable name to the list "prdictors"
predictors.append("ip_day_hour")

# Delete gp and collect garbage
del gp
gc.collect()


# In[ ]:


# Exercise: Very similarly, create a new variable called ip_app_channel, 
# which calculates the total number of clicks for the same ip + app + channel
# Delete and collect garbage


# Note: you can also get grouped mean and variance features by changing count() to mean() or var(). We'll skip this stage today, but you are encouraged to try different features to see how they may affect the final score. 

# ### 3.5 Splitting training and test datasets 
# 
# We have finished our feature engineering today. If you want to create more features, do it before next step.
# 
# Remember that before feature engineering, we have merged the training and test datasets? Now it's time to split them again.
# 

# In[ ]:


# Split training and test data
train = full[:len_train]
test = full[len_train:]

# Set X(predictors) and y(target)
X = train[predictors]
y = train[target]
    
# Delete unused parts 
del train
gc.collect()


# ### 3.6 Splitting training and validation datasets
# 
# The next step is to split the training dataset into training and validation. This is to avoid overfitting. 

# In[ ]:


# Split training and validation data using train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 30)

# delete X and y since we don't need them anymore
del X, y

    


# ## 4. Modelling with Light GBM 
# 
# Finally we are ready for Light GBM! For an introduction of how this model works, please refer to [this link](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/). 
# 
# First, we'll define training and validation dataset again in the way that the lgbm library can understand: 

# In[ ]:


# lgb.Dataset defines the training and validation dataset
# this is a little bit confusing, but in lgb.Dataset, "label" means y(the target variable), 
# because our prediction is actually "labelling" the target
# We also define the feature names for feature importance plotting afterwards
xgtrain = lgb.Dataset(X_train.values, label=y_train.values, feature_name=predictors, categorical_feature = categorical)                          
xgvalid = lgb.Dataset(X_val.values, label=y_val.values, feature_name=predictor,categorical_feature = categoricals)


# Light GBM is a very complex model with tons of parameters to tune. You can leave them as default, but we'll have a brief look at those parameters here. 
# 
# For a very clear introduction (that speaks English) of what those parameters mean and what is the best approach, please refer to [this website](https://sites.google.com/view/lauraepp/parameters). Also check out [here](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst). 
# 
# Next block sets up basic lgb parameters. You can leave them as default if you don't know what to do with them. 

# In[1]:


# Setting lgb model parameters; not mandatory
lgb_params = {
        'boosting_type': 'gbdt', # Gradient Boosted Decision Trees
        'objective': 'binary', # Because we are predicting 0 and 1
        'metric': 'auc', # Method to evaluate the model, auc means "area under the curve". The lower the better
        'learning_rate': 0.03, #Basically the weight for each boosting iteration. A smaller learning_rate may increase accuracy but lead to slower training speed. 
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit. Too deep may lead to overfitting. 
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8, # Number of threads using for training models, better to set it large for large dataset
        'verbose': 0, # Do not affect training, just affect how detailed the information produced during training would be
        'random_state': 42
    }


# Let's officially start training! 

# In[2]:


# Training start! Print a marker for it. 
print("Training...")

# Set start_time as current time
start_time = time.time()

# Create an empty data frame for writing the evaluation results. 
evals_results={}

# This is the real lgb model training process
# There are more parameters to tune! I put those parameters that we are most likely to change here

model_lgb = lgb.train(lgb_params, # The list of parameters that we have already set
                xgtrain,  # The training dataset
                valid_sets= [xgtrain, xgvalid], # We produce evaluation score for both the training and validation dataset
                valid_names=['train','valid'],  # Assign names to the training and validation dataset
                early_stopping_rounds=100, # If there's no improvement after 10 rounds, then stop
                verbose_eval=50,  # Print evaluation scores every 10 rounds
                num_boost_round=5000, # Maximum 200 rounds, even if it does not meet the early_stopping_round requirement
                evals_result=evals_results) # Write evalution results into evals_results
                
# Print current time - start_time, this is the time used for training the model    
print('Model training time: {} seconds'.format(time.time() - start_time))

gc.collect()

# Exercise: change early_Stopping_round, verbose_eval, and num_boost_round, and train the model again
# Print our how much time the training took
# Note: you may need to run the revious block to reset xgtrain and xgvalid because lgbm has already processed categorical data


# ## 5. Feature Importance
# 
# After training the model, we can actually examine how important each feature is. 
# 
# In Light GBM, there are two ways to evaluate the importance of a feature: 
# 
#  1. “split”: the number of times a feature is used in a model 
#  2. “gain”: the total gains of splits which use the feature
# 
# When there are not too many features, gain is usually better.  It's possible that all models used a certain feature (therefore a high score for "split"),  but they only used it once for split in each model (therefore a low score for "gain"). 
# 

# In[ ]:


# List feature importance for all features
print("Features importance...")
gain = model_lgb.feature_importance('gain')
ft = pd.DataFrame({'feature':model_lgb.feature_name(), 
                   'split':model_lgb.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft)


# In[ ]:


# Plot feature importance using the "split" method
split = lgb.plot_importance(model_lgb, importance_type="split")
plt.show(split) # Show the plot
plt.savefig("feature_importance_split.png") # Save the plot in the output


# In[ ]:


# Exercise: Plot feature importance using the "gain" method


# ## 6. Predicting and submission
# 
# ### 6.1 Predicting for the test dataset
# 
# Now that we have trained our model, we will use it to predict the target variable using the predictors in the test dataset. 

# In[ ]:


# Print a mark here
print ("Predicting test data...")

# Creat X_test, which includes all features in the test dataset
X_test = test[predictors]

# Feed X_test to our trained "model_lgb" to predict the target variable (y) in the test dataset
# Store our prediction in ypred
ypred = model_lgb.predict(X_test,num_iteration=model_lgb.best_iteration)

gc.collect()


# ### 6.2 Writing the submission file
# 
# We are ready to write the submission file! 
# 
# However, remember that if you are in the debug mode, your predicted test file will be much shorter than the real test dataset, therefore you will get a warning message "Length of values does not match length of index" . To submit and get a score, you will need to set debug=0. That may take up to 2 hours to train the model, depending on the size of your training data, and how expensive your laptop is. 
# 
# Change debug = 0, adjust the nrows in your training data based on your laptop RAM and CPU, and hit "commit & run". You'll be able to see your output after a while. Submit it to the competition and see where you are on the Leader Board!
# 

# In[ ]:


# Print a mark
print ("Writing submission file...")

# Read the sample submission file
submission = pd.read_csv("../input/sample_submission.csv")

# Change the value in the prediction column into our prediction "ypred"
submission["is_attributed"] = ypred

# Write it into a csv file
submission.to_csv("submission.csv", index = False)

# Print a final mark
print ("Mission Completed")


# 
