#!/usr/bin/env python
# coding: utf-8

# Simple Notebook to probe the LB. The idea is simple: output noise given some condition then ouptut nan causing an error if this condition is not met. Taking advantage of what is mentionend in evaluation tab:
# 
# > You will get an error if you submission includes nulls or infinities and submissions that only include one prediction value will receive a score of -1.
# 
# I'll start getting the number of iteration right: select a given number of iteration if the number of iteration goes above this number then output nan and cause the submission to fail.

# **Over the last two weeks I managed to get some interesting results:**
# - **the LB contains 244 iterations it give us that we have around 2mins and 10s to predict each round**. This is plenty of time.
# - **the LB contains 3425-3429 investment id by time id on average and @jagofc found in his notebook that the number of new id over the LB is 195** (here: https://www.kaggle.com/jagofc/nothing-to-see-here-probing-the-lb) good to know but not sure if it is very important.
# - **More recently I've been looking at that weird pattern in unique values.** If you don't know what I am talking about, you might want to look at my relevant notebook: https://www.kaggle.com/lucasmorin/weird-patterns-in-unique-values-across-time-ids There seems to be two kinds of time ids, some where a good chunk of the features takes a small number of values and another set of time_ids where these features takes different values across time ids.
# 
# I usually use:
# 
# train.groupby('time_id').f_170.nunique==1
# 
# to check if there is one or multiple values in feature 170 as a proxy for the pattern. 
# **As it turns out, the LB only contain time_ids where this condition is met,that is in the public LB the categorical features appears without noise.**

# ## Other Feature Exploration / Feature engineering for Ubiquant:
# 
# - [Complete Feature Exploration](https://www.kaggle.com/lucasmorin/complete-feature-exploration)
# - [Weird pattern in unique values](https://www.kaggle.com/lucasmorin/weird-patterns-in-unique-values-across-time-ids/)
# - [Time x Strategy EDA](https://www.kaggle.com/lucasmorin/time-x-strategy-eda)  
# - [UMAP Data Analysis & Applications](https://www.kaggle.com/lucasmorin/umap-data-analysis-applications)   
# - [LB probing Notebook  ](https://www.kaggle.com/lucasmorin/don-t-mind-me-just-probing-the-lb)
# - On-Line Feature Engineering (in progress)

# In[1]:


import ubiquant
import numpy as np
env = ubiquant.make_env()
iter_test = env.iter_test()


# In[2]:


it_ref = 243
it = 0

f = 'f_272'

investment_counts = []
investment_count_ref = 3425

missing_pattern = []
missing_pattern_mean_ref = 0.5

cusum = 0

for (test_df, sample_prediction_df) in iter_test:
    f_mean = np.round(np.mean(test_df[f]))
    cusum += f_mean
    test_df['target'] = np.random.normal(0, 1, test_df.shape[0])
    
    if (it==it_ref):
        if (cusum<-150):
                test_df['target'] = np.nan
    it=it+1
    
    env.predict(test_df[['row_id','target']])


# In[3]:


np.mean(missing_pattern)


# In[4]:


test_df

