#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
import numpy as np

def read_data(cols):
    
    print('Reading data...')
    
    df = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet', columns=cols)
    
    # simplify cus_id
    unique_cus_ids = df.customer_ID.unique()
    assignment     = dict(zip(unique_cus_ids, list(range(len(unique_cus_ids)))))
    df.customer_ID = df.customer_ID.apply(lambda x: assignment[x]).astype('int32')
    
    print('shape of data:', df.shape)
    
    return df

# method for Information Value
def iv_woe(data, target, bins=20, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:

        print(ivars)

        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>3):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
            
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


# # Smart Brute Force Feature Engineering

# Brute force feature engineering is a effective approach to extract new information from data. [thedevastator](https://www.kaggle.com/thedevastator) has shown in his [notebook](https://www.kaggle.com/code/thedevastator/amex-bruteforce-feature-engineering) that this method indeed is fast and improves the score of your model! However, I want to claim that this method is not efficient. Of course, you can compute all kind of features like:
# 
# - `P_2_Last -  P_2_Mean`
# - `B_6_Last /  B_6_Mean`
# - `S_14_Last  +  S_14_Mean`
# - `S_14_Last   /  S_14_Mean`
# - `P_2_Last  +  P_2_Mean`
# 
# You can even try to compute new features based on different feature aggregation like:
# 
# - `P_2_First  -  B_3_Last`
# - `S_14_First /  B_6_Last`
# - `P_2_Mean   +  B_6_std`
# - `B_3_Max    *  S_14_Min`
# - `B_3_Last   +  P_2_Mean^2`
# 
# **BUT HOW DO YOU KNOW WHETHER THIS IS NOT JUST NOISE?!?!?!**
# 
# Well the is where the 'smart' comes in! In my previous [notebook](https://www.kaggle.com/code/gzguevara/new-features-based-on-information-value) I have introduced the usage of "Information Values" to get a grasp of whether your feature contains good new information. I propose to compute all kind of crazy features, of which we cannot really know, whether they are of good information, and then apply the same approach, based on the "Information Value". This way we can select only those crazy features, which indeed contain useful information. 

# In[2]:


# Features which are know to have high information
high = ['P_2', 'P_3', 'P_4', 
        'D_48', 'D_42', 'D_44', 'D_61',
        'R_1', 'R_3', 'R_10', 'R_5', 'R_16',
        'S_3', 'S_7', 'S_15', 'S_22', 'S_8',
        'B_7', 'B_23', 'B_9', 'B_10', 'B_2']

# Get all possible combination of high information features
all_pairs = []
for i in range(len(high) -1): all_pairs.extend(list(itertools.product([high[i]], high[i+1:])))


# Now we have all possible pairs, based on which we can compute all kind of crazy features. As soon as they are computed, we evalute them based on their Information Value and if the Information Value is high enough, we keep this feature!

# In[3]:


# read only high-information features
data  = read_data(['customer_ID'] + high)
# read targets
target = pd.read_csv('../input/amex-default-prediction/train_labels.csv', usecols=['target'])

def smart_brute_force(info_cutoff):

    all_features = pd.DataFrame()

    for pair in all:

        # Basic aggregations
        group_a = train[['customer_ID', pair[0]]].groupby('customer_ID').agg(['last', 'first', 'median', 'mean', 'std', 'max', 'min'])
        group_b = train[['customer_ID', pair[1]]].groupby('customer_ID').agg(['last', 'first', 'median', 'mean', 'std', 'max', 'min'])
        group_a.columns = [x[1] + '_' for x in group_a.columns]
        group_b.columns = [x[1] + '_' for x in group_b.columns]

        # Combinations
        new_features = pd.DataFrame()

        # Crazy Features and much more are possible if you want - try you own!
        new_features[f'{pair[0]}_last_t_{pair[1]}_std']  = group_a.last_ * group_b.std_
        new_features[f'{pair[0]}_last_d_{pair[1]}_mean'] = group_a.last_ / group_b.mean_
        new_features[f'{pair[0]}_last_p_{pair[1]}_max']  = group_a.last_ + group_b.max_
        new_features[f'{pair[0]}_last_m_{pair[1]}_min']  = group_a.last_ - group_b.min_
        new_features[f'{pair[0]}_last_t_{pair[1]}_median']  = group_a.last_ * group_b.median_
        new_features[f'{pair[0]}_last_t_{pair[1]}_first']  = group_a.last_ * group_b.first_
        new_features[f'{pair[0]}_last_t_{pair[1]}_last']  = group_a.last_ * group_b.last_

        new_features[f'{pair[0]}_mean_t_{pair[1]}_std']  = group_a.mean_ * group_b.std_
        new_features[f'{pair[0]}_mean_d_{pair[1]}_mean'] = group_a.mean_ / group_b.mean_
        new_features[f'{pair[0]}_mean_p_{pair[1]}_max']  = group_a.mean_ + group_b.max_
        new_features[f'{pair[0]}_mean_m_{pair[1]}_min']  = group_a.mean_ - group_b.min_
        new_features[f'{pair[0]}_mean_t_{pair[1]}_median']  = group_a.mean_ * group_b.median_
        new_features[f'{pair[0]}_mean_t_{pair[1]}_first']  = group_a.mean_ * group_b.first_
        new_features[f'{pair[0]}_mean_t_{pair[1]}_last']  = group_a.mean_ * group_b.last_

        # Clean possible missing values and inf's
        new_features = new_features.fillna(0)
        new_features.replace([np.inf, -np.inf], 0, inplace=True)

        # Get Information Value
        new_features['target'] = targets.target
        a, b = iv_woe(new_features, 'target')

        # Select only new features with high information!
        good_ones = a.loc[a.IV > info_cutoff].Variable.values

        # Save new high-information features
        all_features[good_ones] = new_features[good_ones]

        print('\n', all_features.shape, '\n')
        
        return all_features
        
#new_features = smart_brute_force(2.65)


# The result of this method is a collection of over 200 features, from which you know, that they will contribute new & high information to your model! You can find dataset resulting from the method [here](https://www.kaggle.com/datasets/gzguevara/amex-smart-brute-force-features). If you have any further questions - let me know! 
# 
# NOTE! including 20 features, which are base on P_2_last is useless! The information contained in those 20 features will be too similar! 
# 
# Imagine you are at police. Mrs. Smith has been found dead in the forrest. You are telling the police that you saw Mr. Smith on the day before in the forrest. On the next day you go to the police again and tell them: "I saw Mr. Smith and he had a black shoes". On the next day you go to the police again and tell them: "I saw Mr. Smith and he had green pants". etc... The information you are givin to the police is helpfull, yes! But all those little variation do not add significant *new* information! 
# 
# **You need to find features with high information, but also with different information!**

# Hee are the resulting new features:

# In[4]:


pd.read_parquet('../input/amex-smart-brute-force-features')

