#!/usr/bin/env python
# coding: utf-8

# <img width="100%" src="https://images.creativemarket.com/0.1.0/ps/2930173/910/607/m1/fpnw/wm0/credit-card-design-by-jan-baca-.jpg?1499199821&s=9dbd168ee67bf12b45e533800f1f9aed&fmt=webp?1499199821&s=772da0dd5276e72af0dc92f36c2f8149?auto=compress&cs=tinysrgb&w=200&h=200&dpr=1">
# <figcaption>Pop–Art Diamond Bank Card Design</figcaption>

# # Comments
# This Notebook is very similar to my previous notebook ([https://www.kaggle.com/code/schopenhacker75/lightgbm-catboost-0-8-end2end-study?scriptVersionId=106234992](https://www.kaggle.com/code/schopenhacker75/lightgbm-catboost-0-8-end2end-study?scriptVersionId=106234992) whose the feture engineering part is inspired from [Martin's gret notebook](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977). In this notebook I used different feature engineering.
# 
# > **Table of Contents:**
# > * [🚫 Introduction on Payment Default](#1)
# > * [🙏 Credits](#2)
# > * [🛠 Feature Engineering](#3)
# > * [🥢 Feature Selection](#4)
# > * [🤖 Model Training](#5)
# > * [🏅 Feature Importance](#6)
# > * [📈 Cross-Val Roc Curves](#7)
# > * [🤞 Predict Test data & Submission](#8)
# > ---

# <a id="1"></a>
# # <div style='display:fill;color:#2d3a41;background-color:#a3d3eb;padding:20px'>   🚫  <b> Introduction on Payment Default </b> </div>

# Even though credit cards presents many advantages such us avoiding carrying a bulky wallet in your pocket, tracking the spending behaviour, fraud detection... On the other hand the major downside for credit card usage is the increasing tendency to default on their payments. Aggressive marketing strategies can encourage credit card use beyond payment capacity, thus increasing the bearer’s credit risk and resulting in defaults and losses that might have not been properly anticipated

# <a id="2"></a>
# # <div style='display:fill;color:#2d3a41;background-color:#a3d3eb;padding:20px'>   🙏 <b> Credits </b> </div>

# ##### In this notebook we build and train a LightGBM model using [@raddar'dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) (discussion [here](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514) and my notebook explanation [here](https://www.kaggle.com/code/schopenhacker75/data-deanonymization) )
# 
# The feature engineering part  is very inspired from [this notbook](https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created) and the design aspect s from [my previous notebook](https://www.kaggle.com/code/schopenhacker75/fancy-complete-eda/notebook) whose it self ispired from [this notebook](https://www.kaggle.com/code/kellibelcher/amex-default-prediction-eda-lgbm-baseline/notebook). The LightGBM implementation is inspired from [jayjay's notebook](https://www.kaggle.com/code/jayjay75/wids2020-lgb-starter-adversarial-validation)

# In[1]:


## ESSENTIALS
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import sys
import pickle
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
import random
random.seed(75)
from tqdm.notebook import tqdm_notebook 
from functools import partial, reduce

### warnings setting
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

#### model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import catboost
from catboost import Pool, CatBoostClassifier 
import lightgbm as lgb
import joblib
import pickle
from tqdm.notebook import tqdm_notebook 
import uuid

##### LOGGING Stettings #####
import logging
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',)
handler.setFormatter(formatter)
# Set STDERR handler as the only handler 
logger.handlers = [handler]

#### plots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors

sns.set(rc={'axes.facecolor':'#f9ecec', 'figure.facecolor':'#f9ecec'})

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode

### Plotly settings
theme_palette={
    'base': '#a3d3eb',
    'complementary':'#ebbba3',
    'triadic' : '#eba3d3', 
    'backgound' : '#f6fbfd' 
}

temp=dict(layout=go.Layout(font=dict(family="Ubuntu", size=14), 
                           height=600, 
                         legend=dict(#traceorder='reversed',
                            orientation="v",
                            y=1.15,
                            x=0.9),
                    plot_bgcolor = theme_palette['backgound'],
                      paper_bgcolor = theme_palette['backgound']))


SAVED=True



# <a id="3"></a>
# # <div style='display:fill;color:#2d3a41;background-color:#a3d3eb;padding:20px'>   🛠 <b>  Feature Engineering </b> </div>
# 
# <h4 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐    <b> a - Extract date based features </b> </h4>
# 
# The `S_2` is obviously a date type feature we'll extract the date related features such as the **month** and the **day of week**
# 
# <h4 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐    <b> b - Row Rise aggregation based features </b> </h4>
# 
# For each group of variables (Delinquency, Spend, Payment, Balance, Risk variables) we apply agg functions `[mean, sum]` at each row, as well as **count of missing values** by row
# 
# <h4 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐    <b> c - Column Rise aggregation based features</b> </h4>
# 
# Group by `custom_ID` and apply `['mean', 'std', 'min', 'max', 'last']` function for each group of variables
# 
# <h4 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐    <b> c -  Difference based features</b> </h4>
# 
# It [has been shown](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977/notebook) that these features carry a powerful predictive signal:
# 
# * the difference between the last transaction and the second of last
# * the difference between the last transaction and the mean

# <h3 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    <b>-- Get Train Data --</b> </h3>

# In[2]:


if not(SAVED):
    train = pd.read_parquet('/kaggle/input/amex-data-integer-dtypes-parquet-format/train.parquet')
    print(f"Shape = {train.shape}, number of customers = {train['customer_ID'].nunique()}")
    gc.collect()


# <h3 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    <b>-- Feature Engineering --</b> </h3>

# Lets define features by category types:
# * **Delinquency variables** : features starting with **D_**
# * **Spend variables** : features starting with **S_**
# * **Payment variables** : features starting with **P_**
# * **Balance variables** : features starting with **B_**
# * **Risk variables** : features starting with **R_**
# 
# With the following categorical features = `['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']`

# In[3]:


if not(SAVED):
    ## Define some features by category
    features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
    cat_vars = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68', 'month', 'day_of_week' ]
    num_vars = list(filter(lambda x : x not in cat_vars, features))

    # devide nums vars by AmEx Categories
    delequincy_vars = filter(lambda x:x.startswith('D') and x not in cat_vars, features)
    spend_vars = filter(lambda x:(x.startswith('S')) and (x not in cat_vars),features)
    payment_vars = filter(lambda x:x.startswith('P') and x not in cat_vars, features)
    balance_vars = filter(lambda x:x.startswith('B') and x not in cat_vars, features)
    risk_vars = filter(lambda x:x.startswith('R') and x not in cat_vars, features)

    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

    with open('cat_vars.pkl', 'wb') as f:
        pickle.dump(cat_vars, f)

    with open('num_vars.pkl', 'wb') as f:
        pickle.dump(num_vars, f)



# To address the **OOM issues**, I inplemented a batch generator to apply the data processing batch by batch then cancatenate it all. Please note that most of the used aggregation functions are **grouped by customer_ID**, so **all the statements of a given client must be grouped on a single batch.**

# In[4]:


if not SAVED:
    class BatchGenerator:
        def __init__(self, df, batch_feature='customer_ID', keep_features=features, n_batchs=750):
            self.df = df
            self.batch_feature = batch_feature
            self.keep_feature = list(set([batch_feature]+keep_features))
            self.n_batchs = n_batchs

        def __iter__(self):
            unique_vals = self.df[self.batch_feature].unique()
            batch_size = int(np.ceil(len(unique_vals) / self.n_batchs))
            groups = self.df.groupby(self.batch_feature).groups
            n_batchs= min(self.n_batchs, int(np.ceil(len(unique_vals) / batch_size)))
            for i in range(n_batchs):
                keys = unique_vals[i*batch_size:(i+1)*batch_size]
                idx=[i for s in keys for i in groups[s] ]
                if i == n_batchs-1:
                    keys = unique_vals[(i+1)*batch_size:]
                    idx = idx + [i for s in keys for i in groups[s] ]
                yield self.df.loc[idx, self.keep_feature]



    week_days ={1: 'Mon', 2: 'Tue', 3: 'Wen', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}

    def extract_date_vars(df, date_var='S_2', sort_by=['customer_ID','S_2'], week_days=week_days):
        # change to datetime
        df[date_var] = pd.to_datetime(df[date_var])
        # sort by custoner ther by date 
        df = df.sort_values(by=sort_by)
        # extract some date characteristics
        # year has not a very
        # month
        df['month'] = df[date_var].dt.month
        # day of week
        df['day_of_week'] = df[date_var].apply(lambda x : x.isocalendar()[-1])
        return df

    group_names = ["delequincy_vars", "spend_vars", "payment_vars", "balance_vars", "risk_vars"]
    # row rise aggregation
    def row_rise_aggregation(df, 
                             group_vars=[delequincy_vars, spend_vars, payment_vars, balance_vars, risk_vars],
                            group_names=group_names,
                            save=True):
        print('shape before row_rise_aggregation', df.shape )
        for group_name, group_var in zip(group_names, group_vars):
            df[group_name+'_sum'] = df[group_var].sum(axis=1)
            df[group_name+'_mean'] = df[group_var].mean(axis=1)
            df[group_name+'_missing'] = df.isnull().sum(axis=1)
        print('shape after row_rise_aggregation', df.shape )
        if save:    
            df.reset_index(drop=False).to_feather(f"row_agg_{str(uuid.uuid4())}.ftr")
            return df['customer_ID'].nunique()
        return df

    def column_rise_aggregation(df, num_vars=num_vars, cat_vars=cat_vars, save=True):
        print('shape before column_rise_aggregation', df.shape )
        group_names = filter(lambda x: '_vars' in x, df.columns)
        num_agg = df.groupby("customer_ID")[list(set(list(num_vars) + list(group_names)))].agg(['mean', 'std', 'min', 'max', 'last'])
        num_agg.columns = ['_'.join(x) for x in num_agg.columns]

        cat_agg = df.groupby("customer_ID")[list(set(list(cat_vars)+['month', 'day_of_week']))].agg(['count', 'last', 'nunique', pd.Series.mode])
        cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]

        mode_cols = filter(lambda x:x.endswith('_mode'), cat_agg.columns)
        for col in mode_cols:
            cat_agg[col] = cat_agg[col].apply(lambda x: random.choice(str(x).strip('[]').split()))
        #concat the two dataframes
        df = pd.concat([num_agg, cat_agg], axis=1)
        del num_agg, cat_agg

        gc.collect()
        print('shape after column_rise_aggregation', df.shape )
        if save:
            df.reset_index(drop=False).to_feather(f"col_agg_{str(uuid.uuid4())}.ftr")
            return len(df)#df['customer_ID'].nunique()
        return df


    # from https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
    def get_difference(df, num_features):
        res = []
        customer_ids = []
        for customer_id, df in tqdm_notebook(df.groupby(['customer_ID'])):
            # Get the differences
            diff_df = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
            # Append to lists
            res.append(diff_df)
            customer_ids.append(customer_id)
        # Concatenate
        res = np.concatenate(res, axis = 0)
        # Transform to dataframe
        res = pd.DataFrame(res, columns = [col + '_diff1' for col in df[num_features].columns])
        # Add customer id
        res['customer_ID'] = customer_ids
        print('final shape', res.shape)
    #       df = df.merge(res, on='customer_ID', how='inner')
    #       df.reset_index(drop=False).to_feather(f"diff_{str(uuid.uuid4())}.ftr")
        return res#df['customer_ID'].nunique()

    c=0
    def save_partition(df, prefix='train'):
        global c
        df.reset_index(drop=True).to_feather(f'{prefix}_{c}.ftr')
        c=c+1
        return df['customer_ID'].nunique()


# In[5]:


N_BATCHS = 100
c=0
if not SAVED:
    print(train.shape, train.customer_ID.nunique())
    samples_df = BatchGenerator(train, batch_feature='customer_ID', keep_features=features+['S_2'], n_batchs=N_BATCHS)
    processed_elements = sum(map(partial(save_partition, prefix='train'), tqdm_notebook(samples_df, total=N_BATCHS)))
    
    del train
    gc.collect()


# In[6]:


processed = []

if not SAVED:       
    n_paths = 0
    for path in tqdm_notebook(glob.glob('train_*.ftr')):
        # apply on train
        sample_df = pd.read_feather(path)
        diff_df = get_difference(sample_df, num_features=num_vars)
        sample_df = sample_df.merge(diff_df, on='customer_ID', how='inner')
        
     #   if sample_df.shape[1]<300:
        sample_df = extract_date_vars(sample_df)
        all_num_vars = num_vars + list(map(lambda x: x+'_diff1', num_vars))
        sample_df = row_rise_aggregation(sample_df, save=False)
        sample_df = column_rise_aggregation(sample_df, num_vars=all_num_vars, save=False)
        
        print("diff between last and mean transaction")
        for col in tqdm_notebook(num_vars):
            try:
                sample_df[f'{col}_last_mean_diff'] = sample_df[f'{col}_last'] - sample_df[f'{col}_mean']
            except:
                pass

        sample_df = sample_df.reset_index()
        
        sample_df.reset_index(drop=True).to_feather(path)
        print("save processed", path)

        
    train = pd.concat(map(lambda sample_df: pd.read_feather(sample_df), tqdm_notebook(glob.glob('train_*.ftr'))))
        
        
    


# <h3 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    <b>-- Join with Labels--</b> </h3>

# In[7]:


if not(SAVED):
    ## Left join with labels:
    labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')
    print(labels.shape, labels['customer_ID'].nunique())
    labels = labels.set_index('customer_ID')
    train = train.set_index('customer_ID')
    train['target'] = labels['target']
    # del labels
    gc.collect()
    # save result
    train.reset_index().to_feather('feat_eng_agg_train_with_diff.ftr')



# <h2 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    -- Apply preprocess pipeline on Test Dataset -- </h2>
# 

# In[8]:


N_BATCHS = 300
c=0
if not SAVED:
    test = pd.read_parquet('/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet')
    n_cid = test.customer_ID.nunique()
    print(test.shape, n_cid )
    samples_df = BatchGenerator(test, batch_feature='customer_ID', keep_features=features+['S_2'], n_batchs=N_BATCHS)
    processed_elements = sum(map(partial(save_partition, prefix='test'), tqdm_notebook(samples_df, total=N_BATCHS)))
    
    del test
    gc.collect()


# In[9]:


processed = 0

if not SAVED:       
    for path in tqdm_notebook(glob.glob('test_*.ftr')):
        # apply on train
        sample_df = pd.read_feather(path)
        diff_df = get_difference(sample_df, num_features=num_vars)
        sample_df = sample_df.merge(diff_df, on='customer_ID', how='inner')
        
     #   if sample_df.shape[1]<300:
        sample_df = extract_date_vars(sample_df)
        all_num_vars = num_vars + list(map(lambda x: x+'_diff1', num_vars))
        sample_df = row_rise_aggregation(sample_df, save=False)
        sample_df = column_rise_aggregation(sample_df, num_vars=all_num_vars, save=False)
        
        print("diff between last and mean transaction")
        for col in tqdm_notebook(num_vars):
            try:
                sample_df[f'{col}_last_mean_diff'] = sample_df[f'{col}_last'] - sample_df[f'{col}_mean']
            except:
                pass

        sample_df = sample_df.reset_index()
        
        sample_df.reset_index(drop=True).to_feather(path)
        print("save processed", path)
        
        processed += sample_df.customer_ID.nunique()
        
    test = pd.concat(map(lambda sample_df: pd.read_feather(sample_df), tqdm_notebook(glob.glob('test_*.ftr'))))
    test.reset_index(drop=True).to_feather('feat_eng_agg_test_with_diff.ftr')


# In[10]:


# if saved extract the data directly the prepared data anad save time
if SAVED:
    train = pd.read_feather('../input/feat-eng-with-diff/feat_eng_agg_train_with_diff.ftr').set_index('customer_ID')


# In[11]:


def reduce_size(df):
# Transform float64 columns to float32
    print("reduce float data size")
    cols = list(df.dtypes[df.dtypes == 'float64'].index)
    for col in tqdm_notebook(cols):
        df[col] = df[col].astype(np.float32)
    # Transform int64 columns to int32
    print("reduce cat data size")
    cols = list(df.dtypes[df.dtypes == 'int64'].index)
    for col in tqdm_notebook(cols):
        df[col] = df[col].astype(np.int32)
    return df
        
train = reduce_size(train)


# <a id="4"></a>
# # <div style='display:fill;color:#2d3a41;background-color:#a3d3eb;padding:20px'>   🥢 <b> Feature Selection </b> </div>
# For the feature selection part I used some **filter based techniques** because these methods are faster and less computationally expensive than other feature selection methods such as wrapper methods. Basically:
# 
# * Drop Features with high missing values rate
# * Keep the TOP target correlated features
# 
# 
# <h2 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    -- Drop Features with high missing rate -- </h2>
# 
# All features having **more than 80% of missing values are ommited**

# In[12]:


def missing_values_table(df):
    # Total missing values by column
    mis_val = df.isnull().sum()

    # Percentage of missing values by column
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # build a table with the thw columns
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Missing values for training data
missing_values_train = missing_values_table(train)
#cm = sns.color_palette('Set2', as_cmap=True)
#missing_values_train[:20]#.style.background_gradient(cmap=cm)
THRESHOLD = 80
print(train.shape)
drop_cols = missing_values_train[missing_values_train['% of Total Values']>THRESHOLD].index.to_list()
print(f"Drop {len(drop_cols)} features with more than {THRESHOLD}% of missing values")
train = train.drop(drop_cols, axis=1)
print("Training data shape after dropping highly missing values columns", train.shape)


# 
# <h2 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    -- Select top correlated features with the target-- </h2>
# 
# With this method we assume that high predictive features are **highly correlated with the target**
# 
# 

# In[13]:


corr = train.corrwith(train['target'], axis=0)
corr = corr[corr.notna()].sort_values(key=abs, ascending=False)
THRESHOLD = 0.15
CORR_SELECTION=True
if CORR_SELECTION:
    selected_feats = corr[corr.abs()>THRESHOLD].index
    train = train[list(selected_feats)]
    print(f"Training data shape after dropping uncorrelated features"
          f"(threshold Pearson correlation = {THRESHOLD})", 
          train.shape)

    


# In[14]:


gc.collect()


# Let's display the top correlated features to the target 

# In[15]:


sorted_corr = corr.sort_values(key=abs, ascending=False)[:11] # top but we have to  drop corr=1


pos = sorted_corr[(sorted_corr>0) & (sorted_corr<1)]
neg = sorted_corr[(sorted_corr<0)].sort_values(ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(x=pos.index, y= pos.values,
                     orientation='v',
                     name='Positive',
                     marker=dict(color=theme_palette['base'],line=dict(color=theme_palette['complementary'],width=0)),
                     text = ["%.2f" %(round(v ,2) *100) + '%' for v in pos.values],
                     textposition = 'outside',
                     textfont_color = '#212a2f'))

fig.add_trace(go.Bar(x=neg.index, y= neg.values,
                     orientation='v',
                     name='Negative',
                     marker=dict(color=theme_palette['complementary'],line=dict(color=theme_palette['base'],width=0)),
                     text = ["%.2f" %(round(v ,2) *100) + '%' for v in neg.values],
                     textposition = 'outside',
                     textfont_color = '#212a2f'))

#theme_palette['2']
fig.update_layout(template = temp,
                  title={
                      "text": "<b>Top-10 Correlated Features with the Payment Default Feature</b> <BR />Pearson Values > 0.5<br> <br> ",
                      "x":0.035,
                      "font_size": 18,
                      
                  },
                 plot_bgcolor = theme_palette['backgound'],
                      paper_bgcolor = theme_palette['backgound'],
                 legend=dict(
                            y=1.15,
                            x=0.88))


# <h2 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    -- Store Interim Data-- </h2>
# 
# Due to the redundant crashs we ought to store the used columns to use it later

# In[16]:


if True:
    types = train.dtypes
    target_col = 'target'

    cat_cols = list(types[types.apply(lambda x:not(str(x).startswith('float')))].index)
    cat_cols = list(filter(lambda x:x!=target_col, cat_cols))
    features = list(train.drop(target_col, axis=1).columns)
    gc.collect()
    print('len cat_col', len(cat_cols))
    print('len features', len(features))
    
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

    with open('cat_cols.pkl', 'wb') as f:
        pickle.dump(cat_cols, f)


# <a id="5"></a>
# # <div style='display:fill;color:#2d3a41;background-color:#a3d3eb;padding:20px'>   🤖 <b> Model Training </b> </div>
# I implemented a gradient boosting model Wrapper **BaseModel** (inspired from [jayjay's notebook](https://www.kaggle.com/code/jayjay75/wids2020-lgb-starter-adversarial-validation)) in which I defined most of in common methods and attributes of gradient boosting based models, such as cv training, feature importances, prediction ... **LgbModel** and **CatBoost** would inherit from the BaseModel, Later I can add XGboost, hence we get the overall Gradient Boosting models comparison.
# <h2 style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉    -- Compettion Metric -- </h2>

# In[17]:


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    #https://www.kaggle.com/code/inversion/amex-competition-metric-python
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)
    
    y_pred=pd.DataFrame(data={'prediction':y_pred})
    y_true=pd.DataFrame(data={'target':y_true.reset_index(drop=True)})
    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# In[18]:


# a wrapper class  that we can have the same ouput whatever the model we choose
def get_partial_pred_df(fold, y_val, pred):
    df = pd.DataFrame()
    df['y_true'] = y_val
    df['y_pred'] = pred
    df['fold'] = fold
    return df[['fold','y_true', 'y_pred']]

class BaseModel:
    RAND_SEED = 75
    # TODO make a longer list of colors
    COLORS =['#a3d3eb', '#dfa3eb', '#ebbba3', '#afeba3', '#ebe6a3']
    def __init__(self, features, categoricals=[], n_splits=5, verbose=True, ps=None, target_col='target'):
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = target_col
        self.cv_models = []
        self.verbose = verbose
        self.model=None
        if not ps:
            self.params = self.get_default_params()
        else:
            self.params = ps
            
        
    def train_model(self, train_set, val_set, eval_metric=[]):
        raise NotImplementedError
        
    def get_cv(self, train_df):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.RAND_SEED)
        return cv.split(train_df, train_df[self.target])
    
    def get_default_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
            
    def fit_cv(self, train_df, save_cv=False):
        self.oof_pred = np.zeros((len(train_df), ))
    
        cv = self.get_cv(train_df)
        self.partial_oof_scores_=[]
        self.cv_df=pd.DataFrame()
        
        for fold, (train_idx, val_idx) in enumerate(cv):
            
            
            print("*-"*100)
            print(f"*** FOLD == {fold} **")
            print("*-"*100)
            x_train, x_val = train_df[self.features].iloc[train_idx], train_df[self.features].iloc[val_idx]
            y_train, y_val = train_df[self.target][train_idx], train_df[self.target][val_idx]
            train_set = self.convert_dataset(x_train, y_train)
            val_set = self.convert_dataset(x_val, y_val)
            
            model = self.train_model(train_set, val_set)
            self.cv_models.append(model)
            
            conv_x_val = self.convert_x(x_val)
            self.oof_pred[val_idx] = model.predict(conv_x_val).reshape(self.oof_pred[val_idx].shape)
            
            partial_oof_score = amex_metric(y_val, self.oof_pred[val_idx])
            self.partial_oof_scores_.append(partial_oof_score)
            print('Partial score of fold {} is: {}'.format(fold,  partial_oof_score))
            if save_cv:                
                # save model
                joblib.dump(model, f'lgb_{fold}.pkl')
                
            partial_pred_df = get_partial_pred_df(fold, y_val, self.oof_pred[val_idx])
            self.cv_df = pd.concat([self.cv_df, partial_pred_df])

        self.oof_score_ = amex_metric(train_df[self.target], self.oof_pred) 
        if self.verbose:
                print('Our oof score is: ', self.oof_score_)
                
    def fit(self, train):
        x_train = train[self.features]
        y_train = train[self.target]
        train_set = self.convert_dataset(x_train, y_train)
        self.model = self.train_model(train_set)
        print("model trained in all training dataset")
        
    def predict_cv(self, test_df):
        y_pred = np.zeros((len(test_df), ))
        x_test = self.convert_x(test_df[self.features])
        for model in self.cv_models:
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
        return y_pred
    
    def predict(self, test_df):
        x_test = self.convert_x(test_df[self.features])
        y_pred = self.model.predict(x_test).reshape((len(test_df), ))
        return y_pred
    
    def get_cv_feature_importance(self):
        raise NotImplementedError
    
    def plot_cv_roc(self): 
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11), 
                                 name='Random Chance',mode='lines', showlegend=False,
                                 line=dict(color="Black", width=1, dash="dot")))
        for fold, sample_df in self.cv_df.groupby('fold'):
            fpr, tpr, _ = roc_curve(sample_df.y_true, sample_df.y_pred)
            roc_auc = auc(fpr,tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=self.COLORS[fold], width=3), 
                                     hovertemplate = 'True positive rate = %{y:.3f}<br>False positive rate = %{x:.3f}',
                                     name='Fold {}: AUC = {:.3f}'.format(fold+1, roc_auc)))

        fig.update_layout(template = temp,
                      yaxis_automargin=True,
                      height = 800,
                      plot_bgcolor = theme_palette['backgound'],
                      paper_bgcolor = theme_palette['backgound'],
                      title={
                          "text": "<b>Cross-Validation ROC Curves</b> ",
                          "x":0.045,
                          "font_size": 20,                    
                      },
                      margin={'pad':5},
                          xaxis_title='False Positive Rate (1 - Specificity)',
                          yaxis_title='True Positive Rate (Sensitivity)',
                          legend=dict(orientation='v', y=.07, x=1, xanchor="right",
                                      bordercolor="black", borderwidth=.5)
        )

        return fig
    
    def plot_cv_feature_importance(self, top=20):
        feat_imp = self.get_cv_feature_importance()
        data = feat_imp[:top]
        threshold = data.iloc[5]['importance']
        fig = go.Figure()
        fig.add_trace(go.Bar(y=data.index, x= data['importance'],
                             orientation='h',
                             width=[0.6]*len(data),
                             marker=dict(color=(data['importance'] < threshold).astype('int'),
                                         colorscale=[[0, theme_palette['complementary']], [1, theme_palette['base']]], ),

                             text = ["<b>%.2f"%(round(v ,2))+'</b>' for v in data['importance']],
                             textposition = 'inside',
                             textfont_color = theme_palette['backgound']))

        fig.update_layout(template = temp,
                          yaxis_automargin=True,
                          height = 800,
                          plot_bgcolor = theme_palette['backgound'],
                          paper_bgcolor = theme_palette['backgound'],
                          yaxis=dict(autorange="reversed"),
                          title={
                              "text": "<b>Features Importances</b> ",#"— Gain - Threshold = 5.2e+04<BR />the last payment 2 is farway in the top of list<br> <br> ",
                              "x":0.045,
                              "font_size": 20,                    
                          },
                          margin={'pad':5},
        )
        return fig



# In[19]:


#we choose to try a LightGbM using the Base_Model class
class LgbModel(BaseModel):
    
    def train_model(self, train_set, val_set=None):
        verbosity = 100 if self.verbose else 0
        valid_sets=[train_set]
        if val_set:
            valid_sets = [train_set, val_set]
        return lgb.train(self.params, train_set, 
                         valid_sets=valid_sets, 
                         verbose_eval=verbosity)
    
    def convert_dataset(self, x_train, y_train):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        return train_set
    
    
    def get_default_params(self):
        params = {'n_estimators':50000,
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,  
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100,
                    #'is_unbalance' : True ,
                    'scale_pos_weight' : 3
                  
                    }
        return params
    
    def get_cv_feature_importance(self):
        imp_df = pd.DataFrame(index=self.features)
        imp_df['importance'] = 0
        for model in self.cv_models:      
            imp_df['importance'] = (imp_df['importance'] + pd.Series(model.feature_importance(), index=model.feature_name()))/self.n_splits
        return imp_df.sort_values('importance', ascending=False)
                                    
    

#we choose to try a LightGbM using the Base_Model class
class CatBoost(BaseModel):
    def train_model(self, train_set, val_set=None):
#        eval_set=[val_set] if val_set else None            
        verbosity = 100 if self.verbose else 0   
        return catboost.train(pool=train_set, params=self.params, 
                         eval_set=val_set, verbose_eval=verbosity)   
    
    def convert_dataset(self, x_train, y_train):
        train_set = Pool(data=x_train, label=y_train, 
                         cat_features=self.categoricals)
        return train_set
    
    
    def get_default_params(self):
        params = {'iterations' : 5000,
                  'random_seed':self.RAND_SEED
#                  'metric': ['auc','binary_logloss'],
                    }
        return params
    
    def get_cv_feature_importance(self):
        imp_df = pd.DataFrame(index=self.features)
        imp_df['importance'] = 0
        for model in self.cv_models:      
            imp_df['importance'] = (imp_df['importance'] + pd.Series(model.feature_importances_, index=model.feature_names_))/self.n_splits
        return imp_df.sort_values('importance', ascending=False)
                                    
    


# In[ ]:





# In[20]:


train['target'] = train["target"].astype('int')
print('Transform all String features to category.\n')
os.makedirs('label_encoders')

for usecol in tqdm_notebook(cat_cols):
#    print(usecol)
    train[usecol] = train[usecol].astype('str')
#    test[usecol] = test[usecol].astype('str')

    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()))#+
#                      test[usecol].unique().tolist()))

    #At the end 0 will be used for null values so we start at 1 
    train[usecol] = le.transform(train[usecol])+1
#    test[usecol]  = le.transform(test[usecol])+1

    train[usecol] = train[usecol].replace(np.nan, 0).astype('int').astype('category')
#    test[usecol]  = test[usecol].replace(np.nan, 0).astype('int').astype('category')

    joblib.dump(le, f'label_encoders/{usecol}_label_encoder.pkl')
    


# In[21]:


#! zip -r label_encoders.zip label_encoders


# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'> 👉  -- LightGBM Training  -- </h2>
# 

# In[22]:


#https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977#kln-129
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'seed': 75,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
        }


model_path = "../input/modelsv2/lgb_model_with_diff.sav"
if not model_path:
    lgb_model = LgbModel(features=features, n_splits=5, categoricals=cat_cols, ps=params, verbose=None)
    lgb_model.fit_cv(train, save_cv=True)
    joblib.dump(lgb_model, "lgb_model_with_diff.sav")
else:
    lgb_model = joblib.load(model_path)


# In[23]:


gc.collect()


# In[24]:


lgb_model.oof_score_


# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'> 👉  -- CatBoost Training  -- </h2>
# 
# For the Catboost that takes longer time to train, we will apply correlation based feature selection

# In[25]:


# very heavy with all feature => we will select more 
print(THRESHOLD)
deeper_feat_sel = True
if deeper_feat_sel:
    THRESHOLD += 0.05
    print("new threshold", THRESHOLD)
    
corr = train.corrwith(train['target'], axis=0)
corr = corr[corr.notna()].sort_values(key=abs, ascending=False)
selected_feats = corr[corr.abs()>THRESHOLD].index
train = train[list(selected_feats)]

types = train.dtypes
target_col = 'target'

selected_cat_cols = list(types[types.apply(lambda x:not(str(x).startswith('float')))].index)
cb_cat_cols = list(filter(lambda x:x!=target_col, selected_cat_cols))
cb_features = list(train.drop(target_col, axis=1).columns)


# In[26]:


model_path = "../input/modelsv2/cb_model_with_diff.sav"
if not model_path:
    cb_model = CatBoost(features=cb_features, categoricals=cb_cat_cols, n_splits=2)
    cb_model.fit_cv(train, save_cv=True)
    joblib.dump(cb_model, "cb_model_with_diff.sav")
else:
    cb_model = joblib.load(model_path)


# <a id="6"></a>
# # <div style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  🏅  Feature Importance </div>
# 
# Let's explore the feature importance of each model. One of the advantages of gradient boosting based models is that the feature importances can be directly extracted from the splitting gain of tree algorithm
# 
# 
# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   LightGBM Feature Importance      </b>  </h2>
# 

# In[27]:


fig = lgb_model.plot_cv_feature_importance()
fig.show()


# In[28]:


len(features)


# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   CatBoost Feature Importance      </b>  </h2>

# In[29]:


fig = cb_model.plot_cv_feature_importance()
fig.show()


# <a id="7"></a>
# # <div style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  📈   CV Roc Curves </div>
# Lets display Cross validation Roc curves for each model
# 
# 
# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   LightGBM CV roc curves  </b>  </h2>
# 

# In[30]:


fig = lgb_model.plot_cv_roc()
#fig = plot_roc(lgb_model)
fig.show()


# <h2 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   CatBoost CV roc curves  </b>  </h2>
# 

# In[31]:


fig = cb_model.plot_cv_roc()
fig.show()


# In[32]:


del train
gc.collect()


# <a id="8"></a>
# # <div style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  👉 Infer (or Predict Test data)</div>
# 
# Generate the test predictions 

# In[33]:


from functools import partial, reduce

models_folder = '../input/modelsv2'

def get_sample(path):
    test = pd.read_feather(path).set_index('customer_ID')

    test = reduce_size(test)

    test = test[lgb_model.features]
    gc.collect()

    for usecol in tqdm_notebook(cat_cols):
        le = joblib.load(os.path.join(models_folder,f'{usecol}_label_encoder.pkl'))
        test[usecol] = test[usecol].astype('str').apply(lambda x:x.split('.')[0])
        test[usecol] = test[usecol].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        #At the end 0 will be used for null values so we start at 1 
        test[usecol]  = le.transform(test[usecol])+1
        test[usecol] = test[usecol].replace(np.nan, 0).astype('int').astype('category')
    return test


def get_batchs(df, batch_size, keep_features):
    n_batchs = int(len(df)/batch_size)
    cid = list(df.index)
    for i in range(n_batchs):
        idx = cid[i*batch_size:(i+1)*batch_size]
        if i == n_batchs-1:
            idx = idx + cid[(i+1)*batch_size:]
        yield df.loc[idx, keep_features]
        
def predict(sample_df, model):
    y_pred = model.predict_cv(sample_df)
    sub = pd.Series(y_pred, index=sample_df.index, name='prediction')
#    sub.to_frame().to_csv(outfile)
    return sub



# In[34]:


gc.collect()


# In[ ]:





# <h3 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   LightGBM submission </b>  </h3>

# In[35]:


n_batchs=50
sub_lg = pd.Series()
for path in glob.glob("../input/feat-eng-with-diff/feat_eng_agg_test_with_diff_*.ftr"):
    print(path)
    test = get_sample(path)
    samples_df = get_batchs(test,batch_size=n_batchs, keep_features=lgb_model.features)
    sample_sub_lg = pd.concat(map(partial(predict, model=lgb_model), tqdm_notebook(samples_df)))
    sub_lg = pd.concat([sub_lg, sample_sub_lg], ignore_index=True)
    del test
    gc.collect()
    
print(len(sub_lg))

sub_lg.to_frame().to_csv('submission_lgb_with_diff.csv')


# <h3 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   CatBoost Submission  </b>  </h3>

# In[36]:


n_batchs=50
sub_cb = pd.Series()
for path in glob.glob("../input/feat-eng-with-diff/feat_eng_agg_test_with_diff_*.ftr"):
    print(path)
    test = get_sample(path)
    samples_df = get_batchs(test,batch_size=n_batchs, keep_features=cb_model.features)
    sample_sub_cb = pd.concat(map(partial(predict, model=cb_model), tqdm_notebook(samples_df)))
    sub_cb = pd.concat([sub_cb, sample_sub_cb], ignore_index=True)
    del test
    gc.collect()
print(len(sub_cb))

sub_cb.to_frame().to_csv('submission_cb_with_diff.csv')


# <h3 style='color:#2d3a41;background-color:#c8e5f3;padding:10px'>  🫐   <b>   Weighted submission  </b>  </h3>

# In[37]:


# weighted submission
weighted_sub = (sub_lg*lgb_model.oof_score_ + sub_cb*cb_model.oof_score_)/(lgb_model.oof_score_+cb_model.oof_score_)
weighted_sub.to_frame().to_csv('weighted_sub.csv')


# <a id="8"></a>
# # <div style='color:#2d3a41;background-color:#a3d3eb;padding:10px'>  📊 Predictions Distribution</div>

# In[38]:


def get_dirtibution(sub,threshold=0.5):
    target = (sub>threshold).astype(int)
    target = target.value_counts(normalize=True)
    target.rename(index={1:'Default',0:'Paid'},inplace=True)
    fig=go.Figure()
    fig.add_trace(go.Pie(labels=target.index, values=target*100,# hole=.45, 
                         showlegend=True,#sort=True, 
                         marker=dict(colors=list(theme_palette.values())),
    #                     marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                         hovertemplate = "%{label} Amex Acoounts: <b>%{value:.2f}</b>%<extra></extra>"))
    fig.update_layout(template=temp, 
                      title={
                          "text":'<b>Default VS Paid Transaction</b><BR />Unballanced dataset',
                          "x":0.035,
                          "font_size": 20,

                      },

                      uniformtext_minsize=15,# width=700,
    #                  height=800,
                      margin={'t':150, 'l':5})
    return fig


# In[39]:


fig = get_dirtibution(sub_lg)
fig.show()


# In[40]:


fig = get_dirtibution(sub_cb)
fig.show()


# ## <div style='color:#016CC9;text-align:center;font-size:100%'>*** WORK IN PROGRESS ***</div>
# 
# 

# In[ ]:




