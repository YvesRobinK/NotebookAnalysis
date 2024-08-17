#!/usr/bin/env python
# coding: utf-8

# <img width="100%" src="https://img.freepik.com/free-vector/hand-drawn-weather-effects_23-2149117711.jpg?w=1060&t=st=1673001104~exp=1673001704~hmac=183f83c2559786664e6f84b259c3122c36904e0f9c31c53f288d60086a521941">
# <figcaption>Image by <a href="https://www.freepik.com/free-vector/hand-drawn-weather-effects_18895324.htm#query=weather&position=2&from_view=keyword">Freepik</a></figcaption>

# # Introduction:
# 
# ## Overview:
# Weather forecasts have always been based on **quantitative** data related to the atmosphere, land, ocean, and meteorology to project how the atmosphere will change at a given place. These quantitative-based techniques have a limited forecast horizon, so it would be interesting to blend the manual physics-based forecasts with **machine learning** based models to extend **forecast window**.
# 
# For each location and start date, we aim to predict the arithmetic mean of the max and min observed temperature over the **next 14 days** 
# 
# ## Models
# In this Notebook we will Gradient boosting-based algorithms to forecast future temperatures. We will provide a Gradient Boosting template with different model Backbones:
# * LightGBM Model
# * CatBoost Model
# * Xgboost Model
# 
# As these models support GPU processing, we will use **GPU P100** accelerator to run the algorithms
# 
# ## CV
# The Training approach is by **cross validation**, each fold is stratified along with **location and month**.
# By default, we will apply CV with 5 folds
# 
# 
# ## Feature engineering
# For the feature engineering part, we will select the top correlated features with the target and apply **aggregated-based feature engineering** grouped by **location and month**
# 
# 
# 
# 

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
from sklearn.metrics import mean_squared_error

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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import catboost
from catboost import Pool 
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
    'base': '#71A5F8',
    'complementary':'#71A5F8',
    'triadic' : '#eba3d3', 
    'backgound' : "#fcfaf1"
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


# # Read Data

# In[2]:


train = pd.read_csv("/kaggle/input/widsdatathon2023/train_data.csv")
# format datetime feature
train['startdate'] = pd.to_datetime(train['startdate'], format="%m/%d/%y")
train = train.sort_values(by=['lat', 'lon', 'startdate']).reset_index(drop=True)

test = pd.read_csv("/kaggle/input/widsdatathon2023/test_data.csv")
test['startdate'] = pd.to_datetime(test['startdate'], format="%m/%d/%y")
test = test.sort_values(by=['lat', 'lon', 'startdate']).reset_index(drop=True)

train["source"] = "train"
test["source"] = "test"

gc.collect()


# In[3]:


SAMPLE = False
if not SAMPLE:
    df = pd.concat([train, test])
#    del train, test
else:
    sample_keys = list(test.groupby(["lat", 'lon']).groups.keys())[:2]
    df = pd.DataFrame()
    for df_src in [train, test]:
        for gr, df_gr in df_src.groupby(["lat", 'lon']):
            if gr in sample_keys:
                df = pd.concat([df, df_gr])

#del train, test


# In[4]:


target = "contest-tmp2m-14d__tmp2m"
drop_cols = ["index", "startdate", "source"]


def get_num_cat_cols(train, drop_cols=drop_cols):
    cat_cols = list(filter(lambda x : not str(train[x].dtypes).startswith("float"), train.columns))
    cat_cols = [x for x in cat_cols if x not in drop_cols]
    num_cols = [x for x in train.columns if x not in cat_cols+drop_cols+[target]]
    return num_cols, cat_cols

num_cols, cat_cols = get_num_cat_cols(df[df["source"]=="train"])
print(len(num_cols), len(cat_cols))


# # Reduce memory usage

# In[5]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

df = reduce_mem_usage(df)
df["month"] = df['startdate'].dt.month


# # EDA
# 
# In this notebook we will not apply advanced EDA, only a quick glance at the data to get a general
# understanding

# ## Target 
# Lets plot the target distribution with a histogram

# In[6]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train['contest-tmp2m-14d__tmp2m'].to_list(),
    histnorm='percent',
    name='control', # name used in legend and hover labels
    marker_color='#71A5F8',
    opacity=0.75,
    xbins=dict(
        start=train['contest-tmp2m-14d__tmp2m'].min(),
        end=train['contest-tmp2m-14d__tmp2m'].max(),
        size=0.5
    )
))

fig.update_layout(template = temp,
                  yaxis_automargin=False,
                  height = 800,
                  plot_bgcolor = '#fcfaf1',
                  paper_bgcolor = '#fcfaf1',
                  title={
                      "text": "<b>Target Histograms</b> <BR />Bell-shaped curve<br> <br> ",
                      "x":0.035,
                      "font_size": 20,                    
                  },
                  margin={'pad':10},
)
fig.show()


# -> The target Has a normal distribution

# ## Months distribution
# 
# Let's take a look at the months distribution for cold and hot temeratures  

# In[7]:


cold = train[train[target]<-10]
cold["month"] = cold['startdate'].dt.month
cold["year"] = cold['startdate'].dt.year
cold = cold.reset_index().groupby(['year', 'month'])['index'].count()
cold.index = cold.index.to_series().apply(lambda x: f'{x[0]}-{x[1]}').values

hot = train[train[target]>30]
hot["month"] = hot['startdate'].dt.month
hot["year"] = hot['startdate'].dt.year
hot = hot.reset_index().groupby(['year', 'month'])['index'].count()
hot.index = hot.index.to_series().apply(lambda x: f'{x[0]}-{x[1]}').values

fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Bar(x=cold.index, y=cold.values, name='cold', marker_color="#71A5F8"),
              row=1, col=1)
fig.add_trace(go.Bar(x=hot.index, y=hot.values, name='hot', marker_color="#F88171"),
              row=1, col=2)
              
fig.update_layout(template=temp,
                title={
                  "text": "<b>Month Distrubution </b> <BR /> Cold/Hot temperatures<br><br><br>",
                  "x":0.035,
                  "font_size": 20,

              },
                  yaxis_automargin=False,
                  height = 500,
                  plot_bgcolor = '#fcfaf1',
                  paper_bgcolor = '#fcfaf1',
                  margin={'pad':10},

)   



# As expected, the hot temperatures are concentrated around **June, July** and **August**, the cold season is between **December and Junary**
# 
# ## Number of series

# In[8]:


len(train.groupby(["lat", "lon"]).groups)


# In[9]:


train.groupby(["lat", "lon"])["startdate"].nunique().unique()


# We Have Overall **514** time series over 731days=2 years, prediction period = 2 weeks

# ## Missing Values

# In[10]:


# Function to calculate missing values by column# Funct 
# from https://www.kaggle.com/parulpandey/starter-code-with-baseline
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

# Missing values data
missing_values = missing_values_table(df.drop(drop_cols+[target], axis=1))
missing_values[:20].style.background_gradient(cmap='Reds')


# ## Imputing
# For each geo-spatial region defined by lon, lat coordinates and month
# * Mean imputing for numerical features
# * Mode imputing for categorical features

# In[11]:


def impute_missing(train, num_cols=num_cols, cat_cols=cat_cols, imputing_cols=["lat", "lon"]):
    for col in num_cols:
        train[col] = train.groupby(imputing_cols)[col].transform(lambda x: x.fillna(x.mean()))
    for col in cat_cols:
        train[col] = train.groupby(imputing_cols)[col].transform(lambda x: x.fillna(pd.Series.mode))
    return train

df = impute_missing(df, imputing_cols=["lat", "lon", "month"])


# In[12]:


missing_values = missing_values_table(df.drop(drop_cols+[target], axis=1))
missing_values[:20].style.background_gradient(cmap='Reds')


# # Correlation:
# Lets find out the top correlated features with the target

# In[13]:


if os.path.isfile("/kaggle/input/wids-2023-fe/correlation_matrix.csv"):
    corr = pd.read_csv("/kaggle/input/wids-2023-fe/correlation_matrix.csv")
    corr = corr.set_index(corr.columns[0])
else:
    corr = train.corr()

target_corr = corr[target].sort_values(ascending=False)
# take only most correlated features 
most_correlated_cols = list(target_corr[((target_corr>0.9) & (target_corr<1)) | (target_corr<-0.9)].index)

sorted_corr = corr[target].sort_values(key=abs, ascending=False)[:11] # top but we have to  drop corr=1


pos = sorted_corr[(sorted_corr>0) & (sorted_corr<1)]
neg = sorted_corr[(sorted_corr<0)].sort_values(ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(x=pos.index, y= pos.values,
                     orientation='v',
                     name='Positive',
                     marker=dict(color="#71A5F8",line=dict(color="#71A5F8",width=0)),
                     text = ["%.2f" %(round(v ,2) *100) + '%' for v in pos.values],
                     textposition = 'outside',
                     textfont_color = '#4E1C1E'))

fig.add_trace(go.Bar(x=neg.index, y= neg.values,
                     orientation='v',
                     name='Negative',
                     marker=dict(color="#F8C471",line=dict(color="#F8C471",width=0)),
                     text = ["%.2f" %(round(v ,2) *100) + '%' for v in neg.values],
                     textposition = 'outside',
                     textfont_color = '#4E1C1E'))

fig.update_layout(template = temp,
                  title={
                      "text": "<b>Top-10 Correlated Features with the Target Feature</b> <BR />Pearson Values > 0.9<br> <br> ",
                      "x":0.035,
                      "font_size": 18,
                      
                  },
                  height=500,
                 plot_bgcolor = '#fcfaf1',
                      paper_bgcolor = '#fcfaf1',
                 legend=dict(
                            y=1.15,
                            x=0.88))


# # Feature Engineering

# ## Root-mean-square deviation based features
# For each top correlated feature we will compute **Root-mean-square deviation** between the feature and the **target**

# In[14]:


def get_deviations(train, top_corr_feats, groupby_col=['lat', 'lon']):
    deviations = train.groupby(groupby_col).apply(
        lambda s : pd.Series(
            {
                f"{col}" : mean_squared_error(s[target], s[col], squared=False)
                for col in top_corr_feats
            }
        ) 

    )
    suffix = "_".join(groupby_col)
    deviations.columns = [f"{col}_{suffix}_deviation" for col in deviations.columns]
    return deviations.reset_index(drop=False)

if os.path.isfile('/kaggle/input/wids-2023-fe/deviations_loc.csv'):
    deviations = pd.read_csv('/kaggle/input/wids-2023-fe/deviations_loc.csv')
else:
    # only on train
    deviations = get_deviations(
        df[df["source"]=="train"], 
        most_correlated_cols, 
        groupby_col=['lat', 'lon']
    )
    deviations.to_csv("deviations_loc.csv", index=False)
    
print(len(most_correlated_cols))
print(deviations.shape)


# ## Statistics based features

# In[15]:


def get_grouped_statics(train, top_corr_feats, groupby_col=['lat', 'lon']):
    suffix = "_".join(groupby_col)
    
    stats = train.groupby(groupby_col).agg(
        {
            col : ["mean", "std", "min", "max"] for col in top_corr_feats 
        }
    )
    stats.columns = [
        f"{col}_{suffix}_{op}" \
        for col in top_corr_feats \
        for op in ["mean", "std", "min", "max"]
    ]
    
    return stats.reset_index(drop=False)

if os.path.isfile('/kaggle/input/wids-2023-fe/stats_loc.csv'):
    stats = pd.read_csv('/kaggle/input/wids-2023-fe/stats_loc.csv')
else:
    # only on train
    stats = get_grouped_statics(
        df, 
        most_correlated_cols, 
        groupby_col=['lat', 'lon']
    )
    stats.to_csv("stats_loc.csv", index=False)
    
print(len(most_correlated_cols))
print(stats.shape)


# Let summarize all in one function

# In[16]:


def feature_engineering(train, most_correlated_cols, 
                        lags=[1,2, 3, 7, 14, 21, 28, 35, 42, 49], 
                        groupby_col=['lat', 'lon']):
    
    suffix = "_".join(groupby_col)
    ### DEVIATION BASED FEATS ###
    print("### DEVIATION BASED FEATS ###")
    print(train.shape)
    train = train.merge(deviations, how="left", on=groupby_col)
    train = train.merge(stats, how="left", on=groupby_col)
    for col in tqdm_notebook(most_correlated_cols):
        train[f"target_from_{col}_deviation"] = train.apply(
            lambda row : row[col]+row[f"{col}_{suffix}_deviation"] \
                        if row[col]<row[f"{col}_{suffix}_mean"]
                        else row[col]-row[f"{col}_{suffix}_deviation"], 
            axis=1
        )
        train = train.drop(f"{col}_{suffix}_deviation", axis=1)
    print(train.shape)
    
    ## LAG BASED FEATURES
    print("## LAG BASED FEATURES")
    for lag_days in tqdm_notebook(lags) :
        for col in most_correlated_cols:
            train[f'{col}_lag{lag_days}'] = train.groupby(['lat', 'lon'])[col].shift(lag_days)
    print(train.shape)
    
    ### ROLLING BASED FEATURES
    print("### ROLLING BASED FEATURES")
    for lag_days in tqdm_notebook(lags):
        for col in most_correlated_cols:
            train[f'{col}_roll{lag_days}'] = train.groupby(['lat', 'lon'])[col].shift(lag_days).rolling(lag_days).mean()
    print(train.shape)
    
    ## BINNING
    # Select the top most important features from the last catboost model and create the related binning feature
    top_feats = [
        'contest-wind-h500-14d__wind-hgt-500', 
        'contest-wind-h100-14d__wind-hgt-100', 
        "contest-slp-14d__slp", 
        "contest-prwtr-eatm-14d__prwtr", 
        "contest-pevpr-sfc-gauss-14d__pevpr"
    ]
    for col in top_feats:
        train[f"{col}_bins"] = pd.qcut(train[col], q=4, labels=[f'{col}_{i}' for i in range(4)])
    
    return train

if not os.path.isfile("/kaggle/input/wids-2023-fe/df_fe_loc.pkl"):
    ### train
    print("TRAIN FEATURE ENGINEERING")
    train = feature_engineering(
        df[df["source"]=="train"], 
        most_correlated_cols, 
        groupby_col=['lat', 'lon']
    )
    train.to_pickle("df_fe_loc.pkl")
    ## test
    print("TEST FEATURE ENGINEERING")
    test = feature_engineering(
        df[df["source"]=="test"], 
        most_correlated_cols, 
        groupby_col=['lat', 'lon']
    )
    test.to_pickle("df_fe_loc.pkl")
    # concat
    df = pd.concat([train, test])
    # save some memory
    del train, test
    gc.collect()
    
    df.to_pickle("df_fe_loc.pkl")
    
else:
    df = pd.read_pickle("/kaggle/input/wids-2023-fe/df_fe_loc.pkl")


# In[17]:


gc.collect()


# ## Create FOLD feature:
# We will apply stratified cross validation along with **location** and **month**
# 
# Let'st start y creating the FOLD feature

# In[18]:


lat = df['lat'].apply(lambda x : "%.4f" % x)
lon = df['lon'].apply(lambda x : "%.4f" % x)
df['FOLD'] = lat+'_'+lon#+'-'+df["month"].astype(str)


# ## Label Encoding
# For categorical features we chose to apply label encoding

# In[19]:


print('Transform all String features to category.\n')
_, cat_cols = get_num_cat_cols(df)

for usecol in tqdm_notebook(cat_cols):
    print(usecol)   
    df[usecol] = df[usecol].astype('str')

    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(df[usecol].unique().tolist()))

    #At the end 0 will be used for null values so we start at 1 
    df[usecol] = le.transform(df[usecol])+1

    df[usecol] = df[usecol].replace(np.nan, 0).astype('int').astype('category')


# In[20]:


train = df[df["source"]=="train"]
test = df[df["source"]=="test"]
del df
gc.collect()

train = train.set_index("index")
test = test.set_index("index")
# drop unuseful indexes
train = train.drop(['startdate', 'source'], axis=1)
test = test.drop(['startdate', target, 'source'], axis=1)

train["FOLD"] = train["FOLD"].astype('str').astype('category')
test["FOLD"] = test["FOLD"].astype('str').astype('category')


# # Model Training:
# I implemented a gradient boosting model Wrapper BaseModel (inspired from [jayjay's notebook](https://www.kaggle.com/code/jayjay75/wids2020-lgb-starter-adversarial-validation)) in which I defined most of in common methods and attributes of gradient boosting based models, such as cv training, feature importances, prediction ... 
# 
# **LgbModel** model, **CatBoost** model and **XgBoost** Model would inherit from the BaseModel, hence we get the overall Gradient Boosting models comparison.

# In[21]:


N_FLODS = 3
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
    def __init__(self, 
                 features, 
                 categoricals=[], 
                 n_splits=N_FLODS, 
                 fold_feat=None,#"FOLD",
                 keep_fold_feat=True,
                 verbose=True, 
                 ps=None, 
                 target_col=target):
        
        self.features = features
        self.fold_feat = fold_feat
        if not keep_fold_feat and self.fold_feat:
            self.features = [col for col in self.features if col!=self.fold_feat]
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
        if not self.fold_feat:
            print("not stratified CV splitting")
            cv = scores=[]
            cv = KFold(
                n_splits=self.n_splits, 
                random_state=self.RAND_SEED, 
                shuffle=True
            )
            return cv.split(train_df)
        print("Stratified CV splitting")
        cv = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.RAND_SEED
        )
        return cv.split(train_df, train_df[self.fold_feat])
    
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
            
            partial_oof_score = mean_squared_error(y_val, self.oof_pred[val_idx], squared=False) 
            self.partial_oof_scores_.append(partial_oof_score)
            print('Partial score of fold {} is: {}'.format(fold,  partial_oof_score))
            if save_cv:                
                # save model
                joblib.dump(model, f'lgb_{fold}.pkl')
                
            partial_pred_df = get_partial_pred_df(fold, y_val, self.oof_pred[val_idx])
            self.cv_df = pd.concat([self.cv_df, partial_pred_df])

        self.oof_score_ = mean_squared_error(train_df[self.target], self.oof_pred, squared=False) 
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
    
    
    def plot_cv_feature_importance(self, top=20, sub_title=""):
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
                              "text": f"<b>Features Importances</b> *-*-< {sub_title} >-*-*",#
                              "x":0.045,
                              "font_size": 20,                    
                          },
                          margin={'pad':5},
        )
        return fig


# # LightGBM

# In[22]:


#we choose to try a LightGbM using the Base_Model class
class LgbModel(BaseModel):
    def __init__(self, features, categoricals=[], n_splits=5, fold_feat=None,
                 verbose=True, ps=None, target_col=target):
        # call superclass
        super().__init__(features=features, categoricals=categoricals, n_splits=n_splits, 
                         fold_feat=fold_feat,
                         verbose=verbose, ps=ps, target_col=target_col)

    
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
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,  
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100,
                    #'is_unbalance' : True ,
                    'scale_pos_weight' : 3,                  
                    }
        return params
    
    def get_cv_feature_importance(self):
        imp_df = pd.DataFrame(index=self.features)
        imp_df['importance'] = 0
        for model in self.cv_models:      
            imp_df['importance'] = (imp_df['importance'] + pd.Series(model.feature_importance(), index=model.feature_name()))/self.n_splits
        return imp_df.sort_values('importance', ascending=False)
                                    
    


# In[23]:


gc.collect()


# In[24]:


params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "dart",
        "n_estimators": 2000,
        "early_stopping_round": 100,
        "device": "cpu",
#        "gpu_platform_id": 0,
#        "gpu_device_id": 0,
         'lambda_l1': 0.00472279780583036, 
        'lambda_l2': 2.9095205689488508e-05, 
        'num_leaves': 158, 
        'feature_fraction': 0.7386878356648194, 
        'bagging_fraction': 0.8459744550725283, 
        'bagging_freq': 2, 
        'max_depth': 2, 
        'max_bin': 249, 
         'seed': 75,
        'learning_rate': 0.044738463593017294,
        'min_child_samples': 13
    }
    


model_path = "../input/models/lgb_model_loc.sav"
if not os.path.isfile(model_path):
    lgb_model = LgbModel(
        features=test.columns, 
        n_splits=N_FLODS, 
        fold_feat=None, #"FOLD",  
        target_col=target,
        categoricals=cat_cols+['FOLD'], 
        ps=params, 
        verbose=None
    )
    lgb_model.fit_cv(train, save_cv=True)
    joblib.dump(lgb_model, "lgb_model_loc.sav")
else:
    lgb_model = joblib.load(model_path)


# In[25]:


fig = lgb_model.plot_cv_feature_importance(sub_title="LightGBM Model")
fig.show()


# In[26]:


# lgb_model.oof_pred
print("LIGHTGBM OOF RMSE",lgb_model.oof_score_)


# In[27]:


def predict(df, model, name=target):
    y_pred = model.predict_cv(df)
    sub = pd.Series(y_pred, index=df.index, name=name)
#    sub.to_frame().to_csv(outfile)
    return sub

sub1 = predict(test, lgb_model)
sub1.to_csv("submision_lgb.csv")


# # CatBoost

# In[28]:


class CatBoost(BaseModel):
    def __init__(self, features, categoricals=[], n_splits=5, fold_feat=None,
                 verbose=True, ps=None, target_col=target, early_stopping_rounds=400):
        # call superclass
        super().__init__(features=features, categoricals=categoricals, n_splits=n_splits, 
                         fold_feat=fold_feat,
                         verbose=verbose, ps=ps, target_col=target_col)
        self.early_stopping_rounds = early_stopping_rounds
        
    def train_model(self, train_set, val_set=None):
#        eval_set=[val_set] if val_set else None            
        verbosity = 100 if self.verbose else 0
        cb_reg = catboost.CatBoostRegressor(**self.params)
        cb_reg.fit(train_set, 
                   eval_set=val_set,
                   verbose_eval=verbosity,
                  use_best_model=True, 
                   early_stopping_rounds=self.early_stopping_rounds)   
        return cb_reg
    
    def convert_dataset(self, x_train, y_train):
        train_set = Pool(data=x_train, label=y_train, 
                         cat_features=self.categoricals)
        return train_set
    
    
    def get_default_params(self):
        params={'iterations' : 2000,
                "task_type":"GPU",
              'random_seed':self.RAND_SEED,
              'learning_rate': 0.02,
                'depth': 7,
                   'bootstrap_type' : 'Bernoulli',
                   'random_strength': 1,
                   'min_data_in_leaf': 10,
                    'l2_leaf_reg': 3,
                   'loss_function' : 'RMSE', 
                   'eval_metric' : 'RMSE',
                   'grow_policy' : 'Depthwise',
                   'max_bin' : 1024, 
                   'model_size_reg' : 0,
                   'od_type' : 'IncToDec',
                   'od_wait' : 100,
                   'metric_period' : 500,
                   'verbose' : 500,
                   'subsample' : 0.8,
                   'od_pval' : 1e-10,
                   'max_ctr_complexity' : 8,
                   'has_time': False,
        }
                         
        return params
    
    def get_cv_feature_importance(self):
        imp_df = pd.DataFrame(index=self.features)
        imp_df['importance'] = 0
        for model in self.cv_models:      
            imp_df['importance'] = (imp_df['importance'] + pd.Series(model.feature_importances_, index=model.feature_names_))/self.n_splits
        return imp_df.sort_values('importance', ascending=False)
                                    
    


# In[29]:


model_path = "../input/models/cb_loc.sav"
if not os.path.isfile(model_path):
    cb_model = CatBoost(
        features=test.columns, 
        n_splits=N_FLODS, 
        fold_feat=None, #"FOLD",  
        target_col=target,
        categoricals=cat_cols+['FOLD'], 
        ps=None, 
        verbose=None,
        early_stopping_rounds=400
    )
    cb_model.fit_cv(train, save_cv=True)
    joblib.dump(cb_model, "cb_loc.sav")
else:
    cb_model = joblib.load(model_path)


# In[30]:


sub2 = predict(test, cb_model)
sub2.to_csv("submision_cb.csv")


# In[31]:


print("CATBOOST OOF RMSE",cb_model.oof_score_)


# In[32]:


fig = cb_model.plot_cv_feature_importance(sub_title="CatBoost Model")
fig.show()


# # Xgboost

# In[33]:


import xgboost as xgb

class XgBoost(BaseModel):
    def __init__(self, features, categoricals=[], 
                 n_splits=5, fold_feat=None,
                 verbose=True, ps=None, target_col=target, early_stopping_rounds=500):
        # call superclass
        super().__init__(features=features, categoricals=categoricals, n_splits=n_splits, 
                         fold_feat=fold_feat,
                         verbose=verbose, ps=ps, target_col=target_col)
        self.early_stopping_rounds = early_stopping_rounds
        
    def train_model(self, train_set, val_set=None):
#        eval_set=[val_set] if val_set else None            
        verbosity = 100 if self.verbose else 0
        xgb_reg = xgb.XGBRegressor(**self.params)
        xgb_reg.fit(train_set[0],train_set[1], eval_set=[(val_set[0],val_set[1])], verbose=verbosity,
                   early_stopping_rounds=self.early_stopping_rounds)   
        return xgb_reg
    
    # override
    def convert_x(self, x):
        for cat in self.categoricals:
            x[cat] = x[cat].astype(int)
        return x
    
    def convert_dataset(self, x_train, y_train):
        for cat in self.categoricals:
            x_train[cat] = x_train[cat].astype(int)
        return x_train, y_train
    
    
    def get_default_params(self):
        params={'iterations' : 10000,
              'random_seed':self.RAND_SEED,
              'learning_rate': 0.01,
                "base_score": 0.5,
                "booster": 'gbtree',
                "tree_method": 'gpu_hist',
                "objective" : 'reg:linear',
                "max_depth": 3,
                "gpu_id": 0
        }
                         
        return params
    
    def get_cv_feature_importance(self):
        imp_df = pd.DataFrame(index=self.features)
        imp_df['importance'] = 0
        for model in self.cv_models:      
            imp_df['importance'] = (imp_df['importance'] + pd.Series(model.feature_importances_, index=model.get_booster().feature_names))/self.n_splits
        return imp_df.sort_values('importance', ascending=False)


# In[34]:


model_path = "../input/models/xgb_loc.sav"
if not os.path.isfile(model_path):
    xgb_model = XgBoost(
        features=test.columns, 
        n_splits=N_FLODS, 
        fold_feat=None,#"FOLD",  
        target_col=target,
        categoricals=cat_cols+['FOLD'], 
        ps=None, 
        verbose=2,
        early_stopping_rounds=50
    )
    xgb_model.fit_cv(train, save_cv=True)
    joblib.dump(xgb_model, "xgb_loc.sav")
else:
    xgb_model = joblib.load(xgb_model)


# In[35]:


print("XGBOOST OOF RMSE",xgb_model.oof_score_)


# In[36]:


fig = xgb_model.plot_cv_feature_importance(sub_title="XgBoost Model")
fig.show()


# In[37]:


sub3 = predict(test, xgb_model)
sub3.to_csv("submision_xgb.csv")


# ## Save oof preds:
# We will save oof predictions.
# 
# It can be used later to train a second level model such as a linear regression model

# In[38]:


oof = pd.DataFrame(data={
    "fold":lgb_model.cv_df["fold"],
    "lgb" : lgb_model.cv_df["y_pred"],
    "cb" : cb_model.cv_df["y_pred"],
    "xgb": xgb_model.cv_df["y_pred"],
    "y_true": lgb_model.cv_df["y_pred"]
},
                   index=lgb_model.cv_df.index
                  )

oof.to_csv("oof_preds.csv")
oof.head()


# # Predictions distributions
# 
# For each model, we will plot the histogram of the test dataset predictions

# In[39]:


fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Histogram(
    x=sub1.to_list(),
    histnorm='percent',
    name='lgb', # name used in legend and hover labels
    marker_color='#71A5F8',
    opacity=0.75,
    xbins=dict(
        start=sub1.min(),
        end=sub1.max(),
        size=0.5
    )
),
              row=1, col=1)

fig.add_trace(go.Histogram(
    x=sub2.to_list(),
    histnorm='percent',
    name='cb', # name used in legend and hover labels
    marker_color='#F871A5',
    opacity=0.75,
    xbins=dict(
        start=sub2.min(),
        end=sub2.max(),
        size=0.5
    )
),
              row=1, col=2)

fig.add_trace(go.Histogram(
    x=sub3.to_list(),
    histnorm='percent',
    name='xgb', # name used in legend and hover labels
    marker_color='#A5F871',
    opacity=0.75,
    xbins=dict(
        start=sub3.min(),
        end=sub3.max(),
        size=0.5
    )
),
              row=1, col=3)
              
fig.update_layout(template=temp,
                title={
                  "text": "<b>Prediction Distrubution </b> <BR /> <br><br><br>",
                  "x":0.035,
                  "font_size": 20,

              },
                  legend=dict(
                orientation="h"
                ),
                  yaxis_automargin=False,
                  height = 600,
                  plot_bgcolor = '#fcfaf1',
                  paper_bgcolor = '#fcfaf1',
                  margin={'pad':10},

)  


# # Averaged Submission 

# In[40]:


sub = (sub1+sub2+sub3)/3
sub.to_csv("avg_submission.csv")


# In[41]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=sub.to_list(),
    histnorm='percent',
    name='Averaged', # name used in legend and hover labels
    marker_color='#71A5F8',
    opacity=0.75,
    xbins=dict(
        start=sub.min(),
        end=sub.max(),
        size=0.5
    )
))

fig.update_layout(template = temp,
                  yaxis_automargin=False,
                  height = 500,
                  plot_bgcolor = '#fcfaf1',
                  paper_bgcolor = '#fcfaf1',
                  title={
                      "text": "<b>Averaged test prediction Histogram</b><br> ",
                      "x":0.035,
                      "font_size": 20,                    
                  },
                  margin={'pad':10},
)
fig.show()


# In[ ]:




