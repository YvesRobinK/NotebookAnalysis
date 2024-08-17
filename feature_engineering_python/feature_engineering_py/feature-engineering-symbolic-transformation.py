#!/usr/bin/env python
# coding: utf-8

# # ⚡️Summary ⚡️
# 
# This notebook resulted from a re-emergence of an statistical methodology to predict and transform data.  
# **Symbolic regression** is a machine learning technique that aims to identify an underlying mathematical expressions that best describes relationship in data. 
# We can therefore use Symbolic Regression for:
# * Predicting data (Regression and Classification
# * Feature Creation
# 
# The idea came from a fantastic notebook from [@Jano123](https://www.kaggle.com/code/jano123/sr-with-inequalities-simple-model) who used symbolic regression to predict a classification problem \
# In this notebook we use  [GPLEARN, a Symbolic Regression Python Library](https://gplearn.readthedocs.io/en/stable/intro.html) to **Generate Features** using **Symbolic Transformation**

# In[1]:


get_ipython().system('pip install gplearn')


# In[2]:


import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

import gc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, cohen_kappa_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

import lightgbm as lgb
import catboost as cat
import xgboost as xgb


# In[3]:


#suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# parameters 
ADD_DATA = True

EPOCHS = 5000
SCALING = True

SPLITS = 5
SPLITS_REPEAT = 3

name = "lightgbm"#"lightgbm" xgboost rf catboost


# In[5]:


# create a custom metric for lightgbm
def kappa_score(dy_true, dy_pred):
    pred_labels = dy_pred.reshape(len(np.unique(dy_true)),-1).argmax(axis=0)
    
    ks = cohen_kappa_score(dy_true, pred_labels, weights ='quadratic' )
    #ks = accuracy_score(dy_true, pred_labels)
    
    is_higher_better = True

    return "kappa_score", ks, is_higher_better


# ## 💾 Load Data 💾
# Data taken from the [Playground Series Competition S03E05](https://www.kaggle.com/competitions/playground-series-s3e5)

# In[6]:


target = "quality"

project_purpose = "Wine Quality"
metric = "kappa"

train_data_dir = "/kaggle/input/playground-series-s3e5/train.csv"
test_data_dir = "/kaggle/input/playground-series-s3e5/test.csv"
submission_dir = "/kaggle/input/playground-series-s3e5/sample_submission.csv"


# In[7]:


df_train = pd.read_csv(train_data_dir, index_col = 0)
df_test = pd.read_csv(test_data_dir, index_col = 0)
sub = pd.read_csv(submission_dir,index_col = 0)


# In[8]:


if ADD_DATA:
    add_data = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv', index_col = "Id")

    df_train['is_generated'] = 1
    df_test['is_generated'] = 1
    add_data['is_generated'] = 0

    df_train = pd.concat([df_train, add_data],axis=0, ignore_index=True)
df_train


# # 🎯 Manual Feature Engineering 🎯
# We will create a number of manual features as these were identified by some [clever people](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698) in the kaggle competition for this dataset 

# In[9]:


df_trn = df_train.copy(deep = True)
df_tst = df_test.copy(deep = True)


# In[10]:


print(df_trn.duplicated().sum())
df_tst.drop_duplicates(inplace = True,ignore_index  = True)
df_trn.drop_duplicates(inplace = True,ignore_index  = True)
print(df_trn.duplicated().sum())


# In[11]:


def Additional_Features(df_in):
    df = df_in.copy(deep = True)
    #df['mso2'] = df['free sulfur dioxide']/(1+ 10**(df['pH'] -1.81))
    
    df['alcohol_mul_sulphates'] = df['alcohol'] * df['sulphates']
    df['chlorides_mul_sulphates'] = df['chlorides'] * df['volatile acidity']
    
    #old
    df['alcohol_pH'] = df['alcohol'] * df['pH']
    df['alcohol_residual_sugar'] = df['alcohol'] * df['residual sugar']
    df['pH_residual_sugar'] = df['pH'] * df['residual sugar']
    df['alcohol_citric_acid'] = df['alcohol'] * df['citric acid']
    df['total_acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['acid/density'] = df['total_acid']  / df['density']
    df['alcohol_density'] = df['alcohol']  * df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']
    df['sulphates/acid'] = df['sulphates'] / df['volatile acidity']
    df['sulphates/chlorides'] = df['sulphates'] / df['chlorides']
    df['sulphates*alcohol'] = df['sulphates'] / df['alcohol']
    df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
    df['sugar_to_chlorides'] = df['residual sugar'] / df['chlorides']
    df['alcohol_to_density'] = df['alcohol'] / df['density']
    # Create interaction features between alcohol content and volatile acidity
    df['alcohol_volatile_acidity'] = df['alcohol'] * df['volatile acidity']

    # Bin the alcohol content feature
    alcohol_bins = np.linspace(df['alcohol'].min(), df['alcohol'].max(), 5)
    df['alcohol_binned'] = np.digitize(df['alcohol'], alcohol_bins)

    # Bin the volatile acidity feature
    volatile_acidity_bins = np.linspace(df['volatile acidity'].min(), df['volatile acidity'].max(), 5)
    df['volatile_acidity_binned'] = np.digitize(df['volatile acidity'], volatile_acidity_bins)

    # Bin the sulphates feature
    sulphates_bins = np.linspace(df['sulphates'].min(), df['sulphates'].max(), 5)
    df['sulphates_binned'] = np.digitize(df['sulphates'], sulphates_bins)

    # Create interaction features between binned features
    df['alcohol_binned_sulphates_binned'] = df['alcohol_binned'] * df['sulphates_binned']
    df['alcohol_binned_volatile_acidity_binned'] = df['alcohol_binned'] * df['volatile_acidity_binned']
    df['sulphates_binned_volatile_acidity_binned'] = df['sulphates_binned'] * df['volatile_acidity_binned']
    
    for col in ['alcohol_mul_sulphates']: 
        df[f"q01_{col}_quant"] = df[col] - df[col].quantile(q= 0.1)
        df[f"q25_{col}_quant"] = df[col] - df[col].quantile(q= 0.25)
        df[f"q50_{col}_quant"] = df[col] - df[col].quantile(q= 0.5) 
        df[f"q75_{col}_quant"] = df[col] - df[col].quantile(q= 0.75) 
        df[f"q95_{col}_quant"] = df[col] - df[col].quantile(q= 0.95) 
        df[f"q99_{col}_quant"] = df[col] - df[col].quantile(q= 0.99)

    return df

df_trn = Additional_Features(df_trn)
df_tst = Additional_Features(df_tst)
df_trn


# In[12]:


#map the classes to start from 0
df_trn[target] = df_trn[target].map({3:0,
                    4:1,
                    5:2,
                    6:3,
                    7:4,
                    8:5})


# # 🌓 Split 🌓

# In[13]:


# drop the text column and target
X = df_trn.drop([target],axis =1)
y= df_trn[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[14]:


sklearn_weights= compute_class_weight('balanced', classes = np.unique(y),y= y)
xgboost_weights = y.map(pd.DataFrame(sklearn_weights).to_dict()[0])
xgboost_weights


# In[15]:


xgb_params = { 
    'booster' : 'gbtree', # gblinear or dart gbtree
    'tree_method' : 'hist', #exact, approx, hist, gpu_hist auto
    'objective' : "multi:softproba",
    'num_class' : 6,
    'n_estimators' : 500, 
    'custom_metric':kappa_score,
    'colsample_bytree': 0.1, # sample of columns (ratio)
    'subsample' : 0.95, # sample of rows (ratio)
     'learning_rate': 0.04,
    #'min_child_weight' : 10 # larger reduces overfitting
    'gamma' : 1,
    'max_depth': 5,
    'max_delta_step' : 10 #results in a more constrained probability. should help with balancing the dataset
             }


# In[16]:


def single_model(train_df, test_df):
    X = train_df.drop([target],axis =1)
    y= train_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model= xgb.XGBClassifier(**xgb_params)

    model.fit(X_train,y_train,
              sample_weight=xgboost_weights[X_train.index], #Combat class imbalance
                verbose= 0
             )
    y_preds = model.predict_proba(X_test)
    test_preds = model.predict_proba(test_df)
    val_score= cohen_kappa_score(y_test, y_preds.argmax(axis =1), weights ='quadratic')
    val_logloss = log_loss(y_test, y_preds)
    print("Val score:",val_score)
    print("logloss:",val_logloss)
    return model, val_score, val_logloss


# Run a base model to check the score 

# In[17]:


model, score_base, logloss_base =single_model(df_trn, df_tst)


# # 👩‍🔬 Symbolic Transformation (Feature Generation) 👩‍🔬
# * We will use Symbolic transformation to create 100 feature (hall_of_fame) and then select 10 components from these (i.e. non-correlated features) 
# More examples can be found on thie [gplearn website](https://gplearn.readthedocs.io/en/stable/examples.html)

# In[18]:


from gplearn.genetic import SymbolicTransformer


# In[19]:


function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'abs', 'neg', 'inv', 'max', 'min']
gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, metric='pearson') #option to use metric = 'spearman'
gp.fit(X_train, y_train)


# In[20]:


sym_trn = gp.transform(X)
sym_tst = gp.transform(df_tst)

sym_trn = pd.DataFrame(sym_trn, columns = [f"sym_{col}" for col in range(sym_tst.shape[1])])
sym_tst = pd.DataFrame(sym_tst, columns = [f"sym_{col}" for col in range(sym_tst.shape[1])])
sym_trn


# In[21]:


new_df_trn = pd.concat([df_trn,sym_trn],axis =1)
new_df_tst= pd.concat([df_tst,sym_tst],axis =1)
new_df_trn


# # 🤹 Rerun Model 🤹

# In[22]:


model_new, score_new, logloss_new  =single_model(new_df_trn,new_df_tst)


# In[23]:


print("Without Symbolic Features:", score_base)
print("Including Symbolic Features:", score_new)


# # 📌 Conclusion 📌
# As we can see, the additional features improved our model performane 

# In[ ]:




