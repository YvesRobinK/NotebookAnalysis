#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sns.set_palette('Set2')
sns.set_theme(style='darkgrid')


# # About the Competition🚩
# <p style="font-size:15px">Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.<br><br>
# 
# 
# The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.<br><br>
# 
# For this competition, you will predict whether a customer made a claim upon an insurance policy. The ground truth claim is binary valued, but a prediction may be any number from 0.0 to 1.0, representing the probability of a claim. The features in this dataset have been anonymized and may contain missing values.<br><br>
# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
# </p>

# <div class="alert alert-block alert-info" style="font-size:15px; font-family:verdana; line-height: 2.0em;">
# This month comptetion is a binary class classificaton in which we have to predict probiblities for   'claim' feature
# </div>

# # Data Description

# <div style="font-size:15px">
#  We are given 3 csv files:-
# <ul>
#     <li><code>train.csv:</code> the training set</li>
#     <li><code>test.csv:</code> the test set</li>
#     <li><code>sample_submission.csv:</code> sample_submission file in submission format</li>
# </ul>    
# </div>

# # EDA

# # Color Palette

# In[2]:


sns.color_palette('Set2')


# In[3]:


train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')


# let's see how train data looks like

# In[4]:


train.head()


# In[5]:


train.info()


# basic train set statistics

# In[6]:


train.describe()


# <div class="alert alert-block alert-warning" style="font-size:15px; font-family:verdana; line-height: 2.0em;">
# Our train set is very diverse so scaling is necessary
# </div>

# Let's check if missing values are present

# In[7]:


def check_missing(df):
    print(bool(df.isnull))


# In[8]:


check_missing(train)


# missing values are present we will need to deal with them before modeling

# In[9]:


missing_values = pd.DataFrame(train.isna().sum())
missing_values.rename(columns={0:'missing_value'},inplace=True)
def train_missing_perecentage(idx):
    return (idx/len(train))*100
missing_values['missing_value'] = missing_values.apply(train_missing_perecentage)
features = list(train.columns)
percentage = []
for i in features:
    percentage.append(float(missing_values.loc[str(i)]))
missing_values = pd.DataFrame({'Feature':features,'Percentage':percentage})


# let's look perecentage of missing value for each feature

# In[10]:


px.scatter(data_frame=missing_values,x='Feature',y='Percentage',template='plotly_dark')


# it's safe to say around 1.6% of data is missing in every feature exception are Id and claim

# Let's take a look at our target variable:- "claim"

# In[11]:


sns.countplot(x=train.claim,palette='Set2')


# target is evenly distributed so we can simply use K-Fold

# <div class="alert alert-block alert-info" style="font-size:15px; font-family:verdana; line-height: 2.0em;">
# to quickly view detailed EDA on each features we can use pandas profiling
# </div>

# In[12]:


get_ipython().system('pip install pandas-profiling')
from pandas_profiling import ProfileReport
prof = ProfileReport(train,minimal=True) 
prof.to_file(output_file='train_output.html')


# Let's check how features correlate with each other

# In[13]:


features = list(train.columns)
features.remove('id')
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(train[features].corr(), annot=False, linewidths=.5, ax=ax,cmap=sns.color_palette('Set2',as_cmap=True))


# <div class="alert alert-block alert-info" style="font-size:15px; font-family:verdana; line-height: 2.0em;">
# Usually, we keep features that are highly correlated with our target variable and remove the reductant feature but in this case features are not highly correlated to each other
# </div>

# Now let's check our test set

# In[14]:


test.head()


# In[15]:


test.info()


# In[16]:


test.describe()


# Let's check missing value in test set

# In[17]:


check_missing(test)


# In[18]:


missing_values = pd.DataFrame(test.isna().sum())
missing_values.rename(columns={0:'missing_value'},inplace=True)
def test_missing_perecentage(idx):
    return (idx/len(test))*100
missing_values['missing_value'] = missing_values.apply(test_missing_perecentage)
features = list(test.columns)
percentage = []
for i in features:
    percentage.append(float(missing_values.loc[str(i)]))
missing_values = pd.DataFrame({'Feature':features,'Percentage':percentage})


# let's look at percentage of missing value in case of test set

# In[19]:


px.scatter(data_frame=missing_values,x='Feature',y='Percentage',template='plotly_dark')


# it's safe to say around 1.6% of data is missing in every feature exception are Id

# let's generate padnas profiling for test set

# In[20]:


prof = ProfileReport(test,minimal=True) 
prof.to_file(output_file='test_output.html')


# let's see how different features correlate to each other in case of test set

# In[21]:


features = list(test.columns)
features.remove('id')
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(test[features].corr(), annot=False, linewidths=.5, ax=ax,cmap=sns.color_palette('Set2',as_cmap=True))


# # Modeling

# Let's train a simple LightGBM Classifier and setup our Baseline for rest of the competition 

# In[22]:


train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv',index_col='id')
test = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv',index_col='id')
sample_submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
features = list(train.columns)
features.remove('claim')
target = ['claim']


# # Training

# Hyperparameters are optimized using Optuna they can be further optimized by using better suggestions

# In[23]:


params = {
    'task': 'train',    
    'objective': 'binary',
    'verbose':-1,
    'num_leaves': 111, 
    'learning_rate': 0.016206997849237542, 
    'n_estimators': 2641, 
    'min_child_samples': 22,
}
folds = KFold(n_splits=5,shuffle=True,random_state=42)
for fold, (train_idx,valid_idx) in enumerate(folds.split(train)):
    print(f'fold {fold} starting...')
    fold_train = train.iloc[train_idx]
    train_x = fold_train[features]
    train_y = fold_train[target]
    dtrain = lgb.Dataset(train_x,label=train_y)
    
    fold_valid = train.iloc[valid_idx]
    valid_x = fold_valid[features]
    valid_y = fold_valid[target]
    dvalid = lgb.Dataset(valid_x,valid_y)
    
    model = lgb.train(params,train_set=dtrain, 
               valid_sets=dvalid,
              early_stopping_rounds =200,
                     verbose_eval=100)
    oof = model.predict(valid_x)
    score = roc_auc_score(valid_y,oof)
    print(f"Valid score for {fold} is: {score}")
    oof = pd.DataFrame({'id':valid_x.index,'claim':oof})
    oof.to_csv(f'{fold}_oof.csv',index=False)
    model.save_model(f'lightgbm_{fold}.txt')
    print(f' fold {fold} completed')


# # Inference

# In[24]:


for fold in tqdm(range(5)):
    model = lgb.Booster(model_file=f'./lightgbm_{fold}.txt')
    preds = model.predict(test)
    submission = sample_submission.copy()
    submission['claim'] = preds
    submission.to_csv(f'submission_{fold}.csv',index=False)


# # Blending

# In[25]:


sub0 = pd.read_csv('./submission_0.csv')
sub1 = pd.read_csv('./submission_1.csv')
sub2 = pd.read_csv('./submission_2.csv')
sub3 = pd.read_csv('./submission_3.csv')
sub4 = pd.read_csv('./submission_4.csv')
target = (sub0.claim + sub1.claim + sub2.claim + sub3.claim + sub4.claim)/5
sub = sub0.copy()
sub['claim'] = target
sub.to_csv('submission.csv',index=False)


# # Feature Importance

# In[26]:


feature_importance =  model.feature_importance()
feature_importance = (feature_importance - feature_importance.min())/(feature_importance.max() - feature_importance.min())
feature_names = np.array(train_x.columns)
data={'feature_names':feature_names,'feature_importance':feature_importance}
df_plt = pd.DataFrame(data)
df_plt.sort_values(by=['feature_importance'], ascending=False,inplace=True)
plt.figure(figsize=(20,40))
sns.barplot(x=df_plt['feature_importance'], y=df_plt['feature_names'])
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')
plt.show()


# # Possible Next Steps:-

# <div class="alert alert-block alert-info" style="font-size:15px; font-family:verdana; line-height: 1.7em;">
#     📌Hyper Parameter Optimization<br>
#     📌Using Catboost, XGboost and maybe Neural Nets<br>
#     📌Feature Engineering<br>
#     📌Stacking<br>
#     📌Advanced Techniques like Psuedo Labelling, Gaussian optimization
# </div>

# <h2><center>If you learned something new or forked the notebook then please don't forget to upvote<br>Thank You</center>
# </h2>

# <h2><center>Work in Progress ... ⏳</center></h2>
