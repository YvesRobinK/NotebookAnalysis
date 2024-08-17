#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import KFold

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')


# In[2]:


train = pd.read_csv(r'../input/playground-series-s3e11/train.csv')
test_1 = pd.read_csv(r'../input/playground-series-s3e11/test.csv')
orig_train = pd.read_csv(r'../input/media-campaign-cost-prediction/train_dataset.csv')
orig_test = pd.read_csv(r'../input/media-campaign-cost-prediction/test_dataset.csv')

train.drop('id', axis = 1, inplace = True)
test = test_1.drop('id', axis = 1)


# # Knowing Your Data
# 
# ## Descriptive Statistics

# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


orig_train.head()


# In[6]:


orig_test.head()


# In[7]:


train.describe().T


# In[8]:


orig_train.describe().T


# ## Null Values and Duplicates

# In[9]:


print(f'There are {train.isna().sum().sum()} null values in train dataset')
print(f'There are {test.isna().sum().sum()} null values in test dataset')
print(f'There are {orig_train.isna().sum().sum()} null values in original train dataset')
print(f'There are {orig_test.isna().sum().sum()} null values in original test dataset\n')

print(f'There are {train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {train.count()[0]} rows in train dataset')
print(f'There are {test.duplicated().value_counts()[0]} non-duplicate values out of {test.count()[0]} rows in test dataset')
print(f'There are {orig_train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {orig_train.count()[0]} rows in original train dataset')
print(f'There are {orig_test.duplicated().value_counts()[0]} non-duplicate values out of {orig_test.count()[0]} rows in original test dataset')


# In[10]:


orig_train.drop_duplicates(subset = list(train)[0:-1], inplace = True)


# ## Distribution

# In[11]:


fig, ax = plt.subplots(5, 3, figsize = (10, 13), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(test.columns):
    sns.kdeplot(train[column], ax=ax[i], color=pal[0])    
    sns.kdeplot(test[column], ax=ax[i], color=pal[2])
    sns.kdeplot(orig_train[column], ax=ax[i], color=pal[4])
    sns.kdeplot(orig_test[column], ax=ax[i], color=pal[1])
    
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)
    
fig.suptitle('Distribution of Feature\nper Dataset\n', fontsize = 24, fontweight = 'bold')
fig.legend(['Train', 'Test', 'Original Train', 'Original Test'])
plt.tight_layout()


# In[12]:


plt.figure(figsize = (10, 5), dpi = 300)

sns.kdeplot(train['cost'], color = pal[0], fill = True)
sns.kdeplot(orig_train['cost'], color = pal[2], fill = True)

plt.title('Distribution of Cost per Dataset\n', weight = 'bold', fontsize = 25)
plt.legend(['Train', 'Original Train'])
plt.show()


# **Key points:** Both original datasets and the competition datasets have similar distribution!

# ## Correlation

# In[13]:


def heatmap(dataset, label = None):
    corr = dataset.corr()
    plt.figure(figsize = (14, 10), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, annot = True, annot_kws = {'size' : 7}, cmap = 'viridis')
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
    plt.show()


# In[14]:


heatmap(train, 'Train')
heatmap(test, 'Test')
heatmap(orig_train, 'Original Train')
heatmap(orig_test, 'Original Test')


# **Key points:**
# 1. There is perfect correlation between `salad_bar` and `prepared_food`. This means that all prepared foods come from salad bar. Dropping one of them will surely give us better result.
# 2. Moderate correlation is found between `unit_sales` and `store_sales`. We can engineer these features to get average price. (I have tested that this doesn't impact the CV so I won't engineer it)
# 3. We can engineer two new features from `num_children_at_home` and `total_children`. First, we can get the ratio of children. Second, we can get the number of independent children.
# 4. There is moderate correlation between `salad_bar` and `store_sqft`. Out of all building features, it seems salad bar has the biggest space as it increases the store area the most. Strangely, other facilities seem to reduce store area. On another note, moderate to strong correlations are found in between a lot of facilities. Let's try to make one feature that indicates the total facilities available.
# 
# **Additional note:**
# Some features my not have apparent correlation here, but we can still engineer them to create new features, such as units per store area.

# # Feature Engineering

# In[15]:


train = pd.concat([train, orig_train], axis = 0)


# ## Dropping Unused Feature

# In[16]:


train.drop('prepared_food', axis = 1, inplace = True)
test.drop('prepared_food', axis = 1, inplace = True)


# ## Creating New Features
# 
# ### Children Ratio

# In[17]:


train['child_ratio'] = train.eval('total_children / num_children_at_home')
train.replace([np.inf, -np.inf], 10, inplace = True)
train.fillna(0, inplace = True)

test['child_ratio'] = test.eval('total_children / num_children_at_home')
test.replace([np.inf, -np.inf], 10, inplace = True)
train.fillna(0, inplace = True)


# ### Independent Children

# In[18]:


train['independent_children'] = train.eval('total_children - num_children_at_home')
test['independent_children'] = test.eval('total_children - num_children_at_home')


# ### Facilities

# In[19]:


train['facilities'] = train.eval('coffee_bar + video_store + salad_bar + florist')
test['facilities'] = test.eval('coffee_bar + video_store + salad_bar + florist')


# # Model

# In[20]:


X = train.copy()
y = X.pop('cost')

seed = 42
splits = 10
repeats = 1 

np.random.seed(seed)


# In XGBoost you really don't need to do any predictor feature manipulation to be able to use RMSLE. Obviously however, you can't simply just use XGBoost for building actual model.
# 
# **Note:** Actually using RMSLE in XGBoost will slow down the process by a lot. Try enabling GPU P100 accelerator after you finish parameter tuning and pre-processing.

# In[21]:


xgb_params = {
    'seed': seed,
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method' : 'gpu_hist',
    'n_jobs' : -1,
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = KFold(n_splits=splits, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    dtrain = xgb.DMatrix(
        data=X.iloc[train_idx], 
        label=y.iloc[train_idx]
    )
    
    dvalid = xgb.DMatrix(
        data=X.iloc[val_idx], 
        label=y.iloc[val_idx]
    )

    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain,
        verbose_eval = False,
        num_boost_round = 5000,
        evals=[(dtrain, 'train'), 
               (dvalid, 'eval')], 
        callbacks=[xgb.callback.EarlyStopping(rounds=100,
                                              data_name='eval',
                                              maximize=False,
                                              save_best=True)]
    )
    
    best_iter = xgb_model.best_iteration
    
    train_preds = xgb_model.predict(dtrain)
    val_preds = xgb_model.predict(dvalid)
    
    train_score = msle(y.iloc[train_idx], train_preds, squared = False)
    val_score = msle(y.iloc[val_idx], val_preds, squared = False)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += xgb_model.predict(xgb.DMatrix(test)) / splits
    print(f'Fold {fold}: val RMSLE = {val_score:.5f} | train RMSLE = {train_score:.5f} | best_iter = {best_iter}')

print(f'Average val RMSLE = {np.mean(val_scores):.5f} | train RMSLE = {np.mean(train_scores):.5f}')
xgb_preds = predictions.copy()


# Keep in mind that this isn't final model. You have to do your own feature engineering and model building after this!
# 
# # Submission

# In[22]:


test_1.drop(list(test_1.drop('id', axis = 1)), axis = 1, inplace = True)


# In[23]:


test_1['Class'] = xgb_preds
test_1.to_csv('submission.csv', index = False)


# Thank you for reading!
