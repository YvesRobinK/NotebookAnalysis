#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import statsmodels.api as sm

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')


# In[2]:


train = pd.read_csv(r'../input/playground-series-s3e12/train.csv')
test_1 = pd.read_csv(r'../input/playground-series-s3e12/test.csv')
orig_train = pd.read_csv(r'../input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv')

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


desc = train.describe().T
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isna().sum()
desc['type'] = train.dtypes
desc


# In[7]:


desc = test.describe().T
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isna().sum()
desc['type'] = test.dtypes
desc


# In[8]:


desc = orig_train.describe().T
desc['nunique'] = orig_train.nunique()
desc['%unique'] = desc['nunique'] / len(orig_train) * 100
desc['null'] = orig_train.isna().sum()
desc['type'] = orig_train.dtypes
desc


# # Duplicates

# In[9]:


print(f'There are {train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {train.count()[0]} rows in train dataset')
print(f'There are {test.duplicated().value_counts()[0]} non-duplicate values out of {test.count()[0]} rows in test dataset')
print(f'There are {orig_train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {orig_train.count()[0]} rows in original train dataset')


# # Distribution

# In[10]:


fig, ax = plt.subplots(3, 2, figsize = (10, 10), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(test.columns):
    sns.kdeplot(train[column], ax=ax[i], color=pal[0])    
    sns.kdeplot(test[column], ax=ax[i], color=pal[2])
    sns.kdeplot(orig_train[column], ax=ax[i], color=pal[1])
    
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)
    
fig.suptitle('Distribution of Feature\nper Dataset\n', fontsize = 24, fontweight = 'bold')
fig.legend(['Train', 'Test', 'Original Train'])
plt.tight_layout()


# **Key points:**
# 1. Train and test datasets have similar distribution so we can trust the CV.
# 2. Train and original train datasets also have relatively similar distribution, but we need to confirm this with adversarial validation.

# In[11]:


fig, ax = plt.subplots(3, 2, figsize = (10, 10), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(test.columns):
    sns.kdeplot(data = train, x = column, ax=ax[i], color=pal[0], fill = True, legend = False, hue = 'target')
    
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(None)
    
fig.suptitle('Distribution of Features per Class\n', fontsize = 24, fontweight = 'bold')
fig.legend(['Crystal (1)', 'No Crystal (0)'])
plt.tight_layout()


# In[12]:


fig, ax = plt.subplots(3, 2, figsize = (10, 10), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(test.columns):
    sns.kdeplot(data = orig_train, x = column, ax=ax[i], color=pal[0], fill = True, legend = False, hue = 'target')
    
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(None)
    
fig.suptitle('Distribution of Features per Class\n', fontsize = 24, fontweight = 'bold')
fig.legend(['Crystal (1)', 'No Crystal (0)'])
plt.tight_layout()


# **Key points:** It looks like `gravity`, `urea`, and `calc` have positive impact on `target`. We can confirm this causal relationship later.

# In[13]:


fig, ax = plt.subplots(1, 2, figsize = (16, 5))
ax = ax.flatten()

ax[0].pie(
    train['target'].value_counts(), 
    shadow = True, 
    explode = [0, 0.1], 
    autopct = '%1.f%%',
    textprops = {'size' : 20, 'color' : 'white'}
)

sns.countplot(data = train, y = 'target', ax = ax[1])
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count', fontsize = 20)
plt.xticks(fontsize = 12)

fig.suptitle('Target Feature in Train Dataset', fontsize = 25, fontweight = 'bold')
plt.tight_layout()


# In[14]:


fig, ax = plt.subplots(1, 2, figsize = (16, 5))
ax = ax.flatten()

ax[0].pie(
    orig_train['target'].value_counts(), 
    shadow = True, 
    explode = [0, 0.1], 
    autopct = '%1.f%%',
    textprops = {'size' : 20, 'color' : 'white'}
)

sns.countplot(data = orig_train, y = 'target', ax = ax[1])
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count', fontsize = 20)
plt.xticks(fontsize = 12)

fig.suptitle('Target Feature in Original Train Dataset', fontsize = 25, fontweight = 'bold')
plt.tight_layout()


# **Key points:** `target` has relatively balanced distribution

# ## Correlation

# In[15]:


def heatmap(dataset, label = None):
    corr = dataset.corr()
    plt.figure(figsize = (14, 10), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, annot = True, annot_kws = {'size' : 14}, cmap = 'viridis')
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
    plt.show()


# In[16]:


heatmap(train, 'Train')
heatmap(test, 'Test')
heatmap(orig_train, 'Original Train')


# **Key points:**
# 1. All features except `ph` are correlated with each others.
# 2. `target` is moderately correlated with `calc` and `gravity`

# # Causal Relationship

# In[17]:


sm.Logit(train['target'], sm.add_constant(train.drop('target', axis = 1))).fit().summary()


# **Key points:** Only `calc` has significant impact on `target`.
# 
# **Additional note:** There might be a better way to determine a feature importance. However, due to how small our data is, I assume a simple method like this will work well. Don't do this for more complex dataset though.

# # Feature Engineering
# 
# Let's combine both original and train dataset and see how many duplicates are there.

# In[18]:


train = pd.concat([train, orig_train])

print(f'There are {train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {train.count()[0]} rows in train dataset')


# Now we should delete the duplicates.

# In[19]:


train.drop_duplicates(subset = list(train)[0:-1], inplace = True, keep = 'first')


# In[20]:


X = train.copy()
y = X.pop('target')

seed = 42
splits = 10
repeats = 10
k = RepeatedStratifiedKFold(n_splits=splits, random_state=seed, n_repeats = repeats)

np.random.seed(seed)


# Let's try to make our model as simple as possible by only using significant features.

# In[21]:


X = X[['calc']]
test = test[['calc']]


# # Model
# 
# I'll try using `StackingClassifier` for our model. Because the competition uses AUROC to evaluate our submission, we should use `roc_auc_score` as the parameter value for our metric in XGBoost. Because of the small dataset size, I'll use Repeated Stratified KFold for cross-validation.

# In[22]:


xgb_params = {
    'seed': seed,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = RepeatedStratifiedKFold(n_splits=splits, random_state=seed, n_repeats = repeats)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    stack = StackingClassifier(
        [
            ('xgb', XGBClassifier(**xgb_params)),
            ('lr', LogisticRegression(random_state = seed, max_iter = 10000)),
            ('ext', ExtraTreesClassifier(random_state = seed))
        ]
    )
    
    stack.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    train_preds = stack.predict_proba(X.iloc[train_idx])[:,1]
    val_preds = stack.predict_proba(X.iloc[val_idx])[:,1]
    
    train_score = roc_auc_score(y.iloc[train_idx], train_preds)
    val_score = roc_auc_score(y.iloc[val_idx], val_preds)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += stack.predict_proba(test)[:,1] / splits / repeats
    #print(f'Fold {fold // repeats} Iteration {fold % repeats}: val ROC = {val_score:.5f} | train ROC = {train_score:.5f}')

print(f'Average val ROC = {np.mean(val_scores):.5f} | train ROC {np.mean(train_scores):.5f}')
stack_preds = predictions.copy()


# # Submission

# In[23]:


test_1.drop(list(test_1.drop('id', axis = 1)), axis = 1, inplace = True)


# In[24]:


test_1['target'] = stack_preds
test_1.to_csv('submission.csv', index = False)


# Thank you for reading!
