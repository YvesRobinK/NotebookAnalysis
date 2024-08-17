#!/usr/bin/env python
# coding: utf-8

# # Tabular Playground #1: Bimodal Regression
# 
# * [Problem Definition](#Problem-Definition)<br>
# * [Modality](#Modality)<br>
# * [Data Overview](#Data-Overview)<br>
#     * [Target](#Target)<br>
#     * [cont# Features](#cont-Features)<br>
# * [Model Baseline](#Model-Baseline)<br>
# * [Feature Engineering Techniques](#Feature-Engineering-Techniques)<br>
#     * [Gaussian Mixture Modelling](#Gaussian-Mixture-Modelling)<br>
#     * [Binning](#Binning)<br>
#     * [Statistical Features](#Statistical-Features)<br>
#     * [Deep Feature Synthesis](#Deep-Feature-Synthesis)<br>
#     * [Summary](#Summary)<br>
# * [EDA](#EDA)<br>
# 
# # Problem Definition
# 
# In this challenge, we are asked to build a **regression** model. Without further context, we are given some features with **continuous values** to predict a continuous target. This way, we can practice on focussing on the data without requiring any specific domain knowledge because the column names `cont#` do not indicate any further information.
# 
# There is no missing data but instead we have a different obstacle that we have to overcome - **bimodal distribution of the target variable** and multimodal distributions of the features.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Configurations
N_SPLITS = 5
SEED = 42

train_df = pd.read_csv("../input/tabular-playground-series-jan-2021/train.csv")
display(train_df.head())
#print('Train data dimension: ', train_df.shape)

test_df = pd.read_csv("../input/tabular-playground-series-jan-2021/test.csv")
#print('Test data dimension: ', test_df.shape)
#display(test_df.head())

sample_submission = pd.read_csv("../input/tabular-playground-series-jan-2021/sample_submission.csv")
#print('Sample submission dimension: ', sample_submission.shape)
#display(sample_submission.head())


print(f"Missing data in the train data: {train_df.isna().sum(axis=0).any()}")
print(f"Missing data in the test data: {test_df.isna().sum(axis=0).any()}")


# # Modality
# 
# In this challenge, we will learn about the modality of a distribution. You can find the modality of a distribution by counting the number of its peaks. There are unimodal, bimodal and multimodal distributions.
# The most commonly known distribution is unimodal with only one peak. This is probably also the most comfortable distribution to work with. If you have two peaks, it is called bimodal, and if you have three or more peaks, then it is called multimodal.
# 
# Also, **don't get it confused with "multimodal learning"**, which describes problems with mixed feature modalities, such as pictures and text.

# In[2]:


f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

# Unimodal
sns.distplot(np.random.normal(10, 5, 10000), ax=ax[0], hist=False, color='blue')
ax[0].set_title('Unimodal', fontsize=14)
ax[0].set_yticklabels([])
ax[0].set_xticklabels([])

# Bimodal
sample_bimodal = pd.DataFrame({'feature1' : np.random.normal(10, 5, 10000),
                   'feature2' : np.random.normal(40, 10, 10000),
                   'feature3' : np.random.randint(0, 2, 10000),
                  })

sample_bimodal['combined'] = sample_bimodal.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else x.feature2, axis=1)

sns.distplot(sample_bimodal['combined'].values, ax=ax[1], color='blue', hist=False)

ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis

sns.distplot(sample_bimodal.feature1, ax=ax2, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_bimodal.feature2), ax=ax2, color='blue', kde_kws={'linestyle':'--'}, hist=False)

f.tight_layout()  # otherwise the right y-label is slightly clipped

ax[1].set_title('Bimodal', fontsize=14)
ax[1].set_yticklabels([])
ax[1].set_xticklabels([])
ax2.set_yticklabels([])


# Multimodal
sample_multi = pd.DataFrame({'feature1' : np.random.normal(10, 5, 10000),
                   'feature2' : np.random.normal(40, 10, 10000),
                   'feature3' : np.random.randint(0, 3, 10000),
                               'feature4' : np.random.normal(80, 4, 10000),
                  })

sample_multi['combined'] = sample_multi.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else (x.feature2 if x.feature3 == 1 else x.feature4), axis=1 )

sns.distplot(sample_multi['combined'].values, ax=ax[2], color='blue', hist=False)

ax3 = ax[2].twinx()  # instantiate a second axes that shares the same x-axis

sns.distplot(sample_multi.feature1, ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_multi.feature2), ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_multi.feature4), ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)

f.tight_layout()  # otherwise the right y-label is slightly clipped

ax[2].set_title('Multimodal', fontsize=14)
ax[2].set_yticklabels([])
ax[2].set_xticklabels([])
ax3.set_yticklabels([])

plt.show()


# # Data Overview
# 
# ## Target
# * It looks like an overlay of two different distributions - when the data distribution has two peaks, it is called **bimodal distribution**.
# * There is exactly one data point with a target value of 0 - this looks very much like an **outlier**. We should probably drop it since it is only one single data point.
# * The data does not seem to be skewed and therefore does not necessarily need to be transformed (if non-tree-based models are used - for tree-based models this would not matter anyways).

# In[3]:


#display(train_df.target.describe())
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
sns.distplot(train_df.target, ax=ax[0])
sns.boxplot(train_df.target, ax=ax[1])
stats.probplot(train_df['target'], plot=ax[2])
plt.show()


# In[4]:


# Drop one outlier
train_df = train_df[train_df.target != 0].reset_index(drop=True)


# In[5]:


df = pd.DataFrame({'feature1' : np.random.normal(10, 5, 1000),
                   'feature2' : np.random.normal(40, 10, 1000),
                   'feature3' : np.random.randint(0, 2, 1000),
                  })

df['target'] = df.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else x.feature2, axis=1)

display(df.head())
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))
color = 'blue'
sns.distplot(df.target, ax=ax, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'red'
sns.distplot(df.feature1, ax=ax2, color=color, hist=False)
sns.distplot((df.feature2), ax=ax2, color=color, hist=False)

ax2.tick_params(axis='y', labelcolor=color)
ax.set_title('Example of a Bimodal Distribution', fontsize=16)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# ## cont# Features
# Let's look at the `cont#` features in bulk.
# For below plots we can see some **odd distributions: all `cont#` features show multiple 'peaks' with no sign of a normal distribution.**
# 
# Um, ok... what is going on?! We definitely need to dig deeper here before we can start building our model.
# 
# We already saw that our `target` variable has a bimodal distribution. Now we have some feature distributions with multiple peaks. This is called **multimodal distribution**.
# 

# In[6]:


f, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
#f.suptitle('Distribution of Features', fontsize=16)
sns.distplot(train_df['cont1'], ax=ax[0, 0])
sns.distplot(train_df['cont2'], ax=ax[0, 1])
sns.distplot(train_df['cont3'], ax=ax[0, 2])
sns.distplot(train_df['cont4'], ax=ax[0, 3])

sns.distplot(train_df['cont5'], ax=ax[1, 0])
sns.distplot(train_df['cont6'], ax=ax[1, 1])
sns.distplot(train_df['cont7'], ax=ax[1, 2])
sns.distplot(train_df['cont8'], ax=ax[1, 3])

sns.distplot(train_df['cont9'], ax=ax[2, 0])
sns.distplot(train_df['cont10'], ax=ax[2, 1])
sns.distplot(train_df['cont11'], ax=ax[2, 2])
sns.distplot(train_df['cont12'], ax=ax[2, 3])

sns.distplot(train_df['cont13'], ax=ax[3, 0])
sns.distplot(train_df['cont14'], ax=ax[3, 1])
f.delaxes(ax[3, 2])
f.delaxes(ax[3, 3])
plt.tight_layout()
plt.show()

#f, ax = plt.subplots(nrows=14, ncols=3, figsize=(12, 28))
#for i, var in enumerate(train_df.columns[train_df.columns.str.startswith('cont')]):
#    sns.distplot(train_df[var], ax=ax[i, 0])
#    sns.boxplot(train_df[var], ax=ax[i, 1])
#    stats.probplot(train_df[var], plot=ax[i, 2])
#plt.tight_layout()
#plt.show()


# From a top level point of view, we can see that **none of the features seem to be correlated to the `target`** and `cont14`. 
# 
# Additionally, there seems to be a highly correlated cluster consisting of `cont1` and `cont6` through `cont13`.

# In[7]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
ax.set_title("Correlation Matrix", fontsize=16)
sns.heatmap(train_df[train_df.columns[train_df.columns != 'id']].corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
    tick.label.set_rotation(90) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    tick.label.set_rotation(0) 
plt.show()


# # Model Baseline
# 
# Alothough, we did not do any feature engineering so far, let's create a simple LightGBM baseline to get a **benchmark** first. The baselibe in [Tawara](https://www.kaggle.com/ttahara)'s notebook [TPS Jan 2021: GBDTs Baseline](https://www.kaggle.com/ttahara/tps-jan-2021-gbdts-baseline) will be the base for our experiments.

# In[8]:


# Baseline model parameters copied from
# https://www.kaggle.com/ttahara/tps-jan-2021-gbdts-baseline

model_params = {
    "objective": "root_mean_squared_error",
    "learning_rate": 0.1, 
    "seed": 42,
    'max_depth': 7,
    'colsample_bytree': .85,
    "subsample": .85,
}
    
train_params = {
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
}

def visualize_results(y_pred, y_train, features, feature_importances):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

    color = 'blue'
    ax[0].set_ylabel('Ground Truth', color=color, fontsize=14)
    sns.distplot(y_train, ax=ax[0], color=color)
    ax[0].tick_params(axis='y', labelcolor=color)

    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel('Predicted', color=color, fontsize=14)  # we already handled the x-label with ax1
    sns.distplot(pd.Series(y_pred), ax=ax2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax[0].set_title('Distribution of Predicted Values and Ground Truth', fontsize=16)

    pd.DataFrame({'features' : features, 
                  'feature_importance': feature_importances}
                ).set_index('features').sort_values(by='feature_importance', ascending=False).head(10).plot(kind='bar', ax=ax[1])
    ax[1].set_title('Top 10 Most Important Features', fontsize=16)

    for tick in ax[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        tick.label.set_rotation(0) 

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    


# In[9]:


def run_model(X, y, X_test):
    """
    Baseline is based on
    https://www.kaggle.com/ttahara/tps-jan-2021-gbdts-baseline
    
    Arg:
    * X: training data containing features
    * y: training data containing target variables
    * X_test: test data to predict
    
    Returns:
    * predictions for X_test
    """
    # Initialize variables
    y_oof_pred = np.zeros(len(X))
    y_test_pred = np.zeros(len(X_test))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold + 1}:")

        # Prepare training and validation data
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)

        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)  

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Define model
        model = lgb.train(params=model_params,
                          train_set=train_data,
                          valid_sets=[train_data, val_data],
                          **train_params)

        # Calculate evaluation metric: Root Mean Squared Error (RMSE)
        y_val_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f"RMSE: {score:.5f}\n")

        y_oof_pred[val_idx] = y_val_pred

        # Make predictions
        y_test_pred += model.predict(X_test)

    # Calculate evaluation metric for out of fold validation set
    oof_score = np.sqrt(mean_squared_error(y, y_oof_pred))
    print(f"OOF RMSE: {oof_score: 5f}")

    # Average predictions over all folds
    y_test_pred = y_test_pred / N_SPLITS
    visualize_results(y_oof_pred, y, X.columns, model.feature_importance(importance_type="gain"))

    return y_test_pred


features_baseline = train_df.columns[train_df.columns.str.startswith('cont')]
target = ['target']

display(train_df[features_baseline].head().style.set_caption('Training data'))

y_pred = run_model(train_df[features_baseline], 
                   train_df[target], 
                   test_df[features_baseline])


# The out of fold (OOF) RMSE score is **0.703148 - this is our benchmark** for the next steps. This score is quite bad since RMSE of 0 would be ideal. If we look at the distributions of our predictions versus the ground truth, we can see that our model is doing quite poorly.

# **Can tree-based models handle bimodal distributions?**
# 
# TL;DR: Yes, tree-based models in general should be able to handle bimodal distributions.
# 
# Decision trees are insensitive to the targets distribution. Therefore, we do not necessarily need to transform the target to fit a normal distribution. Therefore, we would expect tree-based models to be able to handle bimodal distributions without any transformations as well.
# Let's see if this is true.
# 
# In the following minimal example, we have three features and a target that has a bimodal distribution as shown in the plot. 
# For simplicity reasons, the target equals `feature1` if `feature3` == 0 and else the targets equals `feature2`.

# In[10]:


df = pd.DataFrame({'feature1' : np.random.normal(10, 5, 1000),
                   'feature2' : np.random.normal(40, 10, 1000),
                   'feature3' : np.random.randint(0, 2, 1000),
                  })

df['target'] = df.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else x.feature2, axis=1)

display(df.head())
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))
color = 'blue'
sns.distplot(df.target, ax=ax, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'red'
sns.distplot(df.feature1, ax=ax2, color=color, hist=False)
sns.distplot((df.feature2), ax=ax2, color=color, hist=False)

ax2.tick_params(axis='y', labelcolor=color)
ax.set_title('Example of a Bimodal Distribution', fontsize=16)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Let's use our baseline model as is and use it for modelling the data.
# From the below results, we can see that the LightGBM model is able to handle the bimodal distribution.

# In[11]:


features = df.columns[df.columns.str.startswith('feature')]
target = ['target']

pred = run_model(df[features].head(800), 
          df[target].head(800), 
          df[features].tail(200))


# # Feature Engineering Techniques
# 
# We have seen that our baseline model in theory is able to model the bimodal distribution of our target. However, we can also see that this highly depends on the quality of our features. In the [Data Overview](#Data-Overview), we saw that the features have a low absolute correlation to the target. In this section, we will be exploring different feature engineering techniques.
# 
# ## Gaussian Mixture Modelling (GMM)
# 
# We can use Gaussian Mixture Modelling to separate the two distributions. It is an unsupervised learning algorithm.
# 
# [Discussion: When you have bimodal distribution](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/86951)

# In[12]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)

gmm.fit(train_df.target.values.reshape(-1, 1))

train_df['target_class'] = gmm.predict(train_df.target.values.reshape(-1, 1))


# In[13]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.kdeplot(data=train_df.target, ax=ax[0])
ax[0].set_title('Before GMM', fontsize=16)
sns.kdeplot(data=train_df[train_df.target_class==0].target, label='Component 1', ax=ax[1])
sns.kdeplot(data=train_df[train_df.target_class==1].target, label='Component 2', ax=ax[1])
ax[1].set_title('After GMM', fontsize=16)
plt.show()


# Since the features have multimodal distributions, it would be worthwhile checking what happens if we add a GMM feature for each `cont#` feature. 

# In[14]:


def get_gmm_class_feature(feat, n):
    gmm = GaussianMixture(n_components=n, random_state=42)

    gmm.fit(train_df[feat].values.reshape(-1, 1))

    train_df[f'{feat}_class'] = gmm.predict(train_df[feat].values.reshape(-1, 1))
    test_df[f'{feat}_class'] = gmm.predict(test_df[feat].values.reshape(-1, 1))

get_gmm_class_feature('cont1', 4)
get_gmm_class_feature('cont2', 10)
get_gmm_class_feature('cont3', 6)
get_gmm_class_feature('cont4', 4)
get_gmm_class_feature('cont5', 3)
get_gmm_class_feature('cont6', 2)
get_gmm_class_feature('cont7', 3)
get_gmm_class_feature('cont8', 4)
get_gmm_class_feature('cont9', 4)
get_gmm_class_feature('cont10', 8)
get_gmm_class_feature('cont11', 5)
get_gmm_class_feature('cont12', 4)
get_gmm_class_feature('cont13', 6)
get_gmm_class_feature('cont14', 6)

train_df.head()


# In[15]:


features = list(features_baseline) + list(train_df.columns[train_df.columns.str.contains('class') & ~train_df.columns.str.contains('target')])
target = ['target']

display(train_df[features].head().style.set_caption('Training data'))

y_test_pred = run_model(train_df[features], 
          train_df[target], 
          test_df[features])


# In[16]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
ax.set_title("Correlation Matrix", fontsize=16)
cols = ['target', 'cont1_class', 'cont2_class', 'cont3_class',
       'cont4_class', 'cont5_class', 'cont6_class', 'cont7_class',
       'cont8_class', 'cont9_class', 'cont10_class', 'cont11_class',
       'cont12_class', 'cont13_class', 'cont14_class']
sns.heatmap(train_df[cols].corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
    tick.label.set_rotation(90) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    tick.label.set_rotation(0) 
plt.show()


# With the newly added features, we get a RMSE of **0.703054**. This is only a 0.000094 improvement to our baseline and our distribution is still not close to the distribution of the ground truth.
# 
# Let's try another idea: What happens if we use the classes we got from the GMM to separate each feature:

# In[17]:


for j in range(1,15):
    for i in range(train_df.cont1_class.nunique()):
        train_df[f'cont{j}_class_{i+1}'] = np.where(train_df[f'cont{j}_class'] == (i+1), train_df[f'cont{j}'], np.nan)
        test_df[f'cont{j}_class_{i+1}'] = np.where(test_df[f'cont{j}_class'] == (i+1), test_df[f'cont{j}'], np.nan)
        


# In[18]:


train_df[['cont1', 'cont1_class', 'cont1_class_1', 'cont1_class_2', 'cont1_class_3']].head().style.set_caption('Example of newly created features for cont1 from cont1_class')


# In[19]:


features = list(features_baseline) + list(train_df.columns[train_df.columns.str.contains('class_')])
target = ['target']

display(train_df[features].head().style.set_caption('Training data'))

y_test_pred = run_model(train_df[features], 
          train_df[target], 
          test_df[features])


# ## Binning
# 
# Work in progress

# In[20]:


for i in range(1,15):
    train_df[f'cont{i}_bin_10'] = pd.cut(train_df[f'cont{i}'], bins=10, labels=False)
    test_df[f'cont{i}_bin_10'] = pd.cut(test_df[f'cont{i}'], bins=10, labels=False)


# In[21]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
sns.distplot(train_df['cont1'], ax=ax[0])
ax[0].set_title('Distribution of cont1', fontsize=14)
sns.countplot(train_df['cont1_bin_10'], ax=ax[1], color='steelblue')
ax[1].set_title('cont1 after Binning', fontsize=14)

plt.show()


# In[22]:


features = list(features_baseline) + list(train_df.columns[train_df.columns.str.contains('bin')])
target = ['target']

display(train_df[features].head().style.set_caption('Training data'))

y_test_pred = run_model(train_df[features], 
          train_df[target], 
          test_df[features])


# ## Statistical Features
# Another common approach is to create new features from the data's **statistics (mean, sum, std, etc.)**.

# In[23]:


features = train_df.columns[train_df.columns.str.startswith('cont') & ~train_df.columns.str.contains('class')& ~train_df.columns.str.contains('bin')]

train_df['sum'] = train_df[features].sum(axis=1)
train_df['mean'] = train_df[features].mean(axis=1)
train_df['min'] = train_df[features].min(axis=1)
train_df['max'] = train_df[features].max(axis=1)
train_df['std'] = train_df[features].std(axis=1)
train_df['var'] = train_df[features].var(axis=1)

test_df['sum'] = test_df[features].sum(axis=1)
test_df['mean'] = test_df[features].mean(axis=1)
test_df['min'] = test_df[features].min(axis=1)
test_df['max'] = test_df[features].max(axis=1)
test_df['std'] = test_df[features].std(axis=1)
test_df['var'] = test_df[features].var(axis=1)


# In[24]:


features = list(features_baseline) + list(['sum', 'mean', 'min', 'max', 'std'])
target = ['target']

display(train_df[features].head().style.set_caption('Training data'))

y_test_pred = run_model(train_df[features], 
          train_df[target], 
          test_df[features])


# ## Deep Feature Synthesis
# Another common approach is creating new features by combining features with **basic mathematical operations (addition, subtraction, multiplication, division)**.
# If we were given the names of each feature, we could start by creating new features based on our intuition. For example if you have two features `hours_spent_writing_kernels` and
# `number_of_kernels`, you could combine them by dividion to get a new feature `average_time_to_write_a_kernel`= `hours_spent_writing_kernels`/`number_of_kernels`.
# 
# However, in this challenge we do not have any information about the features. Therefore, we cannot create new features by intuition. An idea would be to randomly combine features and hoping to see a correlation to our target. However, for 14 features that will take a lot of time. We can somewhat automate this with Deep Feature Synthesis.
# 
# If you wanto read more about this topic in depth, I highly recommend this kernel: [Automated Feature Engineering Tutorial](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-tutorial).

# In[25]:


import featuretools as ft
es = ft.EntitySet(id = 'data')

original_cols = ['id', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']


es = es.entity_from_dataframe(entity_id = 'data', 
                              dataframe = pd.concat([train_df, test_df], axis=0)[original_cols], 
                              index = 'id', 
                              time_index = None)
es['data']


# You can list out all possibilities with the function `list_primitives()`.

# In[26]:


ft.list_primitives().head()


# Let's try a couple of primitives. How about dividing each feature by another feature using the primitive `divide_numeric`. You can see that we **end up with 196 features**.

# In[27]:


new_features, new_feature_names = ft.dfs(entityset = es, target_entity = 'data', 
                                 trans_primitives = ['divide_numeric'])

new_features.head()


# In[28]:


new_features = new_features.reset_index(drop=False)
y_test_pred = run_model(new_features[new_features.id.isin(train_df.id)],
                        train_df[target], 
                        new_features[new_features.id.isin(test_df.id)])


# Let's also try it for multiplication:

# In[29]:


new_features, new_feature_names = ft.dfs(entityset = es, target_entity = 'data', 
                                 trans_primitives = ['multiply_numeric'])

new_features.head()


# In[30]:


new_features = new_features.reset_index(drop=False)
y_test_pred = run_model(new_features[new_features.id.isin(train_df.id)],
                        train_df[target], 
                        new_features[new_features.id.isin(test_df.id)])


# ## Summary
# 
# So far, the feature engineering techniques only gave us minor improvements: 
# 
# |           | OOF RMSE | Delta to Benchmark |
# |:---------:|----------|--------------------|
# | **Benchmark** | **0.703148** | n/a                |
# | GMM (class)       | 0.703054 | 0.000094           |
# | GMM (separated)      | 0.703033 | **0.000115**           |
# | Binning   | 0.703237 | -0.000089          |
# | Statistical Features   | 0.703229 | -0.00081|
# | DFS (divide_numeric)  | 0.703145 | 0.00003|
# | DFS (multiply_numeric)  | 0.0.703144 | 0.00004|
# 
# 
# 
# In contrast to the other feature engineering techniques, with the binning features we get a worse score than with the other techniques. This is probably due to the fact that although we have more features now, a. they are highly correlated to each other (e.g. collinearity between `cont1` and `cont1_bin10`) and b. we lose some information when binning features.

# # EDA
# 
# From above section, we understand the urgency of gaining some insights from the data. For this purpose, we will concatenate the training and the testing data and sort them by the column `id`.

# In[31]:


merged_df = pd.concat([train_df, test_df], axis=0).sort_values(by='id').reset_index(drop=True)


# In[32]:


original_cols = ['id', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14', 'target']

display(merged_df[original_cols].head().style.set_caption('Merged Dataframe for EDA'))

merged_df[original_cols].describe()


# Let's see if we can get any interesting insights if we difference the features with `np.diff`. Differencing means that we take the difference between two consecutive datapoints of a column.

# In[33]:


f, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

#f.suptitle('Differenced Features')
j= -1
for i in range(1,15):
    k = (i-1)%4 # ncols
    
    if k == 0:
        j += 1
        
    sns.distplot(np.diff(merged_df[f'cont{i}']), ax=ax[j, k])
    ax[j, k].set_title(f'cont{i}', fontsize=14)
plt.tight_layout()

plt.show()


# Log transform

# In[34]:


f, ax = plt.subplots(nrows=4, ncols=4, figsize=(14, 14))

#f.suptitle('Log Features')
j= -1
for i in range(1,15):
    k = (i-1)%4 # ncols
    
    if k == 0:
        j += 1
        
    sns.distplot(np.log(merged_df[f'cont{i}']), ax=ax[j, k])
    ax[j, k].set_title(f'cont{i}', fontsize=14)
    
plt.tight_layout()
plt.show()


# From the above experiments, we can see that `cont2`, `cont3`, `cont4` and `cont13` seem to be the most important features for our model. Let's explore them a little more.
# 
# Hopefully you now have a few ideas on how to investigate the features. If you need more inspiration, I recommend the discussion section of the [BNP Paribas Cardif Claims Management Competition](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion).
# 
# Happy Kaggling!

# # Submission

# In[35]:


submission = test_df[['id']].copy()
submission['target'] = y_test_pred

submission.to_csv("submission.csv", index=False)

