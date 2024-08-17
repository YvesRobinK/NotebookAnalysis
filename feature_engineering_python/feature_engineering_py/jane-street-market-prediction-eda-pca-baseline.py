#!/usr/bin/env python
# coding: utf-8

# # Jane Street Market Prediction
# ![janestreet](https://www.janestreet.com/assets/logo_horizontal.png)
# 
# ### “Buy low, sell high.” It sounds so easy….
# 
# In reality, trading for profit has always been a difficult problem to solve, even more so in today’s fast-moving and complex financial markets. Electronic trading allows for thousands of transactions to occur within a fraction of a second, resulting in nearly unlimited opportunities to potentially find and take advantage of price differences in real time.
# 
# ## See also the second part of this notebook:
# 
# ## [Jane Street Market Prediction: Baseline (Part 2)](https://www.kaggle.com/maksymshkliarevskyi/jane-street-market-prediction-baseline-part-2)

# In[1]:


get_ipython().system('pip install bioinfokit')


# In[2]:


get_ipython().system('pip install seaborn --upgrade')


# In[3]:


# basic packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample
import gc
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# for PCA
from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

import janestreet


# In[4]:


sns.__version__


# # Loading and a first look at the data

# In[5]:


train_df = pd.read_csv('../input/jane-street-market-prediction/train.csv')
features_df = pd.read_csv('../input/jane-street-market-prediction/features.csv')
example_test = pd.read_csv('../input/jane-street-market-prediction/example_test.csv')
sample_prediction_df = pd.read_csv('../input/jane-street-market-prediction/example_sample_submission.csv')

print('Train dataset shape: {}'.format(train_df.shape))
print('Features dataset shape: {}'.format(features_df.shape))
print('Example test dataset shape: {}'.format(example_test.shape))


# In[6]:


print('Head of the train data:')
train_df.head()


# # EDA

# In[7]:


print('Train dataset dtypes: \n{}'.format(train_df.dtypes.value_counts()))
print('-'*20)
print('Features dataset dtypes: \n{}'.format(features_df.dtypes.value_counts()))
print('-'*20)
print('Example test dtypes: \n{}'.format(example_test.dtypes.value_counts()))


# Almost all columns in train and test datasets are numeric. Features dataset all have bool dtype.

# In[8]:


print('Columns with NaN (Train): %d' %train_df.isna().any().sum())
print('Columns with NaN (Features): %d' %features_df.isna().any().sum())
print('Columns with NaN (Example test): %d' %example_test.isna().any().sum())


# In[9]:


NaN_train = pd.Series(train_df.isna().sum()[train_df.isna().sum() > 0].
                      sort_values(ascending = False) / len(train_df) * 100)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 10))

sns.barplot(y = NaN_train.index[:30], x = NaN_train[:30], 
            edgecolor = 'black', alpha = 0.8,
            palette = sns.color_palette("viridis", len(NaN_train[:30])))
plt.title('NaN values of train dataset (30 columns)', size = 13)
plt.xlabel('NaN values (%)')
plt.show()


# In[10]:


NaN_test = pd.Series(example_test.isna().sum()[example_test.isna().sum() > 0].
                      sort_values(ascending = False) / len(example_test) * 100)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 10))

sns.barplot(y = NaN_test.index[:30], x = NaN_test[:30], 
            edgecolor = 'black', alpha = 0.8,
            palette = sns.color_palette("viridis", len(NaN_test[:30])))
plt.title('NaN values of test dataset (30 columns)', size = 13)
plt.xlabel('NaN values (%)')
plt.show()


# Let's look at the distribution of 'date' column.

# In[11]:


plt.figure(figsize=(10, 5))
plt.title('Date', size = 15)

sns.histplot(data = train_df, x = 'date',
             edgecolor = 'black',
             palette = "viridis")
plt.xlabel('')
plt.show()


# And also 'resp' and 'weight' columns, which together represents a return on the trade.

# In[12]:


def my_plot(feat, ax = None):
    if ax != None:
        sns.histplot(data = train_df, x = feat,
                     palette = "viridis", ax = ax)
        ax.set_xlabel('')
        ax.set_title(f'{feat}')
    else:
        sns.histplot(data = train_df, x = feat,
                     palette = "viridis")
        plt.xlabel('')
        plt.title(f'{feat}')


# In[13]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
sns.set_style("whitegrid")
plt.suptitle('resp 1-4 columns', size = 15)

my_plot('resp_1', ax1)
my_plot('resp_2', ax2)
my_plot('resp_3', ax3)
my_plot('resp_4', ax4)

plt.show()


# In[14]:


plt.figure(figsize=(8, 5))

my_plot('resp')
plt.show()


# In[15]:


plt.figure(figsize=(10, 5))
plt.title('Weight', size = 15)

sns.histplot(data = train_df, x = 'weight',
             edgecolor = 'black',
             palette = "viridis", binwidth = 1)
plt.xlabel('')
plt.show()


# In[16]:


print('Rows with weight==0: \t %d' %len(train_df[train_df.weight == 0]))


# Checking "ts_id" for unique values. The result must be 'True'.

# In[17]:


train_df.ts_id.nunique() == len(train_df)


# In[18]:


sample_df = sample(list(train_df.columns[7:]), 4)

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
sns.set_style("whitegrid")
plt.suptitle('Random feature columns', size = 15)

my_plot(sample_df[0], ax[0, 0])
my_plot(sample_df[1], ax[0, 1])
my_plot(sample_df[2], ax[1, 0])
my_plot(sample_df[3], ax[1, 1])

plt.show()


# We should look at the 'features' dataset too.

# In[19]:


features_df


# Dataset represents a set of bool values. Let's check some of the most frequent features.

# In[20]:


features_tags = features_df.apply(lambda x: x[x == True].count(), axis = 1) \
    .sort_values(ascending = False).astype('str')

sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

sns.histplot(x = features_tags,
             edgecolor = 'black',
             palette = "viridis", binwidth = 1)
plt.xlabel('Tags count')
plt.show()


# # PCA

# In[21]:


pca = PCA()
pca_out = pca.fit(StandardScaler().fit_transform(train_df.iloc[:, 7:-1]
                                                 .dropna()))


# In[22]:


comp = pca_out.components_
num_pc = pca_out.n_features_
pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
comp_df = pd.DataFrame.from_dict(dict(zip(pc_list, comp)))
comp_df['variable'] = train_df.iloc[:, 7:-1].columns.values
comp_df = comp_df.set_index('variable')

comp_df.head(10).style.background_gradient(cmap = 'viridis')


# Positive and negative values in component loadings reflect the positive and negative correlation of the variables with then PCs.

# In[23]:


plt.figure(figsize=(12, 8))
plt.title('Corelation matrix of 10 first feature columns (Train dataset)', size = 15)

sns.heatmap(comp_df.iloc[:10, :10], annot = True, cmap = 'Spectral')
plt.show()


# We should keep only the PCs which explain the most variance. The eigenvalues for PCs can help to retain the number of PCs. It will be useful for future predictions.

# In[24]:


cluster.screeplot(obj = [pc_list[:20], pca_out.explained_variance_ratio_[:20]], 
                  show = True, dim = (16, 5), axlabelfontsize = 13)


# In[25]:


# PCA loadings plots
# 2D
cluster.pcaplot(x = comp[0], y = comp[1], 
                labels = range(0, 129, 1), 
                var1 = round(pca_out.explained_variance_ratio_[0]*100, 2),
                var2 = round(pca_out.explained_variance_ratio_[1]*100, 2),
                show = True, dim = (10, 8), axlabelfontsize = 13)

# 3D
cluster.pcaplot(x = comp[0], y = comp[1], z = comp[2],  
                labels = range(0, 129, 1), 
                var1 = round(pca_out.explained_variance_ratio_[0]*100, 2), 
                var2 = round(pca_out.explained_variance_ratio_[1]*100, 2), 
                var3 = round(pca_out.explained_variance_ratio_[2]*100, 2),
                show = True, dim = (14, 10), axlabelfontsize = 13)


# Also, let's look at the test data example.

# In[26]:


test_pca = PCA()
test_pca_out = test_pca.fit(StandardScaler()
                            .fit_transform(example_test.iloc[:, 2:-1]
                                           .dropna()))


# In[27]:


comp_test = test_pca_out.components_
test_num_pc = test_pca_out.n_features_
test_pc_list = ["PC" + str(i) for i in list(range(1, test_num_pc + 1))]
comp_test_df = pd.DataFrame.from_dict(dict(zip(test_pc_list, comp_test)))
comp_test_df['variable'] = example_test.iloc[:, 2:-1].columns.values
comp_test_df = comp_test_df.set_index('variable')

comp_test_df.head(10).style.background_gradient(cmap = 'viridis')


# In[28]:


plt.figure(figsize=(12, 8))
plt.title('Corelation matrix of 10 first feature columns (Test dataset)', size = 15)

sns.heatmap(comp_test_df.iloc[:10, :10], annot = True, cmap = 'Spectral')
plt.show()


# In[29]:


cluster.screeplot(obj = [test_pc_list[:20], 
                         test_pca_out.explained_variance_ratio_[:20]], 
                  show = True, dim = (16, 5), axlabelfontsize = 13)


# In[30]:


# PCA loadings plots
# 2D
cluster.pcaplot(x = comp_test[0], y = comp_test[1], 
                labels = range(0, 129, 1), 
                var1 = round(test_pca_out.explained_variance_ratio_[0]*100, 2),
                var2 = round(test_pca_out.explained_variance_ratio_[1]*100, 2),
                show = True, dim = (10, 8), axlabelfontsize = 13)

# 3D
cluster.pcaplot(x = comp_test[0], y = comp_test[1], z = comp_test[2],  
                labels = range(0, 129, 1), 
                var1 = round(test_pca_out.explained_variance_ratio_[0]*100, 2), 
                var2 = round(test_pca_out.explained_variance_ratio_[1]*100, 2), 
                var3 = round(test_pca_out.explained_variance_ratio_[2]*100, 2),
                show = True, dim = (14, 10), axlabelfontsize = 13)


# There is no significant difference between 'train' and 'example_test' datasets. Both have three PCs with importance over 10% and some number of less importance (around 5%). For future prediction, I'll use 10 PCs firstly.

# # Baseline prediction
# 
# Now, we'll make a simple prediction, without complicated data preprocessing and feature engineering. We'll use XGBClassifier as a terrific simple but strong algorithm.
# 
# For the training process, we need feature columns with not zero weight values. As a prediction target ('action') we'll use a feature that contains 'weight' and 'resp' columns.

# In[31]:


# Loading prediction work space
env = janestreet.make_env()
iter_test = env.iter_test()


# In[32]:


# Preparing the data
train_df = train_df[train_df['weight'] != 0]
train_df['action'] = ((train_df['weight'].values * train_df['resp']
                       .values) > 0).astype('int')

X_train = train_df.loc[:, train_df.columns.str.contains('feature')]
y_train = train_df.loc[:, 'action']

X_train = X_train.fillna(-999)


# In[33]:


sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

sns.histplot(x = y_train.astype('str'),
             edgecolor = 'black',
             palette = "viridis", binwidth = 1)
plt.xlabel('Action')
plt.show()


# We have balanced targets.

# In[34]:


del train_df
gc.collect()


# In[35]:


# X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, 
#                                                 train_size = 0.85, 
#                                                 random_state = 0)


# In[36]:


# params = {'n_estimators': 1000,
#           'max_depth': 12,
#           'subsample': 0.9,
#           'learning_rate': 0.05,
#           'missing': -999,
#           'random_state': 0,
#           'tree_method': 'gpu_hist'}

# model = XGBClassifier(**params)

# model.fit(X_tr, y_tr)


# #### Model evaluation

# In[37]:


# print('ROC AUC score: %.3f' 
#       %roc_auc_score(y_valid, model.predict(X_valid)))


# # The second part of notebook: 
# ## [Jane Street Market Prediction: Baseline (Part 2)](https://www.kaggle.com/maksymshkliarevskyi/jane-street-market-prediction-baseline-part-2)

# In[38]:


# for (test_df, sample_prediction_df) in iter_test:
#     X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
#     X_test.fillna(-999)
#     preds = model.predict(X_test)
#     sample_prediction_df.action = preds
#     env.predict(sample_prediction_df)


# ## Work in progress...

# In[ ]:




