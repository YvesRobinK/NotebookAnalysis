#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from colorama import Fore

from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn import metrics
from scipy import stats
import math

from tqdm.notebook import tqdm
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder

from umap import UMAP
from sklearn.manifold import TSNE


# In[3]:


# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

primary_green = px.colors.qualitative.Plotly[2]

plt.rcParams['axes.facecolor'] = primary_bgcolor


# In[4]:


colors = [primary_blue, primary_blue2, primary_blue3, primary_grey, primary_black, primary_bgcolor, primary_green]
sns.palplot(sns.color_palette(colors))


# In[5]:


plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'serif'


# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:250%; text-align:center; border-radius: 15px 50px;">Tabular Playground Series 📚 - May 2021 📈</p>
# 
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the category on an eCommerce product given various attributes about the listing. Although the features are anonymized, they have properties relating to real-world features.
# 
# This competition is a classification problem that classifies **4 classes and 50 integer features**.

# <a id='table-of-contents'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">Table of Content</p>
# 
# * [1. Data visualization: Survival Analysis 📊](#1)
#     * [1.1 Target](#1.1)
#     * [1.2 General Feature Analysis](#1.2)
#     * [1.3 Feature distribution by Class](#1.3)
#     * [1.4 Correlation Analysis](#1.4)
# * [2. Dimension Reduction](#2)
#     * [2.1 UMAP](#2.1)
#     * [2.2 t-SNE](#2.2)
# * [3. Feature Engineering](#3)
# * [4. H2O Automl](#4)
# * [5. LightAutoML](#5)

# In[6]:


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/train.csv')
train_df.columns = [column.lower() for column in train_df.columns]
# train_df = train_df.drop(columns=['passengerid'])

test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/test.csv')
test_df.columns = [column.lower() for column in test_df.columns]

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv')
submission.head()

train_df.head()


# In[7]:


feature_columns = train_df.iloc[:, 1:-1].columns.values
target_column = 'target'
feature_columns


# In[8]:


print(train_df.shape)
print(test_df.shape)


# <a id='1'></a>
# [back to top](#table-of-contents)
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">1. Data visualization: First Overview 📊</p>

# In[9]:


train_df.info()


# As we can see, there are no missing values in the dataset and all the features are integer, so we can forget about missings study!

# <a id='1.1'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">1.1 Target Variable</p>
# 
# Now we are going to take a look at the target column to see how balanced the dataset is. This is a important metric to understand if we have to resample or not.

# In[10]:


fig = px.histogram(
    train_df, 
    x=target_column, 
    color=target_column,
    color_discrete_sequence=px.colors.qualitative.G10,
)
fig.update_layout(
    title_text='Target distribution', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    paper_bgcolor=primary_bgcolor,
    plot_bgcolor=primary_bgcolor,
)
fig.update_xaxes(
    title='Target class',
    categoryorder='category ascending',
)
fig.show()


# As we can see, `Class_2` is the majority class but there is not much difference between classes.

# <a id='1.2'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">1.2 General feature analysis</p>
# 
# First of all we will take a look at the train dataset info in terms of features values distribution.

# In[11]:


train_df.drop(columns=['id']).describe().T\
        .style.bar(subset=['mean'], color=px.colors.qualitative.G10[0])\
        .background_gradient(subset=['std'], cmap='Greens')\
        .background_gradient(subset=['50%'], cmap='BuGn')


# The mean and the standard deviation have a large variability that also seem to be related. Aparently, when the mean increase, the standard deviation also does.
# 
# In the case of the median, its seems to be $0$ for most of the cases, but in 2 cases, the medían is $1$.

# <a id='1.3'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">1.3 Feature distribution by Class</p>
# 
# As there are too many variables and plotting all of them will carry us to an unleigble charts, i will plot just 3 of them to see how they behave.

# In[12]:


columns_to_plot = ['feature_9', 'feature_14', 'feature_34']

num_rows, num_cols = 3,1
f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 16), facecolor=primary_bgcolor)
f.suptitle('Distribution of Features', fontsize=20, fontweight='bold', fontfamily='serif', x=0.13)


for index, column in enumerate(train_df[columns_to_plot].columns):
    i,j = (index // num_cols, index % num_cols)
    g = sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_1', column], color=px.colors.qualitative.G10[1], shade=True, ax=axes[i])
    g = sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_2', column], color=px.colors.qualitative.G10[0], label="Skew: %.2f"%(train_df[column].skew()), shade=True, ax=axes[i])
    g = g.legend(loc="best")
    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_3', column], color=px.colors.qualitative.G10[3], shade=True, ax=axes[i])
    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_4', column], color=px.colors.qualitative.G10[2], shade=True, ax=axes[i])

# f.delaxes(axes[-1, -1])
plt.tight_layout()
plt.show()


# As we can see, the distribution of the features based on the class value seems to be the same. So we can go ahead with no problems.
# 
# It's important to mark that all the **features seems to be left skewed**!

# In[13]:


num_rows, num_cols = 10,5
f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 30))
f.suptitle('Distribution of Features', fontweight='bold', fontfamily='serif')

for index, column in enumerate(feature_columns):
    i,j = (index // num_cols, index % num_cols)

    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_1', column], color=px.colors.qualitative.G10[1], shade=True, ax=axes[i,j])
    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_2', column], color=px.colors.qualitative.G10[0], shade=True, ax=axes[i,j])
    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_3', column], color=px.colors.qualitative.G10[3], shade=True, ax=axes[i,j])
    sns.kdeplot(train_df.loc[train_df[target_column] == 'Class_4', column], color=px.colors.qualitative.G10[2], shade=True, ax=axes[i,j])

#f.delaxes(axes[3, 2])
plt.tight_layout()
plt.show()


# <a id='1.4'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">1.4 Correlation Analysis</p>
# 
# Now we are going to see how correlated the features are and how correlated are they with the target variable.

# In[14]:


corr = train_df[feature_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

fig, ax = plt.subplots(figsize=(12, 12), facecolor=primary_bgcolor)
ax.text(-1.1, -0.7, 'Correlation between the Features', fontsize=20, fontweight='bold', fontfamily='serif')
ax.text(-1.1, 0.2, 'There is no features that pass 0.02 correlation within each other', fontsize=13, fontweight='light', fontfamily='serif')


# plot heatmap
sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap='coolwarm',
            cbar_kws={"shrink": .8}, vmin=0, vmax=0.05)
# yticks
plt.yticks(rotation=0)
plt.show()


# As you can see, I have mapped the values (all are absolute values) between $0$ and $0.05$ and any value overpass the $0.02$ mark.

# <a id='2'></a>
# [back to top](#table-of-contents)
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">2. Dimension Reduction</p>

# <a id='2.1'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">2.1 UMAP</p>
# 
# Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data
# 
# 1.     The data is uniformly distributed on Riemannian manifold;
# 2.     The Riemannian metric is locally constant (or can be approximated as such);
# 3.     The manifold is locally connected.
# 
# From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.
# 
# Ref: https://umap-learn.readthedocs.io/en/latest/

# In[15]:


# Take a subsample to reduce computational cost
train_sub = train_df.sample(1000, random_state=2021)


# In[16]:


umap_2d = UMAP(n_components=2, random_state=2021)
proj_2d = umap_2d.fit_transform(train_sub[feature_columns])


# In[17]:


fig_2d = px.scatter(
    proj_2d, x=0, y=1, 
    labels={'color': 'target'},
    color=train_sub.target,
    color_discrete_sequence=px.colors.qualitative.G10,
)
fig_2d.update_layout(
    title='<span style="font-size:24px; font-family:Serif">UMAP</span>',
)

fig_2d.show()


# <a id='2.2'></a>
# [back to top](#table-of-contents)
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">2.2 t-SNE</p>
# 
# **t-distributed stochastic neighbor embedding (t-SNE)** is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. It is based on Stochastic Neighbor Embedding originally developed by Sam Roweis and Geoffrey Hinton, where Laurens van der Maaten proposed the t-distributed variant. It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
# 
# The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric, this can be changed as appropriate.
# 
# Ref: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

# In[18]:


tsne = TSNE(n_components=2, random_state=2021)
projections = tsne.fit_transform(train_sub[feature_columns])


# In[19]:


fig = px.scatter(
    projections, x=0, y=1,
    labels={'color': 'target'},
    color=train_sub.target,
    color_discrete_sequence=px.colors.qualitative.G10,
)
fig.update_layout(
    title='<span style="font-size:24px; font-family:Serif">t-SNE</span>',
)

fig.show()


# In[20]:


tsne = TSNE(n_components=3, random_state=2021)
projections = tsne.fit_transform(train_sub[feature_columns], )


# In[21]:


fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=train_sub.target, labels={'color': 'target'}
)
fig.update_traces(marker_size=8)
fig.show()


# <a id='4'></a>
# [back to top](#table-of-contents)
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">4. H2O AutoML</p>

# In[22]:


import h2o
from h2o.automl import H2OAutoML

h2o.init()


# In[23]:


get_ipython().run_cell_magic('time', '', '\ntrain_hf = h2o.H2OFrame(train_df.copy())\ntest_hf = h2o.H2OFrame(test_df.copy())\n')


# In[24]:


train_hf[target_column] = train_hf[target_column].asfactor()


# In[25]:


get_ipython().run_cell_magic('time', '', '\naml = H2OAutoML(\n    seed=2021, \n    max_runtime_secs=10 * 60,\n    nfolds = 3,\n    exclude_algos = ["DeepLearning"]\n)\n\naml.train(\n    x=list(feature_columns), \n    y=target_column, \n    training_frame=train_hf\n)\n')


# In[26]:


lb = aml.leaderboard 
lb.head(rows = lb.nrows)


# In[27]:


get_ipython().run_cell_magic('time', '', "\npreds = aml.predict(h2o.H2OFrame(test_df[feature_columns].copy()))\npreds_df = h2o.as_list(preds)\npreds_df\n\nsubmission[['Class_1', 'Class_2', 'Class_3', 'Class_4']] = preds_df[['Class_1', 'Class_2', 'Class_3', 'Class_4']]\nsubmission.to_csv('h2o_automl_300s.csv', index=False)\nsubmission.head()\n")


# <a id='5'></a>
# [back to top](#table-of-contents)
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">5. LightAutoML</p>

# In[28]:


pip install -U lightautoml


# In[29]:


# Imports from our package
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.metrics import log_loss


# In[30]:


N_THREADS = 4 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 2021 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 60 * 60 # Time in seconds for automl run


# In[31]:


le = LabelEncoder()
train_df[target_column] = le.fit_transform(train_df[target_column])


# In[32]:


get_ipython().run_cell_magic('time', '', "\ntask = Task('multiclass',)\n\nroles = {\n    'target': target_column,\n    'drop': ['id'],\n}\n\n\nautoml = TabularUtilizedAutoML(task = task, \n                               timeout = TIMEOUT,\n                               cpu_limit = N_THREADS,\n                               reader_params = {'n_jobs': N_THREADS},\n)\n\noof_pred = automl.fit_predict(train_df, roles = roles)\nprint('oof_pred:\\n{}\\nShape = {}'.format(oof_pred[:10], oof_pred.shape))\n")


# In[33]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(test_df)\nprint('Prediction for test set:\\n{}\\nShape = {}'.format(test_pred[:5], test_pred.shape))\n\nprint('Check scores...')\nprint('OOF score: {}'.format(log_loss(train_df[target_column].values, oof_pred.data)))\n")


# In[34]:


submission.iloc[:, 1:] = test_pred.data
submission.to_csv('lightautoml_v1_1hour.csv', index = False)

