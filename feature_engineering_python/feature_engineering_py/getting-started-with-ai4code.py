#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Google AI4Code Competition! #
# 
# In this competition you're challenged to reconstruct the order of Kaggle notebooks whose cells have been shuffled. Check out the [Competition Pages](https://www.kaggle.com/competitions/AI4Code/overview) for a complete overview.
# 
# This notebook will walk you through making a submission with a simple ranking model. We'll look at how to:
# - Wrangle the competition data and create validation splits,
# - Represent the code cell orders with a feature,
# - Build a ranking model with XGBoost,
# - Evaluate predictions with a Python implementation of the competition metric, and,
# - Format predictions to make a successful submission.
# 
# Our model will be able to learn roughly where a cell should go in a notebook based on what words it contains -- that, for example, cells containing "Introduction" or `import` should usually be near the beginning, while cells containing "Submit" or `submission.csv` should usually be near the end. These simple features are effective at reconstructing the global order of typical data science workflows. An understanding of the *interactions* or *relationships between cells*, however, will be required of the most successful solutions. We encourage you therefore to explore things like modern neural network language models for learning the relationships between natural language and computer code.

# # Setup #

# In[1]:


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

pd.options.display.width = 180
pd.options.display.max_colwidth = 120

data_dir = Path('../input/AI4Code')


# # Load Data #

# The notebooks are stored as individiual JSON files. They've been cleaned of the usual metadata present in Jupyter notebooks, leaving only the `cell_type` and `source`. The [Data](https://www.kaggle.com/competitions/AI4Code/data) page on the competition website has the full documentation of this dataset.
# 
# We'll load the notebooks here and join them into a dataframe for easier processing. The full set of training data takes quite a while to load, so we'll just use a subset for this demonstration.

# In[2]:


NUM_TRAIN = 10000


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


paths_train = list((data_dir / 'train').glob('*.json'))[:NUM_TRAIN]
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]
df = (
    pd.concat(notebooks_train)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

df


# Each notebook has all the code cells given first with the markdown cells following. The code cells are in the correct relative order, while the markdown cells are shuffled. In the next section, we'll see how to recover the correct orderings for notebooks in the training set.

# In[3]:


# Get an example notebook
nb_id = df.index.unique('id')[6]
print('Notebook:', nb_id)

print("The disordered notebook:")
nb = df.loc[nb_id, :]
display(nb)
print()


# Your task in this competition is to predict the correct order of the notebook cells, both code and markdown. Since you're given the relative ordering of the code cells among themselves, you could also think of this as predicting where the markdown cells should be placed among the code cells.
# 
# For example, a disordered notebook might be:
# ```
# code_1
# code_2
# code_3
# markdown_1
# markdown_2
# ```
# and the correctly ordered notebook might be:
# ```
# code_1
# markdown_2
# code_2
# code_3
# markdown_1
# ```
# The markdown cells can be in any order, but you would never see `code_2` before `code_1`, for instance.

# # Ordering the Cells #

# In the `train_orders.csv` file we have, for notebooks in the training set, the correct ordering of cells in terms of the cell ids.

# In[4]:


df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list

df_orders


# In[5]:


# Get the correct order
cell_order = df_orders.loc[nb_id]

print("The ordered notebook:")
nb.loc[cell_order, :]


# The correct numeric position of a cell we will call the **rank** of the cell. We can find the ranks of the cells within a notebook by referencing the true ordering of cell ids as given in `train_orders.csv`.

# In[6]:


def get_ranks(base, derived):
    return [base.index(d) for d in derived]

cell_ranks = get_ranks(cell_order, list(nb.index))
nb.insert(0, 'rank', cell_ranks)

nb


# Sorting a notebook by the cell ranks is another way to order the notebook.

# In[7]:


from pandas.testing import assert_frame_equal

assert_frame_equal(nb.loc[cell_order, :], nb.sort_values('rank'))


# The algorithm we'll be using for our baseline model uses the cell ranks as the target, so let's create a dataframe of the ranks for each notebook.

# In[8]:


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ranks


# # Splits #

# The `df_ancestors.csv` file identifies groups of notebooks derived from a common origin, that is, notebooks belonging to the same forking tree.

# In[9]:


df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
df_ancestors


# To prevent leakage, the test set has no notebook with an ancestor in the training set. We therefore form a validation split using `ancestor_id` as a grouping factor.

# In[10]:


from sklearn.model_selection import GroupShuffleSplit

NVALID = 0.1  # size of validation set

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

# Split, keeping notebooks with a common origin (ancestor_id) together
ids = df.index.unique('id')
ancestors = df_ancestors.loc[ids, 'ancestor_id']
ids_train, ids_valid = next(splitter.split(ids, groups=ancestors))
ids_train, ids_valid = ids[ids_train], ids[ids_valid]

df_train = df.loc[ids_train, :]
df_valid = df.loc[ids_valid, :]


# # Feature Engineering #
# 
# Let's generate [tf-idf features](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) to use with our ranking model. These features will help our model learn what kinds of words tend to occur most often at various positions within a notebook.

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Training set
tfidf = TfidfVectorizer(min_df=0.01)
X_train = tfidf.fit_transform(df_train['source'].astype(str))
# Rank of each cell within the notebook
y_train = df_ranks.loc[ids_train].to_numpy()
# Number of cells in each notebook
groups = df_ranks.loc[ids_train].groupby('id').size().to_numpy()


# Now let's add the code cell ordering as a feature. We'll append a column that enumerates the code cells in the correct order, like `1, 2, 3, 4, ...`, while having the dummy value `0` for all markdown cells. This feature will help the model learn to put the code cells in the correct order.

# In[12]:


# Add code cell ordering
X_train = sparse.hstack((
    X_train,
    np.where(
        df_train['cell_type'] == 'code',
        df_train.groupby(['id', 'cell_type']).cumcount().to_numpy() + 1,
        0,
    ).reshape(-1, 1)
))
print(X_train.shape)


# # Train #

# We'll use the ranking algorithm provided by XGBoost.

# In[13]:


from xgboost import XGBRanker

model = XGBRanker(
    min_child_weight=10,
    subsample=0.5,
    tree_method='hist',
)
model.fit(X_train, y_train, group=groups)


# # Evaluate #

# Now let's see how well our model learned to order Kaggle notebook cells. We'll evaluate predictions on the validation set with a variant of the Kendall tau correlation.

# ## Validation set ##

# First we'll create features for the validation set just like we did for the training set.

# In[14]:


# Validation set
X_valid = tfidf.transform(df_valid['source'].astype(str))
# The metric uses cell ids
y_valid = df_orders.loc[ids_valid]

X_valid = sparse.hstack((
    X_valid,
    np.where(
        df_valid['cell_type'] == 'code',
        df_valid.groupby(['id', 'cell_type']).cumcount().to_numpy() + 1,
        0,
    ).reshape(-1, 1)
))


# Here we'll use the model to predict the rank of each cell within its notebook and then convert these ranks into a list of ordered cell ids.

# In[15]:


y_pred = pd.DataFrame({'rank': model.predict(X_valid)}, index=df_valid.index)
y_pred = (
    y_pred
    .sort_values(['id', 'rank'])  # Sort the cells in each notebook by their rank.
                                  # The cell_ids are now in the order the model predicted.
    .reset_index('cell_id')  # Convert the cell_id index into a column.
    .groupby('id')['cell_id'].apply(list)  # Group the cell_ids for each notebook into a list.
)
y_pred.head(10)


# Now let's examine a notebook to see how the model did.

# In[16]:


nb_id = df_valid.index.get_level_values('id').unique()[8]

display(df.loc[nb_id])
display(df.loc[nb_id].loc[y_pred.loc[nb_id]])


# ## Metric ##
# 
# This competition uses a variant of the [Kendall tau correlation](https://www.kaggle.com/competitions/AI4Code/overview/evaluation), which will measure how close to the correct order our predicted orderings are. See this notebook for more on this metric: [Competition Metric - Kendall Tau Correlation](https://www.kaggle.com/code/ryanholbrook/competition-metric-kendall-tau-correlation/notebook).

# In[17]:


from bisect import bisect


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


# Let's test the metric with a dummy submission created from the ids of the shuffled notebooks.

# In[18]:


y_dummy = df_valid.reset_index('cell_id').groupby('id')['cell_id'].apply(list)
kendall_tau(y_valid, y_dummy)


# Comparing this to the score on the predictions, we can see that our model was indeed able to improve the cell ordering somewhat.

# In[19]:


kendall_tau(y_valid, y_pred)


# # Submission #

# To create a submission for this competition, we'll apply our model to the notebooks in the test set. Note that this is a **Code Competition**, which means that the test data we see here is only a small sample. When we submit our notebook for scoring, this example data will be replaced with the full test set of about 20,000 notebooks.

# First we load the data.

# In[20]:


paths_test = list((data_dir / 'test').glob('*.json'))
notebooks_test = [
    read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
]
df_test = (
    pd.concat(notebooks_test)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)


# Then create the tf-idf and code cell features.

# In[21]:


X_test = tfidf.transform(df_test['source'].astype(str))
X_test = sparse.hstack((
    X_test,
    np.where(
        df_test['cell_type'] == 'code',
        df_test.groupby(['id', 'cell_type']).cumcount().to_numpy() + 1,
        0,
    ).reshape(-1, 1)
))


# And then create predictions on the test set.

# In[22]:


y_infer = pd.DataFrame({'rank': model.predict(X_test)}, index=df_test.index)
y_infer = y_infer.sort_values(['id', 'rank']).reset_index('cell_id').groupby('id')['cell_id'].apply(list)
y_infer


# The `sample_submission.csv` file shows what a correctly formatted submission must look like. We'll just use it as a visual check, but you might like to directly modify the values of sample submission instead. (This would help prevent failed submissions due to missing notebook ids or incorrectly named columns, for instance.)

# In[23]:


y_sample = pd.read_csv(data_dir / 'sample_submission.csv', index_col='id', squeeze=True)
y_sample


# We can see that a correctly formatted submission needs the index named `id` and the column of cell orders named `cell_order`. Moreover, we need to convert the list of cell ids into a space-delimited string of cell ids.

# In[24]:


y_submit = (
    y_infer
    .apply(' '.join)  # list of ids -> string of ids
    .rename_axis('id')
    .rename('cell_order')
)
y_submit


# And finally we'll write out the formatted submissions to a file `submission.csv`. When we submit our notebook, it will be rerun on the full test data to create the submission file that's actually scored.

# In[25]:


y_submit.to_csv('submission.csv')

