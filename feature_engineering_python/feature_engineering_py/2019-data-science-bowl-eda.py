#!/usr/bin/env python
# coding: utf-8

# <h1>2019 Data Science Bowl EDA</h1>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#     -<a href='#21'>Load the packages</a>  
#     -<a href='#22'>Load the data</a>  
# - <a href='#3'>Data exploration</a>  
#     -<a href='#30'>Glimpse the data</a>  
#     -<a href='#31'>Missing data</a>  
#     -<a href='#32'>Unique values</a>  
#     -<a href='#33'>Most frequent values</a>      
#     -<a href='#34'>Values distribution</a>   
#     -<a href='#35'>Extract features from train/event_data</a>  
#     -<a href='#36'>Extract features from specs/args</a>      
#     -<a href='#37'>Merged data distribution</a>  
# - <a href='#4'>Next step</a>  
#     

# # <a id="1">Introduction</a>  
# 
# This Kernel objective is to explore the dataset for 2019 Data Science Bowl EDA.   

# # <a id="2">Prepare the data analyisis</a>  
# 
# We load the packages needed for data processing and visualization and we read the data.  

# ## <a id="21">Load the packages</a>  

# In[1]:


import numpy as np
import pandas as pd
import os
import json
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ## <a id="22">Load the data</a>  
# 
# We define a function to read all the data and report the shape of datasets.  
# 

# In[2]:


def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    print(f"train shape: {train_df.shape}")
    print(f"test shape: {test_df.shape}")
    print(f"train labels shape: {train_labels_df.shape}")
    print(f"specs shape: {specs_df.shape}")
    print(f"sample submission shape: {sample_submission_df.shape}")
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df


# In[3]:


train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()


# # <a id="3">Data exploration</a>  

# ## <a id="30">Glimpse the data</a> 
# 
# We will inspect the dataframes to check the data distribution.  
# 
# We will focus on the following data frames:  
# - train_df;  
# - test_df;  
# - train_labels_df;  
# 

# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


train_labels_df.head()


# In[7]:


pd.set_option('max_colwidth', 150)
specs_df.head()


# In[8]:


sample_submission_df.head()


# In[9]:


print(f"train installation id: {train_df.installation_id.nunique()}")
print(f"test installation id: {test_df.installation_id.nunique()}")
print(f"test & submission installation ids identical: {set(test_df.installation_id.unique()) == set(sample_submission_df.installation_id.unique())}")


# We have 17K different installation_id in train and 1K in test sets (these are similar with the ones in sample_submission).

# ## <a id="31">Missing values</a>  
# 
# We define a function to calculate the missing values and also show the type of each column.

# In[10]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[11]:


missing_data(train_df)


# In[12]:


missing_data(test_df)


# In[13]:


missing_data(train_labels_df)


# In[14]:


missing_data(specs_df)


# There are no missing data in the datasets.

# ## <a id="32">Unique values</a>  
# 
# We define a function to show unique values.

# In[15]:


def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))


# ### Train

# In[16]:


unique_values(train_df)


# ### Test

# In[17]:


unique_values(test_df)


# ### Train labels

# In[18]:


unique_values(train_labels_df)


# ### Specs

# In[19]:


unique_values(specs_df)


# ## <a id="32">Most frequent values</a>  
# 
# We define a function for most frequent values.

# In[20]:


def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))


# ### Train

# In[21]:


most_frequent_values(train_df)


# ### Test

# In[22]:


most_frequent_values(test_df)


# ### Train labels

# In[23]:


most_frequent_values(train_labels_df)


# ### Specs

# In[24]:


most_frequent_values(specs_df)


# ## <a id="34">Values distribution</a>  

# We define a function to show the number and percent of each category in the current selected feature.

# In[25]:


def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()    


# In[26]:


plot_count('title', 'title (first most frequent 20 values - train)', train_df, size=4)


# In[27]:


plot_count('title', 'title (first most frequent 20 values - test)', test_df, size=4)


# In[28]:


print(f"Title values (train): {train_df.title.nunique()}")
print(f"Title values (test): {test_df.title.nunique()}")


# In[29]:


plot_count('type', 'type - train', train_df, size=2)


# In[30]:


plot_count('type', 'type - test', test_df, size=2)


# In[31]:


plot_count('world', 'world - train', train_df, size=2)


# In[32]:


plot_count('world', 'world - test', test_df, size=2)


# In[33]:


plot_count('event_code', 'event_code - test', train_df, size=4)


# In[34]:


plot_count('event_code', 'event_code - test', test_df, size=4)


# ### Train_labels

# In[35]:


for column in train_labels_df.columns.values:
    print(f"[train_labels] Unique values of {column} : {train_labels_df[column].nunique()}")


# In[36]:


plot_count('title', 'title - train_labels', train_labels_df, size=3)


# In[37]:


plot_count('accuracy', 'accuracy - train_labels', train_labels_df, size=4)


# In[38]:


plot_count('accuracy_group', 'accuracy_group - train_labels', train_labels_df, size=2)


# In[39]:


plot_count('num_correct', 'num_correct - train_labels', train_labels_df, size=2)


# In[40]:


plot_count('num_incorrect', 'num_incorrect - train_labels', train_labels_df, size=4)


# ### Specs

# In[41]:


for column in specs_df.columns.values:
    print(f"[specs] Unique values of `{column}`: {specs_df[column].nunique()}")


# ## <a id="35">Extract features from train/event_data</a>
# 
# We will parse a subset of train_df to extract features from event_data. We only extract data from 100K random sampled rows. This should be enough to get a good sample of the content.

# In[42]:


sample_train_df = train_df.sample(100000)


# In[43]:


sample_train_df.head()


# Let's look to some of the `event_data` in this sample.

# In[44]:


sample_train_df.iloc[0].event_data


# In[45]:


sample_train_df.iloc[1].event_data


# We use **json** package to normalize the json; we will create one column for each key; the value in the column will be the value associated to the key in the json. The extracted data columns will be quite sparse.

# In[46]:


get_ipython().run_cell_magic('time', '', 'extracted_event_data = pd.io.json.json_normalize(sample_train_df.event_data.apply(json.loads))\n')


# In[47]:


print(f"Extracted data shape: {extracted_event_data.shape}")


# In[48]:


extracted_event_data.head(10)


# Let's check the statistics of the missing values in these columns.

# In[49]:


missing_data(extracted_event_data)


# We modify the `missing_data` function to order the most frequent encountered event data features (newly created function `existing_data`).

# In[50]:


def existing_data(data):
    total = data.isnull().count() - data.isnull().sum()
    percent = 100 - (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    tt = pd.DataFrame(tt.reset_index())
    return(tt.sort_values(['Total'], ascending=False))


# In[51]:


stat_event_data = existing_data(extracted_event_data)


# Let's look to the first 40 values, ordered by percent of existing data (descending).

# In[52]:


plt.figure(figsize=(10, 10))
sns.set(style='whitegrid')
ax = sns.barplot(x='Percent', y='index', data=stat_event_data.head(40), color='blue')
plt.title('Most frequent features in event data')
plt.ylabel('Features')


# In[53]:


stat_event_data[['index', 'Percent']].head(20)


# ## <a id="36">Extract features from specs/args</a>  
# 
# Let's try to extract data from `args` column in `specs_df` similarly we did for `event_data`.

# In[54]:


specs_df.args[0]


# Each row contains a list of key-values pairs (a dictionary), with the keys: `name`, `type` & `info`.
# We will parse this structure and generate new rows for each spec.

# In[55]:


specs_args_extracted = pd.DataFrame()
for i in range(0, specs_df.shape[0]): 
    for arg_item in json.loads(specs_df.args[i]) :
        new_df = pd.DataFrame({'event_id': specs_df['event_id'][i],\
                               'info':specs_df['info'][i],\
                               'args_name': arg_item['name'],\
                               'args_type': arg_item['type'],\
                               'args_info': arg_item['info']}, index=[i])
        specs_args_extracted = specs_args_extracted.append(new_df)


# In[56]:


print(f"Extracted args from specs: {specs_args_extracted.shape}")


# There is a variable number of arguments for each `event_id`.

# In[57]:


specs_args_extracted.head(5)


# Let's see the distribution of the number of arguments for each `event_id`.

# In[58]:


tmp = specs_args_extracted.groupby(['event_id'])['info'].count()
df = pd.DataFrame({'event_id':tmp.index, 'count': tmp.values})
plt.figure(figsize=(6,4))
sns.set(style='whitegrid')
ax = sns.distplot(df['count'],kde=True,hist=False, bins=40)
plt.title('Distribution of number of arguments per event_id')
plt.xlabel('Number of arguments'); plt.ylabel('Density'); plt.show()


# In[59]:


plot_count('args_name', 'args_name (first 20 most frequent values) - specs', specs_args_extracted, size=4)


# In[60]:


plot_count('args_type', 'args_type - specs', specs_args_extracted, size=3)


# In[61]:


plot_count('args_info', 'args_info (first 20 most frequent values) - specs', specs_args_extracted, size=4)


# ## <a id="37">Merged data distribution</a>  
# 
# Let's merge train and train_labels.

# ### Extract time features
# 
# We define a function to extract time features. We will apply this function for both train and test datasets.

# In[62]:


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_month_start'] = df['timestamp'].dt.is_month_start
    return df


# We apply the function to extract time features.

# In[63]:


train_df = extract_time_features(train_df)


# In[64]:


test_df = extract_time_features(test_df)


# In[65]:


train_df.head()


# In[66]:


test_df.head()


# We inspect now the date/time type data.

# In[67]:


plot_count('year', 'year - train', train_df, size=1)


# In[68]:


plot_count('month', 'month - train', train_df, size=1)


# In[69]:


plot_count('hour', 'hour -  train', train_df, size=4)


# In[70]:


plot_count('dayofweek', 'dayofweek - train', train_df, size=2)


# In[71]:


plot_count('weekofyear', 'weekofyear - train', train_df, size=2)


# In[72]:


plot_count('is_month_start', 'is_month_start - train', train_df, size=1)


# In[73]:


plot_count('year', 'year - test', test_df, size=1)


# In[74]:


plot_count('month', 'month - test', test_df, size=1)


# In[75]:


plot_count('hour', 'hour -  test', test_df, size=4)


# In[76]:


plot_count('dayofweek', 'dayofweek - test', test_df, size=2)


# In[77]:


plot_count('weekofyear', 'weekofyear - test', test_df, size=2)


# In[78]:


plot_count('is_month_start', 'is_month_start - test', test_df, size=1)


# Here we define the numerical columns and the categorical columns. We will use these to calculate the aggregated functions for the merge.

# In[79]:


numerical_columns = ['game_time', 'month', 'dayofweek', 'hour']
categorical_columns = ['type', 'world']

comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
comp_train_df.set_index('installation_id', inplace = True)


# In[80]:


def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']
    return df


# Then, we calculate the compacted form of train, by merging the aggregated numerical features from train with the dataset with unique `installation_id`.

# In[81]:


for i in numerical_columns:
    comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)


# In[82]:


print(f"comp_train shape: {comp_train_df.shape}")


# In[83]:


comp_train_df.head()


# In[84]:


# get the mode of the title
labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
# merge target
labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
# replace title with the mode
labels['title'] = labels['title'].map(labels_map)
# join train with labels
comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
print('We have {} training rows'.format(comp_train_df.shape[0]))


# In[85]:


comp_train_df.head()


# In[86]:


print(f"comp_train_df shape: {comp_train_df.shape}")
for feature in comp_train_df.columns.values[3:20]:
    print(f"{feature} unique values: {comp_train_df[feature].nunique()}")


# In[87]:


plot_count('title', 'title - compound train', comp_train_df)


# In[88]:


plot_count('accuracy_group', 'accuracy_group - compound train', comp_train_df, size=2)


# In[89]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of log(`game time mean`) values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(np.log(red_comp_train_df['game_time_mean']), kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[90]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of log(`game time std`) values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(np.log(red_comp_train_df['game_time_std']), kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[91]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `game time skew` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['game_time_skew'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[92]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `hour mean` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['hour_mean'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[93]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `hour std` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['hour_std'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[94]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `hour skew` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['hour_skew'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[95]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `month mean` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['month_mean'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[96]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `month std` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['month_std'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[97]:


plt.figure(figsize=(16,6))
_titles = comp_train_df.title.unique()
plt.title("Distribution of `month skew` values (grouped by title) in the comp train")
for _title in _titles:
    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title]
    sns.distplot(red_comp_train_df['month_skew'], kde=True, label=f'title: {_title}')
plt.legend()
plt.show()


# In[98]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of log(`game time mean`) values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(np.log(red_comp_train_df['game_time_mean']), kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[99]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of log(`game time std`) values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(np.log(red_comp_train_df['game_time_std']), kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[100]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `game time skew` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['game_time_skew'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[101]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `hour mean` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['hour_mean'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[102]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `hour std` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['hour_std'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[103]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `hour skew` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['hour_skew'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[104]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `month mean` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['month_mean'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[105]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `month std` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['month_std'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# In[106]:


plt.figure(figsize=(16,6))
_accuracy_groups = comp_train_df.accuracy_group.unique()
plt.title("Distribution of `month skew` values (grouped by accuracy group) in the comp train")
for _accuracy_group in _accuracy_groups:
    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]
    sns.distplot(red_comp_train_df['month_skew'], kde=True, label=f'accuracy group= {_accuracy_group}')
plt.legend()
plt.show()


# # <a id="4">Next step</a>  
# 
# The next step will be to use the ideas from data exploration to start extracting, selecting, engineering features and prepare models.  
# 
# 
