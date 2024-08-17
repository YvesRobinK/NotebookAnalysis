#!/usr/bin/env python
# coding: utf-8

# Hello Fellow Kagglers,
# 
# This notebook demonstrates the feature engineering, training and inference process, all in one notebook!
# 
# Training takes ~2 hours and inference ~6, expect the submission to take roughly 8 hours.
# 
# A binary approach is used, as in many other notebooks, where, for a pair of points, a match confidence is predicted.
# 
# This notebook uses dataset generated in the follow notebooks:
# 
# 1) [Foursquare 16M Train Pairs Generation](https://www.kaggle.com/code/markwijkhuizen/foursquare-16m-train-pairs-generation)
# 2) [Foursquare USE/MPNET Name Embeddings](https://www.kaggle.com/code/markwijkhuizen/foursquare-use-mpnet-name-embeddings)
# 
# Feel free to give tips and ask questions!

# In[1]:


# Install Reverse Geocode Package to deduce Country/City from coordinates
get_ipython().system('pip install /kaggle/input/reversegeocode/reverse_geocode-1.4.1-py3-none-any.whl')


# In[2]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K
from Levenshtein import distance as lev
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn import metrics
from multiprocessing import cpu_count
from sklearn.neighbors import BallTree
from difflib import SequenceMatcher

import geopy.distance
import reverse_geocode
import math
import scipy
import numba
import warnings
import Levenshtein
import itertools
import gc
import psutil
import sys
import pickle

# Pandas Apply With Progress Bar
tqdm.pandas()

# Plot DPI
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

# Tensorflow Version
print(f'Tensorflow version {tf.__version__}')

# Ignore Warnings
warnings.filterwarnings('ignore')


# In[3]:


# Global Seed
SEED = 42
# Earth Radius in KM to compute Haversine Distance
EARTH_RADIUS = 6371


# # Load Train/Test Data

# In[4]:


usecols = [
    'id',
    'name',
    'latitude',
    'longitude',
    'address',
    'city',
    'state',
    'zip',
    'country',
    'url',
    'phone',
    'categories',
]


# In[5]:


get_ipython().run_cell_magic('time', '', "# Train\ntrain_dtype = {\n    'id': 'category',\n    'name': 'category',\n    'address': 'category',\n    'city': 'category',\n    'state': 'category',\n    'zip': 'category',\n    'country': 'category',\n    'url': 'category',\n    'phone': 'category',\n    'categories': 'category',\n    'latitude': np.float32,\n    'longitude': np.float32,\n}\ntrain = pd.read_csv('/kaggle/input/foursquare-location-matching/train.csv', dtype=train_dtype, usecols=usecols)\ntrain['id'] = train.index.values\ndisplay(train.info(memory_usage=True))\ndisplay(train.head())\ndisplay(train.memory_usage(deep=True) / len(train))\n\ntest = pd.read_csv('/kaggle/input/foursquare-location-matching/test.csv', dtype=train_dtype, usecols=usecols)\ndisplay(test.info())\ndisplay(test.head())\n")


# # Load Pairs

# In[6]:


# Pairs Datype
pairs_dtype = {
    'id_1': 'category',
    'id_2': 'category',
    'name_1': 'category',
    'name_2': 'category',
    'address_1': 'category',
    'address_1': 'category',
    'city_1': 'category',
    'city_2': 'category',
    'state_1': 'category',
    'state_2': 'category',
    'zip_1': 'category',
    'zip_2': 'category',
    'country_1': 'category',
    'country_2': 'category',
    'url_1': 'category',
    'url_2': 'category',
    'phone_1': 'category',
    'phone_2': 'category',
    'categories_1': 'category',
    'categories_2': 'category',
    'latitude_1': np.float32,
    'longitude_1': np.float32,
    'latitude_2': np.float32,
    'longitude_2': np.float32,
}
pd.options.display.max_rows = 99
pd.options.display.max_columns = 99

pairs_sample = pd.read_csv('/kaggle/input/foursquare-location-matching/pairs.csv', dtype=pairs_dtype, skiprows=lambda idx: idx > 5)
display(pairs_sample.info())
display(pairs_sample.head())


# # Load Sample Submission

# In[7]:


# Sample Submission
sample_submission = pd.read_csv('/kaggle/input/foursquare-location-matching/sample_submission.csv')
display(sample_submission.info())
display(sample_submission.head())


# # To Lower

# In[8]:


# Convert String columns to lower case to make features case agnostic
to_lower_columns = [
    'name',
    'state',
    'country',
    'city',
    'address',
    'zip',
    'phone',
    'url',
    'categories',
]

def to_lower(df):
    f = lambda v: '' if v == 'NaN' else v.lower()
    for col in to_lower_columns:
        if f'{col}_1' in df and f'{col}_2' in df:
            df[f'{col}_1'] = df[f'{col}_1'].astype(str, copy=False).str.lower().replace('nan', '').astype('category')
            df[f'{col}_2'] = df[f'{col}_2'].astype(str, copy=False).str.lower().replace('nan', '').astype('category')
        else:
            df[col] = df[col].astype(str, copy=False).str.lower().replace('nan', '').astype('category')
            
to_lower(train)
to_lower(pairs_sample)
to_lower(test)


# In[9]:


display(train.head())


# In[10]:


display(pairs_sample.head())


# In[11]:


display(test.head())


# In[12]:


# Load Pairs generated in other notebook
pairs = pd.read_pickle('/kaggle/input/foursquare-16m-train-pairs-generation-dataset/pairs.pkl')

# Display Memory Usage
display(pairs.memory_usage(deep=True) / len(pairs))

# Display Pairs Data
display(pairs.head(25))
display(pairs.info())

# Display Positive/Negative Sample Ratio's
display(pairs['match'].value_counts(normalize=True).to_frame())

# Unique Names
display(pairs[['name_1', 'name_2']].nunique())


# # Train Explorative Data Analysis

# In[13]:


# Latitude
train[['latitude', 'longitude']].plot(kind='hist', bins=32, alpha=0.50)
plt.title('Latitude and Longitude Distribution', size=24)
plt.show()


# In[14]:


# Most common names are fast food restaurants
print(f'Unique Names in Train: {train["name"].nunique()}\n')
print('===== Top 10 Most Occuring Names =====')
display(train['name'].value_counts(dropna=False, normalize=True).head(10))


# In[15]:


# Cities are really all over the world and, most importantly, sometimes spelled in their native language
# санкт-петербург: Saint Petersburg
print(f'Unique Cities in Train: {train["city"].nunique()}\n')
print('===== Top 10 Most Occuring Cities =====')
display(train['city'].value_counts(dropna=False).head(10))


# In[16]:


# 420K states are empty
print(f'Unique States in Train: {train["state"].nunique()}\n')
print('===== Top 10 Most Occuring States =====')
display(train['state'].value_counts(dropna=False).head(10))


# In[17]:


# Over half of the zip codes are missing
print(f'Unique Zip Codes in Train: {train["zip"].nunique()}\n')
print('===== Top 10 Most Occuring Zip Codes =====')
display(train['zip'].value_counts(dropna=False).head(10))


# In[18]:


# Close to a quarter of the point are located in the US
print(f'Unique States in Countries: {train["country"].nunique()}\n')
print('===== Top 10 Most Occuring Countries =====')
display(train['country'].value_counts(dropna=False).head(10))


# In[19]:


# URLS are rare, over 75% is missing
print(f'Unique URLs in Train: {train["url"].nunique()}\n')
print('===== Top 10 Most Occuring URLs =====')
display(train['url'].value_counts(dropna=False).head(10))


# In[20]:


# Phone is also missing in ~75% of the cases, not a good feature
print(f'Unique Phone Numbers in Train: {train["phone"].nunique()}\n')
print('===== Top 10 Most Occuring Phone Numbers =====')
display(train['phone'].value_counts(dropna=False).head(10))


# In[21]:


# Categories are varying from cafes and hotel to offices and banks
print(f'Unique Categories in Train: {train["categories"].nunique()}\n')
print('===== Top 10 Most Occuring Categories =====')
display(train['categories'].value_counts(dropna=False).head(10))


# # Haversine Distance

# In[22]:


# Numba optimized haversine distance
@numba.jit(nopython=True)
def haversine_np(args):
    lon1, lat1, lon2, lat2 = args
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = EARTH_RADIUS * c
    return km


# # Country Code
# 
# [Reverse Geocode](https://pypi.org/project/reverse-geocode/) package is used to deduce missing country and city based on coordinates

# In[23]:


def get_country_codes(coords):
    data = reverse_geocode.search(coords)
    return [v['country_code'] for v in data]

train['country_code'] = get_country_codes(train[['latitude', 'longitude']])

pairs['country_code_1'] = get_country_codes(pairs[['latitude_1', 'longitude_1']])
pairs['country_code_2'] = get_country_codes(pairs[['latitude_2', 'longitude_2']])

print(f'Unique Country Code 1 in Pairs: {pairs["country_code_1"].nunique()}\n')
print('===== Top 10 Most Occuring Country Codes 1 =====')
display(pairs['country_code_1'].value_counts(dropna=False).head(10))


# # City

# In[24]:


def get_cities(coords):
    data = reverse_geocode.search(coords)
    return [v['city'] for v in data]

train['city_rg'] = get_cities(train[['latitude', 'longitude']])
    
pairs['city_rg_1'] = get_cities(pairs[['latitude_1', 'longitude_1']])
pairs['city_rg_2'] = get_cities(pairs[['latitude_2', 'longitude_2']])

print(f'Unique City Reverse Geocode 1 in Pairs: {pairs["city_rg_1"].nunique()}\n')
print('===== Top 10 Most Occuring City Reverse Geocode 1 =====')
display(pairs['city_rg_1'].value_counts(dropna=False).head(10))


# # Make Ordinal Encoding
# 
# To prevent overfitting on categorical columns the ordinal encoding takes the N most common categories and puts all other categories in an "others" category. Otherwise there would be many categories with just a handfull of samples, which almost guarantees overfitting.

# In[25]:


columns_ordinal = [
    'name',
    'state',
    'country',
    'country_code',
    'city',
    'city_rg',
    'address',
    'zip',
    'url',
]

cat2ord_dict_dicts = dict()
TOP_K = {
    'name': 2048,
    'state': 2048,
    'country': 1024,
    'country_code': 1024,
    'city': 2048,
    'city_rg': 1024,
    'address': 1024,
    'zip': 1024,
    'phone': 1024,
    'url': 1024,
}
for col in tqdm(columns_ordinal):
    n_categories = train[col].nunique()
    cat_codes = train[col].astype('category').cat.codes
    if n_categories < TOP_K[col]:
        train[f'{col}_ordinal'] = cat_codes
    else:
        # Ordinal Encoding + 1 for "Others" category
        train[f'{col}_ordinal'] = cat_codes + 1
    
        # Category Population Count
        train[f'{col}_count'] = train[col].apply(train[col].value_counts().get).astype(np.float32)
        # Set all categories with less than top 1000 population count to 0 "Others" category
        top_k_count = train[col].value_counts()[TOP_K[col] - 1 if cat_codes.max() >= TOP_K[col] else -1]
        train.loc[train[f'{col}_count'] <= top_k_count, f'{col}_ordinal'] = 0
        train[f'{col}_ordinal'] = train[f'{col}_ordinal'].astype('category').cat.codes
    
    # Make Cateogry to Ordinal Dictionary
    cat2ord_dict_dicts[col] = train[[col, f'{col}_ordinal']].set_index(col).squeeze().to_dict()
    n_unique = train[f'{col}_ordinal'].nunique()
    cat2ord_dict_dicts[f'{col}_count'] = n_unique
    print(f'{col} has {n_unique} categories, max: {train[col].nunique()}')


# In[26]:


# Adds ordinal features
def add_ordinal_features(df):
    for col in tqdm(columns_ordinal):
        df[f'{col}_1_ordinal'] = df[f'{col}_1'].apply(cat2ord_dict_dicts[col].get).astype(np.float32, copy=False).fillna(-1).astype(np.int16, copy=False) + 1
        df[f'{col}_2_ordinal'] = df[f'{col}_2'].apply(cat2ord_dict_dicts[col].get).astype(np.float32, copy=False).fillna(-1).astype(np.int16, copy=False) + 1


# # Split Categories And Ordinal Encode

# In[27]:


# Set with all unique categories
CATEGORIES = set()

for cats in tqdm(train['categories'].str.split(', ')):
    if type(cats) is list:
        for c in cats:
                CATEGORIES.add(c)

CATEGORIES_LIST_VALID = np.array(list(CATEGORIES))
# Add the empty category, as some categories are missing
CATEGORIES_LIST = np.sort(['AAA_empty'] + list(CATEGORIES)).tolist()
CATEGORIES = pd.Series(index=CATEGORIES_LIST, data=np.arange(len(CATEGORIES_LIST)))
# Number of Categories, "+1" for NaN
N_CATEGORIES = CATEGORIES.size + 1
N_CATEGORIES_VALID = CATEGORIES_LIST_VALID.size
CAT2ORD_DICT = CATEGORIES.to_dict()
print(f'N_CATEGORIES: {N_CATEGORIES}, N_CATEGORIES_VALID: {N_CATEGORIES_VALID}')


# In[28]:


def add_categories_features(df):
    def get_categories_ordinal(categories, idx):
        # For missing categories, return 0
        if type(categories) is not str:
            return 0
        else:
            # Split categories on comma
            l = np.sort(categories.split(', '))
            # If index of category is larger than number of categories, return 0
            if idx >= len(l):
                return 0
            else:
                # Check if category is in categories dictionary, to prevent keyerror for new categories
                if l[idx] in CAT2ORD_DICT:
                    return CAT2ORD_DICT.get(l[idx]) + 1
                # if category is unknown, return 0
                else:
                    return 0

    # Ordinal encode first 3 categories
    for i in tqdm(range(3)):
        df[f'categories{i}_1_ordinal'] = df[f'categories_1'].apply(get_categories_ordinal, idx=i).astype(np.int16)
        df[f'categories{i}_2_ordinal'] = df[f'categories_2'].apply(get_categories_ordinal, idx=i).astype(np.int16)


# In[29]:


# Stand alone features are computed for a single point and not as relation between pairs
def add_stand_alone_features(df):
    print('===== Ordinal Features =====')
    add_ordinal_features(df)
    add_categories_features(df)
    
# Add all features to the pairs DataFrame
add_stand_alone_features(pairs)


# In[30]:


# Show NaN ratio's for categories
for i in range(3):
    nan_ratio_1 = np.mean(pairs[f'categories{i}_1_ordinal'] == 0) * 100
    nan_ratio_2 = np.mean(pairs[f'categories{i}_2_ordinal'] == 0) * 100
    print(f'{i} | NaN ratio 1: {nan_ratio_1:.1f}%, NaN ratio 2: {nan_ratio_2:.1f}%')


# # Category Embedding

# In[31]:


# Universal Sentence Encoder for English words used to embed categories
def get_categories_embedding():
    embed = hub.load('/kaggle/input/universalsentenceencoderlarge/universal-sentence-encoder-large_5')
    
    EMBEDDING_SIZE = 512
    CATEGORIES_EMBEDDING = np.zeros(shape=[N_CATEGORIES_VALID, EMBEDDING_SIZE], dtype=np.float32)

    for cat_idx, cat in enumerate(tqdm(CATEGORIES_LIST_VALID)):
        CATEGORIES_EMBEDDING[cat_idx] = embed([cat])
        
    return CATEGORIES_EMBEDDING


# In[32]:


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# In[33]:


# Embeddings of Categories
CATEGORIES_EMBEDDING = get_categories_embedding()

# Create Cosine Similarity Matrix
EMBEDDING_DISTANCE = np.full(shape=[N_CATEGORIES, N_CATEGORIES], fill_value=np.nan, dtype=np.float32)
for idx_a, cat_emb_a in enumerate(tqdm(CATEGORIES_EMBEDDING)):
    for idx_b, cat_emb_b in enumerate(CATEGORIES_EMBEDDING):
        EMBEDDING_DISTANCE[idx_a + 2, idx_b + 2] = cosine_similarity(cat_emb_a, cat_emb_b)
        
# Always Save Embedding Distances for Next Time
print(f'EMBEDDING_DISTANCE shape: {EMBEDDING_DISTANCE.shape}, EMBEDDING_DISTANCE dtype: {EMBEDDING_DISTANCE.dtype}')
np.save('EMBEDDING_DISTANCE.npy', EMBEDDING_DISTANCE)


# In[34]:


# Sanity check for embedding distances
display(pd.Series(EMBEDDING_DISTANCE.flatten()).describe().to_frame(name='Value'))


# In[35]:


# Adds distance between a category and the most similar category of the other point
def add_categories_distance_features(df):
    for i in range(3):
        df[f'categories{i}_distance'] = np.nanmax(
            np.stack((
                EMBEDDING_DISTANCE[df[f'categories{i}_1_ordinal'].astype(np.int32), df[f'categories0_2_ordinal'].astype(np.int32)],
                EMBEDDING_DISTANCE[df[f'categories{i}_1_ordinal'].astype(np.int32), df[f'categories1_2_ordinal'].astype(np.int32)],
                EMBEDDING_DISTANCE[df[f'categories{i}_1_ordinal'].astype(np.int32), df[f'categories2_2_ordinal'].astype(np.int32)],
            ))
        , axis=0)


# # Name Embedding
# 
# Embeddings are used from the [Foursquare USE/MPNET Name Embeddings](https://www.kaggle.com/code/markwijkhuizen/foursquare-use-mpnet-name-embeddings/notebook) notebook

# In[36]:


NAMES_EMBEDDINGS_USE = np.load('/kaggle/input/foursquare-usempnet-name-embeddings-dataset/NAMES_EMBEDDINGS_USE.npy')
NAMES_EMBEDDINGS_MPNET = np.load('/kaggle/input/foursquare-usempnet-name-embeddings-dataset/NAMES_EMBEDDINGS_MPNET.npy')

print(f'NAMES_EMBEDDINGS_USE shape: {NAMES_EMBEDDINGS_USE.shape}, dtype: {NAMES_EMBEDDINGS_USE.dtype}')
print(f'NAMES_EMBEDDINGS_MPNET shape: {NAMES_EMBEDDINGS_MPNET.shape}, dtype: {NAMES_EMBEDDINGS_MPNET.dtype}')


# In[37]:


with open('/kaggle/input/foursquare-usempnet-name-embeddings-dataset/name2names_embedding_idx_dict.pkl', 'rb') as f:
    name2names_embedding_idx_dict = pickle.load(f)


# In[38]:


# Computes the cosine similarity as distance between names
def add_name_distance_features(df):
    idxs_1 = df['name_1'].apply(name2names_embedding_idx_dict.get, args=(-1,)).astype(np.int32)
    idxs_2 = df['name_2'].apply(name2names_embedding_idx_dict.get, args=(-1,)).astype(np.int32)
    # Universal Sentence Encoder
    df['name_distance_use'] = np.array([
            np.nan if a < 0 or b < 0 else cosine_similarity(NAMES_EMBEDDINGS_USE[a], NAMES_EMBEDDINGS_USE[b])
                for a, b in zip(idxs_1, idxs_2)
        ], dtype=np.float32)
    
    # MPNET
    df['name_distance_mpnet'] = np.array([
            np.nan if a < 0 or b < 0 else cosine_similarity(NAMES_EMBEDDINGS_MPNET[a], NAMES_EMBEDDINGS_MPNET[b])
                for a, b in zip(idxs_1, idxs_2)
        ], dtype=np.float32)


# # Haversine Distance

# In[39]:


# Adds haversine distance between two points
def add_haversine_distance(df):
    df['haversine_distance'] = np.apply_along_axis(
            haversine_np, 1,
            df[['longitude_1', 'latitude_1', 'longitude_2', 'latitude_2']].values.astype(np.float32)
        ).astype(np.float32)


# # Levenstein Distance

# In[40]:


levenstein_columns = [
    'name',
    'address',
    'categories',
]

# Adds the levenstein distance for the given columns
def add_levenstein_distance(df):
    def get_levenstein_distance(args):
        a, b = args
        if type(a) != str or type(b) != str:
            return np.nan
        else:
            return lev(*args)
        
    for col in levenstein_columns:
        df[f'{col}_ls_distance'] = df[[f'{col}_1', f'{col}_2']].apply(get_levenstein_distance, axis=1, raw=True).astype(np.float32)


# # Equality

# In[41]:


# Checks whether a category occurs in the other categories
def add_equal_features(df):  
    # Category in other category
    for i in range(2):
        df[f'categories{i}_1_isin_2_ordinal'] = (
            (df[f'categories{i}_1_ordinal'] > 0) &
            (
                (df[f'categories{i}_1_ordinal'] == df['categories0_2_ordinal']) |
                (df[f'categories{i}_1_ordinal'] == df['categories1_2_ordinal']) |
                (df[f'categories{i}_1_ordinal'] == df['categories2_2_ordinal'])
            )
        )


# # Longest Substring

# In[42]:


longest_substring_columns = [
    'name',
    'address',
    'categories',
]


# In[43]:


# source: https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
@numba.jit(nopython=True, nogil=True, cache=True)
def longestSubstringFinder(string1: str, string2: str):
    answer = 0
    len1, len2 = len(string1), len(string2)
    
    for i in range(len1):
        for j in range(len2):
            lcs_temp = 0
            match = 0
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and string1[i+lcs_temp] == string2[j+lcs_temp]):
                match += 1
                lcs_temp += 1
            if match > answer:
                answer = match
    return np.uint8(answer)


# In[44]:


# Longest substring feature
def add_longest_substr(df):
    for col in longest_substring_columns:
        df[f'{col}_longest_substr'] = df[[f'{col}_1', f'{col}_2']].apply(lambda args: longestSubstringFinder(*args), axis=1, raw=True).astype(np.uint8)
        df[f'{col}_longest_substr_ratio'] = (
                (df[f'{col}_longest_substr'] * 2) / (df[f'{col}_1'].apply(len) + df[f'{col}_2'].apply(len)) 
            ).astype(np.float32)


# In[45]:


# Drop some columns to reduce memory usage
pairs.drop([
    'city_1',
    'state_1',
    'zip_1',
    'country_1',
    'url_1',
    'phone_1',
    'city_2',
    'state_2',
    'zip_2',
    'country_2',
    'url_2',
    'phone_2',
    'country_code_1',
    'country_code_2',
    'city_rg_1',
    'city_rg_2',
], axis=1, inplace=True)

# Garbage Collect
print(gc.collect())


# # Add Combined Features
# 
# This will take some time, for the complete 16M pairs training set about one and a half hour

# In[46]:


get_ipython().run_cell_magic('time', '', 'def add_combined_features(df):\n    add_categories_distance_features(df)\n    add_haversine_distance(df)\n    add_equal_features(df)\n    add_levenstein_distance(df)\n    add_longest_substr(df)\n    add_name_distance_features(df)\n    \nadd_combined_features(pairs)\ndisplay(pairs.head(30))\ndisplay(pairs.info())\n')


# # LightGBM Dataset

# In[47]:


features_1 = [
    'latitude_1',
    'longitude_1',
    'name_1_ordinal',
    'state_1_ordinal',
    'url_1_ordinal',
    'zip_1_ordinal',
    'country_1_ordinal',
    'country_code_1_ordinal',
    'city_rg_1_ordinal',
    'city_1_ordinal',
    'categories0_1_ordinal',
    'categories1_1_ordinal',
    'categories2_1_ordinal',
]

features_2 = [
    'latitude_2',
    'longitude_2',
    'name_2_ordinal',
    'state_2_ordinal',
    'url_2_ordinal',
    'zip_2_ordinal',
    'country_2_ordinal',
    'country_code_2_ordinal',
    'city_rg_2_ordinal',
    'city_2_ordinal',
    'categories0_2_ordinal',
    'categories1_2_ordinal',
    'categories2_2_ordinal',
]

features_combined = [
    'categories0_distance',
    'categories1_distance',

    
    'name_distance_use',
    'name_distance_mpnet',
    
    'haversine_distance',
    
    'name_ls_distance',
    'address_ls_distance',
    'categories_ls_distance',
    
    'categories0_1_isin_2_ordinal',
    'categories1_1_isin_2_ordinal',
    
    'name_longest_substr',  
    'address_longest_substr',  
    'categories_longest_substr',
    
    'name_longest_substr_ratio',
    'address_longest_substr_ratio',
    'categories_longest_substr_ratio',
]

features = features_1 + features_2 + features_combined

categorical_features_idxs = []
for idx, f in enumerate(features):
    if f.endswith('_ordinal'):
        categorical_features_idxs.append(idx)

target = 'match'

print(f'categorical_features_idxs: {categorical_features_idxs}')


# In[48]:


# Placeholder Matrix for Pairs Features
pairs_features = np.empty(shape=[len(pairs), len(features)], dtype=np.float32)

# Fill up the pairs_features matrix column by column
for f_idx, f in enumerate(features):
    pairs_features[:, f_idx] = pairs[f]

pairs_target = pairs[target].values.astype(np.int8)

# Save Pairs Features and Target
np.save('pairs_features.npy', pairs_features)
np.save('pairs_target.npy', pairs_target)

print(f'pairs_features shape: {pairs_features.shape}, pairs_target shape: {pairs_target.shape}')


# In[49]:


# Train/Validation split, use just 5% for validation as that will already create a validation set larger than the test set
train_idxs, val_idxs = train_test_split(np.arange(len(pairs_target), dtype=np.int32), test_size=0.05, random_state=SEED)
print(f'train_idxs size: {train_idxs.size}, val_idxs size: {val_idxs.size}')


# # Clean Up

# In[50]:


del train, pairs
gc.collect()

ram_usage = psutil.virtual_memory()
print(f'RAM memory % used: {ram_usage[2]:.1f}, ({ram_usage[3] / 2**30:.1f}GB)')


# # Make LightGBM Dataset

# In[51]:


# LightGBM Training Dataset
train_data = lgb.Dataset(
    data = pairs_features[train_idxs],
    label = pairs_target[train_idxs],
    categorical_feature = None,
)

# LightGBM Validation Dataset
val_data_pairs = lgb.Dataset(
    data = pairs_features[val_idxs],
    label = pairs_target[val_idxs],
    categorical_feature = None,
)


# # LightGBM Model

# In[52]:


NUM_BOOST_ROUND = 1000
METRICS = ['binary_logloss', 'binary_error']

# Simple LightGBM parameters
lgbm_params = {
    'objective': 'binary',
    'metric': ','.join(METRICS),
    # Much more than other notebooks, possible due to 16M Training Samples!
    'num_leaves': 256,
    'learning_rate': 0.10,
    'deterministic': True,
    'seed': SEED,
}


# In[53]:


# This is all it takes to train a LightGBM Model!
def train_f():
    evals_result = {}
    model_lgb = lgb.train(
        params = lgbm_params,
        train_set = train_data,
        valid_sets = [train_data, val_data_pairs],
        num_boost_round = NUM_BOOST_ROUND,
        verbose_eval = 10,
        evals_result = evals_result,
        early_stopping_rounds = 7,
        categorical_feature = categorical_features_idxs,
        feature_name = features,
    )

    # save model
    model_lgb.save_model(f'model.lgb')
    
    return model_lgb, evals_result
    
model_lgb, evals_result = train_f()


# In[54]:


# clean up
del train_data, val_data_pairs
gc.collect()


# In[55]:


# Get validation predictions
pred_df = pd.DataFrame({ 'pred': model_lgb.predict(pairs_features[val_idxs]) })
pred_df['pred_correct'] = (pred_df['pred'] > 0.50) == pairs_target[val_idxs]
pred_df['match'] = pairs_target[val_idxs]

# Precision on validation set
display(pred_df['pred_correct'].value_counts(normalize=True).to_frame('Value'))


# In[56]:


# Check predicted value of positive/negative samples in the validation set
display(pred_df.groupby('match')['pred'].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]).T)


# #  Training History

# In[57]:


# plots the training history
def plot_history(evals_result):
    for metric in METRICS:
        plt.figure(figsize=(20,8))
        
        for key in evals_result.keys():
            history_len = len(evals_result.get(key)[metric])
            history = evals_result.get(key)[metric]
            x_axis = np.arange(1, history_len + 1)
            plt.plot(x_axis, history, label=key)
        
        x_ticks = list(filter(lambda e: (e % (history_len // 100 * 10) == 0) or e == 1, x_axis))
        plt.xticks(x_ticks, fontsize=12)
        plt.yticks(fontsize=12)

        plt.title(f'{metric.upper()} History of training', fontsize=18);
        plt.xlabel('EPOCH', fontsize=16)
        plt.ylabel(metric.upper(), fontsize=16)
        
        if metric in ['auc']:
            plt.legend(loc='upper left', fontsize=14)
        else:
            plt.legend(loc='upper right', fontsize=14)
        plt.grid()
        plt.show()

plot_history(evals_result)


# # Feature Importance

# In[58]:


# This is incredibly important to see the performance of features
# If a feature is not used to split and does not provide much gain, remove it!
def show_feature_importances(model, importance_type, max_num_features=10**10):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features
    feature_importances['value'] = pd.DataFrame(model.feature_importance(importance_type))
    feature_importances = feature_importances.sort_values(by='value', ascending=False) # sort feature importance
    feature_importances.to_csv(f'feature_importances_{importance_type}.csv') # write feature importance to csv
    feature_importances = feature_importances[:max_num_features] # only show max_num_features
    
    plt.figure(figsize=(10, len(features) * 0.25))
    plt.xlim([0, feature_importances.value.max()*1.1])
    plt.title(f'Feature {importance_type}', fontsize=18);
    sns.barplot(data=feature_importances, x='value', y='feature', palette='rocket');
    for idx, v in enumerate(feature_importances.value):
        plt.text(v, idx, "  {:.2e}".format(v))

show_feature_importances(model_lgb, 'gain')
show_feature_importances(model_lgb, 'split')


# # Precision/Recall/F1

# In[59]:


# Predictions and true labels of validation dataset
y = pairs_target[val_idxs]
y_pred = model_lgb.predict(pairs_features[val_idxs])
print(f'y shape: {y.shape}, y_pred shape: {y_pred.shape}')


# In[60]:


precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred)
thresholds = np.concatenate(([0], thresholds))

f1 = 2 * (precision * recall) / (precision + recall)
f1_arg_best = np.argmax(f1)
f1_best_threshold = thresholds[f1_arg_best]
f1_best_value = f1.max()
print(f'Best F1({f1_best_value:.3f}) at Threshold {f1_best_threshold:.3f}')


# In[61]:


plt.figure(figsize=(15,8))
plt.plot(precision, recall, color='darkorange', label='Precision/Recall')
plt.scatter(precision[f1_arg_best], recall[f1_arg_best], color='red', s=100, marker='o', label=f'Best F1({f1_best_value:.3f}) at Threshold {f1_best_threshold:.3f}')
plt.title('Precision/Recall Curve', size=24)
plt.xlabel('Precision', size=18)
plt.ylabel('Recall', size=18)
plt.xticks(np.arange(0, 1.1, 0.1), size=16)
plt.yticks(size=16)
plt.grid()
plt.legend(prop={'size': 16})
plt.show()


# In[62]:


plt.figure(figsize=(15,8))
plt.plot(recall, thresholds,  color='darkorange', label='Recall/Threshold')
plt.title('Threshold/Recall Curve', size=24)
plt.xlabel('Threshold', size=18)
plt.ylabel('Recall', size=18)
plt.xticks(np.arange(0, 1.1, 0.1), size=16)
plt.yticks(size=16)
plt.grid()
plt.legend(prop={'size': 16})
plt.show()


# In[63]:


plt.figure(figsize=(15,8))
plt.plot(thresholds, precision,  color='darkorange', label='Precision/Threshold')
plt.title('Threshold/Precision Curve', size=24)
plt.xlabel('Threshold', size=18)
plt.ylabel('Precision', size=18)
plt.xticks(np.arange(0, 1.1, 0.1), size=16)
plt.yticks(np.arange(0, 1.1, 0.1), size=16)
plt.grid()
plt.legend(prop={'size': 16})
plt.show()


# In[64]:


del pairs_features, pairs_target
gc.collect()


# # Inference

# In[65]:


# Let's check how the submission should look like
display(sample_submission)


# In[66]:


# Deduce city and country by coordinates
test['city_rg'] = get_cities(test[['latitude', 'longitude']])
test['country_code'] = get_country_codes(test[['latitude', 'longitude']])

# Make pairs from the test set by simply concatenating the test set with postfix "_1"/"_2"
test_features = pd.concat([test.add_suffix('_1'), test.add_suffix('_2')], axis=1)
add_stand_alone_features(test_features)

display(test_features.head())


# In[67]:


# The famous nearest neighbours lookup tree
tree = BallTree(np.deg2rad(test[['latitude', 'longitude']].values), metric='haversine')


# In[68]:


# Columns to include, because they are needed to compute other features
support_columns_1 = ['name_1', 'address_1', 'categories_1']
support_columns_2 = ['name_2', 'address_2', 'categories_2']


# In[69]:


test_features_np_1 = test_features[features_1 + support_columns_1].values.reshape([len(test_features), 1, -1])
test_features_np_2 = test_features[features_2 + support_columns_2].values


# In[70]:


# Check for submission
IS_DUMMY_TEST = len(test) == 5
# Only the 15 nearest neighbours are used
N_NEIGHBOURS = 3 if IS_DUMMY_TEST else 15
# Threshold to include a point as match
THRESHOLD = 0.50
# Maximum distance to include neighbours from
MAX_DISTANCE_KM = 10
# Matrices to save features in, prediction will be done in 1 go
QUERY_MATRIX = np.zeros(shape=[len(test) * N_NEIGHBOURS, len(features)], dtype=np.float32)
QUERY_DISTANCES = np.zeros(shape=[len(test), N_NEIGHBOURS], dtype=np.float32)
QUERY_INDICES = np.zeros(shape=[len(test), N_NEIGHBOURS], dtype=np.int32)

# Inference loop
for row_idx, row in tqdm(test.iterrows(), total=len(test)):
    # Get 15 neaarest neighbours
    dist, ind = tree.query(np.deg2rad([row['latitude'], row['longitude']]).reshape(1, -1), k=N_NEIGHBOURS)
    # Distance from degrees to KM
    dist = dist.squeeze() * EARTH_RADIUS
    ind = ind.squeeze()

    # Make pairs dataframe
    df_query = pd.DataFrame(
        np.concatenate([
                np.repeat(test_features_np_1[row_idx], N_NEIGHBOURS, 0),
                test_features_np_2[ind]
            ]
        , axis=1)
    , columns=features_1 + support_columns_1 + features_2 + support_columns_2)
    
    # Add Combined Features
    add_combined_features(df_query)

    # Save query, distances and indices
    QUERY_MATRIX[row_idx * N_NEIGHBOURS:(row_idx + 1) * N_NEIGHBOURS] = df_query[features].values
    QUERY_DISTANCES[row_idx] = dist
    QUERY_INDICES[row_idx] = ind


# # Predictions

# In[71]:


# Make a single model call, super efficient!
QUERY_PREDS = model_lgb.predict(QUERY_MATRIX).reshape([len(test), N_NEIGHBOURS])
QUERY_PREDS_SERIES = pd.Series(QUERY_PREDS.flatten())


# In[72]:


display(QUERY_PREDS_SERIES.describe())


# In[73]:


plt.figure(figsize=(10,4))
plt.title('Prediction Distribution', size=24)
QUERY_PREDS_SERIES.plot(kind='hist')
plt.show()


# # Generate Submission DataFrame

# In[74]:


# Rows of the DataFrame are saved as dictionaries
submission_dict = []

for row_idx, row in tqdm(test.iterrows(), total=len(test)):
    ind = QUERY_INDICES[row_idx]
    dist = QUERY_DISTANCES[row_idx]
    pred = QUERY_PREDS[row_idx]
    # Point is included if (mind the brackets):
    # (the confidence is above the threshold or the point refers to itself) and the distance is below the threshold
    pred_id_idxs = ind[((pred > THRESHOLD) | (ind == row_idx)) & (dist < MAX_DISTANCE_KM)]
    pred_ids = ' '.join(test.loc[pred_id_idxs, 'id'].tolist())
    
    submission_dict.append({
        'id': row['id'],
        'matches': pred_ids,
    })


# # Submission

# In[75]:


pd.options.display.max_colwidth = 200
submission_df = pd.DataFrame(submission_dict)
display(submission_df.head())


# In[76]:


submission_df.to_csv('submission.csv', index=False)

