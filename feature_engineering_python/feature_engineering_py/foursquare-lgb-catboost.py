#!/usr/bin/env python
# coding: utf-8

# ## Notebook for Foursquare Competition

# * Get 25 neighbors using Latitude + Longitude
# * Get 25 neighbors using Name tokenized by roberta-base + Latitude + Longitude
# * Concat dataframes and drop duplicates
# * Filter dataframe to reduce its size
# * Create 48 features
# * Predict with LGBM
# * Predict with Catboost
# * Ensemble: 0.2xCatboost + 0.8xLGBM
# * Post-processing 1: if A matches B then B matches A
# * Post-processing 2: if A matches B and B matches C, then A matchs C and C matches A
# * Submit
# * Upvote ;)

# ### Import

# In[1]:


get_ipython().run_cell_magic('capture', '', '# install reverse-geocode\n!mkdir -p /tmp/pip/cache/\n!cp ../input/reverse-geocode/reverse_geocode.xyz /tmp/pip/cache/reverse_geocode.tar.gz\n!pip install /tmp/pip/cache/reverse_geocode.tar.gz\n')


# In[2]:


from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import gc
import random
from sklearn.model_selection import GroupKFold
import warnings
import pickle
from unidecode import unidecode
import reverse_geocode
import string
import Levenshtein
import difflib
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer
import time
import joblib
from math import radians


# In[3]:


tokenizer = AutoTokenizer.from_pretrained('../input/robertabase/roberta-base')


# ### Config and Read

# In[4]:


class CFG:
    seed = 46
    target = "point_of_interest"
    n_neighbors = 25
    n_splits = 3
    threshold = 0.528812
random.seed(CFG.seed)
os.environ["PYTHONHASHSEED"] = str(CFG.seed)
np.random.seed(CFG.seed)


# In[5]:


test = pd.read_csv("../input/foursquare-location-matching/test.csv")
test[CFG.target] = "TEST"


# In[6]:


test.head(3)


# ### Clean data

# In[7]:


# Get address number to create a new feature later
test['number'] = test.address.str.extract('(\d+)')


# In[8]:


test.drop(['state'], axis=1, inplace=True)


# In[9]:


gc.collect()


# In[10]:


def clean_country(cols):
    lat = cols[0]
    long = cols[1]
    country = cols[2]
    
    if country != country: # check if is nan
        coordinates = (lat, long),
        result = reverse_geocode.search(coordinates)
        return result[0]['country_code']
    return country


# In[11]:


test['country'] = test[['latitude','longitude','country']].apply(clean_country, axis=1)


# In[12]:


china = test.loc[test['country'] == 'CN']['name'].values
japan = test.loc[test['country'] == 'JP']['name'].values


# In[13]:


def standard_data(df):
    columns = ['name', 'categories', 'country', 'zip', 'phone', 'url', 'city']
    
    for c in columns:        
        df[c] = df[c].astype(str).str.lower()
        df[c] = df[c].apply(lambda x: unidecode(x)) # will remove for china and japan later
        df[c] = df[c].astype(str).str.lower()
        if c in ['zip', 'phone', 'url']:
            df[c] = df[c].str.replace('[{}]'.format(string.punctuation), '')
            df[c] = df[c].str.replace(' ', '')
        if c == 'url':
            df[c] = df[c].str.replace('http://', '')
            df[c] = df[c].str.replace('https://', '')
            df[c] = df[c].str.replace('http:', '')
            df[c] = df[c].str.replace('https:', '')
            df[c] = df[c].str.replace('http', '')
            df[c] = df[c].str.replace('https', '')
            df[c] = df[c].str.replace('www.', '')
            df[c] = df[c].str.replace('www', '')
        df[c] = df[c].replace('nan', np.nan)
    return df


# In[14]:


test = standard_data(test)


# In[15]:


test.loc[test['country'] == 'cn', 'name'] = china
test.loc[test['country'] == 'jp', 'name'] = japan
del china, japan
gc.collect()


# ### Add count features

# In[16]:


test['latitude_round'] = test['latitude'].round(1)
test['longitude_round'] = test['longitude'].round(1)

test['latitude'] = test['latitude'].apply(lambda x: radians(x))
test['longitude'] = test['longitude'].apply(lambda x: radians(x))


# In[17]:


test['latitude_round_count'] = -99
test['longitude_round_count'] = -99
test['name_count'] = -99
test['country_count'] = -99


# In[18]:


def count_values(df, col):
    freq_encode = df[col].value_counts(dropna=True).to_dict()
    df[col+'_count'] = df[col].map(freq_encode)
    return df[col+'_count']


# In[19]:


for col in ['country']:
    test[col+'_count'] = count_values(test, col)
    test[col+'_count'].fillna(-99, inplace=True)
    test[col+'_count'] = test[col+'_count'].astype(int)


# In[20]:


gc.collect()


# ### Helper functions

# In[21]:


def add_distance_features(cols, *args):
    str1 = cols[0]
    str2 = cols[1]
    feat = ''.join(args)
    
    if str1 == str1 and str2 == str2 and str1 != '' and str2 != '':
        if feat == 'leven':
            if str1 == str2:
                return 0
            return Levenshtein.distance(str1, str2) # Levenshtein
        elif feat == 'jaro':
            if str1 == str2:
                return 1
            return Levenshtein.jaro_winkler(str1, str2) # jaro_winkler
        elif feat == 'lcs':
            return LCS(str1, str2) # LCS
        elif feat == 'jaccard_char':
            if str1 == str2:
                return 1
            return calculate_jaccard_char(str1, str2) # jaccard char
        elif feat == 'jaccard_char_smallest':
            if str1 == str2:
                return 1
            return calculate_jaccard_char_smallest(str1, str2) # jaccard char smallest
        elif feat == 'jaccard_word':
            if str1 == str2:
                return 1
            return calculate_jaccard_word(str1, str2) # jaccard word
        elif feat == 'jaccard_word_smallest':
            if str1 == str2:
                return 1
            return calculate_jaccard_word_smallest(str1, str2) # jaccard word smallest
    return -99


# In[22]:


def calculate_jaccard_char(str1, str2):
    
    # Combine both tokens to find union.
    both_tokens = str1 + str2
    union = set(both_tokens)
    if len(union) == 0:
        return 0
    
    # Calculate intersection.
    intersection = set()
    for w in set(str1):
        if w in set(str2):
            intersection.add(w)

    jaccard_score = len(intersection)/len(union)
    
    return jaccard_score


# In[23]:


def calculate_jaccard_char_smallest(str1, str2):
    str1 = set(str1)
    str2 = set(str2)
    
    small = min(len(str1), len(str2))
    if small == 0:
        return 0
    
    # Calculate intersection.
    intersection = set()
    for w in str1:
        if w in str2:
            intersection.add(w)

    jaccard_score = len(intersection)/small
    
    return jaccard_score


# In[24]:


def calculate_jaccard_word(str1, str2):
    
    # Combine both tokens to find union.
    words1 = str1.split()
    words2 = str2.split()
    union = set(words1 + words2)
    if len(union) == 0:
        return 0
    
    # Calculate intersection.
    intersection = set()
    for word in union:
        if word in words1 and word in words2:
            intersection.add(word)

    jaccard_score = len(intersection)/len(union)
    
    return jaccard_score


# In[25]:


def calculate_jaccard_word_smallest(str1, str2):
    
    if str1 == str2:
        return 1
    
    # Combine both tokens to find union.
    words1 = str1.split()
    words2 = str2.split()
    union = set(words1 + words2)
    small = min(len(set(words1)), len(set(words2)))
    if small == 0:
        return 0
    
    # Calculate intersection.
    intersection = set()
    for word in union:
        if word in words1 and word in words2:
            intersection.add(word)

    jaccard_score = len(intersection)/small
    
    return jaccard_score


# In[26]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[27]:


get_ipython().run_cell_magic('cython', '', 'import numpy as np  # noqa\ncpdef int LCS(str S, str T):\n    cdef int i, j\n    cdef int cost\n    cdef int v1,v2,v3,v4\n    cdef int[:, :] dp = np.zeros((len(S) + 1, len(T) + 1), dtype=np.int32)\n    for i in range(len(S)):\n        for j in range(len(T)):\n            cost = (int)(S[i] == T[j])\n            v1 = dp[i, j] + cost\n            v2 = dp[i + 1, j]\n            v3 = dp[i, j + 1]\n            v4 = dp[i + 1, j + 1]\n            dp[i + 1, j + 1] = max((v1,v2,v3,v4))\n    return dp[len(S)][len(T)]\n')


# In[28]:


from math import sin, cos, sqrt, atan2

def get_real_distance(cols):
    R = 6378.0 # radius of earth in km
    
    lat1 = cols[0]
    lon1 = cols[1]
    lat2 = cols[2]
    lon2 = cols[3]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


# ### Search Candidates

# In[29]:


if len(test) < CFG.n_neighbors:
    CFG.n_neighbors = len(test)


# In[30]:


def get_vectors(df, col):
    """Function to search candidates by name vectorized + latitude + longitude"""
    encoded_input = tokenizer(df[col].fillna('nan').tolist(), return_tensors='np', padding=True)
    encoded_input = encoded_input['input_ids']
    encoded_input = np.c_[encoded_input, df['latitude'].values, df['longitude'].values]
    normalized_encoded_input = encoded_input/np.linalg.norm(encoded_input)
    return normalized_encoded_input


# In[31]:


def get_neighbors_vector(embeddings, n_neighbors, metric, algorithm):
    """Function to search candidates by name vectorized + latitude + longitude"""
    matcher = NearestNeighbors(n_neighbors = n_neighbors,
                       metric = metric,
                       radius=1,
                       algorithm=algorithm,
                       leaf_size=30,
                       p=2,
                       n_jobs=-1)
    
    matcher.fit(embeddings)
    distances, indices = matcher.kneighbors(embeddings)
    return distances, indices


# In[32]:


def get_neighbors_lat_long(df, n_neighbors, metric, algorithm):
    """Function to search candidates by latitude and longitude"""
    matcher = NearestNeighbors(n_neighbors = n_neighbors,
                       metric = metric,
                       radius=1,
                       algorithm=algorithm,
                       leaf_size=30,
                       p=2,
                       n_jobs=-1)
    matcher.fit(df[['latitude', 'longitude']])
    distances, indices = matcher.kneighbors(df[['latitude', 'longitude']])
    return distances, indices


# In[33]:


candidate_df = pd.DataFrame()

for country, country_df in tqdm(test.groupby("country")):
    dfs = []
    country_df = country_df.reset_index(drop=True)
    
    # name count
    freq_encode = country_df['name'].value_counts(dropna=True).to_dict()
    country_df['name_count'] = country_df['name'].map(freq_encode)
    
    # latitude_round count
    freq_encode = country_df['latitude_round'].value_counts(dropna=True).to_dict()
    country_df['latitude_round_count'] = country_df['latitude_round'].map(freq_encode)
    
    # longitude_round count
    freq_encode = country_df['longitude_round'].value_counts(dropna=True).to_dict()
    country_df['longitude_round_count'] = country_df['longitude_round'].map(freq_encode)
    
    # get neighbors by name vectorized + lat + long
    embeddings = get_vectors(country_df, 'name')
    distances, indices = get_neighbors_vector(embeddings, min(len(country_df), CFG.n_neighbors), 'hamming', 'auto')

    # get neighbors by lat + long
    distances2, indices2 = get_neighbors_lat_long(country_df, min(len(country_df), CFG.n_neighbors), 'manhattan', 'auto')
    
    for i in range(min(len(country_df), CFG.n_neighbors)):        
        # name vectorized + lat + long
        tmp_df = pd.DataFrame()
        tmp_df = country_df[["id"]].copy()
        tmp_df["dist"] = distances[:, i]
        tmp_df['dist_mean_neighboors'] = distances[:, :].mean()
        tmp_df["dist_type"] = 0
        tmp_df["country"] = country
        tmp_df['country_count'] = country_df['country_count']
        tmp_df['latitude_round_count'] = country_df['latitude_round_count']
        tmp_df['latitude_round_count_neighbor'] = country_df['latitude_round_count'].values[indices[:, i]]
        tmp_df['longitude_round_count'] = country_df['longitude_round_count']
        tmp_df['longitude_round_count_neighbor'] = country_df['longitude_round_count'].values[indices[:, i]]
        tmp_df['latitude'] = country_df['latitude']
        tmp_df['latitude_neighbor'] = country_df['latitude'].values[indices[:, i]]
        tmp_df['longitude'] = country_df['longitude']
        tmp_df['longitude_neighbor'] = country_df['longitude'].values[indices[:, i]]
        tmp_df['id_neighbor'] = country_df['id'].values[indices[:, i]]
        tmp_df["neighbor_nearest"] = i
        tmp_df['number'] = country_df['number']
        tmp_df['number_neighbor'] = country_df['number'].values[indices[:, i]]
        tmp_df['name'] = country_df['name']
        tmp_df['name_neighbor'] = country_df['name'].values[indices[:, i]]
        tmp_df['name_count'] = country_df['name_count']
        tmp_df['name_count_neighbor'] = country_df['name_count'].values[indices[:, i]]
        tmp_df['categories'] = country_df['categories']
        tmp_df['categories_neighbor'] = country_df['categories'].values[indices[:, i]]
        tmp_df['address'] = country_df['address']
        tmp_df['address_neighbor'] = country_df['address'].values[indices[:, i]]
        tmp_df['zip'] = country_df['zip']
        tmp_df['zip_neighbor'] = country_df['zip'].values[indices[:, i]]
        tmp_df['phone'] = country_df['phone']
        tmp_df['phone_neighbor'] = country_df['phone'].values[indices[:, i]]
        tmp_df['url'] = country_df['url']
        tmp_df['url_neighbor'] = country_df['url'].values[indices[:, i]]
        tmp_df['city'] = country_df['city']
        tmp_df['city_neighbor'] = country_df['city'].values[indices[:, i]]
        tmp_df['name_jaccard_char'] = tmp_df[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_char', axis=1)
        tmp_df['name_jaccard_char_smallest'] = tmp_df[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_char_smallest', axis=1)
        tmp_df['name_jaccard_word'] = tmp_df[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_word', axis=1)
        tmp_df['name_jaccard_word_smallest'] = tmp_df[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_word_smallest', axis=1)
        
        # lat + long
        tmp_df2 = pd.DataFrame()
        tmp_df2 = country_df[["id"]].copy()
        tmp_df2["dist"] = distances2[:, i]
        tmp_df2['dist_mean_neighboors'] = distances2[:, :].mean()
        tmp_df2["dist_type"] = 1
        tmp_df2["country"] = country
        tmp_df2['country_count'] = country_df['country_count']
        tmp_df2['latitude_round_count'] = country_df['latitude_round_count']
        tmp_df2['latitude_round_count_neighbor'] = country_df['latitude_round_count'].values[indices2[:, i]]
        tmp_df2['longitude_round_count'] = country_df['longitude_round_count']
        tmp_df2['longitude_round_count_neighbor'] = country_df['longitude_round_count'].values[indices2[:, i]]
        tmp_df2['latitude'] = country_df['latitude']
        tmp_df2['latitude_neighbor'] = country_df['latitude'].values[indices2[:, i]]
        tmp_df2['longitude'] = country_df['longitude']
        tmp_df2['longitude_neighbor'] = country_df['longitude'].values[indices2[:, i]]
        tmp_df2['id_neighbor'] = country_df['id'].values[indices2[:, i]]
        tmp_df2["neighbor_nearest"] = i
        tmp_df2['number'] = country_df['number']
        tmp_df2['number_neighbor'] = country_df['number'].values[indices2[:, i]]
        tmp_df2['name'] = country_df['name']
        tmp_df2['name_neighbor'] = country_df['name'].values[indices2[:, i]]
        tmp_df2['name_count'] = country_df['name_count']
        tmp_df2['name_count_neighbor'] = country_df['name_count'].values[indices2[:, i]]
        tmp_df2['categories'] = country_df['categories']
        tmp_df2['categories_neighbor'] = country_df['categories'].values[indices2[:, i]]
        tmp_df2['address'] = country_df['address']
        tmp_df2['address_neighbor'] = country_df['address'].values[indices[:, i]]
        tmp_df2['zip'] = country_df['zip']
        tmp_df2['zip_neighbor'] = country_df['zip'].values[indices[:, i]]
        tmp_df2['phone'] = country_df['phone']
        tmp_df2['phone_neighbor'] = country_df['phone'].values[indices[:, i]]
        tmp_df2['url'] = country_df['url']
        tmp_df2['url_neighbor'] = country_df['url'].values[indices[:, i]]
        tmp_df2['city'] = country_df['city']
        tmp_df2['city_neighbor'] = country_df['city'].values[indices[:, i]]
        tmp_df2['name_jaccard_char'] = tmp_df2[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_char', axis=1)
        tmp_df2['name_jaccard_char_smallest'] = tmp_df2[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_char_smallest', axis=1)
        tmp_df2['name_jaccard_word'] = tmp_df2[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_word', axis=1)
        tmp_df2['name_jaccard_word_smallest'] = tmp_df2[['name', 'name_neighbor']].apply(add_distance_features, args='jaccard_word_smallest', axis=1)
        
        # Concat both dataframes
        tmp_final = pd.concat([tmp_df,tmp_df2])
        tmp_final.drop_duplicates(subset=['id', 'id_neighbor'], keep='first', inplace=True, ignore_index=False)
        
        # Calculate real dist
        tmp_final['real_dist'] = tmp_final[['latitude','longitude','latitude_neighbor','longitude_neighbor']].apply(get_real_distance, axis=1)
        tmp_final.drop(['latitude','longitude','latitude_neighbor','longitude_neighbor'], axis=1, inplace=True)
        gc.collect()
        
        # Filter
        tmp_final = tmp_final[(tmp_final['neighbor_nearest'] <= 1) | (tmp_final['real_dist'] < 15) | (tmp_final['name_jaccard_char'] >= 0.75) | (tmp_final['name_jaccard_word'] >= 0.25)]
        tmp_final = tmp_final[(tmp_final['neighbor_nearest'] <= 10) | (tmp_final['real_dist'] < 10) | (tmp_final['name_jaccard_char_smallest'] >= 0.9) | (tmp_final['name_jaccard_word'] >= 0.3)]
        tmp_final = tmp_final[(tmp_final['neighbor_nearest'] <= 20) | (tmp_final['real_dist'] < 1) | (tmp_final['name_jaccard_char_smallest'] >= 0.99) | (tmp_final['name_jaccard_word_smallest'] >= 0.4)]
        
        dfs.append(tmp_final)
        del tmp_df, tmp_df2, tmp_final
        gc.collect()
        
    dfs = pd.concat(dfs)
    candidate_df = pd.concat([candidate_df, dfs])

del dfs, tokenizer
gc.collect()


# In[34]:


del test
gc.collect()


# In[35]:


candidate_df.drop('country', axis=1, inplace=True)
gc.collect()


# In[36]:


candidate_df.reset_index(drop=True, inplace=True)
candidate_df.shape


# ### Reduce Memory

# In[37]:


for col in tqdm(candidate_df.columns):
    if '_count' in col:
        candidate_df[col].fillna(-99, inplace=True)
        candidate_df[col] = candidate_df[col].astype(int)


# In[38]:


def reduce_mem_usage(df, cols, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    if 'int8' not in str(col_type):
                        df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    if 'int16' not in str(col_type):
                        df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    if 'int32' not in str(col_type):
                        df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    if 'int64' not in str(col_type):
                        df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    if 'float16' not in str(col_type):
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    if 'float32' not in str(col_type):
                        df[col] = df[col].astype(np.float32)
                else:
                    if 'float64' not in str(col_type):
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[39]:


candidate_df = reduce_mem_usage(candidate_df, candidate_df.columns)


# In[40]:


gc.collect()


# ### Feature Engineering

# In[41]:


columns = ['name', 'categories']
feats = ['jaccard_char', 'jaccard_char_smallest', 'jaccard_word', 'jaccard_word_smallest',
         'leven', 'jaro', 'lcs']

for col in tqdm(columns):
    for feat in feats:
        if col == 'name' and 'jaccard' in feat:
            continue
        candidate_df[col+'_'+feat] = candidate_df[[col, col+'_neighbor']].apply(add_distance_features, args=feat, axis=1)


# In[42]:


gc.collect()


# In[43]:


columns = ['address']
feats = ['jaccard_char', 'jaccard_word', 'lcs']

for col in tqdm(columns):
    for feat in feats:
        candidate_df[col+'_'+feat] = candidate_df[[col, col+'_neighbor']].apply(add_distance_features, args=feat, axis=1)
    candidate_df.drop([col, col+'_neighbor'], axis=1, inplace=True)


# In[44]:


gc.collect()


# In[45]:


columns = ['zip', 'phone', 'url']
feats = ['jaro']

for col in tqdm(columns):
    for feat in feats:
        candidate_df[col+'_'+feat] = candidate_df[[col, col+'_neighbor']].apply(add_distance_features, args=feat, axis=1)
    candidate_df.drop([col, col+'_neighbor'], axis=1, inplace=True)


# In[46]:


gc.collect()


# In[47]:


columns = ['city']
feats = ['jaccard_char']

for col in tqdm(columns):
    print(col+':')
    for feat in feats:
        print(feat)
        candidate_df[col+'_'+feat] = candidate_df[[col, col+'_neighbor']].apply(add_distance_features, args=feat, axis=1)
    candidate_df.drop([col, col+'_neighbor'], axis=1, inplace=True)


# In[48]:


gc.collect()


# In[49]:


candidate_df = reduce_mem_usage(candidate_df, candidate_df.columns)
gc.collect()


# In[50]:


columns = ['name', 'categories', 'name_neighbor', 'categories_neighbor']
for col in tqdm(columns):
    candidate_df[col+'_len'] = candidate_df[col].astype(str).map(len) 
    candidate_df.loc[candidate_df[col].isnull(), col+'_len'] = -99
    candidate_df[col+'_count_word'] = candidate_df[col].astype(str).apply(lambda x: len(x.split()))
    candidate_df.loc[candidate_df[col].isnull(), col+'_count_word'] = -99
    candidate_df.drop(col, axis=1, inplace=True)
    gc.collect()


# In[51]:


candidate_df['name_lcs_smallest'] = candidate_df['name_lcs'] / candidate_df[['name_len', 'name_neighbor_len']].min(axis=1)
candidate_df['name_lcs_biggest'] = candidate_df['name_lcs'] / candidate_df[['name_len', 'name_neighbor_len']].max(axis=1)
candidate_df['categories_lcs_smallest'] = candidate_df['categories_lcs'] / candidate_df[['categories_len', 'categories_neighbor_len']].min(axis=1)
candidate_df['categories_lcs_biggest'] = candidate_df['categories_lcs'] / candidate_df[['categories_len', 'categories_neighbor_len']].max(axis=1)


# In[52]:


candidate_df['name_leven_biggest'] = candidate_df['name_leven'] / candidate_df[['name_len', 'name_neighbor_len']].max(axis=1)
candidate_df['categories_leven_biggest'] = candidate_df['categories_leven'] / candidate_df[['categories_len', 'categories_neighbor_len']].max(axis=1)


# ### Create Feature About Address Number

# In[53]:


candidate_df['same_number'] = candidate_df['number'] == candidate_df['number_neighbor']
candidate_df['same_number'] = candidate_df['same_number'].astype(int)


# In[54]:


candidate_df.drop(['number', 'number_neighbor'], axis=1, inplace=True)
gc.collect()


# In[55]:


candidate_df.shape


# ### Final Reduce Memory

# In[56]:


candidate_df.replace([np.inf, -np.inf], -99, inplace=True)
candidate_df.fillna(-99, inplace=True)


# In[57]:


gc.collect()
candidate_df = reduce_mem_usage(candidate_df, candidate_df.columns)
gc.collect()


# ### Delete Not Used Columns

# In[58]:


len(candidate_df.columns)


# In[59]:


columns = candidate_df.columns.to_list()
not_use = ['id', 'id_neighbor', 'name', 'name_neighbor', 'address', 'address_neighbor', 'city',
          'city_neighbor', 'country', 'country_neighbor', 'zip', 'zip_neighbor', 'url', 'url_neighbor',
          'phone', 'phone_neighbor', 'categories', 'categories_neighbor', 'point_of_interest',
          'point_of_interest_neighbor', 'target', 'number', 'number_neighbor', 'name_gesh', 'categories_gesh']
features = [item for item in columns if item not in not_use]


# In[60]:


len(features)


# In[61]:


candidate_df = candidate_df[features + ['id', 'id_neighbor']]
candidate_df.reset_index(drop=True, inplace=True)


# In[62]:


gc.collect()


# ### Inference

# In[63]:


from catboost import CatBoostClassifier
chunks = 10
chunk_size = int(len(candidate_df) / chunks)
pred_cat = np.zeros(len(candidate_df))
pred_lgb = np.zeros(len(candidate_df))

for i in tqdm(range(CFG.n_splits)):
    
    # catboost
    model = CatBoostClassifier()
    model.load_model(f'../input/modelcatboostfoursquare2/model2/catboost_fold{i}.cbm')
    for chunk in range(chunks):
        if chunk < chunks - 1:
            pred_cat[chunk*chunk_size:chunk*chunk_size+chunk_size] += model.predict_proba(candidate_df[features][chunk*chunk_size:chunk*chunk_size+chunk_size].to_numpy())[:,1] / CFG.n_splits
        else:
            pred_cat[chunk*chunk_size:] += model.predict_proba(candidate_df[features][chunk*chunk_size:].to_numpy())[:,1] / CFG.n_splits
    del model
    gc.collect()
    
    # lgb
    model = joblib.load(f'../input/modellgbfoursquare29/model29/lgbm_fold{i}.pkl')
    for chunk in range(chunks):
        if chunk < chunks - 1:
            pred_lgb[chunk*chunk_size:chunk*chunk_size+chunk_size] += model.predict(candidate_df[features][chunk*chunk_size:chunk*chunk_size+chunk_size].to_numpy()) / CFG.n_splits
        else:
            pred_lgb[chunk*chunk_size:] += model.predict(candidate_df[features][chunk*chunk_size:].to_numpy()) / CFG.n_splits
    del model
    gc.collect()

candidate_df = candidate_df[['id', 'id_neighbor']]
pred = 0.5*pred_cat + 0.5*pred_lgb
del pred_cat, pred_lgb
gc.collect()
pred = (pred >= CFG.threshold).astype(int)
candidate_df['pred'] = pred


# ### Set matches

# In[64]:


test_original = pd.read_csv("../input/foursquare-location-matching/test.csv", usecols=['id'])


# In[65]:


matches = []
ids_check = []
candidate_df_ones = candidate_df[candidate_df['pred'] == 1][['id','id_neighbor']]
for id, id_df in tqdm(candidate_df_ones.groupby('id')):
    match = id_df['id_neighbor'].to_list()
    if id not in match:
        match = [id] + match
    match = ' '.join(match)
    matches.append(match)
    ids_check.append(id)

del candidate_df, candidate_df_ones
gc.collect()

test_original["matches"] = test_original["id"]
test_original.loc[test_original['id'].isin(ids_check), 'matches'] = matches


# ### Postproccess

# In[66]:


def postprocess(df):
    """ if A matches B then B matches A"""
    id2match = dict(zip(df["id"].values, df["matches"].str.split()))

    for match in tqdm(df["matches"]):
        match = match.split()
        if len(match) == 1:        
            continue

        base = match[0]
        for m in match[1:]:
            if not base in id2match[m]:
                id2match[m].append(base)
    df["matches"] = df["id"].map(id2match).map(" ".join)
    return df 

test_original = postprocess(test_original)


# In[67]:


def postprocess2(df):
    """ if A matches B and B matches C, then A matchs C and C matches A """
    match_map = df.set_index('id')['matches'].to_dict()
    matches = []
    ids_check = []
    for id, id_df in tqdm(df.groupby('id')):
        match = id_df['matches'].str.split().iloc[0]
        new_match = []
        for match_id in match:
            if match_id != id:
                new_match += match_map[match_id].split()

        if len(new_match) > 0:
            match += new_match
            match = list(dict.fromkeys(match)) # remove duplicates
        match = ' '.join(match)
        matches.append(match)
        ids_check.append(id)

    df.loc[df['id'].isin(ids_check), 'matches'] = matches
    return df

test_original = postprocess2(test_original)


# ### Submit

# In[68]:


ssub = pd.read_csv("../input/foursquare-location-matching/sample_submission.csv")
ssub = ssub.drop(columns="matches")
ssub = ssub.merge(test_original[["id", "matches"]], on="id")
ssub.to_csv("submission.csv", index=False)

ssub.head()


# 👍
