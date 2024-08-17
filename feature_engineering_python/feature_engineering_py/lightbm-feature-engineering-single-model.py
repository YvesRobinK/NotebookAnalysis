#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearnex import patch_sklearn

patch_sklearn()

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from IPython.display import display
import collections
from tqdm import tqdm
import string
import Levenshtein
import difflib
import unidecode
import pickle

tqdm.pandas()



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
get_ipython().run_line_magic('load_ext', 'Cython')


# In[2]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[3]:


df_train=pd.read_csv('../input/foursquare-location-matching/test.csv')

with open("../input/baseline-four-square-test/list_models_baseline_foursquare_v17.pkl", 'rb') as f:
    list_models = pickle.load(f)
    

with open("../input/baseline-four-square-test/count_vectorizer_categories_foursquare_v10.pkl", 'rb') as f:
    
    count_vectorizer = pickle.load(f)
    

with open("../input/baseline-four-square-test/variable_vectorizer_name_foursquare_v10.pkl", 'rb') as f:
    name_vectorizer = pickle.load(f)
    
with open("../input/baseline-four-square-test/tsvd_vectorizer_name_foursquare_v10.pkl", 'rb') as f:
    name_tsvd_transformer = pickle.load(f)
    
    
with open("../input/baseline-four-square-test/variable_vectorizer_categories_foursquare_v10.pkl", 'rb') as f:
    categories_vectorizer = pickle.load(f)
    
with open("../input/baseline-four-square-test/tsvd_vectorizer_categories_foursquare_v10.pkl", 'rb') as f:
    categories_tsvd_transformer = pickle.load(f)
    
    




# In[4]:


df_train


# In[5]:


list_models


# In[6]:


#count_vectorizer = CountVectorizer(max_features=10,strip_accents="unicode")
#count_vectorizer = count_vectorizer.fit(df_train["categories"].dropna())


# In[7]:


get_ipython().run_cell_magic('cython', '', 'import numpy as np  # noqa\ncpdef int FastLCS(str S, str T):\n    cdef int i, j\n    cdef int cost\n    cdef int v1,v2,v3,v4\n    cdef int[:, :] dp = np.zeros((len(S) + 1, len(T) + 1), dtype=np.int32)\n    for i in range(len(S)):\n        for j in range(len(T)):\n            cost = (int)(S[i] == T[j])\n            v1 = dp[i, j] + cost\n            v2 = dp[i + 1, j]\n            v3 = dp[i, j + 1]\n            v4 = dp[i + 1, j + 1]\n            dp[i + 1, j + 1] = max((v1,v2,v3,v4))\n    return dp[len(S)][len(T)]\n')


# In[ ]:





# In[8]:


def process_country(df_data):
    
    list_country=  ['US',
     'TR',
     'ID',
     'JP',
     'TH',
     'RU',
     'BR',
     'MY',
     'BE',
     'GB',
     'PH',
     'MX',
     'SG',
     'KR',
     'DE',
     'FR',
     'ES']
        
    df_data["country_popular"]=df_data["country"].copy()
        
    df_data.loc[~df_data["country"].isin(list_country),"country_popular"]="OTHER"
        
        
    return df_data


def create_n_words(data):

    data.loc[data['name'].notnull(),'n_words_name']=data.loc[data['name'].notnull(),'name'].apply(lambda x: len(str(x).split()))

    data.loc[data['name'].notnull(),'n_characters_name']=data.loc[data['name'].notnull(),'name'].apply(lambda x: len(str(x).replace(" ","")))
    
    data.loc[data['name_match_id'].notnull(),'n_words_name_match_id']=data.loc[data['name_match_id'].notnull(),'name_match_id'].apply(lambda x: len(str(x).split()))
    
    data.loc[data['name_match_id'].notnull(),'n_characters_name_match_id']=data.loc[data['name_match_id'].notnull(),'name_match_id'].apply(lambda x: len(str(x).replace(" ","")))
    
    data.loc[data['address'].notnull(),'n_words_address']=data.loc[data['address'].notnull(),'address'].apply(lambda x: len(str(x).split()))
    
    data.loc[data['address'].notnull(),'n_characters_address']=data.loc[data['address'].notnull(),'address'].apply(lambda x: len(str(x).replace(" ","")))
    
    data.loc[data['address_match_id'].notnull(),'n_words_address_match_id']=data.loc[data['address_match_id'].notnull(),'address_match_id'].apply(lambda x: len(str(x).split()))
    
    data.loc[data['address_match_id'].notnull(),'n_characters_address_match_id']=data.loc[data['address_match_id'].notnull(),'address_match_id'].apply(lambda x: len(str(x).replace(" ","")))
    
    data.loc[data['categories'].notnull(),'n_words_categories']=data.loc[data['categories'].notnull(),'categories'].apply(lambda x: len(str(x).split()))
    
    data.loc[data['categories_match_id'].notnull(),'n_words_categories_match_id']=data.loc[data['categories_match_id'].notnull(),'categories_match_id'].apply(lambda x: len(str(x).split()))
    
    return data

    
def create_equal_indicator_features(data,equal_variables):
    
    for name_variable in equal_variables:
        
        name_match_variable=f'{name_variable}_match_id'
        name_equal_indicator=f'equal_{name_variable}'
        
        data[name_equal_indicator]=(data[name_variable]==data[name_match_variable]).astype('int8')
        
        condition=(data[name_variable]==np.nan)|(data[name_match_variable]==np.nan)
        
        data.loc[condition,name_equal_indicator]=np.nan
        
        print(name_variable)
        
    return data



def jaccard_distance(id_str,id_match_str):

    try :
        score = len((id_str & id_match_str)) / len((id_str | id_match_str))
        
    except :
        
        score=np.nan
    
    return score

def first_indicator(id_str,id_match_str):

    try :
        score = int(id_str.split()[0] in id_match_str)
        
    except :
        
        score=np.nan
    
    return score

def last_indicator(id_str,id_match_str):

    try :
        score = int(id_str.split()[-1] in id_match_str)
        
    except :
        
        score=np.nan
    
    return score

def first_match_indicator(id_str,id_match_str):

    try :
        score = len((id_str & id_match_str)) / len(id_str)
        
    except :
        
        score=np.nan
    
    return score

def second_match_indicator(id_str,id_match_str):

    try :
        score = len((id_str & id_match_str)) / len(id_match_str)
        
    except :
        
        score=np.nan
    
    return score



def levenshtein_distance(id_str,id_match_str):
    
    try :
    
        score=Levenshtein.distance(id_str, id_match_str)
        
    except :
        
        score=np.nan
    
    return score

def jaro_winkler(id_str,id_match_str):
    
    try :
    
        score=Levenshtein.jaro_winkler(id_str, id_match_str)
        
    except :
        
        score=np.nan
    
    return score

def sequence_matcher(id_str,id_match_str):
    
    try :
    
        score=difflib.SequenceMatcher(None, id_str,id_match_str).ratio()
        
    except :
        
        score=np.nan
    
    return score

def process_string_variable(id_str):

    if id_str is float :

        id_str=str(int(id_str))
        
    else :
        
        id_str=str(id_str)
        
    return id_str


def metrics_similiarity(data,name_variable,name_match_variable=None,distances_list=["jaccard"],suffix='',make_unidecode=0,remove_vowels=0):
    ## check if there should be separation between words for some punctuations
    
    scores_jaccard = []
    scores_levenshtein=[]
    scores_jaro_winkler=[]
    scores_sequence_matcher=[]
    scores_lcs=[]
    scores_first_match_indicator=[]
    scores_second_match_indicator=[]
    scores_first_indicator=[]
    scores_last_indicator=[]
    str_vowels="aeiou"
    
    
    if name_match_variable is None :
    
        name_match_variable=f'{name_variable}_match_id'
    
    for id_str, id_match_str in tqdm(zip(data[name_variable].fillna("nullvalue").to_numpy(), data[name_match_variable].fillna("nullvalue_match").to_numpy())):
    
        if make_unidecode==1 :
            id_str=unidecode.unidecode(id_str)
            id_match_str=unidecode.unidecode(id_match_str)
            
        id_str=id_str.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
        if remove_vowels==1:
            id_str=id_str.lower().translate(str.maketrans('', '',str_vowels))
        id_str=" ".join(id_str.split()) # this is new
            
        id_match_str=id_match_str.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
        if remove_vowels==1:
            id_match_str=id_match_str.lower().translate(str.maketrans('', '',str_vowels))
        id_match_str=" ".join(id_match_str.split())

        
        id_str_set=set(id_str.split())
        id_match_str_set=set(id_match_str.split())
        
                
        if "jaccard" in distances_list :
                
            score_jaccard=jaccard_distance(id_str_set,id_match_str_set)
                
            scores_jaccard.append(score_jaccard)
            
        if "first_match_indicator" in distances_list :
                
            score_first_match_indicator=first_match_indicator(id_str_set,id_match_str_set)
                
            scores_first_match_indicator.append(score_first_match_indicator)
            
        if "second_match_indicator" in distances_list :
                
            score_second_match_indicator=second_match_indicator(id_str_set,id_match_str_set)
                
            scores_second_match_indicator.append(score_second_match_indicator)
            
        if "first_indicator" in distances_list :
                
            score_first_indicator=first_indicator(id_str,id_match_str)
                
            scores_first_indicator.append(score_first_indicator)
            
        if "last_indicator" in distances_list :
                
            score_last_indicator=last_indicator(id_str,id_match_str)
                
            scores_last_indicator.append(score_last_indicator)
            
            
        if "levenshtein" in distances_list:
                
            score_levenshtein=levenshtein_distance(id_str,id_match_str)
                
            scores_levenshtein.append(score_levenshtein)
                
        if "jaro_winkler" in distances_list:
                
            score_jaro_winkler=jaro_winkler(id_str,id_match_str)
                
            scores_jaro_winkler.append(score_jaro_winkler)
            
        if "sequence_matcher" in distances_list:
            
            score_sequence_matcher=sequence_matcher(id_str,id_match_str)
                
            scores_sequence_matcher.append(score_sequence_matcher)
            
        if "lcs" in distances_list:
            
            score_lcs=FastLCS(id_str,id_match_str)
                
            scores_lcs.append(score_lcs)
                
     
    condition=data[name_variable].isnull() | data[name_match_variable].isnull()
            
            
    if "jaccard" in distances_list :
                            
        scores_jaccard=np.array(scores_jaccard)
        
        data[f'jaccard_distance_match_{name_variable}_{suffix}']=scores_jaccard
        
        data.loc[condition,f'jaccard_distance_match_{name_variable}_{suffix}']=np.nan
        
    if "first_match_indicator" in distances_list :
        
        scores_first_match_indicator=np.array(scores_first_match_indicator)
        
        data[f'first_match_indicator_match_{name_variable}_{suffix}']=scores_first_match_indicator
        
        data.loc[condition,f'first_match_indicator_match_{name_variable}_{suffix}']=np.nan
        
    if "second_match_indicator" in distances_list :
        
        scores_second_match_indicator=np.array(scores_second_match_indicator)
        
        data[f'second_match_indicator_match_{name_variable}_{suffix}']=scores_second_match_indicator
        
        data.loc[condition,f'second_match_indicator_match_{name_variable}_{suffix}']=np.nan
        
        
    if "first_indicator" in distances_list :
                            
        scores_first_indicator=np.array(scores_first_indicator)
        
        data[f'first_indicator_match_{name_variable}_{suffix}']=scores_first_indicator
        
        data.loc[condition,f'first_indicator_match_{name_variable}_{suffix}']=np.nan
        
    if "last_indicator" in distances_list :
                            
        scores_last_indicator=np.array(scores_last_indicator)
        
        data[f'last_indicator_match_{name_variable}_{suffix}']=scores_last_indicator
        
        data.loc[condition,f'last_indicator_match_{name_variable}_{suffix}']=np.nan
            
            
    if "levenshtein" in distances_list:
                
        scores_levenshtein=np.array(scores_levenshtein)
        
        data[f'levenshtein_distance_match_{name_variable}_{suffix}']=scores_levenshtein
        
        data.loc[condition,f'levenshtein_distance_match_{name_variable}_{suffix}']=np.nan
                
    if "jaro_winkler" in distances_list:
        
        scores_jaro_winkler=np.array(scores_jaro_winkler)
        
        data[f'jaro_winkler_distance_match_{name_variable}_{suffix}']=scores_jaro_winkler
        
        data.loc[condition,f'jaro_winkler_distance_match_{name_variable}_{suffix}']=np.nan
        
    if "sequence_matcher" in distances_list:
        
        scores_sequence_matcher=np.array(scores_sequence_matcher)
        
        data[f'sequence_matcher_distance_match_{name_variable}_{suffix}']=scores_sequence_matcher
        
        data.loc[condition,f'sequence_matcher_distance_match_{name_variable}_{suffix}']=np.nan
        
    if "lcs" in distances_list:
        
        scores_lcs=np.array(scores_lcs)
        
        data[f'lcs_distance_match_{name_variable}_{suffix}']=scores_lcs
        
        data.loc[condition,f'lcs_distance_match_{name_variable}_{suffix}']=np.nan
    
    

    return data


def get_numbers_from_string(data,name_variable):
    
    name_number_variable=f'numbers_{name_variable}'
    
    data[name_number_variable]=data[name_variable].fillna('nullvalue').apply(lambda x: (" ".join([''.join(filter(str.isdigit, string)) for string in x.split()])).strip())
    
    data.loc[data[name_number_variable]=='',name_number_variable]=np.nan
    
    return data

def get_numbers_match(data,name_variable):

    data=get_numbers_from_string(data,name_variable)

    data=get_numbers_from_string(data,f'{name_variable}_match_id')

    data=metrics_similiarity(data,f'numbers_{name_variable}')
    
    return data


def get_categories_indicator(data,variable_name,transformer,n_features=10):
    
    data_result=transformer.transform(data[variable_name].fillna("null"))
    
    data_result=data_result.toarray()
    
    data_result=pd.DataFrame(data_result)
    
    dict_names = {transformer.vocabulary_[k] : k for k in transformer.vocabulary_}

    list_names=[f'{dict_names[x]}_{variable_name}' for x in range(0,n_features) ]
    
    data_result.columns=list_names
    
    data = pd.concat([data, data_result], axis=1)

    
    return data

def get_tsvd_lda_vectors(data,variable_name,vector_transformer,tsvd_transformer):
    
    variable_vectors=vector_transformer.transform(data[variable_name].fillna(""))
    
    data_result=tsvd_transformer.transform(variable_vectors)
    
    n_features=np.shape(data_result)[1]
    
    data_result=pd.DataFrame(data_result)
        
    list_names=[f'{variable_name}_component_{x}' for x in range(0,n_features) ]
    
    data_result.columns=list_names
    
    data = pd.concat([data, data_result], axis=1)

    return data




def nearest_neighbors(data,n_nearest):
    
    knn = NearestNeighbors(n_neighbors = n_nearest,metric='haversine')
    knn.fit(data, data.index)
    distances,nearest_ids = knn.kneighbors(data,return_distance = True)
    
    return distances,nearest_ids


# based on this https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/discussion/228814#1253355
def jaccard_similarity(row,variable):
    
    try : 
        l1 = row[variable].split(" ")
        l2 = row[f'{variable}_match_id'].split(" ")    
        intersection = len([s for s in l1 if s in l2])
        union = (len(l1) + len(l2)) - intersection
        return float(intersection) / union
    
    except :
        return np.nan
    
    
def jaccard_similarity_like(data,name_variable):
    
    name_match_variable=f'{name_variable}_match_id'
    
    count_vectorizer = CountVectorizer(binary=True,stop_words=['nullvalue'],token_pattern=r"(?u)\b\w+\b",strip_accents="unicode")
    data_result = count_vectorizer.fit_transform(data[name_variable].fillna('nullvalue'))
    dict_id_index = dict(zip(data[name_variable].values, data.index))
    
    index_match_id=[dict_id_index[element] for element in data[name_match_variable]]
    
    distance_vector=data_result.multiply(data_result[index_match_id]).sum(axis=1).A.ravel()
    
    return distance_vector


def filter_candidates(data,n_neighbor):
    
    #data=data.loc[(data['jaccard_like_distance_name']>0)|(data['jaccard_like_distance_address']>0)].reset_index(drop=True)
    
    #data=data.loc[(data['jaccard_like_distance_name']>0)].reset_index(drop=True)
        
    #data=data.loc[(data['jaccard_like_distance_name']>0)|(data['jaro_winkler_distance_match_name_']>0.7)].reset_index(drop=True)
    
    #data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.7)].reset_index(drop=True)
    
    #data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.4)|(data['jaccard_distance_match_name_unidecode']>0)].reset_index(drop=True)
        
    #data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.7)|(data['jaccard_distance_match_name_unidecode']>0)|(data['jaro_winkler_distance_match_name_unidecode']>0.7)].reset_index(drop=True)
    
    #data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.7)|(data['jaccard_distance_match_name_unidecode']>0)|(data['jaro_winkler_distance_match_name_unidecode']>0.7)|(data["jaccard_distance_match_numbers_address_"]==1)|(data['jaccard_distance_match_address_']==1)].reset_index(drop=True)
    
    #data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.65)|(data['jaccard_distance_match_name_unidecode']>0)|(data['jaro_winkler_distance_match_name_unidecode']>0.65)].reset_index(drop=True)
    
    
    if n_neighbor > 1 :
    
        data=data.loc[(data['jaccard_distance_match_name_']>0)|(data['jaro_winkler_distance_match_name_']>0.7)|(data['jaccard_distance_match_name_unidecode']>0)|(data['jaro_winkler_distance_match_name_unidecode']>0.7)].reset_index(drop=True)
    
    return data


def get_match_variables(data,data_variables):

    data=data.merge(data_variables,on='id',how='left',suffixes=(None, '_id'))

    data=data.merge(data_variables,left_on='id_match',right_on='id',how='left',suffixes=(None, '_match_id'))
    
    return data
    
    

def create_candidates(data_position,measure_variables,n_nearest,data_features):
    
    data_position["longitude"]=np.radians(data_position["longitude"].values)
    
    data_position["latitude"]=np.radians(data_position["latitude"].values)
    
    
    distances,nearest_ids=nearest_neighbors(data_position[measure_variables],n_nearest=n_nearest)
    
    df_neighbors=[]
    
    
    for n_neighbor in range(1,n_nearest):
        
        df_temp=data_position[['id']].copy()
        
        df_temp['distance']=distances[:,n_neighbor]
        
        index_id_match=nearest_ids[:,n_neighbor]
        
        df_temp['id_match']=df_temp[['id']].loc[index_id_match].values
        
        df_temp['n_neighbor']=n_neighbor
        
        df_temp=get_match_variables(df_temp,data_features[['id','name']])
        
        df_temp=metrics_similiarity(df_temp,'name',distances_list=["jaccard","jaro_winkler"])
        
        df_temp=metrics_similiarity(df_temp,'name',distances_list=["jaccard","jaro_winkler"],make_unidecode=1,suffix="unidecode")
        
        print(len(df_temp))

        df_temp=filter_candidates(df_temp,n_neighbor)

        print(len(df_temp))
        print(df_temp["distance"].describe())

        df_neighbors.append(df_temp)
        
        print(n_neighbor)
        
        print('')
        
        
    df_neighbors=pd.concat(df_neighbors,ignore_index=True).reset_index(drop=True)
    
    df_neighbors=df_neighbors.drop(["id_match_id"],axis='columns')
    
    print(len(df_neighbors))
    
    df_temp=df_neighbors.copy()
    
    df_temp[["id","id_match"]]=df_temp[["id_match","id"]].copy()
    
    df_temp["n_neighbor"]=99
    
    df_neighbors=pd.concat([df_neighbors,df_temp],ignore_index=True)
    
    del df_temp ; gc.collect()
    
    df_neighbors=df_neighbors.drop_duplicates(subset=["id","id_match"],keep="first",ignore_index=True)
    
    print(len(df_neighbors))

    return df_neighbors


def get_perfect_jaccard_score(data):
    
    dict_predicted_matches=data.loc[data['target']==1].groupby('id')['id_match'].apply(list).to_dict()    
    #df_real_matchs=data.loc[data['target']==1].groupby('id')['id_match'].apply(list).reset_index()
    
    scores = []
    for id_str, matches in tqdm(zip(df_real_matches['id'].to_numpy(), df_real_matches['id_match'].to_numpy())):
        
        if id_str in dict_predicted_matches:
            targets = dict_predicted_matches[id_str]
            targets.append(id_str)
            targets=set(targets)
        else :
            targets=set([id_str])
        preds = set(matches)
        score = len((targets & preds)) / len((targets | preds))
        scores.append(score)
    scores = np.array(scores)
    
    metric=scores.mean()
    
    print(metric)
    
    return metric


def get_tsvd_vectors(data,variable,n_components):
    
    variable_vectorizer = CountVectorizer(strip_accents="unicode",binary=True)
    variable_vectorizer = variable_vectorizer.fit(data[variable].fillna(""))
    variable_vectors=variable_vectorizer.transform(data[variable].fillna(""))

    tsvd_tranformer=TruncatedSVD(n_components=n_components)
    tsvd_tranformer=tsvd_tranformer.fit(variable_vectors)
    tsvd_vectors=tsvd_tranformer.transform(variable_vectors)
    
    return tsvd_vectors


def nearest_neighbors_arrays(data,n_nearest):
    
    knn = NearestNeighbors(n_neighbors = n_nearest)
    knn.fit(data,[1]*len(data))
    distances,nearest_ids = knn.kneighbors(data,return_distance = True)
    
    return distances,nearest_ids



def create_candidates_strings_vectors(data_id,data_position,n_nearest,data_features,variable="name"):
    
    distances,nearest_ids=nearest_neighbors_arrays(data_position,n_nearest=n_nearest)
    
    df_neighbors=[]
    
    
    for n_neighbor in range(0,n_nearest):
        
        df_temp=data_id[['id']].copy()
        
        df_temp['distance_vectors']=distances[:,n_neighbor]
        
        index_id_match=nearest_ids[:,n_neighbor]
        
        df_temp['id_match']=df_temp[['id']].loc[index_id_match].values
        
        df_temp['n_neighbor']=n_neighbor
        
        print(len(df_temp))
        
        df_temp=get_match_variables(df_temp,data_features)
        
        df_temp=metrics_similiarity(df_temp,variable,distances_list=["jaccard","jaro_winkler"])
        
        df_temp=metrics_similiarity(df_temp,variable,distances_list=["jaccard","jaro_winkler"],make_unidecode=1,suffix="unidecode")
                
        if variable == "name" :
        
            df_temp=filter_candidates_names(df_temp)
            
        elif variable=="address" :
            
            df_temp=filter_candidates_address(df_temp)
            
            
        
        print(len(df_temp))

        df_neighbors.append(df_temp)
        
        print(n_neighbor)
        
        print('')
        
        
    df_neighbors=pd.concat(df_neighbors,ignore_index=True).reset_index(drop=True)
    
    df_neighbors=df_neighbors.drop(["id_match_id"],axis='columns')
    
    print((df_neighbors["id"]==df_neighbors["id_match"]).mean())
    
    df_neighbors=df_neighbors.loc[df_neighbors["id"]!=df_neighbors["id_match"]].reset_index(drop=True)
    
    df_neighbors["n_neighbor"]=999
    
    df_neighbors["distance"]=999
    
    print(len(df_neighbors))

    return df_neighbors

  
def filter_candidates_names(data):
    
    data=data.loc[(data['jaccard_distance_match_name_']>0.875)|(data['jaro_winkler_distance_match_name_']>0.9)|(data['jaccard_distance_match_name_unidecode']>0.875)|(data['jaro_winkler_distance_match_name_unidecode']>0.9)].reset_index(drop=True)
    
    return data


def filter_candidates_address(data):
    
    data=data.loc[(data['jaccard_distance_match_address_']>0.875)|(data['jaro_winkler_distance_match_address_']>0.9)|(data['jaccard_distance_match_address_unidecode']>0.875)|(data['jaro_winkler_distance_match_address_unidecode']>0.9)].reset_index(drop=True)
    
    return data


# In[9]:


n=7 if len(df_train)>100 else 3


df_train_strings=df_train.dropna(subset=["address"])[["id","name","address","latitude","longitude"]].reset_index(drop=True)

name_vectors=get_tsvd_vectors(df_train_strings,"name",n)

address_vectors=get_tsvd_vectors(df_train_strings,"address",n)


n=15 if len(df_train)>100 else 3

neighbors_strings=create_candidates_strings_vectors(df_train_strings[["id"]],name_vectors,n,df_train_strings)

del name_vectors ; gc.collect()


n=10 if len(df_train)>100 else 3

address_neighbors_strings=create_candidates_strings_vectors(df_train_strings[["id"]],address_vectors,n,df_train_strings,variable='address')

del address_vectors
del df_train_strings; gc.collect()


print(len(neighbors_strings))

neighbors_strings=neighbors_strings.merge(address_neighbors_strings[["id","id_match"]],on=["id","id_match"])

print(len(neighbors_strings))

del address_neighbors_strings ; gc.collect()

neighbors_strings=neighbors_strings[['id','distance','id_match','n_neighbor','name','name_match_id','jaccard_distance_match_name_','jaro_winkler_distance_match_name_','jaccard_distance_match_name_unidecode','jaro_winkler_distance_match_name_unidecode']]


# In[10]:


n_nearest=61 if len(df_train)>100 else 2


df_neighbors=create_candidates(df_train[['id','latitude','longitude']],measure_variables=["latitude","longitude"],n_nearest=n_nearest,data_features=df_train[["id","name"]])

print(len(df_neighbors))

df_neighbors=pd.concat([df_neighbors,neighbors_strings],ignore_index=True)

df_neighbors=df_neighbors.drop_duplicates(["id","id_match"]).reset_index(drop=True)

del neighbors_strings ; gc.collect()


# In[11]:


df_train.loc[df_train["zip"].notnull(),"zip"]=df_train.loc[df_train["zip"].notnull(),"zip"].apply(process_string_variable)

df_train.loc[df_train["phone"].notnull(),"phone"]=df_train.loc[df_train["phone"].notnull(),"phone"].apply(process_string_variable)


df_submission=df_train[["id"]].copy()


df_neighbors=get_match_variables(df_neighbors,df_train[['id',"address",'city', 'state', 'zip','country', 'url', 'phone','categories','latitude','longitude']])

del df_train ; gc.collect()

df_neighbors=df_neighbors.drop(["id_match_id"],axis='columns')

print(df_neighbors['distance'].describe())

df_neighbors=create_n_words(df_neighbors)

df_neighbors=create_equal_indicator_features(df_neighbors,equal_variables=[ 'zip',
       'country', 'url', 'phone','categories'])

df_neighbors=metrics_similiarity(df_neighbors,'name',distances_list=["lcs","levenshtein","sequence_matcher","first_match_indicator","second_match_indicator","first_indicator","last_indicator"])


#df_neighbors=metrics_similiarity(df_neighbors,'name',distances_list=["jaccard","jaro_winkler"],make_unidecode=1,suffix="unidecode")
#df_neighbors=metrics_similiarity(df_neighbors,'name',distances_list=["jaccard"],remove_vowels=1,suffix="vowels")

df_neighbors=metrics_similiarity(df_neighbors,'name',distances_list=["jaccard"],remove_vowels=1,suffix="vowels_unicode",make_unidecode=1)


#df_neighbors=metrics_similiarity(df_neighbors,'address',distances_list=["jaccard"],remove_vowels=1,suffix="vowels")

#df_neighbors=metrics_similiarity(df_neighbors,'address',distances_list=["jaccard"],remove_vowels=1,suffix="vowels_unicode",make_unidecode=1)



df_neighbors=metrics_similiarity(df_neighbors,'name',distances_list=["lcs"],make_unidecode=1,suffix="unidecode")

df_neighbors=metrics_similiarity(df_neighbors,'address',distances_list=["jaccard","levenshtein","jaro_winkler","sequence_matcher"])

df_neighbors=metrics_similiarity(df_neighbors,'address',distances_list=["jaccard","jaro_winkler"],make_unidecode=1,suffix="unidecode")

df_neighbors=metrics_similiarity(df_neighbors,'categories',distances_list=["jaccard","levenshtein","jaro_winkler","sequence_matcher"])

df_neighbors=metrics_similiarity(df_neighbors,'city',distances_list=["jaccard"])

df_neighbors=metrics_similiarity(df_neighbors,'city',distances_list=["jaro_winkler"],make_unidecode=1,suffix="unidecode")

df_neighbors=metrics_similiarity(df_neighbors,'state',distances_list=["jaccard"])

df_neighbors=metrics_similiarity(df_neighbors,'state',distances_list=["jaro_winkler"],make_unidecode=1,suffix="unidecode")

df_neighbors=metrics_similiarity(df_neighbors,'zip',distances_list=["jaccard"])

df_neighbors=metrics_similiarity(df_neighbors,'zip',distances_list=["jaro_winkler"],make_unidecode=1,suffix="unidecode")


df_neighbors=metrics_similiarity(df_neighbors,'phone',distances_list=["jaro_winkler","sequence_matcher"])

df_neighbors=metrics_similiarity(df_neighbors,'name','url',distances_list=["jaro_winkler"],suffix='url')

df_neighbors=metrics_similiarity(df_neighbors,'name','url_match_id',distances_list=["jaro_winkler"],suffix='url_match_id')

df_neighbors=metrics_similiarity(df_neighbors,'name_match_id','url',distances_list=["jaro_winkler"],suffix='url')

df_neighbors=metrics_similiarity(df_neighbors,'name_match_id','url_match_id',distances_list=["jaro_winkler"],suffix='url_match_id')

df_neighbors=metrics_similiarity(df_neighbors,'name','address_match_id',distances_list=["jaccard"],suffix='name_address')

df_neighbors=metrics_similiarity(df_neighbors,'name_match_id','address',distances_list=["jaccard"],suffix='name_match_id_address')




#df_neighbors=get_numbers_from_string(df_neighbors,'name')

#df_neighbors=get_numbers_from_string(df_neighbors,'name_match_id')

#df_neighbors=metrics_similiarity(df_neighbors,'numbers_name')

df_neighbors=get_numbers_match(df_neighbors,'name')


#df_neighbors=get_numbers_from_string(df_neighbors,'address')

#df_neighbors=get_numbers_from_string(df_neighbors,'address_match_id')

#df_neighbors=metrics_similiarity(df_neighbors,'numbers_address')

#df_neighbors=get_numbers_match(df_neighbors,'address')



#df_neighbors=process_country(df_neighbors)

df_neighbors=reduce_mem_usage(df_neighbors)


df_neighbors=get_categories_indicator(df_neighbors,"categories",count_vectorizer,10)

df_neighbors=get_categories_indicator(df_neighbors,"categories_match_id",count_vectorizer,10)

del count_vectorizer ; gc.collect()

df_neighbors=get_tsvd_lda_vectors(df_neighbors,"name",name_vectorizer,name_tsvd_transformer)

df_neighbors=get_tsvd_lda_vectors(df_neighbors,"name_match_id",name_vectorizer,name_tsvd_transformer)

del name_vectorizer
del name_tsvd_transformer ; gc.collect()

df_neighbors=get_tsvd_lda_vectors(df_neighbors,"categories",categories_vectorizer,categories_tsvd_transformer)

df_neighbors=get_tsvd_lda_vectors(df_neighbors,"categories_match_id",categories_vectorizer,categories_tsvd_transformer)

del categories_vectorizer
del categories_tsvd_transformer ; gc.collect()



#df_neighbors=get_tsvd_lda_vectors(df_neighbors,"address",address_vectorizer,address_tsvd_transformer)

#df_neighbors=get_tsvd_lda_vectors(df_neighbors,"address_match_id",address_vectorizer,address_tsvd_transformer)

#del address_vectorizer
#del address_tsvd_transformer ; gc.collect()


# In[ ]:





# In[12]:


variables_delete=['name', 'address', 'city', 'state', 'zip', 'country', 'url',
       'phone', 'categories', 
       'name_match_id', "latitude_match_id","longitude_match_id",
       'address_match_id', 'city_match_id', 'state_match_id', 'zip_match_id',
       'country_match_id', 'url_match_id', 'phone_match_id',
       'categories_match_id', 
         'numbers_name','numbers_name_match_id']
    
df_neighbors=df_neighbors.drop(variables_delete,axis="columns")


# In[13]:


df_neighbors=reduce_mem_usage(df_neighbors)


# In[14]:


features=['distance',
 'n_neighbor',
 'jaccard_distance_match_name_',
 'jaro_winkler_distance_match_name_',
 'jaccard_distance_match_name_unidecode',
 'jaro_winkler_distance_match_name_unidecode',
 'latitude',
 'longitude',
 'n_words_name',
 'n_characters_name',
 'n_words_name_match_id',
 'n_characters_name_match_id',
 'n_words_address',
 'n_characters_address',
 'n_words_address_match_id',
 'n_characters_address_match_id',
 'n_words_categories',
 'n_words_categories_match_id',
 'equal_zip',
 'equal_country',
 'equal_url',
 'equal_phone',
 'equal_categories',
 'first_match_indicator_match_name_',
 'second_match_indicator_match_name_',
 'first_indicator_match_name_',
 'last_indicator_match_name_',
 'levenshtein_distance_match_name_',
 'sequence_matcher_distance_match_name_',
 'lcs_distance_match_name_',
 'jaccard_distance_match_name_vowels_unicode',
 'lcs_distance_match_name_unidecode',
 'jaccard_distance_match_address_',
 'levenshtein_distance_match_address_',
 'jaro_winkler_distance_match_address_',
 'sequence_matcher_distance_match_address_',
 'jaccard_distance_match_address_unidecode',
 'jaro_winkler_distance_match_address_unidecode',
 'jaccard_distance_match_categories_',
 'levenshtein_distance_match_categories_',
 'jaro_winkler_distance_match_categories_',
 'sequence_matcher_distance_match_categories_',
 'jaccard_distance_match_city_',
 'jaro_winkler_distance_match_city_unidecode',
 'jaccard_distance_match_state_',
 'jaro_winkler_distance_match_state_unidecode',
 'jaccard_distance_match_zip_',
 'jaro_winkler_distance_match_zip_unidecode',
 'jaro_winkler_distance_match_phone_',
 'sequence_matcher_distance_match_phone_',
 'jaro_winkler_distance_match_name_url',
 'jaro_winkler_distance_match_name_url_match_id',
 'jaro_winkler_distance_match_name_match_id_url',
 'jaro_winkler_distance_match_name_match_id_url_match_id',
 'jaccard_distance_match_name_name_address',
 'jaccard_distance_match_name_match_id_name_match_id_address',
 'jaccard_distance_match_numbers_name_',
 'bars_categories',
 'buildings_categories',
 'cafes_categories',
 'college_categories',
 'food_categories',
 'offices_categories',
 'restaurants_categories',
 'shops_categories',
 'stations_categories',
 'stores_categories',
 'bars_categories_match_id',
 'buildings_categories_match_id',
 'cafes_categories_match_id',
 'college_categories_match_id',
 'food_categories_match_id',
 'offices_categories_match_id',
 'restaurants_categories_match_id',
 'shops_categories_match_id',
 'stations_categories_match_id',
 'stores_categories_match_id',
 'name_component_0',
 'name_component_1',
 'name_component_2',
 'name_component_3',
 'name_component_4',
 'name_component_5',
 'name_component_6',
 'name_match_id_component_0',
 'name_match_id_component_1',
 'name_match_id_component_2',
 'name_match_id_component_3',
 'name_match_id_component_4',
 'name_match_id_component_5',
 'name_match_id_component_6',
 'categories_component_0',
 'categories_component_1',
 'categories_component_2',
 'categories_component_3',
 'categories_component_4',
 'categories_component_5',
 'categories_component_6',
 'categories_match_id_component_0',
 'categories_match_id_component_1',
 'categories_match_id_component_2',
 'categories_match_id_component_3',
 'categories_match_id_component_4',
 'categories_match_id_component_5',
 'categories_match_id_component_6']



categorical=[]


# In[15]:


df_neighbors[features]


# In[16]:


df_neighbors["estimator"]=0

for modelo in list_models:

    df_neighbors["estimator"]=df_neighbors["estimator"]+modelo.predict(df_neighbors[features].values, num_iteration=modelo.best_iteration)/5


# In[17]:


df_neighbors.groupby("id")["id_match"].apply(list)


# In[18]:


def adjust_results(df_results):

    df_results_adjusted=df_results.merge(df_results,left_on=["id","id_match"],right_on=["id_match","id"],suffixes=("_prob_1","_prob_2"))

    del df_results ; gc.collect()

    df_results_adjusted=df_results_adjusted.drop(['id_prob_2','id_match_prob_2'],axis='columns')

    df_results_adjusted=df_results_adjusted.rename(columns={"id_prob_1":"id","id_match_prob_1":"id_match"})

    df_results_adjusted["estimator"]=(df_results_adjusted["estimator_prob_1"]+df_results_adjusted["estimator_prob_2"])/2
    
    return df_results_adjusted


def get_transitive_matches(df_matches):
    
    df_extra_matches=df_matches.merge(df_matches,left_on="id_match",right_on="id",how="left")
        
    df_extra_matches=df_extra_matches.drop(['id_match_x','id_y'],axis="columns")
    
    df_extra_matches=df_extra_matches.rename(columns={"id_x":"id","id_match_y":"id_match"})
    
    df_extra_matches=pd.concat([df_matches,df_extra_matches],ignore_index=True)
    
    df_extra_matches=df_extra_matches.drop_duplicates(["id","id_match"],ignore_index=True)
    
    return df_extra_matches
    
    


# In[19]:


df_results=df_neighbors[['id','id_match','estimator']]

df_results_adjusted=adjust_results(df_results)


condition=df_results_adjusted["estimator"]>0.5

df_matches=df_results_adjusted.loc[condition,["id","id_match"]].reset_index(drop=True)

#condition=df_results["estimator"]>0.5

#df_matches=df_results.loc[condition,["id","id_match"]].reset_index(drop=True)


df_extra_matches=get_transitive_matches(df_matches)


# In[20]:


df_matches


# In[ ]:





# In[21]:


def get_submission(data,condition):
    
    #dict_predicted_matches=data.loc[condition].groupby('id')['id_match'].apply(list).to_dict() 
    
    dict_predicted_matches=data.groupby('id')['id_match'].apply(list).to_dict()
    
    
    predictions = []
    for id_str in tqdm(df_submission['id'].to_numpy()):
        
        if id_str in dict_predicted_matches:
            
            list_predictions=list(dict_predicted_matches[id_str])
            
            if id_str not in list_predictions :

                list_predictions.append(id_str)
            
            list_predictions=' '.join(list_predictions)
            
            predictions.append(list_predictions)
            
        else :
            
            predictions.append(id_str)
            
    df_submission["matches"]=predictions

    
    return df_submission


# In[ ]:





# In[22]:


#condition=df_results_adjusted["estimator"]>0.5

#df_matches=df_results_adjusted.loc[condition,["id","id_match"]].reset_index(drop=True)


df_submission=get_submission(df_matches,condition)

df_submission_2=get_submission(df_extra_matches,condition)


# In[23]:


df_submission


# In[24]:


#df_submission_2


# In[25]:


def post_process(df):
    id2match = dict(zip(df['id'].values, df['matches'].str.split()))

    for base, match in tqdm(df[['id', 'matches']].values):
        match = match.split()
        if len(match) == 1:        
            continue

        for m in match:
            if base not in id2match[m]:
                id2match[m].append(base)
    df['matches'] = df['id'].map(id2match).map(' '.join)
    return df


# In[ ]:





# In[26]:


#df_submission=post_process(df_submission)


# In[27]:


df_submission_2.to_csv("submission.csv", index=False)


# In[28]:


df_submission

