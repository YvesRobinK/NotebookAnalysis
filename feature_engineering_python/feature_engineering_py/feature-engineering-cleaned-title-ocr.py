#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:60px"><center>Shopee Product Matching</center></h1>
# 
# ![shopee logo](https://i.imgur.com/GvmrZK0.png)

# ## Feature Engineering Notebook
# 
# 
#     
# Here I try to create as many features as possible so that in the modelling phase I can get good results. I am using the cleaned title and OCR data that I did in my previous [Notebook](https://www.kaggle.com/mohneesh7/shopee-challenge-eda-nlp-on-title-ocr). If you haven't followed it, please check it out.
# 
# I haven't lemmatized the title text there, I will try to use lemmatized/stemmed words as another feature. Let's see if it gives good results.
#     
# 

# In[ ]:


# import sys
# !cp ../input/rapids/rapids.0.18.0 /opt/conda/envs/rapids.tar.gz
# !cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
# sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
# sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
# sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
# !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/


# ### Import Required Packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import cv2
from joblib import dump, load
from tqdm.notebook import tqdm
import re
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
from cuml.neighbors import NearestNeighbors


# ### Add required paths so as to avoid confusion later on

# In[ ]:


path = '../input/shopee-product-matching'
train_path = '../input/shopee-product-matching/train_images'
test_path = '../input/shopee-product-matching/test_images'
cleaned_data_path = '../input/cleaned-shopee-data-with-ocr/cleaned_title_and_ocr_sw.csv'


# ### Let's load in the data and have a quick look

# In[ ]:


data = pd.read_csv(cleaned_data_path)
data.drop('Unnamed: 0',axis=1,inplace=True)
data.head()


# ### Approaches :
# 
# + **Euclidean Distannces between title/ocr text using word2vec or tfidf vectors.**
# + **Image Similarity using the phash value, setting a threshold for the hamming distance might be a good idea(subject to choosing the best hyperparamter). we will do this in modelling phase**
# + **Use all features to find nearest neighbours and group them together (more resource and time intensive,selecting k neighbours will be key)**
# + **Maybe we can turn this into a clasification problem with number of classes equal to the number of unique label_groups, I am a bit confused on this though.**

# In[ ]:


s_stemmer = SnowballStemmer(language='english')
words = data['cleaned_title'].iloc[4587].split()
lemmatizer = WordNetLemmatizer()
for word in words:
    print(word+' --> '+s_stemmer.stem(word))
    print(word+' --> '+lemmatizer.lemmatize(word))


# **I have decided not to do lemmatization/Stemming as most of the words are not native english words, there are other language words in english that might not get stemmed properly, better to leave it for now, I will come back here if I need to fine tune model and if this helps.**

# In[ ]:


data_train = pd.read_csv(cleaned_data_path)


# ### Text Features
# 
# + **Length & Word count of titles and OCR text**

# In[ ]:


tqdm.pandas()


# In[ ]:


data_train['len_title'] = data_train['cleaned_title'].progress_apply(lambda x: len(x))
data_train['word_count_title'] = data_train['cleaned_title'].progress_apply(lambda x: len(x.split()))
data_train['len_ocr'] = data_train['cleaned_ocr_text'].progress_apply(lambda x: len(str(x)))
data_train['word_count_ocr'] = data_train['cleaned_ocr_text'].progress_apply(lambda x: len(str(x).split()))


# + **Average Word lengths in title and OCR text**

# In[ ]:


data_train['avg_word_length_title'] = data_train['len_title']/data_train['word_count_title']
data_train['avg_word_length_ocr'] = data_train['len_ocr']/data_train['word_count_ocr']


# In[ ]:


def n_gram_count(text,n):
    word_vectorizer = CountVectorizer(ngram_range=(n,n), analyzer='word', stop_words=None,max_df=0.8)
    if len(text.split()) == 1:
        return 1
    if len(text.split()) == 0:
        return 0
    print(text)
    sparse_matrix = word_vectorizer.fit_transform([text])
    frequencies = len(sum(sparse_matrix).toarray()[0])
    return frequencies


# In[ ]:


data_train = cudf.DataFrame.from_pandas(data_train)


# In[ ]:


import gc
_ = gc.collect()


# In[ ]:


tfidf_vec = TfidfVectorizer(stop_words='english', 
                            binary=True, 
#                             max_df = 0.5,
#                             min_df = 2
                           )
title_embeddings = tfidf_vec.fit_transform(data_train['cleaned_title']).toarray().astype(np.float32)
title_embeddings.shape


# In[ ]:


tfidf_vec_2 = TfidfVectorizer(stop_words='english', 
                            binary=True, 
#                             max_df = 0.5,
#                             min_df = 2
                           )
ocr_embeddings = tfidf_vec_2.fit_transform(data_train['cleaned_ocr_text']).toarray().astype(np.float32)
ocr_embeddings.shape


# In[ ]:


# def cosine(v1, v2):
#     v1 = np.array(v1)
#     v2 = np.array(v2)

#     return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
# sims = []
# for i in tqdm(range(len(title_embeddings))):
#     sims_temp = []
#     for j in range(i,len(title_embeddings)):
#         sim = cosine(title_embeddings[i],title_embeddings[j])
#         if sim >= 0.5:
#             sims_temp.append(data_train['posting_id'].iloc[j])
#     sims.append(sims_temp)


# **Not using the above code, It works perfectly but is very slow as it runs on CPU, using RAPIDS is the only alternative for now it seems. the codde from the cell below has been taken from chris Deotte's notebook** [ Check it Out Here](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)

# Title Embeddings Cosine Similarity

# In[ ]:


title_embeddings[0]


# In[ ]:


preds = []
CHUNK = 1024

print('Finding similar titles...')
CTS = len(title_embeddings)//CHUNK
if len(title_embeddings)%CHUNK!=0: CTS += 1
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(title_embeddings))
    print('chunk',a,'to',b)
    
    # COSINE SIMILARITY DISTANCE
    cts = cupy.matmul(title_embeddings, title_embeddings[a:b].T).T
    
    for k in range(b-a):
        IDX = cupy.where(cts[k,]>0.7)[0]
        o = data.iloc[cupy.asnumpy(IDX)]['posting_id'].values
        preds.append(o)
        
# del tfidf_vec, text_embeddings
# _ = gc.collect()
data_train['title_cos_sim>0.7'] = preds


# OCR Text Cosine similarity

# In[ ]:


preds_ocr = []
CHUNK = 1024

print('Finding similar titles...')
CTS = len(ocr_embeddings)//CHUNK
if len(ocr_embeddings)%CHUNK!=0: CTS += 1
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(ocr_embeddings))
    print('chunk',a,'to',b)
    
    # COSINE SIMILARITY DISTANCE
    cts = cupy.matmul(ocr_embeddings, ocr_embeddings[a:b].T).T
    
    for k in range(b-a):
        IDX = cupy.where(cts[k,]>0.7)[0]
        o = data.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds_ocr.append(o)
        
# del tfidf_vec, text_embeddings
# _ = gc.collect()
data_train['ocr_cos_sim>0.7'] = preds_ocr


# **To be on the safe side lets save all features till now (also I dont want to waste GPU time)**

# In[ ]:


data_train.head()
data_train = data_train.to_pandas()
data_train.to_csv('features_till_cos_sim.csv',index=False)


# In[ ]:


data_train = pd.read_csv('features_till_cos_sim.csv')


# # Work in Progress
# 
# I am working on implementing more features.

# In[ ]:




