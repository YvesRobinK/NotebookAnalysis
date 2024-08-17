#!/usr/bin/env python
# coding: utf-8

# # Credit
# 
# BPE Encoding based custom tokenizer by @datafan07
# 
# Ertuğrul Demir - https://www.kaggle.com/code/datafan07/train-your-own-tokenizer
# 
# Dataset: Augmented dataset by @jdragonxherrera https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456729
# 
# Fork from https://www.kaggle.com/code/xiaocao123/ai-generated-text-detection-add-new-data
# 
# https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline
# 
# Inspired by VLADIMIR DEMIDOV's work : <br>
# https://www.kaggle.com/code/yekenot/llm-detect-by-regression <br>
# https://www.kaggle.com/code/x75a40890/ai-generated-text-detection-quick-baseline
# 
# Using new train dataset https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/
# 
# @chenbaoying
# https://www.kaggle.com/code/chenbaoying/0-911-ai-generated-text-detection-test-feature#Voting-Classifier
# 
# And for all others who contributed through discussions/codes

# # Importing library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import random,os,sys
from concurrent.futures import ProcessPoolExecutor
import re
import joblib

from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

import xgboost as xgb
import catboost as ctb
import lightgbm as lgb


# In[2]:


import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


# In[3]:


random.seed(42)
np.random.seed(42)
debug=False
pseudo=False
LOWERCASE = False
VOCAB_SIZE = 30522


# In[4]:


get_ipython().system('pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/')
get_ipython().system('mkdir -p /root/.cache/language_tool_python/')
get_ipython().system('cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7')
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')


# # Importing files and Feature Engineering

# In[5]:


def denoise_text(text):
    # Assuming 'tool' is defined elsewhere in your code
    corrected_text = tool.correct(text)
    return corrected_text

# Function to correct the 'text' column of a DataFrame or Series in parallel
def correct_df(input_data):
    if isinstance(input_data, pd.DataFrame):
        # If input is a DataFrame, correct the 'text' column
        with ProcessPoolExecutor() as executor:
            input_data['text'] = list(executor.map(denoise_text, input_data['text']))
    elif isinstance(input_data, pd.Series):
        # If input is a Series, correct the series
        with ProcessPoolExecutor() as executor:
            input_data = list(executor.map(denoise_text, input_data)) 
    return input_data


# In[6]:


# import spacy

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from collections import Counter

# from tqdm import tqdm


# @torch.no_grad()
# def clean_essay(text): 
#     doc = nlp(text)
#     inputs = tokenizer([s.text for s in doc.sents], truncation=True, padding=True, return_tensors="pt")
#     outputs = deobfuscator.generate(inputs.input_ids.to(DEVICE), max_length=300)
#     sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return " ".join([s.strip() for s in sents])


# MODEL_PATH = "/kaggle/input/essay-gec/deobfuscator-v1"
# DEVICE = "cuda:0"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# deobfuscator = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

# nlp = spacy.load("en_core_web_sm")


# In[7]:


trainx = pd.read_csv('/kaggle/input/daigt-v2-trainessays-cleaned/daigt-v2-train-cleaned.csv')




# In[8]:


## credit @nbroad
not_persuade_df = trainx[trainx['source'] != 'persuade_corpus']
persuade_df = trainx[trainx['source'] == 'persuade_corpus']
sampled_persuade_df = persuade_df.sample(n=6000, random_state=42) 
all_human = set(list(''.join(sampled_persuade_df.text.to_list())))
other = set(list(''.join(not_persuade_df.text.to_list())))
chars_to_remove = ''.join([x for x in other if x not in all_human])
print(chars_to_remove)

translation_table = str.maketrans('', '', chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table)  

trainx['text'] = trainx['text'].str.replace('\n', '') 
trainx['text'] = trainx['text'].apply(remove_chars) 


# In[9]:


# train = pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv')
# train1 = pd.read_csv('/kaggle/input/llm-ai-detect-dataset-5/augmented-typos-introduced-ds5.csv')  # 15% typos on label==0 alone
# # train1 = pd.read_csv("/kaggle/input/llm-ai-detect-dataset-with-typos-2/daigt-v2-test-dataset-with-12%-typos-introduced.csv") #12% typos on both labels
train2 = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
train2.rename(columns={'generated':'label'},inplace=True)  
trainz = pd.concat([trainx,train2])

trainz['text'] = trainz['text'].str.replace('\n', '')  
train1 = trainz.copy()
train1.label.value_counts() 


# In[10]:


if debug:
#     train1 = train.copy() #sample(100)
    test = train1[train1.prompt_name=='Car-free cities']
    train = train1.drop(test.index)
else:
    train = train1.copy()
    test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
    test['text'] = test['text'].str.replace('\n', '') 
    test['text'] = test['text'].apply(remove_chars)  
    correct_df(test)


# In[ ]:





# In[11]:


# test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
# test['text'] = test['text'].str.replace('\n', '')  
# correct_df(test)


# In[12]:


# clean_texts = []
# for i, r in tqdm(test.iterrows(), total=len(test)):
#     clean_texts.append(clean_essay(r.text))
# test["text"] = clean_texts

# del deobfuscator
# torch.cuda.empty_cache()


# In[13]:


# trainx = train.groupby('label').apply(lambda x: x.sample(1000, random_state=42)).reset_index(drop=True)
# data = pd.DataFrame()
# data['text'] = pd.concat([test['text'],trainx['text']])


# In[14]:


# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))


# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens) 

# Creating huggingface dataset object
dataset = Dataset.from_pandas(test[['text']]) 

def train_corp_iter():
    """
    A generator function for iterating over a dataset in chunks.
    """    
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

# Training from iterator REMEMBER it's training on test set...
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)


tokenized_texts_test = []

# Tokenize test set with new tokenizer
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


# ### Functions to vectorize and fit models

# In[15]:


def dummy(text): 
    return text

# Fitting TfidfVectoizer on test set

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')
                            

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_ 

# Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

y_train = train['label'].values

del vectorizer
gc.collect()


# In[16]:


from scipy.sparse import hstack, coo_matrix 

# Add the additional column to the sparse matrix
X1 = hstack([tf_train, np.array(train.label).reshape(-1,1)],format='csr') #.tocsc()


# In[17]:


svm = SGDOneClassSVM(nu=0.1, random_state=42)
iso = IsolationForest(random_state=42,contamination=0.05)
lo = LocalOutlierFactor(n_neighbors=10, contamination=0.05)
# lo = EllipticEnvelope(random_state=42)   

pred = svm.fit_predict(X1) #sdf.drop(['label'],axis=1))
mask = pred != -1
data = X1[mask, :] #.reset_index(drop=True)
print("original number of rows ",X1.shape[0]) 
print('number of rows after outlier removal using SGDsvm',data.shape[0])

# pred = iso.fit_predict(data)
# mask = pred != -1
# data = data[mask, :] #.reset_index(drop=True) 
# print('number of rows after outlier removal using isoforest',data.shape[0])

# pred = lo.fit_predict(data)
# mask = pred != -1
# data = data[mask, :] #.reset_index(drop=True) 
# print('number of rows after outlier removal using localoutlier',data.shape[0])


# In[18]:


y_train = data[:,-1]
data = data[:, :-1]


# In[19]:


clf = MultinomialNB(alpha=0.02)
clf2 = MultinomialNB(alpha=0.01)
sgd_model = SGDClassifier(max_iter=10000, tol=1e-4, loss="modified_huber") 
p6={'n_iter': 3000,'verbose': -1,'objective': 'l2','learning_rate': 0.005670084478292278, 'colsample_bytree': 0.6440444070196796, 'colsample_bynode': 0.637635804565811, 'lambda_l1': 6.29090474401462, 'lambda_l2': 6.775341543233317, 'min_data_in_leaf': 95, 'max_depth': 26, 'max_bin': 630}


lgb=lgb.LGBMClassifier(**p6)

ensemble = VotingClassifier(estimators=[('mnb',clf),('sgd', sgd_model),('lgb',lgb)],
                            weights=[0.25,0.25,0.5], voting='soft', n_jobs=-1)
ensemble.fit(data, y_train.toarray())


# In[20]:


# mnb = MultinomialNB(alpha=0.02)
     
# sgd_model1 = SGDClassifier(max_iter=10000, tol=1e-4,loss="modified_huber", random_state=42) 
# sgd_model2 = SGDClassifier(max_iter=10000, tol=3e-3,loss="modified_huber",  class_weight="balanced",random_state=42) 
# sgd_model3 = SGDClassifier(max_iter=15000, tol=5e-3,loss="modified_huber", early_stopping=True,random_state=42) 

# ensemble = VotingClassifier(estimators=[
                                         
#                                         ('mnb', mnb),
#                                         ('sgd1', sgd_model1),
#                                         ('sgd2', sgd_model2),
#                                         ('sgd3', sgd_model3),
#                                        ],
#                             weights=[0.1, 0.3,0.3,0.3],
#                             voting='soft'
#                            )
 
# ensemble.fit(data, y_train.toarray())


# In[21]:


preds_test = ensemble.predict_proba(tf_test)[:,1]


# ## Submit

# In[22]:


if debug==False:
    pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)
    pd.read_csv('/kaggle/working/submission.csv')
    sub=pd.read_csv('/kaggle/working/submission.csv')
    print(sub)

