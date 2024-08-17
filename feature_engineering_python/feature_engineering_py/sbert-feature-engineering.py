#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import sys
sys.path.append('../input/sentence-transformers/sentence-transformers-master')
sys.path.append('../input/huggingface-bert-variants')
sys.path.append('../input/huggingface-roberta-variants')


# In[ ]:


import random
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas()
from collections import Counter
import statsmodels.api as sm
from scipy.stats import skew, iqr, kurtosis

#notebook formatting
#from rich.jupyter import print
#from rich.console import Console
#from rich.theme import Theme
#from rich import pretty

#visualization imports 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

#sklearn imports
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#nlp imports 
import re
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import spacy
from textblob import TextBlob


# In[ ]:


import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer, models


# In[ ]:





# In[ ]:


nlp_eng_emb = spacy.load("en_core_web_lg")


# In[ ]:


##seed everything
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed = 42
seed_everything(seed)


# In[ ]:


#define your own rmse and set greater_is_better=False
def rmse_custom(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

rmse = make_scorer(rmse_custom, greater_is_better=False)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
print(f"Shape df train: {df_train.shape}")
df_test = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
test_id = df_test[['id']]
print(f"Shape df test: {df_test.shape}")


# In[ ]:


#drop columns that we are not going to use in training:
columns_drop = ['url_legal', 'license', 'id', 'standard_error']
for df in [df_train, df_test]:
    for col in columns_drop:
        try: df = df.drop(col, axis=1)
        except: pass


# In[ ]:


def get_pos(excerpt):
    '''Returns number of nouns, adj and verbs in the text'''
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    adjectives = ['JJ', 'JJR', 'JJS']
    verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
    #first tokenize in words
    text_tokens = word_tokenize(excerpt)
    pos_lst = pos_tag(text_tokens)
    num_nouns = len([noun[0] for noun in pos_lst if noun[1] in nouns])
    num_adj = len([adj[0] for adj in pos_lst if adj[1] in adjectives])
    num_verbs = len([verb[0] for verb in pos_lst if verb[1] in verbs])
    
    return [num_nouns, num_adj, num_verbs]


# In[ ]:


#get polarity and subjecivity 

text_blob_obj = TextBlob('I hate football')


# In[ ]:


text_blob_obj.sentiment


# In[ ]:


text_blob_obj.sentiment.polarity


# In[ ]:


text_blob_obj.sentiment.subjectivity


# In[ ]:


#First let´s generate new features in the train and test set dataset

for df in [df_train, df_test]:
    print(f" ====== Generating Counting features..... =======")
    df['excerpt_length'] = df['excerpt'].apply(lambda x: len(x))
    df['excerpt_num_words'] = df['excerpt'].apply(lambda x: len(word_tokenize(x)))
    df['excerpt_num_sentences'] = df['excerpt'].apply(lambda x: len(sent_tokenize(x)))
    df['count_exclamation_mark'] = df['excerpt'].apply(lambda x: x.count('!'))
    df['count_question_mark'] = df['excerpt'].apply(lambda x: x.count('?'))
    df['count_punctuation'] =  df['excerpt'].apply(lambda x: sum([x.count(punct) for punct in '.,;:']))

    #POS features
    print(f" ====== Generating POS features..... =======")
    df['num_nouns'], df['num_adj'], df['num_verbs'] = zip(*df['excerpt'].apply(lambda x: get_pos(x)))
    # proportion of nouns, adj and verbs with respect to the total number of words
    df['nouns_proportion'] = df['num_nouns'] / df['excerpt_num_words']
    df['adj_proportion'] = df['num_adj'] / df['excerpt_num_words']
    df['verbs_proportion'] = df['num_verbs'] / df['excerpt_num_words']
    
    #Text Blob features: polarity and subjectivity
    print(f" ====== Generating Text Blob features..... =======")
    df['polarity'] = df['excerpt'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['excerpt'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    #More additional features
    print(f" ====== Generating Additional Features.... =======")
    df['num_words_capital'] = df['excerpt'].apply(lambda x: len([word for word in x.split() if word.istitle()]))
    #average lenght of tokens (words) and sentences in each excerpt
    df['avg_len_words'] = df['excerpt'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    df['avg_len_sentences'] = df['excerpt'].apply(lambda x: np.mean([len(sent) for sent in sent_tokenize(x)]))


# In[ ]:


def clean_text(excerpt, remove_stopwords=False, lemmatizer=False):
    #remove punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    excerpt = re.sub(f"[{string.punctuation}]", '', excerpt)
    weird_characters = ['\n', '—', '”', '“', '\xad', '…', '½', '¼', 
                            'æ', '°', '±', '·', '‘', '´', '–', '÷']
    for character in weird_characters:
        excerpt = excerpt.replace(character, "")
    #remove numbers
    excerpt = re.sub('[0-9]', '', excerpt)
    text_token = word_tokenize(excerpt)
    
    clean_words_list = [w for w in text_token if len(w) > 2]
    
    #remove stopwords
    if remove_stopwords:
        stop_words = [w for w in stopwords.words('english')]
        clean_words_list = [w for w in clean_words_list if w not in set(stop_words) and len(w) > 2]
    
    ##TODO: add lemmatizer##
    if lemmatizer:
        lemmatizer = WordNetLemmatizer()
        clean_words_list = [lemmatizer.lemmatize(word) for word in clean_words_list]
        lemmatizer_snow_ball = SnowballStemmer("english")
        clean_words_list = [lemmatizer_snow_ball.stem(word) for word in clean_words_list]
        
    clean_text = ' '.join(clean_words_list)

    return clean_text.lower()


# In[ ]:


def clean_text_sbert(excerpt, remove_stopwords=False, lemmatizer=False):
    #remove punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    remove_punctuation = '"#$%&\'()*+,-/<=>?@[\\]^_`{|}~'
    excerpt = re.sub(f"[{remove_punctuation}]", '', excerpt)
    excerpt = excerpt.lower()
    weird_characters = ['\n', '—', '”', '“', '\xad', '…', '½', '¼', 
                            'æ', '°', '±', '·', '‘', '´', '–', '÷']
    for character in weird_characters:
        excerpt = excerpt.replace(character, "")
    #remove numbers
    excerpt = re.sub('[0-9]', '', excerpt)
    
    #clean_words_list = [w for w in text_token if len(w) > 2]
    
    #clean_text = ' '.join(clean_words_list)

    return excerpt


# In[ ]:


# df_train['excerpt'] = df_train['excerpt'].apply(lambda x: clean_text_sbert(x, remove_stopwords=False, lemmatizer=False))
# df_test['excerpt'] = df_test['excerpt'].apply(lambda x: clean_text_sbert(x, remove_stopwords=False, lemmatizer=False))


# In[ ]:


ner_entities_lst = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                    'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

for ent in ner_entities_lst:
    df_train['count_' + ent] = 0.0
    df_test['count_' + ent] = 0.0


# In[ ]:


def get_ner_tags(excerpt):
    entities = []
    doc = nlp_eng_emb(excerpt)
    for ent in doc.ents:
        entities.append(ent.label_)
    return dict(Counter(entities))


# In[ ]:


df_train['ner'] = df_train['excerpt'].apply(get_ner_tags)
df_test['ner'] = df_test['excerpt'].apply(get_ner_tags)


# In[ ]:


for df in [df_train, df_test]:
    for index, row in tqdm(df.iterrows()):
        for key, value in row['ner'].items():
            df.loc[index, 'count_' + key] = value
    del df['ner']    


# In[ ]:





# In[ ]:





# ## 2. Modeling

# In[ ]:


##SELECT FETAURES 

features_training = []

col_vec_dim = [i for i in range(256)] #dimensions of the vector
features_training.extend(col_vec_dim)

features_eda = ['excerpt_length', 'excerpt_num_words', 'excerpt_num_sentences','num_nouns', 'num_adj', 
                   'num_verbs', 'nouns_proportion','adj_proportion', 'verbs_proportion', 'num_words_capital',
                   'count_exclamation_mark', 'count_question_mark', 'count_punctuation', 'avg_len_words', 'avg_len_sentences']
features_training.extend(features_eda)

features_training.extend(['polarity', 'subjectivity'])

features_ner = ['count_' + ent for ent in ner_entities_lst] #features NER
features_training.extend(features_ner)


# In[ ]:


from torch import nn


# In[ ]:


#BASE_MODEL = '/kaggle/input/huggingface-bert-variants/bert-base-uncased/bert-base-uncased'
BASE_MODEL = '/kaggle/input/huggingface-roberta-variants/roberta-base/roberta-base'
word_embedding_model = models.Transformer(BASE_MODEL)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                           out_features=256, 
                           activation_function=nn.Tanh())
model_sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])


# In[ ]:


df = pd.DataFrame(columns=[i for i in range(256)])
for excerpt in tqdm(list(df_train['excerpt'])):
#     vec = nlp_eng_emb(excerpt).vector
    vec =  model_sbert.encode(excerpt)
    vec_series = pd.Series(list(vec), index=df.columns)
    df = df.append(vec_series, ignore_index=True)
    
    
train = pd.concat([df_train, df], axis=1)
train.head()


# In[ ]:


test = pd.DataFrame(columns=[i for i in range(256)])
for excerpt in tqdm(list(df_test['excerpt'])):
#     vec = nlp_eng_emb(excerpt).vector
    vec =  model_sbert.encode(excerpt)
    vec_series = pd.Series(list(vec), index=df.columns)
    test = test.append(vec_series, ignore_index=True)

df_pred = pd.concat([df_test, test], axis=1)
columns_drop = ['url_legal', 'license', 'id', 'excerpt']
for col in columns_drop:
    df_pred = df_pred.drop(col, axis=1)
df_pred.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, train['target'], test_size=0.01,
                                                   random_state=seed)

print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_test: {X_test.shape}")

X_train = train[features_training]
y_train = train['target']
#filter X_train by columns training


# In[ ]:


pipe_spacy = Pipeline([('scaler', MinMaxScaler()),
                        ('bayesian_ridge', BayesianRidge())]
                        )

param_grid_spacy = {

}

model_spacy = GridSearchCV(estimator=pipe_spacy,
                                param_grid=param_grid_spacy,
                                scoring=rmse,
                                cv=10,
                                verbose=3)


model_spacy.fit(X_train, y_train)
print(f'Best params are : {model_spacy.best_params_}')
print(f'Best training score: {round(model_spacy.best_score_, 5)}')

#y_pred = model_spacy.predict(X_test[features_training])
#print(f"RMSE baseline with testing set: {round(rmse_custom(y_test, y_pred), 5)}")


# ### XGBOOST REGRESSOR
# 
# - gamma: The default is 0. Values of less than 10 are standard. Increasing the value prevents overfitting.
# - reg_alpha: L1 regularization on leaf weights. Larger values mean more regularization and prevent overfitting. The default is 0.
# - reg_lambda: L2 regularization on leaf weights. Increasing the value prevents overfitting. The default is 1
# - booster: gbtree

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_regressor = xgb.XGBRegressor(booster='gbtree', 
                                reg_lambda=10,
                                max_depth=4)


# In[ ]:


xgb_regressor.fit(X_train, y_train)


# In[ ]:


y_train


# In[ ]:


mean_squared_error(np.array(y_train), xgb_regressor.predict(X_train), squared=False)


# In[ ]:


# pipe_spacy = Pipeline([('scaler', MinMaxScaler()),
#                         ('svr', SVR())]
#                         )

# param_grid_spacy = {
#     'svr__C': [1, 2.2, 4, 8],
#     'svr__gamma': [0.1, 0.08, 0.01],
#     'svr__kernel': ['rbf'], 
#     'svr__epsilon': [1, 0.1, 0.01]

# }

# model_spacy = GridSearchCV(estimator=pipe_spacy,
#                                 param_grid=param_grid_spacy,
#                                 scoring=rmse,
#                                 cv=10,
#                                 verbose=3)


# model_spacy.fit(X_train[features_training], y_train)
# print(f'Best params are : {model_spacy.best_params_}')
# print(f'Best training score: {round(model_spacy.best_score_, 5)}')

#y_pred = model_spacy.predict(X_test[features_training])
#print(f"RMSE baseline with testing set: {round(rmse_custom(y_test, y_pred), 5)}")


# ### CROSS VALIDATION STRATEGY
# 
# - out of fold (oof) score based on predictions made by data not used to train a model, using the validations folds.
# - Traget follows a normal distribution, we are going to stratify the training dataset using that column

# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


#create groups/bins in the target 
train['target_bin'] = pd.cut(train['target'], bins=10, labels=[i + 1 for i in range(10)])


# In[ ]:


skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


# In[ ]:


X_train = train[features_training]
y_train = train['target']


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler


# In[ ]:


predictions = []
training_scores = []

df_out_of_fold = train.copy()
df_out_of_fold['out_of_fold'] = 0

for fold, (train_idx, val_idx) in enumerate(list(skfold.split(X=X_train, y=train['target_bin']))):
    print(f"\n Training FOLD: {fold + 1} / {10}")
    
    #bayesian_ridge = BayesianRidge()
    pipe_br = make_pipeline(MinMaxScaler(), BayesianRidge())
    pipe_br.fit(X_train.loc[train_idx], y_train.loc[train_idx])
    train_rmse = mean_squared_error(y_train.loc[train_idx], pipe_br.predict(X_train.loc[train_idx]), squared=False)
    training_scores.append(train_rmse)
    print(f'Fold {fold + 1}: Training score: {round(train_rmse, 4)}')
    
    #precit traget for each fold >> submission values
    #print(pipe_br.predict(df_pred))
    predictions.append(pipe_br.predict(df_pred[features_training]))
    #now let´s predict the results with the validation (not used for training) set of each fold
    pred_oof = pipe_br.predict(X_train.loc[val_idx])
    df_out_of_fold['out_of_fold'].iloc[val_idx] += pred_oof

print(f'Training score: {round(np.mean(training_scores), 4)}, Training STD: {round(np.std(training_scores), 4)}')
print(f'Oout of fold score across folds: {round(mean_squared_error(df_out_of_fold.target, df_out_of_fold.out_of_fold, squared=False), 5)}')


# ### INFERIENCE

# In[ ]:


predictions = xgb_regressor.predict(df_pred)


# In[ ]:


test_id['target'] = np.mean(predictions, axis=0)
df_submission = test_id[['id', 'target']]
df_submission.to_csv('submission.csv', index=False)


# In[ ]:


df_submission


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




