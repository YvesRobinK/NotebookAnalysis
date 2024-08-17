#!/usr/bin/env python
# coding: utf-8

# # Simple blending by averaging the outputs of LightGBM, CatBoost, and SVR models
# 
# #### Credits
# 
# In this notebook, I partially used the code from these notebooks:
# 
# - [Towards TF-IDF in logs features](https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features)
# - [Feature Engineering: Sentence & paragraph features](https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features)
# - [LGBM and NN on sentences](https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences)

# # Import libraries

# In[1]:


import re
import copy
import warnings
import string
import sklearn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost as cb

from tqdm import tqdm
from scipy.stats import kurtosis
from collections import Counter
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# # Set up the environment

# In[2]:


# Turn off warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)

# Matplotlib settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn settings
sns.set_style('whitegrid')
sns.set_palette('pastel')
sns.set_context('notebook')


# # Set up the paths and global variables

# In[2]:


# Set to True if the notebook is running in Kaggle environment
is_in_kaggle = True

# Set to True if the notebook is running in development environment
development_mode = False

# File paths
if is_in_kaggle:
    data_dir = Path('/kaggle/input/linking-writing-processes-to-writing-quality')
    raw_data_dir = data_dir
    processed_data_dir = data_dir / 'processed_data'
    output_dir = Path('./')
else:
    data_dir = Path('../../data')
    raw_data_dir = data_dir / 'competition_data'
    processed_data_dir = data_dir / 'processed_data'
    output_dir = processed_data_dir
    
# Number of outer_folds
num_outer_folds = 1

# Number of folds for cross validation
num_folds = 10

# Number of samples to load from the train data (set to None to load all samples)
num_samples = 5e5

# Global seed for reproducibility
global_seed = 42

# Model with scaled features
model_with_scaled_features = ['svr', 'knn']

# Blending weights
blending_weights = {
    'lgbm': 0.4,
    'catboost': 0.4,
    'svr': 0.2,
    'knn': 0.0,
}

assert sum(blending_weights.values()) == 1.0, "Blending weights must sum to 1.0!"


# # Load the data

# In[3]:


# Load the train_logs.csv file
if is_in_kaggle:
    if development_mode:
        train_logs = pd.read_csv(raw_data_dir / 'train_logs.csv', nrows=num_samples)
    else:
        train_logs = pd.read_csv(raw_data_dir / 'train_logs.csv')
else:
    train_logs = pd.read_csv(raw_data_dir / 'train_logs.csv', nrows=num_samples)

# Outlier essays
#outliers = ['21bbc3f6']

# Remove outliers
#train_logs = train_logs[~train_logs['id'].isin(outliers)].reset_index(drop=True)

# Load the test_logs.csv, train_scores.csv, and sample_submission.csv files
train_scores = pd.read_csv(raw_data_dir / 'train_scores.csv')
test_logs = pd.read_csv(raw_data_dir / 'test_logs.csv')
sample_submission = pd.read_csv(raw_data_dir / 'sample_submission.csv')


# In[4]:


# Have a look at the shape of the train_logs dataframe and number of loaded samples
print(f"Train Logs Shape: {train_logs.shape}")
print(f"Number of essays: {train_logs['id'].nunique()} out of 2471 ({train_logs['id'].nunique()/2471*100:.2f}%)")


# In[5]:


if is_in_kaggle:
    extra_data_dir = Path('/kaggle/input/writing-quality-challenge-constructed-essays')
else:
    extra_data_dir = Path('../../data/extra_data')

train_essays = pd.read_csv(extra_data_dir / 'train_essays_fast.csv')

train_essays.head(10)


# # Feature engineering

# In[6]:


def processingInputs(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        # If activity = Replace
        if Input[0] == 'Replace':
            # splits text_change at ' => '
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue

        # If activity = Paste    
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue

        # If activity = Remove/Cut
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue

        # If activity = Move...
        if "M" in Input[0]:
            # Gets rid of the "Move from to" text
            croppedTxt = Input[0][10:]              
            # Splits cropped text by ' To '
            splitTxt = croppedTxt.split(' To ')              
            # Splits split text again by ', ' for each item
            valueArr = [item.split(', ') for item in splitTxt]              
            # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            # Skip if someone manages to activiate this by moving to same place
            if moveData[0] != moveData[2]:
                # Check if they move text forward in essay (they are different)
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] +\
                    essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] +\
                    essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue                

        # If activity = input
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def getEssays(df):
    # Copy required columns
    textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
    # Get rid of text inputs that make no change
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']     
    # construct essay, fast 
    tqdm.pandas()
    essay=textInputDf.groupby('id')[['activity','cursor_position', 'text_change']].progress_apply(
        lambda x: processingInputs(x))      
    # to dataframe
    essayFrame=essay.to_frame().reset_index()
    essayFrame.columns=['id','essay']
    # Returns the essay series
    return essayFrame


# In[7]:


# Helper functions for feature engineering
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)


# In[8]:


AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', kurtosis, 'sum']

def split_essays_into_words(df):
    essay_df = df
    essay_df['word'] = essay_df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    essay_df = essay_df.explode('word')
    essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
    essay_df = essay_df[essay_df['word_len'] != 0]
    return essay_df

def compute_word_aggregations(word_df):
    word_agg_df = word_df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df

def split_essays_into_sentences(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
    return essay_df

def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index

    # New features intoduced here: https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline-v2
    for sent_l in [50, 60, 75, 100]:
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = df[df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
    
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def split_essays_into_paragraphs(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
    essay_df = essay_df.explode('paragraph')
    # Number of characters in paragraphs
    essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.paragraph_len!=0].reset_index(drop=True)
    return essay_df

def compute_paragraph_aggregations(df):
    paragraph_agg_df = pd.concat(
        [df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    ) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df


# In[9]:


# Word features for train dataset
train_word_df = split_essays_into_words(train_essays)
train_word_agg_df = compute_word_aggregations(train_word_df)
plt.figure(figsize=(15, 1.5))
plt.boxplot(x=train_word_df.word_len, vert=False, labels=['Word length'])
plt.show()


# In[10]:


# Sentence features for train dataset
train_sent_df = split_essays_into_sentences(train_essays)
train_sent_agg_df = compute_sentence_aggregations(train_sent_df)
plt.figure(figsize=(15, 1.5))
plt.boxplot(x=train_sent_df.sent_len, vert=False, labels=['Sentence length'])
plt.show()


# In[11]:


# Paragraph features for train dataset
train_paragraph_df = split_essays_into_paragraphs(train_essays)
train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)
plt.figure(figsize=(15, 1.5))
plt.boxplot(x=train_paragraph_df.paragraph_len, vert=False, labels=['Paragraph length'])
plt.show()


# In[12]:


# Features for test dataset
test_essays = getEssays(test_logs)
test_word_agg_df = compute_word_aggregations(split_essays_into_words(test_essays))
test_sent_agg_df = compute_sentence_aggregations(split_essays_into_sentences(test_essays))
test_paragraph_agg_df = compute_paragraph_aggregations(split_essays_into_paragraphs(test_essays))


# In[13]:


# Keeping the states of these objects global to reuse them in the test data
count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()

# Define the n-gram range
ngram_range = (1, 2)  # Example: Use unigrams and bigrams
count_vect_ngram = CountVectorizer(ngram_range=ngram_range)
tfidf_vect_ngram = TfidfVectorizer(ngram_range=ngram_range)


# In[14]:


def make_text_features(df, name="Train Logs"):

        def count_encoding_ngram(essays_as_string, name=name):
            """Applies Count Encoding to the essay data and returns a DataFrame with prefixed column names."""

            if name == "Train Logs":
                features = count_vect_ngram.fit_transform(essays_as_string)
            else:
                features = count_vect_ngram.transform(essays_as_string)

            feature_names = [f'bow-{ngram}' for ngram in count_vect_ngram.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)

        def tfidf_encoding_ngram(essays_as_string, name=name):
            """Applies TF-IDF Encoding to the essay data and returns a DataFrame with prefixed column names."""

            if name == "Train Logs":
                features = tfidf_vect_ngram.fit_transform(essays_as_string)
            else:
                features = tfidf_vect_ngram.transform(essays_as_string)

            feature_names = [f'tfidf-{ngram}' for ngram in tfidf_vect_ngram.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)


        def count_encoding(essays_as_string, name=name):
            """Applies Count Encoding to the essay data and returns a DataFrame with prefixed column names."""
            
            if name == "Train Logs":
                features = count_vect.fit_transform(essays_as_string)
            else:
                features = count_vect.transform(essays_as_string)
                
            feature_names = [f'bow_{name}' for name in count_vect.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)

        def tfidf_encoding(essays_as_string, name=name):
            """Applies TF-IDF Encoding to the essay data and returns a DataFrame with prefixed column names."""
            
            if name == "Train Logs":
                features = tfidf_vect.fit_transform(essays_as_string)
            else:
                features = tfidf_vect.transform(essays_as_string)
            
            feature_names = [f'tfidf_{name}' for name in tfidf_vect.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)
        
        def custom_feature_engineering(essays):
            """Example custom feature: calculates the length of each essay with prefixed column name."""
            
            custom_text_features = {
                'custom_length': [len(essay) for essay in essays],
                'custom_word_count': [len(essay.split()) for essay in essays],
                'custom_unique_word_count': [len(set(essay.split())) for essay in essays],
                'custom_punctuation_count': [sum([1 for char in essay if char in string.punctuation]) for essay in essays],
                'custom_paragraph_count': [essay.count('\n') + 1 for essay in essays],
                'custom_sentence_count': [essay.count('.') + 1 for essay in essays],
                'custom_comma_count': [essay.count(',') for essay in essays],
                'custom_question_mark_count': [essay.count('?') for essay in essays],
                'custom_exclamation_mark_count': [essay.count('!') for essay in essays],
                'custom_colon_count': [essay.count(':') for essay in essays],
                'custom_semicolon_count': [essay.count(';') for essay in essays],
                'custom_dash_count': [essay.count('-') for essay in essays],
                'custom_quote_count': [essay.count('"') for essay in essays],
                'custom_apostrophe_count': [essay.count("'") for essay in essays],
                'custom_parenthesis_count': [essay.count('(') + essay.count(')') for essay in essays],
                'custom_bracket_count': [essay.count('[') + essay.count(']') for essay in essays],
                'custom_brace_count': [essay.count('{') + essay.count('}') for essay in essays],
                'custom_mathematical_symbol_count': [essay.count('+') + essay.count('-') + essay.count('*') + essay.count('/') for essay in essays],
                'custom_digit_count': [sum([1 for char in essay if char.isdigit()]) for essay in essays],
                'custom_average_word_length': [np.mean([len(word) for word in essay.split()]) for essay in essays],
                'custom_average_sentence_length': [np.mean([len(sentence) for sentence in essay.split('.')]) for essay in essays],
                'custom_average_paragraph_length': [np.mean([len(paragraph) for paragraph in essay.split('\n')]) for essay in essays],
                'custom_average_word_count_per_sentence': [np.mean([len(sentence.split()) for sentence in essay.split('.')]) for essay in essays],
                'custom_average_word_count_per_paragraph': [np.mean([len(paragraph.split()) for paragraph in essay.split('\n')]) for essay in essays],
                'custom_average_sentence_count_per_paragraph': [np.mean([paragraph.count('.') + 1 for paragraph in essay.split('\n')]) for essay in essays],
                'custom_average_punctuation_count_per_sentence': [np.mean([sum([1 for char in sentence if char in string.punctuation]) for sentence in essay.split('.')]) for essay in essays],
                'custom_average_punctuation_count_per_paragraph': [np.mean([sum([1 for char in paragraph if char in string.punctuation]) for paragraph in essay.split('\n')]) for essay in essays],
                'custom_average_digit_count_per_sentence': [np.mean([sum([1 for char in sentence if char.isdigit()]) for sentence in essay.split('.')]) for essay in essays],
                'custom_average_digit_count_per_paragraph': [np.mean([sum([1 for char in paragraph if char.isdigit()]) for paragraph in essay.split('\n')]) for essay in essays],
                'custom_average_mathematical_symbol_count_per_sentence': [np.mean([sum([1 for char in sentence if char in ['+', '-', '*', '/']]) for sentence in essay.split('.')]) for essay in essays],
                'custom_average_mathematical_symbol_count_per_paragraph': [np.mean([sum([1 for char in paragraph if char in ['+', '-', '*', '/']]) for paragraph in essay.split('\n')]) for essay in essays],
                                    }
            return pd.DataFrame(custom_text_features)

        
        def merge_features(data, name):
            """Merges features from different methods into one DataFrame with the id column."""
            essays_as_string = data['essay']

            # Extract features
            bow_df = count_encoding(essays_as_string, name)
            tfidf_df = tfidf_encoding(essays_as_string, name)
            custom_features_df = custom_feature_engineering(data['essay'])
            
            count_encoding_ngram_features = count_encoding_ngram(essays_as_string, name=name)
            #tfidf_encoding_ngram_features = tfidf_encoding_ngram(essays_as_string, name=name)
        
            # Merge all features
            merged_features = pd.concat([data[['id']], bow_df, tfidf_df, custom_features_df, 
                                         #count_encoding_ngram_features, 
                                         #tfidf_encoding_ngram_features
                                        ], axis=1)
            return merged_features
        
        return merge_features(getEssays(df), name)


# In[15]:


class FeatureMaker:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        
        self.idf = defaultdict(float)
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf
            
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def make_feats(self, df, name="Train Logs"):
        
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis])
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']
        
        # print("Engineering text features")
        tmp_df = make_text_features(df, name)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        return feats


# In[16]:


feature_maker = FeatureMaker(seed=global_seed)

train_feats = feature_maker.make_feats(train_logs, name="Train Logs")
test_feats = feature_maker.make_feats(test_logs, name="Test Logs")

nan_cols = train_feats.columns[train_feats.isna().any()].tolist()

train_feats = train_feats.drop(columns=nan_cols)
test_feats = test_feats.drop(columns=nan_cols)


# In[17]:


train_agg_fe_df = train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
train_agg_fe_df.reset_index(inplace=True)

test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)

train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')


# In[18]:


data = []

for logs in [train_logs, test_logs]:
    logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
    logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

    group = logs.groupby('id')['time_diff']
    largest_lantency = group.max()
    smallest_lantency = group.min()
    median_lantency = group.median()
    initial_pause = logs.groupby('id')['down_time'].first() / 1000
    pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
    pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
    pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
    pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
    pauses_3_sec = group.apply(lambda x: (x > 3).sum())

    data.append(pd.DataFrame({
        'id': logs['id'].unique(),
        'largest_lantency': largest_lantency,
        'smallest_lantency': smallest_lantency,
        'median_lantency': median_lantency,
        'initial_pause': initial_pause,
        'pauses_half_sec': pauses_half_sec,
        'pauses_1_sec': pauses_1_sec,
        'pauses_1_half_sec': pauses_1_half_sec,
        'pauses_2_sec': pauses_2_sec,
        'pauses_3_sec': pauses_3_sec,
    }).reset_index(drop=True))

train_eD592674, test_eD592674 = data

train_feats = train_feats.merge(train_eD592674, on='id', how='left')
test_feats = test_feats.merge(test_eD592674, on='id', how='left')
train_feats = train_feats.merge(train_scores, on='id', how='left')


# In[19]:


# Adding the additional features to the original feature set

train_feats = train_feats.merge(train_word_agg_df, on='id', how='left')
train_feats = train_feats.merge(train_sent_agg_df, on='id', how='left')
train_feats = train_feats.merge(train_paragraph_agg_df, on='id', how='left')
train_feats.head(10)


# In[20]:


test_feats = test_feats.merge(test_word_agg_df, on='id', how='left')
test_feats = test_feats.merge(test_sent_agg_df, on='id', how='left')
test_feats = test_feats.merge(test_paragraph_agg_df, on='id', how='left')
test_feats.head(10)


# In[21]:


# Columns in train data but not in test data
set(train_feats.columns) - set(test_feats.columns)


# In[22]:


target_col = ['score']
drop_cols = ['id']
train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]


# In[23]:


best_features = ['id'] + train_cols
created_features = train_feats


# # Model definition

# In[24]:


def make_model():
    
    params = {'reg_alpha': 0.007678095440286993, 
               'reg_lambda': 0.34230534302168353, 
               'colsample_bytree': 0.627061253588415, 
               'subsample': 0.854942238828458, 
               'learning_rate': 0.038697981947473245, 
               'num_leaves': 22, 
               'max_depth': 37, 
               'min_child_samples': 18,
               'random_state': global_seed,
               'n_estimators': 150,
               "objective": "regression",
               "metric": "rmse",
               "verbosity": 0,
              }
    
#     params = {'learning_rate': 0.04740502374845092, 'num_leaves': 254, 'max_depth': 4, 'min_child_samples': 82, 'max_bin': 608, 
#      'subsample': 0.71388444782468, 'subsample_freq': 10, 'colsample_bytree': 0.8456486588339323, 
#      'min_child_weight': 0.0013130497070224434, 'subsample_for_bin': 489817, 'reg_alpha': 0.0016928120651709126, 
#      'reg_lambda': 0.018541096369691805,
#      'random_state': global_seed,
#      'n_estimators': 207,
#      "objective": "regression",
#      "metric": "rmse",
#      "verbosity": 0,
#              }
    
    model1 = lgb.LGBMRegressor(**params)
    
    model2 = cb.CatBoostRegressor(iterations=1000,
                                 learning_rate=0.1,
                                 depth=6,
                                 eval_metric='RMSE',
                                 random_seed = global_seed,
                                 bagging_temperature = 0.2,
                                 od_type='Iter',
                                 metric_period = 50,
                                 od_wait=20,
                                 verbose=False)
    
    model3 = sklearn.svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    model4 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=25, 
                                                    weights='distance', 
                                                    algorithm='auto', 
                                                    leaf_size=20, 
                                                    p=1, 
                                                    metric='minkowski', 
                                                    metric_params=None, 
                                                    n_jobs=-1)
    
    models = []
    
    if 'lgbm' in blending_weights and blending_weights['lgbm'] > 0:
        models.append((model1, 'lgbm'))
    if 'catboost' in blending_weights and blending_weights['catboost'] > 0:
        models.append((model2, 'catboost'))
    if 'svr' in blending_weights and blending_weights['svr'] > 0:
        models.append((model3, 'svr'))
    if 'knn' in blending_weights and blending_weights['knn'] > 0:
        models.append((model4, 'knn'))
    
    return models


# # Train and evaluate the models

# In[25]:


X_y = pd.merge(created_features[best_features], train_scores, on='id', how='left')
X_y['label'] = X_y['score'].apply(lambda x: str(x))

# Replace inf and -inf with nan
X_y.replace([np.inf, -np.inf], np.nan, inplace=True)

features = X_y.iloc[:,1:-2]
target = X_y.iloc[:,-2]
label = X_y.iloc[:,-1]

models_and_errors_dict = {}

for out_fold in range(1, num_outer_folds+1):
    
    print(f'\n--- Out-fold #{out_fold} ---\n')
    
    # Use StratifiedKFold
    #skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=global_seed + out_fold)
    skf = KFold(n_splits=num_folds, shuffle=True, random_state=global_seed + out_fold*10)
    
    # using (stratified) k-fold cross validation to train and evaluate the model
    #for fold, indexes in enumerate(skf.split(features, label), start=1):
    for fold, indexes in enumerate(skf.split(features), start=1):
        
        # Get train and test indexes
        train_index, test_index = indexes
        
        print(f'--- Fold #{fold} ---')       
        
        # Split data into train and test sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        X_train_copy, X_test_copy = X_train.copy(), X_test.copy()
        
        for model, model_type in make_model():
            
            print(f'Training a {model_type} model on fold {fold} in out-fold {out_fold}')
            
            if model_type in model_with_scaled_features:
                # Impute the NaN values
                imputer = SimpleImputer(strategy='mean')  # or median, most_frequent, etc.
                X_train_imputed = imputer.fit_transform(X_train.copy())
                X_test_imputed = imputer.transform(X_test.copy())
                
                # Create the scaler with the desired range
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_test_scaled = scaler.transform(X_test_imputed)
                
                # Drop columns with missing values in the training data
                X_train_copy = X_train_scaled
                X_test_copy = X_test_scaled
            
            if model_type == 'lgb':
                #X_valid, y_valid = X_test, y_test
                early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
                verbose_callback = lgb.log_evaluation(100)
                
                model.fit(X_train_copy, y_train, eval_set=[(X_test_copy, y_test)],  
                          callbacks=[early_stopping_callback, verbose_callback],)
            else:
                model.fit(X_train_copy, y_train)
        
            # Evaluate model
            y_hat = model.predict(X_test_copy)
        
            # Root Mean Squared Error (RMSE)
            rmse = sklearn.metrics.mean_squared_error(y_test, y_hat, squared=False)
            print(f'RMSE: {rmse} on fold {fold}')
            
            if model_type not in models_and_errors_dict:
                models_and_errors_dict[model_type] = []
            
            if model_type in model_with_scaled_features:
                #print(out_fold, fold, model_type, X_train.shape, X_test.shape, X_train_imputed.shape, X_train_copy.shape,
                #      X_test_imputed.shape, X_test_copy.shape)
                models_and_errors_dict[model_type].append((model, rmse, imputer, scaler))
            else:
                models_and_errors_dict[model_type].append((model, rmse, None, None))            


# In[26]:


for model_type, models_info in models_and_errors_dict.items():
    print(f'\n--- {model_type} ---\n')
    print(f'{model_type}: {len(models_info)} models')

    print(f"Number of features: {len(best_features)}")
    print(f"Number of folds: {num_folds}")
    print(f"Number of out folds: {num_outer_folds}")

    # Extracting RMSEs
    rmses = [rmse for _, rmse, _, _ in models_info]

    # Average and standard deviation of RMSE
    mean_rmse = np.round(np.mean(rmses), 5)
    std_rmse = np.round(np.std(rmses), 5)
    print(f'Mean RMSE: {mean_rmse} +/- {std_rmse} using {model_type} models')

    # Best and worst RMSE
    best_rmse = min(rmses)
    worst_rmse = max(rmses)
    print(f'Best RMSE: {best_rmse}')
    print(f'Worst RMSE: {worst_rmse}')
    
    print(50 * '-')


# # Make the predictions

# In[27]:


default_value = 3.7
y_hats = dict()

submission_df = pd.DataFrame(test_feats.copy()['id'])
submission_df['score'] = default_value

X_unseen = test_feats.copy()[best_features]
X_unseen.drop(columns=['id'], inplace=True)

# Replace NaN and infinite values
X_unseen.replace([np.inf, -np.inf], np.nan, inplace=True)

for model_name, model_info in models_and_errors_dict.items():
    print(f'\n--- {model_name} ---\n')
    
    # Deep copy to ensure X_unseen is not modified
    X_unseen_copy = X_unseen.copy()
    y_hats[model_name] = []

    try:
        for ix, (trained_model, error, imputer, scaler) in enumerate(model_info, start=1):
            print(f"Using model {ix} with error {error}")

            # Impute and scale data if necessary
            if model_name in model_with_scaled_features:
                #print(X_unseen_copy.shape)
                X_unseen_imputed = imputer.transform(X_unseen_copy)
                X_unseen_scaled = scaler.transform(X_unseen_imputed)
                y_hats[model_name].append(trained_model.predict(X_unseen_scaled))
            else:
                y_hats[model_name].append(trained_model.predict(X_unseen_copy))
    
    except Exception as err:
        print(f"An error occurred: {err}")
        y_hats[model_name] = [default_value] * len(model_info)
    
    else:
        print("No errors occurred.")
    finally:
        if y_hats[model_name]:
            y_hat_avg = np.mean(y_hats[model_name], axis=0)
            submission_df['score_' + model_name] = y_hat_avg
        else:
            print("No predictions to process.")
        print("Done.")


# # Make the final predictions and the submission file

# In[28]:


# Make the final submission file
blended_score = []
for k, v in blending_weights.items():
    if v > 0:
        if len(blended_score) == 0:
            blended_score = submission_df['score_' + k] * v
        else:
            blended_score += submission_df['score_' + k] * v
submission_df['score'] = blended_score


# In[29]:


submission_df[['id', 'score']].to_csv(output_dir / 'submission.csv', index=False)
submission_df[['id', 'score']].head(10)

