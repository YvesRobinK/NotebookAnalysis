#!/usr/bin/env python
# coding: utf-8

# # 1. Essay Constructor for Tokenization:
# #### One of the key components of my approach is the use of an essay constructor for tokenization. By leveraging this approach, I aimed to extract meaningful features from the textual data, allowing my model to capture intricate patterns within the essays. This technique not only facilitated efficient tokenization but also played a crucial role in enhancing the performance of my model on the given task.
# 
# # 2. Additional Features Shared Among Participants:
# #### In addition to the essay constructor, I incorporated a set of features that were commonly utilized by participants in the competition. These features, identified through rigorous exploration and collaboration within the Kaggle community, added valuable information to the model. The collective wisdom of the participants often leads to powerful feature engineering, and by incorporating these shared features, I aimed to tap into the collective intelligence of the competition.
# 
# # Model Architecture:
# #### I employed a LightGbm  for my predictions. The model was trained on a combination of the engineered features(Tokenization), including those derived from the essay constructor, and the additional features used by the Kaggle community.

# ## Credit
# - https://www.kaggle.com/code/daliimsadek/comeback

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import os
import itertools
import pickle

from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
from transformers import BertTokenizer
import warnings

import os
import gc
import re
import random
from collections import Counter, defaultdict
import pprint
import time
import copy


import seaborn as sns
from tqdm.autonotebook import tqdm

# from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, PowerTransformer, RobustScaler, FunctionTransformer
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna


# # Import Data

# In[ ]:


traindf = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
testdf = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')


# ## Essay Constructor

# In[ ]:


def getEssays(df):
    # Copy required columns
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change']].copy()
    
    # Get rid of text inputs that make no change
    # Note: Shift was unpreditcable so ignored
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']

    # Get how much each Id there is
    valCountsArr = textInputDf['id'].value_counts(sort=False).values

    # Holds the final index of the previous Id
    lastIndex = 0

    # Holds all the essays
    essaySeries = pd.Series()

    # Fills essay series with essays
    for index, valCount in enumerate(valCountsArr):

        # Indexes down_time at current Id
        currTextInput = textInputDf[['activity', 'cursor_position', 'text_change']].iloc[lastIndex : lastIndex + valCount]

        # Update the last index
        lastIndex += valCount

        # Where the essay content will be stored
        essayText = ""

        
        # Produces the essay
        for Input in currTextInput.values:
            
            # Input[0] = activity
            # Input[2] = cursor_position
            # Input[3] = text_change
            
            # If activity = Replace
            if Input[0] == 'Replace':
                # splits text_change at ' => '
                replaceTxt = Input[2].split(' => ')
                
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue

                
            # If activity = Paste    
            if Input[0] == 'Paste':
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

                
            # If activity = Remove/Cut
            if Input[0] == 'Remove/Cut':
                # DONT TOUCH
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
                        # DONT TOUCH
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                    else:
                        # DONT TOUCH
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                continue
                
                
            # If just input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]

            
        # Sets essay at index  
        essaySeries[index] = essayText
     
    
    # Sets essay series index to the ids
    essaySeries.index =  textInputDf['id'].unique()
    
    
    # Returns the essay series
    return essaySeries


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_essays = getEssays(traindf)\n')


# In[ ]:


test_essays = getEssays(testdf)


# In[ ]:


train_essays
test_essays


# In[ ]:


train_essaysdf = pd.DataFrame({'id': train_essays.index, 'essay': train_essays.values})
test_essaysdf = pd.DataFrame({'id': test_essays.index, 'essay': test_essays.values})


# In[ ]:


train_essaysdf.shape


# In[ ]:


merged_data = train_essaysdf.merge(train_scores, on='id')


# ## Using Tokenizer

# In[ ]:


count_vectorizer = CountVectorizer(ngram_range=(1, 2))
X_tokenizer_train = count_vectorizer.fit_transform(merged_data['essay'])
X_tokenizer_test = count_vectorizer.transform(test_essaysdf['essay'])
y = merged_data['score']


# In[ ]:


X_tokenizer_train.shape


# In[ ]:


X_tokenizer_test.shape


# In[ ]:


df_train = pd.DataFrame()
df_test = pd.DataFrame()


# In[ ]:


X_tokenizer_train = X_tokenizer_train.todense()
X_tokenizer_test = X_tokenizer_test.todense()


# ## Create Features from the tokenizer

# In[ ]:


for i in range(X_tokenizer_train.shape[1]) : 
    L = list(X_tokenizer_train[:,i])
    li = [int(x) for x in L ]
    df_train[f'feature {i}'] = li


# In[ ]:


df_train


# In[ ]:


for i in range(X_tokenizer_test.shape[1]) : 
    L = list(X_tokenizer_test[:,i])
    li = [int(x) for x in L ]
    df_test[f'feature {i}'] = li


# In[ ]:


df_train_index = train_essaysdf['id']
df_test_index = test_essaysdf['id']


# In[ ]:


print(df_train_index,df_test_index)


# In[ ]:


df_train.loc[:, 'id'] = df_train_index
df_test.loc[:, 'id'] = df_test_index


# In[ ]:


df_train.sum()


# In[ ]:


(df_test.sum() == 1).sum()


# In[ ]:


traindf


# # Feature Engineering

# In[ ]:


train_agg_fe_df = traindf.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
train_agg_fe_df.reset_index(inplace=True)



# In[ ]:


test_agg_fe_df = testdf.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)


# In[ ]:


from collections import defaultdict

class Preprocessor:
    
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
#         self.gaps = [1, 2]
    
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
    
    def make_feats(self, df):
        
        print("Starting to engineer features")
        
        # initialize features dataframe
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        # get shifted features
        # time shift
        print("Engineering time data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        # cursor position shift
        print("Engineering cursor position data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        # word count shift
        print("Engineering word count data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        # get aggregate statistical features
        print("Engineering statistical summaries for features")
        # [(feature name, [ stat summaries to add ])]
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt])
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

        # counts
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

        # input words
        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        # compare feats
        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']
        
        print("Done!")
        return feats


# In[ ]:


preprocessor = Preprocessor(seed=42)

print("Engineering features for training data")

other_train_feats = preprocessor.make_feats(traindf)

print()
print("-"*25)
print("Engineering features for test data")
print("-"*25)
other_test_feats = preprocessor.make_feats(testdf)


# In[ ]:


other_train_feats.shape


# In[ ]:


other_test_feats.shape


# In[ ]:


len(other_train_feats)


# In[ ]:


df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()


# In[ ]:


df_train_all = df_train.merge(train_agg_fe_df,on='id')


# In[ ]:


df_test_all


# In[ ]:


df_test_all = df_test.merge(test_agg_fe_df,on='id')


# In[ ]:


df_test_all


# In[ ]:


set(df_train_all.columns) - set(df_test_all.columns)


# In[ ]:


test_essaysdf


# In[ ]:


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)


# In[ ]:


AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', pd.DataFrame.kurt, 'sum']

def split_essays_into_sentences(df):
    essay_df = df
    essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',str(x)))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.columns.tolist()].reset_index(drop=True)
    return essay_df

def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def split_essays_into_paragraphs(df):
    essay_df = df
    essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: str(x).split('\n'))
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


# In[ ]:


# Sentence features for train dataset
train_sent_df = split_essays_into_sentences(train_essaysdf)
train_sent_agg_df = compute_sentence_aggregations(train_sent_df)
# plt.figure(figsize=(15, 1.5))
# plt.boxplot(x=train_sent_df.sent_len, vert=False, labels=['Sentence length'])
# plt.show()


# In[ ]:


train_paragraph_df = split_essays_into_paragraphs(train_essaysdf)
train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)


# In[ ]:


test_sent_agg_df = compute_sentence_aggregations(split_essays_into_sentences(test_essaysdf))
test_paragraph_agg_df = compute_paragraph_aggregations(split_essays_into_paragraphs(test_essaysdf))


# In[ ]:


test_sent_agg_df


# In[ ]:


train_paragraph_agg_df.loc[:, 'id'] = df_train_index
train_sent_agg_df.loc[:, 'id'] = df_train_index


# In[ ]:


test_paragraph_agg_df.loc[:, 'id'] = df_test_index
test_sent_agg_df.loc[:, 'id'] = df_test_index


# In[ ]:


new_train_feats = pd.DataFrame()
new_test_feats = pd.DataFrame()


# In[ ]:


new_train_feats = train_paragraph_agg_df.merge(df_train_all,on='id')
new_train_feats = new_train_feats.merge(train_sent_agg_df,on='id')


# In[ ]:


new_test_feats = test_paragraph_agg_df.merge(df_test_all,on='id')
new_test_feats = new_test_feats.merge(test_sent_agg_df,on='id')


# In[ ]:


new_train_feats.shape


# In[ ]:


new_test_feats


# In[ ]:


df_train_all.shape


# In[ ]:


df_test_all.shape


# In[ ]:


#new_train_feats = new_train_feats.merge(train_scores,on='id')


# In[ ]:


train_feats = pd.DataFrame()
test_feats = pd.DataFrame()


# In[ ]:


train_feats = new_train_feats.merge(other_train_feats,on='id')
test_feats = new_test_feats.merge(other_test_feats,on='id')


# In[ ]:


set(train_feats.columns) - set(test_feats.columns)


# In[ ]:


'''train_feats['score'] = train_feats['score_x']
train_feats = train_feats.drop(['score_x','score_y'],axis = 1)'''


# In[ ]:


data = []

for logs in [traindf, testdf]:
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


# In[ ]:


'''train_feats['score'] = train_feats['score_x']
train_feats = train_feats.drop(['score_x','score_y'],axis = 1)'''


# In[ ]:


set(train_feats.columns) - set(test_feats.columns)


# In[ ]:


train_feats.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_feats['score_class'] = le.fit_transform(train_feats['score'])


# In[ ]:


target_col = ['score']

drop_cols = ['id', 'score_class']
train_cols = list()

train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

train_cols.__len__(), target_col.__len__()


# In[ ]:


train_feats.shape , test_feats.shape


# In[ ]:


test_feats


# In[ ]:


train_feats.shape


# In[ ]:


test_feats.shape


# In[ ]:


set(train_feats.columns) - set(test_feats.columns)


# In[ ]:


nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
nan_cols


# In[ ]:


for col in nan_cols:
    mode_value_train = train_feats[col].mode()[0]  # In case there are multiple modes, choose the first one
    train_feats[col].fillna(mode_value_train, inplace=True)


# In[ ]:


for col in test_feats.columns[test_feats.isna().any()].tolist():
    # Find the most frequent value in the training set for the current feature
    most_frequent_value_train = train_feats[col].mode()[0]
    
    # Fill missing values in the test set with the most frequent value from the training set
    test_feats[col].fillna(most_frequent_value_train, inplace=True)

train_feats.shape, test_feats.shape


# In[ ]:


train_feats.columns[train_feats.isna().any()].tolist()


# In[ ]:


nan_values_test = test_feats.columns[test_feats.isna().any()].tolist()


# In[ ]:


nan_values_test


# In[ ]:


#train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]


# In[ ]:


#train_feats = train_feats.drop(['score_x','score',axis=1)


# In[ ]:


#test_feats[train_cols]


# In[ ]:


#OOF_PREDS = np.zeros((len(train_feats), 2))
#TEST_PREDS = np.zeros((len(test_feats), 2))


# # Train the Model using best parameters from Community

# In[ ]:


models_dict = {}
scores = []

test_predict_list = []
best_params = {'boosting_type': 'gbdt', 
        'metric': 'rmse',
        'reg_alpha': 0.003188447814669599, 
        'reg_lambda': 0.0010228604507564066, 
        'colsample_bytree': 0.5420247656839267, 
        'subsample': 0.9778252382803456, 
        'feature_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_fraction': 0.75,
        'learning_rate': 0.01716485155812008, 
        'num_leaves': 19, 
        'min_child_samples': 46,
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': 500,
         'device_type':'gpu'}

for i in range(5): 
    kf = model_selection.KFold(n_splits=10, random_state=42 + i, shuffle=True)

    oof_valid_preds = np.zeros(train_feats.shape[0], )

    X_test = test_feats[train_cols]


    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):

        print("==-"* 50)
        print("Fold : ", fold)

        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]

        print("Trian :", X_train.shape, y_train.shape)
        print("Valid :", X_valid.shape, y_valid.shape)

        params = {
            "objective": "regression",
            "metric": "rmse",
            'random_state': 42,
            "n_estimators" : 12001,
            "verbosity": -1,
            "device_type" : "gpu",
            **best_params
        }

        model = lgb.LGBMRegressor(**params)

        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        verbose_callback = lgb.log_evaluation(100)

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
                  callbacks=[early_stopping_callback, verbose_callback],
        )

        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict

        test_predict = model.predict(X_test)
        test_predict_list.append(test_predict)

        score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
        print("Fold RMSE Score : ", score)

        models_dict[f'{fold}_{i}'] = model


    oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
    scores.append(oof_score)
    print("OOF RMSE Score : ", oof_score)


# ## Plot the important Feature

# 

# In[ ]:


feature_importances_values = np.asarray([model.feature_importances_ for model in models_dict.values()]).mean(axis=0)
feature_importance_df = pd.DataFrame({'name': train_cols, 'importance': feature_importances_values})

feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)


# In[ ]:


np.mean(scores)


# ###  You may observe that the feature generated from the tokenizer is highlighted among the significant features.

# In[ ]:


plt.figure(figsize=(15, 6))

ax = sns.barplot(data=feature_importance_df.head(30), x='name', y='importance')
ax.set_title(f"Mean feature importances")
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)

plt.show()


# In[ ]:


test_feats['score'] = np.mean(test_predict_list, axis=0)


# In[ ]:


#test_feats.head()


# In[ ]:


test_feats[['id', 'score']].to_csv("submission.csv", index=False)


# In[ ]:


test_feats[['id','score']]


# In[ ]:


#test_feats[['id','score']]

