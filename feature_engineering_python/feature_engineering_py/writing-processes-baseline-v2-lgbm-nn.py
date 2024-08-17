#!/usr/bin/env python
# coding: utf-8

# # Linking Writing Processes to Writing Quality: Baseline - Electric Boogaloo
# 
# This notebook is my second iteration of a public baseline for the Writing Processes to Quality competition.
# 
# ## References
# 
# I used these as my references. **Please go upvote these as well!**
# 
# - https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline (original baseline)
# - https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline (my v1 baseline)
# - https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features (TF-IDF features)
# - https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs ({SPACE} features)
# - https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features (Essay features)
# - https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences (LightAutoML)
# - https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor (initial essay constructor)
# - https://www.kaggle.com/code/yuriao/fast-essay-constructor (fast essay constructor)
# 
# ## Changes
# 
# ### Encapsulation
# 
# I have encapsulated much of the code into classes, which makes it easier to isolate logic, which adheres to the Single Responsibility Principle as much as possible. 
# 
# ### Visualizations
# 
# I have added graphs to help analyze the performance of our models. Feel free to iterate on these.
# 
# ### Feature Engineering
# 
# We use the new [fast essay constructor](https://www.kaggle.com/code/yuriao/fast-essay-constructor) by @yuriao to speed up feature engineering. We also add new features:
# - the number of words whose length exceed a given value N;
# - the number of sentences whose length exceed a given value N;
# - more activity/event counts;
# - the inclusion of both the TF-IDF and regular versions of counts for activity and events.
# 
# To save time, we also include the pre-engineered features for the training logs in a separate dataset and provide an option to load these in if desired.
# 
# ### Modelling
# 
# We use tweaked LGBM parameters and the same setup for LightAutoML as the 0.586 notebook. As the LightAutoML was trained on GPU, we will need to run the notebook using GPU as well.
# 
# We will use an ensemble of 6 models using hill climbing: LightGBM, CatBoost, LightAutoML (DenseLight), Ridge, Lasso and Random Forest.

# # Set Global Configuration Options

# In[ ]:


class CONFIG:
    '''
    > General Options
    '''
    # global seed
    seed = 42
    # the number of samples to use for testing purposes
    # if None, we use the full dataset
    samples_testing = None #None
    # max rows to display for pandas dataframes
    display_max_rows = 200
    # name of the response variate we are trying to predict
    response_variate = 'score'
    # minimum value for response variate
    min_possible_response_value = 0.5
    # maximum value for response variate
    max_possible_response_value = 6.0
    
    '''
    > Feature Engineering Options
    '''
    # whether to use pre feature engineered data or not
    use_pre_fe_data = True
    # fe data saved path
    pre_fe_data_filepath = '/kaggle/input/writing-quality-baseline-v2-train-data/feat_eng_train_feats.csv'
    
    '''
    > Preprocessing Options
    '''
    # number of folds to split the data for CV
    num_folds = 10
    
    '''
    > Modelling + Training Options
    '''
    # the names of the models to use
    # either a list of model names, or 'all', in which case all models are used
    model_names = 'all'
    # number of trials to use for early stopping
    num_trials_early_stopping = 50
    # model path for lightautoml
    lightautoml_model_path = '/kaggle/input/writing-quality-baseline-v2-lightautoml/denselight.model'
    # oof preds path for lightautoml
    lightautoml_oof_preds_path = '/kaggle/input/writing-quality-baseline-v2-lightautoml/denselight_oof_preds'
    
    '''
    > Post-Modelling Options
    '''
    # number of most important features to display
    # for feature importances plots
    num_features_to_display = 50


# # Import Libraries

# ## Download LightAutoML v0.3.8 and Pandas v2.0.3

# In[ ]:


get_ipython().system('pip install --no-index -Uq --find-links=/kaggle/input/writing-quality-pip-install-with-lightautoml lightautoml==0.3.8')
get_ipython().system('pip install --no-index -Uq --find-links=/kaggle/input/writing-quality-pip-install-with-lightautoml pandas==2.0.3')


# ## Import Other Libraries

# In[ ]:


import warnings

import os
import gc
import re
import random
from collections import Counter, defaultdict
import pprint
import pickle
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# ## Set Some Options

# In[ ]:


tqdm.pandas()

sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', CONFIG.display_max_rows)
warnings.simplefilter('ignore')

random.seed(CONFIG.seed)


# # Load Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "INPUT_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality'\n\ntrain_logs = pd.read_csv(f'{INPUT_DIR}/train_logs.csv')\ntrain_scores = pd.read_csv(f'{INPUT_DIR}/train_scores.csv')\ntest_logs = pd.read_csv(f'{INPUT_DIR}/test_logs.csv')\n")


# ## Subsample Data (If Specified)

# In[ ]:


if CONFIG.samples_testing is not None:
    ids = list(train_logs["id"].unique())
    sample_ids = random.sample(ids, CONFIG.samples_testing)
    train_logs = train_logs[train_logs["id"].isin(sample_ids)]


# ## Looking At Data

# In[ ]:


train_logs.head()


# In[ ]:


train_scores.head()


# In[ ]:


test_logs.head()


# # Feature Engineering

# ## Essay Constructor
# 
# We encapsulate the functionality of the essay construction into a class, which allows for easy reusability.
# 
# Function taken from https://www.kaggle.com/code/yuriao/fast-essay-constructor.

# In[ ]:


class EssayConstructor:
    
    def processingInputs(self,currTextInput):
        # Where the essay content will be stored
        essayText = ""
        # Produces the essay
        for Input in currTextInput.values:
            # Input[0] = activity
            # Input[1] = cursor_position
            # Input[2] = text_change
            # Input[3] = id
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
                
            # If activity = input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        return essayText
            
            
    def getEssays(self,df):
        # Copy required columns
        textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
        # Get rid of text inputs that make no change
        textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']     
        # construct essay, fast 
        tqdm.pandas()
        essay=textInputDf.groupby('id')[['activity','cursor_position', 'text_change']].progress_apply(lambda x: self.processingInputs(x))      
        # to dataframe
        essayFrame=essay.to_frame().reset_index()
        essayFrame.columns=['id','essay']
        # Returns the essay series
        return essayFrame


# ## Preprocessor Class
# 
# The original preprocessor comes from my 0.604 baseline notebook. I have added the new {SPACE} features and essay aggregation features into the preprocessor as well, which may have been separate in the other public notebooks.

# In[ ]:


# nth percentile function for agg
def percentile(n):
    def percentile_(x):
        return x.quantile(n/100)
    percentile_.__name__ = 'pct_{:02.0f}'.format(n)
    return percentile_

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

class Preprocessor:
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes_dict = {
            'q': 'q', 
            ' ': 'space', 
            'NoChange': 'NoChange', 
            '.': 'full_stop', 
            ',': 'comma', 
            '\n': 'newline', 
            "'": 'single_quote', 
            '"': 'double_quote', 
            '-': 'dash', 
            '?': 'question_mark', 
            ';': 'semicolon', 
            '=': 'equals', 
            '/': 'slash', 
            '\\': 'double_backslash', 
            ':': 'colon'
        }
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 70, 100]
        self.percentiles = [5, 10, 25, 50, 75, 90, 95]
        self.percentiles_cols = [percentile(n) for n in self.percentiles]
        self.aggregations = ['mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', pd.DataFrame.kurt, 'sum']
        self.idf = defaultdict(float)
        
        self.essay_constructor = EssayConstructor()
    
    def get_essay_aggregations(self, essay_df):
        cols_to_drop = ['essay']
        # Total essay length
        essay_df['essay_len'] = essay_df['essay'].apply(lambda x: len(x))
        essay_df = essay_df.drop(columns=cols_to_drop)
        return essay_df
    
    def split_essays_into_words(self, essay_df):
        essay_df['word'] = essay_df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
        essay_df = essay_df.explode('word')
        # Word length (number of characters in word)
        essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
        essay_df = essay_df[essay_df['word_len'] != 0]
        return essay_df
    
    def compute_word_aggregations(self, word_df):
        word_agg_df = word_df[['id','word_len']].groupby(['id']).agg(self.aggregations)
        word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
        word_agg_df['id'] = word_agg_df.index
        # New features: computing the # of words whose length exceed word_l
        for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
            word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
            word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
        word_agg_df = word_agg_df.reset_index(drop=True)
        return word_agg_df
    
    def split_essays_into_sentences(self, essay_df):
        essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
        essay_df = essay_df.explode('sent')
        essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
        # Number of characters in sentences
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        # Number of words in sentences
        essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
        return essay_df

    def compute_sentence_aggregations(self, sent_df):
        sent_agg_df = sent_df[['id','sent_len','sent_word_count']].groupby(['id']).agg(self.aggregations)
        sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
        sent_agg_df['id'] = sent_agg_df.index
        # New features: computing the # of sentences whose (character) length exceed sent_l
        for sent_l in [50, 60, 75, 100]:
            sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_df[sent_df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
            sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
        sent_agg_df = sent_agg_df.reset_index(drop=True)
        return sent_agg_df

    def split_essays_into_paragraphs(self, essay_df):
        essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
        essay_df = essay_df.explode('paragraph')
        # Number of characters in paragraphs
        essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x)) 
        # Number of sentences in paragraphs
        essay_df['paragraph_sent_count'] = essay_df['paragraph'].apply(lambda x: len(x.split('\\.|\\?|\\!')))
        # Number of words in paragraphs
        essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.paragraph_len!=0].reset_index(drop=True)
        return essay_df

    def compute_paragraph_aggregations(self, paragraph_df):
        paragraph_agg_df = paragraph_df[['id','paragraph_len', 'paragraph_sent_count', 'paragraph_word_count']].groupby(['id']).agg(self.aggregations)
        paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
        paragraph_agg_df['id'] = paragraph_agg_df.index
        paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
        return paragraph_agg_df
        
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            # make dictionary entry for "move from X to Y"
            di["move_to"] = 0
            
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
                else:
                    # we can do this because there are no missing values
                    di["move_to"] += v
            ret.append(di)
        
        ret = pd.DataFrame(ret)
        # using tfidf
        ret_tfidf = pd.DataFrame(ret)
        # returning counts as is
        ret_normal = pd.DataFrame(ret)
        
        tfidf_cols = [f'activity_{act}_tfidf_count' for act in ret.columns]
        normal_cols = [f'activity_{act}_normal_count' for act in ret.columns]
        
        ret_tfidf.columns = tfidf_cols
        ret_normal.columns = normal_cols
        
        '''
        Credit: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features
        '''
        cnts = ret_tfidf.sum(1)

        for col in tfidf_cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret_tfidf[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret_tfidf[col] = 1 + np.log(ret_tfidf[col] / cnts)
            ret_tfidf[col] *= idf
        
        ret_agg = pd.concat([ret_tfidf, ret_normal], axis=1)
        return ret_agg

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
        # using tfidf
        ret_tfidf = pd.DataFrame(ret)
        # returning counts as is
        ret_normal = pd.DataFrame(ret)
        
        tfidf_cols = [f'{colname}_{event}_tfidf_count' for event in ret.columns]
        normal_cols = [f'{colname}_{event}_normal_count' for event in ret.columns]
        
        ret_tfidf.columns = tfidf_cols
        ret_normal.columns = normal_cols
        
        '''
        Credit: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features
        '''
        cnts = ret_tfidf.sum(1)

        for col in tfidf_cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret_tfidf[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret_tfidf[col] = 1 + np.log(ret_tfidf[col] / cnts)
            ret_tfidf[col] *= idf
        
        ret_agg = pd.concat([ret_tfidf, ret_normal], axis=1)
        return ret_agg

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes_dict.keys():
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
            
        ret = pd.DataFrame(ret)
        # using tfidf
        ret_tfidf = pd.DataFrame(ret)
        # returning counts as is
        ret_normal = pd.DataFrame(ret)
        
        tfidf_cols = [f'text_change_{self.text_changes_dict[txt_change]}_tfidf_count' for txt_change in ret.columns]
        normal_cols = [f'text_change_{self.text_changes_dict[txt_change]}_normal_count' for txt_change in ret.columns]
        
        ret_tfidf.columns = tfidf_cols
        ret_normal.columns = normal_cols
        
        '''
        Credit: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features
        '''
        cnts = ret_tfidf.sum(1)

        for col in tfidf_cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret_tfidf[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret_tfidf[col] = 1 + np.log(ret_tfidf[col] / cnts)
            ret_tfidf[col] *= idf
        
        ret_agg = pd.concat([ret_tfidf, ret_normal], axis=1)
        return ret_agg
    
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

    # Credit: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs/notebook
    def make_space_features(self, df):
        df['up_time_lagged'] = df.groupby('id')['up_time'].shift(1).fillna(df['down_time'])
        df['time_diff'] = abs(df['down_time'] - df['up_time_lagged']) / 1000

        group = df.groupby('id')['time_diff']
        largest_lantency = group.max()
        smallest_lantency = group.min()
        median_lantency = group.median()
        initial_pause = df.groupby('id')['down_time'].first() / 1000
        pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
        pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
        pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
        pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
        pauses_3_sec = group.apply(lambda x: (x > 3).sum())
        
        result = pd.DataFrame({
            'id': df['id'].unique(),
            'largest_lantency': largest_lantency,
            'smallest_lantency': smallest_lantency,
            'median_lantency': median_lantency,
            'initial_pause': initial_pause,
            'pauses_half_sec': pauses_half_sec,
            'pauses_1_sec': pauses_1_sec,
            'pauses_1_half_sec': pauses_1_half_sec,
            'pauses_2_sec': pauses_2_sec,
            'pauses_3_sec': pauses_3_sec,
        }).reset_index(drop=True)
        return result
    
    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        for percentile in self.percentiles:
            tmp_df[f'input_word_length_pct_{percentile}'] = tmp_df['text_change'].apply(lambda x: np.percentile([len(i) for i in x] if len(x) > 0 else 0, 
                                                                                                               percentile))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def make_feats(self, df: pd.DataFrame, save_essays_path: str):
        
        print("Starting to engineer features")
        
        # initialize features dataframe
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        # get essay feats
        print("Getting essays")
        essay_df = self.essay_constructor.getEssays(df)
        essay_df.to_csv(save_essays_path, index=False)

        print("Getting essay aggregations data")
        essay_agg_df = self.get_essay_aggregations(essay_df)
        feats = feats.merge(essay_agg_df, on='id', how='left')

        print("Getting essay word aggregations data")
        word_df = self.split_essays_into_words(essay_df)
        word_agg_df = self.compute_word_aggregations(word_df)
        feats = feats.merge(word_agg_df, on='id', how='left')

        print("Getting essay sentence aggregations data")
        sent_df = self.split_essays_into_sentences(essay_df)
        sent_agg_df = self.compute_sentence_aggregations(sent_df)
        feats = feats.merge(sent_agg_df, on='id', how='left')

        print("Getting essay paragraph aggregations data")
        paragraph_df = self.split_essays_into_paragraphs(essay_df)
        paragraph_agg_df = self.compute_paragraph_aggregations(paragraph_df)
        feats = feats.merge(paragraph_agg_df, on='id', how='left')
        
        # engineer counts data
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
        
        # space features
        print("Engineering space-related data")
        tmp_df = self.make_space_features(df)
        feats = feats.merge(tmp_df, on='id', how='left')
        
        # get shifted features
        # time shift
        print("Engineering time data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)
        
        # cursor position shift
        print("Engineering cursor position data - gaps")
        for gap in self.gaps: 
            print(f"> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)
        
        # word count shift
        print("Engineering word count data - gaps")
        for gap in self.gaps: 
            print(f"> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        # get aggregate statistical features
        print("Engineering statistical summaries for features")
        # [(feature name, [ stat summaries to add ])]
        percentiles_cols = [percentile(n) for n in self.percentiles]
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['first', 'last', 'max']),
            ('down_time', ['first', 'last', 'max']),
            ('action_time', ['max', 'mean', 'std', 'sem', 'skew', pd.DataFrame.kurt ] + self.percentiles_cols),
            ('activity', ['nunique']),
            ('down_event', [ 'nunique']),
            ('up_event', [ 'nunique']),
            ('text_change', [ 'nunique']),
            ('cursor_position', ['max']),
            ('word_count', ['max'])] 

        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['first','last', 'max', 'min', 'mean', 'std', 'sem', 'skew', pd.DataFrame.kurt]+ percentiles_cols),
                (f'cursor_position_change{gap}', ['first','last','max', 'mean', 'std','sem', 'skew', pd.DataFrame.kurt]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'sum', 'sem', 'skew', pd.DataFrame.kurt] + percentiles_cols),
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

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')
        
        # compare feats
        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_mean'] / feats['up_time_max']
        
        print("Done!")
        return feats


# ## Run Preprocessor

# In[ ]:


if CONFIG.use_pre_fe_data:
    print("-"*25)
    print("Loading pre-engineered features for training data")
    print("-"*25)
    train_feats = pd.read_csv(CONFIG.pre_fe_data_filepath)
else:
    preprocessor_train = Preprocessor(
        seed = CONFIG.seed,
    )
    print("-"*25)
    print("Engineering features for training data")
    print("-"*25)
    train_feats = preprocessor_train.make_feats(
        train_logs,
        save_essays_path = 'train_essays.csv'
    )
    del preprocessor_train
    gc.collect()

print()
print("-"*25)
print("Engineering features for test data")
print("-"*25)
preprocessor_test = Preprocessor(
    seed = CONFIG.seed,
)
test_feats = preprocessor_test.make_feats(
    test_logs,
    save_essays_path = 'test_essays.csv'
)
del preprocessor_test
gc.collect()


# In[ ]:


train_feats.to_csv('feat_eng_train_feats.csv', index=False)


# In[ ]:


print(f"Shape of training data: {train_feats.shape}")
print(f"Shape of test data: {test_feats.shape}")


# In[ ]:


assert train_feats.shape[1] == test_feats.shape[1], "Train and test data must have same number of features."


# In[ ]:


train_feats.head()


# In[ ]:


test_feats.head()


# In[ ]:


train_feats = train_feats.merge(train_scores, on='id', how='left')


# # Split Data Into Folds

# In[ ]:


kfold = KFold(n_splits=CONFIG.num_folds, shuffle=True, random_state=CONFIG.seed)
    
for fold, (_, val_idx) in enumerate(kfold.split(train_feats)):
    train_feats.loc[val_idx, "fold"] = fold


# # Modelling

# We encapsulate the model into a class, which allows us to
# - use the same class for different types of models (LGBM/XGB/Cat/Sklearn/etc)
# - cleans up the code, since we can just call one function to train/validate/predict

# In[ ]:


class WritingQualityModel:
    
    def __init__(self, model_name: str, params: dict):
        self.model_name = model_name
        self.model = self.create_model(model_name, params)
    
    def make_pipeline(self, model: str):
        return Pipeline([
            ('remove_infs', FunctionTransformer(lambda x: np.nan_to_num(x, nan=np.nan, posinf=0, neginf=0))),
            ('imputer', SimpleImputer(strategy='mean')),
            ('normalizer', FunctionTransformer(lambda x: np.log1p(np.abs(x)))),
            ('scaler', RobustScaler()),
            ('model', model)
        ])
    
    def create_model(self, model_name: str, params: dict):
        model = None
        if 'lgbm' in model_name:
            model = lgb.LGBMRegressor(**params)
        elif 'cat' in model_name:
            model = cb.CatBoostRegressor(**params)
        elif 'rfr' in model_name:
            model = RandomForestRegressor(**params)
        elif 'lasso' in model_name:
            model = self.make_pipeline(Lasso(**params))
        elif 'ridge' in model_name:
            model = self.make_pipeline(Ridge(**params))
        return model

    def train(self, X_train, Y_train, X_val, Y_val):
        if any(x in self.model_name for x in ['lgbm']):
            early_stopping_callback = lgb.early_stopping(CONFIG.num_trials_early_stopping, first_metric_only=True, verbose=False)

            self.model.fit(X_train,
                      Y_train,
                      eval_set=[(X_val, Y_val)],
                      eval_metric='rmse',
                      verbose=0,
                      callbacks=[early_stopping_callback])

        elif any(x in self.model_name for x in ['cat']):
            self.model.fit(X_train,
                      Y_train,
                      eval_set=[(X_val, Y_val)],
                      verbose=0,
                      early_stopping_rounds=CONFIG.num_trials_early_stopping)
        else:
            X_train = np.nan_to_num(X_train, posinf=-1, neginf=-1)
            self.model.fit(X_train, Y_train)

        return self.model
    
    def validate(self, X_val, Y_val):
        if any(x in self.model_name for x in ['rfr', 'ridge', 'lasso']):
            X_val = np.nan_to_num(X_val, posinf=-1, neginf=-1)
            
        pred = self.model.predict(X_val)
        score = mean_squared_error(pred, Y_val, squared=False)
        return pred, score
    
    def predict(self, X_test):
        if any(x in self.model_name for x in ['rfr', 'ridge', 'lasso']):
            X_test = np.nan_to_num(X_test, posinf=-1, neginf=-1)
            
        return self.model.predict(X_test)


# In[ ]:


model_params_dict = {
    'lgbm': {
        'boosting_type': 'gbdt', 
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
    },
    'cat': {
        'learning_rate': 0.027407502438096695,
        'min_child_samples': 23,
        'iterations': 500,
        'random_state': 419610,
        'reg_lambda': 0.24381872974603785,
        'subsample': 0.889148863756771,
    },
    'rfr': {
        'max_depth': 6,
        'max_features': 'sqrt',
        'min_impurity_decrease': 0.0016295128631816343,
        'n_estimators': 200,
        'random_state': 42,
    },
    'ridge': {
        'alpha': 1,
        'random_state': 42,
    },
    'lasso': {
        'alpha': 0.04198227921905038, 
        'max_iter': 2000, 
        'random_state': 42,
    },
}


# # Select Models

# In[ ]:


if CONFIG.model_names == 'all':
    model_names = list(model_params_dict.keys())
else:
    model_names = CONFIG.model_names


# # Select Features

# In[ ]:


default_feature_names = list(
        filter(lambda x: x not in [CONFIG.response_variate, 'id', 'fold'], train_feats.columns))

def get_features_for_model():
    feature_names = default_feature_names
    # filter out features with zero std deviation
    feature_names = [name for name in feature_names if np.std(train_feats[name]) > 0]
    return feature_names


# # KFold Cross-Validation Trainer 

# We also encapsulate the training, validation and prediction code into a Trainer class.

# In[ ]:


class KFoldTrainer:
    """Responsible for training a model using k-fold cross validation"""
    
    def __init__(self, seed: int, model_name: str, model_params: dict):
        self.seed = seed
        self.model_name = model_name
        self.model_params = model_params
        
        # [fold]: model
        self.saved_models = {}
        
    def train_by_fold(self, df: pd.DataFrame, feature_names: list[str], verbose: bool = False):
        for fold in range(CONFIG.num_folds):
            if verbose:
                print(f"Training for FOLD {fold}")
                
            X_train = df[df["fold"] != fold][feature_names]
            Y_train = df[df["fold"] != fold][CONFIG.response_variate]

            X_val = df[df["fold"] == fold][feature_names]
            Y_val = df[df["fold"] == fold][CONFIG.response_variate]

            model = WritingQualityModel(self.model_name, self.model_params)
            model.train(X_train.values, Y_train.values, X_val.values, Y_val.values)

            self.saved_models[fold] = model
    
    def get_saved_models(self):
        return self.saved_models
    
    # returns (df_pred, metric, fold_metrics, mean_fold_metric, sd_fold_metric)
    def get_oof_predictions_and_metric(self, df: pd.DataFrame, feature_names: list[str], verbose: bool = False):
        df_pred = pd.DataFrame({'index': df.index}).set_index('index')
        fold_metrics = []
        for fold in range(CONFIG.num_folds):
            if verbose:
                print(f"Validating for FOLD {fold}")
                
            X_val = df[df["fold"] == fold][feature_names]
            Y_val = df[df["fold"] == fold][CONFIG.response_variate]
            pred_fold = self.saved_models[fold].predict(X_val.values)
            df_pred.loc[X_val.index, "pred"] = pred_fold
            
            fold_metric = mean_squared_error(Y_val.values, pred_fold, squared=False)
            fold_metrics.append(fold_metric)
            
        metric = mean_squared_error(df[CONFIG.response_variate], df_pred["pred"], squared=False)
        mean_fold_metric, sd_fold_metric = np.mean(fold_metrics), np.std(fold_metrics)
        return df_pred['pred'].values, metric, fold_metrics, mean_fold_metric, sd_fold_metric
    
    def predict(self, df: pd.DataFrame, feature_names: list[str], verbose: bool = False):
        df_pred = pd.DataFrame({'index': df.index}).set_index('index')
        for fold in range(CONFIG.num_folds):
            if verbose:
                print(f"Predicting for FOLD {fold}")
            X_test = df[feature_names]
            pred = self.saved_models[fold].predict(X_test.values)
            df_pred[f"pred{fold}"] = pred
            
        df_pred["pred"] = df_pred[[f"pred{fold}" for fold in range(CONFIG.num_folds)]].mean(axis=1)
        return df_pred["pred"].values


# # Training / Validation / Test Loop

# In[ ]:


print(f"Out of a possible {len(default_feature_names)} features, we are using {len(get_features_for_model())} for training.")


# In[ ]:


feature_names = ['id'] + get_features_for_model()
train_feats[feature_names].to_csv('feat_eng_train_feats_select_features.csv', index=False)


# In[ ]:


def postprocess_preds(preds):
    # clipping
    post_proc_preds = np.clip(preds, 
                              a_min=CONFIG.min_possible_response_value, 
                              a_max=CONFIG.max_possible_response_value)
    
    return post_proc_preds

validation_rmses = {}
model_scores = []
model_fold_scores = []
models_dict = {}

for idx, model_name in enumerate(model_names):
    
    print("="*25)
    print(f"Starting training, validation and prediction for model {model_name} [MODEL {idx+1}/{len(model_names)}]")
    print("="*25)
    
    print("-"*25)
    print(f"Training model:")
    print("-"*25)
    
    feature_names = get_features_for_model()
    start_time = time.process_time()
    trainer = KFoldTrainer(
        seed = CONFIG.seed,
        model_name = model_name,
        model_params = model_params_dict[model_name]
    )
    # training
    trainer.train_by_fold(train_feats, feature_names, verbose=True)
    print("-"*25)
    
    models_dict[model_name] = trainer.get_saved_models()
    
    # validation
    pred_train, metric, fold_metrics, mean_fold_metric, sd_fold_metric = trainer.get_oof_predictions_and_metric(train_feats, feature_names)
    train_feats[f'pred_{CONFIG.response_variate}_{model_name}'] = postprocess_preds(pred_train)
    
    for fold in range(CONFIG.num_folds):
        print(f"RMSE for FOLD {fold}: {fold_metrics[fold]:6f}")
    print()
    print(f"OOF RMSE for {model_name}: {metric:.6f}")
    print(f"Mean/SD RMSE for {model_name}: {mean_fold_metric:.6f} ± {sd_fold_metric:.6f}")
    
    # for plotting later on
    validation_rmses[model_name] = metric 
    model_scores.append({
        'model_name': model_name,
        'score': metric,
    })
    model_fold_scores.extend([{
        'model_name': model_name,
        'fold': fold,
        'score': fold_metrics[fold]
    } for fold in range(CONFIG.num_folds)])
    
    # prediction (test set)
    print("-"*25)
    print(f"Predicting test set with model:")
    print("-"*25)
    test_feats[f'pred_{CONFIG.response_variate}_{model_name}'] = postprocess_preds(
        trainer.predict(test_feats, feature_names, verbose=True))
    
    # cleanup
    del trainer, pred_train, metric, fold_metrics, mean_fold_metric, sd_fold_metric
    gc.collect()


# ## Add LightAutoML Predictions

# In[ ]:


import joblib
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

light_automl_model = joblib.load(CONFIG.lightautoml_model_path)
light_automl_oof_preds = joblib.load(CONFIG.lightautoml_oof_preds_path)

train_feats[f'pred_{CONFIG.response_variate}_lightautoml'] = light_automl_oof_preds.data.reshape(-1)
test_feats[f'pred_{CONFIG.response_variate}_lightautoml'] = light_automl_model.predict(test_feats[get_features_for_model()]).data

lightautoml_rmse = mean_squared_error(train_feats[f'pred_{CONFIG.response_variate}_lightautoml'], train_feats[CONFIG.response_variate], squared=False)
validation_rmses['lightautoml'] = lightautoml_rmse
print(f"OOF RMSE for LightAutoML RMSE: {lightautoml_rmse}")

model_names.append('lightautoml')
model_scores.append({ 'model_name': 'lightautoml', 'score': lightautoml_rmse })


# # Mean Feature Importances Of (Decision Tree) Models

# In[ ]:


model_decision_trees = set(['lgbm', 'cat', 'rfr']).intersection(set(model_names))


# In[ ]:


# model_name: feat_df
model_feat_dfs = {}

all_feature_importances_df = pd.DataFrame({'name': feature_names})

for model_name in model_decision_trees:
    fold_models = models_dict[model_name].values()
    
    feature_importances_values = np.asarray([model.model.feature_importances_ for model in fold_models]).mean(axis=0)
    feature_importance_df = pd.DataFrame({'name': feature_names, 'importance': feature_importances_values})
    all_feature_importances_df = all_feature_importances_df.merge(
        feature_importance_df.rename(columns={'importance': f'importance_{model_name}'}), 
        on='name', 
        suffixes=(None, None))

    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(CONFIG.num_features_to_display)
    model_feat_dfs[model_name] = feature_importance_df
    
fig, axes = plt.subplots(nrows=len(model_decision_trees), 
                         figsize=(12, len(model_decision_trees)*8))
plt.subplots_adjust(hspace=1)

if len(model_decision_trees) == 1:
    axes = [axes]

for ax, model_name in zip(axes, model_decision_trees):
    
    feature_importance_df = model_feat_dfs[model_name]
    
    sns.barplot(data=feature_importance_df, x='name', y='importance', ax=ax)
    ax.set_title(f"Mean feature importances for model {model_name}")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)

plt.show()


# We can see that our new essay features have good feature importance.

# # Visualizations of Model Performance

# ## Graph of OOF RMSEs

# In[ ]:


plt.figure(figsize=(len(model_names)*1, 8))

pred_df = pd.DataFrame.from_records(model_scores)
min_score = pred_df['score'].min()
max_score = pred_df['score'].max()

order = pred_df.sort_values('score', ascending=True)['model_name']
ax = sns.barplot(data=pred_df, x='model_name', y='score', order=order)
ax.bar_label(ax.containers[0], fmt='{:,.5f}')
ax.set(ylim=(min_score-0.05, max_score+0.05))
plt.show()


# ## Correlation Matrix of Model OOF Predictions and Score
# 
# Reasoning behind this: highly correlated models do not ensemble as well as (relatively) lowly correlated models.

# In[ ]:


pred_and_score_cols = [f"pred_{CONFIG.response_variate}_{model_name}" for model_name in model_names] + [CONFIG.response_variate]

plt.figure(figsize=(10, 10))
sns.heatmap(train_feats[pred_and_score_cols].corr(numeric_only=True), annot=True, fmt='.4f')
plt.show()


# ## Boxplots of Fold RMSEs
# 
# Reasoning: this can help investigate the spread of the RMSEs for each fold.

# In[ ]:


plt.figure(figsize=(15, 8))

fold_pred_df = pd.DataFrame.from_records(model_fold_scores)
order = fold_pred_df.groupby('model_name').median().sort_values('score', ascending=True).index
ax = sns.boxplot(data=fold_pred_df, x='model_name', y='score', order=order)
plt.show()


# ## Boxenplots of Residuals For Each Model
# 
# Reasoning: this can help investigate where our models perform worse. In particular, we see that our models perform worse in the extremes.

# In[ ]:


# make score into category so we can use it to categorize residuals
train_feats['score_cat'] = train_feats['score'].astype('category')
# make residuals
for model_name in model_names:
    train_feats[f'diff_{model_name}'] = train_feats[f'pred_{CONFIG.response_variate}_{model_name}'] - train_feats[CONFIG.response_variate]
diff_cols = [f'diff_{model_name}' for model_name in model_names]
    
fig, axes = plt.subplots(nrows=len(model_names), figsize=(15, len(model_names)*6))

for idx, col in enumerate(diff_cols): 
    sns.boxenplot(data=train_feats, y=col, ax=axes[idx], x='score_cat')
    axes[idx].set_title(f'{col} by score')
    
plt.show()


# # Ensembling Using Hill Climbing

# In[ ]:


# [(model_name, weight)]
model_weights = {}
# preds
best_ensemble_train_preds = None
best_ensemble_test_preds = None

print("-"*25)
print(f"Running hill climbing")
print("-"*25)

# Initialise
STOP = False
# [model_name]: model_weight
model_weights = {}
best_model_order = [a[0] for a in sorted(validation_rmses.items(), key=lambda x: x[1])]
i = 0

cur_model_names = list(set(model_names).difference(set([best_model_order[0]]))).copy()
y_target = train_feats[CONFIG.response_variate].values

best_ensemble_train_preds = train_feats[f"pred_{CONFIG.response_variate}_{best_model_order[0]}"]
best_ensemble_test_preds = test_feats[f"pred_{CONFIG.response_variate}_{best_model_order[0]}"]

potential_new_best_cv_score = mean_squared_error(y_target, best_ensemble_train_preds, squared=False)
model_weights[best_model_order[0]] = 1
print(f"Initial best single model RMSE ({best_model_order[0]}): {potential_new_best_cv_score}")

weight_range = np.arange(-0.6, 0.6, 0.001)

# Hill climbing
while not STOP:
    i += 1
    potential_new_best_cv_score = mean_squared_error(y_target, best_ensemble_train_preds, squared=False)
    k_best, model_name_best, wgt_best = None, None, None
    for k, model_name in enumerate(cur_model_names):
        for wgt in weight_range:
            potential_ensemble = (1-wgt) * best_ensemble_train_preds + wgt * train_feats[f"pred_{CONFIG.response_variate}_{model_name}"]
            cv_score = mean_squared_error(y_target, potential_ensemble, squared=False)
            if cv_score < potential_new_best_cv_score:
                potential_new_best_cv_score = cv_score
                k_best, model_name_best, wgt_best = k, model_name, wgt

    if k_best is not None:
        best_ensemble_train_preds = (1-wgt_best) * best_ensemble_train_preds + wgt_best * train_feats[f"pred_{CONFIG.response_variate}_{model_name_best}"]
        best_ensemble_test_preds = (1-wgt_best) * best_ensemble_test_preds + wgt_best * test_feats[f"pred_{CONFIG.response_variate}_{model_name_best}"]

        model_weights = {model_name: model_weight * (1-wgt_best) for model_name, model_weight in model_weights.items()}
        model_weights[model_name_best] = wgt_best

        cur_model_names.remove(model_name_best)
        if len(cur_model_names) == 0:
            STOP = True
        print(f'Iteration: {i}, Model added: {model_name_best}, Best weight: {wgt_best:.5f}, Best RMSE: {potential_new_best_cv_score:.7f}')
    else:
        STOP = True


# ## Model Weights

# In[ ]:


model_weights_df = pd.DataFrame(model_weights.items(), columns=['model_name', 'weight'])
model_weights_df.sort_values('weight', ascending=False, inplace=True)

plt.figure(figsize=(18, 6))
ax = sns.barplot(data=model_weights_df, x='model_name', y='weight')
ax.bar_label(ax.containers[0], fmt='{:,.3f}')

plt.show()


# # Predict Train and Test Set with Ensemble Predictions

# In[ ]:


train_feats[f"pred_{CONFIG.response_variate}"] = best_ensemble_train_preds
test_feats[CONFIG.response_variate] = best_ensemble_test_preds


# # Scatter Plot of Predictions vs Actual

# In[ ]:


plt.figure(figsize=(6, 6))
ax = sns.scatterplot(data=train_feats, x=CONFIG.response_variate, y=f"pred_{CONFIG.response_variate}")
sns.lineplot(data=train_feats, x=CONFIG.response_variate, y=CONFIG.response_variate, ax=ax, color="black")
ax.set_title("Predicted vs Actual")
plt.show()


# # Submission

# In[ ]:


submission = test_feats[["id", CONFIG.response_variate]]
submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


# # Thanks for reading! Make sure to upvote if you forked/liked this!
