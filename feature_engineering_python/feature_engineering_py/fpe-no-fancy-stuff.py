#!/usr/bin/env python
# coding: utf-8

# #### Experiment Log:
# 
# 
# | Version | Models Used | CV Score | LB Score | Comments |
# | --- | --- | --- | --- | --- |
# | v1 | LogisticRegression <br> RandomForestClassifier | - | - | Baseline (Errored Out)
# | v2 | LogisticRegression <br> RandomForestClassifier | 0.8898984 <br> 0.8877021 | - | Baseline
# | v3 | LogisticRegression <br> RandomForestClassifier | 0.8898984 <br> 0.8877021 | 0.868 | Baseline
# | v4 | LogisticRegression <br> RandomForestClassifier | 0.7463457 <br> 0.8374394 | 0.784 | Text Preprocessing <br> TF-IDF
# | v5 | LogisticRegression <br> RandomForestClassifier | 0.7534682 <br> 0.7823450 | 0.761 | Text Preprocessing <br> Lemmatization <br> TF-IDF <br> Glove Embeddings
# | v6 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7534682 <br> 0.6215837 <br> 0.5861382 | 0.695 | Text Preprocessing <br> Lemmatization <br> TF-IDF <br> Glove Embeddings
# | v7 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7932787 <br> 0.6045388 <br> 0.5755040 | 0.685 | Text Preprocessing <br> Lemmatization <br> Glove Embeddings
# | v8 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7932841 <br> 0.6041433 <br> 0.5748984 | 0.677 | Text Preprocessing <br> Lemmatization <br> Glove Embeddings <br> New features added
# | v9 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7013439 <br> 0.6039653 <br> 0.5749525 | 0.673 | Text Preprocessing <br> Lemmatization <br> Glove Embeddings <br> Quantile Transformer
# | v10 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7012418 <br> 0.5843301 <br> 0.5770625 | 0.679 | Text Preprocessing <br> Lemmatization <br> Glove Embeddings <br> Quantile Transformer
# | v11 | LogisticRegression <br> XGBoost <br> LightGBM <br> Voting Classifier | Error | - | Text Preprocessing <br> Lemmatization <br> Glove Embeddings <br> Quantile Transformer
# | v13 | LogisticRegression <br> XGBoost <br> LightGBM <br> Voting Classifier | 0.2121148 | 0.681 | Text Preprocessing <br> Lemmatization <br> Glove Embeddings <br> Quantile Transformer
# | v14 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7034141 <br> 0.6002171 <br> 0.5768558 | 0.669 | Text Preprocessing <br> Lemmatization <br> Glove + FastText Embeddings <br> Quantile Transformer
# | v15 | LogisticRegression <br> XGBoost <br> LightGBM | 0.6929070 <br> 0.6058480 <br> 0.5769126 | 0.685 | Sentence-Transformers
# | v16 | XGBoost <br> LightGBM | Error | - | Text Preprocessing (handle OOV words) <br> Lemmatization <br> Glove + FastText Embeddings <br> Quantile Transformer
# | v17 | XGBoost <br> LightGBM | 0.5981064 <br> 0.5738860 | 0.673 | Text Preprocessing (handle OOV words) <br> Lemmatization <br> Glove + FastText Embeddings <br> Quantile Transformer
# | v18 | LogisticRegression <br> XGBoost <br> LightGBM | 0.8130772 <br> 0.6800559 <br> 0.6760178 | 0.672 | Text Preprocessing <br> Lemmatization <br> Glove + FastText Embeddings <br> Quantile Transformer <br> GroupKFold
# | v19 | LogisticRegression <br> XGBoost <br> LightGBM | 0.8135349 <br> 0.6804664 <br> 0.6747677 | - | Text Preprocessing <br> Lemmatization <br> Glove + FastText Embeddings <br> TextBlob to handle OOV tokens <br> Quantile Transformer <br> GroupKFold
# | v21 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7428862 <br> 0.6817771 <br> 0.6779332 | 0.672 | Preprocessing <br> Lemmatization <br> Glove + FastText Embeddings <br> Min Max Scaler <br> Stratified GroupKFold
# | v22 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7022837 <br> 0.5999362 <br> 0.5741513 | 0.669 | Text Preprocessing <br> Lemmatization <br> Glove + FastText Embeddings <br> Quantile Transformer
# | v25 | LogisticRegression <br> XGBoost <br> LightGBM | 0.6991969 <br> 0.5993221 <br> 0.5729388 | 0.666 | Text Preprocessing <br> Glove + FastText Embeddings <br> Quantile Transformer
# | v26 | LogisticRegression <br> XGBoost <br> LightGBM | 0.7008197 <br> 0.6003453 <br> 0.5755974 | 0.667 | Text Preprocessing <br> Glove + Paragram Embeddings <br> Quantile Transformer
# | v27 | LogisticRegression <br> XGBoost <br> LightGBM | - | - | Text Preprocessing <br> Glove + Paragram Embeddings <br> Quantile Transformer <br> New features added

# ## Import libraries

# In[1]:


import gc
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import re
import nltk
import string
from textblob import TextBlob
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

tqdm.pandas()
np.random.seed(42)


# ## Load source datasets

# In[2]:


train = pd.read_csv('../input/feedback-prize-effectiveness/train.csv')
train["essay_text"] = train["essay_id"].apply(lambda x: open(f'../input/feedback-prize-effectiveness/train/{x}.txt').read())
print(f"train: {train.shape}")
train.head()


# In[3]:


test = pd.read_csv('../input/feedback-prize-effectiveness/test.csv')
test["essay_text"] = test["essay_id"].apply(lambda x: open(f'../input/feedback-prize-effectiveness/test/{x}.txt').read())
print(f"test: {test.shape}")
test.head()


# In[4]:


train["discourse_effectiveness"] = train["discourse_effectiveness"].map({
    "Adequate":1,
    "Effective":2,
    "Ineffective":0
})
train['discourse_effectiveness'].value_counts()


# In[5]:


Ytrain = train['discourse_effectiveness'].values
train.drop(['discourse_effectiveness'], inplace=True, axis=1)

print(f"train: {train.shape} \ntest: {test.shape} \nYtrain: {Ytrain.shape}")


# ## Feature Engineering

# ### Helper Functions

# In[6]:


misspell_mapping = {
    'studentdesigned': 'student designed',
    'teacherdesigned': 'teacher designed',
    'genericname': 'generic name',
    'winnertakeall': 'winner take all',
    'studentname': 'student name',
    'driveless': 'driverless',
    'teachername': 'teacher name',
    'propername': 'proper name',
    'bestlaid': 'best laid',
    'genericschool': 'generic school',
    'schoolname': 'school name',
    'winnertakesall': 'winner take all',
    'elctoral': 'electoral',
    'eletoral': 'electoral',
    'genericcity': 'generic city',
    'elctors': 'electoral',
    'venuse': 'venue',
    'blimplike': 'blimp like',
    'selfdriving': 'self driving',
    'electorals': 'electoral',
    'nearrecord': 'near record',
    'egyptianstyle': 'egyptian style',
    'oddnumbered': 'odd numbered',
    'carintensive': 'car intensive',
    'elecoral': 'electoral',
    'oction': 'auction',
    'electroal': 'electoral',
    'evennumbered': 'even numbered',
    'mesalandforms': 'mesa landforms',
    'electoralvote': 'electoral vote',
    'relativename': 'relative name',
    '22euro': 'twenty two euro',
    'ellectoral': 'electoral',
    'thirtyplus': 'thirty plus',
    'collegewon': 'college won',
    'hisher': 'higher',
    'teacherbased': 'teacher based',
    'computeranimated': 'computer animated',
    'canadidate': 'candidate',
    'studentbased': 'student based',
    'gorethanks': 'gore thanks',
    'clouddraped': 'cloud draped',
    'edgarsnyder': 'edgar snyder',
    'emotionrecognition': 'emotion recognition',
    'landfrom': 'land form',
    'fivedays': 'five days',
    'electoal': 'electoral',
    'lanform': 'land form',
    'electral': 'electoral',
    'presidentbut': 'president but',
    'teacherassigned': 'teacher assigned',
    'beacuas': 'because',
    'positionestimating': 'position estimating',
    'selfeducation': 'self education',
    'diverless': 'driverless',
    'computerdriven': 'computer driven',
    'outofcontrol': 'out of control',
    'faultthe': 'fault the',
    'unfairoutdated': 'unfair outdated',
    'aviods': 'avoid',
    'momdad': 'mom dad',
    'statesbig': 'states big',
    'presidentswing': 'president swing',
    'inconclusion': 'in conclusion',
    'handsonlearning': 'hands on learning',
    'electroral': 'electoral',
    'carowner': 'car owner',
    'elecotral': 'electoral',
    'studentassigned': 'student assigned',
    'collegefive': 'college five',
    'presidant': 'president',
    'unfairoutdatedand': 'unfair outdated and',
    'nixonjimmy': 'nixon jimmy',
    'canadates': 'candidate',
    'tabletennis': 'table tennis',
    'himher': 'him her',
    'studentsummerpacketdesigners': 'student summer packet designers',
    'studentdesign': 'student designed',
    'limting': 'limiting',
    'electrol': 'electoral',
    'campaignto': 'campaign to',
    'presendent': 'president',
    'thezebra': 'the zebra',
    'landformation': 'land formation',
    'eyetoeye': 'eye to eye',
    'selfreliance': 'self reliance',
    'studentdriven': 'student driven',
    'winnertake': 'winner take',
    'alliens': 'aliens',
    '2000but': '2000 but',
    'electionto': 'election to',
    'candidatesas': 'candidates as',
    'electers': 'electoral',
    'winnertakes': 'winner takes',
    'isfeet': 'is feet',
    'incar': 'incur',
    'covid19': 'something',
    'aflcio': '',
    'outdatedand': 'outdated and',
    'httpswww': '',
    '51998': '',
    'iswing': '',
    'ascertainments': '',
    'athome': '',
    'risorius': '',
    'votes538': '',
    '41971': '',
    'palpabraeus': '',
    'figurelandform': 'figure landform',
    'possibleit': 'possible it',
    'takeall': 'take all',
    'inschool': 'in school',
    'fouces': 'focus',
    'presidentand': 'president and',
    'elecotrs': 'electoral',
    'formationwhich': 'formation which',
    'electorswho': 'electoral who',
    'presidnt': 'president',
    'eletors': 'electoral',
    'sinceraly': 'sincerely',
    'emotionshappiness': 'emotions happiness',
    'carterbob': 'carter bob',
    'donÃ£Ã¢t': 'do not',
    'eyesnose': 'eyes nose',
    'smartroad': 'smart road',
    'systemvoters': 'system voters',
    'emtions': 'emotions',
    'statedemocrats': 'state democrats',
    'lowcar': 'low car',
    'elcetoral': 'electoral',
    'expressivefor': 'expressive for',
    'animails': 'animals',
    'oppertonuty': 'opportunity',
    'tempetures': 'temperature',
    'recevies': 'receives',
    'twoseat': 'two seat',
    'consistution': 'constitution',
    'horsesyoung': 'horses young',
    'semidriverless': 'semi driverless',
    'presisdent': 'president',
    'exspression': 'expression',
    'valcanoes': 'volcano',
    'actiry': '',
    'lifejust': 'life just',
    'selfreliant': 'self reliant',
    'comcaraccidentcauseofaccidentcellphonecellphonestatistics': 'car accident cause of accident cellphone statistics',
    'vaubangermany': 'germany',
    'fourtyfour': 'fourty four',
    'atomspheric': 'atmospheric',
    'mid1990': '',
    'activitis': 'activities',
    'paragrpah': 'paragraph',
    'electora': 'electoral',
    'elcetion': 'election',
    'stressfree': 'stress free',
    'seegoing': 'see going',
    'coferencing': 'conferencing',
    'ctrdot': '',
    'segoing': '',
    'teacherdesign': 'teacher design',
    'kidsteens': 'kids teens',
    'elcetors': 'electoral',
    'poulltion': 'pollution',
    'surportive': 'supportive',
    'presisent': 'president',
    'technollogy': 'technology',
    'precidency': 'president',
    'voteswhile': 'votes while',
    'headformed': 'head formed',
    'swingstates': 'swing states',
    'candates': 'candidate',
    'locationname': 'location name',
    'venuss': 'venues',
    'astronmers': 'astronomers',
    'democtratic': 'democratic',
    'canadent': 'candidate',
    'cyndonia': '',
    'computure': 'computer',
    'nasas': 'nasa',
    'onehalf': 'one half',
    'preident': 'president',
    'ressons': 'reasons',
    'presidentvice': 'president vice',
    'nonswing': 'non swing',
    'thirtyeight': 'thirty eight',
    'processnot': 'process not',
    'facetoface': 'face to face',
    'teendriversource': 'teen driver source',
    'sadnessand': 'sadness and',
    'abloish': 'abolish',
    'driveing': 'driving',
    'navagating': 'navigating',
    'electorsthe': 'electoral',
    'vothing': 'voting',
    'callage': 'college',
    'senseit': 'sense it',
    'mercedesbenz': 'mercedes benz',
    'electorall': 'electoral'
}


# In[7]:


def contraction_count(sent):
    count = 0
    count += re.subn(r"won\'t", '', sent)[1]
    count += re.subn(r"can\'t", '', sent)[1]
    count += re.subn(r"n\'t", '', sent)[1]
    count += re.subn(r"\'re", '', sent)[1]
    count += re.subn(r"\'s", '', sent)[1]
    count += re.subn(r"\'d", '', sent)[1]
    count += re.subn(r"\'ll", '', sent)[1]
    count += re.subn(r"\'t", '', sent)[1]
    count += re.subn(r"\'ve", '', sent)[1]
    count += re.subn(r"\'m", '', sent)[1]
    return count


# In[8]:


def pos_count(sent):
    nn_count = 0   #Noun
    pr_count = 0   #Pronoun
    vb_count = 0   #Verb
    jj_count = 0   #Adjective
    uh_count = 0   #Interjection
    cd_count = 0   #Numerics
    
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)

    for token in sent:
        if token[1] in ['NN','NNP','NNS']:
            nn_count += 1

        if token[1] in ['PRP','PRP$']:
            pr_count += 1

        if token[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            vb_count += 1

        if token[1] in ['JJ','JJR','JJS']:
            jj_count += 1

        if token[1] in ['UH']:
            uh_count += 1

        if token[1] in ['CD']:
            cd_count += 1
    
    return pd.Series([nn_count, pr_count, vb_count, jj_count, uh_count, cd_count])


# In[9]:


def decontraction(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"he's", "he is", phrase)
    phrase = re.sub(r"there's", "there is", phrase)
    phrase = re.sub(r"We're", "We are", phrase)
    phrase = re.sub(r"That's", "That is", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"they're", "they are", phrase)
    phrase = re.sub(r"Can't", "Cannot", phrase)
    phrase = re.sub(r"wasn't", "was not", phrase)
    phrase = re.sub(r"don\x89Ûªt", "do not", phrase)
    phrase = re.sub(r"donãât", "do not", phrase)
    phrase = re.sub(r"aren't", "are not", phrase)
    phrase = re.sub(r"isn't", "is not", phrase)
    phrase = re.sub(r"What's", "What is", phrase)
    phrase = re.sub(r"haven't", "have not", phrase)
    phrase = re.sub(r"hasn't", "has not", phrase)
    phrase = re.sub(r"There's", "There is", phrase)
    phrase = re.sub(r"He's", "He is", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"You're", "You are", phrase)
    phrase = re.sub(r"I'M", "I am", phrase)
    phrase = re.sub(r"shouldn't", "should not", phrase)
    phrase = re.sub(r"wouldn't", "would not", phrase)
    phrase = re.sub(r"i'm", "I am", phrase)
    phrase = re.sub(r"I\x89Ûªm", "I am", phrase)
    phrase = re.sub(r"I'm", "I am", phrase)
    phrase = re.sub(r"Isn't", "is not", phrase)
    phrase = re.sub(r"Here's", "Here is", phrase)
    phrase = re.sub(r"you've", "you have", phrase)
    phrase = re.sub(r"you\x89Ûªve", "you have", phrase)
    phrase = re.sub(r"we're", "we are", phrase)
    phrase = re.sub(r"what's", "what is", phrase)
    phrase = re.sub(r"couldn't", "could not", phrase)
    phrase = re.sub(r"we've", "we have", phrase)
    phrase = re.sub(r"it\x89Ûªs", "it is", phrase)
    phrase = re.sub(r"doesn\x89Ûªt", "does not", phrase)
    phrase = re.sub(r"It\x89Ûªs", "It is", phrase)
    phrase = re.sub(r"Here\x89Ûªs", "Here is", phrase)
    phrase = re.sub(r"who's", "who is", phrase)
    phrase = re.sub(r"I\x89Ûªve", "I have", phrase)
    phrase = re.sub(r"y'all", "you all", phrase)
    phrase = re.sub(r"can\x89Ûªt", "cannot", phrase)
    phrase = re.sub(r"would've", "would have", phrase)
    phrase = re.sub(r"it'll", "it will", phrase)
    phrase = re.sub(r"we'll", "we will", phrase)
    phrase = re.sub(r"wouldn\x89Ûªt", "would not", phrase)
    phrase = re.sub(r"We've", "We have", phrase)
    phrase = re.sub(r"he'll", "he will", phrase)
    phrase = re.sub(r"Y'all", "You all", phrase)
    phrase = re.sub(r"Weren't", "Were not", phrase)
    phrase = re.sub(r"Didn't", "Did not", phrase)
    phrase = re.sub(r"they'll", "they will", phrase)
    phrase = re.sub(r"they'd", "they would", phrase)
    phrase = re.sub(r"DON'T", "DO NOT", phrase)
    phrase = re.sub(r"That\x89Ûªs", "That is", phrase)
    phrase = re.sub(r"they've", "they have", phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"should've", "should have", phrase)
    phrase = re.sub(r"You\x89Ûªre", "You are", phrase)
    phrase = re.sub(r"where's", "where is", phrase)
    phrase = re.sub(r"Don\x89Ûªt", "Do not", phrase)
    phrase = re.sub(r"we'd", "we would", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"weren't", "were not", phrase)
    phrase = re.sub(r"They're", "They are", phrase)
    phrase = re.sub(r"Can\x89Ûªt", "Cannot", phrase)
    phrase = re.sub(r"you\x89Ûªll", "you will", phrase)
    phrase = re.sub(r"I\x89Ûªd", "I would", phrase)
    phrase = re.sub(r"let's", "let us", phrase)
    phrase = re.sub(r"it's", "it is", phrase)
    phrase = re.sub(r"can't", "cannot", phrase)
    phrase = re.sub(r"don't", "do not", phrase)
    phrase = re.sub(r"you're", "you are", phrase)
    phrase = re.sub(r"i've", "I have", phrase)
    phrase = re.sub(r"that's", "that is", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"doesn't", "does not",phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"didn't", "did not", phrase)
    phrase = re.sub(r"ain't", "am not", phrase)
    phrase = re.sub(r"you'll", "you will", phrase)
    phrase = re.sub(r"I've", "I have", phrase)
    phrase = re.sub(r"Don't", "do not", phrase)
    phrase = re.sub(r"I'll", "I will", phrase)
    phrase = re.sub(r"I'd", "I would", phrase)
    phrase = re.sub(r"Let's", "Let us", phrase)
    phrase = re.sub(r"you'd", "You would", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"Ain't", "am not", phrase)
    phrase = re.sub(r"Haven't", "Have not", phrase)
    phrase = re.sub(r"Could've", "Could have", phrase)
    phrase = re.sub(r"youve", "you have", phrase)  
    phrase = re.sub(r"donå«t", "do not", phrase)
    return phrase


# In[10]:


def remove_punctuations(text):
    for punctuation in list(string.punctuation):
        text = text.replace(punctuation, '')
    return text


# In[11]:


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {
        "N": wordnet.NOUN, 
        "V": wordnet.VERB, 
        "J": wordnet.ADJ, 
        "R": wordnet.ADV
    }
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


# In[12]:


def clean_number(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    return text


# In[13]:


def clean_misspell(text):
    for bad_word in misspell_mapping:
        if bad_word in text:
            text = text.replace(bad_word, misspell_mapping[bad_word])
    return text


# In[14]:


def sent2vec(text):
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            M.append(embeddings_index['unk'])
            continue
    
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    
    return v / np.sqrt((v ** 2).sum())


# ### Create basic text features

# In[15]:


def text_features(df, col):
    df[f"{col}_num_words"] = df[col].progress_apply(lambda x: len(str(x).split()))
    df[f"{col}_num_unique_words"] = df[col].progress_apply(lambda x: len(set(str(x).split())))
    df[f"{col}_num_chars"] = df[col].progress_apply(lambda x: len(str(x)))
    df[f"{col}_num_stopwords"] = df[col].progress_apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords.words('english')]))
    df[f"{col}_num_punctuations"] = df[col].progress_apply(lambda x: len([c for c in str(x) if c in list(string.punctuation)]))
    df[f"{col}_num_words_upper"] = df[col].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df[f"{col}_num_words_title"] = df[col].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df[f"{col}_mean_word_len"] = df[col].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df[f"{col}_num_paragraphs"] = df[col].progress_apply(lambda x: len(x.split('\n')))
    df[f"{col}_num_contractions"] = df[col].progress_apply(contraction_count)
    df[f"{col}_polarity"] = df[col].progress_apply(lambda x: TextBlob(x).sentiment[0])
    df[f"{col}_subjectivity"] = df[col].progress_apply(lambda x: TextBlob(x).sentiment[1])
    df[[f'{col}_nn_count',f'{col}_pr_count',f'{col}_vb_count',f'{col}_jj_count',f'{col}_uh_count',f'{col}_cd_count']] = df[col].progress_apply(pos_count)
    return df


# In[16]:


discourse_train = train[['discourse_id','discourse_text']].copy()
discourse_train.drop_duplicates(inplace=True)
print(f"discourse_train: {discourse_train.shape}")

discourse_train = text_features(discourse_train, "discourse_text")
discourse_train.head()


# In[17]:


essay_train = train[['essay_id','essay_text']].copy()
essay_train.drop_duplicates(inplace=True)
print(f"essay_train: {essay_train.shape}")

essay_train = text_features(essay_train, "essay_text")
essay_train.head()


# In[18]:


discourse_test = test[['discourse_id','discourse_text']].copy()
discourse_test.drop_duplicates(inplace=True)
print(f"discourse_test: {discourse_test.shape}")

discourse_test = text_features(discourse_test, "discourse_text")
discourse_test.head()


# In[19]:


essay_test = test[['essay_id','essay_text']].copy()
essay_test.drop_duplicates(inplace=True)
print(f"essay_test: {essay_test.shape}")

essay_test = text_features(essay_test, "essay_text")
essay_test.head()


# ### Text Preprocessing

# In[20]:


def clean_text(text):
    text = decontraction(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = remove_punctuations(text)
    text = clean_number(text)
    text = clean_misspell(text)
    return text


# In[21]:


discourse_train['discourse_text'] = discourse_train['discourse_text'].progress_apply(clean_text)
discourse_test['discourse_text'] = discourse_test['discourse_text'].progress_apply(clean_text)


# In[22]:


essay_train['essay_text'] = essay_train['essay_text'].progress_apply(clean_text)
essay_test['essay_text'] = essay_test['essay_text'].progress_apply(clean_text)


# ### Glove Embeddings

# In[23]:


with open("../input/nlp-word-embeddings/Glove_Embeddings.txt", 'rb') as handle: 
    data = handle.read()

processed_data = pickle.loads(data)
embeddings_index = processed_data['glove_embeddings_index']
print('Word vectors found: {}'.format(len(embeddings_index)))

del processed_data
gc.collect()


# In[24]:


discourse_train.set_index('discourse_id', inplace=True)

glove_vec = [sent2vec(x) for x in tqdm(discourse_train["discourse_text"].values)]
col_list = ['discourse_glove_'+str(i) for i in range(300)]
glove_vec_df = pd.DataFrame(np.array(glove_vec), columns=col_list, index=discourse_train.index)
print(f"glove_vec_df: {glove_vec_df.shape}")

discourse_train = pd.merge(
    discourse_train, 
    glove_vec_df, 
    how="inner", 
    on="discourse_id", 
    sort=False
)

del glove_vec, glove_vec_df
gc.collect()

print(f"discourse_train: {discourse_train.shape}")
discourse_train.head()


# In[25]:


essay_train.set_index('essay_id', inplace=True)

glove_vec = [sent2vec(x) for x in tqdm(essay_train["essay_text"].values)]
col_list = ['essay_glove_'+str(i) for i in range(300)]
glove_vec_df = pd.DataFrame(np.array(glove_vec), columns=col_list, index=essay_train.index)
print(f"glove_vec_df: {glove_vec_df.shape}")

essay_train = pd.merge(
    essay_train, 
    glove_vec_df, 
    how="inner", 
    on="essay_id", 
    sort=False
)

del glove_vec, glove_vec_df
gc.collect()

print(f"essay_train: {essay_train.shape}")
essay_train.head()


# In[26]:


discourse_test.set_index('discourse_id', inplace=True)

glove_vec = [sent2vec(x) for x in tqdm(discourse_test["discourse_text"].values)]
col_list = ['discourse_glove_'+str(i) for i in range(300)]
glove_vec_df = pd.DataFrame(np.array(glove_vec), columns=col_list, index=discourse_test.index)
print(f"glove_vec_df: {glove_vec_df.shape}")

discourse_test = pd.merge(
    discourse_test, 
    glove_vec_df, 
    how="inner", 
    on="discourse_id", 
    sort=False
)

del glove_vec, glove_vec_df
gc.collect()

print(f"discourse_test: {discourse_test.shape}")
discourse_test.head()


# In[27]:


essay_test.set_index('essay_id', inplace=True)

glove_vec = [sent2vec(x) for x in tqdm(essay_test["essay_text"].values)]
col_list = ['essay_glove_'+str(i) for i in range(300)]
glove_vec_df = pd.DataFrame(np.array(glove_vec), columns=col_list, index=essay_test.index)
print(f"glove_vec_df: {glove_vec_df.shape}")

essay_test = pd.merge(
    essay_test, 
    glove_vec_df, 
    how="inner", 
    on="essay_id", 
    sort=False
)

del glove_vec, glove_vec_df
gc.collect()

print(f"essay_test: {essay_test.shape}")
essay_test.head()


# In[28]:


del embeddings_index
gc.collect()


# ### Paragram Embeddings

# In[29]:


with open("../input/nlp-word-embeddings/Para_Embeddings.txt", 'rb') as handle: 
    data = handle.read()

processed_data = pickle.loads(data)
embeddings_index = processed_data['para_embeddings_index']
print('Word vectors found: {}'.format(len(embeddings_index)))

del processed_data
gc.collect()


# In[30]:


para_vec = [sent2vec(x) for x in tqdm(discourse_train["discourse_text"].values)]
col_list = ['discourse_para_'+str(i) for i in range(300)]
para_vec_df = pd.DataFrame(np.array(para_vec), columns=col_list, index=discourse_train.index)
print(f"para_vec_df: {para_vec_df.shape}")

discourse_train = pd.merge(
    discourse_train, 
    para_vec_df, 
    how="inner", 
    on="discourse_id", 
    sort=False
)

del para_vec, para_vec_df
gc.collect()

discourse_train.drop('discourse_text', axis=1, inplace=True)
print(f"discourse_train: {discourse_train.shape}")
discourse_train.head()


# In[31]:


para_vec = [sent2vec(x) for x in tqdm(essay_train["essay_text"].values)]
col_list = ['essay_para_'+str(i) for i in range(300)]
para_vec_df = pd.DataFrame(np.array(para_vec), columns=col_list, index=essay_train.index)
print(f"para_vec_df: {para_vec_df.shape}")

essay_train = pd.merge(
    essay_train, 
    para_vec_df, 
    how="inner", 
    on="essay_id", 
    sort=False
)

del para_vec, para_vec_df
gc.collect()

essay_train.drop('essay_text', axis=1, inplace=True)
print(f"essay_train: {essay_train.shape}")
essay_train.head()


# In[32]:


para_vec = [sent2vec(x) for x in tqdm(discourse_test["discourse_text"].values)]
col_list = ['discourse_para_'+str(i) for i in range(300)]
para_vec_df = pd.DataFrame(np.array(para_vec), columns=col_list, index=discourse_test.index)
print(f"para_vec_df: {para_vec_df.shape}")

discourse_test = pd.merge(
    discourse_test, 
    para_vec_df, 
    how="inner", 
    on="discourse_id", 
    sort=False
)

del para_vec, para_vec_df
gc.collect()

discourse_test.drop('discourse_text', axis=1, inplace=True)
print(f"discourse_test: {discourse_test.shape}")
discourse_test.head()


# In[33]:


para_vec = [sent2vec(x) for x in tqdm(essay_test["essay_text"].values)]
col_list = ['essay_para_'+str(i) for i in range(300)]
para_vec_df = pd.DataFrame(np.array(para_vec), columns=col_list, index=essay_test.index)
print(f"para_vec_df: {para_vec_df.shape}")

essay_test = pd.merge(
    essay_test, 
    para_vec_df, 
    how="inner", 
    on="essay_id", 
    sort=False
)

del para_vec, para_vec_df
gc.collect()

essay_test.drop('essay_text', axis=1, inplace=True)
print(f"essay_test: {essay_test.shape}")
essay_test.head()


# In[34]:


del embeddings_index
gc.collect()


# ### Merge all datasets

# In[35]:


train = pd.merge(
    train,
    discourse_train,
    how='inner',
    on='discourse_id',
    sort=False
)

train = pd.merge(
    train,
    essay_train,
    how='inner',
    on='essay_id',
    sort=False
)

print(f"train: {train.shape}")
train.head()


# In[36]:


test = pd.merge(
    test,
    discourse_test,
    how='inner',
    on='discourse_id',
    sort=False
)

test = pd.merge(
    test,
    essay_test,
    how='inner',
    on='essay_id',
    sort=False
)

print(f"test: {test.shape}")
test.head()


# In[37]:


del discourse_train, essay_train
del discourse_test, essay_test
gc.collect()


# ### Additional features

# In[38]:


train['discourse_index'] = train.apply(lambda x: x['essay_text'].find(x['discourse_text'].strip()), axis=1)
train['num_words_ratio'] = train['discourse_text_num_words']/train['essay_text_num_words']
train['num_unique_words_ratio'] = train['discourse_text_num_unique_words']/train['essay_text_num_unique_words']
train['num_chars_ratio'] = train['discourse_text_num_chars']/train['essay_text_num_chars']
train['num_stopwords_ratio'] = train['discourse_text_num_stopwords']/train['essay_text_num_stopwords']
train['num_punctuations_ratio'] = train['discourse_text_num_punctuations']/train['essay_text_num_punctuations']
train['mean_word_len_ratio'] = train['discourse_text_mean_word_len']/train['essay_text_mean_word_len']
train.head()


# In[39]:


test['discourse_index'] = test.apply(lambda x: x['essay_text'].find(x['discourse_text'].strip()), axis=1)
test['num_words_ratio'] = test['discourse_text_num_words']/test['essay_text_num_words']
test['num_unique_words_ratio'] = test['discourse_text_num_unique_words']/test['essay_text_num_unique_words']
test['num_chars_ratio'] = test['discourse_text_num_chars']/test['essay_text_num_chars']
test['num_stopwords_ratio'] = test['discourse_text_num_stopwords']/test['essay_text_num_stopwords']
test['num_punctuations_ratio'] = test['discourse_text_num_punctuations']/test['essay_text_num_punctuations']
test['mean_word_len_ratio'] = test['discourse_text_mean_word_len']/test['essay_text_mean_word_len']
test.head()


# In[40]:


df = train.groupby('essay_id')\
        .agg({'discourse_id':'count'})\
        .reset_index()\
        .rename(columns={'discourse_id':'discourse_count'})

train = pd.merge(
    train,
    df,
    how='inner',
    on='essay_id',
    sort=False
)

train.head()


# In[41]:


df = train.groupby(['essay_id','discourse_type']).agg({
    'discourse_id': 'count',
    'discourse_text_num_words': 'mean',
    'discourse_text_num_chars': 'mean'
}).reset_index().rename(columns={
    'discourse_id':'discourse_type_count',
    'discourse_text_num_words':'discourse_text_num_words_mean',
    'discourse_text_num_chars':'discourse_text_num_chars_mean'
})

train = pd.merge(
    train,
    df,
    how='inner',
    on=['essay_id','discourse_type'],
    sort=False
)

train.head()


# In[42]:


df = test.groupby('essay_id')\
        .agg({'discourse_id':'count'})\
        .reset_index()\
        .rename(columns={'discourse_id':'discourse_count'})

test = pd.merge(
    test,
    df,
    how='inner',
    on='essay_id',
    sort=False
)

test.head()


# In[43]:


df = test.groupby(['essay_id','discourse_type']).agg({
    'discourse_id': 'count',
    'discourse_text_num_words': 'mean',
    'discourse_text_num_chars': 'mean'
}).reset_index().rename(columns={
    'discourse_id':'discourse_type_count',
    'discourse_text_num_words':'discourse_text_num_words_mean',
    'discourse_text_num_chars':'discourse_text_num_chars_mean'
})

test = pd.merge(
    test,
    df,
    how='inner',
    on=['essay_id','discourse_type'],
    sort=False
)

test.head()


# ### Label Encoding and Feature Scaling

# In[44]:


le = LabelEncoder().fit(train['discourse_type'].append(test['discourse_type']))
train['discourse_type'] = le.transform(train['discourse_type'])
test['discourse_type'] = le.transform(test['discourse_type'])
train.head()


# In[45]:


train.drop([
    'discourse_id',
    'essay_id',
    'discourse_text',
    'essay_text'
], axis=1, inplace=True)


test.drop([
    'discourse_id',
    'essay_id',
    'discourse_text',
    'essay_text'
], axis=1, inplace=True)

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# In[46]:


features = test.columns.tolist()

qt = QuantileTransformer(n_quantiles=1000, 
                         output_distribution='normal', 
                         random_state=42).fit(train[features])

train[features] = qt.transform(train[features])
test[features] = qt.transform(test[features])


# In[47]:


Xtrain = train.copy()
Xtest = test.copy()
print(f"Xtrain: {Xtrain.shape} \nXtest: {Xtest.shape}")


# In[48]:


del train, test, qt
gc.collect()


# ## Models Training

# ### Logistic Regression

# In[49]:


FOLD = 5
SEEDS = [42]

counter = 0
oof_score = 0
y_pred_final_lr = np.zeros((Xtest.shape[0], 3))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain[val]

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(train_x, train_y)
        
        y_pred = model.predict_proba(val_x)
        y_pred_final_lr += model.predict_proba(Xtest)
        
        score = log_loss(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("Seed-{} | Fold-{} | OOF Score: {}".format(seed, idx, score))
        
        with open(f'FPE_LR_Model_{counter}.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_final_lr = y_pred_final_lr / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ### XGBoost

# In[50]:


FOLD = 5
SEEDS = [42]

counter = 0
oof_score = 0
y_pred_final_xgb = np.zeros((Xtest.shape[0], 3))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain[val]

        model = XGBClassifier(
            objective='multi:softproba',
            eval_metric='mlogloss',
            booster='gbtree',
            sample_type='weighted',
            tree_method='hist',
            grow_policy='lossguide',
            use_label_encoder=False,
            num_round=5000,
            num_class=3,
            max_depth=9, 
            max_leaves=36,
            learning_rate=0.095,
            subsample=0.7024,
            colsample_bytree=0.5289,
            min_child_weight=15,
            reg_lambda=0.05465,
            verbosity=0,
            random_state=42
        )
        
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                  early_stopping_rounds=100, verbose=50)
        
        y_pred = model.predict_proba(val_x, iteration_range=(0, model.best_iteration))
        y_pred_final_xgb += model.predict_proba(Xtest, iteration_range=(0, model.best_iteration))
        
        score = log_loss(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("Seed-{} | Fold-{} | OOF Score: {}".format(seed, idx, score))
        
        with open(f'FPE_XGB_Model_{counter}.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_final_xgb = y_pred_final_xgb / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ### LightGBM

# In[51]:


params = {}
params["objective"] = 'multiclass'
params['metric'] = 'multi_logloss'
params['boosting'] = 'gbdt'
params['num_class'] = 3
params['is_unbalance'] = True
params["learning_rate"] = 0.05
params["lambda_l2"] = 0.0256
params["num_leaves"] = 52
params["max_depth"] = 10
params["feature_fraction"] = 0.503
params["bagging_fraction"] = 0.741
params["bagging_freq"] = 8
params["bagging_seed"] = 10
params["min_data_in_leaf"] = 10
params["verbosity"] = -1
params["random_state"] = 42
num_rounds = 5000


# In[52]:


FOLD = 5
SEEDS = [42]

counter = 0
oof_score = 0
y_pred_final_lgb = np.zeros((Xtest.shape[0], 3))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain[val]
        
        lgtrain = lgb.Dataset(train_x, label=train_y.ravel())
        lgvalidation = lgb.Dataset(val_x, label=val_y.ravel())

        model = lgb.train(params, lgtrain, num_rounds, 
                          valid_sets=[lgtrain, lgvalidation], 
                          early_stopping_rounds=100, verbose_eval=100)
        
        y_pred = model.predict(val_x, num_iteration=model.best_iteration)
        y_pred_final_lgb += model.predict(Xtest, num_iteration=model.best_iteration)
        
        score = log_loss(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("Seed-{} | Fold-{} | OOF Score: {}".format(seed, idx, score))
        
        with open(f'FPE_LGB_Model_{counter}.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_final_lgb = y_pred_final_lgb / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ## Create submission file

# In[53]:


y_pred_final = (y_pred_final_lr * 0.1) + (y_pred_final_xgb * 0.5) + (y_pred_final_lgb * 0.4)

submission = pd.read_csv("../input/feedback-prize-effectiveness/sample_submission.csv")
submission['Ineffective'] = y_pred_final[:,0]
submission['Adequate'] = y_pred_final[:,1]
submission['Effective'] = y_pred_final[:,2]
submission.to_csv("./submission.csv", index=False)
submission.head()


# In[54]:


## Good Day!!

