#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebook was my attempt to achieve a score comparable to baseline neural networks such as pretrained transformers with only a simple ridge regression algorithm. Even more important than the score, is the feature engineering in this notebook. 
# 
# This notebook uses the readability library ([link to offline version dataset](https://www.kaggle.com/ravishah1/readability-package) - see how I use it below as submissions must be without internet) to generate 24 powerful traditional features. These features range from common statistics such as words per sentence to readability scoring measures such as kincaid. I also use the spacy libraries en_core_web_lg to generate 300 features. Lastly I incorporate 31 part of speech tag features using nltk. 
# 
# Hopefully combining these features with more advanced models will help you improve your score.
# 
# I hope you find these features useful. Upvote if you use these features. Comment questions and suggestions. I'll probably add more features to this notebook in the future.
# 
# Version Summary:
# 
# V1-6: Incomplete versions may have errors and bugs
# 
# V7: The original version
# 
# V8-9: Some experimental ideas with pearson's correlation and feature transformations - currently no improvement
# 
# V10/11: Same as V7 but fixed duplicate column name bug and typos

# In[1]:


import pandas as pd
import numpy as np
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag, pos_tag_sents
import string
import re
import math

#!pip install readability
import sys
sys.path = [
    '../input/readability-package',
] + sys.path
import readability
import spacy

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

import warnings
warnings.filterwarnings('ignore')


# In[2]:


nlp_sm = spacy.load('en_core_web_sm')


# In[3]:


get_ipython().system('pip install ../input/textstats/textstat-master')
get_ipython().system('pip install ../input/pyphen/Pyphen-master')


# In[4]:


import textstat


# # Peeking at the Data

# In[5]:


train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
train_df.head(2)


# In[6]:


ex_excerpt = train_df.iloc[0].excerpt
ex_excerpt


# In[7]:


sns.distplot(train_df["target"])


# In[8]:


sns.distplot(train_df["standard_error"])


# In[9]:


test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
test_df.head(2)


# # Feature Engineering

# In[10]:


#word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)
#print(word2vec_model.vectors.shape)


# In[11]:


#glove_words = list(word2vec_model.index_to_key)


# In[12]:


def TF_IDF_W2V(text):
    '''Calculate TF-IDF with word2vec
    '''
    #Load TF-IDF from sklearn
    TFIDF_model = TfidfVectorizer()
    #fit on text
    TFIDF_model.fit(text)
    #create dictionary with word as key
    #and idf as value
    dictionary = dict(zip(TFIDF_model.get_feature_names(), list(TFIDF_model.idf_)))
    #apply set as we need unique features
    TFIDF_words = set(TFIDF_model.get_feature_names())
    #create list which stores TFIDF_W2V
    TFIDF_W2V_vectors = []
    for sentence in text:
        #create empty vector to store result
        vector = np.zeros(300)
        #number of words with valid vector in sentence
        TFIDF_weight =0
        for word in sentence.split(): 
            #if word exist in glove_words and TFIDF_words
            if (word in glove_words) and (word in TFIDF_words):
                #get its vector from glove_words
                vec = word2vec_model[word]
                #calculate TF-IDF for each word
                TFIDF = dictionary[word]*(sentence.count(word)/len(sentence.split()))
                #calculate TF-IDF weighted W2V
                vector += (vec * TFIDF)
                TFIDF_weight += TFIDF
                
        if TFIDF_weight != 0:
            vector /= TFIDF_weight
        TFIDF_W2V_vectors.append(vector)
    return TFIDF_W2V_vectors 


# In[13]:


import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
def preprocess_text(text):
    text=str(text).lower()
    text = re.sub('\n', '', text)
    return text

def removestop(text):
    stop_words = set(stopwords.words('english')) 
  
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence
STOP_WORDS = set(stopwords.words('english')) # stop words
def rem_stop(text):
    text = ' '.join([i for i in text.split() if i not in STOP_WORDS])
    return text
def remove_punctuation(text):
    text_clean="".join([i for i in text if i not in string.punctuation])
    return text_clean



# In[14]:


def number_sentence(text: str):
    

    
    about_doc = nlp_sm(text) 
    sentences = list(about_doc.sents)
    log_len_sentence = math.log(len(sentences))
    len_sentence = len(sentences)
    difficult_words = textstat.difficult_words(text)
    log_difficult_words =  -1 if difficult_words==0 else math.log(difficult_words)
    
    word_list = []

    for i in about_doc:
        if not i.is_punct:
            word_list.append(str(i).lower())

    unique = set(word_list)
    
    unique_toker_per_text =  len(unique)
    
    log_unique_toker_per_text =  -1 if unique_toker_per_text==0 else math.log(unique_toker_per_text)
    
    text_len = len(text)
    log_text_len = math.log(len(text))
    
    return [log_len_sentence, len_sentence,difficult_words, log_difficult_words, unique_toker_per_text, log_unique_toker_per_text, text_len,log_text_len]
    


# In[15]:


def text_stats(text: str):
    

    
    difficult_words = textstat.difficult_words(text)
    
    return [difficult_words]


# In[16]:


number_sentence(train_df.excerpt[0])


# In[17]:


def readability_measurements(passage: str):
    """
    This function uses the readability library for feature engineering.
    It includes textual statistics, readability scales and metric, and some pos stats
    """
    results = readability.getmeasures(passage, lang='en')
    
    chars_per_word = results['sentence info']['characters_per_word']
    syll_per_word = results['sentence info']['syll_per_word']
    words_per_sent = results['sentence info']['words_per_sentence']
    complex_words  = results['sentence info']['complex_words']
    long_words  = results['sentence info']['long_words']
    
    log_chars_per_word  =  -1 if chars_per_word==0 else math.log(chars_per_word)
    log_syll_per_word   =  -1 if syll_per_word==0 else math.log(syll_per_word)
    log_words_per_sent  =  -1 if words_per_sent==0 else math.log(words_per_sent)
    log_complex_words   =  -1 if complex_words==0 else math.log(complex_words)
    log_long_words      =  -1 if long_words==0 else math.log(long_words)
    
    
    kincaid = results['readability grades']['Kincaid']
    ari = results['readability grades']['ARI']
    coleman_liau = results['readability grades']['Coleman-Liau']
    flesch = results['readability grades']['FleschReadingEase']
    gunning_fog = results['readability grades']['GunningFogIndex']
    lix = results['readability grades']['LIX']
    smog = results['readability grades']['SMOGIndex']
    rix = results['readability grades']['RIX']
    dale_chall = results['readability grades']['DaleChallIndex']
    
    tobeverb = results['word usage']['tobeverb']
    auxverb = results['word usage']['auxverb']
    conjunction = results['word usage']['conjunction']
    pronoun = results['word usage']['pronoun']
    preposition = results['word usage']['preposition']
    nominalization = results['word usage']['nominalization']
    
    pronoun_b = results['sentence beginnings']['pronoun']
    interrogative = results['sentence beginnings']['interrogative']
    article = results['sentence beginnings']['article']
    subordination = results['sentence beginnings']['subordination']
    conjunction_b = results['sentence beginnings']['conjunction']
    preposition_b = results['sentence beginnings']['preposition']

    
    return [chars_per_word, syll_per_word, words_per_sent,
            kincaid, ari, coleman_liau, flesch, gunning_fog, lix, smog, rix, dale_chall,
            tobeverb, auxverb, conjunction, pronoun, preposition, nominalization,
            pronoun_b, interrogative, article, subordination, conjunction_b, preposition_b,complex_words,long_words,
           log_chars_per_word,log_syll_per_word,log_words_per_sent,log_complex_words,log_long_words]


# In[18]:


def spacy_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    
    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])
        
    return vectors

def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")
        
    return names


# In[19]:


def tf_idf_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    
    df['excerpt2']=df['excerpt'].apply(lambda x:preprocess_text(x))
    df['excerpt2']=df['excerpt2'].apply(lambda x:remove_punctuation(x))
    df['excerpt2']=df['excerpt2'].apply(lambda x:rem_stop(x))
    
    tfidf_w2v_excerpt_train = TF_IDF_W2V(df['excerpt2'])
    vectors = np.array(tfidf_w2v_excerpt_train)
        
    return vectors

def get_tf_idf_col_names():
    names = list()
    for i in range(300):
        names.append(f"tf_idf_{i}")
        
    return names


# In[20]:


#get_tf_idf_col_names()


# In[21]:


def pos_tag_features(passage: str):
    """
    This function counts the number of times different parts of speech occur in an excerpt
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
    
    tags = pos_tag(word_tokenize(passage))
    tag_list= list()
    
    for tag in pos_tags:
        tag_list.append(len([i[0] for i in tags if i[1] == tag]))
    
    return tag_list


# In[22]:


def generate_other_features(passage: str):
    """
    This function is where I test miscellaneous features
    This is experimental
    """
    # punctuation count
    periods = passage.count(".")
    commas = passage.count(",")
    semis = passage.count(";")
    exclaims = passage.count("!")
    questions = passage.count("?")
    
    # Some other stats
    num_char = len(passage)
    num_words = len(passage.split(" "))
    unique_words = len(set(passage.split(" ") ))
    word_diversity = unique_words/num_words
    
    word_len = [len(w) for w in passage.split(" ")]
    longest_word = np.max(word_len)
    avg_len_word = np.mean(word_len)
    
    return [periods, commas, semis, exclaims, questions,
            num_char, num_words, unique_words, word_diversity,
            longest_word, avg_len_word]


# In[23]:


def create_folds(data: pd.DataFrame, num_splits: int, seed=42):
    """ 
    This function creates a kfold cross validation system based on this reference: 
    https://www.kaggle.com/abhishek/step-1-create-folds
    """
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


# In[24]:


class CLRDataset:
    """
    This is my CommonLit Readability Dataset.
    By calling the get_df method on an object of this class,
    you will have a fully feature engineered dataframe
    """
    def __init__(self, df: pd.DataFrame, train: bool, n_folds=2):
        self.df = df
        self.excerpts = df["excerpt"]
        
        self._extract_features()
        
        if train:
            self.df = create_folds(self.df, n_folds)
        
    def _extract_features(self):
        scores_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : readability_measurements(p)).tolist(), 
                                 columns=["chars_per_word", "syll_per_word", "words_per_sent",
                                          "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
                                          "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization",
                                          "pronoun_b", "interrogative", "article", "subordination", "conjunction_b",
                                          "preposition_b","complex_words","long_words",
                                         "log_chars_per_word","log_syll_per_word","log_words_per_sent","log_complex_words","log_long_words"])
        self.df = pd.merge(self.df, scores_df, left_index=True, right_index=True)
        
        spacy_df = pd.DataFrame(spacy_features(self.df), columns=get_spacy_col_names())
        self.df = pd.merge(self.df, spacy_df, left_index=True, right_index=True)
        
        #tf_id = pd.DataFrame(tf_idf_features(self.df), columns=get_tf_idf_col_names())
        #self.df = pd.merge(self.df, tf_id, left_index=True, right_index=True)
        
        add_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : number_sentence(p)).tolist(),
                              columns=["log_number_sentence","number_sentence","difficult_words","log_difficult_words",
                                      "unique_toker_per_text","log_text_len","text_len","log_unique_toker_per_text"])
        self.df = pd.merge(self.df, add_df, left_index=True, right_index=True)
        
        pos_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : pos_tag_features(p)).tolist(),
                              columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                                       "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                                       "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
        self.df = pd.merge(self.df, pos_df, left_index=True, right_index=True)
        
        other_df = pd.DataFrame(self.df["excerpt"].apply(lambda p : generate_other_features(p)).tolist(),
                                columns=["periods", "commas", "semis", "exclaims", "questions",
                                         "num_char", "num_words", "unique_words", "word_diversity",
                                         "longest_word", "avg_len_word"])
        self.df = pd.merge(self.df, other_df, left_index=True, right_index=True)
        
    def get_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        pass


# In[25]:


dataset = CLRDataset(train_df, train=True)
df = dataset.get_df()
df.head() # train dataframe


# In[26]:


notplot = ['id', 'url_legal','license','excerpt']

#for col in df.columns:
#    if col not in notplot:
#        df.plot.hist(col)


# In[27]:


plt.hist(df["chars_per_word"])


# In[28]:


# This is just here to investigate different features
plt.scatter((df["flesch"]), df["target"])
plt.show()


# In[29]:


test_dataset = CLRDataset(test_df, train=False)
test_df = test_dataset.get_df()
test_df.head(2) # test dataframe


# In[30]:


test_df


# # Modeling

# In[31]:


def set_seed(seed=42):
    """ Sets the Seed """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
set_seed(42)


# In[32]:


features = ["chars_per_word", "syll_per_word", "words_per_sent",
            "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
            "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization", 
            "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b",
            "number_sentence","log_number_sentence","difficult_words","complex_words","long_words","log_difficult_words"
            ,"log_text_len", "text_len","log_chars_per_word","log_syll_per_word","log_words_per_sent","log_complex_words","log_long_words"
           ]
           
features+=get_spacy_col_names()
#features+=get_tf_idf_col_names()
features+=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",  "MD", 
            "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
            "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
#features+=["avg_len_word"]
#features+= [#"periods", "commas", "semis", "exclaims", "questions",
            #"num_char", "num_words", "unique_words", 
#    "word_diversity",
            #"longest_word", "avg_len_word"
 #          ]
# currently better results without the other_df features


# In[33]:


""" I normalize the data here, could be useful depending on your model"""
#scaler = MinMaxScaler()
#df[features] = scaler.fit_transform(df[features])
#test_df[features] = scaler.transform(test_df[features])


# In[34]:


def train_pred_one_fold(model_name: str, fold: int, df: pd.DataFrame, test_df: pd.DataFrame, features: list, rmse: list):
    """
    This function trains and predicts on one fold of your selected model
    df is the train df, test_df is the test_df
    X features are defined in features
    y output is target
    oof score is printed and stored in the rmse list
    """
    train = df[df.kfold == fold]
    X_train = train[features]
    y_train = train["target"]
 
    valid = df[df.kfold != fold]
    X_valid = valid[features]
    y_valid = valid["target"]
    
    X_test = test_df[features]

    if model_name == 'ridge':
        model = Ridge(alpha=.7)    
        model.fit(X_train, y_train)
        oof = model.predict(X_valid)
        print(np.sqrt(mean_squared_error(y_valid, oof)))
        rmse.append(np.sqrt(mean_squared_error(y_valid, oof)))
        test_preds = model.predict(X_test)
        
    elif model_name == 'Gradiend':
        model = GradientBoostingRegressor(n_estimators=500,max_depth=2)    
        model.fit(X_train, y_train)
        oof = model.predict(X_valid)
        print(np.sqrt(mean_squared_error(y_valid, oof)))
        rmse.append(np.sqrt(mean_squared_error(y_valid, oof)))
        test_preds = model.predict(X_test)
        
    else:
        test_preds = 0
        raise Exception("Not Implemented")
        
    return test_preds


# In[35]:


def train_pred(model_name: str, df: pd.DataFrame, test_df: pd.DataFrame, features: list):
    """
    This function trains and predicts multiple fold using train_pred_one_fold
    The average rmse is printed the the test data predictions are returned
    The last column is the average result from all folds to be submitted
    """
    print(f"model_name: {model_name}")
    all_preds = pd.DataFrame()
    rmse = list()
    for f in range(2):
        all_preds[f"{model_name}_{f}"] = train_pred_one_fold(model_name, f, df, test_df, features, rmse)

    all_preds[f"{model_name}"] = all_preds.mean(axis=1)
    print("---------")
    print(f"avg rmse: {np.mean(rmse)}")
    return all_preds


# In[36]:


def prep_sub(preds: pd.DataFrame, col_name: str):
    """
    This function takes an output prediction df from train_pred
    and sets it to a format that can be submitted to the competition
    """
    sub = pd.read_csv("../input/commonlitreadabilityprize/sample_submission.csv")
    sub["target"] = preds[col_name]
    sub.to_csv("submission.csv", index=False)


# In[37]:


#df = dataset.get_df()


# In[38]:


#ridge_preds = train_pred('ridge', df, test_df, features)
#ridge_preds


# In[39]:


preds =  []
for i in range(10):
    n_df = create_folds(df, num_splits=2, seed=i)
    #print(n_df.kfold.value_counts())

    ridge_preds = train_pred('ridge', n_df, test_df, features)
    #print(ridge_preds)
    preds.append(np.array(ridge_preds['ridge']))

pred_ridge = np.mean(preds,axis=0)


# In[40]:


preds_g =  []
for i in range(10):
    n_df = create_folds(df, num_splits=2, seed=i)
    #print(n_df.kfold.value_counts())

    ridge_preds = train_pred('Gradiend', n_df, test_df, features)
    #print(ridge_preds)
    preds_g.append(np.array(ridge_preds['Gradiend']))

pred_gradien = np.mean(preds_g,axis=0)


# In[41]:


#gradien_pres = train_pred('Gradiend', df, test_df, features)
#gradien_pres


# In[42]:


np.mean(preds,axis=0)


# In[43]:


target = pred_gradien*0.3 + pred_ridge*0.7
target


# In[44]:


df_test = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')

df_test


# In[45]:


df_test['target'] = target

df_test[['id','target']].to_csv('submission.csv',index = False)


# In[46]:


df_test


# In[ ]:





# In[ ]:




