#!/usr/bin/env python
# coding: utf-8

# ## NLP with disaster tweets:
# NLP with disaster tweets is the beginner's dataset for learning and applying NLP techniques. In this notebook, we will explore several nlp applications and libraries. Here is a list of things we have covered in this notebook:<br/>
# ### sections:
# (1) [basic data exploration](#section1)<br/>
# We go over the different features present in the dataset. Some of the codes are commented out currently; which you can uncomment and explore. In this part, we normally explore the dataset and do some basic transformations.<br/>
# (2) [extensive text cleaning](#section2)<br/>
# In this part, we import NLTK( a text cleaning library), beautifulsoup, regex( text string manipulation library) and then perform extensive text cleaning. We have taken part of the cleaning code from another high score(0.84) notebook. If you are just starting out in NLP, this part will give you a good lesson on text cleaning as well as you can reuse some of this code in normal NLP also.<br/>
# (3)[feature-generation](#section3)<br/>
# In this part, we explore the location feature and generate several features out of it to capture extra information about the tweet's location.<br/>
# (4)[NLTK frequency analysis and tf-idf](#section4)<br/>
# In this part, we have used the NLTK library to find out the top n words appearing in each of the classes; i.e. disaster tweets and non-disaster tweets. Top frequency words make a very good feature in identifying text classes; as their presence denotes high signal for the class in which they occur. Check this part's code to understand in details.<br/>
# We have also sklearn's tfidf vectorizer to generate tfidf features from keyword, location and main text. check the code to understand how we have done it.<br/>
# (5)[vector embedding creation using spacy](#section5)<br/>
# In this part, we use spacy's large english model to create 300 dimensional vector embedding for the twitter texts; and add these as feature to the dataset.<br/>
# (6)[Modeling with random forest, naive bayes and xgboost](#section6)<br/>
# In this part, we have done modeling and rough fine tuning with random forest; added commented code and experimentation guidance for naive-bayes and finally trained and created submission file using xgboost model.<br/>
# ### Resources:
# (1) [Deep understanding of tfidf vectorizer](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)<br/>
# (2) [introduction to spacy](https://shyambhu20.blogspot.com/2020/09/introduction-to-spacy-basic-nlp-usage.html)<br/>
# (3)[xgboost modeling documentation](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)<br/>
# (4)[sklearn naive bayes documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)<br/>
# (5)[why use naive bayes for text classification](https://monkeylearn.com/text-classification-naive-bayes/)<br/>
# (6) [how to create inbound links in kaggle notebook](https://sebastianraschka.com/Articles/2014_ipython_internal_links.html#top)

# In[1]:


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


# In[2]:


train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train_data.shape)
print(train_data.columns)
print(test_data.shape)
train_data.head()


# In[3]:


train_data = train_data.drop('id',axis = 1)
test_data = test_data.drop('id',axis = 1)


# In[4]:


test_data = test_data.fillna('')
train_data = train_data.fillna('')


# In[5]:


#keywords = list(train_data['keyword'].unique())
#print(keywords)


# ## <a id = section1>Basic data exploration</a>
# In this part, we will read keyword, location and texts; do some preliminary cleaning. For example, on printing keyword in the above code block you can see that in keyword, gaps are filled with %20 sign. So in this next function we are cleaning the keywords by replacing that with space.

# In[6]:


import re
def keyword_correction(x):
    try:
        x = x.split("%20")
        x = ' '.join(x)
        return x
    except:
        return x


# In[7]:


train_data['keyword'] = train_data['keyword'].apply(lambda x: keyword_correction(x))
test_data['keyword'] = test_data['keyword'].apply(lambda x: keyword_correction(x))


# In[8]:


#train_data['keyword'].unique()


# In[9]:


#list(train_data['location'].unique())


# ## <a id ='section2'>Extensive Text cleaning</a>
# In this part, we will follow some extensive text cleaning, part of which we have taken from [this awesome notebook](https://www.kaggle.com/nxhong93/tweet-predict1). 

# In[10]:


from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
def text_cleaning(text):
    forbidden_words = set(stopwords.words('english'))
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() if word not in forbidden_words]
        return text
    return []
#clean data
#this following cleaning is taken from https://www.kaggle.com/nxhong93/tweet-predict1
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"}

puncts = puncts + list(string.punctuation)

def clean_text(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('\d+', ' ', x)
    return x


def replace_typical_misspell(text):
    mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    def replace(match):
        return mispell_dict[match.group(0)]

    return mispellings_re.sub(replace, text)

def remove_space(string):
    string = BeautifulSoup(string).text.strip().lower()
    string = re.sub(r'((http)\S+)', 'http', string)
    string = re.sub(r'\s+', ' ', string)
    return string


def clean_data(df, columns: list):
    
    for col in columns:
        df[col] = df[col].apply(lambda x: remove_space(x).lower())        
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
        df[col] = df[col].apply(lambda x: clean_text(x))
        
    return df


# In[11]:


for col in ['location','text']:
    train_data[col] = train_data[col].apply(lambda x: ' '.join(text_cleaning(x)))
    test_data[col] = test_data[col].apply(lambda x: ' '.join(text_cleaning(x)))
train_data = clean_data(train_data,['keyword','text'])
test_data = clean_data(test_data,['keyword','text'])


# In[12]:


import spacy
nlp = spacy.load('en_core_web_lg')


# ## <a id = 'section3'>feature-generation</a>
# As we have seen and explored the location feature, we will generate a few features out of that to capture information.<br/>
# So location seems to have legit locations, as well as garbage words. So we will detect if there is entities in the location, as legit location should give more credibility to the tweets.

# In[13]:


def location_detection(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent)
    if len(entities)>0:
        return 1
    return 0


# In[14]:


train_data['original_locations'] = train_data['location'].apply(lambda x: location_detection(x))
test_data['original_locations'] = test_data['location'].apply(lambda x: location_detection(x))


# In[15]:


#list(train_data['location'].unique())


# there seems to be web addresses available as locations as well. We will create a feature to capture if there is any web address in the location text. On reading the locations manually, www, amazon, youtube, twitch,gmail are some associated words to website addresses.<br/>
# Also, most spammy locations contain these words: <br/>
# place, room, home, somewhere, dope,nowhere,location,kidding,moon,searching bae,gotham city,wherever,5th dimension,anywhere,idn,spying thoughts,beside,happily married 2 kids,'c h c g','playa','visit youtube channel',fvck,fuck,world.<br/>
# Some of these are too specific, but other words we will use as bag of words for signalling spam location.<br/>
# social media locations are also mentioned:<br/>
# usually the locations are mentioned with instagram, snapchat.<br/>
# Also, another thing is that, if location is full numeric, then it maybe very well be latitude and longitudes.<br/>

# In[16]:


spam_locations = ['place','room','home','somewhere','nowhere','everywhere','location',
                  'dope','kidding','moon','wherever','dimension','world','fvck','fuck','beside']
def is_location_spammy(text):
    for word in spam_locations:
        if word in text:
            return 1
    return 0


# In[17]:


train_data['Is_location_spam'] = train_data['location'].apply(lambda x: is_location_spammy(x))
test_data['Is_location_spam'] = test_data['location'].apply(lambda x: is_location_spammy(x))


# Also, as we noted if all the words in the location are numbers, then it can be a latitude longitude point. As there are different version of that, we will count the number of digit tokens, as well as we will count if the whole text is just digits.

# In[18]:


def digit_counter(text):
    """detects any digit in any token and counts
       once par token."""
    sum_number = 0
    doc = nlp(text)
    for token in doc:
        sum_number += bool(re.search(r'\d', token.text))*1
    return sum_number    


# In[19]:


train_data['digit_count_location'] = train_data['location'].apply(lambda x: digit_counter(x))
test_data['digit_count_location'] = test_data['location'].apply(lambda x: digit_counter(x))


# ## <a id = 'section4'>NLTK frequency analysis and TF-IDF feature generation</a>
# Now that we have sort of exhausted the location feature; let's check the original tweets. We will first check the different frequent words occuring in disaster tweets vs non-disaster tweets. We will use NLTK's freqdist() function to do this. Then we will generate the tfidf features using sklearn's tfidf vectorizer.

# In[20]:


disaster_tweets =' '.join(train_data[train_data['target'] == 1]['text'].tolist())
non_disaster_tweets = ' '.join(train_data[train_data['target'] == 0]['text'].tolist())


# In[21]:


import nltk
def return_top_words(text,words = 10):
    allWords = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)    
    mostCommontuples= allWordExceptStopDist.most_common(words)
    mostCommon = [tupl[0] for tupl in mostCommontuples]
    return mostCommon


# In[22]:


top_50_disaster_words = return_top_words(disaster_tweets,50)
top_50_nondisaster_words = return_top_words(non_disaster_tweets,50)


# In[23]:


#top_50_disaster_words


# In[24]:


#top_50_nondisaster_words


# So there are some more negative words in the disaster text, and words in non-disaster text are less negative. We will now check what are the top words not occurring in the other categories. For that we will use set differences with top 400 words. The name of the variable is top_200 as initially I created using top 200 words; but to increase accuracy I included 400. You will have to reset the values according to your vocabulary size for the problem in hand.

# In[25]:


top_200_disaster_words = return_top_words(disaster_tweets,400)
top_200_nondisaster_words = return_top_words(non_disaster_tweets,400)
top_disaster_exclusive = list(set(top_200_disaster_words).difference(set(top_200_nondisaster_words)))
top_nondisaster_exclusive = list(set(top_200_nondisaster_words).difference(set(top_200_disaster_words)))


# In[26]:


#top_disaster_exclusive


# In[27]:


#top_nondisaster_exclusive


# In[28]:


total_vocab = top_disaster_exclusive + top_nondisaster_exclusive


# In[29]:


for word in total_vocab:
    train_data['Is_'+word+'_present'] = train_data['text'].apply(lambda x: (word in x)*1)
    test_data['Is_'+word+'_present'] = test_data['text'].apply(lambda x: (word in x)*1)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         max_features = 5000,
                         smooth_idf=False)
X_train_tfidf = tf_idf.fit_transform(train_data['text'])
X_test_tfidf = tf_idf.transform(test_data['text'])
tf_kw = TfidfVectorizer(ngram_range = (1,2),
                        binary = True,
                        max_features = 1500,
                        smooth_idf = False)
kw_train_tfidf = tf_kw.fit_transform(train_data['keyword'])
kw_test_tfidf = tf_kw.transform(test_data['keyword'])
tf_location = TfidfVectorizer(ngram_range = (1,2),
                              binary = True,
                              max_features = 1500,
                              smooth_idf = False)
location_train_tfidf = tf_location.fit_transform(train_data['location'])
location_test_tfidf = tf_location.transform(test_data['location'])


# In[31]:


train_data = pd.concat([train_data,pd.DataFrame(X_train_tfidf.toarray(),
                                                columns = ['text_contains_'+ str(text) for text in tf_idf.get_feature_names()]),
                        pd.DataFrame(kw_train_tfidf.toarray(),
                                     columns = ['keyword_contains_'+str(text) for text in tf_kw.get_feature_names()]),
                        pd.DataFrame(location_train_tfidf.toarray(),
                                     columns = ['location_contains_'+str(text) for text in tf_location.get_feature_names()])],axis = 1)
test_data = pd.concat([test_data,pd.DataFrame(X_test_tfidf.toarray(),
                                              columns = ['text_contains_'+ str(text) for text in tf_idf.get_feature_names()]),
                       pd.DataFrame(kw_test_tfidf.toarray(),
                                    columns = ['keyword_contains_'+str(text) for text in tf_kw.get_feature_names()]),
                       pd.DataFrame(location_test_tfidf.toarray(),
                                    columns = ['location_contains_'+str(text) for text in tf_location.get_feature_names()])],axis = 1)


# In[32]:


#for col in train_data.columns:
#    if col == 'text':
#        print(train_data[col].describe())


# ## <a id='section5'>vector embedding</a>
# In this section we create 300 dimensional vector embedding from the text feature; and include these as 300 features to the original dataset. Reason we include vector embedding is to give a sense of original meaning; and to gather any semantic structure in the disaster tweets if present. Here we are using spacy's en_core_web_lg model's embeddings. For learning more about vector embeddings using spacy, read [this post](https://shyambhu20.blogspot.com/2020/10/calculate-word-similarity-spacy-nlp.html).<br/>

# In[33]:


def create_vec(dataframe):
    texts = dataframe['text'].tolist()
    vectors = []
    for doc in nlp.pipe(texts):
        vectors.append(list(doc.vector))
    df = pd.DataFrame(vectors,columns = ['vec_'+str(i) for i in range(300)])
    return df
vec_train = create_vec(train_data)
vec_test = create_vec(test_data)
train_data = pd.concat([train_data,vec_train],axis = 1)
test_data = pd.concat([test_data,vec_test],axis = 1)


# In[34]:


train_data = train_data.drop(['keyword','location','text'],axis = 1)
test_data = test_data.drop(['keyword','location','text'],axis = 1)


# In[35]:


X_train = train_data.drop('target',axis = 1)
Y_train = train_data['target']
print('target' in test_data.columns)


# ## <a id='section6'>Extensive Modeling zone</a>
# In this portion, we will try out random forest classifier model, which achieves around 77% accuracy with fine-tuning.<br/>
# Then we have the code for naive bayes but it is commented out, as with vector embeddings having negative values; naive bayes can't run. If you do want to run and experiment it, use maxabscaler() to scale all the features as to non-negative values and then you can run it. <br/>
# Finally, we will run the xgboost code. We have not done much xgboost fine-tuning, but you can experiment on that to try and get a higher score in it.<br/>

# In[36]:


len(train_data.columns)


# In[37]:


from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report
forest = rfc(n_estimators = 128,max_depth = 8,min_samples_split = 15,
             class_weight = {0:1,1:1.6},oob_score = True)
forest.fit(X_train,Y_train)
print(forest.oob_score_)
Y_pred_train = forest.predict(X_train)
print(classification_report(Y_pred_train,Y_train))


# So we can approach a 77% accuracy using these features. Let's check the feature importances for the features and drop the low performing features.

# In[38]:


#features = list(X_train.columns)
#feature_importances = forest.feature_importances_
#data = pd.DataFrame()
#data['features'] = features
#data['feature_importances'] = feature_importances
#data = data.sort_values(by = 'feature_importances',ascending = False)
#print(data)


# In[39]:


#bad_features = data[data['feature_importances']<0.001]['features'].tolist()


# In[40]:


#X_train_reduced = X_train.drop(bad_features,axis = 1)
#test_data_reduced = test_data.drop(bad_features,axis = 1)


# In[41]:


#X_train_reduced.shape


# In[42]:


#forest = rfc(n_estimators = 128,max_depth = 5,min_samples_split = 15,
#             class_weight = {0:1,1:1.53},
#             oob_score = True)
#forest.fit(X_train_reduced,Y_train)
#print(forest.oob_score_)
#Y_pred_train = forest.predict(X_train_reduced)
#print(classification_report(Y_pred_train,Y_train))


# In[43]:


#taken from https://www.kaggle.com/vishalsiram50/fine-tuning-bert-88-accuracy
from sklearn.model_selection import StratifiedKFold, cross_val_score

def get_auc_CV(model,X_train,Y_train):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(
        model, X_train, Y_train, scoring="roc_auc", cv=kf)

    return auc.mean()


# We have added the vector embedding with negative values as features. That's why we can't use naive bayes anymore; as that only allows non-negative values of features. You can use maxabsscaler as we suggested above and try out this part. Also tune the naivebayes for alpha value. 

# In[44]:


#from sklearn.naive_bayes import MultinomialNB as MNB
#from sklearn.metrics import classification_report
#for alpha in [0.001,0.1,1]:
#    print(alpha)
#    clf = MNB(alpha = alpha)
#    auc = get_auc_CV(clf,X_train,Y_train)
#    print(auc)
#    clf.fit(X_train,Y_train)
#    Y_pred_train = clf.predict(X_train)
#    print(classification_report(Y_train,Y_pred_train))


# so 0.1 has the highest auc, so we will go with alpha = 0.1. Let's train this model and create submission.

# In[45]:


#clf = MNB(alpha = 0.1)
#clf.fit(X_train,Y_train)
#Y_pred_train = clf.predict(X_train)
#print(classification_report(Y_train,Y_pred_train))


# Let's train an Xgboost model with this data too.

# In[46]:


#code taken from https://www.kaggle.com/lucidlenn/data-analysis-and-classification-using-xgboost
import time
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
xgb = XGBClassifier(n_estimators=200,learning_rate = 0.2,max_depth = 8)
training_start = time.perf_counter()
xgb.fit(X_train, Y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
pred_final = xgb.predict(test_data)
pred_train = xgb.predict(X_train)
print(classification_report(Y_train,pred_train))
prediction_end = time.perf_counter()
#acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
#print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))


# In[47]:


get_auc_CV(xgb,X_train,Y_train)


# In[48]:


#test_data.isna().sum().sum()


# In[49]:


#test_data.shape


# In[50]:


#test_prediction = clf.predict(test_data)


# In[51]:


sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print(sample_submission.columns)


# In[52]:


dataframe = pd.DataFrame()
dataframe['id'] = sample_submission['id']
dataframe['target'] = pred_final
dataframe.to_csv("final_submission.csv",index = False)

