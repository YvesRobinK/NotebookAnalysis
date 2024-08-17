#!/usr/bin/env python
# coding: utf-8

# ## Fundamental and Intuitive note on Bayes Theorem
# 
# ![Imgur](https://imgur.com/brkFzR6.png)
# 
# Same formulae with some notes
# 
# ![Imgur](https://imgur.com/pU0s86X.png)
# 
# To understand why Bayes’ theorem is so important, let’s look at a general form of this problem. Our beliefs describe the world we know, so when we observe something, its conditional probability represents the likelihood of what we’ve seen given what we believe, or:
# 
# #### P(observed | belief)
# 
# 
# For example, suppose you believe in climate change, and therefore you expect that the area where you live will have more droughts than usual over a 10-year period. Your belief is that climate change is taking place, and our observation is the number of droughts in our area; let’s say there were 5 droughts in the last 10 years. Determining how likely it is that you’d see exactly 5 droughts in the past 10 years if there were climate change during that period may be difficult. One way to do this would be to consult an expert in climate science and ask them the probability of droughts given that their model assumes climate change.
# 
# At this point, all we’ve done is ask, “What is the probability of what I’ve observed, given that I believe climate change is true?” But what you want is some way to quantify how strongly you believe climate change is really happening, given what you have observed. Bayes’ theorem allows you to reverse P(observed | belief), which you asked the climate scientist for, and solve for the likelihood of our beliefs given what you’ve observed, or:
# 
# 
# #### P(belief | observed)
# 
# 
# In this example, Bayes’ theorem allows us to transform our observation of five droughts in a decade into a statement about how strongly you believe in climate change after you have observed these droughts. The only other pieces of information you need are the general probability of 5 droughts in 10 years (which could be estimated with historical data) and our initial certainty of our belief in climate change. And while most people would have a different initial probability for climate change, Bayes’ theorem allows you to quantify exactly how much the data changes any belief.
# 
# For example, if the expert says that 5 droughts in 10 years is very likely if we assume that climate change is happening, most people will change their previous beliefs to favor climate change a little, whether they’re skeptical of climate change or they’re Al Gore.
# However, suppose that the expert told you that in fact, 5 droughts in 10 years was very unlikely given our assumption that climate change is happening. In that case, our prior belief in climate change would weaken slightly given the evidence. The key takeaway here is that Bayes’ theorem ultimately allows evidence to change the strength of our beliefs.
# 
# Bayes’ theorem allows us to take our beliefs about the world, combine them with data, and then transform this combination into an estimate of the strength of our beliefs given the evidence we’ve observed. Very often our beliefs are just our initial certainty in an idea; this is the P(A) in Bayes’ theorem. We often debate topics such as whether gun control will reduce violence, whether increased testing increases student performance, or whether public health care will reduce overall health care costs. But we seldom think about how evidence should change our minds or the minds of those we’re debating. Bayes’ theorem allows us to observe evidence about these beliefs and quantify exactly how much this evidence changes our beliefs.
# 
# ## Simplest Proof of Bayes Theorem
# 
# Let’s consider two probabilistic events A and B. We can correlate the marginal probabilities P(A) and P(B) with the conditional probabilities P(A|B) and P(B|A) using the product rule:
# 
# ![Imgur](https://imgur.com/1fGQOXC.png)
# 
# First of all, let’s consider the marginal probability P(A): this is normally a value that determines how probable a target event is, like P(Spam) or P(Rain). As there are no other elements, this kind of probability is called Apriori, because it’s often determined by mathematical considerations or simply by a frequency count. For example, imagine we want to implement a very simple spam filter and we’ve collected 100 emails. We know that 30 are spam and 70 are regular. So we can say that P(Spam) = 0.3.
# 
# However, we’d like to evaluate using some criteria (for simplicity, let’s consider a single one), for example, e-mail text is shorter than 50 characters. Therefore, our query becomes:
# 
# ---
# 
# ### The time complexity of Naive Bayes
# 
# Let, n = no of data points, d = no of features in an input vector x. and c = number of classes.
# 
# So, if we have ‘d’ features then we calculate ‘d’ likelihoods for 1 class. Then for ‘c’ class **d \* c** likelihood probabilities
# 
# #### T (n) = O (ndc)
# 
# All it needs to do is computing the frequency of every feature value di for each class.
# 
# If d is small we can neglect the presence of d and we can tell O(n).
# 
# #### For space complexity, we store only the likelihoods for ‘c’ classes,
# 
# S(n) = O(dc)
# 
# For Testing phase : Since all the likelihood probabilities are already calculated in training phase then at
# 
# the time of evaluation we just need to lookup.
# 
# #### S(n) = O(dc)
# 
# ---
# 
# ### Difference between naive Bayes & multinomial naive Bayes
# 
# In general, to train Naive Bayes for n-dimensional data, and k classes you need to estimate $P(x_i | c_j)$ for each $1 \leq i \leq n$, $1 \leq j \leq k$ . You can assume any probability distribution for any pair $(i,j)$ (although it's better to not assume discrete distribution for $P(x_i|c_{j_1})$ and continuous for $P(x_i | c_{j_2})$). You can have Gaussian distribution on one variable, Poisson on other and some discrete on yet another variable.
# 
# #### Multinomial Naive Bayes simply assumes multinomial distribution for all the pairs, which seem to be a reasonable assumption in some cases, i.e. for word counts in documents.
# 
# ---
# 
# The general term **Naive Bayes** refers the the strong independence assumptions in the model, rather than the particular distribution of each feature. A Naive Bayes model assumes that each of the features it uses are conditionally independent of one another given some class. More formally, if I want to calculate the probability of observing features $f_1$ through $f_n$, given some class c, under the Naive Bayes assumption the following holds:
# 
# $$ p(f_1,..., f_n|c) = \prod_{i=1}^n p(f_i|c)$$
# 
# This means that when I want to use a Naive Bayes model to classify a new example, the posterior probability is much simpler to work with:
# 
# $$ p(c|f_1,...,f_n) \propto p(c)p(f_1|c)...p(f_n|c) $$
# 
# Of course these assumptions of independence are rarely true, which may explain why some have referred to the model as the "Idiot Bayes" model, but in practice Naive Bayes models have performed surprisingly well, even on complex tasks where it is clear that the strong independence assumptions are false.
# 
# Up to this point we have said nothing about the distribution of each feature. In other words, we have left $p(f_i|c)$ undefined. The term **Multinomial Naive Bayes** simply lets us know that each $p(f_i|c)$ is a multinomial distribution, rather than some other distribution. This works well for data which can easily be turned into counts, such as word counts in text.
# 
# In summary, Naive Bayes classifier is a general term which refers to conditional independence of each of the features in the model, while Multinomial Naive Bayes classifier is a specific instance of a Naive Bayes classifier which uses a multinomial distribution for each of the features.
# 
# References:
# 
# Stuart J. Russell and Peter Norvig. 2003. Artificial Intelligence: A Modern Approach (2 ed.). Pearson Education. _See p. 499 for reference to "idiot Bayes" as well as the general definition of the Naive Bayes model and its independence assumptions_
# 
# ---
# 
# ## Note on Bag of Words
# 
# The name “bag-of-words” comes from the algorithm simply seeking to know the number of times a given word is present within a body of text. The order or context of the words is not analyzed here. Similarly, if we have a bag filled with six pencils, eight pens, and four notebooks, the algorithm merely cares about recording the number of each of these objects, not the order in which they are found, or their orientation. You typically want to use the bag-of-words feature extraction technique for document classification. Why is this the case? We assume that documents of certain classifications contain certain
# words. For example, we expect a document referencing political science to perhaps feature jargon such as dialectical materialism or free market capitalism; whereas a document that is referring to classical music will
# have terms such as crescendo, diminuendo, and so forth. In these instances of document classification, the location of the word itself is not terribly important. It’s important to know what portion of the vocabulary is present in one class of document vs. another.
# 
# 

# In[1]:


# Start of Google Colab Import related codes
# Keep these lines commented out in Local Drive and in Kaggle

# from google.colab import drive
# drive.mount('/content/drive')
# import pandas as pd
# import os
# # Navigate into Drive where you want to store your Kaggle data
# os.chdir('/content/drive/MyDrive/Kaggle_Datasets')

# !pip install kaggle

# The private folder in G-Drive where my Kaggle API Token is saved - kaggle.json
# os.environ['KAGGLE_CONFIG_DIR']='/content/drive/MyDrive/Kaggle_Datasets'

# this is the copied API command, the data will download to the current directory
# !kaggle competitions download -c donorschoose-application-screening

# End of Google Colab Import related codes

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import StandardScaler, Normalizer
from io import StringIO
import requests
import pickle
from tqdm import tqdm
import os
# from plotly import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from collections import Counter
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from collections import Counter
from wordcloud import WordCloud


# In[2]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/donorschooseorg-application-screening/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1.1 Reading Data

# In[3]:


# Start of Google Colab related Imports from G-Drive
# Keep this commented out in Local Drive and in Kaggle
# train_project_data_original = pd.read_csv("./train.zip")
# test_project_data_original = pd.read_csv('./test.zip')
# resource_data = pd.read_csv('./resources.zip')
# End of Google Colab related Imports from G-Drive

# Start of Deepnote related Imports
# test_project_data_original = pd.read_csv('/work/test.zip')
# train_project_data_original = pd.read_csv ('/work/train.zip')
# resource_data = pd.read_csv('/work/resources.zip')
# End of Deepnote related Imports

# Imports for both Kaggle and Local Machine
train_project_data_original = pd.read_csv("../input/donorschooseorg-application-screening/train.csv")
test_project_data_original = pd.read_csv('../input/donorschooseorg-application-screening/test.csv')
resource_data = pd.read_csv('../input/donorschooseorg-application-screening/resources.csv')

# Reading smaller set of data for doing just experimentation
# train_project_data_original = pd.read_csv("../input/donorschoose-application-screening/train.zip", nrows=200)
# test_project_data_original = pd.read_csv('../input/donorschoose-application-screening/test.zip', nrows=200)
# resource_data = pd.read_csv('../input/donorschoose-application-screening/resources.zip', nrows=200)

print('Column names from train_project_data_original is : ', train_project_data_original.columns )
train_project_data_original.head()


# In[4]:


print('Shape of train_project_data_original: ', train_project_data_original.shape)
print('Shape of test_project_data_original: ', test_project_data_original.shape)
# Test data will NOT have any column for 'project_is_approved'


# In[5]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]



# ## 2. Pre-processing Categorical Features: project_grade_category

# In[6]:


train_project_data_original['project_grade_category'].value_counts()


# In[7]:


# Removing all the spaces, replace the '-' with '_' and convert all the letters to small

train_project_data_original['project_grade_category'] = train_project_data_original['project_grade_category'].str.replace(' ','_').str.replace('-', '_')
train_project_data_original['project_grade_category'] = train_project_data_original['project_grade_category'].str.lower()
train_project_data_original['project_grade_category'].value_counts()


# In[8]:


# Same above text-Preprocessing for test dataset -
test_project_data_original['project_grade_category'] = test_project_data_original['project_grade_category'].str.replace(' ','_').str.replace('-', '_')
test_project_data_original['project_grade_category'] = test_project_data_original['project_grade_category'].str.lower()
test_project_data_original['project_grade_category'].value_counts()


# ## Preprocessing Categorical Features: project_subject_categories

# In[9]:


train_project_data_original['project_subject_categories'].value_counts()


# In[10]:


# remove spaces, 'the'
# replace '&' with '_', and ',' with '_'
train_project_data_original['project_subject_categories'] = train_project_data_original['project_subject_categories'].str.replace(' The ','').str.replace(' ','').str.replace('&','_').str.replace(',','_')
train_project_data_original['project_subject_categories'] = train_project_data_original['project_subject_categories'].str.lower()
train_project_data_original['project_subject_categories'].value_counts()


# In[11]:


# Same above text-Preprocessing for test dataset - project_subject_categories
test_project_data_original['project_subject_categories'] = test_project_data_original['project_subject_categories'].str.replace(' The ','').str.replace(' ','').str.replace('&','_').str.replace(',','_')
test_project_data_original['project_subject_categories'] = test_project_data_original['project_subject_categories'].str.lower()
test_project_data_original['project_subject_categories'].value_counts()


# ## Preprocessing Categorical Features: project_subject_subcategories

# In[12]:


train_project_data_original['project_subject_subcategories'].value_counts()


# In[13]:


# Same kind of cleaning as we did in 'project_subject_categories'

train_project_data_original['project_subject_subcategories'] = train_project_data_original['project_subject_subcategories'].str.replace(' The ','').str.replace(' ','').str.replace('&','_').str.replace(',','_').str.lower()

train_project_data_original['project_subject_subcategories'].value_counts()


# In[14]:


# Same above text-Preprocessing for test dataset: project_subject_subcategories
test_project_data_original['project_subject_subcategories'] = test_project_data_original['project_subject_subcategories'].str.replace(' The ','').str.replace(' ','').str.replace('&','_').str.replace(',','_').str.lower()

test_project_data_original['project_subject_subcategories'].value_counts()


# ## Preprocessing Categorical Features: teacher_prefix

# In[15]:


train_project_data_original['teacher_prefix'].value_counts()


# In[16]:


# check if we have any nan values are there
print(train_project_data_original['teacher_prefix'].isnull().values.any())
print("number of nan values",train_project_data_original['teacher_prefix'].isnull().values.sum())

# If there's indeed any "NA" values then fill them up
train_project_data_original['teacher_prefix']=train_project_data_original['teacher_prefix'].fillna('Mrs.')

train_project_data_original['teacher_prefix'].value_counts()


# In[17]:


# Remove '.'
# convert all the chars to small

train_project_data_original['teacher_prefix'] = train_project_data_original['teacher_prefix'].str.replace('.','')
train_project_data_original['teacher_prefix'] = train_project_data_original['teacher_prefix'].str.lower()
train_project_data_original['teacher_prefix'].value_counts()


# In[18]:


# Same above text-Preprocessing for test dataset: 'teacher_prefix'
test_project_data_original['teacher_prefix'] = test_project_data_original['teacher_prefix'].str.replace('.','')
test_project_data_original['teacher_prefix'] = test_project_data_original['teacher_prefix'].str.lower()
test_project_data_original['teacher_prefix'].value_counts()


# ## Preprocessing Categorical Features: school_state

# In[19]:


train_project_data_original['school_state'].value_counts()


# In[20]:


# convert all of them into small letters
train_project_data_original['school_state'] = train_project_data_original['school_state'].str.lower()
train_project_data_original['school_state'].value_counts()


# In[21]:


# Same above text-Preprocessing for test dataset: 'school_state'
test_project_data_original['school_state'] = test_project_data_original['school_state'].str.lower()
test_project_data_original['school_state'].value_counts()


# ## Preprocessing Categorical Features: project_title
# 
# #### First Expanding English language contractions in Python
# 
# The English language has [a couple of contractions](http://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions). For instance:
# 
#     you've -> you have
#     he's -> he is

# In[22]:


# https://stackoverflow.com/a/47091490/4084039
import re

def remove_eng_lang_contraction(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# For few other ways to remove contraction check below
# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python


# ## Stopword Removal
# 
# Stop words are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. The intuition behind using stop words is that, by removing low information words from text, we can focus on the important words instead.
# 
# For example, in the context of a search system, if your search query is “what is text preprocessing?”, you want the search system to focus on surfacing documents that talk about text preprocessing over documents that talk about what is. This can be done by preventing all words from your stop word list from being analyzed. Stop words are commonly applied in search systems, text classification applications, topic modeling, topic extraction and others.
# In my experience, stop word removal, while effective in search and topic extraction systems, showed to be non-critical in classification systems. However, it does help reduce the number of features in consideration which helps keep your models decently sized.

# In[23]:


train_project_data_original['project_title'].head(5)


# In[24]:


print("printing some random reviews before Pre-Processing ")
print(3, train_project_data_original['project_title'].values[3])
print(15, train_project_data_original['project_title'].values[15])
print(10, train_project_data_original['project_title'].values[10])


# In[25]:


# Now applying all the above pre-processing steps and functions
# Expanding English language contractions with the function we defined above
# Removing all stopwords

from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for current_sentenceence in tqdm(text_data):

        # Expanding English language contractions
        current_sentence = remove_eng_lang_contraction(current_sentenceence)
        current_sentence = current_sentence.replace('\\r', ' ')
        current_sentence = current_sentence.replace('\\n', ' ')
        current_sentence = current_sentence.replace('\\"', ' ')
        current_sentence = re.sub('[^A-Za-z0-9]+', ' ', current_sentence)

        # Removing all stopwords
        # https://gist.github.com/sebleier/554280
        current_sentence = ' '.join(e for e in current_sentence.split() if e.lower() not in stopwords)
        preprocessed_text.append(current_sentence.lower().strip())
    return preprocessed_text


# In[26]:


preprocessed_titles_train = preprocess_text(train_project_data_original['project_title'].values)

train_project_data_original['project_title'] = preprocessed_titles_train

print("printing few random reviews AFTER Pre-Processing ")
print(3, train_project_data_original['project_title'].values[3])
print(15, train_project_data_original['project_title'].values[15])
print(10, train_project_data_original['project_title'].values[10])


# In[27]:


# Same above text-Preprocessing for test dataset: 'project_title'

preprocessed_titles_test = preprocess_text(test_project_data_original['project_title'].values)

test_project_data_original['project_title'] = preprocessed_titles_test

print("printing few random reviews AFTER Pre-Processing ")
print(3, test_project_data_original['project_title'].values[3])
print(15, test_project_data_original['project_title'].values[15])
print(10, test_project_data_original['project_title'].values[10])


# ## Preprocessing Categorical Features: essay

# In[28]:


# First combine all the 4 essay columns into a single one:
#  'project_essay_1',  'project_essay_2',   'project_essay_3',   'project_essay_4',
train_project_data_original["essay"] = train_project_data_original["project_essay_1"].map(str) +\
                        train_project_data_original["project_essay_2"].map(str) + \
                        train_project_data_original["project_essay_3"].map(str) + \
                        train_project_data_original["project_essay_4"].map(str)

test_project_data_original["essay"] = test_project_data_original["project_essay_1"].map(str) +\
                        test_project_data_original["project_essay_2"].map(str) + \
                        test_project_data_original["project_essay_3"].map(str) + \
                        test_project_data_original["project_essay_4"].map(str)


print("Checking out few random essay BEFORE Pre-Processing ")
print(9, train_project_data_original['essay'].values[9])
print('*'*50)
print(34, train_project_data_original['essay'].values[34])
print('*'*50)
print(147, train_project_data_original['essay'].values[147])


# In[29]:


preprocessed_essays_train = preprocess_text(train_project_data_original['essay'].values)

train_project_data_original['essay'] = preprocessed_essays_train

print("Checking out few random essay AFTER Pre-Processing ")
print(9, train_project_data_original['essay'].values[9])
print('*'*50)
print(34, train_project_data_original['essay'].values[34])
print('*'*50)
print(147, train_project_data_original['essay'].values[147])

print('*'*50)


# In[30]:


# Same above text-Preprocessing for test dataset: 'essay'
preprocessed_essays_test = preprocess_text(test_project_data_original['essay'].values)
test_project_data_original['essay'] = preprocessed_essays_test


# ## Preprocessing Numerical Values: price
# 

# In[31]:


print('Shape of Resource datesset: ', resource_data.shape) #15.42mn rows


# As we can see there's around 15.42mn rows in the original resource_data.csv. And this is because the same project will require multiple resources.
# So, now first I want to sum all the Price and Quantity that belongs to the same Project ID.
# 
# ### How reset_index() function works.
# 
# Say, I have a dataframe from which I remove some rows. As a result, I get a dataframe in which index is something like that: `[1,5,6,10,11]` and I would like to reset it to `[0,1,2,3,4]`. This is where I will use `reset_index()`
# 
# `df.reset_index(drop=True)`

# In[32]:


price_data_from_resource = resource_data.groupby('id').agg({'price': 'sum', 'quantity': 'sum'}).reset_index()
price_data_from_resource.head(2)


# In[33]:


# Multiply two columns and then create new column with values
price_data_from_resource['resource_cost'] = price_data_from_resource['price'] * price_data_from_resource['quantity']
price_data_from_resource.head(2)


# In[34]:


price_data_from_resource = price_data_from_resource.drop(['price', 'quantity'], axis=1)
price_data_from_resource.head(2)


# In[35]:


train_project_data_original = pd.merge(train_project_data_original, price_data_from_resource, on='id', how='left')
test_project_data_original = pd.merge(test_project_data_original, price_data_from_resource, on='id', how='left')
train_project_data_original['resource_cost'].head()
print(train_project_data_original.columns)
print(test_project_data_original.columns)


# In[36]:


train_project_data_original['resource_cost'].head()


# In[37]:


test_project_data_original['resource_cost'].head()


# ## Deciding which columns to drop - project_resource_summary
# 
# Lets checkout, which words are used in summaries ('project_resource_summary') of approved and non-approved projects.

# In[38]:


word_cloud_for_project_resource_summary = ' '.join(train_project_data_original.loc[train_project_data_original['project_is_approved'] == 1, 'project_resource_summary'].values)

wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000).generate(word_cloud_for_project_resource_summary)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for approved projects')
plt.axis("off")
plt.show()


# In[39]:


word_cloud_for_project_resource_summary = ' '.join(train_project_data_original.loc[train_project_data_original['project_is_approved'] == 0, 'project_resource_summary'].values)

wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000).generate(word_cloud_for_project_resource_summary)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for approved projects')
plt.axis("off")
plt.show()


# Looks almost there's not real difference, hence I shall include 'project_resource_summary' within deleted-columns
# 
# ### I will drop following columns from both train and test dataset
# 

# In[40]:


cols_to_drop_from = [
    'id',
    'teacher_id',
    'project_submitted_datetime',
    'project_resource_summary'
]

train_df_pre_processed_original = train_project_data_original.drop(cols_to_drop_from, axis=1)
test_df_pre_processed_original = test_project_data_original.drop(cols_to_drop_from, axis=1)

# test_project_data_original.drop(cols_to_drop)
print('after dropping ', train_df_pre_processed_original.columns)
# train_project_data_original.to_csv('preprocessed-train.csv')
# test_project_data_original.to_csv('preprocessed-test.csv')


# In[41]:


train_df_pre_processed_original.head(1)


# In[42]:


print('All Features names in preprocessed-train.csv ', train_df_pre_processed_original.columns )
print('All Features names in preprocessed-test.csv ', test_df_pre_processed_original.columns )


# ## From hereon, below all work is completely on the above Pre-Processed dataset (both train and test)
# 
# #### Split the train data into Train and Validation Set, so I can do all the experimentation below on these sets itself,
# 
# This is specifically for applying GridSearchCV to get the best alpha to be applied finally on the original test dataset.
# 

# In[43]:


y_train_pre_processed_original = train_df_pre_processed_original['project_is_approved'].values
x_train_pre_processed_original = train_df_pre_processed_original.drop(['project_is_approved'], axis=1)

X_donor_choose_train, X_donor_choose_validation, y_donor_choose_train, y_donor_choose_validation = train_test_split(x_train_pre_processed_original, y_train_pre_processed_original, test_size=0.33, stratify=y_train_pre_processed_original)
print('X_donor_choose_train shape ', X_donor_choose_train.shape)

X_donor_choose_train.head(1)


# In[44]:


print('x_train_pre_processed_original name of all columns values ', x_train_pre_processed_original.columns)


# ## Vectorization ( convert text to word count vectors with CountVectorizer ) of below Categorical features
# 
# - teacher_prefix
# - project_grade_category
# - school_state
# - project_subject_categories
# - project_subject_subcategories
# 
# For eac of the above categorical variable, I have to vectorize from X_donor_choose_train, X_donor_choose_validation and the original test set given in the Kaggle dataset i.e. test_df_pre_processed_original
# 
# `CountVectorizer()` will basically create a vector from word count
# 
# As to the workflow in the next part, after vectorizing I will merge  all these ONE HOT features with `hsstack()`
# 
# CountVectorizer, an implementation of bag-of-words in which we code text data as a representation of features/words. The values of each of these features represent the occurrence counts of words across all documents.
# 
# ### Vectorizing Categorical data: project_subject_categories

# In[45]:


vectorizer_clean_categories = CountVectorizer(lowercase=False, binary=True)

train_vectorized_ohe_clean_categories = vectorizer_clean_categories.fit_transform(X_donor_choose_train['project_subject_categories'].values)
print(train_vectorized_ohe_clean_categories.shape)

validation_vectorized_ohe_clean_categories = vectorizer_clean_categories.transform(X_donor_choose_validation['project_subject_categories'].values)
print(validation_vectorized_ohe_clean_categories.shape)

test_df_vectorized_ohe_clean_categories = vectorizer_clean_categories.transform(test_df_pre_processed_original['project_subject_categories'].values)
print(test_df_vectorized_ohe_clean_categories.shape)


# ### Vectorizing Categorical data: project_subject_subcategories

# In[46]:


vectorizer_clean_subcategories = CountVectorizer(lowercase=False, binary=True)

train_vectorized_ohe_clean_subcategories = vectorizer_clean_subcategories.fit_transform(X_donor_choose_train['project_subject_subcategories'].values)
print(train_vectorized_ohe_clean_subcategories.shape)

validation_vectorized_ohe_clean_subcategories = vectorizer_clean_subcategories.transform(X_donor_choose_validation['project_subject_subcategories'].values)
print(validation_vectorized_ohe_clean_subcategories.shape)

test_df_vectorized_ohe_clean_subcategories = vectorizer_clean_subcategories.transform(test_df_pre_processed_original['project_subject_subcategories'].values)
print(test_df_vectorized_ohe_clean_subcategories.shape)


# ### Vectorizing Categorical data: teacher_prefix

# In[47]:


vectorizer_teacher_prefix = CountVectorizer(lowercase=False, binary=True)

train_vectorized_ohe_teacher_prefix = vectorizer_teacher_prefix.fit_transform(X_donor_choose_train['teacher_prefix'].values)
print(train_vectorized_ohe_teacher_prefix.shape)

validation_vectorized_ohe_teacher_prefix = vectorizer_teacher_prefix.transform(X_donor_choose_validation['teacher_prefix'].values)
print(validation_vectorized_ohe_teacher_prefix.shape)

test_df_pre_processed_original['teacher_prefix'] = test_df_pre_processed_original['teacher_prefix'].fillna('None')

test_df_vectorized_ohe_teacher_prefix = vectorizer_teacher_prefix.transform(test_df_pre_processed_original['teacher_prefix'].values)
print(test_df_vectorized_ohe_teacher_prefix.shape)


# ### Vectorizing Categorical data: project_grade_category

# In[48]:


vectorizer_project_grade_category = CountVectorizer(lowercase=False, binary=True)

train_vectorized_ohe_project_grade_category = vectorizer_project_grade_category.fit_transform(X_donor_choose_train['project_grade_category'].values)
print(train_vectorized_ohe_project_grade_category.shape)

validation_vectorized_ohe_project_grade_category = vectorizer_project_grade_category.transform(X_donor_choose_validation['project_grade_category'].values)
print(validation_vectorized_ohe_project_grade_category.shape)

test_df_vectorized_ohe_project_grade_category = vectorizer_project_grade_category.transform(test_df_pre_processed_original['project_grade_category'].values)
print(test_df_vectorized_ohe_project_grade_category.shape)


# ### Vectorizing Categorical data: school_state

# In[49]:


vectorizer_school_state = CountVectorizer(lowercase=False, binary=True)

train_vectorized_ohe_school_state = vectorizer_school_state.fit_transform(X_donor_choose_train['school_state'].values)
print(train_vectorized_ohe_school_state.shape)

validation_vectorized_ohe_school_state = vectorizer_school_state.transform(X_donor_choose_validation['school_state'].values)
print(validation_vectorized_ohe_school_state.shape)

test_df_vectorized_ohe_school_state = vectorizer_school_state.transform(test_df_pre_processed_original['school_state'].values)
print(test_df_vectorized_ohe_school_state.shape)


# ## Normalizing numerical features (resource_cost, teacher_number_of_previously_posted_projects)
# 
# 
# Here is how normalizer works and why we should use reshape(1, -1) instead of (-1, 1)
# 
# #### Normalizer by default normalizes on each sample(row) while StandardScaler standardises on each feature(column)
# 
# Here if I use (-1, 1) it means any number of rows, which is the responsibility of numpy to figure out, while I am specifying that I need to have one column. Remember -1 lets numpy to calculate the unknown dimension for the resultant that will match with the original matrix.
# 
# And vice versa, if I do reshape(1, -1) means, that I am specifying row to be 1 while leaving column numbers to be calculated by Numpy.
# 
# So for the case, that I use (-1, 1) => i.e. Rows are unknown while columns required is 1
# 
# ### Note, for normalizing numerical data, we got to use reshape(1, -1) and NOT reshape(-1, 1).
# 
# So now below, I shall derived the vectors for 'resource_cost' and 'teacher_number_of_previously_posted_projects' for both train and validation, and these will be later merged with the other vectorized categorical variables (that I calculated above)  to form the final matrix.
# 
# And for the reshape, first I will reshape(1, -1) i.e. one row and unknown columns > Then apply Normalization > and then reshape again to (-1, 1) i.e. unknown rows, and 1 column
# 
# ### 2.1 Normalizing numerical feature: resource_cost
# 
# ( First I am printing some extra stuff to see the implementation of the above principle, so the below cell's code is only for experimentation and not used outside of the cell )

# In[50]:


# ****** THE CODE OF THIS CELL IS ONLY FOR SHOWING HOW TO - Correct way to implement Normalization of Numerical Features. AND CODES IN THIS CELL ARE NOT USED ANYWHERE OUTSIDE OF THIS CELL *****

normalizer = Normalizer()

""" printing below just to inspect the shape so I know in which form I need to apply the Normalizer()

Per the above note, we can see that shape_a produces a 2-d column vector, and given Normalizer() normalizes on each sample(row)
So shape_a normalized will produce column vector with each value being 1

"""

resource_cost_arr = X_donor_choose_train['resource_cost'].values
# print( 'resource_cost_arr ',  resource_cost_arr)

shape_a = X_donor_choose_train['resource_cost'].values.reshape(-1, 1)
shape_b = X_donor_choose_train['resource_cost'].values.reshape(1, -1)
# print('shape_a ', shape_a )
# print('shape_b ', shape_b )

# Below is WRONG reshape and should NOT be done
# Below 2 variables will produce resource_cost column vectors with each value being 1 and so is worthless
train_normalized_resource_cost_wrong = normalizer.transform(shape_a)
train_normalized_resource_cost_correct = normalizer.transform(shape_b)


print('train_normalized_resource_cost_wrong is : ', train_normalized_resource_cost_wrong[0:10])
# print('train_normalized_resource_cost_correct is : ', train_normalized_resource_cost_correct)
# Above line will print like below, i.e. a single column vector
'''[[1.83906133e-02 3.96946770e-03 1.12896957e-03 3.67034849e-03
  1.48897540e-01 6.30706441e-04 1.41765263e-02 2.00495074e-03
  4.53619601e-03 6.88785883e-03 1.37011017e-03 1.73935829e-03
  5.91764982e-03 1.02206206e-02 4.50035010e-03 3.96785438e-03
  3.93200848e-03 8.99717107e-03 6.63265212e-03 1.50744385e-03
  1.91811391e-02 3.99129791e-03 6.66461598e-03 4.00891837e-02
  ...
  ]]
'''


# In[51]:


# Our first Numerical feature - 'resource_cost'
normalizer = Normalizer()

# As explainEd above first I will reshape(1, -1)
normalizer.fit(X_donor_choose_train['resource_cost'].values.reshape(1, -1))

train_normalized_resource_cost = normalizer.transform(X_donor_choose_train['resource_cost'].values.reshape(1, -1))

validation_normalized_resource_cost = normalizer.transform(X_donor_choose_validation['resource_cost'].values.reshape(1, -1))

test_df_normalized_resource_cost = normalizer.transform(test_df_pre_processed_original['resource_cost'].values.reshape(1, -1))

# After normalization reshape again to (-1, 1) i.e. this time unknown rows (i.e. leaving it to Numpy to decide), and specifying I need 1 column
train_normalized_resource_cost = train_normalized_resource_cost.reshape(-1, 1)
print(train_normalized_resource_cost.shape)

validation_normalized_resource_cost = validation_normalized_resource_cost.reshape(-1, 1)
print(validation_normalized_resource_cost.shape)

test_df_normalized_resource_cost = test_df_normalized_resource_cost.reshape(-1, 1)
print(test_df_normalized_resource_cost.shape)


# ### Normalizing next numerical feature: teacher_number_of_previously_posted_projects

# In[52]:


# Second Numerical feature - 'teacher_number_of_previously_posted_projects'
normalizer = Normalizer()

normalizer.fit(X_donor_choose_train['teacher_number_of_previously_posted_projects'].values.reshape(1, -1))

train_normalized_teacher_number_of_previously_posted_projects = normalizer.transform(X_donor_choose_train['teacher_number_of_previously_posted_projects'].values.reshape(1, -1))

validation_normalized_teacher_number_of_previously_posted_projects = normalizer.transform(X_donor_choose_validation['teacher_number_of_previously_posted_projects'].values.reshape(1, -1))

test_df_normalized_teacher_number_of_previously_posted_projects = normalizer.transform(test_df_pre_processed_original['teacher_number_of_previously_posted_projects'].values.reshape(1, -1))

# After normalization reshape again to (-1, 1) i.e. this time unknown rows (i.e. leaving it to Numpy to decide), and specifying I need 1 column
train_normalized_teacher_number_of_previously_posted_projects = train_normalized_teacher_number_of_previously_posted_projects.reshape(-1, 1)
print(train_normalized_teacher_number_of_previously_posted_projects.shape)

validation_normalized_teacher_number_of_previously_posted_projects = validation_normalized_teacher_number_of_previously_posted_projects.reshape(-1, 1)
print(validation_normalized_teacher_number_of_previously_posted_projects.shape)

test_df_normalized_teacher_number_of_previously_posted_projects = test_df_normalized_teacher_number_of_previously_posted_projects.reshape(-1, 1)
print(test_df_normalized_teacher_number_of_previously_posted_projects.shape)


# ## Encoding Essay column using Bag Of Words

# In[53]:


vectorizer_essay_bow = CountVectorizer(min_df=10)

train_vectorized_bow_essay = vectorizer_essay_bow.fit_transform(X_donor_choose_train['essay'])
print(train_vectorized_bow_essay.shape)

validation_vectorized_bow_essay = vectorizer_essay_bow.transform(X_donor_choose_validation['essay'])
print(validation_vectorized_bow_essay.shape)

test_df_vectorized_bow_essay = vectorizer_essay_bow.transform(test_df_pre_processed_original['essay'])
print(test_df_vectorized_bow_essay.shape)


# ## 5. Merging (with hstack) all the above vectorized features that we created above
# 
# #### You can use the scipy.sparse.hstack to concatenate sparse matrices with the same number of rows (horizontal concatenation):
# 
# `hstack((X1, X2))`
# 
# ### We need to merge all the numerical vectors i.e categorical, text, numerical vectors, once for Naive Bayes on BOW and then for Naive Bayes on TFIDF
# 
# ## Merging all categorical, text, numerical vectors based on BOW

# In[54]:


X_train_hstacked_all_bow_features_vectorized = hstack((train_vectorized_ohe_clean_categories, train_vectorized_ohe_clean_subcategories, train_vectorized_ohe_teacher_prefix, train_vectorized_ohe_project_grade_category, train_vectorized_ohe_school_state, train_normalized_resource_cost, train_normalized_teacher_number_of_previously_posted_projects, train_vectorized_bow_essay))

print('X_train_hstacked_all_bow_features_vectorized.shape is ', X_train_hstacked_all_bow_features_vectorized.shape)

X_validation_hstacked_all_bow_features_vectorized = hstack((validation_vectorized_ohe_clean_categories, validation_vectorized_ohe_clean_subcategories, validation_vectorized_ohe_teacher_prefix, validation_vectorized_ohe_project_grade_category, validation_vectorized_ohe_school_state, validation_normalized_resource_cost, validation_normalized_teacher_number_of_previously_posted_projects, validation_vectorized_bow_essay))

print('X_validation_hstacked_all_bow_features_vectorized.shape is ', X_validation_hstacked_all_bow_features_vectorized.shape)

test_df_hstacked_all_bow_features_vectorized = hstack((test_df_vectorized_ohe_clean_categories, test_df_vectorized_ohe_clean_subcategories, test_df_vectorized_ohe_teacher_prefix, test_df_vectorized_ohe_project_grade_category, test_df_vectorized_ohe_school_state, test_df_normalized_resource_cost, test_df_normalized_teacher_number_of_previously_posted_projects, test_df_vectorized_bow_essay))

print('test_df_hstacked_all_bow_features_vectorized.shape is ', test_df_hstacked_all_bow_features_vectorized.shape)


# ## A Note on GridSearchCV() function of sklearn
# 
# #### when using cross validation with cv=10 the data is split into 10 parts i.e. 10% /90% then each part is used for training while the rest used for validation. I recommend setting the grid search parameter
# 
# The grid search returns a dictionary (accessible through `.cv_results_`) containing the scores for each fold train/test scores as well as the time it took to train/test each fold. Also a summary of that data is included using the mean and the standard deviation. PS. in newer version of pandas you'll need to include return_train_score=True PS.S. when using grid search, splitting the data to train/test is not necessary for model selection, because the grid search splits the data automatically (cv=10 means that the data is split to 10 folds)
# 
# More on the above point - from this very detailed official [doc](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) -
# 
# A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:
# 
# A model is trained using of the folds as training data;
# 
# the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
# 
# The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set), which is a major advantage in problems such as inverse inference where the number of samples is very small.
# 
# ## 6. Applying Multinomial Naive Bayes on BOW
# 
# #### Using GridSearchCV with MultinomialNB on BOW based data to find best alpha that I shall apply in the next step on the test dataset to find the probabilities.

# In[55]:


multinomial_nb_bow = MultinomialNB(class_prior=[0.5, 0.5], fit_prior=False)

'''fit_prior bool, default=True
Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
Whenever we initialize the 'class_prior' parameter to any value (other than None), then it is a good practice to initialize fit_prior = False.

'''

parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}

clf = GridSearchCV(multinomial_nb_bow, parameters, cv=10, scoring='roc_auc', verbose=1, return_train_score=True)

clf.fit(X_train_hstacked_all_bow_features_vectorized, y_donor_choose_train )

# cv_results_dict
train_auc_bow = clf.cv_results_['mean_train_score']
train_auc_std_bow = clf.cv_results_['std_train_score']

cv_auc_bow = clf.cv_results_['mean_test_score']
cv_auc_std_bow = clf.cv_results_['std_test_score']

best_alpha_1_bow = clf.best_params_['alpha']
best_score_1_bow = clf.best_score_

print('Best Alpha BOW: ', best_alpha_1_bow )
print('Best Score BOW : ', best_score_1_bow)


# 
# ### Summary of the above GridSearchCV Implementation
# 
# We have started with hyperparameter alpha with as low as 0.0001 to 1000.Since it is difficult to plot the given range we have used log alphas on x-axis and Auc on y axis as shown in the plot.
# 
# we observe that as log alpha approaches close to 4, both train AUc and cv AUC lines converge
# 
# ## Plotting Naive Bayes on BOW - Log of Alpha on X-axis and AUC on Y-axis

# In[56]:


alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
log_of_alphas = []

for alpha in tqdm(alphas):
  alpha_log = np.log10(alpha)
  log_of_alphas.append(alpha_log)

plt.figure(figsize=(10, 5))
plt.plot(log_of_alphas, train_auc_bow, label='Train AUC Curve' )

'''
[mean - std, mean + std] => It creates a shaded area in between
Taken from - https://stackoverflow.com/a/48803361/1902852

The Axes.fill_between() function in axes module of matplotlib library is used to fill the area between two horizontal curves

And pyplot.gca() Get the current Axes instance on the current figure matching the given keyword args, or create one.

"Current" here means that it provides a handle to the last active axes. If there is no axes yet, an axes will be created. If you create two subplots, the subplot that is created last is the current one.

'''
plt.gca().fill_between(log_of_alphas, train_auc_bow - train_auc_std_bow, train_auc_bow + train_auc_std_bow, alpha=0.3, color='darkblue' )


# similarly for CV_AUC
plt.plot(log_of_alphas, cv_auc_bow, label='CV AUC Curve for BOW Set' )
plt.gca().fill_between(log_of_alphas, cv_auc_bow - cv_auc_std_bow, cv_auc_bow+cv_auc_std_bow + cv_auc_std_bow, alpha=0.3, color='darkorange' )


plt.scatter(log_of_alphas, train_auc_bow, label='Train AUC points-BOW' )
plt.scatter(log_of_alphas, cv_auc_bow, label='CV AUC points-BOW' )

plt.legend()
plt.xlabel('Hyperparameter: Log of alpha for BOW Set')
plt.ylabel('AUC-BOW ')
plt.title('Log of Alpha-Hyperparameter with Train and CV AUC for BOW Set')
plt.grid()
plt.show()


# ### Reason of using logathims_of_alphas in the above
# 
# One of the main reason for using log scale is log scales allow a large range to be displayed without small values being compressed down into bottom of the graph.
# 
# Genearally, in ML/DS log transformation is done on many continuous variables very often. Mostly because of skewed distribution. When the variables span several orders of magnitude. Income is a typical example: its distribution is "power law", meaning that the vast majority of incomes are small and very few are big. This type of "fat tailed" distribution is studied in logarithmic scale because of the mathematical properties of the logarithm:
# 
# Logarithm naturally reduces the dynamic range of a variable so the differences are preserved while the scale is not that dramatically skewed. Imagine some people got 100,000,000 loan and some got 10000 and some 0. Any feature scaling will probably put 0 and 10000 so close to each other as the biggest number anyway pushes the boundary. Logarithm solves the issue.
# 
# If you take values 1000,000,000 and 10000 and 0 into account. In many cases, the first one is too big to let others be seen properly by your model. But if you take logarithm you will have 9, 4 and 0 respectively. As you see the dynamic range is reduced while the differences are almost preserved. It comes from any exponential nature in your feature
# 
# $$log(x^n)= n log(x)$$
# 
# which implies
# 
# $$log(10^4) = 4 * log(10)$$
# 
# and
# 
# $$log(10^3) = 3 * log(10)$$
# 
# ### A note on the alpha parameter
# 
# We add a small smoothing value, alpha to prevent from the probabilities going to zero
# 
# In Multinomial Naive Bayes, the `alpha` parameter is what is known as a [_hyperparameter_](https://en.wikipedia.org/wiki/Hyperparameter_optimization); i.e. a parameter that controls the form of the model itself. In most cases, the best way to determine optimal values for hyperparameters is through a [grid search](http://scikit-learn.org/stable/modules/grid_search.html) over possible parameter values, using [cross validation](http://scikit-learn.org/stable/modules/cross_validation.html) to evaluate the performance of the model on your data at each value. Read the above links for details on how to do this with scikit-learn.
# 
# Check out the wikipedia page http://en.wikipedia.org/wiki/Additive_smoothing.
# 
# ![Imgur](https://imgur.com/v5Vx17J.png)
# 
# ### How to choose alpha
# 
# Basically the idea is that you want to decrease the effect of rare words: for example if you have one spam email with the word 'multinomialNB' in it, and no nonspam emails with this word, then without additive smoothing, your spam filter will classify every email with this keyword as spam.
# 
# ---
# 
# ### class_prior : array-like of shape (n_classes,), default=None
# 
# Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
# 
# Now that GridSearchCV has completed its run with MultinomialNB, I have the best alpha that I shall now apply in the next step on the test dataset to find its class-probabilities.
# 
# ## Run MultinomialNB with the best hyperparameter value (categorical + numerical + essay features)
# 
# From our above GridSearchCV we saw the best_alpha_1_bow was at 0.5
# 
# Best Alpha:  0.5
# Best Score :  0.7049241508387795

# In[57]:


naive_bayes_results_for_bow_with_best_alpha = MultinomialNB(alpha = best_alpha_1_bow, class_prior=[0.5, 0.5], fit_prior=False )

'''fit_prior bool, default=True
Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

Whenever we initialize the 'class_prior' parameter to any value (other than None), then it is a good practice to initialize fit_prior = False.
'''

naive_bayes_results_for_bow_with_best_alpha.fit(X_train_hstacked_all_bow_features_vectorized, y_donor_choose_train)

# Now instead of .best_param like previous implementation will be using .predict_proba
# Using .predict_proba directly yields the same results as using .best_param to get the best hyper-parameter through
y_predicted_for_bow_with_best_alpha_train = naive_bayes_results_for_bow_with_best_alpha.predict_proba(X_train_hstacked_all_bow_features_vectorized)[:, 1]
print('y_predicted_for_bow_with_best_alpha_train.shape is ', y_predicted_for_bow_with_best_alpha_train.shape)

y_predicted_for_bow_with_best_alpha_validation = naive_bayes_results_for_bow_with_best_alpha.predict_proba(X_validation_hstacked_all_bow_features_vectorized)[:, 1]
print('y_predicted_for_bow_with_best_alpha_validation.shape is ', y_predicted_for_bow_with_best_alpha_validation.shape)


# In[58]:


fpr_train_bow, tpr_train_bow, thresholds_train_bow = roc_curve(y_donor_choose_train, y_predicted_for_bow_with_best_alpha_train )

fpr_validation_bow, tpr_validation_bow, thresholds_validation_bow = roc_curve(y_donor_choose_validation, y_predicted_for_bow_with_best_alpha_validation )

print('fpr_train_bow: ', fpr_train_bow)

ax = plt.subplot()

auc_bow_train = auc(fpr_train_bow, tpr_train_bow)
auc_bow_validation = auc(fpr_validation_bow, tpr_validation_bow)

ax.plot(fpr_train_bow, tpr_train_bow, label='Train AUC ='+str(auc(fpr_train_bow, tpr_train_bow)))
ax.plot(fpr_validation_bow, tpr_validation_bow, label='Test AUC ='+str(auc(fpr_validation_bow, tpr_validation_bow)))

plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('AUC')
plt.grid(b=True, which='major', color='k', linestyle=':')
ax.set_facecolor('white')
plt.show()


# ## Bonus Calculation - Derive top 30 Important features - BOW

# In[59]:


# using feature_log_prob_ to get the most important features
print('naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_ ', naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_)
# Its basically, just the log probability of each word, given in a 2-d array structure
# Note, feature_log_prob_ will give me the actual log probabilities of each features,
# but I need the index position of those probabilities.
# So, that with that same index position I can get the corresponding feature-names.

print('LEN of naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_ ', len(naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_[1]))

# So, first, I will sort the result from feature_log_prob_ , and ONLY after that
# get the corresponding positional-index of the top 20 features.
# And then I can apply the same positional-index number to get the corresponding features names (which are the labels / words )
# https://stackoverflow.com/a/50530697/1902852

probabilities_sorted_of_all_negative_class = naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_[0, :].argsort()
#class 0

positive_class_sorted_proba = naive_bayes_results_for_bow_with_best_alpha.feature_log_prob_[1, :].argsort()
#class 1

'''numpy.argsort() function is used to perform an indirect sort along the given axis. It returns an array of indices of the same shape as arr that that would sort the array. Basically it gives me the sorted array indices'''

print("Shape of probabilities_sorted_of_all_negative_class ", probabilities_sorted_of_all_negative_class.shape)
print("Shape of positive_class_sorted_proba ", positive_class_sorted_proba.shape)

# Now - Lets get all horizontally stacked features which were individually derived from CounterVectorizer earlier
# Using itertools.chain() method to chain multiple arrays/lists at once
# https://stackoverflow.com/a/34665782/1902852
from itertools import chain

vectorized_all_features_horizontally_stacked = list(chain(vectorizer_clean_categories.get_feature_names(), vectorizer_clean_subcategories.get_feature_names(), vectorizer_teacher_prefix.get_feature_names(),  vectorizer_project_grade_category.get_feature_names(), vectorizer_school_state.get_feature_names(), vectorizer_essay_bow.get_feature_names()))

# After adding all the vectorized features (which includes both categorical and text features)
# And now I also need to append the two left-over Numerical features which are
# 'resource_cost' and 'teacher_number_of_previously_posted_projects'
vectorized_all_features_horizontally_stacked.extend(['resource_cost', 'teacher_number_of_previously_posted_projects'])

# The above extend() is equivalent to below append()
# vectorized_all_features_horizontally_stacked.append('resource_cost')
# vectorized_all_features_horizontally_stacked.append('teacher_number_of_previously_posted_projects' )

print('Length of vectorized_all_features_horizontally_stacked', ' ( should be ', X_train_hstacked_all_bow_features_vectorized.shape[1], ') : ' , len(vectorized_all_features_horizontally_stacked))

# FINALLY THE TOP 20 FEATURES
top_20_negative_class_features_labels = np.take(vectorized_all_features_horizontally_stacked, probabilities_sorted_of_all_negative_class[-20: -1])
print('top_20_negative_class_features_labels ', top_20_negative_class_features_labels)


top_20_positive_class_features_labels = np.take(vectorized_all_features_horizontally_stacked, positive_class_sorted_proba[-20: -1])
print('top_20_positive_class_features_labels ', top_20_positive_class_features_labels)


# 
# ## What is `feature_log_prob_` in the naive_bayes MultinomialNB()
# 
# From Doc
# 
# feature_log_prob_ is and array of shape (n_classes, n_features) => Empirical log probability of features given a class, P(x_i|y).
# 
# #### Models like logistic regression, or Naive Bayes algorithm, predict the probabilities of observing some outcomes. In standard binary regression scenario the models give you probability of observing the "success" category. In multinomial case, the models return probabilities of observing each of the outcomes. Log probabilities are simply natural logarithms of the predicted probabilities.
# 
# Let's take an example feature "best" for the purpose of this illustration, the `log` probability of this feature for class `1` is `-2.14006616` (as you pointed out), now if we were to convert it into actual probability score it will be `np.exp(1)**-2.14006616 = 0.11764`. Let's take one more step back to see how and why the probability of "best" in class `1` is `0.11764`. As per the documentation of [Multinomial Naive Bayes][2], we see that these probabilities are computed using the formula below:
# 
# ![img](https://i.stack.imgur.com/gyokC.png)
# 
# Where, the numerator roughly corresponds to the number of times feature "best" appears in the class `1` (which is of our interest in this example) in the training set, and the denominator corresponds to the total count of all features for class `1`. Also, we add a small smoothing value, `alpha` to prevent from the probabilities going to zero and `n` corresponds to the total number of features i.e. size of vocabulary.
# 
# ## Predicting Probabilities for the actual test dataset of Kaggle

# In[60]:


# We already have the variable "naive_bayes_results_for_bow_with_best_alpha" fitted as below earlier
# naive_bayes_results_for_bow_with_best_alpha.fit(X_train_hstacked_all_bow_features_vectorized, y_donor_choose_train)
y_predicted_for_bow_with_best_alpha_test_df = naive_bayes_results_for_bow_with_best_alpha.predict_proba(test_df_hstacked_all_bow_features_vectorized)[:, 1]

print('y_predicted_for_bow_with_best_alpha_test_df.shape is ', y_predicted_for_bow_with_best_alpha_test_df.shape)


# ## Creating file for final submission

# In[61]:


sample_sub = pd.read_csv("../input/donorschooseorg-application-screening/sample_submission.csv")

# y_predicted_for_bow_with_best_alpha_test_df


# In[62]:


submission_naive_bayes = pd.concat([sample_sub, pd.Series(y_predicted_for_bow_with_best_alpha_test_df , name='project_is_approved')] , axis=1 ).iloc[:,[0,2]]
submission_naive_bayes.head()
# submission_naive_bayes.to_csv('submission_naive_bayes.csv', index=False)

