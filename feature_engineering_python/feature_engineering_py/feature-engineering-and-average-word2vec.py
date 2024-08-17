#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup
import shutil
import gensim
import re
import spacy
from gensim.models import Word2Vec


# In[3]:


shutil.unpack_archive('../input/quora-question-pairs/train.csv.zip', '.')
shutil.unpack_archive('../input/quora-question-pairs/test.csv.zip', '.')


# In[4]:


df = pd.read_csv("./train.csv")


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


new_df = df.sample(30000, random_state=2)


# In[8]:


new_df.sample(10)


# In[9]:


new_df.isnull().sum()


# In[10]:


# ques_df = new_df[['question1','question2']]
# ques_df.head()


# In[11]:


df = df.dropna().reset_index(drop=True)
df.isnull().sum()


# In[12]:


nlp = spacy.load('en_core_web_sm')


# In[13]:


def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace special characters with their string equivalents.
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q


# In[14]:


new_df['question1'] = new_df['question1'].apply(preprocess)
new_df['question2'] = new_df['question2'].apply(preprocess)


# In[15]:


new_df


# In[16]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

new_df["ques1_lemi"] = new_df["question1"].apply(lambda text: lemmatize_words(text))
new_df["ques2_lemi"] = new_df["question2"].apply(lambda text: lemmatize_words(text))


# In[17]:


new_df = new_df.drop(['question1','question2'], axis=1)


# In[18]:


new_df


# In[19]:


# merge texts
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(new_df['ques1_lemi']) + list(new_df['ques2_lemi'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[20]:


temp_df1 = pd.DataFrame(q1_arr, index= new_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= new_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[21]:


temp_df.sample(10)


# In[22]:


temp_df['is_duplicate'] = new_df['is_duplicate']


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size=0.2,random_state=1)


# In[24]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# rf = RandomForestClassifier()
# rf.fit(X_train,y_train)
# y_pred = rf.predict(X_test)
# accuracy_score(y_test,y_pred)


# In[25]:


# from xgboost import XGBClassifier
# xgb = XGBClassifier(n_jobs= -1, max_depth=10, learning_rate=0.1, subsample = 0.8, min_child_weight=3)
# xgb.fit(X_train,y_train)
# y_pred = xgb.predict(X_test)
# accuracy_score(y_test,y_pred)


# ### Approach 2. With Feature creation and Average Word2vec

# In[26]:


def feature_extraction(q):
    q = q.str.len()
    
    
    


# In[27]:


new_df['q1_len'] = new_df['ques1_lemi'].str.len() 
new_df['q2_len'] = new_df['ques2_lemi'].str.len()


# In[28]:


new_df['q1_num_words'] = new_df['ques1_lemi'].apply(lambda row: len(row.split(" ")))
new_df['q2_num_words'] = new_df['ques2_lemi'].apply(lambda row: len(row.split(" ")))
new_df.sample(5)


# In[29]:


def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['ques1_lemi'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['ques2_lemi'].split(" ")))    
    return len(w1 & w2)


# In[30]:


new_df['word_common'] = new_df.apply(common_words, axis=1)
new_df.head()


# In[31]:


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['ques1_lemi'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['ques2_lemi'].split(" ")))    
    return (len(w1) + len(w2))


# In[32]:


new_df['word_total'] = new_df.apply(total_words, axis=1)
new_df.head()


# In[33]:


new_df['word_share'] = round(new_df['word_common']/new_df['word_total'],2)
new_df.head()


# In[34]:


# Analysis of features
sns.displot(new_df['q1_len'])
print('minimum characters',new_df['q1_len'].min())
print('maximum characters',new_df['q1_len'].max())
print('average num of characters',int(new_df['q1_len'].mean()))


# In[35]:


# common words
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_common'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_common'],label='duplicate')
plt.legend()
plt.show()


# In[36]:


# total words
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_total'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_total'],label='duplicate')
plt.legend()
plt.show()


# In[37]:


# word share
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_share'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_share'],label='duplicate')
plt.legend()
plt.show()


# In[38]:


df_new =new_df.copy()


# In[39]:


final_df = df_new.drop(columns=['id','qid1','qid2','ques1_lemi','ques2_lemi'])
print(final_df.shape)
final_df.head()


# In[40]:


# final_df = final_df.drop('is_duplicate', axis=1)


# In[41]:


final_df = pd.concat([final_df, temp_df], axis=1)


# In[42]:


final_df.head()


# In[43]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)


# In[44]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[45]:


new_df


# In[46]:


def read_questions(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))
    
documents = []
for index, row in new_df.iterrows():
    documents.append(read_questions(row,"ques1_lemi"))
    if row["is_duplicate"] == 0:
        documents.append(read_questions(row,"ques2_lemi"))


# In[47]:


w2v_model = Word2Vec(min_count=10,
                     window=3,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=4)


# In[48]:


w2v_model.build_vocab(documents, progress_per=10000)


# In[49]:


w2v_model.train(documents, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


# In[50]:


words = list(w2v_model.wv.key_to_index)


# In[51]:


w2v_model.wv.most_similar(positive=["india"])


# In[52]:


splitted = []
for row in questions: 
    splitted.append([word for word in row.split()]) 


# In[53]:


avg_data = []
for row in splitted:
    vec = np.zeros(300)
    count = 0
    for word in row:
        try:
            vec = vec+ w2v_model.wv[word]
            count = count+ 1
        except:
            pass
    avg_data.append(vec/count)


# In[54]:


avg_data[0]


# In[55]:


length = len(avg_data)
half = int(length/2)
first_half = avg_data[:half]
second_half = avg_data[length-half:]


# In[56]:


q1_arry = np.array(first_half)
q2_arry = np.array(second_half)


# In[57]:


q1_arry.shape, q2_arry.shape


# In[58]:


tem_df1 = pd.DataFrame(q1_arry, index= new_df.index)
tem_df2 = pd.DataFrame(q2_arry, index= new_df.index)
tem_df = pd.concat([tem_df1, tem_df2], axis=1)


# In[59]:


finl_df = pd.concat([tem_df, temp_df], axis=1)


# In[60]:


finl_df.head()


# In[61]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(finl_df.iloc[:,0:-1].values,finl_df.iloc[:,-1].values,test_size=0.2,random_state=1)


# In[62]:


from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200,n_jobs=4, max_depth=30, learning_rate=0.1, subsample = 0.5)
xgb.fit(Xtrain,ytrain)
y_pred = xgb.predict(Xtest)
accuracy_score(ytest,y_pred)


# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:




