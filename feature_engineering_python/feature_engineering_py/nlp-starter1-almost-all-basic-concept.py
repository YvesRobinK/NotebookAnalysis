#!/usr/bin/env python
# coding: utf-8

# # <h1><center><b> U.S. Patent Phrase to Phrase Matching</b></center></h1>
# 
# 
# 
# <img src='https://www.visor.ai/wp-content/uploads/artigo-nlp-27.jpg'>
# 
# 
# 
# ### **Hello Guys! This is my first competition Based on NLP --> Two members(Me and my friend Edoziem Enyinnaya)** 
# 
# ***I try to apply the concept in Competition data later! Now learn basics of NLP and Apply the Simple_text***
# 
# ### ***I am trying in Coming weeks (below the content) and finally apply the concepts in Competition data | If anyone interested in the process suggest your idea and topics guys***
# 
# ## **Part 1_Process**
# 
# ### **1. Data Cleaning and Preprocessing:**
# 
# - Data-Punctuations,stopwords,number,caseconversion (Remove noise)
# - word normalization(stemming,tokenization,lemmatization,POS,NER)
# - word standartization(tables,regularexp)--> Clean text data
# 
# ### **2. Text Representation and word_Embedding:**
# 
# - BOW
# - TF-IDF
# - Word-Embedding (word2vec,doc2vec,Glove)
# - TSNE visualization
# 
# ## **Part 2_Process**
# 
# ### **3. Transformer**
# 
# - BERT, DEBERT, ROBERT, HUGGINGFACE and more process coming soon ....
# 
# ----------------------------------------------------------------------------------------------------------------------------------------

# # **Steps: D/D**
# 
# ### ***Import the Necessay library (Choose one NLTK,Spacy,etc....) But I'm choosing NLTK***
# 
# ## ***-->Part 1<--***
# 
# ### 1. Tokenization (Word and Sentence)
# 
# ### 2. Stopword
# 
# ### 3. Stemming
# 
# ### 4. Part of speech tagging
# 
# ### 5. Chunking
# 
# ### 6. Chinking
# 
# ### 7. Name Entity Recognition
# 
# ### 8. Lemmatization
# 
# ### 9. Corpora
# 
# ### 10. WordNet
# 
# ### 11. Word_Semantic_Similariy
# 
# ### 12. Apply above knowledge in one Movie review data-----> Text_Classification (Simple_NLTK_Data & NaiveBayes algo and predict output)
# 
# ### 13. Text Representation (One_Hot_Encoder, BOW, BOW-NGram, TF-IDF, WordEmbed(FastText, Glove, wor2vec, doc2vec))
# 
# ### 13. TSNE,PCA visualization
# 

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# In[2]:


ex_text = "Hello Mr. VK I am there, how are you doing today? The weather is great and Nltk is awesome. The Nltk is super compare to other library. Its oops based developed."


# # **1. Tokenizing - word tokenizers|sentence tokenizers**
# ## **lexicon and corporas**
# 
# - corpus -> A corpus can be defined as a collection of text documents. It can be thought as just a bunch of text files in a directory,               often alongside many other directories of text files
# - corpora -> body of text, ex: medical journals, presidential speeches, english language
# - lexicon -> words and their means
# - Token -> Each "entity" that is a part of whatever was split up based on rules. For examples, each word is a token when a sentence is             "tokenized" into words. Each sentence can also be a token, if you tokenized the sentences out of a paragraph.

# ## **Sentence_Tokenize**

# In[3]:


print(sent_tokenize(ex_text))


# ## **Word_Tokenize**

# In[4]:


print(word_tokenize(ex_text))


# In[5]:


for i in word_tokenize(ex_text):
    print(i)


# # **2. Stopwords**
# 
# Stopwords are the words in any language which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For some search engines, these are some of the most common, short function words, such as the, is, at, which, and on. In this case, stop words can cause problems when searching for phrases that include them, particularly in names such as “The Who” or “Take That”.
# 

# In[6]:


from nltk.corpus import stopwords


# In[7]:


stop_words = set(stopwords.words("english"))
print(stop_words)


# In[8]:


words = word_tokenize(ex_text)

filtered_sentence =[]

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
        
print(filtered_sentence)

#filtered_sentence = [w for w in words if not w in stop_words]


# # **3. Stemming**
# 
# The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning, other than when tense is involved.
# 
# The reason why we stem is to shorten the lookup, and normalize sentences.
# 
# 
# 
# eg.. 
# i was taking a ride in the car.
# i was riding in the car.
# 
# Porterstemmer(1979)

# In[9]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[10]:


ps = PorterStemmer()

ex_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for s in ex_words:
    print(ps.stem(s))


# In[11]:


n_t = "It is very important to be pythoniy while you are pythoning with python. All pythoners have pythoned poorly atleast once."

words = word_tokenize(n_t)

for st in  words:
    print(ps.stem(st))


# # **4. Part of speech tagging**
# 
# This means labeling words in a sentence as nouns, adjectives, verbs...etc. Even more impressive, it also labels by tense, and more. Here's a list of the tags, what they mean, and some examples:
# 
# #### **POS tag list:**
# 
# - CC	coordinating conjunction
# - CD	cardinal digit
# - DT	determiner
# - EX	existential there (like: "there is" ... think of it like "there exists")
# - FW	foreign word
# - IN	preposition/subordinating conjunction
# - JJ	adjective	'big'
# - JJR	adjective, comparative	'bigger'
# - JJS	adjective, superlative	'biggest'
# - LS	list marker	1)
# - MD	modal	could, will
# - NN	noun, singular 'desk'
# - NNS	noun plural	'desks'
# - NNP	proper noun, singular	'Harrison'
# - NNPS	proper noun, plural	'Americans'
# - PDT	predeterminer	'all the kids'
# - POS	possessive ending	parent\'s
# - PRP	personal pronoun	I, he, she
# - RB	adverb	very, silently,
# - RBR	adverb, comparative	better
# - RBS	adverb, superlative	best
# - RP	particle	give up
# - TO	to	go 'to' the store.
# - UH	interjection	errrrrrrrm
# - VB	verb, base form	take
# - VBD	verb, past tense	took
# - VBG	verb, gerund/present participle	taking
# - VBN	verb, past participle	taken
# - VBP	verb, sing. present, non-3d	take
# - VBZ	verb, 3rd person sing. present	takes
# - WDT	wh-determiner	which
# - WP	wh-pronoun	who, what
# - WRB	wh-abverb	where, when

# In[12]:


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


# In[13]:


train_text = "Engineers, as practitioners of engineering, are professionals who invent, design, analyze, build and test machines, complex systems, structures, gadgets and materials to fulfill functional objectives and requirements while considering the limitations imposed by practicality, regulation, safety and cost.[1][2] The word engineer (Latin ingeniator[3]) is derived from the Latin words ingeniare (to create, generate, contrive, devise and ingenium) (cleverness).[4][5] The foundational qualifications of an engineer typically include a four-year bachelor's degree in an engineering discipline, or in some jurisdictions, a master's degree in an engineering discipline plus four to six years of peer-reviewed professional practice (culminating in a project report or thesis) and passage of engineering board examinations.The work of engineers forms the link between scientific discoveries and their subsequent applications to human and business needs and quality of life.[1]"
sample_text = "Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s. NLP's creators claim there is a connection between neurological processes (neuro-), language (linguistic) and behavioral patterns learned through experience (programming), and that these can be changed to achieve specific goals in life.[1][2]: 2  Bandler and Grinder also claim that NLP methodology can model the skills of exceptional people, allowing anyone to acquire those skills.[3]: 5–6 [4] They claim as well that, often in a single session, NLP can treat problems such as phobias, depression, tic disorders, psychosomatic illnesses, near-sightedness,[5] allergy, the common cold,[Note 1] and learning disorders.[7][8] NLP has been adopted by some hypnotherapists and also by companies that run seminars marketed as leadership training to businesses and government agencies.[9][10]There is no scientific evidence supporting the claims made by NLP advocates, and it has been discredited as a pseudoscience.[11][12][13] Scientific reviews state that NLP is based on outdated metaphors of how the brain works that are inconsistent with current neurological theory and contain numerous factual errors.[10][14] Reviews also found that all of the supportive research on NLP contained significant methodological flaws and that there were three times as many studies of a much higher quality that failed to reproduce the extraordinary claims made by Bandler, Grinder, and other NLP practitioners.[12][13]"


# In[14]:


# train_text = state_union.raw('train_text.txt')
# sample_text = state_union.raw('sample_text.txt')


# In[15]:


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)


# In[16]:


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()


# # **5. Chunking**
# 
# Now that we know the parts of speech, we can do what is called chunking, and group words into hopefully meaningful chunks. One of the main goals of chunking is to group into what are known as "noun phrases." These are phrases of one or more words that contain a noun, maybe some descriptive words, maybe a verb, and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.
# 
# In order to chunk, we combine the part of speech tags with regular expressions. Mainly from regular expressions, we are going to utilize the following:

# ## **Regular expression**
# 
# Here is a quick cheat sheet for various rules in regular expressions:
# 
# ### ***Identifiers:***
# 
# - \d = any number
# - \D = anything but a number
# - \s = space
# - \S = anything but a space
# - \w = any letter
# - \W = anything but a letter
# - . = any character, except for a new line
# - \b = space around whole words
# - \. = period. must use backslash, because . normally means any character.
# 
# ### ***Modifiers:***
# 
# - {1,3} = for digits, u expect 1-3 counts of digits, or "places"
# - + = match 1 or more
# - ? = match 0 or 1 repetitions.
# - * = match 0 or MORE repetitions
# - ^ = matches start of a string
# - | = matches either/or. Example x|y = will match either x or y
# - [] = range, or "variance"
# - {x} = expect to see this amount of the preceding code.
# - {x,y} = expect to see this x-y amounts of the precedng code
# 
# ### ***White Space Charts:***
# 
# - \n = new line
# - \s = space
# - \t = tab
# - \e = escape
# - \f = form feed
# - \r = carriage return
# 
# ***Characters to REMEMBER TO ESCAPE IF USED!***
# 
# - . + * ? [ ] $ ^ ( ) { } | \
# 
# ### ***Brackets:***
# 
# - [] = quant[ia]tative = will find either quantitative, or quantatative.
# - [a-z] = return any lowercase letter a-z
# - [1-5a-qA-Z] = return all numbers 1-5, lowercase letters a-q and uppercase A-Z
# - """<RB.?>* = "0 or more of any tense of adverb," followed by:
# - <VB.?>* = "0 or more of any tense of verb," followed by:
# - <NNP>+ = "One or more proper nouns," followed by
# - <NN>? = "zero or one singular noun."
# - """

# In[17]:


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #print(tagged)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""" 
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            

    except Exception as e:
        print(str(e))


process_content()


# In[18]:


import matplotlib.pyplot as plt
from sklearn import tree
import os
from IPython.display import Image, display
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
get_ipython().run_line_magic('matplotlib', 'inline')

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #print(tagged)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""" 
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #print(chunked)
            chunked.draw()

    except Exception as e:
        print(str(e))


process_content()


# This kind of output is shows! but jupyter got didnot display
# 
# <img src='https://naadispeaks.files.wordpress.com/2017/02/cap_4.gif'>

# # **6. Chinking**
# Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk. The chunk that you remove from your chunk is your chink.

# In[19]:


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #print(tagged)
            
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            #chunked.draw()

    except Exception as e:
        print(str(e))


process_content()


# # **7. Named Entity Recognition:**
# 
# One of the most major forms of chunking in natural language processing is called "Named Entity Recognition." The idea is to have the machine immediately be able to pull out "entities" like people, places, things, locations, monetary figures, and more.
# 
# 
# #### **Example:**
# 
# NE Type and Examples
# 
# 1. ORGANIZATION - Georgia-Pacific Corp., WHO
# 2. PERSON - Eddy Bonte, President Obama
# 3. LOCATION - Murray River, Mount Everest
# 4. DATE - June, 2008-06-29
# 5. TIME - two fifty a m, 1:30 p.m.
# 6. MONEY - 175 million Canadian Dollars, GBP 10.40
# 7. PERCENT - twenty pct, 18.75 %
# 8. FACILITY - Washington Monument, Stonehenge
# 9. GPE - South East Asia, Midlothian

# In[20]:


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            print('Without Binary NER')
            print('-----------------------------------------------------------------------------------')
            # Try without apply binary=True
            namedEntity = nltk.ne_chunk(tagged)
            #namedEntity.draw()
            print(namedEntity)
            print('----------------------------------------------------------------------------------')
            
            print('With Binary NER')
            print('-----------------------------------------------------------------------------------')
            # try with apply binary= True
            namedEntity_Bin = nltk.ne_chunk(tagged,binary=True)
            print(namedEntity_Bin)
            
    except Exception as e:
        print(str(e))

print("Go and watch the output like name entity recognition e.g., organization(NNP/NN),....")
print('')
process_content()


# # ***8. Lemmatization***
# A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
# 
# So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma.
# 
# 

# In[21]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


# In[22]:


print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("dogs"))
print(lemmatizer.lemmatize("corpora"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))


# In[23]:


# different approach based on noun,adjective,verb

#general
print('General_lemma:',lemmatizer.lemmatize("worst"))

# Adjective
print('Adjective_change_lemma -> worst to :',lemmatizer.lemmatize("worst", pos='a'))
print('Adjective_change_lemma -> better to:',lemmatizer.lemmatize("better",pos='a'))

#verb
print('lemma_v:',lemmatizer.lemmatize("run"))
print('lemma_verb_initiated:',lemmatizer.lemmatize("run",pos='v'))


# # **9. Corpora**
# 
# Corpus - Simply said that corpus means inbuilt data of text
# 
# Almost all of the files in the NLTK corpus follow the same rules for accessing them by using the NLTK module, but nothing is magical about them. These files are plain text files for the most part, some are XML and some are other formats, but they are all accessible by you manually, or via the module and Python

# In[24]:


import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

print(nltk.__file__)


# In[25]:


sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])


# # **10. WordNet:**
# 
# WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus.
# 
# You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more

# In[26]:


from nltk.corpus import wordnet


# In[27]:


syns = wordnet.synsets("program")
print('All kind of syns:',syns)


# In[28]:


#synset
print('Only first index of synset:',syns[0])
print('------------------------------------')

# lemmas
print('Apply lemmas in first index of synset:',syns[0].lemmas())
print('------------------------------------')
# just the one word
print('Name of synset:',syns[0].lemmas()[0].name())
print('------------------------------------')
# definition of particular word (eg.. plan)
print('Definition of synsets:',syns[0].definition())
print('------------------------------------')
#examples
print('Example of synsets:',syns[0].examples())
print('------------------------------------')


# In[29]:


# Synonyms and Antonyms

synonyms = []
antonyms = []

for syn in wordnet.synsets("unknown"):
    for l in syn.lemmas(): #lemmas means synonyms of word
#         print("l:",l) # meaning of words in nltk library is lot
#         print('---------------------------------------------')
        synonyms.append(l.name())
        
        if l.antonyms(): #opposite of meaning
            antonyms.append(l.antonyms()[0].name())

print('Synonyms:',set(synonyms))
print('----------------------------------------------------')
print('antonyms:',set(antonyms))
print('----------------------------------------------------')


# # **11. Similarity identification**

# In[30]:


#Similarity identification

#"Wu and Palmer method: WordNet::Similarity::wup - Perl module for computing semantic relatedness of word senses using the edge counting method of the of Wu & Palmer (1994)"
#Resnik (1999) revises the Wu & Palmer (1994) method of measuring semantic relatedness. Resnik uses use an edge distance method by taking into account the most specific node subsuming the two concepts. Here we have implemented the original Wu & Palmer method, which uses node-counting.
# If you want more details refer this one: "https://metacpan.org/release/TPEDERSE/WordNet-Similarity-1.03/view/lib/WordNet/Similarity/wup.pm'


word1 = wordnet.synset("computer.n.01") # n means Noun
word2 = wordnet.synset("system.n.01")

word3 = wordnet.synset("cycle.n.01") # n means Noun
word4 = wordnet.synset("bike.n.01")

word5 = wordnet.synset("ship.n.01") # n means Noun
word6 = wordnet.synset("person.n.01")

word7 = wordnet.synset("ship.n.01") # n means Noun
word8 = wordnet.synset("boat.n.01")



print('Similarity_Score of first_second words: ',word1.wup_similarity(word2))
print('Similarity_Score of Third_fourth words: ',word3.wup_similarity(word4))
print('Similarity_Score of fifth_sixth words: ',word5.wup_similarity(word6))
print('Similarity_Score of seventh_eigth words: ',word7.wup_similarity(word8))


# # ***Apply above knowledge in simple nltk corpus data***
# 
# # **12. Text_Classification**
# 
# - Text classification can be pretty broad. Maybe we're trying to classify text as about politics or the military,Culture, stock etc... 
# - Maybe we're trying to classify it by the gender of the author who wrote it. In our case, we're going to try to create a sentiment analysis algorithm.
# 
# **I thought NLTK corpus is awesome! Because all custom data and easy to work**
# 
# 1. Taken the Movie review data
# 2. Identify the common negative words (convert word to features)
# 3. Apply naive bayes algo (How to works?)

# In[31]:


# Now I am trying to movie review(sentimental analysis)
import nltk
import random
from nltk.corpus import movie_reviews


# In[32]:


# documents_review = []

# for category in movie_reviews.categories():
#     for fileid in movie_reviews.categories():
#         documents_review.append(list(movie_reviews.words(fileid)),category)


documents = [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
print(documents[1])


# In[33]:


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# Print Most common words in the movie reviews (15) - How many time used
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))


# In[34]:


print('How many time word "happy" comes in movie reviews: ',all_words["happy"])


# # **12.1. Converting Words to features**

# In[35]:


# Already we know this below code only change is visualize the 3000 common words
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]


# In[36]:


# Identify the which one commonly positive and negative

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

feature_sets = [(find_features(rev), category) for (rev,category) in documents ]


# # **12.2. Apply Algorithm and predict output**

# In[37]:


#Feature separation(dont said category of data and predict the algo)

train_set = feature_sets[:1900]
test_set = feature_sets[1900:]


# In[38]:


#Naive bayes algo best 

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("NaiveBayes Algorithm accuracy % :", (nltk.classify.accuracy(classifier,test_set))*100)
print('----------------------------------------------------------')
print('')
classifier.show_most_informative_features(15)


# # **12.3. Save_Load the Pickle file**

# In[39]:


import pickle
#save the model

save_class = open("naivebayes.pickle","wb")
pickle.dump(classifier,save_class)
save_class.close()

classifier_f = open("./naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("NaiveBayes Algorithm accuracy % :", (nltk.classify.accuracy(classifier,test_set))*100)
print('----------------------------------------------------------')
print('')
classifier.show_most_informative_features(30)


# # **13. Text Representation**
# 
#  Text Representation is a way to convert text in its natural form to vector form  – Machines like it and understand it in this way only! The numbers/vectors form. This is the second step in an NLP pipeline after Text Pre-processing. Let’s get started with a sample corpus, pre-process
# 
# <img src= 'https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-1.png'>

# ## **13.1. One Hot Encoding of text (First manual function and using simple scikitlearn)**

# In[40]:


# take input text doc and remove puncuation,dot, cvt lower all input data

documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs


# In[41]:


# build the vocabulary 
vocab = {}
count = 0
for doc in processed_docs:
    for word in doc.split():
        if word not in vocab:
            count = count +1
            vocab[word] = count
print(vocab)

#Get one hot representation for any string based on this vocabulary. 
#If the word exists in the vocabulary, its representation is returned. 
#If not, a list of zeroes is returned for that word. 
def get_onehot_vector(somestring):
    onehot_encoded = []
    for word in somestring.split():
        temp = [0]*len(vocab)
        if word in vocab:
            temp[vocab[word]-1] = 1 # -1 is to take care of the fact indexing in array starts from 0 and not 1
        onehot_encoded.append(temp)
    return onehot_encoded


print('-------------------------------------------------')
print(processed_docs[1])
print('-------------------------------------------------')
print(get_onehot_vector(processed_docs[1])) #one hot representation for a text from our corpus.
print('---------------------------------------------------')
#try your own words
get_onehot_vector("man and man are good") 


# ## **Scikit-learn pack - OHE**

# In[42]:


S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0]+data[1]+data[2]+data[3]
print("The data: ",values)
print('-------------------------------------------------')

#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)
print('----------------------------------------------------')

#One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n",onehot_encoded)


# ## **13.2. Bag of words**
# 
# A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. It is called a “bag” of words because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document.
# 
# 
# <img src='https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Feditor.analyticsvidhya.com%2Fuploads%2F34521bagofwords.jpg&f=1&nofb=1'>

# In[43]:


#Now, let's do the main task of finding bag of words representation. We will use CountVectorizer from sklearn.

from sklearn.feature_extraction.text import CountVectorizer

#look at the documents list
print("Our corpus: ", processed_docs)

count_vect = CountVectorizer()
#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and man are friends"])
print("Bow representation for 'dog bites are friends':", temp.toarray())


# In[44]:


#In the above code, we represented the text considering the frequency of words into account. However, sometimes, we don't care about frequency much, but only want to know whether a word appeared in a text or not. That is, each document is represented as a vector of 0s and 1s. We will use the option binary=True in CountVectorizer for this purpose.
#BoW with binary vectors
count_vect = CountVectorizer(binary=True)
count_vect.fit(processed_docs)
temp = count_vect.transform(["dog man are friends"])
print("Bow representation for 'dog bites man are friends':", temp.toarray())


# ## **13.3 Bag of N-Gram**
# 
# One hot encoding, BoW and TF-IDF treat words as independent units. There is no notion of phrases or word ordering. Bag of Ngrams (BoN) approach tries to remedy this. It does so by breaking text into chunks of n countigous words/tokens. This can help us capture some context, which earlier approaches could not do
# 
# Example:
# 
# So, the respective vectors for these sentences are:
# 
# “This is a good job. I will not miss it for anything”=[1,1,1,1,0]
# 
# ”This is not good at all”=[1,0,0,1,1]
# 
# Can you guess what is the problem here? Sentence 2 is a negative sentence and sentence 1 is a positive sentence. Does this reflect in any way in the vectors above? Not at all. So how can we solve this problem? Here come the N-grams to our rescue.
# 
# An N-gram is an N-token sequence of words: a 2-gram (more commonly called a bigram) is a two-word sequence of words like “really good”, “not good”, or “your homework”, and a 3-gram (more commonly called a trigram) is a three-word sequence of words like “not at all”, or “turn off light”.
# 
# <img src = 'https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fconglang.github.io%2Fimg%2Fml_feature_extraction_ngram.png&f=1&nofb=1'>

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer

#Ngram vectorization example with count vectorizer and uni, bi, trigrams
count_vect = CountVectorizer(ngram_range=(1,3))

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])

print("Bow representation for 'dog and dog are friends':", temp.toarray())

#Note that the number of features (and hence the size of the feature vector) increased a lot for the same data, compared to the ther single word based representations!!


# ## **13.4. TF-IDF**
# 
# ***In all the other approaches we saw so far, all the words in the text are treated equally important. There is no notion of some words in the document being more important than others. TF-IDF addresses this issue. It aims to quantify the importance of a given word relative to other words in the document and in the corpus. It was commonly used representation scheme for information retrieval systems, for extracting relevant documents from a corpus for given text query.***
# 
# TF-IDF for a word in a document is calculated by multiplying two different metrics:
# 
# **The term frequency (TF)** of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are other ways to adjust the frequency. For example, by dividing the raw count of instances of a word by either length of the document, or by the raw frequency of the most frequent word in the document. The formula to calculate Term-Frequency is
# 
# **TF(i,j)=n(i,j)/Σ n(i,j)**
# 
# Where,
# 
# n(i,j )= number of times nth word  occurred in a document
# Σn(i,j) = total number of words in a document. 
# 
# **The inverse document frequency(IDF)** of the word across a set of documents. This suggests how common or rare a word is in the entire document set. The closer it is to 0, the more common is the word. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
# 
# <img align = 'center' src="https://mungingdata.files.wordpress.com/2017/11/equation.png?w=430&h=336">

# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["vk hit bat.", "vk ran street.", "Dog eats carrot.", "vk dog love very much in vk."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
print("processed_docs:",processed_docs)
print('-----------------------------------------------------')
print('')
#Initialize the TFIDF
tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(processed_docs)

#IDF for all words in the vocabulary
print("IDF for all words in the vocabulary",tfidf.idf_)
print("-"*10)
#All words in the vocabulary.
print("All words in the vocabulary",tfidf.get_feature_names())
print("-"*10)

#TFIDF representation for all documents in our corpus 
print("TFIDF representation for all documents in our corpus\n",bow_rep_tfidf.toarray()) 
print("-"*10)

temp = tfidf.transform(["vk and dog are friends"])
print("Tfidf representation for 'dog and vk are friends':\n", temp.toarray())


# # **13.5. Word Embedding**
# 
# Word embeddings are an approach to representing text in NLP. In this notebook we will demonstrate how to train embeddings using Genism. 
# 
# one of the great reference: https://towardsdatascience.com/word2vec-explained-49c52b4ccb71

# In[47]:


get_ipython().system('pip install gensim==3.6.0')
get_ipython().system('pip install requests==2.23.0')


# In[48]:


from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')


# In[49]:


# define training data
#Genism word2vec requires that a format of ‘list of lists’ be provided for training where every document contained in a list.
#Every list contains lists of tokens of that document.
corpus = [['vk','birds','dog','bites','man'], ["man", "bites" ,"dog","love"],["dog","eats","meat","cat"],["man", "eats","food"]]

#Training the model
model_cbow = Word2Vec(corpus, min_count=1,sg=0) #using CBOW Architecture for trainnig
model_skipgram = Word2Vec(corpus, min_count=1,sg=1)#using skipGram Architecture for training 


# ### ***There are two main architectures which yield the success of word2vec. The skip-gram and CBOW architectures***
# 
# ## **13.5.1 CBOW (Continuous BOW)**
# 
# This architecture is very similar to a feed forward neural network. This model architecture essentially tries to predict a target word from a list of context words.
# 
# <img src='https://miro.medium.com/max/392/1*_8Ul4ICaCtmZWPrWqH32Ow.png'>

# In[50]:


#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.vocab)
print(words)

#Acess vector for one word
print(model_cbow['dog'])


# In[51]:


#Compute similarity 
print("Similarity between eats and bites,vk:",model_cbow.similarity('eats',"dog"))
print("Similarity between eats and man, dog:",model_cbow.similarity('eats','vk'))
print('------------------------------------')
print('')
#Most similarity
print(model_cbow.most_similar('meat'))
print('-----------------------------------')
print('')

# save model
model_cbow.save('model_cbow.bin')
print('---------------------------------------')
print('')
# load model
new_model_cbow = Word2Vec.load('model_cbow.bin')
print(new_model_cbow)


# ## **13.5.2.Skipgram**
# 
# In skipgram, the task is to predict the context words from the center word.
# 
# 'This model essentially tries to learn and predict the context words around the specified input word'
# 
# 
# <img src='https://miro.medium.com/max/700/1*M6UxaLSbNMeoDFWRN_kPeQ.png'>

# In[52]:


#Summarize the loaded model
print(model_skipgram)
print('-------------------------------------------------')
print('')
#Summarize vocabulary
words = list(model_skipgram.wv.vocab)
print(words)
print('-------------------------------------------------')
print('')

#Acess vector for one word
print(model_skipgram['dog'])
print('-------------------------------------------------')
print('')

#Compute similarity 
print("Similarity between eats and bites:",model_skipgram.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_skipgram.similarity('eats', 'man'))
print('-------------------------------------------------')
print('')
#Most similarity
print(model_skipgram.most_similar('meat'))
print('-------------------------------------------------')
print('')

# save model
model_skipgram.save('model_skipgram.bin')

# load model
new_model_skipgram = Word2Vec.load('model_skipgram.bin')
print(new_model_skipgram)


# ## **13.5.3 DOC2VEC**
# 
# **Everyone got onedoubt in doc2vec and word2vec! same thing happen me! this pic easily know what happen in two process i think**
# 
# Doc2Vec is another widely used technique that creates an embedding of a document irrespective to its length. While Word2Vec computes a feature vector for every word in the corpus, Doc2Vec computes a feature vector for every document in the corpus
# 
# <img src='https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs40537-018-0139-2/MediaObjects/40537_2018_139_Fig8_HTML.png'>

# In[53]:


import warnings
warnings.filterwarnings('ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pprint import pprint
import nltk
nltk.download('punkt')


# In[54]:


data = ["vk love nature",
        "vk love dogs",
        "dog eats carrot",
        "vk eats food"]
tagged_data = [TaggedDocument(words=word_tokenize(word.lower()), tags=[str(i)]) for i, word in enumerate(data)]
tagged_data


# In[55]:


#dbow
model_dbow = Doc2Vec(tagged_data,vector_size=20, min_count=1, epochs=2,dm=0)
print(model_dbow.infer_vector(['vk','eats','meat']))#feature vector of man eats food


# In[56]:


model_dbow.wv.most_similar("vk",topn=5)#top 5 most simlar words.


# In[57]:


model_dbow.wv.n_similarity(["dog"],["vk"])


# In[58]:


#dm
model_dm = Doc2Vec(tagged_data, min_count=1, vector_size=20, epochs=2,dm=1)

print("Inference Vector of vk eats food\n ",model_dm.infer_vector(['vk','eats','food']))

print("Most similar words to vk in our corpus\n",model_dm.wv.most_similar("vk",topn=5))
print("Similarity between vk and dog: ",model_dm.wv.n_similarity(["dog"],["vk"]))


# ## **13.5.4 T-SNE visualize embedding**
# 
# T-SNE algorithm to visualize embeddings. This is a nice cheap technique for understanding the nature of your embeddings.
# 
# 
# ### ***I am taking competition data and visualize***
# 
# Reference: 
# 1. https://www.kaggle.com/code/colinmorris/visualizing-embeddings-with-t-sne/notebook
# 
# 2. https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne/notebook

# In[59]:


import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# import zipfile
# with zipfile.ZipFile('../input/quora-question-pairs/train.csv.zip', 'r') as zip_ref:
#     zip_ref.extractall()


# data = pd.read_csv('./train.csv').sample(50000, random_state=23)
data = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')


# In[60]:


STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['context', 'target']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

data = clean_dataframe(data)
data.head(5)


# In[61]:


def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['context','target']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(data)        
corpus[0:2]


# In[62]:


model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model.wv['material']


# In[63]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[64]:


tsne_plot(model)


# In[65]:


# A more selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
tsne_plot(model)


# In[66]:


# A less selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=100, workers=4)
tsne_plot(model)


# In[67]:


model.most_similar('material')


# ## **13.5.5 FastText**
# 
# FastText create Facebook research team, One of best embedded method
# 
# As Word2Vec and Doc2Vec models rely on the vocabulary they had been trained on if the new text data that we want to vectorized contains words that were not previously present in the training vocabulary then these models fail to vectorized the unseen words accurately. FastText overcomes this problem.
# 
# FastText does this by vectorizing each word as a combination of character n-grams. The keyword to remember when working with FastText is character n-grams. If you don’t know what n-grams are, they are a number of words taken into consideration when working with text.
# 
# ### ***Uses of FastText:***
# 
# - Very useful for finding semantic similarities
# - Large datasets can be trained in minutes
# - Can be used for the purpose of text classification.
# 
# <img src='https://kavita-ganesan.com/wp-content/uploads/fastText-vs.-Word2Vec.png'>
# 
# Thus the biggest advantage of using FastText over other models such as Word2Vec is that FastText can generate embeddings for sentences with words not present in the training vocabulary with the help of character n-grams whereas other models fail to do so.
# 
# Full credit-Reference: https://pythonwife.com/fasttext-in-nlp/

# In[68]:


get_ipython().system('pip install gensim==4.0.1')
from gensim.models import FastText
from gensim.test.utils import common_texts


# In[69]:


common_texts


# In[70]:


#build the model
model1 = FastText(size=5, window=3, min_count=1)


# In[71]:


model1.build_vocab(common_texts)
model1.train(common_texts, total_examples=len(common_texts), epochs=10)


# In[72]:


print('Predict one word:',model1.wv['human'])
print('---------------------------')
print('')
print(model1.wv.most_similar(positive=['computer','interface'], negative=['human']))


# ## **13.5.6 Glove**
# 
# @Ruchi Bhatia (Reference: https://www.kaggle.com/discussions/getting-started/164408)
# 
# GloVe: GloVe is based on matrix factorization techniques on the word-context matrix. It first constructs a large matrix of (words x context) co-occurrence information, i.e. for each “word” (the rows), you count how frequently we see this word in some “context” (the columns) in a large corpus. The number of “contexts” is of course large, since it is essentially combinatorial in size. So then we factorize this matrix to yield a lower-dimensional (word x features) matrix, where each row now yields a vector representation for each word. In general, this is done by minimizing a “reconstruction loss”. This loss tries to find the lower-dimensional representations which can explain most of the variance in the high-dimensional data.
# 
# ### **Why GloVe is better than Word2Vec?**
# 
# ### **Rather than using a window to define local context, GloVe constructs an explicit word-context or word co-occurrence matrix using statistics across the whole text corpus. The result is a learning model that may result in generally better word embeddings.**
# 
# <img src='https://miro.medium.com/max/1400/1*gcC7b_v7OKWutYN1NAHyMQ.png'>
# 
# Reference: https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html

# ## **Hello Guys! I'm trying Glove embed, Build lstm model and predict output**
# 
# ### **Features selection: I'm simply take x is target and y is score**
# 

# In[73]:


#Here we will use GloVe pretrained corpus model to represent our words.It is available in 3 varieties :50D ,100D and 200 Dimentional.We will try 100 D here.

import numpy as np
import pandas as pd 


import os
print(os.listdir("../input"))

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
df.sample(2)


# ### **Count the score attribute**

# In[74]:


print(df['score'].value_counts())
sns.countplot(df['score'])


# ### **Feature selection**

# In[75]:


x = df['target']
y = df['score']


# ### **Tokenize,pad,vocb the input**

# In[76]:


token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)


# In[77]:


pad_seq = pad_sequences(seq,maxlen=300)


# In[78]:


vocab_size = len(token.word_index)+1


# ### **Initialize the glove 300d input txt file and apply the input data**

# In[79]:


embedding_vector = {}
f = open('../input/glove840b300dtxt/glove.840B.300d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef


# In[80]:


embedding_matrix = np.zeros((vocab_size,300))
for word,i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value


# ### **Build the model**

# In[81]:


model = Sequential()
model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))
model.add(Bidirectional(CuDNNLSTM(75)))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(pad_seq,y,epochs = 50,batch_size=32)


# ### **Save the model**

# In[82]:


from keras.models import load_model
model.save("My_Glove_LSTM_Model.h5")
#del model  # deletes the existing model
    
# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')


# ### **Apply testdata and predict** 

# In[83]:


testing = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
sample = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
print('sample_shape',sample.shape)
testing.sample(2)


# In[84]:


x_test = testing['target']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test,maxlen=300)


#  ### **Predict output** 

# In[85]:


predict = model.predict(testing_seq)
testing['label'] = predict
testing.head()


# # **Generate Output_Submission CSV file**

# In[86]:


final_predict = testing.label
sample['score'] = final_predict
sample.to_csv("submission.csv",index=False)
print("Final achieve to send Glove_Predict output data")


# In[ ]:





# ***First of all this learning process in NLP is nice for me and I believe this notebook useful for all,This not just basics of NLP "How to start nlp in begineers?" in my experience perspective***
# 
# **This Notebook is first step to learn NLP (different basic concepts and how to apply the text data)**
# 
# ## **"My next step is move forward in Transformer concept"** 
# (https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/320962)

# ***Reference: Full credits -----> Sendex_NLTK video series, Nlp practical book, more resourse like blog,articles,research paper & my team member*** 
# 
# **1. https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/**
# 
# **2. https://youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL**
# 
# **3. https://github.com/practical-nlp/practical-nlp-code/blob/master/**
# 
# **4. https://www.mygreatlearning.com/blog/bag-of-words/**
# 
# **5. https://towardsdatascience.com/word2vec-explained-49c52b4ccb71**
# 
# **6. https://www.kaggle.com/code/shahules/basic-eda-cleaning-and-glove#GloVe-for-Vectorization**
# 
# **7. https://medium.com/@sarin.samarth07/glove-word-embeddings-with-keras-python-code-52131b0c8b1d**
# 
# 
# ### **"if you see any errors and your opinion! feel free to share with me"**
# 
# 
# ## ⭐️⭐️⭐️NextProcess_🔥Transformer🔥_Nextprocess⭐️⭐️⭐️   

# # ***⭐️⭐️⭐️Thankyou for visiting Guys!⭐️⭐️⭐️***
