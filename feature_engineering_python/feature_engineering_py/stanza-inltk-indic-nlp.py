#!/usr/bin/env python
# coding: utf-8

# In[39]:


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


# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMcqi3PiHDWsNk5YrwALttBllQQdpvfgsOxg&usqp=CAU)amazon.in

# #Three Important NLP Libraries for Indian Languages
# 
# Author: MOHD SANAD ZAKI RIZVI, JANUARY 23, 2020 
# 
# Text Processing for Indian Languages using Python:
# 
# iNLTK
# 
# Indic NLP Library
# 
# Stanza - https://stanfordnlp.github.io/stanza/index.html
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[40]:


get_ipython().system('pip install spacy')


# In[41]:


train = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")
train.head()


# In[42]:


train.iloc[9,1]


# In[43]:


paragraphs = ["நெல்சன் மண்டேலா (Nelson Rolihlahla Mandela, 18 சூலை 1918 – 5 திசம்பர் 2013)", "தென்னாப்பிரிக்காவின் மக்களாட்சி முறையில் தேர்ந்தெடுக்கப்பட்ட முதல் குடியரசுத் தலைவர் ஆவார்.", "அதற்கு","முன்னர் நிறவெறிக்கு எதிராகப் போராடிய முக்கிய தலைவர்களுள் ஒருவராக இருந்தார்.","MPWOLKE""தொடக்கத்தில்","அறப்போர் (வன்முறையற்ற) வழியில் நம்பிக்கை கொண்டிருந்த இவர், பிறகு  ஆப்பிரிக்க தேசிய காங்கிரஸின் இராணுவப்","MPWOLKE"," பிரிவுக்கு தலைமை தாங்கினார்.","இவர்கள்", "மரபுசாரா கொரில்லாப் போர்முறைத் தாக்குதலை நிறவெறி","MPWOLKE"," அரசுக்கு எதிராக நடத்தினர்.", "மண்டேலாவின் 27", "ஆண்டு சிறைவாசம், நிறவெறிக் கொடுமையின் பரவலாக அறியப்பட்ட சாட்சியமாக விளங்குகிறது.", "சிறையின்" , "பெரும்பாலான காலத்தை இவர் ராபன் தீவில் சிறிய சிறை அறையில் கழித்தார். ","1990 இல்", "அவரது விடுதலைக்கு பிறகு அமைதியான முறையில் புதிய தென்னாப்பிரிக்கக் குடியரசு மலர்ந்தது.", "மண்டேலா", "உலகில் அதிகம் மதிக்கப்படும் தலைவர்களில் ஒருவராக விளங்கினார்.\nமண்டேலா, இனவெறி ஆட்சியில் ஊறிக்கிடந்த தென்னாபிரிக்காவை மக்களாட்சியின் மிளிர்வுக்கு இட்டுச் சென்றவர்.", "அமைதிவழிப்"," போராளியாக, ஆயுதப் போராட்டத் தலைவனாக, தேசத்துரோகக் குற்றம் சுமத்தப்பட்ட குற்றவாளியாக, 27 ஆண்டுகள் சிறையில் வாடி பின்னர் விடுதலையாகி குடியரசு தலைவராக, அமைதிக்கான நோபல் பரிசு பெற்றவராக இவரின் அரசியல் பயணம் தொடர்ந்தது.", "சூன் 2008ல்" "பொது வாழ்க்கையிலிருந்து விலகுவதாக அறிவித்தார்.\n இளமை \nநெல்சன்" "மண்டேலா 1918" "ஆம் ஆண்டு சூலை மாதம் 18 ஆம்" "தியதி தென்னாப்பிரிக்காவில் உள்ள குலு கிராமத்தில் பிறந்தார்." "இவரது தந்தை சோசா பழங்குடி இன மக்கள் தலைவர் ஆவார்[1]."" இவரின் தந்தைக்கு நான்கு மனைவிகள்."" 4 ஆண்களும் 9 பெண்களுமாக 13 பிள்ளைகள். மூன்றாவது மனைவிக்கு மகனாகப் பிறந்தவர் தான் மண்டேலா. இவரின் முழுப்பெயர் \நெல்சன் ரோபிசலா மண்டேலா\. நெல்சன் மண்டேலா என்றே பொதுவாக அழைப்பார்கள். இவரின் சிறுவயதில் குத்துச் சண்டை வீரராகவே அறியப் பெற்றார்.\n\nஅந்தக் குடும்பத்திலிருந்து  முதன் முதலில் பள்ளி சென்ற மண்டேலா, இளம் வயதில் ஆடு, மாடு மேய்த்துக்கொண்டே பள்ளிக்கூடத்தில் படித்தார். போர் புரியும் கலைகளையும் பயின்றார். இவரின் பெயரின் முன்னால் உள்ள ""நெல்சன்" "இவர் கல்வி கற்ற முதல் பள்ளியின் ஆசிரியரினால் சூட்டப்பட்டது என்பது குறிப்பிடத்தக்கது. கல்வியறிவைப் பெறுவதில் பெரிதும் நாட்டம் கொண்ட மண்டேலா, லண்டன் மற்றும் தென்னாபிரிக்கா பல்கலைக்கழகங்களிலும் பட்டப்படிப்பை மேற்கொண்டார். 1941 ஆம் ஆண்டு ஜோகானஸ்பேர்க் சென்று பகுதி நேரத்தில் சட்டக்கல்வி படித்தார். ஒரு தங்கச் சுரங்க பாதுகாப்பு அதிகாரியாகவும், தோட்ட முகவராகவும் பணியாற்றி வந்தார்.\nஅப்போது \'நோமதாம் சங்கர்\' என்ற செவிலியரைத்  திருமணம் செய்து கொண்டார். மண்டேலா ஆப்பிரிக்க தேசிய காங்கிரஸ் இயக்கத்தில் தீவிரமாக ஈடுபட்டு இருந்ததால் மனைவிக்கும் அவருக்கும் இடையே கருத்து வேறுபாடு ஏற்பட்டது. பின்னர் தென்னாப்பிரிக்க அரசு, ஆப்பிரிக்க தேசிய காங்கிரஸ் கட்சியைத் தடை செய்தது. மண்டேலா மீது வழக்கு தொடரப்பட்டது. ஐந்தாண்டுகளாக அந்த வழக்கு விசாரணை நடந்து கொண்டு இருந்தபோது 1958 ஆம் ஆண்டு வின்னி மடிகி லேனா என்பவரை மணந்தார். வின்னி தனது கணவரின் கொள்கைகளுக்காகப் போராடி வ (Truth and Reconciliation Commission)"]  


# In[44]:


text = " ".join(paragraphs)
words = text.split(" ")


# In[45]:


from collections import Counter 
cnt = Counter(words)

cnt.most_common(10)
# print 


# In[46]:


from spacy.lang.ta import STOP_WORDS as STOP_WORDS_TA

#from spacy.lang.hi import STOP_WORDS as hindi_stopwords
#from spacy.lang.ta import STOP_WORDS as tamil_stopwords


# In[47]:


#https://colab.research.google.com/github/rahul1990gupta/indic-nlp-datasets/blob/master/examples/Getting_started_with_processing_hindi_text.ipynb#scrollTo=ghnpMGzngcOn

# Let's remove the stop words before printing most common words 

from spacy.lang.ta import Tamil

nlp = Tamil()

doc = nlp(text)

not_stop_words = []
for token in doc:
  if token.is_stop:
    continue
  if token.is_punct or token.text =="|":
    continue 
  not_stop_words.append(token.text)


not_stop_cnt = Counter(not_stop_words)

not_stop_cnt.most_common(10)


# In[48]:


# Cancel Wordcloud Go to plan B 
# Stanza, iNLTK and Indic NLP now are plan A  
from wordcloud import WordCloud
from spacy.lang.ta import STOP_WORDS as STOP_WORS_TA
import matplotlib.pyplot as plt


# <h1><span class="label label-default" style="background-color:black;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:#03e8fc; padding:10px">iNLTK Toolkit for Indic Languages</span></h1><br>
# 
# iNLTK (Natural Language Toolkit for Indic Languages)
# 
# iNLTK has a dependency on PyTorch 1.3.1, to use iNLTK is necessary to install that below:
# 
# https://inltk.readthedocs.io/en/latest/

# In[49]:


get_ipython().system('pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html')


# #Install inltk

# In[50]:


get_ipython().system('pip install inltk')


# #Hi means hindi language. Not hello.

# In[51]:


from inltk.inltk import setup
setup('hi')


# #Tokenization with iNLTK
# 
# The first step we do to solve any NLP task is to break down the text into its smallest units or tokens. iNLTK supports tokenization of all the 12 languages I showed earlier:
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[52]:


from inltk.inltk import tokenize

hindi_text = """अमेरिकी क्रन्तिकारी युद्ध (1775–1783), जिसे संयुक्त राज्य में अमेरिकी स्वतन्त्रता युद्ध या क्रन्तिकारी युद्ध भी कहा जाता है, ग्रेट ब्रिटेन और उसके तेरह उत्तर अमेरिकी उपनिवेशों के बीच एक सैन्य संघर्ष था, जिससे वे उपनिवेश स्वतन्त्र संयुक्त राज्य अमेरिका बने। शुरूआती लड़ाई उत्तर अमेरिकी महाद्वीप पर हुई। सप्तवर्षीय युद्ध में पराजय के बाद, बदले के लिए आतुर फ़्रान्स ने 1778 में इस नए"""

# tokenize(input text, language code)
tokenize(hindi_text, "hi") #Hi is Hindi language


# #Generate similar sentences from a given text input
# 
# Since iNLTK is internally based on a Language Model for each of the languages it supports, we can do interesting stuff like generate similar sentences given a piece of text!
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[53]:


from inltk.inltk import get_similar_sentences

# get similar sentences to the one given in hindi
output = get_similar_sentences('मैं आज बहुत खुश हूं', 5, 'hi')

print(output)


# #Extract embedding vectors
# 
# "When we are training machine learning or deep learning-based models for NLP tasks, we usually represent the text data by an embedding like TF-IDF, Word2vec, GloVe, etc. These embedding vectors capture the semantic information of the text input and are easier to work with for the models (as they expect numerical input)."
# 
# "iNLTK under the hood utilizes the ULMFiT method of training language models and hence it can generate vector embeddings for a given input text. Here’s an example:"
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[54]:


from inltk.inltk import get_embedding_vectors

# get embedding for input words
vectors = get_embedding_vectors("अमेरिकी उपनिवेशों केया", "hi") #hi is Hindi language

print(vectors)
# print shape of the first word
print("shape:", vectors[0].shape)


# Notice that each word is denoted by an embedding of 400 dimensions.

# #Finding similarity between two sentences
# 
# iNLTK provides an API to find semantic similarities between two pieces of text. This is a really useful feature! We can use the similarity score for feature engineering and even building sentiment analysis systems. Here’s how it works:
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[55]:


from inltk.inltk import get_sentence_similarity

# similarity of encodings is calculated by using cmp function whose default is cosine similarity
get_sentence_similarity('मुझे भोजन पसंद है।', 'मैं ऐसे भोजन की सराहना करता हूं जिसका स्वाद अच्छा हो।', 'hi')


# #The model gives out a cosine similarity of 0.67 which means that the sentences are pretty close, and that’s correct.

# In[56]:


get_sentence_similarity("शुरूआती लड़ाई उत्तर अमेरिकी", "महाद्वीप पर हुई। सप्तवर्षीय युद्", 'hi')


# #Above the similarity is 0.09326018

# <h1><span class="label label-default" style="background-color:black;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:#03e8fc; padding:10px">Indic NLP Library</span></h1><br>

# In[57]:


get_ipython().system('pip install indic-nlp-library')


# In[58]:


# download the resource
get_ipython().system('git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git')


# In[59]:


# download the repo
get_ipython().system('git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git')


# #Set the path so that Python knows where to find these on your computer:

# In[60]:


import sys
from indicnlp import common

# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=r"indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=r"indic_nlp_resources"

# Add library to Python path
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))

# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


# #Splitting input text into sentences
# 
# "Indic NLP Library supports many basic text processing tasks like normalization, tokenization at the word level, etc. But sentence level tokenization is what I find interesting because this is something that different Indian languages follow different rules for."
# 
# "Here is an example of how to use this sentence splitter:"
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[61]:


from indicnlp.tokenize import sentence_tokenize

indic_string="""பெண்கள் இறப்பதும், பிறந்தபின் குழந்தைகள் இறப்பதும் சர்வ சாதாரணம். லேசான சிராய்ப்புகளும் கீறல்களும் கூட மரணத்திற்கு இட்டுச் சென்றன. ஒரு நுண்ணுயிரை வைத்து இன்னொன்றைக் கொல்லமுடிகிற பெனிஸிலின் போன்ற நச்சுமுறி மருந்துகள் """
# Split the sentence, language code "hi" is passed for hingi
sentences=sentence_tokenize.sentence_split(indic_string, lang='hi')

# print the sentences
for t in sentences:
    print(t)


# #perform transliteration using the Indic NLP Library: Transliterating from Hindi to Tamil

# In[62]:


from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Input text "
input_text='பெண்கள் இறப்பதும், பிறந்தபின் குழந்தைகள் இறப்பதும் சர்வ சாதாரணம்.'

# Transliterate from Hindi to Tamil
print(UnicodeIndicTransliterator.transliterate(input_text,"hi","ta"))


# #Phonetics for Indian Sub-Continent languages Alphabet
# 
# "The Indian Sub-Continent languages have strong phonetics for their alphabet and that’s why in the Indic NLP Library, each character has a phonetic vector associated with it that defines its properties."
# 
# "An example where we take the simple Hindi character ‘उ’ :
# 
# https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# In[63]:


from indicnlp.langinfo import *

# Input character 
c='उ'
# Language is Hindi or 'hi'
lang='hi'

print('Is vowel?:  {}'.format(is_vowel(c,lang)))
print('Is consonant?:  {}'.format(is_consonant(c,lang)))
print('Is velar?:  {}'.format(is_velar(c,lang)))
print('Is palatal?:  {}'.format(is_palatal(c,lang)))
print('Is aspirated?:  {}'.format(is_aspirated(c,lang)))
print('Is unvoiced?:  {}'.format(is_unvoiced(c,lang)))
print('Is nasal?:  {}'.format(is_nasal(c,lang)))


# <h1><span class="label label-default" style="background-color:black;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:#03e8fc; padding:10px">Stanza NLP Tookit</span></h1><br>

# ![](https://i.ytimg.com/vi/Mtkktl0kHV0/mqdefault.jpg)youtube.com

# #Installing Stanza

# Stanza is a collection of accurate and efficient tools for the linguistic analysis of many human languages. Starting from raw text to syntactic analysis and entity recognition, Stanza brings state-of-the-art NLP models to languages of your choosing.
# 
# Citation: Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.
# 
# https://stanfordnlp.github.io/stanza/

# In[64]:


get_ipython().system('pip install stanza')


# In[65]:


import stanza
stanza.download('hi')


# In[66]:


nlp = stanza.Pipeline('hi')


# In[67]:


doc = nlp("जिसे संयुक्त राज्य में अमेरिकी स्वतन्त्रता युद्ध या क्रन्तिकारी युद्ध भी कहा जाता")
print(doc)
print(doc.entities)


# #Document to Python Object

# In[68]:


nlp = stanza.Pipeline('hi', processors='tokenize,pos')
doc = nlp('जिसे संयुक्त राज्य में अमेरिकी स्वतन्त्रता युद्ध या क्रन्तिकारी युद्ध भ') # doc is class Document
dicts = doc.to_dict() # dicts is List[List[Dict]], representing each token / word in each sentence in the document


# #Python Object to Document

# In[69]:


from stanza.models.common.doc import Document

dicts = [[{'id': 1, 'text': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=0|end_char=4'}, {'id': 2, 'text': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=5|end_char=13'}, {'id': 3, 'text': '.', 'upos': 'PUNCT', 'xpos': '.', 'misc': 'start_char=13|end_char=14'}]] # dicts is List[List[Dict]], representing each token / word in each sentence in the document
doc = Document(dicts) # doc is class Document


# #Tokenization and Sentence Segmentation

# In[70]:


import stanza

nlp = stanza.Pipeline(lang='hi', processors='tokenize')
doc = nlp('केंद्र की मोदी सरकार ने शुक्रवार को अपना अंतरिम बजट पेश किया. कार्यवाहक वित्त')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')


# #Use the tokenizer just for sentence segmentation. To access segmented sentences, simply use

# In[71]:


print([sentence.text for sentence in doc.sentences])


# #Tokenization without Sentence Segmentation
# 
# "Sometimes you might want to tokenize your text given existing sentences (e.g., in machine translation). You can perform tokenization without sentence segmentation, as long as the sentences are split by two continuous newlines (\n\n) in the raw text. Just set tokenize_no_ssplit as True to disable sentence segmentation. Here is an example"
# 
# https://stanfordnlp.github.io/stanza/tokenize.html

# #I included my account name in the doc below:

# In[75]:


import stanza

nlp = stanza.Pipeline(lang='hi', processors='tokenize', tokenize_no_ssplit=True)
doc = nlp('வழியில் நம்பிக்கை கொண்டிருந்த இவர்,MPWOLKE, பிறக')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')


# #Start with Pretokenized Text
# 
# "In some cases, you might have already tokenized your text, and just want to use Stanza for downstream processing. In these cases, you can feed in pretokenized (and sentence split) text to the pipeline, as newline (\n) separated sentences, where each sentence is space separated tokens. Just set tokenize_pretokenized as True to bypass the neural tokenizer."
# 
# "The code below shows an example of bypassing the neural tokenizer:"
# 
# https://stanfordnlp.github.io/stanza/tokenize.html

# In[73]:


import stanza

nlp = stanza.Pipeline(lang='hi', processors='tokenize', tokenize_pretokenized=True)
doc = nlp('केंद्र की मोदी सरकार ने शुक्रवार को अपना अंतरिम बजट पेश किया. कार्यवाहक वित्त')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')


# #Exception: spaCy tokenizer is currently only allowed in English pipeline.

# In[76]:


import stanza

nlp = stanza.Pipeline(lang='hi', processors={'tokenize': 'spacy'}) # spaCy tokenizer is currently only allowed in English pipeline.
doc = nlp('केंद्र की मोदी सरकार ने शुक्रवार को अपना अंतरिम बजट पेश किया. कार्यवाहक वित्त')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')


# #After the snippet above I couldn't make anything more with Stanza since Hindi is not supported yet.
# 
# Exception: spaCy tokenizer is currently only allowed in English pipeline. 

# Sources:
#     
#  Stanza Package: A Python NLP Package for Many Human Languages
#  https://stanfordnlp.github.io/stanza/sentiment.html   
#  
#  
#  Text Processing for Indian Languages using Python
#  https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

# #Journeys of a thousand miles begin with a single step!  BioBERT wait for me with Stanza!
