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


get_ipython().system('pip install rich')


# In[3]:


import re
import warnings
import string
import numpy as np 
import random
import pandas as pd 
from scipy import stats
import missingno as msno
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from rich.console import Console
from rich.theme import Theme
from rich import pretty

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

from collections import defaultdict
from collections import Counter

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error as mse, 
    make_scorer, 
    accuracy_score, 
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import optimizers, losses, metrics, Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM, 
                          Embedding, 
                          BatchNormalization,
                          Dense, 
                          TimeDistributed, 
                          Dropout, 
                          Bidirectional,
                          Flatten, 
                          GlobalMaxPool1D)

from transformers import TFAutoModelForSequenceClassification, TFAutoModel, AutoTokenizer


# In[4]:


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 2021
seed_everything(seed)


# In[5]:


warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 150)


# In[6]:


# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

primary_green = px.colors.qualitative.Plotly[2]

plotly_discrete_sequence = px.colors.qualitative.G10


# In[7]:


colors = [primary_blue, primary_blue2, primary_blue3, primary_grey, primary_black, primary_bgcolor, primary_green]
sns.palplot(sns.color_palette(colors))


# In[8]:


sns.palplot(sns.color_palette(plotly_discrete_sequence))


# In[9]:


plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.facecolor'] = primary_bgcolor


# In[10]:


custom_theme = Theme({
    "info" : "italic bold blue",
    "succeed": "italic bold green",
    "danger": "bold red"
})

console = Console(theme=custom_theme)

pretty.install()


# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:250%; text-align:center; border-radius: 15px 50px;">CommonLit Readability 📝 A complete Analysis</p>
# 
# ![nlp-header.png](attachment:4a916c95-e05d-4636-98fa-3e82cb1066ce.png)
# 
# **Natural Language Processing or NLP** is a branch of Artificial Intelligence which deal with bridging the machines understanding humans in their Natural Language. Natural Language can be in form of text or sound, which are used for humans to communicate each other. NLP can enable humans to communicate to machines in a natural way.
# 
# **Text Classification** is a process involved in Sentiment Analysis. It is classification of peoples opinion or expressions into different sentiments. Sentiments include Positive, Neutral, and Negative, Review Ratings and Happy, Sad. Sentiment Analysis can be done on different consumer centered industries to analyse people's opinion on a particular product or subject.
# 
# Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence.

# <a id="table-of-content"></a>
# 
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center; border-radius: 15px 50px;">Table of Content</p>
# 
# * [1. Loading Data 💎](#1)
# * [2. EDA 📊](#2)
#     * [2.1 Missing values](#2.1)
#     * [2.2 Target and Std_err Distributions 📸](#2.2)
#     * [2.3 Excert overview 🔎](#2.3)
# * [3. Data Preprocessing ⚙️](#3)
#     * [3.1 Cleaning the corpus 🛠](#3.1)
#     * [3.2 Stemming 🛠](#3.2)
#     * [3.3 All together 🛠](#3.3)
# * [4. Tokens visualization 📊](#4)
#     * [4.1 Top Words 📝](#4.1)
#     * [4.2 WordCloud 🌟](#4.2)
# * [5. Vectorization](#5)
#     * [5.1 Tunning CountVectorizer](#5.1)
#     * [5.2 TF-IDF](#5.2)
#     * [5.3 Word Embeddings: GloVe](#5.3)
# * [6. Modeling](#6)
#     * [6.1 XGBoost](#6.1)
# * [7. LSTM with Glove](#7)
# * [8. RoBERTa](#8)
# 
# To be continued..

# <a id='1'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">1. Loading Data 💎</p>
# 
# Just load the dataset and global variables for colors and so on.

# In[11]:


train_df = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
test_df = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
sub_df = pd.read_csv('/kaggle/input/commonlitreadabilityprize/sample_submission.csv')

train_df.shape


# In[12]:


train_df.head()


# In[13]:


target_column = 'target'

train_df.head()


# <a href="#table-of-content">back to table of content</a>
# <a id='2'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">2. EDA 📊</p>
# 
# Now we are going to take a look about the target distribution, missings, messages length and so on.

# <a id='2.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">2.1 Missing values</p>
# 
# As we can see, the only missing values are in: `url_legal` and `license`. For now, we are going to do an analysis based on the `excerpt` text so we can go ahead.

# In[14]:


msno.bar(train_df, color=primary_blue, sort="ascending", figsize=(10,5), fontsize=12)
plt.show()


# <a id='2.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">2.2 Target and Std_err Distributions 📸</p>
# 

# In[15]:


fig2 = ff.create_distplot([train_df[target_column]], [target_column], colors=[primary_blue],
                             bin_size=.05, show_rug=False)
plot_title = f"<span style='font-size:30px; font-family:Serif'><b>{target_column.capitalize()}</b> resume</span>"

fig = go.Figure()

mean_value = train_df[target_column].mean()

fig.add_vrect(
    x0=train_df[target_column].quantile(0.25), 
    x1=train_df[target_column].quantile(0.75), 
    annotation_text="IQR", 
    annotation_position="top left",
    fillcolor=primary_grey, 
    opacity=0.25, 
    line_width=2,
)

fig.add_trace(go.Scatter(
    fig2['data'][1],
    line=dict(
        color=primary_blue, 
        width=1.5,
    ),
    fill='tozeroy'
))

fig.add_vline(
    x=mean_value, 
    line_width=2, 
    line_dash="dash", 
    line_color=primary_black
)
fig.add_annotation(
    yref="y domain",
    x=mean_value,
    # The arrow head will be 40% along the y axis, starting from the bottom
    y=0.5,
    axref="x",
    ayref="y domain",
    ax=mean_value + 0.2*mean_value,
    ay=0.6,
    text=f"<span>{target_column.capitalize()} mean</span>",
    arrowhead=2,
)
fig.add_annotation(
    xref="x domain", yref="y domain",
    x=0.98, y=0.98,
    text=f"<span><b>Skew: %.2f</b></span>"%(train_df[target_column].skew()),
    bordercolor=primary_black,
    borderwidth=1.5, borderpad=2.5,
    showarrow=False,
)

fig.update_layout(
    showlegend=True,
    title_text=plot_title
)

fig.show()


# In[16]:


###### Helpers not used:

#fig.add_annotation(
#    yref="y3 domain",
#    xref="x3",
#    x=q1_value,
#    # The arrow head will be 40% along the y axis, starting from the bottom
#    y=0.95,
#    axref="x3",
#    ayref="y3 domain",
#    ay=0.95,
#    ax=q1_value + abs(0.2*q1_value),
#    text="Interquartile range (IQR)",
#    arrowhead=3,
#)


#fig.add_annotation(
#    yref="y3 domain",
#    xref="x3",
#    x=mean_value,
#    y=0.5,
#    axref="x3",
#    ayref="y3 domain",
#    ax=mean_value + 0.2*mean_value,
#    ay=0.6,
#    text=f"<span>{feature.capitalize()} mean</span>",
#    arrowhead=3,
#)


#fig.add_shape(go.layout.Shape(
#    type="line",
#    yref="y3 domain",
#    xref="x",
#    x0=mean_value,
#    y0=0,
#    x1=mean_value,
#    y1=1,
#    line=dict(
#        color=primary_black, 
#        width=2, 
#        dash="dash"
#    ),
#), row=3, col=1)


# In[17]:


def generate_feature_resume(df, feature):
    
    plot_title = f"<span style='font-size:30px; font-family:Serif'><b>{feature.capitalize()}</b> resume</span>"
    (osm, osr), (slope, intercept, r) = stats.probplot(df[feature], plot=None)
    
    q1_value = train_df[feature].quantile(0.25)
    mean_value = train_df[feature].mean()
    fig2 = ff.create_distplot([df[feature]], [feature], colors=[primary_blue],
                             bin_size=.05, show_rug=False)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"rowspan": 2}, {"rowspan": 2}],
            [None, None],
            [{"colspan": 2}, None]
        ],
        subplot_titles=(
            "Quantile-Quantile Plot",
            "Box Plot",
            "Distribution Plot"
        )
    )

    fig.add_trace(go.Scatter(
        x=osm,
        y=slope*osm + intercept,
        mode='lines',
        line={
            'color': '#c81515',
            'width': 2.5
        }

    ), row=1, col=1)
    
    ## QQ-Plot
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker={
            'color': primary_blue
        }
    ), row=1, col=1)

    ## Box Plot
    fig.add_trace(go.Box(
        y=df[feature], 
        name='',
        marker_color = primary_blue
    ), row=1, col=2)

    ## Distribution plot
    fig.add_trace(go.Scatter(
        fig2['data'][1],
        line=dict(
            color=primary_blue, 
            width=1.5,
        ),
        fill='tozeroy'
    ), row=3, col=1)
    
    ## InterQuartile Range (IQR)
    fig.add_vrect(
        x0=df[feature].quantile(0.25), 
        x1=df[feature].quantile(0.75), 
        annotation_text="IQR", 
        annotation_position="top left",
        fillcolor=primary_grey, 
        opacity=0.25, 
        line_width=2,
        row=3, col=1,
    )
    
    ## Mean line
    fig.add_vline(
        x=mean_value,
        line_width=2, 
        line_dash="dash", 
        line_color=primary_black,
        annotation_text="Mean", 
        annotation_position="bottom right",
        row=3, col=1,
    )
    
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.98, y=0.98,
        text=f"<span style='font-family:Serif>Skew: %.2f</span>"%(df[feature].skew()),
        showarrow=False,
        bordercolor=primary_black,
        borderwidth=1, borderpad=2,
        row=3, col=1,
    )
    
    fig.update_layout(
        showlegend=False, 
        title_text=plot_title,
        height=720,
    )

    fig.show()


# In[18]:


generate_feature_resume(train_df, target_column)


# In[19]:


# As there is a f*** 0 in the standard_error feature, we are going to change it with the value of the next lowest element
train_df.loc[train_df['standard_error'] == 0, 'standard_error'] = train_df['standard_error'].sort_values(ascending=True).iloc[1]


# In[20]:


generate_feature_resume(train_df, "standard_error")


# In[21]:


sns.jointplot(
    x=train_df['target'], 
    y=train_df['standard_error'], 
    kind='hex',
    height=8,
    edgecolor=primary_grey,
    color=primary_blue
)
plt.suptitle("Target vs Standard error ",font="Serif", size=20)
plt.subplots_adjust(top=0.95)
plt.show()


# <a id='2.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">2.3 Excert overview 🔎</p>

# In[22]:


train_df['excerpt_len'] = train_df['excerpt'].apply(
    lambda x : len(x)
)
train_df['excerpt_word_count'] = train_df['excerpt'].apply(
    lambda x : len(x.split(' '))
)


# In[23]:


fig = ff.create_distplot(
    [train_df['excerpt_len']], 
    ['excerpt_len'], 
    bin_size=12, 
    show_rug=False,
    colors=[primary_blue],
)
fig.update_layout(
    showlegend=False, 
    title_text=f"<span style='font-size:30px; font-family:Serif'><b>Excerpt</b> length</span>",
)
fig.show()


# In[24]:


fig = ff.create_distplot(
    [train_df['excerpt_word_count']], 
    ['excerpt_word_count'], 
    bin_size=2, 
    show_rug=False,
    colors=[primary_blue],
)
fig.update_layout(
    showlegend=False, 
    title_text=f"<span style='font-size:30px; font-family:Serif'><b>Excerpt</b> word count</span>",
)
fig.show()


# <a href="#table-of-content">back to table of content</a>
# <a id='3'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">3. Data Preprocessing ⚙️</p>
# 
# Now we are going to engineering the data to make it easier for the model to clasiffy.
# 
# This section is very important to reduce the dimensions of the problem.

# <a id='3.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">3.1 Cleaning the corpus 🛠</p>

# In[25]:


# Special thanks to https://www.kaggle.com/tanulsingh077 for this function
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[26]:


train_df['excerpt_clean'] = train_df['excerpt'].apply(clean_text)
train_df.head()


# ### Stopwords
# Stopwords are commonly used words in English which have no contextual meaning in an sentence. So therefore we remove them before classification. Some examples removing stopwords are:
# 
# ![stopwords.png](attachment:a023a8e1-af19-4555-875a-8fc533b0c580.png)

# In[27]:


stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    
train_df['excerpt_clean'] = train_df['excerpt_clean'].apply(remove_stopwords)
train_df.head()


# In[ ]:





# <a id='3.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">3.2 Stemming 🛠</p>
# 
# ### Stemming/ Lematization
# For grammatical reasons, documents are going to use different forms of a word, such as *write, writing and writes*. Additionally, there are families of derivationally related words with similar meanings. The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.
# 
# **Stemming** usually refers to a process that chops off the ends of words in the hope of achieving goal correctly most of the time and often includes the removal of derivational affixes.
# 
# **Lemmatization** usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base and dictionary form of a word
# 
# ![stemming-lematization.png](attachment:e2f093fa-6bec-42b3-963b-a9fbc60761f0.png)
# 
# As far as the meaning of the words is not important for this study, we will focus on stemming rather than lemmatization.
# 
# ### Stemming algorithms
# 
# There are several stemming algorithms implemented in NLTK Python library:
# 1. **PorterStemmer** uses *Suffix Stripping* to produce stems. **PorterStemmer is known for its simplicity and speed**. Notice how the PorterStemmer is giving the root (stem) of the word "cats" by simply removing the 's' after cat. This is a suffix added to cat to make it plural. But if you look at 'trouble', 'troubling' and 'troubled' they are stemmed to 'trouble' because *PorterStemmer algorithm does not follow linguistics rather a set of 05 rules for different cases that are applied in phases (step by step) to generate stems*. This is the reason why PorterStemmer does not often generate stems that are actual English words. It does not keep a lookup table for actual stems of the word but applies algorithmic rules to generate stems. It uses the rules to decide whether it is wise to strip a suffix.
# 2. One can generate its own set of rules for any language that is why Python nltk introduced **SnowballStemmers** that are used to create non-English Stemmers!
# 3. **LancasterStemmer** (Paice-Husk stemmer) is an iterative algorithm with rules saved externally. One table containing about 120 rules indexed by the last letter of a suffix. On each iteration, it tries to find an applicable rule by the last character of the word. Each rule specifies either a deletion or replacement of an ending. If there is no such rule, it terminates. It also terminates if a word starts with a vowel and there are only two letters left or if a word starts with a consonant and there are only three characters left. Otherwise, the rule is applied, and the process repeats.

# In[28]:


stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


# In[29]:


train_df['excerpt_clean'] = train_df['excerpt_clean'].apply(stemm_text)
train_df.head()


# <a id='3.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">3.3 All together 🛠</p>

# In[30]:


def preprocess_data(text, strip=False):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    
    if strip:
        text = text.strip()
    
    return text


# In[31]:


train_df['excerpt_clean'] = train_df['excerpt_clean'].apply(preprocess_data)
train_df.head()


# In[32]:


console.print('First, lets see the original text:')
console.print(train_df['excerpt'][0], style='info')

console.print('And now, lets see the clean text:')
console.print(train_df['excerpt_clean'][0], style='succeed')


# <a href="#table-of-content">back to table of content</a>
# <a id='4'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">4. Tokens visualization 📊</p>
# 
# Let's see which are the top tockens using count vectorizer and ranking based on the appearence. They idea is to have a first overview of the relevance of each word or tuple of words.

# <a id='4.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">4.1 Top Words 📝</p>

# In[33]:


def get_top_n_grams(vect, corpus, n):
    # Use the CountVectorizer to create a document-term matrix
    dtm = vect.fit_transform(corpus)

    dtm_sum = dtm.sum(axis=0) 
    dtm_freq = [(word, dtm_sum[0, idx]) for word, idx in vect.vocabulary_.items()]
    dtm_freq =sorted(dtm_freq, key = lambda w: w[1], reverse=True)
    return dict(sorted(dtm_freq[:n], key = lambda w: w[1], reverse=False))

def plot_top_grams(grams, groups, title):
    fig = go.Figure(go.Bar(
        x=list(grams.values()), y=list(grams.keys()),
        orientation='h',
    ))
    # Customize aspect
    fig.update_traces(
        marker_color=groups[2]*[primary_grey] + groups[1]*[primary_blue2] + groups[0]*[primary_blue], 
        marker_line_color=primary_blue3,
        marker_line_width=1, 
        opacity=0.6
    )
    fig.update_layout(
        title_text=f"<span style='font-size:30px; font-family:Serif'>{title}</span>"
    )
    fig.show()


# In[34]:


vect = CountVectorizer()
top_unigrams = get_top_n_grams(vect, train_df['excerpt_clean'], 15)

plot_top_grams(top_unigrams, [1,5,9], "Top 15 <b>Unigrams</b>")


# In[35]:


vect = CountVectorizer(ngram_range=(2, 2))
top_bigrams = get_top_n_grams(vect, train_df['excerpt_clean'], 15)

plot_top_grams(top_bigrams, [3,7,5], "Top 15 <b>Bigrams</b>")


# In[36]:


vect = CountVectorizer(ngram_range=(3, 3))
top_trigrams = get_top_n_grams(vect, train_df['excerpt_clean'], 15)

plot_top_grams(top_trigrams, [3,4,8], "Top 15 <b>Trigrams</b>")


# <a id='4.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">4.2 WordCloud 🌟</p>

# In[37]:


book_mask = np.array(Image.open('/kaggle/input/masksforwordclouds/book-logo-1.jpg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=book_mask,
)
wc.generate(' '.join(text for text in train_df.loc[:, 'excerpt_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()


# <a href="#table-of-content">back to table of content</a>
# <a id='5'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">5. Baseline Model and Comparison</p>
# 
# Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
# 
# We'll do that in three steps using the bag-of-words model:
# 
# 1. Count how many times does a word occur in each message (Known as term frequency)
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
# 
# Let's begin the first step:
# 
# Each vector will have as many dimensions as there are unique words in the SMS corpus. We will first use SciKit Learn's **CountVectorizer**. This model will convert a collection of text documents to a matrix of token counts.
# 
# We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message.
# 
# ![vectorizer.png](attachment:fe0ac5b9-f924-4a7c-b8e3-306485322784.png)

# In[38]:


rmse = lambda y_true, y_pred: np.sqrt(mse(y_true, y_pred))
rmse_loss = lambda Estimator, X, y: rmse(y, Estimator.predict(X))


# In[39]:


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
x = train_df['excerpt_clean']
y = train_df['target']

print(len(x), len(y))


# In[40]:


# Split into train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


# In[41]:


model = make_pipeline(
    CountVectorizer(ngram_range=(1,1)),
    LinearRegression(),
)

val_score = cross_val_score(
    model, 
    train_df['excerpt_clean'], 
    train_df['target'], 
    scoring=rmse_loss
).mean()

console.print(f'Train Score for CountVectorizer(1,1): {val_score}')


# In[42]:


model = make_pipeline(
    CountVectorizer(ngram_range=(2,2)),
    LinearRegression(),
)

val_score = cross_val_score(
    model, 
    train_df['excerpt_clean'], 
    train_df['target'], 
    scoring=rmse_loss
).mean()

console.print(f'Train Score for CountVectorizer(1,1): {val_score}')


# In[43]:


model = make_pipeline(
    CountVectorizer(ngram_range=(1,2)),
    LinearRegression(),
)

val_score = cross_val_score(
    model, 
    train_df['excerpt_clean'], 
    train_df['target'], 
    scoring=rmse_loss
).mean()

console.print(f'Train Score for CountVectorizer(1,1): {val_score}')


# As we can see, it seems that the best result is achived using `ngram_range=(1,2)` which has much sense as we are going to see in the following section.>

# In[44]:


# Now create the train and test dtm
vect = CountVectorizer()
vect.fit(x_train)

# Use the trained to create a document-term matrix from train and test sets
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)


# <a id='5.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">5.1 Tunning CountVectorizer</p>
# 
# CountVectorizer has a few parameters you should know.
# 
# 1. **stop_words**: Since CountVectorizer just counts the occurrences of each word in its vocabulary, extremely common words like ‘the’, ‘and’, etc. will become very important features while they add little meaning to the text. Your model can often be improved if you don’t take those words into account. Stop words are just a list of words you don’t want to use as features. You can set the parameter stop_words=’english’ to use a built-in list. Alternatively you can set stop_words equal to some custom list. This parameter defaults to None.
# 
# 2. **ngram_range**: An n-gram is just a string of n words in a row. E.g. the sentence ‘I am Groot’ contains the 2-grams ‘I am’ and ‘am Groot’. The sentence is itself a 3-gram. Set the parameter ngram_range=(a,b) where a is the minimum and b is the maximum size of ngrams you want to include in your features. The default ngram_range is (1,1). In a recent project where I modeled job postings online, I found that including 2-grams as features boosted my model’s predictive power significantly. This makes intuitive sense; many job titles such as ‘data scientist’, ‘data engineer’, and ‘data analyst’ are 2 words long.
# 
# 3. **min_df, max_df**: These are the minimum and maximum document frequencies words/n-grams must have to be used as features. If either of these parameters are set to integers, they will be used as bounds on the number of documents each feature must be in to be considered as a feature. If either is set to a float, that number will be interpreted as a frequency rather than a numerical limit. min_df defaults to 1 (int) and max_df defaults to 1.0 (float).
# 
# 4. **max_features**: This parameter is pretty self-explanatory. The CountVectorizer will choose the words/features that occur most frequently to be in its’ vocabulary and drop everything else. 
# 
# You would set these parameters when initializing your CountVectorizer object as shown below.

# In[45]:


vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)


# <a id='5.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">5.2 TF-IDF</p>
# 
# In information retrieval, tf–idf, **TF-IDF**, or TFIDF, **short for term frequency–inverse document frequency**, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. 
# 
# **tf–idf** is one of the most popular term-weighting schemes today. A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use tf–idf.
# 
# ![tf-idf.png](attachment:ed3d959e-bd1b-446f-af2f-775e4364b2b7.png)

# In[46]:


model = make_pipeline(
    TfidfVectorizer(ngram_range=(1,1)),
    LinearRegression()
)

val_score = cross_val_score(
    model, 
    train_df['excerpt_clean'], 
    train_df['target'], 
    scoring=rmse_loss
).mean()

console.print(f'Train Score for TfidfVectorizer(1,1): {val_score}')


# In[47]:


model = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2)),
    LinearRegression()
)

val_score = cross_val_score(
    model, 
    train_df['excerpt_clean'], 
    train_df['target'], 
    scoring=rmse_loss
).mean()

console.print(f'Train Score for TfidfVectorizer(1,1): {val_score}')


# Again, the best selection for `ngram` parameter is the tuple `(1,2)`.

# In[48]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

x_train_tfidf


# <a id='5.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">5.3 Word Embeddings: GloVe</p>

# In[49]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


# In[50]:


texts = train_df['excerpt_clean']
target = train_df['target']


# We need to perform **tokenization** - the processing of segmenting text into sentences of words. The benefit of tokenization is that it gets the text into a format that is easier to convert to raw numbers, which can actually be used for processing.
# 
# ![tokenization.jpeg](attachment:156686a1-ffd0-4f85-be0a-d1fd31ea8a64.jpeg)

# In[51]:


# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length


# ### Pad_sequences
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
# 
# ```python
# tf.keras.preprocessing.sequence.pad_sequences(
#     sequences, maxlen=None, dtype='int32', padding='pre',
#     truncating='pre', value=0.0
# )
# ```
# 
# This function transforms a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the longest sequence in the list.
# 
# ```python
# >>> sequence = [[1], [2, 3], [4, 5, 6]]
# >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
# array([[1, 0, 0],
#        [2, 3, 0],
#        [4, 5, 6]], dtype=int32)
# ```

# In[52]:


def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)

train_padded_sentences


# ### GloVe
# 
# GloVe method is built on an important idea,
# 
# > You can derive semantic relationships between words from the co-occurrence matrix.
# 
# To obtain a vector representation for words we can use an unsupervised learning algorithm called **GloVe (Global Vectors for Word Representation)**, which focuses on words co-occurrences over the whole corpus. Its embeddings relate to the probabilities that two words appear together.
# 
# Word embeddings are basically a form of word representation that bridges the human understanding of language to that of a machine. They have learned representations of text in an n-dimensional space where words that have the same meaning have a similar representation. Meaning that two similar words are represented by almost similar vectors that are very closely placed in a vector space.
# 
# Thus when using word embeddings, all individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values are learned in a way that resembles a neural network.

# In[53]:


embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions


# In[54]:


# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
embedding_matrix


# <a href="#table-of-content">back to table of content</a>
# <a id='6'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">6. Modeling</p>

# <a id='6.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:140%; text-align:center; border-radius: 15px 50px;">6.1 TF-IDF with XGBoost</p>

# In[55]:


from sklearn.pipeline import Pipeline
import xgboost as xgb

pipe = Pipeline([
    ('tfid', TfidfVectorizer(ngram_range=(1,2))),  
    ('model', xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='rmse',
    ))
])

# Fit the pipeline with the data
pipe.fit(x_train, y_train)


# In[56]:


y_pred = pipe.predict(x_test)

console.print(f'Score for XGBoost with TfidfVectorizer(1,2): {rmse(y_test, y_pred)}')


# <a href="#table-of-content">back to table of content</a>
# <a id='7'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">7. LSTM</p>

# In[57]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    target, 
    test_size=0.25
)


# In[58]:


# Model from https://www.kaggle.com/mariapushkareva/nlp-disaster-tweets-with-glove-and-lstm/data

def glove_lstm():
    model = Sequential()
    
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0], 
        output_dim=embedding_matrix.shape[1], 
        weights = [embedding_matrix], 
        input_length=length_long_sentence
    ))
    
    model.add(Bidirectional(LSTM(
        length_long_sentence, 
        return_sequences = True, 
        recurrent_dropout=0.2
    )))
    
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'linear'))
    
    model.compile(Adam(lr=1e-5), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    
    return model

model = glove_lstm()
model.summary()


# In[59]:


# Load the model and train!!

model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss', 
    verbose = 1, 
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)
history = model.fit(
    X_train, 
    y_train, 
    epochs = 10,
    batch_size = 32,
    validation_data = (X_test, y_test),
    verbose = 1,
    callbacks = [reduce_lr, checkpoint]
)


# In[60]:


def plot_learning_curves(history, arr):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for idx in range(2):
        ax[idx].plot(history.history[arr[idx][0]])
        ax[idx].plot(history.history[arr[idx][1]])
        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)
        ax[idx].set_xlabel('A ',fontsize=16)
        ax[idx].set_ylabel('B',fontsize=16)
        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)


# In[61]:


plot_learning_curves(history, [['loss', 'val_loss'],['root_mean_squared_error', 'val_root_mean_squared_error']])


# <a href="#table-of-content">back to table of content</a>
# <a id='8'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">8. ToBERTa TF 🛠</p>
# 
# Special thanks to @sauravmaheshkar and @dimitreoliveira for such amazing work and thanks for sharing your code. It was ery helpful.

# In[62]:


# Sampling Function
def sample_target(features, target):
    mean, stddev = target
    sampled_target = tf.random.normal([], mean=tf.cast(mean, dtype=tf.float32), 
                                      stddev=tf.cast(stddev, dtype=tf.float32), dtype=tf.float32)
    
    return (features, sampled_target)

# Convert to tf.data.Dataset
def get_dataset(df, tokenizer, labeled=True, ordered=False, repeated=False, 
                is_sampled=False, batch_size=32, seq_len=256):
    
    texts = [preprocess_data(text, True) for text in df['excerpt']]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(texts, max_length=seq_len, truncation=True, 
                                 padding='max_length', return_tensors='tf')
    
    if labeled:
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': tokenized_inputs['input_ids'], 
                                                      'attention_mask': tokenized_inputs['attention_mask']}, 
                                                      (df[target_column], df['standard_error'])))
        if is_sampled:
            dataset = dataset.map(sample_target, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices({'input_ids': tokenized_inputs['input_ids'], 
                                                      'attention_mask': tokenized_inputs['attention_mask']})
        
    if repeated:
        dataset = dataset.repeat()
    if not ordered:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# In[63]:


# TPU or GPU detection
# Detect hardware and create the ad-hoc strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU {tpu.master()}')
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    print('Using GPU strategy...')
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


# In[64]:


BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 5
STEPS=50
N_FOLDS = 5
ES_PATIENCE = 7
SEQ_LEN = 256
BASE_MODEL = '/kaggle/input/huggingface-roberta/roberta-base/'
proper_names = ['fayre', 'roger', 'blaney']


# In[65]:


def model_fn(encoder, seq_len=256):
    input_ids = layers.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_attention_mask = layers.Input(shape=(seq_len,), dtype=tf.int32, name='attention_mask')
    
    outputs = encoder({'input_ids': input_ids, 
                       'attention_mask': input_attention_mask})
    
    model = Model(
        inputs=[input_ids, input_attention_mask], 
        outputs=outputs
    )

    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, 
        loss=losses.MeanSquaredError(), 
        metrics=[metrics.RootMeanSquaredError()]
    ) 
    return model


with strategy.scope():
    encoder = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model = model_fn(encoder, SEQ_LEN)
    
model.summary()


# In[66]:


tf.keras.utils.plot_model(
    model,
    show_shapes=True, show_dtype=False,
    show_layer_names=False, rankdir='TB', 
    expand_nested=False
)


# In[67]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\ntokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)\n\nskf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)\n\noof_pred = []; oof_labels = []\nhistory_list = []; test_pred = []\n\nfor fold,(idxT, idxV) in enumerate(skf.split(train_df)):\n    if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)\n    \n    print(f\'\\nFOLD: {fold+1}\')\n    print(f\'TRAIN: {len(idxT)} VALID: {len(idxV)}\')\n\n    # Model\n    backend.clear_session()\n    with strategy.scope():\n        encoder = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)\n        model = model_fn(encoder, SEQ_LEN)\n        \n    model_path = f\'model_{fold}.h5\'\n    es = EarlyStopping(monitor=\'val_root_mean_squared_error\', mode=\'min\', \n                       patience=ES_PATIENCE, restore_best_weights=True, verbose=1)\n    checkpoint = ModelCheckpoint(model_path, monitor=\'val_root_mean_squared_error\', mode=\'min\', \n                                 save_best_only=True, save_weights_only=True)\n\n    # Train\n    history = model.fit(\n        x=get_dataset(\n            train_df.loc[idxT], \n            tokenizer, \n            repeated=True, \n            is_sampled=True, \n            batch_size=BATCH_SIZE, \n            seq_len=SEQ_LEN\n        ), \n        validation_data=get_dataset(\n            train_df.loc[idxV], \n            tokenizer, \n            ordered=True, \n            batch_size=BATCH_SIZE, seq_len=SEQ_LEN\n        ), \n        steps_per_epoch=STEPS, \n        callbacks=[es, checkpoint], \n        epochs=EPOCHS,  \n        verbose=2\n    ).history\n      \n    history_list.append(history)\n    # Save last model weights\n    model.load_weights(model_path)\n    \n    # Results\n    print(f"#### FOLD {fold+1} OOF RMSE = {np.min(history[\'val_root_mean_squared_error\']):.4f}")\n\n    # OOF predictions\n    valid_ds = get_dataset(\n        train_df.loc[idxV], \n        tokenizer, \n        ordered=True, batch_size=BATCH_SIZE, seq_len=SEQ_LEN\n    )\n    oof_labels.append([target[0].numpy() for sample, target in iter(valid_ds.unbatch())])\n    x_oof = valid_ds.map(lambda sample, target: sample)\n    oof_pred.append(model.predict(x_oof)[\'logits\'])\n\n    # Test predictions\n    test_ds = get_dataset(\n        test_df, \n        tokenizer, \n        labeled=False, ordered=True, batch_size=BATCH_SIZE, seq_len=SEQ_LEN\n    )\n    x_test = test_ds.map(lambda sample: sample)\n    test_pred.append(model.predict(x_test)[\'logits\'])\n')


# In[ ]:





# In[ ]:




