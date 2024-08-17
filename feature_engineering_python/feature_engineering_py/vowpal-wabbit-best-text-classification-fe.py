#!/usr/bin/env python
# coding: utf-8

# ![](https://kaggle2.blob.core.windows.net/competitions/kaggle/10737/logos/thumb76_76.png?t=2018-10-22-18-50-58t=2018-10-24-17-14-05)
# <h2 style="color:  #aa2200;text-align: center;" id="Quora-Insincere-Questions-Classification">Quora Insincere Questions Classification</h2>
# <ul style="align:justify;color: #aa2200;">
#  <li><span style="text-color: #aa2200;">An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.<a style="color: #aa2200;" href="https://www.quora.com/" rel="nofollow">Quora</a>&nbsp;is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.</span></li>
# <li><span style="text-color: #aa2200;">In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.</span></li>
# <li><span style="text-color: #aa2200;">Here's your chance to combat online trolls at scale. Help Quora uphold their policy of &ldquo;Be Nice, Be Respectful&rdquo; and continue to be a place for sharing and growing the world&rsquo;s knowledge.</span></li>
# </ul>
# <hr>
# <h2 style="color: #aa2200;text-align: Left;" id="Quora-Insincere-Questions-Classification">Outline of the Notebook</h2>
# <hr>
# <h4 style="color: #aa2200;text-align: Left;" >Objective : Predict whether a question asked on Quora is sincere or not. This is a kernels only comeptition.</h4>
# <ul>
# <li><a href="#1.Data-Reading-and-Investigation" style="color:#aa2200;">1.Data Reading and Investigation</a></li>
# <li><a href="#2.Target-variable-Distribution-Check" style="color:#aa2200;">2.Target variable Distribution Check</a></li>
# <li><a href="#3.Word-Cloud-of-Sincere-and-Insincere-Target-Class" style="color:#aa2200;">3.Word Cloud of Sincere and Insincere Target Class</a></li>
# <li><a href="#4.-Word-Count-by-Sincere-Vs-Insincere" style="color:#aa2200;">4. Word Count by Sincere Vs Insincere</a></li>
# <li><a href="#5.-Bi-gram-by-Sincere-Vs-Insincere" style="color:#aa2200;">5. Bi-gram by Sincere Vs Insincere</a></li>
# <li><a href="#6.-Tri-gram-by-Sincere-Vs-Insincere" style="color:#aa2200;">6. Tri-gram by Sincere Vs Insincere</a></li>
# <li><a href="#7.-Meta-Feature-Engineering" style="color:#aa2200;">7. Meta Feature Engineering</a></li>
# <li><a href="#8.-Model-Training-using-Vowpal-Wabbit-Algorithm" style="color:#aa2200;">8. Model Training using Vowpal Wabbit Algorithm</a></li>
# <li><a href="#9.-Model-Training" style="color:#aa2200;">9. Model Training</a></li>
# <li><a href="#10.-F1-Score-Graph" style="color:#aa2200;">10. F1-Score Graph</a></li>
# <li><a href="#11.-Final-Submission" style="color:#aa2200;">11. Final Submission</a></li>
# </ul>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# https://www.kaggle.com/arunkumarramanan/market-data-nn-baseline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from vowpalwabbit import pyvw
import sklearn.metrics as metrics
from gensim.parsing import preprocessing as prep
import plotly.tools as tls
import warnings
# from plotly.tools import FigureFactory as FF 
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff


######### Function
def mis_value_graph(data):
#     data.isnull().sum().plot(kind="bar", figsize = (20,10), fontsize = 20)
#     plt.xlabel("Columns", fontsize = 20)
#     plt.ylabel("Value Count", fontsize = 20)
#     plt.title("Total Missing Value By Column", fontsize = 20)
#     for i in range(len(data)):
#          colors.append(generate_color())
            
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'Unknown Assets',
        textfont=dict(size=20),
        marker=dict(
#         color= colors,
        line=dict(
            color=generate_color(),
            width=2,
        ), opacity = 0.45
    )
    ),
    ]
    layout= go.Layout(
        title= '"Total Missing Value By Column"',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='skin')
    

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color


# <h3 style="color: #aa2200;text-align: Left;" > 1.Data Reading and Investigation</h3>

# In[ ]:


train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
mis_value_graph(train)
print("Train Shape:",train.shape)
display(train.isna().sum().to_frame())
print("=====Train Data Column Types=====")
display(train.dtypes)
print("=====Train Data=====")
display(train.head())


# In[ ]:


test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
mis_value_graph(train)
print("Test Shape:",test.shape)
display(test.isna().sum().to_frame())
print("=====Test Data Column Types=====")
display(test.dtypes)
print("=====Test Data=====")
display(train.head())


# <h3 style="color: #aa2200;text-align: Left;" >2.Target variable Distribution Check</h3>

# In[ ]:


colors = ['#FEBFB3', '#aa2200', '#aa2222', '#aa22aa']
trace1 = go.Pie(
labels = ['Sincere','Insincere'],
values = train.target.value_counts(),
textfont=dict(size=20),
marker=dict(colors=colors,line=dict(color='#aa2200', width=2)), hole = 0.45)
layout = dict(title = "Sincere vs Insincere Comments")
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


## target count ##
cnt_srs = train['target'].value_counts()
trace = go.Bar(
    x=['Sincere','Insincere'],
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'RdBu', # ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
#             'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
#             'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']       
        reversescale = True
    ),
)
layout = go.Layout(
    title='Target Count',
    font=dict(size=18)
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")
display('We can see that clearly here class imbalance Problem')


# <h3 style="color: #aa2200;text-align: Left;" >3.Word Cloud of Sincere and Insincere Target Class</h3>

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=500, 
                    height=300,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': '#aa2200', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    

comments_mask = np.array(Image.open("../input/quora24/img_44218.png"))
plot_wordcloud(train["question_text"], comments_mask,title = 'Question Words Frequency of Quora')


# <h3 style="color: #aa2200;text-align: Left;" >4. Word Count by Sincere Vs Insincere</h3>

# In[ ]:


# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
from collections import defaultdict
train1_df = train[train["target"]==1]
train0_df = train[train["target"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=[i for j in range(100) for i in ['#aa2200','#2b6dad',generate_color(),generate_color(),generate_color()]],
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Sincere Question")
display(fd_sorted.head(10))
trace0 = horizontal_bar_chart(fd_sorted.head(30), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Insincere Question")
display(fd_sorted.head(10))
trace1 = horizontal_bar_chart(fd_sorted.head(50), color = 'blue')

# Creating two subplots
fig = tls.make_subplots(rows=1, cols=2, vertical_spacing=0.04,subplot_titles=["Frequent words of sincere questions","Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900,title="Word Count Plots")
py.iplot(fig, filename='word-plots')


# <h6 style="align:justify;color: #aa2200;"> Words Counts Insights</h6>
# 
# <ul style="align:justify;color: #aa2200;">
# <li><span style="text-color: #aa2200;">We can see that their are so many comman words people use to write their question here.</span></li>
# <li><span style="text-color: #aa2200;"><b>In Insincere Graph</b> you can check insincere bar plot <b>`people,trump, women, will, think, many, white, men, indian. muslims` are most used words more than 4k times</b>.</span></li>
# <li><span style="text-color: #aa2200;"><b>In Sincere Graph</b>you can check Sincere bar plot <b>`Best, will, people, good, one, make, think, many, much, someone etc.`  are most used words more than 30k times.</b></span></li>
# </ul>
# <h3 style="color: #aa2200;text-align: Left;" >5. Bi-gram by Sincere Vs Insincere</h3>

# In[ ]:


freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Sincere Question")
display(fd_sorted.head(10))
trace0 = horizontal_bar_chart(fd_sorted.head(30), generate_color())


freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Insincere Question")
display(fd_sorted.head(10))
trace1 = horizontal_bar_chart(fd_sorted.head(30), generate_color())

# Creating two subplots
fig = tls.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams of sincere questions", 
                                          "Frequent bigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900,title="Bigram Word pair Count Plots")
py.iplot(fig, filename='word-plots')


# <h6 style="align:justify;color: #aa2200;"> Bigram Insights</h6>
# <ul style="align:justify;color: #aa2200;">
# <li><span style="text-color: #aa2200;">A <b>bigram</b> is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words. A bigram is an n-gram for n=2</span></li>
# <li><span style="text-color: #aa2200;"><b>In Sincere Graph</b> you can check Sincere bar plot <b>`Best way, Year old, will happen` are most used words more than 2000 times.</b></span></li>
# <li><span style="text-color: #aa2200;"><b>In Insincere Graph</b> you can check Insincere bar plot <b>`Donald Trump, White Perople, black people, many people etc.` are most used words more than 500 times.</b></span></li></ul>
# 
# <h3 style="color: #aa2200;text-align: Left;" >6. Tri-gram by Sincere Vs Insincere</h3>

# In[ ]:


freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Sincere Question")
display(fd_sorted.head(10))
trace0 = horizontal_bar_chart(fd_sorted.head(30), 'green')


freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
print("Frequency Trigram for Insincere Question")
display(fd_sorted.head(10))
trace1 = horizontal_bar_chart(fd_sorted.head(30), 'green')

# Creating two subplots
fig = tls.make_subplots(rows=1, cols=2, vertical_spacing=0.5, horizontal_spacing=0.2,
                          subplot_titles=["Frequent trigrams of sincere questions", 
                                          "Frequent trigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, title="Trigram Count Plots")
py.iplot(fig, filename='word-plots')


# <h6 style="align:justify;color: #aa2200;"> Trigram Insights</h6>
# <ul style="align:justify;color: #aa2200;">
# <li><span style="text-color: #aa2200;">A <b>Trigra</b> is a sequence of <b>three adjacent elements</b>from a <b>string of tokens</b>, which are typically <b>letters, syllables, or words</b>. A <b>Trigram</b>is an n-gram for <b>n=3</b></span></li>
# <li><span style="text-color: #aa2200;"><b>In Sincere Graph</b> you can check Sincere bar plot <b>`tips someone starting, someone starting work, useful tips someone, advice give someone,short-term business travelers, hotels short-term business, good hotels short-term, give someone moving, good bad neighborhoods, best known for? etc.`*are most used words more than 400 times.</b></span></li>
# <li><span style="text-color: #aa2200;"><b>In Insincere Graph</b> you can check Sincere bar plot <b>`will donald trump, black lives matter, long will take, kim jong un,12 year old, people still believe,14 year old,united states america.`are most used words more than 30 times.</b></span></li></ul>

# <h3 style="color: #aa2200;text-align: Left;" >7. Meta Feature Engineering</h3>
# <p style="color: #aa2200;text-align: Left;">(From SRK's Diary)</p>
# 
# <ul style="align:justify;color: #aa2200;">
# <p style="color: #aa2200;text-align: Left;">Now let us create some meta features and then look at how they are distributed between the classes. The ones that we will create are</p>
# <li><span style="text-color: #aa2200;">Number of words in the text       </span></li>
# <li><span style="text-color: #aa2200;">Number of unique words in the text</span></li>
# <li><span style="text-color: #aa2200;">Number of characters in the text  </span></li>
# <li><span style="text-color: #aa2200;">Number of stopwords               </span></li>
# <li><span style="text-color: #aa2200;">Number of punctuations            </span></li>
# <li><span style="text-color: #aa2200;">Number of upper case words        </span></li>
# <li><span style="text-color: #aa2200;">Number of title case words        </span></li>
# <li><span style="text-color: #aa2200;">Average length of the words       </span></li>
# </ul>

# In[ ]:


import string
## Number of words in the text ##
train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

## Number of punctuations in the text ##
train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["num_words_upper"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_title"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

print("Train shape:",train.shape)
print("Test Shape:",test.shape)


# <h3 style="color: #aa2200;text-align: Left;" >8. Model Training using Vowpal Wabbit Algorithm</h3>
# <ul style="align:justify;color: #aa2200;">
# <li><span style="text-color: #aa2200;">Vowpal Wabbit. Vowpal Wabbit (also known as "VW") is an open source fast out-of-core machine learning system library and program developed originally at Yahoo! Research, and currently at Microsoft Research. It was started and is led by John Langford.</span></li>
# <li><span style="text-color: #aa2200;">The Vowpal Wabbit (VW) is a project started at Yahoo! Research and now sponsored by Microsoft Research. Started and led by John Langford, VW focuses on fast learning by building an intrinsically fast learning algorithm. John gave two guest lectures to us on AllReduce and Bandits during NYU Big Data class this semester. From what I see, he is a reputed researcher and really passionate about online learning algorithm.</span></li>
# <li><span style="text-color: #aa2200;">The Vowpal Wabbit name is oddly pronounced and strange. Langford explained that Vowpal Wabbit is how Elmer Fudd would pronounce “Vorpal Rabbit”. As for “Vorpal”, if you Google it, you will find “Vorpal Bunny”, which is also known as a “killer rabbit” in a popular computer game. Maybe this is exactly what he wants VW to be – cute but also powerful and fast.</span></li>
# <li><span style="text-color: #aa2200;">VW supports a number of machine learning problems, importance weighting, a selection of loss functions and optimization algorithms, like SGD (Stochastic Gradient Descent), BFGS (a a popular algorithm for parameter estimation), conjugate gradient etc. It has been used to learn a sparse terafeature (i.e. 1012 sparse features) dataset on 1000 nodes in one hour, which beats all current machine linear learning algorithms. According to its tutorial on John Langford’s GitHub, VW is about a factor of 3 faster than svmsgd on the RCV1 example, which is a collection for text categorization. </span></li>
# <li><span style="text-color: #aa2200;">For More Reading : https://www.zinkov.com/posts/2013-08-13-vowpal-tutorial/</span></li>
# </ul>
# 
# 
# 

# In[ ]:


# https://www.kaggle.com/hippskill/vowpal-wabbit-starter-pack
class Tokenizer(object):
    def __call__(self, doc): 
        striped = prep.strip_punctuation(doc)
        striped = prep.strip_tags(striped)
        striped = prep.strip_multiple_whitespaces(striped).lower()
        return striped
    
class FilterRareWords(object):
    def __init__(self):
        self.cv = defaultdict(int)
    def fit(self, texts):
        for text in texts:
            for word in text.split():
                self.cv[word] += 1
    def __call__(self, text):
        return ' '.join([self.filter_word(word) for word in text.split()])
    def filter_word(self, word):
        return '' if self.cv[word] < 2 else word

tokenizer = Tokenizer()
filter_words = FilterRareWords()

display(train[train['target'] == 1].head())

train['question_text'] = train['question_text'].apply(tokenizer)

filter_words.fit(train['question_text'])
train['question_text'] = train['question_text'].apply(filter_words)
pos_weight = train['target'].sum() / train.shape[0]
display(train.head())


# In[ ]:


def make_vw_feature_line(label, importance, text):
    return '{} {} |text {}'.format(label, importance, text)

def make_vw_corpus(texts, labels):
    for text, label in zip(texts, labels):
        if label == 1.0:
            cur_feautre = make_vw_feature_line('1', 1 - pos_weight, text)
        else:
            cur_feautre = make_vw_feature_line('-1', pos_weight, text)
        yield cur_feautre


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train, test_size=0.1, shuffle=True, random_state=42)


# In[ ]:


vw = pyvw.vw(
    quiet=True,
    loss_function='logistic',
    link='logistic',
    b=29,
    ngram=2,
    skips=1,
    random_seed=42,
    l1=3.4742122764e-09,
    l2=1.24232077629e-11,
    learning_rate=0.751849318433,
)


# In[ ]:


def get_pred(feature):
    ex = vw.example(feature)
    pred = vw.predict(ex)
    ex.finish()
    return pred


# In[ ]:


feature_map =  ['question_text', 'num_words', 'num_unique_words',
       'num_chars', 'num_stopwords', 'num_punctuations', 'num_words_upper',
       'num_words_title', 'mean_word_len']


# <h3 style="color: #aa2200;text-align: Left;" >9. Model Training</h3>

# In[ ]:


get_ipython().run_cell_magic('time', '', "for fit_iter in range(7):\n    for num, feature in enumerate(make_vw_corpus(X_train['question_text'], X_train['target'])):\n        ex = vw.example(feature)\n        vw.learn(ex)\n        ex.finish()\n        \n    print('pass num {} done'.format(fit_iter))\n")


# In[ ]:


pred = np.array([get_pred(x) for x in make_vw_corpus(X_test['question_text'], X_test['target'])])


# In[ ]:


thresholds = np.linspace(0, 1, 100)
f1_scores = [metrics.f1_score(X_test['target'], pred > threshold) for threshold in thresholds]


# <h3 style="color: #aa2200;text-align: Left;" >10. F1-Score Graph</h3>

# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(thresholds, f1_scores)
plt.grid(True)
plt.show()


# In[ ]:


print('best f1 score is {} with threshold {}'.format(np.max(f1_scores), thresholds[np.argmax(f1_scores)]))


# <h3 style="color: #aa2200;text-align: Left;" >11. Final Submission</h3>

# In[ ]:


test['question_text'] = test['question_text'].apply(tokenizer)
test['question_text'] = test['question_text'].apply(filter_words)

pred = np.array([get_pred(x) for x in make_vw_corpus(test['question_text'], [1] * len(test))])

example = pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv')
example['prediction'] = (pred > thresholds[np.argmax(f1_scores)]).astype(int)
example.to_csv('submission.csv', index=False)


# ### Suggestions are welcome...I want to improve this kernel more...Upvote if you like it!!!
# ### Thanks for reading🙏🙏🙏
