#!/usr/bin/env python
# coding: utf-8

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Introduction</h1>
# 
# 

# In the following notebook, we dive into some more refined analysis on text-derived features and create two baseline models.
# A small experiment will be performed to uncover some of the differences between the hardest texts and the easiest ones.
# About the experiment:
# We will divide the texts into the given data set into two groups which we will call "Hard" and "Easy," the "Easy" group will contain all the texts whose reading ease score is below two standard deviations, and the "Hard" group will include those who are above two standard deviations.
# Our goal will be to analyze and determine which features differ the most between the two groups, an attribute that increases the importance of such a feature when predicting a reading ease score.
# 
# In the process of creating and evaluating the models, we will engage the problem from 2 perspectives; the first will be a correlation-based approach where we will use the insight gained through the data analysis to construct simple models and test their performance.
# The second approach will be a deep neural network based on the embeddings of the text in our dataset.
# The neural network will be a simple linear descent dense head connected to an embeddings layer.
# 

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Libraries and Data Loading</h1>
# 
# 

# In[1]:


import scipy
import re
import string
import nltk
import random
import os
import pymc3                                                as pm
import tensorflow                                           as tf
import numpy                                                as np 
import pandas                                               as pd 
import matplotlib.pyplot                                    as plt
import matplotlib.cm                                        as cm
import seaborn                                              as sns
import plotly.express                                       as ex
import plotly.graph_objs                                    as go
import plotly.offline                                       as pyo
import spacy                                                as sp
from plotly.subplots                                        import make_subplots
from sklearn.decomposition                                  import TruncatedSVD,PCA
from sklearn.cluster                                        import DBSCAN
from sklearn.manifold                                       import Isomap
from sklearn.feature_extraction.text                        import CountVectorizer,TfidfVectorizer
from sklearn.cluster                                        import KMeans
from nltk.sentiment.vader                                   import SentimentIntensityAnalyzer as SIA
from wordcloud                                              import WordCloud,STOPWORDS
from nltk.util                                              import ngrams
from nltk                                                   import word_tokenize
from nltk.stem                                              import PorterStemmer
from nltk.stem                                              import WordNetLemmatizer
from tqdm.notebook                                          import tqdm
from keras                                                  import backend as K
import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()
sns.set_style('darkgrid')
#nltk.download('vader_lexicon')
pyo.init_notebook_mode()
nlps = sp.load('en')
plt.rc('figure',figsize=(16,8))
sns.set_context('paper',font_scale=1.5)
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()


# In[2]:


train = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
test  = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')

c_df = pd.concat([train,test])[['excerpt','target']]
c_df.head(3)


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Data Preprocessing and Feature Engineering</h1>
# 
# 

# In[3]:


#Preprocessing and Feature Engineering 



#===========================Feature Engineering =========================================================================================

#Naive Features
c_df['# Of Words']              = c_df['excerpt'].apply(lambda x: len(x.split(' ')))
c_df['# Of StopWords']          = c_df['excerpt'].apply(lambda x: len([word for word in x.split(' ') if word in list(STOPWORDS)]))
c_df['# Of Sentences']          = c_df['excerpt'].apply(lambda x: len(re.findall('\.',x)))
c_df['Average Word Length']     = c_df['excerpt'].apply(lambda x: np.mean(np.array([len(va) for va in x.split(' ') if va not in list(STOPWORDS)])))
c_df['Average Sentence Length'] = c_df['excerpt'].apply(lambda x: np.mean(np.array([len(va) for va in x.split('.')])))


#Recored Entities
ENTS_PER_TEXT = []

for E in tqdm(c_df['excerpt']):
    
    TAGGED_ENTS = { 'GPE'    : [],
                    'MONEY'  : [],
                    'PERSON' : [],
                    'EVENT'  : [],
                    'FAC'    : []
              }
    for tok in nlps(E).ents:
        if tok.label_ in ['GPE','MONEY','PERSON','EVENT','FAC']:
            TAGGED_ENTS[tok.label_].append(tok)
    ENTS_PER_TEXT.append(TAGGED_ENTS)   

#Named Entity Extraction ("CAN BE SKIPPED BY USING DATA COLLECTED IN THE PERIVOUS FOR LOOP")
c_df['# Of Different Countries Mentioned']    = c_df['excerpt'].progress_apply(lambda x: len([tok for tok in nlps(x).ents if tok.label_ == 'GPE' ]))
c_df['# Of Times Money Was Mentioned']        = c_df['excerpt'].progress_apply(lambda x: len([tok for tok in nlps(x).ents if tok.label_ == 'MONEY' ]))
c_df['# Of Different People Mentioned']       = c_df['excerpt'].progress_apply(lambda x: len([tok for tok in nlps(x).ents if tok.label_ == 'PERSON' ]))
c_df['# Of Different Events Mentioned']       = c_df['excerpt'].progress_apply(lambda x: len([tok for tok in nlps(x).ents if tok.label_ == 'EVENT' ]))
c_df['# Of Different Facilities Mentioned']   = c_df['excerpt'].progress_apply(lambda x: len([tok for tok in nlps(x).ents if tok.label_ == 'FAC' ]))


        


#Sentiment Analysis
sid = SIA()
c_df['sentiments']           = c_df['excerpt'].progress_apply(lambda x: sid.polarity_scores(x))
c_df['Positive Sentiment']   = c_df['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
c_df['Neutral Sentiment']    = c_df['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
c_df['Negative Sentiment']   = c_df['sentiments'].apply(lambda x: x['neg']+1*(10**-6))

c_df.drop(columns=['sentiments'],inplace=True)

#===========================Data Preprocessing=========================================================================================

# Remove all the special characters
c_df.excerpt              = c_df.excerpt.apply(lambda x: ''.join(re.sub(r'\W', ' ', x))) 
# Substituting multiple spaces with single space 
c_df.excerpt              = c_df.excerpt.apply(lambda x: ''.join(re.sub(r'\s+', ' ', x, flags=re.I)))
# Converting to Lowercase 
c_df.excerpt              = c_df.excerpt.str.lower() 


# In[4]:


c_df.head(5)


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Exploratory Data Analysis</h1>
# 
# 

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Global Analysis</h1>
# 
# 

# In[5]:


plt.subplot(211)
plt.title('Distribution of Reading Ease Score in Our Data')
ax = sns.kdeplot(c_df.target,lw=2)
plt.vlines(c_df.target.mean(),0,0.36,color='red',label='Mean',lw=3,ls='--')
plt.vlines(c_df.target.median(),0,0.36,color='tab:green',label='Median',lw=3,ls='-.')
plt.vlines(scipy.stats.mode(c_df.target)[0],0,0.36,color='tab:Purple',label='Mode',lw=3,ls='-')

plt.xlabel('')
plt.legend()
plt.subplot(212)
sns.kdeplot(c_df.target,cumulative=True,lw=3,color='tab:orange')
plt.xlabel('Reading Ease')
plt.tight_layout()


# In[6]:


mask = np.zeros_like(c_df.corr())
mask[np.abs(c_df.corr()) < 0.1] = 1
sns.heatmap(c_df.corr(),cmap='coolwarm',annot=True,mask=mask,linewidth=2)


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Text Based Analysis</h1>
# 
# 

# In[7]:


ALL_GPS    = []
ALL_MONEY  = []
ALL_PERSON = []
ALL_EVENT  = []
ALL_FAC    = []
for dic in ENTS_PER_TEXT:
    for key in ['GPE','MONEY','PERSON','EVENT','FAC']:
        if len(dic['GPE']) != 0:
            ALL_GPS+=list(dic['GPE'])
        if len(dic['MONEY']) != 0:
            ALL_MONEY+=(dic['MONEY'])
        if len(dic['PERSON']) != 0:
            ALL_PERSON+=(dic['PERSON'])
        if len(dic['EVENT']) != 0:
            ALL_EVENT+=(dic['EVENT'])
        if len(dic['FAC']) != 0:
            ALL_FAC+=(dic['FAC'])
top_10_places = pd.Series([i.text for i in ALL_GPS]).value_counts()[:10]
top_10_money = pd.Series([i.text for i in ALL_MONEY]).value_counts()[:10]
top_10_person = pd.Series([i.text for i in ALL_PERSON]).value_counts()[:10]
top_10_event = pd.Series([i.text for i in ALL_EVENT]).value_counts()[:10]
top_10_fac = pd.Series([i.text for i in ALL_FAC]).value_counts()[:10]


# In[8]:


plt.subplot(211)
plt.title('Top 10 Most Frequently Mentioned Locations')
sns.barplot(x=top_10_places.index,y=top_10_places.values,palette=cm.twilight(top_10_places.values))
plt.subplot(212)
plt.title('Top 10 Most Frequently Currency Related Words/Phrases')
sns.barplot(x=top_10_money.index,y=top_10_money.values,palette=cm.twilight(top_10_money.values*10))


# In[9]:


plt.subplot(311)
plt.title('Top 10 Most Frequently Mentioned Pepole')
sns.barplot(x=top_10_person.index,y=top_10_person.values,palette=cm.twilight(top_10_person.values))
plt.subplot(312)
plt.title('Top 10 Most Frequently Mentioned Events')
ax = sns.barplot(x=top_10_event.index,y=top_10_event.values,palette=cm.twilight(top_10_event.values*10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.subplot(313)
plt.title('Top 10 Most Frequently Mentioned Facilites')
sns.barplot(x=top_10_fac.index,y=top_10_fac.values,palette=cm.twilight(top_10_fac.values*10))

plt.tight_layout()


# In[10]:


NUMBER_OF_COMPONENTS = 1200

CV = CountVectorizer(stop_words='english',ngram_range=(1,1))
cv_df = CV.fit_transform(c_df.excerpt)
cv_df = pd.DataFrame(cv_df.toarray(),columns=CV.vocabulary_)

svd = TruncatedSVD(NUMBER_OF_COMPONENTS)
decomposed = svd.fit_transform(cv_df)

evr = svd.explained_variance_ratio_
total_var = evr.sum() * 100
cumsum_evr = np.cumsum(evr)

trace1 = {
    "name": "individual explained variance", 
    "type": "bar", 
    'y':evr}
trace2 = {
    "name": "cumulative explained variance", 
    "type": "scatter", 
     'y':cumsum_evr}
data = [trace1, trace2]
layout = {
    "xaxis": {"title": "Principal components"}, 
    "yaxis": {"title": "Explained variance ratio"},
  }
fig = go.Figure(data=data, layout=layout)
fig.update_layout(     title='{:.2f}% of the Song Lyrics Variance Can Be Explained Using {} Words out of {} Unique words'.format(np.sum(evr)*100,NUMBER_OF_COMPONENTS,len(CV.vocabulary_)))
fig.show()


# In[11]:


tf_idf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))

trans_df = tf_idf.fit_transform(c_df.excerpt[:-7])

isomap = Isomap(n_components=3)

isomap_dec = isomap.fit_transform(trans_df)

temp  =  pd.DataFrame(isomap_dec)
temp = temp.rename(columns={0:'Dim1',1:'Dim2',2:'Dim3'})
temp['target'] = c_df.target[:-7].values

ex.scatter_3d(temp,x='Dim1',y='Dim2',z='Dim3',color='target',title='Spread of Tfidf Vectorized Samples in R^3')


# In[12]:


db = DBSCAN(eps=1.2,min_samples=80)
db.fit(temp)
temp['cluster'] =db.labels_

ex.scatter_3d(temp,x='Dim1',y='Dim2',z='Dim3',color='cluster',title='Tfidf Vectorized Samples in R^3 Clusterd By Density')


# In[13]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
dec_df= temp.copy()
r = lambda: np.random.randint(0,255)
def random_color_hex():
    return ('#%02X%02X%02X' % (r(),r(),r()))
countries = pd.Series(db.labels_).unique()
colors = [random_color_hex() for _ in range(0,len(countries))]
gs = grid_spec.GridSpec(len(countries),1)
fig = plt.figure(figsize=(16,9))

i = 0

ax_objs = []
for cluster in countries:
    x = np.array(dec_df[dec_df.cluster == cluster]['target'])
    x_d = np.linspace(-4,4, 1000)

    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # creating new axes object
    ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

    # plotting the distribution
    ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
    ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])

    #ax_objs[-1].set_xlim(0,1.)
    #ax_objs[-1].set_ylim(0,2.5)

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])
    ax_objs[-1].grid(False)
    if i == len(countries)-1:
        ax_objs[-1].set_xlabel("target", fontsize=16,fontweight="bold")
    else:
        ax_objs[-1].set_xticklabels([])

    spines = ["top","right","left","bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    #adj_country = country.replace(" ","\n")
    ax_objs[-1].text(-4,0,f'Cluster: {cluster}',fontweight="bold",fontsize=14,ha="right")


    i += 1

gs.update(hspace=-0.6)

fig.text(0.07,0.89,"Distribution Reading Ease Score in Each Cluster",fontsize=20)

plt.grid(False)
plt.tight_layout()
plt.show()


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Hardest/Easiest Expirement</h1>
# 
# 

# In[14]:


plt.title('Distribution of Reading Ease Score in Our Data')
ax = sns.kdeplot(c_df.target,lw=2)
kde_x, kde_y = ax.lines[0].get_data()
p1 = plt.axvline(x=c_df.target.mean()+2*c_df.target.std(),color='tab:red')
p2 = plt.axvline(x=c_df.target.mean()-2*c_df.target.std(),color='tab:green')


ax.fill_between(kde_x, kde_y, where=((kde_x<=c_df.target.mean()-2*c_df.target.std()) ) , interpolate=True, color='tab:green',label='Easiest to Read')
ax.fill_between(kde_x, kde_y, where=((kde_x>=c_df.target.mean()+2*c_df.target.std())) , interpolate=True, color='tab:red',label='Hardest to Read')

plt.xlabel('')
plt.legend()


# In[15]:


e_df = c_df[c_df.target <= c_df.target.mean()-2*c_df.target.std() ].copy()
h_df = c_df[c_df.target >= c_df.target.mean()+2*c_df.target.std() ].copy()
a_df = c_df[(c_df.target > c_df.target.mean()-2*c_df.target.std()) & (c_df.target < c_df.target.mean()+2*c_df.target.std())].copy()

_e_ENTS = pd.DataFrame(ENTS_PER_TEXT,index=c_df.index).loc[e_df.index,:]
_h_ENTS = pd.DataFrame(ENTS_PER_TEXT,index=c_df.index).loc[h_df.index,:]

for col in e_df.columns[1:]:
    e_df[col] = (e_df[col]-a_df[col].mean())/a_df[col].std()
    h_df[col] = (h_df[col]-a_df[col].mean())/a_df[col].std()
    
E_WC,H_WC = None,None
E_WC = WordCloud(background_color='white',width=800,height=400,stopwords=STOPWORDS,collocations=False).generate(' '.join(e_df.excerpt))
H_WC = WordCloud(background_color='white',width=800,height=400,stopwords=STOPWORDS,collocations=False).generate(' '.join(h_df.excerpt))

plt.subplot(121)
plt.title('Most Common Words in Easiest Samples')
plt.imshow(E_WC)
plt.axis('off')
plt.subplot(122)
plt.title('Most Common Words in Hardest Samples')
plt.imshow(H_WC)
plt.axis('off')

plt.tight_layout()


# In[16]:


medf = e_df.iloc[:,1:].melt()
mhdf = h_df.iloc[:,1:].melt()
medf['Set'] = 'Easy'
mhdf['Set'] = 'Hard'
ax = sns.boxplot(x='variable',y='value',hue='Set',data=pd.concat([medf,mhdf]),showfliers = False,notch=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()


# Observation: Looking at the boxplot above, we can see that in terms of z-score, we have some significant differences between the two groups.
# Features like "Number of Sentences," "Average Sentence Length," and Positive Sentiment differ by at least one standard deviation from each other; such an analysis saves us the time of performing t-tests or ANOVA as we can see a significant difference between the means.

# In[17]:


EASY_GPS    = []
EASY_MONEY  = []
EASY_PERSON = []
EASY_EVENT  = []
EASY_FAC    = []

HARD_GPS    = []
HARD_MONEY  = []
HARD_PERSON = []
HARD_EVENT  = []
HARD_FAC    = []
for index,row in _e_ENTS.iterrows():
    dic = {'GPE':row['GPE'],'MONEY':row['MONEY'],'PERSON':row['PERSON'],'EVENT':row['EVENT'],'FAC':row['FAC']}
    for key in ['GPE','MONEY','PERSON','EVENT','FAC']:
        if len(dic['GPE']) != 0:
            EASY_GPS+=list(dic['GPE'])
        if len(dic['MONEY']) != 0:
            EASY_MONEY+=(dic['MONEY'])
        if len(dic['PERSON']) != 0:
            EASY_PERSON+=(dic['PERSON'])
        if len(dic['EVENT']) != 0:
            EASY_EVENT+=(dic['EVENT'])
        if len(dic['FAC']) != 0:
            EASY_FAC+=(dic['FAC'])
            
for index,row in _h_ENTS.iterrows():
    dic = {'GPE':row['GPE'],'MONEY':row['MONEY'],'PERSON':row['PERSON'],'EVENT':row['EVENT'],'FAC':row['FAC']}
    for key in ['GPE','MONEY','PERSON','EVENT','FAC']:
        if len(dic['GPE']) != 0:
            HARD_GPS+=list(dic['GPE'])
        if len(dic['MONEY']) != 0:
            HARD_MONEY+=(dic['MONEY'])
        if len(dic['PERSON']) != 0:
            HARD_PERSON+=(dic['PERSON'])
        if len(dic['EVENT']) != 0:
            HARD_EVENT+=(dic['EVENT'])
        if len(dic['FAC']) != 0:
            HARD_FAC+=(dic['FAC'])
            
EASY_top_10_places  = pd.Series([i.text for i in EASY_GPS]).value_counts()[:10]
EASY_top_10_money   = pd.Series([i.text for i in EASY_MONEY]).value_counts()[:10]
EASY_top_10_person  = pd.Series([i.text for i in EASY_PERSON]).value_counts()[:10]
EASY_top_10_event   = pd.Series([i.text for i in EASY_EVENT]).value_counts()[:10]
EASY_top_10_fac     = pd.Series([i.text for i in EASY_FAC]).value_counts()[:10]

HARD_top_10_places  = pd.Series([i.text for i in HARD_GPS]).value_counts()[:10]
HARD_top_10_money   = pd.Series([i.text for i in HARD_MONEY]).value_counts()[:10]
HARD_top_10_person  = pd.Series([i.text for i in HARD_PERSON]).value_counts()[:10]
HARD_top_10_event   = pd.Series([i.text for i in HARD_EVENT]).value_counts()[:10]
HARD_top_10_fac     = pd.Series([i.text for i in HARD_FAC]).value_counts()[:10]


# In[18]:


plt.subplot(211)
plt.title('Top 10 Most Frequently Mentioned Locations in "Easiest" Text')
sns.barplot(x=EASY_top_10_places.index,y=EASY_top_10_places.values,palette=cm.twilight(EASY_top_10_places.values*10))
plt.subplot(212)
plt.title('Top 10 Most Frequently Currency Related Words/Phrases in "Easiest" Text')
ax = sns.barplot(x=EASY_top_10_money.index,y=EASY_top_10_money.values,palette=cm.twilight(EASY_top_10_money.values*10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.tight_layout()


# In[19]:


plt.subplot(311)
plt.title('Top 10 Most Frequently Mentioned Pepole in "Easiest" Text')
sns.barplot(x=EASY_top_10_person.index,y=EASY_top_10_person.values,palette=cm.twilight(EASY_top_10_person.values*20))
plt.subplot(312)
plt.title('Top 10 Most Frequently Mentioned Events in "Easiest" Text')
ax = sns.barplot(x=EASY_top_10_event.index,y=EASY_top_10_event.values,palette=cm.twilight(EASY_top_10_event.values*20))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.subplot(313)
plt.title('Top 10 Most Frequently Mentioned Facilites in "Easiest" Text')
sns.barplot(x=EASY_top_10_fac.index,y=EASY_top_10_fac.values,palette=cm.twilight(EASY_top_10_fac.values*10))

plt.tight_layout()


# In[20]:


plt.subplot(211)
plt.title('Top 10 Most Frequently Mentioned Locations in "Hardest" Text')
sns.barplot(x=HARD_top_10_places.index,y=HARD_top_10_places.values,palette=cm.twilight(HARD_top_10_places.values*30))
plt.tight_layout()
plt.subplot(212)
plt.title('Top 10 Most Frequently Mentioned Pepole in "Hardest" Text')
sns.barplot(x=HARD_top_10_person.index,y=HARD_top_10_person.values,palette=cm.twilight(HARD_top_10_person.values*20))
plt.tight_layout()


# In[21]:


fixed_df = c_df[['Average Word Length','target']].copy()
fixed_df['Average Word Length'] = np.round(fixed_df['Average Word Length'],1)
fixed_df = fixed_df.sort_values(by='Average Word Length')
ns_df = fixed_df.groupby(by='Average Word Length').mean().reset_index()

plt.title('Average Score For Every Average Word Length Bin Scaled To 1 Decimel Point')
ax= sns.barplot(x=ns_df['Average Word Length'],y=ns_df['target'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=-45)
plt.grid()
plt.show()


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Probabilistic Inference</h1>
# 
# 

# In[22]:


n_df = c_df.iloc[:-8,:].copy()
#n_df['Average Word Length'] = (n_df['Average Word Length']-n_df['Average Word Length'].min())/(n_df['Average Word Length'].max()-n_df['Average Word Length'].min())+0.0001
#n_df['target'] = (n_df['target']-n_df['target'].min())/(n_df['target'].max()-n_df['target'].min())+0.0001

fig = ex.scatter(n_df,x='Average Word Length',y='target',trendline='ols')
fig.update_layout(title='<b>The linear relationship between average word length in a given text and the reading ease score<b>')
fig.show()


# In general, the classic frequentists way to think about Linear Regression as follows:
# $$Y=Xβ+ϵ$$
# where $Y$ is the output we want to predict (or dependent variable), X is our predictor (or independent variable), and $β$ are the coefficients of the model we want to estimate. $ϵ$ is an error term which is assumed to be normally distributed.
# 
# We can then use Ordinary Least Squares or Maximum Likelihood to find the best fitting $β$.
# 
# We will preform the Bayesian equivalent which takes a probabilistic view of the problem and express this model in terms of probability distributions:
# 
# $$Y∼N(Xβ,σ^{2})$$
# In our case $Y$ is a random variable of which each entry is distributed according to a Normal distribution. The mean of this normal distribution is provided by our linear predictor with variance $σ^{2}$.
# 

# In[23]:


with pm.Model() as breg: 
    sigma = pm.Uniform("sigma", 0,10)
    intercept = pm.Normal("Intercept", 0, sigma=20)
    x_coeff = pm.Normal("x", 0, sigma=20)

    likelihood = pm.Normal("y", mu=(intercept + x_coeff * n_df.iloc[:,5]), sigma=sigma, observed=n_df.iloc[:,1])
    
    step = pm.Metropolis()
    trace = pm.sample(5000, cores=2,step=step)


# In our Bayesian model, we have four random variables that origin in different distributions, those four random variables construct our 4D posterior landscape from which we will sample using MCMC.
#  

# In[24]:


plt.figure(figsize=(15, 10))
pm.plot_trace(trace[3000:])
plt.tight_layout();


# In[25]:


plt.figure(figsize=(17, 10))
plt.plot(n_df.iloc[:,5], n_df.iloc[:,1], "x", label="data")
pm.plot_posterior_predictive_glm(trace[3000:], samples=3000, label="posterior predictive regression lines",eval=n_df.iloc[:,5])

plt.title("Posterior predictive regression lines")
plt.legend(loc=0)
plt.xlabel("Average Word Length")
plt.ylabel("Reading Ease");


# **Observation**: After constructing a bayesian model to sample from the posterior space of all possible linear regression intercepts and coefficients, we learn that when using the intercept and coefficient of the average word length to predict the reading ease score, we can be the most confident in our estimate when the average word length is around six letters wide.
# As we go further away from six in both directions, we lose confidence in our estimate as the variance grows larger and the average error in our estimate increases.

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Baseline Model Testing</h1>
# 
# 

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Correlation Oriented</h1>
# 
# 

# In[26]:


from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score

LR_MODEL = Pipeline(steps=[('Scale',StandardScaler()),('model',LinearRegression())])
DT_MODEL = Pipeline(steps=[('Scale',StandardScaler()),('model',DecisionTreeRegressor())])
RF_MODEL = Pipeline(steps=[('Scale',StandardScaler()),('model',RandomForestRegressor())])

x_train,x_test,y_train,y_test = train_test_split(c_df.iloc[:-8,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']],c_df.iloc[:-8,1])


# In[27]:


LR_RESULTS = -1*cross_val_score(LR_MODEL,c_df.iloc[:-8,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']],
                   c_df.iloc[:-8,:][['target']],cv=10,scoring='neg_root_mean_squared_error')
DT_RESULTS = -1*cross_val_score(DT_MODEL,c_df.iloc[:-8,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']],
                   c_df.iloc[:-8,:][['target']],cv=10,scoring='neg_root_mean_squared_error')
RF_RESULTS = -1*cross_val_score(RF_MODEL,c_df.iloc[:-8,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']],
                   c_df.iloc[:-8,:]['target'].values,cv=10,scoring='neg_root_mean_squared_error')


# In[28]:


fig = go.Figure()
lr_trace =  go.Scatter(x=list(range(0,len(LR_RESULTS))),y=LR_RESULTS,name='Linear Regression')
dt_trace =  go.Scatter(x=list(range(0,len(DT_RESULTS))),y=DT_RESULTS,name='Decision Tree')
rf_trace =  go.Scatter(x=list(range(0,len(RF_RESULTS))),y=RF_RESULTS,name='Random Forest')

fig.add_trace(lr_trace)
fig.add_trace(dt_trace)
fig.add_trace(rf_trace)

fig.update_layout(title='Different Baseline Model 10 Fold Cross Validation')
fig.update_yaxes(title_text="RMSE")
fig.update_xaxes(title_text="Fold #")
fig.show()


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Text Oriented</h1>
# 
# 

# In[29]:


from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
validation = c_df.iloc[:c_df.shape[0]-8,0].sample(20,random_state=42)
tarin_df = c_df.iloc[:c_df.shape[0]-8,0].drop(index=validation.index)

vocab = [i for i in list(nltk.FreqDist(' '.join(c_df.excerpt).split(' ')).keys()) ]
vocab_size = len(vocab)
encoded_docs = [one_hot(d, vocab_size) for d in tarin_df]
max_length = max(tarin_df.str.len())
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


encoded_valid= [one_hot(d, vocab_size) for d in validation]
padded_docs_valid = pad_sequences(encoded_valid, maxlen=max_length, padding='post')


encoded_test= [one_hot(d, vocab_size) for d in test.excerpt]
padded_docs_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')


# In[30]:


model = Sequential()
model.add(Embedding(vocab_size, 950, input_length=max_length))
model.add(Flatten())
model.add(Dense(250, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(25, activation='linear'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='linear'))


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['mse'])

model.summary()


# In[31]:


history = model.fit(padded_docs,
          c_df.iloc[tarin_df.index,1],
          epochs=5,
          batch_size=25,
          validation_data=(padded_docs_valid,c_df.iloc[validation.index,1]),
          verbose=1)


# In[32]:


plt.subplot(211)
plt.title('Model loss over epochs')
pd.DataFrame(history.history).loss.plot()
plt.subplot(212)
plt.title('Model MSE over epochs')
pd.DataFrame(history.history).mse.plot(color='red')


# In[33]:


prediction = model.predict(padded_docs_test,verbose=0)
prediction = prediction.flatten()
LR_MODEL.fit(x_train,y_train)
RF_MODEL.fit(x_train,y_train)
final_prediction = prediction*0.8+RF_MODEL.predict(c_df.iloc[-7:,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']])*0.1
final_prediction += LR_MODEL.predict(c_df.iloc[-7:,:][['Average Word Length','# Of Sentences','# Of Different People Mentioned']])*0.1
submit_df = pd.DataFrame({'id':test.id,'target':final_prediction})
submit_df


# In[34]:


submit_df.to_csv("submission.csv", index = False)


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Conclusions</h1>
# 
# 

# <p style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;"><span data-preserver-spaces="true" style='color: rgb(14, 16, 26); background: transparent; margin-top: 0pt; margin-bottom: 0pt; font-size: 22px; font-family: "Times New Roman", Times, serif;'>During the work on the following notebook, we derived several key points worth taking into account in feature works.</span></p>
# <p style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;"><span style="font-family: 'Times New Roman', Times, serif;"><span style="font-size: 22px;"><br></span></span></p>
# <ul style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;">
#     <li style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt; list-style-type:disc;"><span style="font-family: 'Times New Roman', Times, serif;"><span style="font-size: 22px;"><span data-preserver-spaces="true" style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;">Basic text structure features such as &quot;Average Word Length&quot; significantly affect the overall reading ease compared to more fine detail text attributes such as the sentiments or the different elements of the text&apos;s theme.</span></span></span></li>
#     <li style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt; list-style-type:disc;"><span style="font-family: 'Times New Roman', Times, serif;"><span style="font-size: 22px;"><span data-preserver-spaces="true" style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;">When looking at the most frequent entities from a list of types, we see that many labels reappear and that the distribution of those labels is far from uniform (&apos;a penny&apos;, for example, is the most frequent currency mentioned in our texts).</span></span></span></li>
#     <li style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt; list-style-type:disc;"><span style="font-family: 'Times New Roman', Times, serif;"><span style="font-size: 22px;"><span data-preserver-spaces="true" style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;">From the conducted experiment, I was surprised to see that the &quot;Hardest&quot; texts &quot;Event.&quot; labeled entities are not existing and that the &quot;French Revolution&quot; has been the most frequent event in the &quot;Easiest&quot; texts. One thing to note for future work is that there is a topic and theme difference between the two groups, as shown by the different distribution of top 10 entities in all labels.</span></span></span></li>
#     <li style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt; list-style-type:disc;"><span style="font-family: 'Times New Roman', Times, serif;"><span style="font-size: 22px;"><span data-preserver-spaces="true" style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt;">When comparing different &quot;naive&quot; models, surprisingly, the Linear Regression model has scored the lowest total RMSE when fitting on different folds of the data.</span></span></span></li>
#     <li style="color: rgb(14, 16, 26); background: transparent; margin-top:0pt; margin-bottom:0pt; list-style-type:disc;"><span data-preserver-spaces="true" style='color: rgb(14, 16, 26); background: transparent; margin-top: 0pt; margin-bottom: 0pt; font-size: 22px; font-family: "Times New Roman", Times, serif;'>The most simple NN based on word embeddings has converged in about four iterations and has overscored the &quot;naive&quot; models.</span></li>
# </ul>

# In[ ]:




