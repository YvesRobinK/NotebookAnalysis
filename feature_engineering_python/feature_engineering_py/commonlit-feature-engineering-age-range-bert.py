#!/usr/bin/env python
# coding: utf-8

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Introduction</h1>
# 
# 

# Greetings fellow Kagglers,
# Many different approaches are taken in the pursuit of the perfect model to solve the CommonLit reading ease problem.
# The following notebook suggests yet another approach that can potentially increase the performance of already stable models,
# The following notebook will use data collected from a book recommendation website that targets children and teens and provides the target reading age for each book in its database.
# We will use the books' synopsis from that website to train a dense neural network that will predict the minimum recommended reading age and the maximum recommended reading age.
# The hypothesis is that knowing the reading age bound of a text is a strong indicator of the hardness of a given text. 
# 
# 
# 

# **Our Hypothesis and Intuition for Achieving Such A Feature**
# 
# ![](https://i.ibb.co/4mFwSng/h16.png)

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Libraries And Utilities</h1>
# 
# 

# In[1]:


import scipy
import re
import string
import nltk
import random
import os
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
from sklearn.manifold                                       import Isomap
from sklearn.feature_extraction.text                        import CountVectorizer,TfidfVectorizer
from sklearn.cluster                                        import KMeans
from tqdm.notebook                                          import tqdm
from keras                                                  import backend as K
from keras                                                  import Sequential
from keras.layers                                           import Dense,LSTM,Input,Dropout,SimpleRNN
from sklearn.model_selection                                import train_test_split
from scipy.spatial                                          import distance_matrix
from sklearn.metrics.pairwise                               import cosine_similarity
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
tqdm.pandas()
sns.set_style('darkgrid')
pyo.init_notebook_mode()
plt.rc('figure',figsize=(16,8))
sns.set_context('paper',font_scale=1.5)
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Data Loading and Preprocessing</h1>
# 
# 

# In[2]:


train = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
test  = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
clit_df = pd.concat([train,test])[['excerpt']]
cs_df = pd.read_csv('/kaggle/input/highly-rated-children-books-and-stories/children_books.csv',encoding='ISO-8859-1')


# In[3]:


# Remove all the special characters
clit_df.excerpt              = clit_df.excerpt.apply(lambda x: ''.join(re.sub(r'\W', ' ', x)))
cs_df.Desc                   = cs_df.Desc.apply(lambda x: ''.join(re.sub(r'\W', ' ', x))) 
# Substituting multiple spaces with single space 
clit_df.excerpt              = clit_df.excerpt.apply(lambda x: ''.join(re.sub(r'\s+', ' ', x, flags=re.I)))
cs_df.Desc                   = cs_df.Desc.apply(lambda x: ''.join(re.sub(r'\s+', ' ', x, flags=re.I)))
# Converting to Lowercase 
clit_df.excerpt              = clit_df.excerpt.str.lower() 
cs_df.Desc                   = cs_df.Desc.str.lower() 


# **Example of The Data we will train our models on as stated in the dataset mentioned below**
# 
# ![](https://i.ibb.co/SyLjSvc/h15.png)

# In[4]:


cs_df.head(5)


# **Note:** Above is the data collected from the book recommendation website as seen in the following [dataset]( https://www.kaggle.com/thomaskonstantin/highly-rated-children-books-and-stories)

# In[5]:


def min_age(sir):
    if sir.find('+')!=-1:
        return np.int(sir[:sir.find('+')])
    elif sir.find('-')!=-1:
        return np.int(sir.split('-')[0])
    elif bool(re.match(r'[0-9]+',sir)):
        return np.int(sir)
    else:
        return 'else'
def max_age(sir):
    if sir.find('+')!=-1:
        return 99
    elif sir.find('-')!=-1:
        return np.int(sir.split('-')[1])
    elif bool(re.match(r'[0-9]+',sir)):
        return np.int(sir)
    else:
        return 'else'
    
cs_df['Min_Age'] = cs_df.Reading_age.apply(min_age)
cs_df['Max_Age'] = cs_df.Reading_age.apply(max_age)


# **Note:** After extracting bounds from the recommended reading age, we can advance and create models to predict each side of the interval.

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Similarity Analysis</h1>
# 
# 

# The following short section will test the similarity between the texts in both datasets, the first test we will use is calculating the cosine similarity between the two corpora (the children's book synopsis vs. the CommonLit texts).
# We will also cluster the resulting values to see if any visible cluster emerges; such a result will increase our confidence in the usability of the text from the children's book dataset and may reveal any underlying topics that may reside in our dataset.
# 
# The second test we will perform is using an $R^2$ representation of the Tfidf vectorized texts from both datasets, plotting them and visually inspecting how they are located in $R^2$ space; a visual confirmation of such proximity can confirm a certain degree of similarity between both corpora.

# In[6]:


all_text = pd.concat([cs_df.Desc,clit_df.excerpt]).values
VCT = TfidfVectorizer(stop_words='english')

VCT.fit(all_text)
ISOMAP = Isomap(n_components =2)
ISOMAP.fit(VCT.transform(clit_df.excerpt))
dict_size = len(VCT.vocabulary_)
cs_tf = ISOMAP.transform(VCT.transform(cs_df.Desc))
clit_tf = ISOMAP.transform(VCT.transform(clit_df.excerpt))
CS = cosine_similarity(clit_tf,cs_tf)


# In[7]:


sns.clustermap(CS)
plt.show()


# In[8]:


fig = go.Figure()
scatter_cs = go.Scatter(x=cs_tf[:,0],y=cs_tf[:,1],mode='markers',name='Children Books')
scatter_clit = go.Scatter(x=clit_tf[:,0],y=clit_tf[:,1],mode='markers',name='CommoinLit')
fig.add_trace(scatter_clit)
fig.add_trace(scatter_cs)
fig.show()


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Model Fitting and Evaluation</h1>
# 
# 

# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Baseline Architecture</h1>
# 
# 

# In[9]:


all_text = pd.concat([cs_df.Desc,clit_df.excerpt]).values
VCT = TfidfVectorizer(stop_words='english')

VCT.fit(all_text)

dict_size = len(VCT.vocabulary_)

cs_tf = VCT.transform(cs_df.Desc)
clit_tf = VCT.transform(clit_df.excerpt)

train_x_min,test_x_min,train_y_min,test_y_min = train_test_split(cs_tf.todense(),cs_df.Min_Age.astype(np.float32))


# In[10]:


min_age_model = Sequential()
min_age_model.add(Input(train_x_min.shape))
min_age_model.add(Dense(500,activation='linear'))
min_age_model.add(Dense(100,activation='linear'))
min_age_model.add(Dense(50,activation='linear'))
min_age_model.add(Dense(20,activation='linear'))
min_age_model.add(Dense(1,activation='linear'))

min_age_model.compile(optimizer='adam',loss=root_mean_squared_error)
min_age_model.summary()


# In[11]:


history = min_age_model.fit(train_x_min,train_y_min,epochs=200,batch_size=128,verbose=0)
plt.plot(history.history['loss'])
plt.title('min age model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[12]:


pred=  pd.DataFrame({'Pred':np.round(min_age_model.predict(test_x_min).squeeze(),0),'True':test_y_min})
pred.head(10)


# In[13]:


train_x_max,test_x_max,train_y_max,test_y_max = train_test_split(cs_tf.todense(),cs_df.Max_Age.astype(np.float32))


# In[14]:


max_age_model = Sequential()
max_age_model.add(Input(train_x_max.shape))
max_age_model.add(Dense(500,activation='linear'))
max_age_model.add(Dense(100,activation='linear'))
max_age_model.add(Dense(50,activation='linear'))
max_age_model.add(Dense(20,activation='linear'))
max_age_model.add(Dense(1,activation='linear'))

max_age_model.compile(optimizer='adam',loss=root_mean_squared_error)
max_age_model.summary()


# In[15]:


history = max_age_model.fit(train_x_max,train_y_max,epochs=200,batch_size=128,verbose=0)
plt.plot(history.history['loss'])
plt.title('max age model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[16]:


pred=  pd.DataFrame({'Pred_Max_Age':np.round(max_age_model.predict(test_x_max).squeeze(),0),'True_Max_Age':test_y_max})
pred.head(10)


# In[17]:


hist_min = min_age_model.fit(cs_tf.todense(),cs_df.Min_Age.astype(np.float32),epochs=200,batch_size=128,verbose=0)
hist_max = max_age_model.fit(cs_tf.todense(),cs_df.Max_Age.astype(np.float32),epochs=200,batch_size=128,verbose=0)


# In[18]:


min_ages = np.round(min_age_model.predict(clit_tf.todense()).squeeze(),0)
max_ages = np.round(max_age_model.predict(clit_tf.todense()).squeeze(),0)


# In[19]:


train = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
test  = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
train['Min_Age'] =min_ages[:-7]
train['Max_Age'] =max_ages[:-7]
test['Min_Age'] =min_ages[-7:]
test['Max_Age'] =max_ages[-7:]


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">BERT Architecture</h1>
# 
# 

# In[20]:


#LIBS
from transformers import TFAutoModel
from transformers import BertTokenizer

#VARS
SEQ_LENGTH = 512
N_SAMPLES  = cs_df.shape[0]
BATCH_SIZE = 8
SPLIT = 0.8
SIZE  = int((N_SAMPLES/BATCH_SIZE)*SPLIT)



#FUNCS
def map_function(input_ids,mask,labels=None):
    if labels != None:
        return {'input_ids':input_ids,'attention_mask':mask},labels
    else:
        return {'input_ids':input_ids,'attention_mask':mask}

#IMPLEMENTATION


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def create_bert_dataset(corpus,SEQ_LENGTH,BATCH_SIZE,labels=None):
    N_SAMPLES = len(corpus)
    XIDS = np.zeros((N_SAMPLES,SEQ_LENGTH))
    XMASK= np.zeros((N_SAMPLES,SEQ_LENGTH))
    for aux,desc in tqdm(enumerate(corpus),leave=False):
        tokens = tokenizer.encode_plus(desc,max_length=SEQ_LENGTH,truncation=True,
                                       padding='max_length',
                                       add_special_tokens=True,
                                       return_tensors='tf')
        XIDS[aux,:]  = tokens['input_ids']
        XMASK[aux,:] = tokens['attention_mask']


    if type(labels) != None:
        dataset = tf.data.Dataset.from_tensor_slices((XIDS,XMASK,labels))
        dataset = dataset.map(map_function)
        dataset = dataset.shuffle(10000).batch(BATCH_SIZE,drop_remainder=True)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((XIDS,XMASK))
        dataset = dataset.map(map_function)
        dataset = dataset.shuffle(10000).batch(BATCH_SIZE,drop_remainder=True)
    return dataset


# In[21]:


def convert_data(corpus):
        tokens = tokenizer.encode_plus(corpus,max_length=SEQ_LENGTH,truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   return_tensors='tf')
        return {'input_ids':tf.cast(tokens['input_ids'],np.float64),'attention_mask':tf.cast(tokens['attention_mask'],np.float64)}


# In[22]:


#Min Age
labels = np.zeros((cs_df.shape[0],100))
labels[np.arange(cs_df.shape[0]),cs_df.Min_Age] = 1
min_dataset = create_bert_dataset(cs_df.Desc,SEQ_LENGTH,BATCH_SIZE,labels)

train_min_ds = min_dataset.take(SIZE)
val_min_ds   = min_dataset.skip(SIZE)

#Max Age
labels = np.zeros((cs_df.shape[0],100))
labels[np.arange(cs_df.shape[0]),cs_df.Max_Age] = 1
max_dataset = create_bert_dataset(cs_df.Desc,SEQ_LENGTH,BATCH_SIZE,labels)

train_max_ds = max_dataset.take(SIZE)
val_max_ds   = max_dataset.skip(SIZE)


# In[23]:


BERT = TFAutoModel.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape=(SEQ_LENGTH,),name='input_ids',dtype='int32')
masks     = tf.keras.layers.Input(shape=(SEQ_LENGTH,),name='attention_mask',dtype='int32')

EMBDS = BERT.bert(input_ids,attention_mask=masks)[1]

x = tf.keras.layers.Dense(1440,activation='relu')(EMBDS)
y = tf.keras.layers.Dense(100,activation='softmax',name='outputs')(x)

bert_min_age_model = tf.keras.Model(inputs=[input_ids,masks],outputs=y)
bert_max_age_model = tf.keras.models.clone_model(bert_min_age_model)
bert_min_age_model.summary()


# In[24]:


optimizer = tf.keras.optimizers.Adam(lr=1e-5,decay=1e-6)
loss      = tf.keras.losses.CategoricalCrossentropy()
accuracy  = tf.keras.metrics.CategoricalAccuracy('accuracy')
bert_min_age_model.compile(optimizer=optimizer,loss=loss,metrics=[accuracy])
bert_max_age_model.compile(optimizer=optimizer,loss=loss,metrics=[accuracy])


# In[25]:


from tqdm.keras import TqdmCallback
history_min = bert_min_age_model.fit(
    train_min_ds,
    validation_data = val_min_ds,
    epochs=3, verbose=0, callbacks=[TqdmCallback(verbose=1)]
)


# In[26]:


history_max = bert_max_age_model.fit(
    train_max_ds,
    validation_data = val_max_ds,
    epochs=3, verbose=0, callbacks=[TqdmCallback(verbose=1)]
)


# In[27]:


plt.title('Min Age Bert Model Loss')
plt.plot(history_min.history['loss'],label='Min Age')
plt.plot(history_max.history['loss'],label='Max Age')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[28]:


#bert_min_age_model.save('Min_Age_BERT.h5')
#bert_max_age_model.save('Max_Age_BERT.h5')


# <h1 style="background-color:orange;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 15px 50px;">Conclusion</h1>
# 
# 

# In[29]:


plt.title('Min Age Connection with Target')
sns.boxplot(y=train['target'],x=train['Min_Age'])
plt.show()


# In[30]:


plt.title('Max Age Connection with Target')
ax = sns.boxplot(y=train['target'],x=train['Max_Age'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()


# In[31]:


plt.title('Pearson Correlation Between Features')
sns.heatmap(train.corr(),annot=True,cmap='mako')


# In[32]:


train.to_csv('train_with_minmax_ages.csv')
test.to_csv('test_with_minmax_ages.csv')
min_age_model.save('Min_Age_Model.mdl')
max_age_model.save('Max_Age_Model.mdl')


# After training a fairly basic NN model to predict the minimum and maximum recommended reading ages of a given text, we saw a fair connection between the reading ease metric and the minimum and maximum reading age.
# Although by themselves, those two features may not be strong predictors of the reading ease metric but we do see a confirmation of our initial hypothesis, which states a connection between the target reading age interval and the reading ease metric,
# It may be beneficial to involve these new age parameters as part of any robust model and test their contribution to the overall RMSE reduction in future works.
# 
