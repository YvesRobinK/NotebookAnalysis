#!/usr/bin/env python
# coding: utf-8

# ![cc](data:image/png;base64,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)

# # Introduction
# ---

# **Competition's Objective:**
# 
# The objective of this competition is to assess the language proficiency of English Language Learners (ELLs). Using a data set of articles written by ELLs will help develop competency models that best support all students.
# 
# Your work will help English language learners receive more accurate feedback on their language development and speed up the grading cycle for teachers. These results can enable English language learners to receive more appropriate learning tasks that will help them improve their English language proficiency.

# **Data Description**
# 
# The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.
# 
# Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set.

# ### **Importing packages ⬇️**

# In[52]:


import pandas as pd
import numpy as np
import re
# for removing accented and special chracters
import unicodedata
# for stopwords Removal
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# for calculating Polarity and Subjectivity
from textblob import TextBlob
# for Wordscloud
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet

import keras
import tensorflow as tf
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
from textblob import Word
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import set_config
set_config(print_changed_only = False)


# In[6]:


train_df=pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')


# In[54]:


# read the data
train_df.head(10)


# In[7]:


test_df=pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
test_df


# Show the Information about the data

# In[56]:


# Data information
train_df.info()


# In[57]:


# Basic Statistic
train_df[['cohesion', 'syntax', 'vocabulary',
       'phraseology', 'grammar', 'conventions']].describe().T.style.background_gradient(cmap='Blues')


# In[58]:


# show the some of text from column 'Full Text'
train_df['full_text'][0]


# ## Clean Data

# In[59]:


train_df['full_text'] = train_df["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
test_df['full_text'] = test_df["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)


# In[60]:


# lets create a function to remove accented characters
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text

# lets apply the function
train_df['full_text'] = train_df.apply(lambda x: remove_accented_chars(x['full_text']), axis = 1)


# In[61]:


# Create a function to remove special characters
def remove_special_characters(text):
    pat = r'[^a-zA-z0-9]' 
    return re.sub(pat, ' ', text)
 
# lets apply this function
train_df['full_text'] = train_df.apply(lambda x: remove_special_characters(x['full_text']), axis = 1)


# In[62]:


train_df['full_text'][0]


# That is the data after make cleaning

# In[63]:


# lets print the Stopwords
print(stopwords.words('english'))


# In[64]:


# Now lets Remove the Stopwords

# targeting only English Stopwords
stop = stopwords.words('english')
stop_words = []
from nltk.tokenize import word_tokenize

text_tokens = word_tokenize(train_df['full_text'][0])

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

print(tokens_without_sw)


# # Feature Engineering

# In[65]:


data=train_df[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]
def count(x):
    for i in x:
        return x.value_counts()
count(data)


# In[66]:


# Lets calculate the length of the Reviews
train_df['length'] = train_df['full_text'].apply(len)


# ### Text Polarity

# In[67]:


# Lets calculate the Polarity of the Reviews
def get_polarity(text):
    textblob = TextBlob(str(text.encode('utf-8')))
    pol = textblob.sentiment.polarity
    return pol

# lets apply the function
train_df['polarity'] = train_df['full_text'].apply(get_polarity)


# ### Text Subjectivity

# In[68]:


# Lets calculate the Subjectvity of the Reviews
def get_subjectivity(text):
    textblob = TextBlob(str(text.encode('utf-8')))
    subj = textblob.sentiment.subjectivity
    return subj

# lets apply the Function
train_df['subjectivity'] = train_df['full_text'].apply(get_subjectivity)


# In[69]:


## lets summarize the Newly Created Features
train_df[['length','polarity','subjectivity']].describe()


# In[70]:


## Visualizing Polarity and Subjectivity

plt.rcParams['figure.figsize'] = (10, 4)

plt.subplot(1, 2, 1)
sns.distplot(train_df['polarity'])

plt.subplot(1, 2, 2)
sns.distplot(train_df['subjectivity'])

plt.suptitle('Distribution of Polarity and Subjectivity')
plt.show()


# In[71]:


# lets check relation between Polarity and Subjectivity

sns.scatterplot(train_df['polarity'], train_df['subjectivity'])
plt.title('Polarity vs Subjectivity')
plt.show()


# In[72]:


## Visualizing the Most Frequent Words

from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train_df['full_text'])
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Most Frequently Occuring Words - Top 20")
plt.show()


# In[73]:


## Visualizing the Least Frequent Words

from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train_df['full_text'])
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 20))
frequency.tail(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Least Frequently Occuring Words - Top 20")
plt.show()


# In[74]:


# lets plot the Wordscloud

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train_df['full_text'])
sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000).generate_from_frequencies(dict(words_freq))

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Vocabulary from Reviews", fontsize = 20)
plt.show()


# ## Character Count

# In[75]:


train_df['ncharacters'] = train_df['full_text'].str.len()
avg_char = round(train_df['ncharacters'].mean())
max_char = round(train_df['ncharacters'].max())
print('Average length: {}'.format(avg_char))
print('Max length: {}'.format(max_char))


# In[76]:


plt.figure(figsize = (22,5))
sns.distplot(train_df['ncharacters'])
plt.axvline(x = avg_char, color = 'red')
plt.title('Character Count')


# ## Word Count
# 

# In[77]:


train_df['nwords'] = train_df['full_text'].apply(lambda x: len(x.split()))
avg_words = round(train_df['nwords'].mean())
max_words = round(train_df['nwords'].max())
print('Average length: {}'.format(avg_words))
print('Max length: {}'.format(max_words))


# In[78]:


plt.figure(figsize = (22,5))
sns.distplot(train_df['nwords'])
plt.axvline(x = avg_words, color = 'red')
plt.title('Word count')


# In[79]:


def plot_distribution_per_score(c): 
    scores = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
    figure, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (22,5))
    for i, col in enumerate(scores):
        conditionlist = [
        (train_df[col] >= 4.5) ,
        (train_df[col] >= 2) & (train_df[col] < 4.5),
        (train_df[col] < 2)]
        choicelist = ['High', 'Mid', 'Low']
        train_df['performance'] = np.select(conditionlist, choicelist, default='Not Specified')

        mask = train_df.performance != 'Mid'
        sns.kdeplot(train_df[mask][c], hue = train_df.performance, ax = axes[i])
        axes[i].set_title(col)

        mask_low = train_df.performance == 'Low'
        avg_low = train_df[mask_low][c].mean()
        axes[i].axvline(x = avg_low, color = 'green', linestyle = '--')

        mask_high = train_df.performance == 'High'
        avg_high = train_df[mask_high][c].mean()
        axes[i].axvline(x = avg_high, color = 'orange', linestyle = '--')

        del train_df['performance']


# In[80]:


plot_distribution_per_score('nwords')


# ## Correlations

# In[81]:


corr= train_df.select_dtypes(['int','float']).corr()
# Getting the Upper Triangle of the co-relation matrix
matrix = np.triu(corr)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (22,8))
# Heatmap without absolute values
sns.heatmap(corr, mask=matrix, center = 0, cmap = 'vlag', ax = axes[0], 
            annot=True, fmt='.2f').set_title('Without absolute values')
# Heatmap with absolute values
sns.heatmap(abs(corr), mask=matrix, center = 0, cmap = 'vlag', ax = axes[1], 
           annot=True, fmt='.2f').set_title('With absolute values')

fig.tight_layout(h_pad=1.0, w_pad=0.5)


# In[82]:


from textblob import TextBlob

train_df['polarity'] = train_df['full_text'].apply(lambda x: TextBlob(x).sentiment[0])
train_df['subjetivity'] = train_df['full_text'].apply(lambda x: TextBlob(x).sentiment[1])
figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16,4))
sns.kdeplot(train_df['polarity'], ax = axes[0])
axes[0].set_title('Polarity Distribution')
sns.kdeplot(train_df['subjetivity'], ax = axes[1])
axes[1].set_title('Subjetivity Distribution')


# # Modeling 

# In[1]:


from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

# ----------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


import transformers
print('transformers version:',transformers.__version__)


# In[3]:


config = {
    'model': 'bert-base-uncased',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 16,
    'epochs': 5,
    'lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'scheduler': 'CosineAnnealingWarmRestarts'
}


# # Tokenizer, Dataset

# In[19]:


df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
test_df = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')


# In[20]:


tokenizer = AutoTokenizer.from_pretrained(config['model'])


# In[22]:


class EssayDataset:
    def __init__(self, df, config, tokenizer=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.classes = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
        self.max_len = config['max_length']
        self.tokenizer = tokenizer
        self.is_test = is_test
        
    def __getitem__(self,idx):
        sample = self.df['full_text'][idx]
        tokenized = tokenizer.encode_plus(sample,
                                          None,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          truncation=True,
                                          padding='max_length'
                                         )
        inputs = {
            "input_ids": torch.tensor(tokenized['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        }
        
        if self.is_test == True:
            return inputs
        
        label = self.df.loc[idx,self.classes].to_list()
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }
        
        return inputs, targets
    
    def __len__(self):
        return len(self.df)


# In[24]:


train_df, val_df = train_test_split(df,test_size=0.2,random_state=1357,shuffle=True)
print('dataframe shapes:',train_df.shape, val_df.shape)


# In[25]:


train_ds = EssayDataset(train_df, config, tokenizer=tokenizer)
val_ds = EssayDataset(val_df, config, tokenizer=tokenizer)
test_ds = EssayDataset(test_df, config, tokenizer=tokenizer, is_test=True)


# In[26]:


train_ds[0][0]['input_ids'].shape


# In[28]:


train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=config['batch_size'],
                                           shuffle=True,
                                           num_workers=2,
                                           pin_memory=True
                                          )
val_loader = torch.utils.data.DataLoader(val_ds,
                                         batch_size=config['batch_size'],
                                         shuffle=True,
                                         num_workers=2,
                                         pin_memory=True
                                        )


# In[31]:


print('loader shapes:',len(train_loader), len(val_loader))


# In[33]:


class EssayModel(nn.Module):
    def __init__(self,config,num_classes=6):
        super(EssayModel,self).__init__()
        self.model_name = config['model']
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.encoder.config.hidden_size,64)
        self.fc2 = nn.Linear(64,num_classes)
        
    def forward(self,inputs):
        _,outputs = self.encoder(**inputs, return_dict=False)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs


# In[34]:


# Puilding the model
class Trainer:
    def __init__(self, model, loaders, config):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.input_keys = ['input_ids','token_type_ids','attention_mask']
        
        self.optim = self._get_optim()
        
        self.scheduler_options = {
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5,eta_min=1e-7),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', min_lr=1e-7),
            'StepLR': torch.optim.lr_scheduler.StepLR(self.optim,step_size=2)
        }
        
        self.scheduler = self.scheduler_options[self.config['scheduler']]
        
        self.train_losses = []
        self.val_losses = []
        
    def _get_optim(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'])
        return optimizer

        
    def loss_fn(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss
    
    def train_one_epoch(self,epoch):
        
        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))
        
        for i,(inputs,targets) in enumerate(progress):
            
            self.optim.zero_grad()
            
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            targets = targets['labels'].to(device=config['device'])
            
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()
            
            loss.backward()
            self.optim.step()
            
            if self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch-1+i/len(self.train_loader)) # as per pytorch docs
            
            del inputs, targets, outputs, loss
            
        if self.config['scheduler'] == 'StepLR':
            self.scheduler.step()
            
        train_loss = running_loss/len(self.train_loader)
        self.train_losses.append(train_loss)
        
    @torch.no_grad()
    def valid_one_epoch(self,epoch):
        
        running_loss = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))
        
        for (inputs, targets) in progress:
            
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            targets = targets['labels'].to(device=config['device'])
            
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()
            
            del inputs, targets, outputs, loss
            
        
        val_loss = running_loss/len(self.val_loader)
        self.val_losses.append(val_loss)
        
        if config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)
            
    
    def test(self, test_loader):
        
        preds = []
        for (inputs) in test_loader:
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
            
        preds = torch.concat(preds)
        return preds
    
    def fit(self):
        
        fit_progress = tqdm(
            range(1, self.config['epochs']+1),
            leave = True,
            desc="Training..."
        )
        
        for epoch in fit_progress:
            
            self.model.train()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()
            
            self.model.eval()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.valid_one_epoch(epoch)
            self.clear()

            print(f"{'-'*30} EPOCH {epoch} / {self.config['epochs']} {'-'*30}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"valid loss: {self.val_losses[-1]}\n\n")
            
    
    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()


# In[35]:


model = EssayModel(config).to(device=config['device'])
trainer = Trainer(model, (train_loader, val_loader), config)


# In[36]:


# Fit the model
trainer.fit()


# In[37]:


losses_df = pd.DataFrame({'epoch':list(range(1,config['epochs'] + 1)),
                          'train_loss':trainer.train_losses, 
                          'val_loss': trainer.val_losses
                         })
losses_df


# ### Plot

# In[38]:


plt.plot(trainer.train_losses, color='red')
plt.plot(trainer.val_losses, color='orange')
plt.title('MCRMSE Loss')
plt.legend(['Train', 'Validation'], loc='upper right')


# In[ ]:




