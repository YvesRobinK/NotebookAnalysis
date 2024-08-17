#!/usr/bin/env python
# coding: utf-8

# ## imports and functions

# In[1]:


# imports
import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import pprint
import re

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import cv2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.preprocessing import StandardScaler, RobustScaler

import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

import lightgbm as lgb
import xgboost as xgb
import time

from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.models import LdaMulticore
from gensim import corpora
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE

import itertools

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from nltk.tokenize import TweetTokenizer
import nltk
isascii = lambda s: len(s) == len(s.encode())
tknzr = TweetTokenizer()
import jieba
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")


# In[3]:


# torch imports
from torch import nn
import torch
from torch.nn import functional as F
from torchvision.models import resnet50, resnet34, densenet201, densenet121
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import TensorDataset


# In[4]:


# util funcs
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    min_rating, max_rating =None, None
    rater_a, rater_b = np.array(y, dtype=int), np.array(y_pred, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator, denominator = 0.0, 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = np.square(i - j) / np.square(num_ratings - 1)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def val_kappa(preds, train_data):
    labels = train_data.get_label()
    preds = np.argmax(preds.reshape((-1,5)), axis=1)
    
    return 'qwk', quadratic_weighted_kappa(labels, preds), True

def val_kappa_reg(preds, train_data, cdf):
    labels = train_data.get_label()
    preds = getTestScore2(preds, cdf)
    return 'qwk', quadratic_weighted_kappa(labels, preds), True

def get_cdf(hist):
    return np.cumsum(hist/np.sum(hist))

def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([4]*num, dtype=int)
    rank = pred.argsort()
    output[rank[:int(num*cdf[0]-1)]] = 0
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 1
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 2
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 3
    if valid:
        cutoff = [ pred[rank[int(num*cdf[i]-1)]] for i in range(4) ]
        return output, cutoff
    return output

def getTestScore(pred, cutoff):
    num = pred.shape[0]
    output = np.asarray([4]*num, dtype=int)
    for i in range(num):
        if pred[i] <= cutoff[0]:
            output[i] = 0
        elif pred[i] <= cutoff[1]:
            output[i] = 1
        elif pred[i] <= cutoff[2]:
            output[i] = 2
        elif pred[i] <= cutoff[3]:
            output[i] = 3
    return output

def getTestScore2(pred, cdf):
    num = pred.shape[0]
    rank = pred.argsort()
    output = np.asarray([4]*num, dtype=int)
    output[rank[:int(num*cdf[0]-1)]] = 0
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 1
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 2
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 3
    return output

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

isascii = lambda s: len(s) == len(s.encode())

def custom_tokenizer(text):
    init_doc = tknzr.tokenize(text)
    retval = []
    for t in init_doc:
        if isascii(t): 
            retval.append(t)
        else:
            for w in t:
                retval.append(w)
    return retval

def build_emb_matrix(word_dict, emb_dict):
    embed_size = 300
    nb_words = len(word_dict)+1000
    nb_oov = 0
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    for key in tqdm(word_dict):
        word = key
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        nb_oov+=1
        embedding_matrix[word_dict[key]] = unknown_vector                    
    return embedding_matrix, nb_words, nb_oov

def _init_esim_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        if isinstance(module, nn.LSTM):
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
        else:
            hidden_size = module.bias_hh_l0.data.shape[0] // 3
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif '_ih_' in name:
                nn.init.xavier_normal_(param)
            elif '_hh_' in name:
                nn.init.orthogonal_(param)
                param.data[hidden_size:(2 * hidden_size)] = 1.0


# In[5]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# In[6]:


class ResnetModel(nn.Module):
    def __init__(self, resnet_fun = resnet50, freeze_basenet = True):
        super(ResnetModel, self).__init__()
        self.resnet = resnet_fun(pretrained=False)
        if freeze_basenet:
            for p in self.resnet.parameters():
                p.requires_grad = False
       
    def init_resnet(self, path):
        state = torch.load(path)
        self.resnet.load_state_dict(state)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x/255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        return x
    
class DenseModel(nn.Module):
    def __init__(self, dense_func = densenet201, freeze_basenet = True):
        super(DenseModel, self).__init__()
        self.densenet = dense_func(pretrained=False)
#         model_name = 'se_resnet50'
#         self.resnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
        if freeze_basenet:
            for p in self.densenet.parameters():
                p.requires_grad = False
        
    def init_densenet(self, path):
        state = torch.load(path)
        self.densenet.load_state_dict(state)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x/255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)
        x = self.densenet.features(x)
        x = F.adaptive_avg_pool2d(F.relu(x,inplace=True), output_size=1).view(batch_size, -1)
        return x


# In[7]:


from torch.utils.data import Dataset, DataLoader

def img_to_torch(image):
    return torch.from_numpy(np.transpose(image, (2, 0, 1)))

def pad_to_square(image):
    h, w = image.shape[0:2]
    new_size = max(h, w)
    delta_top = (new_size-h)//2
    delta_bottom = new_size-h-delta_top
    delta_left = (new_size-w)//2
    delta_right = new_size-delta_left-w
    new_im = cv2.copyMakeBorder(image, delta_top, delta_bottom, delta_left, delta_right, 
                                cv2.BORDER_CONSTANT,  value=[0,0,0])
    return new_im

class PetDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.path = df['path'].tolist()
        self.cache = {}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.df)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        # only take on channel
#         if index not in self.cache:
        image = cv2.imread(self.path[index])
        image = pad_to_square(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#             self.cache[index] = img_to_torch(image)

        return img_to_torch(image)


# ## read data

# In[8]:


os.listdir('../input/')


# In[9]:


labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')


# In[10]:


# read data
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
train_len = len(train)
data_df = pd.concat([train, test], sort=False).reset_index(drop=True)
data_df['Breed_full'] = data_df['Breed1'].astype(str)+'_'+data_df['Breed2'].astype(str)
data_df['Color_full'] = data_df['Color1'].astype(str)+'_'+data_df['Color2'].astype(str)+'_'+data_df['Color3'].astype(str)

data_df['Breed_full'],_ = pd.factorize(data_df['Breed_full'])
data_df['Color_full'],_ = pd.factorize(data_df['Color_full'])
data_df['State'],_ = pd.factorize(data_df['State'])

data_df['hard_interaction'] = data_df['Type'].astype(str)+data_df['Gender'].astype(str)+ \
                              data_df['Vaccinated'].astype(str)+'_'+ \
                              data_df['Dewormed'].astype(str)+'_'+data_df['Sterilized'].astype(str)
data_df['hard_interaction'],_ = pd.factorize(data_df['hard_interaction'])

data_df['MaturitySize'] = data_df['MaturitySize'].replace(0, np.nan)
data_df['FurLength'] = data_df['FurLength'].replace(0, np.nan)

data_df['Vaccinated'] = data_df['Vaccinated'].replace(3, np.nan)
data_df['Vaccinated'] = data_df['Vaccinated'].replace(2, 0)

data_df['Dewormed'] = data_df['Dewormed'].replace(3, np.nan)
data_df['Dewormed'] = data_df['Dewormed'].replace(2, 0)

data_df['Sterilized'] = data_df['Sterilized'].replace(3, np.nan)
data_df['Sterilized'] = data_df['Sterilized'].replace(2, 0)


data_df['Health'] = data_df['Health'].replace(0, np.nan)
data_df['age_in_year'] = (data_df['Age']//12).astype(np.int8)
data_df['avg_fee'] = data_df['Fee']/data_df['Quantity']
data_df['avg_photo'] = data_df['PhotoAmt']/data_df['Quantity']

# name feature
pattern = re.compile(r"[0-9\.:!]")
data_df['empty_name'] = data_df['Name'].isnull().astype(np.int8)
data_df['Name'] =data_df['Name'].fillna('')
data_df['name_len'] = data_df['Name'].apply(lambda x: len(x))
data_df['strange_name'] = data_df['Name'].apply(lambda x: len(pattern.findall(x))>0).astype(np.int8)


# In[11]:


data_df['color_num'] = 1
data_df['color_num'] += data_df['Color2'].apply(lambda x: 1 if x!=0 else 0)
data_df['color_num'] += data_df['Color3'].apply(lambda x: 1 if x!=0 else 0)


# In[12]:


# breed feature
labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
labels_breed.rename(index=str, columns={'BreedName':'Breed1Name'},inplace=True)
labels_breed['Breed2Name'] = labels_breed['Breed1Name'].values


# In[13]:


data_df = data_df.merge(labels_breed[['BreedID','Breed1Name']], left_on='Breed1', right_on='BreedID', how='left')
data_df.drop('BreedID',axis=1,inplace=True)
data_df = data_df.merge(labels_breed[['BreedID','Breed2Name']], left_on='Breed2', right_on='BreedID', how='left')
data_df.drop('BreedID',axis=1,inplace=True)
data_df['Breed2Name'].fillna('',inplace=True)
data_df['BreedName_full'] = data_df['Breed1Name']+' '+data_df['Breed2Name']


# In[14]:


data_df['breed_noname'] = data_df['BreedName_full'].isnull().astype(np.int8)
data_df['BreedName_full'].fillna('',inplace=True)
data_df['BreedName_full'] = data_df['BreedName_full'].str.lower()


# In[15]:


data_df['breed_num'] = 1
data_df['breed_num'] += data_df['Breed2'].apply(lambda x: 1 if x!=0 else 0)
data_df['breed_mixed'] = data_df['BreedName_full'].apply(lambda x: x.find('mixed')>=0).astype(np.int8)
data_df['breed_Domestic'] = data_df['BreedName_full'].apply(lambda x: x.find('domestic')>=0).astype(np.int8)
data_df['pure_breed'] = ((data_df['breed_num']==1)&(data_df['breed_mixed']==0)).astype(np.int8)


# In[16]:


# 
# data_df['breed_American'] = data_df['BreedName_full'].apply(lambda x: x.find('american')>=0).astype(np.int8)
# data_df['breed_Australian'] = data_df['BreedName_full'].apply(lambda x: x.find('australian')>=0).astype(np.int8)
# data_df['breed_Belgian'] = data_df['BreedName_full'].apply(lambda x: x.find('belgian')>=0).astype(np.int8)
# data_df['breed_English'] = data_df['BreedName_full'].apply(lambda x: x.find('english')>=0).astype(np.int8)
# data_df['breed_German'] = data_df['BreedName_full'].apply(lambda x: x.find('german')>=0).astype(np.int8)
# data_df['breed_irish'] = data_df['BreedName_full'].apply(lambda x: x.find('irish')>=0).astype(np.int8)
# 
# data_df['breed_Oriental'] = data_df['BreedName_full'].apply(lambda x: x.find('Oriental')>=0).astype(np.int8)


# ### external data

# In[17]:


import json
# add features from ratings 
with open('../input/cat-and-dog-breeds-parameters/rating.json', 'r') as f:
    ratings = json.load(f)


# In[18]:


cat_ratings = ratings['cat_breeds']
dog_ratings = ratings['dog_breeds']
catdog_ratings = {**cat_ratings, **dog_ratings} 


# In[19]:


parameters_df=pd.DataFrame()
i = 0
for breed in catdog_ratings.keys():
    for key in catdog_ratings[breed].keys():
        parameters_df.at[i,'breed'] = breed
        parameters_df.at[i,key] = catdog_ratings[breed][key] 
    i = i+1


# In[20]:


parameters_df.rename(index=str, columns={'breed':'Breed1Name'},inplace=True)


# In[21]:


data_df = data_df.merge(parameters_df[['Breed1Name','Affectionate with Family','Amount of Shedding',
                                      'Easy to Groom','General Health','Intelligence','Kid Friendly',
                                      'Pet Friendly','Potential for Playfulness']], on='Breed1Name', how='left')


# ### description

# In[22]:


data_df['Description'] = data_df['Description'].fillna(' ')


# In[23]:


english_desc, chinese_desc = [], []
tokens = set()
word_dict = {}
pos_count, word_count = 1, 1 # starts from 1, 0 for padding token
pos_dict = {}
eng_sequences = []
pos_sequences = []
for i in range(len(data_df)):
    e_d, c_d, eng_seq, pos_seq = [], [], [], []
    doc = custom_tokenizer(data_df['Description'].iloc[i])
    for token in doc:
        if not isascii(token):
            c_d.append(token)
        else:
            e_d.append(token)
            if token not in word_dict:
                word_dict[token] = word_count
                word_count +=1
    english_desc.append(' '.join(e_d))
    chinese_desc.append(' '.join(c_d))
    pos_seq = nltk.pos_tag(e_d)
    for t in pos_seq:
        if t[1] not in pos_dict:
            pos_dict[t[1]] = pos_count
            pos_count += 1
    pos_seq = [pos_dict[t[1]] for t in pos_seq]
    eng_seq = [word_dict[t] for t in e_d]
    if len(eng_seq)==0:
        eng_seq.append(0)
        pos_seq.append(0)
    eng_sequences.append(eng_seq)
    pos_sequences.append(pos_seq)


# In[24]:


data_df['English_desc'] = english_desc
data_df['Chinese_desc'] = chinese_desc

data_df['e_description_len'] = data_df['English_desc'].apply(lambda x:len(x))
data_df['e_description_word_len'] = data_df['English_desc'].apply(lambda x: len(x.split(' ')))
data_df['e_description_word_unique'] = data_df['English_desc'].apply(lambda x: len(set(x.split(' '))))

data_df['c_description_len'] = data_df['Chinese_desc'].apply(lambda x:len(x))
data_df['c_description_word_len'] = data_df['Chinese_desc'].apply(lambda x:len(x.split(' ')))
data_df['c_description_word_unique'] = data_df['Chinese_desc'].apply(lambda x: len(set(x)))

data_df['description_len'] = data_df['Description'].apply(lambda x:len(x))
data_df['description_word_len'] = data_df['Description'].apply(lambda x: len(x.split(' ')))


# In[25]:


print(len(eng_sequences))
print(len(pos_sequences))


# In[26]:


nb_pos = len(pos_dict)
print(nb_pos)


# In[27]:


# build embedding
def load_glove():
    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in (open(EMBEDDING_FILE)))
    return embeddings_index

glove_emb = load_glove()

embedding_matrix, nb_words, nb_oov = build_emb_matrix(word_dict, glove_emb)
print(nb_words, nb_oov)
del glove_emb
gc.collect()


# In[28]:


# np.save('mini_embedding.npy',embedding_matrix)


# In[29]:


# embedding_matrix = np.load('mini_embedding.npy')
# nb_words = 33947


# ### split data

# In[30]:


len(set(train.index.tolist()))


# In[31]:


n_splits = 5
# kfold = GroupKFold(n_splits=n_splits)
split_index = []
# for train_idx, valid_idx in kfold.split(train, train['AdoptionSpeed'], train['RescuerID']):
#     split_index.append((train_idx, valid_idx))

kfold = StratifiedKFold(n_splits=n_splits, random_state=1991)
for train_idx, valid_idx in kfold.split(train, train['AdoptionSpeed']):
    split_index.append((train_idx, valid_idx))


# ### load mapping dictionaries:

# In[32]:


train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

print('num of train images files: {}'.format(len(train_image_files)))
print('num of train metadata files: {}'.format(len(train_metadata_files)))
print('num of train sentiment files: {}'.format(len(train_sentiment_files)))

test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

print('num of test images files: {}'.format(len(test_image_files)))
print('num of test metadata files: {}'.format(len(test_metadata_files)))
print('num of test sentiment files: {}'.format(len(test_sentiment_files)))

image_files = train_image_files+test_image_files
metadata_files = train_metadata_files+test_metadata_files
sentiment_files = train_sentiment_files+test_sentiment_files


# ## image feature

# In[33]:


IMG_SIZE = 256
BATCH_SIZE = 256

def get_petid(path):
    basename = os.path.basename(path)
    return basename.split('-')[0]
def get_picid(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    return basename.split('-')[1]

image_df = pd.DataFrame(image_files, columns=['path'])
image_df['PetID'] = image_df['path'].apply(get_petid)
image_df['PicID'] = image_df['path'].apply(get_picid)

gc.collect()


# In[34]:


image_df[image_df['PicID']=='1'].shape


# In[35]:


pet_image_dataset = PetDataset(image_df)
pet_image_loader = DataLoader(pet_image_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=1, pin_memory=True)


# In[36]:


# cat and dog model
resnet34_feature = []
model = ResnetModel(resnet_fun=resnet34)
model.init_resnet('../input/pretrained-pytorch-dog-and-cat-models/resnet34.pth')
model.cuda()
model.eval()
with torch.no_grad():
    for img_batch in tqdm(pet_image_loader):
        img_batch = img_batch.float().cuda()
        y_pred = model(img_batch)
        resnet34_feature.append(y_pred.cpu().numpy()) 
resnet34_feature = np.vstack(resnet34_feature)
del model
gc.collect()
torch.cuda.empty_cache()

densenet121_feature = []
model = DenseModel(dense_func=densenet121)
model.init_densenet('../input/pretrained-pytorch-dog-and-cat-models/densenet121.pth')
model.cuda()
model.eval()
with torch.no_grad():
    for img_batch in tqdm(pet_image_loader):
        img_batch = img_batch.float().cuda()
        y_pred = model(img_batch)
        densenet121_feature.append(y_pred.cpu().numpy()) 
densenet121_feature = np.vstack(densenet121_feature)
del model
# del pet_image_loader
# del pet_image_dataset
gc.collect()
torch.cuda.empty_cache()


# In[37]:


densenet_feature = []
model = DenseModel()
model.init_densenet('../input/pytorch-pretrained-image-models/densenet201.pth')
model.cuda()
model.eval()
with torch.no_grad():
    for img_batch in tqdm(pet_image_loader):
        img_batch = img_batch.float().cuda()
        y_pred = model(img_batch)
        densenet_feature.append(y_pred.cpu().numpy()) 
densenet_feature = np.vstack(densenet_feature)

del model
gc.collect()
torch.cuda.empty_cache()

resnet50_feature = []
model = ResnetModel()
model.init_resnet('../input/pytorch-pretrained-image-models/resnet50.pth')
model.cuda()
model.eval()
with torch.no_grad():
    for img_batch in tqdm(pet_image_loader):
        img_batch = img_batch.float().cuda()
        y_pred = model(img_batch)
        resnet50_feature.append(y_pred.cpu().numpy()) 
resnet50_feature = np.vstack(resnet50_feature)


# In[38]:


del img_batch


# In[39]:


del model
del pet_image_loader
del pet_image_dataset
gc.collect()
torch.cuda.empty_cache()


# In[40]:


RES50_IMG_FEATURE_DIM = resnet50_feature.shape[1]
DENSE_IMG_FEATURE_DIM = densenet_feature.shape[1]
RES34_IMG_FEATURE_DIM = resnet34_feature.shape[1]
DENSE121_IMG_FEATURE_DIM = densenet121_feature.shape[1]
print(RES50_IMG_FEATURE_DIM)
print(DENSE_IMG_FEATURE_DIM)
print(RES34_IMG_FEATURE_DIM)
print(DENSE121_IMG_FEATURE_DIM)


# In[41]:


resnet50_feature_df = pd.DataFrame(resnet50_feature, dtype=np.float32,
                                   columns=['resnet50_%d'%i for i in range(RES50_IMG_FEATURE_DIM)])
resnet50_feature_df['PetID'] = image_df['PetID'].values
resnet50_feature_df['PicID'] = image_df['PicID'].values
resnet50_feature_df_avg = resnet50_feature_df.drop('PicID', axis=1).groupby('PetID').agg('mean').reset_index()
resnet50_feature_df_avg.columns = ['PetID']+['resnet50_mean_%d'%i for i in range(RES50_IMG_FEATURE_DIM)]
resnet50_feature_df_1 = resnet50_feature_df[resnet50_feature_df['PicID']=='1'].drop('PicID', axis=1)
resnet50_feature_df = resnet50_feature_df_1.merge(resnet50_feature_df_avg, on='PetID', how='outer')
resnet50_feature_df = data_df[['PetID','RescuerID','AdoptionSpeed']].merge(resnet50_feature_df, on='PetID', how='left')
for c in resnet50_feature_df.columns:
    if c in ['PetID','RescuerID','AdoptionSpeed']:continue
    resnet50_feature_df[c] = resnet50_feature_df[c].fillna(-1)
del resnet50_feature_df_avg
del resnet50_feature_df_1
gc.collect()


# In[42]:


dense_feature_df = pd.DataFrame(densenet_feature, dtype=np.float32,
                                   columns=['densenet_%d'%i for i in range(DENSE_IMG_FEATURE_DIM)])
dense_feature_df['PetID'] = image_df['PetID'].values
dense_feature_df['PicID'] = image_df['PicID'].values
dense_feature_df_avg = dense_feature_df.drop('PicID', axis=1).groupby('PetID').agg('mean').reset_index()
dense_feature_df_avg.columns = ['PetID']+['densenet_mean_%d'%i for i in range(DENSE_IMG_FEATURE_DIM)]
dense_feature_df_1 = dense_feature_df[dense_feature_df['PicID']=='1'].drop('PicID', axis=1)
dense_feature_df = dense_feature_df_1.merge(dense_feature_df_avg, on='PetID', how='left')
dense_feature_df = data_df[['PetID','RescuerID','AdoptionSpeed']].merge(dense_feature_df, on='PetID', how='left')
for c in dense_feature_df.columns:
    if c in ['PetID','RescuerID','AdoptionSpeed']:continue
    dense_feature_df[c] = dense_feature_df[c].fillna(-1)
del dense_feature_df_1
del dense_feature_df_avg
gc.collect()


# In[43]:


# res34 on pet and dogs
res34_feature_df = pd.DataFrame(resnet34_feature, dtype=np.float32,
                                   columns=['densenet_%d'%i for i in range(RES34_IMG_FEATURE_DIM)])
res34_feature_df['PetID'] = image_df['PetID'].values
res34_feature_df['PicID'] = image_df['PicID'].values
res34_feature_df_avg = res34_feature_df.drop('PicID', axis=1).groupby('PetID').agg('mean').reset_index()
res34_feature_df_avg.columns = ['PetID']+['res34_mean_%d'%i for i in range(RES34_IMG_FEATURE_DIM)]
res34_feature_df_1 = res34_feature_df[res34_feature_df['PicID']=='1'].drop('PicID', axis=1)
res34_feature_df = res34_feature_df_1.merge(res34_feature_df_avg, on='PetID', how='left')
res34_feature_df = data_df[['PetID','RescuerID','AdoptionSpeed']].merge(res34_feature_df, on='PetID', how='left')
for c in res34_feature_df.columns:
    if c in ['PetID','RescuerID','AdoptionSpeed']:continue
    res34_feature_df[c] = res34_feature_df[c].fillna(-1)
del res34_feature_df_1
del res34_feature_df_avg
gc.collect()


# In[44]:


# densenet121 on pet and dogs
dense121_feature_df = pd.DataFrame(densenet121_feature, dtype=np.float32,
                                   columns=['densenet_%d'%i for i in range(DENSE121_IMG_FEATURE_DIM)])
dense121_feature_df['PetID'] = image_df['PetID'].values
dense121_feature_df['PicID'] = image_df['PicID'].values
dense121_feature_df_avg = dense121_feature_df.drop('PicID', axis=1).groupby('PetID').agg('mean').reset_index()
dense121_feature_df_avg.columns = ['PetID']+['dense121_mean_%d'%i for i in range(DENSE121_IMG_FEATURE_DIM)]
dense121_feature_df_1 = dense121_feature_df[dense121_feature_df['PicID']=='1'].drop('PicID', axis=1)
dense121_feature_df = dense121_feature_df_1.merge(dense121_feature_df_avg, on='PetID', how='left')
dense121_feature_df = data_df[['PetID','RescuerID','AdoptionSpeed']].merge(dense121_feature_df, on='PetID', how='left')
for c in dense121_feature_df.columns:
    if c in ['PetID','RescuerID','AdoptionSpeed']:continue
    dense121_feature_df[c] = dense121_feature_df[c].fillna(-1)
del dense121_feature_df_1
del dense121_feature_df_avg
gc.collect()


# In[45]:


class ImgExtractor(nn.Module):
    def __init__(self, raw_feature_dim = 4096, feature_dim=64):
        super(ImgExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.BatchNorm1d(raw_feature_dim),
            nn.Dropout(0.5),
            nn.Linear(raw_feature_dim, raw_feature_dim//4),
            nn.ELU(inplace=True),
#             nn.BatchNorm1d(raw_feature_dim//4),
            nn.Dropout(0.1),
            nn.Linear(raw_feature_dim//4, raw_feature_dim//8),
            nn.ELU(inplace=True),
#             nn.Dropout(0.1),
#             nn.BatchNorm1d(raw_feature_dim//8),
            nn.Linear(raw_feature_dim//8, feature_dim),
            nn.ELU(inplace=True)
        )
        self.logit = nn.Sequential(
            nn.Linear(feature_dim, 1)
        )
        
#         self.apply(_init_esim_weights)
        
    def forward(self, x):
        batch_size = x.shape[0]
        feat = self.extractor(x)
        out = self.logit(feat)
        return out, feat


# In[46]:


IMG_FEATURE_DIM_NN = 128
IMG_FEATURE_DIM_NN2 = 64

def get_image_feature(img_train_data, img_test_data,img_feature_dim):
    loss_fn = torch.nn.MSELoss().cuda()

    oof_train_img = np.zeros((img_train_data.shape[0], img_feature_dim+1))
    oof_test_img = []

    # X_test = raw_img_features_test.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1)
    test_dataset = TensorDataset(torch.tensor(img_test_data))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=1, pin_memory=True)

    qwks = []
    rmses = []

    for n_fold, (train_idx, valid_idx) in enumerate(split_index): 
        print('fold:',n_fold)
        X_tr = img_train_data[train_idx]
        X_val = img_train_data[valid_idx]

        y_tr = train.iloc[train_idx]['AdoptionSpeed'].values    
        y_val = train.iloc[valid_idx]['AdoptionSpeed'].values

        hist = histogram(y_tr.astype(int), 
                         int(np.min(train['AdoptionSpeed'])), 
                         int(np.max(train['AdoptionSpeed'])))
        tr_cdf = get_cdf(hist)

        tra_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        training_loader = DataLoader(tra_dataset, batch_size=1024, shuffle=True, num_workers=1, pin_memory=True)
        validation_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=1, pin_memory=True)

        model = ImgExtractor(raw_feature_dim=img_train_data.shape[1], feature_dim=img_feature_dim)
        model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7, eta_min=0.0003)
        iteration = 0
        min_val_loss = 100
        since = time.time()
        test_feats = None
        for epoch in range(15):
            scheduler.step()
            model.train()
            for x, y in training_loader:
                iteration += 1
                x = x.cuda()
                y = y.type(torch.FloatTensor).cuda().view(-1, 1)

                pred, feat = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_predicts = []
            val_feats = []
            with torch.no_grad():
                for x, y in validation_loader:
                    x = x.cuda()
                    y = y.type(torch.FloatTensor).cuda()#.view(-1, 1)
                    v_pred, feat = model(x)
                    val_predicts.append(v_pred.cpu().numpy())
                    val_feats.append(feat.cpu().numpy())

            val_predicts = np.concatenate(val_predicts) 
            val_feats = np.vstack(val_feats)
            pred_test_y_k = getTestScore2(val_predicts.flatten(), tr_cdf)
            qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
            val_loss = rmse(y_val,val_predicts)

            if val_loss<min_val_loss:
                min_val_loss = val_loss
                oof_train_img[valid_idx,:] = np.hstack([val_feats,val_predicts])
                test_feats = []
                test_preds = []
                with torch.no_grad():
                    for x, in test_loader:
                        x = x.cuda()
                        v_pred, feat = model(x)
                        test_preds.append(v_pred.cpu().numpy())
                        test_feats.append(feat.cpu().numpy())
                test_feats = np.hstack([np.vstack(test_feats), np.concatenate(test_preds)])
                print(epoch, "best loss! val loss:", val_loss, 'qwk:', qwk, "elapsed time:", time.time()-since)
        oof_test_img.append(test_feats)
        rmses.append(min_val_loss)
        qwks.append(qwk)
        del model
        del x
        del y
        del tra_dataset
        del val_dataset
        del training_loader
        del validation_loader
        gc.collect()
        torch.cuda.empty_cache()
    print('overall rmse: %.5f'%rmse(oof_train_img[:,-1], train['AdoptionSpeed']))
    print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
    print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))
    del test_loader
    del test_dataset
    gc.collect()
    return oof_train_img, np.mean(oof_test_img, axis=0)


# In[47]:


res34_array = res34_feature_df.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1).values
# raw_img_array  = stdscaler.fit_transform(raw_img_array)
res34_tra = res34_array[0:train_len]
res34_test = res34_array[train_len:]
oof_train_res34, oof_test_res34 = get_image_feature(res34_tra, res34_test, IMG_FEATURE_DIM_NN2)
del res34_tra
del res34_test
gc.collect()
dnn_resnet34_features = np.vstack([oof_train_res34, oof_test_res34])
dnn_resnet34_features = pd.DataFrame(dnn_resnet34_features, columns=['res34_%d'%c for c in range(dnn_resnet34_features.shape[1])])
dnn_resnet34_features['PetID'] = data_df['PetID'].values
del oof_train_res34
del oof_test_res34
gc.collect()


# In[48]:


stdscaler = RobustScaler()
res50_array = resnet50_feature_df.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1).values
# raw_img_array  = stdscaler.fit_transform(raw_img_array)
res50_tra = res50_array[0:train_len]
res50_test = res50_array[train_len:]
oof_train_res, oof_test_res = get_image_feature(res50_tra, res50_test, IMG_FEATURE_DIM_NN)
del res50_tra
del res50_test
gc.collect()
dnn_resnet50_features = np.vstack([oof_train_res, oof_test_res])
dnn_resnet50_features = pd.DataFrame(dnn_resnet50_features, columns=['resnet_%d'%c for c in range(dnn_resnet50_features.shape[1])])
dnn_resnet50_features['PetID'] = data_df['PetID'].values
del oof_train_res
del oof_test_res
gc.collect()


# In[49]:


dense121_array = dense121_feature_df.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1).values
dense121_tra = dense121_array[0:train_len]
dense121_test = dense121_array[train_len:]
oof_train_dense121, oof_test_dense121 = get_image_feature(dense121_tra, dense121_test, IMG_FEATURE_DIM_NN)
del dense121_tra
del dense121_test
gc.collect()
dnn_dense121_features = np.vstack([oof_train_dense121, oof_test_dense121])
dnn_dense121_features = pd.DataFrame(dnn_dense121_features, columns=['dense121_%d'%c for c in range(dnn_dense121_features.shape[1])])
dnn_dense121_features['PetID'] = data_df['PetID'].values
del oof_train_dense121
del oof_test_dense121
gc.collect()


# In[50]:


dense_array = dense_feature_df.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1).values
dense_tra = dense_array[0:train_len]
dense_test = dense_array[train_len:]
oof_train_dense, oof_test_dense = get_image_feature(dense_tra, dense_test, IMG_FEATURE_DIM_NN)
del dense_tra
del dense_test
gc.collect()
dnn_dense_features = np.vstack([oof_train_dense, oof_test_dense])
dnn_dense_features = pd.DataFrame(dnn_dense_features, columns=['dense_%d'%c for c in range(dnn_dense_features.shape[1])])
dnn_dense_features['PetID'] = data_df['PetID'].values
del oof_train_dense
del oof_test_dense
gc.collect()


# In[51]:


n_components = 64


# features = raw_img_features.drop(['PetID','RescuerID','AdoptionSpeed'], axis=1).values
svd_ = TruncatedSVD(n_components=32, random_state=1337)
svd_resnet50_features = svd_.fit_transform(res50_array)
svd_resnet50_features = pd.DataFrame(svd_resnet50_features)
svd_resnet50_features = svd_resnet50_features.add_prefix('resnet50_SVD_')
svd_resnet50_features['PetID'] = resnet50_feature_df['PetID'].values


# In[52]:


svd_ = TruncatedSVD(n_components=32, random_state=1337)
svd_resnet34_features = svd_.fit_transform(res34_array)
svd_resnet34_features = pd.DataFrame(svd_resnet34_features)
svd_resnet34_features = svd_resnet34_features.add_prefix('resnet34_SVD_')
svd_resnet34_features['PetID'] = resnet50_feature_df['PetID'].values


# In[53]:


svd_ = TruncatedSVD(n_components=32, random_state=1337)
svd_dense121_features = svd_.fit_transform(dense121_array)
svd_dense121_features = pd.DataFrame(svd_dense121_features)
svd_dense121_features = svd_dense121_features.add_prefix('dense121_SVD_')
svd_dense121_features['PetID'] = resnet50_feature_df['PetID'].values


# In[54]:


svd_ = TruncatedSVD(n_components=32, random_state=1337)
svd_dense_features = svd_.fit_transform(dense_array)
svd_dense_features = pd.DataFrame(svd_dense_features)
svd_dense_features = svd_dense_features.add_prefix('dense_SVD_')
svd_dense_features['PetID'] = dense_feature_df['PetID'].values


# In[55]:


svd_ = TruncatedSVD(n_components=64, random_state=1337)
svd_img_features = svd_.fit_transform(np.hstack([res50_array, dense_array, dense121_array, res34_array]))
svd_img_features = pd.DataFrame(svd_img_features)
svd_img_features = svd_img_features.add_prefix('IMG_SVD_')
svd_img_features['PetID'] = resnet50_feature_df['PetID'].values


# In[56]:


#img clustering
from sklearn.cluster import DBSCAN, FeatureAgglomeration, KMeans

features = svd_img_features.drop(['PetID'], axis=1).values

cluster = KMeans(n_clusters=32)
cluster_label = cluster.fit_predict(features)

cluster_img_features = pd.DataFrame(cluster_label)
cluster_img_features = cluster_img_features.add_prefix('img_CLUSTER_')
cluster_img_features['PetID'] = dense_feature_df['PetID'].values

# features = svd_dense_features.drop(['PetID'], axis=1).values

# cluster = KMeans(n_clusters=32)
# cluster_label = cluster.fit_predict(features)

# cluster_dense_features = pd.DataFrame(cluster_label)
# cluster_dense_features = cluster_dense_features.add_prefix('dense_CLUSTER_')
# cluster_dense_features['PetID'] = dense_feature_df['PetID'].values


# In[57]:


svd_img_features.head()


# In[58]:


# svd_resnet50_features.to_pickle('image_features_svd.pkl')
# dnn_resnet50_features.to_pickle('image_features_nn.pkl')
# cluster_resnet50_features.to_pickle('image_features_cluster.pkl')


# ## meta & senti feature
# 
# After taking a look at the data, we know its structure and can use it to extract additional features and concatenate them with basic train/test DFs.

# In[59]:


class PetFinderParser(object):
    def __init__(self, debug=False):
        self.debug = debug
        self.sentence_sep = '; '
        
        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = False
        
    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file
            
    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file
            
    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(Image.open(filename))
        return image
        
    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """
        # documentSentiment
        ret_val = {}
        ret_val['doc_mag'] = file['documentSentiment']['magnitude']
        ret_val['doc_score']= file['documentSentiment']['score']
        ret_val['doc_language'] = file['language']
        ret_val['doc_stcs_len'] = len(file['sentences'])
        if ret_val['doc_stcs_len']>0:
            ret_val['doc_first_score'] = file['sentences'][0]['sentiment']['score']
            ret_val['doc_first_mag'] = file['sentences'][0]['sentiment']['magnitude']
            ret_val['doc_last_score'] = file['sentences'][-1]['sentiment']['score']
            ret_val['doc_last_mag'] = file['sentences'][-1]['sentiment']['magnitude']
        else:
            ret_val['doc_first_score'] = np.nan
            ret_val['doc_first_mag'] = np.nan
            ret_val['doc_last_score'] = np.nan
            ret_val['doc_last_mag'] = np.nan
        ret_val['doc_ent_num'] = len(file['entities'])
        
        # sentence score
        mags, scores = [], []
        for s in file['sentences']:
            mags.append(s['sentiment']['magnitude'])
            scores.append(s['sentiment']['score'])
        
        if len(scores)==0:
            ret_val['doc_score_sum'] = np.nan
            ret_val['doc_mag_sum'] = np.nan
            ret_val['doc_score_mena'] = np.nan
            ret_val['doc_mag_mean'] = np.nan
            ret_val['doc_score_max'] = np.nan
            ret_val['doc_mag_max'] = np.nan
            ret_val['doc_score_min'] = np.nan
            ret_val['doc_mag_min'] = np.nan
            ret_val['doc_score_std'] = np.nan
            ret_val['doc_mag_std'] = np.nan
        else:
            ret_val['doc_score_sum'] = np.sum(scores)
            ret_val['doc_mag_sum'] = np.sum(mags)
            ret_val['doc_score_mena'] = np.mean(scores)
            ret_val['doc_mag_mean'] = np.mean(mags)
            ret_val['doc_score_max'] = np.max(scores)
            ret_val['doc_mag_max'] = np.max(mags)
            ret_val['doc_score_min'] = np.min(scores)
            ret_val['doc_mag_min'] = np.min(mags)
            ret_val['doc_score_std'] = np.std(scores)
            ret_val['doc_mag_std'] = np.std(mags)

        # entity type
        ret_val['sentiment_entities'] = []
        ret_val['doc_ent_person_count'] = 0
        ret_val['doc_ent_location_count'] = 0
        ret_val['doc_ent_org_count'] = 0
        ret_val['doc_ent_event_count'] = 0
        ret_val['doc_ent_woa_count'] = 0
        ret_val['doc_ent_good_count'] = 0
        ret_val['doc_ent_other_count'] = 0
        key_mapper = {
            'PERSON':'doc_ent_person_count',
            'LOCATION':'doc_ent_location_count',
            'ORGANIZATION':'doc_ent_org_count',
            'EVENT':'doc_ent_event_count',
            'WORK_OF_ART':'doc_ent_woa_count',
            'CONSUMER_GOOD':'doc_ent_good_count',
            'OTHER':'doc_ent_other_count'
        }
        for e in file['entities']:
            ret_val['sentiment_entities'].append(e['name'])
            if e['type'] in key_mapper:
                ret_val[key_mapper[e['type']]]+=1
        
        ret_val['sentiment_entities'] = ' '.join(ret_val['sentiment_entities'])
        return ret_val
    
    def parse_metadata_file(self, file, img):
        """
        Parse metadata file. Output DF with metadata features.
        """
        file_keys = list(file.keys())
        if 'textAnnotations' in file_keys:
#             textanno = 1
            textblock_num = len(file['textAnnotations'])
            textlen = np.sum([len(text['description']) for text in file['textAnnotations']])
        else:
#             textanno = 0
            textblock_num = 0
            textlen = 0
        if 'faceAnnotations' in file_keys:
            faceanno = 1
        else:
            faceanno = 0
        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations']#[:len(file['labelAnnotations'])]
            if len(file_annots)==0:
                file_label_score_mean = np.nan
                file_label_score_max = np.nan
                file_label_score_min = np.nan
            else:
                temp = np.asarray([x['score'] for x in file_annots])
                file_label_score_mean = temp.mean()
                file_label_score_max = temp.max()
                file_label_score_min = temp.min()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_label_score_mean = np.nan
            file_label_score_max = np.nan
            file_label_score_min = np.nan
            file_top_desc = ['']
        
        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']
        if len(file_colors)==0:
            file_color_score = np.nan
            file_color_pixelfrac = np.nan
            color_red_mean = np.nan
            color_green_mean = np.nan
            color_blue_mean = np.nan
            color_red_std = np.nan
            color_green_std = np.nan
            color_blue_std = np.nan
        else:
            file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
            file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()
            file_color_red = np.asarray([x['color']['red'] if 'red' in x['color'] else 0 for x in file_colors])
            file_color_green = np.asarray([x['color']['green'] if 'green' in x['color'] else 0for x in file_colors])
            file_color_blue = np.asarray([x['color']['blue'] if 'blue' in x['color'] else 0 for x in file_colors])
            color_red_mean = file_color_red.mean()
            color_green_mean = file_color_green.mean()
            color_blue_mean = file_color_blue.mean()
            color_red_std = file_color_red.std()
            color_green_std = file_color_green.std()
            color_blue_std = file_color_blue.std()
        
        if len(file_crops)==0:
            file_crop_conf=np.nan
            file_crop_importance = np.nan
            file_crop_fraction_mean = np.nan
            file_crop_fraction_sum = np.nan
            file_crop_fraction_std = np.nan
            file_crop_num = 0
        else:
            file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
            file_crop_num = len(file_crops)
            if 'importanceFraction' in file_crops[0].keys():
                file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
            else:
                file_crop_importance = np.nan
            crop_areas = []
            image_area = img.shape[0]*img.shape[1]
            for crophint in file_crops:
                v_x, v_y = [], []
                for vertices in crophint['boundingPoly']['vertices']:
                    if 'x' not in vertices:
                        v_x.append(0)
                    else:
                        v_x.append(vertices['x'])
                    if 'y' not in vertices:
                        v_y.append(0)
                    else:
                        v_y.append(vertices['y'])
                crop_areas.append((max(v_x)-min(v_x))*(max(v_y)-min(v_y))/image_area)
            file_crop_fraction_mean = np.mean(crop_areas)
            file_crop_fraction_sum = np.sum(crop_areas)
            file_crop_fraction_std = np.std(crop_areas)

        df_metadata = {
            'label_score_mean': file_label_score_mean,
            'label_score_max': file_label_score_max,
            'label_score_min': file_label_score_min,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'color_red_mean':color_red_mean,
            'color_green_mean':color_green_mean,
            'color_blue_mean':color_blue_mean,
            'color_red_std':color_red_std,
            'color_green_std':color_green_std,
            'color_blue_std':color_blue_std,
#             'crop_area_mean':file_crop_fraction_mean,
            'crop_area_sum':file_crop_fraction_sum,
#             'crop_area_std':file_crop_fraction_std,
            'annots_top_desc': self.sentence_sep.join(file_top_desc),
            'img_aratio':img.shape[0]/img.shape[1],
#             'text_annotation':textanno,
            'text_len':textlen,
            'textblock_num':textblock_num,
            'face_annotation':faceanno
        }
        
        return df_metadata
    
# Helper function for parallel data processing:
def extract_additional_features(pet_id, mode='train'):
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = None

    dfs_metadata = []
    for ind in range(1,200):
        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)
        image_filename = '../input/petfinder-adoption-prediction/{}_images/{}-{}.jpg'.format(mode, pet_id, ind)
        try:
            image = cv2.imread(image_filename)
            metadata_file = pet_parser.open_metadata_file(metadata_filename)
            df_metadata = pet_parser.parse_metadata_file(metadata_file, image)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        except FileNotFoundError:
            break
    return [df_sentiment, dfs_metadata]
    
pet_parser = PetFinderParser()


# In[60]:


# Unique IDs from train and test:
train_pet_ids = train.PetID.unique()
test_pet_ids = test.PetID.unique()
n_jobs = 8

# Train set:
# Parallel processing of data:
dfs_train = Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

# Extract processed data and format them as DFs:
train_dicts_sentiment = [x[0] for x in dfs_train if x[0] is not None]
train_dfs_metadata = [x[1] for x in dfs_train if len([x[1]])>0]

train_dfs_sentiment = pd.DataFrame(train_dicts_sentiment)
train_dfs_metadata = list(itertools.chain.from_iterable(train_dfs_metadata))
train_dfs_metadata = pd.DataFrame(train_dfs_metadata)

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)


# In[61]:


# Test set:
# Parallel processing of data:
dfs_test = Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

# Extract processed data and format them as DFs:
test_dicts_sentiment = [x[0] for x in dfs_test if x[0] is not None]
test_dfs_metadata = [x[1] for x in dfs_test if len(x[1])>0]

test_dfs_sentiment = pd.DataFrame(test_dicts_sentiment)
test_dfs_metadata = list(itertools.chain.from_iterable(test_dfs_metadata))
test_dfs_metadata = pd.DataFrame(test_dfs_metadata)

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


# ### group extracted features by PetID:

# In[62]:


meta_df = pd.concat([train_dfs_metadata, test_dfs_metadata], sort=False).reset_index(drop=True)
senti_df = pd.concat([train_dfs_sentiment, test_dfs_sentiment], sort=False).reset_index(drop=True)


# In[63]:


# meta_df.to_pickle('./meta_df.pkl')
# senti_df.to_pickle('./senti_df.pkl')


# In[64]:


# meta_df = pd.read_pickle('./meta_df.pkl')
# senti_df = pd.read_pickle('./senti_df.pkl')


# In[65]:


metadata_desc = meta_df.groupby(['PetID'])['annots_top_desc'].unique().reset_index()
metadata_desc['meta_annots_top_desc'] = metadata_desc['annots_top_desc'].apply(lambda x: '; '.join(x))
metadata_desc.drop('annots_top_desc', axis=1, inplace=True)

possible_annots = set()
for i in range(len(meta_df)):
    possible_annots = possible_annots.union(set(meta_df['annots_top_desc'].iloc[i].split('; ')))
annot_mapper = {}
for idx, a in enumerate(possible_annots):
    annot_mapper[a] = str(idx)
metadata_desc['meta_desc'] = metadata_desc['meta_annots_top_desc'].apply(lambda x: ' '.join(annot_mapper[i] for i in x.split('; ')))


# In[66]:


# sentiment feature
senti_df['sentiment_entities'].fillna('', inplace=True)
senti_df['sentiment_entities'] = senti_df['sentiment_entities'].str.lower()
senti_df['sentiment_len'] = senti_df['sentiment_entities'].apply(lambda x:len(x))
senti_df['sentiment_word_len'] = senti_df['sentiment_entities'].apply(lambda x: len(x.replace(';',' ').split(' ')))
senti_df['sentiment_word_unique'] = senti_df['sentiment_entities'].apply(lambda x: len(set(x.replace(';',' ').split(' '))))

senti_df['doc_language'] = pd.factorize(senti_df['doc_language'])[0]


# In[67]:


# meta 需要agg
aggregates = {
    'color_blue_mean':['mean','std'],
    'color_blue_std':['mean'],
    'color_green_mean':['mean','std'], 
    'color_green_std':['mean'],
    'color_pixelfrac':['mean','std'],
    'color_red_mean':['mean','std'],
    'color_red_std':['mean'],
    'color_score':['mean','max'], 
#     'crop_area_mean':['mean','std','max'],
#     'crop_area_std':['mean'], 
    'crop_area_sum':['mean','std','min'], 
    'crop_conf':['mean','std','max'],
    'crop_importance':['mean','std'],
    'label_score_max':['mean','std','max'],
    'label_score_mean':['mean','max','std'],
    'label_score_min':['mean','max','std'],
    'img_aratio':['nunique','std','max','min'],
    'textblock_num':['mean','max'],
#     'text_len':['mean','max'],
    'face_annotation':['mean','nunique']
}

# Train
metadata_gr = meta_df.drop(['annots_top_desc'], axis=1)
for i in metadata_gr.columns:
    if 'PetID' not in i:
        metadata_gr[i] = metadata_gr[i].astype(float)
metadata_gr = metadata_gr.groupby(['PetID']).agg(aggregates)
metadata_gr.columns = pd.Index(['{}_{}_{}'.format('meta', c[0], c[1].upper()) for c in metadata_gr.columns.tolist()])
metadata_gr = metadata_gr.reset_index()


# In[68]:


meta_df = metadata_desc.merge(metadata_gr, on='PetID', how='left')


# In[69]:


# annotation feature
meta_df['meta_annots_top_desc'].fillna(' ', inplace=True)


# In[70]:


meta_df[meta_df['meta_textblock_num_MEAN']>0.8].head()


# ## feature engineering:

# In[71]:


feat_df = data_df[['PetID','Color1','Breed1','State','RescuerID','Name','Breed_full','Color_full','hard_interaction']]


# In[72]:


# color feature
agg = {
    'Fee':['mean','std','max'],
    'avg_fee':['mean','std','max'],
    'Breed1':['nunique'],
    #'Gender':['nunique'],
    'Age':['mean','std','max'], #,'min'
    'Quantity':['std'],#'mean',,'min','max'
    'PetID':['nunique']
}
feat = data_df.groupby('Color1').agg(agg)
feat.columns = pd.Index(['COLOR_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Color1', how='left')

agg = {
    'Fee':['mean','std','max'],
    'avg_fee':['mean','std','max'],
    'Breed_full':['nunique'],
    'Quantity':['sum'],
}
feat = data_df.groupby('Color_full').agg(agg)
feat.columns = pd.Index(['COLORfull_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Color_full', how='left')


# In[73]:


# Breed feature
agg = {
    'Color_full':['nunique'],
    'Breed2':['nunique'],
    'FurLength':['nunique'],
    'Fee':['mean','max'],#,'min'
    'avg_fee':['mean','std','max'],
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','max','sum'],#'min'
    'PetID':['nunique'],
    'FurLength':['mean'],
    'Health':['mean'],
    'MaturitySize':['mean','std','min','max'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean']
}
feat = data_df.groupby('Breed1').agg(agg)
feat.columns = pd.Index(['BREED1_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Breed1', how='left')

# Breed feature
agg = {
    'Color_full':['nunique'],
    'Fee':['mean','min','max'],
    'avg_fee':['mean','std','max'],
    'Quantity':['sum'],
    'PetID':['nunique']
}
feat = data_df.groupby('Breed_full').agg(agg)
feat.columns = pd.Index(['BREEDfull_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='Breed_full', how='left')


# In[74]:


# State feature
agg = {
    'Color_full':['nunique'],
    'Breed_full':['nunique'],
    'PetID':['nunique'],
    'RescuerID':['nunique'],
    'Fee':['mean','max'],
    'avg_fee':['mean','std','max'],
    'Age':['mean','std','max'],
    'Quantity':['mean','std','max'],#,'min','sum'
    'FurLength':['mean','std'],
    'Health':['mean'],
    'MaturitySize':['mean','std'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean'],
    'VideoAmt':['mean','std'],
    'PhotoAmt':['mean','std'],
    'avg_photo':['mean','std']
}
feat = data_df.groupby('State').agg(agg)
feat.columns = pd.Index(['STATE_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='State', how='left')


# In[75]:


# multiple agg
agg = {
    'Fee':['mean','min','max'],
    'avg_fee':['mean','min','max'],
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','sum'],
    'PetID':['nunique'],
    'FurLength':['mean'],
    'Health':['mean'],
    'MaturitySize':['mean','std'],
    'Vaccinated':['mean'],
    'Dewormed':['mean'],
    'Sterilized':['mean']
}
feat = data_df.groupby(['State','Breed1','Color1']).agg(agg)
feat.columns = pd.Index(['MULTI_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on=['State','Breed1','Color1'], how='left')

agg = {
    'Fee':['mean','min','max'],
    'avg_fee':['mean','min','max'],
    'Age':['mean','std','min','max'],
    'Quantity':['mean','std','sum'],
    'PetID':['nunique'],
}
feat = data_df.groupby(['State','Breed_full','Color_full']).agg(agg)
feat.columns = pd.Index(['MULTI2_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on=['State','Breed_full','Color_full'], how='left')


# In[76]:


# name feature
feat = data_df.groupby('Name')['PetID'].agg({'name_count':'nunique'}).reset_index()
feat_df = feat_df.merge(feat, on='Name', how='left')


# In[77]:


# Count RescuerID occurrences:
agg = {
#     'avg_fee':['mean','std'], # hurt
#     'Age':['mean','std'], #,'min','max' hurt
#     'Quantity':['mean','std','sum'],
    'PetID':['nunique'],
    'Breed_full':['nunique'],
    'VideoAmt':['mean','std'],
    'PhotoAmt':['mean','std'],
    'avg_photo':['mean','std'],
    'Sterilized':['mean'],
    'Dewormed':['mean'],
    'Vaccinated':['mean']
#     'description_word_len':['mean','std'] # hurt
}
rescuer_count = data_df.groupby(['RescuerID']).agg(agg)
rescuer_count.columns = pd.Index(['RESCUER_' + e[0] + "_" + e[1].upper() for e in rescuer_count.columns.tolist()])
rescuer_count.reset_index(inplace=True)
feat_df = feat_df.merge(rescuer_count, how='left', on='RescuerID')


# In[78]:


# State feature
agg = {
    'Fee':['mean','min','max'],
    'avg_fee':['mean','std','max']
}
feat = data_df.groupby('hard_interaction').agg(agg)
feat.columns = pd.Index(['INTERACTION_' + e[0] + "_" + e[1].upper() for e in feat.columns.tolist()])
feat.reset_index(inplace=True)
feat_df = feat_df.merge(feat, on='hard_interaction', how='left')


# In[79]:


feat_df.drop(['Color1','Breed1','State','RescuerID','Name','Breed_full','Color_full','hard_interaction'], axis=1, inplace=True)


# ## text features

# In[80]:


X_text = data_df[['PetID','Description']].merge(data_df[['PetID','Chinese_desc']], on='PetID', how='left')
X_text = X_text.merge(senti_df[['PetID','sentiment_entities']], on='PetID', how='left')
X_text = X_text.merge(metadata_desc[['PetID','meta_annots_top_desc','meta_desc']], on='PetID', how='left')
text_columns = ['Description','Chinese_desc','sentiment_entities','meta_annots_top_desc','meta_desc']
print(text_columns)
X_text['meta_annots_top_desc'].fillna(' ',inplace=True)
X_text['meta_desc'].fillna(' ',inplace=True)
X_text['sentiment_entities'].fillna(' ',inplace=True)


# In[81]:


text_features = []
ngram_ranges = [(1,3),(1,3),(1,1),(1,3),(1,1)]
n_components = [80, 24, 10, 32,16]

# Generate text features:
for idx, i in enumerate(text_columns):
    # Initialize decomposition methods:
    print('generating features from: {}'.format(i))
    svd_ = TruncatedSVD(
        n_components=n_components[idx], random_state=1337)
    
    tfidf_col = TfidfVectorizer(ngram_range = ngram_ranges[idx], stop_words = 'english', #lowercase=False,
                                tokenizer=custom_tokenizer,
                               strip_accents='unicode').fit_transform(X_text.loc[:, i].values)
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
    
    text_features.append(svd_col)

# Combine all extracted features:
# text_features = pd.concat(text_features, axis=1)


# In[82]:


docs = [custom_tokenizer(x) for x in data_df['Description'].values]
dictionary = corpora.Dictionary(docs)
name = 'Description'
corpus = [dictionary.doc2bow(text) for text in docs]
lda = LdaMulticore(corpus, id2word=dictionary, num_topics=20, random_state = 999)
docres = [dict(lda[doc_bow]) for doc_bow in corpus]
doc_df = pd.DataFrame(docres,dtype=np.float16).fillna(0.001)
doc_df.columns = ['%s_lda_%d'%(name,x) for x in range(20)]


# In[83]:


text_features.append(doc_df)


# In[84]:


X_text_char = data_df[['PetID','Name','BreedName_full']].merge(metadata_desc[['PetID','meta_annots_top_desc']], on='PetID', how='left')
X_text_char['meta_annots_top_desc'].fillna(' ',inplace=True)
for c in ['Name','BreedName_full','meta_annots_top_desc']:
    svd_ = TruncatedSVD(
        n_components=16, random_state=1337)

    tfidf_col = TfidfVectorizer(ngram_range = (1,5), analyzer='char',#lowercase=False,
                               strip_accents='unicode').fit_transform(X_text_char[c])
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_CHAR_{}_'.format(c))

    text_features.append(svd_col)
del X_text_char
gc.collect()


# In[85]:


text_features = pd.concat(text_features, axis=1)


# In[86]:


text_features['PetID'] = X_text['PetID'].values


# ## LDA feature

# In[87]:


def get_lda_feature(data, target, source, name, n_topic = 10, random_state = 999):
    retval = pd.DataFrame(data[[target, source]])
    x = retval.groupby(target, as_index=False)[source].agg({'list':(lambda x: list(x))})
    x['sentence'] = x['list'].apply(lambda x: list(map(str,x)))
    docs = x['sentence'].tolist() #.apply(lambda x:x.split()).tolist()
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=n_topic, random_state = random_state)
    docres = [dict(lda[doc_bow]) for doc_bow in corpus]
    doc_df = pd.DataFrame(docres,dtype=np.float16).fillna(0.001)
    doc_df.columns = ['%s_lda_%d'%(name,x) for x in range(n_topic)]
    doc_df[target] = x[target]
    return doc_df


# In[88]:


breed_color_lda = get_lda_feature(data_df, 'Breed1', 'Color_full', 'breed', n_topic=5)
breed_breed_lda = get_lda_feature(data_df, 'Breed1', 'Breed2', 'breed_breed', n_topic=5)
state_breed_lda = get_lda_feature(data_df, 'State', 'Breed_full', 'State_breed', n_topic=5)
state_color_lda = get_lda_feature(data_df, 'State', 'Color_full', 'State_color',n_topic=5)
# rescuer_breed_lda = get_lda_feature(data_df, 'RescuerID', 'Breed_full', 'rescuer_breed',n_topic=30)
# rescuer_color_lda = get_lda_feature(data_df, 'RescuerID', 'Color_full', 'rescuer_color',n_topic=10)


# ## merge all features

# In[89]:


# Train merges:
data_df_proc = data_df.copy()
data_df_proc = data_df_proc.merge(senti_df, how='left', on='PetID')
data_df_proc = data_df_proc.merge(meta_df, how='left', on='PetID')
data_df_proc = data_df_proc.merge(feat_df, how='left', on='PetID')

data_df_proc = data_df_proc.merge(dnn_resnet50_features[['PetID','resnet_%d'%IMG_FEATURE_DIM_NN]], how='left', on='PetID')#
data_df_proc = data_df_proc.merge(dnn_dense_features[['PetID','dense_%d'%IMG_FEATURE_DIM_NN]], how='left', on='PetID')#
data_df_proc = data_df_proc.merge(dnn_resnet34_features[['PetID','res34_%d'%IMG_FEATURE_DIM_NN2]], how='left', on='PetID')#
data_df_proc = data_df_proc.merge(dnn_dense121_features[['PetID','dense121_%d'%IMG_FEATURE_DIM_NN]], how='left', on='PetID')#

data_df_proc = data_df_proc.merge(svd_resnet34_features, how='left', on='PetID')
data_df_proc = data_df_proc.merge(svd_resnet50_features, how='left', on='PetID')
data_df_proc = data_df_proc.merge(svd_dense_features, how='left', on='PetID')
data_df_proc = data_df_proc.merge(svd_dense121_features, how='left', on='PetID')

data_df_proc = data_df_proc.merge(cluster_img_features, how='left', on='PetID')

# lda features
data_df_proc = data_df_proc.merge(breed_color_lda, how='left', on='Breed1')
data_df_proc = data_df_proc.merge(breed_breed_lda, how='left', on='Breed1')
data_df_proc = data_df_proc.merge(state_breed_lda, how='left', on='State')
data_df_proc = data_df_proc.merge(state_color_lda, how='left', on='State')

# Concatenate with main DF:
data_df_proc = data_df_proc.merge(text_features, how='left', on='PetID')


print(data_df_proc.shape)
assert data_df_proc.shape[0] == data_df.shape[0]


# ### train/test split:

# In[90]:


# Split into train and test again:
X_train = data_df_proc.iloc[0:train_len]
X_test = data_df_proc.iloc[train_len:]
# X_train_dummy = data_df_proc_dummy.iloc[0:train_len]
# X_test_dummy = data_df_proc_dummy.iloc[train_len:]

# Remove missing target column from test:
X_test = X_test.drop(['AdoptionSpeed'], axis=1)

print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]

# Check if columns between the two DFs are the same:
train_cols = X_train.columns.tolist()
train_cols.remove('AdoptionSpeed')

test_cols = X_test.columns.tolist()

assert np.all(train_cols == test_cols)


# ### model training:

# In[91]:


# Additional parameters:
early_stop = 300
verbose_eval = 100
num_rounds = 10000

to_drop_columns = ['PetID', 'Name', 'RescuerID', 'AdoptionSpeed', 'target2', 
                   'main_breed_Type', 'main_breed_BreedName', 'second_breed_Type', 'second_breed_BreedName',
                   'Description', 'sentiment_entities', 'meta_annots_top_desc','meta_desc',
                   'Chinese_desc', 'English_desc','BreedName_full','Breed1Name','Breed2Name']


# ## NN model

# In[92]:


torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
torch.backends.cudnn.deterministic = True


# In[93]:


fm_cols = ['Type','age_in_year','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
           'FurLength','Vaccinated','Dewormed','Sterilized','Health','State','Breed_full',
           'Color_full', 'hard_interaction','img_CLUSTER_0']
fm_data = data_df_proc[fm_cols]
fm_values = []
for c in fm_cols:
    fm_data.loc[:,c] = fm_data[c].fillna(0)
    fm_data.loc[:,c] = c+'_'+fm_data[c].astype(str)
    fm_values+=fm_data[c].unique().tolist()


# In[94]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit(fm_values)
for c in fm_cols:
    fm_data.loc[:,c] = lbe.transform(fm_data[c])


# In[95]:


numerical_cols = [x for x in X_train.columns if x not in to_drop_columns+fm_cols+svd_dense_features.columns.tolist()]


# In[96]:


len(numerical_cols)


# In[97]:


numerical_feats = []
for c in numerical_cols:
    numerical_feats.append(data_df_proc[c].fillna(0))
    
# for c in range(1920):
#     numerical_feats.append(raw_img_features['resnet50_%d'%c].fillna(0))

numerical_feats = np.vstack(numerical_feats).T
# numerical_feats = stdscaler.fit_transform(numerical_feats)


# In[98]:


numerical_feats.shape


# In[99]:


MAX_LEN = 400
class PetDesDataset(Dataset):
    def __init__(self, sentences, pos, fm_data, numerical_feat,
                 mode='train', target=None):
        super(PetDesDataset, self).__init__()
        self.data = sentences
        self.pos = pos
        self.target = target
        self.mode = mode
        self.fm_data = fm_data
        self.fm_dim = fm_data.shape[1]
        self.numerical_feat = numerical_feat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index not in range(0, self.__len__()):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        sentence_len = min(MAX_LEN, len(self.data[index]))
        sentence = torch.tensor(self.data[index][:sentence_len])
        fm_data = self.fm_data[index,:]
        pos = torch.tensor(self.pos[index][:sentence_len])

        if self.mode != 'test':  # , pos, tag
            return sentence, pos, sentence_len, fm_data, self.numerical_feat[index], self.target[index]  # , clf_label
        else:
            return sentence, pos, sentence_len, fm_data, self.numerical_feat[index]
        
def nn_collate(batch):
    has_label = len(batch[0]) == 6
    if has_label:
        sentences, poses, lengths, fm_data, numerical_feats, label = zip(*batch)
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).type(torch.LongTensor)
        poses = nn.utils.rnn.pad_sequence(poses, batch_first=True).type(torch.LongTensor)
        lengths = torch.LongTensor(lengths)
        fm_data = torch.LongTensor(fm_data)
        numerical_feats = torch.FloatTensor(numerical_feats)
        label = torch.FloatTensor(label)
        return sentences, poses, lengths, fm_data, numerical_feats, label
    else:
        sentences, poses, lengths, fm_data, numerical_feats = zip(*batch)
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).type(torch.LongTensor)
        poses = nn.utils.rnn.pad_sequence(poses, batch_first=True).type(torch.LongTensor)
        lengths = torch.LongTensor(lengths)
        fm_data = torch.LongTensor(fm_data)
        numerical_feats = torch.FloatTensor(numerical_feats)
        return sentences, poses, lengths, fm_data, numerical_feats


# In[100]:


def get_mask(sequences_batch, sequences_lengths, cpu=False):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if cpu:
        return mask
    else:
        return mask.cuda()
class Attention(nn.Module):
    def __init__(self, feature_dim, bias=True, head_num=1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.head_num = head_num
        weight = torch.zeros(feature_dim, self.head_num)
        bias = torch.zeros((1, 1, self.head_num))
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(bias)

    def forward(self, x, mask=None):
        batch_size, step_dim, feature_dim = x.size()
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),  # B*L*H
            self.weight  # B*H*1
        ).view(-1, step_dim, self.head_num)  # B*L*head
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        if mask is not None:
            eij = eij * mask - 99999.9 * (1 - mask)
        a = torch.softmax(eij, dim=1)

        weighted_input = torch.bmm(x.permute((0,2,1)),
                                   a).view(batch_size, -1)
        return weighted_input


# In[101]:


embed_size = 300
class FM(nn.Module):

    def __init__(self, max_features, feat_len, embed_size):
        super(FM, self).__init__()
        self.bias_emb = nn.Embedding(max_features, 1)
        self.fm_emb = nn.Embedding(max_features, embed_size)
        self.feat_len = feat_len
        self.embed_size = embed_size

    def forward(self, x):
        bias = self.bias_emb(x)
        bias = torch.sum(bias,1) # N * 1

        # second order term
        # square of sum
        emb = self.fm_emb(x)
        sum_feature_emb = torch.sum(emb, 1) # N * k
        square_sum_feature_emb = sum_feature_emb*sum_feature_emb

        # sum of square
        square_feature_emb = emb * emb
        sum_square_feature_emb = torch.sum(square_feature_emb, 1) # N * k

        second_order = 0.5*(square_sum_feature_emb-sum_square_feature_emb) # N *k
        return bias+second_order, emb.view(-1, self.feat_len*self.embed_size)


# In[102]:


class FmNlpModel(nn.Module):
    def turn_on_embedding(self):
        self.embedding.weight.requires_grad = True

    def __init__(self, hidden_size=64, init_embedding=None, head_num=3,
                 fm_embed_size=8, fm_feat_len=10, fm_max_feature = 300, numerical_dim = 300,
                 nb_word = 40000, nb_pos = 200, pos_emb_size = 10):
        super(FmNlpModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(nb_word, 300, padding_idx=0)
        self.pos_embedding = nn.Embedding(nb_pos+100, pos_emb_size, padding_idx=0)
        
        if init_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.embedding.weight.requires_grad = False
        
        self.fm = FM(fm_max_feature, fm_feat_len, fm_embed_size)

        self.dropout = nn.Dropout(0.1)
        self.attention_gru = Attention(feature_dim=self.hidden_size * 2, head_num=head_num)
        self.gru = nn.GRU(embed_size+pos_emb_size, hidden_size, bidirectional=True, batch_first=True) #
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(numerical_dim),
            nn.Dropout(0.1),
            nn.Linear(numerical_dim, 256),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
        )
        
        self.rnn_dnn = nn.Sequential(
            nn.BatchNorm1d(fm_embed_size+fm_feat_len*fm_embed_size +2*head_num * hidden_size+128), #
            nn.Dropout(0.3),
            nn.Linear(fm_embed_size+fm_feat_len*fm_embed_size+2*head_num * hidden_size+128, 32),
            nn.ELU(inplace=True),
        )
        self.logit = nn.Sequential(
            nn.Linear(32,1)
        )
#         self.apply(_init_esim_weights)

    def forward(self, x, pos_x, len_x, fm_x, numerical_x):
        
        fm_result, fm_embed = self.fm(fm_x)
        
        sentence_mask = get_mask(x, len_x)
        x = x * sentence_mask.long()
        sentence_mask = torch.unsqueeze(sentence_mask, -1)

        h_embedding = self.embedding(x)
        h_pos_embedding = self.pos_embedding(pos_x)
        h_embedding = torch.cat([h_embedding, h_pos_embedding],2)
        
        h_embedding = self.dropout(h_embedding)
        
        sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
        # 排序前的顺序是对下标再排一次序
        _, desorted_indices = torch.sort(indices, descending=False)
        h_embedding = h_embedding[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(h_embedding, sorted_seq_lengths, batch_first=True)
        
        h_gru, _ = self.gru(packed_inputs)
        h_gru2, _ = self.gru2(h_gru)  # sentence_mask.expand_as(h_lstm)
        
        h_gru2, _ = nn.utils.rnn.pad_packed_sequence(h_gru2, batch_first=True)
        h_gru2 = h_gru2[desorted_indices]
        att_pool_gru = self.attention_gru(h_gru2, sentence_mask)
        
        numerical_x = self.dnn(numerical_x)

        x = torch.cat([att_pool_gru,fm_result,fm_embed,numerical_x],1) # 
        feat = self.rnn_dnn(x)
        out = self.logit(feat)

        return out, feat


# In[103]:


X_train_numerical = numerical_feats[0:len(train)]
X_test_numerical = numerical_feats[len(train):]

X_train_seq = pd.Series(eng_sequences[0:len(train)])
X_test_seq = pd.Series(eng_sequences[len(train):])

X_train_pos_seq = pd.Series(pos_sequences[0:len(train)])
X_test_pos_seq = pd.Series(pos_sequences[len(train):])

X_train_fm = fm_data.iloc[0:len(train)].values
X_test_fm = fm_data.iloc[len(train):].values

Y_train = data_df.iloc[0:len(train)]['AdoptionSpeed'].values


# In[104]:


train_epochs = 6
loss_fn = torch.nn.MSELoss().cuda()
oof_train_nlp = np.zeros((X_train.shape[0], 32+1))
oof_test_nlp = []

test_set = PetDesDataset(X_test_seq.tolist(), X_test_pos_seq.tolist(), X_test_fm, X_test_numerical, mode='test')
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=1, pin_memory=True,
                                collate_fn=nn_collate)
qwks = []
rmses = []

for n_fold, (train_idx, valid_idx) in enumerate(split_index): 
        
    print('fold:', n_fold)
    hist = histogram(Y_train[train_idx].astype(int), 
                     int(np.min(X_train['AdoptionSpeed'])), 
                     int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    training_set = PetDesDataset(X_train_seq[train_idx].tolist(), 
                                 X_train_pos_seq[train_idx].tolist(),
                                 X_train_fm[train_idx], 
                                 X_train_numerical[train_idx], target = Y_train[train_idx])
    
    validation_set = PetDesDataset(X_train_seq[valid_idx].tolist(), 
                                   X_train_pos_seq[valid_idx].tolist(),
                                   X_train_fm[valid_idx], 
                                  X_train_numerical[valid_idx],target = Y_train[valid_idx])
    
    training_loader = DataLoader(training_set, batch_size=512, shuffle=True, num_workers=1,
                                collate_fn=nn_collate)
    validation_loader = DataLoader(validation_set, batch_size=512, shuffle=False, num_workers=1,
                                collate_fn=nn_collate)
    
    model = FmNlpModel(hidden_size=48, init_embedding=embedding_matrix, head_num=10, 
                      fm_embed_size=10, fm_feat_len=X_train_fm.shape[1], fm_max_feature=len(fm_values),
                      numerical_dim=X_train_numerical.shape[1],
                      nb_word=nb_words, nb_pos=nb_pos, pos_emb_size=10)
    model.cuda()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.0001)
    
    iteration = 0
    min_val_loss = 100
    since = time.time()
    
    for epoch in range(train_epochs):       
        scheduler.step()
        model.train()
        for sentences, poses, lengths, x_fm, x_numerical, labels in training_loader:
            iteration += 1
            sentences = sentences.cuda()
            poses = poses.cuda()
            lengths = lengths.cuda()
            x_fm = x_fm.cuda()
            x_numerical = x_numerical.cuda()
            labels = labels.type(torch.FloatTensor).cuda().view(-1, 1)

            pred,_ = model(sentences, poses, lengths, x_fm, x_numerical)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_predicts = []
        val_feats = []
        with torch.no_grad():
            for sentences, poses, lengths, x_fm, x_numerical, labels in validation_loader:
                sentences = sentences.cuda()
                poses = poses.cuda()
                lengths = lengths.cuda()
                x_fm = x_fm.cuda()
                x_numerical = x_numerical.cuda()
                labels = labels.type(torch.FloatTensor).cuda()#.view(-1, 1)
                v_pred, v_feat = model(sentences, poses, lengths, x_fm, x_numerical)
                val_predicts.append(v_pred.cpu().numpy())
                val_feats.append(v_feat.cpu().numpy())

        val_predicts = np.concatenate(val_predicts)
        val_feats = np.vstack(val_feats)
        val_loss = rmse(Y_train[valid_idx], val_predicts)
        if val_loss<min_val_loss:
            min_val_loss = val_loss
            oof_train_nlp[valid_idx,:] = np.hstack([val_feats, val_predicts])
            test_feats = []
            test_preds = []
            with torch.no_grad():
                for sentences, poses, lengths, x_fm, x_numerical in test_loader:
                    sentences = sentences.cuda()
                    poses = poses.cuda()
                    lengths = lengths.cuda()
                    x_fm = x_fm.cuda()
                    x_numerical = x_numerical.cuda()
                    v_pred, feat = model(sentences, poses, lengths, x_fm, x_numerical)
                    test_preds.append(v_pred.cpu().numpy())
                    test_feats.append(feat.cpu().numpy())
            test_feats = np.hstack([np.vstack(test_feats), np.concatenate(test_preds)])
            pred_test_y_k = getTestScore2(val_predicts.flatten(), tr_cdf)
            qwk = quadratic_weighted_kappa(Y_train[valid_idx], pred_test_y_k)
            print(epoch, "val loss:", val_loss, "val QWK_2 = ", qwk, "elapsed time:", time.time()-since)
    oof_test_nlp.append(test_feats)
    del model
    del training_set
    del validation_set 
    del sentences
    del lengths
    del x_fm
    del x_numerical
    del poses
    gc.collect()
    torch.cuda.empty_cache()
    
    qwks.append(qwk)
    rmses.append(min_val_loss)


# In[105]:


print('overall rmse: %.5f'%rmse(oof_train_nlp[:,-1], X_train['AdoptionSpeed']))
print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))


# In[106]:


# del model
# del training_set
# del validation_set 
# del sentences
# del lengths
# del x_fm
# del x_numerical
# gc.collect()
# torch.cuda.empty_cache()


# In[107]:


oof_test_nlp = np.mean(oof_test_nlp, axis=0)


# ## LGB Model

# In[108]:


features = [x for x in X_train.columns if x not in to_drop_columns+svd_resnet50_features.columns.tolist()]


# In[109]:


print(len(features))


# In[110]:


def run_lgb(X_train, X_test, features, split_index):
    params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'learning_rate': 0.02,
          'subsample': 0.85,
          'feature_fraction': 0.7,
          'lambda_l1':0.01,
          'verbosity': -1,
         }
    oof_train_lgb = np.zeros((X_train.shape[0]))
    oof_test_lgb = []
    qwks = []
    rmses = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(split_index):
        since = time.time()
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]

        y_tr = X_tr['AdoptionSpeed'].values    
        y_val = X_val['AdoptionSpeed'].values

        d_train = lgb.Dataset(X_tr[features], label=y_tr,
    #                          categorical_feature=['Breed1','Color1','Breed2','State','Breed_full','Color_full']
                             )
        d_valid = lgb.Dataset(X_val[features], label=y_val, reference=d_train)
        watchlist = [d_valid]

        print('training LGB:')
        lgb_model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=500,
                          early_stopping_rounds=100,
                         )

        val_pred = lgb_model.predict(X_val[features])
        test_pred = lgb_model.predict(X_test[features])
        train_pred = lgb_model.predict(X_tr[features])

        oof_train_lgb[valid_idx] = val_pred
        oof_test_lgb.append(test_pred)

        hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
                         int(np.min(X_train['AdoptionSpeed'])), 
                         int(np.max(X_train['AdoptionSpeed'])))
        tr_cdf = get_cdf(hist)
        _, cutoff = getScore(train_pred, tr_cdf, True)

        pred_test_y_k = getTestScore2(val_pred, tr_cdf)
        qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
        qwks.append(qwk)
        rmses.append(rmse(X_val['AdoptionSpeed'].values, val_pred))
        print("QWK_2 = ", qwk,'elapsed time:',time.time()-since)
    
    oof_test_lgb = np.mean(oof_test_lgb, axis=0)
    print('overall rmse: %.5f'%rmse(oof_train_lgb, X_train['AdoptionSpeed']))
    print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
    print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))
    return oof_train_lgb, oof_test_lgb


# In[111]:


base_cols = [x for x in senti_df.columns.tolist()+meta_df.columns.tolist()+feat_df.columns.tolist() \
             +cluster_img_features.columns.tolist() \
             +breed_color_lda.columns.tolist()+breed_breed_lda.columns.tolist() \
             +state_breed_lda.columns.tolist() \
             +state_color_lda.columns.tolist()+text_features.columns.tolist() if x not in to_drop_columns]


# In[112]:


cols34 = base_cols+['res34_%d'%IMG_FEATURE_DIM_NN2]+['resnet34_SVD_%d'%i for i in range(32)]
lgb34_oof, lgb34_test = run_lgb(X_train, X_test, cols34, split_index)


# In[113]:


cols50 = base_cols+['resnet_%d'%IMG_FEATURE_DIM_NN]+['resnet50_SVD_%d'%i for i in range(32)]
lgb50_oof, lgb50_test = run_lgb(X_train, X_test, cols50, split_index)


# In[114]:


cols121 = base_cols+['dense121_%d'%IMG_FEATURE_DIM_NN]+['dense121_SVD_%d'%i for i in range(32)]
lgb121_oof, lgb121_test = run_lgb(X_train, X_test, cols121, split_index)


# In[115]:


cols201 = base_cols+['dense_%d'%IMG_FEATURE_DIM_NN]+['dense_SVD_%d'%i for i in range(32)]
lgb201_oof, lgb201_test = run_lgb(X_train, X_test, cols201, split_index)


# ## Catboost

# In[116]:


# from catboost import CatBoostRegressor, Pool


# In[117]:


# features = [x for x in X_train.columns if x not in to_drop_columns+svd_dense121_features.columns.tolist()]


# In[118]:


# cat_index = []
# for idx, c in enumerate(features):
#     if c in ['Type','Breed1','Breed2','Gender','Color1','Color2','Color3','State','Breed_full',
#            'Color_full', 'hard_interaction','img_CLUSTER_0']:
#         cat_index.append(idx)


# In[119]:


# oof_train_cat = np.zeros((X_train.shape[0]))
# oof_test_cat = []
# qwks = []
# rmses = []

# for n_fold, (train_idx, valid_idx) in enumerate(split_index):
#     since = time.time()
#     X_tr = X_train.iloc[train_idx]
#     X_val = X_train.iloc[valid_idx]

#     y_tr = X_tr['AdoptionSpeed'].values    #apply(target_transform).
#     y_val = X_val['AdoptionSpeed'].values #apply(target_transform)
        
    
#     eval_dataset = Pool(X_val[features].values,
#                     y_val,
#                    cat_index)
#     print('training Catboost:')
#     model = CatBoostRegressor(learning_rate=0.01,  depth=8, task_type = "GPU", l2_leaf_reg=1)
#     model.fit(X_tr[features].values,
#               y_tr,
#               eval_set=eval_dataset,
#               cat_features= cat_index,
#               verbose=False)
    
#     val_pred = model.predict(eval_dataset)
#     test_pred = model.predict(X_test[features])
    
#     oof_train_cat[valid_idx] = val_pred
#     oof_test_cat.append(test_pred)
               
#     hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
#                      int(np.min(X_train['AdoptionSpeed'])), 
#                      int(np.max(X_train['AdoptionSpeed'])))
#     tr_cdf = get_cdf(hist)
    
#     pred_test_y_k = getTestScore2(val_pred, tr_cdf)
#     qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
#     qwks.append(qwk)
#     rmses.append(rmse(X_val['AdoptionSpeed'].values, val_pred))
#     print('rmse=',rmses[-1],"QWK_2 = ", qwk,'elapsed time:',time.time()-since)


# In[120]:


# print('overall rmse: %.5f'%rmse(oof_train_cat, X_train['AdoptionSpeed']))
# print('mean rmse =', np.mean(rmses), 'rmse std =', np.std(rmses))
# print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))


# ## xgb model

# In[121]:


def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kf = StratifiedKFold(folds, shuffle=True, random_state=1)
            kfold = kf.split(train_data, train_data[target_col].values) 
            parts = []
            for tr_in, val_ind in kfold:
                # divide data
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + 
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded[:train_data.shape[0]], 
            all_encoded[train_data.shape[0]:])


# In[122]:


features = [x for x in X_train.columns if x not in to_drop_columns+svd_dense_features.columns.tolist()]


# In[123]:


xgb_features = features


# In[124]:


len(xgb_features)


# In[125]:


params = {
        'objective': 'reg:linear', #huber
        'eval_metric':'rmse',
        'eta': 0.01,
        'tree_method':'gpu_hist',
        'max_depth': 9,  
        'subsample': 0.85,  
        'colsample_bytree': 0.7,     
        'alpha': 0.01,  
    } 

oof_train_xgb = np.zeros((X_train.shape[0]))
oof_test_xgb = []
qwks = []

i = 0
test_set = xgb.DMatrix(X_test[xgb_features])

for n_fold, (train_idx, valid_idx) in enumerate(split_index):  
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[valid_idx]
    
    y_tr = X_tr['AdoptionSpeed'].values    
    y_val = X_val['AdoptionSpeed'].values
        
    d_train = xgb.DMatrix(X_tr[xgb_features], y_tr)
    d_valid = xgb.DMatrix(X_val[xgb_features], y_val)
    watchlist = [d_valid]
    since = time.time()
    print('training XGB:')
    model = xgb.train(params, d_train, num_boost_round = 10000, evals=[(d_valid,'val')],
                     early_stopping_rounds=100, #feval=xgb_eval_kappa,
                     verbose_eval=500)
    
    val_pred = model.predict(d_valid)
    test_pred = model.predict(test_set)
    
    oof_train_xgb[valid_idx] = val_pred
    oof_test_xgb.append(test_pred)
    
#     hist = histogram(X_tr['AdoptionSpeed'].astype(int), 
#                      int(np.min(X_train['AdoptionSpeed'])), 
#                      int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    pred_test_y_k = getTestScore2(val_pred, tr_cdf)
    qwk = quadratic_weighted_kappa(X_val['AdoptionSpeed'].values, pred_test_y_k)
    qwks.append(qwk)
    print("QWK_2 = ", qwk,'elapsed time:',time.time()-since)


# In[126]:


print('overall rmse: %.5f'%rmse(oof_train_xgb, X_train['AdoptionSpeed']))
print('mean QWK =', np.mean(qwks), 'std QWK =', np.std(qwks))


# ## voting

# In[127]:


hist = histogram(X_train['AdoptionSpeed'].astype(int), 
                 int(np.min(X_train['AdoptionSpeed'])), 
                 int(np.max(X_train['AdoptionSpeed'])))
tr_cdf = get_cdf(hist)


# In[128]:


# valid
final_pred = pd.DataFrame()
final_pred['lgb34'] = getTestScore2(lgb34_oof, tr_cdf) 
final_pred['lgb50'] = getTestScore2(lgb50_oof, tr_cdf)
final_pred['lgb121'] = getTestScore2(lgb121_oof, tr_cdf)
final_pred['lgb201'] = getTestScore2(lgb201_oof, tr_cdf)
final_pred['xgb'] = getTestScore2(oof_train_xgb, tr_cdf)
final_pred['nlp_pred'] = getTestScore2(oof_train_nlp[:,-1], tr_cdf)


# In[129]:


final_pred['pred'] = final_pred[['lgb34','lgb50','lgb121','lgb201','xgb','nlp_pred']].mode(axis=1)[0]


# In[130]:


qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, final_pred['pred'])
print(qwk)


# In[131]:


final_pred = pd.DataFrame()
final_pred['lgb34'] = getTestScore2(lgb34_test, tr_cdf) 
final_pred['lgb50'] = getTestScore2(lgb50_test, tr_cdf)
final_pred['lgb121'] = getTestScore2(lgb121_test, tr_cdf)
final_pred['lgb201'] = getTestScore2(lgb201_test, tr_cdf)
final_pred['xgb'] = getTestScore2(np.mean(oof_test_xgb,axis=0), tr_cdf)
final_pred['nlp_pred'] = getTestScore2(oof_test_nlp[:,-1], tr_cdf)
final_pred['pred'] = final_pred[['lgb34','lgb50','lgb121','lgb201','xgb','nlp_pred']].mode(axis=1)[0]


# ## stacking

# In[132]:


np.corrcoef([lgb34_test, 
             lgb50_test,
             lgb121_test,
             lgb201_test,
             np.mean(oof_test_xgb,axis=0),
             oof_test_nlp[:,-1]])


# In[133]:


X_train_stacking = np.vstack([lgb34_oof, 
                              lgb50_oof,
                              lgb121_oof, 
                              lgb201_oof, 
                              oof_train_xgb,
                              oof_train_nlp[:,-1]]).T
X_test_stacking = np.vstack([lgb34_test,
                             lgb50_test,
                             lgb121_test,
                             lgb201_test,
                             np.mean(oof_test_xgb,axis=0),
                             oof_test_nlp[:,-1]]).T

stacking_train = np.zeros((X_train.shape[0]))
stacking_test = []
rmses, qwks = [], []

for n_fold, (train_idx, valid_idx) in enumerate(split_index):
    
    X_tr = X_train_stacking[train_idx]
    X_val = X_train_stacking[valid_idx]
    
    y_tr = X_train.iloc[train_idx]['AdoptionSpeed'].values    
    y_val = X_train.iloc[valid_idx]['AdoptionSpeed'].values
        
    since = time.time()
    
    print('training Ridge:')
    model = Ridge(alpha=1)
    model.fit(X_tr, y_tr)
    print(model.coef_)
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_stacking)
    
    stacking_train[valid_idx] = val_pred
    stacking_test.append(test_pred)
    loss = rmse(Y_train[valid_idx], val_pred)
    hist = histogram(y_tr.astype(int), 
                     int(np.min(X_train['AdoptionSpeed'])), 
                     int(np.max(X_train['AdoptionSpeed'])))
    tr_cdf = get_cdf(hist)
    
    pred_test_y_k = getTestScore2(val_pred, tr_cdf)
    qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
    qwks.append(qwk)
    rmses.append(loss)
    print("RMSE=",loss, "QWK_2 = ", qwk,'elapsed time:',time.time()-since)
stacking_test = np.mean(stacking_test, axis=0)
print('mean rmse:',np.mean(rmses), 'rmse std:', np.std(rmses))
print('mean qwk:', np.mean(qwks), 'qwk std:', np.std(qwks))


# In[134]:


# Compute QWK based on OOF train predictions:
hist = histogram(X_train['AdoptionSpeed'].astype(int), 
                 int(np.min(X_train['AdoptionSpeed'])), 
                 int(np.max(X_train['AdoptionSpeed'])))
tr_cdf = get_cdf(hist)
train_predictions = getTestScore2(stacking_train, tr_cdf)
test_predictions = getTestScore2(stacking_test, tr_cdf)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, train_predictions)
print("QWK = ", qwk)


# In[135]:


# Distribution inspection of original target and predicted train and test:

# print("True Distribution:")
# print(pd.value_counts(X_train['AdoptionSpeed'], normalize=True).sort_index())
# print("\nTrain Predicted Distribution:")
# print(pd.value_counts(train_predictions, normalize=True).sort_index())
# print("\nTest Predicted Distribution:")
# print(pd.value_counts(test_predictions, normalize=True).sort_index())


# In[136]:


# Generate submission:

submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': stacking_test.astype(np.int32)})
# submission.head()
submission.to_csv('submission.csv', index=False)


# In[137]:


# result = pd.read_csv('../input/test_solution.csv')

# predictions = pd.DataFrame({'PetID': test['PetID'].values, 
#                             'vote': final_pred['pred'].values.astype(np.int32),
#                             'stacking': stacking_test.astype(np.int32)})

# pb = result[result['Usage']=='Public']
# pb = pb.merge(predictions, on='PetID')

# pb.head()

# qwk = quadratic_weighted_kappa(pb['AdoptionSpeed'], pb['vote'])
# print("QWK = ", qwk)

# qwk = quadratic_weighted_kappa(pb['AdoptionSpeed'], pb['stacking'])
# print("QWK = ", qwk)


# In[ ]:




