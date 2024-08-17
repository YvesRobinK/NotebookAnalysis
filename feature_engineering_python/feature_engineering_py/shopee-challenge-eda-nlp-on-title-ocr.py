#!/usr/bin/env python
# coding: utf-8

# # Shopee Product Matching EDA and Cleaning
# ### Import Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from textwrap import wrap
import pytesseract
import re,string
from wordcloud import WordCloud, STOPWORDS
from tqdm.notebook import tqdm
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[2]:


path = '../input/shopee-product-matching'
train_path = '../input/shopee-product-matching/train_images'
test_path = '../input/shopee-product-matching/test_images'


# **Let's have a look at the data head**

# In[3]:


data = pd.read_csv(path+'/'+'train.csv')
data.head()


# ### Basic Details about the data

# In[4]:


print(f"The Shape of the train data : {data.shape}")
print(f"Duplicate Rows : {data.shape[0] - len(data['posting_id'].unique())}")


# In[5]:


print("Number unique label_groups = {}".format( len(data["label_group"].unique()) ))


# **Here one label group indicates similar products, i.e: all products with same label_id are similar**

# In[6]:


num_label_groups = {}
for i in data['label_group']:
    num_label_groups[i] = data[data['label_group'] == i]


# In[7]:


len_label_groups = {}
for i in num_label_groups:
    len_label_groups[i] = len(num_label_groups[i])

print(f"Max of all the label groups : {max(len_label_groups.values())}")
print(f"Min of all the label groups : {min(len_label_groups.values())}")


# In[8]:


label_names = list(num_label_groups.keys())
num_label_groups[label_names[5]]


# **Let's Visualize some Similar products**
# 
# This Function allows you to input label_id and view top 10 (or less where applicable) similar products.

# In[9]:


def visualize_sim_products(label_id):
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = '28'
    plt.figure(figsize=(50,50))
    length = len_label_groups[label_id]
    if length > 10:
        length = 10
    for i in range(length):
        img = plt.imread(train_path + '/' + num_label_groups[label_id]['image'].iloc[i])
        plt.subplot(length,2,i+1)
        plt.imshow(img)
        plt.title("\n".join(wrap(num_label_groups[label_id]['title'].iloc[i],60)))
        plt.axis('off')
    plt.show()


# In[10]:


visualize_sim_products(label_names[14])


# ### Perceptual Hashing
# 
# In this challenge perpetual hashing is provided, so I read up on it, According to wikipedia : 
# > Perceptual hashing is the use of an algorithm that produces a snippet or fingerprint of various forms of multimedia.[1][2] Perceptual hash functions are analogous if features of the multimedia are similar, whereas cryptographic hashing relies on the avalanche effect of a small change in input value creating a drastic change in output value. Perceptual hash functions are widely used in finding cases of online copyright infringement as well as in digital forensics because of the ability to have a correlation between hashes so similar data can be found (for instance with a differing watermark).
# 
# *So maybe similar objects have similar Perceptual hash values.* 
# This is my hypothesis, let's check if this is true.

# In[11]:


num_label_groups[label_names[17]]['image_phash']


#  **Seems like my hypotheses was wrong.** 
# 
# But we can use the hamming distance between these phashes as a feature (Feature Engineering), that will be for another notebook though. 
# 
# Seems like exact same images have same pHash.

# In[12]:


def hamming(s1, s2):
    return float(sum(c1 != c2 for c1, c2 in zip(s1, s2))) / float(len(s1))
hamming('bf38f0e08397c712','bf38f0e083d7c710')


# ### Let's find out how many same exact images are there

# In[13]:


copies = {}
for i in data['image_phash']:
    copies[i] = data[data['image_phash'] == i]
phash_list = list(copies.keys())

copies[phash_list[14]]


# In[14]:


copies_len = {}
for i in copies.keys():
    copies_len[i] = len(copies[i])


# In[15]:


copies_len = pd.DataFrame({'phash':copies_len.keys(),'count':copies_len.values()})
# copies_len.reset_index(inplace=True)
copies_len.head()


# In[16]:


copies_len.sort_values(by='count',ascending = False, inplace = True)


# ### Top 10 duplicate images' phashes

# In[17]:


fig = plt.figure(figsize=(70,50))
sns.barplot(x = copies_len.iloc[:10]['phash'],y=copies_len.iloc[:10]['count'])
plt.show()


# ### Let's view some exact copies

# In[18]:


def visualize_sim_phashes(phash):
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = '28'
    plt.figure(figsize=(50,50))
    length = len(copies[phash])
    if length > 10:
        length = 10
    for i in range(length):
        img = plt.imread(train_path + '/' + copies[phash]['image'].iloc[i])
        plt.subplot(length,2,i+1)
        plt.imshow(img)
        plt.title("\n".join(wrap(copies[phash]['title'].iloc[i],60)))
        plt.axis('off')
    plt.show()


# In[19]:


visualize_sim_phashes(phash_list[14])


# # NLP Based EDA

# In[20]:


title_text = data['title'].values


# In[21]:


def clean(title):
    stop = stopwords.words('english')
    title = [x for x in title.split() if not x in stop]
    title = " ".join(title)
    title = title.lower()
    title = re.sub(r"\-","",title)
    title = re.sub(r"\+","",title)
    title = re.sub (r"&","and",title)
    title = re.sub(r"\|","",title)
    title = re.sub(r"\\","",title)
    title = re.sub(r"\W"," ",title)
    for p in string.punctuation :
        title = re.sub(r"f{p}","",title)
    
    title = re.sub(r"\s+"," ",title)
    
    return title


# In[22]:


data.head()


# ## WordClouds

# In[23]:


stopwords_wc = set(STOPWORDS) 
token_text = ''

for i in tqdm(title_text):
    token_l = i.split()
    token_text += " ".join(token_l) + " " 


# ### WordCloud for Title Text

# In[24]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords_wc, 
                min_font_size = 10).generate(token_text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# **Let's Extract everything we can from the images, maybe used as a feature in the future**

# In[25]:


# Use this for OCR extraction
# ocr_text = []
# for i in tqdm(range(data.shape[0])):
#     img = cv2.imread(train_path + '/' + data['image'].iloc[i])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     text = pytesseract.image_to_string(img)
#     text = " ".join(text.split())
#     if len(text) != 0:
#         ocr_text.append(text)
#     else:
#         ocr_text.append('Nothing Found')

# data['ocr_text'] = ocr_text


# In[26]:


# data.to_csv('cleaned_and _raw_ocr.csv')
data = pd.read_csv('../input/cleaned-shopee-data-with-ocr/cleaned_title_and_ocr.csv')
# Cleaning titles and ocr text again for stopwords, which I missed before
data.drop(['cleaned_title','cleaned_ocr_text'],axis=1,inplace=True)
data['cleaned_title'] = data['title'].map(clean)


# In[27]:


data['cleaned_ocr_text'] = data['ocr_text'].map(clean)
data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
data.to_csv('cleaned_title_and_ocr_sw.csv')


# In[28]:


data.head()


# ### WordCloud For the OCR data

# In[29]:


title_text = data['cleaned_ocr_text'].values
stopwords_wc = list(STOPWORDS)
token_text = ''

for i in tqdm(title_text):
    if i.strip() != 'Nothing Found'.lower():
        token_l = i.split()
        token_text += " ".join(token_l) + " "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords_wc, 
                min_font_size = 10,contour_color='steelblue').generate(token_text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud,interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()     


# ## Let's have a look at title and OCR text as well
# 
# ### Distribution of Title text Lengths

# In[30]:


plt.figure(figsize = (10, 6))
sns.set_style("whitegrid")
sns.kdeplot(data['cleaned_title'].apply(lambda x: len(x)),fill = True,edgecolor='black',alpha=0.9)
plt.xlabel('Title Text Length')
plt.show()


# ### Distribution of Title text Tokens Count

# In[31]:


plt.figure(figsize = (10, 6))
sns.set_style("whitegrid")
sns.kdeplot(data['cleaned_title'].apply(lambda x: len(x.split())),fill = True,edgecolor='black',alpha=0.9,color='cyan')
plt.xlabel('Title Text Tokens Count')
plt.show()


# ### Distribution of OCR text lengths

# In[32]:


plt.figure(figsize = (10, 6))
sns.set_style("whitegrid")
sns.kdeplot(data['cleaned_ocr_text'].apply(lambda x: len(x)),fill = True,color = 'red',edgecolor='black',alpha=0.9)
plt.xlabel('OCR Text Length')
plt.show()


# ### Distribution of OCR text tokens count

# In[33]:


plt.figure(figsize = (10, 6))
sns.set_style("whitegrid")
sns.kdeplot(data['cleaned_ocr_text'].apply(lambda x: len(x.split())),fill = True,color = 'maroon',edgecolor='black',alpha=0.9)
plt.xlabel('OCR Text Tokens Count')
plt.show()


# ## Image shapes EDA

# In[34]:


image_shapes_h = []
image_shapes_w = []
image_shapes_c = []
for i in tqdm(range(data.shape[0])):
    img = cv2.imread(train_path + '/' + data['image'].iloc[i])
    h, w, c = img.shape
    image_shapes_h.append(h)
    image_shapes_w.append(w)
    image_shapes_c.append(c)


# In[35]:


dump(image_shapes_h,'heights.pkl')
dump(image_shapes_w,'widths.pkl')
dump(image_shapes_c,'channels.pkl')

image_shapes_h = load('heights.pkl')
image_shapes_w = load('widths.pkl')
image_shapes_c = load('channels.pkl')


# In[36]:


set(image_shapes_c)


# **There aren't any B&W/Gray images**

# In[37]:


sns.set_style("white")
sns.axes_style('whitegrid')
h = sns.JointGrid(x =  image_shapes_h,y = image_shapes_w,height=8)
h.plot_joint(sns.scatterplot)
h.plot_marginals(sns.histplot, kde=True)
plt.show()


# ## Insights to the Data (Compilation)
# 
# + There are a total of 34250 products in the database which are unique, that means there are no duplicate rows.
# + There are 11014 label_groups provided in the dataset.
# + Max of all the label groups : 51
# + Min of all the label groups : 2
# + Perceptual hash of all the images in a label group are not the same, but there are some identical images with different titles.
# + Phash hamming distance might be a good feature in the future to use.
# + We can refer the respective wordclouds to get an idea of frequenty used words in the title and ocr text.
# + The OCR text might be a good feature for the model.
# + Title text length  seem to be less than 90 characters for most of the data.
# + Title text tokens count seems to be less than 20 words for most of the data.
# + Similarly OCR text length also seems to be less than 90-100 characters.
# + The count of OCR text tokens seem to be less than 18-20 tokens (most of these tokens are garbled noise from bad OCR)
# + All images are 3-channeled RGB images, there are no B&W or gray images in the dataset. The jointplot can be refered to get and idea of images heights and widths distribution.
