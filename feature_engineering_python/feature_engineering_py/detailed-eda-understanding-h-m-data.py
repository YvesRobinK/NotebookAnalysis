#!/usr/bin/env python
# coding: utf-8

# # **Detailed EDA - Understanding H&M data**  
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/HM-Logo.png" alt="drawing" width="400"/>
# 
# 
# Understanding data is the most important part of any analysis. It's the basis of all decisions related to data preprocessing and later the ML machine model creation and finally the interpretation of the results.
# 
# In this notebook, you will find a detailed description of this dataset. I hope you find it useful.
# 
# As an introduction let's see who our client/stakeholder actually is:
# 
# **H&M** is a very popular Swedish clothing company with a headquarter in Stockholm. The company was established in 1947 and was originally named Hennes. In 1968 they acquired the hunting and fishing store named Mauritz Widfoss. From this point in time it operated under name Hennes& Mauritz or simply H&M. Six years later the had a debut on the Stockholm Stock Exchange which allowed them soon after in 1976 to open their first shop outside Sweden - in UK, London. In 2000 they entered the US market. As of 2022 it operates shops in 74 countries what you can see on the map below:  
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/f/f0/H%26M_Global_map_%281%29.png" alt="drawing" width="450"/>
# 
# 
# More about history of H&M you can read on their article [here](https://about.hm.com/content/dam/hmgroup/groupsite/documents/en/Digital%20Annual%20Report/2017/Annual%20Report%202017%20Our%20history.pdf).
# 
# **Recommender Systems** are powerful, successful and widespread applications for almost every business selling products or services. It's especially useful for companies with a wide offer and diverse clients. Ideal examples are retail companies, like H&M, Zalando, etc. as well as these selling services or digital products - the best examples are Netflix or Spotify. If you visit websites of any of these firms you'll notice that after creating an account the service will start to recommend you other products, movies or songs that the algorithm thinks will suit you the best. It's their way to personalise the offer and who doesn't like to get such care. That's why these systems are precious for business owners. The more you buy, watch and listen the better it gets. Also, the more users the better it gets.  
# 
# However, because this is a dynamic environment - both clients and product change it is quite a hassle to tune, manage and maintain these systems. That's why companies using this tool keep a big group of machine learning engineers and backend SE in a dedicated department or they outsource this work to other companies.
# 
# More about **Netflix Recommender System (NRS)** you can read [here](https://research.netflix.com/research-area/recommendations).  
# 
# To learn more about the **Zalando** recommender system check their [Engineering Blog](https://engineering.zalando.com/posts/2016/12/recommendations-galore-how-zalando-tech-makes-it-happen.html).
# 
# Recommender Systems usually are classified into two groups:
# 1. *Collaborative-filtering* which is based on users behaviours. However, it needs a lot of traffic data per customer to be fully successfull.
# 2. *Content-based filtering* which is based on similarity and complementariness (what fits what) of products. This is good if you just started to collect traffic data but you have a good described and labeled products.
# Many companies create their custom-made, hybrid systems. In this competition we're free to decide what methodology we're are going to use.
# 
# Now let's go to the data itself.
# The first step: loading necessary libraries and 3 databases (articles, transactions and customers).

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # nice visualisations
import matplotlib.pyplot as plt # basic visualisation library
import datetime as dt # library to opearate on dates

print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("seaborn version: {}".format(sns.__version__))


# In[2]:


art = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
cust = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
trans = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")


# # 1. Articles database  
# 
# This databasecontains information about the assortiment of H&M shops. It's important not to confuse it with the number of transactions for each article what is given in a different database.

# In[3]:


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

data = art
art_dtypes = art.dtypes.value_counts()

fig = plt.figure(figsize=(5,2),facecolor='white')

ax0 = fig.add_subplot(1,1,1)
font = 'monospace'
ax0.text(1, 0.8, "Key figures",color='black',fontsize=28, fontweight='bold', fontfamily=font, ha='center')

ax0.text(0, 0.4, "{:,d}".format(data.shape[0]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax0.text(0, 0.001, "# of rows \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily=font,ha='center')

ax0.text(0.6, 0.4, "{}".format(data.shape[1]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax0.text(0.6, 0.001, "# of features \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily=font,ha='center')

ax0.text(1.2, 0.4, "{}".format(art_dtypes[0]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax0.text(1.2, 0.001, "# of text columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily=font, ha='center')

ax0.text(1.9, 0.4,"{}".format(art_dtypes[1]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax0.text(1.9, 0.001,"# of numeric columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily=font,ha='center')

ax0.set_yticklabels('')
ax0.tick_params(axis='y',length=0)
ax0.tick_params(axis='x',length=0)
ax0.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax0.spines[direction].set_visible(False)

fig.subplots_adjust(top=0.9, bottom=0.2, left=0, hspace=1)

fig.patch.set_linewidth(5)
fig.patch.set_edgecolor('#8c8c8c')
fig.patch.set_facecolor('#f6f6f6')
ax0.set_facecolor('#f6f6f6')
    
plt.show()


# Unique indentifier of an article:
# * ```article_id``` (int64) - an unique 9-digit identifier of the article, 105 542 unique values (as the length of the database)
# 
# 5 product related columns:
# * ```product_code``` (int64) - 6-digit product code (the first 6 digits of ```article_id```, 47 224 unique values
# * ```prod_name``` (object) - name of a product, 45 875 unique values
# * ```product_type_no``` (int64) - product type number, 131 unique values
# * ```product_type_name``` (object) - name of a product type, equivalent of ```product_type_no```
# * ```product_group_name``` (object) - name of a product group, in total 19 groups
# 
# 2 columns related to the pattern:
# * ```graphical_appearance_no``` (int64) - code of a pattern, 30 unique values
# * ```graphical_appearance_name``` (object) - name of a pattern, 30 unique values
# 
# 2 columns related to the color:
# * ```colour_group_code``` (int64) - code of a color, 50 unique values
# * ```colour_group_name``` (object) - name of a color, 50 unique values
# 
# 4 columns related to perceived colour (general tone):
# * ```perceived_colour_value_id``` - perceived color id, 8 unique values
# * ```perceived_colour_value_name``` - perceived color name, 8 unique values
# * ```perceived_colour_master_id``` - perceived master color id, 20 unique values
# * ```perceived_colour_master_name``` - perceived master color name, 20 unique values
# 
# 2 columns related to the department:
# * ```department_no``` - department number, 299 unique values
# * ```department_name``` - department name, 299 unique values
# 
# 4 columns related to the index, which is actually a top-level category:
# * ```index_code``` - index code, 10 unique values
# * ```index_name``` - index name, 10 unique values
# * ```index_group_no``` - index group code, 5 unique values
# * ```index_group_name``` - index group code, 5 unique values
# 
# 2 columns related to the section:
# * ```section_no``` - section number, 56 unique values
# * ```section_name``` - section name, 56 unique values
# 
# 2 columns related to the garment group:
# * ```garment_group_n``` - section number, 56 unique values
# * ```garment_group_name``` - section name, 56 unique values
# 
# 1 column with a detailed description of the article:
# * ```detail_desc``` - 43 404 unique values

# Let's check how many missing values we have (in pct).

# In[4]:


art.isna().sum()/len(art)*100


# Only one column - ```detail desc``` - has missing values but this is a very small fraction of the dataset - about 0.4%.
# 
# Let's visualise some of the articles. To do so I will create a helper function below. As mentioned in the competition description not all articles have an image. Therefore, this function will show selected amount of images from a given folder. It will also write an article name. What I've found is that a leading zero in the name of the folder does not correspond to the first digit of the ```article_id``` and it has to be stripped when looking for a product in the articles database. E.g.: *0108775015.jpg* is ```article_id``` *108775015*.

# In[5]:


from os import walk

def show_articles(folder, no_images=3):
    folder_path = '../input/h-and-m-personalized-fashion-recommendations/images/{}/'.format(folder)
    # extracting all image names from a folder
    files = []
    for _, _, filenames in walk(folder_path):
        files.extend(filenames)
    no_files = len(files)
    if no_images > no_files:
        no_images = no_files
        print("Warning! In the folder there are less images than requested.")
        
    # plotting selected number of pictures
    images = files[:no_images]
    fig, ax = plt.subplots(1,no_images, figsize=(12,4))
    for i, img in enumerate(images):
        art_id = img.split('.')[0]
        img = plt.imread(folder_path+img)
        ax[i].imshow(img, aspect='equal')
        ax[i].grid(False)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
        ax[i].set_xlabel(art[art['article_id']==int(art_id[1:])]['prod_name'].iloc[0])
    plt.show()
    
show_articles('020',4)


# In a hidden cell below there's a helper function for plotting sorted horizontal barplots.

# In[6]:


import matplotlib.ticker as mtick

def plot_bar(database, col, figsize=(13,5), pct=False, label='articles'):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#f6f6f6')
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if pct:
        data = database[col].value_counts()
        data = data.div(data.sum()).mul(100)
        data = data.reset_index()
        ax = sns.barplot(data=data, x=col, y='index', color='#2693d7', lw=1.5, ec='black', zorder=2)
        ax.set_xlabel('% of ' + label, fontsize=10, weight='bold')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        data = database[col].value_counts().reset_index()
        ax = sns.barplot(data=data, x=col, y='index', color='#2693d7', lw=1.5, ec='black', zorder=2)        
        ax.set_xlabel('# of articles' + label)
        
    ax.grid(zorder=0)
    ax.text(0, -0.75, col, color='black', fontsize=10, ha='left', va='bottom', weight='bold', style='italic')
    ax.set_ylabel('')
        
    plt.show()


# In[7]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plot_bar(art, 'index_group_name', pct=True)


# Most of the articles are in categories (index) of *Ladieswear* and *Baby/Children*. The smallest amount of articles is in *Sport* group.

# In[8]:


plot_bar(art, 'index_name', pct=True)


# Once again the dominant group is *Ladieswear*. However, this time the second group is *Divided*. Let's check what this group actually is.  
# 
# After asking a subject matter expert, I know that this is a category for teenagers.

# In[9]:


art_divided = art[art['index_name']=='Divided']
art_divided[['prod_name','product_type_name','detail_desc','index_name','section_name','garment_group_name']].drop_duplicates().head(12)


# Let's visualise some of articles from this category. To do so I will create a helper function that can be reused later.

# In[10]:


def show_items_in_category(column, value, no_imgs=4, title=None):
    data = art[art[column]==value]
    cat_ids = data['article_id'].iloc[:no_imgs].to_list()
    
    fig, ax = plt.subplots(1, no_imgs, figsize=(12,4))

    for i, prod_id in enumerate(cat_ids):
        folder = str(prod_id)[:2]
        file_path = '../input/h-and-m-personalized-fashion-recommendations/images/0{}/0{}.jpg'.format(folder, prod_id)

        img = plt.imread(file_path)       
        ax[i].imshow(img, aspect='equal')
        ax[i].grid(False)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
        ax[i].set_xlabel(art[art['article_id']==int(prod_id)]['prod_name'].iloc[0])
    
    fig.suptitle(title)
    plt.show()


# In[11]:


show_items_in_category('index_name', 'Divided', 5, 'Articles from a "Divided" category')


# We see now also that index is a sub-category of ```index_group```. We can create now a multi-index fram with groupings to see the counts:

# In[12]:


art.groupby(['index_group_name', 'index_name']).size()


# Even finer category is ```product_group_name```. Let's see what is it's structure and articles counts.

# In[13]:


plot_bar(art, 'product_group_name', pct=True)


# The barchart above shows that most of articles lays in only few groups. Let's see the cumulative sum and a Pareto graph.

# In[14]:


data = art['product_group_name'].value_counts()
data = data.div(data.sum()).mul(100)
pareto = data.cumsum().rename('cumulative_pct')
pareto


# In[15]:


data = pareto.reset_index()
data.columns = ['group', 'cumulative_pct']
data.index += 1
data['cumulative_pct'][6]/100


# In[16]:


fig, ax = plt.subplots(figsize=(15,5))

data = pareto.reset_index()
data.columns = ['features', 'cumulative_pct']
data.index += 1

sns.lineplot(data=data, x=data.index, y='cumulative_pct')

for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
    
ax.set_xticks(pareto.reset_index().index)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('cumulative percentage')
ax.set_xlabel('number of product groups')
ax.set_xlim(1)
ax.set_ylim(data['cumulative_pct'][1])

ax.vlines(7, data['cumulative_pct'][1], data['cumulative_pct'][7], color='orange', ls='--')
ax.hlines(data['cumulative_pct'][7], 1, 7, color='orange', ls='--')
ax.text(0, 1.05, 'Pareto graph of the product groups', color='black', fontsize=10, ha='left', va='bottom', weight='bold', style='italic', transform=ax.transAxes)
plt.show()


# Subcategories of ```product_group``` is ```product_type```. Taking into consideration there are 131 types plotting a barchar may be useless. So let's check how many product types are in each group.

# In[17]:


for group in art['product_group_name'].unique():
    print('Number of subcategories in "{}"" is {}.'.format(group, len(art.groupby(['product_group_name', 'product_type_name']).size()[group])))


# We see that the accessories and shoes have the biggest number of subcategories.
# 
# The key takeaways here are:
# * The hierarhy of categories is: index_group --> index --> group --> type
# * Over 80% of the products lays in 4 product groups (out of 19)
# 
# 
# Let's investigate now some other feature of the H&M products.  
# First in what colours are H&M products:

# In[18]:


plot_bar(art, 'colour_group_name', figsize=(15,12), pct=True)


# Next let's see the perceived color, which is general color range like: dark, light, bright, etc.

# In[19]:


plot_bar(art, 'perceived_colour_value_name', pct=True)


# I'm wondering what ```Dusty Light``` actually is. Let's visualise.

# In[20]:


show_items_in_category('perceived_colour_value_name', 'Dusty Light', 5, 'Dusty Light articles')


# After looking at the pictures above we see what does it mean. Interestingly the last picture shows two items where one is grey-ish while the other is black.

# In[21]:


plot_bar(art, 'perceived_colour_master_name', pct=True)


# Perceived colour is more or less in line with the colour itself but it's more generic. 
# 
# Now it's time for patterns.

# In[22]:


plot_bar(art, 'graphical_appearance_name', figsize=(14,7), pct=True)


# Most of H&M articles are without any pattern - they fall into a category ```solid```. Let's visualise next two categories.

# In[23]:


show_items_in_category('graphical_appearance_name', 'All over pattern', 5,  'All over pattern')


# In[24]:


show_items_in_category('graphical_appearance_name', 'Melange', 5,  'Melange')


# As I'm a total layman in fashion from pure curiosity I'll check what are some other patterns.

# In[25]:


show_items_in_category('graphical_appearance_name', 'Lace', 5,  'Lace')


# In[26]:


show_items_in_category('graphical_appearance_name', 'Embroidery', 5,  'Embroidery')


# Let's see some exemplary descriptions.

# In[27]:


art['detail_desc'].drop_duplicates().to_list()[:10]


# Nice way to visualise descriptions is to create a word cloud. To do this we have to tokenize the descriptions and later join all tokens into a bag of words or like in this case into a one long text.

# In[28]:


import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS # library to create a wordcloud
from PIL import Image

# creating cloud of words
words_raw = art['detail_desc'].dropna().apply(nltk.word_tokenize)
bag_of_words = " ".join(words_raw.explode())
stopwords = set(STOPWORDS)

# creating cloud of words
fig, ax1 = plt.subplots(figsize=(8,6))
wordcloud = WordCloud(stopwords=stopwords, background_color="white", height=300, contour_width=3).generate(bag_of_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # 2. Customers database
# 
# This database contains data about customers. Collecting this type of data allows company to tune their recommender system. This data contains data which can be treated as 'static' or slowly-changing, usually these are features like sex, age, address, hight, etc. Let's look what H&M gave us.

# In[29]:


mpl.rcParams.update(mpl.rcParamsDefault)

cust_dtypes = cust.dtypes.value_counts()
data = cust

fig = plt.figure(figsize=(5,2),facecolor='white')

ax0 = fig.add_subplot(1,1,1)
ax0.text(1.0, 1, "Key figures",color='black',fontsize=28, fontweight='bold', fontfamily='monospace',ha='center')

ax0.text(0, 0.4, "{:,d}".format(data.shape[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(0, 0.001, "# of rows \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(0.6, 0.4, "{}".format(data.shape[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(0.6, 0.001, "# of features \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.2, 0.4, "{}".format(cust_dtypes[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(1.2, 0.001, "# of text columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.9, 0.4,"{}".format(cust_dtypes[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(1.9, 0.001,"# of numeric columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.set_yticklabels('')
ax0.tick_params(axis='y',length=0)
ax0.tick_params(axis='x',length=0)
ax0.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax0.spines[direction].set_visible(False)
    
fig.patch.set_linewidth(5)
fig.patch.set_edgecolor('#8c8c8c')
fig.patch.set_facecolor('#f6f6f6')
ax0.set_facecolor('#f6f6f6')

plt.show()


# Unique indentifier of a customer:
# * ```customer_id``` - an unique identifier of the customer
# 
# 5 product related columns:
# * ```FN``` - binary feature (1 or NaN)
# * ```Active``` - binary feature (1 or NaN)
# * ```club_member_status``` - status in a club, 3 unique values
# * ```fashion_news_frequency``` - frequency of sending communication to the customer, 4 unique values
# * ```age```  - age of the customer
# * ```postal_code``` - postal code (anonimized), 352 899 unique values

# Let's check the missing values:

# In[30]:


cust.isna().sum()


# Here I'll replace NaN for columns ```FN``` and ```Active``` with 0.

# In[31]:


cust_backup = cust.copy()
cust[['FN','Active']] = cust[['FN','Active']].fillna(0)


# In[32]:


fig, ax = plt.subplots(figsize=(5,5))
explode = (0, 0.1)
colors = sns.color_palette('Paired')
ax.pie(cust['FN'].value_counts(), explode=explode, labels=['Not-FN','FN'],
       autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
ax.axis('equal')
plt.show()


# In[33]:


fig, ax = plt.subplots(figsize=(5,5))
explode = (0, 0.1)
colors = sns.color_palette('Paired')
ax.pie(cust['Active'].value_counts(), explode=explode, labels=['Not-active','Active'],
       autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
ax.axis('equal')
plt.show()


# In[34]:


FN_Active = len(cust[(cust['FN']==1) & (cust['Active']==1)])/cust.shape[0]*100
print('Percentage of customers that have both FN and Active status: {}%'.format(round(FN_Active,2)))


# It look that all custmoers that are Active have also FN status. But reverse is not true - not all users with FN status are active. Let's check this by substaraction of these two sets. If it's correct we would expect to see a percentage difference in order of 0.9%.

# In[35]:


FN_not_active = len(cust[(cust['FN']==1) & (cust['Active']!=1)])/cust.shape[0]*100
print('Percentage of customers that have FN status but are not Active: {}%'.format(round(FN_not_active,2)))


# As we remember from the missing values analysis there is also a group of people where we do not have any data. Perhaps, these people are the ones without membership at all - to add them to the visialisation I'll fill NaN with 'N/A'.

# In[36]:


cust['club_member_status'] = cust['club_member_status'].fillna('N/A')
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plot_bar(cust, 'club_member_status', pct=True, label='customers')


# Most of customers have an ```active``` membership status, others are with the ```pre-create``` status. Interestingly, there is nobody with ```left club``` status.

# In[37]:


cust['fashion_news_frequency'] = cust['fashion_news_frequency'].fillna('N/A')
plot_bar(cust, 'fashion_news_frequency', pct=True, label='customers')


# We see here that there are two statuses that can be merged: *NONE* and *None* to improve data quality and to reduce one dimension. Most of customers do not receive any communication from H&M.
# 
# Let's see now what is the age distribution.

# In[38]:


fig, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=cust, x='age', bins=cust['age'].nunique(), color='orange', stat="percent")
ax.set_xlabel('Distribution of the customers age')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
median = cust['age'].median()
ax.axvline(x=median, color="green", ls="--")
ax.text(median, 3.5, 'median: {}'.format(round(median,1)), rotation='vertical', ha='right')
ax.text(12, 5.5, 'Distribution of customers age', color='black', fontsize=10, ha='left', va='bottom', weight='bold', style='italic')
plt.show()


# The distribution shows that there are two main age-groups of customers: around 20-30 years old and 45-55 years old. Let's check how old is the oldest customer.

# In[39]:


print('The olders customer is {} years old.'.format(cust['age'].max()))


# In[40]:


fig, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=cust, x='age', bins=cust['age'].nunique(), hue='Active', stat="percent")
ax.set_xlabel('Distribution of the customers age')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# Let's check what is the share of the active customers per age.

# In[41]:


active_age_ratio = cust.groupby('age')['Active'].value_counts(normalize=True).mul(100)
active_age_ratio = active_age_ratio.rename('Active_ratio', inplace=True).reset_index()
active_age_ratio = active_age_ratio[active_age_ratio['Active']==1]
active_age_ratio['Active'] = active_age_ratio['Active'].astype(int)
active_age_ratio['age'] = active_age_ratio['age'].astype(int)


# In[42]:


fig, ax = plt.subplots(figsize=(16,8))
sns.barplot(x='age', y='Active_ratio', data=active_age_ratio)
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
    
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
ax.set_title("Share of active users per age", color='black', fontsize=12, weight='bold')
plt.show()


# Having these data it may be worth considering during data engineering part to perform a customer segmentation. It can be performed in many ways - some techinques I describe in my other [Kaggle notebook](https://www.kaggle.com/datark1/customers-clustering-k-means-dbscan-and-ap).

# # 3. Transactions
# 
# This is the biggest database containing all transactions every day.

# In[43]:


mpl.rcParams.update(mpl.rcParamsDefault)

trans_dtypes = trans.dtypes.value_counts()
data = trans

fig = plt.figure(figsize=(5,2),facecolor='white')

ax0 = fig.add_subplot(1,1,1)
ax0.text(1.0, 1, "Key figures",color='black',fontsize=28, fontweight='bold', fontfamily='monospace',ha='center')

ax0.text(0, 0.4, "{:,d}".format(data.shape[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(0, 0.001, "# of rows \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(0.6, 0.4, "{}".format(data.shape[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(0.6, 0.001, "# of features \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.2, 0.4, "{}".format(trans_dtypes[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(1.2, 0.001, "# of text columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.9, 0.4,"{}".format(trans_dtypes[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
ax0.text(1.9, 0.001,"# of numeric columns \nin the dataset",color='dimgrey',fontsize=15, fontweight='light', fontfamily='monospace',ha='center')

ax0.set_yticklabels('')
ax0.tick_params(axis='y',length=0)
ax0.tick_params(axis='x',length=0)
ax0.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax0.spines[direction].set_visible(False)

fig.patch.set_linewidth(5)
fig.patch.set_edgecolor('#8c8c8c')
fig.patch.set_facecolor('#f6f6f6')
ax0.set_facecolor('#f6f6f6')
    
plt.show()


# Columns description:
# * ```t_dat``` - date of a transaction in format YYYY-MM-DD but provided as a string
# * ```customer_id``` - identifier of the customer which can be mapped to the ```customer_id```  column in the ```customers``` table
# * ```article_id``` - identifier of the product which can be mapped to the ```article_id```  column in the ```articles``` table
# * ```price``` - price paid
# * ```sales_channel_id``` - sales channel, 2 unique values

# In[44]:


trans.head()


# In[45]:


trans.isna().sum()


# No missing data!
# 
# Let's investigate the price column.

# In[46]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
fig, ax = plt.subplots(figsize=(10,5), facecolor='#f6f5f5')
ax = sns.histplot(data=trans, x='price', bins=50, stat="percent")
ax.set_xlabel('Distribution of the price')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# It's clear from the above graph that we have a lot o outliers. Let's look at the price after cutting the values above 0.1.

# In[47]:


fig, ax = plt.subplots(figsize=(10,5), facecolor='#f6f5f5')
data = trans[trans['price']<0.1]
ax = sns.histplot(data=data, x='price', bins=20, stat="percent")
ax.set_xlabel('Distribution of the price')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# Now let's see the data distribution over time. First what dates range is provided. It will be usefull to change the datatype of 

# In[48]:


trans['t_dat'] = pd.to_datetime(trans['t_dat'])


# In[49]:


begin = trans['t_dat'].min()
end = trans['t_dat'].max()
print('Date range is from {} to {}.'.format(begin.date(), end.date()))


# We have full 2 years of data. Let's plot now number of transactions per day over the full period of time.

# In[50]:


t_per_day = trans.groupby('t_dat',as_index=False).count()


# In[51]:


fig, ax = plt.subplots(figsize=(16,8))

sns.lineplot(data=t_per_day, x='t_dat',y='customer_id')

ax.set_xlabel('date')
ax.set_ylabel('number of transactions')

ax.axvline(x=dt.datetime(2019,1,1), c='green')
ax.axvline(x=dt.datetime(2020,1,1), c='green')

max_t = t_per_day['customer_id'].max()
max_t_date = t_per_day[t_per_day['customer_id']==max_t]['t_dat']
ax.scatter(max_t_date, max_t, c='red')
ax.text(max_t_date+pd.DateOffset(days=5), max_t-4000, '{}\n{:,d}'.format(max_t_date.iloc[0].date(), max_t))

min_t = t_per_day['customer_id'].min()
min_t_date = t_per_day[t_per_day['customer_id']==min_t]['t_dat']
ax.scatter(min_t_date, min_t, c='red')
ax.text(min_t_date+pd.DateOffset(days=5), min_t-4000, '{}\n{:,d}'.format(min_t_date.iloc[0].date(), min_t))
ax.set_xlim(trans['t_dat'].min(),trans['t_dat'].max())

for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')

plt.show()


# From the graph above we see that there are distinct variations and spikes in the number of transactions per day. It would be clearer to visualise this by using box plots with monthly aggregations.

# In[52]:


trans_gr_month = trans.groupby('t_dat').size().rename("no_transactions")
trans_gr_month = trans_gr_month.reset_index()
trans_gr_month['month_year'] = trans_gr_month['t_dat'].dt.to_period('M')


# In[53]:


fig, ax = plt.subplots(figsize=(16,8))
ax = sns.boxplot(x="month_year", y='no_transactions', data=trans_gr_month)
plt.xticks(rotation=90)
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.set_xlabel('Month-Year')
ax.set_ylabel('Number of transactions')
plt.show()


# The bar chart above show us that per day usuall number of transactions lays in range about between 25 000 and 80 000 transactions per day. We see also that sales spikes during summertime and drops during winter.
# 
# Now, let's see how many transactions, on average, customers do.

# In[54]:


t_by_customer = trans.groupby('customer_id', as_index=False).size()

fig, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=t_by_customer, x='size', bins=50, stat="percent")
ax.set_xlabel('Distribution of total transactions per customer')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# Clearly there's a lot of outliers. Let's look at the distribution after cutting everything above 50 trasactions per customer.

# In[55]:


t_by_customer_50tr = t_by_customer[t_by_customer['size'] < 50]

fig, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=t_by_customer_50tr, x='size', bins=50, stat="percent")
ax.set_xlabel('Distribution of total transactions per customer')
for loc in ['bottom', 'left']:
    ax.spines[loc].set_visible(True)
    ax.spines[loc].set_linewidth(2)
    ax.spines[loc].set_color('black')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# The graph above shows us that most of customers, on average, bought only few items during these 2 years.
# 
# Let's see now the popularity of sale channels.

# In[56]:


fig, ax = plt.subplots(figsize=(5,5))
explode = (0, 0.1)
colors = sns.color_palette('Paired')
ax.pie(trans['sales_channel_id'].value_counts(), explode=explode, labels=['1','2'],
       autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
ax.axis('equal')
ax.set_title('Sale channel')
plt.show()


# # 4. Combined databases EDA

# All 3 databases can be joined together. However, if you look closely it's not possible to connect directly databases ```art``` (with ```article_id``` as the primary key) and  ```cust``` (with ```customer_id``` as the primary key). They can be joined with a bridge table ```trans``` which contains both keys ```article_id``` and ```customer_id```.  
# 
# We can join them separately or all at once - it depends what your approach will be. Note that joining all tables will create one mega-table with a significant size which can slow-down your calculations or event it may not fit into memory allocated to you.
# 
# Below I'll join pre-filtered ```trans``` with ```art``` to visualise amount of transactions per article top-level group each month. I'll merge dataframes using ```pd.merge()``` with SQL-like logic.

# In[57]:


#trans['month'] = trans['t_dat'].dt.month
#trans['year'] = trans['t_dat'].dt.year
trans['year_month'] = trans['t_dat'].dt.to_period('M')
trans.head()


# In[58]:


trans_grouped = trans.groupby(['year_month', 'article_id']).size().rename('total_per_article').to_frame()


# In[59]:


trans_grouped.head()


# In[60]:


trans_grouped.reset_index(inplace=True)


# In[61]:


art_trans = pd.merge(art[['article_id', 'index_name']], trans_grouped, on='article_id')


# In[62]:


art_trans = art_trans.groupby(['year_month','index_name'])['total_per_article'].sum().to_frame()
art_trans.head()


# # UNDER CONSTRUCTION - TO BE CONTINUED

# In[ ]:




