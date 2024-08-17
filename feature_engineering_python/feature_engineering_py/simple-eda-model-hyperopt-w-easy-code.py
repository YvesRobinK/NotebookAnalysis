#!/usr/bin/env python
# coding: utf-8

# # Exploring and Predicting Sales
# 
# ## Descrition of this competition:
# This challenge serves as final project for the "How to win a data science competition" Coursera course.
# 
# In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 
# 
# We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
# 
# <b>Data fields</b><br>
# <b>ID</b> - an Id that represents a (Shop, Item) tuple within the test set<br>
# <b>shop_id</b> - unique identifier of a shop<br>
# <b>item_id</b> - unique identifier of a product<br>
# <b>item_category_id</b> - unique identifier of item category<br>
# <b>item_cnt_day</b> - number of products sold. You are predicting a monthly amount of this measure<br>
# <b>item_price</b> - current price of an item<br>
# <b>date</b> - date in format dd/mm/yyyy<br>
# <b>date_block_num</b> - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33<br>
# <b>item_name</b> - name of item<br>
# <b>shop_name</b> - name of shop<br>
# <b>item_category_name</b> - name of item category

# ## Some questions to guide the initial exploration:
# - Data shape, missings, first rows, 
# - What are the entropy of each column
# - what are the principal shops? 
# - What are the distributions of items price and total items sold by each item;
# - What are the more sold items and which are their categorys;
# - What's the range of date sales;
# - How many items was sold by each day. Could we see any peak in christhmas, valentine's day or another special day.
# - Crossing some of this features 
# - And a lot of more questions that probably will raise through the exploration.

# English is not my first language, so sorry for any mistake.

# ## NOTE: This kernel is under construction. 
# If you think that this kernel was useful for you, please votes up the kernel, and if you want see all codes, fork it. 

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import random

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf

import lightgbm
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import gc
import warnings
from itertools import product
warnings.filterwarnings("ignore")


# # Importing the Datasets

# In[2]:


df_train = pd.read_csv('../input/sales_train.csv')

df_categories = pd.read_csv("../input/item_categories.csv")
df_items = pd.read_csv("../input/items.csv")
df_shops = pd.read_csv("../input/shops.csv")

df_test = pd.read_csv("../input/test.csv")


# ## Joining the tables to our train dataset

# In[3]:


df_train = pd.merge(df_train, df_items, on='item_id', how='inner')
df_train = pd.merge(df_train, df_categories, on='item_category_id', how='inner')
df_train = pd.merge(df_train, df_shops, on='shop_id', how='inner')

df_test = pd.merge(df_test, df_items, on='item_id', how='inner')
df_test = pd.merge(df_test, df_categories, on='item_category_id', how='inner')
df_test = pd.merge(df_test, df_shops, on='shop_id', how='inner')

# del df_items, df_categories, df_shops

gc.collect()


# ## First look at our data

# In[4]:


df_train.head()


# Wow, it's russian !!! I will try to do someting to handle with the data easiest. 

# In[5]:


dict_categories = ['Cinema - DVD', 'PC Games - Standard Editions',
                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',
                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',
                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',
                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',
                    'Programs - Home and Office', 'Gifts - Development',
                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',
                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',
                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',
                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',
                    'Accessories - PS4', 'Games - PSVita',
                    'Books - Methodical materials 1C', 'Payment cards - PSN',
                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',
                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',
                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',
                    'Gifts - Attributes', 'Service Tools',
                    'Music - branded production CD', 'Payment cards - Live!',
                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',
                    'Music - Music Video', 'Game Consoles - PS3',
                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',
                    'Books - Audiobooks 1C', 'Books - Digit',
                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',
                    'Accessories - XBOX ONE', 'Pure media (piece)',
                    'Programs - Home and Office (Digital)', 'Programs - Educational',
                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',
                    'Programs - Educational (Digit)', 'Accessories - PSP',
                    'Gaming consoles - XBOX ONE', 'Delivery of goods',
                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',
                    'Music - Gift Edition', 'Service Tools - Tickets',
                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',
                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',
                    'Gifts - Certificates, Services', 'Android Games - Digit',
                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',
                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',
                    'Books - Computer Literature', 'Books - Travel Guides',
                    'PC - Headsets / Headphones', 'Books - Fiction',
                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',
                    'Books - Cognitive literature']

dict_shops = ['Moscow Shopping Center "Semenovskiy"', 
              'Moscow TRK "Atrium"', 
              "Khimki Shopping Center",
              'Moscow TC "MEGA Teply Stan" II', 
              'Yakutsk Ordzhonikidze, 56',
              'St. Petersburg TC "Nevsky Center"', 
              'Moscow TC "MEGA Belaya Dacha II"',
              'Voronezh (Plekhanovskaya, 13)', 
              'Yakutsk Shopping Center "Central"',
              'Chekhov SEC "Carnival"', 
              'Sergiev Posad TC "7Ya"',
              'Tyumen TC "Goodwin"',
              'Kursk TC "Pushkinsky"', 
              'Kaluga SEC "XXI Century"',
              'N.Novgorod Science and entertainment complex "Fantastic"',
              'Moscow MTRC "Afi Mall"',
              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',
              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',
              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',
              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',
              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',
              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',
              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',
              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',
              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',
              "Volzhsky Shopping Center 'Volga Mall' ",
              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',
              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',
              'Online store of emergencies', 'Adygea Shopping Center "Mega"',
              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',
              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',
              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',
              'RostovNaDonu TRK "Megacenter Horizon"',
              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',
              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",
              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',
              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',
              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",
              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']


# In[6]:


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y") # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday #extracting week day
    df["_day"] = df['date'].dt.day # extracting day
    df["_month"] = df['date'].dt.month # extracting month
    
    return df #returning the df after the transformations

def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):
    temp = cols
    cm = sns.light_palette("green", as_cmap=True)
    return pd.crosstab(df[temp[0]], df[temp[1]], 
                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)

def quantiles(df, columns):
    for name in columns:
        print(name + " quantiles")
        print(df[name].quantile([.01,.25,.5,.75,.99]))
        print("")

def chi2_test(col ):
    stat, p, dof, expected = stats.chi2_contingency((pd.crosstab(df_train[col], df_train.item_cnt_day)))
    # interpret test-statistic
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print(f"Testing the {col} by Total Items Sold")
    print('dof=%d' % dof)
    print(p)
    print("Critical Result: ")
    if abs(stat) >= critical:
        print(f"Critical {round(critical,4)}")
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
    print("")
    alpha = 1.0 - prob
    print("P Value: ")
    if p <= alpha:
        print(f"P-Value: {round(p,8)}")
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
def log_transforms(df, cols):
    for col in cols:
        df[col+'_log'] = np.log(df[col] + 1)
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df

def knowningData(df, limit=5): #seting the function with df, 
    print(f"Dataset Shape: {df.shape}")
    print('Unique values per column: ')
    print(df.nunique())
    print("################")
    print("")    
    for column in df.columns: #initializing the loop
        print("Column Name: ", column )
        entropy = round(stats.entropy(df[column].value_counts(), base=2),2)
        print("entropy ", entropy, 
              " | Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | Total unique values: ", df.nunique()[column], #print the data and % of nulls
              " | Missing: ", df[column].isna().sum())
        print("Top 5 most frequent values: ")
        print(round(df[column].value_counts()[:limit] / df[column].value_counts().sum() * 100,2))
        print("")
        print("####################################")


# ## First look at some informations of our data
# - to see the output click on "show output" button >>>

# In[7]:


knowningData(df_train)


#     

# ## Mapping our dictionary
# - as the data are in russian, I decided to translate it to english. 

# In[8]:


df_train.item_category_name = df_train.item_category_name.map(dict(zip(df_train.item_category_name.value_counts().index, dict_categories)))
df_train.shop_name = df_train.shop_name.map(dict(zip(df_train.shop_name.value_counts().index, dict_shops)))


# ## I will start exploring our target (item_cnt_day) that refers to items sold and the item_price

# In[9]:


plt.figure(figsize=(16,12))

plt.subplot(221)
g = sns.distplot(np.log(df_train[df_train['item_cnt_day'] >0]['item_cnt_day']))
g.set_title("Item Sold Count Distribuition", fontsize=18)
g.set_xlabel("")
g.set_ylabel("Frequency", fontsize=12)

plt.subplot(222)
g1 = plt.scatter(range(df_train.shape[0]), np.sort(df_train.item_cnt_day.values))
g1= plt.title("Item Sold ECDF Distribuition", fontsize=18)
g1 = plt.xlabel("Index")
g1 = plt.ylabel("Total Items", fontsize=15)

plt.subplot(223)
g2 = sns.distplot(np.log(df_train[df_train['item_price'] > 0]['item_price']))
g2.set_title("Items Price Log Distribuition", fontsize=18)
g2.set_xlabel("")
g2.set_ylabel("Frequency", fontsize=15)

plt.subplot(224)
g3 = plt.scatter(range(df_train.shape[0]), np.sort(df_train.item_price.values))
g3= plt.title("Item Price ECDF Distribuition", fontsize=18)
g3 = plt.xlabel("Index")
g3 = plt.ylabel("Item Price Distribution", fontsize=15)

plt.subplots_adjust(wspace = 0.3, hspace = 0.3,
                    top = 0.9)

plt.show()


# - Interesting... Almost all of items sold are 1.
# Let's take some descriptions about the quantiles

# ## Quantiles of continuous features and target
# - I will also create a total amount column, that will be the price * qtd. sold

# In[10]:


df_train['total_amount'] = df_train['item_price'] * df_train['item_cnt_day']


# In[11]:


quantiles(df_train, ['item_cnt_day','item_price', 'total_amount'])


# Very cool and interesting values distribution.
# Let's investigate it further trought the other features and try to find some interesting patterns

# ## Knowing the Shop, category and items columns

# In[12]:


import squarify

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(30)]
shop_name = df_train["shop_name"].value_counts() #counting the values of shop names

print("Description most frequent countrys: ")
print(shop_name[:10]) #printing the 15 top most 

shop = round((df_train["shop_name"].value_counts()[:20] \
                       / len(df_train["shop_name"]) * 100),2)

plt.figure(figsize=(20,10))
g = squarify.plot(sizes=shop.values, label=shop.index, 
                  value=shop.values,
                  alpha=.8, color=color)
g.set_title("'TOP 20 Stores/Shop - % size of total",fontsize=20)
g.set_axis_off()
plt.show()


# Cool. it's a well distributed market where no one has a great monopoly. <br>
# Let's keep understanding the Shop Names

# ## Looking the Total Amount sold by the Stores

# In[13]:


print("Percentual of total sold by each Shop")
print((df_train.groupby('shop_name')['item_price'].sum().nlargest(25) / df_train.groupby('shop_name')['item_price'].sum().sum() * 100)[:5])

df_train.groupby('shop_name')['item_price'].sum().nlargest(25).iplot(kind='bar',
                                                                     title='TOP 25 Shop Name by Total Amount Sold',
                                                                     xTitle='Shop Names', 
                                                                     yTitle='Total Sold')


# It's interesting to note that the difference in values aren't different as the total solds<br>
# the difference betweeen Moscow Shopping Center and Moscow TRK aren't so different 

# ## The Item Solds by Shop Names

# In[14]:


print("Percentual of total sold by each Shop")
print((df_train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25) / df_train.groupby('shop_name')['item_cnt_day'].sum().sum() * 100)[:5])

df_train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25).iplot(kind='bar',
                                                                       title='TOP 25 Shop Name by Total Amount Sold',
                                                                       xTitle='Shop Names', 
                                                                       yTitle='Total Sold')


# 

# In[15]:


df_train.columns


# In[16]:


df_train = log_transforms(df_train, ['item_price', 'item_cnt_day', 'total_amount'])


# In[17]:


df_train[['item_cnt_day', 'item_price', 'item_name']].sort_values('item_cnt_day', ascending=False).head(20)


# 

# ## Items category
# - Let's see some distributions of the top values in our data

# In[18]:


top_cats = df_train.item_category_name.value_counts()[:15]

plt.figure(figsize=(15,20))

plt.subplot(311)
g1 = sns.countplot(x='item_category_name', 
                   data=df_train[df_train.item_category_name.isin(top_cats.index)])
g1.set_xticklabels(g1.get_xticklabels(),rotation=70)
g1.set_title("TOP 15 Principal Products Sold", fontsize=22)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=18)

plt.subplot(312)
g2 = sns.boxplot(x='item_category_name', y='item_cnt_day', 
                   data=df_train[df_train.item_category_name.isin(top_cats.index)])
g2.set_xticklabels(g2.get_xticklabels(),rotation=70)
g2.set_title("Principal Categories by Item Solds Log", fontsize=22)
g2.set_xlabel("")
g2.set_ylabel("Items Sold Log Distribution", fontsize=18)

plt.subplot(313)
g3 = sns.boxplot(x='item_category_name', y='total_amount', 
                   data=df_train[df_train.item_category_name.isin(top_cats.index)])
g3.set_xticklabels(g3.get_xticklabels(),rotation=70)
g3.set_title("Category Name by Total Amount Log", fontsize=22)
g3.set_xlabel("")
g3.set_ylabel("Total Amount Log", fontsize=18)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)
plt.show()


# It's very meaningfull. <br>
# we can see what are the most sold items and the distribution of solds. <br>
# I will see the prices further and the possibility to cross our data

# ## Taking a look at the highest purchases in the data
# - First I will create a subsample to see the outliers
# - I will get top 5K highest total amounts and see if we can have some insight

# In[19]:


sub_categorys_5000 = df_train.sort_values('total_amount',
                                          ascending=False)[['item_category_name', 'item_name', 
                                                            'shop_name',
                                                            'item_cnt_day','item_price',
                                                            'total_amount']].head(5000)
sub_categorys_5000.head(10)


# Loking the price of PS4 I can infer that it's not about dollars.<br>
# Let's see the distribution of our features in the subsample;

# # Descriptions of the top 5k most exepensive sales

# In[20]:


sub_categorys_5000.describe(include='all')


# Very cool!! We can see that the most expensive item Price is 308k and the highest item sold a

# ## Total revenue Representation of total sales

# In[21]:


print("TOTAL REPRESENTATION OF TOP 5k Most Expensive orders: ", 
      f'{round((sub_categorys_5000.item_price.sum() / df_train.item_price.sum()) * 100, 2)}%')


# Altough it contains high values, the top 5k most expensive bills represents just 2.45% of total amount sold. <br>
# I think that it happens because are retails shops that sells to "normal" people and not other business
# 

# ## Shops and items categorys of the most expensive trades

# In[22]:


plt.figure(figsize=(14,26))

plt.subplot(311)
g = sns.countplot(x='shop_name', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Shop Names in Top Bills ", fontsize=22)
g.set_xlabel('Shop Names', fontsize=18)
g.set_ylabel("Total Count in expensive bills", fontsize=18)

plt.subplot(312)
g = sns.countplot(x='item_category_name', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Category Name in Top Bills", fontsize=22)
g.set_xlabel('Category Names', fontsize=18)
g.set_ylabel("Total Count in expensive bills", fontsize=18)

plt.subplot(313)
g = sns.boxenplot(x='item_category_name', y='item_cnt_day', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Category Name in Top Bills by Total items Sold", fontsize=22)
g.set_xlabel('Most top Category Names', fontsize=18)
g.set_ylabel("Total sold distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.2, hspace = 1.6,top = 0.9)

plt.show()


# Some of this values has just one unit sold, and I will see it now

# ## The most expensive products

# In[23]:


df_train[['item_category_name', 'item_name', 'shop_name', 'item_cnt_day', 'item_price']].nlargest(15, 'item_price')


# Hummm... We have a high GAP between the most expensive item to other items. 
# I will see how many of total are of 1 unit item sold
# 

# In[24]:


print(f"Total of solds that have only one unit: {round(len(df_train[df_train.item_cnt_day == 1]) / len(df_train) * 100,2)}%")


# ## Category's by items sold by day - Crosstab

# In[25]:


cross_heatmap(sub_categorys_5000, ['item_category_name', 'item_cnt_day'])


# Very interesting patterns. We can see that some items aren't sold only one unit. Maybe some people buy to resell the games or consoles

# ## TOP 25 items Solds
# - Let's understand the principal itens sold at the dataset
# - The distribution of Total amount and Item Solds in the bill

# In[26]:


count_item = df_train.item_name.value_counts()[:25]

plt.figure(figsize=(14,50))

plt.subplot(311)
g = sns.countplot(x='item_name', data=df_train[df_train.item_name.isin(count_item.index)])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Most sold Items", fontsize=22)
g.set_xlabel('', fontsize=18)
g.set_ylabel("Total Count of ", fontsize=18)

plt.subplot(312)
g1 = sns.boxenplot(x='item_name', y='total_amount',
                  data=df_train[df_train.item_name.isin(count_item.index)])
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Count of Category Name in Top Bills", fontsize=22)
g1.set_xlabel('', fontsize=18)
g1.set_ylabel("Total Count in expensive bills", fontsize=18)

plt.subplot(313)
g2 = sns.boxenplot(x='item_name', y='item_cnt_day', 
                  data=df_train[df_train.item_name.isin(count_item.index)])
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Items Sold Distribution", fontsize=22)
g2.set_xlabel('Item Names', fontsize=18)
g2.set_ylabel("Total sold distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.2, hspace = 1.2,top = .8)

plt.show()


# 

# ## Understanding the revenue for each Shop Name. 

# In[27]:


cross_heatmap(df_train.sample(500000), ['item_category_name', 'shop_name'], 
              normalize='columns', aggfunc='sum', values=df_train['total_amount'])


# 

# ## Time series
# - Exploring some metrics abuot the datetime feature

# In[28]:


# Calling the function to transform the date column in datetime pandas object
df_train = date_process(df_train)

#seting some static color options
color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']


dates_temp = df_train['date'].value_counts().to_frame().reset_index().sort_values('index') 
# renaming the columns to apropriate names
dates_temp = dates_temp.rename(columns = {"date" : "Total_Bills"}).rename(columns = {"index" : "date"})

# creating the first trace with the necessary parameters
trace = go.Scatter(x=dates_temp.date.astype(str), y=dates_temp.Total_Bills,
                    opacity = 0.8, line = dict(color = color_op[7]), name= 'Total tickets')

# Below we will get the total amount sold
dates_temp_sum = df_train.groupby('date')['item_price'].sum().to_frame().reset_index()

# using the new dates_temp_sum we will create the second trace
trace1 = go.Scatter(x=dates_temp_sum.date.astype(str), line = dict(color = color_op[1]), name="Total Amount",
                        y=dates_temp_sum['item_price'], opacity = 0.8)

# Getting the total values by Transactions by each date
dates_temp_count = df_train[df_train['item_cnt_day'] > 0].groupby('date')['item_cnt_day'].sum().to_frame().reset_index()

# using the new dates_temp_count we will create the third trace
trace2 = go.Scatter(x=dates_temp_count.date.astype(str), line = dict(color = color_op[5]), name="Total Items Sold",
                        y=dates_temp_count['item_cnt_day'], opacity = 0.8)

#creating the layout the will allow us to give an title and 
# give us some interesting options to handle with the outputs of graphs
layout = dict(
    title= "Informations by Date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)

# creating figure with the both traces and layout
fig = dict(data= [trace, trace1, trace2], layout=layout)

#rendering the graphs
iplot(fig) #it's an equivalent to plt.show()


# We can see that one specific day had an peak in sales. I put it on google and I don't find anything about this date. 

# ## Understanding the sales by month

# In[29]:


def generate_random_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

#shared Xaxis parameter can make this graph look even better
fig = tls.make_subplots(rows=2, cols=1)

layout1 = cf.Layout(
    height=500,
    width=200
)
animal_color = generate_random_color()
fig1 = df_train.groupby(['_month'])['item_cnt_day'].count().iplot(kind='bar',barmode='stack',
                                                                  asFigure=True,showlegend=False,
                                                                  title='Total Items Sold By Month',
                                                                  xTitle='Months', yTitle='Total Items Sold',
                                                                  color = 'blue')
fig1['data'][0]['showlegend'] = False
fig.append_trace(fig1['data'][0], 1, 1)


fig2 = df_train.groupby(['_month'])['item_cnt_day'].sum().iplot(kind='bar',barmode='stack',
                                                                title='Total orders by Month',
                                                                xTitle='Months', yTitle='Total Orders',
                                                                asFigure=True, showlegend=False, 
                                                                color = 'blue')

#if we do not use the below line there will be two legend
fig2['data'][0]['showlegend'] = False


fig.append_trace(fig2['data'][0], 2, 1)

layout = dict(
    title= "Informations by Date",
    )

fig['layout']['height'] = 800
fig['layout']['width'] = 1000
fig['layout']['title'] = "TOTAL ORDERS AND TOTAL ITEMS BY MONTHS"
fig['layout']['yaxis']['title'] = "Total Items Sold"
fig['layout']['xaxis']['title'] = "Months"
fig['layout']

iplot(fig)


# In[30]:


df_train['diff_days'] = df_train.groupby(['shop_name','item_category_name']).date.diff().dt.days.fillna(0, downcast='infer')


# In[31]:


df_train['diff_days'].hist(bins=50)


# In[32]:


grouped_blocks = df_train.groupby(["date_block_num",
                                    "shop_name","item_category_name"])["item_name",
                                                                       "item_price",
                                                                       "item_cnt_day"].agg({"item_name":["nunique", 'count'],
                                                                                            "item_price":["min",'mean','max'],
                                                                                            "item_cnt_day":["min",'mean','max','sum']})


# In[33]:


grouped_blocks.head(15)


# ## Start modeling to ML 

# - For the modelling part I am using the codes of some another kernels that I will put as resources on the final of kernel
# 
# Test Chi2
# - Let's see if the categorys are independent or dependent of the target

# - HO is that the feature hasn't influence in item_cnt_day
# - H1 is that the feature has some influence in item_cnt_day

# In[34]:


chi2_test('shop_name')


# In[35]:


chi2_test('item_category_name')


# In[36]:


chi2_test('item_price')


# - We can see that item_price, shop_name and item category are important to explain the items sold

# In[37]:


## Deleting the datasets that was used to explore the data
del df_train
del df_test

gc.collect()

## Importing the df's again to modelling
df_train = pd.read_csv('../input/sales_train.csv')
df_test = pd.read_csv("../input/test.csv")


# In[38]:


df_train = df_train[df_train.item_price<100000]
df_train = df_train[df_train.item_cnt_day<1001]


# In[39]:


median = df_train[(df_train.shop_id==32)&(df_train.item_id==2973)&(df_train.date_block_num==4)&(df_train.item_price>0)].item_price.median()
df_train.loc[df_train.item_price<0, 'item_price'] = median


# In[40]:


df_train.loc[df_train.shop_id == 0, 'shop_id'] = 57
df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
df_train.loc[df_train.shop_id == 1, 'shop_id'] = 58
df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
df_train.loc[df_train.shop_id == 10, 'shop_id'] = 11
df_test.loc[df_test.shop_id == 10, 'shop_id'] = 11


# In[41]:


from sklearn.preprocessing import LabelEncoder

df_shops.loc[df_shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
df_shops['city'] = df_shops['shop_name'].str.split(' ').map(lambda x: x[0])
df_shops.loc[df_shops.city == '!Якутск', 'city'] = 'Якутск'
df_shops['city_code'] = LabelEncoder().fit_transform(df_shops['city'])
df_shops = df_shops[['shop_id','city_code']]

df_categories['split'] = df_categories['item_category_name'].str.split('-')
df_categories['type'] = df_categories['split'].map(lambda x: x[0].strip())
df_categories['type_code'] = LabelEncoder().fit_transform(df_categories['type'])
# if subtype is nan then type
df_categories['subtype'] = df_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
df_categories['subtype_code'] = LabelEncoder().fit_transform(df_categories['subtype'])
df_categories = df_categories[['item_category_id','type_code', 'subtype_code']]

df_items.drop(['item_name'], axis=1, inplace=True)


# In[42]:


import time

# Creating the Matrix
matrix = []

# Column names
cols = ['date_block_num','shop_id','item_id']

# Creating the pairwise for each date_num_block
for i in range(34):
    # Filtering sales by each month
    sales = df_train[df_train.date_block_num==i]
    # Creating the matrix date_block, shop_id, item_id
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
# Seting the matrix to dataframe
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

# Seting the features to int8 to reduce memory usage
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)


# In[43]:





# In[43]:


# Creating the revenue column
df_train['revenue'] = df_train['item_price'] *  df_train['item_cnt_day']


# In[44]:


# getting the total itens sold by each date_block for each shop_id and item_id pairs
group = df_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
# Renaming columns
group.columns = ['item_cnt_month']
# Reset the index 
group.reset_index(inplace=True)

# Merging the grouped column to our matrix
matrix = pd.merge(matrix, group, on=cols, how='left')
# Filling Na's and clipping the values to have range 0,20
# seting to float16 to reduce memory usage
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))

matrix.head()


# In[45]:


gc.collect()


# In[46]:


# Creating the date_block in df_test
df_test['date_block_num'] = 34

# Seting the df test columns to int8 to reduce memory usage
df_test['date_block_num'] = df_test['date_block_num'].astype(np.int8)
df_test['shop_id'] = df_test['shop_id'].astype(np.int8)
df_test['item_id'] = df_test['item_id'].astype(np.int16)


# In[47]:


## Concatenating the df test into our matrix and filling Na's with zero
matrix = pd.concat([matrix, df_test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month


# In[48]:


# merging the df shops, df items, and df categories in our matrix
matrix = pd.merge(matrix, df_shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, df_items, on=['item_id'], how='left')
matrix = pd.merge(matrix, df_categories, on=['item_category_id'], how='left')

# Seting the new columns to int8 to reduce memory
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

matrix.head()


# In[49]:


# Function to calculate lag features
def lag_feature(df, lags, col):
    # Columns to get lag
    tmp = df[['date_block_num','shop_id','item_id',col]]
    # loop for each lag value in the list
    for i in lags:
        # Coping the df
        shifted = tmp.copy()
        # Creating the lag column
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        # getting the correct date_num_block to calculation
        shifted['date_block_num'] += i
        # merging the new column into the matrix
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        
    return df


# In[50]:


# Creating the lag columns 
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

matrix.head()


# In[51]:


# Getting the mean item_cnt_month by date_bock
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
# Renaming
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the grouped object into the matrix
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
# creating the lag column to average itens solds
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
# Droping the date_avg_item_cnt
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[52]:


## Getting the mean item solds by date_blocks and item_id 
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
# Renaming column
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)

# Geting the lag feature to the new column
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[53]:


# Grouping the mean items sold by shop id for each date_block
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
# Renaming Columns
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the grouped into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)

# Geting the lag of the new column
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[54]:


## Getting the mean items sold by item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})

# Renaming column
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[55]:


## Getting the mean items sold by shop_id and item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[56]:


## Getting the mean items sold by shop_id and item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[57]:


## Getting the mean items sold by shop_id and subtype_code for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[58]:


## Getting the mean items sold by city_code for each date_block_num
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[59]:


## Getting the mean items sold by item_id and city_code for each date_block_num
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[60]:


## Getting the mean items sold by type_code for each date_block_num
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[61]:


## Getting the mean items sold by subtype_code for each date_block_num
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')

matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

matrix.head()


# In[62]:


del df_items
del df_shops
del df_categories


# In[63]:


matrix = reduce_mem_usage(matrix)


# In[64]:


# Getting the mean item price by item_id
group = df_train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['item_id'], how='left')

matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

## Getting the mean item price by item_id for each date_block_num
group = df_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

# Geting the lags of date item avg item price
lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

# seting the delta price lag for each lag price
for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

# Selecting trends
def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
# Applying the trend selection
matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

# Getting the features to drop
fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)
gc.collect()

matrix.head()


# In[65]:





# In[65]:


# Getting the revenue sum by shop_id and date_block
group = df_train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

# Merging the new group into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

# Getting the mean item price by item_id
group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)


# In[66]:


matrix['month'] = matrix['date_block_num'] % 12


# In[67]:


days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)


# In[68]:


cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         


# In[69]:


cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num


# In[70]:


matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')


# In[71]:


matrix = matrix[matrix.date_block_num > 11]


# In[72]:


def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)


# In[73]:


matrix.info()


# In[74]:


matrix.to_pickle('data.pkl')

del matrix
del cache
del group
del df_train

# leave test for submission
gc.collect();


# In[75]:


data = pd.read_pickle('data.pkl')


# In[76]:


data = data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'city_code', 'item_category_id',
             'type_code', 'subtype_code', 'item_cnt_month_lag_1', 'item_cnt_month_lag_2',
             'item_cnt_month_lag_3', 'item_cnt_month_lag_6', 'item_cnt_month_lag_12',
             'date_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
             'date_item_avg_item_cnt_lag_3', 'date_item_avg_item_cnt_lag_6', 'date_item_avg_item_cnt_lag_12',
             'date_shop_avg_item_cnt_lag_1', 'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
             'date_shop_avg_item_cnt_lag_6', 'date_shop_avg_item_cnt_lag_12', 'date_cat_avg_item_cnt_lag_1',
             'date_shop_cat_avg_item_cnt_lag_1', 'item_shop_first_sale', 'item_first_sale',
             #'date_shop_type_avg_item_cnt_lag_1','date_shop_subtype_avg_item_cnt_lag_1',
             'date_city_avg_item_cnt_lag_1', 'date_item_city_avg_item_cnt_lag_1',
             #'date_type_avg_item_cnt_lag_1', #'date_subtype_avg_item_cnt_lag_1',
             'delta_price_lag', 'month', 'days', 'item_shop_last_sale', 'item_last_sale']]


# X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# y_train = data[data.date_block_num < 33]['item_cnt_month']
# X_val = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# y_val = data[data.date_block_num == 33]['item_cnt_month']
# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# In[77]:


X_train = data[data.date_block_num < 34].drop(['item_cnt_month'], axis=1)
#Y_train = train_set['item_cnt']
y_train = data[data.date_block_num < 34]['item_cnt_month'].clip(0.,20.)

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

X_val = X_train[X_train.date_block_num > 30]
X_train = X_train[X_train.date_block_num <= 30]

y_val = y_train[~y_train.index.isin(X_train.index)]
y_train = y_train[y_train.index.isin(X_train.index)]

X_val_test = X_val[X_val.date_block_num > 32]
X_val = X_val[X_val.date_block_num <= 32]

y_val_test = y_val[~y_val.index.isin(X_val.index)]
y_val = y_val[y_val.index.isin(X_val.index)]

X_train.head()


# X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# y_train = data[data.date_block_num < 33]['item_cnt_month']
# X_val = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# y_val = data[data.date_block_num == 33]['item_cnt_month']
# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# In[78]:


del data
gc.collect()


# In[79]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

thresh = 5 * 10**(-4)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
#select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)

X_train = selection.transform(X_train)
X_val = selection.transform(X_val)
X_val_test = selection.transform(X_val_test)
X_test = selection.transform(X_test)

del model, selection

gc.collect()


# In[80]:


print("X_important_train Shape: ", X_train.shape)
print("X_important_val Shape: ", X_val.shape)
print("X_important_val_test Shape: ", X_val_test.shape)
print("X_important_test Shape: ", X_test.shape)


# ## Preprocessing and spliting the dataset

# ## Setting the X_test

# test_set = df_test.copy()
# test_set['date_block_num'] = 34
# 
# test_set = pd.merge(test_set, test_price_a, on=['shop_id','item_id'], how='left')
# test_set = mergeFeature(test_set)
# 
# test_set['item_order'] = test_set['order_prev']
# test_set.loc[test_set['item_order'] == 0, 'item_order'] = 1
# 
# X_test = test_set.drop(['ID'], axis=1)
# X_test.head()
# 
# assert(X_train.columns.isin(X_test.columns).all())

# In[81]:





# In[81]:


# Define searched space
hyper_space = {'objective': 'regression',
               'metric':'rmse',
               'boosting':'gbdt',
               #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),
               'max_depth':  hp.choice('max_depth', [3, 5, 8, 10, 12, 15]),
               'num_leaves': hp.choice('num_leaves', [25, 50, 75, 100, 125, 150, 225, 250, 350, 400, 500]),
               'subsample': hp.choice('subsample', [.3, .5, .7, .8, .9, 1]),
               'colsample_bytree': hp.choice('colsample_bytree', [.5, .6, .7, .8, .9, 1]),
               'learning_rate': hp.choice('learning_rate', [.01, .1, .05, .2]),
               'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6, .7]),
               'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]), 
                # 'bagging_fraction': hp.choice('bagging_fraction', [.5, .6, .7, .8, .9, 1]),
               'feature_fraction':  hp.choice('feature_fraction', [.6, .7, .8, .9, 1]), 
               'bagging_frequency':  hp.choice('bagging_frequency', [.3, .4, .5, .6, .7, .8, .9]),                  
               'min_child_samples': hp.choice('min_child_samples', [10, 20, 30, 40])}


# In[82]:


def rmse(y_pred, y):
    return np.sqrt(np.mean(np.square(y - y_pred)))


# In[83]:


lgtrain = lightgbm.Dataset(X_train, label=y_train)
lgval = lightgbm.Dataset(X_val, label=y_val)

def evaluate_metric(params):
    
    model_lgb = lightgbm.train(params, lgtrain, 600, 
                          valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 
                          verbose_eval=300)

    pred = model_lgb.predict(X_val_test, num_iteration=1000)

    score = rmse(pred, y_val_test)
    
    print(score, params)
 
    return {
        'loss': score,
        'status': STATUS_OK,
        'stats_running': STATUS_RUNNING
    }


# In[84]:





# In[84]:


# Trail
trials = Trials()

# Set algoritm parameters
algo = partial(tpe.suggest, 
               n_startup_jobs=-1)

# Seting the number of Evals
MAX_EVALS= 30

# Fit Tree Parzen Estimator
best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1,
                 algo=algo, max_evals=MAX_EVALS, trials=trials)

# Print best parameters
best_params = space_eval(hyper_space, best_vals)


# ## Getting the best parameters

# In[85]:


print("BEST PARAMETERS: " + str(best_params))


# ## Trainning the model with best parameters and predicting the X_test to submission

# In[86]:


model_lgb = lightgbm.train(best_params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], early_stopping_rounds=500, 
                      verbose_eval=250)

lgb_pred = model_lgb.predict(X_test).clip(0, 20)


# In[87]:


lgb_pred = model_lgb.predict(X_test).clip(0, 20)

# rmse(lgb_pred, y_val_test)


# 

# In[88]:


result = pd.DataFrame({
    "ID": df_test["ID"],
    "item_cnt_month": lgb_pred.clip(0. ,20.)
})
result.to_csv("submission.csv", index=False)


# In[89]:





# ## I'm working on this kernel, so stay tuned and votesup the kernel =) 

# Some references:<br>
# https://www.kaggle.com/plasticgrammer/future-sales-prediction-playground <br>
# https://www.kaggle.com/jatinmittal0001/predict-future-sales-part-2<br>
# https://www.kaggle.com/dlarionov/feature-engineering-xgboost<br>
