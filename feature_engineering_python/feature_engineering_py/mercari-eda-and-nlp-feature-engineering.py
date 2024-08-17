#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, I do some basic EDA and feature engineering on the Mercari data. So far, I am not getting great results, but there may be some interesting ideas here for future improvements.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/train.tsv',delimiter='\t',index_col=0)


# In[3]:


train.head()


# There are only a small number of features here. I'm going to ignore the item description for now to help save on space.

# In[4]:


train = train.drop('item_description',axis=1)
train['item_condition_id'] = train['item_condition_id'].astype(np.int8)
train['shipping'] = train['shipping'].astype(np.int8)
train.info()


# In[5]:


len(train)


# There are around 1.5 million entries, so things run pretty slowly on a regular CPU. It is at least still small enough to be held in memory.

# In[6]:


len(train.category_name.unique())


# In[7]:


len(train.brand_name.unique())


# There are about 1300 different categories and 5000 different brand names found in the data. The categories are actually mostly combinations of several categories separated by slashes.
# 
# # Some Basic Plots and Features of the Dataset
# 
# We should first make some basic plots, such as the distribution of prices.

# In[8]:


plt.hist(train.price,bins=50,range=(0,2000),color='b',alpha=0.6)
plt.xlabel('Price [$]')
plt.ylabel('Number')
#plt.yscale('log')
plt.show()


# The logarithmic axis isn't displaying for some reason, but we see that prices are heavily weighted toward the low end of the distribution. Prices range as high as several thousand dollars.

# In[9]:


plt.hist(np.log10(train[train['price']>0].price),bins=25,range=(0.5,3.1),color='b',alpha=0.6)
plt.xlabel(r'log_{10}(Price) [$]')
plt.ylabel('Number')
#plt.yscale('log')
plt.show()


# In a log-log plot (it's a histogram so I transformed the x-axis to keep the bins the same size), we see that above a log-Price of around 1.3, there is an approximately linear relationship between the count and the log of the price. Here, the log axis for some reason causes the bars not to be displayed. The linear relationship indicates an approximately power-law relationship between the two. It's not quite linear though, and it looks like the the shape is almost linear but more like a downward parabola.
# 
# We can also look at the price for each value of "Shipping."

# In[10]:


plt.hist(train[train['shipping']==0].price,bins=100,range=(0,100),
         normed=True,label='Shipping=0',color='b',alpha=0.6)
plt.hist(train[train['shipping']==1].price,bins=100,range=(0,100),
         normed=True,label='Shipping=1',color='r',alpha=0.6)

plt.xlabel('Price [$]')
plt.ylabel('Fraction')
#plt.yscale('log')
plt.legend(loc='upper right')
plt.show()


# You in fact can see that the price distribution is noticeably different between Shipping=1 and Shipping=0. The cheapest products seem to all be Shipping=1. We also see an odd effect where the peaks that appear every $5 are a single bin for Shipping=1 but 2 separate bins for Shipping=0. So, the price would allow us to predict Shipping with a reasonably good accuracy, although it looks like the opposite is probably not true.
# 
# There are 5 values for item_condition_id, so putting them on a single histogram plot is not so useful.

# In[11]:


print(train['item_condition_id'].value_counts())
train.groupby('item_condition_id')['price'].describe()


# But, we see that the basic statistical properties are quite similar. The quartiles, mean, and standard deviation all lie within a fairly narrow range. This means that item_condition_id is not going to be so useful unless combined with other features.

# # Categorical Variables
# 
# It might be good to look at the different categories that are available. There are 1300 total categories, but most of these are compound categories made up of several different basic categories. We can extract these by splitting the strings at a slash.

# In[12]:


train['category_name'] = train['category_name'].fillna('')

counter = Counter()
cats = [str(x).split('/') for x in train.category_name]
for c in cats:
    counter.update(c)

print('Number of categories: {}'.format(len(counter)))
print(counter.most_common())


# From the counter, we see that the 1288 original categories are built from 951 more basic categories. So splitting things is not going to help too much. We could potentially reduce this a lot by taking only the top X% of categories. So, if the category is the form A/B/C where A & B are common and C is not, we could turn it into A/B. We would want to tune the threshold to get things working properly. However, with 1.5M features, we have enough data where even this many categories is not a huge issue as long as there are a decent number of entries for each category.
# 
# We can do the same for the brand name.

# In[13]:


train['brand_name'] = train['brand_name'].fillna('')

counter = Counter()
cats = [str(x).split('/') for x in train.brand_name]
for c in cats:
    counter.update(c)

print('Number of brands: {}'.format(len(counter)))
print(counter.most_common())


# We can quickly see that most brands are not very well represented. Out of nearly 5000 brands, it looks like there's at most a few hundred with even 100 products in the dataset. Furthermore, nearly 50% of products don't even have any brand information. As above, it may be useful to set a threshold on the minimum number of products and remove the brand information from everything else.
# 
# There are other things that might be harder to automate such as combining different sub-brands from the same company. For example, Nike and Nike Golf are counted as two different brands. We could also try to split the brands into different categories. Things like "clothing," "jewelry," and "electronics" are good broad categories while things like "luxury" and "discount" are good subjective feaures. Again, this would be simple but tedious to do manually. Some ways to automate this would be to maybe do some sentiment analysis on the item descriptions for each brand (i.e. vectorize, run something like SVD and then maybe do clustering of brands with KMeans). It's probably also possible to do an analysis on top search results for each brand.

# It's also good to see how these categorical variables relate to the price.

# In[14]:


train.groupby('brand_name').price.mean().sort_values()


# It turns out that there are actually a couple brands where the mean price is $0. I haven't bothered checking but my guess is that these brands are represented by only a single item. But, it looks like there are many brands where the mean price is quite low. At the upper end we see a mix of things like luxury clothing brands and appliances. There is a lot of variation here, so I would expect brand to be very useful even without doing much feature engineering.

# In[15]:


train.groupby('category_name').price.mean().sort_values()


# Category is similar. There are some categories with typically low prices (<$5) and others with prices over $100. Because we're looking at log differences, the most important thing is getting prices to around the right order of magnitude, so the fact that there are a small number of products with prices over $1000 but no brands or categories anywhere near that is not a problem.

# # Making Predictions Using Group Averages
# 
# One of the simplest things that we can do is to group like categorical features and use some statistic based on the group as our prediction. The simplest thing to do would be to take the average price, although that is not necessarily the best choice. First, we should define our scoring function since sklearn does not already have it.

# In[16]:


def rmsle(ytrue,y):
    return np.sqrt(mean_squared_log_error(ytrue,y))


# Now, we can start picking different features to see what kind of results we get. I'll do a 5-fold cross validation here and print out the result from each fold. It's important that we only train things using the training set and not the validation set so that our model will be capable of handling new samples.
# 
# # Brand and Category Treated Separately

# In[17]:


kf = KFold(n_splits=5,random_state=123)
i = 0
def get_val(series,x):
    try:
        return series[x]
    except:
        pass
    return series['']
    
for train_idx,val_idx in kf.split(train):
    print('Fold {}'.format(i))
    i+=1
    cols = [1,2,3,4,5]
    X_train = train.iloc[train_idx,cols]
    X_val = train.iloc[val_idx,cols]

    brand_price = X_train.groupby('brand_name').price.mean()
    cat_price = X_train.groupby('category_name').price.mean()
    
    X_train['brand_price'] = [brand_price[x] for x in X_train.brand_name ]
    X_train['category_price'] = [cat_price[x] for x in X_train.category_name ]

    X_val['brand_price'] = [get_val(brand_price,x) for x in X_val.brand_name ]
    X_val['category_price'] = [get_val(cat_price,x) for x in X_val.category_name ]
    
    print('RMSLE, train, brand price: {:0.4}'
          .format(rmsle(X_train.price,X_train.brand_price)))
    print('RMSLE, train, category price: {:0.4}'
          .format(rmsle(X_train.price,X_train.category_price)))
    print('RMSLE, test, brand price: {:0.4}'
          .format(rmsle(X_val.price,X_val.brand_price)))
    print('RMSLE, test, category price: {:0.4}'
          .format(rmsle(X_val.price,X_val.category_price)))


# It turns out that, at least without doing any feature engineering, the brand and category give essentially identical results: around 0.7-0.71 for both train and test/validation sets.
# 
# # Brand and Category Together

# In[18]:


kf = KFold(n_splits=5,random_state=123)
i = 0
def get_val(series,x):
    try:
        return series[x]
    except:
        pass
    return series['Cat: Brand:']

train['CatBrand'] = 'Cat:'+train.category_name+' Brand:'+train.brand_name

for train_idx,val_idx in kf.split(train):
    print('Fold {}'.format(i))
    i+=1
    cols = [1,2,3,4,5]
    X_train = train.iloc[train_idx,:]
    X_val = train.iloc[val_idx,:]

    cb_price = X_train.groupby('CatBrand').price.mean()
    
    cb_price_train = np.array([cb_price[x] for x in X_train.CatBrand ])
    cb_price_val = np.array([get_val(cb_price,x) for x in X_val.CatBrand])

    X_val['cb_price'] = [get_val(cb_price,x) for x in X_val.CatBrand ]
    
    print('RMSLE, train, cat/brand price: {:0.4}'
          .format(rmsle(X_train.price,cb_price_train)))
    print('RMSLE, test, cat/brand price: {:0.4}'
          .format(rmsle(X_val.price,cb_price_val)))


# Combining the two into a single feature gives better results: 0.61 on the training set and 0.625 on the validation set.
# 
# # Brand, Category, Shipping, and Condition

# In[19]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=5,random_state=123)
i = 0
def get_val(series,x,default):
    try:
        return series[x]
    except:
        pass
    return default

train['All4'] = ['Cat:'+c+' Brand:'+b+ \
                ' condition:'+str(i) + ' shipping:'+str(s)
                for c,b,i,s in zip(train.category_name,train.brand_name,
                                   train.item_condition_id,train.shipping)]

for train_idx,val_idx in kf.split(train):
    print('Fold {}'.format(i))
    i+=1
    cols = [1,2,3,4,5]
    X_train = train.iloc[train_idx,:]
    X_val = train.iloc[val_idx,:]

    cb_price = X_train.groupby('All4').price.mean()

    mean = train.price.mean()
    
    cb_price_train = np.array([cb_price[x] for x in X_train.All4 ])
    cb_price_val = np.array([get_val(cb_price,x,mean) for x in X_val.All4])
    
    print('RMSLE, train, all feature price: {:0.4}'
          .format(rmsle(X_train.price,cb_price_train)))
    print('RMSLE, test, all feature price: {:0.4}'
          .format(rmsle(X_val.price,cb_price_val)))


# If we combine all of our categorical features together our results continue to improve a bit, but we do start seeing a larger difference between the test and validation sets. Things are quite consistent between folds, though, so the numbers are stable. This gives an error of 0.56 on the training set and 0.60 on the test set.

# # Beyond Group Statistics
# 
# To go beyond just using group statistics, we'll have to start feature engineering. I already mentioned some ways to transform the brand and category labels that could provide some use. But, we haven't looked at all at the item description and item name. These potentially hold a lot of information, but in a fairly unstructured form.
# 
# Here, I will look at doing feature reduction on the item name. It turns out that this is not too helpful yet, but I have also not been able to do much tuning of the data.
# 
# First, let's read in the data again from the file. We'll then do a train/test split to get our training and validation sets. I'm doing this here because it turns out that some algorithms take a very long time to run so k-fold cross validation will take too long for what I want to do now. If you have a cluster of computers or some GPUs, this probably won't be a problem.
# 
# After doing the split, I will remove non-alphabetic characters, lower any capitalized letters, and then pass the text through a count vectorizer to get a sparse term frequency matrix. I've just picked a minimum document frequency without tuning at this point.

# In[20]:


train = pd.read_csv('../input/train.tsv',delimiter='\t',index_col=0)
train = train.drop('item_description',axis=1)
cvec = CountVectorizer(min_df=25,stop_words='english')

X_tr,X_te = train_test_split(train,test_size=0.3,random_state=234)

names_tr = X_tr.name
names_tr = [n.lower() for n in names_tr]
names_tr = [ re.sub(r'[^A-Za-z]',' ',n) for n in names_tr]

names_te = X_te.name
names_te = [n.lower() for n in names_te]
names_te = [ re.sub(r'[^A-Za-z]',' ',n) for n in names_te]
cvec.fit(names_tr)

X_tr_names = cvec.transform(names_tr)
X_te_names = cvec.transform(names_te)
    
print(len(cvec.vocabulary_))


# We end up with a vocabulary of 8580 words. That's far too many to directly put into most models, so I will reduce that down to 50 using truncated singular value decomposition. Again, I just chose 50 here without tuning it.

# In[21]:


svd = TruncatedSVD(n_components=50,n_iter=10)
svd.fit(X_tr_names)
X_tr_svd = svd.transform(X_tr_names)
X_te_svd = svd.transform(X_te_names)


# Now I can prepare my training and validation sets for input into machine learning models. I'll first take the categorical variables and price. Then, for the brand name and category name, I will do the following:
#   - Make list of all values with at least 10 entries (the 10 needs tuning)
#   - Replace all other values with an empty string
#   - Sort the altered names by mean price
#   - Use the sorted prices to get a map from name to an integer index
#   - Replace the names with the indices
#   
# This leaves us with a new numerical brand_name and category_name feature that is more or less an ordinal feature. The numbers now represent the typical price for a given name. Having everything ordered should allow for some machine learning algorithms (decision trees in particular) to work efficiently on these features without having to use a one hot encoder.

# In[22]:


y_tr = X_tr['price']
X_tr = X_tr.loc[:,['item_condition_id','category_name','brand_name','shipping','price']]
y_te = X_te['price']
X_te = X_te.loc[:,['item_condition_id','category_name','brand_name','shipping','price']]


# In[23]:


cat_counts = X_tr.groupby('category_name')['price'].count()
to_keep = []
for i in range(len(cat_counts)):
    if (cat_counts.iloc[i]>10):
        to_keep.append(cat_counts.index.values[i])
def filter_vals(x,alist):
    if x in alist:
        return x
    return ''
X_tr.loc[:,'category_name'] = [filter_vals(x,to_keep) for x in X_tr['category_name']]  
X_te.loc[:,'category_name'] = [filter_vals(x,to_keep) for x in X_te['category_name']]    


# In[24]:


brand_counts = X_tr.groupby('brand_name')['price'].count()
to_keep = []
for i in range(len(brand_counts)):
    if (brand_counts.iloc[i]>10):
        to_keep.append(brand_counts.index.values[i])

X_tr.loc[:,'brand_name'] = [filter_vals(x,to_keep) for x in X_tr['brand_name']]
X_te.loc[:,'brand_name'] = [filter_vals(x,to_keep) for x in X_te['brand_name']]  


# In[25]:


brands_sorted = X_tr.groupby('brand_name')['price'].mean().sort_values()
cat_sorted = X_tr.groupby('category_name')['price'].mean().sort_values()

brand_dict = {}
cat_dict = {}
for i in range(len(brands_sorted)):
    brand_dict[brands_sorted.index.values[i]] = i
    
for i in range(len(cat_sorted)):
    cat_dict[cat_sorted.index.values[i]] = i
    
X_tr['brand'] = X_tr['brand_name'].map(brand_dict)
X_tr['category'] = X_tr['category_name'].map(cat_dict)
X_te['brand'] = X_te['brand_name'].map(brand_dict)
X_te['category'] = X_te['category_name'].map(cat_dict)


# In[26]:


X_tr.head()


# In[27]:


X_tr = X_tr.loc[:,['item_condition_id','shipping','brand','category']]
X_te = X_te.loc[:,['item_condition_id','shipping','brand','category']]


# Now that we have numerical versions of all our categorical features, we can combine them with the SVD results into a final feature set, now with 54 features.

# In[28]:


X_tr_fin = np.concatenate((X_tr,X_tr_svd),axis=1)
X_te_fin = np.concatenate((X_te,X_te_svd),axis=1)


# # Random Forest Regression Model
# 
# We can now start analyzing the data. I'll use a random forest model here. Because we have many categorical features, many common models such as linear regression will not work properly, at least without expanding the categorical features into one hot encoded numbers. Decision trees are able to deal with this sort of data, so a random forest model should be a decent choice.
# 
# It turns out that it takes nearly an hour to train the model on my laptop, so I won't attempt to do any tuning here. I'm also reducing the size of the set to make sure things finish within Kaggle's time limit. In my full version, I use a maximum depth of 12.

# In[29]:


from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor(n_estimators=50,min_samples_leaf=10,max_depth=10)
n_reduced = int(2./3*len(X_tr_fin))
X_tr_fin = X_tr_fin[:n_reduced,:]
y_tr = y_tr.iloc[:n_reduced]
rfc.fit(X_tr_fin,y_tr)


# In[30]:


def rmsle(ytrue,y):
    return np.sqrt(mean_squared_log_error(ytrue,y))

y_tr_pred = rfc.predict(X_tr_fin)
score_tr = rmsle(y_tr,y_tr_pred)

y_te_pred = rfc.predict(X_te_fin)
score_te = rmsle(y_te,y_te_pred)
print(score_tr)
print(score_te)


# If trained on the full training set in our training/validation split the model only gained about 0.01 from our best group average model. That is a little disappointing. We see good agreement between the training and validation score, so the model does seem to be generalizing well. Overfitting is always a danger with decision trees. The random forest method helps avoid this, as do our choices of hyperparameters, which limit the minimum size of a split and the maximum depth of the tree.

# In[31]:


rfc.feature_importances_


# Looking at the feature importances, we see that our original feature set still provides most of our predictive power. Brand is most important by far, then category, then item condition. It is interesting that item condition is 3rd, since it didn't look very useful when we looked at its basic statistics. After that, there are several SVD features with an importance of at least 0.01.
# 
# But, this result does not mean that we haven't constructed better features. Tuning may give significant improvements in the results but would take a long time without having a good computing setup.

# In[32]:




