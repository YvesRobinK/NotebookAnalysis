#!/usr/bin/env python
# coding: utf-8

# # Just A Theory...
# 
# When this competition launched first, I though that we were working with pre-encoded categorical data and I created this notebook [Lessons Learned from Previous Cat Competitions](https://www.kaggle.com/iamleonie/lessons-learned-from-previous-cat-competitions). Now a few days have passed and I am beginning to think that we might not be working with categorical data at all.
# 
# Let's have a look at the data again. Below you can see that:
# - There are 50 features, which is more than in previous TPS Challenges according to [this disucssion](https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/236128)
# - The data is **sparse**, which means that we mostly have the value zero
# - There are no missing values
# - Most features are positive integers. Some can also be negative.
# 
# Hm... where have we seen this type of data before? This seems oddly similar to something you might have seen previously in **Natural Language Processing (NLP)** problems. The data almost looks like it is a **Bag of Words (BoW)** or **Document Term Matrix (DTM)** representation.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Encoders
import category_encoders as ce

# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, make_scorer

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("../input/tabular-playground-series-may-2021/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2021/test.csv")

# Drop id column
#train_df.drop('id', axis=1, inplace=True)
#test_df.drop('id', axis=1, inplace=True)

# Only for visualization purposes
temp = train_df.copy()
temp.columns = list(['id']) + list([f"f{c}" for c in range(50)]) + list(['target'])
feature_cols = temp.columns[temp.columns.str.startswith('f')]

display(temp[feature_cols].head(10).style.background_gradient(cmap='Greens_r', vmin=-1, vmax=1))
del(temp)


# # Bag of Words / Document Term Matrix
# 
# The BoW or DTM is a way to represent text in a numerical way. Since this data is based on eCommerce listings, I will try to explain it with some eCommerce related examples:
# Let's say you have two product reviews:
# > "This book was a great read." - Review 1
# 
# > "I read this book in a day. Easy read." - Review 2
# 
# Now you take all unique words from both reviews and make them as features: {'this', 'book', 'was', 'great', 'read', 'i', 'in', 'a', 'day', 'easy'}. And then you assign the number of times this word appears in the review to the feature as follows:
# 
# |          | this | book | was | great |read | i | in | a | day |easy |
# |:--------:|------|------|-----|-------|-------|---|----|---|-----|-----|
# | Review 1 | 1    | 1    | 1   | 1     |  1 |0 | 0  | 0 | 0   |0   |
# | Review 2 | 1    | 1    | 0   | 0     | 2 | 1 | 1  | 1 | 1   |1   |
# 
# Now this looks quite similar to what we have above. 
# 
# This is just a **wild guess but maybe we are trying to classify an eCommerce product given its product reviews**. For example, maybe we are trying to classify whether a book is a thriller, drama, novel or non-fiction based on its reviews. Of course this could be completely wrong but let's try this hypothesis.
# 
# Some notes of precaution:
# * One point that does not fit this hypothesis fully at the moment is that we also have some features with negative values. BoW usually do not have negative values (how can a word appear negative times in a text?"). 
# * As [@melanie7744](https://www.kaggle.com/melanie7744) has mentioned in the comments, 50 features for BoW seems a little bit too small for a vocabulary.
# * Some "words" have a high frequency in the "sentences". For example, feature_43 is appearing 21 times in the first sample. I cannot imagine what kind of "text" that would be.
# 
# **If you have an idea how this could relate to the idea, please share it!**

# # Exploratory Data Analysis
# We have seen many great notebooks in this challenge sofar showcasing dimensionality reductions for the features using UMAP and t-SNE. I want to take a slightly different approach and use dimensionality reduction to see which features are similar to each other.
# 
# From below plot you can see that most features are close together in one big cluster and that feature_38 and feature_42 are both far off to the sides. This might be a hint to evaluate these features a little bit more in depth.
# 
# Also, we can see that classes 1, 3, 4 are in close approximity to each other while class 2 is a little bit further away from the other three classes.

# In[2]:


# One Hot Encode target variable
ohe = ce.OneHotEncoder(handle_unknown='value', use_cat_names=True)
OH_target = pd.DataFrame(ohe.fit_transform(train_df['target'].astype(str)))

display(OH_target.head())

# Merge new OH encoded target to train_df 
train_df = pd.concat([train_df, OH_target], axis=1)


# In[3]:


from sklearn.manifold import TSNE
feature_cols = train_df.columns[train_df.columns.str.startswith('feature_') | train_df.columns.str.startswith('target_')]

X_embedded = TSNE(n_components=2, init='pca', perplexity = 5, random_state=42).fit_transform(train_df[feature_cols].T.values)

plt.figure(figsize=(10, 10))
x = X_embedded[:,0]
y = X_embedded[:,1]
sns.scatterplot(x[:-4], y[:-4], color='green')
sns.scatterplot(x[-4:], y[-4:], color='red')

plt.title('Clusters of similar features', fontsize=14)
plt.grid(True)
    
for i, word in enumerate(feature_cols):
    if word.startswith('feature'):
        plt.annotate(f"f{word.split('_')[1]}", xy=(x[i], y[i]) , size=10,  alpha=0.8, xytext=(5, 2), 
                 textcoords='offset points', ha='left', va='bottom')
    else: 
        plt.annotate(f"c{word.split('_')[2]}", xy=(x[i], y[i]) , size=13,  alpha=0.8, xytext=(5, 2), 
                 textcoords='offset points', ha='left', va='bottom')
         
plt.show()


# As a first experiment, let's see what happens if we drop feature_38 and/or feature_42.
# * drop feature_19 only: slightly increased performance
# * drop feature_38 only: decreased performance
# * drop feature_42 only: decreased performance
# * drop feature_38 and feature_42: slightly increased performance
# 

# In[4]:


# Experiment

#drop_cols = ['feature_19', 'feature_38', 'feature_42']
#train_df.drop(drop_cols, axis=1, inplace=True)
#test_df.drop(drop_cols, axis=1, inplace=True)


# # Feature Engineering
# 
# ## Length and Count Features
# 
# Common length and count features for NLP problems are:
# * word count
# * character count
# * sentence count
# * average word length
# * average sentence length
# 
# However, since we only have the BoW, we can only create a new feature for word count. For the remaining length and count features, we would need the original texts.

# In[5]:


feature_cols = train_df.columns[train_df.columns.str.startswith('feature_')]

train_df['feature_number_of_words'] = train_df[feature_cols].sum(axis=1)
test_df['feature_number_of_words'] = test_df[feature_cols].sum(axis=1)


# In[6]:


#train_df['feature_number_of_words'] = np.log(train_df.feature_number_of_words)
sns.kdeplot(train_df[train_df.target == 'Class_1']['feature_number_of_words'], label='Class_1')
sns.kdeplot(train_df[train_df.target == 'Class_2']['feature_number_of_words'], label='Class_2')
sns.kdeplot(train_df[train_df.target == 'Class_3']['feature_number_of_words'], label='Class_3')
sns.kdeplot(train_df[train_df.target == 'Class_4']['feature_number_of_words'], label='Class_4')
plt.legend()
plt.show()


# There are 2 datapoints in the training data with 0 words. Both are categorized as different classes. Since this does not happen in the test data, we will for now assume that this might be some faulty data and we will drop these from the training data.

# In[7]:


display(train_df[train_df.feature_number_of_words == 0])
train_df = train_df[train_df.feature_number_of_words != 0].reset_index(drop=True)


# ## Term Frequency—Inverse Dense Frequency (TF-IDF) 
# 
# The following explanation is largely based on this [medium blog post](https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76).
# 
# ### Term Frequency
# 
# > The number of times a word appears in a document ($n_{i,j}$) divded by the total number of words ($\sum_k n_{i,j}$) in the document. 
# 
# $tf_{i,j} = \dfrac{n_{i,j}}{\sum_k n_{i,j}}$
# 
# ### Inverse Data Frequency
# 
# > The log of the number of documents ($N$) divided by the number of documents that contain the word $w$ ($df_t$). 
# 
# $idf(w)=\log(\dfrac{N}{df_t})$

# In[8]:


idf = np.log(len(train_df) / (train_df != 0).sum(axis=0))
idf.head(5)


# > Lastly, the TF-IDF is simply the TF multiplied by IDF.
# 
# TF-IDF = $tf_{i,j} \cdot idf(w)$

# In[9]:


for f in feature_cols:
    train_df[f"{f}_tfidf"] = train_df[f] / train_df['feature_number_of_words'] * idf[f]
    test_df[f"{f}_tfidf"] = test_df[f] / test_df['feature_number_of_words'] * idf[f]
    train_df[f"{f}_tfidf"] = train_df[f"{f}_tfidf"].fillna(0)
    test_df[f"{f}_tfidf"] = test_df[f"{f}_tfidf"].fillna(0)
    
display(train_df[train_df.columns[train_df.columns.str.endswith('tfidf')]].head(5))


# # Baseline
# 
# This baseline is copied from my initial notebook for this competition ([Lessons Learned from Previous Cat Competitions](https://www.kaggle.com/iamleonie/lessons-learned-from-previous-cat-competitions))

# In[10]:


feature_cols = train_df.columns[train_df.columns.str.startswith('feature_')]
X = train_df[feature_cols]
y = train_df['target']
X_test = test_df[feature_cols]

N_SPLITS = 5

display(X.head(5))


# In[11]:


# Initialize variables
y_oof_pred = np.zeros((len(X), 4))
y_test_pred = np.zeros((len(X_test), 4))

kf = StratifiedKFold(n_splits = N_SPLITS)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):    
    # Prepare training and validation data
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)  

    # Define model
    model = LogisticRegression(random_state = 42, 
                               #C = 0.8, 
                              # max_iter = 100, # default value
                              )
    model.fit(X_train, y_train)
    
    # Calculate evaluation metric
    y_val_pred = model.predict_proba(X_val)

    print(f"Fold {fold + 1} Log Loss: {log_loss(y_val, y_val_pred)}")

    # Make predictions
    y_oof_pred[val_idx] = y_val_pred
    y_test_pred += model.predict_proba(X_test)


# Calculate evaluation metric for out of fold validation set
y_test_pred = y_test_pred / N_SPLITS

print(f"Overall OOF Log Loss: {log_loss(y, y_oof_pred)}")


# In[12]:


# Visualize
y_pred = pd.Series(y_test_pred.argmax(axis=1)).replace({0 : 'Class_1', 1 : 'Class_2', 2 : 'Class_3', 3 : 'Class_4'})

print(y_pred.value_counts())

fig = plt.figure(figsize=(8,4))
sns.countplot(y_pred,
              palette='Greens', 
              order=['Class_1', 'Class_2', 'Class_3', 'Class_4'], )
plt.show()


# In[13]:


submission_df = pd.DataFrame(y_test_pred)
submission_df.columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4']
submission_df['id'] = test_df['id']
submission_df = submission_df[['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4']]

submission_df.to_csv("submission.csv", index=False)
display(submission_df.head())


# So far, this approach is not working very well... 
# 
# Further experiments are on going and will be updated shortly.
