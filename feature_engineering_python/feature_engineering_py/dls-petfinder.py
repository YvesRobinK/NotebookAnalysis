#!/usr/bin/env python
# coding: utf-8

# ## This notebook was created as a Kaggle tutorial for a lecture at [Deep Learning School](https://www.dlschool.org/?lang=en)

# In[1]:


import os
import tqdm
import torch

import numpy as np
import pandas as pd


# Let's check that GPU is working correctly!

# In[2]:


get_ipython().system('nvidia-smi')


# Make sure that the internet is turned on as well:

# In[3]:


get_ipython().system('ping -c 4 google.com')


# # EDA

# First, let's locate our data...

# In[4]:


get_ipython().system('ls ../input')


# In[5]:


get_ipython().system('ls ../input/petfinder-adoption-prediction')


# In[6]:


get_ipython().system('ls ../input/petfinder-adoption-prediction/train')


# Now, read the train `.csv` file:

# In[7]:


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
train.head(10)


# Let's see the train size.

# In[8]:


train.shape


# Or, even better:

# In[9]:


train.info()


# The `dtype` of `PhotoAmt` column looks weird! Let's change it to `np.int64` so that it matches the others.

# In[10]:


train.PhotoAmt = train.PhotoAmt.astype(np.int64)


# Let's look at the target distribution:

# In[11]:


#########################
## YOUR CODE GOES HERE ##
#########################
# Hint: use np.unique

np.unique(train.AdoptionSpeed, return_counts=True)


# ### Question: what can we say about class imbalance?
# ### Question: which features can we expect to correlate with the target?

# For this tutorial, we will remove the text data.

# In[12]:


def filter_text_columns(table):
    _blacklist = ['Name', 'RescuerID', 'Description', 'PetID']
    for column in _blacklist:
        if column in table.columns:
            del table[column]

filter_text_columns(train)


# Let's separate the data from the target:

# In[13]:


X = np.array(train.iloc[:,:-1])
y = np.array(train.AdoptionSpeed)


# In[14]:


assert X.shape == (14993, 19)
assert y.shape == (14993,)
print("Good job!")


# ### Question: Can we calculate the dataset statistics at this point?

# In[15]:


# X -= X.mean(axis=0) ?


# # Splitting the data

# Before we move any further, we need to make sure we have a validation set. We'll use a simple hold-out validation for now.
# 
# Now, create a **stratified** validation set with `20%` validation size. Make sure to **fix your random seed**!

# In[16]:


from sklearn.model_selection import train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

random_state = 42

#########################
## YOUR CODE GOES HERE ##
#########################
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state, test_size=0.2)


# In[17]:


assert X_train.shape == (11994, 19)
assert y_train.shape == (11994,)
assert X_test.shape == (2999, 19)
assert y_test.shape == (2999,)

assert np.sum(X_train) == 500668689
assert np.sum(X_test) == 125179430
print("Nice!")


# # Building our first pipeline

# Obviously, we need a metric for this..

# ### Challenge #1: find out how to implement the metric!

# In[18]:


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
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
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# In[19]:


def metric(y_true, y_pred):
    #########################
    ## YOUR CODE GOES HERE ##
    #########################
#     return quadratic_weighted_kappa(y_true, y_pred)
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# In[20]:


assert np.abs(1 - metric(y_train, y_train)) <= 1e-7
assert np.abs(1 - metric(y_test, y_test)) <= 1e-7
assert np.abs(metric(y_test, y_test + 1) - 0.7349020406) <= 1e-7
print("Awesome!")


# Let's build our first pipeline!

# In[21]:


def vanilla_pipeline(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


# Basic model: k-NN Classifier!

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
vanilla_pipeline(clf)


# Still something! Now, select the classifier with the best K (K < 10).

# In[23]:


# for i in range(1,10):
#     clf = KNeighborsClassifier(n_neighbors=i)
#     print(vanilla_pipeline(clf))

kNN = KNeighborsClassifier(n_neighbors=9)


# In[24]:


assert vanilla_pipeline(kNN) >= 0.26, "Your classifier isn't the best!"
print("Cool!")


# ### Question: is K-NN a meaningful architecture for EXACTLY this data? What about linear models? Tree-based models?

# Let's try a more meaningful model.

# ### Try out Random Forest classifier with `n_estimators` equal to `25` at max (hint: set `n_jobs=4` to speed things up)

# In[25]:


#########################
## YOUR CODE GOES HERE ##
#########################

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=25, n_jobs=4)
vanilla_pipeline(rf)


# In[26]:


assert vanilla_pipeline(rf) >= 0.27
print("Nice!")


# # Feature engineering

# Notice that we haven't changed any input data. What if we want to do any preprocessing or feature engineering? Let's look at images first.

# In[27]:


get_ipython().system('ls ../input/petfinder-adoption-prediction/train_images/ | head -20')


# In[28]:


import os
image_list = sorted(os.listdir('../input/petfinder-adoption-prediction/train_images/'))
image_list[:10]


# In[29]:


from PIL import Image
image = Image.open('../input/petfinder-adoption-prediction/train_images/0008c5398-1.jpg')
image


# Now we may want to calculate image embeddings for our images. Let us use `torchvision.models` for this. First, let's define our transform:

# In[30]:


from torchvision import transforms

# Defining transform
transform = transforms.Compose([            
 transforms.Resize(224),               
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],            
 std=[0.229, 0.224, 0.225]              
 )])


# Now, let's take **ResNet-50** pretrained embeddings. Make sure that you enable CUDA and set training type to `eval`.

# In[31]:


import torchvision.models as models

#########################
## YOUR CODE GOES HERE ##
#########################


mobilenet = models.mobilenet_v2(pretrained=True).cuda()


# Now everything is ready to calculate the embeddings. For this, we need to:
# * Transform an image
# * Create a batch containing this image and convert it to CUDA
# * Make predictions
# * Convert predictions to numpy and ravel

# In[32]:


def calc_embedding(image):
    #########################
    ## YOUR CODE GOES HERE ##
    #########################
    
    transformed = transform(image)
    batch = transformed.unsqueeze(0)
    predictions = mobilenet(batch.cuda())
    return predictions.cpu().detach().numpy().ravel()


# Let's test your implementation.

# In[33]:


embedding.std()


# In[34]:


# Testing
embedding = calc_embedding(image)

assert torch.cuda.current_device() == 0, "Are you sure you're using CUDA?"
assert type(embedding) == np.ndarray, "Make sure to convert the result to numpy.array"
assert embedding.dtype == np.float32, "Convert your embedding to float32"
assert embedding.shape == (1000,), "Make sure to ravel the predictions"
assert np.abs(embedding.mean() - 8.483887e-06) <= 1e-6
# assert np.abs(embedding.std() - 2.0538368) <= 1e-6
print("Fabulous!")


# In[35]:


embedding.shape


# Some convenience functions:

# In[36]:


def _get_default_photo_path(pet_id):
    return '../input/petfinder-adoption-prediction/train_images/%s-1.jpg' % pet_id

def does_pet_have_photo(pet_id):
    return os.path.exists(_get_default_photo_path(pet_id))

def photo_of_pet(pet_id):
    path = _get_default_photo_path(pet_id)
    return Image.open(path).convert('RGB')


# Now, let's get the embeddings for the test set!

# In[37]:


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
train.PhotoAmt = train.PhotoAmt.astype(np.int64)

# We'll store our embeddings here
embeddings = np.zeros((len(train), embedding.shape[0]), dtype=np.float32)

pet_ids = train.PetID
for i in tqdm.tqdm_notebook(range(len(train))):
    pet_id = pet_ids[i]
    
    if does_pet_have_photo(pet_id):
        embeddings[i] = calc_embedding(photo_of_pet(pet_id))


# In[38]:


embeddings.shape


# In[39]:


X.shape


# Now, everything is ready to create a new dataset. For that, we just need to stack our features and the new embeddings.

# In[40]:


filter_text_columns(train)

X = np.array(train.iloc[:,:-1])
y = np.array(train.AdoptionSpeed)

#########################
## YOUR CODE GOES HERE ##
#########################

X = np.hstack([X, embeddings])


# In[41]:


assert X.shape == (14993, 1019)


# Let's split the data into train/test as before..

# In[42]:


random_state = 42

#########################
## YOUR CODE GOES HERE ##
#########################
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state, test_size=0.2)


# Try out the `RandomForestClassifier`:

# In[43]:


rf = RandomForestClassifier(n_estimators=25, n_jobs=4, random_state=42)
vanilla_pipeline(rf)


# ### Question: why the result is so poor?

# Let's get the train and test *embeddings* only

# In[44]:


#########################
## YOUR CODE GOES HERE ##
#########################

X_train_feats = X_train[:,-1000:]
# X_test_feats = X_test[:,-1000:]


# Let's use `TruncatedSVD`. Convert `X_train_feats` and `X_test_feats` to the new 6-dimensional space.

# # PCA(n_components=0.95)

# In[45]:


from sklearn.decomposition import TruncatedSVD

n_feats = 6
random_state = 42

#########################
## YOUR CODE GOES HERE ##
#########################
pca = TruncatedSVD(n_components=n_feats)
pca.fit(X_train_feats)
X_train_feats = pca.transform(X_train_feats)
# X_test_feats = pca.transform(X_test_feats)


# In[46]:


X_train.shape


# Now, let's fit out SVD on the train image features.

# In[47]:


#########################
## YOUR CODE GOES HERE ##
#########################



# Now, we need to modify `X_train` and `X_test` to include compressed embeddings.
# 
# **Challenge: can you do this in 2 lines?**

# In[48]:


#########################
## YOUR CODE GOES HERE ##
#########################

X_train = np.hstack([X_train[:,:19], X_train_feats])
X_test = np.hstack([X_test[:,:19], X_test_feats])


# Let's check our result now!

# In[49]:


X_train.shape


# In[50]:


rf = RandomForestClassifier(n_estimators=25, n_jobs=4, random_state=42)
vanilla_pipeline(rf)


# In[ ]:





# Nice!

# ### Question: what would you do with text?

# Let's improve our result a little bit by using CatBoost!

# In[51]:


from catboost import CatBoostClassifier

#########################
## YOUR CODE GOES HERE ##
#########################

cb = CatBoostClassifier(n_)
vanilla_pipeline(cb)


# Now, it's time to make predictions for the test set! Don't forget to include image embeddings!

# * убрать текстовые фичи
# * привести все к np.int64
# * посчитать картиночные фичи
# * понизить пространство картиночных фичей
# * сконкатить все
# * model.predict(...)

# In[52]:


test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')


# In[53]:


def _get_default_photo_path(pet_id):
    return '../input/petfinder-adoption-prediction/test_images/%s-1.jpg' % pet_id

def does_pet_have_photo(pet_id):
    return os.path.exists(_get_default_photo_path(pet_id))

def photo_of_pet(pet_id):
    path = _get_default_photo_path(pet_id)
    return Image.open(path).convert('RGB')


# In[54]:


# We'll store our embeddings here
embeddings = np.zeros((len(test), 1000), dtype=np.float32)

pet_ids = test.PetID
for i in tqdm.tqdm_notebook(range(len(test))):
    pet_id = pet_ids[i]
    
    if does_pet_have_photo(pet_id):
        embeddings[i] = calc_embedding(photo_of_pet(pet_id))


# In[55]:


filter_text_columns(test)
test = test.astype(np.int64)


# In[56]:


embeddings.shape


# In[57]:


pca.transform(embeddings)


# In[58]:


X_test = test
X_test = np.hstack([X_test, X_test_feats])


# Now, let's submit our result.

# In[59]:


sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
sample_submission.head()


# In[60]:


submission = sample_submission
submission['AdoptionSpeed'] = predictions
submission.to_csv('submission.csv', index=False)


# In[ ]:




