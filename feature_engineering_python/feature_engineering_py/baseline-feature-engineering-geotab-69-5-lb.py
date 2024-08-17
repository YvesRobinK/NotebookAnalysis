#!/usr/bin/env python
# coding: utf-8

# * Forked from : https://www.kaggle.com/pulkitmehtawork1985/beating-benchmark
# * Copies feature code over from my other kernel; https://www.kaggle.com/danofer/basic-features-geotab-intersections
# 
# * V6 - try  a multitask model in addition to a model per target. Likely to have worse performance, but will be faster

# In[1]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Load Data

train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv").sample(frac=0.15,random_state=42)#,nrows=123456)
test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")


# ## Data Cleaning
# 

# In[3]:


train.nunique()


# In[4]:


print(train["City"].unique())
print(test["City"].unique())


# In[5]:


# test.groupby(["City"]).apply(np.unique)
test.groupby(["City"]).nunique()


# In[6]:


train.isna().sum(axis=0)


# In[7]:


test.isna().sum(axis=0)


# ## Add features
# 
# ##### turn direction: 
# The cardinal directions can be expressed using the equation: $$ \frac{\theta}{\pi} $$
# 
# Where $\theta$ is the angle between the direction we want to encode and the north compass direction, measured clockwise.
# 
# * This is an **important** feature, as shown by janlauge here : https://www.kaggle.com/janlauge/intersection-congestion-eda
# 
# * We can fill in this code in python (e.g. based on: https://www.analytics-link.com/single-post/2018/08/21/Calculating-the-compass-direction-between-two-points-in-Python , https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python , https://gist.github.com/RobertSudwarts/acf8df23a16afdb5837f ) 
# 
# * TODO: circularize / use angles

# In[8]:


directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}


# In[9]:


train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)


# In[10]:


train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

test['diffHeading'] = test['EntryHeading']-test['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(10)


# In[11]:


### code if we wanted the diffs, without changing the raw variables:

# train['diffHeading'] = train['ExitHeading'].map(directions) - train['EntryHeading'].map(directions)
# test['diffHeading'] = test['ExitHeading'].map(directions) - test['EntryHeading'].map(directions)


# In[12]:


train.head()


# * entering and exiting on same street
# * todo: clean text, check if on same boulevard, etc' 

# In[13]:


train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)
test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)


# ### Skip OHE intersections for now - memory issues
# * Intersection IDs aren't unique  etween cities - so we'll make new ones
# 
# * Running fit on just train reveals that **the test data has a "novel" city + intersection!** ( '3Atlanta'!) (We will fix this)
#      * Means we need to be careful when OHEing the data
#      
#  * There are 2,796 intersections, more if we count unique by city (~4K) = many, many columns. gave me memory issues when doing one hot encoding
#      * Could try count or target mean encoding. 
#      
# * For now - ordinal encoding

# In[14]:


le = preprocessing.LabelEncoder()
# le = preprocessing.OneHotEncoder(handle_unknown="ignore") # will have all zeros for novel categoricals, [can't do drop first due to nans issue , otherwise we'd  drop first value to avoid colinearity


# In[15]:


train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]

print(train["Intersection"].sample(6).values)


# In[16]:


train.head()


# In[17]:


test.head()


# In[18]:


# pd.concat([train,le.transform(train["Intersection"].values.reshape(-1,1)).toarray()],axis=1).head()


# #### with ordinal encoder - ideally we'd encode all the "new" cols with a single missing value, but it doesn't really matter given that they're Out of Distribution anyway (no such values in train). 
# * So we'll fit on train+Test in order to avoid encoding errors - when using the ordinal encoder! (LEss of a n issue with OHE)

# In[19]:


pd.concat([train["Intersection"],test["Intersection"]],axis=0).drop_duplicates().values


# In[20]:


le.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
train["Intersection"] = le.transform(train["Intersection"])
test["Intersection"] = le.transform(test["Intersection"])


# In[21]:


train.head()


# ### ORIG  OneHotEncode
# ##### We could Create one hot encoding for entry , exit direction fields - but may make more sense to leave them as continous
# 
# 
# * Intersection ID is only unique within a city

# In[22]:


pd.get_dummies(train["City"],dummy_na=False, drop_first=False).head()


# In[23]:


# pd.get_dummies(train[["EntryHeading","ExitHeading","City"]].head(),prefix = {"EntryHeading":'en',"ExitHeading":"ex","City":"city"})


# In[24]:


train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)


# In[25]:


train.shape,test.shape


# In[26]:


test.head()


# In[27]:


train.columns


#  #### Approach: We will make 6 predictions based on features we derived - IntersectionId , Hour , Weekend , Month , entry & exit directions .
#  * Target variables will be TotalTimeStopped_p20 ,TotalTimeStopped_p50,TotalTimeStopped_p80,DistanceToFirstStop_p20,DistanceToFirstStop_p50,DistanceToFirstStop_p80 .
#  
#  * I leave in the original IntersectionId just in case there's meaning accidentally encoded in the numbers

# In[28]:


FEAT_COLS = ["IntersectionId",
             'Intersection',
           'diffHeading',  'same_street_exact',
           "Hour","Weekend","Month",
          'Latitude', 'Longitude',
          'EntryHeading', 'ExitHeading',
            'Atlanta', 'Boston', 'Chicago',
       'Philadelphia']


# In[29]:


train.head()


# In[30]:


train.columns


# In[31]:


X = train[FEAT_COLS]
y1 = train["TotalTimeStopped_p20"]
y2 = train["TotalTimeStopped_p50"]
y3 = train["TotalTimeStopped_p80"]
y4 = train["DistanceToFirstStop_p20"]
y5 = train["DistanceToFirstStop_p50"]
y6 = train["DistanceToFirstStop_p80"]


# In[32]:


y = train[['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',
        'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]


# In[33]:


testX = test[FEAT_COLS]


# In[34]:


## kaggle kernel performance can be very unstable when trying to use miltuiprocessing
# lr = LinearRegression()
lr = RandomForestRegressor(n_estimators=100,min_samples_split=3)#,n_jobs=3) #different default hyperparams, not necessarily any better


# In[35]:


## Original: model + prediction per target
#############

lr.fit(X,y1)
pred1 = lr.predict(testX)
lr.fit(X,y2)
pred2 = lr.predict(testX)
lr.fit(X,y3)
pred3 = lr.predict(testX)
lr.fit(X,y4)
pred4 = lr.predict(testX)
lr.fit(X,y5)
pred5 = lr.predict(testX)
lr.fit(X,y6)
pred6 = lr.predict(testX)


# Appending all predictions
all_preds = []
for i in range(len(pred1)):
    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:
        all_preds.append(j[i])   

sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
sub["Target"] = all_preds
sub.to_csv("benchmark_beat_rfr_multimodels.csv",index = False)

print(len(all_preds))


# * ALT : multitask model

# In[36]:


## New/Alt: multitask -  model for all targets

lr.fit(X,y)
print("fitted")

all_preds = lr.predict(testX)


# In[37]:


## convert list of lists to format required for submissions
print(all_preds[0])

s = pd.Series(list(all_preds) )
all_preds = pd.Series.explode(s)

print(len(all_preds))
print(all_preds[0])


# In[38]:


sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
print(sub.shape)
sub.head()


# In[39]:


sub["Target"] = all_preds.values
sub.sample(5)


# In[40]:


sub.to_csv("benchmark_beat_rfr_multitask.csv",index = False)


# # Export featurized data
# 
# * Uncomment this to get the features exported for further use. 

# In[41]:


train.drop("Path",axis=1).to_csv("train_danFeatsV1.csv.gz",index = False,compression="gzip")
test.drop("Path",axis=1).to_csv("test_danFeatsV1.csv.gz",index = False,compression="gzip")

