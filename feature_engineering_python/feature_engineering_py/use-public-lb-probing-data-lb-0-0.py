#!/usr/bin/env python
# coding: utf-8

# # Train With Public LB Probing Data
# This notebook demonstrates that using the targets extracted from public test data (via probing) to train our model does not improve our private LB score. In version 1 of this notebook, we verify that the list of probed public test targets are correct. Then in version 2 we train and infer CatBoost with only train data. Then in version 3 we train CatBoost with both train data plus public test data. We observe that using the probed public test does not help.

# In[1]:


import pandas as pd, numpy as np
from catboost import CatBoostClassifier

train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
greeks['epsilon_date'] = pd.to_datetime(greeks['Epsilon'].replace('Unknown', None), format='%m/%d/%Y')

# mega feature engineering )
train.EJ = train.EJ.eq('B').astype('int')
test.EJ = test.EJ.eq('B').astype('int')

# just use all features
features = [i for i in train.columns if i not in ['Id', 'Class']]

# drop null and old data
train = train.merge(greeks, on='Id')
train = train[train['epsilon_date'].dt.year >= 2017]


# # Probed Public Test Labels

# In[2]:


test = test.sort_values('Id')

PUBLIC1 = np.array([
        14, 26, 32, 42, 71, 101, 103, 105, 106, 170, 173, 176, 179, 187, 
        194, 210, 238, 246, 258, 264, 270, 302, 329, 338, 355, 359])
PUBLIC0 = np.array([  
         1,   2,   6,   9,  10,  12,  15,  16,  20,  23,  24,  27,  28,
        34,  37,  39,  44,  45,  46,  50,  53,  55,  57,  58,  66,  69,
        72,  76,  79,  83,  85,  86,  87,  90,  91,  92,  95,  96, 100,
       107, 109, 111, 112, 113, 114, 123, 125, 126, 129, 132, 133, 134,
       135, 145, 148, 150, 153, 156, 157, 158, 163, 164, 166, 174, 177,
       178, 181, 182, 183, 185, 188, 190, 192, 195, 196, 197, 208, 211,
       214, 217, 219, 221, 222, 224, 225, 227, 229, 232, 234, 236, 237,
       239, 240, 241, 245, 249, 254, 257, 259, 260, 266, 269, 280, 282,
       286, 289, 295, 299, 306, 307, 309, 315, 321, 323, 326, 327, 328,
       330, 333, 343, 344, 346, 351, 352])

pred = np.array([0.5]*len(test))

PUBLIC1 = PUBLIC1[ PUBLIC1<len(test) ]
PUBLIC0 = PUBLIC0[ PUBLIC0<len(test) ]

pred[PUBLIC1] = 1
pred[PUBLIC0] = 0


# # Add Probed Data To Train Data

# In[3]:


test['Class'] = pred
more_data = test.loc[test.Class != 0.5]
print('Shape of more data:', more_data.shape )
more_data.head()


# In[4]:


print('Shape of train before:', train.shape )
X_train = pd.concat([train[features],more_data[features]],axis=0)
y_train = pd.concat([train['Class'],more_data['Class']],axis=0)
print('Shape of train after:', X_train.shape )


# # Train and Predict with CatBoost
# We only use CatBoost in versions 2 and 3 of this notebook

# In[5]:


if 1:
    model = CatBoostClassifier(verbose=100)
    model.fit(X_train, y_train)


# In[6]:


# POST PROCESS TO BALANCE PREDICTIONS
# from Samuel @muelsamu
# https://www.kaggle.com/code/muelsamu/simple-tabpfn-approach-for-score-of-15-in-1-min
if 1:
    p1 = model.predict_proba(test[features])[:, 1]
    p0 = 1 - p1
    p = np.stack([p0, p1]).T
    class_0_est_instances = p[:,0].sum()
    others_est_instances = p[:,1:].sum()
    pred = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
    pred = pred / np.sum(pred, axis=1, keepdims=1)
    pred = pred[:,1]


# # Save Predictions to Submission CSV

# In[7]:


submission = pd.DataFrame(test["Id"], columns=["Id"])
submission["class_1"] = pred
submission["class_0"] = 1 - submission["class_1"]
submission.to_csv('submission.csv', index=False)
submission.head()

