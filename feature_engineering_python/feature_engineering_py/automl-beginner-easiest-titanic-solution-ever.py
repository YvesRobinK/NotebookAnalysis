#!/usr/bin/env python
# coding: utf-8

# 
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> Titanic using PyCaret</p>

# So you have a Kaggle account,what next?
# 
# What if I tell you you can create your very first submission in less than 100 lines of code? Perhaps you're looking to become a contributor from a Novice.
# 
# No,I'm not talking the usual Logistic regression. I'm talking advanced Kaggle concepts like Feature engineering,Blending,Stackimg and Ensembling?
# 
# Welcome Pycaret,a low Code library developed by Moez Ali,which helps professional data scientists develop prototypes quickly with very few lines of code.
# 
# It provides a great starting point to rule out what works for your data and what doesn't,so I highly recommend this.
# In this code, We will read the data and create models and final predictions.
# I do recommend reading the official documentation while following along,and typing your own code by reading this notebook.
# 
# *If you find this useful, Consider upvoting .If you have any feedbacks, please leave it in the comments.*

# * [1. Data Dictionary](#1)
#     
# * [2. Feature Engineering](#2)
#     
# * [3. Setting up Pycaret](#3)  
#     
# * [4. Model comparison](#4)  
#     
# * [5. Model selection](#5) 
# 
# * [6. Model tuning](#6) 
# 
# * [7. Model ensembling](#7) 
# 
# * [8. References](#9) 

# # <a id="1"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> Data Dictionary</p>

# * survival - Survival (0 = No; 1 = Yes)
# * class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# * name - Name
# * sex - Sex
# * age - Age
# * sibsp - Number of Siblings/Spouses Aboard
# * parch - Number of Parents/Children Aboard
# * ticket - Ticket Number
# * fare - Passenger Fare
# * cabin - Cabin
# * embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[1]:


import pandas as pd 


# In[2]:


# #Pycaret needs to be installed
get_ipython().system('pip install pycaret')


# In[3]:


#Let's read the data
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# # <a id="2"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;">Feature Engineering</p>

# **Feature engineering using these rules,and a few mentioned in the notebook in the reference:**
# 
# Predict live for all males titled “Master” whose entire family, excluding adult males, all live.
# Predict die for all females whose entire family, excluding adult males, all die.

# In[4]:


train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())


# In[5]:


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}


# In[6]:


train['title']=train.title.map(newtitles)
test['title']=test.title.map(newtitles)


# In[7]:


train['Relatives']=train.SibSp+train.Parch
test['Relatives']=test.SibSp+test.Parch

train['Ticket2']=train.Ticket.apply(lambda x : len(x))
test['Ticket2']=test.Ticket.apply(lambda x : len(x))


# In[8]:


train['Cabin2']=train.Cabin.apply(lambda x : len(str(x)))
test['Cabin2']=test.Cabin.apply(lambda x : len(str(x)))


# In[9]:


train['Name2']=train.Name.apply(lambda x: x.split(',')[0].strip())
test['Name2']=test.Name.apply(lambda x: x.split(',')[0].strip())


# # <a id="3"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> Setting up Pycaret</p>
# 
# 
# This is where magic happens.One line does all of these things:
# 
# * I will tell the model to ignore certain ID features with high cardinality,the target column,and give my session an id.
# * I will also pass sex as a categorical feature here,and try rebalancing to see how it turns out.
# * I will pass multicollinearity handling as true so that it takes care of it.
# * I will normalize the data
# 
# 

# In[10]:


from pycaret import classification
classification_setup = classification.setup(data = train,target = 'Survived',silent=True,)


# # <a id="4"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;">Model Comparison</p>

# In[11]:


classification.compare_models()


# # <a id="5"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;">Model Selection</p>

# In[12]:


lgb_classifier = classification.create_model('lightgbm')


# In[13]:


import numpy as np
params = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
          'n_estimators':[100,250,500,750,1000,1250,1500,1750],
          'max_depth': np.random.randint(1, (len(train.columns)*.85),20),
          'max_features': np.random.randint(1, len(train.columns),20),
          'min_samples_split':[2,4,6,8,10,20,40,60,100], 
          'min_samples_leaf':[1,3,5,7,9],
          'criterion': ["gini", "entropy"]}

tune_lgb = classification.tune_model(lgb_classifier, custom_grid = params)


# # <a id="6"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> Model Tuning</p>

# In[14]:


# Tune the model
params = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
tune_ridge = classification.tune_model(classification.create_model('ridge'), custom_grid = params, n_iter=50, fold=50)


# # <a id="7"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> Model ensembling</p>

# In[15]:


# ensemble boosting
bagging = classification.ensemble_model(tune_lgb, method= 'Bagging')


# In[16]:


from pycaret.classification import blend_models
# blending all models
blend_all = blend_models(method='hard',estimator_list=classification.compare_models(sort='Accuracy',n_select=10))


# In[17]:


# create individual models for stacking
ridge_cls = classification.create_model('ridge')
extre_tr = classification.create_model('et')
lgb = classification.create_model('lightgbm')
cat_cls = classification.create_model('catboost')
lg_cls = classification.create_model('lr')


# In[18]:


from pycaret.classification import stack_models
# stacking models
stacker = stack_models(estimator_list = [ridge_cls, extre_tr, lgb, cat_cls, lg_cls])


# This function returns the best model out of all models created in the current active environment based on metric defined in optimize parameter. Run this code at the end of  your script.
# Let's see the best model up until now.

# In[19]:


best = classification.automl(optimize = 'auc')


# In[20]:


best
# A stacked classifier it is!!


# In[21]:


# Validation Curve
classification.plot_model(tune_lgb, plot = 'vc')


# In[22]:


# AUC Curve
classification.plot_model(tune_lgb, plot = 'auc')


# In[23]:


# error Curve
classification.plot_model(tune_lgb, plot = 'error')


# In[24]:


y_pred = classification.predict_model(tune_lgb, data=test)


# In[25]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred['Label']
    })
submission.to_csv("submission.csv", index=False)


# Happy Kaggling :)

# # <a id="9"></a>
# # <p style="background-color:#A291A7;font-family:newtimeroman;color:#444160;font-size:150%;text-align:center;border-radius:20px 30px;"> References</p>
# * https://www.kaggle.com/aditi81k/titanic-prediction-using-pycaret
# (Thanks Aditi)
# * https://pycaret.org/
# (Official Documentation)
# * https://www.kaggle.com/goldens/titanic-on-the-top-with-a-simple-model
# (Feature engineering)

# In[ ]:




