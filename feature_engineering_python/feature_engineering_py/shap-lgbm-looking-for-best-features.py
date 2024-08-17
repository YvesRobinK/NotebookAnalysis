#!/usr/bin/env python
# coding: utf-8

# # My path to finding for best features ....
# 
# **This is attempt for finding TPS-05 solution and best feature for model building. I decided to create simple model using LGBM and then analize what features drive model for each particular class.**
# 
# Interesting in my TPS-05 notebooks?
# - [Pytorch NN for tabular - step by step](https://www.kaggle.com/remekkinas/tps-5-pytorch-nn-for-tabular-step-by-step)
# - [CNN (2D Convolution) for solving TPS-05](https://www.kaggle.com/remekkinas/cnn-2d-convolution-for-solving-tps-05)
# - [Weighted training - XGB, RF, LR, ... SMOTE](https://www.kaggle.com/remekkinas/tps-5-weighted-training-xgb-rf-lr-smote)
# - [HydraNet!! ... Keras Stacked Ensemble ..](https://www.kaggle.com/remekkinas/tps-5-hydranet-keras-stacked-ensemble)
# 

# # Reference - ["Interpretable Machine Learning, A Guide for Making Black Box Models Explainable."](https://christophm.github.io/interpretable-ml-book/) Christoph Molnar (updated 09.05.2021)
# 

# #### Shap in my opinion is absolutely one of the best tool you can use for model understanding and hacking (improving). In this competition I made many analysis using Shap which lead me to interesting solutions.  

# In[1]:


import shap
import pandas as pd
import numpy as np
import seaborn as sns

from tqdm import tqdm
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# ### LOAD TPS-04 DATA AND PREPROCESS (QUICK WAY)

# In[2]:


train = pd.read_csv("../input/tabular-playground-series-may-2021/train.csv", index_col = 'id')
test = pd.read_csv("../input/tabular-playground-series-may-2021/test.csv", index_col = 'id')
train = train[~train.drop('target', axis = 1).duplicated()]

X = pd.DataFrame(train.drop("target", axis = 1))

lencoder = LabelEncoder()
y = pd.DataFrame(lencoder.fit_transform(train['target']), columns=['target'])


# ### LGBM CROSS VALIDATED TRAINING LOOP
# LGBM is one of the most popular algorith used in TPS-05. Let's look how it predict. Let's look how it sees the TPS-05 data.

# In[3]:


params = { 
       'objective': 'multiclass', 
       'num_class' : 4, 
       'metric': 'multi_logloss' 
   } 


# In[4]:


test_preds = None
train_rmse = 0
val_rmse = 0
n_splits = 10

model =  LGBMClassifier(**params)

skf = StratifiedKFold(n_splits = n_splits, shuffle = True,  random_state = 0)

for tr_index , val_index in tqdm(skf.split(X.values , y.values), total=skf.get_n_splits(), desc="k-fold"):

    x_train_o, x_val_o = X.iloc[tr_index] , X.iloc[val_index]
    y_train_o, y_val_o = y.iloc[tr_index] , y.iloc[val_index]
    
    eval_set = [(x_val_o, y_val_o)]
    
    model.fit(x_train_o, y_train_o, eval_set = eval_set, early_stopping_rounds=100, verbose=False)

    train_preds = model.predict(x_train_o)
    train_rmse += mean_squared_error(y_train_o ,train_preds , squared = False)

    val_preds = model.predict(x_val_o)
    val_rmse += mean_squared_error(y_val_o , val_preds , squared = False)

    if test_preds is None:
        test_preds = model.predict_proba(test.values)
    else:
        test_preds += model.predict_proba(test.values)

print(f"\nAverage Training RMSE : {train_rmse / n_splits}")
print(f"Average Validation RMSE : {val_rmse / n_splits}\n")

test_preds /= n_splits


# # SHAP model explainer

# In[5]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test)


# ### HOW FEATURES IMPACT MODEL

# In[6]:


shap.summary_plot(shap_values, X)


# As we can see:
# - Feature_14 and feature_15 affect the model the most, but you can also see the balance between classes (feature_15 plays big role in class_2 prediction) - they are definitely important variables
# - The most important feature for class_0 is feature_25, class_1 -> feature_14, class_2 -> feature_15, class_3 -> feature_31 
# 

# In[7]:


selected_features = ["feature_25", "feature_14", "feature_15", "feature_31"]

plt.figure(figsize=(20,5))
c = 1
for feat in selected_features:
    plt.subplot(1, 4, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# - Looks like most of values is ZERO(!) This is why you can observe some patterns later during class prediction analysis.
# - This could be a potential problem with the model. Which will divide classes with respect to 0. And if there is a majority of them then ...
# - In my opinion they in this competition is feature modeling - not dataset imbalance, not super model building ...

# In[8]:


test[selected_features].describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# ### Lets look inside TOP6 features .... 

# In[9]:


selected_features = ["feature_14", "feature_15", "feature_6", "feature_16", "feature_31", "feature_37"]

plt.figure(figsize=(15,10))
c = 1
for feat in selected_features:
    plt.subplot(2, 3, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# - The same situation we can see here - ZERO rulez! ZERO vs ........ REST!!!

# In[10]:


test[selected_features].describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# ## CLASS 0 - FEATURES IMPORTANCE

# In[11]:


shap.summary_plot(shap_values[0], test)


# In[12]:


selected_features = ["feature_25", "feature_37", "feature_23", "feature_38"]

plt.figure(figsize=(30,5))
c = 1
for feat in selected_features:
    plt.subplot(1, 4, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# Conclusion:
# - there is no obvious patters here, this is why model has problem with class 0 prediction - it works almost randomly (I think but I could be wrong)
# - as you can see reds are on the left and right - but we can see that most "high" values are on right side of chart 
# - almost all features are 0 balanced ... so 0 can drive model to Class 0 and .... rest of Class -> ZERO plays main role in this competition. 
# - Feature 25 - we can see that "medium" feature value (purple) drives model to predict class 0 
# - Feature 37 - the highest value (red) then class 0 is predicted,
# - Feature 23 - high values drive model to predict class 0
# - there is almost no uniform distribution

# ## CLASS 1 - FEATURES IMPORTANCE

# In[13]:


shap.summary_plot(shap_values[1], test)


# Conclusion:
# - We see some interesting patters here - blue blobs
# - if most of features (till 20) are "low" they drives model to other class (not class 1, probably for class 3)
# - feature 14 and "low" values says that probaly there is no Class_1
# - feature 14->37 are uniform - "higher" values drive model to predict Class_1
# - feature_34 - "higher" values drive model to Class_0

# In[14]:


selected_features = ["feature_14", "feature_6", "feature_28", "feature_37"]

plt.figure(figsize=(30,5))
c = 1
for feat in selected_features:
    plt.subplot(1, 4, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# ## CLASS 2 - FEATURES IMPORTANCE

# In[15]:


shap.summary_plot(shap_values[2], test)


# Conclusion:
# - We see some interesting patters here - blue blobs
# - The most important feature is 15 and 14. Here we can see obvious pattens. In feature_15 small values (probalby 0) drives model to Class 2. Feature_14 - high values drive to Class 2.
# - Feature_2 there is seen good separation - all positive values (blue blob is 0) highly influence model in Class_2 direction.
# - Almost in all features from list we can see that "high" values drives model to Class_2

# In[16]:


selected_features = ["feature_15", "feature_14", "feature_2", "feature_11"]

plt.figure(figsize=(30,5))
c = 1
for feat in selected_features:
    plt.subplot(1, 4, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# ## CLASS 3 - FEATURES IMPORTANCE

# In[17]:


shap.summary_plot(shap_values[3], test)


# Conclusion:
# - We see some interesting patters here - blue blobs
# - From feature 31 to 23 ... we can see that small values drive model to Class_3. This is probably 0 (we shold see data in these feature).
# - The bigger number in features the less chance that Class 3 will be predicted.

# In[18]:


selected_features = ["feature_31", "feature_24", "feature_14", "feature_16"]

plt.figure(figsize=(30,5))
c = 1
for feat in selected_features:
    plt.subplot(1, 4, c)
    sns.histplot(x = feat, data = test, bins=10)
    c = c + 1    
plt.show()


# # CONCLUTIONS
# 
# In my opinion ....  focus points:
# - not model .... (low priority) -  same results you can achive using XGBoost, LightGBM, CatBoost (slight differences) ... and even poorly designed NN :) 
# - not imbalance ..... (low priority) - weighted training is not a solution (see this notebook: https://www.kaggle.com/remekkinas/tps-5-weighted-training-xgb-rf-lr-smote ), sampling (uder/over) is not a solution 
# 
# Feature 
# - feature engineering (HIGH priority) - dealing with 0 which drives models to biased prediction
# - finding better ML model to deal with sparsity ...
# 
# Of course, you may have a different opinion on this. Conversations below are welcome.
