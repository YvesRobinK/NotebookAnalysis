#!/usr/bin/env python
# coding: utf-8

# <h3>Introduction</h3><br>
# <b>Hello,</b><br>
# In this kernel I will be showing you the fastest way to make your first submission to a Kaggle competition I will be using the data from the <b>'House prices: Advanced Regression Techniques'</b> competition.<br> 
# For this competition, we are predicting the sale price of property. The data is splited into two parts Training and testing sets both contains 1470 observations and 80 features.<br>
# <b style="color:red">This approach with no feature engineering does ok on leaderboard </b>
# <br><br><br>
# 
# I would like to recommend some kernels and courses that helped me begin my journey on Kaggle:
# <ul>
#   <li>Machine learning <a href="https://course.fast.ai/ml.html">Fastai</a> it’s free and available on YouTube</li>
#   <li>For advance feature engineering and parameter tuning those 2 kernels are well detailed 
#       <a href="https://www.kaggle.com/josh24990/simple-stacking-approach-top-12-score/notebook">Simple stacking approach</a>,
#       <a href="https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard">Stacked Regressions</a>
# </li>
# </ul>  
# 
# 

# ## Imports

# <h4><a href="https://github.com/fastai/fastai">Fastai download Installation guide/documentation</a></h4>

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null')
# !apt update && apt install -y libsm6 libxext6


# In[ ]:


from fastai.structured import rf_feat_importance


# In[ ]:


from fastai.structured import train_cats,proc_df


# #### Import ML models 

# In[ ]:


#RandomForest
import math 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
#Xgboost
import xgboost as xgb
#CatBoost 
from catboost import CatBoostRegressor


# ### Reading the data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# #### Saving the Id column for later use 

# In[ ]:


Id = test['Id']


# #### combinion the traning and testing dataset to maintain consistency between the sets

# In[ ]:


test_copy = test.copy()


# In[ ]:


test_copy["SalePrice"] = np.nan


# In[ ]:


train_set_data = [train,test_copy]
train_set_data = pd.concat(train_set_data)


# In[ ]:


len(train_set_data) == len(train)+len(test)


# #### train_cats is a function in the fastai library that  convert strings to pandas categories

# In[ ]:


train_cats(train_set_data)


# #### proc_df will replace categories with their numeric codes, handle missing continuous values, and split the dependent variable into a separate variable for the max_n_cat is to create dummy variables for the categorical column with less or equal to 10 categories 

# In[ ]:


df, y, nas = proc_df(train_set_data, 'SalePrice',max_n_cat=10)


# ####  resplite the training and test set

# In[ ]:


test_df = df[1460:2919]
df = df[0:1460]
y=y[0:1460]


# #### Train a quick randomForest Resressor to check the feature importance 

# In[ ]:


m = RandomForestRegressor(n_jobs=-1,verbose=0)
m.fit(df, y)


# #### rf_feat_importance that uses the feature_importances_ attribute from the RandomForestRegressor to return a dataframe with the columns and their importance in descending order.

# In[ ]:


fi = rf_feat_importance(m, df)


# In[ ]:


len(df.columns)


# In[ ]:


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# #### creating a plot of the most relevant features 

# In[ ]:


plot_fi(fi[fi.imp>0.005])


# #### keep only the column that have a acceptable information gain 

# In[ ]:


df = df[fi[fi.imp>0.0005].cols]


# #### Splite the data to training set and a validation set 

# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 400  
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# #### creating a function that evaluate the algorithmes performance <a href="https://en.wikipedia.org/wiki/Out-of-bag_error">Learn more about OOB score</a> 

# In[ ]:


def rmse(x,y): return math.sqrt(((np.log(x)-np.log(y))**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# ### <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForest</a>

# #### optimizing hyperparameters of a random forest with the grid search 

# In[ ]:


rf_param_grid = {
                 'max_depth' : [4, 6, 8,12],
                 'n_estimators': [5,10,20,60,100],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10,20],
                 'min_samples_leaf': [1, 3, 10,18,25],
                 'bootstrap': [True, False],
                 }


# In[ ]:


m = RandomForestRegressor()


# In[ ]:


m_r = RandomizedSearchCV(param_distributions=rf_param_grid, 
                                    estimator = m,  
                                    verbose = 0, n_iter = 50, cv = 4)


# #### fitting the model to the training set 

# In[ ]:


m_r.fit(X_train, y_train)


# In[ ]:


print_score(m_r)


# ### <a href="https://xgboost.readthedocs.io/en/latest/">Xgboost</a>

# In[ ]:


xgb_classifier = xgb.XGBRegressor()


# #### optimizing hyperparameters of a random forest with the grid search 

# In[ ]:


gbm_param_grid = {
    'n_estimators': range(1,100),
    'max_depth': range(1, 15),
    'learning_rate': [.1,.13, .16, .19,.3,.6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}


# In[ ]:


xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = xgb_classifier, 
                                    verbose = 0, n_iter = 50, cv = 4)


# In[ ]:


xgb_random.fit(X_train,y_train)


# In[ ]:


print_score(xgb_random)


# ### <a href="https://tech.yandex.com/catboost/doc/dg/concepts/python-installation-docpage/">CatBoost</a>

# ##### The below parameters come from <a href="https://www.kaggle.com/josh24990/simple-stacking-approach-top-12-score/notebook">this kernel </a>, we can optimizing hyperparameter but it will take a long time a lot of processing power (<a href="https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/">catBoost parameter tuning</a>)

# In[ ]:


m_c = CatBoostRegressor(iterations=2000,learning_rate=0.1,depth=3,loss_function='RMSE',l2_leaf_reg=4,border_count=15,verbose=False)


# In[ ]:


m_c.fit(X_train,y_train)


# In[ ]:


print_score(m_c)


# #### get only the column used on the training set to predict on the test set 

# In[ ]:


test_df = test_df[list(X_train.columns)]


# ### combining the 3 models to predict on the test set  

# In[ ]:


y_pred = (m_c.predict(test_df) + m_r.predict(test_df)+xgb_random.predict(test_df))/3


# #### concatenate the saved Id with the predicted values to create a csv file for submussion 

# In[ ]:


submission = pd.DataFrame({"Id": Id,"SalePrice": y_pred})
submission.to_csv('submission.csv', index=False)


# ![](submission.PNG)

# ### For further parameter engineering they is a great python library "<a href="https://github.com/pandas-profiling/pandas-profiling">Pandas profiling</a>" that Generates profile reports from a pandas DataFrame.

# ### Thank you for reading ( ͡ᵔ ͜ʖ ͡ᵔ )
