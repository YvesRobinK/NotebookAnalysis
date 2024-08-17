#!/usr/bin/env python
# coding: utf-8

# # TabNet: A very simple regression example using the House Prices data
# [**TabNet**](https://arxiv.org/pdf/1908.07442.pdf) brings deep learning to tabular data. TabNet has been developed by researchers at Google Cloud AI and achieves SOTA performance on a number of test cases.
# This notebook is a simple example of performing a regression using the [pyTorch implementation](https://pypi.org/project/pytorch-tabnet/). 
# 
# `TabNetRegressor()` has a number of options for the `device_name`: `cpu`, `cuda`, `mkldnn`, `opengl`, `opencl`, `ideep`, `hip`, `msnpu`, and `xla`.
# The `fit()` has a variety of `eval_metric`: `auc`, `accuracy`, `balanced_accuracy`, `logloss`, `mae`, `mse`, and `rmse`. TabNet can also perform classification using `TabNetClassifier()` as well as perform [multi-task learning](https://en.wikipedia.org/wiki/Multi-task_learning).
# 
# We shall use the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) data for this demonstration. In this example I undertake no feature engineering, nor data cleaning, such as the removal of outliers *etc*., and perform  only the most basic imputation simply to account for any missing values.
# 
# #### Install TabNet:

# In[1]:


get_ipython().system('pip install pytorch-tabnet')
import pandas as pd
import numpy  as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold


# In[2]:


#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample     = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/solution.csv')


# In[3]:


#===========================================================================
# select some features
#===========================================================================
features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
            'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 
            '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
            'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr',  'Fireplaces', 
            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
            'EnclosedPorch',  'PoolArea', 'YrSold']


# In[4]:


X      = train_data[features]
y      = np.log1p(train_data["SalePrice"])
X_test = test_data[features]
y_true = solution["SalePrice"]


# We shall impute any missing data with a simple mean value. As to the relative merits of doing this *before* using cross-validation see [Byron C. Jaeger, Nicholas J. Tierney, and Noah R. Simon "*When to Impute? Imputation before and during cross-validation*" arXiv:2010.00718](https://arxiv.org/pdf/2010.00718.pdf).
# For a much better imputation method take a look at the notebook ["MissForest - The best imputation algorithm"](https://www.kaggle.com/lmorgan95/missforest-the-best-imputation-algorithm) by [Liam Morgan](https://www.kaggle.com/lmorgan95). It deals with the R implementation, and MissForest can also be used in python via the [missingpy](https://github.com/epsilon-machine/missingpy) package.

# In[5]:


X      =      X.apply(lambda x: x.fillna(x.mean()),axis=0)
X_test = X_test.apply(lambda x: x.fillna(x.mean()),axis=0)


# Convert the data to [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)

# In[6]:


X      = X.to_numpy()
y      = y.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()


# run the TabNet deep neural network, averaging over 5 folds:

# In[7]:


kf = KFold(n_splits=5, random_state=42, shuffle=True)
predictions_array =[]
CV_score_array    =[]
for train_index, test_index in kf.split(X):
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    regressor = TabNetRegressor(verbose=0,seed=42)
    regressor.fit(X_train=X_train, y_train=y_train,
              eval_set=[(X_valid, y_valid)],
              patience=300, max_epochs=2000,
              eval_metric=['rmse'])
    CV_score_array.append(regressor.best_cost)
    predictions_array.append(np.expm1(regressor.predict(X_test)))

predictions = np.mean(predictions_array,axis=0)


# calculate our average CV score

# In[8]:


print("The CV score is %.5f" % np.mean(CV_score_array,axis=0) )


# now calculate our leaderboard score (See: ["House Prices: How to work offline"](https://www.kaggle.com/carlmcbrideellis/house-prices-how-to-work-offline)).

# In[9]:


from sklearn.metrics import mean_squared_log_error
RMSLE = np.sqrt( mean_squared_log_error(y_true, predictions) )
print("The LB score is %.5f" % RMSLE )


# We can see that our CV score corresponds nicely with our leaderboard score, so we do not seem to be [overfitting or underfitting](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic) by too much.
# 
# Finally write out a `submission.csv` file:

# In[10]:


sample.iloc[:,1:] = predictions
sample.to_csv('submission.csv',index=False)


# # Related reading
# * [Sercan O. Arik and Tomas Pfister "TabNet: Attentive Interpretable Tabular Learning", arXiv:1908.07442 (2019)](https://arxiv.org/pdf/1908.07442.pdf)
# * [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet) (GitHub)
# * ["TabNet on AI Platform: High-performance, Explainable Tabular Learning"](https://cloud.google.com/blog/products/ai-machine-learning/ml-model-tabnet-is-easy-to-use-on-cloud-ai-platform) (Google Cloud)
# * Notebook: [TabNet: A simple binary classification example](https://www.kaggle.com/carlmcbrideellis/tabnet-simple-binary-classification-example) (using the Santander Customer Satisfaction data on kaggle)
