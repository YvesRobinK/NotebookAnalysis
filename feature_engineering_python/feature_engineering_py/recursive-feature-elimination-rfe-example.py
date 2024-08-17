#!/usr/bin/env python
# coding: utf-8

# ## Recursive Feature Elimination (RFE) example
# Recursive feature elimination [1] is an example of *backward feature elimination* [2] in which we essentially first fit our model using *all* the features in a given set, then progressively one by one we remove the *least* significant features, each time re-fitting, until we are left with the desired number of features, which is set by the parameter `n_features_to_select`.
# 
# This simple script uses the scikit-learn *Recursive Feature Elimination* routine [sklearn.feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html). In this example we shall use the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) data.
# For the regressor we shall use the [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) routine, also from scikit-learn.
# 
# Scikit-learn also has a variant of this routine that incorporates [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics&#41;), see: [sklearn.feature_selection.RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).
# 
# ### Forward feature selection
# RFE has its counterpart in *forward feature selection*, which does the opposite: accrete features rather than eliminate them, usually via some form of [greedy algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm) [3]. Scikit-learn has the routine [sklearn.feature_selection.f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) to facilitate this. 
# 
# Note that both of these *wrapper* methods can be beaten if one has access to "domain knowledge", i.e. understanding the problem and having a good idea as to which features will be important in the model one is constructing.
# 
# ### The python code:

# In[1]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a simple script to perform recursive feature elimination on 
# the kaggle 'House Prices' data set using the scikit-learn RFE
# Carl McBride Ellis (2.V.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd
import numpy as np

#===========================================================================
# read in the data and specify the target feature
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)
test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col=0)
target     = 'SalePrice'

#===========================================================================
#===========================================================================
X_train = train_data.select_dtypes(include=['number']).copy()
X_train = X_train.drop([target], axis=1)
y_train = train_data[target]
X_test  = test_data.select_dtypes(include=['number']).copy()

#===========================================================================
# simple preprocessing: imputation; substitute any 'NaN' with mean value
#===========================================================================
X_train = X_train.fillna(X_train.mean())
X_test  = X_test.fillna(X_test.mean())

#===========================================================================
# set up our regressor. Today we shall be using the random forest
#===========================================================================
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)

#===========================================================================
# perform a scikit-learn Recursive Feature Elimination (RFE)
#===========================================================================
from sklearn.feature_selection import RFE
# here we want only one final feature, we do this to produce a ranking
n_features_to_select = 1
rfe = RFE(regressor, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

#===========================================================================
# now print out the features in order of ranking
#===========================================================================
from operator import itemgetter
features = X_train.columns.to_list()
for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
    print(x, y)

#===========================================================================
# ok, this time let's choose the top 10 featues and use them for the model
#===========================================================================
n_features_to_select = 10
rfe = RFE(regressor, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
predictions = rfe.predict(X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({"Id":test_data.index, target:predictions})
output.to_csv('submission.csv', index=False)


# ### References
# 
# 1. [Isabelle Guyon, Jason Weston, Stephen Barnhill and Vladimir Vapnik "Gene Selection for Cancer Classification using Support Vector Machines", Machine Learning volume 46, pages 389–422 (2002) (doi: 10.1023/A:1012487302797)](https://doi.org/10.1023/A:1012487302797)
# 2. [Ron Kohavi and George H. John "Wrappers for feature subset selection", Artificial Intelligence Volume 97 pages 273-324 (1997) (doi: 10.1016/S0004-3702(97)00043-X)](https://www.sciencedirect.com/science/article/pii/S000437029700043X)
# 3. [Haleh Vafaie and Ibrahim F. Imam "Feature Selection Methods: Genetic Algorithms vs. Greedy-like Search", Proceedings of the 3rd International Conference on Fuzzy and Intelligent Control Systems (1994)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.8452)
