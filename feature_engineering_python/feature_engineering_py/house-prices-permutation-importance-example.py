#!/usr/bin/env python
# coding: utf-8

# **Author: [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis)** ([LinkedIn](https://www.linkedin.com/in/carl-mcbride-ellis/))
# 
# ## Permutation importance: Example
# This is a simple example script to perfom permutation importance on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition data. The idea for this notebook comes directly from the magnificient introduction to [permutation importance](https://www.kaggle.com/dansbecker/permutation-importance) by [Dan Becker](https://www.kaggle.com/dansbecker), as part of the kaggle [Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) micro-course. It makes use of the [ELI5 python library](https://eli5.readthedocs.io/en/latest/) in conjunction with the [sklearn Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
# 
# Finally, we shall compare the results to those obtained from the scikit-learn Recursive Feature Elimination routine [sklearn.feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html). 

# In[1]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd
import numpy as np

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#===========================================================================
# select some features to rank. These are all numeric fields for today.
#===========================================================================
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 
        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


# Here we shall use the *Random Bar* idea by Damien Benveniste to insert a column of random numbers into the dataset. This column
# obviously has no predictive power whatsoever, and as such, will provide a marker in the feature ranking output
# clearly indicating that any features below this marker can be safely discarded.

# In[2]:


train_data['RANDOM'] = np.random.normal(size=len(train_data))

features.append("RANDOM")


# In[3]:


X_train       = train_data[features]
y_train       = train_data["SalePrice"]

#===========================================================================
# simple preprocessing: imputation; substitute any 'NaN' with mean value
#===========================================================================
X_train      = X_train.fillna(X_train.mean())

#===========================================================================
# set up our regressor + fit. 
# Today we shall be using the random forest regressor
#===========================================================================
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, max_depth=6)
regressor.fit(X_train, y_train)


# In[4]:


#===========================================================================
# perform the PermutationImportance
#===========================================================================
import eli5
from eli5.sklearn import PermutationImportance

perm_import = PermutationImportance(regressor, random_state=1).fit(X_train, y_train)

# visualize the results
eli5.show_weights(perm_import, top=None, feature_names = X_train.columns.tolist())


# and now using Recursive Feature Elimination:

# In[5]:


#===========================================================================
# perform a scikit-learn Recursive Feature Elimination (RFE)
#===========================================================================
from sklearn.feature_selection import RFE
# here we want only one final feature, we do this to produce a ranking
rfe = RFE(regressor, n_features_to_select=1)
rfe.fit(X_train, y_train)

#===========================================================================
# now print out the features in order of ranking
#===========================================================================
from operator import itemgetter
for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
    print(x, y)


# We can see that both techniques coincide in selecting the same four most important features:
# 
# 1. OverallQual
# 2. GrLivArea
# 3. TotalBsmtSF
# 4. BsmtFinSF1
# 
# after which the order of the less important features change between both techniques and between runs.
# 
# ### **Notes**: 
# > Features that are deemed of **low importance for a bad model** (low cross-validation score) could be **very important for a good model**. Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but **how important this feature is for a particular model**. ([source](https://scikit-learn.org/stable/modules/permutation_importance.html))
# 
# * **Extrapolation**: It should also be mentioned that tree based methods (*i.e.* Random Forest, XGBoost *etc*) are unable to extrapolate. See ["Extrapolation: Do not stray out of the forest!"](https://www.kaggle.com/code/carlmcbrideellis/extrapolation-do-not-stray-out-of-the-forest/). This has important implications when it comes to performing permutation importance using such estimators. For example imagine there are two highly correlated features. Permuting one of these features will create new data points that are now *outside* of the region that the estimator was originally trained on, which would require being able to extrapolate in order to produce a reliable predictions. This will lead to an incorrect estimate of the importance of the permuted feature, and a computationally costly re-training for each permuted feature would really be required to produce an accurate estimate.
# 
# For more details see: ["*Unrestricted Permutation forces Extrapolation: Variable Importance Requires at least One More Model, or There Is No Free Variable Importance*",  arXiv:1905.03151 (2021)](https://arxiv.org/pdf/1905.03151.pdf)
