#!/usr/bin/env python
# coding: utf-8

# **Author: [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis)** ([LinkedIn](https://www.linkedin.com/in/carl-mcbride-ellis/))
# 
# # Feature importance or selection using the LASSO
# 
# When creating a model not all of the features in our training data are of equal importance. If we have sufficient computational resources at our disposal then we could indeed include all of the available features in our model, but this has (at least) two drawbacks; this can lead to [overfitting](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic), and also reduces the interpretability of our model. It is much more informative to create our model on a subset of the most influential features, in other words create a *sparse* model. In this short notebook we shall be ranking the features in a given dataset using the LASSO. Other notable feature importance techniques are 
# 
# * [Boruta-SHAP](https://www.kaggle.com/carlmcbrideellis/feature-selection-using-the-boruta-shap-package)
# * [Recursive Feature Elimination (RFE)](https://www.kaggle.com/carlmcbrideellis/recursive-feature-elimination-rfe-example)
# * [Permutation Importance](https://www.kaggle.com/carlmcbrideellis/house-prices-permutation-importance-example)
# 
# ## Regularization, or 'shrinkage'
# 
# A very common *loss function* ($\mathcal{L}$) is the square of the residual or error
# 
# $$ \mathcal{L}_2 = (\mathtt{y\_true} - \mathtt{y\_pred} )^2= \left( y_i - \hat{y_i}\right)^2 $$
# 
# which is known as the *squared error* (SE) or *L2 loss*. It examines the difference between the predicted value ($\hat{y}$) and the true value ($y$) for a given datapoint $i$.
# Now let us start by looking at our old friend the linear regression. When finding the best fit it is usual to minimise a *cost function* (the sum of the *loss function* for all points), in this case the *residual sum of squares* (RSS), given by (ESL Eq. 3.2):
# 
# $$ \mathrm{RSS}(\beta) = \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \right)^2  \equiv \sum_{i=1}^n \left( y_i - \hat{y_i}\right)^2 \equiv \sum \mathcal{L}_2$$
# 
# where $\beta_j$ are the regression coefficients, also known as the **Weights**, associated with each feature.
# 
# ### Ridge regression
# We can shrink these regression coefficients by additionally imposing a *penalty term* (or *regularization term*) in our cost function (ESL Eq. 3.41):
# 
# $$ \mathrm{PRSS} = \mathrm{RSS} + \lambda \sum_{j=1}^p \beta_j^2 $$
# 
# where $\lambda \gt 0 $ is the '*tuning parameter*'. If $\lambda = 0$ we revert to the original residual sum of squares cost function. The optimal value for the tuning parameter is found via cross-validation. Note the similarity between the penalty term and the  $\ell^2$ vector norm. With Ridge regression we may well find that many of the coefficients become very small, however they are generally non-zero.
# 
# ### LASSO regression
# With LASSO (Least Absolute Shrinkage and Selection Operator) regression we now have the following penalty term (ESL Eq. 3.53):
# 
# $$ \mathrm{PRSS} = \mathrm{RSS} + \lambda \sum_{j=1}^p |\beta_j| $$
# 
# note now the use of the $\ell^1$ norm, which is sometimes known as the [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry). This change has the effect of actually forcing some of the coefficients to become zero, in other words this is actually removing features from the model, and is effectively performing feature selection.
# 
# ## Example
# We shall now apply LASSO to the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset

# In[1]:


#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd
import numpy   as np

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

#===========================================================================
# here, for this simple demonstration we shall only use the numerical columns 
# and ingnore the categorical features
#===========================================================================
X_train = train_data.select_dtypes(include=['number']).copy()
X_train = X_train.drop(['SalePrice'], axis=1)
y_train = train_data["SalePrice"]
# fill in any missing data with the mean value
X_train = X_train.fillna(X_train.mean())


# with the following number of features

# In[2]:


print(X_train.shape[1])


# We shall use the [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) whose details are given [here](https://scikit-learn.org/stable/modules/linear_model.html#lasso).
# 
# Firstly we should standardize the features, *i.e.* divide each feature by the standard deviation of that feature, thus each feature that is passed to the LASSO now has a standard deviation of one.

# In[3]:


std      = np.std(X_train, axis=0)
X_train /= std


# note that scikit-learn represents $\lambda$ by `alpha`

# In[4]:


from sklearn import linear_model
regressor = linear_model.Lasso(alpha=5000,
                               positive=True,
                               fit_intercept=False, 
                               max_iter=1000,
                               tol=0.0001)
regressor.fit(X_train, y_train)


# Finally we shall use the [ELI5](https://eli5.readthedocs.io/en/latest/autodocs/eli5.html) library to show the results, the greater the weight the more important the feature

# In[5]:


import eli5
eli5.show_weights(regressor, top=-1, feature_names = X_train.columns.tolist())


# we have now selected, using this particular value of $\lambda$, the 10 most important features from the original set of 36 features, the other features now have a Weight of zero, and have effectively been dropped.
# 
# Note that by keeping strong features and dropping the weaker features may well slightly harm the overall (LB) score, but the introduction of sparsity will lead to a much more interpretable model.
# 
# # Related reading
# * ESL:=  ["*The Elements of Statistical Learning*" by Trevor Hastie, Robert Tibshirani  and Jerome Friedman (2nd Ed.) Springer (2009)](https://web.stanford.edu/~hastie/ElemStatLearn/)
# * [Arthur E. Hoerl and Robert W. Kennard "*Ridge Regression: Biased Estimation for Nonorthogonal Problems*", Technometrics **vol 12** pp. 55-67 (1970)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634)
# * [Arthur E. Hoerl and Robert W. Kennard "*Ridge Regression: Applications to Nonorthogonal Problems*", Technometrics **vol 12** pp. 69-82 (1970)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488635)
# * [Robert Tibshirani "*Regression Shrinkage and Selection Via the Lasso*", Journal of the Royal Statistical Society: Series B (Methodological) **vol 58** pp. 267-288 (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
# * [*Introduction to shrinkage and Ridge*](https://youtu.be/I8bPQ272Pbs) YouTube video (for ESL § 3.4)
# * [*LASSO regression*](https://youtu.be/FlSQgXv7Dvw) YouTube video (for ESL § 3.4.3)
# * [*Selecting the tuning parameter for Ridge regression and LASSO*](https://youtu.be/8oEZkHqf_Rk) YouTube video
