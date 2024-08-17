#!/usr/bin/env python
# coding: utf-8

# # Introduction to the Regularized Greedy Forest (RGF)
# The RGF is a powerful technique developed by Rie Johnson and Tong Zhang in the paper ["Learning Nonlinear Functions Using Regularized Greedy Forest"](https://arxiv.org/pdf/1109.0887.pdf). It is on a par with gradient boosting tools like [XGBoost](https://xgboost.ai/). An ensemble of the solutions produced form these methods may well be good enough to win a kaggle competition.
# ## Decision Trees
# [Decision trees](https://scikit-learn.org/stable/modules/tree.html) are perhaps one of the most venerable techniques used in machine learning, notably the [ID3](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf) and **C4.5** algorithms by Ross Quinlan. 
# Decision trees are simple to implement and to explain, but are prone to [overfitting](https://en.wikipedia.org/wiki/Overfitting).
# They can be used for both classification, for example see [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), and regression, see [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), and hence the acronym CART. 
# Decision trees are what are now known as *weak learners*.
# ## Ensembles and the Decision Forest
# It was shown by that one is able to create a *strong learner* from a collection, or '*ensemble*', of weak learners in a famous paper by Robert Schapire ["The Strength of Weak Learnability"](https://link.springer.com/content/pdf/10.1007/BF00116037.pdf). It was from this idea that came the **decision forest**, in which, as the name suggests, one by one a collection of decision trees is created. This goes by the name of [*boosting*](https://en.wikipedia.org/wiki/Boosting_&#40;machine_learning&#41;). The boosting process is what is known as being [*greedy*](https://en.wikipedia.org/wiki/Greedy_algorithm); each individual step is optimal (for example, each tree added to the forest) at the time, but this does not necessarily lead to an overall optimal solution.
# 
# ## Regularization
# [Regularization](https://en.wikipedia.org/wiki/Regularization_&#40;mathematics&#41;) is a technique designed to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting). In gradient boosting, an implicit regularization effect is achieved by small step size $s$ or [*shrinkage parameter*](https://en.wikipedia.org/wiki/Shrinkage_&#40;statistics&#41;), which for best results should tend to be infinitesimally small. However, as one can imagine this is not viable in practice. In the end one chooses as small an $s$ as possible, in conjunction with an *early stopping* criteria.
# In the RGF however,  an explicit regularization is used to prevent overfitting using [structured sparsity](https://en.wikipedia.org/wiki/Structured_sparsity_regularization) where the underlying forest structure is viewed as a graph of sparsity structures. In RGF one has an ensemble of forest nodes rather than individual trees.
# 
# ## What is RGF?
# In the words of the authors of RGF:
# 
# > "RGF integrates two ideas: one is to include tree-structured regularization into the learning formulation; and the other is to employ the  fully-corrective  regularized  greedy  algorithm.  Since  in  this  approach  we  are  able  to  take  advantage  of  the special  structure  of  the  decision  forest"
# 
# # Why use the RGF?
# The regularized greedy forest has been shown to out-perform [gradient boosting decision trees](https://en.wikipedia.org/wiki/Gradient_boosting) (GBDT), which is a technique used by [XGBoost](https://xgboost.ai/), [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/), and [CatBoost](https://catboost.ai/). Indeed, RGF was used by the kagglers [infty](https://www.kaggle.com/infty36878) and [random modeler](https://www.kaggle.com/randommodeler) to come 1st in the kaggle [Benchmark Bond Trade Price Challenge](https://www.kaggle.com/c/benchmark-bond-trade-price-challenge) and the [Heritage Health Prize](https://www.kaggle.com/c/hhp/) competitions, and 4th place in the [Predicting a Biological Response](https://www.kaggle.com/c/bioresponse) competition.
# 
# # How to use RGF:
# We shall use the [`rgf_python`](https://github.com/RGF-team/rgf/tree/master/python-package) package, written by the [RGF-team](https://github.com/RGF-team), applied first to a simple classification example; the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition data, and then to a regression example, using the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition data.

# In[1]:


get_ipython().system('pip install rgf_python')


# # Classification example: Titanic

# In[2]:


import pandas  as pd
import numpy   as np

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')
solution   = pd.read_csv('../input/submission-solution/submission_solution.csv')

#===========================================================================
# select some features
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train       = pd.get_dummies(train_data[features])
y_train       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])

#===========================================================================
# perform the classification 
#===========================================================================
from rgf.sklearn import RGFClassifier

classifier = RGFClassifier(max_leaf=300, algorithm="RGF_Sib", test_interval=100)

#===========================================================================
# and the fit 
#===========================================================================
classifier.fit(X_train, y_train)

#===========================================================================
# use the model to predict 'Survived' for the test data
#===========================================================================
predictions = classifier.predict(final_X_test)

#===========================================================================
# now calculate our score
#===========================================================================
from sklearn.metrics import accuracy_score
print("The score is %.5f" % accuracy_score( solution['Survived'] , predictions ) )


# # Regression example: House Prices

# In[3]:


#===========================================================================
# read in the competition data 
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/solution.csv')
                         
#===========================================================================
# select some features
#===========================================================================
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 
        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

#===========================================================================
#===========================================================================
X_train       = train_data[features]
y_train       = train_data["SalePrice"]
final_X_test  = test_data[features]
y_true        = solution["SalePrice"]

#===========================================================================
# essential preprocessing: imputation; substitute any 'NaN' with mean value
#===========================================================================
X_train      = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())

#===========================================================================
# perform the regression
#===========================================================================
from rgf.sklearn import RGFRegressor

regressor = RGFRegressor(max_leaf=300, algorithm="RGF_Sib", test_interval=100, loss="LS")

#===========================================================================
# and the fit 
#===========================================================================
regressor.fit(X_train, y_train)

#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
y_pred = regressor.predict(final_X_test)

#===========================================================================
# compare your predictions with the 'solution' using the 
# root of the mean_squared_log_error
#===========================================================================
from sklearn.metrics import mean_squared_log_error
RMSLE = np.sqrt( mean_squared_log_error(y_true, y_pred) )
print("The score is %.5f" % RMSLE )


# It almost goes without saying that to produce a medal winning score one does not only need a powerful estimator such as the RGF, but also perform data cleaning, judicious feature selection (perhaps using the new [Boruta-SHAP package](https://www.kaggle.com/carlmcbrideellis/feature-selection-using-borutashap)), if required then also perform  [feature engineering](http://www.feat.engineering/) as well as the necessary hyperparameter tuning and, just maybe, add a little *magic*.
# 
# ## RGF hyperparameters
# Here we shall mention two of the [RGF parameters](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst#432-parameters-to-control-training) that control training:
# 
# `algorithm=`
# * `RGF`: RGF with $L_2$ regularization on leaf-only models. (default)
# * `RGF_Opt`: RGF with min-penalty regularization.
# * `RGF_Sib`: RGF with min-penalty regularization with the sum-to-zero sibling constraints.
# 
# `loss=`
# * `LS`: square loss (default)
# * `Expo`: exponential loss
# * `Log`: logistic loss
# 
# # References
# * [rgf_python](https://github.com/RGF-team/rgf/tree/master/python-package) on GitHub
# * [Regularized Greedy Forest in C++: User Guide](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst)
# * [FastRGF](https://github.com/RGF-team/rgf/tree/master/) A variant developed to be used with large (and sparse) datasets.
# * [Rie Johnson and Tong Zhang "Learning Nonlinear Functions Using Regularized Greedy Forest", IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume: 36 , Issue: 5  pp. 942-954 (2014)](https://dx.doi.org/10.1109/TPAMI.2013.159) ([arXiv](https://arxiv.org/abs/1109.0887))
# 
# # Related reading
# * [Regularized Greedy Forest – The Scottish Play (Act I)](https://www.statworx.com/at/blog/regularized-greedy-forest-the-scottish-play-act-i/) by Fabian Müller
# * [Regularized Greedy Forest – The Scottish Play (Act II)](https://www.statworx.com/de/blog/regularized-greedy-forest-the-scottish-play-act-ii/) by Fabian Müller
# * [An Introductory Guide to Regularized Greedy Forests (RGF) with a case study in Python](https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/) by Ankit Choudhary
