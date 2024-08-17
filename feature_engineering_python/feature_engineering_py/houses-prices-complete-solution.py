#!/usr/bin/env python
# coding: utf-8

# # House Prices - Kaggle Copetitions
# ![image](https://i0.wp.com/nycdatascience.com/blog/wp-content/uploads/2017/11/neighborhood-puzzle.jpg)
# __Introduction__:
# 
# Develop your skills in data science through a useful and common case to everyone, the forecast of the sale prices of houses. With this premise there is a plethora of available data, where we can highlight the data set of [Boston Housing](http://lib.stat.cmu.edu/datasets/boston) from Harrison and Rubinfeld (1978) as the most famous for beginners and already much explored.
# 
# But after so many years would this data set present enough complexity and challenge for students to practice a whole range of new knowledge acquired in their courses in statistics and data science? Would it have a sufficient number of observations and features to make it necessary to analyze outliers, collinearity, multicollinearity, the need for selection and reduction of dimensionality of features, it to not mention its applicability to more modern machine learning techniques and algorithms?
# 
# Not in the opinion of [Dr Dean De Cock](https://www.linkedin.com/in/dean-de-cock-b5336537/), Professor of Statistics at Truman State University. He was looking for a data set that would allow students the opportunity to display the skills they had learned within the class.
# 
# In his quest, he finally found with the Ames City Assessor's Office a potential data set. The initial Excel file contained 113 variables describing 3970 property sales that had occurred in Ames, Iowa between 2006 and 2010. The variables were a mix of nominal, ordinal, continuous, and discrete variables used in calculation of assessed values and included physical property measurements in addition to computation variables used in the city's assessment process. For Dr Dean purposes, a "layman's" data set that could be easily understood by users at all levels was desirable; so He began his project by removing any variables that required special knowledge or previous calculations for their use. Most of these deleted variables were related to weighting and adjustment factors used in the city's current modeling system. 
# 
# In the end, the selected dataset has 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The case was so interesting that it was introduced as one of the competitions for beginners of Kaggle, quickly becoming one of the most popular. If you interest to get some insights and suggestions directly from Dr Dean before start, I recommend that you read his [paper](http://ww2.amstat.org/publications/jse/v19n3/decock.pdf).
# 
# With this spirit, I create this material, not with the purpose of competing, but of intended to cover most of the techniques of data analysis in Python for regression analysis. That is why it follows the natural flow of ML and contains many texts and links to the techniques, made your conference and references easy. as it can be extended over time.
# 
# In order not to fall into monotony, sometimes I take some liberties and apply a little humor, but nothing that compromises the accuracy of knowledge that is being acquired by the reader. Of course, anyone who wants to contribute some addition or even a claim or choreography will be very welcome. But what should you be thinking now? So, enough of tangle and let's go.
# [![image](http://www.conradopaulinoadv.com.br/v2/wp-content/uploads/2016/05/2-600x350.jpg)](https://www.youtube.com/watch?v=xdt13wtIlVs)
# __Competition Description__:
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this [competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle challenges you to predict the final price of each home.
# 
# Practice Skills
# - Creative feature engineering 
# - Advanced regression techniques like random forest and gradient boosting
# 
# For a detailed description of the data set, click [here](../input/data_description.txt).

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Preparing-environment-and-uploading-data" data-toc-modified-id="Preparing-environment-and-uploading-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Preparing environment and uploading data</a></span><ul class="toc-item"><li><span><a href="#Import-Packages" data-toc-modified-id="Import-Packages-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Import Packages</a></span></li><li><span><a href="#Load-Datasets" data-toc-modified-id="Load-Datasets-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Load Datasets</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis-(EDA)" data-toc-modified-id="Exploratory-Data-Analysis-(EDA)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis (EDA)</a></span><ul class="toc-item"><li><span><a href="#Take-a-First-Look-of-our-Data:" data-toc-modified-id="Take-a-First-Look-of-our-Data:-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Take a First Look of our Data:</a></span></li><li><span><a href="#Some-Observations-from-the-STR-Details:" data-toc-modified-id="Some-Observations-from-the-STR-Details:-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Some Observations from the STR Details:</a></span></li><li><span><a href="#First-see-of-some-stats-of-Numeric-Data" data-toc-modified-id="First-see-of-some-stats-of-Numeric-Data-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>First see of some stats of Numeric Data</a></span></li><li><span><a href="#Overall-Quality" data-toc-modified-id="Overall-Quality-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Overall Quality</a></span></li><li><span><a href="#Total-Rooms-above-Ground-and-Living-Area" data-toc-modified-id="Total-Rooms-above-Ground-and-Living-Area-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Total Rooms above Ground and Living Area</a></span></li><li><span><a href="#Garage-areas-and-parking" data-toc-modified-id="Garage-areas-and-parking-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Garage areas and parking</a></span></li><li><span><a href="#Total-Basement-Area-Vs-1st-Flor-Area" data-toc-modified-id="Total-Basement-Area-Vs-1st-Flor-Area-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Total Basement Area Vs 1st Flor Area</a></span></li><li><span><a href="#Year-Built-Vs-Garage-Year-Built" data-toc-modified-id="Year-Built-Vs-Garage-Year-Built-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>Year Built Vs Garage Year Built</a></span></li><li><span><a href="#Bathrooms-Features" data-toc-modified-id="Bathrooms-Features-2.9"><span class="toc-item-num">2.9&nbsp;&nbsp;</span>Bathrooms Features</a></span></li><li><span><a href="#Reviwe-Porch-Features:" data-toc-modified-id="Reviwe-Porch-Features:-2.10"><span class="toc-item-num">2.10&nbsp;&nbsp;</span>Reviwe Porch Features:</a></span></li><li><span><a href="#Slope-of-property-and-Lot-area" data-toc-modified-id="Slope-of-property-and-Lot-area-2.11"><span class="toc-item-num">2.11&nbsp;&nbsp;</span>Slope of property and Lot area</a></span></li><li><span><a href="#Neighborhood" data-toc-modified-id="Neighborhood-2.12"><span class="toc-item-num">2.12&nbsp;&nbsp;</span>Neighborhood</a></span></li><li><span><a href="#Check-the-Dependent-Variable---SalePrice:" data-toc-modified-id="Check-the-Dependent-Variable---SalePrice:-2.13"><span class="toc-item-num">2.13&nbsp;&nbsp;</span>Check the Dependent Variable - SalePrice:</a></span></li><li><span><a href="#Test-hypothesis-of-better-feature:-Construction-Area" data-toc-modified-id="Test-hypothesis-of-better-feature:-Construction-Area-2.14"><span class="toc-item-num">2.14&nbsp;&nbsp;</span>Test hypothesis of better feature: Construction Area</a></span></li></ul></li><li><span><a href="#3.-Check-Data-Quality:" data-toc-modified-id="3.-Check-Data-Quality:-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>3. Check Data Quality:</a></span><ul class="toc-item"><li><span><a href="#Nulls-Check:" data-toc-modified-id="Nulls-Check:-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Nulls Check:</a></span></li><li><span><a href="#Some-Observations-Respect-Data-Quality:" data-toc-modified-id="Some-Observations-Respect-Data-Quality:-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Some Observations Respect Data Quality:</a></span></li><li><span><a href="#Identify-the-Most-Common-Electrical:" data-toc-modified-id="Identify-the-Most-Common-Electrical:-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Identify the Most Common Electrical:</a></span></li><li><span><a href="#Fill-Missing-Values-of-Garage-Features:" data-toc-modified-id="Fill-Missing-Values-of-Garage-Features:-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Fill Missing Values of Garage Features:</a></span><ul class="toc-item"><li><span><a href="#Group-by-GarageType" data-toc-modified-id="Group-by-GarageType-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>Group by GarageType</a></span></li><li><span><a href="#Check-if-all-nulls-of-Garage-features-are-inputed" data-toc-modified-id="Check-if-all-nulls-of-Garage-features-are-inputed-3.4.2"><span class="toc-item-num">3.4.2&nbsp;&nbsp;</span>Check if all nulls of Garage features are inputed</a></span></li></ul></li><li><span><a href="#Masonry-veneer" data-toc-modified-id="Masonry-veneer-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Masonry veneer</a></span><ul class="toc-item"><li><span><a href="#Correct-masonry-veneer-types" data-toc-modified-id="Correct-masonry-veneer-types-3.5.1"><span class="toc-item-num">3.5.1&nbsp;&nbsp;</span>Correct masonry veneer types</a></span></li><li><span><a href="#Check-if-all-nulls-of-masonry-veneer-types-are-updated" data-toc-modified-id="Check-if-all-nulls-of-masonry-veneer-types-are-updated-3.5.2"><span class="toc-item-num">3.5.2&nbsp;&nbsp;</span>Check if all nulls of masonry veneer types are updated</a></span></li></ul></li><li><span><a href="#Check-and-Input-Basement-Features-Nulls:" data-toc-modified-id="Check-and-Input-Basement-Features-Nulls:-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Check and Input Basement Features Nulls:</a></span></li><li><span><a href="#Lot-Frontage---Check-and-Fill-Nulls" data-toc-modified-id="Lot-Frontage---Check-and-Fill-Nulls-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Lot Frontage - Check and Fill Nulls</a></span></li><li><span><a href="#Pool-Quality---Fill-Nulls" data-toc-modified-id="Pool-Quality---Fill-Nulls-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Pool Quality - Fill Nulls</a></span></li><li><span><a href="#Functional---Miss-Values-Treatment" data-toc-modified-id="Functional---Miss-Values-Treatment-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Functional - Miss Values Treatment</a></span></li><li><span><a href="#Fireplace-Quality---Miss-Values-Treatment" data-toc-modified-id="Fireplace-Quality---Miss-Values-Treatment-3.10"><span class="toc-item-num">3.10&nbsp;&nbsp;</span>Fireplace Quality - Miss Values Treatment</a></span></li><li><span><a href="#Kitchen-Quality---Miss-Values-Treatment" data-toc-modified-id="Kitchen-Quality---Miss-Values-Treatment-3.11"><span class="toc-item-num">3.11&nbsp;&nbsp;</span>Kitchen Quality - Miss Values Treatment</a></span></li><li><span><a href="#Alley,-Fence-and-Miscellaneous-Feature---Miss-Values-Treatment" data-toc-modified-id="Alley,-Fence-and-Miscellaneous-Feature---Miss-Values-Treatment-3.12"><span class="toc-item-num">3.12&nbsp;&nbsp;</span>Alley, Fence and Miscellaneous Feature - Miss Values Treatment</a></span></li><li><span><a href="#Back-to-the-Past!-Garage-Year--Build-from-2207" data-toc-modified-id="Back-to-the-Past!-Garage-Year--Build-from-2207-3.13"><span class="toc-item-num">3.13&nbsp;&nbsp;</span>Back to the Past! Garage Year  Build from 2207</a></span></li><li><span><a href="#Final-Check-and-Filling-Nulls" data-toc-modified-id="Final-Check-and-Filling-Nulls-3.14"><span class="toc-item-num">3.14&nbsp;&nbsp;</span>Final Check and Filling Nulls</a></span></li></ul></li><li><span><a href="#Mapping-Ordinal-Features" data-toc-modified-id="Mapping-Ordinal-Features-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Mapping Ordinal Features</a></span></li><li><span><a href="#Feature-Engineering:-Create-New-Features:" data-toc-modified-id="Feature-Engineering:-Create-New-Features:-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature Engineering: Create New Features:</a></span><ul class="toc-item"><li><span><a href="#Include-pool-in-the-Miscellaneous-features" data-toc-modified-id="Include-pool-in-the-Miscellaneous-features-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Include pool in the Miscellaneous features</a></span></li><li><span><a href="#Points-Review" data-toc-modified-id="Points-Review-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Points Review</a></span></li><li><span><a href="#One-Hot-Encode-Categorical-Features" data-toc-modified-id="One-Hot-Encode-Categorical-Features-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>One Hot Encode Categorical Features</a></span><ul class="toc-item"><li><span><a href="#Removing-Dummies-with--none-observations-in-train-or-test-datasets" data-toc-modified-id="Removing-Dummies-with--none-observations-in-train-or-test-datasets-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>Removing Dummies with  none observations in train or test datasets</a></span></li></ul></li><li><span><a href="#Transform-Years-to-Ages-and-Create-Flags-to-New-and-Remod" data-toc-modified-id="Transform-Years-to-Ages-and-Create-Flags-to-New-and-Remod-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Transform Years to Ages and Create Flags to New and Remod</a></span></li><li><span><a href="#Check-for-any-correlations-between-features" data-toc-modified-id="Check-for-any-correlations-between-features-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Check for any correlations between features</a></span><ul class="toc-item"><li><span><a href="#Drop-the-features-with-highest-correlations-to-other-Features:" data-toc-modified-id="Drop-the-features-with-highest-correlations-to-other-Features:-5.5.1"><span class="toc-item-num">5.5.1&nbsp;&nbsp;</span>Drop the features with highest correlations to other Features:</a></span></li><li><span><a href="#Identify--and-treat-multicollinearity:" data-toc-modified-id="Identify--and-treat-multicollinearity:-5.5.2"><span class="toc-item-num">5.5.2&nbsp;&nbsp;</span>Identify  and treat multicollinearity:</a></span></li></ul></li><li><span><a href="#Defining-Categorical-and-Boolean-Data-as-unit8-types" data-toc-modified-id="Defining-Categorical-and-Boolean-Data-as-unit8-types-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Defining Categorical and Boolean Data as unit8 types</a></span></li><li><span><a href="#Box-cox-transformation-of-highly-skewed-features" data-toc-modified-id="Box-cox-transformation-of-highly-skewed-features-5.7"><span class="toc-item-num">5.7&nbsp;&nbsp;</span>Box cox transformation of highly skewed features</a></span></li><li><span><a href="#Evaluate-Apply-Polynomials-by-Region-Plots-on-the-more-Correlated-Features" data-toc-modified-id="Evaluate-Apply-Polynomials-by-Region-Plots-on-the-more-Correlated-Features-5.8"><span class="toc-item-num">5.8&nbsp;&nbsp;</span>Evaluate Apply Polynomials by Region Plots on the more Correlated Features</a></span><ul class="toc-item"><li><span><a href="#Evaluating-Polynomials-Options-Performance" data-toc-modified-id="Evaluating-Polynomials-Options-Performance-5.8.1"><span class="toc-item-num">5.8.1&nbsp;&nbsp;</span>Evaluating Polynomials Options Performance</a></span></li><li><span><a href="#Create-Degree-3-Polynomials-Features" data-toc-modified-id="Create-Degree-3-Polynomials-Features-5.8.2"><span class="toc-item-num">5.8.2&nbsp;&nbsp;</span>Create Degree 3 Polynomials Features</a></span></li></ul></li></ul></li><li><span><a href="#Separate-Train,-Test-Datasets,-identifiers-and-Dependent-Variable" data-toc-modified-id="Separate-Train,-Test-Datasets,-identifiers-and-Dependent-Variable-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Separate Train, Test Datasets, identifiers and Dependent Variable</a></span></li><li><span><a href="#Select-Features" data-toc-modified-id="Select-Features-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Select Features</a></span><ul class="toc-item"><li><span><a href="#Prepare-Data-to-Select-Features" data-toc-modified-id="Prepare-Data-to-Select-Features-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Prepare Data to Select Features</a></span></li><li><span><a href="#Wrapper-Methods" data-toc-modified-id="Wrapper-Methods-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Wrapper Methods</a></span><ul class="toc-item"><li><span><a href="#Backward-Elimination" data-toc-modified-id="Backward-Elimination-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>Backward Elimination</a></span><ul class="toc-item"><li><span><a href="#Backward-Elimination-By-P-values" data-toc-modified-id="Backward-Elimination-By-P-values-7.2.1.1"><span class="toc-item-num">7.2.1.1&nbsp;&nbsp;</span>Backward Elimination By P-values</a></span></li></ul></li><li><span><a href="#Select-Features-by-Recursive-Feature-Elimination" data-toc-modified-id="Select-Features-by-Recursive-Feature-Elimination-7.2.2"><span class="toc-item-num">7.2.2&nbsp;&nbsp;</span>Select Features by Recursive Feature Elimination</a></span></li><li><span><a href="#Sequential-feature-selection" data-toc-modified-id="Sequential-feature-selection-7.2.3"><span class="toc-item-num">7.2.3&nbsp;&nbsp;</span>Sequential feature selection</a></span></li></ul></li><li><span><a href="#Feature-Selection-by-Filter-Methods" data-toc-modified-id="Feature-Selection-by-Filter-Methods-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Feature Selection by Filter Methods</a></span><ul class="toc-item"><li><span><a href="#Univariate-feature-selection" data-toc-modified-id="Univariate-feature-selection-7.3.1"><span class="toc-item-num">7.3.1&nbsp;&nbsp;</span>Univariate feature selection</a></span></li></ul></li><li><span><a href="#Select-Features-by-Embedded-Methods" data-toc-modified-id="Select-Features-by-Embedded-Methods-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Select Features by Embedded Methods</a></span><ul class="toc-item"><li><span><a href="#Feature-Selection-by-Gradient-Boosting" data-toc-modified-id="Feature-Selection-by-Gradient-Boosting-7.4.1"><span class="toc-item-num">7.4.1&nbsp;&nbsp;</span>Feature Selection by Gradient Boosting</a></span></li></ul></li><li><span><a href="#Separate-data-for-modeling" data-toc-modified-id="Separate-data-for-modeling-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Separate data for modeling</a></span><ul class="toc-item"><li><span><a href="#Feature-Selection-into-the-Pipeline" data-toc-modified-id="Feature-Selection-into-the-Pipeline-7.5.1"><span class="toc-item-num">7.5.1&nbsp;&nbsp;</span>Feature Selection into the Pipeline</a></span></li></ul></li></ul></li><li><span><a href="#Compressing-Data-via-Dimensionality-Reduction" data-toc-modified-id="Compressing-Data-via-Dimensionality-Reduction-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Compressing Data via Dimensionality Reduction</a></span><ul class="toc-item"><li><span><a href="#PCA" data-toc-modified-id="PCA-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>PCA</a></span></li></ul></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Modeling</a></span><ul class="toc-item"><li><span><a href="#Model-Hyper-Parametrization" data-toc-modified-id="Model-Hyper-Parametrization-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Model Hyper Parametrization</a></span><ul class="toc-item"><li><span><a href="#Evaluate-Results" data-toc-modified-id="Evaluate-Results-9.1.1"><span class="toc-item-num">9.1.1&nbsp;&nbsp;</span>Evaluate Results</a></span></li></ul></li><li><span><a href="#Residuals-Plots" data-toc-modified-id="Residuals-Plots-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Residuals Plots</a></span></li><li><span><a href="#Model-Hiperparametrization" data-toc-modified-id="Model-Hiperparametrization-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Model Hiperparametrization</a></span><ul class="toc-item"><li><span><a href="#Lasso-(Least-Absolute-Shrinkage-and-Selection-Operator)" data-toc-modified-id="Lasso-(Least-Absolute-Shrinkage-and-Selection-Operator)-9.3.1"><span class="toc-item-num">9.3.1&nbsp;&nbsp;</span>Lasso (Least Absolute Shrinkage and Selection Operator)</a></span></li><li><span><a href="#XGBRegressor" data-toc-modified-id="XGBRegressor-9.3.2"><span class="toc-item-num">9.3.2&nbsp;&nbsp;</span>XGBRegressor</a></span></li><li><span><a href="#Gradient-Boosting-Regressor" data-toc-modified-id="Gradient-Boosting-Regressor-9.3.3"><span class="toc-item-num">9.3.3&nbsp;&nbsp;</span>Gradient Boosting Regressor</a></span></li><li><span><a href="#ElasticNet" data-toc-modified-id="ElasticNet-9.3.4"><span class="toc-item-num">9.3.4&nbsp;&nbsp;</span>ElasticNet</a></span></li><li><span><a href="#Bayesian-Ridge-Regression" data-toc-modified-id="Bayesian-Ridge-Regression-9.3.5"><span class="toc-item-num">9.3.5&nbsp;&nbsp;</span>Bayesian Ridge Regression</a></span></li><li><span><a href="#Linear-Regression" data-toc-modified-id="Linear-Regression-9.3.6"><span class="toc-item-num">9.3.6&nbsp;&nbsp;</span>Linear Regression</a></span></li><li><span><a href="#Orthogonal-Matching-Pursuit-model-(OMP)" data-toc-modified-id="Orthogonal-Matching-Pursuit-model-(OMP)-9.3.7"><span class="toc-item-num">9.3.7&nbsp;&nbsp;</span>Orthogonal Matching Pursuit model (OMP)</a></span></li><li><span><a href="#Robust-Regressor" data-toc-modified-id="Robust-Regressor-9.3.8"><span class="toc-item-num">9.3.8&nbsp;&nbsp;</span>Robust Regressor</a></span></li><li><span><a href="#Passive-Aggressive-Regressor" data-toc-modified-id="Passive-Aggressive-Regressor-9.3.9"><span class="toc-item-num">9.3.9&nbsp;&nbsp;</span>Passive Aggressive Regressor</a></span></li><li><span><a href="#SGD-Regressor" data-toc-modified-id="SGD-Regressor-9.3.10"><span class="toc-item-num">9.3.10&nbsp;&nbsp;</span>SGD Regressor</a></span></li></ul></li></ul></li><li><span><a href="#Check-the-best-results-from-the-models-hyper-parametrization" data-toc-modified-id="Check-the-best-results-from-the-models-hyper-parametrization-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Check the best results from the models hyper parametrization</a></span></li><li><span><a href="#Stacking-the-Models" data-toc-modified-id="Stacking-the-Models-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Stacking the Models</a></span></li><li><span><a href="#Create-Submission-File:" data-toc-modified-id="Create-Submission-File:-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Create Submission File:</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# ## Preparing environment and uploading data
# ### Import Packages

# In[ ]:


import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import pylab 

from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from scipy.special import boxcox1p
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso, LassoLarsIC, ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ### Load Datasets

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

train.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
test.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

test['SalePrice'] = 0


# ## Exploratory Data Analysis (EDA)
# ![image](http://visualoop.com/media/2013/09/Dilbert-680x490.png)
# ### Take a First Look of our Data:
# I created the function below to simplify the analysis of general characteristics of the data. Inspired on the str function of R, this function returns the types, counts, distinct, count nulls, missing ratio and uniques values of each field/feature.
# 
# If the study involve some supervised learning, this function can return the study of the correlation, for this we just need provide the dependent variable to the pred parameter.
# 
# Also, if its return is stored in a variable you can evaluate it in more detail, focus on specific field, or sort them from different perspectives.

# In[ ]:


def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# In[ ]:


details = rstr(train, 'SalePrice')
display(details.sort_values(by='corr SalePrice', ascending=False))


# ### Some Observations from the STR Details:
# ![image](https://imgs.xkcd.com/comics/science_valentine.png)
# - The dependent variabel, **SalePrice**, are ***skewed*** and ***heavy-tailed distribution***. We need investigate its distribution with a plot and check if a **transformation by Log 1P** could correct it, withou drop most of the **outiliers**.
# <p>
#  
# - Nulls: The data have 19 features with nulls, five of then area categorical and with more then 47% of missing ration. They are candidates to drop or use them to create another more interesting feature:
#  - PoolQC
#  - MiscFeature
#  - Alley
#  - Fence
#  - FireplaceQu
# 
# <p>
#     
# - Features ***high skewed right***, ***heavy-tailed distribution***, and with ***high correlation*** to Sales Price. It is important to treat them (boxcox 1p transformation, Robustscaler, and drop some outliers):
#  - TotalBsmtSF
#  - 1stFlrSF
#  - GrLivArea
# 
# <p>
#     
# - Features ***skewed***, ***heavy-tailed distribution***, and with ***good correlation*** to Sales Price. It is important to treat them (boxcox 1p transformation, Robustscaler, and drop some outliers):
#  - LotArea
#  - KitchenAbvGr
#  - ScreenPorch
#  - EnclosedPorch
#  - MasVnrArea
#  - OpenPorchSF
#  - LotFrontage
#  - BsmtFinSF1
#  - WoodDeckSF
#  - MSSubClass
# 
# <p>
#     
# - Features ***high skewed***, ***heavy-tailed distribution***, and with ***low correlation*** to Sales Price. Maybe we can drop these features, or just use they with other to create a new more importants feature:
#  - MiscVal
#  - TSsnPorch
#  - LowQualFinSF
#  - BsmtFinSF2
#  - BsmtHalfBa
# 
# <p>
#     
# - Features ***low skewed***, and with ***good to low correlation*** to Sales Price. Just use a Robustscaler probably reduce the few  distorcions:
#  - BsmtUnfSF
#  - 2ndFlrSF
#  - TotRmsAbvGrd
#  - HalfBath
#  - Fireplaces
#  - BsmtFullBath
#  - OverallQual
#  - BedroomAbvGr
#  - GarageArea
#  - FullBath
#  - GarageCars
#  - OverallCond
# 
# <p>
#     
# - Transforme from Yaer Feature to Age, 2011 - Year feature, or YEAR(TODAY()) - Year Feature
#  - YearRemodAdd: 
#  - YearBuilt
#  - GarageYrBlt
#  - YrSold
# 
# If we apply this data to a Keras, first we need to chnage the float64 and Int64 to float32 and Int32!

# ### First see of some stats of Numeric Data
# So, for the main statistics of our numeric data describe the function (like the summary of R)

# In[ ]:


display(train.describe().transpose())


# ### Overall Quality

# It is not surprise that overall quality has the highest correlation with SalePrice among the numeric variables (0.79). It rates the overall material and finish of the house on a scale from 1 (very poor) to 10 (very excellent). The positive correlation is certainly there indeed, and seems to be a slightly upward curve. Regarding outliers, I do not see any extreme values. If there is a candidate to take out as an outlier later on, it seems to be the expensive house with grade 4.
# 
# Especially the two houses with really big living areas and low SalePrices seem outliers. I will not take them out yet, as taking outliers can be dangerous. For instance, a low score on the Overall Quality could explain a low price. However, as you can see below, these two houses actually also score maximum points on Overall Quality. Therefore, I will keep theses houses in mind as prime candidates to take out as outliers.
# ![quality](https://rew-feed-images.global.ssl.fastly.net/cimls_rspearman/a/residential/49908-1-m.jpg)

# In[ ]:


fig = plt.figure(figsize=(20, 15))
sns.set(font_scale=1.5)

# (Corr= 0.817185) Box plot overallqual/salePrice
fig1 = fig.add_subplot(221); sns.boxplot(x='OverallQual', y='SalePrice', data=train[['SalePrice', 'OverallQual']])

# (Corr= 0.700927) GrLivArea vs SalePrice plot
fig2 = fig.add_subplot(222); 
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

# (Corr= 0.680625) GarageCars vs SalePrice plot
fig3 = fig.add_subplot(223); 
sns.scatterplot(x = train.GarageCars, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

# (Corr= 0.650888) GarageArea vs SalePrice plot
fig4 = fig.add_subplot(224); 
sns.scatterplot(x = train.GarageArea, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

fig5 = plt.figure(figsize=(16, 8))
fig6 = fig5.add_subplot(121); 
sns.scatterplot(y = train.SalePrice , x = train.TotalBsmtSF, hue=train.OverallQual, palette= 'YlOrRd')

fig7 = fig5.add_subplot(122); 
sns.scatterplot(y = train.SalePrice, x = train['1stFlrSF'], hue=train.OverallQual, palette= 'YlOrRd')

plt.tight_layout(); plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(121)
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, ax = ax)

#Deleting outliers
train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<300000)].index)

#Check the graphic again
ax = fig.add_subplot(122)
sns.scatterplot(x =train.GrLivArea, y = train.SalePrice, ax = ax)
plt.show()


# ### Total Rooms above Ground and Living Area
# From a previews experience with Boston data set, you probably main expect to much from the total rooms above ground, as its 'RM' feature (the average number of rooms per dwelling), but here is not the same scenario. Our common sense make to think that live area maybe has some correlation to it and probably we can combine this two features to produce a better predictor. Let's see.
# ![image](https://www.housing.iastate.edu/sites/default/files/imported//images/floorplans/Frederiksen-4BR.gif)

# In[ ]:


sns.reset_defaults()
sns.set(style="ticks", color_codes=True)

df = train[['SalePrice', 'GrLivArea', 'TotRmsAbvGrd']]
df['GrLivAreaByRms'] = train.GrLivArea/train.TotRmsAbvGrd
df['GrLivArea_x_Rms'] = train.GrLivArea*train.TotRmsAbvGrd
fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='GrLivAreaByRms', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GrLivAreaByRms.corr(df['SalePrice'])))
fig2 = fig.add_subplot(122); sns.regplot(x='GrLivArea_x_Rms', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=30000, y=100000, s='Correlation with SalePrice: {:1.4f}'.format(df.GrLivArea_x_Rms.corr(df['SalePrice'])))

print('                                                                  Outliers:',(df.GrLivArea_x_Rms>=45000).sum())
df = df.loc[df.GrLivArea_x_Rms<45000]
sns.regplot(x='GrLivArea_x_Rms', y='SalePrice', data=df); 
plt.title('Living Area has beter correlation ({:1.2f}) than It Multply by Rooms!'.format(df.GrLivArea.corr(df.SalePrice)))
plt.text(x=30000, y=50000, s='Correlation withou Outliers: {:1.4f}'.format(df.GrLivArea_x_Rms.corr(df['SalePrice'])))
plt.show()
del df


# As we can see, the interaction between the two features did not present a better correlation than that already seen in the living area, include it improves to 0.74 with the cut of the outliers.
# 
# On the other hand, the ***multiplication*** not only demonstrated the living area **outliers** already identified, but it still **emphasized another**. If the strategy is to ***drop the TotRmsAbvGrd***, we should also ***exclude this additional outlier***.

# In[ ]:


train = train[train.GrLivArea * train.TotRmsAbvGrd < 45000]
print('Train observations after remove outliers:',train.shape[0])


# ### Garage areas and parking
# From the boxplot below, we can note that more than 3 parking cars and more than 900 of area are outliers, since a few number of their observations. Although there is a relationship between them, most likely with a smaller number of parking spaces, there may be more garage area for other purposes, reason why the correlation between them is 0.88 and not 1.
# ![image](https://thumbs.gfycat.com/TangiblePleasingHousefly-size_restricted.gif)

# In[ ]:


fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(131); sns.boxplot(train.GarageCars)
fig2 = fig.add_subplot(132); sns.boxplot(train.GarageArea)
fig3 = fig.add_subplot(133); sns.boxplot(train.GarageCars, train.GarageArea)
plt.show()


# In[ ]:


df = train[['SalePrice', 'GarageArea', 'GarageCars']]
df['GarageAreaByCar'] = train.GarageArea/train.GarageCars
df['GarageArea_x_Car'] = train.GarageArea*train.GarageCars

fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='GarageAreaByCar', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageAreaByCar.corr(df['SalePrice'])))

fig2 = fig.add_subplot(122); sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=-100, y=750000, s='Correlation with SalePrice: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
print('                                                                 Outliers:',(df.GarageArea_x_Car>=3700).sum())
df = df.loc[df.GarageArea_x_Car<3700]
sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.title('Garage Area Multiply By Cars is the best!')
plt.text(x=-100, y=700000, s='Correlation withou Outliers: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
plt.show()
del df


# As can be seen the area by car is little useful, but contrary to common sense the multiplication of the area by the number of vacancies yes is. In the division we lose the magnitude and we have to maintain one or another functionality to recover it. With the multiplication we solve the problem of 1 parking space of 10 square feet against another of 10 with 1 square feet each. We could still ***improve the correlation*** by **0.06**, already considering the exclusion of only 4 outliers. 
# 
# The identification of the outliers was facilitated, note that before we would have a greater number of outliers, since the respective of each features alone are not coincident.
# ![garage](https://ap.rdcpix.com/517018206/36a0594038b86ae431068cc483092fe6l-m0xd-w480_h480_q80.jpg)
# So let's continue with the multiplication strategy, remove the two original metrics that have high correlation with each other, and exclude the 4 outliers from the training base.

# In[ ]:


train = train[train.GarageArea * train.GarageCars < 3700]
print('Total observatiosn after outliers cut:', train.shape[0])


# ### Total Basement Area Vs 1st Flor Area
# In our country it is not common to have Basement, I think we thought it was a little spooky. So I looked a bit more "carefully" at this variable...
# ![image](https://lparchive.org/Scooby-Doo-Mystery/Update%2002/46-Fusion_2012-08-28_01-59-20-18.png)
# I noticed that in Ames has a lot of variation, but the predictive effect is very small, so I decided to study its composition with the first floor.

# In[ ]:


df = train[['SalePrice', 'TotalBsmtSF', '1stFlrSF']]
df['TotalBsmtSFByBms'] = train.TotalBsmtSF/train['1stFlrSF']
df['TotalBsmtSF_x_Bsm'] = train.TotalBsmtSF*train['1stFlrSF']
fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='TotalBsmtSFByBms', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalBsmtSFByBms.corr(df['SalePrice'])))
fig2 = fig.add_subplot(122); sns.regplot(x='TotalBsmtSF_x_Bsm', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=7e06, y=90000, s='Correlation with SalePrice: {:1.4f}'.format(df.TotalBsmtSF_x_Bsm.corr(df['SalePrice'])))

print('                                                             Outliers:',(df.TotalBsmtSF_x_Bsm>=0.9e07).sum())
df = df.loc[df.TotalBsmtSF_x_Bsm<0.9e07]
sns.regplot(x='TotalBsmtSF_x_Bsm', y='SalePrice', data=df); 
plt.title('The multiplacation is better than total basement correlation ({:1.2f}) after outliers cut!'.format(df.TotalBsmtSF.corr(df.SalePrice)))
plt.text(x=7e06, y=50000, s='Correlation withou Outliers: {:1.4f}'.format(df.TotalBsmtSF_x_Bsm.corr(df['SalePrice'])))
plt.show()
del df


# [![image](https://www.wcibasementrepair.com/wp-content/themes/wci/images/home-popupdots.png)](https://www.manta.com/cost-basement-waterproofing-ames-ia)
# Similar to what we saw in the garage analysis, we again have a better correlation by multiplying the variables, but now we don't have a significant gain with outliers exclusion. So let's continue with the multiplication strategy and remove only the two original metrics that have high correlation with each other.

# ### Year Built Vs Garage Year Built
# Of course when we buy a property the date of its construction makes a lot of difference as it can be a source of great headaches. Depending on the age and conditions there will be need for renovations and very old houses there may be cases where the garage has been built or refit after the house itself.
# ![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROzLQY3lcdYJimrBS7fHjLE0vhecqf1HTCfBANuDX5_5ZGBv0b)
# Well, I'd be more worried about the plumbing, the electricity, ... the garage is only for car and trunk, or is it not? Is that so? it will be?
# 
# So, let's see the graphs below, and confirm that this two features are highly correlated, but as expect is not easy to find a good substitute by iteration.

# In[ ]:


df = train[['SalePrice', 'YearBuilt', 'GarageYrBlt']]
df['YearBuilt_x_Garage'] = train.YearBuilt*train.GarageYrBlt
df['Garage_Newest'] = train.YearBuilt < train.GarageYrBlt

fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.scatterplot(y = df.SalePrice, x = df.YearBuilt, hue=df.GarageYrBlt, palette= 'YlOrRd')
fig2 = fig.add_subplot(122); sns.regplot(x='YearBuilt_x_Garage', y='SalePrice', data=df); 
plt.text(x=3700000, y=600000, s='YearBuilt Correlation with SalePrice: {:1.4f}'.format(df.YearBuilt.corr(df['SalePrice'])))
plt.text(x=3700000, y=550000, s='Correlation with SalePrice: {:1.4f}'.format(df.YearBuilt_x_Garage.corr(df['SalePrice'])))
plt.show()


# However, by making the year of construction of the garage an indicator of whether it is newer, it becomes easiest to identify a pattern of separation. 
# 
# And more, note that we have a rising price due to the lower age. Maybe the old cars had the garage would only be for themselves...
# ![image](https://www.corsia.us/wp-content/uploads/2016/05/cars-period-1909-taft-white-steam-car-1-800x533-538x218.jpg)
# ..., or put it in the barn. Today we must have other more usable uses for garage, right...?
# ![image](http://bonjourmini.com/wp-content/uploads/2018/05/garage-man-cave-how-to-create-a-man-cave-garage-more-best-flooring-for-garage-man-cave.jpg)

# In[ ]:


sns.lmplot(y = 'SalePrice', x = 'YearBuilt', data=df, markers='.', 
           aspect=1.4, height=4, hue= 'Garage_Newest', palette= 'YlOrRd')
plt.show();  
del df


# But note that although we have a rising price the newer the house, the growth rate is very smooth, even with the rate gain with a newer garage. This makes sense, given that the prices of these regressors are meeting with the mean price of each year.

# ### Bathrooms Features
# It's time to take a break and go to the toilet, to our luck there are 4 bathroom variables in our data set. FullBath has the largest correlation with SalePrice between than. The others individually, these features are not very important. 
# 

# In[ ]:


fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(221); sns.regplot(x='FullBath', y='SalePrice', data=train)
plt.title('Correlation with SalePrice: {:6.4f}'.format(train.FullBath.corr(train['SalePrice'])))

fig2 = fig.add_subplot(222); sns.regplot(x='HalfBath', y='SalePrice', data=train);
plt.title('Correlation with SalePrice: {:6.4f}'.format(train.HalfBath.corr(train['SalePrice'])))

fig3 = fig.add_subplot(223); sns.regplot(x='BsmtFullBath', y='SalePrice', data=train)
plt.title('Correlation with SalePrice: {:6.4f}'.format(train.BsmtFullBath.corr(train['SalePrice'])))

fig4 = fig.add_subplot(224); sns.regplot(x='BsmtHalfBath', y='SalePrice', data=train);
plt.title('Correlation with SalePrice: {:6.4f}'.format(train.HalfBath.corr(train['SalePrice'])))

plt.show()


# ![image](http://www.danlanephotography.com/wp-content/uploads/20-funny-toilet-paper-holders-funny-toilet-paper-holders.jpg)
# However, I assume that I if I add them up into one predictor, this predictor is likely to become a strong one. A half-bath, also known as a powder room or guest bath, has only two of the four main bathroom components-typically a toilet and sink. Consequently, I will also count the half bathrooms as half.

# In[ ]:


df = train[['SalePrice', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']]
df['TotBathrooms'] = df.FullBath + (df.HalfBath*0.5) + df.BsmtFullBath + (df.BsmtHalfBath*0.5)


# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.regplot(x='TotBathrooms', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=1, y=680000, s='Correlation with SalePrice: {:6.4f}'.format(df.TotBathrooms.corr(df['SalePrice'])))
print('                                                                 Outliers:',(df.TotBathrooms>=5).sum())
df = df.loc[df.TotBathrooms<5]
sns.regplot(x='TotBathrooms', y='SalePrice', data=df); plt.title('Cut Total Bathrooms Outliers is the best!')
plt.text(x=1, y=630000, s='Correlation withou Outliers: {:6.4f}'.format(df.TotBathrooms.corr(df['SalePrice'])))
plt.show()


# So, with our best predictor, we can cut only two outliers, use it and substitute all others bath features with a existence indicator.

# In[ ]:


train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]
print('Data observations after outliers deletion:', train.shape[0])


# ### Reviwe Porch Features:
# The porch is where many people feel more comfortable to watch life go by, or you prefer the sofa in front of the TV, I think there are people that solved this to the family don't can fighting about this...
# ![image](http://ginormasource.com/humor/wp-content/uploads/2011/10/redneck-porch-swing.jpg)
# ... this idea should make a house worth more, should not it?

# In[ ]:


def PorchPlots():
    fig = plt.figure(figsize=(20,10))
    fig1 = fig.add_subplot(231); sns.regplot(x='OpenPorchSF', y='SalePrice', data=df)
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.OpenPorchSF.corr(df['SalePrice'])))

    fig2 = fig.add_subplot(232); sns.regplot(x='EnclosedPorch', y='SalePrice', data=df);
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.EnclosedPorch.corr(df['SalePrice'])))

    fig3 = fig.add_subplot(233); sns.regplot(x='TSsnPorch', y='SalePrice', data=df)
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TSsnPorch.corr(df['SalePrice'])))

    fig4 = fig.add_subplot(234); sns.regplot(x='ScreenPorch', y='SalePrice', data=df);
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.ScreenPorch.corr(df['SalePrice'])))

    fig5 = fig.add_subplot(235); sns.regplot(x='WoodDeckSF', y='SalePrice', data=df);
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.WoodDeckSF.corr(df['SalePrice'])))

    fig6 = fig.add_subplot(236); sns.regplot(x='TotalPorchSF', y='SalePrice', data=df);
    plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalPorchSF.corr(df['SalePrice'])))

    plt.show()

df = train[['SalePrice', 'OpenPorchSF', 'EnclosedPorch', 'TSsnPorch', 'ScreenPorch', 'WoodDeckSF']]
df['TotalPorchSF'] = df.OpenPorchSF + df.EnclosedPorch + df.TSsnPorch + df.ScreenPorch + df.WoodDeckSF
#df = df[df.TotalPorchSF<=600] # A possible outlier cut!
PorchPlots()


# In[ ]:


df.OpenPorchSF = df.OpenPorchSF > 0
df.EnclosedPorch =  df.EnclosedPorch > 0
df.TSsnPorch = df.TSsnPorch > 0
df.ScreenPorch = df.ScreenPorch > 0
df.WoodDeckSF = df.WoodDeckSF > 0
df.TotalPorchSF = np.sqrt(df.TotalPorchSF) * (np.log1p(np.sqrt(df.TotalPorchSF))**2)

PorchPlots()


# As we have seen, porch features have low correlation with price, and by the graphics we see all most has low bas and high variance, being a high risk to end complex models and fall into ouverfit.
# 
# ### Slope of property and Lot area
# Everyone knows that the size of the lot matters, but has anyone seen any ad talking about the slope?
# ![image](https://www.abedward.com/wp-content/uploads/2015/10/upside-down-house1-1024x730.jpg)

# In[ ]:


# LandSlope: Slope of property
LandSlope = {}
LandSlope['Gtl'] = 3 #'Gentle slope'
LandSlope['Mod'] = 2 #'Moderate Slope'
LandSlope['Sev'] = 1 #'Severe Slope'

df = train[['SalePrice', 'LandSlope', 'LotArea']]
df.LandSlope = df.LandSlope.map(LandSlope)
df['LotAreaMultSlope'] = (df.LotArea * df.LandSlope)

fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(231); sns.regplot(x='LandSlope', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LandSlope.corr(df['SalePrice'])))

fig2 = fig.add_subplot(232); sns.regplot(x='LotArea', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LotArea.corr(df['SalePrice'])))

fig3 = fig.add_subplot(233); sns.regplot(x='LotAreaMultSlope', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LotAreaMultSlope.corr(df['SalePrice'])))
plt.show()


# It is interesting to note that the slope has a low correlation, but as an expected negative. On the other hand, the lot size does not present such a significant correlation, contrary to the interaction between these two characteristics, which is better and also allow us to identify some outliers. Let's take a look at the effect of removing the outliers.

# In[ ]:


print(df[df.LotArea>155000])
df = df[df.LotArea<155000]
fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(234); sns.regplot(x='LandSlope', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LandSlope.corr(df['SalePrice'])))

fig2 = fig.add_subplot(235); sns.regplot(x='LotArea', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LotArea.corr(df['SalePrice'])))

fig3 = fig.add_subplot(236); sns.regplot(x='LotAreaMultSlope', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.LotAreaMultSlope.corr(df['SalePrice'])))
plt.show()


# ### Neighborhood
# Let's watch how much the neighborhood may be influencing the price.
# ![image](https://files.sharenator.com/634178883988989120_NeighborhoodWatch_FailsWins_and_Motis-s800x600-87267-1020.jpg)

# In[ ]:


figa = plt.figure(figsize=(20, 5))
g = train.Neighborhood.value_counts().plot(kind='bar', title='Number of Sales by Neighborhood')

figb = plt.figure(figsize=(20, 5))
plt.tight_layout()
df = train[['SalePrice', 'YrSold', 'Neighborhood']]

df['TotalArea'] = (train.TotalBsmtSF.fillna(0) + train.WoodDeckSF.fillna(0) + train.GrLivArea.fillna(0) + 
                   train.LotArea.fillna(0) + train.MasVnrArea.fillna(0) + train.GarageArea.fillna(0) + 
                   train.OpenPorchSF.fillna(0) + train.TSsnPorch.fillna(0) + train.ScreenPorch.fillna(0) + 
                   train.EnclosedPorch.fillna(0) + train.PoolArea.fillna(0) )
 
df = df.groupby(by=['Neighborhood', 'YrSold'], as_index=False).sum()
Neig = df[['SalePrice', 'TotalArea', 'Neighborhood']].groupby(by='Neighborhood', as_index=False).sum()
Neig['NeigPrice'] = Neig.SalePrice / Neig.TotalArea
Neig.drop(['TotalArea', 'SalePrice'], axis=1, inplace=True)
g = Neig.groupby('Neighborhood').NeigPrice.sum().sort_values(ascending = False).\
    plot(kind='bar', title='Mean Sales Prices per Area (Constructed + Lot) by Neighborhood')
Neig = Neig.groupby(by='Neighborhood', as_index=True).NeigPrice.sum().sort_values(ascending = False)


# As we can see prices are affected by the neighborhood, yes, if more similar more they attract. But we will delve a little and see how the year and month of the sale also has great influence on the price variation and confirm the seasonality.
# ![image](http://blogs.tallahassee.com/community/wp-content/uploads/2015/10/real-estate-seasonality-impact.gif)

# In[ ]:


# Yearly Sales Price per Area (Constructed + Lot) by Neighborhood:
df['HistPriceByNeighborhood'] = df.SalePrice / df.TotalArea
df.drop(['TotalArea', 'SalePrice'], axis=1, inplace=True)

# Fill the gaps
df = df.append(pd.DataFrame([['NPkVill', 2006, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))

df = df.append(pd.DataFrame([['Veenker', 2009, df.HistPriceByNeighborhood[df.Neighborhood=='Veenker'].mean()]], 
                            columns=df.columns))

df = df.append(pd.DataFrame([['Veenker', 2010, df.HistPriceByNeighborhood[df.Neighborhood=='Veenker'].mean()]], 
                            columns=df.columns))

df = df.append(pd.DataFrame([['Blueste', 2006, df.HistPriceByNeighborhood[df.Neighborhood=='Blueste'].min()]], 
                            columns=df.columns))

df = df.append(pd.DataFrame([['Blueste', 2007, df.HistPriceByNeighborhood[df.Neighborhood=='Blueste'].min()]], 
                            columns=df.columns))

df = df.append(pd.DataFrame([['Blueste', 2010, df.HistPriceByNeighborhood[df.Neighborhood=='Blueste'].max()]], 
                            columns=df.columns))

# Reserve data to merge with all data set of train and test data
YearlyPrice = df
YearlyPrice.columns = ['Neighborhood', 'YrSold', 'YearlyPriceByNeighborhood']

print('                              Yearly Sales Prices per Area (Constructed + Lot) by Neighborhood:')
g = sns.catplot(y= 'YearlyPriceByNeighborhood', x = 'YrSold', col='Neighborhood', data=YearlyPrice, 
               kind="point", aspect=.6, col_wrap=7, height=4, col_order=Neig.index)


# In[ ]:


# Monthly Sales Prices per Area (Constructed + Lot) by Neighborhood:
df = train[['SalePrice', 'MoSold', 'Neighborhood']]

df['TotalArea'] = (train.TotalBsmtSF.fillna(0) + train.WoodDeckSF.fillna(0) + train.GrLivArea.fillna(0) + 
                   train.LotArea.fillna(0) + train.MasVnrArea.fillna(0) + train.GarageArea.fillna(0) + 
                   train.OpenPorchSF.fillna(0) + train.TSsnPorch.fillna(0) + train.ScreenPorch.fillna(0) + 
                   train.EnclosedPorch.fillna(0) + train.PoolArea.fillna(0) )

df = df.groupby(by=['Neighborhood', 'MoSold'], as_index=False).sum()
df['HistPriceByNeighborhood'] = df.SalePrice / df.TotalArea
df.drop(['TotalArea', 'SalePrice'], axis=1, inplace=True)

print('                                 Monthly Sales Prices per Area (Constructed + Lot) by Neighborhood:')
g = sns.catplot(y= 'HistPriceByNeighborhood', x = 'MoSold', col='Neighborhood', data=df, 
               kind="point", aspect=.6, col_wrap=7, height=4, col_order=Neig.index )


# In[ ]:


# Outliers from Crawfor Neighborhood
df = train[train.Neighborhood=='Crawfor'][['SalePrice', 'MoSold', 'Neighborhood']]

df['TotalArea'] = (train.TotalBsmtSF.fillna(0) + train.WoodDeckSF.fillna(0) + train.GrLivArea.fillna(0) + 
                   train.LotArea.fillna(0) + train.MasVnrArea.fillna(0) + train.GarageArea.fillna(0) + 
                   train.OpenPorchSF.fillna(0) + train.TSsnPorch.fillna(0) + train.ScreenPorch.fillna(0) + 
                   train.EnclosedPorch.fillna(0) + train.PoolArea.fillna(0) )

df['HistPriceByNeighborhood'] = df.SalePrice / df.TotalArea
df[df.HistPriceByNeighborhood>30]


# In[ ]:


train = train.loc[~(train.SalePrice==392500.0)]
train = train.loc[~((train.SalePrice==275000.0) & (train.Neighborhood=='Crawfor'))]
print('Data observations after outliers deletion:', train.shape[0])


# In[ ]:


# Bin neighborhood for trade cases with low observations on monthly sales prices per Area (Constructed + Lot) by Neighborhood:
Neigb = {}
Neigb['Blueste'] = 'Top'     # 32.212721
Neigb['Blmngtn'] = 'Top'     # 28.364756
Neigb['BrDale']  = 'BrDale'  # 24.903923
Neigb['NPkVill'] = 'NPkVill' # 23.681105
Neigb['MeadowV'] = 'High'    # 22.034923
Neigb['StoneBr'] = 'High'    # 20.475090
Neigb['NridgHt'] =  'High'   # 20.209245
Neigb['Somerst'] = 'Somerst' # 19.551888
Neigb['NoRidge'] = 'NoRidge' # 17.038145
Neigb['CollgCr'] = 'CollgCr' # 15.134767
Neigb['SawyerW'] = 'SawyerW' # 13.992995
Neigb['Crawfor'] = 'Crawfor' # 13.773418
Neigb['Gilbert'] = 'Gilbert' # 13.260281
Neigb['BrkSide'] = 'BrkSide' # 12.785202
Neigb['SWISU']   = 'SVN'     # 12.635171
Neigb['Veenker'] = 'SVN'     # 12.343735
Neigb['NWAmes']  = 'SVN'     # 12.066590
Neigb['OldTown'] = 'OldTown' # 11.571331
Neigb['NAmes']   = 'NAmes'   # 11.091393
Neigb['Mitchel'] = 'Mitchel' # 10.936368
Neigb['Edwards'] = 'Edwards' # 10.614919
Neigb['Sawyer']  = 'Sawyer'  #10.334445
Neigb['IDOTRR']  = 'IDOTRR'  # 9.880838
Neigb['Timber']  = 'Timber'  # 8.723326
Neigb['ClearCr'] = 'ClearCr' # 6.113654

# Preper dataset for Sales Price per Area (Constructed + Lot) by Neighborhood:
df = train[['SalePrice', 'MoSold', 'Neighborhood']]

df['TotalArea'] = (train.TotalBsmtSF.fillna(0) + train.WoodDeckSF.fillna(0) + train.GrLivArea.fillna(0) + 
                   train.LotArea.fillna(0) + train.MasVnrArea.fillna(0) + train.GarageArea.fillna(0) + 
                   train.OpenPorchSF.fillna(0) + train.TSsnPorch.fillna(0) + train.ScreenPorch.fillna(0) + 
                   train.EnclosedPorch.fillna(0) + train.PoolArea.fillna(0) )
df['Price'] = df.SalePrice/df.TotalArea

# Cut Outliers from Crawfor Neighborhood
df = df[(((df.Neighborhood == 'Crawfor') & (df.Price<30.)) | (df.Neighborhood != 'Crawfor'))]
df.drop(['Price'], axis=1, inplace=True)

df.Neighborhood = train.Neighborhood.map(Neigb)

df = df.groupby(by=['Neighborhood', 'MoSold'], as_index=False).sum()
df['HistPriceByNeighborhood'] = df.SalePrice / df.TotalArea

# Get the index for order by value
Neig = df[['SalePrice', 'TotalArea', 'Neighborhood']].groupby(by='Neighborhood', as_index=False).sum()
Neig['NeigPrice'] = Neig.SalePrice / Neig.TotalArea
Neig.drop(['TotalArea', 'SalePrice'], axis=1, inplace=True)
Neig = Neig.groupby(by='Neighborhood', as_index=True).NeigPrice.sum().sort_values(ascending = False)

df.drop(['TotalArea', 'SalePrice'], axis=1, inplace=True)

# Fill the gaps
df = df.append(pd.DataFrame([['Top', 1, df.HistPriceByNeighborhood[df.Neighborhood=='Top'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Top', 8, df.HistPriceByNeighborhood[df.Neighborhood=='Top'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Top', 11, df.HistPriceByNeighborhood[df.Neighborhood=='Top'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Top', 12, df.HistPriceByNeighborhood[df.Neighborhood=='Top'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['BrDale', 1, df.HistPriceByNeighborhood[df.Neighborhood=='BrDale'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['BrDale', 10, df.HistPriceByNeighborhood[df.Neighborhood=='BrDale'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['BrDale', 12, df.HistPriceByNeighborhood[df.Neighborhood=='BrDale'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 1, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 3, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 5, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 8, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 9, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 11, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NPkVill', 12, df.HistPriceByNeighborhood[df.Neighborhood=='NPkVill'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NoRidge', 2, df.HistPriceByNeighborhood[df.Neighborhood=='NoRidge'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NoRidge', 9, df.HistPriceByNeighborhood[df.Neighborhood=='NoRidge'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['NoRidge', 10, df.HistPriceByNeighborhood[df.Neighborhood=='NoRidge'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Crawfor', 1, df.HistPriceByNeighborhood[df.Neighborhood=='Crawfor'].max()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Crawfor', 4, df.HistPriceByNeighborhood[df.Neighborhood=='Crawfor'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Timber', 10, df.HistPriceByNeighborhood[df.Neighborhood=='Timber'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['ClearCr', 1, df.HistPriceByNeighborhood[df.Neighborhood=='ClearCr'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['ClearCr', 2, df.HistPriceByNeighborhood[df.Neighborhood=='ClearCr'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['ClearCr', 8, df.HistPriceByNeighborhood[df.Neighborhood=='ClearCr'].mean()]], 
                            columns=df.columns))
df = df.append(pd.DataFrame([['Edwards', 12, df.HistPriceByNeighborhood[df.Neighborhood=='Edwards'].mean()]], 
                            columns=df.columns))


# Reserve data to merge with all data set of train and test data
MonthlyPrice = df
MonthlyPrice.columns = ['Neigb', 'MoSold', 'MonthlyPriceByNeighborhood']

print('                         Monthly Hist Sales Prices per Area (Construct + Lot) by Neighborhood:')
g = sns.catplot(y= 'MonthlyPriceByNeighborhood', x = 'MoSold', col='Neigb', data=MonthlyPrice, 
               kind="point", aspect=.6, col_wrap=7, height=4, col_order=Neig.index )


# ![image](https://savemax.ca/assets/uploads/pageuploads/what-are-your-neighbours-selling-for.gif)
# As we expected, the seasonality does have some effect, but of course we draw this conclusion based only on the above graphs is precipitated if not erroneous, given that even having restricted the views still exist houses with different characteristics in the same neighborhood.
# 
# However, this is sufficient to understand that the timing of the sale matters, so the model will probably have to take this into account, or this will be part of the residual errors.

# ### Check the Dependent Variable - SalePrice:
# Since most of the machine learning algorithms start from the principle that our data has a normal distribution, we first take a look at the distribution of our **dependent variable**. For this, I create a procedure to plot the **Sales Distribution** and **QQ-plot** to identify substantive departures from normality, likes ***outliers***, ***skewness*** and ***kurtosis***.
# ![image](https://www.radionz.co.nz/assets/news/44599/eight_col_0508-House-prices-1610-gif.gif?1438725455)

# In[ ]:


def QQ_plot(data, measure):
    fig = plt.figure(figsize=(20,7))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()


# In[ ]:


QQ_plot(train.SalePrice, 'Sales Price')


# In[ ]:


#We use the numpy fuction log1p which applies log(1+x) to all elements of the column
train.SalePrice = np.log1p(train.SalePrice)

QQ_plot(train.SalePrice, 'Log1P of Sales Price')


# From the first graph above we can see that Sales Price distribution is ***skewed***, has a **peak**, it **deviates from normal distribution** and is **positively biased**.
# From the **Probability Plot**, we could see that **Sales Price** also does **not align with the diagonal  <span style="color:red">red line</span>** which represent normal distribution. The form of its distribution confirm that is a skewed right. 
# 
# With ***skewness positive of 1.9***, we confirm the **lack of symmetry** and indicate that Sales Price are **skewed right**, as we can see too at the Sales Distribution plot, skewed right means that the right tail is **long relative to the left tail**. The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.
# 
# **Kurtosis** is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers, and **positive** kurtosis indicates a **heavy-tailed distribution** and **negative** kurtosis indicates a **light tailed distribution**. So, with 6.5 of positive kurtosis **Sales Price** are definitely heavy-tailed and has some **outliers** that we need take care.
# 
# Note that in contrast to common belief, training a linear regression model does not require that the explanatory or target variables are normally distributed. The normality assumption is only a requirement for certain statistical tests and hypothesis tests.
# 
# So, I try some linear regressors with both, with and without transformation of SalePrice to check their results.

# ### Test hypothesis of better feature: Construction Area
# Let's call a specialist to help us create a new feature that sum all area features, the construct area, and evaluates if is better than their parcels. 
# ![image](https://im.ziffdavisinternational.com/ign_fr/screenshot/default/simpsonblackboard_h9ga.jpg)

# In[ ]:


df = train[['SalePrice', 'GrLivArea']]
df['ConstructArea'] = (train.TotalBsmtSF.fillna(0) + train.WoodDeckSF.fillna(0) + train.GrLivArea.fillna(0) + 
                       train.MasVnrArea.fillna(0) + train.GarageArea.fillna(0) + train.OpenPorchSF.fillna(0) + 
                       train.TSsnPorch.fillna(0) + train.ScreenPorch.fillna(0) + train.EnclosedPorch.fillna(0) + 
                       train.PoolArea.fillna(0) )
                         
fig8 = plt.figure(figsize=(20,5))
fig9 = fig8.add_subplot(121); sns.regplot((df.ConstructArea), df.SalePrice)
plt.title('Cosntruct Area correlation {:1.2f}'.format(df.ConstructArea.corr(df.SalePrice)))

fig10 = fig8.add_subplot(122); sns.regplot((df.GrLivArea.fillna(0)), df.SalePrice)
tit = 'Livig Area correlation is {:1.2f} and is {:1.2f} correlated to Construct Area'
plt.title(tit.format(df.GrLivArea.fillna(0).corr(df.SalePrice), df.GrLivArea.corr(df.ConstructArea)))
plt.show()


# As we can see, our built metric performs better than its parcels, even more than the living area. Besides better correlation, it presents less bias and variance.
# 
# This may lead us to think of a model option that uses only the constructed area, without including any of the parcels, that would be replaced by an indication variable of existence or not if there is no categorical variable associated with it.
# 
# We can also use them to compose other variables and finally remove them.
# 
# Anyway the **living area** seems ***useless*** now, to prove it let's go see how a single linear regressor perform with this options:

# In[ ]:


def print_results():
    # The coefficients
    print('Coefficients: \n', lr.coef_)
    # The mean squared error
    print("Root mean squared error: %.4f"
          % np.expm1(np.sqrt(mean_squared_error(y_test, y_pred))))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.4f' % r2_score(y_test, y_pred))
    print('--------------------------------------------------------------------------------\n')
    
scale = RobustScaler()
y = df.SalePrice

X = scale.fit_transform(df[['ConstructArea', 'GrLivArea']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

lr = LinearRegression()

print('1. Linear regressor with only living Area:')
lr.fit(X_train[: , 1].reshape(-1, 1), y_train)

# Make predictions using the testing set
y_pred = lr.predict(X_test[: , 1].reshape(-1, 1))
print_results()

print('2. Linear regressor with bouth features:')
lr.fit(X_train, y_train)

# Makepredictions using the testing set
y_pred = lr.predict(X_test)
print_results()

print('3. Linear regressor with only Construct Area:')
lr = LinearRegression()
lr.fit(X_train[: , 0].reshape(-1, 1), y_train)

# Makepredictions using the testing set
y_pred = lr.predict(X_test[: , 0].reshape(-1, 1))
print_results()

print('4. Polinomial regressor of orden 3 with only Construction Area:')
# create polynomial features
cubic = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_cubic = cubic.fit_transform(X_train[: , 0].reshape(-1, 1))

# cubic fit
lr = lr.fit(X_cubic, y_train)
y_pred = lr.predict(cubic.fit_transform(X_test[: , 0].reshape(-1, 1)))
print_results()

print('5. Polinomial regressor of orden 3 with both features:')
# create polynomial features
cubic = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_cubic = cubic.fit_transform(X_train)

# cubic fit
lr = lr.fit(X_cubic, y_train)
y_pred = lr.predict(cubic.fit_transform(X_test))
print_results()


# According to our specialist, the above results show to us that:
# 
# - we can safely eliminate the living area, and as there are no records with it zero or null we will not create an existence indicator for it.
# - We may be able to discard other area metrics, especially those that have many zeros for nulls, which contribute little to accuracy and even to reduce multicollinearity.
# - the **polynomial transformation of 3<sup>th</sup> degree** presents ***improvements*** of **2.6%** and **1.8%** of  RMSE and R<sup>2</sup> respectively. So, now we have a simple regressor from our specialist to bit!
# 
# Before create new features and other test, is better to make the data cleaning and fill nulls.

# ## 3. Check Data Quality:
# ![image](https://i.imgur.com/ZswCHWs.png)
# ### Nulls Check:
# In this section, I am going to fix the 34 predictors that contains missing values. I will go through them working my way down from most NAs until I have fixed them all. If I stumble upon a variable that actually forms a group with other variables, I will also deal with them as a group. For instance, there are multiple variables that relate to Pool, Garage, and Basement.

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)

print("All data observations have {0} rows and {1} columns".format(all_data.shape[0], all_data.shape[1]))
details = rstr(all_data)
print("All data have {1:2.2%} of null at {0} features".format(details[details.nulls>0].shape[0], 
                                                   details.nulls[details.nulls>0].sum()/all_data.size))
print('\nBelow the table with all columns with nulls oredered by missin ration:')
display(details.loc[(details.nulls>0), 'types':'uniques'].sort_values(by= 'missing ration', ascending=False))


# In[ ]:


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """
        Impute missing values:
        - Columns of dtype object are imputed with the most frequent value in column.
        - Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# ### Some Observations Respect Data Quality:
# <p>The total training observations are 1,460 and have 79 features ( 3 float64, 33 int64, 43 object ) with 19 columns with nulls.
#     
# All data observations, including test data set, have 2919 rows and 79 features, where 6.06% are nulls, but found nulls at 34 different features. So the test dataset has null in features that training dataset doesn't have!
# 
# ***Based on feature description provide, A feature that has NA means it is absent***
# <p>
# First, before we assume this as a total reality, we need check some quality issues, like the record has Garage, but doesn't have Garage Quality and vice versa.
# 
# 
#  - 14 has few null, so are good candidates for imputer strategies:
#    - GarageFinish  1379  object.  Interior finish of the garage.
#    - GarageQual    1379  object:  Garage quality. 
#    - GarageCond    1379  object:  Garage condition. 
#    - GarageType    1379  object:  Garage location
#    - GarageYrBlt   1379  float64: Year garage was built
#    - Electrical    1459  object.  Only one, can apply the most common.
#    - MasVnrType    1452  object:  is the masonry veneer type, hasn't CBlock!
#    - MasVnrArea    1452  float64: Masonry veneer area in square feet.   
#    - BsmtExposure  1422  object:  Refers to walkout or garden level walls.
#    - BsmtFinType2  1422  object:  Rating of basement finished area (if multiple types).
#    - BsmtQual      1423  object:  Evaluates the height of the basement Doesn't have PO.
#    - BsmtCond      1423  object:  Evaluates the general condition of the basement. 
#    - BsmtFinType1  1423  object:  Rating of basement finished area (if multiple types).    
#    - LotFrontage   1201  float64: is the linear feet of street connected to property.
# <p>
#  - 5 has miss ration grater than 47%, maybe candidates to exclude, especially if their have below correlation with price.
#     - Fence         281  object: Fence quality.
#     - FireplaceQu   770  object: Fireplace quality.
#     - MiscFeature    54  object: Miscellaneous feature not covered in other categories. 
#     - Alley          91  object: is the type of alley access to property.
#     - PoolQC          7  object: Pool quality. Attention for the related other feature PoolArea: Pool area in square feet
# 
# Some numeric data are ordinal or categorical already translate to codes. We need correct identify the ordinal from the description and can maintain as is, but need to change categorical.<p>
#     
# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modeling. We can then safely remove it.

# In[ ]:


all_data.drop('Utilities', axis=1, inplace=True)


# ### Identify the Most Common Electrical:
# ![image](https://i.ytimg.com/vi/uXUs7hlTmp0/hqdefault.jpg)

# In[ ]:


display(all_data.Electrical.value_counts())

all_data.Electrical = all_data.Electrical.fillna('SBrkr')


# ### Fill Missing Values of Garage Features:
# Identify if has some special cases where we find some garage feature inputted, where's others garages features are null.
# ![image](https://i.pinimg.com/originals/54/bc/0e/54bc0e3a4223c98049c312167f9b727a.jpg)

# In[ ]:


feat = ['GarageYrBlt', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageArea', 'GarageCars']
print(all_data[feat].isnull().sum())
print("GarageArea equal a 0: ", (all_data.GarageArea==0).sum())
print("GarageCars equal a 0: ", (all_data.GarageCars==0).sum())
c = all_data[~all_data.GarageType.isnull()][feat]
c[c.GarageYrBlt.isnull()]


# #### Group by GarageType
# Fill missing value with median or mode where GarageType equal a Detchd and 0 or NA for the others.

# In[ ]:


all_data.GarageType = all_data.GarageType.fillna('NA')

# Group by GarageType and fill missing value with median where GarageType=='Detchd' and 0 for the others
cmedian = all_data[all_data.GarageType=='Detchd'].GarageArea.median()
print("GarageArea median of Type Detchd:", cmedian)
all_data.loc[all_data.GarageType=='Detchd', 'GarageArea'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                         'GarageArea'].fillna(cmedian)
all_data.GarageArea = all_data.GarageArea.fillna(0)

cmedian = all_data[all_data.GarageType=='Detchd'].GarageCars.median()
print("GarageCars median of Type Detchd:", cmedian)
all_data.loc[all_data.GarageType=='Detchd', 'GarageCars'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                         'GarageCars'].fillna(cmedian)
all_data.GarageCars = all_data.GarageCars.fillna(0)

cmedian = all_data[all_data.GarageType=='Detchd'].GarageYrBlt.median()
print("GarageYrBlt median of Type Detchd:", cmedian)
all_data.loc[all_data.GarageType=='Detchd', 'GarageYrBlt'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                          'GarageYrBlt'].fillna(cmedian)
all_data.GarageYrBlt = all_data.GarageYrBlt.fillna(0)

# Group by GarageType and fill missing value with mode where GarageType=='Detchd' and 'NA' for the others
cmode = all_data[all_data.GarageType=='Detchd'].GarageFinish.mode()[0]
print("GarageFinish mode of Type Detchd:", cmode)
all_data.loc[all_data.GarageType=='Detchd', 'GarageFinish'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                           'GarageFinish'].fillna(cmode)
all_data.GarageFinish = all_data.GarageFinish.fillna('NA')

cmode = all_data[all_data.GarageType=='Detchd'].GarageQual.mode()[0]
print("GarageQual mode of Type Detchd: %s" %cmode)
all_data.loc[all_data.GarageType=='Detchd', 'GarageQual'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                         'GarageQual'].fillna(cmode)
all_data.GarageQual = all_data.GarageQual.fillna('NA')

cmode = all_data[all_data.GarageType=='Detchd'].GarageCond.mode()[0]
print("GarageCond mode of Type Detchd:", cmode)
all_data.loc[all_data.GarageType=='Detchd', 'GarageCond'] = all_data.loc[all_data.GarageType=='Detchd', 
                                                                         'GarageCond'].fillna(cmode)
all_data.GarageCond = all_data.GarageCond.fillna('NA')


# #### Check if all nulls of Garage features are inputed
# ![image](http://www.danielthiebaut.com/wp-content/uploads/2013/03/Cool-Garage-Idea.jpg)

# In[ ]:


print(all_data[feat].isnull().sum())


# ### Masonry veneer
# Check Nulls and Data Quality Problems:
# ![image](https://torontorealtyblog.com/wp-content/uploads/2014/11/Pigs.jpg)

# In[ ]:


feat = ['MasVnrArea', 'MasVnrType']
c = all_data[~all_data.MasVnrArea.isnull()][feat]
print('Masonry veneer Nulls:')
print(all_data[feat].isnull().sum(), '\n')
print("Has MasVnrType but not has MasVnrArea:",all_data[~all_data.MasVnrType.isnull()].MasVnrArea.isnull().sum())
print("Has MasVnrArea but not has MasVnrType:",c[c.MasVnrType.isnull()].MasVnrArea.count())
print(c[c.MasVnrType.isnull()], '\n')

print("Has MasVnrType but MasVnrArea is equal a Zero:",c[c.MasVnrArea==0].MasVnrType.count())
print("MasVnrArea equal a 0: ", (all_data.MasVnrArea==0).sum(), '\n')
print("Has Type and Area == 0:")
print(c[c.MasVnrArea==0].MasVnrType.value_counts(), '\n')

print("Type None with Area > 0 ?")
print(all_data.loc[(all_data.MasVnrType=='None') & (all_data.MasVnrArea>0), ['MasVnrType','MasVnrArea']])

print('\n What is the most comumn MasVnrType after None?')
print(all_data.MasVnrType.value_counts())


# #### Correct masonry veneer types
# Change to BrkFace the masonry veneer types Nulls and Nones wheres records has masonry Veneer Area

# In[ ]:


# All None Types with Are greater than 0 update to BrkFace type
all_data.loc[(all_data.MasVnrType=='None') & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

# All Types null with Are greater than 0 update to BrkFace type
all_data.loc[(all_data.MasVnrType.isnull()) & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

# All Types different from None with Are equal to 0 update to median Area of no None types with Areas
all_data.loc[(all_data.MasVnrType!='None') & 
             (all_data.MasVnrArea==0), ['MasVnrArea']] = all_data.loc[(all_data.MasVnrType!='None') & 
                                                                      (all_data.MasVnrArea>0), ['MasVnrArea']].median()[0]
# Filling 0 and None for records wheres both are nulls
all_data.MasVnrArea = all_data.MasVnrArea.fillna(0)
all_data.MasVnrType = all_data.MasVnrType.fillna('None')


# #### Check if all nulls of masonry veneer types are updated

# In[ ]:


c = all_data[~all_data.MasVnrType.isnull()][feat]
print('Masonry veneer Nulls:')
print(all_data[feat].isnull().sum(), '\n')


# ### Check and Input Basement Features Nulls:
# ![image](https://jokideo.com/wp-content/uploads/2012/07/418275_421511294568548_302301786_n.jpg)

# In[ ]:


feat = ['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath', 
        'BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
print(all_data[feat].isnull().sum())
print("BsmtFinSF1 equal a 0: ", (all_data.BsmtFinSF1==0).sum())
print("BsmtFinSF2 equal a 0: ", (all_data.BsmtFinSF2==0).sum())
print("BsmtUnfSF equal a 0: ", (all_data.BsmtUnfSF==0).sum())
print("TotalBsmtSF equal a 0: ", (all_data.TotalBsmtSF==0).sum())
print("BsmtFullBath equal a 0: ", (all_data.BsmtFullBath==0).sum())
print("BsmtHalfBath equal a 0: ", (all_data.BsmtHalfBath==0).sum())


# In[ ]:


# No Basement Av is the most comumn BsmtExposure. 
display(all_data.BsmtExposure.value_counts())

# Update nulls Exposure to Av wheres TotalBsmntSF is grenter tham zero 
all_data.loc[(~all_data.TotalBsmtSF.isnull()) & 
             (all_data.BsmtExposure.isnull()) & 
             (all_data.TotalBsmtSF>0), 'BsmtExposure'] = 'Av'


# In[ ]:


# TA is the most comumn BsmtQual. 
display(all_data.BsmtQual.value_counts())

# We use TA for all cases wheres has same evidenci that the house has Basement
all_data.loc[(~all_data.TotalBsmtSF.isnull()) & 
             (all_data.BsmtQual.isnull()) & 
             (all_data.TotalBsmtSF>0), 'BsmtQual'] = 'TA'


# In[ ]:


# TA is the most comumn BsmtCond. 
display(all_data.BsmtCond.value_counts())

# We use TA for all cases wheres has same evidenci that the house has Basement
all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtCond.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtCond'] = 'TA'


# In[ ]:


# Unf is the most comumn BsmtFinType2. 
display(all_data.BsmtFinType2.value_counts())

# We use Unf for all cases wheres BsmtFinType2 is null but BsmtFinSF2 is grater than Zro
all_data.loc[(all_data.BsmtFinSF2>0) & (all_data.BsmtFinType2.isnull()) , 'BsmtFinType2'] = 'Unf'


# In[ ]:


# See below that we have one case where BsmtFinType2 is BLQ and the Area is Zero, but its area was inputed at Unfinesh
display(all_data[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull())][feat])

# Correct BsmtFinSF2 and BsmtUnfSF:
all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtFinSF2'] = 354.0
all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtUnfSF'] = 0.0


# In[ ]:


# All these cases are clear don´t have basement. 
print("Rest cases where Cond is Null", (all_data[all_data.BsmtCond.isnull()]).shape[0], '\n')
print('Others categories basement features are Null when Cond is Null:\n',
      (all_data[all_data.BsmtCond.isnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure',
                                             'BsmtFinType1' , 'BsmtFinType2']]).isnull().sum())
print('\nOthers numerics basement features are Null or Zero when Cond is Null:\n',
      all_data[all_data.BsmtCond.isnull()][['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF',
                                            'BsmtFullBath', 'BsmtHalfBath']].sum())
print("\nThe particular cases where's numeric basement features see below are Null were included in the previous groups:") 
display(all_data[all_data.BsmtFullBath.isnull()][feat])

# So, we update these Zero or NA according to their dictionary:
nulls_cols = {'BsmtExposure': 'NA', 'BsmtFinType2': 'NA', 'BsmtQual': 'NA', 'BsmtCond': 'NA', 'BsmtFinType1': 'NA',
              'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0 ,'TotalBsmtSF': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0}

all_data = all_data.fillna(value=nulls_cols)

print('\nFinal Check if all nulls basement features are treated:', all_data[feat].isnull().sum().sum())


# ### Lot Frontage - Check and Fill Nulls
# ![image](https://s.abcnews.com/images/US/HT-frozen-house-lake-ontario-jt-170312_16x9_992.jpg)

# In[ ]:


# Group by Neigborhood and fill missing value with Lot frontage median of the respect Neigborhood
NegMean = all_data.groupby('Neighborhood').LotFrontage.mean()

all_data.loc.LotFrontage = all_data[['Neighborhood', 'LotFrontage']].\
                           apply(lambda x: NegMean[x.Neighborhood] if np.isnan(x.LotFrontage) else x.LotFrontage, axis=1)


# ### Pool Quality - Fill Nulls
# ![image](https://msr7.net/images/funny-pool-10.jpg)
# Probably models won't use Pools Features, since they has few correlation to price (0.069798) and more than 99% of missing.
# But for the moment, we still filling null of PoolQC that has Area with based on the Overall Quality of the houses divided by 2.

# In[ ]:


PoolQC = {0: 'NA', 1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}

all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['PoolQC']] =\
        ((all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['OverallQual']]/2).round()).\
        apply(lambda x: x.map(PoolQC))

all_data.PoolQC = all_data.PoolQC.fillna('NA')


# ### Functional - Miss Values Treatment
# ![image](http://patscolor.com/wp-content/uploads/2014/11/doggy-trailer-luxury-dog-house.png)
# Since Functional description include the statement "Assume typical unless deductions are warranted", we assume "Typ" for nulls cases.

# In[ ]:


all_data.Functional = all_data.Functional.fillna('Typ')


# ### Fireplace Quality - Miss Values Treatment
# Since all Fireplace Quality nulls has Fireplaces equal a zero, its sure that Fireplace Quality could be update to NA.
# ![image](https://img.memey.com/1/1/cute-happy-cat-fireplace.jpg)

# In[ ]:


all_data.loc[(all_data.Fireplaces==0) & (all_data.FireplaceQu.isnull()), ['FireplaceQu']] = 'NA'


# ### Kitchen Quality - Miss Values Treatment
# ![image](http://blog.qualitybath.com/wp-content/uploads/2010/11/12.jpg)
# Since all Kitchen Quality nulls has Kitchen Above Ground grater than zero, we assume mode for Kitchen Quality.

# In[ ]:


all_data.loc[(all_data.KitchenAbvGr>0) & (all_data.KitchenQual.isnull()), 
             ['KitchenQual']] = all_data.KitchenQual.mode()[0]


# ### Alley, Fence and Miscellaneous Feature - Miss Values Treatment
# - Miscellaneous feature not covered in other categories. 
# - Alley has a few records, and is not really common to have alley in properties.
# - It's not uncommon to see properties without fence at USA.
# ![image](http://i.imgur.com/A3qTE.png)
# So, don't lose time and update nulls for NA.

# In[ ]:


all_data.Alley = all_data.Alley.fillna('NA')
all_data.Fence = all_data.Fence.fillna('NA')
all_data.MiscFeature = all_data.MiscFeature.fillna('NA')


# ### Back to the Past! Garage Year  Build from 2207
# ![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROZIDDdbhzPVNBsbArn0LZCniU1_LJf0OLXQ3CEOSw6B3ZY25PPw)

# In[ ]:


display(all_data.loc[all_data.GarageYrBlt==2207, ['GarageYrBlt', 'YearBuilt']])
all_data.loc[all_data.GarageYrBlt==2207.0, 'GarageYrBlt'] = 2007.0


# ### Final Check and Filling Nulls
# ![image](https://i.pinimg.com/236x/f8/8a/f2/f88af268195e24088347d98b2fd78dcf--slide-stairs-basement-stairs.jpg)

# In[ ]:


all_data = DataFrameImputer().fit_transform(all_data)

# Final check if we have some NA
print("Data nulls:", all_data.isnull().sum().sum())


# ## Mapping Ordinal Features
# Any attribute or feature that is categorical in nature represents discrete values that belong to a specific finite set of categories or classes. The ordinal are special category type that can also be ordered based on rules on the context. So, ordinal categorical variables can be ordered and sorted on the basis of their values and hence these values have specific significance such that their order makes sense. 
# ![image](https://i.pinimg.com/originals/45/67/85/45678591a8d7d31cd6c83e3f7edbd8ad.jpg)
# Like nominal features, even ordinal features might be present in text form and you need to map and transform them into their numeric representation. 
# 
# Below, you can see that it is really easy to build your own transformation mapping scheme with the help of Python dictionaries and use the map function from pandas to transform the ordinal feature, and preserve its significance.

# In[ ]:


def map_ordinals(data):
    
    # LandSlope: Slope of property
    LandSlope = {}
    LandSlope['Gtl'] = 3 #'Gentle slope'
    LandSlope['Mod'] = 2 #'Moderate Slope'
    LandSlope['Sev'] = 1 #'Severe Slope'

    data.LandSlope = data.LandSlope.map(LandSlope)
        
    # ExterQual: Evaluates the quality of the material on the exterior 
    ExterQual = {}
    ExterQual['Ex'] = 5 #'Excellent'
    ExterQual['Gd'] = 4 #'Good'
    ExterQual['TA'] = 3 #'Average/Typical'
    ExterQual['Fa'] = 2 #'Fair'
    ExterQual['Po'] = 1 #'Poor'
    ExterQual['NA'] = 0 #'NA'

    data.ExterQual = data.ExterQual.map(ExterQual)

    # ExterCond: Evaluates the present condition of the material on the exterior
    data.ExterCond = data.ExterCond.map(ExterQual)

    #HeatingQC: Heating quality and condition
    data.HeatingQC = data.HeatingQC.map(ExterQual)

    # KitchenQual: Kitchen quality
    data.KitchenQual = data.KitchenQual.map(ExterQual)

    # FireplaceQu: Fireplace quality
    data.FireplaceQu = data.FireplaceQu.map(ExterQual)

    # GarageCond: Garage Conditionals
    data.GarageCond = data.GarageCond.map(ExterQual)

    PavedDrive = {}
    PavedDrive['Y'] = 3 #'Paved'
    PavedDrive['P'] = 2 #'Partial Pavement'
    PavedDrive['N'] = 1 #'Dirt/Gravel'

    data.PavedDrive = data.PavedDrive.map(PavedDrive)

    # LotShape: General shape of property
    LotShape = {}
    LotShape['Reg'] = 4 #'Regular'
    LotShape['IR1'] = 3 #'Slightly irregular'
    LotShape['IR2'] = 2 #'Moderately Irregular'
    LotShape['IR3'] = 1 #'Irregular'

    data.LotShape = data.LotShape.map(LotShape)

    # BsmtQual: Evaluates the height of the basement
    BsmtQual = {}
    BsmtQual['Ex'] = 5 #'Excellent (100+ inches)'
    BsmtQual['Gd'] = 4 #'Good (90-99 inches)'
    BsmtQual['TA'] = 3 #'Typical (80-89 inches)'
    BsmtQual['Fa'] = 2 #'Fair (70-79 inches)'
    BsmtQual['Po'] = 1 #'Poor (<70 inches'
    BsmtQual['NA'] = 0 #'No Basement'

    data.BsmtQual = data.BsmtQual.map(BsmtQual)

    # BsmtCond: Evaluates the general condition of the basement
    data.BsmtCond = data.BsmtCond.map(BsmtQual)

    # GarageQual: Garage quality
    data.GarageQual = data.GarageQual.map(BsmtQual)

    # PoolQC: Pool quality
    data.PoolQC = data.PoolQC.map(BsmtQual)
    
    # BsmtExposure: Refers to walkout or garden level walls
    BsmtExposure = {}
    BsmtExposure['Gd'] = 4 #'Good Exposure'
    BsmtExposure['Av'] = 3 #'Average Exposure (split levels or foyers typically score average or above)'
    BsmtExposure['Mn'] = 2 #'Mimimum Exposure'
    BsmtExposure['No'] = 1 #'No Exposure'
    BsmtExposure['NA'] = 0 #'No Basement'

    data.BsmtExposure = data.BsmtExposure.map(BsmtExposure)

    # BsmtFinType1: Rating of basement finished area
    BsmtFinType1 = {}
    BsmtFinType1['GLQ'] = 6 #'Good Living Quarters'
    BsmtFinType1['ALQ'] = 5 # 'Average Living Quarters'
    BsmtFinType1['BLQ'] = 4 # 'Below Average Living Quarters'
    BsmtFinType1['Rec'] = 3 # 'Average Rec Room'
    BsmtFinType1['LwQ'] = 2 # 'Low Quality'
    BsmtFinType1['Unf'] = 1 # 'Unfinshed'
    BsmtFinType1['NA'] = 0 #'No Basement'

    data.BsmtFinType1 = data.BsmtFinType1.map(BsmtFinType1)

    # BsmtFinType2: Rating of basement finished area (if multiple types)
    data.BsmtFinType2 = data.BsmtFinType2.map(BsmtFinType1)

    #CentralAir: Central air conditioning
    # Since with this transformatio as the same as binarize this feature
    CentralAir = {}
    CentralAir['N'] = 0
    CentralAir['Y'] = 1

    data.CentralAir = data.CentralAir.map(CentralAir)

    # GarageFinish: Interior finish of the garage
    GarageFinish = {}
    GarageFinish['Fin'] = 3 #'Finished'
    GarageFinish['RFn'] = 2 #'Rough Finished'
    GarageFinish['Unf'] = 1 #'Unfinished'
    GarageFinish['NA'] = 0 #'No Garage'
    
    data.GarageFinish = data.GarageFinish.map(GarageFinish)
    
    # Functional: Home functionality
    Functional = {}
    Functional['Typ'] = 7   # Typical Functionality
    Functional['Min1'] = 6  # Minor Deductions 1
    Functional['Min2'] = 5  # Minor Deductions 2
    Functional['Mod'] = 4   # Moderate Deductions
    Functional['Maj1'] = 3  # Major Deductions 1
    Functional['Maj2'] = 2  # Major Deductions 2
    Functional['Sev'] = 1   # Severely Damaged
    Functional['Sal'] = 0   # Salvage only

    data.Functional = data.Functional.map(Functional)
    
    #Street: Type of road access to property
    # Since with this transformatio as the same as binarize this feature
    Street = {}
    Street['Grvl'] = 0 # Gravel 
    Street['Pave'] = 1 # Paved

    data.Street = data.Street.map(Street)


    # Fence: Fence quality
    Fence = {}
    Fence['GdPrv'] = 5 #'Good Privacy'
    Fence['MnPrv'] = 4 #'Minimum Privacy'
    Fence['GdWo'] = 3 #'Good Wood'
    Fence['MnWw'] = 2 #'Minimum Wood/Wire'
    Fence['NA'] = 1 #'No Fence'

    data.Fence = data.Fence.map(Fence)
    #But No Fence has the higest median Sales Price. So I try to use it as categorical
            
    return data

all_data = map_ordinals(all_data)


# ## Feature Engineering: Create New Features:
# ![image](http://assets.amuniversal.com/a87892a06cb801301d46001dd8b71c47)

# ### Include pool in the Miscellaneous features

# In[ ]:


display(all_data.loc[(all_data.PoolArea>0) & (all_data.MiscVal>0), ['MiscFeature', 'MiscVal', 'PoolArea', 'PoolQC']])


# Check if we had others "TenC" in the dataset:

# In[ ]:


display(all_data.loc[(all_data.MiscFeature=='TenC'), ['MiscFeature', 'MiscVal', 'PoolArea', 'PoolQC']])


# Since we don't have other "TenC" and others Pools don't coincide with any miscellaneous feature, we include the pools into the Misc Features and drop Pools columns after used it in the creation of others features.

# In[ ]:


all_data.loc[(all_data.PoolArea>0), ['MiscFeature']] = 'Pool'
all_data.loc[(all_data.PoolArea>0), ['MiscVal']] = all_data.loc[(all_data.PoolArea>0), 
                                                               ['MiscVal', 'PoolArea']].\
                                                                apply(lambda x: (x.MiscVal + x.PoolArea), axis=1)


# ### Points Review
# Since there are many punctuation characteristics in our base, and I believe that each person has a specific preference depending on the stage and moment of life, I believe that these variables have importance, but they present a lot of variation and bias.
# 
# So, it is important to consider a general score calculated from all the points that are agreed upon. But then you'll say, "What do you mean, you did not know that the base has general scores for condition and quality, you're dumb!".
# ![image](https://i.pinimg.com/236x/eb/5a/fc/eb5afcbd19f72317f266cc52a93c0a2a--exam-humor-cpa-review.jpg)
# "Patience young grasshopper", I know of them and I believe that there are some criteria for them, but they sum up a very narrow and discreet spectrum, and frankly who accepts criteria that express a point of view and are not simply natural laws without questioning them. So building the metric will give us a counterpoint based on the different grades, and it is not surprising that it is better for our model than all the grades, even the overall.
# 
# And if you still doubt, I put charts and correlation numbers to help you understand the benefits, and of course, then you can also question my own criteria and establish yours to calculate your overall score.

# In[ ]:


all_data['TotalExtraPoints'] = all_data.HeatingQC + all_data.PoolQC + all_data.FireplaceQu  + all_data.KitchenQual
all_data['TotalPoints'] =  (all_data.ExterQual + all_data.FireplaceQu + all_data.GarageQual + all_data.KitchenQual +
                            all_data.BsmtQual + all_data.BsmtExposure + all_data.BsmtFinType1 + all_data.PoolQC + 
                            all_data.ExterCond + all_data.BsmtCond + all_data.GarageCond + all_data.OverallCond +
                            all_data.BsmtFinType2 + all_data.HeatingQC ) + all_data.OverallQual**2
                         
df = all_data.loc[(all_data.SalePrice>0), ['TotalPoints', 'TotalExtraPoints', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 
               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'PoolQC', 'KitchenQual', 
               'FireplaceQu', 'GarageQual', 'GarageCond', 'SalePrice']]


# In[ ]:


fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(231); sns.regplot(x='BsmtExposure', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.BsmtExposure.corr(df['SalePrice'])))

fig2 = fig.add_subplot(232); sns.regplot(x='BsmtFinType1', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.BsmtFinType1.corr(df['SalePrice'])))

fig3 = fig.add_subplot(233); sns.regplot(x='BsmtFinType2', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.BsmtFinType2.corr(df['SalePrice'])))

fig4 = fig.add_subplot(234); sns.regplot(x='GarageQual', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageQual.corr(df['SalePrice'])))

fig5 = fig.add_subplot(235); sns.regplot(x='GarageCond', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageCond.corr(df['SalePrice'])))

fig6 = fig.add_subplot(236); sns.regplot(x='TotalPoints', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalPoints.corr(df['SalePrice'])))
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(231); sns.regplot(x='OverallQual', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.OverallQual.corr(df['SalePrice'])))

fig2 = fig.add_subplot(232); sns.regplot(x='OverallCond', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.OverallCond.corr(df['SalePrice'])))

fig3 = fig.add_subplot(233); sns.regplot(x='ExterQual', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.ExterQual.corr(df['SalePrice'])))

fig4 = fig.add_subplot(234); sns.regplot(x='ExterCond', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.ExterCond.corr(df['SalePrice'])))

fig5 = fig.add_subplot(235); sns.regplot(x='BsmtQual', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.BsmtQual.corr(df['SalePrice'])))

fig6 = fig.add_subplot(236); sns.regplot(x='BsmtCond', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.BsmtCond.corr(df['SalePrice'])))
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,10))
fig1 = fig.add_subplot(231); sns.regplot(x='HeatingQC', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.HeatingQC.corr(df['SalePrice'])))

fig2 = fig.add_subplot(232); sns.regplot(x='PoolQC', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.PoolQC.corr(df['SalePrice'])))

fig3 = fig.add_subplot(233); sns.regplot(x='KitchenQual', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.KitchenQual.corr(df['SalePrice'])))

fig4 = fig.add_subplot(234); sns.regplot(x='FireplaceQu', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.FireplaceQu.corr(df['SalePrice'])))

fig5 = fig.add_subplot(235); sns.regplot(x='TotalExtraPoints', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalExtraPoints.corr(df['SalePrice'])))

plt.show()


# In[ ]:


all_data['GarageArea_x_Car'] = all_data.GarageArea * all_data.GarageCars

all_data['TotalBsmtSF_x_Bsm'] = all_data.TotalBsmtSF * all_data['1stFlrSF']

# We don´t have a feature with all construct area, maybe it is an interesting feature to create.
all_data['ConstructArea'] = (all_data.TotalBsmtSF + all_data.WoodDeckSF + all_data.GrLivArea +
                             all_data.OpenPorchSF + all_data.TSsnPorch + all_data.ScreenPorch + all_data.EnclosedPorch +
                             all_data.MasVnrArea + all_data.GarageArea + all_data.PoolArea )

#all_data['TotalArea'] = all_data.ConstructArea + all_data.LotArea

all_data['Garage_Newest'] = all_data.YearBuilt > all_data.GarageYrBlt
all_data.Garage_Newest =  all_data.Garage_Newest.apply(lambda x: 1 if x else 0)

all_data['TotalPorchSF'] = all_data.OpenPorchSF + all_data.EnclosedPorch + all_data.TSsnPorch + all_data.ScreenPorch + all_data.WoodDeckSF
all_data.EnclosedPorch = all_data.EnclosedPorch.apply(lambda x: 1 if x else 0)

all_data['LotAreaMultSlope'] = all_data.LotArea * all_data.LandSlope


all_data['BsmtSFPoints'] = (all_data.BsmtQual**2 + all_data.BsmtCond + all_data.BsmtExposure + 
                            all_data.BsmtFinType1 + all_data.BsmtFinType2)


all_data['BsmtSFMultPoints'] = all_data.TotalBsmtSF * (all_data.BsmtQual**2 + all_data.BsmtCond + all_data.BsmtExposure + 
                                                       all_data.BsmtFinType1 + all_data.BsmtFinType2)

all_data['TotBathrooms'] = all_data.FullBath + (all_data.HalfBath*0.5) + all_data.BsmtFullBath + (all_data.BsmtHalfBath*0.5)
all_data.FullBath = all_data.FullBath.apply(lambda x: 1 if x else 0)
all_data.HalfBath = all_data.HalfBath.apply(lambda x: 1 if x else 0)
all_data.BsmtFullBath = all_data.BsmtFullBath.apply(lambda x: 1 if x else 0)
all_data.BsmtHalfBath = all_data.BsmtHalfBath.apply(lambda x: 1 if x else 0)


# ### One Hot Encode Categorical Features
# You might now be wondering, if we have a data set wheres all categorical data already transformed and mapped them into numeric representations, why would we need more levels of encoding again? 
# ![image](https://i.pinimg.com/236x/ee/bd/70/eebd70b6a2a1bc9d403bb92af62ef8bd--symbol-logo-open-data.jpg)
# To traditional people and liberals, I am not discussing a question of gender or option, but rather that If we directly fed these transformed numeric representations of categorical features into any algorithm, the model will essentially try to interpret these as raw numeric features and hence the notion of magnitude will be wrongly introduced in the system.
# 
# There are several schemes and strategies where dummy features are created for each unique value or label out of all the distinct categories in any feature, like one hot encoding, dummy coding, effect coding, and feature hashing schemes.
# 
# In one hot encoding strategy we considering have numeric representation of any categorical feature with m labels, the one hot encoding scheme, encodes or transforms the feature into m binary features, which can only contain a value of 1 or 0. Each observation in the categorical feature is thus converted into a vector of size m with only one of the values as 1 (indicating it as active). 
# 
# From sklearn in preprocessing you can use the  LabelEncoder to create a cod map for the category feature than use the OneHotEncoder to apply the one hot encode strategy above then. This is interesting for most real cases where your model will be applied to data that will be fed and updated continuously in a pipeline.
# 
# As our case is restricted to the data provided and it fits on a pandas data frame, we will make use of the function get_dummies from pandas, but attention this function does not transform the data to a vector as in the case of the previous or as in R.
# 
# The dummy coding scheme is similar to the one hot encoding scheme, except in the case of dummy coding scheme, when applied on a categorical feature with m distinct labels, we get m-1 binary features. Thus each value of the categorical variable gets converted into a vector of size m-1. The extra feature is completely disregarded and thus if the category values range from {0, 1, ..., m-1} the 0th or the m-1th feature is usually represented by a vector of all zeros (0).

# In[ ]:


def one_hot_encode(df):
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(len(categorical_cols), "categorical columns")
    print(categorical_cols)
    # Remove special charactres and withe spaces. 
    for col in categorical_cols:
        df[col] = df[col].str.replace('\W', '').str.replace(' ', '_') #.str.lower()

    dummies = pd.get_dummies(df[categorical_cols], columns = categorical_cols).columns
    df = pd.get_dummies(df, columns = categorical_cols)

    print("Total Columns:",len(df.columns))
    print(df.info())
    
    return df, dummies

# Correct Categorical from int to str types
all_data.MSSubClass = all_data.MSSubClass.astype('str')
all_data.MoSold = all_data.MoSold.astype('str')

all_data, dummies = one_hot_encode(all_data)


# #### Removing Dummies with  none observations in train or test datasets
# This is such a simple action, we often find it to be obvious, but note that few books or articles make them as standard. In fact, it does not make sense to use categorical for you to train your model if there is no record in one of the training set or test, do not you agree?
# ![image](https://images-na.ssl-images-amazon.com/images/I/517sKy1FGPL._SX258_BO1,204,203,200_.jpg)

# In[ ]:


# Find Dummies with all test observatiosn are equal to 0
ZeroTest = all_data[dummies][ntrain:].sum()==0
all_data.drop(dummies[ZeroTest], axis=1, inplace=True)
print('Dummins in test dataset with all observatios equal to 0:',len(dummies[ZeroTest]),'of \n',dummies[ZeroTest],'\n')
dummies = dummies.drop(dummies[ZeroTest])

# Find dummies with all training observatiosn are equal to 0
ZeroTest = all_data[dummies][:ntrain].sum()==0
all_data.drop(dummies[ZeroTest], axis=1, inplace=True)
print('Dummins in trainig dataset with all observatios equal to 0:',len(dummies[ZeroTest]),'of \n',dummies[ZeroTest],'\n')
dummies = dummies.drop(dummies[ZeroTest])

del ZeroTest


# ### Transform Years to Ages and Create Flags to New and Remod
# Instead of falling into the discussion of whether years are ordinal or not, why not work with age?
# ![image](http://myplace.frontier.com/~pboakes/images/zits-053108.gif)

# In[ ]:


def AgeYears(feature): 
    return feature.apply(lambda x: 0 if x==0 else (2011 - x))

all_data.YearBuilt = AgeYears(all_data.YearBuilt)
all_data.YearRemodAdd = AgeYears(all_data.YearRemodAdd)
all_data.GarageYrBlt = AgeYears(all_data.GarageYrBlt) 
all_data.YrSold =  AgeYears(all_data.YrSold) 


# Altogether, there are 3 variables that are relevant with regards to the Age of a house; YearBlt, YearRemodAdd, and YearSold. YearRemodAdd defaults to YearBuilt if there has been no Remodeling/Addition. I will use YearRemodeled and YearSold to determine the Age. However, as parts of old constructions will always remain and only parts of the house might have been renovated, I will also introduce a Remodeled Yes/No variable. This should be seen as some sort of penalty parameter that indicates that if the Age is based on a remodeling date, it is probably worth less than houses that were built from scratch in that same year.

# In[ ]:


all_data['Remod'] = 2
all_data.loc[(all_data.YearBuilt==all_data.YearRemodAdd), ['Remod']] = 0
all_data.loc[(all_data.YearBuilt!=all_data.YearRemodAdd), ['Remod']] = 1

all_data.Age = all_data.YearRemodAdd - all_data.YrSold # sice I convert both to age

all_data["IsNew"] = 2
all_data.loc[(all_data.YearBuilt==all_data.YrSold), ['IsNew']] = 1
all_data.loc[(all_data.YearBuilt!=all_data.YrSold), ['IsNew']] = 0


# ### Check for any correlations between features
# ![image](http://flowingdata.com/wp-content/uploads/2011/07/Cancer-causes-cell-phones-625x203.png)
# To quantify the linear relationship between the features, I will now create a correlation matrix. 
#  
# The correlation matrix is identical to a covariance matrix computed from standardized data. The correlation matrix is a square matrix that contains the Pearson product-moment correlation coefficients (often abbreviated as [Pearson's r](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)), which measure the linear dependence between pairs of features:
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/602e9087d7a3c4de443b86c734d7434ae12890bc)
# Pearson's correlation coefficient can simply be calculated as the covariance between two features x and y (numerator) divided by the product of their standard deviations (denominator):
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/f76ccfa7c2ed7f5b085115086107bbe25d329cec)
# The covariance between standardized features is in fact equal to their linear correlation coefficient.
# Use NumPy's corrcoef and seaborn's heatmap functions to plot the correlation matrix array as a heat map.
# 
# To fit a linear regression model, we are interested in those features that have a high correlation with our target variable. So, I will make a zoom in these features in order of their correlation with SalePrice.

# In[ ]:


corr = all_data[all_data.SalePrice>1].corr()
top_corr_cols = corr[abs((corr.SalePrice)>=.26)].SalePrice.sort_values(ascending=False).keys()
top_corr = corr.loc[top_corr_cols, top_corr_cols]
dropSelf = np.zeros_like(top_corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
plt.figure(figsize=(20, 20))
sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
sns.set(font_scale=0.5)
plt.show()
del corr, dropSelf, top_corr


# We can see that our target variable SalePrice shows the largest correlation with the OverallQual variable (0.79), followed by.  GrLivArea (0.71). This seems to make sense, since in fact we expect the overall quality and size of the living area to have a greater influence on our value judgments about a property.
# 
# From the graph above, it also becomes clear the multicollinearity is an issue. 
#  - The correlation between GarageCars and GarageArea is very high (0.88), and has very close correlation with the SalePrice. 
#  - From total square feet of basement area (TotalBsmtSF) and first Floor square feet (1stFlrSF), we found 0.81 of correlation and same correlation with sale price (0.61).
#  - Original construction date (YearBuilt) has a little more correlation with price (0.52) than GarageYrBlt (0.49), and a high correlation between them (0.83)
#  - 0.83 is the correlation between total rooms above grade not include bathrooms (TotRmsAbvGrd) and GrLivArea, but TotRmsAbvGrd has only 0.51 of correlation with sale price.
#  
# Let's see their distributions and type of relation curve between the 10th features with largest correlation with sales price,

# #### Drop the features with highest correlations to other Features:
# 
# **Colinearity** is the state where two variables are highly correlated and contain similar information about the variance within a given dataset. And as you see above, it is easy to find ***highest colinearities***.
# ![image](https://s-media-cache-ak0.pinimg.com/originals/aa/1b/3d/aa1b3d19f534c2fccbd5a46c7887b924.jpg)
# You should always be concerned about the collinearity, regardless of the model/method being linear or not, or the main task being prediction or classification.
# 
# Assume a number of linearly correlated covariates/features present in the data set and Random Forest as the method. Obviously, random selection per node may pick only (or mostly) collinear features which may/will result in a poor split, and this can happen repeatedly, thus negatively affecting the performance.
# 
# Now, the collinear features may be less informative of the outcome than the other (non-collinear) features and as such they should be considered for elimination from the feature set anyway. However, assume that the features are ranked high in the 'feature importance' list produced by RF. As such they would be kept in the data set unnecessarily increasing the dimensionality. So, in practice, I'd always, as an exploratory step (out of many related) check the pairwise association of the features, including linear correlation.

# In[ ]:


all_data.drop(['FireplaceQu', 'BsmtSFPoints', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OverallQual', 'GrLivArea', 
               'TotalBsmtSF_x_Bsm', '1stFlrSF', 'PoolArea', 'LotArea', 'SaleCondition_Partial', 'Exterior1st_VinylSd',
               'GarageCond', 'HouseStyle_2Story', 'BsmtSFMultPoints', 'ScreenPorch', 'LowQualFinSF', 'BsmtFinSF2',
               'TSsnPorch'], axis=1, inplace=True) 


# In[ ]:


corr = all_data[all_data.SalePrice>1].corr()
top_corr_cols = corr[abs((corr.SalePrice)>=.26)].SalePrice.sort_values(ascending=False).keys()
top_corr = corr.loc[top_corr_cols, top_corr_cols]
dropSelf = np.zeros_like(top_corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
plt.figure(figsize=(20, 20))
sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
sns.set(font_scale=0.5)
plt.show()
del corr, dropSelf, top_corr


# In[ ]:


sns.set(font_scale=1.0)
g = sns.pairplot(all_data.loc[all_data.SalePrice>0, top_corr_cols[:12]])


# From the pair plot above we note some points the need attention, like:
#  - We can confirm the treatment of some outliers, most of then is area features. 
#    - So, we confirm the reduce with a little cut of outliers.
#    - On the other hand, if we continue to see some of others outliers we need attention to:
#      - Possible use of a robust scaler.
#      - Search, evaluate and cut outliers on base of residuals plot.
#      - As an alternative to throwing out outliers, we will look at a robust method of regression using the RANdom SAmple Consensus (RANSAC) algorithm, which fits a regression model to a subset of the data.
#  - We can see that most of data appears skewed and some of than has peaks long tails. It is suggest a use of box cox transformation
#  - The Sale Price appears skewed and has a long right tail.

# #### Identify  and treat multicollinearity:
# **Multicollinearity** is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model, leading to unreliable and unstable estimates of regression coefficients. To make matters worst multicollinearity can emerge even when isolated pairs of variables are not collinear.
# 
# To identify, we need start with the coefficient of determination, r<sup>2</sup>, is the square of the Pearson correlation coefficient r. The coefficient of determination, with respect to correlation, is the proportion of the variance that is shared by both variables. It gives a measure of the amount of variation that can be explained by the model (the correlation is the model). It is sometimes expressed as a percentage (e.g., 36% instead of 0.36) when we discuss the proportion of variance explained by the correlation. However, you should not write r<sup>2</sup> = 36%, or any other percentage. You should write it as a proportion (e.g., r<sup>2</sup> = 0.36).
# ![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTS_TVxaBpLmAGthSUAS9w7SVKsmLOtocz7ts-MXioJwa-Se0U)
# 
# Already the **Variance Inflation Factor** (**VIF**) is a measure of collinearity among predictor variables within a multiple regression.  It is may be calculated for each predictor by doing a linear regression of that predictor on all the other predictors, and then obtaining the R<sup>2</sup> from that regression.  It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variance of a single beta if it were fit alone [1/(1-R<sup>2</sup>)]. Thus, a VIF of 1.8 tells us that the variance (the square of the standard error) of a particular coefficient is 80% larger than it would be if that predictor was completely uncorrelated with all the other predictors. The VIF has a lower bound of 1 but no upper bound. Authorities differ on how high the VIF has to be to constitute a problem (e.g.: 2.50 (R<sup>2</sup> equal to 0.6), sometimes 5 (R<sup>2</sup> equal to .8), or greater than 10 (R<sup>2</sup> equal to 0.9) and so on). 
# 
# But there are several situations in which multicollinearity can be safely ignored:
# 
#  - ***Interaction terms*** and ***higher-order terms*** (e.g., ***squared*** and ***cubed predictors***) ***are correlated*** with main effect terms because they include the main effects terms. **Ops!** Sometimes we use ***polynomials*** to solve problems, **indeed!** But keep calm, in these cases,  **standardizing** the predictors can **removed the multicollinearity**. 
#  - ***Indicator***, like ***dummy*** or ***one-hot-encode***, that represent a ***categorical variable with three or more categories***. If the proportion of cases in the reference category is small, the indicator will necessarily have high VIF's, even if the categorical is not associated with other variables in the regression model. But, you need check if some dummy is collinear or has multicollinearity with other features outside of their dummies.
#  - ***Control feature** if the ***feature of interest*** **do not have high VIF's**. Here's the thing about multicollinearity: it's only a problem for the features that are **collinear**. It increases the standard errors of their coefficients, and it may make those coefficients unstable in several ways. But so long as the collinear feature are only used as control feature, and they are not collinear with your feature of interest, there's no problem. The coefficients of the features of interest are not affected, and the performance of the control feature as controls is not impaired.
# 
# So, generally, we could run the same model twice, once with severe multicollinearity and once with moderate multicollinearity. This provides a great head-to-head comparison and it reveals the classic effects of multicollinearity. However, when standardizing your predictors doesn't work, you can try other solutions such as:
# - removing highly correlated predictors
# - linearly combining predictors, such as adding them together
# - running entirely different analyses, such as partial least squares regression or principal components analysis
# 
# When considering a solution, keep in mind that all remedies have potential drawbacks. If you can live with less precise coefficient estimates, or a model that has a high R-squared but few significant predictors, doing nothing can be the correct decision because it won't impact the fit.
# 
# Given the potential for correlation among the predictors, we'll have display the variance inflation factors (VIF), which indicate the extent to which multicollinearity is present in a regression analysis. Hence such variables need to be removed from the model. Deleting one variable at a time and then again checking the VIF for the model is the best way to do this.
# 
# So, I start the analysis already having removed the features with he highest collinearities and run VIF.

# In[ ]:


all_data.rename(columns={'2ndFlrSF':'SndFlrSF'}, inplace=True)

def VRF(predict, data, y):
   
    scale = StandardScaler(with_std=False)
    df = pd.DataFrame(scale.fit_transform(data), columns= cols)
    features = "+".join(cols)
    df['SalePrice'] = y.values

    # get y and X dataframes based on this regression:
    y, X = dmatrices(predict + ' ~' + features, data = df, return_type='dataframe')

   # Calculate VIF Factors
    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    # Inspect VIF Factors
    display(vif.sort_values('VIF Factor'))
    return vif

# Remove the higest correlations and run a multiple regression
cols = all_data.columns
cols = cols.drop(['SalePrice'])
vif = VRF('SalePrice', all_data.loc[all_data.SalePrice>0, cols], all_data.SalePrice[all_data.SalePrice>0])


# From the results, I conclude that we need work in the features where's the VIF stated as inf. This is done to prevent multicollinearity, or the dummy variable trap caused by including a dummy variable for every single category. let's try remove one or two dummies for every category and check if it solves the other dummies from its category:

# In[ ]:


# Remove one feature with VIF on Inf from the same category and run a multiple regression
cols = cols.drop(['Condition1_PosN', 'Neighborhood_NWAmes', 'Exterior1st_CBlock', 'BldgType_1Fam', 'RoofStyle_Flat',
                  'MSZoning_Call', 'Alley_Grvl', 'LandContour_Bnk', 'LotConfig_Corner', 'GarageType_2Types', 'MSSubClass_45',
                  'MasVnrType_BrkCmn', 'Foundation_CBlock', 'MiscFeature_Gar2', 'SaleType_COD', 'Exterior2nd_CBlock'])

vif = VRF('SalePrice', all_data.loc[all_data.SalePrice>0, cols], all_data.SalePrice[all_data.SalePrice>0])


# Excellent, we are in the good path, but ...
# ![image](https://media1.tenor.com/images/c91d63cf0b36c261720637f8b61e9c6a/tenor.gif?itemid=5090669)
# ... we need continuing work on the remaining features to reduce the VIF, lets to continue and try to get less then 50.

# In[ ]:


# Remove one feature with highest VIF from the same category and run a multiple regression
cols = cols.drop(['PoolQC', 'BldgType_TwnhsE', 'BsmtFinSF1', 'BsmtUnfSF', 'Electrical_SBrkr', 'Exterior1st_MetalSd',
                  'Exterior2nd_VinylSd', 'GarageQual', 'GarageType_Attchd', 'HouseStyle_1Story', 'MasVnrType_None',
                  'MiscFeature_NA', 'MSZoning_RL', 'RoofStyle_Gable', 'SaleCondition_Normal', 'MoSold_10',
                  'SaleType_New', 'SndFlrSF', 'TotalPorchSF', 'WoodDeckSF', 'BldgType_Duplex', 'MSSubClass_90'])
              
vif = VRF('SalePrice', all_data.loc[all_data.SalePrice>0, cols], all_data.SalePrice[all_data.SalePrice>0])


# ### Defining Categorical and Boolean Data as unit8 types
# Remember we used the panda for one hot encode and some categorical ones had already been provided or created as numbers. So, in order that the models do not make inappropriate use of features transformed into numbers and apply only calculations relevant to categorical, we have to transform their type into category type or in unit8 to leave some calculations.
# ![image](http://www.coaltocashhomebuyers.com/wp-content/uploads/2015/01/driveway-300x246.gif)

# In[ ]:


# Reserve a copy for futer analysis:
df_copy = all_data[all_data.SalePrice>0].copy()

all_data.CentralAir = all_data.CentralAir.astype('uint8')
all_data.Garage_Newest = all_data.Garage_Newest.astype('uint8')
all_data.EnclosedPorch = all_data.EnclosedPorch.astype('uint8')
all_data.FullBath = all_data.FullBath.astype('uint8')
all_data.HalfBath = all_data.HalfBath.astype('uint8')
all_data.BsmtFullBath = all_data.BsmtFullBath.astype('uint8')
all_data.BsmtHalfBath = all_data.BsmtHalfBath.astype('uint8')
all_data.Remod = all_data.Remod.astype('uint8')
all_data.IsNew = all_data.IsNew.astype('uint8') 
all_data.Street = all_data.Street.astype('uint8') # orinal
all_data.PavedDrive = all_data.PavedDrive.astype('uint8') # ordinal
all_data.Functional = all_data.Functional.astype('uint8') # ordinal
all_data.LandSlope = all_data.LandSlope.astype('uint8') # ordinal

'''
for feat in cols:
    if all_data[feat].dtype=='uint8':
        all_data[feat] = all_data[feat].astype('category')
'''


# ### Box cox transformation of highly skewed features
# A Box Cox transformation is a way to transform non-normal data distribution into a normal shape. 
# ![image](https://i.pinimg.com/originals/d1/9f/7c/d19f7c7f5daaed737ab2516decea9874.png)
# Why does this matter?
# - **Model bias and spurious interactions**: If you are performing a regression or any statistical modeling, this asymmetrical behavior may lead to a bias in the model. If a factor has a significant effect on the average, because the variability is much larger, many factors will seem to have a stronger effect when the mean is larger. This is not due, however, to a true factor effect but rather to an increased amount of variability that affects all factor effect estimates when the mean gets larger. This will probably generate spurious interactions due to a non-constant variation, resulting in a **very complex model** with many **spurious** and **unrealistic** interactions.
# - **Normality is an important assumption for many statistical techniques**: such as individuals control charts, Cp/Cpk analysis, t-tests and analysis of variance (ANOVA). A substantial departure from normality will bias your capability estimates.
# 
# One solution to this is to transform your data into normality using a [Box-Cox transformation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) means that you are able to run a broader number of tests.
# 
# At the core of the Box Cox transformation is an exponent, lambda (λ), which varies from -5 to 5. All values of λ are considered and the optimal value for your data is selected; The 'optimal value' is the one which results in the best approximation of a normal distribution curve. The transformation of Y has the form:
# <a href="http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/boxcox.png"><img src="http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/boxcox.png" alt="" width="222" height="70" class="aligncenter size-full wp-image-13940" /></a>
# 
# The scipy implementation proceeded with this formula, then you need before take care of negatives values if you have. A common technique for handling negative values is to add a constant value to the data prior to applying the log transform. The transformation is therefore log(Y+a) where a is the constant. Some people like to choose a so that min(Y+a) is a very small positive number (like 0.001). Others choose a so that min(Y+a) = 1. For the latter choice, you can show that a = b – min(Y), where b is either a small number or is 1.
# This test only works for positive data. However, Box and Cox did propose a second formula that can be used for negative y-values, not implemented in scipy:
# <a href="http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/boxcoxNeg.png"><img src="http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/boxcoxNeg.png" alt="" width="272" height="76" class="aligncenter size-full wp-image-13941" /></a>
# The formula are deceptively simple. Testing all possible values by hand is unnecessarily labor intensive.
# 
# <p align='center'> Common Box-Cox Transformations 
# </p>
# 
# | Lambda value (λ) | Transformed data (Y') |
# |------------------|-----------------------|
# |        -3	       | Y\*\*-3 = 1/Y\*\*3    |
# |        -2        | Y\*\*-2 = 1/Y\*\*2    |
# |        -1        | Y\*\*-1 = 1/Y         |
# |       -0.5       | Y\*\*-0.5 = 1/(√(Y))  |
# |         0        | log(Y)(\*)            |
# |        0.5       | Y0.5 = √(Y)           |
# |         1        | Y\*\*1 = Y            |
# |         2        | Y\*\*2                |
# |         3        | Y\*\*3                |
# 
# (\*)Note: the transformation for zero is log(0), otherwise all data would transform to Y\*\*0 = 1.
# The transformation doesn't always work well, so make sure you check your data after the transformation with a normal probability plot or if the skew are reduced, tending to zero.

# In[ ]:


numeric_features = list(all_data.loc[:, cols].dtypes[(all_data.dtypes != "category") & (all_data.dtypes !='uint8')].index)

'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
'''
skewed_features = all_data[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness
skewness = pd.DataFrame({'Skew' :skewed_features})   

# Get only higest skewed features
skewness = skewness[abs(skewness) > 0.7]
skewness = skewness.dropna()
print ("There are {} higest skewed numerical features to box cox transform".format(skewness.shape[0]))

l_opt = {}

for feat in skewness.index:
    all_data[feat], l_opt[feat] = boxcox((all_data[feat]+1))

skewed_features2 = all_data[skewness.index].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness
skewness2 = pd.DataFrame({'New Skew' :skewed_features2})   
display(pd.concat([skewness, skewness2], axis=1).sort_values(by=['Skew'], ascending=False))


# As you can see, we were able at first to bring most the numerical values closer to normal. Maybe you're not satisfied with the results of MiscVal and Kitchener and want to understand if we really need to continue to transform some discrete data. So, let's take a look at the QQ test of these features.

# In[ ]:


def QQ_plot(data, measure):
    fig = plt.figure(figsize=(12,4))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.\
                   format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()
    
for feat in skewness.index:
    QQ_plot(all_data[feat], ('Boxcox1p of {}'.format(feat)))


# As you have seen, really MiscVal and Kitchener really do not seem to be good results, especially MiscVal, but it is a fact that both variables do not look good indifferent to their distribution. 
# 
# As for the other discrete variables, in addition to having presented significant improvements, they also pass the QQ test and present interesting distributions as we can observe in their respective graphs.
# 
# So, we can continue to apply the BoxCox on this features and leave to feature selection algorithms to decide if we continue with some of then or not.

# ### Evaluate Apply Polynomials by Region Plots on the more Correlated Features 

# In[ ]:


fig = plt.figure(figsize=(20,15)) 
ax = fig.add_subplot(331); g = sns.regplot(y='SalePrice', x='ConstructArea', data = all_data[all_data.SalePrice>0], order=3) 
ax = fig.add_subplot(332); g = sns.regplot(y='SalePrice', x='GarageArea_x_Car', data = all_data[all_data.SalePrice>0], order=3)
ax = fig.add_subplot(333); g = sns.regplot(y='SalePrice', x='LotAreaMultSlope', data = all_data[all_data.SalePrice>0], order=3) 
ax = fig.add_subplot(334); g = sns.regplot(y='SalePrice', x='TotalExtraPoints', data = all_data[all_data.SalePrice>0], order=4) 
ax = fig.add_subplot(335); g = sns.regplot(y='SalePrice', x='TotalPoints', data = all_data[all_data.SalePrice>0], order=3) 

plt.tight_layout() 


# #### Evaluating Polynomials Options Performance
# One way to account for the violation of linearity assumption is to use a polynomial regression model by adding polynomial terms.
# ![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQQIQ1HTrA1PzE7sw5CwOiV3XWhKXz-rGLj7FMmxYZO_CsU1Iz)
# Although we can use polynomial regression to model a nonlinear relationship, it is still considered a multiple linear regression model because of the linear regression coefficients w.
# 
# Moreover, as we have seen, some of our features are better when interacting with each other than with just observed ones, but some have a negative effect.
# 
# So, let's check it more carefully.

# In[ ]:


def poly(X, y, feat=''):

    # Initializatin of regression models
    regr = LinearRegression()
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X)
    linear_r2 = r2_score(y, regr.predict(X))

    # create polynomial features
    quadratic = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    cubic = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
    fourth = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)
    fifth = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)
    X_fourth = fourth.fit_transform(X)
    X_fifth = fifth.fit_transform(X)

    # quadratic fit
    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X))
    quadratic_r2 = r2_score(y, y_quad_fit)

    # cubic fit
    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X))
    cubic_r2 = r2_score(y, y_cubic_fit)

    # Fourth fit
    regr = regr.fit(X_fourth, y)
    y_fourth_fit = regr.predict(fourth.fit_transform(X))
    four_r2 = r2_score(y, y_fourth_fit)

    # Fifth fit
    regr = regr.fit(X_fifth, y)
    y_fifth_fit = regr.predict(fifth.fit_transform(X))
    five_r2 = r2_score(y, y_fifth_fit)
    
    if len(feat)>0:
        fig = plt.figure(figsize=(20,5))
        # Plot lowest Polynomials
        fig1 = fig.add_subplot(121)
        plt.scatter(X[feat], y, label='training points', color='lightgray')
        plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=0.5, linestyle=':')
        plt.plot(X[feat], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2, color='red', lw=0.5, linestyle='-')
        plt.plot(X[feat], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2,  color='green', lw=0.5, linestyle='--')

        plt.xlabel(feat)
        plt.ylabel('Sale Price')
        plt.legend(loc='upper left')

        # Plot higest Polynomials
        fig2 = fig.add_subplot(122)
        plt.scatter(X[feat], y, label='training points', color='lightgray')
        plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=2, linestyle=':')
        plt.plot(X[feat], y_fifth_fit, label='Fifth (d=5), $R^2=%.3f$' % five_r2, color='yellow', lw=2, linestyle='-')
        plt.plot(X[feat], y_fifth_fit, label='Fourth (d=4), $R^2=%.3f$' % four_r2, color='red', lw=2, linestyle=':')

        plt.xlabel(feat)
        plt.ylabel('Sale Price')
        plt.legend(loc='upper left')
    else:
        # Plot initialisation
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, s=40)

        # make lines of the regressors:
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, 
                 color='blue', lw=2, linestyle=':')
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2, 
                 color='red', lw=0.5, linestyle='-')
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2, 
                 color='green', lw=0.5, linestyle='--')
        # label the axes
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel('Sales Price')
        ax.set_title("Poly up to 3 degree")
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# In[ ]:


y = all_data.SalePrice[all_data.SalePrice>0]
X = all_data.loc[all_data.SalePrice>0, ['ConstructArea']] 
poly(X, y, 'ConstructArea')

X = all_data.loc[all_data.SalePrice>0, ['ConstructArea', 'TotalPoints']] 
poly(X, y)

X = all_data.loc[all_data.SalePrice>0, ['ConstructArea', 'TotalPoints', 'LotAreaMultSlope',  'GarageArea_x_Car']] 
poly(X, y)


# As you can see, our third-degree polynomial with only Construct Area provides an improvement of 0.6%, while the one with  Construct Area and Total Points has a significant improvement of 10.5% and final R<sup>2</sup> of 85,6% in third-degree, but it represents only 0.6% again from without polynomials to a third-degree. We should be aware that adding more and more polynomial features increases the complexity of a model and therefore increases the chance of overfit. Se that our 4 features polynomials present a R<sup>2</sup> of 87.2%, a increase of only 1.6% from the polynomial only with 2 features and it represent a increase of 0.9% in the gain (without and with poly).
# 
# If you have downloaded this notebook to run it, try to include some other variables with high correlation index, one at a time. You will find that some cause an opposite effect, reducing the accuracy of the model, for example change the TotalPoints by TotalExtraPoints. So, you can see that is better include a controlled polynomial interaction then just include the polynomial above all features in the pipeline.

# #### Create Degree 3 Polynomials Features
# As you saw, it is not appropriate to apply the polynomial to all of our variables, so let's create a code that applies the third-degree polynomial only in our previous 4 features and concatenate its result to our data set.

# In[ ]:


poly_cols = ['ConstructArea', 'TotalPoints', 'LotAreaMultSlope',  'GarageArea_x_Car']

pf = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
res = pf.fit_transform(all_data.loc[:, poly_cols])

target_feature_names = [feat.replace(' ','_') for feat in pf.get_feature_names(poly_cols)]
output_df = pd.DataFrame(res, columns = target_feature_names,  index=all_data.index).iloc[:, len(poly_cols):]
print('Polynomial Features included:', output_df.shape[1])
display(output_df.head())
all_data = pd.concat([all_data, output_df], axis=1)
print('Total Features after Polynomial Features included:', all_data.shape[1])
colsP = output_df.columns

del output_df, target_feature_names, res, pf


# ## Separate Train, Test Datasets, identifiers and Dependent Variable
# 
# In the next steps we will select features, reduce dimensions and run models, so it is important to separate our data sets again in training, test, id and dependent variable.

# In[ ]:


y_train = (all_data.SalePrice[all_data.SalePrice>0].reset_index(drop=True, inplace=False))

# Data with Polynomials
train = all_data.loc[(all_data.SalePrice>0), cols].reset_index(drop=True, inplace=False)
test = all_data.loc[(all_data.SalePrice==0), cols].reset_index(drop=True, inplace=False)


# ## Select Features
# It is important to consider feature selection a part of the model selection process. If you do not, you may inadvertently introduce bias into your models which can result in overfitting.
# 
# Compare several feature selection methods, including your new idea, correlation coefficients, backward selection and embedded methods. Use linear and non-linear predictors. Select the best approach with model selection.
# 
# Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.
# 
# All of the features we find in the dataset might not be useful in building a machine learning model to make the necessary prediction. Using some of the features might even make the predictions worse. 
# ![selectFeat](http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/Select_features.png)
# Often in data science we have hundreds or even millions of features and we want a way to create a model that only includes the most important features. This has three benefits.
# 
# 1. It reduces the variance of the model, and therefore overfitting.
# 2. It reduces the complexity of a model and makes it easier to interpret.
# 3. It improves the accuracy of a model if the right subset is chosen.
# 4.Finally, it reduces the computational cost (and time) of training a model.
# So, an alternative way to reduce the complexity of the model and avoid overfitting is dimensionality reduction via feature selection, which is especially useful for unregularized models. There are two main categories of dimensionality reduction techniques: feature selection and feature extraction. Using feature selection, we select a subset of the original features. In feature extraction, we derive information from the feature set to construct a new feature subspace.
# ![image](http://www.bigdata-madesimple.com/wp-content/uploads/2015/01/bigdata-knows-everything.jpg)
# Exist various methodologies and techniques that you can use to subset your feature space and help your models perform better and efficiently. So, let's get started.

# ### Prepare Data to Select Features
# Let's create a data set, at first without applying our third-degree polymorph and already robust scaled.

# In[ ]:


scale = RobustScaler()
# Data without Polynomials
df = pd.DataFrame(scale.fit_transform(train[cols]), columns= cols)


# ### Wrapper Methods
# In wrapper methods, we try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, we decide to add or remove features from your subset. The problem is essentially reduced to a search problem. 
# ![image](https://static.wixstatic.com/media/0ecdaf_80b92d491f82441cb886f5787ea67f24.gif)
# 
# The two main disadvantages of these methods are : 
# - The increasing overfitting risk when the number of observations is insufficient.
# - These methods are usually computationally very expensive.
# 
# #### Backward Elimination
# In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.
# 
# We will see below row implementation of backward elimination, one to select by P-values and other based on the accuracy of a model the we submitted to it.
# 
# ##### Backward Elimination By P-values
# 
# The **P-value**, or probability value, or asymptotic significance, is a **probability** value for a given **statistical model** that, ***if the null hypothesis is true***, a set of statistical observations more commonly known as **the statistical summary** <i>is greater than or equal in magnitude to</i> **the observed results**.
# 
# The **null hypothesis** is a general statement that **there is no relationship between two measured phenomena**.
# 
# For example, if the correlation is very small and furthermore, the p-value is high meaning that it is very likely to observe such correlation on a dataset of this size purely by chance.
# 
# But you need to be careful how you interpret the statistical significance of a correlation. If your correlation coefficient has been determined to be statistically significant this does not mean that you have a strong association. It simply tests the null hypothesis that there is no relationship. By rejecting the null hypothesis, you accept the alternative hypothesis that states that there is a relationship, but with no information about the strength of the relationship or its importance.
# 
# Since removal of different features from the dataset will have different effects on the p-value for the dataset, we can remove different features and measure the p-value in each case. These measured p-values can be used to decide whether to keep a feature or not.
# 
# From statsmodels.api, the [linear models](https://www.statsmodels.org/dev/regression.html) allows estimation by ordinary least squares ([OLS](https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html)), weighted least squares ([WLS](https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html)), generalized least squares ([GLS](https://www.statsmodels.org/dev/examples/notebooks/generated/gls.html)), and feasible generalized [recursive least squares](https://www.statsmodels.org/dev/examples/notebooks/generated/recursive_ls.html) with autocorrelated AR(p) errors.
# 
# Next we make the test of a ***Linear regression*** to check the result and **select features** based on its the **P-value**:

# In[ ]:


ln_model=sm.OLS(y_train,df)
result=ln_model.fit()
print(result.summary2())


# Like before, I excluded one by one of the features with the highest P-value and run again until get only P-values up to 0.05, but here I use a backward elimination process.

# In[ ]:


pv_cols = cols.values

def backwardElimination(x, Y, sl, columns):
    ini = len(columns)
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor = sm.OLS(Y, x).fit()
        maxVar = max(regressor.pvalues) #.astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor.pvalues[j].astype(float) == maxVar):
                    columns = np.delete(columns, j)
                    x = x.loc[:, columns]
                    
    print('\nSelect {:d} features from {:d} by best p-values.'.format(len(columns), ini))
    print('The max p-value from the features selecte is {:.3f}.'.format(maxVar))
    print(regressor.summary())
    
    # odds ratios and 95% CI
    conf = np.exp(regressor.conf_int())
    conf['Odds Ratios'] = np.exp(regressor.params)
    conf.columns = ['2.5%', '97.5%', 'Odds Ratios']
    display(conf)
    
    return columns, regressor

SL = 0.051

pv_cols, LR = backwardElimination(df, y_train, SL, pv_cols)


# From the results, we can highlight:
# - we're very confident about some relationship between the probability of raising prices:
#  - there is an sgnificante inverse relationship with Fireplaces and  Roof Material CompShg.
#  - there is an positive relationship, from greater to low, with HalfBath, BsmtFinType2, BldgType, Foundation, MasVnrType, YearRemodAdd, Fence, BsmtHalfBath, HouseStyle, ExterCond, Exterior2nd, FullBath, TotalExtraPoints, BsmtFullBath, Exterior1st, KitchenAbvGr, Neighborhood
# ![image](https://media.giphy.com/media/3o6MbaBBOIlKBk9ZvO/giphy.gif) 
# - From the coefficient:
#  - As expected, the neighborhood makes a lot of difference, which is confirmed by the presence of all 24 dummy dummies after only one exclusion by the FIV.
#  - From MSSubClass category we can see that  thirteen of them are among the first seventeen highest coefficients.
#  - Is interest to note that KitchenAbvGr is the most highest coefficient, you can imagine this before? 
#  - Note that from our chosen polynomial variables were selected only construction area and Total Extra Points.
#  
# Take the **exponential** of each of the **coefficients** to generate the ***odds ratios***. This tells you how a 1 unit increase or decrease in a variable affects the odds of raising prices. For example, we can expect the odds of price to decreases n 18.5% with the numbers of Fireplaces.

# ![image](http://vignette1.wikia.nocookie.net/disney/images/e/e5/Asf.gif/revision/latest?cb=20160317185039)
# 
# Let's take a look at the graphs of some of the interactions of the selected features:

# In[ ]:


pred = LR.predict(df[pv_cols])

df_copy['proba'] = pred

y_pred = pred.apply(lambda x: 1 if x > 0.5 else 0)

print('Fvalue: {:.6f}'.format(LR.fvalue))
print('MSE total on the train data: {:.4f}'.format(LR.mse_total))
def plot_proba(continous, predict, discret, data):
    grouped = pd.pivot_table(data, values=[predict], index=[continous, discret], aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in data[discret].unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        plt.plot(plt_data.index.get_level_values(0), plt_data[predict], color=colors[int(col)])
    plt.xlabel(continous)
    plt.ylabel("Probabilities")
    plt.legend(np.sort(data[discret].unique()), loc='upper left', title=discret)
    plt.title("Probabilities with " + continous + " and " + discret)

fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(241); plot_proba('ConstructArea', 'SalePrice', 'Remod', df_copy)
ax = fig.add_subplot(242); plot_proba('TotalPoints', 'SalePrice', 'ExterCond', df_copy)
ax = fig.add_subplot(243); plot_proba('BsmtFinType1', 'SalePrice', 'BsmtCond', df_copy)
ax = fig.add_subplot(245); plot_proba('TotalExtraPoints', 'SalePrice', 'LotShape', df_copy)

plt.show()
del df_copy


# As expected, we can see that prices grow with the growth of the built area, although the reform does not seem to contribute higher prices, in fact we have to remember that if a house went through renovation it is indeed old enough to have needed, and New homes tend to be more expensive. If you wish, switch from Remod to IsNew and see for yourself.
# 
# Something similar can be seen in relation to the total points segmented by the external condition, while we see that prices grow with the growth of the points, we see that although the external condition presents a small positive coefficient, the graph may be suggesting something different, but note that level 3 stands out at the beginning and around the mean, which would explain a small positive coefficient.
# 
# Also see that more important than basement conditions is its purpose in itself. Basements with living conditions present higher prices, curiously unfinished ones too, perhaps because they get the new owners to make them what they want.
# 
# As for the lot multiplied by the slope, as we already know we see the trend of price increase with lot size, but much variation, since other aspects influence the price as well as the slope itself.
# 
# Finally, the total of extra points, there is nothing new when we see that the bigger the better, segmented by the format of the lot, we see that the more regular the better it is, but if the terrain is unregulated the extra high score will not work.
# 
# This is the beauty of linear models, even with many features it is possible to understand them when evaluating their coefficients, significance and graphs , as we can see how certain variables present noise and its influence on variance and bias.

# #### Select Features by Recursive Feature Elimination
# The goal of [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) (RFE) is to select features by recursively considering smaller and smaller sets of features.
# 
# RFE is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. 
# 
# Other option is sequential Feature Selector (SFS) from mlxtend, a separate Python library that is designed to work well with scikit-learn, also provides a S that works a bit differently.
# 
# RFE is computationally less complex using the feature's weight coefficients (e.g., linear models) or feature importances (tree-based algorithms) to eliminate features recursively, whereas SFSs eliminate (or add) features based on a user-defined classifier/regression performance metric.
# 
# The scikit-learn has two implementations of RFE, let's see the [RFECV](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV) through a Lasso model to make the feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
# 
# In fact, this algorithm is quite efficient and makes a selection that will produce the best results when we go through the hyper parametrization phase.

# In[ ]:


ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
rfecv = RFECV(estimator=ls, n_jobs = -1, step=1, scoring = 'neg_mean_squared_error' ,cv=5)
rfecv.fit(df, y_train)

select_features_rfecv = rfecv.get_support()
RFEcv = cols[select_features_rfecv]
print('{:d} Features Select by RFEcv:\n{:}'.format(rfecv.n_features_, RFEcv.values))


# #### Sequential feature selection
# 
# **Sequential feature selection algorithms** are a family of **greedy search algorithms** that are used to reduce an initial d-dimensional feature space to a k-dimensional feature subspace where k < d. The motivation behind feature selection algorithms is to automatically select a subset of features that are most relevant to the problem to improve computational efficiency or reduce the generalization error of the model by removing irrelevant features or noise, ***which can be useful for algorithms that don't support regularization***.
# 
# Greedy algorithms make locally optimal choices at each stage of a combinatorial search problem and generally yield a suboptimal solution to the problem in contrast to exhaustive search algorithms, which evaluate all possible combinations and are guaranteed to find the optimal solution. However, in practice, an exhaustive search is often computationally not feasible, whereas greedy algorithms allow for a less complex, computationally more efficient solution.
# 
# As you saw in the previous topic, RFE is computationally less complex using the feature weight coefficients (e.g., linear models) or feature importance (tree-based algorithms) to eliminate features recursively, whereas SFSs eliminate (or add) features based on a user-defined classifier/regression performance metric.
# 
# The SBS aims to reduce the dimensionality of the initial feature subspace with a minimum decay in performance of the regressor or classifier to improve upon computational efficiency. In certain cases, SBS can even improve the predictive power of the model if a model suffers from overfitting.
# 
# SBS sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features. In order to determine which feature is to be removed at each stage, we need to define criterion function J that we want to minimize. The criterion calculated by the criterion function can simply be the difference in performance of the classifier after and before the removal of a particular feature. Then the feature to be removed at each stage can simply be defined as the feature that maximizes this criterion.
# 
# So, let's see a example of SBS in our data, 

# In[ ]:


from itertools import combinations
class SBS():
    def __init__(self, estimator, k_features, scoring=r2_score, test_size=0.25, random_state=101):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = list(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, list(p))
                scores.append(score)
                subsets.append(list(p))
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
            
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X.iloc[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train.iloc[:, indices], y_train)
        y_pred = self.estimator.predict(X_test.iloc[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

score = r2_score
ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
sbs = SBS(ls, k_features=1, scoring= score)
sbs.fit(df, y_train)

k_feat = [len(k) for k in sbs.subsets_]
fig = plt.figure(figsize=(10,5))
plt.plot(k_feat, sbs.scores_, marker='o')
plt.xlim([1, len(sbs.subsets_)])
plt.xticks(np.arange(1, len(sbs.subsets_)+1))
plt.ylabel('R2 Score')
plt.xlabel('Number of features')
plt.grid(b=1)
plt.show()

print('Best Score: {:2.2%}\n'.format(max(sbs.scores_)))
print('Best score with:{0:2d}.\n'.\
      format(len(list(df.columns[sbs.subsets_[np.argmax(sbs.scores_)]]))))
SBS = list(df.columns[list(sbs.subsets_[max(np.arange(0, len(sbs.scores_))[(sbs.scores_==max(sbs.scores_))])])])
print('\nBest score with {0:2d} features:\n{1:}'.format(len(SBS), SBS))


# As you saw, the SBS is straightforward code to understand, but is computationally expensive. In a nutshell, SFAs remove or add one feature at the time based on the classifier or regressior performance until a feature subset of the desired size k is reached. There are 4 different flavors of SFAs available via the SequentialFeatureSelector from [mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/):
# - Sequential Forward Selection (SFS)
# - Sequential Backward Selection (SBS)
# - Sequential Forward Floating Selection (SFFS)
# - Sequential Backward Floating Selection (SBFS)
# 
# The next code use the SBS from the mlxten. It has interest features to explore, but is more computationally expensive than previous code, so, take care if you try running it.
# Sequential Backward Selection
ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
#lr = LinearRegression()

sbs = SFS(estimator = ls, k_features = 52, forward= False, floating = False, scoring = 'neg_mean_squared_error', 
          cv = 3, n_jobs = -1)
sbs = sbs.fit(df.values, y_train.values)

print('\nSequential Backward Selection:')
print(sbs.k_feature_idx_)
print('\nCV Score:', sbs.k_score_)
display(pd.DataFrame.from_dict(sbs.get_metric_dict()).T)

fig1 = plot_sfs(sbs.get_metric_dict(), kind='std_err')
plt.title('Sequential Backward Selection (w. StdErr)')
plt.grid()
plt.show()
# SBS is actually computationally expensive, but also generated models with better performance when we go through the hyper parameterization phase.

# ### Feature Selection by Filter Methods
# Filter methods use statistical methods for evaluation of a subset of features, they are generally used as a preprocessing step. These methods are also known as **univariate feature selection**, they examines each feature individually to determine the strength of the relationship of the feature with the dependent variable. These methods are **simple to run and understand** and are in general particularly **good for gaining a better understanding** of data, but **not necessarily for optimizing the feature set for better generalization**.
# 
# ![image](http://imgs.xkcd.com/comics/boyfriend.png)
# 
# So, the features are selected on the basis of their scores in various statistical tests for their correlation with the outcome variable. The correlation is a subjective term here. For basic guidance, you can refer to the following table for defining correlation co-efficients.
# 
# | Feature/Response |       Continuous      | Categorical
# |------------------|-----------------------|------------
# | Continuous       | Pearson's Correlation | LDA
# | Categorical      | Anova                 | Chi-Square
# 
# One thing that should be kept in mind is that filter methods do not remove multicollinearity. So, you must deal with multicollinearity of features as well before training models for your data.
# 
# There are lot of different options for univariate selection. Some examples are:
# - Univariate feature selection
# - Model Based Ranking

# #### Univariate feature selection
# On scikit-learn we find variety of implementation oriented to regression tasks to select features according to the [k highest scores](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html), see below some of that:
# - [f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) The Pearson's Correlation are covert to F score then to a p-value. So, the selection is based on the F-value between label/feature for regression tasks.
# - [mutual_info_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression) estimate mutual information for a continuous target variable. The MI between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency. The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances.
# ![image](https://blogradiusagent.files.wordpress.com/2018/07/tenor.gif?w=770)
# 
# The methods based on F-test estimate the degree of linear dependency between two random variables. On the other hand, mutual information methods can capture any kind of statistical dependency, but being nonparametric, they require more samples for accurate estimation.
# 
# Other important point is if you use sparse data, for example if we continue consider hot-encode of some categorical data with largest number of distinct values, mutual_info_regression will deal with the data without making it dense.
# 
# Let's see the SelectKBest of f_regression and mutual_info_regression for our data:

# In[ ]:


skb = SelectKBest(score_func=f_regression, k=80)
skb.fit(df, y_train)
select_features_kbest = skb.get_support()
kbest_FR = cols[select_features_kbest]
scores = skb.scores_[select_features_kbest]
feature_scores = pd.DataFrame([(item, score) for item, score in zip(kbest_FR, scores)], columns=['feature', 'score'])
fig = plt.figure(figsize=(40,20))
f1 = fig.add_subplot(121)
feature_scores.sort_values(by='score', ascending=True).plot(y = 'score', x = 'feature', kind='barh', 
                                                            ax = f1, fontsize=10, grid=True) 

skb = SelectKBest(score_func=mutual_info_regression, k=80)
skb.fit(df, y_train)
select_features_kbest = skb.get_support()
kbest_MIR = cols[select_features_kbest]
scores = skb.scores_[select_features_kbest]
feature_scores = pd.DataFrame([(item, score) for item, score in zip(kbest_FR, scores)], columns=['feature', 'score'])
f2 = fig.add_subplot(122)
feature_scores.sort_values(by='score', ascending=True).plot(y = 'score', x = 'feature', kind='barh', 
                                                            ax = f2, fontsize=10, grid=True) 
plt.show()


# ### Select Features by Embedded Methods
# In addition to the return of the performance itself, some models has in their internal process some step to features select that best fit their proposal, and returns the features importance too. Thus, they provide two straightforward methods for feature selection and combine the qualities' of filter and wrapper methods. 
# ![image](https://paulbromford.files.wordpress.com/2018/02/c88e5e569aa7b412bff3f848ec9f7c53.gif)
# Some of the most popular examples of these methods are LASSO, RIDGE, SVM, Regularized trees, Memetic algorithm, and Random multinomial logit.
# 
# In the case of Random Forest, some other models base on trees, we have two basic approaches implemented in the packages:
# 1. Gini/Entropy Importance or Mean Decrease in Impurity (MDI)
# 2. Permutation Importance or Mean Decrease in Accuracy 
# 3. Permutation with Shadow Features
# 4. Gradient Boosting
# 
# Others models has concerns om **multicollinearity** problem and adding additional **constraints** or **penalty** to **regularize**. When there are multiple correlated features, as is the case with very many real life datasets, the model becomes unstable, meaning that small changes in the data can cause large changes in the model, making model interpretation very difficult on the regularization terms. 
# 
# This applies to regression models like LASSO and RIDGE. In classifier cases, you can use 'SGDClassifier' where you can set the loss parameter to 'log' for Logistic Regression or 'hinge' for 'SVM'. In 'SGDClassifier' you can set the penalty to either of 'l1', 'l2' or 'elasticnet' which is a combination of both.
# 
# Let's start with more details and examples:

# #### Feature Selection by Gradient Boosting
# The LightGBM model the importance is calculated from, if 'split', result contains numbers of times the feature is used in a model, if 'gain', result contains total gains of splits which use the feature.
# 
# On the [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html) model the importance is calculated by:
# - 'weight': the number of times a feature is used to split the data across all trees.
# - 'gain': the average gain across all splits the feature is used in.
# - 'cover': the average coverage across all splits the feature is used in.
# - 'total_gain': the total gain across all splits the feature is used in.
# - 'total_cover': the total coverage across all splits the feature is used in.
# 
# First measure is split-based and is very similar with the one given by for Gini Importance. But it doesn't take the number of samples into account.
# 
# The second measure is gain-based. It's basically the same as the Gini Importance implemented in R packages and in scikit-learn with Gini impurity replaced by the objective used by the gradient boosting model.
# 
# The cover, implemented exclusively in XGBoost, is counting the number of samples affected by the splits based on a feature.
# 
# get_score(fmap='', importance_type='weight')
# Get feature importance of each feature. Importance type can be defined as:
# 
# The default measure of both XGBoost and LightGBM is the split-based one. I think this measure will be problematic if there are one or two feature with strong signals and a few features with weak signals. The model will exploit the strong features in the first few trees and use the rest of the features to improve on the residuals. The strong features will look not as important as they actually are. While setting lower learning rate and early stopping should alleviate the problem, also checking gain-based measure may be a good idea.
# 
# Note that these measures are purely calculated using training data, so there's a chance that a split creates no improvement on the objective in the holdout set. This problem is more severe than in the random forest since gradient boosting models are more prone to over-fitting. 
# 
# Feature importance scores can be used for feature selection in scikit-learn.
# 
# This is done using the SelectFromModel class that takes a model and can transform a dataset into a subset with selected features.
# 
# This class can take a previous trained model, such as one trained on the entire training dataset. It can then use a threshold to decide which features to select. This threshold is used when you call the transform() method on the SelectFromModel instance to consistently select the same features on the training dataset and the test dataset.

# In[ ]:


warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# split data into train and test sets
X_train, X_test, y, y_test = train_test_split(df, y_train, test_size=0.30, random_state=101)

# fit model on all training data
#importance_type='gain'
model =  XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, max_delta_step=0, 
                      random_state=101, min_child_weight=1, missing=None, n_jobs=4,  
                      scale_pos_weight=1, seed=None, silent=True, subsample=1)


model.fit(X_train, y)
fig=plt.figure(figsize=(20,20))
ax = fig.add_subplot(121)
g = plot_importance(model, height=0.5, ax=ax)

# Using each unique importance as a threshold
thresholds = np.sort(np.unique(model.feature_importances_)) 
best = 1e36
colsbest = 31
my_model = model
threshold = 0

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model =  XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, max_delta_step=0, 
                                    random_state=101, min_child_weight=1, missing=None, n_jobs=4, 
                                    scale_pos_weight=1, seed=None, silent=True, subsample=1)
    selection_model.fit(select_X_train, y)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print("Thresh={:1.3f}, n={:d}, R2: {:2.2%} with MSE: {:.4f}".format(thresh, select_X_train.shape[1], r2, mse))
    if (best >= mse):
        best = mse
        colsbest = select_X_train.shape[1]
        my_model = selection_model
        threshold = thresh
        
ax = fig.add_subplot(122)
g = plot_importance(my_model,height=0.5, ax=ax, 
                    title='The best MSE: {:1.4f} with {:d} features'.\
                    format(best, colsbest))

feature_importances = [(score, feature) for score, feature in zip(model.feature_importances_, cols)]
XGBest = pd.DataFrame(sorted(sorted(feature_importances, reverse=True)[:colsbest]), columns=['Score', 'Feature'])
g = XGBest.plot(x='Feature', kind='barh', figsize=(20,10), fontsize=14, grid= True,
     title='Original feature importance from selected features')
plt.tight_layout(); plt.show()
XGBestCols = XGBest.iloc[:, 1].tolist()


# In[ ]:


bcols = set(pv_cols).union(set(RFEcv)).union(set(kbest_FR)).union(set(kbest_MIR)).union(set(XGBestCols)).union(set(SBS))
print('Features Selected by Filter Methods:\n')
print("Extra features select by Kbest_FR:", set(kbest_FR).\
     difference(set(pv_cols).union(set(RFEcv)).union(set(kbest_MIR)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by Kbest_MIR:", set(kbest_MIR).\
      difference(set(pv_cols).union(set(RFEcv)).union(set(kbest_FR)).union(set(XGBestCols)).union(set(SBS))), '\n')
print('_'*75,'\nFeatures Selected by Wrappers Methods:\n')
print("Extra features select by pv_cols:", set(pv_cols).\
      difference(set(SBS).union(set(RFEcv)).union(set(kbest_MIR)).union(set(kbest_FR)).union(set(XGBestCols))),'\n')
print("Extra features select by RFEcv:", set(RFEcv).\
      difference(set(pv_cols).union(set(kbest_FR)).union(set(kbest_MIR)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by SBS:", set(SBS).\
      difference(set(pv_cols).union(set(RFEcv)).union(set(kbest_MIR)).union(set(kbest_FR)).union(set(XGBestCols))), '\n')
print('_'*75,'\nFeatures Selected by Embedded Methods:\n')
print("Extra features select by XGBestCols:", set(XGBestCols).\
      difference(set(pv_cols).union(set(RFEcv)).union(set(kbest_MIR)).union(set(kbest_FR)).union(set(SBS))), '\n')
print('_'*75,'\nIntersection Features Selected:')
intersection = set(SBS).intersection(set(kbest_MIR)).intersection(set(RFEcv)).intersection(set(pv_cols)).\
               intersection(set(kbest_FR)).intersection(set(XGBestCols))
print(intersection, '\n')
print('_'*75,'\nUnion All Features Selected:')
print('Total number of features selected:', len(bcols))
print('\n{0:2d} features removed if use the union of selections: {1:}'.\
      format(len(cols.difference(bcols)), cols.difference(bcols)))


# ### Separate data for modeling
# ![image](https://frinkiac.com/gif/S09E09/266832/270803.gif?b64lines=IFlFUywgVEhFIE1PTkVZIElTIEdPT0QgQlVUCiBUSEUgQkVBVVRZIElTIFlPVSBHRVQgVE8KIFNUQVkgSU4gVEhFIEhPVVNFIFVOVElMCiBJVCdTIFNPTEQu)

# In[ ]:


totalCols = list(bcols.union(set(colsP)))
train = all_data.loc[all_data.SalePrice>0 , list(totalCols)].reset_index(drop=True, inplace=False)
y_train = all_data.SalePrice[all_data.SalePrice>0].reset_index(drop=True, inplace=False)
test = all_data.loc[all_data.SalePrice==0 , list(totalCols)].reset_index(drop=True, inplace=False)


# #### Feature Selection into the Pipeline
# Since we have a very different selection of features selection methods, from the results it may be interesting keeping only the removal of collinear and multicollinear, and can decide with we must have the pre polynomials and apply PCA or not. We can still improve the results through hyper parameterization and cross-validation.

# In[ ]:


class select_fetaures(object): # BaseEstimator, TransformerMixin, 
    def __init__(self, select_cols):
        self.select_cols_ = select_cols
    
    def fit(self, X, Y ):
        print('Recive {0:2d} features...'.format(X.shape[1]))
        return self

    def transform(self, X):
        print('Select {0:2d} features'.format(X.loc[:, self.select_cols_].shape[1]))
        return X.loc[:, self.select_cols_]    

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        df = self.transform(X)
        return df 
        #X.loc[:, self.select_cols_]    

    def __getitem__(self, x):
        return self.X[x], self.Y[x]
        


# ## Compressing Data via Dimensionality Reduction
# ![image](https://i.imgur.com/tLA1EhY.jpg)
# ### PCA
# **Principal component analysis** ([PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. If there are n observations with p variables, then the number of distinct principal components is `min(n-1,p)`. This transformation is defined in such a way that the first principal component has the largest possible variance, and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.
# ![image.png](http://vitarts3.hospedagemdesites.ws/wp-content/uploads/2018/09/DimRed.png)
# 
# Let's see how PCA can reduce the dimensionality of our dataset with minimum of lose information:

# In[ ]:


scale = RobustScaler() 
df = scale.fit_transform(train)

pca = PCA().fit(df) # whiten=True
print('With only 120 features: {:6.4%}'.format(sum(pca.explained_variance_ratio_[:120])),"%\n")

print('After PCA, {:3} features only not explained {:6.4%} of variance ratio from the original {:3}'.format(120,
                                                                                    (sum(pca.explained_variance_ratio_[120:])),
                                                                                    df.shape[1]))
del df,all_data


# ## Modeling 
# ![image](https://cdn-images-1.medium.com/max/1600/1*B29frkr87GXv70nrUGj7oQ.png)
# First, we start to looking at different approaches to implement linear regression models, and use hyper parametrization, cross validation and compare the results between different erros measures.
# 
# ### Model Hyper Parametrization
# #### Evaluate Results
# **Mean Squared Error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error))**
# 
# In statistics, MSE or mean squared deviation (MSD) of an estimator measures the average of the squares of the errors. MSE is a risk function, corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive (and not zero) is because of randomness or because the estimator does not account for information that could produce a more accurate estimate.
# ![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e)
# Which is simply the average value of the SSE cost function that we minimize to fit the linear regression model. The MSE is useful to for comparing different regression models or for tuning their parameters via a grid search and cross-validation.
# 
# **Root-Mean-Square Error ([RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation))**
# 
# The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values predicted by a model or an estimator and the values observed. 
# ![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/6d689379d70cd119e3a9ed3c8ae306cafa5d516d)
# 
# **Mean Absolute Error ([MAE](https://en.wikipedia.org/wiki/Mean_absolute_error))**
# 
# In statistics, mean absolute error (MAE) is a measure of difference between two continuous variables, is also the average horizontal distance between each point and the identity line. 
# ![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/3ef87b78a9af65e308cf4aa9acf6f203efbdeded)
# 
# **Coefficient of determination ( [R<sup>2</sup>](https://en.wikipedia.org/wiki/Coefficient_of_determination) )**
# ![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/0ab5cc13b206a34cc713e153b192f93b685fa875)
# Wheres:
# SSres is the sum of squares of residuals, also called the residual sum of squares: 
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/2669c9340581d55b274d3b8ea67a7deb2225510b)
# and SStot is the total sum of squares (proportional to the variance of the data):
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/aec2d91094ee54fbf0f7912d329706ff016ec1bd)
# Which can be understood as a standardized version of the MSE, for better [interpretability](https://www.youtube.com/watch?v=jXiLXjv02XY) of the model performance (try to say that tree times and faster!). In other words, R<sup>2</sup> is the fraction of response variance that is captured by the model. 
# 
# For the training dataset, R<sup>2</sup> is bounded between 0 and 1, but it can become negative for the test set. If R<sup>2</sup> =1 , the model fits the data perfectly with a corresponding MSE = 0 .

# In[ ]:


def get_results(model, name='NAN', log=False):
    
    rcols = ['Name','Model', 'BestParameters', 'Scorer', 'Index', 'BestScore', 'BestScoreStd', 'MeanScore', 
             'MeanScoreStd', 'Best']
    res = pd.DataFrame(columns=rcols)
    
    results = gs.cv_results_
    modelo = gs.best_estimator_

    scoring = {'MEA': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_mean_squared_error'}

    for scorer in sorted(scoring):
        best_index = np.nonzero(results['rank_test_%s' % scoring[scorer]] == 1)[0][0]
        if scorer == 'RMSE': 
            best = np.sqrt(-results['mean_test_%s' % scoring[scorer]][best_index])
            best_std = np.sqrt(results['std_test_%s' % scoring[scorer]][best_index])
            scormean = np.sqrt(-results['mean_test_%s' % scoring[scorer]].mean())
            stdmean = np.sqrt(results['std_test_%s' % scoring[scorer]].mean())
            if log:
                best = np.expm1(best)
                best_std = np.expm1(best_std)
                scormean = np.expm1(scormean)
                stdmean = np.expm1(stdmean)
        elif scorer == 'MEA':
            best = (-results['mean_test_%s' % scoring[scorer]][best_index])
            best_std = results['std_test_%s' % scoring[scorer]][best_index]
            scormean =(-results['mean_test_%s' % scoring[scorer]].mean())
            stdmean = results['std_test_%s' % scoring[scorer]].mean()
            if log:
                best = np.expm1(best)
                best_std = np.expm1(best_std)
                scormean = np.expm1(scormean)
                stdmean = np.expm1(stdmean)
        else:
            best = results['mean_test_%s' % scoring[scorer]][best_index]*100
            best_std = results['std_test_%s' % scoring[scorer]][best_index]*100
            scormean = results['mean_test_%s' % scoring[scorer]].mean()*100
            stdmean = results['std_test_%s' % scoring[scorer]].mean()*100
        
        r1 = pd.DataFrame([(name, modelo, gs.best_params_, scorer, best_index, best, best_std, scormean, 
                            stdmean, gs.best_score_)],
                          columns = rcols)
        res = res.append(r1)
        
    if log:
        bestscore = np.expm1(np.sqrt(-gs.best_score_))
    else:
        bestscore = np.sqrt(-gs.best_score_)
        
    print("Best Score: {:.6f}".format(bestscore))
    print('---------------------------------------')
    print('Best Parameters:')
    print(gs.best_params_)
    
    return res


# ### Residuals Plots
# The plot of differences or vertical distances between the actual and predicted values. Commonly used graphical analysis for diagnosing regression models to detect nonlinearity and outliers, and to check if the errors are randomly distributed.
# ![image.png](https://i1.wp.com/condor.depaul.edu/sjost/it223/documents/resid-plots.gif)
# Some points for help you in your analysis:
# - Since `Residual = Observed – Predicted` ***positive values*** for the residual (on the y-axis) mean the ***prediction was too low***, and ***negative values*** mean the ***prediction was too high***; 0 means the guess was exactly correct.
# 
# - ***They're pretty symmetrically distributed, tending to cluster towards the middle of the plot.***
# 
#     <p>For a good regression model, we would expect that the errors are randomly distributed and the residuals should be randomly scattered around the centerline. <p>
#     
# - ***Detect outliers, which are represented by the points with a large deviation from the centerline.***
# 
#     Now, you might be wondering how large a residual has to be before a data point should be flagged as being an outlier. The answer is not straightforward, since the magnitude of the residuals depends on the units of the response variable. That is, if your measurements are made in pounds, then the units of the residuals are in pounds. And, if your measurements are made in inches, then the units of the residuals are in inches. Therefore, there is no one "rule of thumb" that we can define to flag a residual as being exceptionally unusual.
# 
#     There's a solution to this problem. We can make the residuals **unitless**by dividing them by their standard deviation. In this way we create what are called **standardized residuals**. They tell us how many standard deviations above — if positive — or below — if negative — a data point is from the estimated regression line. <p>
#     
# - ***They're clustered around the lower single digits of the y-axis (e.g., 0.5 or 1.5, not 30 or 150).***
# 
#     Again, doesn't exist a unique rule for all cases. But, recall that the empirical rule tells us that, for data that are normally distributed, 95% of the measurements fall within 2 standard deviations of the mean. Therefore, any observations with a standardized residual greater than 2 or smaller than -2 might be flagged for further investigation. It is important to note that by using this "greater than 2, smaller than -2 rule," approximately 5% of the measurements in a data set will be flagged even though they are perfectly fine. It is in your best interest not to treat this rule of thumb as a cut-and-dried, believe-it-to-the-bone, hard-and fast rule! So, in most cases it may be more practical to investigate further any observations with a standardized residual greater than 3 or smaller than -3. Using the empirical rule we would expect only 0.2% of observations to fall into this category.<p>
#     
# - ***If we see patterns in a residual plot, it means that our model is unable to capture some explanatory information.***
# 
#     A special case is  any systematic (non-random) pattern. It is sufficient to suggest that the regression function is not linear. For example, if the residuals depart from 0 in some systematic manner, such as being positive for small x values, negative for medium x values, and positive again for large x values. <p>
#     
# - ***Non-constant error variance shows up on a residuals vs. fits (or predictor) plot in any of the following ways:***
#     - The plot has a "fanning" effect. That is, the residuals are close to 0 for small x values and are more spread out for large x values.
#     - The plot has a "funneling" effect. That is, the residuals are spread out for small x values and close to 0 for large x values.
#     - Or, the spread of the residuals in the residuals vs. fits plot varies in some complex fashion.

# In[ ]:


def resilduals_plots(lr, X, Y, log=False):
    y_pred = lr.predict(X)
    residual = pd.DataFrame()
    residual['Predict'] = y_pred
    residual['Residual'] = Y - y_pred
    residual['Predicted'] = np.expm1(residual.Predict)
    residual['StdResidual'] = np.expm1(Y) - residual.Predicted
    residual.StdResidual = residual.StdResidual / residual.StdResidual.std()
    residual['IDX'] = X.index
    
    if log:
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(121)
        g = sns.regplot(y='Residual', x='Predict', data = residual, order=1, ax = ax) 
        plt.xlabel('Log Predicted Values')
        plt.ylabel('Log Residuals')
        plt.hlines(y=0, xmin=min(Y)-1, xmax=max(Y)+1, lw=2, color='red')
        plt.xlim([min(Y)-1, max(Y)+1])

        ax = fig.add_subplot(122)
        g = sns.regplot(y='StdResidual', x='Predicted', data = residual, order=1, ax = ax) 
        plt.xlabel('Predicted Values')
        plt.ylabel('Standardized Residuals')
        plt.hlines(y=0, xmin=np.expm1(min(Y))-1, xmax=np.expm1(max(Y))+1, lw=2, color='red')
        plt.xlim([np.expm1(min(Y))-1, np.expm1(max(Y))+1])
    else:
        residual.StdResidual = residual.Residual / residual.Residual.std()
        residual.drop(['Residual', 'Predicted'], axis = 1, inplace=True)
        g = sns.regplot(y='StdResidual', x='Predict', data = residual, order=1) 
        plt.xlabel('Predicted Values')
        plt.ylabel('Standardized Residuals')
        plt.hlines(y=0, xmin=min(Y)-1, xmax=max(Y)+1, lw=2, color='red')
        plt.xlim([min(Y)-1, max(Y)+1])

    plt.show()  

    return residual


# ### Model Hiperparametrization
# #### Lasso (Least Absolute Shrinkage and Selection Operator)
# [Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) was introduced in order to improve the prediction accuracy and interpretability of regression models by include a with L1 prior as regularizer and altering the model fitting process to select only a subset of the provided covariates for use in the final model rather than using all of them. [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) was originally formulated for least squares models and this simple case reveals a substantial amount about the behavior of the estimator, including its relationship to ridge regression and best subset selection and the connections between Lasso coefficient estimates and so-called soft thresholding. It also reveals that the coefficient estimates need not be unique if covariates are collinear.
# 
# Prior to lasso, the most widely used method for choosing which covariates to include was stepwise selection, which only improves prediction accuracy in certain cases, such as when only a few covariates have a strong relationship with the outcome. However, in other cases, it can make prediction error worse. Also, at the time, ridge regression was the most popular technique for improving prediction accuracy. Ridge regression improves prediction error by shrinking large regression coefficients in order to reduce overfitting, but it does not perform covariate selection and therefore does not help to make the model more interpretable.
# 
# Lasso is able to achieve both of these goals by forcing the sum of the absolute value of the regression coefficients to be less than a fixed value, which depending on the regularization strength, certain weights can become zero, which makes the Lasso also useful as a supervised feature selection technique, by effectively choosing a simpler model that does not include those coefficients. However, a limitation of the Lasso is that it selects at most n variables if m > n.
# 
# This idea is similar to ridge regression, in which the sum of the squares of the coefficients is forced to be less than a fixed value, though in the case of ridge regression, this only shrinks the size of the coefficients, it does not set any of them to zero.
# 
# The optimization objective for Lasso is: `(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1`
# 
# Technically the Lasso model is optimizing the same objective function as the Elastic Net with `l1_ratio=1.0`,  no L2 penalty.
# 
# This characteristics turn Lasso a interesting alternative approach that can lead to sparse models.
# 
# From sklearn its most important parameters are: 
# - alpha: Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Given this, you should use the LinearRegression object.
# - max_iter: The maximum number of iterations
# - selection: If set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to 'random') often leads to significantly faster convergence especially when tol is higher than 1e-4.
# - tol: The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
# 
# So, I run many times my model, with different parameters, selection features, reduction or not and with and without log1P transformation of Sales Price. Below I preserve the code with the best options and with few possibilities for you can see the grid search cv in action, but I encourage you to make changes and see for yourself. 

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', Lasso(random_state = 101))]) 

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-5, len(SEL)-3, len(SEL)] 
whiten = [False, True]
max_iter = [5] #, 10, 100, 200, 300, 400, 500, 600]  
alpha = [0.0003, 0.0007, 0.0005, 0.05, 0.5, 1.0]
selection = ['random', 'cyclic'] 
tol = [2e-03, 0.003, 0.001, 0.0005]
param_grid =\
            dict(
                  model__alpha = alpha
                  ,model__max_iter = max_iter
                  ,model__selection = selection
                  ,model__tol = tol
                  ,pca__n_components = n_components
                  ,pca__whiten = whiten 
                ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error' #, iid=False
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)

lasso = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)), 
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

lasso.fit(train,y_train)

results = get_results(lasso, 'lasso Lg1', log=True)
display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
r = resilduals_plots(lasso, train, y_train, log=True)


# As you can see our Lasso has good performance with RFEcv selection features plus polynomials features, with: MAE 0.080, RMSE 0.1164 and R<sup>2</sup> of 92.36%. 
# 
# From the residuals plot with log sales price:
# We saw that most are plot randomly scattered around the centerline. You can think that some outliers that we didn't remove it can be detect! Are they the points with a large deviation from the centerline, the most extern points? However, notice that our deviation s no grater the 0.42 and no below to -0.8, they're clustered around the lower single digits of the y-axis. 
# 
# In general, there aren't any clear patterns, but with more attention we can observe some patterns in a few points, it means that our model is unable to capture some explanatory information, but as you can see, it is not easy to solve then. Maybe other model, or stack models or some feature that we drooped can help?
# 
# In addition, the log transformation is also used to handle cases where the expected distribution of the dependent variable leads to a funnel-like residue plot. Note that our residue plot without the log actually has a slight funnel look, but note that the model was trained and validated with the log transformation of the sales prices. So the transformation actually dealt with the problem of the distribution of the variable, but seems to have had little effect on the deviations of the residual
# 
# So, you could decide to cut some of these outliers, or simply go ahead and believe that your model is performing well and that the transformation in log1P of the selling price was successful. But calm, as you can see from the chart of errors on the right, things are not quite like that.
# 
# In fact I have seen many books and some colleagues do just that, turn the dependent variable into something like log, check for QQ testing that there has been improvement by reversing it to normal distribution, plotting errors and cutting or moving on believing it to be correct, but when you has applied a transformation on your response variable it is recommended that you reverse it when evaluating the errors, and this is what we did in the right chart.
# 
# So you can see that the residues have a pattern, curiously linear. and is more clear to see some outliers that really important to drop. But for now, let's ignore that we already know this, in order to show that if we cut some of the biggest deviations from the log observations perspective:

# In[ ]:


fica =  list(r.IDX[abs(r.Residual)<=0.3])
print('Outliers removed:', r.shape[0]-len(fica))
t = train.iloc[fica, :].reset_index(drop=True, inplace=False)
y_t = y_train.iloc[fica].reset_index(drop=True, inplace=False)

lasso.fit(t, y_t)
results = get_results(lasso, 'lasso Lg2', log=True)
display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
r = resilduals_plots(lasso, t, y_t, log=True)
del  t, y_t, fica


# We would have an improvement as you can see from the MAE with 0.072, RMSE is 0.094 and R<sup>2</sup> is 94.58%. However, our residual plot without log shows that this actually didn't had a good effect, where we continue have outliers, the linear pattern has a little increase and and the funnel shape intensified. 
# 
# So, let's see how it performs a model without transforming the sales price:

# In[ ]:


y = np.expm1(y_train)
lasso.fit(train, y)
results = get_results(lasso, 'lasso N1')
display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
r = resilduals_plots(lasso, train, y)


# You then see these results and then you are disappointed, so much discussion to make R<sup>2</sup> better only from 92.364 to 92.60% and didn't leave to have the funnel shape. But it confirm that we don't need transform our sales price. In the other hand, note that the residual graph does not have the slightly linear pattern and the funnel shape is more strangled. And more, now we can also apply the outliers detection rule.

# In[ ]:


fica =  list(r.IDX[abs(r.StdResidual)<3]) # =2.7
print('Outliers removed:', r.shape[0]-len(fica))
t = train.iloc[fica, :].reset_index(drop=True, inplace=False)
y_l = y_train.iloc[fica].reset_index(drop=True, inplace=False)
y_n = np.expm1(y_l)

lasso.fit(t, y_n)
results = get_results(lasso, 'lasso N2')
display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
r2 = resilduals_plots(lasso, t, y_n)
del fica, r2


# Note that although our R2 is not higher than what we get in the cut over the log observations, now you can see that the deletion of only 23 outliers made more sense, being more effective in improving the model and did not create any slightly linear pattern in residues, but it seems to widen a bit to the right of our funnel.
# 
# So you see, this confirms that this does not mean that anyone using log1p has failed, but shows that without the help of the residuals plot and the use of standardized metrics, it would be very difficult to identify these 23 outliers, more still decide to cut them, as well as require more tests to confirm the model.
# 
# So what we have just seen is another procedure for identifying and cutting outliers with the intention of improving the performance and generalization of the model. However, this procedure has to be taken care of; in addition, we might not have done it now, but the one left for a final stage where we would have already selected the model or built our stacked model. Thus, we do not run the risk of excluding observations that are outliers in this model, but which may be treated in another. Remember that the main outliers, the most damaging, had already been identified and eliminated in the EDA phase.
# 
# So at this point we will not cut any additional outlier, but we will not make use of the sales price transformation in your log1p, and thus avoid the linear pattern of the residuals.

# In[ ]:


y_log = y_train.copy()
y_train = np.expm1(y_train)

lasso.fit(train, y_train)

results = get_results(lasso, 'lasso', log=False)
display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
r = resilduals_plots(lasso, train, y_train, log=False)


# #### XGBRegressor
# [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html) is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, and Julia. It works on Linux, Windows, and macOS. From the project description, it aims to provide a "Scalable, Portable and Distributed Gradient Boosting (GBM, GBRT, GBDT) Library". Other than running on a single machine, it also supports the [distributed processing frameworks](http://vitarts.com.br/uma-introducao-ao-ciclo-de-vida-de-data-science-sobre-o-ecossistema-de-big-data/) Apache Hadoop, Apache Spark, and Apache Flink. It has gained much popularity and attention recently as it was the algorithm of choice for many winning teams of a number of machine learning competitions.
# 
# The XGBRegressor is a scikit-learn wrapper interface for running a regressor on XGBoost. Its most important parameters are:
# - max_depth: Maximum tree depth for base learners.
# - learning_rate: Boosting learning rate (the xgb's "eta")
# - n_estimators: Number of boosted trees to fit.
# - silent: Whether to print messages while running boosting.
# - objective: Specify the learning task and the corresponding learning objective or a custom objective function to be used.
# - booster: Specify which booster to use: 'gbtree', 'gblinear' or 'dart'.
# - n_jobs: Number of parallel threads used to run xgboost. 
# - gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
# - min_child_weight: Minimum sum of instance weight needed in a child.
# - max_delta_step: Maximum delta step we allow each tree's weight estimation to be.
# - subsample: Subsample ratio of the training instance.
# - colsample_bytree: Subsample ratio of columns when constructing each tree.
# - colsample_bylevel: Subsample ratio of columns for each split, in each level.
# - reg_alpha: L1 regularization term on weights.
# - reg_lambda: L2 regularization term on weights (xgb's lambda).
# - scale_pos_weight: Balancing of positive and negative weights.
# - base_score: The initial prediction score of all instances, global bias.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', XGBRegressor(random_state=101, silent=False))])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [90] # [len(SEL)-18, len(SEL)-19, len(SEL)-20] 
whiten = [True] #, False]
n_est = [3500] # [500, 750, 1000, 2000, 2006] # np.arange(1997, 2009, 3) # 
max_depth = [3] #, 4]
learning_rate = [0.01] #, 0.03] #, 0.1, 0.05
reg_lambda = [1] #0.1, 1e-06, 1e-04, 1e-03, 1e-02, 1e-05, 1, 0.0] 
reg_alpha= [1] # , 0.5, 1, 0.0]
booster = ['gblinear'] #'dart', 'gbtree']  
objective = ['reg:tweedie'] #, 'reg:linear', 'reg:gamma']

param_grid =\
            dict(
                  pca__n_components = n_components,
                  pca__whiten = whiten, 
                  model__n_estimators= n_est
                  ,model__booster = booster
                  ,model__objective = objective
                  ,model__learning_rate = learning_rate
                  ,model__reg_lambda = reg_lambda
                  ,model__reg_alpha = reg_alpha
                  ,model__max_depth = max_depth
                ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
XGBR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

XGBR.fit(train, y_train)

res = get_results(XGBR, 'XGBRegressor', log=False)
resilduals_plots(XGBR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# From the results, we can saw that XGB was better than Lasso, but it is much more computationally expensive, especially for hyper parameterization. 
# 
# From the residuals plots we can observe some linear pattern. Although this model has generated this unwanted residue pattern it seems to have been able to capture some nonlinear patterns. We will have to check if a stake model will be able to produce better results than the two models individually.

# ####  Gradient Boosting Regressor
# Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.
# 
# On the [sklearn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), in each stage a regression tree is fit on the negative gradient of the given loss function. Its most important parameters are:
# - loss: loss function to be optimized. 'ls' refers to least squares regression. 'lad' (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. 'huber' is a combination of the two. 'quantile' allows quantile regression (use alpha to specify the quantile).
# - learning_rate: learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
# - n_estimators: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# - criterion: The function to measure the quality of a split. Supported criteria are 'friedman_mse' for the mean squared error with improvement score by Friedman, 'mse' for mean squared error, and 'mae' for the mean absolute error. The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases.
# - min_samples_split: The minimum number of samples required to split an internal node:
# - min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
# - max_depth: maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
# - alpha: The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.
# - tol: Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', GradientBoostingRegressor(random_state=101))])

SEL = list(set(XGBestCols).union(set(colsP)))
# n_components = [len(SEL)] 
whiten = [True] #, False]
n_est = [3000]
learning_rate = [0.05] #, 0.01, 0.1, 0.005]
loss = ['huber'] #, 'ls', 'lad', 'quantile']
max_features = ['auto'] #, 'sqrt', 'log2']
max_depth = [3] #, 2] # , 5]
min_samples_split = [3] #, 4] 
min_samples_leaf = [3] # , 3, 2 ,4 ]
criterion = ['friedman_mse'] #, 'mse', 'mae']
alpha = [0.8] #, 0.75, 0.9, 0.7] 

param_grid =\
            dict(
                  #pca__n_components = n_components,
                  pca__whiten = whiten, 
                   model__n_estimators= n_est 
                  ,model__learning_rate = learning_rate
                  ,model__loss = loss
                  ,model__criterion = criterion
                  ,model__max_depth = max_depth
                  ,model__alpha = alpha
                  ,model__max_features = max_features
                  ,model__min_samples_split = min_samples_split
                  ,model__min_samples_leaf = min_samples_leaf
                   )

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                  , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                  ,cv=5, verbose=1, n_jobs=4)
 
GBR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

GBR.fit(train, y_train)
res = get_results(GBR, 'GBR' , log=False)
resilduals_plots(GBR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# We already have models with better performance than this one, so let's continue searching for another that can help more.

# #### ElasticNet
# In statistics and, in particular, in the fitting of linear or logistic regression models, the elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.
# The elastic net method overcomes the limitations of the Lasso method which uses a penalty function based on
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/5a188f4b162086fb06a4485f3336baefc22e18b3)
# Use of this penalty function has several limitations. For example, in the 'large p, small n' case (high-dimensional data with few examples, what has looked like this case), the Lasso selects at most n variables before it saturates. Also if there is a group of highly correlated variables, then the Lasso tends to select one variable from a group and ignore the others. To overcome these limitations, the elastic net adds a quadratic part to the penalty. The estimates from the elastic net method are defined by:
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/a66c7bfcf201d515eb71dd0aed5c8553ce990b6e)
# - Has a L1 penalty to generate sparsity. if we set l1_ratio to 1.0, the ElasticNet regressor would be equal to Lasso regression. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.
# - alpha: Constant that multiplies the penalty terms. 
# - Has a L2 penalty to overcome some of the limitations of the Lasso, such as the number of selected variables.
# - The quadratic penalty term makes the loss function strictly convex, and it therefore has a unique minimum. 
# 
# On [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), if you are interested in controlling the L1 and L2 penalty separately, keep in mind that this is equivalent to::
# 
#         a * L1 + b * L2    where:  alpha = a + b and l1_ratio = a / (a + b)
#         
# Its most important parameters are:
# - alpha: Constant that multiplies the penalty terms. Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter.``alpha = 0`` is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Given this, you should use the LinearRegression object.
# - l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
# - max_iter: The maximum number of iterations
# - selection: If set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to 'random') often leads to significantly faster convergence especially when tol is higher than 1e-4.
# - tol: The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', ElasticNet(random_state=101))])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-5, len(SEL)-3, len(SEL)] 
whiten = [False] #, True]
max_iter = [5] #, 100] 
alpha = [1e-05] #, 0.001, 0.01, 0.003, 0.00001] 
l1_ratio =  [0.00003] 
selection = ['cyclic'] #, 'random', 'cyclic']

param_grid =\
            dict(
                  model__max_iter= max_iter
                  ,pca__n_components = n_components
                  ,pca__whiten = whiten 
                  ,model__alpha = alpha
                  ,model__l1_ratio = l1_ratio
                  ,model__selection = selection
               ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
ELA = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

ELA.fit(train, y_train)

res = get_results(ELA, 'ELA', log=False)
resilduals_plots(ELA, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# This model looks promising and have good computational performance. Again we use PCA for performance improvement not for dimension reduction.

# #### Bayesian Ridge Regression
# Ridge regression is an L2 penalized model where we simply add the squared sum of the weights to our least-squares cost function:
# ![image](https://cdn-images-1.medium.com/max/1200/1*jgWOhDiGjVp-NCSPa5abmg.png)
# By increasing the value of the hyper parameter λ , we increase the regularization strength and shrink the weights of our model. Please note that we don't regularize the intercept term ß<sub>0,/sub>.
# 
# Bayesian ridge regression fit a Bayesian ridge model and optimize the regularization parameters lambda (precision of the weights) and alpha (precision of the noise).
# 
# The advantages of Bayesian Regression are:
# - It adapts to the data at hand.
# - It can be used to include regularization parameters in the estimation procedure.
# The disadvantages of Bayesian regression include:
# - Inference of the model can be time consuming.
# 
# The main parameter on [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge) are:
# - n_iter: Maximum number of iterations. Default is 300.
# - tol: Stop the algorithm if w has converged. Default is 1.e-3.
# - alpha_1: Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Default is 1.e-6
# - alpha_2: Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Default is 1.e-6.
# - lambda_1: Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Default is 1.e-6.
# - lambda_2: parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Default is 1.e-6

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', BayesianRidge())]) #compute_score=False, fit_intercept=True, normalize=False

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-9] #, len(SEL)-8, len(SEL)-7] 
whiten = [True] # , False]
n_iter=  [36] # np.arange(36, 45) # [40, 35, 45, 70, 100, 200, 300, 500, 700, 1000] #  
alpha_1 = [1e-06] #0.1, 1e-04, 1e-03, 1e-02, 1e-05]
alpha_2 = [0.1] # 1e-06 , , 1e-02, 1e-04, 1e-03]
lambda_1 = [0.001] # 0.1, 1e-06, 1e-04, 1e-02, 1e-05] 
lambda_2 = [0.01] # 0.1, 1e-06, 1e-04, 1e-03, 1e-05]

param_grid =\
            dict(
                   model__n_iter = n_iter
                  ,model__alpha_1 = alpha_1
                  ,model__alpha_2 = alpha_2
                  ,model__lambda_1 = lambda_1
                  ,model__lambda_2 = lambda_2
                  ,pca__n_components = n_components
                  ,pca__whiten = whiten 
              ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
BayR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

BayR.fit(train, y_train)
res = get_results(BayR, 'BayR', log=False)
resilduals_plots(BayR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# Other good model and without linear pattern on the residuals.

# #### Linear Regression
# In statistics, ordinary least squares (OLS) is a type of linear least squares method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by the principle of least squares: minimizing the sum of the squares of the differences between the observed dependent variable (values of the variable being predicted) in the given dataset and those predicted by the linear function.
# 
# On sklearn the ordinary least squares are implemented as [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), and its main important parameters are:
# - fit_intercept: whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).
# - normalize: This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', LinearRegression())])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-10, len(SEL)-11, len(SEL)-9] 
whiten = [True, False]

param_grid =\
            dict(
                   pca__n_components = n_components,
                   pca__whiten = whiten
               ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
LR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

LR.fit(train, y_train)

res = get_results(LR, 'LR', log=False)
resilduals_plots(LR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# As you can see, this model performs very well. This is expected, since we work on eliminating the problems of collinearity, multicollinearity and maximization of significance. Of course, when we do this, we take care of the properties required by linear regressions, and try give some flexibility to the model to identify other patterns by the inclusion of some polynomials.

# #### Orthogonal Matching Pursuit model (OMP)
# OMP is based on a greedy algorithm that includes at each step the atom most highly correlated with the current residual. It is similar to the simpler matching pursuit (MP) method, but better in that at each iteration, the residual is recomputed using an orthogonal projection on the space of the previously chosen dictionary elements.
# 
# On [skelearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html) the main parameters are:
# n_nonzero_coefs: Desired number of non-zero entries in the solution. If None (by default) this value is set to 10% of n_features.
# tol: Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', OrthogonalMatchingPursuit())])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [100] # [len(SEL)-11, len(SEL)-10, len(SEL)-9] 
whiten = [False]
tol = [5e-05] # [None, 0.00005, 0.0001, 0.00000, 0.002]

param_grid =\
            dict(
                   model__tol = tol
                   ,model__n_nonzero_coefs = [2] # range(2, 6) # [10, 20, 30, 40, 50, 60, 70, 80, 90, None] # 
                   ,pca__n_components = n_components
                   ,pca__whiten = whiten
                   ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
ORT = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

ORT.fit(train, y_train)
res = get_results(ORT, 'ORT', log=False)
resilduals_plots(ORT, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# As can be seen, we have achieved that this model performance equated to LR and close to ELA and Lasso.

# #### Robust Regressor
# In robust statistics, robust regression is a form of regression analysis designed to overcome some limitations of traditional parametric and non-parametric methods. Regression analysis seeks to find the relationship between one or more independent variables and a dependent variable. Certain widely used methods of regression, such as ordinary least squares, have favourable properties if their underlying assumptions are true, but can give misleading results if those assumptions are not true; thus ordinary least squares is said to be not robust to violations of its assumptions. Robust regression methods are designed to be not overly affected by violations of assumptions by the underlying data-generating process.
# 
# In particular, least squares estimates for regression models are highly sensitive to (i.e. not robust against) outliers. While there is no precise definition of an outlier, outliers are observations which do not follow the pattern of the other observations. This is not normally a problem if the outlier is simply an extreme observation drawn from the tail of a normal distribution, but if the outlier results from non-normal measurement error or some other violation of standard ordinary least squares assumptions, then it compromises the validity of the regression results if a non-robust regression technique is used.
# 
# From the sklearn the [HuberRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor) implements a robust regressor strategy. The HuberRegressor is different to Ridge because it applies a linear loss to samples that are classified as outliers. A sample is classified as an inlier if the absolute error of that sample is lesser than a certain threshold. It differs from TheilSenRegressor and RANSACRegressor because it does not ignore the effect of the outliers but gives a lesser weight to them.
# 
# Its main parameters are:
# - epsilon: The parameter epsilon controls the number of samples that should be classified as outliers. The smaller the epsilon, the more robust it is to outliers.
# - max_iter: Maximum number of iterations that scipy.optimize.fmin_l_bfgs_b should run for.
# - alpha: Regularization parameter.
# - tol: The iteration will stop when `max{|proj g_i | i = 1, ..., n} <= tol` where pg_i is the i-th component of the projected gradient.

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', HuberRegressor())])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-9] #, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
whiten = [True] #, False]
max_iter = [2000] 
alpha = [0.0001] #, 5e-05, 0.01, 0.00005, 0.0005, 0.5, 0.001] 
epsilon = [1.005] #, 1.05, 1.01, 1.001] 
tol = [1e-01, 1e-02] #, 2e-01, 3e-01, 4e-01, 5e-01, 6e-01] 

param_grid =\
            dict(
                  model__max_iter= max_iter
                  ,pca__n_components = n_components
                  ,pca__whiten = whiten 
                  ,model__alpha = alpha
                  ,model__epsilon = epsilon
                  ,model__tol = tol
               ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=3)
 
Hub = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

Hub.fit(train, y_train)

res = get_results(Hub, 'Hub', log=False)
resilduals_plots(Hub, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# This model had a good performance, but some linear pattern of escape given the most scattered points on the top right.

# #### Passive Aggressive Regressor
# The passive-aggressive algorithms are a family of algorithms for large-scale learning. They are similar to the Perceptron in that they do not require a learning rate. However, contrary to the Perceptron, they include a regularization parameter C.
# 
# For regression, [PassiveAggressiveRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor) can be used with loss='epsilon_insensitive' (PA-I) or loss='squared_epsilon_insensitive' (PA-II).
# 
# The main parameters are:
# - C: Maximum step size (regularization). Defaults to 1.0.
# - max_iter: The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit.  Defaults to 1000 from 0.21 version, or if tol is not None.
# - tol: The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to 1e-3 from 0.21 version.
# - epsilon: If the difference between the current prediction and the correct label is below this threshold, the model is not updated.
# - n_iter_no_change: Number of iterations with no improvement to wait before early stopping. Default=5

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', PassiveAggressiveRegressor(random_state = 101))])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-9, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
whiten = [True] #, False]
loss = ['squared_epsilon_insensitive'] #, 'epsilon_insensitive']
C = [0.001] #, 0.005, 0.003]
max_iter = [1000] 
epsilon = [0.00001] # , 0.00005
tol = [1e-03] #, 1e-05,1e-02, 1e-01, 1e-04, 1e-06]

param_grid =\
            dict(
                  pca__n_components = n_components,
                  pca__whiten = whiten, 
                  model__loss = loss
                  ,model__epsilon = epsilon
                  ,model__C = C
                  ,model__tol = tol
                  ,model__max_iter = max_iter
               ) 
 
gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
PassR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

PassR.fit(train, y_train)
res = get_results(PassR, 'PassR', log=False)
resilduals_plots(PassR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# #### SGD Regressor
# Linear model fitted by minimizing a regularized empirical loss with SGD
# 
# Stochastic gradient descent (often shortened to SGD), also known as incremental gradient descent, is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization. The gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
# 
# The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.
# 
# This implementation works with data represented as dense numpy arrays of floating point values for the features.
# 
# The main [parameters](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) are:
# - loss: The loss function to be used. The 'squared_loss' refers to the ordinary least squares fit. 'huber' modifies 'squared_loss' to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is linear past that; this is the loss function used in SVR. 'squared_epsilon_insensitive' is the same but becomes squared loss past a tolerance of epsilon.
# - penalty: The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'.
# - alpha: Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to 'optimal'.
# - l1_ratio: The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
# - max_iter: The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit. Defaults to 1000 from 0.21 version, or if tol is not None.
# - tol: The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to 1e-3 from 0.21 version.
# - epsilon: Epsilon in the epsilon-insensitive loss functions; only if loss is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'. For 'huber', determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
# - eta0: The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules. The default value is 0.0 as eta0 is not used by the default schedule 'optimal'.
# - power_t:The exponent for inverse scaling learning rate. Default 0.5.
# - learning_rate: The learning rate schedule:
#   - 'constant': eta = eta0
#   - 'optimal': eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
#   - 'invscaling': eta = eta0 / pow(t, power_t)
#   - 'adaptive': eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.
# 

# In[ ]:


model = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('model', SGDRegressor(random_state = 101))])

SEL = list(set(RFEcv).union(set(colsP)))
n_components = [len(SEL)-9, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
whiten = [True] #, False]
loss = ['squared_loss'] #, 'huber', 'squared_epsilon_insensitive', 'epsilon_insensitive']
penalty = ['l2'] #, 'elasticnet', 'l1']
l1_ratio = [0.7] #, 0.8] #[0.2, 0.5, 0.03]
learning_rate = ['invscaling'] #, 'constant', 'optimal']
alpha = [0.001] # [1e-01, 1e-2, 1e-03, 1e-4, 1e-05]
epsilon =  [1e-01] #, 1e-2, 1e-03, 1e-4, 1e-05]
tol = [0.001] #, 0.003] 
eta0 = [0.01] #, 1e-1, 1e-03, 1e-4, 1e-05] 
power_t = [0.5]
 
param_grid =\
            dict(
                   pca__n_components = n_components
                   ,pca__whiten = whiten, 
                   model__penalty = penalty
                   ,model__l1_ratio = l1_ratio
                   ,model__loss = loss
                   ,model__alpha = alpha
                   ,model__epsilon = epsilon
                   ,model__tol = tol
                   ,model__eta0 = eta0
                   ,model__power_t = power_t
                   ,model__learning_rate = learning_rate
               ) 

gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
                   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
                   ,cv=5, verbose=1, n_jobs=4)
 
SGDR = Pipeline([
        ('sel', select_fetaures(select_cols=SEL)),
        ('scl', RobustScaler()),
        ('gs', gs)
 ])

SGDR.fit(train, y_train)
res = get_results(SGDR, 'SGDR', log=False)
resilduals_plots(SGDR, train, y_train, log=False)
results = pd.concat([results, res], axis=0)
res.loc[:, 'Scorer' : 'MeanScoreStd']


# ## Check the best results from the models hyper parametrization

# In[ ]:


results.loc[results.Scorer=='RMSE', ['Name','BestScore', 'BestScoreStd']].sort_values(by='BestScore', ascending=True)


# ## Stacking the Models

# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)   

#defining RMSLE evaluation function
def RMSLE (y, y_pred):
    return (np.sqrt(mean_squared_error(y, y_pred)))

# Averaged base models score
averaged_models = AveragingModels(models = (XGBR, BayR, PassR)) # Hub, ELA,  lasso, ARDR, LGBM, GBR

averaged_models.fit(train, y_train) 
stacked_train_pred = averaged_models.predict(train)

stacked_pred = (averaged_models.predict(test))
rmsle = RMSLE(y_train,stacked_train_pred)

print('RMSLE score on the train data: {:.4f}'.format(rmsle))

print('Accuracy score: {:.6%}'.format(averaged_models.score(train, y_train)))


# ## Create Submission File:

# In[ ]:


# Preper Submission File
ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
submit['SalePrice'] = ensemble
# ----------------------------- Create File to Submit --------------------------------
submit.to_csv('SalePrice_N_submission.csv', index = False)
submit.head()


# ## Conclusion
# 
# 
# As we can see, through a method we were able to generate models that could present good generalization and better performance when combined.
# 
# The big challenge was the proper detection and decision-cutting of outliers, the reassessment of noise-generating features, and the combined combination of selection and data engineering strategies.
# 
# We can put into practice a great number of techniques and methods, from EDA to the generation of stacked models, covering a broad conceptual and practical expectation as desired.
# 
# If you run this kernel as it is and submit the file, you will be able to be among the top 34.7% with 0.12559 points, but as you noted, we had to make decisions that impacted the performance better or worse depending on the characteristics of each model . In this sense, I invite you to download this notebook and practice it, send your suggestions and comments, knowing that it is possible to be among the top 16%.
# 
# For next steps, I suggest:
# - Applies it to a deep learning model like TensorFlow
# - Try a Random Forest Regressor, with and without the transformations like box cox and without polynomials.
# - Try eliminate outliers only through the residuals
# - Try find a good model with Sales price on log1p transformation
# - Run some models without polynomials and try other combinations of polynomials, with more and less features
# - Apply BboxCox on others features, and try boxcox1p with different lambdas
# - Some categorical data has better performance if you ignore that it is categorical. Since this is a wrong approach, you need find than empirical and try give some meaning, if you really can.
# 
# Thank you for your attention so far, I also ask you to share and accept any comments and feedback.
# 
# My best regards and good luck.
