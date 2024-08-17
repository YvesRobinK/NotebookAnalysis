#!/usr/bin/env python
# coding: utf-8

# # Advanced House Pricing Prediction: Feature Importance Case Study

# In[ ]:


get_ipython().system('pip install pdpipe')


# In[ ]:


# Loading neccesary packages

import numpy as np
import pandas as pd
import pdpipe as pdp

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

#

from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p

#

from typing import Dict, List, Tuple, Sequence


# # Feature Importance in the Age of Interpretable AI/ML
# 
# In the recent years, we see increasing demand in interpretable AI/ML. Human decision-makers would like to trust their AI-based decision support on the ground of rationale rather than via religious belief in what AI system calculates/suggests/forecasts.
# 
# It was quite easy in good old days when the stage was preoccupied by the easily interpretable ML algorithms (like Linear regression, polynomial regression, logistic regression, CART-based decision trees etc.).
# 
# Unfortunately such algorithms lacked accuracy in many real-world business forecasting and decision support scenarios. It resulted in the advent of highly accurate and complicated algorithms (starting from Random Forests through Gradient Boosting Machine-like algorithms up to the wide spectrum of Neural Networks of the second generation). 
# 
# As everything in this world, however, the accuracy came at a price. There was no more easy way to interpret the decision-making flow of such AI/ML algorithms in a human-friendly and rational way.
# 
# One of the early attempts to address the challenge was adding the supplementary capability to calculate the feature importance scores by some of the modern algorithms (this is featured, for instance, by Random Forest, GBM, or lightgbm).
# 
# The poor side of the feature importance scores is that they are sometimes confusing / misleading due to the fact they are calculated separately from the ML model training itself.
# 
# Such a collision gave a birth to several analytical algorithms to calculate the feature importance / do the feature selection for ML models. As opposed to the classical statistical (filtering) approaches (where feature importance is determined on a basis of a certain statistical metric, whether it is a Pierson correlation or Chi Square), such techniques embrace a series of model training experiments under certain feature space tweaks. In such a way, they relatively score the importance of each feature for a specific model to train.
# 
# In this notebook, we are going to review and implement such feature selection / feature importance detection methods. They all will be useful in the strive to build the industrial culture of interpretable AI/ML. 
# 
# With this direction, we are on a par with the industry giant like Google (who recently launched the services of Explainable AI – see https://cloud.google.com/explainable-ai).
# 
# **Note**: in addition to building comprehensive interpretable ML models, the relevant feature selection will also help you to handle other ML challenges as follows
# - Dropping the garbage (non-informative) features from your modelling pipeline
# - Tackling the curse of dimensionality as well as minimizing the impact of the model overfitting
# - Improve accuracy/performance of our model forecasting
# 
# Now, we are ready to get our hands dirty in the code, to orchestrate a solid ML solution that benefits from industrial feature selection techniques.
# 

# # Load Initial Data

# In[ ]:


def get_data_file_path(in_kaggle: bool) -> Tuple[str, str]:
    train_set_path = ''
    test_set_path = ''
    
    if in_kaggle:
        # running in Kaggle, inside 
        # 'House Prices: Advanced Regression Techniques' competition kernel container
        train_set_path = '../input/house-prices-advanced-regression-techniques/train.csv'
        test_set_path = '../input/house-prices-advanced-regression-techniques/test.csv'
    else:
        # running locally
        train_set_path = 'data/train.csv'
        test_set_path = 'data/test.csv'
    
    return train_set_path,test_set_path


# In[ ]:


# Loading datasets
in_kaggle = True
train_set_path, test_set_path = get_data_file_path(in_kaggle)

train = pd.read_csv(train_set_path)
test = pd.read_csv(test_set_path)


# In[ ]:


# check train dimension
display(train.shape)


# In[ ]:


# check test dimension
display(test.shape)


# # Initial Data Transformation: Dropping Id col
# 
# Starting this section and down the activities in **'Setting Model Data and Log Transforming the Target'** section below, we are going to reuse the feature enginering steps defined, justified and explained in https://www.kaggle.com/datafan07/beginner-eda-with-feature-eng-and-blending-models

# In[ ]:


# dropping unneccessary columns, merging training and test sets

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# # Dropping Outliers From Training Set

# In[ ]:


# Dropping outliers after detecting them by eye

train = train.drop(train[(train['OverallQual'] < 5)
                                  & (train['SalePrice'] > 200000)].index)
train = train.drop(train[(train['GrLivArea'] > 4000)
                                  & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['GarageArea'] > 1200)
                                  & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['TotalBsmtSF'] > 3000)
                                  & (train['SalePrice'] > 320000)].index)
train = train.drop(train[(train['1stFlrSF'] < 3000)
                                  & (train['SalePrice'] > 600000)].index)
train = train.drop(train[(train['1stFlrSF'] > 3000)
                                  & (train['SalePrice'] < 200000)].index)


# # Merging Train and Test Sets

# In[ ]:


# Backing up target variables and dropping them from train data

y = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Merging features

features = pd.concat([train_features, test_features]).reset_index(drop=True)
print(features.shape)


# # Imputing Missing Values

# In[ ]:


# List of NaN including columns where NaN's mean none.
none_cols = [
    'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

# List of NaN including columns where NaN's mean 0.

zero_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
]

# List of NaN including columns where NaN's actually missing gonna replaced with mode.

freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
    'SaleType', 'Utilities'
]

# Filling the list of columns above:

for col in zero_cols:
    features[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    features[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    features[col].replace(np.nan, features[col].mode()[0], inplace=True)
    
# Filling MSZoning according to MSSubClass
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

# Filling LotFrontage according to Neighborhood
features['LotFrontage'] = features.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# # Numeric Features to be Treated as Categories
# 
# Below we cast some numeric features to categories based on the logical assessment of the feature essence

# In[ ]:


# Features which numerical on data but should be treated as category.
features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)


# # New Feature Engineering

# ## Binning the Rare Category Values

# In[ ]:


# Transforming rare values(less than 10) into one group - dimensionality reduction

others = [
    'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'Heating', 'Electrical', 'Functional', 'SaleType'
]

for col in others:
    mask = features[col].isin(
        features[col].value_counts()[features[col].value_counts() < 10].index)
    features[col][mask] = 'Other'


# ##  Converting Some Categorical Variables to Numeric Ones

# In[ ]:


# Converting some of the categorical values to numeric ones.

neigh_map = {
    'MeadowV': 1,
    'IDOTRR': 1,
    'BrDale': 1,
    'BrkSide': 2,
    'OldTown': 2,
    'Edwards': 2,
    'Sawyer': 3,
    'Blueste': 3,
    'SWISU': 3,
    'NPkVill': 3,
    'NAmes': 3,
    'Mitchel': 4,
    'SawyerW': 5,
    'NWAmes': 5,
    'Gilbert': 5,
    'Blmngtn': 5,
    'CollgCr': 5,
    'ClearCr': 6,
    'Crawfor': 6,
    'Veenker': 7,
    'Somerst': 7,
    'Timber': 8,
    'StoneBr': 9,
    'NridgHt': 10,
    'NoRidge': 10
}

features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype('int')
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')
features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')
bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')
features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')
bsmf_map = {
    'None': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')
features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')
heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')
features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')
features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')
features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')
features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')


# ## Creating New Features

# In[ ]:


# Creating new features  based on previous observations

features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                       features['1stFlrSF'] + features['2ndFlrSF'])
features['TotalBathrooms'] = (features['FullBath'] +
                              (0.5 * features['HalfBath']) +
                              features['BsmtFullBath'] +
                              (0.5 * features['BsmtHalfBath']))

features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                            features['EnclosedPorch'] +
                            features['ScreenPorch'] + features['WoodDeckSF'])

features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])

# Merging quality and conditions

features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])
features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] +
                            features['BsmtFinType1'] +
                            features['BsmtFinType2'])
features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])
features['TotalQual'] = features['OverallQual'] + features[
    'TotalExtQual'] + features['TotalBsmQual'] + features[
        'TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']

# Creating new features by using new quality indicators

features['QualGr'] = features['TotalQual'] * features['GrLivArea']
features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +
                                                  features['BsmtFinSF2'])
features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']
features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']
features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']
features['QlLivArea'] = (features['GrLivArea'] -
                         features['LowQualFinSF']) * (features['TotalQual'])
features['QualSFNg'] = features['QualGr'] * features['Neighborhood']


# In[ ]:


# Creating some simple features

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)

features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)

features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)


# # Transforming The Skewed Features

# In[ ]:


possible_skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal'
]

# Finding skewness of the numerical features

skew_features = np.abs(features[possible_skewed].apply(lambda x: skew(x)).sort_values(
    ascending=False))

# Filtering skewed features

high_skew = skew_features[skew_features > 0.3]

# Taking indexes of high skew

skew_index = high_skew.index

# Applying boxcox transformation to fix skewness

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))


# # Dropping Unnecessary Features

# In[ ]:


# Features to drop

to_drop = [
    'Utilities',
    'PoolQC',
    'YrSold',
    'MoSold',
    'ExterQual',
    'BsmtQual',
    'GarageQual',
    'KitchenQual',
    'HeatingQC',
]

# Dropping ML-irrelevant features

features.drop(columns=to_drop, inplace=True)


# # Label Encoding The Categorical Variables

# In[ ]:


# Getting dummy variables for ategorical data
features = pd.get_dummies(data=features)


# # Final Check of The Data Before Feature Selection and ML Experiments

# In[ ]:


print(f'Number of missing values: {features.isna().sum().sum()}')


# In[ ]:


features.shape


# #  Separating Train and Test Sets Again

# In[ ]:


# Separating train and test set

train = features.iloc[:len(y), :]
test = features.iloc[len(train):, :]


# # Setting Model Data and Log Transforming the Target

# In[ ]:


# Setting model data

X = train
X_test = test
y = np.log1p(y)


# In[ ]:





# # What’s In Your Toolbox?
# 
# Before we jump on the practical implementation of instrumental Feature Importance calculation and Feature selection techniques, I would like to briefly outline the essence of each of the methods we are going to use down the road.
# 
# In real machine-learning problems, tackling the curse of dimensionality as well as increasing the model interpretability are translated to solid feature selection techniques.
# 
# Apart from the filtering-based methods (correlation- or chi-square-based) or PCA (the latter is mostly applicable to linear regression problems), there is a set of analytical (computational) feature selection methods. 
# 
# In the subsections below, we will review the available analytical feature selection options, along with the technical implementation details for them in Python.
# 
# ## Wrapper-based methods
# Wrapper-based methods treat the selection of a set of features as a search problem. 
# 
# **RFE** and its implementation in sckit-learn can be referred to as one of the good options to go on with it, for example. 
# The utility function to benefit from RFE feature importance score calculations is provided below
# 
# https://gist.github.com/gvyshnya/7349198e74b4c5fc6caad18ac150ff07
# 
# Other options in the wrapper-based feature selection domain are Backward Elimination, Forward Selection, and Bidirectional Elimination.
# 
# ## Embedded methods
# Embedded methods are applicable to ML algorithms that have built-in feature selection methods. For instance, **Lasso, Random Forest, or lightgbm** have them.
# 
# From the technical standpoint, feature selection with the embedded methods relies on scikit-learn’s SelectFromModel. You can see it in action with the demo code snippet below
# 
# https://gist.github.com/gvyshnya/de775c04f7f752eb66c1d40ed40bcb05
# 
# ## Permutation method
# 
# Put simply, this method changes the data in a column and then tests how much that affects the model accuracy. If changing the data in a column drops the accuracy, then that column is assumed to be important.
# 
# You can benefit from the out-of-the box scikit-learn utility to facilitate permutable feature importance calculation. I wrapped it up in a utility function below
# https://gist.github.com/gvyshnya/abe6c06767922f8762bd288c2d897ce5
# 
# ## Drop-Column Method
# 
# This method is focused on measuring performance of a ML model on a full-feature dataset of predictors vs. the set of smaller datasets (each one of them to drop exactly one feature from the original fully featured dataset). In such a way, the difference between the performance of the model on the original dataset and every dataset with a dropped feature will be a metric of the dropped feature importance for the model performance.
# 
# I could not find the stable Pythonic implementation of such a feature selection / feature importance measurement therefore I ended up with the custom implementation below
# 
# https://gist.github.com/gvyshnya/513080f611491b8baa08cc1bf6987144
# 
# ## Several Practical Considerations
# When deciding which feature selection method is the best one for your specific problem, you should keep in mind the points below
# - there is no a single silver bullet-proof method of feature selection that worsks well for each and every project - typically, you will always have to undertake several feature selection experiments (using different methods) to figure out which one leads to the ML model with the highest performance metric score
# - all methods except filter-based ones have its computational time tall, and it may take too much time to go through them appropriately for large datasets
# - embedded methods sometimes introduce the confusion (or even a misinterpretation on the feature importance) as the feature importance scores are often calculated separately from the model training
# 
# With that said, let's get back to coding.
# 

# # Modelling and Feature Selection Pre-Requisite

# In[ ]:


# Loading neccesary packages for modelling and feature selection
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Setting kfold for future use
kf = KFold(10, random_state=42, shuffle=True)

# Train our baseline RF Regression model for feature importance scoring/feature selection
n_trees = 100
rf = RandomForestRegressor(n_jobs=-1, n_estimators=n_trees, verbose=1)
rf.fit(X, y)


# In[ ]:


def rfe_select_featurs(X, y, estimator, num_features) -> List[str]:
    rfe_selector = RFE(estimator=estimator, 
                       n_features_to_select=num_features, 
                       step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    
    return rfe_feature


# In[ ]:


# total list of features
colnames = X.columns
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# # Feature Importance Scores From Model and via RFE

# In[ ]:


# Do FRE feature importance scoring - 
# stop the search when only the last feature is left
rfe = RFE(rf, n_features_to_select=1, verbose =3 )
rfe.fit(X, y)
ranks["RFE_RF"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


# In[ ]:


# Extract feature importance coefficients as calculated by the trained model
ranks["RF"] = ranking(rf.feature_importances_, colnames);


# # Creating the Feature Importance Dataframe
# 
# Let's build the initial version of the dataframe to collect feature importance scores calculated by different methods.
# 
# Initially, it will contain the RFE scores as well as the scores calculated by the model directly.
# 
# In future, we will extend it with the additional metrics (permutational feature importance, and drop-column feature importance).

# In[ ]:


# all ranks
# Put the mean scores into a Pandas dataframe
rfe_rf_df = pd.DataFrame(list(ranks['RFE_RF'].items()), columns= ['Feature','rfe_importance'])
rf_df = pd.DataFrame(list(ranks['RF'].items()), columns= ['Feature','alg_importance'])

all_ranks = pd.merge(rfe_rf_df, rf_df, on=['Feature'])

display(all_ranks.head(10))


# # Embedded Feature Selection: Selecting Features From a Model

# In[ ]:


from sklearn.feature_selection import SelectFromModel

embeded_rf_selector = SelectFromModel(rf, max_features=200)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[ ]:


print(embeded_rf_feature)


# # Permutable Feature Importance

# In[ ]:


from sklearn.inspection import permutation_importance


# Here's how you use permutation importance
def get_permutation_importance(X, y, model) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=1,
                                random_state=0)
    
    # permutational importance results
    result_df = pd.DataFrame(colnames,  columns=['Feature'])
    result_df['permutation_importance'] = result.get('importances')
    
    return result_df


# In[ ]:


permutate_df = get_permutation_importance(X, y, rf)
permutate_df.sort_values('permutation_importance', 
                   ascending=False)[
                                    ['Feature','permutation_importance'
                                    ]
                                  ][:30].style.background_gradient(cmap='Blues')


# # Drop-Column Importance
# 
# Drop-column importance can be done simply by dropping each column and re-training, which you already know how to do if you’ve trained a model.

# In[ ]:


from sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append( round( (benchmark_score - drop_col_score)/benchmark_score, 4) )
    
    importances_df = pd.DataFrame(X_train.columns, columns=['Feature'])
    importances_df['drop_col_importance'] = importances
    return importances_df

drop_col_impt_df = drop_col_feat_imp(rf, X, y)



# In[ ]:


drop_col_impt_df.sort_values('drop_col_importance', 
                   ascending=False)[
                                    ['Feature','drop_col_importance'
                                    ]
                                  ][:30].style.background_gradient(cmap='Blues')


# # Merging All Feature Importance Metrics Into a Single Results Dataframe
# 
# Now we are going to add *permutation_importance* and *drop_col_importance* columns to *all_ranks* dataframe as well as calculate the average (mean) feature importance score across all of 4 methods.

# In[ ]:


# merge drop_col_impt_df
all_ranks = pd.merge(all_ranks, drop_col_impt_df, on=['Feature'])

# merge permutate_df
all_ranks = pd.merge(all_ranks, permutate_df, on=['Feature'])

# calculate average feature importance
average_fi_pipeline = pdp.PdPipeline([
    pdp.ApplyToRows(
        lambda row: (row['drop_col_importance'] + row['permutation_importance'] + row['rfe_importance'] + row['alg_importance'])/4, 
        colname='mean_feature_importance') # 'mean_feature_importance
])

all_ranks = average_fi_pipeline.apply(all_ranks)

display(all_ranks.reset_index().drop(['index'], axis=1).style.background_gradient(cmap='summer_r'))


# # Forecasting Model Experiments
# 
# Based on the feature selection experiments above, we are going to train a bunch of different Random Forests (RF) models (using different feature sets and number of trees to train) to see which one performs better.
# 
# In the course of the activities down the road, we will have to repeat two core activities below
# 
# - subset top N features from the entire set of available features, based on a certain feature importance metrics
# - measure an individual model performace as well as output it in a format ready to aggregate in a single dataframe (to compare each model with its counterparts  as well as identify the one with the best performance)
# 
# So we are going to automate it with two auxiliary functions below
# - *get_top_features_by_rank*
# - *model_check*

# In[ ]:


def get_top_features_by_rank(metric_col_name: str, feature_number: int):
    features_df = all_ranks.copy()
    
    # features_df = features_df.sort_values(by=['feature_number'])
    
    # TODO: [:feature_number]
    
    # top n rows ordered by multiple columns
    features_df = features_df.nlargest(feature_number, [metric_col_name])
    
    result_list = list(features_df['Feature'])
    return result_list

def model_check(X, y, estimator, model_name, model_description, cv):
    model_table = pd.DataFrame()

    cv_results = cross_validate(estimator,
                                X,
                                y,
                                cv=cv,
                                scoring='neg_root_mean_squared_error',
                                return_train_score=True,
                                n_jobs=-1)

    train_rmse = -cv_results['train_score'].mean()
    test_rmse = -cv_results['test_score'].mean()
    test_std = cv_results['test_score'].std()
    fit_time = cv_results['fit_time'].mean()

    attributes = {
        'model_name': model_name,
        'train_score': train_rmse,
        'test_score': test_rmse,
        'test_std': test_std,
        'fit_time': fit_time,
        'description': model_description,
    }
    
    model_table = pd.DataFrame(data=[attributes])
    return model_table


# First of all, we are going to train a baseline RF model. It is going to be rather simple and not likely to perform in a good fashion. However, it will provide as the reasonable indication on how the more advanced models improve their performace vs. the baseline.

# In[ ]:


# check the baseline RF
baseline = model_check(X, y, rf, 'Baseline RF', "Baseline RF (100 trees, all features)", kf)
result_df = baseline


# After that, we are going to train a set of RF models, using the entire feature set we have after the  feature engineering above

# In[ ]:


n_estimators = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in n_estimators:
    rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=n, verbose=1)
    description = "RF with n_trees = {}".format(n)
    model_check_df = model_check(X, y, rf2, 'RF - All Features', description, kf)
    
    # concatenate
    frames = [result_df, model_check_df]
    result_df = pd.concat(frames)


# In the next series of experiemnts, we are going to train a number of RF models that use only top 50 features as ranked by **RFE** feature importance/feature selection algorithm.

# In[ ]:


# subset of features selected by RFE feature importance
top_rfe_features = 50
rfe_features = get_top_features_by_rank('rfe_importance', top_rfe_features)
X_important_features = X[rfe_features]
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in n_estimators:
    rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=n, verbose=1)
    description = "RF with top {} RFE features, n_trees = {}".format(top_rfe_features, n)
    model_check_df = model_check(X_important_features, y, rf2, 'RFE Features', description, kf)
    
    # concatenate
    frames = [result_df, model_check_df]
    result_df = pd.concat(frames)


# As our next step, we are going to train a set of RF models that utilize only the tiny subset of features selected from the model via embedded feature selection algorithm.

# In[ ]:


# subset of features selected by RF embedded Feature Selection
X_embedded_features = X[embeded_rf_feature]
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in n_estimators:
    rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=n, verbose=1)
    description = "RF witn n_trees = {}".format(n)
    model_check_df = model_check(X_embedded_features, y, rf2, 'RF - Embedded Features', description, kf)
    
    # concatenate
    frames = [result_df, model_check_df]
    result_df = pd.concat(frames)


# Now we are ready to train one more set of RF models that use top 50 features selected by **permutation method**

# In[ ]:


# train RF with the top importance feautres selected via the permutation method
top_features = 50
important_features = get_top_features_by_rank('permutation_importance', top_features)
X_important_features = X[important_features]

n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in n_estimators:
    rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=n, verbose=1)
    description = "RF with top {} permutatively important features, n_trees = {}".format(top_rfe_features, n)
    model_check_df = model_check(X_important_features, y, rf2, 'RF - Permutatively Important Features', description, kf)
    
    # concatenate
    frames = [result_df, model_check_df]
    result_df = pd.concat(frames)


# As a final step in our experiments, we are going to train a set of RF models with the top 50 features selected by **drop-column method**

# In[ ]:


# train RF with the top importance feautres selected via the drop-column method
top_features = 50
important_features = get_top_features_by_rank('drop_col_importance', top_features)
X_important_features = X[important_features]

n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in n_estimators:
    rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=n, verbose=1)
    description = "RF with top {} drop-col-important features, n_trees = {}".format(top_rfe_features, n)
    model_check_df = model_check(X_important_features, y, rf2, 'RF - Drop-Column Important Features', description, kf)
    
    # concatenate
    frames = [result_df, model_check_df]
    result_df = pd.concat(frames)


# Now we are going to display and compare the results of scoring each of the trained model in the tabular view below

# In[ ]:


display(result_df.reset_index().drop(['index'], axis=1).style.background_gradient(cmap='summer_r'))


# We will find that
# 
# - almost every model we trained (except for the set of models to use the tiny subset of features selected  via embedded feature  selection method)
# - feature importance scores calculated by RF algorithm directly are really misleading, and they do not reflect the actual feature importance for the model trained
# - the reason why embedded feature selection led to a poor result (see the chart above) can be explained by the fact RF needs reasonable variance in the feature space (to train diverse set of poor predictors - individual decision tree model estimators - to represent the  regression problem  complexity in a proper way)
# - among the bunch of models we trained, the best performance on the CV testing sets (withe testing score of 0.127509) was demonstrated by the model that used top 50 RFE features and n_trees = 800
# - in this case, the appropriate feature selection not only improved the interpretability of our model but also added the edge in its forecasting performance (vs. the set of RF models that used the entire set of features in training)

# **Note**: the model training piece above generates the weird job termination issues, if running the notebook here at Kaggle; therefore you could not see the final dataframe with each model scoring rendered properly in the views above. You can generate it, however, if you download the code and the competion data to run it locally.

# # References
# 
# You can refer to the blog posts below if you like to undertake the deeper dive into industrial feature selection/feature importance calculation techniques
# 
# - Feature Selection with pandas - https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b 
# - 5 feature selection methods every data scientist need to know - https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2 
# - Feature Importance May Be Lying to You - https://towardsdatascience.com/feature-importance-may-be-lying-to-you-3247cafa7ee7 
# - Beware Default Random Forest Importance - https://explained.ai/rf-importance/ 
# 
# 
