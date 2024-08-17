#!/usr/bin/env python
# coding: utf-8

# # Feature Selection
# 
# In this notebook we compile feature engineering ideas found in other notebooks and from the [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) course. We then use techniques to try to select a subset of these features. In some cases, there is a scikit-learn compatible transformer which we can use to perform this feature selection, otherwise, we write our own transformers to do this. We will consider the following techniques:
# 
# 1. Models w/ built-in feature selection
# 2. Removing multicollinear features
# 3. Recursive Feature Elimination (RFECV)
# 4. SelectKBest w/ various scoring functions
# 5. Permutation Importance
# 6. Sequential Feature Selection (SFS)

# In[1]:


# Update mlxtend (for feature_groups method for SFS)
get_ipython().system('pip install mlxtend --upgrade --no-deps')

# Global Variables
DEBUG = False  # notebook runs faster if debug = True
NUM_FOLDS = 3 if DEBUG else 12
RANDOM_SEED = 153

# Imports
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from functools import partial
from collections import defaultdict, Counter
from itertools import chain
from heapq import heappush

# Speedup some scikit-learn algorithms
from sklearnex import patch_sklearn
patch_sklearn()
import sklearn

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns

# Correlation/Clustering
from scipy.stats import spearmanr, pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist

# Preprocessing 
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from category_encoders import OneHotEncoder, OrdinalEncoder

# Feature Selection
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFECV
from sklearn.feature_selection import mutual_info_regression, f_regression, r_regression

# Models
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, check_cv
from sklearn.linear_model import Ridge, RidgeCV, HuberRegressor
from sklearn.linear_model import OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso, LassoCV, Lars, LarsCV
from sklearn.linear_model import LassoLars, LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor

# VIF
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


# Original Data
original_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
original_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col = 'Id')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

# Preprocessed Data
train = pd.read_csv('../input/house-prices-ames-cleaned-dataset/new_train.csv')
test = pd.read_csv('../input/house-prices-ames-cleaned-dataset/new_test.csv', index_col = 'Id')
original_cols = list(test.columns)

# Log transform the target
target = train['SalePrice'].apply(np.log1p)

# Cross Validation Splits
skf = list(StratifiedKFold(
    n_splits = NUM_FOLDS, 
    shuffle = True, 
    random_state = RANDOM_SEED
).split(
    train[test.columns], 
    pd.cut(target, bins = 20, labels = False)
))


# # 1. Feature Engineering
# 
# These features are taken from public notebooks and the Kaggle learn course. On their own, we don't know the value each feature brings to our model (if any). We will use several techniques to evaluate the feature importance in successive sections. We consider two types of models:
# 
# 1. Linear Model w/ L2 Regularization (`RidgeCV`)
# 2. Gradient Boosting (`GradientBoostingRegressor`)
# 
# In some cases a feature selection method will benefit one model more than the other. In the case of very slow methods, we only consider a linear model which trains much faster.

# In[3]:


# Indicator Features

train["HasShed"] = (train["MiscFeature"] == "Shed").astype(int)
test["HasShed"] = (test["MiscFeature"] == "Shed").astype(int)

train['MiscFeature'] = (train['MiscFeature'].notna()).astype(int)
test['MiscFeature'] = (test['MiscFeature'].notna()).astype(int)

train['HasGarage'] = (train['GarageType'].notna()).astype(int)
test['HasGarage'] = (test['GarageType'].notna()).astype(int)

train['HasPool'] = (train['PoolArea'] > 0).astype(int)
test['HasPool'] = (test['PoolArea'] > 0).astype(int)

porch_cols = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','WoodDeckSF']
train['HasPorch'] = (train[porch_cols].sum(axis = 1) > 0).astype(int)
test['HasPorch'] = (test[porch_cols].sum(axis = 1) > 0).astype(int)

train['HasFireplace'] = (train['Fireplaces'] > 0).astype(int)
test['HasFireplace'] = (test['Fireplaces'] > 0).astype(int)

train['HasFence'] = (train['Fence'] > 0).astype(int)
test['HasFence'] = (test['Fence'] > 0).astype(int)

train['Has2ndFloor'] = (train['2ndFlrSF'] > 0).astype(int)
test['Has2ndFloor'] = (test['2ndFlrSF'] > 0).astype(int)

train['HasBasement'] = (train['BsmtCond'] > 0).astype(int)
test['HasBasement'] = (test['BsmtCond'] > 0).astype(int)

train['Remodel'] = (train['YearRemodAdd'] != train['YearBuilt']).astype(int)
test['Remodel'] = (test['YearRemodAdd'] != test['YearBuilt']).astype(int)


# Combine counts and square footage

train['TotalSF'] = train[['TotalBsmtSF',"1stFlrSF", "2ndFlrSF"]].sum(axis = 1)
test['TotalSF'] = test[['TotalBsmtSF',"1stFlrSF", "2ndFlrSF"]].sum(axis = 1)

train['TotalBath'] = train[['FullBath','BsmtFullBath']].sum(axis = 1) + 0.5 * train[['HalfBath','BsmtHalfBath']].sum(axis = 1)
test['TotalBath'] = test[['FullBath','BsmtFullBath']].sum(axis = 1) + 0.5 * test[['HalfBath','BsmtHalfBath']].sum(axis = 1)

train['TotalPorch'] = train[porch_cols].sum(axis = 1)
test['TotalPorch'] = test[porch_cols].sum(axis = 1)

train["PorchTypes"] = train[porch_cols].gt(0.0).sum(axis=1)
test["PorchTypes"] = test[porch_cols].gt(0.0).sum(axis=1)

train['TotalLot'] = train['LotFrontage'] + train['LotArea']
test['TotalLot'] = test['LotFrontage'] + test['LotArea']

train['TotalBsmtFin'] = train['BsmtFinSF1'] + train['BsmtFinSF2']
test['TotalBsmtFin'] = test['BsmtFinSF1'] + test['BsmtFinSF2']

train["PCA_Feature1"] = train['GrLivArea'] + train['TotalBsmtSF']
test["PCA_Feature1"] = test['GrLivArea'] + test['TotalBsmtSF']

# Misc ratio and mutliplicative features

train["LivLotRatio"] = train["GrLivArea"] / train["LotArea"]
test["LivLotRatio"] = test["GrLivArea"] / test["LotArea"]

train["Spaciousness"] = (train["1stFlrSF"]+train["2ndFlrSF"]) / train["TotRmsAbvGrd"]
test["Spaciousness"] = (test["1stFlrSF"]+test["2ndFlrSF"]) / test["TotRmsAbvGrd"]

# PCA features (PCA feature engineering course) 
# Other PCA see: https://www.kaggle.com/code/muntasirphy/house-prices-top-1/notebook?scriptVersionId=103817570

train["PCA_Feature1"] = train['GrLivArea'] + train['TotalBsmtSF']
test["PCA_Feature1"] = test['GrLivArea'] + test['TotalBsmtSF']

train["PCA_Feature2"] = np.sqrt(train['YearRemodAdd'] * train['TotalBsmtSF'])
test["PCA_Feature2"] = np.sqrt(test['YearRemodAdd'] * test['TotalBsmtSF'])

train["PCA_Feature3"] = np.sqrt(train['GrLivArea'] * train['TotRmsAbvGrd'])
test["PCA_Feature3"] = np.sqrt(test['GrLivArea'] * test['TotRmsAbvGrd'])

train["PCA_Feature4"] = np.sqrt(train['GrLivArea'] * train['1stFlrSF'])
test["PCA_Feature4"] = np.sqrt(test['GrLivArea'] * test['1stFlrSF'])
    
train["PCA_Feature5"] = np.sqrt(train['GarageCars'] * train['GarageArea'])
test["PCA_Feature5"] = np.sqrt(test['GarageCars'] * test['GarageArea'])

train["PCA_Feature6"] = np.sqrt(train['GrLivArea'] * train['FullBath'])
test["PCA_Feature6"] = np.sqrt(test['GrLivArea'] * test['FullBath'])

train["PCA_Feature7"] = np.sqrt(train['GrLivArea'] * train['2ndFlrSF'])
test["PCA_Feature7"] = np.sqrt(test['GrLivArea'] * test['2ndFlrSF'])

# Age features

train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']

train['Remodel_Rel'] = (train['YrSold'] - train['YearRemodAdd']) / train['HouseAge']
test['Remodel_Rel'] = (test['YrSold'] - test['YearRemodAdd']) / test['HouseAge']
train['Remodel_Rel'] = np.nan_to_num(train['Remodel_Rel'], nan = 0, posinf = 1, neginf = 0)
test['Remodel_Rel'] = np.nan_to_num(test['Remodel_Rel'], nan = 0, posinf = 1, neginf = 0)

train['Garage_Rel'] = (train['YrSold'] - train['GarageYrBlt']) / train['HouseAge']
test['Garage_Rel'] = (test['YrSold'] - test['GarageYrBlt']) / train['HouseAge']
train['Garage_Rel'] = np.nan_to_num(train['Garage_Rel'], nan = 0, posinf = 1, neginf = 0)
test['Garage_Rel'] = np.nan_to_num(test['Garage_Rel'], nan = 0, posinf = 1, neginf = 0)


# Multiplicative interactions

train['OverallGrade'] = np.sqrt(train['OverallQual'] * train['OverallCond'])
test['OverallGrade'] = np.sqrt(test['OverallQual'] * test['OverallCond'])

train['ExterGrade'] = np.sqrt(train['ExterQual'] * train['ExterCond'])
test['ExterGrade'] = np.sqrt(test['ExterQual'] * test['ExterCond'])

train['BsmtGrade'] = np.sqrt(train['BsmtQual'] * train['BsmtCond'])
test['BsmtGrade'] = np.sqrt(test['BsmtQual'] * test['BsmtCond'])

train['GarageGrade'] = np.sqrt(train['GarageQual'] * train['GarageCond'])
test['GarageGrade'] = np.sqrt(test['GarageQual'] * test['GarageCond'])

train['MiscGrade'] = np.sqrt(train['KitchenQual'] * train['HeatingQC'])
test['MiscGrade'] = np.sqrt(test['KitchenQual'] * test['HeatingQC'])

# Other additive interactions

train['OverallSum'] = train[['OverallQual','OverallCond']].sum(axis = 1)
test['OverallSum'] = test[['OverallQual','OverallCond']].sum(axis = 1)

train['ExterSum'] = train[['ExterQual','ExterCond']].sum(axis = 1)
test['ExterSum'] = test[['ExterQual','ExterCond']].sum(axis = 1)

train['BsmtFinTypeSum'] = train[['BsmtFinType1','BsmtFinType2']].sum(axis = 1)
test['BsmtFinTypeSum'] = test[['BsmtFinType1','BsmtFinType2']].sum(axis = 1)

train['GarageSum'] = train[['GarageQual','GarageCond']].sum(axis = 1)
test['GarageSum'] = test[['GarageQual','GarageCond']].sum(axis = 1)

train['MiscSum'] = train[['KitchenQual','HeatingQC']].sum(axis = 1)
test['MiscSum'] = test[['KitchenQual','HeatingQC']].sum(axis = 1)

# Interaction Features w/ Qual/Cond columns

train['Wow_Factor'] = np.sqrt(train['OverallGrade'] * train['GrLivArea']).round()
test['Wow_Factor'] = np.sqrt(test['OverallGrade'] * test['GrLivArea']).round()

train['Wow_Basement'] = np.sqrt(train['BsmtGrade'] * train['TotalBsmtFin']).round()
test['Wow_Basement'] = np.sqrt(test['BsmtGrade'] * test['TotalBsmtFin']).round()

train['Wow_Exterior'] = np.sqrt(train['ExterGrade'] * train['MasVnrArea']).round()
test['Wow_Exterior'] = np.sqrt(test['ExterGrade'] * test['MasVnrArea']).round()

train['Wow_Garage'] = np.sqrt(train['GarageGrade'] * train['GarageArea']).round()
test['Wow_Garage'] = np.sqrt(test['GarageGrade'] * test['GarageArea']).round()

train.drop(columns = ['PoolQC','MoSold','YearBuilt','YrSold'], inplace = True)
test.drop(columns = ['PoolQC','MoSold','YearBuilt','YrSold'], inplace = True)


# # 2. Preprocessing
# 
# To avoid repetitive actions and to speedup our iterative cross-validation, we perform some of the preprocessing steps now rather than in our pipelines. 

# In[4]:


# Define categorical columns
category_cols = [x for x in test.columns if test[x].dtype == 'object']
binary_cols = [x for x in test.columns if len(test[x].unique()) == 2]
numerical_cols = [x for x in test.columns if (x not in category_cols + binary_cols)]

# Dictionary for saving scores
cv_scores = defaultdict(list)
cv_scores['Scheme'] = [
    *[f'Split {i}' for i in range(len(skf))], 'Average', 'Median'
]

# Saves ensembled predictions on the test set
preds = defaultdict(lambda: np.zeros(len(test)))

# One Hot Encoding (+ drop target)
encoder = OneHotEncoder(cols = category_cols, use_cat_names = True)
train = encoder.fit_transform(train[test.columns])
test = encoder.transform(test[test.columns])

# Transform skewed variables
temp = train[numerical_cols].skew()
skew_features = list(temp[temp > 0.75].index)
for col in skew_features:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

print('Training set size:', len(test.columns), 'columns')


# ## 2.1 Linear Model Baseline
# 
# This is the main model we will consider as it runs quickly and linear models do well in the competitions. L2 regularization preferred over L1 regularization so we can be sure the model itself doesn't exclude any features.

# In[5]:


get_ipython().run_cell_magic('time', '', "\n# Define generic linear model pipeline\npipeline = Pipeline([\n    ('scaler', RobustScaler()), \n    ('imputer', SimpleImputer()), \n    ('model', RidgeCV(alphas = np.logspace(-1,2,200)))\n])\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = pipeline,\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['ridge'] += model.predict(test) / NUM_FOLDS\n    cv_scores['ridge'].append(score)\n    print(f'Fold {fold}:', round(-score,6))\ncv_scores['ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# ## 2.2 Gradient Boosting Baseline
# 
# For all the examples in this notebook, we will use the linear model, however for some methods, the linear models is bad for demonstration so we will use a gradient boosting model instead.

# In[6]:


get_ipython().run_cell_magic('time', '', "\n# Define generic gradient boosting pipeline\ngb_pipeline = Pipeline([\n    ('imputer', SimpleImputer()), \n    ('model', GradientBoostingRegressor())\n])\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = gb_pipeline,\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['boost'] += model.predict(test) / NUM_FOLDS\n    cv_scores['boost'].append(score)\n    print(f'Fold {fold}:', round(-score,6))\ncv_scores['boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n \n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# # 3. Linear Models w/ Variable Selection
# 
# These models are taking from the **Regressors with variable selection** subsection of scikit-learn's [linear model classes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model). These models are sparse in the sense that the coefficients (e.g. `model.coef_`) corresponding to the features that are unused by the model correspend to zero. We chose those models which have a CV class which finds the best parameters using cross-validation. In particular, we consider the following models:
# 
# 1. LassoCV
# 2. LassoLarsCV
# 3. ElasticNetCV
# 4. OrthogonalMatchingPursuitCV
# 
# The results from these models seem to depend heavily on the amount of training data given, when ran with 5 folds these models performed worse than the `RidgeCV` baseline but with 8 or more folds they outperform the baseline (with the exception of OMP).
# 
# ## 3.1 LassoCV
# 
# A linear model with L1 regularization (as opposed to the L2 regularization using in the `Ridge` model). We first run `LassoCV` to determine a good regularization term `alpha`. We then use this same alpha to test our model and get our predictions.

# In[7]:


get_ipython().run_cell_magic('time', '', "\n# Get best alpha value\nestimator = Pipeline([\n    ('scaler', RobustScaler()), \n    ('imputer', SimpleImputer()), \n    ('model', LassoCV(cv = NUM_FOLDS, n_jobs = -1))\n])\nestimator.fit(train, target)\nbest_alpha = estimator.named_steps.model.alpha_\nprint(f'Best alpha: {round(best_alpha, 6)}\\n')\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = Pipeline([\n        ('scaler', RobustScaler()), \n        ('imputer', SimpleImputer()), \n        ('model', Lasso(alpha = best_alpha))\n    ]),\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['lasso'] += model.predict(test) / NUM_FOLDS\n    temp = pd.Series(model.named_steps.model.coef_, index = train.columns)\n    temp = temp[temp > 0]\n    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')\n    cv_scores['lasso'].append(score)\ncv_scores['lasso'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n \n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# ## 3.2 LassoLarsCV
# 
# LARS stands for "Least Angle Regression" and is supposedly efficient in the case of high dimension data relative to the number of samples. There is a `LarsCV` and `LassoLarsCV`, we only include the latter. You can read more about this in the scikit-learn [user guide](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression). We first run `LassoLarsCV` to determine a good regularization term `alpha`, we then train `LassoLars` models using this `alpha`.

# In[8]:


get_ipython().run_cell_magic('time', '', "\n# Get best alpha value\nestimator = Pipeline([\n    ('scaler', RobustScaler()), \n    ('imputer', SimpleImputer()), \n    ('model', LassoLarsCV(cv = NUM_FOLDS))\n])\nestimator.fit(train, target)\nbest_alpha = estimator.named_steps.model.alpha_\nprint(f'Best alpha: {round(best_alpha,6)}\\n')\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = Pipeline([('scaler', RobustScaler()), ('imputer', SimpleImputer()), ('model', LassoLars(alpha = best_alpha))]),\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['lassolars'] += model.predict(test) / NUM_FOLDS\n    temp = pd.Series(model.named_steps.model.coef_, index = train.columns)\n    temp = temp[temp > 0]\n    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')\n    cv_scores['lassolars'].append(score)\ncv_scores['lassolars'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n \n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# ## 3.3 ElasticNetCV
# 
# ElasticNet combines the `L1` regularization using the `Lasso` models with the `L2` regularization of the `Ridge` model. We run `ElasticNetCV` to determine an optimal `l1_ratio` (a trade-off for how much L1 vs L2 regularization we want), then we use this ratio for all the folds.

# In[9]:


get_ipython().run_cell_magic('time', '', "\n# Get optimal l1_ratio value\nparams = dict(\n    l1_ratio = np.logspace(-2,0,20),\n    cv = NUM_FOLDS,\n    n_jobs = -1\n)\nestimator = Pipeline([ \n    ('scaler', RobustScaler()), \n    ('imputer', SimpleImputer()), \n    ('model', ElasticNetCV(**params))\n])\nestimator.fit(train, target)\nbest_ratio = estimator.named_steps.model.l1_ratio_\nbest_alpha = estimator.named_steps.model.alpha_\nprint(f'Best L1 ratio: {round(best_ratio,6)}')\nprint(f'Best alpha: {round(best_alpha,6)}\\n')\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = Pipeline([\n        ('scaler', RobustScaler()), \n        ('imputer', SimpleImputer()), \n        ('model', ElasticNet(alpha = best_alpha, l1_ratio = best_ratio))\n    ]),\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['elasticnet'] += model.predict(test) / NUM_FOLDS\n    temp = pd.Series(model.named_steps.model.coef_, index = train.columns)\n    temp = temp[temp > 0]\n    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')\n    cv_scores['elasticnet'].append(score)\ncv_scores['elasticnet'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n \n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# ## 3.4 Orthogonal Matching Pursuit
# 
# Attempts to find the optimal number of non-zero coefficients in a greedy manner. From the scikit-learn [user guide](https://scikit-learn.org/stable/modules/linear_model.html#omp) it seems like the OMP algorithm greedily picks the next feature most correlated with the current residual (resulting from the already chosen features). This is a very sparse model (~15 features), yet it gives surprisingly decent cv scores.

# In[10]:


get_ipython().run_cell_magic('time', '', "\n# Optimal # of features\nestimator = Pipeline([\n    ('scaler', RobustScaler()), \n    ('imputer', SimpleImputer()), \n    ('model', OrthogonalMatchingPursuitCV(cv = NUM_FOLDS))\n])\nestimator.fit(train, target)\nbest_n_features = estimator.named_steps.model.n_nonzero_coefs_\n\n# Cross-validation for benchmarking\nscores = cross_validate(\n    estimator = Pipeline([\n        ('scaler', RobustScaler()), \n        ('imputer', SimpleImputer()), \n        ('model', OrthogonalMatchingPursuit(n_nonzero_coefs = best_n_features))\n    ]),\n    X = train,\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    return_estimator = True,\n)\n\n# Create test predictions and save cv scores \nfor fold, (score, model) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['ohp'] += model.predict(test) / NUM_FOLDS\n    temp = pd.Series(model.named_steps.model.coef_, index = train.columns)\n    temp = temp[temp > 0]\n    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')\n    cv_scores['ohp'].append(score)\ncv_scores['ohp'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n\n    \nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# # 4. Handling Multicollinearity
# 
# In this section, we consider feature selection methods that attempt to remove multicollinear features (e.g. features that are highly correlated with each other). If there are many features highly correlated with one another then removing one of these features may given misleading results since our model still has access to the other features. We consider two approaches aimed as reducing this multicollinearity:
# 
# 1. Clustering based on Multicollinearity
# 2. Iteratively removing features based on [Variance Inflation Factor](https://en.wikipedia.org/wiki/Variance_inflation_factor)
# 
# Neither of these methods are from the scikit-learn library however, we can use the `SelectorMixin` to create our own classes that perform the feature selection. Keep in mind that these methods do not attempt to use the target variable at all and don't remove general "bad" features, only remove redundant features.

# In[11]:


# Helper function for visualizing GridSearch
def plot_gridcv_results(clf, var):
    
    # Melt dataframe
    data = pd.DataFrame(clf.cv_results_).rename(columns = {f'param_selector__{var}': var})
    data = pd.melt(
        data, 
        id_vars = [var],
        value_vars = [x for x in data.columns if x.startswith('split')],
        var_name = 'cv_fold',
        value_name = 'cv_score'
    )
    data[var] = data[var].astype(float)
    data['cv_fold'] = data['cv_fold'].apply(lambda x: int(x.split('_')[0][5:]))
    data = data.sort_values(var).reset_index(drop = True)
    
    # Plot CV scores vs k features
    k = clf.best_params_[f'selector__{var}']
    fig, ax = plt.subplots(figsize=(9,5))
    sns.lineplot(x = data[var], y = data['cv_score'], ci = 'sd', ax = ax)
    plt.title('Feature Selection')
    plt.axvline(x = k, color='k', linestyle='--')
    plt.ylabel('Average RMSE')
    plt.xlabel(f'{var} ({round(k,3)})')
    plt.grid()
    plt.show()


# ## 4.1 Hierarchical Clustering
# 
# We adapt the heirarchical clustering code from the scikit-learn [user guide](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py). How this works:
# 
# 1. Calculate [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) on all features
# 2. Convert this into a distance where the correlated features are "closer"
# 3. Cluster features using this distance
# 4. Assign the feature most correlated with the target variable as the representative of the cluster (and drop all the other features)
# 4. Vary the distance threshold and look at the clusters formed
# 5. Return the clustering resulting in the best cross-validated score

# In[12]:


class CorrelatedClusters(BaseEstimator, SelectorMixin):
    '''
    Assumes input is a pandas dataframe with numerical features
    '''
    def __init__(self, threshold = 1, verbose = True):
        self.threshold = threshold
        self.verbose = verbose
    
    def fit(self, X, y):
        
        temp = pd.DataFrame(SimpleImputer().fit_transform(X), columns = range(X.shape[1]))
        
        # Calculate correlation
        self.corr_ = {ft:val for val, ft in  zip(r_regression(temp, y), temp.columns)}
        self.corr_mat_ = spearmanr(temp, nan_policy = 'omit').correlation
        self.corr_mat_ = (self.corr_mat_ + self.corr_mat_.T) / 2
        np.fill_diagonal(self.corr_mat_, 1)
        dist = np.nan_to_num(1 - np.abs(self.corr_mat_), nan = 0, posinf = 1e6, neginf = -1e6)
    
        # Cluster features
        self.dist_linkage_ = hierarchy.ward(squareform(dist))
        cluster_ids = hierarchy.fcluster(self.dist_linkage_, self.threshold, criterion="distance")
        self.clusters_ = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            col = temp.columns[idx]
            heappush(self.clusters_[cluster_id], (self.corr_[col], col))
        self.features_ = {clust[0][1] for clust in self.clusters_.values()}
        self.mask_ = np.asarray([(x in self.features_) for x in temp.columns])
        if self.verbose: print(f'Created {len(self.clusters_.keys())} clusters out of {len(temp.columns)} features')
        return self
    
    def _get_support_mask(self):
        return self.mask_
    
    def plot_dendrogram(self, columns):
        
        # Create dendrogram
        assert len(columns) == len(self.mask_)
        fig, ax = plt.subplots(figsize=(24, 10))
        dendro = hierarchy.dendrogram(
            self.dist_linkage_, p = 7, labels = columns, ax=ax, leaf_rotation=90, leaf_font_size = 8, truncate_mode = 'level'
        )
        fig.tight_layout()
        plt.show()
        
    def plot_heatmap(self, columns):
        
        # Plot heatmap
        assert len(columns) == len(self.mask_)
        fig, ax = plt.subplots(figsize=(20,20))
        dendro = hierarchy.dendrogram(
            self.dist_linkage_, labels = columns, no_plot = True, leaf_rotation=90, leaf_font_size = 8
        )
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        ax.imshow(self.corr_mat_[dendro["leaves"], :][:, dendro["leaves"]])
        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax.set_yticklabels(dendro["ivl"])
        fig.tight_layout()
        plt.show()
        
    def get_clusters(self, columns):
        
        assert len(columns) == len(self.mask_)
        idx = {i:x for i,x in enumerate(columns)}
        return [[idx[i[1]] for i in clust] for clust in self.clusters_.values()]


# ### 4.1.1 Linear Model
# 
# The linear model benefits modestly from clustering, however we save the clusters formed as feature groups for the SequentialFeatureSelector.

# In[13]:


get_ipython().run_cell_magic('time', '', "\n# Define pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()),\n    ('scaler', RobustScaler()),  \n    ('selector', CorrelatedClusters(verbose = False)),\n    ('model', RidgeCV(alphas = np.logspace(-1,2,200)))\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__threshold': np.arange(0.0, 0.7, 0.02),\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n)\n\n# Data structure of cv scores\nclf.fit(train, target)\nplot_gridcv_results(clf, 'threshold')\n")


# In[14]:


# Remaining features
clustering = CorrelatedClusters(threshold = clf.best_params_['selector__threshold'])
clustering.fit(X = train, y = target)
features = clustering.get_feature_names_out(test.columns)
clusters = clustering.get_clusters(test.columns)

# Fit model with remaining features 
scores = cross_validate(
    estimator = pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Dendrogram
clustering.plot_dendrogram(train.columns)


# In[15]:


# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['cluster_ridge'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['cluster_ridge'].append(score)
cv_scores['cluster_ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])


print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ### 4.1.2. Gradient Boosting
# 
# The gradient boosting model appears to benefit slightly from clustering features

# In[16]:


get_ipython().run_cell_magic('time', '', "\n# Define pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()), \n    ('selector', CorrelatedClusters(verbose = False)),\n    ('model', GradientBoostingRegressor())\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__threshold': np.arange(0.0, 0.4, 0.01),\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n)\n\n# Data structure of cv scores\nclf.fit(train, target)\nplot_gridcv_results(clf, 'threshold')\n")


# In[17]:


# Remaining features
clustering = CorrelatedClusters(threshold = clf.best_params_['selector__threshold'])
clustering.fit(X = train, y = target)
features = clustering.get_feature_names_out(test.columns)

# Fit model with remaining features 
scores = cross_validate(
    estimator = gb_pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Dendrogram
clustering.plot_dendrogram(train.columns)


# In[18]:


# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['cluster_boost'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['cluster_boost'].append(score)
cv_scores['cluster_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
  

print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ## 4.2 Variance Inflation Factor
# 
# This methods was inspired by the following two notebooks:
# 
# 1. [dealing-with-multicollinearity](https://www.kaggle.com/code/robertoruiz/dealing-with-multicollinearity) 
# 2. [sklearn-multicollinearity-class](https://www.kaggle.com/code/ffisegydd/sklearn-multicollinearity-class)
# 
# This works as follows, starting with the full set of features:
# 
# 1. Calculate VIF for each feature
# 2. Remove the feature with the highest VIF
# 3. In the case of ties pick the feature least correlated with the target
# 4. Pick the feature set resulting in the best cross-validated score
# 
# We rewrite the function from the [statsmodels](https://www.statsmodels.org/0.6.1/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html) library to use scikit-learn functions `LinearRegression()` and `r2_score` which results in a significant speedup. This is still a fairly slow method since it has to compute k least squares regressions for all k remaining features at each iteration.
# 

# In[19]:


# Variance inflation factor
def variance_inflation(X):
    k_vars = X.shape[1]
    vif = np.zeros(k_vars)
    for idx in range(k_vars):
        mask = np.arange(k_vars) != idx
        preds = LinearRegression().fit(X[:, mask], X[:, idx]).predict(X[:, mask])
        vif[idx] = 1. / (1. - r2_score(X[:, idx], preds))
    return vif

# Feature selection using VIF
class VIFSelector(BaseEstimator, SelectorMixin):
    
    def __init__(self, estimator, threshold =  100 if DEBUG else 10, cv = 5):
        self.threshold = threshold
        self.cv = cv
        self.estimator = estimator
        
    def fit(self, X, y):

        temp = pd.DataFrame(
            RobustScaler().fit_transform(
                SimpleImputer().fit_transform(X)
            )
        ) 
        dist = {ft: 1 - abs(val) for val, ft in  zip(r_regression(temp, y), temp.columns)}
        cols = set(temp.columns)
        self.scores_ = list()
        self.n_vars_ = list()
        self.heap_ = list()
        columns = dict()
        while True:
            max_vif, _, col = max(zip(variance_inflation(temp[cols].values),[dist[x] for x in cols], cols))
            if max_vif < self.threshold: break
            cols.remove(col)
            n_vars = len(cols)
            columns[n_vars] = set(cols)
            cv_dict = cross_validate(
                estimator = self.estimator,
                X = temp[cols],
                y = y,
                scoring = 'neg_root_mean_squared_error',
                cv = self.cv,
                n_jobs = -1,
            )
            heappush(self.heap_, (-np.mean(cv_dict['test_score']), n_vars))
            self.scores_.append(np.mean(cv_dict['test_score']))
            self.n_vars_.append(n_vars)
        self.best_score, self.best_n_vars = self.heap_[0]
        self.mask_ = np.asarray([(x in columns[self.best_n_vars]) for x in temp.columns])
        return self
    
    def _get_support_mask(self):
        return self.mask_
    
    def plot_scores(self):
            fig, ax = plt.subplots(figsize=(9,5))
            sns.lineplot(x = self.n_vars_, y = self.scores_, ax = ax)
            plt.title('Iterative Feature Elimination w/ VIF')
            plt.axvline(x = self.best_n_vars, color='k', linestyle='--')
            plt.ylabel('Average RMSE')
            plt.xlabel(f'Number of Features')
            plt.grid()
            plt.show()


# ### 4.2.1 Linear Model
# 
# This appears to slightly benefit the linear model.

# In[20]:


get_ipython().run_cell_magic('time', '', '\n# Remaining features\nselector = VIFSelector(estimator = RidgeCV(alphas = np.logspace(-1,2,200)), cv = skf)\nselector.fit(X = train, y = target)\nfeatures = selector.get_feature_names_out(test.columns)\n\n# Plot\nselector.plot_scores()\n')


# In[21]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['vif_ridge'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['vif_ridge'].append(score)
cv_scores['vif_ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
   

print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ### 4.2.2 Gradient Boosting
# 
# This may result in modest improvement for the gradient boosting model.

# In[22]:


get_ipython().run_cell_magic('time', '', '\n# Remaining features\nselector = VIFSelector(estimator = GradientBoostingRegressor(), cv = skf)\nselector.fit(X = train, y = target)\nfeatures = selector.get_feature_names_out(test.columns)\n\n# Plot\nselector.plot_scores()\n')


# In[23]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = gb_pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['vif_boost'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['vif_boost'].append(score)
cv_scores['vif_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 

print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# # 5. Recursive Feature Elimination
# 
# This technique uses the built-in `.coef_` or `.feature_importance` attribute of our fitted models. I couldn't get good results with this method with either gradient boosting or the linear model.

# In[24]:


get_ipython().run_cell_magic('time', '', "\n# Each iteration remove 5 worst features by feature importance\nselector = RFECV(\n    gb_pipeline,\n    step = 10,\n    scoring = 'neg_root_mean_squared_error', \n    cv = skf, \n    n_jobs = -1,\n    importance_getter = 'named_steps.model.feature_importances_'\n)\n\n# Remaining features\nselector.fit(X = SimpleImputer().fit_transform(train), y = target)\nfeatures = selector.get_feature_names_out(train.columns)\n\n# Fit model with remaining features \nscores = cross_validate(\n    estimator = gb_pipeline,\n    X = train[features],\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Make predictions\nfor fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['rfecv_boost'] += estimator.predict(test[features]) / NUM_FOLDS\n    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')\n    cv_scores['rfecv_boost'].append(score)\ncv_scores['rfecv_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n \n\nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# # 6. SelectKBest
# 
# In this section we consider methods using`SelectKBest` selector with the following scoring schemes:
# 
# 1. Mutual Information
# 2. F-score
# 
# For each method, we vary the number of chosen features and pick the one which results in the best cross-validated scores.
# 
# ## 6.1 Mutual Information - Linear Model

# In[25]:


get_ipython().run_cell_magic('time', '', "\n# Pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()),\n    ('scaler', RobustScaler()),\n    ('selector', SelectKBest()),\n    ('ridge', RidgeCV(alphas = np.logspace(-1,2,200)))\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__score_func': [mutual_info_regression],\n    'selector__k': np.arange(10, len(test.columns), 5)\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n)\n\n# Data structure of cv scores\nclf.fit(SimpleImputer().fit_transform(train), target)\n\n# Plot scores\nplot_gridcv_results(clf , 'k')\n")


# In[26]:


# Fit model with optimal parameters
scores = cross_validate(
    estimator = estimator.set_params(**clf.best_params_),
    X = train,
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['mi_ridge'] += estimator.predict(test) / NUM_FOLDS
    temp = estimator.named_steps.selector.get_feature_names_out(train.columns)
    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')
    cv_scores['mi_ridge'].append(score)
cv_scores['mi_ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 
    
print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ## 6.2 Mutual Information - Gradient Boosting

# In[27]:


get_ipython().run_cell_magic('time', '', "\n# Pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()),\n    ('scaler', RobustScaler()),\n    ('selector', SelectKBest()),\n    ('model', GradientBoostingRegressor())\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__score_func': [mutual_info_regression],\n    'selector__k': np.arange(10, len(test.columns), 5)\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n)\n\n# Data structure of cv scores\nclf.fit(SimpleImputer().fit_transform(train), target)\n\n# Plot scores\nplot_gridcv_results(clf , 'k')\n")


# In[28]:


# Fit model with optimal parameters
scores = cross_validate(
    estimator = estimator.set_params(**clf.best_params_),
    X = train,
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['mi_boost'] += estimator.predict(test) / NUM_FOLDS
    temp = estimator.named_steps.selector.get_feature_names_out(train.columns)
    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')
    cv_scores['mi_boost'].append(score)
cv_scores['mi_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 
    
print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ## 6.3. F-score - Linear Model

# In[29]:


get_ipython().run_cell_magic('time', '', "\n# Pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()),\n    ('scaler', RobustScaler()),\n    ('selector', SelectKBest()),\n    ('ridge', RidgeCV(alphas = np.logspace(-1,2,200)))\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__score_func': [f_regression],\n    'selector__k': np.arange(10, len(test.columns), 10)\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n)\n\n# Data structure of cv scores\nclf.fit(SimpleImputer().fit_transform(train), target)\n\n# Plot scores\nplot_gridcv_results(clf, 'k')\n")


# In[30]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = estimator.set_params(**clf.best_params_),
    X = train,
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['f_ridge'] += estimator.predict(test) / NUM_FOLDS
    temp = estimator.named_steps.selector.get_feature_names_out(train.columns)
    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')
    cv_scores['f_ridge'].append(score)
cv_scores['f_ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 
    
print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ## 6.4. F-score - Gradient Boosting

# In[31]:


get_ipython().run_cell_magic('time', '', "\n# Pipeline\nestimator = Pipeline([\n    ('imputer', SimpleImputer()),\n    ('scaler', RobustScaler()),\n    ('selector', SelectKBest()),\n    ('ridge', GradientBoostingRegressor())\n])\n\n# Parameter Grid\nparam_grid = {\n    'selector__score_func': [f_regression],\n    'selector__k': np.arange(10, len(test.columns), 10)\n}\n\n# GridSearch\nclf = GridSearchCV(\n    estimator = estimator,\n    param_grid = param_grid,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n)\n\n# Data structure of cv scores\nclf.fit(SimpleImputer().fit_transform(train), target)\n\n# Plot scores\nplot_gridcv_results(clf, 'k')\n")


# In[32]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = estimator.set_params(**clf.best_params_),
    X = train,
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['f_boost'] += estimator.predict(test) / NUM_FOLDS
    temp = estimator.named_steps.selector.get_feature_names_out(train.columns)
    print(f'Fold {fold}:', round(-score,6), f'using {len(temp)} columns.')
    cv_scores['f_boost'].append(score)
cv_scores['f_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 
    
print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# # 7. Permutation Importance
# 
# There is not a canonical way to use permutation importance for feature selection (e.g. within `SelectKBest`) so we have to do this manually. We accomplish this as follows:
# 
# 1. Train several models using cross-validation
# 2. Evaluate permutation importance on each fold
# 3. Do the equivalent of `SelectKBest` using the permutation importance averaged over the folds
# 
# This method has to train a bunch of models to determine permutation importance so it may take a while to run.

# In[33]:


class PermutationSelector(BaseEstimator, SelectorMixin):
    
    def __init__(self, estimator, cv = NUM_FOLDS, min_features = 20, step = 2):
        self.estimator = estimator
        self.cv = cv
        self.min_features = 20
        self.step = step
        
    def fit(self, X, y):
        
        train = pd.DataFrame(X)
        target = pd.Series(y)
        
        # Fit models to be used for perm importance
        scores = cross_validate(
            estimator = self.estimator,
            X = train,
            y = target,
            scoring = 'neg_root_mean_squared_error',
            cv = self.cv,
            return_estimator = True,
        )
        
        # Use fitted estimators to calculate permutation importance
        pi_scores = pd.Series(np.zeros(len(train.columns)), index = train.columns)
        for fold, (model, (train_idx, valid_idx)) in enumerate(zip(scores['estimator'], check_cv(self.cv).split(X))):
            pi_scores += permutation_importance(
                estimator = model, 
                X = train.loc[valid_idx],
                y = target.loc[valid_idx],
                scoring = 'neg_root_mean_squared_error',
                random_state = RANDOM_SEED,
                n_jobs = -1,
            )['importances_mean']

        # Data structure of cv scores
        self.num_features_ = list()
        self.scores_ = list()

        sorted_pi = pi_scores.sort_values(ascending = False)
        # Manual SelectKBest for k = 10, 20,...,
        for k in range(10, len(train.columns), self.step):
            
            features = list(sorted_pi.index[:k])
            # Fit models to be used for perm importance
            scores = cross_validate(
                estimator = self.estimator,
                X = train[features],
                y = target,
                scoring = 'neg_root_mean_squared_error',
                cv = self.cv,
                return_estimator = True,
            )
            self.num_features_.append(k)
            self.scores_.append(np.mean(scores['test_score']))
    
        self.best_score_, self.best_k_ = max(zip(self.scores_, self.num_features_))
        self.mask_ = np.asarray([(x in sorted_pi.index[:self.best_k_]) for x in train.columns])
    
    def _get_support_mask(self):
        return self.mask_
    
    def plot_scores(self):
            fig, ax = plt.subplots(figsize=(9,5))
            sns.lineplot(x = self.num_features_, y = self.scores_, ax = ax)
            plt.title('Feature Elimination w/ Permutation Importance')
            plt.axvline(x = self.best_k_, color='k', linestyle='--')
            plt.ylabel('Average RMSE')
            plt.xlabel(f'Number of Features')
            plt.grid()
            plt.show()
        


# ## 7.1 Linear Model

# In[34]:


get_ipython().run_cell_magic('time', '', '\n# Permutation Importance Selector\nselector = PermutationSelector(\n    pipeline, \n    cv = NUM_FOLDS, \n    min_features = 20, \n    step = 5\n)\n\n# Linear pipeline\nsfs = selector.fit(X = train, y = target)\nfeatures = selector.get_feature_names_out(train.columns)\n\nselector.plot_scores()\n')


# In[35]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['perm_ridge'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['perm_ridge'].append(score)
cv_scores['perm_ridge'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 

print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# ## 7.2 Gradient Boosting

# In[36]:


get_ipython().run_cell_magic('time', '', '\n# Permutation Importance Selector\nselector = PermutationSelector(\n    gb_pipeline, \n    cv = NUM_FOLDS, \n    min_features = 20, \n    step = 5\n)\n\n# Linear pipeline\nsfs = selector.fit(X = train, y = target)\nfeatures = selector.get_feature_names_out(train.columns)\n\nselector.plot_scores()\n')


# In[37]:


# Fit model with remaining features 
scores = cross_validate(
    estimator = gb_pipeline,
    X = train[features],
    y = target,
    scoring = 'neg_root_mean_squared_error',
    cv = skf,
    n_jobs = -1,
    return_estimator = True,
)

# Make predictions
for fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):
    preds['perm_boost'] += estimator.predict(test[features]) / NUM_FOLDS
    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')
    cv_scores['perm_boost'].append(score)
cv_scores['perm_boost'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])
 

print('\nBest    (RSME):', round(-np.max(scores['test_score']), 6))
print('Median  (RSME):', round(-np.mean(scores['test_score']), 6))
print('Average (RSME):', round(-np.median(scores['test_score']), 6))
print('Worst   (RSME):', round(-np.min(scores['test_score']), 6))


# # 8. Sequential Feature Selection
# 
# We use an extended version of the scikit-learn SequentialFeatureSelector. Forward SFS is a greedy algorithm which adds the feature which results in the greatest cv score improvement at each iteration. To speed up the algorithm and hopefully give better results we group certain features together based on clustering and categories using the `feature_group` option.
# 
# ## 8.1. SFS - Clusters
# 
# We consider highly correlated features together using the best clustering we found earlier (using the `CorrelatedClusters` class). This algorithm gives good results but runs very slowly.

# In[38]:


get_ipython().run_cell_magic('time', '', "\n# Define sequential feature selector\nselector = SequentialFeatureSelector(\n    pipeline, \n    k_features = 2 if DEBUG else 'parsimonious',\n    scoring = 'neg_root_mean_squared_error', \n    cv = skf, \n    feature_groups = clusters,\n    n_jobs = -1,\n)\n\n# Linear pipeline\nsfs = selector.fit(X = train, y = target)\nfeatures = list(selector.k_feature_names_)\n\n# Fit model with best found features \nscores = cross_validate(\n    estimator = pipeline,\n    X = train[features],\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Make predictions\nfor fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['sfs_clusters'] += estimator.predict(test[features]) / NUM_FOLDS\n    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')\n    cv_scores['sfs_clusters'].append(score)\ncv_scores['sfs_clusters'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n\nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# ## 8.2 Feature Groups - Categories
# 
# In this section, we group the one-hot encoded features together which should reduce the dimensionality significantly.

# In[39]:


get_ipython().run_cell_magic('time', '', "\n# Define groups (group one-hot encoded things together)\ncategory_cols += ['Condition', 'Exterior']\ngroups = [[x for x in train.columns if x.startswith(y)] for y in category_cols]\ngroups += [[x] for x in train.columns if x not in set(chain.from_iterable(groups))]\n\n# Define sequential feature selector\nselector = SequentialFeatureSelector(\n    pipeline, \n    k_features = 2 if DEBUG else 'parsimonious',\n    scoring = 'neg_root_mean_squared_error', \n    cv = skf, \n    feature_groups = groups,\n    n_jobs = -1,\n)\n\n# Linear pipeline\nsfs = selector.fit(X = train, y = target)\nfeatures = list(selector.k_feature_names_)\n\n# Fit model with best found features \nscores = cross_validate(\n    estimator = pipeline,\n    X = train[features],\n    y = target,\n    scoring = 'neg_root_mean_squared_error',\n    cv = skf,\n    n_jobs = -1,\n    return_estimator = True,\n)\n\n# Make predictions\nfor fold, (score, estimator) in enumerate(zip(scores['test_score'], scores['estimator'])):\n    preds['sfs_categories'] += estimator.predict(test[features]) / NUM_FOLDS\n    print(f'Fold {fold}:', round(-score,6), f'using {len(features)} columns.')\n    cv_scores['sfs_categories'].append(score)\ncv_scores['sfs_categories'].extend([np.mean(scores['test_score']),np.median(scores['test_score'])])\n\nprint('\\nBest    (RSME):', round(-np.max(scores['test_score']), 6))\nprint('Median  (RSME):', round(-np.mean(scores['test_score']), 6))\nprint('Average (RSME):', round(-np.median(scores['test_score']), 6))\nprint('Worst   (RSME):', round(-np.min(scores['test_score']), 6))\n")


# # 9. Summary

# In[40]:


temp = pd.DataFrame(cv_scores)
temp.set_index('Scheme', inplace = True)
temp = temp.T
temp.sort_values('Average', ascending = False)


# # 10. Submissions
# 
# Finally, we used our saved predictions to make submissions.

# In[41]:


for key in preds.keys():

    # Create submission for each method/model
    submission['SalePrice'] = np.expm1(preds[key])
    submission.to_csv(f'{key}_submission.csv', index = False)


# Thank you for reading, I hope you found this useful.
