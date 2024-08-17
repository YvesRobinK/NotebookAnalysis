#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Model Stacking for House Price Modelling
# 
# In this notebook, I use the open source Python library [Feature-engine](https://feature-engine.readthedocs.io/en/latest/) to create 3 different pipelines of variable transformation. Then, I train several machine learning models utilizing the transformed datasets, tuning their parameters with a grid search with cross-validation. And finally, I will combine the models through stacking.
# 
# I analysed the house prices data set in a [different notebook](https://www.kaggle.com/solegalli/predict-house-price-with-feature-engine) in case you are interested in getting more familiar with the variables. There are also a number of notebooks in Kaggle with good data exploration.
# 
# ### This notebook is based on the following resources:
# 
# - [Feature-engine](https://feature-engine.readthedocs.io/en/latest/), Python open source library
# - [Feature Engineering for Machine Learning](https://www.courses.trainindata.com/p/feature-engineering-for-machine-learning), online course.
# - [Packt Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)
# - [Kaggle ensembling guide](https://mlwave.com/kaggle-ensembling-guide/)

# In[1]:


# let's install Feature-engine

get_ipython().system('pip install feature_engine')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint

# Scikit-learn metrics and handling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Scikit-learn models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Other models
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# feature engineering
from feature_engine import creation
from feature_engine import discretisation as disc
from feature_engine import encoding as enc
from feature_engine import imputation as imp
from feature_engine import selection as sel
from feature_engine import transformation as tf


# ## Load data

# In[4]:


# load training data
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# load data for competition submission
# this data does not have the target
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[5]:


# split training data into train and test

X_train, X_test, y_train, y_test = train_test_split(data.drop(
    ['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.05,
    random_state=0)

X_train.shape, X_test.shape


# In[6]:


# drop id
id_ = submission['Id']

submission.drop('Id', axis=1, inplace=True)


# In[7]:


# let's transform the target with the log

y_train = np.log(y_train)

y_test = np.log(y_test)


# ## Quick setting
# 
# I will create lists with the variable names for which I will carry common pre-processing and transformations.

# In[8]:


# let's identify the categorical variables

categorical = [var for var in X_train.columns if data[var].dtype == 'O']

# MSSubClass is also categorical by definition, despite its numeric values
categorical = categorical + ['MSSubClass']

# number of categorical variables
len(categorical)


# In[9]:


# cast all variables as categorical, for automatic use with Feature-engine

X_train[categorical] = X_train[categorical].astype('O')
X_test[categorical] = X_test[categorical].astype('O')
submission[categorical] = submission[categorical].astype('O')


# In[10]:


master_data = pd.concat([data, submission], axis=0)


# In[11]:


# variables to impute with the most frequent category
categorical_mode = [var for var in categorical 
                    if master_data[var].isnull().sum()>0 
                    and master_data[var].isnull().mean()<0.1]

# variables to impute with the string missing
categorical_missing = [var for var in categorical 
                       if master_data[var].isnull().sum()>0 
                       and master_data[var].isnull().mean()>=0.1]

len(categorical_mode), len(categorical_missing)


# In[12]:


# some variables refer to years, we are better off if we combine them into new features

year_vars = [var for var in X_train.columns if 'Yr' in var or 'Year' in var]

year_vars


# In[13]:


# when I create new features automatically using feature engine, these
# 2 new variables will contain missing data, as they come from garageYrBlt, which
# shows na.

new_vars = ['YrSold_sub_GarageYrBlt', 'GarageYrBlt_sub_YearBuilt']


# In[14]:


# let's find the numerical variables

numerical = [var for var in X_train.columns if var not in categorical+year_vars]

len(numerical)


# In[15]:


# variables to impute with the most frequent category
numerical_median = [var for var in numerical 
                    if master_data[var].isnull().sum()>0 
                    and master_data[var].isnull().mean()<0.1]

# variables to impute with the string missing
numerical_arbitrary = [var for var in numerical 
                       if master_data[var].isnull().sum()>0 
                       and master_data[var].isnull().mean()>=0.1]

len(numerical_median), len(numerical_arbitrary)


# In[16]:


# let's find non-discrete variables

discretize = [
    var for var in numerical if len(X_train[var].unique()) >= 20
]

# number of discrete variables
len(discretize)


# ## Feature Engineering Pipelines

# In[17]:


linear_pipe = Pipeline([
    
    # === feature creation ===
    
    # this transformer substracts the reference variables from YrSold, 
    # one at a time, to create 3 new variables with the elapsed time between the 2
    
    ('elapsed_time', creation.CombineWithReferenceFeature(
        variables_to_combine = ['YrSold'],
        reference_variables = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
        operations=['sub'],
    )),
    
    # this transformer substracts YearBuilt from the 2 variables to combine
    # to capture the time passed between when the house was built and then 
    # subsequently remodelled, or when the garage was built
    
    ('elapsed_time2', creation.CombineWithReferenceFeature(
    variables_to_combine = ['YearRemodAdd', 'GarageYrBlt'],
    reference_variables = ['YearBuilt'],
    operations=['sub'],
     )),
    
    # the following 2 steps are inspired of the engineering in this notebook:
    # https://www.kaggle.com/marto24/beginners-prediction-top3
    
    # this transformer summs the 4 variables to combine in a new variable with the
    # indicated name in new_variables_name
    ('total_surface', creation.MathematicalCombination(
        variables_to_combine=['TotRmsAbvGrd', 'FullBath','HalfBath', 'KitchenAbvGr'],
        math_operations=['sum'],
        new_variables_names=['Total_Surface']
    )),
    
    # this transformer takes the ratio of the variable to combine and the reference
    # into a new variable
    
    ('surface_room', creation.CombineWithReferenceFeature(
        variables_to_combine = ['GrLivArea'],
        reference_variables = ['Total_Surface'],
        operations=['div'],
    )),
    
    
    # this transformer summs the 2 variables to combine in a new variable with the
    # indicated name in new_variables_name
    # idea also taken from https://www.kaggle.com/marto24/beginners-prediction-top3
    ('qual_sf', creation.MathematicalCombination(
        variables_to_combine=['1stFlrSF', '2ndFlrSF'],
        math_operations=['sum'],
        new_variables_names=['HighQualSF']
    )),
    
    # === drop year vars ===
    
    # now that I have used these variables to derive the above features, I
    # drop them from the data
    
    ('drop_features', sel.DropFeatures(
        features_to_drop =['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']
    )),
    
    # === missing data imputation ====
    
    # adds binary variables when data is missing for the indicated variables
    
    ('missing_ind', imp.AddMissingIndicator(
        missing_only=True, variables=numerical_arbitrary+categorical_mode+new_vars
    )),
    
    # replaces NA by a value placed at the 75th quantile + 3 * IQR of the variable
    
    ('arbitrary_number', imp.EndTailImputer(
        imputation_method='iqr', tail='right', fold=3, variables=numerical_arbitrary
    )),
    
    # replaces NA with the median value of the variable
    
    ('median', imp.MeanMedianImputer(
        imputation_method='median', variables=numerical_median+new_vars
    )),
    
    # replaces NA with the most frequent category
    
    ('frequent', imp.CategoricalImputer(
        imputation_method='frequent', variables=categorical_mode, return_object=True
    )),
    
    # replaces NA with the string 'Missing'
    
    ('missing', imp.CategoricalImputer(
        imputation_method='missing', variables=categorical_missing, return_object=True
    )),
    
    # === transformation ==
    
    # applies Yeo-Johnson transformation to the indicated variables
    
    ('transformation', tf.YeoJohnsonTransformer(variables=discretize)),
     
     # === categorical encoding 

    # one hot encoding of the 10 most frequent categories of each categorical
    # variable
    # (Feature-engine recognises categorical variables automatically if they 
    # are casted as object)
    
    ('encoder', enc.OneHotEncoder(top_categories=10)),
     
    # === feature Scaling ===
    
    ('scaler', StandardScaler()),
])


# In[18]:


# fit pipeline, learns all necessary parameters
linear_pipe.fit(X_train, y_train)

# transform the data
X_train_linear = linear_pipe.transform(X_train)
X_test_linear = linear_pipe.transform(X_test)
submission_linear = linear_pipe.transform(submission)


# In[19]:


monotonic_pipe = Pipeline([
    
    # === feature creation ===
    
    ('elapsed_time', creation.CombineWithReferenceFeature(
        variables_to_combine = ['YrSold'],
        reference_variables = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
        operations=['sub'],
    )),
    
    ('elapsed_time2', creation.CombineWithReferenceFeature(
    variables_to_combine = ['YearRemodAdd', 'GarageYrBlt'],
    reference_variables = ['YearBuilt'],
    operations=['sub'],
     )),
    
    # the following 2 steps are inspired of the engineering in this notebook:
    # https://www.kaggle.com/marto24/beginners-prediction-top3
    
    # this transformer summs the 4 variables to combine in a new variable with the
    # indicated name in new_variables_name
    ('total_surface', creation.MathematicalCombination(
        variables_to_combine=['TotRmsAbvGrd', 'FullBath','HalfBath', 'KitchenAbvGr'],
        math_operations=['sum'],
        new_variables_names=['Total_Surface']
    )),
    
    # this transformer takes the ratio of the variable to combine and the reference
    # into a new variable
    
    ('surface_room', creation.CombineWithReferenceFeature(
        variables_to_combine = ['GrLivArea'],
        reference_variables = ['Total_Surface'],
        operations=['div'],
    )),
    
    
    # this transformer summs the 2 variables to combine in a new variable with the
    # indicated name in new_variables_name
    # idea also taken from https://www.kaggle.com/marto24/beginners-prediction-top3
    ('qual_sf', creation.MathematicalCombination(
        variables_to_combine=['1stFlrSF', '2ndFlrSF'],
        math_operations=['sum'],
        new_variables_names=['HighQualSF']
    )),
    
    # === drop year vars ===
    
    ('drop_features', sel.DropFeatures(
        features_to_drop =['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']
    )),
    
    # === missing data imputation ====
    
    ('missing_ind', imp.AddMissingIndicator(
        missing_only=True, variables=numerical_arbitrary+categorical_mode+new_vars
    )),
    
    ('arbitrary_number', imp.EndTailImputer(
        imputation_method='iqr', tail='right', fold=3, variables=numerical_arbitrary
    )),
       
    ('median', imp.MeanMedianImputer(
        imputation_method='median', variables=numerical_median+new_vars
    )),
    
    ('frequent', imp.CategoricalImputer(
        imputation_method='frequent', variables=categorical_mode, return_object=True
    )),
     
    ('missing', imp.CategoricalImputer(
        imputation_method='missing', variables=categorical_missing, return_object=True
    )),
    
    
    # == rare category grouping ==
    
    # we group categories that appear in less than 10% of the observations
    # into a new label called 'Rare'
    
    ('rare_grouping', enc.RareLabelEncoder(
        tol = 0.1,n_categories=1,
    )),
    
    # === discretization ==
    
    # sort continuous variables into discrete bins of equal number of observations
    # returns variables cast as objects, that will be automatically captured 
    # by the encoder later on
    ('discretizer', disc.EqualWidthDiscretiser(
        bins=3, variables=discretize,return_object=True
    )),
     
     # === categorical encoding
    
    # transform the categories of categorical variables and the bins of the disctetized
    # variables into integers, that go from 0 to the number of unique values, in the order
    # of the target mean per category or per bin
    
    ('encoder', enc.OrdinalEncoder(encoding_method='ordered')),
     
    # === feature Scaling ===
    
    ('scaler', StandardScaler()),
])


# In[20]:


monotonic_pipe.fit(X_train, y_train)

X_train_monotonic = monotonic_pipe.transform(X_train)
X_test_monotonic = monotonic_pipe.transform(X_test)
submission_monotonic = monotonic_pipe.transform(submission)


# In[21]:


tree_pipe = Pipeline([
    
    # === feature creation ===
    
    ('elapsed_time', creation.CombineWithReferenceFeature(
        variables_to_combine = ['YrSold'],
        reference_variables = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
        operations=['sub'],
    )),
    
    ('elapsed_time2', creation.CombineWithReferenceFeature(
    variables_to_combine = ['YearRemodAdd', 'GarageYrBlt'],
    reference_variables = ['YearBuilt'],
    operations=['sub'],
     )),
    
    # the following 2 steps are inspired of the engineering in this notebook:
    # https://www.kaggle.com/marto24/beginners-prediction-top3
    
    # this transformer summs the 4 variables to combine in a new variable with the
    # indicated name in new_variables_name
    ('total_surface', creation.MathematicalCombination(
        variables_to_combine=['TotRmsAbvGrd', 'FullBath','HalfBath', 'KitchenAbvGr'],
        math_operations=['sum'],
        new_variables_names=['Total_Surface']
    )),
    
    # this transformer takes the ratio of the variable to combine and the reference
    # into a new variable
    
    ('surface_room', creation.CombineWithReferenceFeature(
        variables_to_combine = ['GrLivArea'],
        reference_variables = ['Total_Surface'],
        operations=['div'],
    )),
    
    
    # this transformer summs the 2 variables to combine in a new variable with the
    # indicated name in new_variables_name
    # idea also taken from https://www.kaggle.com/marto24/beginners-prediction-top3
    ('qual_sf', creation.MathematicalCombination(
        variables_to_combine=['1stFlrSF', '2ndFlrSF'],
        math_operations=['sum'],
        new_variables_names=['HighQualSF']
    )),
    
    # === drop year vars ===
    
    ('drop_features', sel.DropFeatures(
        features_to_drop =['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']
    )),
    
    # === missing data imputation ====
    
    ('missing_ind', imp.AddMissingIndicator(
        missing_only=True, variables=numerical_arbitrary+categorical_mode+new_vars
    )),
    
    ('arbitrary_number', imp.EndTailImputer(
        imputation_method='iqr', tail='right', fold=3, variables=numerical_arbitrary
    )),
       
    ('median', imp.MeanMedianImputer(
        imputation_method='median', variables=numerical_median+new_vars
    )),
    
    ('frequent', imp.CategoricalImputer(
        imputation_method='frequent', variables=categorical_mode, return_object=True
    )),
     
    ('missing', imp.CategoricalImputer(
        imputation_method='missing', variables=categorical_missing, return_object=True
    )),
    
    
    # == rare category grouping ==
    
    ('rare_grouping', enc.RareLabelEncoder(
        tol = 0.1,n_categories=1,
    )),
    
    # === discretization ==
    
    # we replace the values of continuous variables by the predictions made by a 
    # decision tree
    ('discretizer', disc.DecisionTreeDiscretiser(
        cv=3, scoring='neg_mean_squared_error', variables=discretize,
        regression=True, random_state=0
    )),
     
     # === categorical encoding
    
    # we replace the categories of categorical variables by the predictions made by a 
    # decision tree
    ('encoder', enc.DecisionTreeEncoder(
        encoding_method='arbitrary', cv=3, scoring='neg_mean_squared_error',
        regression=True, random_state=0
    )),
     
    # === feature Scaling ===
    ('scaler', StandardScaler()),
])


# In[22]:


tree_pipe.fit(X_train, y_train)

X_train_tree = tree_pipe.transform(X_train)
X_test_tree = tree_pipe.transform(X_test)
submission_tree = tree_pipe.transform(submission)


# ## Machine Learning Model Library

# In[23]:


# gradient boosting regressor

gbm_param = dict(
    loss=['ls', 'huber'],
    n_estimators=[10, 20, 50, 100, 200],
    min_samples_split=[0.01, 0.1, 0.3],
    max_depth=[1,2,3,None],
    )

gbm = GradientBoostingRegressor(
    loss='ls',
    n_estimators=100,
    criterion='friedman_mse',
    min_samples_split=2,
    max_depth=3,
    random_state=0,
    n_iter_no_change=2,
    tol=0.0001,
    )


# In[24]:


gbm_grid = RandomizedSearchCV(gbm, gbm_param, scoring='neg_mean_squared_error', n_iter=100, random_state=1)

# gbm_grid = GridSearchCV(gbm, gbm_param, scoring='neg_mean_squared_error')

gbm_linear = gbm_grid.fit(X_train_linear, y_train)
gbm_monotonic = gbm_grid.fit(X_train_monotonic, y_train)
gbm_tree = gbm_grid.fit(X_train_tree, y_train)


# In[25]:


gbm_linear.best_params_


# In[26]:


# function to plot the results of the grid search

def plot_grid(grid, title):
    
    # make df with results
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='mean_test_score', ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    
    # plot results
    results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)
    plt.ylabel('Mean test score')
    plt.title(title)
    plt.show()
    
    return None


# In[27]:


# plot results

plot_grid(gbm_linear, 'gbm search - linear data')
plot_grid(gbm_monotonic, 'gbm search - monotonic data')
plot_grid(gbm_tree, 'gbm search - tree data')


# In[28]:


# Elastic Net - Linear Model

elastic_param = dict(
    max_iter=[50000, 100000],
    alpha=[0.001, 0.01],
    l1_ratio=[0, 0.2, 0.5, 0.7, 1]
    )

elastic = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=100000,
    random_state=0
)


# In[29]:


# elastic_grid = RandomizedSearchCV(elastic, elastic_param, scoring='neg_mean_squared_error', n_iter=5, random_state=1)

elastic_grid = GridSearchCV(elastic, elastic_param, scoring='neg_mean_squared_error')

elastic_linear = elastic_grid.fit(X_train_linear, y_train)
elastic_monotonic = elastic_grid.fit(X_train_monotonic, y_train)
elastic_tree = elastic_grid.fit(X_train_tree, y_train)


# In[30]:


elastic_linear.best_params_


# In[31]:


plot_grid(elastic_linear, 'elastic search - linear data')
plot_grid(elastic_monotonic, 'elastic search - monotonic data')
plot_grid(elastic_tree, 'elastic search - tree data')


# In[32]:


# Nearest Neighbours

knn_param = dict(
    n_neighbors=[3,5,10], 
    algorithm=['ball_tree', 'kd_tree', 'brute'],
    p=[1,2],
    )

knn = KNeighborsRegressor(
    n_neighbors=5, 
    algorithm='auto',
    leaf_size=30,
    p=2,
    )


# In[33]:


# knn_grid = RandomizedSearchCV(knn, knn_param, scoring='neg_mean_squared_error', n_iter=5, random_state=1)

knn_grid = GridSearchCV(knn, knn_param, scoring='neg_mean_squared_error')

knn_linear = knn_grid.fit(X_train_linear, y_train)
knn_monotonic = knn_grid.fit(X_train_monotonic, y_train)
knn_tree = knn_grid.fit(X_train_tree, y_train)


# In[34]:


knn_linear.best_params_


# In[35]:


plot_grid(knn_linear, 'knn search - linear data')
plot_grid(knn_monotonic, 'knn search - monotonic data')
plot_grid(knn_tree, 'knn search - tree data')


# In[36]:


# Light GBM

lgbm_param = {
    "num_leaves": [20,30,40],
    "max_depth": [4, 6, 10, 20],
    "n_estimators": [20, 64, 100],
}

lgbm = LGBMRegressor(
    learning_rate = 0.16060612646519587, 
    min_child_weight = 0.4453842422224686,
    objective='regression', 
    random_state=0)


# In[37]:


lgbm_grid = GridSearchCV(lgbm, lgbm_param, scoring='neg_mean_squared_error')

lgbm_linear = lgbm_grid.fit(X_train_linear, y_train)
lgbm_monotonic = lgbm_grid.fit(X_train_monotonic, y_train)
lgbm_tree = lgbm_grid.fit(X_train_tree, y_train)


# In[38]:


lgbm_linear.best_params_


# In[39]:


plot_grid(lgbm_linear, 'lgbm search - linear data')
plot_grid(lgbm_monotonic, 'lgbm search - monotonic data')
plot_grid(lgbm_tree, 'lgbm search - tree data')


# In[40]:


# Support Vector Regressor

svr_param = {
    "kernel": ["poly",'rbf'],
    "C": [0.053677105521141605, 0.1],
    "epsilon": [0.03925943476562099, 0.1],
    "coef0": [0.9486751042886584, 0.5],
}

svr = SVR(
    kernel='rbf',
    degree=3,
    C=1.0,
    )


# In[41]:


# svr_grid = RandomizedSearchCV(svr, svr_param, scoring='neg_mean_squared_error', n_iter=5, random_state=1)

svr_grid = GridSearchCV(svr, svr_param, scoring='neg_mean_squared_error')

svr_linear = svr_grid.fit(X_train_linear, y_train)
svr_monotonic = svr_grid.fit(X_train_monotonic, y_train)
svr_tree = svr_grid.fit(X_train_tree, y_train)


# In[42]:


svr_linear.best_params_


# In[43]:


plot_grid(svr_linear, 'svr search - linear data')
plot_grid(svr_monotonic, 'svr search - monotonic data')
plot_grid(svr_tree, 'svr search - tree data')


# In[44]:


# gradient boosting regressor

rf_param = dict(
    n_estimators=[100, 200, 500, 1000],
    min_samples_split=[0.1, 0.3, 0.5, 1.0],
    max_depth=[1,2,3,None],
    )

rf = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=2,
    max_depth=3,
    random_state=0,
    n_jobs=-1,
    )


# In[45]:


rf_grid = RandomizedSearchCV(rf, rf_param, scoring='neg_mean_squared_error', n_iter=10, random_state=0)

# rf_grid = GridSearchCV(rf, rf_param, scoring='neg_mean_squared_error')

rf_linear = rf_grid.fit(X_train_linear, y_train)
rf_monotonic = rf_grid.fit(X_train_monotonic, y_train)
rf_tree = rf_grid.fit(X_train_tree, y_train)


# In[46]:


rf_linear.best_params_


# In[47]:


plot_grid(rf_linear, 'svr search - linear data')
plot_grid(rf_monotonic, 'svr search - monotonic data')
plot_grid(rf_tree, 'svr search - tree data')


# In[48]:


def select_best_score(grid):
    
    results = pd.DataFrame(grid.cv_results_)
    
    results.sort_values(by='mean_test_score', ascending=False, inplace=True)
    
    results = results[['mean_test_score', 'std_test_score']]
    
    return results.head(1)

# test function
select_best_score(rf_linear)


# In[49]:


results = pd.concat([
    select_best_score(gbm_linear),
    select_best_score(gbm_monotonic),
    select_best_score(gbm_tree),
    
    select_best_score(elastic_linear),
    select_best_score(elastic_monotonic),
    select_best_score(elastic_tree),
    
    select_best_score(knn_linear),
    select_best_score(knn_monotonic),
    select_best_score(knn_tree),
    
    select_best_score(svr_linear),
    select_best_score(svr_monotonic),
    select_best_score(svr_tree),
    
    select_best_score(lgbm_linear),
    select_best_score(lgbm_monotonic),
    select_best_score(lgbm_tree),
    
    select_best_score(rf_linear),
    select_best_score(rf_monotonic),
    select_best_score(rf_tree),
    ], axis=0)

results.index = [
    'gbm_linear','gbm_monotonic', 'gbm_tree',
    'elastic_linear','elastic_monotonic', 'elastic_tree',
    'knn_linear','knn_monotonic', 'knn_tree',
    'svr_linear','svr_monotonic', 'svr_tree',
    'lgbm_linear','lgbm_monotonic', 'lgbm_tree',
    'rf_linear','rf_monotonic', 'rf_tree',
]

results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.head()


# In[50]:


results['mean_test_score'].plot.bar(
    yerr=[results['std_test_score'], results['std_test_score']],
    subplots=True, figsize=(10,5))

plt.ylabel('Score')
plt.show()


# ## Compare performance

# In[51]:


# let's get the predictions from the elastic net
X_train_preds = elastic_monotonic.predict(X_train_monotonic)
X_test_preds = elastic_monotonic.predict(X_test_monotonic)
submission_preds = elastic_monotonic.predict(submission_monotonic)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))

my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_elastic.csv', index=False)


# In[52]:


# let's get the predictions from the SVR
X_train_preds = svr_monotonic.predict(X_train_monotonic)
X_test_preds = svr_monotonic.predict(X_test_monotonic)
submission_preds = svr_monotonic.predict(submission_monotonic)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# In[53]:


# let's get the predictions from the light GBM
X_train_preds = lgbm_tree.predict(X_train_tree)
X_test_preds = lgbm_tree.predict(X_test_tree)
submission_preds = lgbm_tree.predict(submission_tree)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))

my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_lgbm.csv', index=False)


# In[54]:


# let's get the predictions from the gradient boosting regressor
X_train_preds = gbm_tree.predict(X_train_tree)
X_test_preds = gbm_tree.predict(X_test_tree)
submission_preds = gbm_tree.predict(submission_tree)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# ## Model Stacking

# In[55]:


estimators = [
    ('gbm_linear',gbm_linear.best_estimator_),
    ('gbm_monotonic',gbm_monotonic.best_estimator_),
    ('gbm_tree',gbm_tree.best_estimator_),
    ('elastic_linear',elastic_linear.best_estimator_),
    ('elastic_monotonic',elastic_monotonic.best_estimator_),
    ('elastic_tree',elastic_tree.best_estimator_),
    ('knn_linear',knn_linear.best_estimator_),
    ('knn_monotonic',knn_monotonic.best_estimator_),
    ('knn_tree',knn_tree.best_estimator_),
    ('svr_linear',svr_linear.best_estimator_),
    ('svr_monotonic',svr_monotonic.best_estimator_),
    ('svr_tree',svr_tree.best_estimator_),
    ('lgbm_linear',lgbm_linear.best_estimator_),
    ('lgbm_monotonic',lgbm_monotonic.best_estimator_),
    ('lgbm_tree',lgbm_tree.best_estimator_),
    ('rf_linear',rf_linear.best_estimator_),
    ('rf_monotonic',rf_monotonic.best_estimator_),
    ('rf_tree', rf_tree.best_estimator_),
]

stacked = StackingRegressor(
    estimators=estimators,
    final_estimator=LGBMRegressor(random_state=1)
)


# In[56]:


stacking = cross_validate(
    stacked, X_train_linear, y_train, cv=5,
    scoring='neg_mean_squared_error', return_estimator=True)

stacking['test_score'].mean(), stacking['test_score'].std()


# In[57]:


stacking = cross_validate(
    stacked, X_train_monotonic, y_train, cv=5,
    scoring='neg_mean_squared_error', return_estimator=True)

stacking['test_score'].mean(), stacking['test_score'].std()


# In[58]:


stacking = cross_validate(
    stacked, X_train_tree, y_train, cv=5,
    scoring='neg_mean_squared_error', return_estimator=True)

stacking['test_score'].mean(), stacking['test_score'].std()


# In[59]:


stacked.fit(X_train_tree, y_train)


# In[60]:


# let's get the predictions from the stacked models

X_train_preds = stacked.predict(X_train_tree)
X_test_preds = stacked.predict(X_test_tree)
submission_preds = stacked.predict(submission_tree)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))

my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_stacke.csv', index=False)


# In[ ]:




