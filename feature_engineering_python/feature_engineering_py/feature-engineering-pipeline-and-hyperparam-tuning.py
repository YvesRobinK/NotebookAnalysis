#!/usr/bin/env python
# coding: utf-8

# ## Assembling a Feature Engineering Pipeline with Hyperparameter Optimization
# 
# In this notebook, I will assemble a feature engineering pipeline followed by a Gradient Boosting Classifier. I will search for the best hyperparameters both for the machine learning model and the feature engineering steps using Grid Search.
# 
# In summary, we will:
# 
# - set up a series of feature engineering steps using [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html)
# - train a Gradient Boosting Classifier
# - train the pipeline with cross-validation, looking over different feature-engineering and model hyperparameters
# 
# For more details on feature engineering and hyperparameter optimization feel free to check my [online courses](https://www.trainindata.com/).

# In[1]:


# Let's install Feature-engine
# an open-source Python library for feature engineering
# it allows us to select which features we want to transform
# straight-away from within each transformer

# https://feature-engine.readthedocs.io/en/latest/index.html

get_ipython().system('pip install feature-engine')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# for feature engineering
from feature_engine import imputation as mdi
from feature_engine import encoding as ce


# ## Load the data

# In[3]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")

data.head()


# In[4]:


# the aim of this notebook is to show how to optimize hyperparameters of an
# entire machine learning pipeline.

# So I will take a shortcut and remove some features to make things simpler

cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked', 'Survived'
]

data = data[cols]

data.head()


# In[5]:


# Cabin: extract numerical and categorical part and delete original variable

data['cabin_num'] = data['Cabin'].str.extract('(\d+)') # captures numerical part
data['cabin_num'] = data['cabin_num'].astype('float')
data['cabin_cat'] = data['Cabin'].str[0] # captures the first letter

data.drop(['Cabin'], axis=1, inplace=True)

data.head()


# In[6]:


# make list of variables types
# we need these lists to indicate Feature-engine which variables it should modify

# numerical: discrete
discrete = [
    var for var in data.columns if data[var].dtype != 'O' and var != 'Survived'
    and data[var].nunique() < 10
]

# numerical: continuous
continuous = [
    var for var in data.columns
    if data[var].dtype != 'O' and var != 'Survived' and var not in discrete
]

# categorical
categorical = [var for var in data.columns if data[var].dtype == 'O']

print('There are {} discrete variables'.format(len(discrete)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(categorical)))


# In[7]:


# discrete variables

discrete


# In[8]:


# continuous variables

continuous


# In[9]:


# categorical variables

categorical


# In[10]:


# separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Survived', axis=1),  # predictors
    data['Survived'],  # target
    test_size=0.1,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# ### Set up the pipeline
# 
# We assemble a pipeline with default or some hyperparameters for each step. But we will modify this during the hyperparameter search later on.

# In[11]:


titanic_pipe = Pipeline([

    # missing data imputation - we replace na in numerical variables
    # with an arbitrary value. 
    ('imputer_num',
     mdi.ArbitraryNumberImputer(arbitrary_number=-1,
                                variables=['Age', 'Fare', 'cabin_num'])),
    
    # for categorical variables, we can either replace na with the string
    # missing or with the most frequent category
    ('imputer_cat',
     mdi.CategoricalImputer(variables=['Embarked', 'cabin_cat'])),

    # categorical encoding - we will group rare categories into 1
    ('encoder_rare_label', ce.RareLabelEncoder(
        tol=0.01,
        n_categories=2,
        variables=['Embarked', 'cabin_cat'],
    )),
    
    # we replace category names by numbers
    ('categorical_encoder', ce.OrdinalEncoder(
        encoding_method='ordered',
        variables=['cabin_cat', 'Sex', 'Embarked'],
    )),

    # Gradient Boosted machine
    ('gbm', GradientBoostingClassifier(random_state=0))
])


# ## Grid Search with Cross-validation
# 
# For hyperparameter search we need:
# 
# - the machine learning pipeline that we want to optimize (in the previous cell)
# - the hyperparamter space that we want to sample (next cell)
# - a metric to optimize
# - an algorithm to sample the hyperparameters (Grid Search in this case)
# 
# Let's do that.

# In[12]:


# now we create the hyperparameter space that we want to sample
# that is, the hyperparameter values that we want to test.

# to perform Grid search, we need to specifically provide the hyperparameter
# values that we want to test

# to opimize hyperparameters within a pipeline, we assemble the space
# as follows:

param_grid = {
    # try different feature engineering parameters:
    
    # test different parameters to replace na with numbers
    'imputer_num__arbitrary_number': [-1, 99],
    
    # test imputation with frequent category or string missing
    'imputer_cat__imputation_method': ['missing','frequent'],
    
    # test different thresholds to group rare labels
    'encoder_rare_label__tol': [0.1, 0.2],
    
    # test 2 different encoding strategies
    'categorical_encoder__encoding_method': ['ordered', 'arbitrary'],
    
    # try different gradient boosted tree model paramenters
    'gbm__max_depth': [None, 1, 3],
    'gbm__n_estimators': [10, 20, 50, 100, 200]
}

# (note how we call the step name in the pipeline followed by __
# followed by the name of the hyperparameter that we want to modify)

# for more details on the Feature-engine transformers hyperparameters, visit
# Feature-engine documentation


# In[13]:


# now we set up the grid search with cross-validation

# we are optimizing over few hyperparameters altogether, so 
# a GridSearch should be more than enough

grid_search = GridSearchCV(
    titanic_pipe, # the pipeline
    param_grid, # the hyperparameter space
    cv=3, # the cross-validation
    scoring='roc_auc', # the metric to optimize
)

# for more details in the grid parameters visit:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


# In[14]:


# and now we train over all the possible combinations of the parameters
# specified above
grid_search.fit(X_train, y_train)

# and we print the best score over the train set
print(("best roc-auc from grid search: %.3f"
       % grid_search.score(X_train, y_train)))


# In[15]:


# and finally let's check the performance over the test set

print(("best linear regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))


# In[16]:


# we can find the best pipeline with its parameters like this

grid_search.best_estimator_


# In[17]:


# and find the best fit parameters like this

grid_search.best_params_


# In[18]:


# we also find the data for all models evaluated

results = pd.DataFrame(grid_search.cv_results_)

print(results.shape)

results.head()


# In[19]:


# we can order the different models based on their performance

results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)


# plot model performance and the generalization error

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score - ROC-AUC')
plt.xlabel('Hyperparameter combinations')


# In[20]:


# and to wrap up:
# let's explore the importance of the features

importance = pd.Series(grid_search.best_estimator_['gbm'].feature_importances_)
importance.index = data.drop('Survived', axis=1).columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))


# If you liked this notebook and would like to know more about feature engineering and hyperparameter optimization feel free to check my [online courses](https://www.trainindata.com/).

# In[ ]:




