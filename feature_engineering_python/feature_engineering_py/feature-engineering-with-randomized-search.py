#!/usr/bin/env python
# coding: utf-8

# ## Finding the best data transformation with Randomized Search
# 
# In a [previous notebook](https://www.kaggle.com/solegalli/feature-engineering-pipeline-and-hyperparam-tuning), I made a grid search to optimize the hyperparameters of various feature engineering transformers and a gradient boosting classifier.
# 
# What if I am not sure which transformer to use to begin with? Can I also make a search to find the best transformation?
# 
# Yes, we can!
# 
# In this notebook, I will:
# 
# - assemble a feature engineering pipeline
# - automatically find out the best data transformation
# - train a Logistic Regression 
# 
# Using Randomized search.
# 
# We will:
# 
# - set up a series of feature engineering steps using [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html)
# - train a Logistic Regression
# - train the pipeline with cross-validation, looking over different feature-engineering transformation and model hyperparameters
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
from scipy import stats

# for the model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler,
    RobustScaler,
    )

# for feature engineering
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from feature_engine import discretisation as disc
from feature_engine import transformation as t


# ## Load the data

# In[3]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")

data.head()


# In[4]:


# the aim of this notebook is to show how to select the best data
# transformations

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
# we need these lists to tell Feature-engine which variables it should modify

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
# I want to assemble a pipeline that contains the following steps:
# 
# - impute numerical variables
# - impute categorical variables
# - encode categorical variables
# - either discretise or transform continuous variables
# - scale all variables
# - train a logistic regression
# 
# But I am unsure of the way to select the best imputation methods, the best encoding method, or if I should transform or discretise the continuous variables. 
# 
# Let's take it one step at the time.

# In[11]:


# Numerical imputation:
#----------------------

# Should I do mean, median imputation or imputation with an arbitrary value?

mean_imputer = mdi.MeanMedianImputer(imputation_method = 'mean', variables=['Age', 'Fare', 'cabin_num'])

median_imputer = mdi.MeanMedianImputer(imputation_method = 'median', variables=['Age', 'Fare', 'cabin_num'])

arbitrary_imputer = mdi.EndTailImputer(variables=['Age', 'Fare', 'cabin_num'])

num_imputer = [mean_imputer, median_imputer, arbitrary_imputer]


# In[12]:


# Categorical encoding

# Should I do one hot? ordinal imputation or mean encoding?

onehot_enc = ce.OneHotEncoder(variables=categorical)
ordinal_enc = ce.OrdinalEncoder(encoding_method='ordered', variables=categorical)
mean_enc = ce.MeanEncoder(variables=categorical)

cat_encoder = [onehot_enc, ordinal_enc, mean_enc]


# In[13]:


# Continuous variables

# should I discretise them or transform them?

efd = disc.EqualFrequencyDiscretiser(q=5, variables=continuous)
dtd = disc.DecisionTreeDiscretiser(variables=continuous)

yj = t.YeoJohnsonTransformer(variables=continuous)

transformers = [efd, dtd, yj]


# In[14]:


# finally, I want to scale the variables before passing them
# to the logit:

scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]


# In[15]:


# Now I set up the pipeline with some parameters.
# We will modify the steps later during the random search


titanic_pipe = Pipeline([

    # missing data imputation - numerical
    ('imputer_num', mean_imputer),
    
    # missind data imputation - categorical
    ('imputer_cat', mdi.CategoricalImputer(variables=['Embarked', 'cabin_cat'])),

    # categorical encoding - we will group rare categories into 1
    ('encoder_rare_label', ce.RareLabelEncoder(
        tol=0.01,
        n_categories=2,
        variables=['Embarked', 'cabin_cat'],
    )),
    
    # categorical encoding (into numbers)
    ('categorical_encoder', onehot_enc),
    
    # continuous variable transformation
    ('transformation', efd),
    
    # variable scaling
    ('scalers', StandardScaler),

    # Logistic regression
    ('logit', LogisticRegression(random_state=0))
])


# ## Random Search with Cross-validation
# 
# For hyperparameter search we need:
# 
# - the machine learning pipeline that we want to optimize (in the previous cell)
# - the hyperparamter space that we want to sample (next cell)
# - a metric to optimize
# - an algorithm to sample the hyperparameters (Random Search in this case)
# 
# Let's do that.

# In[16]:


# not we enter into the param_grid, the different options
# that we want to test

param_grid = {
    
    # test different numerical variable imputation
    'imputer_num': num_imputer,
    
    # test imputation with frequent category or string missing
    # we modify the paramater of the feature-engine transformer directly
    'imputer_cat__imputation_method': ['missing','frequent'],
    
    # test different thresholds to group rare labels
    # we modify the paramater of the feature-engine transformer directly
    'encoder_rare_label__tol': stats.uniform(0.1, 0.2),
    
    # test different encoding strategies
    'categorical_encoder': cat_encoder,
    
    # test different variable transformation strategies
    'transformation': transformers,
    
    # test different scalers
    'scalers': scalers,
    
    # try different logistic regression hyperparamenters
    'logit__C': stats.uniform(0, 1),
}

# (note how we call the step name in the pipeline followed by __
# followed by the name of the hyperparameter that we want to modify
# when we want to access directly the parameters inside the pipeline step)


# In[17]:


# now we set up the randomized search with cross-validation

search = RandomizedSearchCV(
    titanic_pipe, # the pipeline
    param_grid, # the hyperparameter space
    cv=3, # the cross-validation
    scoring='roc_auc', # the metric to optimize
    n_iter = 20, # the number of combinations to sample, 
)

# for more details in the randomized search visit:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html


# In[18]:


# and now we train over all the possible combinations of the parameters
# specified above

search.fit(X_train, y_train)

# and we print the best score over the train set
print(("best roc-auc from search: %.3f"
       % search.score(X_train, y_train)))


# In[19]:


# and finally let's check the performance over the test set

print(("best linear regression from grid search: %.3f"
       %search.score(X_test, y_test)))


# In[20]:


# we can find the best pipeline with its parameters like this

search.best_estimator_


# In[21]:


# and find the best fit parameters like this

search.best_params_


# In the previous cell we can see which were the data transformations that worked best.

# In[22]:


# we also find the data for all models evaluated

results = pd.DataFrame(search.cv_results_)

print(results.shape)

results.head()


# In[23]:


# we can order the different models based on their performance

results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)


# plot model performance and the generalization error

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score - ROC-AUC')
plt.xlabel('Hyperparameter combinations')


# If you liked this notebook and would like to know more about feature engineering and hyperparameter optimization feel free to check my [online courses](https://www.trainindata.com/).

# In[ ]:




