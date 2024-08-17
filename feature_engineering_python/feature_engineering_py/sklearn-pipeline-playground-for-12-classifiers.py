#!/usr/bin/env python
# coding: utf-8

# # Pipeline playground for 12+ classifiers (0.11860 LB)
# I want to **provide a clean, simple and beginner friendly template** that makes use of a flexible [scikit-learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). I aim to **automate the datacleaning and feature engineering** as much as possible while **allowing for fast iteration**. 
# 
# The **total training and processing time of the pipeline for 12 classifiers is around 20secs** (w/o scoring the models) and gets you **0.11860 on the Leaderboard** with practically no fuss.

# ## Imports and globals 🤖
# First we **import all the necessary libraries** and **set a base file path to the data sets**.

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

BASE_PATH = "/kaggle/input/house-prices-advanced-regression-techniques/"


# ## Superquick intro: What is a Pipeline?
# A pipeline is a [supercool class that scikit-learn provides](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), which allows to chain so called transformers (that – uhmmm.... – transform your data) with a final estimator at the end. Let's look a a simple example.  

# In[2]:


df = pd.read_csv(f"{BASE_PATH}train.csv")
X = df.select_dtypes("number").drop("SalePrice", axis=1)
y = df.SalePrice
pipe = make_pipeline(SimpleImputer(), RobustScaler(), LinearRegression())
print(f"The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}")


# ## <font color="darkred">How cool is that? 
# > With **only 5 lines of code we imported our training data, separated describing features from the target variable, setup a pipeline with an Imputer (that fills in missing values), a Scaler and a LinearRegression classifier. We crossvalidated and printed out the result.** 
#     
# It can't be easier than that I think...  
# 
# Now let's setup a pipeline that is able to **work on our categorical data as well.**

# In[3]:


num_cols = df.drop("SalePrice", axis=1).select_dtypes("number").columns
cat_cols = df.select_dtypes("object").columns

# we instantiate a first Pipeline, that processes our numerical values
numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', RobustScaler())])

# the same we do for categorical data
categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
# a ColumnTransformer combines the two created pipelines
# each tranformer gets the proper features according to «num_cols» and «cat_cols»
preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearRegression())])

X = df.drop("SalePrice", axis=1)
y = df.SalePrice
print(f"The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}")


# ## <font color="darkred">Even cooler... 
# > With **only 9 lines of code** we processed all our features automagically. Our score improves accordingly.
#     
# Now we expand this to a playground for all the regression classifiers you can think of.

# ## Choosing the estimators 🤓
# Since this is meant as a sandbox for experimentation we set a list with 10 common classifiers (and their respective names) which we will use in our pipeline. 
# 
# The initial hyperparameters I have [grid-searched](https://www.kaggle.com/chmaxx/extensive-data-exploration-modelling-python).

# In[4]:


# comment out all classifiers that you don't want to use
# and do so for clf_names accordingly
classifiers = [
               DummyRegressor(),
               LinearRegression(n_jobs=-1), 
               Ridge(alpha=0.003, max_iter=30), 
               Lasso(alpha=.0005), 
               ElasticNet(alpha=0.0005, l1_ratio=.9),
               KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
               SGDRegressor(),
               SVR(kernel="linear"),
               LinearSVR(),
               RandomForestRegressor(n_jobs=-1, n_estimators=350, 
                                     max_depth=12, random_state=1),
               GradientBoostingRegressor(n_estimators=500, max_depth=2),
               lgb.LGBMRegressor(n_jobs=-1, max_depth=2, n_estimators=1000, 
                                 learning_rate=0.05),
               xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, 
                                max_depth=2, n_estimators=1500, learning_rate=0.075),
]

clf_names = [
            "dummy", 
            "linear", 
            "ridge",
            "lasso",
            "elastic",
            "kernlrdg",
            "sgdreg",
            "svr",
            "linearsvr",
            "randomforest", 
            "gbm", 
            "lgbm", 
            "xgboost"
]


# ## Setting up the Pipeline 💡
# We will now setup simple functions to:
# *     **clean and prepare the data**
# *     **build the Pipeline**     
# *     **score models** 
# *     **train models**
# *     **predict from models**

# ### <span style="color:darkgreen">🐚 Encapsulate all our feature cleaning and engineering 
# To experiment, just add your code to this function.</span>

# In[5]:


def clean_data(data, is_train_data=True):
    # add your code for data cleaning and feature engineering here
    # e.g. create a new feature from existing ones
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    # add here the code that you only want to apply to your training data and not the test set
    # e.g. removing outliers from the training data works... 
    # ...but you cannot remove samples from your test set.
    if is_train_data == True:
        data = data[data.GrLivArea < 4000]
        
    return data


# ### <span style="color:darkgreen">👷‍♂️ Prepare our data for the pipeline 

# In[6]:


def prepare_data(df, is_train_data=True):
    
    # split data into numerical & categorical in order to process seperately in the pipeline 
    numerical   = df.select_dtypes("number").copy()
    categorical = df.select_dtypes("object").copy()
    
    # for training data only...
    # ...convert SalePrice to log values and drop "Id" and "SalePrice" columns
    if is_train_data == True :
        SalePrice = numerical.SalePrice
        y = np.log1p(SalePrice)
        numerical.drop(["Id", "SalePrice"], axis=1, inplace=True)
        
    # for the test data: just drop "Id" and set "y" to None
    else:
        numerical.drop(["Id"], axis=1, inplace=True)
        y = None
    
    # concatenate numerical and categorical data to X (our final training data)
    X = pd.concat([numerical, categorical], axis=1)
    
    # in addition to X and y return the separated columns to use these separetely in our pipeline
    return X, y, numerical.columns, categorical.columns


# ### <span style="color:darkgreen">👷‍♂️ Create the pipeline 

# In[7]:


def get_pipeline(classifier, num_cols, cat_cols):
    # the numeric transformer gets the numerical data acording to num_cols
    # the first step is the imputer which imputes all missing values to the mean
    # in the second step all numerical data gets scaled by the StandardScaler()
    numeric_transformer = Pipeline(steps=[
        ('imputer', make_pipeline(SimpleImputer(strategy='mean'))),
        ('scaler', StandardScaler())])
    
    # the categorical transformer gets all categorical data according to cat_cols
    # again: first step is imputing missing values and one hot encoding the categoricals
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # the column transformer creates one Pipeline for categorical and numerical data each
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    # return the whole pipeline with the classifier provided in the function call    
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])


# ## <span style="color:darkgreen">🌡 Score the models with crossvalidation 

# In[8]:


def score_models(df):
    # retrieve X, y and the seperate columns names
    X, y, num_cols, cat_cols = prepare_data(df)
    
    # since we converted SalePrice to log values, we use neg_mean_squared_error... 
    # ...rather than *neg_mean_squared_log_error* 
    scoring_metric = "neg_mean_squared_error"
    scores = []
    
    for clf_name, classifier in zip(clf_names, classifiers):
        # create a pipeline for each classifier
        clf = get_pipeline(classifier, num_cols, cat_cols)
        # set a kfold with 3 splits to get more robust scores. 
        # increase to 5 or 10 to get more precise estimations on models score
        kfold = KFold(n_splits=3, shuffle=True, random_state=1)  
        # crossvalidate and return the square root of the results
        results = np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=scoring_metric))
        scores.append([clf_name, results.mean()])

    scores = pd.DataFrame(scores, columns=["classifier", "rmse"]).sort_values("rmse", ascending=False)
    # just for good measure: add the mean of all scores to dataframe
    scores.loc[len(scores) + 1, :] = ["mean_all", scores.rmse.mean()]
    return scores.reset_index(drop=True)
    


# ## <span style="color:darkgreen">  🏋️‍♂️ Finally: Train the models
# For each classifier we create and fit a pipeline.

# In[9]:


def train_models(df): 
    X, y, num_cols, cat_cols = prepare_data(df)
    pipelines = []
    
    for clf_name, classifier in zip(clf_names, classifiers):
        clf = get_pipeline(classifier, num_cols, cat_cols)
        clf.fit(X, y)
        pipelines.append(clf)
    
    return pipelines


# ## <span style="color:darkgreen">🔮 Make predictions with trained models  
# For each fitted pipeline we retrieve predictions for SalePrice

# In[10]:


def predict_from_models(df_test, pipelines):
    X_test, _ , _, _ = prepare_data(df_test, is_train_data=False)
    predictions = []
    
    for pipeline in pipelines:
        preds = pipeline.predict(X_test)
        # we return the exponent of the predictions since we have log converted y for training
        predictions.append(np.expm1(preds))
    
    return predictions


# ## 🚀 And now: Let's use our pipeline... 
# 

# In[11]:


df = pd.read_csv(f"{BASE_PATH}train.csv")
df_test = pd.read_csv(f"{BASE_PATH}test.csv")

# We clean the data
df = clean_data(df)
df_test = clean_data(df_test, is_train_data=False)


# In[12]:


# We score the models on the preprocessed training data
my_scores = score_models(df)
display(my_scores)


# In[13]:


# We train the models on the whole training set and predict on the test data
models = train_models(df)
predictions = predict_from_models(df_test, models)
# We average over the results of all 12 classifiers (simple ensembling)
# we exclude the DummyRegressor and the SGDRegressor: they perform worst...
prediction_final = pd.DataFrame(predictions[2:]).mean().T.values

submission = pd.DataFrame({'Id': df_test.Id.values, 'SalePrice': prediction_final})
submission.to_csv(f"submission.csv", index=False)


# ### Have feedback? Found errors? Please let me know in the comments. 👌😎
