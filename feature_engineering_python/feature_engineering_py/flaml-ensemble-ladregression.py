#!/usr/bin/env python
# coding: utf-8

# <font color = 'blue'>
# Content: 
# 
# 
# 1. [Load Python Pakages and Data](#1)
# 2. [First look to data](#2)
# 3. [Exploratory Data Analysis](#3)   
# 4. [Feature Engineering](#4)   
# 5. [Preprocesing](#5)
#    * [A custom pipeline for Feature Engineering](#7)
# 6. [Putting pieces together](#8)
# 9. [Submission](#13)
# 
# 

# <a id = "1"></a><br>
# # Load Python Pakages
# 

# In[1]:


#basics
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

import warnings
warnings.filterwarnings("ignore")


#preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, quantile_transform


#statistics
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import randint

#feature engineering
from sklearn.feature_selection import mutual_info_regression


#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config


#algorithms
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


#model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


#stacking
from sklearn.ensemble import StackingRegressor




# <a id = "2"></a><br>
# #  First look to data

# In[2]:


# Read the data
train = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv', index_col=[0])
test = pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv', index_col=[0])
original = pd.read_csv('/kaggle/input/crab-age-prediction/CrabAgePrediction.csv')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e16/sample_submission.csv")

# reserved for pipeline
pipe_data = train.copy()
pipe_test = test.copy()
pipe_original = original.copy()

# use for preliminary analysis
train_df = train.copy()
test_df = test.copy()
original_df = original.copy()
train_df.head()


# In[3]:


test_df.head()


# In[4]:


original_df.head()


# In[5]:


original_df.index.names = ['id']
original_df.head()


# In[6]:


train_df = pd.concat([train_df, original_df])
train_df.head()


# In[7]:


# is there any missing value?
train_df.isnull().any()


# ## Descpriptive statistics

# In[8]:


#numerical feature descriptive statistics
train_df.describe().T


# ## Grouping features for preprocessing purposes

# In[9]:


train_df.nunique().sort_values()


# In[10]:


# Just bookkeeping
feature_list = [feature for feature in train_df.columns if not feature  == "Age"]
categorical_features= ['Sex']
numerical_features = list(set(feature_list) - set(categorical_features))

assert feature_list.sort() == (numerical_features + categorical_features).sort()


# <a id = "3"></a><br>
# # Exploratory Data Analysis

# In[11]:


fig, ax = plt.subplots(3, 3, figsize=(20, 20))
for var, subplot in zip(numerical_features, ax.flatten()):
    sns.scatterplot(x=var, y='Age',  data=train_df, ax=subplot, hue = 'Age' )
    


# In[12]:


# Display correlations between features and Age on heatmap.

sns.set(font_scale=1.1)
correlation_train = train_df.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False);


# In[13]:


y= train_df['Age']


# In[14]:


# determine the mutual information for numerical features
mutual_df = train_df[numerical_features]

mutual_info = mutual_info_regression(mutual_df, y, random_state=1)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["MI_score"] ).style.background_gradient("cool")


# In[15]:


#categorical features must be encoded to get mutual information
mutual_df_categorical = train_df[categorical_features]
for colname in mutual_df_categorical:
    mutual_df_categorical[colname], _ = mutual_df_categorical[colname].factorize()
mutual_info = mutual_info_regression(mutual_df_categorical, y, random_state=1)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df_categorical.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Categorical_Feature_MI"] ).style.background_gradient("cool")


# <a id = "4"></a><br>
# # Feature Engineering

# In[16]:


train_df ["volume"] = train_df["Height"] * train_df["Diameter"] * train_df["Length"]
train_df ["dim1"] = train_df["Height"] * train_df["Diameter"] 
train_df ["dim2"] = train_df["Height"] * train_df["Length"] 
train_df ["dim3"] = train_df["Diameter"] * train_df["Length"]
train_df ["total_weight"] = train_df["Shell Weight"] + train_df["Viscera Weight"] + train_df["Shucked Weight"]
train_df ["weight_volume_ratio"] = train_df["Weight"] / (train_df["Diameter"] + 1e-8 )
train_df ["shell_to_total_weight"] = train_df["Shell Weight"] / train_df["Weight"]
train_df ["viscera_to_total_weight"] = train_df["Viscera Weight"] / train_df["Weight"]
train_df ["shucked_to_total_weight"] = train_df["Shucked Weight"] / train_df["Weight"]




new_features = ["volume", 'dim1', 'dim2', 'dim3', 'total_weight', 'weight_volume_ratio', 'shell_to_total_weight','viscera_to_total_weight','shucked_to_total_weight']


# Let's check new features mutual information scores...

# In[17]:


mutual_df = train_df[new_features]

mutual_info = mutual_info_regression(mutual_df, y, random_state=1)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["New_Feature_MI"] ).style.background_gradient("cool")


# In[18]:


fig, ax = plt.subplots(2, 2, figsize=(20, 20))
for var, subplot in zip(new_features, ax.flatten()):
    sns.scatterplot(x=var, y='Age',  data=train_df, ax=subplot, hue = 'Age')


# In[19]:


categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[20]:


#tree preprocessor
tree_preprocessor = ColumnTransformer(remainder='passthrough',
    transformers=[
        ('categorical_transformer', categorical_transformer, categorical_features)

    ])

tree_preprocessor


# <a id = "7"></a><br>
# ## A custom pipeline for Feature Engineering

# In[21]:


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        
        self.add_attributes = add_attributes
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        if self.add_attributes:
            X_copy = X.copy()
            
            
            X_copy ["volume"] = X_copy["Height"] * X_copy["Diameter"] * X_copy["Length"]
            X_copy ["dim1"] = X_copy["Height"] * X_copy["Diameter"] 
            X_copy ["dim2"] = X_copy["Height"] * X_copy["Length"] 
            X_copy ["dim3"] = X_copy["Diameter"] * X_copy["Length"]
            X_copy ["total_weight"] = X_copy["Shell Weight"] + X_copy["Viscera Weight"] + X_copy["Shucked Weight"]
            X_copy ["weight_volume_ratio"] = X_copy["Weight"] / (X_copy["Diameter"] + 1e-8 )
            X_copy ["shell_to_total_weight"] = X_copy["Shell Weight"] / X_copy["Weight"]
            X_copy ["viscera_to_total_weight"] = X_copy["Viscera Weight"] / X_copy["Weight"]
            X_copy ["shucked_to_total_weight"] = X_copy["Shucked Weight"] / X_copy["Weight"]
            
            return X_copy
        else:
            return X_copy


# In[22]:


Creator = FeatureCreator(add_attributes = True)


# <a id = "8"></a><br>
# # Putting pieces together

# In[23]:


pipe_original.index.names = ['id']
pipe_original.head()


pipe_data = pd.concat([pipe_data, pipe_original])
pipe_data.info()



# In[24]:


y = pipe_data['Age']
pipe_data = pipe_data.drop('Age', axis=1)
pipe_data.head()


# In[25]:


pip install flaml


# In[26]:


#flaml
from flaml import AutoML


# In[27]:


pip install sklego


# In[28]:


from sklego.linear_model import LADRegression


# In[29]:


automl = AutoML()

automl_pipeline = Pipeline([
    ('Creator', Creator),
    ('tree_preprocessor', tree_preprocessor),
    ("automl", automl)
])
automl_pipeline


# In[30]:


# Specify automl goal and constraint
automl_settings = {
    "time_budget": 10800,  # total running time in seconds
    "task": 'regression',  # task type
    "seed": 24545678,  # random seed
    "metric" : 'mae',
    "eval_method" : 'cv',
    "n_splits" : 5,
    "ensemble" : True,
        "ensemble": {
        "final_estimator": LADRegression(),
        "passthrough": True,
    },

    
}

pipeline_settings = {f"automl__{key}": value for key, value in automl_settings.items()}


# In[31]:


automl_pipeline = automl_pipeline.fit(pipe_data, y, **pipeline_settings)


# In[32]:


preds_test =  automl_pipeline.predict(pipe_test)


# In[33]:


preds_test = [round(x) for x in preds_test]


# <a id = "13"></a><br>
# # Submission

# In[34]:


output = pd.DataFrame({'id': pipe_test.index,
                       'Age': preds_test})
output.to_csv('submission.csv', index=False)


# In[35]:


output.head()

