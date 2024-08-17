#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# In this notebook we will predict the house prices located in Ames, Iowa. The biggest challenge in this project is the number of columns. There are total of 81 columns in the training dataset out of which 79 columns just contains the features of the houses. Let's start this project and make our predictions.
# 

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

# ## Importing Libraries

# In[1]:


# Importing Libraries
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import  KFold, cross_val_score
from xgboost import XGBRegressor

# Setting matplotlib defaults 
plt.style.use('seaborn-whitegrid')
plt.rc('figure' , autolayout = True)
plt.rc(
    'axes',
    labelweight = 'bold',
    labelsize = 'large',
    titleweight = 'bold',
    titlesize = 14,
    titlepad = 10
 
  )

#Mute warnings
warnings.filterwarnings('ignore')


# # DATA PREPROCESSING
# 
# Before we train our model we have to prepare our data - we have to fill the missing values, we have to make sure that our data is in numerical form etc.

# ## Loading Data 
# 
# First we will read the data given to us in the competition datasets. We will use a function to load/read the data.

# In[2]:


def load_data():
    #Reading data
    data_dir = Path("../input/house-prices-advanced-regression-techniques")
    df_train = pd.read_csv(data_dir / 'train.csv' , index_col = 'Id')
    df_test = pd.read_csv(data_dir / 'test.csv' , index_col = 'Id')
    
    #Merging the train and test to process them together
    df = pd.concat([df_train , df_test])
    
    # Preprocessing 
    df = clean(df)
    df = encode(df)
    df = impute(df)
    
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index , :]
    return df_train , df_test
    
    


# ## Clean Data 
# 
# Some of the categorical features in this dataset have what are apparently typos in their categories:

# In[3]:


data_dir = Path("../input/house-prices-advanced-regression-techniques")
df = pd.read_csv(data_dir / 'train.csv' , index_col = 'Id')

df.Exterior2nd.unique()


# We can see the values in the Exterior2nd columns for example. These  are short forms but they look like typos.

# In[4]:


# Cleaning data
def clean(df):
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn' : 'BrkComm'})
    
    # Some values of GarageYrBlt are corrupt, so we'll replace them  with the year the house was built
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt <= 2010 , df.YearBuilt)
    
    # Names beginning with numbers are awkward to work with
    df.rename(columns = {
        '1stFlrSF' : 'FirstFlrSF',
        '2ndFlrSF' : 'SecondFlrSF',
        '3SsnPorch' : 'Threeseasonporch'
    }, inplace = True,)
    return df


# ## Encoding Data
# 

# In[5]:


# The nominative (unordered) categorical features
features_nom = ['MSSubClass' , 'MSZoning' , 'Street' , 'Alley' , 'LandContour' , 'LotConfig' , 'Neighborhood' ,'Condition1' ,
               'Condition2' , 'BldgType' , 'HouseStyle' , 'RoofStyle' , 'RoofMatl' , 'Exterior1st' ,'Exterior2nd' , 'MasVnrType',
               'Foundation' ,'Heating' ,'CentralAir' , 'GarageType' ,'MiscFeature' , 'SaleType' , 'SaleCondition']

# Pandas calls the categories "levels"
five_levels = ['Po' , 'Fa' , 'Ta' , 'Gd' , 'Ex']
ten_levels = list(range(10))

ordered_levels = {
    'OverallQual' : ten_levels,
    'OverallCond' : ten_levels,
    'ExterQual' :  five_levels,
    'ExterCond' :  five_levels,
    'BsmtQual' : five_levels,
    'BsmtCond' : five_levels,
    'HeatingQC' : five_levels,
    'KitchenQual' : five_levels,
    'FireplaceQu' : five_levels,
    'GarageQual' : five_levels,
    'GarageCond' : five_levels,
    'PoolQC' : five_levels,
    'LotShape' : ['Reg' , 'IR1' , 'IR2' , 'IR3'],
    'LandSlope' : ['Sev' , 'Mod' , 'Gtl'],
    'BsmtExposure' : ['No' , 'Mn' , 'Av' , 'Gd'],
    'BsmtFinType1' : ['Unf' , 'LwQ' , 'Rec' , 'BLQ' , 'ALQ' , 'GLQ'],
    'BsmtFinType2' : ['Unf' , 'LwQ' , 'Rec' , 'BLQ' , 'ALQ' , 'GLQ'],
    'Functional' : ['Sal' , 'Sev' , 'Maj2' , 'Maj1' , 'Mod' , 'Min2' , 'Min1' , 'Typ'],
    'GarageFinish' : ['Unf' , 'RFn' , 'Fin'],
    'PavedDrive' : ['N' , 'P' , 'Y'],
    'Utilities' : ['NoSeWa' , 'ELO' , 'NoSewr' , 'AllPub'],
    'CentralAir' : ['N' , 'Y'],
    'Electrical' : ['Mix' , 'FuseP' , 'FuseF' , 'FuseA' , 'SBrkr'],
    'Fence' : ['MnWw' , 'GdWo' , 'MnPrv' , 'GdPrv']
    
}

# Add a None level for missing values
ordered_levels = {key: ['None'] + value for key , value in ordered_levels.items()}

# Defining encoding function
def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype('category')
        # Add a None category for missing values
        if 'None' not in df[name].cat.categories:
            df[name].cat.add_categories('None' , inplace = True)
            
     # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype (levels , ordered = True))
    return df


# ## Handling missing values
# 
# We'll impute 0 for missing numeric values and "None" for missing categorical values.

# In[6]:


# Defining impute function
def impute(df):
    for name in df.select_dtypes('number'):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes('category'):
        df[name] = df[name].fillna('None')
    return df
    


# ## Loading Data
# 
# And now we can call the data loader and get the processed data splits.

# In[7]:


# getting our train and test datasets
df_train , df_test = load_data()


# In[8]:


# Viewing values
display(df_train)
print(' ')
print('-'*20)
print(' ')
display(df_test)


# In[9]:


# Display information about dtypes and missing values
display(df_train.info())
print(' ')
print('-'*20)
print(' ')
display(df_test.info())


# ## Establishing Baseline
# 
# Now we will establish a baseline score to judge our feature engineering against.

# In[10]:


# Function for calculating baseline score
def score_dataset(X , y , model = XGBRegressor()):
    
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    log_y = np.log(y)
    score = cross_val_score(model , X , log_y , cv = 5 , scoring = 'neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# In[11]:


# Getting baseline score
X = df_train.copy()
y = X.pop('SalePrice')

baseline_score = score_dataset(X,y)
print(f"Baseline_score: {baseline_score: .5f}  RMSLE")


# # Feature Utility Score
# 
# Feature Utility Score gives us an indication of how much potential that particular feature has.

# In[12]:


# Function for making mi_scores
def make_mi_scores(X,y):
    X = X.copy()
    for colname in X.select_dtypes(['object' , 'category']):
        X[colname] , _= X[colname].factorize()
     # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X , y , discrete_features = discrete_features , random_state = 0)
    mi_scores = pd.Series(mi_scores , name = 'MI Scores' , index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores

# Function for plotting mi_scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width , scores)
    plt.yticks(width ,  ticks)
    plt.title('Mutual Information Scores')
    


# In[13]:


# Calling make_mi_score
X = df_train.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X,y)
mi_scores


# The features that have 0 mi_score we don't have any use for them, therefore we will remove them.

# In[14]:


# function for dropping uninformation mi_scores
def drop_uninformative(df , mi_scores):
    return df.loc[: , mi_scores> 0.0]


# In[15]:


X = df_train.copy()
y = X.pop('SalePrice')
# Calling drop_uninformative
X = drop_uninformative(X , mi_scores)


# In[16]:


#Making new baseline score
score_dataset(X,y)


# # Create Features
# 
# Now we'll start developing our feature set.

# In[17]:


# Label Encoding
def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    return X


# ### Creating Features with Pandas
# 
# 

# In[18]:


def mathematical_transforms(df):
    X = pd.DataFrame()
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X

def interactions(df):
    X = pd.get_dummies(df.BldgType , prefix = 'Bldg')
    X = X.mul(df.GrLivArea , axis = 0)
    return X

def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[[
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        'Threeseasonporch',
        'ScreenPorch'
     ]].gt(0.0).sum(axis = 1)
    return X

def break_down(df):
    X = pd.DataFrame()
    X['MSClass'] = df.MSSubClass.str.split('_' , n =1 , expand = True)[0]
    return X 

def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    return X
    


# # K - Means Clustering

# In[19]:


# K - Means Clustering
cluster_features = [
    'LotArea' ,
    'TotalBsmtSF' ,
    'FirstFlrSF' ,
    'SecondFlrSF' ,
    'GrLivArea'
]

def cluster_labels(df , features , n_clusters = 20):
    X = df.copy()
    X_scaled = X.loc[: , features]
    X_scaled = (X_scaled - X_scaled.mean(axis = 0)) / X_scaled.std(axis = 0)
    kmeans = KMeans(n_clusters = n_clusters , n_init = 50 , random_state = 0)
    X_new = pd.DataFrame()
    X_new['Cluster'] = kmeans.fit_predict(X_scaled)
    return X_new

def cluster_distance(df , features , n_clusters = 20):
    X = df.copy()
    X_scaled = X.loc[: , features]
    X_scaled = (X_scaled - X_scaled.mean(axis = 0)) / X_scaled.std(axis = 0)
    kmeans = KMeans(n_clusters = n_clusters , n_init = 50 , random_state = 0)
    X_cd = kmeans.fit_transform(X_scaled)
    X_cd = pd.DataFrame(
        X_cd , columns = [f'centroid_{i}' for i in range(X_cd.shape[1])]
      )
    return X_cd


# ## Principal Component Analysis

# In[20]:


# Principal Component Analysis
def apply_pca(X  ,standardize = True):
    if standardize:
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    #Creating Principal Components
    pca = PCA()
    X_pca  = pca.fit_transform(X)
    # Convert to DataFrame
    component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca , columns = component_names)
    # Create Loadings
    loadings = pd.DataFrame(
        pca.components_.T , # transpose the matrix of loadings
        columns = component_names,  # so the columns are the principal components
        index = X.columns         # and the rows are the original features
        
      )
    return pca , X_pca , loadings

def plot_variance(pca , width = 8 , dpi = 100):
    #Creating figure
    fig , axis = plt.subplots(1,2)
    n = pca.n_components_
    grid = np.arange(1 , n+1 )
    # Explained Variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid , evr)
    axs[0].set(
        xlabel = 'Component' , title = '%Explained Variance' , ylim = (0.0 , 1.0)
        )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0 , grid] , np.r[0 , cv] , 'o-')
    axs[1].set(
        xlabel = 'Component' , title = '%Cumulative Variance' , ylim = (0.0 , 1.0)
    )
    # Set up figure
    fig.set(figwidth = 8 , dpi = 100)
    return axs
    


# In[21]:


# Transforms
def pca_inspired(df):
    X = pd.DataFrame()
    X['Feature1'] = df.GrLivArea + df.TotalBsmtSF
    X['Feature2'] = df.YearRemodAdd * df.TotalBsmtSF
    return X

def pca_components(df , features):
    X = df.loc[: , features]
    _ , X_pca , _ = apply_pca(X)
    return X_pca

pca_features = [
    'GarageArea',
    'YearRemodAdd' ,
    'TotalBsmtSF' , 
    'GrLivArea'  
]


# In[22]:


# Correlation Matrix
def corrplot(df , method = 'pearson' , annot = True , **kwargs):
    sns.clustermap(
        df.corr(method),
        vmin = -1.0,
        vmax = 1.0,
        cmap = 'icefire',
        method = 'complete',
        annot = annot , 
        **kwargs
    )
    
corrplot(df_train , annot= None)


# ### PCA Application - Indicate Outliers

# In[23]:


def indicate_outliers(df):
    X_new = pd.DataFrame()
    X_new['Outlier'] = (df.Neighborhood == 'Edwards') & (df.SaleCondition == 'Partial')
    return X_new


# ### Target Encoding
# 
# To do the target encoding we will- 
# 
# 
# 1.) Split the data into folds, each fold having two splits of the dataset.
# 
# 
# 2.) Train the encoder on one split but transform the values of the other.
# 
# 
# 3.) Repeat for all the splits.
# 

# In[24]:


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


# ### Creating Final Feature Set

# In[25]:


def create_features(df , df_test = None):
    X = df.copy()
    y = X.pop('SalePrice')
    mi_scores = make_mi_scores(X,y)
    
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop('SalePrice')
        X = pd.concat([X, X_test])
    # Mutual Information
    X = drop_uninformative(X, mi_scores)
    
    # Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    X = X.join(break_down(X))
    X = X.join(group_transforms(X))
    
    # Clustering
    X = X.join(cluster_labels(X, cluster_features , n_clusters = 20))
    X = X.join(cluster_distance(X, cluster_features , n_clusters = 20))
    
    # PCA
    X = X.join(pca_inspired(X))
    X = X.join(pca_components(X , pca_features))
    X = X.join(indicate_outliers(X))
    
    X = label_encode(X)
    
    # Reform Splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index , inplace = True)
        
    # Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder , m = 1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))
        
    if df_test is not None:
        return X , X_test
    else:
        return X
    
df_train , df_test = load_data()
X_train = create_features(df_train)
y_train = df_train.loc[:,'SalePrice']

score_dataset(X_train  , y_train)
        


# # Hyperparameter Tuning

# In[26]:


X_train = create_features(df_train)
y_train = df_train.loc[:, 'SalePrice']

xgb_params = dict(
    max_depth = 9,
    learning_rate = 0.01,
    n_estimators = 3000,
    min_child_weight = 1,
    colsample_bytree = 0.7,
    subsample = 0.7,
    reg_alpha = 0.3,
    reg_lambda = 1.0,
    num_parallel_tree = 1,
)
xgb = XGBRegressor(**xgb_params)
score_dataset(X_train, y_train , xgb)


# # Training Model and making Submission

# In[27]:


X_train , X_test = create_features(df_train , df_test)
y_train = df_train.loc[:, 'SalePrice']

xgb = XGBRegressor(**xgb_params)
# XGB minimizes MSE, but competition loss is RMSLE  So, we need to log-transform y to train and exp-transform the predictions
xgb.fit(X_train , np.log(y))
predictions = np.exp(xgb.predict(X_test))

output = pd.DataFrame({'Id': X_test.index , 'SalePrice' : predictions})
output.to_csv('my_submission.csv' , index = False)
print('Submission is successfully saved')


# # Thank You 
# Thank you if you have come this far, please give this notebooks an upvote as it will encourage me to make more notebooks like these. 
# 
# **Reference**
# 
# https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices/notebook

# In[ ]:





# In[ ]:




