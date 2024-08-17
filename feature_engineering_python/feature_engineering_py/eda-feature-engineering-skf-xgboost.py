#!/usr/bin/env python
# coding: utf-8

# # House Prices - EDA, FE & StratifiedKFold XGBoost
# 
# Welcome! [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) is a perfect competition to practise a variety of data analysis methods and machine learning models. In this notebook, I would like to share with you:
# 
# * Exploratory data analysis
# 
# * Feature engineering
# 
# * StratifiedKFold CV split
# 
# * XGBoost model training and inference
# 
# If you have any questions or suggestions, please leave a comment. Please upvote this notebook if you like it! Thank you.

# # Imports and Loading Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

plt.style.use('ggplot')
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)


# Let's load the data and show the first 5 rows

# In[2]:


df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()


# In[3]:


print(df.shape)
print(df.columns)


# The goal is to build a model to predict a house price taking these 80 features as inputs. We have 1460 examples and no duplicated data.

# In[4]:


len(df.loc[df.duplicated()])


# # Exploratory Data Analysis
# 
# ## Numetic Features

# In[5]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_df.describe()


# Some features don't have the full 1460 values. The missing value won't be calculated in the features' statistics. 

# In[6]:


numeric_df.isna().sum()


# 'LotFrontage' has 259 missing values. 'MasVnrArea' has 8. 'GarageYrBlt' has 81.
# 
# To understand each feature's meaning, we can check "data_description.txt" in the competition data tab. Here I extract some lines.
# 
# > OverallQual: Rates the overall material and finish of the house
# >
# > 1stFlrSF: First Floor square feet
# >
# > GrLivArea: Above grade (ground) living area square feet
# >
# > TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 
# These numeric features probably have considerable influences on house prices according to common sense. We will explore them next.

# In[7]:


def num_tar_plot(df, numeric, target='SalePrice'):
    ax = df.plot(kind='scatter', 
                 x=numeric, 
                 y=target, 
                 figsize=(15,5),
                 title=numeric+' VS '+target)
    plt.show()


# In[8]:


features_to_plot = ['OverallQual', '1stFlrSF', \
                    'GrLivArea', 'TotRmsAbvGrd' ]
for feature in features_to_plot:
    num_tar_plot(df, feature)


# From the scatter plots, we can see clearly that these 4 features all have positive correlations with the sale prices.
# 
# ## Categorical Features
# 
# 

# In[9]:


cat_df = df.select_dtypes(include=['object'])
cat_df.describe()


# In[10]:


cat_df.isna().sum()


# For most categorical features, encoding them as one-hot vectors is best practice, such as: 'MSZoning', 'Neighborhood', 'HouseStyle', 'SaleType'. We plot box of them and encode them using scikit-learn OneHotEncoder later in the training process.

# In[11]:


def cat_tar_plot(df, cat, target='SalePrice'):
    ax = df.boxplot(column=target, by=cat, 
                    rot=90, figsize=(15,5))
    plt.show()


# In[12]:


features_to_plot = ['MSZoning', 'Neighborhood', \
                    'HouseStyle', 'SaleType' ]
for feature in features_to_plot:
    cat_tar_plot(df, feature)


# # Feature Engineering
# 
# Some categorical features have linear relationship in their values. We can convert them to integers, handle them as numerical features. Furthermore, we replace 'NA' value with 0. It's reasonable to do that because 'NA' means 'no basement' or 'no garage' according to the data description file.
# 

# In[13]:


def cat2num(df):
    df['BsmtQual'] = df['BsmtQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).fillna(0)
    df['BsmtCond'] = df['BsmtCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).fillna(0)
    df['BsmtExposure'] = df['BsmtExposure'].map({'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).fillna(0)
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}).fillna(0)
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}).fillna(0)
    df['GarageQual'] = df['GarageQual'].map({'Po': 1,'Fa': 2,'TA': 3,'Gd': 4,'Ex': 5}).fillna(0)
    df['GarageCond'] = df['GarageCond'].map({'Po': 1,'Fa': 2,'TA': 3,'Gd': 4,'Ex': 5}).fillna(0)
    return df

df = cat2num(df)


# In[14]:


features_to_plot = ['BsmtQual', 'BsmtCond', \
                    'BsmtExposure', 'BsmtFinType1' ]
for feature in features_to_plot:
    cat_tar_plot(df, feature)


# We can see from above figures, with the value of each feature increases, the interquartile (box) rises. We hope the conversion will help our model learn patterns easier.
# 
# Additionally, I create 3 new features: 'bedrooms_per_room', 'fireplace_per_room', 'live_area_ratio'. Their definitions are self-explanatory. I put them in 'feature_e' function. Hopefully, they would provide more useful information for the model.

# In[15]:


def feature_e(df):
    df['bedrooms_per_room'] = df['BedroomAbvGr'] / df['TotRmsAbvGrd']
    df['fireplaces_per_room'] = df['Fireplaces'] / df['TotRmsAbvGrd']
    df['live_area_ratio'] = df['GrLivArea'] / df['LotArea']
    return df

df = feature_e(df)


# # StratifiedKFold CV Split
# 
# Now let's look at the target: 'SalePrice'.

# In[16]:


ax = df['SalePrice'].plot(kind='hist', bins=15,
                          figsize=(15,5),
                          title='SalePrice Distribution')
ax.set_xlabel('SalePrice')
plt.show()


# We separate 'SalePrice' into 15 quantile range as bins, save to 'SalePrice_Bin' column, then stratified split examples into 4 folds by 'SalePrice_Bin'.

# In[17]:


df['SalePrice_Bin'] = pd.qcut(x=df['SalePrice'], q=15, labels=False)
df['SalePrice_Bin'].value_counts().sort_index()


# In[18]:


N_FOLD = 4
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(skf.split(df, df['SalePrice_Bin'])):
    df.loc[val_index, 'fold'] = int(n)
df['fold'] = df['fold'].astype(int)
df['fold'].value_counts()


# In[19]:


df.to_csv('./skf_split.csv', index=False)


# In[20]:


plt.figure(figsize=(15, 10))
for fold in range(N_FOLD):
    plt.subplot(2, 2, fold + 1)
    plt.title(f"FOLD {fold} SalePrice Distribution")
    df[df['fold'] == fold]['SalePrice'].plot.hist()
plt.tight_layout()
plt.show()


# 'SalePrice' distributions of the 4 folds are roughly balanced so we can expect:
# 
# * Every model in cross validation will generalize well through the whole price range.
# 
# * Out-of-fold metric will be a reliable criterion, and will align with the test score.
# 
# # XGBoost Training
# 
# Most features are included, some with missing values are dropped.

# In[21]:


target = 'SalePrice'

numeric_features = \
    [feature for feature in df.select_dtypes(include=['int64', 'float64']).columns.values \
     if feature not in ['Id', 'MSSubClass', 'LotFrontage', 'MasVnrArea', \
                        'GarageYrBlt', 'MoSold', 'SalePrice', 'SalePrice_Bin', 'fold']
    ]

cat_features = \
    [feature for feature in df.select_dtypes(include=['object']).columns.values \
     if feature not in ['Alley', 'MasVnrType',\
                        'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', \
                        'PoolQC', 'Fence', 'MiscFeature']
    ]

print(f'Numerical features: {numeric_features}')
print(f'\nCategorical features: {cat_features}')


# To compare with test score, we define root mean squared logarithm error as our metric.

# In[22]:


def rmse_log(y_true, y_pred):
    return mean_squared_error(np.log(y_true+1), np.log(y_pred+1), squared=False)


# Cross validation training loop starts. XGBoost hyperparameters can be tuned to get the best out-of-fold RMSLE.

# In[23]:


# Fit the one-hot encoder
OH_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_enc.fit(df[cat_features])

rmsles = []
models = []

for fold in range(N_FOLD):
    print(f'\n-----------FOLD {fold} ------------')
    
    #Create dataset
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    
    # One-hot encode categorical features
    cat_X_train = pd.DataFrame(OH_enc.transform(train_df[cat_features]))
    cat_X_valid = pd.DataFrame(OH_enc.transform(valid_df[cat_features]))

    cat_X_train.index = train_df.index
    cat_X_valid.index = valid_df.index

    num_X_train = train_df[numeric_features]
    num_X_valid = valid_df[numeric_features]
    
    # Concatenate numerical features and categorical features
    X_train = pd.concat([num_X_train, cat_X_train], axis=1)
    X_valid = pd.concat([num_X_valid, cat_X_valid], axis=1)
    y_train = train_df[target]
    y_valid = valid_df[target]
    
    print(f'\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}\n')
    
    #Model training
    model = XGBRegressor(n_estimators=1000, 
                         learning_rate=0.05,
                         max_depth=3,
                         subsample=0.8,
                         colsample_bytree=0.6,
                         eval_metric='rmsle',
                         random_state=0)

    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_valid, y_valid)], 
              verbose=200)
    
    #Validation
    predictions = model.predict(X_valid)
    rmsle = rmse_log(y_valid, predictions)
    rmsles.append(rmsle)
    models.append(model)
    print(f'\nFold{fold} root mean square log error: {rmsle}')

# Aggregate validation rmsles
oof_rmsle = np.sqrt(np.square(rmsles).mean())
print(f'\nOut of fold RMSLE: {oof_rmsle}')


# # 4 Folds Bagging Inference

# In[24]:


test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df = cat2num(test_df)
test_df = feature_e(test_df)

test_preds = []
for fold in range(N_FOLD):
    cat_X_test = pd.DataFrame(OH_enc.transform(test_df[cat_features]))
    cat_X_test.index = test_df.index
    num_X_test = test_df[numeric_features]
    X_test = pd.concat([num_X_test, cat_X_test], axis=1)
    
    pred = models[fold].predict(X_test)
    print(f'\nModel{fold} prediction finished.')
    test_preds.append(pred)

test_preds = np.mean(test_preds, axis=0)

sub_df = pd.DataFrame({'Id': test_df['Id'],
                       'SalePrice': test_preds})
sub_df.to_csv('submission.csv', index=False)


# In[25]:


sub_df.head()

