#!/usr/bin/env python
# coding: utf-8

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Introduction</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# This notebook dives deep into building a complete pipeline for this competition. Starting with a solid baseline of (Top 15%), I've steadily climbed the leaderboard through preprocessing and feature engineering. My second submission already cracked the top 10%. Now, my latest iteration sits in the top 4%, and I'm eager to push it even further. I'd love to hear your thoughts on tackling the remaining challenges and welcome any suggestions for optimization. Let's explore the path to the top together! Happy Kaggling!
# </p>

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Table of contents</div>
# 
# * [1-Libraries](#section-one)
# * [2-Data loading](#section-two)
# * [3-EDA](#section-three)
# * [4-Preprocessing and Feature engineering](#section-four)
# * [5-Training](#section-five)
# * [6-Blending](#section-six)

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 1 | Libraries</div>

# In[1]:


get_ipython().system('pip install lofo-importance -q')
get_ipython().system('pip install feature_engine -q')


# In[2]:


# Data handling
import pandas as pd
import numpy as np

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn
from sklearn import model_selection, preprocessing, impute, metrics 

# Encoding
from feature_engine.encoding import *
from feature_engine.outliers import *
from category_encoders import TargetEncoder, CatBoostEncoder,GLMMEncoder, CountEncoder

# Feature selection
import shap
from lofo import LOFOImportance, Dataset, plot_importance

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn import kernel_ridge as kr
from sklearn import linear_model, neighbors, naive_bayes, ensemble, svm, neural_network

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas formatting
pd.options.display.float_format = '{:20,.2f}'.format

# Viz style
sns.set_style ('darkgrid')


# <a id="section-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 2 | Data loading and Folds</div>

# In[3]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[4]:


def kfold(df):
    df = df.copy()
    train['kfold'] = -1
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
        df.loc[valid_idx,'kfold'] = i
    return df


# In[5]:


train = kfold(train)


# <a id="section-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 3 | EDA</div>

# In[6]:


train.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[7]:


test.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# <p style="font-size:25px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Feature distribution</b></p>

# In[8]:


features = [feature for feature in train.columns if feature not in ('Id', 'SalePrice', 'kfold')]


# In[9]:


categorical_features = [feature for feature in train.columns if train[feature].dtype=='O']
numerical_features = [feature for feature in train[features].columns if feature not in categorical_features]


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Numerical features distribution</p>

# In[10]:


len(numerical_features)


# In[11]:


sns.set(rc={"figure.figsize":(20, 30)})
for i, feature in enumerate(numerical_features):
    plt.subplot(12,3, i*1 + 1)
    plt.ylabel(feature)
    g1 = sns.histplot(data=train, x=train[feature], bins=30,  color='darkblue') 
    g2 = sns.histplot(data=test, x=test[feature], bins=30, color='lightblue')
    g1.set(xticklabels=[])  
    g1.set(xlabel=None)
    g1.tick_params(bottom=False)  
    g2.set(xticklabels=[])  
    g2.set(xlabel=None)
    g2.tick_params(bottom=False);


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> The distribution of numeric features varies quite a lot, there is some skewed features. Also we can clearly see what features are actually continous and which are categoricals. From the graphs above we can remove the following from the numerical variables:</p>
# 
# 
# <ol><b>
#     
#   <li>'MSSubClass'</li>
#   <li>'OverallQual'</li>
#   <li>'OverallCond'</li>
#   <li>'BsmtHalfBath'</li>
#   <li>'BsmtFullBath'</li>
#   <li>'FullBath'</li>
#   <li>'HalfBath'</li>
#   <li>'BedroomAbvGr'</li>
#   <li>'KitchenAbvGr'</li>
#   <li>'TotRmsAbvGrd'</li>
#   <li>'Fireplaces'</li>
#   <li>'GarageCars'</li>
#   <li>'MoSold'</li>
#   <li>'YrSold'</li></b>
# </ol> 
# 
# 

# In[12]:


other_categorical_feat = ['MSSubClass','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                          'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','MoSold','YrSold']
new_numerical_features = [feature for feature in numerical_features if feature not in other_categorical_feat]
new_categorical_features = [feature for feature in train[features].columns if feature not in new_numerical_features]


# In[13]:


len(new_numerical_features)


# In[14]:


sns.set(rc={"figure.figsize":(20, 30)})
for i, feature in enumerate(new_numerical_features):
    plt.subplot(8,3, i*1 + 1)
    plt.ylabel(feature)
    g1 = sns.histplot(data=train, x=train[feature], bins=30,  color='darkblue') 
    g2 = sns.histplot(data=test, x=test[feature], bins=30, color='lightblue')
    g1.set(xticklabels=[])  
    g1.set(xlabel=None)
    g1.tick_params(bottom=False)  
    g2.set(xticklabels=[])  
    g2.set(xlabel=None)
    g2.tick_params(bottom=False);


# In[15]:


sns.set(rc={"figure.figsize":(20, 30)})
for i, feature in enumerate(new_numerical_features):
    plt.subplot(8,3, i*1 + 1)
    plt.ylabel(feature)
    g1 = sns.boxenplot(data=train, x=train[feature],color='darkblue')
    g2 = sns.boxenplot(data=test, x=test[feature], color='lightblue') 
    g1.set(xlabel=None)
    g1.tick_params(bottom=False)    
    g2.set(xlabel=None)
    g2.tick_params(bottom=False)  


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> There is a lot of outliers on the data, we need to handle that, this also depend of the model you choose. Also some columns has a lot of NaN values, like 3SsnPorch, PoolArea, MiscVal.</p>
# 

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Categorical features distribution</p>

# In[16]:


len(new_categorical_features)


# In[17]:


sns.set(rc={"figure.figsize":(20, 30)})
for i, feature in enumerate(new_categorical_features[:29]):
    plt.subplot(10,3, i*1 + 1)
    plt.ylabel(feature)
    g1 = sns.histplot(data=train, x=train[feature], color='darkblue')
    g2 = sns.histplot(data=test, x=test[feature], color='lightblue')
    g1.set(xlabel=None)
    g2.set(xlabel=None)
    plt.xticks(rotation=25)


# In[18]:


sns.set(rc={"figure.figsize":(20, 30)})
for i, feature in enumerate(new_categorical_features[29:]):
    plt.subplot(10,3, i*1 + 1)
    plt.ylabel(feature)
    g1 = sns.histplot(data=train, x=train[feature], color='darkblue')
    g2 = sns.histplot(data=test, x=test[feature], color='lightblue')
    g1.set(xlabel=None)
    g2.set(xlabel=None)


# <p style="font-size:25px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Features cardinality</b></p>

# In[19]:


train_cardinality = (pd.Series({feature: len(train[feature].unique()) for feature in train[categorical_features]})
                     .reset_index().rename(columns={'index':'Feature',0:'Cardinality'}))


# In[20]:


sns.set(rc={"figure.figsize":(10, 12)})
g=sns.barplot(data=train_cardinality.sort_values(by='Cardinality',ascending=False), 
              x = 'Cardinality', 
              y = 'Feature', 
              palette = 'rainbow')
g.set(title='Train Cardinality');


# In[21]:


test_cardinality = (pd.Series({feature: len(test[feature].unique()) for feature in test[categorical_features]}).reset_index()
                     .rename(columns={'index':'Feature', 0:'Cardinality'}))


# In[22]:


g=sns.barplot(data=test_cardinality.sort_values(by='Cardinality', ascending=False), 
              x = 'Cardinality', 
              y = 'Feature', 
              palette = 'rainbow')
g.set(title='Test Cardinality');


# In[23]:


(pd.merge(train_cardinality,test_cardinality, on='Feature', how='left')
 .sort_values(by=['Cardinality_x', 'Cardinality_y'], ascending=[False,False])
 .rename(columns={'Cardinality_x':'Train Cardinality','Cardinality_y':'Test Cardinality'}))


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> Here we can see that the cardinality is not the same in the 2 datasets, it varies in some features, especially in Condition2 and RoofMatl. Also it's quite high in some of the features.</p>

# <p style="font-size:25px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Missing Values</b></p>
# 

# In[24]:


train_missing_vals = (pd.Series(train.isna().sum()).reset_index().rename(columns={'index': 'Feature', 0: 'Missing Values'}))


# In[25]:


sns.set(rc={"figure.figsize":(10, 12)})
g=sns.barplot(data=train_missing_vals[train_missing_vals['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False), 
              x = 'Missing Values', 
              y = 'Feature', 
              palette = 'rainbow')
g.set(title='Train missing values');


# In[26]:


test_missing_vals = (pd.Series(test.isna().sum()).reset_index().rename(columns={'index': 'Feature', 0: 'Missing Values'}))


# In[27]:


sns.set(rc={"figure.figsize":(10, 12)})
g=sns.barplot(data=test_missing_vals[test_missing_vals['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False), 
              x = 'Missing Values', 
              y = 'Feature', 
              palette = 'rainbow')
g.set(title='Test missing values');


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> There are a large number of missing values that there are in some features, but that does not mean that this missing value does not represent anything, but rather that it may be poorly encoded. For example, the first feature with missing values is PoolQC and the missing values are due to the fact that many houses do not have a pool (NA=No Pool). Therefore, if we drop the column, we're losing valuable information about whether a house has a pool or not, which will possibly have an impact on the SalePrice. This is going to require manual handling of missing values, and leaving out sklearn's imputation. On the other hand, we can saw that the number of missing value features is greater on the test dataset. </p>
# 

# <p style="font-size:25px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Features relationships with the target value</b></p>
# 

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em;margin-left:20px">   
# Numerical</p>

# In[28]:


sns.set(rc={"figure.figsize":(22, 30)})
for i, feature in enumerate(new_numerical_features):
    plt.subplot(8,3, i*1 + 1)
    g1 = sns.scatterplot(data=train,
                         x = train[feature], 
                         y = np.log(train['SalePrice']), 
                         hue =  np.log(train['SalePrice']), 
                         palette = 'rainbow')


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> We can see that there are several features with a direct relationship in the SalePrice such as Basement, Floors, Ground Area, Garage.</p>

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Categorical</p>

# In[29]:


sns.set(rc={"figure.figsize":(22, 34)})
for i, feature in enumerate(new_categorical_features[:29]):
    plt.subplot(10,3, i*1 + 1)
    g1 = sns.violinplot(data=train, x=train[feature], y=np.log(train['SalePrice']) ,palette='rainbow')
    g1.set(xlabel=None)
    plt.xticks(rotation=20)
    plt.title(feature, x=0.9, y=0.9)


# In[30]:


for i, feature in enumerate(new_categorical_features[29:]):
    plt.subplot(10,3, i*1 + 1)
    g1 = sns.violinplot(data=train, x=train[feature], y=np.log(train['SalePrice']) ,palette='rainbow') 
    g1.set(xlabel=None)
    plt.xticks(rotation=25)
    plt.title(feature, x=0.9, y=0.9)


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> Here also we can see that there are several features with a direct relationship in the SalePrice such as Overal quality, Total rooms.</p>

# <a id="section-four"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 4 | Preprocessing and Feature engineering</div>

# In[31]:


# Creation of a fake target value (-1) for concatenation purposes
test['SalePrice'] = -1
combined_df = pd.concat([train,test], axis = 0)


# In[32]:


def preprocessing_inputs(df):
    df = df.copy()
    
    # Filling NA categoric features
    na_features = ['Alley', 'MasVnrType', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                   'FireplaceQu','GarageType', 'GarageFinish','GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
 
    df['MasVnrType'].replace({'None': 'No'}, inplace=True)
    for feature in na_features:
        df[feature] = df[feature].fillna('No')
    
    na_test_features = ['MSZoning', 'Utilities', 'Exterior1st','Exterior2nd','KitchenQual', 'Functional', 'SaleType' ]
    for feature in na_test_features:
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        
    df.loc[df['GarageYrBlt']==2207.0, 'GarageYrBlt']=2007.0   
        
    # Encoding
    qual_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'No':0}
    qual_features = ['PoolQC','HeatingQC','KitchenQual','ExterQual','ExterCond','BsmtQual', 'GarageQual','GarageCond','FireplaceQu','BsmtCond']
    for col in qual_features:
        df[col] = df[col].map(qual_dict)
        
    df['Alley'] = df['Alley'].map({'No':0, 'Grvl':1, 'Pave':2})        
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'No':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'Unf' : 1 , 'No' : 0, 'LwQ' : 2, 'Rec': 3 , 'BLQ' : 4, 'ALQ':5, 'GLQ':6})
    df['BsmtExposure'] = df['BsmtExposure'].map({'Gd':4, 'Av' : 3, 'Mn':2, 'No':0 })
    df['Fence'] = df['Fence'].map({'No' :0, 'MnWw':1, 'GdWo':2, 'MnPrv': 3, 'GdPrv':4})
    df['Functional'] = df['Functional'].map({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8})
    df['LandSlope'] = df['LandSlope'].map({'Sev':1, 'Mod':2, 'Gtl':3})
    df['Street'] = df['Street'].map({'Grvl':1, 'Pave':2})
    df['PavedDrive'] = df['PavedDrive'].map({'N':1, 'P':2, 'Y':3})
    df['Utilities'] = df['Utilities'].map({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4})
        
    # Feature engineering
    df['GarageYrBltn'] = abs(df['YrSold'] - df['GarageYrBlt'])
    df['YearRemodAddn'] = abs(df['YrSold'] - df['YearRemodAdd'])
    df['YearBuiltn'] = abs(df['YrSold'] - df['YearBuilt'])
    df['SF'] = df['1stFlrSF']+df['2ndFlrSF']+df['TotalBsmtSF']+df['GrLivArea']+df['HalfBath']+df['FullBath']
    
    # Transformation of skew features
    df[numerical_features] = np.log(df[numerical_features] + 1)
    
    # Split the dataframe
    train = df.query("SalePrice != -1").copy()
    test = df.query("SalePrice == -1").copy()
        
    test.drop(['SalePrice'], axis = 1, inplace=True)
    
    return train, test


# In[33]:


train_df, test_df = preprocessing_inputs(combined_df)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Feature selection</p>
# 
# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# LOFO</p>

# In[34]:


features = [feature for feature in train_df.columns if feature not in ('Id', 'SalePrice', 'kfold')]
categorical_features = [feature for feature in train_df[features].columns if train_df[feature].dtype=='O']
numerical_features = [feature for feature in train_df[features].columns if feature not in categorical_features]


# In[35]:


def get_lofo_importance(df, numerical_features, categorical_features, target,metric):
    
    df=df.copy()
    df[categorical_features] = preprocessing.OrdinalEncoder().fit_transform(df[categorical_features])
    
    dataset = Dataset(df=df, target=target, features=numerical_features+categorical_features)

    model = lgb.LGBMRegressor()
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    lofo_imp = LOFOImportance(dataset, cv=kf, scoring=metric)
    
    importance_df = lofo_imp.get_importance()
    return importance_df


# In[36]:


lofo_imp_df = get_lofo_importance(train_df, numerical_features, categorical_features, 'SalePrice', 'neg_mean_squared_error')


# In[37]:


plot_importance(lofo_imp_df, figsize=(12, 20))


# In[38]:


len(lofo_imp_df[lofo_imp_df.importance_mean >=0])


# In[39]:


lofo_features = lofo_imp_df[lofo_imp_df.importance_mean >=0]['feature'].to_list()


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Shap</p>

# In[40]:


models = {'B-XGB': xgb.XGBRegressor(n_estimators=1000)}


# In[41]:


def shap_feature_selection(df, subset, numerical_features, categorical_features, target, folds, models):
    
    list_shap_values = []
    
    df = df.sample(frac=subset, random_state=42).reset_index()
    
    df[categorical_features] = preprocessing.OrdinalEncoder().fit_transform(df[categorical_features])

    for fold in range(folds):
        print(f' Fold number = {fold}')
        X_train = df[df.kfold != fold].reset_index(drop=True)
        X_valid = df[df.kfold == fold].reset_index(drop=True)

        y_train = X_train[target]
        y_valid = X_valid[target]

        X_train = X_train[numerical_features+categorical_features]
        X_valid = X_valid[numerical_features+categorical_features]
        
        model = models.fit(X_train, y_train)
            
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(X_valid)
        list_shap_values.append(shap_value)
        
    feature_name = df[numerical_features+categorical_features].columns
    shap_values = np.vstack([sv[1] for sv in list_shap_values])
    sv = np.abs(shap_values).mean(0)
    
    importance_df = pd.DataFrame({"feature": feature_name, "shap_values": sv})
    
    return importance_df, shap_values, df


# In[42]:


for name, model in models.items():
    print(f'Model {name}')
    shap_imp_df, shap_values, df = shap_feature_selection(train_df,1., numerical_features, categorical_features, 'SalePrice', 5, model)


# In[43]:


shap.summary_plot(
    shap_values, features=features, feature_names=train_df[features].columns, plot_type="bar",max_display=20
)


# In[44]:


shap_imp_df.sort_values(by='shap_values', ascending=False)


# In[45]:


len(shap_imp_df[shap_imp_df.shap_values > 0])


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> We can see how LOFO is more restrictive about the features who select between a range of 30 to 60, since I know more about Shap, I'm going to use his selection of features (83 total).</p>

# <a id="section-five"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 5 | Training </div>

# In[46]:


class Trainer:
    
    """
    Args:
        - model: Any ML model to train.
        - model_name: The corresponding model name to be used to identify it in the training process.
        - fold: Fold number.
        - model_params: Hyperparameters of the respective model.
        
    """
    
    def __init__(self, model, model_name, fold):
     
        self.model_ = model
        self.model_name = model_name
        self.fold = fold
        
        self.test_preds = []
              
    def fit(self, xtrain, ytrain, xvalid, yvalid):
        
        """
        Fits an instance of the model for a particular dataset.
        Args:
            - xtrain: Train data.
            - ytrain: Train target.
            - xvalid: Validation data.
            - yvalid: Validation target.
        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid
        
        if self.model_name.startswith('B'):
            self.model_.fit(self.xtrain, self.ytrain, early_stopping_rounds=10, eval_set=[(self.xvalid, self.yvalid)],verbose=False)
        if self.model_name.startswith('T'):
            self.model_.fit(self.xtrain, self.ytrain.values.reshape(-1,1), eval_set=[(self.xvalid, self.yvalid.values.reshape(-1,1))])
        else:
            self.model_.fit(self.xtrain, self.ytrain)
        
        return self.model_
        
    def pred_evaluate(self,name, oofs,val_idx):
        
        """
        Makes predictions for each model on the valid data provided.
        Args:
            - name: Model name.
            - oofs: oofs data.
            - val_idx: Validation indicies.

        """
        
        preds_valid = self.model_.predict(self.xvalid)
        score = metrics.mean_squared_error(self.yvalid, preds_valid, squared=False)
        oofs.loc[val_idx, f"{name}_preds"] = preds_valid
        
        print(f'fold = {self.fold} | score = {score:.4f}')
        
        return score, preds_valid, oofs
    
    def blend(self, models, n_folds, xtest, column_id, features):
        
        """
        Makes a blend of the trained models.
        Args:
            - models: Models to blend.
            - n_folds: Number of folds.
            - xtest: Test data preprocess.
            - column_id: Id.
            - features: final 
        """
        
        y_test_pred = pd.DataFrame(xtest.loc[:,column_id])    
        X_test = xtest[features]

        models_n = np.array_split(list(models.items()), len(models)/n_folds)
        for model_folds in models_n:
            model_test_preds = []
            for fold, model in model_folds:
                preds_test = model.predict(X_test)
                model_test_preds.append(preds_test)
                
            # Get the model name
            name =  fold.split('_')[0]
            # Assign mean folds predictions to preds dataframe
            y_test_pred.loc[:,f'{name}_preds'] = np.mean(np.column_stack (model_test_preds), axis=1)
        
        return y_test_pred


# In[47]:


class Models:
    BXGB = xgb.XGBRegressor(random_state=42)
    BLGBM = lgb.LGBMRegressor(random_state=42)
    BCB = cb.CatBoostRegressor(random_state=42, verbose=False)
    RR = linear_model.Ridge(random_state=42)
    AR = linear_model.ARDRegression()
    KR = kr.KernelRidge(alpha = 0.68, coef0 = 3.5, degree = 2, kernel = 'polynomial')


# In[48]:


features = [feature for feature in train_df.columns if feature not in ('Id', 'SalePrice', 'kfold')]
categorical_features = [feature for feature in train_df[features].columns if train_df[feature].dtype=='O']
numerical_features = [feature for feature in train_df[features].columns if feature not in categorical_features]


# In[49]:


len(numerical_features+categorical_features)


# In[50]:


oofs_dfs = []
models_trained = {}

iterator = iter([[getattr(Models, attr), attr] for attr in dir(Models) if not attr.startswith("__")])

for i, mdls in enumerate(iterator):

    oofs_scores = []
    
    model = mdls[0]
    name = mdls[1]
    
#     if name.startswith('B'):
#         model.set_params(random_state=i)
    
    oofs = train_df[["Id","SalePrice", "kfold"]].copy()
    oofs[f"{name}_preds"] = None

    print(f' Model {name}')
    
    for fold in range(5):
    
        X_train = train_df[train_df.kfold != fold]
        X_valid = train_df[train_df.kfold == fold]
                
        val_idx = X_valid.index.to_list()

        y_train = np.log(X_train['SalePrice'])
        y_valid = np.log(X_valid['SalePrice'])
        
        X_train = X_train[numerical_features+categorical_features]
        X_valid = X_valid[numerical_features+categorical_features]
        
        # Scaling
        scl = preprocessing.RobustScaler()
        X_train[numerical_features] = scl.fit_transform(X_train[numerical_features])
        X_valid[numerical_features] = scl.transform(X_valid[numerical_features])
        
        # Imputing
            # Numerical
        imp_num = impute.KNNImputer(weights = 'uniform', metric = 'nan_euclidean')
        X_train[numerical_features] = imp_num.fit_transform(X_train[numerical_features])
        X_valid[numerical_features] = imp_num.transform(X_valid[numerical_features])
            
            # Categorical
        imp_cat = impute.SimpleImputer(strategy='most_frequent')
        X_train[categorical_features] = imp_cat.fit_transform(X_train[categorical_features])
        X_valid[categorical_features] = imp_cat.transform(X_valid[categorical_features])
        
        # Outliers handlingh
        capper = Winsorizer(capping_method='gaussian', tail='right', fold=3, variables=numerical_features)
        X_train[numerical_features] = capper.fit_transform(X_train[numerical_features])
        X_valid[numerical_features] = capper.transform(X_valid[numerical_features])

        # Encoding       
            # OHE
        ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore').fit(X_train[categorical_features])
        encoded_cols = list(ohe.get_feature_names(categorical_features))
    
        X_train[encoded_cols] = ohe.transform(X_train[categorical_features])
        X_valid[encoded_cols] = ohe.transform(X_valid[categorical_features])

        # Preprocessed's dfs
        X_train = X_train[numerical_features+encoded_cols]
        X_valid = X_valid[numerical_features+encoded_cols]
               
        # Trainer class initialization
        trainer = Trainer(model=model, model_name=name,fold=fold)
        
        # Fit the trainer
        model_trained = trainer.fit(X_train, y_train, X_valid, y_valid)
        
        # Evaluate
        scores, valid_preds, oofs = trainer.pred_evaluate(name, oofs, val_idx)
        models_trained[f'{name}_{fold}'] = model_trained
        oofs_scores.append(scores)
   
    oofs_dfs.append(oofs)
    print(f' oofs score = {np.mean(oofs_scores):.4f}')
    print()


# In[51]:


final_valid_df = pd.concat(oofs_dfs, axis=1).iloc[:, [0,1,2,3,7,11,15,19]]


# In[52]:


final_valid_df.sample(5)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Preprocessing test dataframe</p>

# In[53]:


X_test = test_df.copy()
X_test[numerical_features] = scl.transform(X_test[numerical_features])
X_test[numerical_features] = imp_num.transform(X_test[numerical_features])
X_test[numerical_features] = capper.transform(X_test[numerical_features])
X_test[categorical_features] = imp_cat.transform(X_test[categorical_features])
X_test[encoded_cols] = ohe.transform(X_test[categorical_features])


# In[54]:


final_test_df = trainer.blend(models_trained, 5, X_test, 'Id', numerical_features+encoded_cols)


# In[55]:


final_test_df.sample(5)


# <a id="section-six"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 6 | Blending</div>
# 
# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Approach 1: Find the best weights with LinearRegression</b></p>

# In[56]:


features_lr = [feature for feature in final_valid_df.columns if 'preds' in feature]


# In[57]:


# Meta Model LR

train_lr = final_valid_df.copy()
test_lr = final_test_df.copy()

scores = []
valid_preds_lr = {}
test_preds_lr = []

for fold in range(5):
    
    X_train = train_lr[train_lr.kfold != fold].reset_index(drop=True)
    X_valid = train_lr[train_lr.kfold == fold].reset_index(drop=True)
    
    X_test = test_lr[features_lr].copy() #
    
    X_valid_ids = X_valid.Id.values.tolist()

    y_train = np.log(X_train['SalePrice'])
    y_valid = np.log(X_valid['SalePrice'])
    
    X_train = X_train[features_lr]
    X_valid = X_valid[features_lr]
    
    # Model
    model = linear_model.LinearRegression()
    
    model.fit(X_train, y_train) 
        
    preds_valid = model.predict(X_valid)
    preds_test = model.predict(X_test)
    
    valid_preds_lr.update(dict(zip(X_valid_ids,preds_valid )))
    test_preds_lr.append(preds_test)
    
    rmse = metrics.mean_squared_error(y_valid, preds_valid, squared=False)
    scores.append(rmse)
    
    print(f' Fold = {fold}, RMSE = {rmse:.10f}')
print(f'Mean score = {np.mean(scores):.10f} Std = {np.std(scores):.3f} ')


# In[58]:


level1_valid_preds_lr = pd.DataFrame.from_dict(valid_preds_lr, orient='index').reset_index().rename(columns = {'index':'Id', 0:'lr_pred_1'})

level1_test_preds_lr = submission.copy()
level1_test_preds_lr.SalePrice = np.exp(np.mean(np.column_stack (test_preds_lr), axis=1 ))
level1_test_preds_lr.to_csv('lr-submission.csv', index = False)


# In[59]:


level1_test_preds_lr.sample(5)


# ![hp LB.jpg](attachment:06921962-25ed-4330-8eaa-59575beab60a.jpg)

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Linear Regression gives me a higher score compared to weighted averaging</p>

# <p style="font-size:25px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>Approach 2: Weighted average of the predictions</b></p> 

# In[60]:


wavg_preds = final_test_df[features_lr].mean(axis=1)


# In[61]:


submission.SalePrice = np.exp(wavg_preds)


# In[62]:


submission.to_csv('submission.csv', index=False)


# In[63]:


submission.sample(5)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Weighted average submission</p>

# ![we LB.jpg](attachment:81709da8-5a5c-4f90-9cae-a9cc5061a4df.jpg)

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b>To improve the score we need to do a proper encoding, improve the outliers treatment, do more feature engineering, feature selection, etc. In the coming updates, I'll do all of this stuff. Thanks for taking the time to read my notebook. </b></p>
