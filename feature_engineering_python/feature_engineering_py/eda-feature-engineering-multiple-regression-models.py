#!/usr/bin/env python
# coding: utf-8

# <h1> Restaurant Revenue Regression
#     
# <h4> Data fields
#     
# * **Id** : Restaurant id. 
# * **Open Date** : opening date for a restaurant
# * **City** : City that the restaurant is in. Note that there are unicode in the names. 
# * **City Group**: Type of the city. Big cities, or Other. 
# * **Type**: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
# * **P1, P2 - P37**: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
# * **Revenue**: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values. 

# <h1> 0. unzip the .csv dataset

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_dir = '../input/restaurant-revenue-prediction/train.csv.zip'
test_dir = '../input/restaurant-revenue-prediction/test.csv.zip'

import zipfile
with zipfile.ZipFile(train_dir,"r") as z:
    z.extractall('.')

with zipfile.ZipFile(test_dir,"r") as z:
    z.extractall('.')


# In[5]:


from datetime import datetime


# <h1> 1. Load the Dataset

# In[6]:


# load the train and test dataframe
train_df=pd.read_csv('./train.csv')
test_df=pd.read_csv('./test.csv')

# look at the information of the train_df
display(train_df.head())
display(train_df.info())


# <h1> 2. Pre-Process the Dataset

# <h2> 2.1 convert some features data type
#     
# <li> convert some features into [datetime] datatypes
# <li> convert some int/float to [str] datatypes
#     

# In[7]:


# select the columns convert to [datetime] object in Python
def time_feature_convert(df,time_features,time_format):
    df[time_features]=df[time_features].apply(
    pd.to_datetime,
    format=time_format)
    print(f'Feature Type Conversion Informatoin: {(df[time_features]).dtypes}\n')
    return df

# transfer the cols in train and test df     
time_feature_convert(df=train_df,
                     time_features=['Open Date'],
                     time_format='%m/%d/%Y')
time_feature_convert(df=test_df,
                     time_features=['Open Date'],
                     time_format='%m/%d/%Y')
pass


# In[8]:


# select the columns convert to [str] object in Python
def str_feature_convert(df,str_features):
    df[str_features]=df[str_features].astype(str)
    print(f'Feature Type Conversion Informatoin: {(df[str_features]).dtypes}\n')
    return df

# transfer the cols in train and test df     
str_feature_convert(df=train_df,
                     str_features=['Id'])
str_feature_convert(df=test_df,
                     str_features=['Id'])
pass


# In[9]:


# now check the df datatypes
display(train_df.info())
display(train_df.head())


# <h2> 2.2 Drop some irrelevant features (such as ID)

# In[10]:


# indicate the cols that need to be dropped
drop_features=['Id','Open Date']

# drop the selected cols
train_df.drop(drop_features,axis=1,inplace=True)
test_df.drop(drop_features,axis=1,inplace=True)


# In[11]:


train_df


# <h1> 3. EDA

# In[12]:


pip install dataprep


# In[13]:


from dataprep.datasets import get_dataset_names
from dataprep.datasets import load_dataset
from dataprep.eda import create_report,plot,plot_missing
import scipy.stats as stats


# <h2> 3.1 Overall Statistics

# In[14]:


# function for overall statistical report
def overall_stat(df):
    # display the overall stat report
    display(plot(df, display=['Stats', 'Insights']))
    # display(df.info())

    # store and display the numerical and nonn-numerical cols in df
    num_cols=list(df.select_dtypes(include=['number']).columns)
    non_num_cols=list((set(df.columns)-set(num_cols)))

    print(f'Num cols = {num_cols}')
    print(f'Non-num cols = {non_num_cols}')


# In[15]:


# display the overall stats
overall_stat(train_df)


# **As can be seen above, for train_df:**
# <li> 137 rows
# <li> no missing value
# <li> 41 features, 1 target (numerical)

# <h2> 3.2 Univariate Analysis

# <h3> 3.2.1 Numerical Feature Univariate Analysis

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Relationship_between_mean_and_median_under_different_skewness.png/434px-Relationship_between_mean_and_median_under_different_skewness.png)

# In[16]:


# define the interest feature you want to explore
inter_features='revenue'

# define the function for univariate analysis
def num_uni_analysis(df,inter_features):
    display(plot(df,inter_features,display=['Stats','KDE Plot','Normal Q-Q Plot','Box Plot']))
    skewness=df[inter_features].skew()
    kurtosis=df[inter_features].kurtosis()
    print(f'-The Skewness = {skewness}')
    if abs(skewness)<1:
        print(f'The [{inter_features}] distribution is nearly normal')
    elif skewness>1:
        print(f'The [{inter_features}] distribution is right skewed ')
    else:
        print(f'The [{inter_features}] distribution is left skewed ')
    print(f'-The Kurtosis = {kurtosis}')


# In[17]:


# display the univariate analysis result for feature [revenue]
num_uni_analysis(train_df,inter_features)


# <h3> 3.2.2 Categorical Feature Univaraite Analysis

# In[18]:


# define the function for univariate analysis
def cat_uni_analysis(df,inter_features):
    print(f'The Non-Numerical Column You Choose is: [{inter_features}]\n')
    display(plot(df,inter_features,display=['Stats','Pie Chart','Value Table']))


# In[19]:


# display the univariate categorical analysis result
cat_uni_analysis(train_df,inter_features='Type')
cat_uni_analysis(train_df,inter_features='City')
cat_uni_analysis(train_df,inter_features='City Group')


# <h2> 3.3 Bivariate Analysis

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# <h3> 3.3.1 Overall num-num relationship (corr heatmap)

# In[21]:


# overall num-num relationship: correlation heatmap
def heatmap(df,figsize):
    fig, axs=plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(),annot=True, linewidths=.7,cmap='coolwarm',fmt='.1f',ax=axs)
    
# display the overall correlation heatmap
heatmap(df=train_df,figsize=(25,25))


# <h3> 3.3.2 Categorical - Numerical relationship (1 vs 1)

# In[22]:


pip install scikit-posthocs


# In[23]:


# this is for the Kruskal test used for categorical-numerical relationship
import scikit_posthocs as sp 


# In[24]:


# categoircal-numerical relationship (cat_feature - target)

def cat_num_relationship(df,cat_col,num_col):
    # visualization
    print(f'[{cat_col}] --- [{num_col}] relationship')
    display(plot(df,num_col,cat_col))
    
    # hypothesis testing for catgorical-numerical relationship (Kruskal test)
    pc = sp.posthoc_conover(df, val_col=num_col, group_col=cat_col,p_adjust = 'holm')
    # visualization of the heatmap
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

    # plot
    fig, ax = plt.subplots(ncols=1)
    fig.suptitle('Significance Plot')
    sp.sign_plot(pc,**heatmap_args,ax=ax) 
    fig.show()


# In[25]:


cat_num_relationship(df=train_df,
                    cat_col='City',
                    num_col='revenue')

cat_num_relationship(df=train_df,
                    cat_col='Type',
                    num_col='revenue')

cat_num_relationship(df=train_df,
                    cat_col='City Group',
                    num_col='revenue')


# <h3> 3.3.3 Categorical-Categorical Relationship (1 vs 1)

# In[26]:


from scipy.stats import chi2_contingency


# In[27]:


# categoircal-categorical relationship

def cat_cat_relationship(df,cat_col_1,cat_col_2):
    # visualization
    plot(df,cat_col_1,
         cat_col_2,
         display=['Stacked Bar Chart','Heat Map'])
    
    # Chi-square test
    
    # 1st step convert the data into a contingency table with frequencies
    chi_contigency=pd.crosstab(df[cat_col_1],df[cat_col_2])
    print(f'Selected cols [{cat_col_1}] and [{cat_col_2}]')
    print('chi2-contingency table')
    display(chi_contigency)
    
    # 2nd step: Chi-square test of independence.
    c, p, dof, expected = chi2_contingency(chi_contigency)
    if p<0.05:
      print('Reject Null Hypothesis')
      print(f'The:\n [{cat_col_1}],[{cat_col_2}] are not independent\n')
    else:
      print('Fail to Reject Null Hypothesis')
      print(f'The:\n [{cat_col_1}],[{cat_col_2}] are independent\n') 
    print(f'The P-value = {p}')


# In[28]:


# display the result
cat_cat_relationship(df=train_df,
                    cat_col_1='City Group',
                    cat_col_2='Type')


# <h1> 4. Label Encoding

# In[29]:


from sklearn import preprocessing


# In[30]:


train_df.head()


# In[31]:


# define the function for label or one-hot encoding
def label_encode_transform(df,cols):
    cols=cols
    le = preprocessing.LabelEncoder()
    df[cols]=df[cols].apply(le.fit_transform)
    return df
    
def onehot_encode_transform(df,cols):
    cols=cols
    df=pd.get_dummies(df,columns=cols)
    return df


# In[32]:


train_df_encode=label_encode_transform(df=train_df,
                        cols=['City'])
train_df_encode=onehot_encode_transform(df=train_df_encode,
                        cols=['City Group','Type'])

test_df_encode=label_encode_transform(df=test_df,
                        cols=['City'])
test_df_encode=onehot_encode_transform(df=test_df_encode,
                        cols=['City Group','Type'])


# In[33]:


train_df_encode.info()


# <h1> 5. Feature Selection

# In[34]:


from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, roc_auc_score


# In[35]:


# seperate the source and the target variables
feature_cols = [x for x in train_df_encode.columns if x != 'revenue']
X_train = train_df_encode[feature_cols]
y_train = train_df_encode['revenue']

X_test  = test_df_encode[feature_cols]


# In[36]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.transform (X_test)


# In[62]:


model= Lasso()
feature_selector = RFECV(model,
                         scoring='neg_mean_squared_error',
                         cv=3)
feature_selector.fit(X_train_scaled,y_train)


# In[64]:


feature_selector.n_features_


# In[100]:


# length check
len(feature_selector.support_)==len(X_train.columns)

# store the selected features
select_feature=pd.DataFrame(
    {'tf':feature_selector.support_,
    'feature':X_train.columns})

selected_feature=list(select_feature.loc[select_feature['tf']==True]['feature'])
feature_coef=feature_selector.estimator_.coef_


# In[107]:


# visualize the selected feature and coefficient
ax=sns.barplot(y=selected_feature,x=feature_coef)
ax.set_title('Selected Feature Coefficient Plot')
print(f'Selected features are: {selected_feature}')


# In[79]:


# Visualization
fig,axs=plt.subplots(ncols=1,figsize=(15,5))
fig.suptitle('Number of features --- RMSE')
sns.lineplot(range(1,len(feature_selector.grid_scores_)+1),
             feature_selector.grid_scores_,
             marker='o',
             ax=axs)


# <H1> 6. Modelling

# * Regularization method (Lasso, Ridge, Elastic Net)
# * Tree based model (XGBoost etc)

# In[154]:


# select the features from the dataset (3 features)
X_train_fs=X_train[selected_feature]
X_test_fs=X_test[selected_feature]

X_train_fs_scaled= scaler.fit_transform(X_train_fs)
X_test_fs_scaled = scaler.transform (X_test_fs)


# In[113]:


pip install xgboost


# In[114]:


pip install lightgbm


# In[128]:


from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import mean_squared_error


# In[135]:


cv=3

# fit the Models
lassoCV=LassoCV(cv=cv).fit(X_train_fs_scaled,y_train)
ridgeCV=RidgeCV(cv=cv).fit(X_train_fs_scaled,y_train)
elasticnetCV=ElasticNetCV(cv=cv).fit(X_train_fs_scaled,y_train)
lightgbm=lgb.LGBMRegressor().fit(X_train_fs_scaled,y_train)
xgboost=xgb.XGBRegressor().fit(X_train_fs_scaled,y_train)

# generate the prediction for train dataset
lasso_train_pred=lassoCV.predict(X_train_fs_scaled)
ridge_train_pred=ridgeCV.predict(X_train_fs_scaled)
elasticnet_train_pred=elasticnetCV.predict(X_train_fs_scaled)
lgbm_train_pred=lightgbm.predict(X_train_fs_scaled)
xgb_train_pred=xgboost.predict(X_train_fs_scaled)

# generate RMSE for each models
lasso_RMSE= np.sqrt(mean_squared_error(y_train, lasso_train_pred))
ridge_RMSE= np.sqrt(mean_squared_error(y_train, ridge_train_pred))
elasticnet_RMSE= np.sqrt(mean_squared_error(y_train, elasticnet_train_pred))
lgbm_RMSE= np.sqrt(mean_squared_error(y_train, lgbm_train_pred))
xgb_RMSE= np.sqrt(mean_squared_error(y_train, xgb_train_pred))


# In[153]:


model_list=['Lasso','Ridge','ElasticNet','LGBM','XGBoost']
rmse_list=[lasso_RMSE,ridge_RMSE,elasticnet_RMSE,lgbm_RMSE,xgb_RMSE]

# plot the RMSE for each model
ax=sns.barplot(y=model_list,x=rmse_list)
ax.set_title('Model RMSE Result')

# print the result RMSE number
print(f' lasso={lasso_RMSE} \n ridge = {ridge_RMSE}\n Elastic_Net = {elasticnet_RMSE}\n LGBM = {lgbm_RMSE}\n XGBoost= {xgb_RMSE}\n')


# **Therefore here i choose the XGBoost as the final model for prediction**

# In[157]:


# generate prediction for test dataset
xgb_test_pred=xgboost.predict(X_test_fs_scaled)


# In[161]:


# store the result
submission_df=pd.DataFrame(
{'Id':test_df.index,
'Prediction':xgb_test_pred}
)


# In[165]:


submission_df.to_csv('submission_dcx.csv',index=False)

