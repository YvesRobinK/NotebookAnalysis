#!/usr/bin/env python
# coding: utf-8

# # <u>Steps Performed:</u>
#    - Segregated dataset into 3 groups
#    - Converted temp columns into seasons
#    - Imputed the missing values
#    - Handled the 0s in year_build feature
#    - Label encoded the 3 categorical features
#    - Feature Scaling
#    - Feature Selection
#    - Modeling
#    - Tuning and remodelling
#    - Conclusions

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/widsdatathon2022/train.csv')
test = pd.read_csv('../input/widsdatathon2022/test.csv')


# In[3]:


train.head(3)


# In[4]:


target = train['site_eui']                 # capturing target variable
row_id = test.iloc[:,62:]                  # capturing the ids for submission file


# In[5]:


# drop features from dataframe
def drop(df, *features):
    for i in features:
        df.drop(i, axis=1, inplace=True)


# In[6]:


test.head(3)


# In[7]:


drop(train, 'id')
drop(test, 'id')


# In[8]:


# Segreggating the features into 3 sectors for better analysis
train_temp = train.iloc[:, 8:44]
train_fog = train.iloc[:, 44:62]
train_imp = train.iloc[:,:8]

test_temp = test.iloc[:, 8:44]
test_fog = test.iloc[:, 44:62]
test_imp = test.iloc[:,:8]


# ### 1. Temp columns 

# From the EDA done earlier: https://www.kaggle.com/pranalibose/wids-temp-feature-engineering it was clear that the temperature columns are highly correlated. We can remove the min and max columns for each month and proceed with just the avg temp of each month. Below, we have dropped the min max columns and grouped the remaining temp features into different season which brought down the temp features count from 36 to just 4.

# In[9]:


def temp_to_seasons(df):
    '''
    This function takes a dataframe and performs 3 tasks:
        1. Drop the min and max columns
        2. Converts the remaining into 4 seasons
        3. Drops the avg columns
    '''
    to_drop = []   
    
    # drop all the min and max columns
    for i in df.columns:
        if 'avg' not in i:
            to_drop.append(i)
    drop(df, to_drop)                        
    
    # grouping in seasons
    df['spring'] = (df['march_avg_temp'] + df['april_avg_temp'] + df['may_avg_temp'])/3
    df['summer'] = (df['june_avg_temp'] + df['july_avg_temp'] + df['august_avg_temp'])/3
    df['fall'] = (df['september_avg_temp'] + df['october_avg_temp'] + df['november_avg_temp'])/3
    df['winter'] = (df['december_avg_temp'] + df['january_avg_temp'] + df['february_avg_temp'])/3
    
    #dropping the avg columns
    to_drop = []
    for i in df.columns:
        if 'avg' in i:
            to_drop.append(i)
    drop(df, to_drop)


# In[10]:


temp_to_seasons(train_temp)
temp_to_seasons(test_temp)


# ### 2. Fog Columns 

# In[11]:


train_fog.head(2)


# In[12]:


missing = [i for i in train_fog.columns if train_fog[i].isnull().sum() != 0]    # capturing the cols with missing values


# In[13]:


# As we have a couple of outliers we will impute using the median
train_fog.fillna(train_fog.median(), inplace=True)
test_fog.fillna(test_fog.median(), inplace=True)


# ### 3. Other Features

# In[14]:


train_imp[train_imp['year_built'] == 0]                # year_built 0 doesn't make any sense


# We see a couple of 0's in year_built. Replacing that with median

# In[15]:


train_imp.replace(to_replace=0, value=train_imp['year_built'].median(), inplace=True)
test_imp.replace(to_replace=0, value=test_imp['year_built'].median(), inplace=True)


# In[16]:


# Imputing missing values with median

train_imp.fillna(train_imp.median(), inplace=True)
test_imp.fillna(test_imp.median(), inplace=True)


# In[17]:


train_imp.head(3)


# In[18]:


# Label Encoding
columns = [i for i in train_imp.columns if train_imp[i].dtypes == 'object']

def LE(train, test):
    le = LabelEncoder()
    for col in train.columns:
        if train[col].dtypes == 'object':
            train[col] = le.fit_transform(train[col])
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            test[col] = test[col].apply(lambda x: le_dict.get(x, -1))
    return train, test

train_df, test_df = LE(train_imp, test_imp)


# In[19]:


train_imp.head(3)


# ### Joining the 3 datasets

# Combining the 3 datasets

# In[20]:


train_clean = pd.concat([train_temp, train_fog, train_imp], axis=1)
test_clean = pd.concat([test_temp, test_fog, test_imp], axis=1)


# In[21]:


# adding the target variable in train df
train_clean['site_eui'] = target


# In[22]:


# Saving the clean data
# train_clean.to_csv('clean_train.csv', index=False)
# test_clean.to_csv('clean_test.csv', index=False)


# ### Feature Scaling 

# In[23]:


# Before Scaling
train_clean.head(3)


# In[24]:


# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_features = [i for i in train_clean.columns]
num_features.remove('site_eui')
num_features

train_clean[num_features] = scaler.fit_transform(train_clean[num_features])
test_clean[num_features] = scaler.transform(test_clean[num_features])


# In[25]:


# After Scaling
train_clean.head(3)


# The target variable normalization didn't help in improving model performance!

# In[26]:


# Check target column
train_clean['site_eui'].hist(color='navy')


# In[27]:


np.sqrt(train_clean['site_eui']).hist(color='green')


# In[28]:


# train_clean['site_eui'] = np.sqrt(train_clean['site_eui']) 


# ### Feature Selection 

# In[29]:


# creating a correlation dataframe with the corr values
corr_df = train_clean.select_dtypes('number').drop('site_eui', axis=1).corrwith(train_clean['site_eui']).sort_values().reset_index().rename(columns = {'index':'feature' ,0:'correlation'})

#Plotting
fig , ax = plt.subplots(figsize  = (8,10))
ax.barh(y =corr_df.feature , width = corr_df.correlation, color='lightgreen' )
ax.set_title('correlation between feature and target'.title() ,
            fontsize = 16 , fontfamily = 'serif' , fontweight = 'bold')
plt.show()
type(corr_df.feature[0])


# In[30]:


# List of features having minimal correlation
columns_with_low_correlation = corr_df[(corr_df.correlation >-0.03) & (corr_df.correlation<0.03)].feature.tolist()
columns_with_low_correlation


# In[31]:


# Dropping the irrelevant columns
drop(train_clean, columns_with_low_correlation)
drop(test_clean, columns_with_low_correlation)


# ### Modeling

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb


# In[33]:


def model_score(model, txt):
    '''
    Printing the performance metrics:
    R2 Square, Mean Absolute, Mean Squared Error, Root Mean Squared Error
    
    Parameters:
    model: Trained Model
    txt: To print the results for each model
    '''
    #Predicting the SalePrices using test set 
    y_pred = model.predict(x_test)            
    x_pred = model.predict(x_train)

    # Printing the metrics, comparing model's performance on seen and unseen data
    print('MSE on train:',metrics.mean_squared_error(y_train, x_pred))
    print('MSE on test:',metrics.mean_squared_error(y_test, y_pred))
    return x_pred, y_pred


# In[34]:


# train test split
X= train_clean.drop('site_eui',axis=1)
y= train_clean['site_eui']
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)


# ### Light GBM

# In[35]:


lgb= lgb.LGBMRegressor(n_estimators = 900 , max_depth=10, learning_rate=0.01, importance_type='gain', colsample_bytree = 0.4)
lgb.fit(x_train,y_train)

lgbm_x_pred, lgbm_y_pred = model_score(lgb, 'LGBM Regression Scores: ')


# ### CatBoost

# In[36]:


# Tuning

# params = { 'max_depth': range(3,11,1),
#            'learning_rate': np.arange(0.01,0.1,0.1),
#            'n_estimators': np.arange(100,1000,100),
#            'colsample_bytree': np.arange(0.3, 0.7,0.1)}

# xgbr = XGBRegressor(seed = 20)
# clf = RandomizedSearchCV(estimator=xgbr,
#                    cv=3,
#                    param_distributions=params,
#                    scoring='neg_root_mean_squared_error', 
#                    verbose=1,n_jobs=-1)
# clf.fit(x_train,y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest mean squared wrr: ", (clf.best_score_))


# In[37]:


cbr = CatBoostRegressor(learning_rate=0.01, max_depth=10, n_estimators=900, subsample=0.5)
cbr.fit(x_train, y_train)


# In[38]:


cbr_x_reg, cbr_y_reg = model_score(cbr, 'CatBoost Regression Scores: ')


# ### Random Forest

# In[39]:


rf = RandomForestRegressor(n_estimators = 50 ,  min_samples_split = 6, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 100, bootstrap=False)
rf.fit(x_train,y_train)

rf_x_predcv, rf_y_predcv = model_score(rf, 'Random Forest Scores: ')


# `Note:` Random Forest overfits the train set

# ### XGB Regressor

# In[40]:


xgb = XGBRegressor( n_estimators = 800 , max_depth=8, learning_rate=0.01, random_state = 0, colsample_bytree = 0.4)
xgb.fit(x_train,y_train)

xgb_x_pred, xgb_y_pred = model_score(xgb, 'XGB Regression Scores: ')


# ### Preparing the Submission File

# In[41]:


submission = cbr.predict(test_clean)         # Predict with the best model
submission                                  # This contains our predicted values


# In[42]:


row = row_id.to_numpy().flatten()


# In[43]:


# Formatting to create a submission csv file with just the id and target

submit = pd.DataFrame()
submit['id'] = row
submit['site_eui'] = submission
submit.head()


# In[44]:


# submit.to_csv('Submission.csv', index=False)


# ## Conclusions

# 1. year_built had a lot of 0 values. Handle it.
# 1. Scaling reduced the model performance drastically.
# 1. Removing the irrelevant features (correlation value less than 0.3) did not improve model's performance
# 1. CatBoost performs the best
# 1. Random Forest overfits the train set
