#!/usr/bin/env python
# coding: utf-8

# # Bike Shring Demand Top 6.6% Solution for Everyone !!

# ### This is a simple modeling notebook using Random Forest Regression. This model reaches the top 6.6%. If you think it's useful, please upvote ^^ 

# ### I also shared [basic EDA notebook for everyone](https://www.kaggle.com/werooring/bike-sharing-demand-basic-eda-for-everyone)

# - [Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand)
# 
# - [Modeling Reference Notebook](https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile)

# ### Load Data

# In[1]:


import numpy as np
import pandas as pd

train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")
submission = pd.read_csv("/kaggle/input/sampleSubmission.csv")


# ### Concatenate train and test data

# In[2]:


all_data_temp = pd.concat([train, test])
all_data_temp


# In[3]:


all_data = pd.concat([train, test], ignore_index=True)
all_data


# ## Feature Engineering

# ### Create new features

# In[4]:


from datetime import datetime

all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0]) # Create date feature
all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0]) # Create year feature
all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1]) # Create month feature
all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0]) # Create hour feature
all_data["weekday"] = all_data['date'].apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday()) # Create weekday feature


# ### Change categorical data type for memory reduction

# In[5]:


categorical_features = ['season', 'holiday', 'workingday', 'weather', 'weekday', 'month', 'year', 'hour']

for feature in categorical_features:
    all_data[feature] = all_data[feature].astype("category")


# ### Separate train and test data. Assign train target value(y)

# In[6]:


train = all_data[pd.notnull(all_data['count'])]
test = all_data[~pd.notnull(all_data['count'])]
y = train['count']


# ### Drop useless features

# In[7]:


drop_features = ['count', 'casual', 'registered', 'datetime', 'date', 'datetime', 'windspeed', 'month']

X_train = train.drop(drop_features, axis=1)
X_test = test.drop(drop_features, axis=1)


# ### Check final features and types

# In[8]:


X_train.info()


# ## Train Model and Measure Model Performance

# ### Evaluation score(RMSLE) function

# In[9]:


def rmsle(y_true, y_pred, convertExp=True):
    # Apply exponential transformation function
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
        
    # Convert missing value to zero after log transformation
    log_true = np.nan_to_num(np.array([np.log(y+1) for y in y_true]))
    log_pred = np.nan_to_num(np.array([np.log(y+1) for y in y_pred]))
    
    # Compute RMSLE
    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output


# ### Linear regression model

# In[10]:


from sklearn.linear_model import LinearRegression

# Step 1: Create Model
linear_reg_model = LinearRegression()

# Step 2: Train Model
log_y = np.log1p(y)  # Log Transformation of Target Value y
linear_reg_model.fit(X_train, log_y) 

# Step 3 : Predict
preds = linear_reg_model.predict(X_train)

# Step 4 : Evaluate
print ('Linear Regression RMSLE:', rmsle(log_y, preds, True))


# ### Ridge Model (Apply Gridsearch)

# In[11]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Step 1: Create Model
ridge_model = Ridge()

# Step 2-1 : Create GridSearchCV Object
# Hyper-parameter List
ridge_params = {'max_iter':[3000], 'alpha':[0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]}
# Evaluate Function for Cross-Validation (RMSLE score)
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False) 
# Create GridSearchCV Object (with Ridge)
gridsearch_ridge_model = GridSearchCV(estimator=ridge_model,
                                      param_grid=ridge_params,
                                      scoring=rmsle_scorer,
                                      cv=5)

# Step 2-2 : Perform Grid Search
log_y = np.log1p(y) # Log Transformation of Target Value y
gridsearch_ridge_model.fit(X_train, log_y) # Train (Grid Search)

print('Best Parameter:', gridsearch_ridge_model.best_params_)

# Step 3 : Predict
preds = gridsearch_ridge_model.best_estimator_.predict(X_train)

# Step 4 : Evaluate
print('Ridge Regression RMSLE:', rmsle(log_y, preds, True))


# ### Lasso Model (Apply Gridsearch)

# In[12]:


from sklearn.linear_model import Lasso

# Step 1: Create Model
lasso_model = Lasso()

# Step 2-1 : Create GridSearchCV Object
# Hyper-parameter List
lasso_alpha = 1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
lasso_params = {'max_iter':[3000], 'alpha':lasso_alpha}
# Create GridSearchCV Object (with Lasso)
gridsearch_lasso_model = GridSearchCV(estimator=lasso_model,
                                      param_grid=lasso_params,
                                      scoring=rmsle_scorer,
                                      cv=5)


# Step 2-2 : Perform Grid Search
log_y = np.log1p(y)
gridsearch_lasso_model.fit(X_train, log_y) # Train (Grid Search)

print('Best Parameter:', gridsearch_lasso_model.best_params_)

# Step 3 : Predict
preds = gridsearch_lasso_model.best_estimator_.predict(X_train)

# Step 4 : Evaluate
print('Lasso Regression RMSLE:', rmsle(log_y, preds, True))


# ### Random Forest Regression Model (Apply Grid Search)

# In[13]:


from sklearn.ensemble import RandomForestRegressor

# Step 1: Create Model
randomforest_model = RandomForestRegressor()

# Step 2-1 : Create GridSearchCV Object
# Hyper-parameter List
rf_params = {'random_state':[42], 'n_estimators':[100, 120, 140]}
# Create GridSearchCV Object (with Random Forest Regression)
gridsearch_random_forest_model = GridSearchCV(estimator=randomforest_model,
                                              param_grid=rf_params,
                                              scoring=rmsle_scorer,
                                              cv=5)

# Step 2-2 : Perform Grid Search
log_y = np.log1p(y)
gridsearch_random_forest_model.fit(X_train, log_y)

print('Best Parameter:', gridsearch_random_forest_model.best_params_)

# 스텝 3 : 예측
preds = gridsearch_random_forest_model.best_estimator_.predict(X_train)

# 스텝 4 : 평가
print('Random Forest Regression RMSLE:', rmsle(log_y, preds, True))


# ### Random Forest Regression Model is the Best among four models!

# ## Submit

# ### Compare train data vs predicted test data distribution 

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

randomforest_preds = gridsearch_random_forest_model.best_estimator_.predict(X_test)

figure, axes = plt.subplots(ncols=2)
figure.set_size_inches(10, 4)

sns.distplot(y, ax=axes[0], bins=50)
axes[0].set_title('Train Data Distribution')
sns.distplot(np.exp(randomforest_preds), ax=axes[1], bins=50)
axes[1].set_title('Predicted Test Data Distribution');


# ### submit final predictions

# In[15]:


submission['count'] = np.exp(randomforest_preds)
submission.to_csv('submission.csv', index=False)


# # I would appreciate it if you upvote my notebook. Thank you!
