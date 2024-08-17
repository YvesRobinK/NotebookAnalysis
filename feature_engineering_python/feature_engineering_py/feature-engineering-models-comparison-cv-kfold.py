#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


sample_sub = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/sample_submission.csv")
sample_sub.head()


# In[3]:


train_scores = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
train_scores.head()


# In[4]:


train_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
train_logs.head()


# In[5]:


test_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
test_logs.head()


# In[6]:


train_logs.info(show_counts=True)


# In[7]:


train_df = pd.merge(train_logs, train_scores, on = "id", how = "inner")
train_df.head()


# # Feature Engineering

# In[8]:


df_agg_new = train_df.groupby(['id']).agg(
    #event_id_max=pd.NamedAgg(column="event_id", aggfunc="max"),
    action_time_mean=pd.NamedAgg(column="action_time", aggfunc="mean"),
    action_time_sum=pd.NamedAgg(column="action_time", aggfunc="sum"),
    action_time_min=pd.NamedAgg(column="action_time", aggfunc="min"),
    action_time_max=pd.NamedAgg(column="action_time", aggfunc="max"),
    word_count_max=pd.NamedAgg(column="word_count", aggfunc="max"),
    cursor_position_max=pd.NamedAgg(column="cursor_position", aggfunc="max"),
    activity_count=pd.NamedAgg(column="activity", aggfunc="count"),
    text_change_counts = pd.NamedAgg(column="text_change", aggfunc="count"),
    down_event_counts = pd.NamedAgg(column="down_event", aggfunc="count"),
    up_event_counts = pd.NamedAgg(column="up_event", aggfunc="count"),
    score = pd.NamedAgg(column="score", aggfunc="max")
).reset_index()


# In[9]:


df_agg_new.head()


# In[10]:


y = df_agg_new.score
X = df_agg_new.drop(['id','score'],axis=1)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


# In[12]:


# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Cross Valuation

# In[13]:


# Define a function that compares the CV perfromance of a set of predetrmined models 
def cv_comparison(models, X, y, cv):
    # Initiate a DataFrame for the averages and a list for all measures
    cv_accuracies = pd.DataFrame()
    maes = []
    mses = []
    r2s = []
    accs = []
    # Loop through the models, run a CV, add the average scores to the DataFrame and the scores of 
    # all CVs to the list
    for model in models:
        mae = -np.round(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv), 4)
        maes.append(mae)
        mae_avg = round(mae.mean(), 4)
        mse = -np.round(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv), 4)
        mses.append(mse)
        mse_avg = round(mse.mean(), 4)
        r2 = np.round(cross_val_score(model, X, y, scoring='r2', cv=cv), 4)
        r2s.append(r2)
        r2_avg = round(r2.mean(), 4)
        acc = np.round((100 - (100 * (mae * len(X))) / sum(y)), 4)
        accs.append(acc)
        acc_avg = round(acc.mean(), 4)
        cv_accuracies[str(model)] = [mae_avg, mse_avg, r2_avg, acc_avg]
    cv_accuracies.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'Accuracy']
    cv_accuracies.columns = ["Linear Regression","Random Forest","Extreme Gradient Boost"]
    return cv_accuracies, maes, mses, r2s, accs


# In[14]:


# Create the models to be tested
mlr_reg = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42)
xgb_reg = xgb.XGBRegressor(random_state=42)

# Put the models in a list to be used for Cross-Validation
models = [mlr_reg, rf_reg, xgb_reg]

# Run the Cross-Validation comparison with the models used in this analysis
comp, maes, mses, r2s, accs = cv_comparison(models, X_train, y_train, 4)


# In[15]:


comp


# Random Forest seems to be producing the best results

# ### Look at the results of all the folds

# In[16]:


# Create DataFrame for all maes
maes_comp = pd.DataFrame(maes, index=comp.columns, columns=['1st Fold', '2nd Fold', '3rd Fold', 
                                                         '4th Fold'])

# Add a column for the averages
maes_comp['Average'] = np.round(maes_comp.mean(axis=1),4)

maes_comp


# # HyperParameter Tuning with RandomSearch

# In[17]:


# Number of trees in Random Forest
rf_n_estimators = [int(x) for x in np.linspace(100, 500, 5)]

# Maximum number of levels in tree
rf_max_depth = [int(x) for x in np.linspace(2, 20, 10)]
# Add the default as a possible value
rf_max_depth.append(None)

# Number of features to consider at every split
rf_max_features = ['auto', 'sqrt', 'log2']

# Criterion to split on
#rf_criterion = ['squared_error','absolute_error']
rf_criterion = ['squared_error']

# Minimum number of samples required to split a node
rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

# Minimum decrease in impurity required for split to happen
rf_min_impurity_decrease = [0.0, 0.05, 0.1]

# Method of selecting samples for training each tree
rf_bootstrap = [True, False]

# Create the grid
rf_grid = {'n_estimators': rf_n_estimators,
               'max_depth': rf_max_depth,
               'max_features': rf_max_features,
               'criterion': rf_criterion,
               'min_samples_split': rf_min_samples_split,
               'min_impurity_decrease': rf_min_impurity_decrease,
               'bootstrap': rf_bootstrap}


# In[18]:


# Create the model to be tuned
rf_base = RandomForestRegressor()

# Create the random search Random Forest
rf_random = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid, 
                               n_iter = 50, cv = 3, verbose = 2, random_state = 42, 
                               n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

# View the best parameters from the random search
rf_random.best_params_


# In[19]:


# Number of trees to be used
xgb_n_estimators = [int(x) for x in np.linspace(100, 500, 5)]

# Maximum number of levels in tree
xgb_max_depth = [int(x) for x in np.linspace(2, 20, 10)]

# Minimum number of instaces needed in each node
xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]

# Tree construction algorithm used in XGBoost
xgb_tree_method = ['auto', 'exact', 'approx', 'hist']

# Learning rate
xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]

# Minimum loss reduction required to make further partition
xgb_gamma = [int(x) for x in np.linspace(0, 0.5, 6)]

# Learning objective used
xgb_objective = ['reg:squarederror']

# Create the grid
xgb_grid = {'n_estimators': xgb_n_estimators,
            'max_depth': xgb_max_depth,
            'min_child_weight': xgb_min_child_weight,
            'tree_method': xgb_tree_method,
            'eta': xgb_eta,
            'gamma': xgb_gamma,
            'objective': xgb_objective}


# In[20]:


# Create the model to be tuned
xgb_base = xgb.XGBRegressor()

# Create the random search Random Forest
xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = xgb_grid, 
                                n_iter = 50, cv = 3, verbose = 2, 
                                random_state = 420, n_jobs = -1)

# Fit the random search model
xgb_random.fit(X_train, y_train)

# Get the optimal parameters
xgb_random.best_params_


# # Training models with the best parameters

# In[21]:


mlr_final = LinearRegression()

# Create the final Random Forest
rf_final = RandomForestRegressor(n_estimators = 400,
                                 min_samples_split = 10,
                                 min_impurity_decrease = 0.0,
                                 max_features = 'log2',
                                 max_depth = 8,
                                 criterion = 'squared_error',
                                 bootstrap = True,
                                 random_state = 42)

# Create the fnal Extreme Gradient Booster
xgb_final = xgb.XGBRegressor(tree_method = 'auto',
                         objective = 'reg:squarederror',
                         n_estimators = 200,
                         min_child_weight = 1,
                         max_depth = 6,
                         gamma = 0,
                         eta = 0.1,
                         random_state = 42)

# Train the models using 80% of the original data
mlr_final.fit(X_train, y_train)
rf_final.fit(X_train, y_train)
xgb_final.fit(X_train, y_train)


# # Final Comparision between the models on the hold out set

# In[22]:


# Define a function that compares all final models
def final_comparison(models, test_features, test_labels):
    scores = pd.DataFrame()
    for model in models:
        predictions = model.predict(test_features)
        mae = round(mean_absolute_error(test_labels, predictions), 4)
        mse = round(mean_squared_error(test_labels, predictions), 4)
        r2 = round(r2_score(test_labels, predictions), 4)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = round(100 - mape, 4)
        scores[str(model)] = [mae, mse, r2, accuracy]
    scores.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'Accuracy']
    return scores


# In[23]:


# Call the comparison function with the three final models
final_scores = final_comparison([mlr_final, rf_final, xgb_final], X_test, y_test)

# Adjust the column headers
final_scores.columns  = ['Linear Regression', 'Random Forest', 'Extreme Gradient Boosting']


# In[24]:


final_scores


# Again, RF seems to be producing the best results

# # Final training with the whole training data

# In[25]:


rf_final.fit(X, y)


# # Submission

# In[26]:


df_test_agg = test_logs.groupby(['id']).agg(
    #event_id_max=pd.NamedAgg(column="event_id", aggfunc="max"),
    action_time_mean=pd.NamedAgg(column="action_time", aggfunc="mean"),
    action_time_sum=pd.NamedAgg(column="action_time", aggfunc="sum"),
    action_time_min=pd.NamedAgg(column="action_time", aggfunc="min"),
    action_time_max=pd.NamedAgg(column="action_time", aggfunc="max"),
    word_count_max=pd.NamedAgg(column="word_count", aggfunc="max"),
    cursor_position_max=pd.NamedAgg(column="cursor_position", aggfunc="max"),
    activity_count=pd.NamedAgg(column="activity", aggfunc="count"),
    text_change_counts = pd.NamedAgg(column="text_change", aggfunc="count"),
    down_event_counts = pd.NamedAgg(column="down_event", aggfunc="count"),
    up_event_counts = pd.NamedAgg(column="up_event", aggfunc="count")
).reset_index()


# In[27]:


sub_ids = df_test_agg['id']
df_test_agg.drop(['id'],axis=1,inplace=True)


# In[28]:


# lets make prediction on test dataset
y_sub = rf_final.predict(df_test_agg)


# In[29]:


# lets prepare for the prediction submission
sub = pd.DataFrame()
sub['id'] = sub_ids
sub['score'] = y_sub
sub.to_csv('submission.csv',index=False)


# In[ ]:




