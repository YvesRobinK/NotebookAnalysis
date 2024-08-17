#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# # Importing Libraries

# In[6]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
import lightgbm
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# # Importing Dataset

# In[7]:


# Importing Dataset

train_data = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv') # Importing training data
test_data = pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv') # Importing test data

# Displaying Data Shape
print(f"Training Data shape :{train_data.shape}")
print(f"Test Data shape :{test_data.shape}")


# # Data Profiling

# 1. Data Summary
# 2. Missing Values
# 3. Outliers
# 4. Zero/ Non Zero Values

# In[8]:


# Understanding the summary statistics of the data

def data_summary(df):
    # Summary statistics
    summary_stats = df.describe()

    # Count of missing values
    missing_values = df.isnull().sum()

    # Count of outliers
    outliers = {}
    for col in df.columns:
        if df[col].dtype != 'object':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            num_outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            outliers[col] = num_outliers

    # Other measures
    measures = {}
    for col in df.columns:
        if df[col].dtype != 'object':
            unique_values = df[col].nunique()
            zero_values = len(df[df[col] == 0])
            measures[col] = {'Unique Values': unique_values, 'Zero Values': zero_values}
            
    combined_table = pd.DataFrame(pd.concat([summary_stats.T, pd.Series(missing_values, name='Missing'), pd.Series(outliers, name='Outliers')], axis=1))

    return combined_table, measures


# In[9]:


# Analysising the training data

stats_train, measures_train = data_summary(train_data)
display(stats_train, measures_train)


# In[10]:


# Analysising the test data

stats_test, measures_test = data_summary(test_data)
display(stats_test, measures_test)


# # Data Visualisation

# 1. Univariate Analysis
# * Pie Chart
# * Box Plots

# In[11]:


values = train_data["Sex"].value_counts()
labels = ["M", "I", "F"]
fig = px.pie(train_data, values=values, names=labels, title="Distribution of Sex")
fig.show()


# In[12]:


values = test_data["Sex"].value_counts()
labels = ["M", "I", "F"]
fig = px.pie(test_data, values=values, names=labels, title="Distribution of Sex")
fig.show()


# In[13]:


numeric_columns = train_data.select_dtypes(include='number')

# Determine the number of numeric columns
num_columns = numeric_columns.shape[1]

# Create subplots for each numeric column
fig, axs = plt.subplots(1, num_columns, figsize=(12, 4))

# Plot boxplots for each numeric column
for i, column in enumerate(numeric_columns):
    axs[i].boxplot(numeric_columns[column])
    axs[i].set_title(column)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[14]:


numeric_columns = test_data.select_dtypes(include='number')

# Determine the number of numeric columns
num_columns = numeric_columns.shape[1]

# Create subplots for each numeric column
fig, axs = plt.subplots(1, num_columns, figsize=(12, 4))

# Plot boxplots for each numeric column
for i, column in enumerate(numeric_columns):
    axs[i].boxplot(numeric_columns[column])
    axs[i].set_title(column)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# 2. Bivariate Analysis
# * Pairplots

# In[16]:


# Pairplot on Train Data

fig = px.scatter_matrix(train_data.drop('id',axis=1),
                        dimensions=['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight',
                                    'Viscera Weight', 'Shell Weight', 'Age'],
                        color='Sex')
fig.update_layout(title='Pairplot on Training Data', width=1500, height=1200)
fig.show()


# In[17]:


# Pairplot on Test Data

fig = px.scatter_matrix(test_data.drop('id',axis=1),
                        dimensions=['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight',
                                    'Viscera Weight', 'Shell Weight'],
                        color='Sex')
fig.update_layout(title='Pairplot on Test Data', width=1500, height=1200)
fig.show()


# Correlation Analysis

# In[21]:


plt.figure(figsize=(10,8))
sns.heatmap(train_data.drop('Sex',axis=1).corr(),cmap = "Reds", annot=True,)


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(test_data.drop('Sex',axis=1),cmap = "Reds", annot=True)


# # Data Modelling

# 1. Fine-tune XGBoost, LightGBM and CatBoost
# 2. Stack XGBoost, LightGBM and CatBoost
# 3. Compute Median of output from three models and round up

# In[20]:


def evaluate_model_with_stack(X, y, model1,model2,model3, num_folds=10, n_repeats=5, random_state=1):
    kf = RepeatedKFold(n_splits=num_folds, n_repeats=n_repeats, random_state=random_state)
    maes = []
    
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_val)
        
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_val)
        
        model3.fit(X_train, y_train)
        y_pred3 = model3.predict(X_val)
        
        X_stack = np.column_stack((y_pred1, y_pred2, y_pred3))
        stacked_predictions = np.round(np.median(X_stack, axis=1))

        mae = mean_absolute_error(y_val, stacked_predictions)
        maes.append(mae)

    mean_mae = np.mean(maes)
    return mean_mae


data = pd.get_dummies(train_data.drop(['id'],axis=1).drop_duplicates(),drop_first=True, dtype='int32')
data['Height'] = data['Height'].replace('0.0','0.0125')
X = data.drop('Age',axis=1)
y = data['Age']

# Evaluate the model
xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror',learning_rate=0.1, max_depth=7, min_child_weight=3, n_estimators=200, subsample=0.7, gamma=0.1, reg_lambda=0, reg_alpha=0)
lgb_model = lightgbm.LGBMRegressor(max_depth=5,num_leaves=120, n_estimators=1800,objective='mae', metric='mean_absolute_error',reg_alpha=0.000012, reg_lambda=0.45)
catboost_model = CatBoostRegressor(loss_function='MAE',eval_metric='MAE',bagging_temperature=2.5, colsample_bylevel=0.75,learning_rate=0.067,od_wait=40,max_depth=6,l2_leaf_reg=1.575,min_data_in_leaf=28,random_strength=0.55, max_bin=256, logging_level='Silent')


score = evaluate_model_with_stack(X,y, xgb_model, lgb_model, catboost_model)
print("Average MAE after Stacking: ",score)


# <h2> Thank you for reading till end! If you like it, please give a upvote </h2>

# # Future work
# 
# 1. **Feature Engineering**- Compute the feature and perform the analysis again
#     
#     1.1 Volume: Calculate the volume of the crab using the length, diameter, and height. The formula for calculating the volume of a cylindrical object (approximating the shape of a crab) is V = π * (Diameter/2)^2 * Height. This derived feature takes into account the overall size and shape of the crab.
#     
#     1.2 Density: Calculate the density of the crab by dividing the weight by the volume. Density can provide insights into the crab's growth and development.
#     
#     1.3 Shell-to-Body Ratio: Calculate the ratio of shell weight to the total weight of the crab (weight + shell weight). This ratio can indicate the proportion of the crab's weight that is attributed to the shell, which may have implications for its age and development.
#     
#     1.4 Meat-Yield: Calculate the ratio of shucked weight (weight without the shell) to the total weight of the crab. This feature represents the proportion of edible meat relative to the total weight and can be relevant for commercial fishing or aquaculture contexts.
#     
#     1.5 Body-Condition-Index: Create a composite feature that combines length, weight, and shucked weight. For example, you can calculate the square root of the product of length, weight, and shucked weight. This index may provide an indication of the crab's overall health and condition.
#     
# 
# 2. **Modelling**
# 
#      2.1 Implmenting voting regressor
#      
#      2.2 Attempt NN model
#      
# 3. **Focus on quasi duplicate data treamtment**

# In[ ]:




