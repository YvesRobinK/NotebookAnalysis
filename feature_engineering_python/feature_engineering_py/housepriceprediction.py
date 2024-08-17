#!/usr/bin/env python
# coding: utf-8

# ## Load the data

# In[ ]:


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


# In[ ]:


train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# # Exploratory Data Analysis

# In[ ]:


train_data.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.heatmap(train_data.corr(), center = 0)
plt.title("Correlations Between Columns")
plt.show()


# ## Split input and target variables

# In[ ]:


y = train_data.SalePrice
X = train_data.drop(columns=["SalePrice"], axis=1)


# In[ ]:


y.shape, X.shape, test_data.shape


# # Feature Engineering

# ## Choose only the significant features, discard those with correlation score < 0.5 with the target variable

# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


corr_matrix['SalePrice'][(corr_matrix["SalePrice"] > 0.40) | (corr_matrix["SalePrice"] < -0.40)]


# In[ ]:


important_num_cols = list(corr_matrix['SalePrice'][(corr_matrix["SalePrice"] > 0.5) | (corr_matrix["SalePrice"] < -0.5)].index)

important_num_cols.remove('SalePrice')
len(important_num_cols)


# In[ ]:


important_num_cols


# In[ ]:


X_num_only = X[important_num_cols]


# In[ ]:


X_num_only.shape


# ## Remove the feautures which are highly correlated with each other

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(X_num_only.corr(), center = 0)
plt.title("Correlations Between Columns")
plt.show()


# In[ ]:


corr_X = X_num_only.corr()
len(corr_X)


# In[ ]:


for i in range(0, len(corr_X) - 1):
    for j in range(i + 1, len(corr_X)):
        if(corr_X.iloc[i, j] < -0.6 or corr_X.iloc[i, j] > 0.6):
            print(corr_X.iloc[i, j], i, j, corr_X.index[i], corr_X.index[j])
            


# In[ ]:


# Based on the above information, we further discard the features 1stFlrSF, FullBath, TotRmsAbvGrd, GarageArea
#num_cols = [i for i in X_modified.columns if i not in ['1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageArea']]
num_cols = [i for i in X_num_only.columns if i not in ['1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageArea']]


# In[ ]:


# Categorical columns - choose the important ones

cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]


# In[ ]:


X_final = X[num_cols]


# In[ ]:


X_final.shape


# ## Modify 'YearRemodAdd' feature - make it more informative

# In[ ]:


X_final['YearRemodAdd'] = X_final['YearRemodAdd'] - X_final['YearBuilt']


# In[ ]:


X_final.head()


# In[ ]:





# # Handling missing data

# In[ ]:


X_final.isna().sum()


# In[ ]:


#X_final['MasVnrArea'] = X_final['MasVnrArea'].fillna(X_final['MasVnrArea'].median())


# In[ ]:


X[cat_cols].isna().sum()


# In[ ]:





# # Encoding Categorical data

# In[ ]:


X_categorical_df = pd.get_dummies(X[cat_cols], columns=cat_cols)


# In[ ]:


X_categorical_df


# In[ ]:


# Create final dataframe


# In[ ]:


X_final = X_final.join(X_categorical_df)


# In[ ]:


X_final


# # Normalizing the data

# In[ ]:


from sklearn import preprocessing
standardize = preprocessing.StandardScaler().fit(X_final[num_cols])


# In[ ]:


#See mean per column
standardize.mean_


# In[ ]:


#transform
X_final[num_cols] = standardize.transform(X_final[num_cols])


# In[ ]:


X_final


# In[ ]:





# In[ ]:


X_final.head()


# ## Split training data into training and validation

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=1)


# In[ ]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# # Regression Using Machine Learning 

# In[ ]:


from sklearn.metrics import r2_score 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


perf = []
method = []


# In[ ]:


from sklearn.metrics import mean_squared_log_error


# In[ ]:





# In[ ]:


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)
rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
method.append('Linear Regression')
perf.append(rmsle)


# In[ ]:


# Ridge regression
ridge = Ridge()
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)
method.append('Ridge Regression')

rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
perf.append(rmsle)


# In[ ]:


# Ridge regression
lasso = Lasso()
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)
method.append('Lasso Regression')

rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
perf.append(rmsle)


# In[ ]:


# support vector regression
from sklearn.svm import SVR
svr = SVR(C=1000000)
svr.fit(X_train, y_train)
predictions = svr.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)
#method.append('SVM')
rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
#perf.append(rmsle)


# In[ ]:


svr_rbf = SVR(kernel="rbf", C=1000000, gamma=0.01, epsilon=0.1)
svr_rbf.fit(X_train, y_train)
predictions = svr_rbf.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)

method.append('SVR')
rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
perf.append(rmsle)


# In[ ]:


#Random forest regressor
for i in range(50 , 500, 50):
    random_forest = RandomForestRegressor(n_estimators=i)
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_val)

    r_squared = r2_score(predictions, y_val)

    print("R2 Score:", r_squared)
    method.append('Random Forest Regressor')
    rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
    print("RMSLE:", rmsle)
    perf.append(rmsle)


# In[ ]:





# In[ ]:


# xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_val)

r_squared = r2_score(predictions, y_val)

print("R2 Score:", r_squared)
method.append('XGBoost Regressor')
rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
perf.append(rmsle)


# In[ ]:


# ANN
'''
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError



hidden_units1 = 400
#hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01
# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
    model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
      ])
    return model
# build the model
model = build_model_using_sequential()

# loss function
msle = MeanSquaredLogarithmicError()
model.compile(
    loss=msle, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[msle]
)

# train the model
history = model.fit(
    X_final.values, 
    y.values, 
    epochs=1000, 
    batch_size=64,
    validation_split=0.2
)
predictions = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(predictions, y_val))
print("RMSLE:", rmsle)
method.append('ANN')
perf.append(rmsle)
'''


# In[ ]:


# Compare performances of models
plt.barh(method, perf)
plt.title('RMSLE comparison of models')


# # Testing

# In[ ]:


# Test Data Preprocessing

X_test = test_data[num_cols + cat_cols]
X_test['YearRemodAdd'] = X_test['YearRemodAdd'] - X_test['YearBuilt']


# In[ ]:


X_test.shape


# In[ ]:


# Encode categorical similar to train
X_test = pd.get_dummies(X_test)


# In[ ]:


X_test


# In[ ]:


# Add missed columns missed due to get dummies on X_test
X_test = X_test.reindex(columns = X_final.columns, fill_value=0)


# In[ ]:


X_test


# In[ ]:


#transform
X_test[num_cols] = standardize.transform(X_test[num_cols])


# In[ ]:


X_test


# ## Handling missing data in test data

# In[ ]:


X_test.isna().sum()


# In[ ]:


# we will use median for missing values
X_test['TotalBsmtSF'] = X_test['TotalBsmtSF'].fillna(train_data['TotalBsmtSF'].median())


# In[ ]:


# mode for cars
X_test['GarageCars'] = X_test['GarageCars'].fillna(train_data['GarageCars'].mode()[0])


# In[ ]:


# Submission using SVR

preds = svr_rbf.predict(X_test)
submit = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': preds})
submit.to_csv('submission.csv',index=False)


# In[ ]:


# Submission using ANN
'''
preds = model.predict(X_test)
preds_2 = [i[0] for i in preds]
out = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': preds_2}) 
out.to_csv('submission.csv',index=False)
'''


# In[ ]:





# In[ ]:





# In[ ]:




