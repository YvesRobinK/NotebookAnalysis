#!/usr/bin/env python
# coding: utf-8

# # Tabular Playground Series - Aug 2022
# 
# **About the data**
# This data represents the results of a large product testing study. For each product_code you are given a number of product attributes (fixed for the code) as well as a number of measurement values for each individual product, representing various lab testing methods. Each product is used in a simulated real-world environment experiment, and and absorbs a certain amount of fluid (loading) to see whether or not it fails.
# 
# Your task is to use the data to predict individual product failures of new codes with their individual lab test results.
# 
# **About the notebook**
# In this notbook, the `Poisson Regressor` model will be used to classify the target value based on the categories and numerical variables. The following are the steps present within this notebook:
# 
# 1. Installing Peripheral libraries
#     * Installing dataprep package for visualization and EDA
# 2. Importing Necessary Libraries
# 3. Import the dataset
# 4. Exploratory Data Analysis
#     * Visulize the correlation
#     * Plot statistics for the failure column of the train data
#     * Report generated for the train data
#     * Train and test data comparison
# 5. Data Preprocessing
#     * Feature Engineering the number code
#     * Encode categorical dimensions using `LabelEncoder()`
#     * Fill missing values using `KNNImputer()`
#     * Scale data using `StandardScaler()`
# 6. Finding the best parameters using Grid Search
# 7. Submit predictions

# # Installing peripheral libraries

# In[1]:


# !pip install dataprep


# # Importing Necessary Libraries

# In[2]:


# Data Wrangling libraries
import numpy as np
import pandas as pd

# Visualization Libraries
from IPython.display import display,HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing libraries
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Machine Learning Estimators
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# As per this notebook, we disabled dataprep viz due to viewing errors.
# You could uncomment the code upon forking to use the module
# from dataprep.eda import plot, plot_correlation, create_report, plot_missing, plot_diff

import warnings
warnings.filterwarnings('ignore')


# # Import the datasets
# 
# Data wasa taken from Kaggle's Tabular Playground Series - Aug 2022.

# In[3]:


train_data = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv') 


# # Exploratory Data Analysis
# 
# The following are interactive visualizations as provided by [dataprep.ai](https://dataprep.ai/). You can visit their website for more information. In this Notebook, this would be the primary tool for the exploratory data analysis. The outputs are interactive and clickable. Therefore, you could view different graphs and analysis by hovering on the columns and figures.

# In[4]:


# Visulize the correlation
# plot_correlation(train_data)


# In[5]:


# Plot statistics for the failure column of the train data
# plot(train_data, 'failure')


# In[6]:


# Report generated for the train data
# Note: This code make take a couple of minutes depending on the speed of your device.
# create_report(train_data)


# In[7]:


# View missing data
# plot_missing(train_data)


# In[8]:


# Train and test data comparison
# plot_diff([train_data, test_data])


# # Data Preprocessing
# 
# **Steps:**
# 
# 1. Feature Engineering the number code
# 2. Encode categorical dimensions using `LabelEncoder()`
# 3. Fill missing values using `KNNImputer()`
# 4. Scale data using `StandardScaler()`

# ## Feature Engineering the number code

# In[9]:


def extract_num_code(data, features = ['attribute_0', 'attribute_1']):
    for col in data[features].columns:
        data[col] = data[col].str.split('_', 1).str[1].astype('int')
    return data

train_data = extract_num_code(train_data, features = ['attribute_0', 'attribute_1'])
test_data = extract_num_code(test_data, features = ['attribute_0', 'attribute_1'])


# ## Encode categorical dimensions using `LabelEncoder()`

# In[10]:


# List of the categorical features.
cat_feat = ['product_code',
 'attribute_0',
 'attribute_1',
 'attribute_2',
 'attribute_3',
 'measurement_0',
 'measurement_1',
 'measurement_2']


# In[11]:


def encode(data, cat_fatures=cat_feat):
    encoder = LabelEncoder()
    for feat in cat_feat:
        data[feat] = encoder.fit_transform(data[[feat]])
    return data

train_data = encode(train_data)
test_data = encode(test_data)


# ## Fill missing values using `KNNImputer()`

# In[12]:


# Credits to TheDevastors' notebook for this imputing method
def fill_missing(data):
    imputer = KNNImputer(n_neighbors=3)
    for col in data.columns:
        data[col] = imputer.fit_transform(data[[col]])
    return data

train_data = fill_missing(train_data)
test_data = fill_missing(test_data)


# In[13]:


train_data.isna().sum()


# ## Scale data using `StandardScaler()`

# In[14]:


X_train = train_data.drop(['failure', 'id'], axis=1)
y_train = train_data['failure']


# In[15]:


def scale_data(data):
    scaler = StandardScaler()
    data.loc[:] = scaler.fit_transform(data)
    return data

X_train = scale_data(X_train)
X_test = scale_data(test_data.drop('id', axis=1))


# # Finding the best parameters using Grid Search
# 
# Note that since the Poisson Regressor does not have a predict_proba method, it is necessary to set the `fit_intercept` parameter as `True`. More on this can be found on the sci-kit learn documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html).

# In[16]:


parameters = {
    "alpha":[1, 2, 3, 4],
    "fit_intercept": [True],  # Important to be True since the PoissonRegressor do not have a predict_proba method
    "max_iter": [500, 1000, 1500],
    "tol":[1e-2, 1e-3, 1e-4, 1e-5],
    "verbose":[0],
    
}

model_poisson = PoissonRegressor()

model_poisson = GridSearchCV(
    model_poisson, 
    parameters, 
    cv=5,
    scoring='accuracy',
)

model_poisson.fit(X_train, y_train)

print('-----')
print(f'Best parameters {model_poisson.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + 
    f'{model_poisson.best_score_:.3f}'
)

y_preds_poisson = model_poisson.best_estimator_.predict(X_test)


# ## Optional: 
# 
# If the Grid Search is taking too long. You could opt to just run the following line of code. This includes the best parameters in my case.

# In[17]:


# poisson = PoissonRegressor(alpha=1, fit_intercept=True, max_iter=500, tol=0.01, verbose=0)
# poisson.fit(X_train, y_train)
# preds = poisson.predict(X_test)
# preds


# ## Submit Predictions

# In[18]:


test_data = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')

submission_dataframe = pd.DataFrame(test_data['id'])
submission_dataframe['failure'] = y_preds_poisson


# In[19]:


submission_dataframe.to_csv('submission_test.csv', index=False)


# In[20]:


submission_dataframe


# In[ ]:




