#!/usr/bin/env python
# coding: utf-8

# <div style="color:#D81F26;
#            display:fill;
#            border-style: solid;
#            border-color:#C1C1C1;
#            font-size:14px;
#            font-family:Calibri;
#            background-color:#373737;">
# <h2 style="text-align: center;
#            padding: 10px;
#            color:#FFFFFF;">
# ======= Novozymes Enzyme Stability Prediction =======
# </h2>
# </div>

# # About this notebook

# This notebook is for submission to the Novozymes Enzymes Stability Prediction.
# 
# I just simply use n-letter sequence as features from the protein sequence and experiment if this feature can have predictive power to the target label "tm".
# 
# A XGBoost Regressor with hyperparametering is used to train the model. The best model is used for optimal model for regression.

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


# Basic library
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

# Scaler
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cross-Validation
from sklearn.model_selection import StratifiedKFold

#  For Modelling
from sklearn.model_selection import train_test_split
import xgboost


# # Load the data

# In[3]:


# Load the training, testing and submission data
TRAIN = "/kaggle/input/novozymes-enzyme-stability-prediction/train.csv"
TEST = "/kaggle/input/novozymes-enzyme-stability-prediction/test.csv"
SUBMISSION = "/kaggle/input/novozymes-enzyme-stability-prediction/sample_submission.csv"

df_train = pd.read_csv(TRAIN)
df_test = pd.read_csv(TEST)
df_submission = pd.read_csv(SUBMISSION)

print('Shape of the training dataset {}'.format(df_train.shape))
print('Shape of the testing dataset {}'.format(df_test.shape))
print('Shape of the submission dataset {}'.format(df_submission.shape))                                                


# # Explortory Data Analysis

# In[4]:


df_train.info()


# In[5]:


df_train.describe()


# ## Distribution of numerical variables

# In[6]:


numerical = ['pH', 'tm']
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

for col in numerical:
    if pd.api.types.is_numeric_dtype(df_train[col]) == True:
        df_train[col].plot.hist(bins=50, grid=True, legend=None)
        plt.title(col)
        plt.show()


# #### Observation: The pH data is skewed to the right.   We will remove the outlier (e.g. top X% quantile) and analyze the distribution again. 

# In[7]:


df_pH = df_train.pH
print('Shape of the pH temp dataset {}'.format(df_pH.shape))


# ## Remove outliers by percentile analysis

# In[8]:


# Get the X% percentile of the pH to remove outlier and plot the distribution again 
pH_05 = df_pH.quantile(0.05)
pH_10 = df_pH.quantile(0.10)
pH_95 = df_pH.quantile(0.95)
pH_99 = df_pH.quantile(0.99)
print('p05 = {}, p10 = {}'.format(pH_05, pH_10))
print('p95 = {}, p99 = {}'.format(pH_95, pH_99))


# In[9]:


df_train[(df_train['pH']<pH_99) & (df_train['pH']>pH_05)]['pH'].plot.hist(bins=50, grid=True, legend=None)
plt.title('pH')
plt.show()


# #### Observation: after setting the cap and floor for pH feature,  we can see that most pH values are closed to 7

# In[10]:


df_train['data_source'].value_counts()


# #### to review the distribution of the features in test data set to determine if pH and data_source has discriminatory information

# In[11]:


df_test['data_source'].value_counts()


# #### Observation: data_source has no irrelevant to the target label.

# In[12]:


# Get the X% percentile of the pH to remove outlier and plot the distribution again 
pH_05 = df_test['pH'].quantile(0.05)
pH_10 = df_test['pH'].quantile(0.10)
pH_95 = df_test['pH'].quantile(0.95)
pH_99 = df_test['pH'].quantile(0.99)
print('p05 = {}, p10 = {}'.format(pH_05, pH_10))
print('p95 = {}, p99 = {}'.format(pH_95, pH_99))


# In[13]:


df_test['pH'].plot.hist(bins=50, grid=True, legend=None)
plt.title('pH')
plt.show()


# #### Observation: the pH value has no predictive power to the target label

# 
# # Feature engineering

# In[14]:


# Now, we analyze if 1-letter sequence code is representative. E.g. we know that  L, A, V and G are popular sequence code 
Protein_Seq = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

Null_Seq = []

for col in Protein_Seq:
    if df_train['protein_sequence'].str.contains(col).sum() == 0 :
        print('{}: {}'.format(col, df_train['protein_sequence'].str.contains(col).sum()))
        Null_Seq.append(col)
        
print('Sequence with no occurence : {}'.format(Null_Seq))    


# #### Observation: Code sequence of B, J, O, U, X and Z are not representative.  

# In[15]:


# We only include non-zero 1-letter sequence code in the feature extraction
Letter_1_Seq = []

for i in Protein_Seq:
    if i not in Null_Seq:
        Letter_1_Seq.append(i)

print('1-letter Protein Sequence : {}'.format(Letter_1_Seq))

print('Size of training and testing dataset before extracting 1-letter sequence: {} and {}'.format(df_train.shape, df_test.shape))

for col in Letter_1_Seq:
#     print('{}: {}'.format(col, df_train['protein_sequence'].str.contains(col).sum()))
    df_train[col] = df_train['protein_sequence'].str.count(col)
    df_test[col] = df_test['protein_sequence'].str.count(col)
    
print('Size of training and testing dataset after extracting 1-letter sequence: {} and {}'.format(df_train.shape, df_test.shape))


# In[16]:


print(df_train.head(10))    


# In[17]:


Protein_Seq = Letter_1_Seq


# ## Standardization for numerical labels

# In[18]:


standardScaler = StandardScaler()
encoder_num = standardScaler.fit_transform(df_train[Protein_Seq])
encoded_num = pd.DataFrame(encoder_num, columns =Protein_Seq)
print(encoded_num.shape)
print(encoded_num.head(5))


# # Model - XGBooster

# In[19]:


X = encoded_num.copy()
y = df_train['tm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [10, 15, 20, 25],
        'min_child_weight': [5, 10, 20, 30],
        'subsample': [0.3, 0.5, 0.7],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'n_estimators' : [500], # , 500, 750, 1000
        'objective': ['reg:squarederror']
    }
    
    xgb_model = xgboost.XGBRegressor()

    gsearch = RandomizedSearchCV(estimator = xgb_model,
                           param_distributions = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 3,
                           n_iter=35, 
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_


# In[21]:


#Run only in the first run of the kernel.
# best_model = hyperParameterTuning(X_train, y_train)


# In[22]:


# print(best_model)


# In[23]:


# best fit
# xgb_model_best = xgboost.XGBRegressor(**best_model)

# %time xgb_model_best.fit(X_train, y_train,early_stopping_rounds=300, eval_set=[(X_test, y_test)], verbose=False)


# In[24]:


model_XGBoost = xgboost.XGBRegressor(n_estimators=500, max_depth=15, learning_rate= 0.001)

model_XGBoost.fit(X_train, y_train)

print('Model model_XGBoost Training is done!')


# ## Spearman's rank correlation coefficient

# In[25]:


from scipy import stats

y_pred_xgboost = model_XGBoost.predict(X_test)

stats.spearmanr(y_test, y_pred_xgboost)


# In[26]:


print(y_pred_xgboost)
print(y_test)


# ## Prediction 

# In[27]:


# Do the same data preparatioon for the test data
# Now, we analyze if 1-letter sequence code is representative. E.g. we know that  L, A, V and G are popular sequence code 

df_pred = df_test.copy()

encoded_num = standardScaler.transform(df_pred[Protein_Seq])

df_pred = pd.DataFrame(encoded_num, columns =Protein_Seq)

print(df_pred.head(5))


# In[28]:


submission = pd.DataFrame()
submission['seq_id'] = df_test['seq_id']
submission['tm'] =model_XGBoost.predict(df_pred)

print(submission.head(10))

submission.to_csv("submission.csv", index=False)
print('Submission Done')

