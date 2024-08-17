#!/usr/bin/env python
# coding: utf-8

# The trend of anonymized data for online competitions is increasing day by day as companies want their data to be secure and thus maintaining the privacy of their customers. Santander has released an anonymized dataset for predicting the value of transactions for each potential customer.
# 
# So in this notebook I will be focusing on gathering insights from the unknown data and selecting appropriate subset of features.

# # Importing modules and getting a glimpse of the data

# In[ ]:


import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os
print(os.listdir("../input"))
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


print("The shape of the training set is:",train_data.shape)


# In[ ]:


print("The shape of the test set is:", test_data.shape)


# - It is quiet interesting to see that the number of features in the train dataset is greater than the number of data points i.e. **the curse of dimensionality**.
# - The test set is 10 times bigger than the train set in shape.
# - Thus, feature extraction is very important and will substantially improve the score of the model.

# In[ ]:


feature_cols = [c for c in train_data.columns if c not in ["ID", "target"]]
flat_values = train_data[feature_cols].values.flatten()

labels = 'Zero_values','Non-Zero_values'
values = [sum(flat_values==0), sum(flat_values!=0)]
colors = ['rgba(55, 12, 233, .6)','rgba(125, 42, 123, .2)']

Plot = go.Pie(labels=labels, values=values,marker=dict(colors=colors,line=dict(color='#fff', width= 3)))
layout = go.Layout(title='Value distribution', height=400)
fig = go.Figure(data=[Plot], layout=layout)
iplot(fig)


# In[ ]:


train_data.info()


# The memory usage of the data is approx 170MB and the datatypes for features are distributed as:
# - **float64** - 1845
# -   **int64** - 3147
# -  **object** - 1

# In[ ]:


test_data.info()


# The memory usage for test data is 1.8GB and the datatypes for features are:
# 
# - **float64** - 4991
# -  **object** - 1

# In[ ]:


train_data.describe()


# ## Missing Data

# In[ ]:


def missing_data(data): #calculates missing values in each column
    total = data.isnull().sum().reset_index()
    total.columns  = ['Feature_Name','Missing_value']
    total_val = total[total['Missing_value']>0]
    total_val = total.sort_values(by ='Missing_value')
    return total_val


# In[ ]:


missing_data(train_data).head()


# In[ ]:


missing_data(test_data).head()


# There are no missing values in the train and test dataset. This is reasonably good as it is nearly impossible to fill missing data with certain values. 

# ## Histogram view of the log transformed dependent quantitative variable

# In[ ]:


sns.distplot(np.log1p(train_data['target']))


# In[ ]:


X_train = train_data.drop(['ID','target'],axis=1)
y_train = np.log1p(train_data["target"])
X_test = test_data.drop('ID', axis = 1)


# # Outlier detection using Isolation Forest 

# Outlier detection is one of the most important aspects of regression analysis. If not removed it can hamper the performance of the model which we will fit to the data for continuous value prediction. So I have used a method which is highly suitable for high dimensional datasets i.e. Isolation forest algorithm, an ensemble method which returns anomaly scores of each sample in the dataset.
# 
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
# +1 indicates that the sample is an inlier whereas -1 indicates that the sample is an outlier.

# In[ ]:


#X = X_train.copy()
#Xa = X_test.copy()
#clf = IsolationForest(max_samples=100, random_state= 0)
#clf.fit(X)


# In[ ]:


#y_pred = clf.predict(X)
#y_pred_df = pd.DataFrame(data=y_pred,columns = ['Values'])
#y_pred_df['Values'].value_counts()


# In[ ]:


#anomaly_score = clf.decision_function(X_train)
#anomaly_score


# In[ ]:


#y_test_pred = clf.predict(Xa)
#y_test_pred_df = pd.DataFrame(data=y_test_pred,columns = ['Out_Values'])
#y_test_pred_df['Out_Values'].value_counts()


# In[ ]:


#anomaly_score = clf.decision_function(X_test)
#anomaly_score


# ## Feature selection using Mutual Information & SelectKBest method

# Feature Selection is an essential part of feature engineering. The method I have used here is a univariate feature selection method in which there is a scoring function and a    selection method.
# 
# - Mutual_info_regression is used as a scoring function which calculates the mutual information between each feature and the target by estimating the entropy from K-Nearest   Neighbors. Mutual Information is the measure of dependency between two random variables which is a non-negative value. It is equal to zero if two variables are independent and higher value means higher dependency.
# 
# - SelectKBest is the univariate selection method which takes scoring function as an input that returns univariate scores and p values. It removes all the features except the K-highest scoring features.

# In[ ]:


feat = SelectKBest(mutual_info_regression,k=200)


# In[ ]:


X_tr = feat.fit_transform(X_train,y_train)
X_te = feat.transform(X_test)


#  ## Modelling with Lasso

# In[ ]:


tr_data = scaler.fit_transform(X_tr)
te_data = scaler.fit_transform(X_te)
reg = Lasso(alpha=0.0000001, max_iter = 10000)


# In[ ]:


reg.fit(tr_data,y_train)


# In[ ]:


y_pred = reg.predict(te_data)
y_pred


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
#y_pred = np.clip(y_pred,y_train.min(),y_train.max())
sub["target"] = np.expm1(y_pred)
print(sub.head())
sub.to_csv('sub_las.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




