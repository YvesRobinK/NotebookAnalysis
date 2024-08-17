#!/usr/bin/env python
# coding: utf-8

# > **Table of Contents:**
# > * [Overview of the Competition](#1)
# > * [Importing Libraries](#2)
# > * [Loading Data](#3)
# > * [EDA - Exploratory Data Analysis](#4)
# > * [PreProcessing](#5)
# > * [Data Visualization](#6)
# > * [Data Modeling](#7)
# > * [Submission](#8)
# > ---

# <a id="1"></a> 
# # Overview of the Competition
# The competing Kaggle merchandise stores we saw in January's Tabular Playground are at it again. This time, they're selling books!
# 
# 
# The task for this month's competitions is a bit more complicated. Not only are there six countries and four books to forecast, but you're being asked to forecast sales during the tumultuous year 2021. Can you use your data science skills to predict book sales when conditions are far from the ordinary?
# 
# 
# In this notebook, we will be exploring the dataset and visualize the relationships of every column through graphs are representation.We will then preprocess the data and perform some feature engineering in order to be able to fit the train data into different machine learning models. After that, we will evaluate the models using different metrics to determine which model will perform best on the current dataset.

# Note:
# This notebook is inspired from this [notebook](http://www.kaggle.com/code/vencerlanz09/tps-eda-9-models-explanation#%F0%9F%93%8AAll-Model-Results)

# ---

# <a id="2"></a> 
# # 1. Importing Libraries 😀

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Machine Learning Estimators
import xgboost as xgb

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_style('darkgrid')

import warnings
warnings.filterwarnings("ignore")


# <a id="3"></a> 
# # 2. Loading the Data 📅

# In[2]:


df_train =pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv',index_col=0,parse_dates=['date']) # index col is set to zero as it is the index, so no need to have 2 indexes, just set it as index.
df_train.head()


# In[3]:


df_test =pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv',index_col=0,parse_dates=['date'])
df_test.head()


# <a id="4"></a> 
# # 3. Let's Explore 👓 - Exploratory Data Analysis

# In[4]:


df_train.info()


# In[5]:


df_train.shape, df_test.shape 


# ## 3.B. Checking for Null values 🤔

# In[6]:


#checking for null values in train data
df_train.isnull().sum()


# In[7]:


#checking for null values in test data
df_test.isnull().sum()


# ## 3.C. Checking for Duplicate values

# In[8]:


#Checking for duplicate rows
df_train.duplicated().sum()


# In[9]:


df_train.describe()


# In[10]:


display('Train Data',df_train.describe(include='object')),display('Test Data',df_test.describe(include='object'))


# - There are 6 unique countries.
# - There are 2 stores in this dataframe.
# - We are studying data for 4 products.
# - Test data is for an year.

# In[11]:


# checking for timeline of train and test data
display('Train_data:' ,'---------',df_train.date.min(),'---------',df_train.date.max())


# In[12]:


display('Test_data:','---------',df_test.date.min(),'---------',df_test.date.max())


# **Train Data is for year 2017,2018,2019 and 2020 while
# Test Data is for year 2021**

# In[13]:


df_train.nunique()


# In[14]:


countries = df_train['country'].unique().tolist()
countries


# In[15]:


products = df_train['product'].unique().tolist()
products


# In[16]:


stores =  df_train['store'].unique().tolist()
stores


# In[17]:


df_train.head()


# <a id="5"></a> 
# # 4. Preprocessing  🤠

# In[18]:


df_train["year"] = df_train["date"].dt.year
df_train["month"] = df_train["date"].dt.month
df_train["day_of_week"] = df_train["date"].dt.dayofweek
df_train["day_of_year"] = df_train["date"].dt.dayofyear


# In[19]:


df_test["year"] = df_test["date"].dt.year
df_test["month"] = df_test["date"].dt.month
df_test["day_of_week"] = df_test["date"].dt.dayofweek
df_test["day_of_year"] = df_test["date"].dt.dayofyear


# <a id="6"></a> 
# # 5. Data Visualization 🤩

# In[20]:


sns.histplot(df_train['num_sold']);


# In[21]:


product_df = df_train.groupby(["date","product"])["num_sold"].sum().reset_index()
product_df


# In[22]:


product_df.describe()


# In[23]:


plt.figure(figsize=(12,8))
sns.lineplot(data=product_df, x="date", y="num_sold", hue="product");


# In[24]:


sns.boxplot(x='store',y='num_sold',data=df_train);


# In[25]:


sns.boxplot(x='product',y='num_sold',data=df_train)
plt.xticks(rotation=90);


# In[26]:


sns.boxplot(x='country',y='num_sold',data=df_train);


# In[27]:


data=df_train.groupby('country').sum('num_sold')
sns.barplot(data=data,x=data.index,y='num_sold')
plt.title('Total Sales by Country');


# In[28]:


weekly_df = df_train.groupby(["country","store", "product", pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = df_train.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()


# In[29]:


def plot_all(df):
    f,axes = plt.subplots(2,2,figsize=(20,15), sharex = True, sharey=True)
    f.tight_layout()
    for n,prod in enumerate(df["product"].unique()):
        plot_df = df.loc[df["product"] == prod]
        sns.lineplot(data=plot_df, x="date", y="num_sold", hue="country", style="store",ax=axes[n//2,n%2])
        axes[n//2,n%2].set_title("Product: "+str(prod))


# In[30]:


plot_all(weekly_df)


# In[31]:


plot_all(monthly_df)


# ### The Data is constant for the year 2020 for both stores and products. This is very suspicious and makes the date column for prediction unuseful.

# In[32]:


df_train.info()


# In[33]:


#encoder = LabelEncoder()
#def encode_data(data, categories=['country', 'store', 'product']):
    #for cat in categories:
        #data[cat] = encoder.fit_transform(data[[cat]])
    #return data


# In[34]:


#train = encode_data(df_train)
#test = encode_data(df_test)


# In[35]:


train = pd.get_dummies(df_train, columns=['country', 'store', 'product'], drop_first=True)
test =  pd.get_dummies(df_test, columns=['country', 'store', 'product'], drop_first=True)


# In[36]:


# Splitting the data
X = train.drop(['num_sold', 'date'], axis=1)
y = train.num_sold

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)


# In[37]:


# Function to calculate metric results
def calculate_results(y_true, y_pred):
    # Calculate model Mean Absolute Error (MAE)
    model_mae = mean_absolute_error(y_val, y_pred)
    # Calculate model Mean Squrared Error (MSE)
    model_mse = mean_squared_error(y_val, y_pred)
    # Calculate model Root Mean Squared Error (RMSE)
    model_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # Calculate Adjusted R2_score
    model_r2 = r2_score(y_val, y_pred)
    # Calculate Root Mean Squared Log Error
    model_rmsle = np.log(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    
    model_results = {"Mean Absolute Error (MAE)": model_mae,
                     "Mean Squared Error (MSE)": model_mse,
                     "Root Mean Squared Error (RMSE)": model_rmse,
                     "Adjusted R^2 Score": model_r2,
                     "Root Mean Squared Log Error": model_rmsle}
    return model_results


# <a id="7"></a> 
# # 6. Data Modeling  🤖

# I used GridSearchCV to get the best parameters: {'regressor__n_estimators': 100,
#  'regressor__max_depth': 6,
#  'regressor__learning_rate': 0.1}

# In[38]:


# Predict using Ridge Regression
xgb_model = xgb.XGBRegressor(params={'regressor__n_estimators': 120, 'regressor__max_depth': 6, 'regressor__learning_rate': 0.1})
y_pred = xgb_model.fit(X_train, y_train).predict(X_val)
    
xgb_results = calculate_results(y_val, y_pred)
pd.DataFrame(xgb_results ,index=['values']).T


# In[39]:


predictions= xgb_model.fit(X,y).predict(test.drop('date',axis=1))


# In[40]:


# Feature importances
pd.DataFrame({'Feature': X.columns,'Importance': xgb_model.feature_importances_}).sort_values(by=['Importance'], ascending=False).reset_index(drop=True)


# <a id="8"></a> 
# # 7. Submission  -Finally 😎

# In[41]:


submissions=pd.read_csv('../input/tabular-playground-series-sep-2022/sample_submission.csv')
submissions['num_sold']=predictions//1


# In[42]:


submissions.head()


# In[43]:


submissions.to_csv('submission.csv', index = False)


# ### <center>Thanks for reading:)</center>
# ### <center>Upvote! and Leave some suggestions</center>

# In[ ]:




