#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius: 10px; border: #6B8E23 solid; padding: 15px; background-color: #F5F5DC; font-size: 100%; text-align: left">
# 
# <h3 align="left"><font color='#556B2F'>📜 Introduction : </font></h3>
#     
# **What is feature engineering?**
#     
# Feature engineering is the process of transforming and enriching data to improve the performance of machine learning algorithms used to train models using that data.
# 
# Feature engineering includes steps such as scaling or normalizing data, encoding non-numeric data (such as text or images), aggregating data by time or entity, joining data from different sources, or even transferring knowledge from other models. The goal of these transformations is to increase the ability of machine learning algorithms to learn from the data set and thus make more accurate predictions.
#     
# **Why is feature engineering important?**
#     
# Feature engineering is important for several reasons. Firstly, as mentioned earlier, machine learning models sometimes can't operate on raw data, and so the data must be transformed into a numeric form that the model can understand. This could involve converting text or image data into numeric form, or creating aggregate features such as average transaction values for a customer.
# 
# Sometimes relevant features for a machine learning problem may exist across multiple data sources, and so effective feature engineering involves joining these data sources together to create a single, usable data set. This allows you to use all of the available data to train your model, which can improve its accuracy and performance.
# 
# Another common scenario is that other models' output and learning can sometimes be reused in the form of features for a new problem, using a process known as transfer learning. This allows you to leverage the knowledge gained from previous models to improve the performance of a new model. Transfer learning can be particularly useful when dealing with large, complex data sets where it is impractical to train a model from scratch.
# 
# Effective feature engineering also enables reliable features at inference time, when the model is being used to make predictions on new data. This is important because the features used at inference time must be the same as the features used at training time, in order to avoid "online/offline skew," where the features used at the time of prediction are calculated differently than those used for training.
# 
# **How is feature engineering different from other data transformations?**
#     
# The goal of feature engineering is to create a data set that can be trained to build a machine learning model. Many of the tools and techniques used for data transformations are also used for feature engineering.
# 
# Because the emphasis of feature engineering is to develop a model, there are several requirements that are not present with all feature transformations. For example, you may want to reuse features across multiple models or across teams in your organization. This requires a robust method for discovering features.
# 
# Also, as soon as features are reused, you will need a way to track where and how features are computed. This is called feature lineage. Reproducible feature computations are of particular importance for machine learning, since the feature not only must be computed for training the model but also must be recomputed in exactly the same way when the model is used for inference.
# 
# **What are the benefits of effective feature engineering?**
#     
# Having an effective feature engineering pipeline means more robust modeling pipelines, and ultimately more reliable and performant models. Improving the features used both for training and inference can have an incredible impact on model quality, so better features means better models.
# 
# From a different perspective, effective feature engineering also encourages reuse, not only saving practitioners time but also improving the quality of their models. This feature reuse is important for two reasons: it saves time, and having robustly defined features helps prevent your models from using different feature data between training and inference, which typically leads to "online/offline" skew.

# <center><img src="https://i.imgur.com/UVB8v8B.png" width="800" height="800"></center>

# # Content
# 
# 1. [☠️Outliers☠️](#1)
#     * [Detecting Outliers](#2)
#     * [Accessing Outliers](#3)
#     * [Removing Outliers](#4)
#     * [Handling Outliers through Capping / Re-Assignment with Thresholds](#5)
#     * [Multivariate Outlier Analysis (Local Outlier Factor)](#6)
# 1. [🤔Missing Values🤔](#7)
#     * [Detecting Missing Values](#8)
#     * [Dealing with the Missing Values Problem](#9)
#     * [Predicting Missing Values with Machine Learning](#10)
#     * [Exploring the Relationship between Missing Values and Dependent Variables](#11)
# 1. [0️⃣Encoding1️⃣](#12)
#     * [Label Encoding](#13)
#     * [One Hot Encoding](#14)
#     * [Rare Encoding](#15)
#     * [Feature Scaling](#16)
# 1. [🔍Feature Extraction🔎](#17)
#     * [Binary Features: Flag, Bool, True-False](#18)
#     * [Text Features](#19)
#     * [Regex Features](#20)
#     * [Date Features](#21)
#     * [Feature Interaction](#22)

# <a id="1"></a>
# # <p style="border-radius: 10px; border: 4px solid #2E1A47; background-color: #F5F5DC; font-family: 'Rockwell', cursive; font-weight: bold; font-size: 150%; text-align: center; border-radius: 15px 50px; padding: 5px; box-shadow: 4px 4px 4px #556B2F; color: #556B2F;">☠️ Outliers ☠️</p>

# **What is an outlier?**
# 
# In data analytics, outliers are values within a dataset that vary greatly from the others—they’re either much larger, or significantly smaller. Outliers may indicate variabilities in a measurement, experimental errors, or a novelty.
# 
# In a real-world example, the average height of a giraffe is about 16 feet tall. However, there have been recent discoveries of two giraffes that stand at 9 feet and 8.5 feet, respectively. These two giraffes would be considered outliers in comparison to the general giraffe population. 
# 
# When going through the process of data analysis, outliers can cause anomalies in the results obtained. This means that they require some special attention and, in some cases, will need to be removed in order to analyze data effectively.
# 
# * There are two main reasons why giving outliers special attention is a necessary aspect of the data analytics process:
# 
#     * Outliers may have a negative effect on the result of an analysis.
#     * Outliers—or their behavior—may be the information that a data analyst requires from the analysis.
#     
# **Types of Outliers**
# 
# * There are two kinds of outliers:
# 
#     * **A univariate outlier** is an extreme value that relates to just one variable. For example, Sultan Kösen is currently the tallest man alive, with a height of 8ft, 2.8 inches (251cm). This case would be considered a univariate outlier as it’s an extreme case of just one factor: height. 
#     * **A multivariate outlier** is a combination of unusual or extreme values for at least two variables. For example, if you’re looking at both the height and weight of a group of adults, you might observe that one person in your dataset is 5ft 9 inches tall—a measurement that would fall within the normal range for this particular variable. You may also observe that this person weighs 110lbs. Again, this observation alone falls within the normal range for the variable of interest: weight. However, when you consider these two observations in conjunction, you have an adult who is 5ft 9 inches and weighs 110lbs—a surprising combination. That’s a multivariate outlier.
# 
# * Besides the distinction between univariate and multivariate outliers, you’ll  see outliers categorized as any of the following:
# 
#     * Global outliers (otherwise known as point outliers) are single data points that lay far from the rest of the data distribution. 
#     * Contextual outliers (otherwise known as conditional outliers) are values that significantly deviate from the rest of the data points in the same context, meaning that the same value may not be considered an outlier if it occurred in a different context. Outliers in this category are commonly found in time series data. 
#     * Collective outliers are seen as a subset of data points that are completely different with respect to the entire dataset.

# <a id = "2"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Detecting Outliers✨</p>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# !pip install missingno
import missingno as msno

from datetime import date

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

import warnings
warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.set_option('display.width', 500)

get_ipython().system('wget https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/main/themes/rose-pine-dawn.mplstyle -P /tmp')
plt.style.use("/tmp/rose-pine-dawn.mplstyle")


# In[2]:


home_credit = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
titanic_ = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic = titanic_.copy()


# In[3]:


titanic.describe([0.01,0.99]).T


# In[4]:


# A boxplot provides distribution information for a numerical variable. Another method could be a Histogram plot.

sns.boxplot(x = titanic["Age"])
plt.show()


# In[5]:


q1 = titanic["Age"].quantile(0.25)
q3 = titanic["Age"].quantile(0.75)
IQR = q3 - q1
up = q3 + 1.5 * IQR
low = q1 - 1.5 * IQR


# In[6]:


# If we want to obtain outliers, what should we do?;

titanic[(titanic["Age"] < low) | (titanic["Age"] > up)]


# In[7]:


# We obtained the index information of rows containing outliers;

titanic[(titanic["Age"] < low) | (titanic["Age"] > up)].index


# In[8]:


titanic[(titanic["Age"] < low) | (titanic["Age"] > up)].any(axis=None)


# In[9]:


titanic[(titanic["Age"] < low)].any(axis=None)


# ----

# In[10]:


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    IQR = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * IQR
    low_limit = quartile1 - 1.5 * IQR
    
    return low_limit, up_limit


# ---

# In[11]:


def check_outlier(dataframe, col_name):
    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        
        return True
    
    else:
        
        return False


# In[12]:


check_outlier(titanic, "Age")


# -----

# In[13]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numerical-looking categorical variables are also included in the categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe for which variable names are to be obtained.
        cat_th: int, optional
                Class threshold value for numerical but categorical variables.
        car_th: int, optional
                Class threshold value for categorical but cardinal variables.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numerical variables.
        cat_but_car: list
                List of categorical but cardinal variables.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is within cat_cols.
        The sum of the 3 lists returned is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    return cat_cols, num_cols, cat_but_car


# In[14]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic)


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * The number of num_but_cat is already included within the cat_cols, representing only the quantity identified.

# In[15]:


num_cols = [col for col in num_cols if col not in "PassengerId"]
num_cols


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * We need to distinguish ID or date variables since they can also be considered as numerical.

# In[16]:


for col in num_cols:
    
    print(col, check_outlier(titanic, col))


# ----

# In[17]:


def outliner_detecter(df,cols):
    
    temp = pd.DataFrame()
    
    for col in cols:
        
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        IQR = q3 - q1
        up = q3 + 1.5 * IQR
        low = q1 - 1.5 * IQR
        temp.loc[col, "Min"] = df[col].min()
        temp.loc[col, "Low_Limit"] = low
        temp.loc[col,"Up_Limit"] = up
        temp.loc[col, "Max"] = df[col].max()
        
    return print(temp.astype(int).to_markdown())


# In[18]:


outliner_detecter(titanic,num_cols)


# ------

# <div style="border-radius:10px; border:#867070 solid; padding: 15px; background-color: #F5EBEB; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>📄 Notes: </font></h3>
#     
# * **outlier_thresholds**: With this function, we can display the lower and upper limits.
# * **check_outlier**: This function can be used to check whether a variable contains outliers.
# * **grab_col_names**: With this function, we can identify numerical, categorical, numerical-looking categorical, or non-informative categorical variables. The parameters may vary depending on the dataset.
# * **outlier_detector**: In this function, we can view not only the lower and upper limits but also the minimum and maximum values. Additional statistics like median can also be included if desired.

# <a id = "3"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Accessing Outliers✨</p>

# In[19]:


def grab_outliers(dataframe, col_name, index=False):
    
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]

    if len(outliers) > 10:
        print(outliers.head())
        
    else:
        print(outliers)

    if index:
        outlier_index = outliers.index
        
        return outlier_index


# In[20]:


grab_outliers(titanic, "Fare")


# In[21]:


age_indexes = grab_outliers(titanic, "Fare", True)


# In[22]:


age_indexes


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * If we detect more than 10 outliers, we will print the first 5 rows. If we want to access index information, we can set the 'index=False' parameter to True.

# <a id = "4"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Removing Outliers✨</p>

# In[23]:


titanic.shape


# -----

# In[24]:


def remove_outlier(dataframe, col_name):
    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    
    return df_without_outliers


# In[25]:


for col in num_cols:
    
    new_df = remove_outlier(titanic, col)


# ---

# In[26]:


titanic.shape[0] - new_df.shape[0]


# In[27]:


titanic.describe([0.01,0.99]).T


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * When we remove the outliers in one cell, the data in other variables is also affected. Therefore, sometimes it may be more appropriate to apply a capping process instead of deletion.

# <a id = "5"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Re-Assignment with Thresholds✨</p>

# In[28]:


titanic = titanic_.copy()


# In[29]:


titanic.shape


# In[30]:


for col in num_cols:
    
    print(col, check_outlier(titanic, col))


# ---

# In[31]:


def corr_skew_outliner(df, cols):

    for col in cols:
        
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        df.loc[df[col] < Q1, col] = Q1
        df.loc[df[col] > Q3, col] = Q3
        #df[col] = np.sqrt(df[col])
        
    return df


# In[32]:


corr_skew_outliner(titanic, num_cols).head()


# -----

# In[33]:


titanic.describe([0.01,0.99]).T


# In[34]:


titanic.shape


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * As seen, this time we preserved all the data using the capping method instead of losing data. However, the choice between deletion or capping methods can vary depending on the dataset. It is a preference process based entirely on interpretation.

# <a id = "6"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Local Outlier Factor✨</p>

# Local Outlier Factor detects the outliers or deviation of data points in a distribution with respect to the density of its neighbors. It identifies local outliers in a dataset that are not outliers in another region of the dataset.
# 
# For example, consider a very dense cluster of data points in a dataset. One of the data points is at a small distance from the dense cluster. This data point is considered an outlier. In the same dataset, a data point in a sparse cluster might appear to be an outlier but is detected to be at a similar distance from each of its neighbors.
# 
# A normal data point has a LOF between 1 and 1.5, while an outlier has a much higher LOF. If the LOF of a data point is 10, it means that the average density of its neighbors is ten times higher than the local density of the data point.
# 
# The Local Outlier Factor method is used in detecting outliers in geographic data, video streams, or network intrusion detection.
# 
# * The LOF score of a data point is determined using the following:
# 
#     * Number of neighbors
#     * A tree algorithm used for structuring the data
#     * Leaf size to define the depth of the tree algorithm
#     * A metric function to define the distance between two points
#     * Hyperparameter tuning
#     * Dimensionality reduction and variance
#     
# An example: It allows us to define outliers by scoring observations based on their density at their respective locations. Density is the semantic similarity of that observation unit with its neighbors. For example, being 17 years old on its own may not appear as an outlier. However, being 17 years old and having been married 3 times may not appear quite normal. We can consider it as an outlier.

# <center><img src="https://i.imgur.com/5GNOE8s.png" width="800" height="800"></center>

# In[35]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])


# In[36]:


df.isnull().sum() # seems there is not any NaN value.


# In[37]:


df.head()


# In[38]:


df.shape


# In[39]:


for col in df.columns:
    
    print(col, check_outlier(df, col))


# In[40]:


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)


# In[41]:


# Local Outlier Factor Scores

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores (If we want to keep positive values)


# In[42]:


scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * The closer the LOF score is to -10, the worse it is. However, we need to set a threshold.
# * As we can see, there is a significant change in slope after the third point.

# In[43]:


th = np.sort(df_scores)[3]
th


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * We have set this value as the threshold. We will consider those that are smaller than this value as outliers.

# In[44]:


df[df_scores < th]


# In[45]:


df[df_scores < th].shape


# In[46]:


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# Why did it consider these 3 observations as outliers? Let's examine.
# 
# * The first noticeable detail is that observation number "41918" has a "depth" value of "78.200." In our main dataset, the maximum value of the "depth" variable is displayed as "79.000." But why did it not take the highest value and instead made a choice lower than the maximum value? It's just an inference. Because there is a multivariate effect here, there may have been an inconsistency in the interaction with other variables when the depth value was as it is. Let's continue the examination.
# * Observation number "48410" has a "z" value of "31.800." At the same time, this z value corresponds to the maximum value. Of course, this situation also falls within an outlier with a connection to other variables, just like the previous one. It can be inferred that a variable becomes inconsistent with another variable's value when a variable takes a value.

# In[47]:


df[df_scores < th].index


# In[48]:


df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * If we are interested in tree methods, at this point, we should prefer not to touch outlier removal and capping methods at all, at the most basic level. However, if there is a need for some adjustment, very small adjustments (0.01 - 0.99) can be made.
# * If linear methods are preferred; instead of filling, it may be preferred to delete if the number of outliers is low or capping can be done as a univariate approach.

# In[49]:


def take_a_look(df):
    
    print(pd.DataFrame({'Rows': [df.shape[0]], 'Columns': [df.shape[1]]}, index=["Shape"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.dtypes, columns=["Type"]).to_markdown())
    print("\n")
    print(df.head().to_markdown(index=False))
    print("\n")
    print(df.isnull().sum().to_markdown())
    print("\n")
    print(df.describe([0.01, 0.25, 0.75, 0.99]).astype(int).T.to_markdown())


# In[50]:


take_a_look(df)


# <a id="7"></a>
# # <p style="border-radius: 10px; border: 4px solid #2E1A47; background-color: #F5F5DC; font-family: 'Rockwell', cursive; font-weight: bold; font-size: 150%; text-align: center; border-radius: 15px 50px; padding: 5px; box-shadow: 4px 4px 4px #556B2F; color: #556B2F;">🤔 Missing Values 🤔</p>

# **What Are Missing Values?**
# 
# Missing values refer to the situation in a dataset where one or more variables have missing or undefined (NULL) values. These missing values can result from errors in the data collection or data entry process. Missing values can pose a significant problem for data analysis and machine learning models because they can lead to incorrect results or misleading analyses. Dealing with missing values is a crucial aspect of data science and statistics.
# 
# **Types of Missing Data:**
# 
# 1. **Missing Completely At Random (MCAR):** In this type of missingness, missing values are random and unrelated to other variables. For example, forgetting to fill out a section of a survey is an example of this type.
# 
# 2. **Missing At Random (MAR):** This type of missingness means that missing values are related to other observed variables, but there are other variables available that can be used to predict the level of missing values. For example, if age data is missing, but gender data is available, it might be possible to predict age using the relationship between age and gender.
# 
# 3. **Missing Not At Random (MNAR):** This type of missingness indicates that missing values are related to other variables, and there are no other variables available to predict the level of missing values. MNAR is the most challenging type of missing data and often requires a deeper understanding of the reasons for missing data.
# 
# **Handling Missing Data:**
# 
# Missing data can be handled using several different methods:
# 
# 1. **Deleting Missing Values:** This involves removing observations with missing data from the dataset. However, this method can lead to the loss of important information and should not be used if the dataset is small.
# 
# 2. **Imputing Missing Values:** Missing values can be imputed or filled using various methods. Imputation can be done using statistical values such as median, mean, mode, or by using machine learning models to predict missing values.
# 
# 3. **Categorizing Missing Values:** Missing data can be treated as a separate category based on the missingness. This can be a useful approach to prevent data loss.
# 
# **Impact of Missing Data on Modeling:**
# 
# Missing data can impact model results and outcomes. Observations with missing data can lead to misleading results. Additionally, the management of missing data can also affect model performance. Therefore, the handling of missing data can have a significant impact on model accuracy.

# <a id = "8"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Detecting Missing Values✨</p>

# In[51]:


# It checks whether there are missing observations;

titanic.isnull().values.any()


# In[52]:


# The number of missing values in variables;

titanic.isnull().sum()


# In[53]:


# The number of complete values in variables;

titanic.notnull().sum()


# In[54]:


# The total number of missing values in the dataset;

titanic.isnull().sum().sum()


# In[55]:


# Observations with at least one missing value;

titanic[titanic.isnull().any(axis=1)].head()


# In[56]:


# Observations with complete values;

titanic[titanic.notnull().all(axis=1)].head()


# ----

# In[57]:


def missing_values_table(dataframe, na_name=False):
    
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']).to_markdown()
    
    print(missing_df, end="\n")
    
    if na_name:
        
        return na_columns


# In[58]:


missing_values_table(titanic)


# <a id = "9"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Dealing with the Missing Values Problem✨</p>

# In[59]:


titanic.shape


# In[60]:


titanic.isnull().sum().sum()


# In[61]:


titanic.dropna().shape


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * We can use the `dropna` method to delete rows with missing values. However, as seen, it can lead to a significant loss of data.

# ----

# In[62]:


titanic = titanic_.copy()


# In[63]:


titanic["Age"].fillna(titanic["Age"].mean()).isnull().sum()


# In[64]:


titanic["Age"].fillna(titanic["Age"].median()).isnull().sum()


# In[65]:


titanic["Age"].fillna(0).isnull().sum()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * It can also be filled with the mean, median, or a desired fixed value.

# ----

# In[66]:


titanic = titanic_.copy()


# In[67]:


# For numerical features;

titanic.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()


# In[68]:


# For handling missing values in categorical variables;
# titanic["Embarked"].fillna(titanic["Embarked"].mode()[0]).isnull().sum()
# titanic["Embarked"].fillna("missing")
# You can fill the missing values with the mode of the "Embarked" variable or with the string "missing."


# In[69]:


titanic.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


# In[70]:


for col in cat_cols:
    if titanic[col].isnull().sum() > 0:
        mode_value = titanic[col].mode()[0]
        titanic[col].fillna(mode_value, inplace=True)


# In[71]:


titanic.isnull().sum()


# ----

# In[72]:


titanic.loc[(titanic["Age"].isnull()) & (titanic["Sex"]=="female"), "Age"] = titanic.groupby("Sex")["Age"].mean()["female"]

titanic.loc[(titanic["Age"].isnull()) & (titanic["Sex"]=="male"), "Age"] = titanic.groupby("Sex")["Age"].mean()["male"]


# In[73]:


titanic.isnull().sum()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * Missing age values can be replaced with median values by gender.

# <a id = "10"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Predicting Missing Values with Machine Learning✨</p>

# In[74]:


titanic = titanic_.copy()


# In[75]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic)


# In[76]:


print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)
print("Cardinals (Categorical but not informative):", cat_but_car)


# In[77]:


num_cols = [col for col in num_cols if col not in "PassengerId"]


# In[78]:


for_predict = pd.get_dummies(titanic[cat_cols + num_cols], drop_first=True)
for_predict.head()


# In[79]:


scaler = MinMaxScaler()
for_predict = pd.DataFrame(scaler.fit_transform(for_predict), columns=for_predict.columns)
for_predict.head()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * We applied standardization to the values in the data, scaling them between 0 and 1. We needed to bring the data into this form before using a machine learning model.

# In[80]:


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

for_predict = pd.DataFrame(imputer.fit_transform(for_predict), columns=for_predict.columns)
for_predict.head()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * We can use a machine learning model to impute missing values.
# * KNN (K-Nearest Neighbors) is a distance-based method that looks at the neighbors of a point (value) and predicts the most suitable value for it.
# * By taking the average of the five closest neighbors, a new value is added.
# * Up to this point, filling in the missing values is complete. However, if you look at the appearance, it still remains in standardized form. Therefore, we need to reverse the process and transform it back to its normal form.

# In[81]:


for_predict = pd.DataFrame(scaler.inverse_transform(for_predict), columns=for_predict.columns)
for_predict.head()


# <div style="border-radius:10px; border:#435560 solid; padding: 15px; background-color: #C8C6A7; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>📄 Notes: </font></h3>
#     
# * So, if you want to see what values have been assigned or observe the changes made, what can you do?

# In[82]:


titanic["Age_Imputed_KNN"] = for_predict[["Age"]]


# In[83]:


titanic.loc[titanic["Age"].isnull(), ["Age", "Age_Imputed_KNN"]].head()


# In[84]:


titanic.loc[titanic["Age"].isnull()].head()


# <a id = "11"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Exploring the Relationship between Missing Values and Dependent Variables✨</p>

# In[85]:


titanic = titanic_.copy()


# In[86]:


msno.bar(titanic)
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()


# In[87]:


msno.heatmap(titanic)
fig = plt.gcf()
fig.set_size_inches(9, 4)
plt.show()


# In[88]:


plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.boxplot(x="Pclass", y=column, data=titanic,palette="pastel")
    plt.title(f"{column} Distribution")
    plt.tight_layout()


# In[89]:


missing_cols = missing_values_table(titanic, True)


# In[90]:


missing_cols


# ----

# In[91]:


def missing_vs_target(dataframe, target, na_columns):
    
    temp_df = dataframe.copy()

    for col in na_columns:
        
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        
        result_df = pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                                  "Count": temp_df.groupby(col)[target].count()})
        print(result_df.to_markdown(), end="\n\n\n")


# In[92]:


missing_vs_target(titanic, "Survived", missing_cols)


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * Here, "1" represents those with missing values, and "0" represents those without missing values.
# * If we look at the Age_NA_FLAG values, those with missing values have an average survival rate of "0.293785," while those with complete age values have a survival rate of "0.406162."
# * Let's check the Cabin_NA_FLAG variable; this variable had a 77.1% missing value rate. The survival rate of observations with missing values is "0.299854," while the survival rate of observations without missing values is "0.666667." There is a significant difference.
# * In fact, there is a general story behind this; one reason for such a high level of missing values in the "Cabin" variable could be that many of the ship's crew do not have cabin numbers. They are not assigned such a value because they do not stay in the cabins. The majority of those staying in the cabins are passengers. It can be inferred that some of these missing values belong to ship employees.
# * All of these comments are an attempt to perform analysis and interpretation.

# <a id="12"></a>
# # <p style="border-radius: 10px; border: 4px solid #2E1A47; background-color: #F5F5DC; font-family: 'Rockwell', cursive; font-weight: bold; font-size: 150%; text-align: center; border-radius: 15px 50px; padding: 5px; box-shadow: 4px 4px 4px #556B2F; color: #556B2F;">0️⃣ Encoding 1️⃣</p>

# **Encoding** is a data preprocessing technique used to convert categorical data into numerical format. Categorical data are data expressed in text-based or symbol-based values, and many machine learning algorithms cannot handle such data. Here are commonly used encoding methods:
# 
# 1. **Label Encoding:** This method assigns a unique number to each different category. For example, categories like "Red," "Blue," and "Green" are encoded as 0, 1, and 2. Label Encoding is suitable for ordinal categorical data, where the order of data matters.
# 
# 2. **One Hot Encoding:** This method creates a column for each category and assigns a value of 1 or 0 within each column for each observation. For example, for categories "Red," "Blue," and "Green," three separate columns are created. An observation that is "Red" has a value of 1 in the "Red" column and 0 in other columns. One Hot Encoding is suitable for nominal categorical data where there is no inherent order.
# 
# 3. **Rare Encoding:** This method combines infrequent or low-frequency categorical values. It is especially useful for variables with a large number of different categories. Rare Encoding groups rare categories under a single "Rare" category while preserving other categories. This can help the model better learn rare classes.
# 
# The choice of encoding method depends on the characteristics of the data and the specific preprocessing requirements. Selecting the right encoding method and parameters is a crucial part of data preprocessing, as it can significantly impact the model's performance.
# 
# When using encoding methods, there are several important parameters to consider:
# 
# 1. **Number of Categories:** It's essential to understand how many different categories exist in categorical variables before encoding. A large number of categories in One Hot Encoding can lead to many columns and increased model complexity. Rare Encoding requires information about the frequency of rare categories.
# 
# 2. **Ordinality:** If the data is ordinal (e.g., "Low," "Medium," "High"), Label Encoding can be suitable. For unordered data (e.g., "Red," "Blue," "Green"), One Hot Encoding might be a better choice.
# 
# 3. **Discriminatory Power:** The discriminatory power of categorical variables determines how much influence a category has on the target variable. This is particularly important when using Rare Encoding, as rare categories may have a strong impact on the target.
# 
# 4. **Rare Encoding Threshold:** When using Rare Encoding, you need to choose a threshold for what is considered "rare" and should be combined under a "Rare" category. This threshold can vary depending on the data's structure and problem domain.
# 
# 5. **One Hot Encoding Dummy Variable Trap:** When using One Hot Encoding, it's essential to use the "drop_first" parameter to combine columns representing the same category and avoid the dummy variable trap.
# 
# 6. **Label Encoding and Relationship with the Target Variable:** When using Label Encoding, consider the relationship between the categorical variable and the target variable. If there is no or weak relationship, the use of Label Encoding may be limited.

# <a id = "13"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Label Encoding✨</p>

# In[93]:


titanic = titanic_.copy()


# In[94]:


titanic["Sex"].head()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * For machine learning, we need to convert "male" and "female" observations into mathematical expressions (binary).

# In[95]:


le = LabelEncoder()
le.fit_transform(titanic["Sex"])[0:5]


# In[96]:


# If we wonder about the equivalents of 0 and 1;

le.inverse_transform([0, 1])


# ----

# In[97]:


def label_encoder(dataframe, binary_col):
    
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    
    return dataframe


# ----

# In[98]:


titanic = titanic_.copy()


# In[99]:


binary_cols = [col for col in titanic.columns if titanic[col].dtype not in [int, float]
               and titanic[col].nunique() == 2]

binary_cols


# In[100]:


titanic[binary_cols].head()


# In[101]:


for col in binary_cols:
    
    label_encoder(titanic, col)
    
titanic. head()


# <a id = "14"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨One Hot Encoding✨</p>

# In[102]:


titanic = titanic_.copy()


# In[103]:


titanic["Embarked"].value_counts()


# In[104]:


pd.get_dummies(titanic, columns=["Embarked"], drop_first=True).head()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * To avoid the dummy variable trap, as variables should not be generatable from each other, we remove the first class with "drop_first = True."

# In[105]:


pd.get_dummies(titanic, columns=["Embarked"], dummy_na=True).head()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * If we want missing values in the variable to be treated as a class, we can use "dummy_na = True."

# In[106]:


pd.get_dummies(titanic, columns=["Sex", "Embarked"], drop_first=True).head()


# ----

# In[107]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    
    return dataframe


# ----

# In[108]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic)


# In[109]:


cat_cols


# In[110]:


ohe_cols = [col for col in titanic.columns if 10 >= titanic[col].nunique() > 2] # one-hot-encoding cols
ohe_cols


# In[111]:


one_hot_encoder(titanic, ohe_cols).head() # to make a permanent change, it should be assigned to "titanic."


# In[112]:


titanic.head()


# <a id = "15"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Rare Encoding✨</p>

# > **Analyzing categorical variables based on their frequency of occurrence;**

# In[113]:


home_credit["NAME_EDUCATION_TYPE"].value_counts()


# In[114]:


cat_cols, num_cols, cat_but_car = grab_col_names(home_credit)


# -----

# In[115]:


def cat_summary(dataframe, col_name, plot=False):
    
    summary_df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                               "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    
    print(summary_df.to_markdown(), end="\n\n")
    
    if plot:
        
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# In[116]:


for col in cat_cols:
    
    cat_summary(home_credit, col)


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * As seen, there are observations with a very low probability of appearing within the variables. Instead of applying one-hot encoding to all of these values, we can group the observations with low frequencies into one group and then apply encoding.

# ----

# > **Analyzing the relationship between rare categories and the dependent variable;**

# In[117]:


home_credit["NAME_INCOME_TYPE"].value_counts()


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# We can consider some comments:
# 
# * The classes "Unemployed, Student, Businessman, and Maternity leave" can be grouped into a single category.
# * Of course, at this point, personal opinions come into play. This merging done before machine learning can be perceived as noise by the model.
# * Conversely, it can also lead to successful modeling results.
# * During the rare encoding process, it is important to proceed with caution and carefully monitor personal decisions.

# In[118]:


home_credit.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


# -----

# In[119]:


def rare_analyser(dataframe, target, cat_cols):
    
    for col in cat_cols:
        
        print(f"{col} : {len(dataframe[col].value_counts())}\n")
        
        summary_df = pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        })
        
        print(summary_df.to_markdown(), end="\n\n\n")


# In[120]:


# rare_analyser(home_credit, "TARGET", cat_cols)


# ----

# > **Rare Encoding Function;**

# In[121]:


def rare_encoder(dataframe, rare_perc):
    
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


# In[122]:


new_df = rare_encoder(home_credit, 0.01)


# In[123]:


rare_analyser(new_df, "TARGET", cat_cols)


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * As observed, the Rare class is a consolidated version of classes with an average of less than 1%.

# <a id = "16"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Feature Scaling✨</p>

# Feature scaling is a term used in data mining and machine learning to bring numerical variables with different ranges of values to the same range or scale. It aids in data processing and modeling processes, improving the performance of many algorithms. Here are some commonly used feature scaling methods:
# 
# 1. **StandardScaler (Z-Score Scaling):** This method transforms the data for each feature to have a mean of 0 and a standard deviation of 1. It assumes that the data is normally distributed and is sensitive to outliers. StandardScaler is the default scaling method used in most cases.
# 
# 2. **RobustScaler:** This method scales features using medians and percentiles. It is more robust against outliers because medians and percentiles are robust statistics, limiting the impact of outliers.
# 
# 3. **MinMaxScaler:** This method scales data to a specific range (usually between 0 and 1). The data is transformed to minimum and maximum values. MinMaxScaler preserves the original range of features.
# 
# 4. **MaxAbsScaler:** This method scales each feature to have a maximum absolute value of 1. It preserves the original sign (positive or negative) of features, so it has no effect on the mean and variance of the data.
# 
# These methods can yield different results depending on the data and the type of algorithm used. The choice of which method to use depends on the characteristics of the data and the requirements of the model. Scaling is an important part of the data preprocessing stage and can significantly impact the success rate of a model.

# <div style="border-radius:10px; border:#867070 solid; padding: 15px; background-color: #F5EBEB; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>📄 Notes: </font></h3>
#     
# * When using feature scaling methods, there are some important parameters and considerations to be aware of:
# 
#    1. **Robustness:** Pay attention to how resistant the method is to outliers. Particularly, RobustScaler is more resistant to outliers and may be more suitable for data containing outliers. 
#    2. **Data Distribution:** Examine the distribution of the data. StandardScaler works under the assumption that the data is normally distributed. If the data deviates from normality or has a more complex distribution, it may be better to choose a different scaling method. 
#    3. **Preservation of Original Data:** MinMaxScaler and MaxAbsScaler preserve the original values of the data. Other methods may alter the data. Therefore, if it is essential to retain the original data values, these scaling methods may be preferred. 
#    4. **Centering Around Zero:** StandardScaler centers the data around zero. This is necessary for some models, but it may be unnecessary in other cases. Consider how centering the data around zero affects the model's performance. 
#    5. **Normalization:** Some methods are used to compress features into a specific range, while others are used to normalize the data. Determine which type of scaling is better for your data.
#    6. **Number of Outliers:** Consider the quantity and impact of outliers in your data. The presence and quantity of outliers can influence the choice of a scaling method.

# In[124]:


standard = StandardScaler()
titanic["Age_Standard_Scaler"] = standard.fit_transform(titanic[["Age"]])
titanic.head()


# In[125]:


rs = RobustScaler()
titanic["Age_Robuts_Scaler"] = rs.fit_transform(titanic[["Age"]])
titanic.head()


# In[126]:


titanic.head()


# In[127]:


mms = MinMaxScaler()
titanic["Age_Min_Max_Scaler"] = mms.fit_transform(titanic[["Age"]])
titanic.describe().T


# In[128]:


titanic.head()


# -----

# In[129]:


def summarize_numeric_data(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary = dataframe[numerical_col].describe(quantiles)
    
    if plot:
        plt.figure(figsize=(8, 6))
        dataframe[numerical_col].hist(bins=20, color='lightseagreen', edgecolor='black')
        plt.xlabel(numerical_col)
        plt.title(f'{numerical_col} Distribution')
        plt.show()

    return summary

age_cols = [col for col in titanic.columns if "Age" in col]

for col in age_cols:
    print(f"Summary for {col}:")
    summary = summarize_numeric_data(titanic, col, plot=True)
    print(summary)
    print('\n' + '-'*50 + '\n')


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * Numeric to Categorical (Binning) transformation is used to convert numerical variables into categorical ones. For example, the age variable is divided into quartiles and transformed into a categorical variable. This can be useful for grouping or modeling the data in a more meaningful way.

# ----

# In[130]:


titanic["Age_qcut"] = pd.qcut(titanic['Age'], 5)
titanic.head()


# <a id="17"></a>
# # <p style="border-radius: 10px; border: 4px solid #2E1A47; background-color: #F5F5DC; font-family: 'Rockwell', cursive; font-weight: bold; font-size: 150%; text-align: center; border-radius: 15px 50px; padding: 5px; box-shadow: 4px 4px 4px #556B2F; color: #556B2F;">🔍 Feature Extraction 🔎</p>

# <a id = "18"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Binary Features: Flag, Bool, True-False✨</p>

# In[131]:


titanic = titanic_.copy()


# In[132]:


titanic["Cabin_Bool"] = titanic["Cabin"].notnull().astype('int')
titanic.head()

# NaN values were converted to 0, and the rest were set to 1.


# In[133]:


titanic.groupby("Cabin_Bool").agg({"Survived": "mean"})


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * We need to statistically examine the relationship between this newly created variable and the dependent variable. It might be a randomly occurring situation, so we need to prove it.

# ---

# In[134]:


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[titanic.loc[titanic["Cabin_Bool"] == 1, "Survived"].sum(),
                                             titanic.loc[titanic["Cabin_Bool"] == 0, "Survived"].sum()],

                                      nobs=[titanic.loc[titanic["Cabin_Bool"] == 1, "Survived"].shape[0],
                                            titanic.loc[titanic["Cabin_Bool"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * Our test is whether there is a difference between the proportions of p1 and p2.
# * p1 and p2 represent the survival rates of those with and without cabin numbers.
# * Since the p-value is less than 0.05, H0 has been rejected.
# * There is a statistically significant difference between p1 and p2.

# ----

# In[135]:


titanic.loc[((titanic['SibSp'] + titanic['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
titanic.loc[((titanic['SibSp'] + titanic['Parch']) == 0), "NEW_IS_ALONE"] = "YES"


# In[136]:


titanic.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
#     
# * NO: Those with family, YES: Those alone
# * It appears that the survival rate of those with family is higher. However, we need to statistically prove this, as it may have occurred by chance or due to multivariate effects.

# In[137]:


test_stat, pvalue = proportions_ztest(count=[titanic.loc[titanic["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             titanic.loc[titanic["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[titanic.loc[titanic["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            titanic.loc[titanic["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * Since p-value is less than 0.05, H0 is rejected.
# * This variable may have a meaningful impact.

# <a id = "19"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Text Features✨</p>

# In[138]:


titanic = titanic_.copy()


# In[139]:


# Letter count;

titanic["New_Name_Count"] = titanic["Name"].str.len()


# In[140]:


# Word count;

titanic["New_Name_Word_Count"] = titanic["Name"].apply(lambda x: len(str(x).split(" ")))


# In[141]:


# Catching Special Words;

titanic["New_Special"] = titanic["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

titanic.groupby("New_Special").agg({"Survived": ["mean","count"]})


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * In this section, we experimented with deriving new variables from an existing variable.
# * From a variable that contains names, we can determine the length of names, the number of words, or the count of individuals with special titles in their names and assign these values to new variables.
# 
# For example, there are 10 people with "Dr" in their names.

# <a id = "20"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Regex Features✨</p>

# In[142]:


titanic.head()


# In[143]:


titanic['New_Title'] = titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[144]:


titanic[["New_Title", "Survived", "Age"]].groupby(["New_Title"]).agg({"Survived": "mean", "Age": ["count", "mean"]})


# <div style="border-radius:10px; border:#65647C solid; padding: 15px; background-color: #F8EDE3; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#7D6E83'><b>🗨️ Comment: </b></font></h3>
# 
# * Here, you can observe the frequencies and averages of the newly created classes and make inferences.

# <a id = "21"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Date Features✨</p>

# In[145]:


course = pd.read_csv("/kaggle/input/course-reviewsdataset/course_reviews.csv")


# In[146]:


course.head()


# In[147]:


course.info()


# In[148]:


course['Timestamp'] = pd.to_datetime(course["Timestamp"], format="%Y-%m-%d %H:%M:%S")


# In[149]:


course['Year'] = course['Timestamp'].dt.year


# In[150]:


course['Month'] = course['Timestamp'].dt.month


# In[151]:


# The year difference between two dates;
course['Year_Diff'] = date.today().year - course['Timestamp'].dt.year


# In[152]:


# The month difference between two dates;

course['Month_Diff'] = (date.today().year - course['Timestamp'].dt.year) * 12 + date.today().month - course['Timestamp'].dt.month


# In[153]:


# Day names;

course['Day_Name'] = course['Timestamp'].dt.day_name()


# In[154]:


course.head()


# <a id = "22"></a><br>
# <p style="font-family: 'Rockwell', cursive; font-weight: bold; letter-spacing: 2px; color: #556B2F; font-size: 180%; text-align: left; padding: 0px; border-bottom: 3px solid">✨Feature Interaction✨</p>

# In[155]:


titanic = titanic_.copy()


# In[156]:


titanic.head()


# In[157]:


titanic["NEW_AGE_PCLASS"] = titanic["Age"] * titanic["Pclass"] # Age için stanartlaştırma işlemi gerekebilir.

titanic["NEW_FAMILY_SIZE"] = titanic["SibSp"] + titanic["Parch"] + 1 

titanic.loc[(titanic['Sex'] == 'male') & (titanic['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

titanic.loc[(titanic['Sex'] == 'male') & (titanic['Age'] > 21) & (titanic['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

titanic.loc[(titanic['Sex'] == 'male') & (titanic['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

titanic.loc[(titanic['Sex'] == 'female') & (titanic['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

titanic.loc[(titanic['Sex'] == 'female') & (titanic['Age'] > 21) & (titanic['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

titanic.loc[(titanic['Sex'] == 'female') & (titanic['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


# In[158]:


titanic.head()


# <div style="border-radius:10px; border:#867070 solid; padding: 15px; background-color: #F5EBEB; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>👻 Analysis Results: </font></h3>
# 
# Feature Engineering is a crucial stage in data science projects and encompasses a range of techniques to make datasets more meaningful and suitable for modeling. In this section, we've focused on topics such as the detection and handling of outliers, addressing missing values, encoding variables, scaling features, extracting new features, and working with date, text, and regular expression features.
# 
# Handling outliers is important for cleaning the dataset and improving model performance. Addressing missing values correctly can minimize data loss while enhancing the reliability of the model. Encoding variables is used to convert categorical data into numerical formats, which can assist in better model performance. Creating new features can add more meaning to the data and boost model performance. Finally, working with date, text, and regular expressions allows handling more complex data types.
# 
# These topics provide essential tools for data engineers and data scientists to organize datasets and enhance modeling capabilities. Feature Engineering is a critical step in gaining better results by understanding datasets better and improving modeling capabilities.

# **Sources;**
# 
# * https://www.databricks.com/glossary/feature-engineering
# * https://careerfoundry.com/en/blog/data-analytics/what-is-an-outlier/#:~:text=In%20data%20analytics%2C%20outliers%20are,experimental%20errors%2C%20or%20a%20novelty.
# * https://help.rubiscape.io/rarg/machine-learning/anomaly-detection/local-outlier-factor
# * https://www.kaggle.com/dumanmesut
