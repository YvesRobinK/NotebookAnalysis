#!/usr/bin/env python
# coding: utf-8

# ## Types of Missing Values
# There are mainly three types of missing values- MCAR, MNAR and MAR.
# 
# **1. MCAR (Missing Completly At Random)**: 
# - A variable is missing completely at random if the probability of being missing is the same for all the observations.
# - When data is MCAR, which Means there is absolutely no relationship between the data missing and any other observed or missing value in the dataset. 
# - In other words, those missing data points are a random subset of the dataset.
# 
# **2. MNAR (Missing Data Not At Random)**:
# - As the name suggests their will be some relationship between the data missing and any other value in the dataset.
# 
# **3. MAR(Missing At Random)**:
# - Missing at Random means, the propensity for a data point to be missing is not related to the missing data, but it is related to some of the observed data

# ### **Explanation for Missing Values**
# 
# There can be many reasons for missing values inside datasets. For example, in a dataset of height and age, there will be more missing values in the age column because girls are not comfortable talking about their age. Similarly, if we prepare a dataset for predicting an employee's salary that includes columns like experience and salary, the salary column will have more missing values because most people do not want to share information about their salary. In a bigger scenario where we are preparing data for a large population, some diseases, people who died in an accident, the number of taxpayers inside a city, etc. People participating in all these and similar scenarios usually hesitate to put down their personal information and sometimes hide the real numbers. Even if you download the data from a third-party resource, there is still some chance of missing values due to some corruption in the file while downloading.

# ### **Why Handle Missing Values ?**
# * Many machine learning algorithms faild to perform on the dataset if it contains missing values. However there are some algorithms that works even with missing values like K-nearest neighbours and Naive Bayes.
# 
# * You may end up building a biased model that will lead to incorrect results.
# * Missing values reduces accuracy of the model.
# * Missing values can lead to less precision.

# In[1]:


## Sample Datasets & Libraries
import pandas as pd
titanic = pd.read_csv ('../input/titanic/train.csv')


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# before imputing missing values let's see how to find missing values inside dataset.

# ## **Checking For Missing Values Inside The Dataset**

# In[3]:


## 1. Using isna() or isnull() To Find Count
titanic.isna().sum()


# In[4]:


## 2. Missing Values Percentage
round((titanic.isna().sum() / len(titanic))*100,2)


# In[5]:


## 3. Visulizing Missing Values
import missingno as mn
mn.matrix(titanic,figsize=(10,8));


# # How To Handle Missing Values?

# There are two ways to handle missing values-
# 1. Dropping Missing Values
# 2. Imputing Missing Values With Some Other Value (Preferred)

# ### 1. Dropping Missing Values
# * if missing values is of type Missing At Random(MAR) or Missing Completly At Random(MCAR) then it can be deleted. 
# * One of the biggest disadvantage of dropping missing values is one might end up deleting some useful data as well with missing values. 
# 
# There are two ways to remove missing values
# 1. Deleting the entire row containing missing values
# 2. Deleting the entire column containing missing values

# #### 1. Deleting The Entire Row

# In[6]:


## Copy of data
df = titanic.copy()
df = df.dropna(axis=0)
df.isna().sum()


# In[7]:


print('Dataset Size With Missing Values',titanic.shape)


# In[8]:


print('Dataset Size Without Missing Values',df.shape)


# There is a big loss of data, because Cabin column contains more than 75% missing values.

# #### 2. Deleting The Entire Column

# In[9]:


## Copy of data
df = titanic.copy()
df = df.drop(['Cabin','Age','Embarked'],axis=1) 
df.isna().sum()


# Although We Have Removed Missing Values, But With Them a big part of data is also lost. Like Age is a very important column for survival of a person. 

# > Deleting Missing Values From The Dataset Is Only An Option if we have less than 10% of missing values on a big dataset. 
# 
# The Best and Evergreen option is to Impute missing values with some other similar value.

# ## 2. Imputing Missing Values

# ## METHOD 1 → Mean Value Imputation 

# In this technique we replace missing values with the `mean` value of the column containing missing values

# In[10]:


## First Step : Check The Distribution
sns.distplot(titanic.Age);


# In[11]:


def impute_nan_mean(df,column,mean):
    df[column+'_mean'] = df[column].fillna(mean)
    return df

mean_val = titanic.Age.mean()
titanic = impute_nan_mean(titanic,'Age',mean_val)


# In[12]:


titanic[titanic.Age.isna()][['Age','Age_mean']]


# Mean Imputation is a great choice, if distribution is normal or close to normal. 
# 
# In case of skewed distribution we shoud go with meadian value.

# ## METHOD 2: Median Value Imputation
# 
# In this technique we replace all the missing value with median value of the column

# In[13]:


titanic


# In[14]:


## Imputing missing value with median value of the column
def impute_nan_median(df,column,median):
    df[column+'_median'] = df[column].fillna(median)
    return df

median_val = titanic.Age.median()
titanic = impute_nan_median(titanic,'Age',median_val)


# In[15]:


titanic[titanic.Age.isna()][['Age','Age_median']]


# ## METHOD 3: Mode Imputation (Frequent Category Imputation)
# 
# Mode value imputation is mostly used for categorical data. it can also be used for numerical variables as well.
# 
# * In this technique we replace all the missing values with the most frequent value of the column.

# In[16]:


## Imputing missing value with median value of the column
def impute_nan_mode(df,column,mode):
    df[column+'_mode'] = df[column].fillna(mode)
    return df

mode_val = titanic.Age.mode()
titanic = impute_nan_mode(titanic,'Age',mode_val)


# In[17]:


titanic[titanic.Age.isna()][['Age','Age_mode']]


# you see how most NAN values gets replaced with NAN values because the most frequent value of this numerical column is `NAN`. To use mode here we need to ignore all the NAN values and then apply mode on remaining values. 

# In[18]:


titanic[titanic.Age.notna()]['Age'].mode()[0]


# In[19]:


## Imputing missing value with median value of the column
def impute_nan_mode(df,column,mode):
    df[column+'_mode'] = df[column].fillna(mode)
    return df

mode_val = titanic[titanic.Age.notna()]['Age'].mode()[0]    ## find mode of all the non missing values of Age column
titanic = impute_nan_mode(titanic,'Age',mode_val)


# In[20]:


titanic[titanic.Age.isna()][['Age','Age_mode']]


# Although the above code worked but you should never use mode for a numerical column until there is some exception.
# 
# Mode works better for categorical data with less categories.
# 
# * One of the biggest advantage of using mode as categorical data imputer is that we don't need to convert categories into numerical data.
# 
# Let's use it to fill missing values in Embarked Column.

# In[21]:


titanic.Embarked.isna().sum()


# In[22]:


#### STEP 1: Find Mode Values
mode_cat_embarked = titanic.Embarked.mode()[0]

#### STEP 2: Fill Missing Values With Most Frequent Category
titanic['Embarked_mode'] = titanic['Embarked'].fillna(mode_cat_embarked)

#### Check For Results
titanic['Embarked_mode'].isna().sum()


# In[23]:


mode_cat_embarked


# ## METHOD 4:  Random Sample Imputation
# 
# In this method we will replace all the missing values with  a random sample from the data.

# In[24]:


### STEP 1: Generating Random Sample
sample = titanic.Age.sample().values[0]

### STEP 2: Filling nan values with random sample value
titanic['Age'+'_random_sample'] = titanic['Age'].fillna(sample)

titanic[titanic['Age'].isna()][['Age','Age_random_sample']]


# In[25]:


### STEP 1: Generating Random Sample
sample = titanic.Age.sample().values[0]

### STEP 2: Filling nan values with random sample value
titanic['Age'+'_random_sample'] = titanic['Age'].fillna(sample)

titanic[titanic['Age'].isna()][['Age','Age_random_sample']]


# You see everytime we run the codeblock we get a new random sample value.
# 
# To avoid this we can make use of `np.seed()`
# 
# * Note: There are chances that our random sample will pick nan as a sample, to avoid this we can use the same method we have used in mode.

# In[26]:


import numpy as np
np.random.seed(42)

### STEP 1: Generating Random Sample
sample = titanic[titanic.Age.notna()]['Age'].sample().values[0]

### STEP 2: Filling nan values with random sample value
titanic['Age'+'_random_sample'] = titanic['Age'].fillna(sample)

titanic[titanic['Age'].isna()][['Age','Age_random_sample']]


# no matter how many time we run the above codeblock the value will remain same, until whole execution resets.

# ## METHOD 5: End of Distribution

# If Missing Value is not at random then we can use this method.  In this we replace all the missing values with 3rd std deviation value

# In[27]:


sns.boxplot(x = 'Age',data=titanic);


# In[28]:


### STEP 1: Find Extreme Value
extreme = titanic.Age.mean() + 3*titanic.Age.std()

### STEP 2: Fill nan with extreme value
titanic['Age_end_distribution']  = titanic['Age'].fillna(extreme)

titanic
titanic[titanic['Age'].isna()][['Age','Age_end_distribution']]


# we can also use a least value based on our data distribution.

# #### In all above methods we are replacing missing values with some other value from the sample. 
# What if missing values are not at random?
# 
# What if you are required to showcase the importance of missing values?
# 
# In all above and similar cases we can use Arbitrary Value Imputation.

# ## METHOD 6.1: Arbitrary Value Imputation (Numerical)
# 
# In this method we replace missing value with lowest or highest value of the distribution. (-infinity, +infinity) 
# 
# In case of Age we can replace missing values either with 0 or 100.(least and highest) 
# 
# * This method not only fill missing values but also captures the importance of it.

# In[29]:


### Filling Values using 0
titanic['Age_0'] = titanic['Age'].fillna(0)


# In[30]:


### Filling Values using 100
titanic['Age_100'] = titanic['Age'].fillna(100)


# In[31]:


titanic[titanic.Age.isna()][['Age','Age_0','Age_100']]


# ## METHOD 6.2: Arbitrary Value Imputation (Categorical)

# In[32]:


titanic['Cabin_Missing'] = titanic['Cabin'].fillna('Missing')


# In[33]:


titanic[titanic.Cabin.isna()][['Cabin','Cabin_Missing']]


# This method works fine but the only problem with this it it creates a new category `Missing`. If count of missing values is higher than other known categories than `Missing`category will impact the result more that will leads to an bad performing model. 
# 
# To prevent this problem we can create a new feature containing information about missing values, where 1 will represent a missing value.

# ## METHOD 7: Capturing Missing Values with new feature.
# 
# It works well if the data are not missing completely at random.

# In[34]:


import numpy as np
titanic['Age_nan']=np.where(titanic['Age'].isnull(),1,0)


# In[35]:


### STEP 1: Creating a new feature
titanic['Cabin_nan']=np.where(titanic['Cabin'].isnull(),1,0)


# In[36]:


titanic.iloc[:5][['Cabin','Cabin_nan']]


# Once a new feature is added we can make use of any method to fill missing values. 

# In[37]:


titanic['Cabin'].mode()[0]


# In[38]:


titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])


# We can try same for numerical column `Age` as well.

# In[39]:


titanic['Age_nan']=np.where(titanic['Age'].isnull(),1,0)


# In[40]:


titanic[['Age','Age_nan']]


# All above methods we have discussed makes use of manual approach to find missing values and then fill using fillna. There are some advanced methods as well that makes use of machine learning models to predict missing values. 
# 
# Let's see some of the most famous technique of predicting missing values using machine learning.

# ## METHOD 8: KNN Imputer
# 
# In this technique, we predict missing values using an algorithm that uses the values of nearby data points to impute, or predict, missing values. KNN stands for K-Nearest Neighbors, with K referring to the (user-defined) number of neighbors the algorithm will take into account.
# 
# KNN only works with numerical data. In some cases, however, non-numerical (e.g., categorical) data may be converted to numerical data for use with the KNN Imputer as well. We can also use the KNN Imputer with multiple features (or columns) at once.

# In[41]:


titanic


# In[42]:


df = titanic[["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]]
df["Sex"] = [1 if x=="male" else 0 for x in df["Sex"]]


# In[43]:


### Defining Cols
cols = ['Age','Survived']

### Defining KNN imputer with neighbors
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=5)

knn.fit_transform(df[cols])


# In[44]:


df2 = pd.DataFrame(knn.transform(df[cols]),columns=['Age', 'Survived'])


# In[45]:


df2


# In[46]:


titanic['Age_KNN'] = df2['Age']


# In[47]:


titanic[titanic.Age.isna()][['Age','Age_KNN']]


# ## METHOD 9: Predicting NAN values using Linear Regression

# In[48]:


titanic.Age.head(8)


# In[49]:


train_data = df[df['Age'].notna()]
test_data = df[df["Age"].isnull()]

X_train= train_data.drop('Age',axis=1)
y_train = train_data['Age']

X_test = test_data.drop('Age',axis=1)
y_test = test_data['Age']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred


# In[50]:


a = y_train.to_list()
b = list(y_pred)
a.extend(b)


# In[51]:


titanic['Age_LR'] = a


# In[52]:


titanic.head(10)


# In[53]:


titanic[titanic.Age.isna()][['Age','Age_LR']]


# ## Method 10: Imputation Using Multivariate Imputation by Chained Equation (MICE)

# In[54]:


get_ipython().system('pip install impyute')


# In[55]:


from impyute.imputation.cs import mice

# start the MICE training
data_imputed=mice(df.values)


# In[56]:


cleaned_df = pd.DataFrame(data_imputed,columns=['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age'])


# In[57]:


titanic['Age_MICE'] = cleaned_df['Age']


# In[58]:


titanic[titanic.Age.isna()][['Age','Age_MICE']]


# ### Let's Compare Each Method Imputations...........

# In[59]:


titanic[titanic.Age.isna()][['Age','Age_mean','Age_median', 'Age_mode', 'Age_random_sample','Age_end_distribution', 'Age_0', 'Age_100','Age_KNN','Age_LR','Age_MICE']]


# ## Thanks For Reading
# I hope You Found Someting useful.
# 
# ### Don't Forgot to give the notebook a 🔼

# ### **----Check Out My Other Notebooks As Well---**
# 
# [Best Numpy Functions For Data Science](https://www.kaggle.com/code/abhayparashar31/best-numpy-functions-for-data-science)<br>
# [Spaceship Titanic Complete Analysis + Prediction Using Plotly & XGBClassifier](https://www.kaggle.com/code/abhayparashar31/spaceship-titanic-complete-analysis-prediction)
