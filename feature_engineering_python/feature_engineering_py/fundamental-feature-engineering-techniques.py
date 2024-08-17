#!/usr/bin/env python
# coding: utf-8

# ---
# 
# <h1 style="text-align: center;font-size: 40px;">Feature Engineering Techniques you should know</h1>
# 
# ---
# 
# <center><img src="https://repository-images.githubusercontent.com/184386548/2ed13600-4a1a-11eb-8cd9-1e23e4e66ed1" width="500" height="600"></center>
# 
# ---

# Feature engineering is the pre-processing step of machine learning, which is used to transform raw data into features that can be used for creating a predictive model using Machine learning or statistical Modelling.It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy. Feature engineering is required when working with machine learning models. Regardless of the data or architecture, a terrible feature will have a direct impact on your model. It helps to represent an underlying problem to predictive models in a better way, which as a result, improve the accuracy of the model for unseen data. Here, in this kernel we are going to learn some fundamental feature engineering techniques.

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv("/kaggle/input/house-rent-prediction-dataset/House_Rent_Dataset.csv")
df2 = pd.read_csv("/kaggle/input/world-happiness-report-2021/world-happiness-report.csv")
df3 = pd.read_csv("/kaggle/input/titanic/train.csv")
df4 = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df.head(2)


# <h3> 1. Imputation:</h3>

# > Editing and imputing are both methods of data processing. Editing refers to the detection and correction of errors in the data. Whilst imputation is a method of correcting errors and estimating and filling in missing values in a data set. Model based imputation refers to a class of methods that estimate missing values using assumptions about the distribution of the data, which include mean and median imputation. To deal with missing values you need to apply different techniques of imputation.

# In[3]:


df3.isnull().mean()


# - Here, we can see that on df3 theres a few  missing values on Age,Cabin and Embarked. We can fill those missing values if it is very less compared to our dataframe. But if the  number of missing values are very high, you need to be very carefull to deal with these values.

# In[4]:


df3.isnull().sum()/len(df3)


# > Here, we're dropping columns in which the percentage of missing values are greater than 70%

# In[5]:


threshold = 0.7
df3 = df3[df3.columns[df3.isnull().sum()/len(df3)< threshold]]
df3.columns


# - we can see that the column cabin has been dropped, because it has missing values greater than 70%

# * **1.1 Numerical Imputation:**
# 
# > - But what about the other two columns Age and Embarked , which also contains missing values. Since column age is numeric, you need to perform numerical imputation, which is to fill these columns missing values with mean, median or mode. But, if you have a column that only has 1 and NA, then it is likely that the column is a binary attribute and hence the NA corresponds to 0. And for column Embarked you need to perform categorical imputation since it is categorical.
# > - Since Age is not a binary attribute, hence you can fill the missing values with the mean, median or mode of this column. I think the best imputation way is to use the medians of the columns. As the mean of the columns are sensitive to the outlier values, while medians are more solid in this respect.

# In[6]:


#Filling missing values with medians of the columns
df3['Age'].fillna(df3['Age'].median(), inplace = True)


# In[7]:


df3['Age'].isnull().sum()


# > Now, we can see that theres no null values in Age column, all values have been filled with the median of this column.

# > **1.2 Categorical Imputation**

# - Replacing the missing values of a categorical column with the maximum occurred value which is called mode of the column is a good option for handling categorical columns. But if the values of the column are uniformly distributed then adding an extra category namely anything i.e. "Unknown" might be more sensible.
# 

# In[8]:


#Replacing with the maximum occurred value
df3['Embarked'].fillna(df3['Embarked'].mode().values[0], inplace = True)


# > Now, let's check the missing values

# In[9]:


df3.isnull().sum()


# - So, we've successfully removed all the missing values

# <h3> 2. Handling Outliers:</h3>
# 
# > Outliers are those data points that are significantly different from the rest of the dataset. They are often abnormal observations that skew the data distribution, and arise due to inconsistent data entry, or erroneous observations.  As outliers are very different values—abnormally low or abnormally high—their presence can often skew the results of statistical analyses on the dataset. This could lead to less effective and less useful models. So, before applying dataset to build your model, you need to deal with these outliers. You can detect and drop outliers by statistical methods or simply detect outliers by data visualization.

# * **2.1 Using Boxplot:**

# ![](https://miro.medium.com/max/700/1*0MPDTLn8KoLApoFvI0P2vQ.png)

# > For boxplot:
# - Lower Limit = Q1 - 1.5 * IQR
# - Upper Limit = Q3 + 1.5 * IQR</br>
# Anything less than lower limit or greater than upper limit is considered as outliers

# In[10]:


plt.figure(figsize=(9,6))
sns.set_style("whitegrid")
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = df4)
plt.show();


# > Here we can see that there are some values specifically for sale condition which are Normal are significantly different from the rest of the datapoints. These datapoints are considered as outliers.

# * **2.2 Using Standard Deviation:**
# 
# > For Standard Deviation:
# >- Lower Limit = mean - x * standard deviation
# >- Upper Limit = mean + x * standard deviation
# >- Practically the value of x is considered between 2 and 4 </br>
# Anything less than lower limit or greater than upper limit will be considered as outliers

# In[11]:


x = 2
upper_lim = df4['SalePrice'].mean () + df4['SalePrice'].std () * x
lower_lim = df4['SalePrice'].mean () - df4['SalePrice'].std () * x

data = df4[(df4['SalePrice'] < upper_lim) & (df4['SalePrice'] > lower_lim)]
data.shape


# > So, by using standard deviation, we have dropped 63 outlier values from df4

# * **2.3 Using Percentiles**
# 
# >Another method of detecting and removing outliers is using percentiles. You can consider a certain percentage from top and bottom of your data as an outlier. Let’s define a custom range that accommodates all data points that lie anywhere between 0.5 and 95 percentile of the datapoints. Datapoints outside this range will consider as outliers. 

# In[12]:


#Dropping rows with  outliers using Percentiles
upper_lim = df4['SalePrice'].quantile(.95)
lower_lim = df4['SalePrice'].quantile(.05)

data = df4[(df4['SalePrice'] < upper_lim) & (df4['SalePrice'] > lower_lim)]
data.shape


# > So, by using percentiles, we have dropped 148 outlier values from df4
# 
# - Note: Don't just drop values, first detect if there is any outliers or not. To do this, the best option will be visualization. You can use boxplot or scatterplot for this.

# <h3> 3. Log Transformation: </h3>
# 
# > Log transformation is a data transformation method in which it replaces each variable x with a log(x). The choice of the logarithm base is usually left up to the analyst and it would depend on the purposes of statistical modeling. When our original continuous data do not follow the bell curve, we can log transform this data to make it as “normal” as possible so that the statistical analysis results from this data become more valid. In other words, the **log transformation reduces the skewness of our original data.** People may use logs because they think it compresses the scale or something, but the principled use of logs is that you are working with data that has a lognormal distribution. This will tend to be things like salaries, housing prices, etc, where all values are positive and most are relatively modest, but some are very large. If you can take the log of the data and it becomes normalish, then you can take advantage of many features of a normal distribution, like well-defined mean, standard deviation (and hence z-scores), etc.</br>
# > **Note:** Log transformat will give you an error, if your data has any negative values. Also, you can add 1 to your data before transform it. Thus, you ensure the output of the transformation to be positive.

# In[13]:


import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.figure(figsize=(12,5))
sns.distplot(df["Rent"])
plt.show()


# > Here, we can see that feature "Rent" in dataframe df is positively skewed, which is very large. We can reduce this skewness using log transformation.

# In[14]:


import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.figure(figsize=(12,5))
sns.distplot(np.log(df["Rent"]+1))
plt.show()


# - We can see that, log transformation reduced the skewness of the feature Rent

# <h3> 4. Binning: </h3>
# 
# > Data binning, bucketing is a data pre-processing method used to minimize the effects of observation errors. Binning is the process of transforming numerical variables into categorical counterparts. Binning improves accuracy of the predictive models by reducing the noise or non-linearity in the dataset. Finally, binning lets easy identification of outliers, invalid and missing values of numerical variables. Binning is a quantization technique in Machine Learning to handle continuous variables. It is one of the important steps in Data Wrangling. An example could be grouping a person’s age into interval where 1-18 falls under a minor, 19- 29 under young, 30-49 under old, and 50-100 in very old.

# In[15]:


#Creating bins with labels
bins = [1.0,18.0,30.0,50.0,100.0]
labels = ['Minor','Young','Old','Very Old']

df3['age_category'] = pd.cut(df3['Age'],bins = bins, labels = labels)
df3.head(3)


# > For more information about binning check this [link](https://towardsdatascience.com/binning-for-feature-engineering-in-machine-learning-d3b3d76f364a)

# <h3> 5. One-hot encoding:</h3>
# 
# > Encoding is a technique of converting categorical variables into numerical values so that it could be easily fitted to a machine learning model. Categorical variables must be changed in the pre-processing section since machine learning models require numeric input variables. Nominal or ordinal data can be found in categorical data. The two most popular technique of encodings are **Ordinal Encoding** and **One-Hot Encoding**.

# * **5.1 Ordinal Encoding:**
# > In ordinal encoding, each unique category value is assigned an integer value. For example, “Mango” is 1, “Banana” is 2, and “Apple” is 3. This is called an ordinal encoding or an integer encoding and is easily reversible. Often, integer values starting at zero are used. For some variables, an ordinal encoding may be enough. The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship. It is a natural encoding for ordinal variables. For categorical variables, it imposes an ordinal relationship where no such relationship may exist. This can cause problems and a one-hot encoding may be used instead.

# In[16]:


# example of a ordinal encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Area Type"] = le.fit_transform(df["Area Type"])
df.head(2)


# - Each unique category of Area Type has been is assigned with an integer value. For more information about Encodings check this [link](https://towardsdatascience.com/6-ways-to-encode-features-for-machine-learning-algorithms-21593f6238b0).

# * **5.2 One Hot Encoding:**
# > One-hot encoding in machine learning is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy. It is a common method for dealing with categorical data in machine learning.
# This approach creates a new column for each unique value in the original category column.
# Because this procedure generates several new variables, it is prone to causing a large problem (too many predictors) if the original column has a large number of unique values.
# 
# []()
# <img src="https://e6v4p8w2.rocketcdn.me/wp-content/uploads/2022/01/One-Hot-Encoding-for-Scikit-Learn-in-Python-Explained-1024x576.png" alt="One Hot Encoding" style="width:900px;height:300px;">
# 

# In[17]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse=False)
ohe_feat = pd.DataFrame(ohe.fit_transform(df[["City"]]))

df = df.join(ohe_feat)
df.head(2)


# * Another easy way to do one hot encoding, is to use pandas  get_dummies function

# In[18]:


encoding = pd.get_dummies(df['City'], drop_first = True)
df = df.join(encoding)
df.head(2)


# > You can see, the last 5 columns, are the results of One Hot Encoding
# 
# > - You must have noticed that we have dropped the first features column. But why we did that????
# Because, drop_first=True is important to use, as it helps in reducing the extra column created during dummy variable creation. Let’s say we have 3 types of values in Categorical column and we want to create dummy variable for that column. If one variable is "not furnished" and another one is "semi_furnished", then it is obvious, the 3rd variable will be "unfurnished". So we dont need 3rd variable to identify "unfurnished"

# <h4> When to use What??</h4></br>
# 
# * Label encoder is used when:
# 1. The number of categories is quite large as one-hot encoding can lead to high memory consumption.
# 2. When the order does not matter in categorical feature.
# 
# * One Hot encoder is used when:
# 1. When the order does not matter in categorical features
# 2. Categories in a feature are fewer.
# 
# **Note**: The model will misunderstand the data to be in some kind of order, 0 < 1 < 2. For e.g. In the above six classes’ example for “City” column, the model misunderstood a relationship between these values as follows: 0 < 1 < 2 < 3 < 4 < 5. To overcome this problem, we can use one-hot encoding as explained.

# <h3>6. Scaling: </h3>
# 
# > Scaling of the data comes under the set of steps of data pre-processing when we are performing machine learning algorithms in the data set. As we know all of the supervised learning methods make decisions according to the data sets applied to them and often the algorithms calculate the distance between the data points to make better inferences out of the data. In real life, variations in the same thing makes us confused. If we take an example of purchasing apples from a bunch of apples, we go close to the shop, examine various apples and pick various apples of the same attributes. Because we have learned about the attributes of apples and we know which are better, and which are not good. So if most of the apples consist of pretty similar attributes we will take less time in the selection of the apples which directly affect the time of purchasing taken by us. The moral of the example is that, if every apple in the shop is good we will take less time to purchase or if the apples are not good enough then we will take more time in the selection process which means that if the values of attributes are closer we will work faster and the chances of selecting good apples also strong. Similarly in machine learning algorithms if the values of the features are closer to each other there are chances for the algorithm to get trained well and faster instead of the data set where the data points or feature values have high differences with each other which will take more time to understand the data and the accuracy will be lower. So if the data in any conditions has data points far from each other, scaling is a technique to make them closer to each other or in simpler words, we can say that the scaling is used for making data points generalized so that the distance between them will be lower. To know more about scaling read [this](https://analyticsindiamag.com/why-data-scaling-is-important-in-machine-learning-how-to-effectively-do-it/#:~:text=Scaling%20the%20target%20value%20is,learn%20and%20understand%20the%20problem.&text=Scaling%20of%20the%20data%20comes,algorithms%20in%20the%20data%20set.).

# * **6.1 Data Normalization:**
# 
# > In statistics, normalization is the method of rescaling data where we try to fit all the data points between the range of 0 to 1 so that the data points can become closer to each other. It is a very common approach to scaling the data. In this method of scaling the data, the minimum value of any feature gets converted into 0 and the maximum value of the feature gets converted into 1. To do this you can use scikit learns package MinMaxScaler.
# ![](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/08/image-288-1024x253.png)
# 

# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
sc_feat = pd.DataFrame(scaler.fit_transform(df[["Rent","Size"]]), columns = ["Rent","Size"])
sc_feat.head(5)


# > We can see that the Rent and Size column has been scaled, according to their minimum and maximum values

# * **6.2 Data Standardization:**
# 
# > Like normalization, standardization is also required in some forms of machine learning when the input data points are scaled in different scales. Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation. This technique also tries to scale the data point between zero to one but in it, we don’t use max or minimum. Here we are working with the mean and the standard deviation.
# 
# ![](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/08/image-289.png)

# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc_feat = pd.DataFrame(scaler.fit_transform(df[["Rent","Size"]]), columns = ["Rent","Size"])
sc_feat.head(3)


# References:
# 1. https://bookdown.org/v_anandkumar88/docs2/what-is-imputation.html
# 2. https://medium.com/@agarwal.vishal819/outlier-detection-with-boxplots-1b6757fafa21
# 3. https://medium.com/@kyawsawhtoon/log-transformation-purpose-and-interpretation-9444b4b049c9
# 4. https://bit.ly/3eQW9oX
# 5. https://towardsdatascience.com/6-ways-to-encode-features-for-machine-learning-algorithms-21593f6238b0
# 6. https://www.naukri.com/learning/articles/one-hot-encoding-vs-label-encoding/
# 7. https://bit.ly/3BETI1K

# ---
# 
# <h1 style="text-align: center;font-size: 25px;">Thanks for Reading</h1>
# 
# ---

# In[ ]:




