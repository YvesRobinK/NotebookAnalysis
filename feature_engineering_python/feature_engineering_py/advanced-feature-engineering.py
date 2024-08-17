#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# 
# What is a feature and why we need the engineering of it? 
# 
# Basically, all machine learning algorithms use some input data to create outputs. This input data comprise features, which are usually in the form of structured columns. Algorithms require features with some specific characteristic to work properly. Here, the need for feature engineering arises. Feature engineering efforts mainly have two goals:
# 
# - Preparing the proper input dataset, compatible with the machine learning algorithm requirements.
# 
# - Improving the performance of machine learning models.
# 
# The important point is that machine learning algorithms desire structured dataset because of that reason feature engineering is a key indicator for data science life cyle. __Harward Business Review article__ stated that, "Poor data quality is enemy number one to the widespread, profitable use of machine learning. The quality demands of machine learning are steep, and bad data can rear its ugly head twice both in the historical data used to train the predictive model and in the new data used by that model to make future decisions. To ensure you have the right data for machine learning, you must have an aggressive, well-executed quality program."
# 
# Besides that, according to a survey in Forbes, data scientists spend 80% of their time on data preparation:
# 
# ![image.png](https://miro.medium.com/max/1400/0*-dn9U8gMVWjDahQV.jpg)
# 
# 
# 
# Source:
# 
# [Harward Business Review](https://hbr.org/2018/04/if-your-data-is-bad-your-machine-learning-tools-are-useless)
# 
# [Towards Data Science](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
# 
# [Forbes Survey](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/?sh=28c8fe0c6f63)
# 

# In this notebook we will deeply analyze feature engineering topics as below.
# 
# - Outliers
# - Missing Values
# - Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# - Feature Scaling
# - Feature Extraction
# - Feature Interactions
# - End-to-End Application

# # 1. Outliers
# 
# An outlier is an observation that lies an abnormal distance from other values in a random sample from a population. In a sense, this definition leaves it up to the analyst (or a consensus process) to decide what will be considered abnormal. Before abnormal observations can be singled out, it is necessary to characterize normal observations.
# 
# 👉 __Trimming__: It excludes the outlier values from our analysis. By applying this technique our data becomes thin when there are more outliers present in the dataset. Its main advantage is its fastest nature.
# 
# 👉 __Capping__: In this technique, we cap our outliers data and make the limit i.e, above a particular value or less than that value, all the values will be considered as outliers, and the number of outliers in the dataset gives that capping number.
# 
# For Example, if you’re working on the income feature, you might find that people above a certain income level behave in the same way as those with a lower income. In this case, you can cap the income value at a level that keeps that intact and accordingly treat the outliers.
# 
# 👉 __Treat outliers as a missing value__: By assuming outliers as the missing observations, treat them accordingly i.e, same as those of missing values.
# 
# 👉 __Discretization__: In this technique, by making the groups we include the outliers in a particular group and force them to behave in the same manner as those of other points in that group. This technique is also known as Binning.
# 
# __How to Detect Outliers ?__
# 
# 👉 __For Normal distributions__: Use empirical relations of Normal distribution.
# 
#        The data points which fall below mean-3*(sigma) or above mean+3*(sigma) are outliers.
# 
# where mean and sigma are the average value and standard deviation of a particular column.
# 
# 👉 __For Skewed distributions__: Use Inter-Quartile Range (IQR) proximity rule __(Box Plot).__
# 
#     The data points which fall below Q1 – 1.5 IQR or above Q3 + 1.5 IQR are outliers.
# 
# where Q1 and Q3 are the 25th and 75th percentile of the dataset respectively, and IQR represents the inter-quartile range and given by Q3 – Q1.
# 
# ![image.png](https://miro.medium.com/max/1400/1*NRlqiZGQdsIyAu0KzP7LaQ.png)
# 
# Source:
# 
# [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/)

# ## Lets Code and Practice 🚀👨🏼‍💻

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")
df.head()


# In[3]:


# Defining Interquartile Range
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr


# In[4]:


# less than the lower limit or greater than the upper limit
df[(df["Age"] < low) | (df["Age"] > up)]


# In[5]:


# lets find the index of the outliers
df[(df["Age"] < low) | (df["Age"] > up)].index


# In[6]:


# Lets find do I have any outliers ?
df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)


# In[7]:


# Lets add functionalty

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")


# In[8]:


# Lets add check outlier function for further needs (Return Boolean)
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")


# In[9]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
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

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# In[10]:


for col in num_cols:
    print(col, check_outlier(df, col))


# In[11]:


# Lets add function to grab the outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age", True)


# In[12]:


sns.boxplot(df["Age"])


# __Solving Outliers Problem__
# 
# We will check dropping and capping methods in order to solve outlier problems

# In[13]:


# Dropping the outlier data points
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

remove_outlier(df, "Fare").shape


# In[14]:


for col in ["Age", "Fare"]:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]


# In[15]:


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")


# In[16]:


# We can see that all the outlier data points have gone
sns.boxplot(df["Age"])


# __Multivariate Outlier Analysis (Local Outlier Factor)__

# In[17]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()


# In[18]:


for col in df.columns:
    print(col, check_outlier(df, col))


# In[19]:


# The higher LOF score means the more normal
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5] # selecting the worst 5 five scores


# In[20]:


scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()


# In[21]:


th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index


# In[22]:


df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


# # 2.Missing Values
# 
# The imputation method develops reasonable guesses for missing data. It’s most useful when the percentage of missing data is low. If the portion of missing data is too high, the results lack natural variation that could result in an effective model.
# 
# The other option is to remove data. When dealing with data that is missing at random, related data can be deleted to reduce bias. Removing data may not be the best option if there are not enough observations to result in a reliable analysis. In some situations, observation of specific events or factors may be required.
# 
# 💎 Direct removal of missing value observations from the data set and not examining the randomness will lose the statistical reliability of inferences and modelling studies (Alpar, 2011).
# 
# ![image.png](https://d35fo82fjcw0y8.cloudfront.net/2016/04/03210550/missing-values-.jpg)
# 

# ## Lets Code and Practice 🚀👨🏼‍💻

# In[23]:


V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IT", "IT", "IT"])

dff = pd.DataFrame(
    {"salary": V1,
     "V2": V2,
     "V3": V3,
     "departmant": V4}
)


# In[24]:


# Lets catch the missing values
dff.isnull().values.any()


# In[25]:


# Catching the missing value counts for each columns
dff.isnull().sum()


# In[26]:


# Catching the not null data counts for each columns
dff.notnull().sum()


# In[27]:


# Catching total missing value counts for all the dataset
dff.isnull().sum().sum()


# In[28]:


# Catching the columns that have at least 1 misssing value 
dff[dff.isnull().any(axis=1)]


# In[29]:


# Lets add functionality

def missing_values_table(dataframe, na_name=False):
    # The columns name that contains missing value
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # Number of missing data
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # Ration of the missing data points over the dataset
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Missing dataframe
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
    
missing_values_table(dff, True)


# ## We have defined the missing data points, so how can we solve the missing data problem ? 🙋
# 
# There are some approach to achieve this goal as below.
# 
# - Dropping the missing data points
# 
# - Assigning mean, median value of the related column of the dataset
# 
# - Using imputer to fill the missing data points
# 
# - Value Assignment in Categorical Variable Breakdown
# 
# - Using predictive methods to fill missing data points

# In[30]:


df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")
df.isnull().any()


# In[31]:


# Dropping the missing data points
df.dropna()


# In[32]:


# Assigning mean, median value of the related column of the dataset
df["Age"].fillna(0)
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())


# In[33]:


df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)


# In[34]:


dff["Embarked"].fillna(dff["Embarked"].mode()[0])
dff["Embarked"].fillna(dff["Embarked"].mode()[0]).isnull().sum()
dff["Embarked"].fillna("missing")


# In[35]:


dff.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

dff.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


# In[ ]:


# Using imputer to fill the missing data points
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')  # mean, median, most_frequent, constant
imp_mean.fit(df)
imp_mean.transform(df)


# In[37]:


# Value Assignment in Categorical Variable Breakdown
# Using predictive methods to fill missing data points
df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


# In[38]:


# Using predictive methods to fill missing data points
df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()


# In[39]:


# Scaling the dataset
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# In[40]:


# Lets use KNN imputer for predictive filling the missing data points

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()


# In[41]:


# Lets use inverse transform to reach the raw dataset
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df["age_imputed_knn"] = dff[["Age"]]
df.head()


# In[42]:


# Lets check the null age values with computed age_imputed_knn
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]


# ## Advanced Analysis for Missing Data Points
# 
# 💎 We will use missingo library for plotting and interpreting the figure.
# 
# 💎 And, we will check the missing value correlations and the missing values relation between each other

# In[43]:


msno.bar(df)
plt.show()


# In[44]:


msno.matrix(df)
plt.show()


# In[45]:


msno.heatmap(df)
plt.show()


# In[46]:


missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)


# # 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)

# ## Label Encoding & Binary Encoding
# 
# Encoding or continuization is the transformation of categorical variables to binary or numerical counterparts. An example is to treat male or female for gender as 1 or 0. Categorical variables must be encoded in many modeling methods (e.g., linear regression, SVM, neural networks)
# 
# ![image.png](https://womaneng.com/wp-content/uploads/2018/09/onehotencoding.jpg)

# In[47]:


df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")

df["Sex"].head()


# In[48]:


le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])


# In[49]:


# Lets add fuctionality

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# First defining the binary columns using categorical columns
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


# In[50]:


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)
    
df.head(10)


# ## One-Hot Encoding
# 
# Though label encoding is straight but it has the disadvantage that the numeric values can be misinterpreted by algorithms as having some sort of hierarchy/order in them. This ordering issue is addressed in another common alternative approach called ‘One-Hot Encoding’. In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column.
# 
# ![image.png](https://miro.medium.com/max/2000/1*WHM-sZuVQBOZzZv64fMgow.png)
# 
# __Advantages of one-hot encoding__
# - Does not assume the distribution of categories of the categorical variable.
# 
# - Keeps all the information of the categorical variable.
# 
# - Suitable for linear models.
# 
# __Limitations of one-hot encoding__
# 
# - Expands the feature space.
# 
# - Does not add extra information while encoding.
# 
# - Many dummy variables may be identical, and this can introduce redundant information.
# 
# Source:
# 
# [One-Hot Encoding](https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd)

# In[51]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()


# ## Rare Encoding
# 
# Rare labels are those that appear only in a tiny proportion of the observations in a dataset. Rare labels may cause some issues, especially with overfitting and generalization.
# The solution to that problem is to group those rare labels into a new category like other or rare—this way, the possible issues can be prevented.
# 
# ![image.png](https://miro.medium.com/max/2000/1*wmgHrdrZ3fXvlYL5zHpt7A.png)
# 
# This way, categories that are new in the test set are treated as rare, and the model can know how to handle those categories as well, even though they weren’t present in the train set.
# 
# Source: 
# 
# [Rare Encoding](https://heartbeat.comet.ml/hands-on-with-feature-engineering-techniques-encoding-categorical-variables-be4bc0715394)

# In[52]:


# 1. Target Frequency
# 2. Traget Ratio
# 3. Group by columns for target column

# Lets use large dataset to understand better the rare encoding
df = pd.read_csv("../input/home-credit-default-risk/application_train.csv")

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# In[54]:


df.TARGET.head()


# In[55]:


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)


# In[56]:


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# With the ouput, we can analyze the rare columns
rare_analyser(new_df, "TARGET", cat_cols)


# In[57]:


rare_analyser(df, "TARGET", cat_cols)


# # 4. Feature Scaling
# 
# ![image.png](https://miro.medium.com/max/2000/1*yR54MSI1jjnf2QeGtt57PA.png)
# 
# Feature scaling in machine learning is one of the most critical steps during the pre-processing of data before creating a machine learning model. Scaling can make a difference between a weak machine learning model and a better one.
# The most common techniques of feature scaling are Normalization and Standardization.
# Normalization is used when we want to bound our values between two numbers, typically, between [0,1] or [-1,1]. While Standardization transforms the data to have zero mean and a variance of 1, they make our data unitless. Refer to the below diagram, which shows how data looks after scaling in the X-Y plane.
# 
# ![image.png](https://i.stack.imgur.com/lggVP.png)
# 
# __Type of feature scaling:__
# 
# - __StandardScaler__: z = (x - u) / s
# 
# - __RobustScaler__: value = (value – median) / (p75 – p25)
# 
# - __MinMaxScaler__:  
#         
#         X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#         X_scaled = X_std * (max - min) + min
# 
# 
# - __Logaritmic Scaler__: Taking the log of the value. But, if we have a negative values we couldn't take the log. So we need to be careful abaout it.
# 
# 
# Source:
# 
# [Towards Data Science](https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)

# In[59]:


df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")

scaler = StandardScaler()

df["Age_standard_scaler"] = scaler.fit_transform(df[["Age"]])

df.head()


# In[60]:


rscaler = RobustScaler()

df["Age_robuts_scaler"] = rscaler.fit_transform(df[["Age"]])

df.head()


# In[61]:


mmscaler = MinMaxScaler()

df["Age_min_max_scaler"] = mmscaler.fit_transform(df[["Age"]])

df.head()


# In[62]:


df["Age_log"] = np.log(df["Age"])

df.head()


# # 5. Feature Extraction
# 
# Feature extraction is a process of dimensionality reduction by which an initial set of raw data is reduced to more manageable groups for processing. A characteristic of these large data sets is a large number of variables that require a lot of computing resources to process.
# 
# ![image.png](https://www.shopfactory.com/contents/media/feature-people.png)
# 
# [Deep AI](https://deepai.org/machine-learning-glossary-and-terms/feature-extraction#:~:text=Feature%20extraction%20is%20a%20process,of%20computing%20resources%20to%20process.)

# In[65]:


# If the cabin is Nan we will assign as 0 otherwise 1
# We know that employees att Titanic dont have Cabin

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# So let's analyze them if they survive or not

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


# In[66]:


# Lets check the used method for Cabin column using proportion test

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Ho rejected. That means, there is no difference between the cabin breakdown for Survived target as statistically


# In[68]:


#sibsp	# of siblings / spouses aboard the Titanic	
#parch	# of parents / children aboard the Titanic

# Lets check the relation as alone or not using feature extraction

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()


# In[69]:


df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


# In[71]:


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Ho rejected. That means, there is no difference between the cabin breakdown for Survived target as statistically


# In[72]:


# Lets check the title of the titanic crew and analyze

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})


# In[73]:


# Lets use the same method for all title using regex
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})


# # 6. Feature Interactions
# 
# ![image.png](https://miro.medium.com/max/620/1*SGai7lOKRn9YhM2p-8SChQ.jpeg)
# 
# If a machine learning model makes a prediction based on two features, we can decompose the prediction into four terms: a constant term, a term for the first feature, a term for the second feature and a term for the interaction between the two features.
# The interaction between two features is the change in the prediction that occurs by varying the features after considering the individual feature effects.
# 

# In[74]:


df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df["NEW_AGExPCLASS"] = df["Age"] * df["Pclass"]


# In[75]:


df.head()


# # 7. End-to-End Application
# 
# We will use Titanic dataset and we will use all the feature engineering methods. After all this process, we will use Random Forest Classifier to predict Survive or not.
# 
# ![image.png](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

# In[76]:


df = pd.read_csv("../input/data-science-day1-titanic/DSB_Day1_Titanic_train.csv")


# In[81]:


# Feature Engineering


# Cabin bool
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["Name"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
# is alone
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()


# In[82]:


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

df.shape


# In[83]:


#############################################
# 2. Outliers (Aykırı Değerler)
#############################################


for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))


# In[86]:


#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)
df.head()


df.drop("Cabin", inplace=True, axis=1)
missing_values_table(df)



remove_cols = ["Ticket", "Name"]
df.drop(remove_cols, inplace=True, axis=1)
df.head()

missing_values_table(df)


# In[88]:


df["Age"] = df["Age"].fillna(df.groupby("NEW_TITLE")["Age"].transform("median"))
missing_values_table(df)

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

missing_values_table(df)

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
missing_values_table(df)


# In[89]:


# Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# In[90]:


# Rare Encoding

rare_analyser(df, "Survived", cat_cols)

df = rare_encoder(df, 0.01)
df["NEW_TITLE"].value_counts()
rare_analyser(df, "Survived", cat_cols)


# In[92]:


#############################################
# One-Hot Encoding
#############################################


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PassengerId" not in col]


rare_analyser(df, "Survived", cat_cols)
df.head()



(df["Sex"].value_counts() / len(df) < 0.01).any()
(df["NEW_NAME_WORD_COUNT_9"].value_counts() / len(df) < 0.01).any()


useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


# In[93]:


#############################################
# Standart Scaler
#############################################


num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

# son kontrol:
df.head()
df.shape
df.tail()


# In[96]:


#############################################
# Model
#############################################


y = df["Survived"]
X = df.drop(["PassengerId", "Survived"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# MODEL
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


# In[ ]:




