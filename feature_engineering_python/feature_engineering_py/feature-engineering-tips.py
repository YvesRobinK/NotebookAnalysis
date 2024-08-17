#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:lightgray; font-family:verdana; font-size:250%; text-align:center; border-radius: 15px 20px;">🟠Feature Engineering🟠</p>

# ### **The main purpose of feature engineering is to manipulate existing features or create new ones in order to improve the performance of a machine learning model or better represent information in the dataset. When data that has undergone a good feature engineering process is fed into machine learning models, it facilitates the model in making more effective and accurate predictions.**
# 
# ### **Feature engineering is one of the most critical steps in a data science or machine learning project. It can help prevent the model from overfitting to the training data. Unnecessary or excessive features may lead the model to learn random noise in the dataset, making it less adaptable to new data. Removing unnecessary complex features results in a faster-performing model.**
# 
# ### **In conclusion, feature engineering provides significant advantages such as better representation of the dataset, improved model performance, and prevention of overfitting. Therefore, it is crucial to carry out this step meticulously to achieve successful results in a data science or machine learning project.**

# <center><span style="color:#f7c297;font-family:cursive;font-size:100%"> </span></center>
# <center><img src="https://i.imgur.com/6tJmccK.png" width="800" height="800"></center>

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


# # <p style="border-radius:10px; border:#DEB887 solid; padding:25px; background-color: #FFFAF0; font-size:100%;color:#52017A;text-align:center;"> Outliers</p>

# ### Outliers are values that significantly deviate from the general trend in a dataset. Dealing with outliers can involve visual inspections 
# 
# > such as using boxplots
# 
# ### and statistical methods
# 
# > for examplecalculating the Z-score or applying the IQR method. 
# 
# ### Outliers can be addressed by capping them suppressing them or removing them from the dataset. The fewer outliers there are, the more balanced the dataset becomes.
# 
# ### For instance, in a dataset with hundreds of outliers, if we choose to suppress them, it may alter the course of the dataset. This action could potentially introduce duplicate records, leading to serious issues later on. When working with tree-based methods, it's often advisable not to tamper with outliers. If there are only a few outliers, they can be removed, although this decision is subjective and depends on the context.

# In[93]:


dff = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
dff.head()


# In[3]:


df= pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> Catching the outliers</font></h3>

# In[4]:


#as we can see there is outliers after  65 
sns.boxplot(x=df["Age"])
plt.show()


# In[5]:


# 25th percentile of the 'Age' variable
q1 = df["Age"].quantile(0.25)
# 75th percentile of the 'Age' variable
q3 = df["Age"].quantile(0.75)
# Calculate the interquartile range (IQR) by subtracting q1 from q3
iqr = q3 - q1
# Calculate the upper limit by multiplying 1.5 with IQR and adding it to q3
up = q3 + 1.5 * iqr
# Calculate the lower limit by multiplying 1.5 with IQR and subtracting it from q1
low = q1 - 1.5 * iqr


# In[14]:


# Rows where the 'Age' variable is less than the lower limit (low) or greater than the upper limit (up)
outliers_df = df[(df["Age"] < low) | (df["Age"] > up)]
# Index values of the rows with 'Age' outliers
outliers_index = df[(df["Age"] < low) | (df["Age"] > up)].index
print(outliers_index)
print("--------------------------------------------------------------")
print(outliers_df.head())


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>Are There Outliers or Not?</font></h3>

# In[15]:


# We asked if there are any outliers, and it returned True
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)


# In[16]:


# We asked about the lower values, and it returned False
df[(df["Age"] < low)].any(axis=None)


# > 1- We set a threshold value.
# 
# > 2- We accessed the outliers.
# 
# > 3- Quickly asked whether there are any outliers or not.

# In[18]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

print(outlier_thresholds(df, "Age"))
print(outlier_thresholds(df, "Fare"))


# In[21]:


#so we get outliers like this.
df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low) | (df["Fare"] > up)].index


# In[26]:


#this function answers the question of is there any outlier or not?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# In[24]:


#columns ayırma fonksiyonu
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


#891 observations, 12 variables, 6 categorical, 3 numeric, 3 categorical with high cardinality, and 4 numeric but treated as categorical
#(for observation purposes, hence printed).
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# In[25]:


# 'PassengerId' is our exception, and date variable could be treated similarly, so we exclude 'PassengerId'
num_cols = [col for col in num_cols if col not in ["PassengerId"]]

# We check for outliers, and it says there are outliers in 'Age' and 'Fare'
for col in num_cols:
    print(col, check_outlier(df, col))


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>Accessing Outlier Values</font></h3>

# In[28]:


def grab_outliers(dataframe, col_name, index=False):  # We will input the dataframe name, column name, and variable name, default is False
    low, up = outlier_thresholds(dataframe, col_name)   # Capture low and up values
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:   # If there are more than 10 outliers
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())   # Print 5 of them
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])          # If there are 10 or fewer, print all
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index  
        # If the index argument is True, print the indices
        return outlier_index


# There are more than 10 outliers in the 'Age' variable, so it returns 5 of them
grab_outliers(df, "Age")


# In[29]:


# It returns only the indices
grab_outliers(df, "Age", True)


# In[30]:


# If we want to store it for later use, we can do it like this
age_index = grab_outliers(df, "Age", True)


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>
# Addressing Outlier Issue</font></h3>
# 
#  <h3 align="left"><font color='#5E5273'>Deletion</font></h3>

# In[31]:


# Taking those other than below the lower limit and above the upper limit
low, up = outlier_thresholds(df, "Fare")
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


# In[32]:


# Removing those below the lower limit and above the upper limit and assigning it to df
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# Capturing the columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Excluding 'passengerId'
num_cols = [col for col in num_cols if col not in "PassengerId"]


# In[33]:


# Looping through, and removing all outliers
for col in num_cols:
    new_df = remove_outlier(df, col)
df.shape[0]  -  new_df.shape[0]
# 891 values in the original df, 775 values in the new df


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> Capping Method (re-assignment with thresholds)</font></h3>

# In[36]:


# Replacing the variable that is below the lower limit with the lower limit, and the one above the upper limit with the upper limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
# Asking if there are outliers
for col in num_cols:
    print(col, check_outlier(df, col))


# In[37]:


# Replacing the outliers
for col in num_cols:
    replace_with_thresholds(df, col)
# Asking again
for col in num_cols:
    print(col, check_outlier(df, col))


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>  Local Outlier Factor</font></h3>

# ### Let's say we have a variable representing marital status, having been married three times is not an outlier And there is an age variable, being 17 years old is normal But being 17 years old and having been married three times is abnormal
# 
# >  Local Outlier Factor provides a chance to calculate the distance based on neighborhoods, the farther the given observations are from 1, the more likely they are outliers
# 

# In[38]:


df = sns.load_dataset('diamonds')  #diamonds data set
df = df.select_dtypes(include=['float64', 'int64'])    #taking only numeric
df = df.dropna()  #droping nan's
df.head()


# In[40]:


# Checking for outliers in all columns
for col in df.columns:
    print(col, check_outlier(df, col))

# 1889 outliers
low, up = outlier_thresholds(df, "carat")
print(df[((df["carat"] < low) | (df["carat"] > up))].shape)

# 2545 outliers
low, up = outlier_thresholds(df, "depth")
print(df[((df["depth"] < low) | (df["depth"] > up))].shape)


# In[41]:


# When we look individually, a very high number of outliers come up
# Setting the number of neighbors to 20
clf = LocalOutlierFactor(n_neighbors=20)
# Fitting and predicting
clf.fit_predict(df)

# Keeping the scores
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

# We can determine according to the elbow method
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()


# In[42]:


# Setting the threshold value to 3
th = np.sort(df_scores)[3]

# Defining values smaller than the threshold as outliers
df[df_scores < th]
# There are 3 observations
df[df_scores < th].shape

# There are 3 observations, why did it flag as outliers?
# 1st observation may be flagged due to having a depth of 78 and a low price
# 2nd observation has a z value of 31.800 but the mean is 3.5
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T


# In[43]:


# Capturing the index information
df[df_scores < th].index
# We can delete them if we want
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


# ### While capping can be applied, what will be used for capping? 
# 
# > In this scenario, there are only 3 observations, but if there are hundreds of outliers, capping can significantly alter the trajectory of the dataset and lead to duplicate records. This can cause serious problems. If working with tree-based methods, it's advisable not to touch them. Perhaps, for outlier thresholds, a slight trimming can be applied. Deleting can be done based on the minority or majority situation, but it is subjective.
# 

# # <p style="border-radius:10px; border:#DEB887 solid; padding:25px; background-color: #FFFAF0; font-size:100%;color:#52017A;text-align:center;"> Missing Values </p>

# ### It indicates the presence of missing values in observations.
# ### How can missing values be addressed? 
# 
# > Deletion 
# 
# >Imputation 
# 
# >Prediction-based methods

# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>  Catching the missing values</font></h3>

# In[44]:


df= pd.read_csv("/kaggle/input/titanic/train.csv")

# Query to check if there are any missing observations
df.isnull().values.any()


# In[45]:


# Number of missing values in each variable
df.isnull().sum()


# In[46]:


# Observations with at least one missing value
df[df.isnull().any(axis=1)]


# In[47]:


# Observations without any missing values
df[df.notnull().all(axis=1)]


# In[48]:


# Sorting in descending order
df.isnull().sum().sort_values(ascending=False)


# In[49]:


# Calculating the percentage of missing values in each variable
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)


# In[50]:


# Assigning features with missing values to na_cols, and when we print na_cols, we see the names of features with missing values
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# In[51]:


def missing_values_table(dataframe, na_name=False):  # If set to True, it provides the names of variables with missing values
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]   # Capture columns with missing values
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)   # Number of missing values
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)      # Ratio of missing values
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])       # Concatenate
    print(missing_df, end="\n")  # Print the DataFrame
    if na_name:
        return na_columns

missing_values_table(df, True)


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> Solving the Missing Value Issue</font></h3>

# In[52]:


# Solution 1: Quick Deletion
# After dropping the missing values with dropna(), we can assign it
# However, when dropping, if there is any 'nan' anywhere in an observation, we drop the entire observation, so a lot of data can be lost
df.dropna().shape


# In[53]:


# Solution 2: Filling with Simple Assignment Methods
###################
# With fillna, we can fill missing values with mean, median, or 0
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()


# In[62]:


# If we have a dataset with many missing variables
# Here, we say fill only the ones that are different from 'O' because they are numerical variables
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "object" else x, axis=0).head()


# In[63]:


# Filling with assignment
df4 = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
# Sorting from large to small
df4.isnull().sum().sort_values(ascending=False)


# In[64]:


# The most logical filling for categorical variables is to take the mode
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()


# In[65]:


# Or we can fill with a specific expression, such as 'missing'
df["Embarked"].fillna("missing")


# In[66]:


# Our condition is, if it is a categorical variable and at the same time the number of unique values is less than or equal to 10, 
#fill with mode, otherwise (else) leave it as it is
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


# In[72]:


#Imputation in the Breakdown of Categorical Variables
###################
# Taking some variables in the dataset as breakdown and making assignments accordingly, for example, the average age of females is 27, males 30
df.groupby("Sex")["Age"].mean()


# In[68]:


# Average age is 29
df["Age"].mean()
# Instead of assigning the value of 29 to all missing values, when there is a missing value in females, we can assign 27, and for males, 
#we can assign 30 to the missing ones


# In[69]:


# Filling the missing values in the age variable with the averages coming from the breakdown by gender, i.e., the average of females to females and the average of males to males
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))


# In[70]:


# Doing the same thing without fillna, assigning the average of females to females
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]


# In[71]:


# Assigning the average of males to males
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


# In[73]:


# Solution 3: Imputation with Prediction-Based Assignment
#############################################
# Capturing columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# It took passenger id numerically, we drop it
num_cols = [col for col in num_cols if col not in "PassengerId"]


# In[74]:


# We separate the variables with the get_dummies method
df2 = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# Trying to write two-class or more-class variables numerically
df2.head()


# In[76]:


# Standardization of variables
# We say bring the values ​​between 0-1
scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
df2.head()


# In[77]:


# Applying knn
from sklearn.impute import KNNImputer
# Filling missing values with knn, prediction-based
imputer = KNNImputer(n_neighbors=5)
df2 = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)
# We standardized it, but since we minmax, it becomes between 0-1
# In this part, we take the minmax back
df2 = pd.DataFrame(scaler.inverse_transform(df2), columns=df2.columns)


# In[78]:


# We assign the first df to the second df
df["age_imputed_knn"] = df2[["Age"]]
# We look at where we assigned what
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> Examining the Structure of Missing Data</font></h3>

# In[79]:


# With the missingno library, we can see missing data
msno.bar(df)
plt.show()
# We can also look with the matrix method, if the missingness in the variables comes in a range, we can observe it
msno.matrix(df)
plt.show()
# The heat map is built on missingness, have the missing values ​​come out with a certain correlation?
msno.heatmap(df)
plt.show()


# In[80]:


##################
# Examining the Relationship of Missing Values with the Dependent Variable
###################
# Does the missingness in my dataset have a counterpart on the dependent variable,
missing_values_table(df, True)
na_cols = missing_values_table(df, True)
# We pulled the variables with missing values

def missing_vs_target(dataframe, target, na_columns):  # df, target, and empty columns
    temp_df = dataframe.copy()  # we open a temporary df and give a copy of the df
    for col in na_columns:  # we say FLAG to the points with na
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)  # we wrote 1 to points with missing value, 0 to empty ones
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns  # select and do in the temp df, select all columns but bring those with the expression NA
    for col in na_flags:  # loop through these variables
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")  # take the mean of the target and the mean, count of the variable

missing_vs_target(df, "Survived", na_cols)


# # <p style="border-radius:10px; border:#DEB887 solid; padding:25px; background-color: #FFFAF0; font-size:100%;color:#52017A;text-align:center;"> Encoding (Label Encoding, One-Hot Encoding, Rare Encoding) </p>

# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> Label Encoding</font></h3>

# ### Encoding: Changing the representation of variables
# 
# ### Label Encoding & Binary Encoding
# 
# > For example, changing a variable that was previously male/female to 0/1or let's say there is an ordinal education variable, ordered from 0 to 5, we can do this by making the variable numerical manually. It can also be done with one-hot encoder.

# In[81]:


# Label encoder is defined
le = LabelEncoder()
# We fit and transform the label encoder
le.fit_transform(df["Sex"])[0:5]
# Here we reverse it
le.inverse_transform([0, 1])

# We can do it quickly with a function
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# In[82]:


# When we make it unique, it brings the missing value, nunique does not see the missing value as a class
# We capture values ​​with binary or only 2 different variables
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
# We pass all of them from the label encoder with the function in the for loop
for col in binary_cols:
    label_encoder(df, col)


# In[83]:


# If there is a missing value, for example, there are 3 classes in the label encoded, female 0, male 1, NAN 2.

# There are 3 classes, but when we take len, 4 classes appear
print(df["Embarked"].value_counts())
print(df["Embarked"].nunique())
print(len(df["Embarked"].unique()))


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'> One-Hot Encoding</font></h3>

# ###  In variables where there is no difference between classes, such as teams (real madrid, barca, bjk), if we assign 0-1-2 to them, 2 will be the highest, so we convert all classes into variables.

# In[84]:


df["Embarked"].value_counts()  # There are 3 classes
# By using drop_first, we escape from the dummy variable trap, i.e., the first class is dropped.
# So, it drops Real Madrid, leaving only Barca and BJK. If there is no score for Barca and BJK in the dataset, the model understands 
#that the score belongs to Real Madrid.


# In[85]:


# Convert the variable into dummy variables with the get_dummies method.
pd.get_dummies(df, columns=["Embarked"]).head()


# In[86]:


# Drop the first observation with drop_first to avoid the dummy variable trap.
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()


# In[87]:


# If we want the missing variables in the relevant variable to come as a class, it creates a class for missing values if we specify dummy_na as True.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()


# In[88]:


# The gender variable was binary (2 classes), but here only male comes, Is it male? (1 and 0), and similarly, we added embarked from the side, and it also encoded it.
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

# Function application for the ohe_cols we have defined, where unique value is more than 2 and less than 10.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# In[89]:


# Apply the function to ohe_cols.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>Rare Encoding</font></h3>

# > 1. Analyzing the rarity and commonality of categorical variables.
# 
# > 2. Analyzing the relationship between rare categories and the dependent variable.
# 
# > 3. Writing a Rare Encoder.

# In[95]:


# 1. Analyzing the rarity and commonality of categorical variables.
# Run the large dataset as the second dataset above.
# Capture the columns.
cat_cols, num_cols, cat_but_car = grab_col_names(dff)


# In[96]:


def cat_summary(dataframe, col_name, plot=False):                                          
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))       
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)                                      
        plt.show()

for col in cat_cols:
    cat_summary(dff, col)


# In[98]:


# 2- Analyzing the relationship between rare categories and the dependent variable.
# There were rare categories in this variable with 5-10 values.
dff["NAME_INCOME_TYPE"].value_counts()


# In[99]:


# Check the relationship between rare categories and the target.
dff.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


# In[100]:


# A function to perform rare analysis for all categorical variables.
def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyzer(dff, "TARGET", cat_cols)


# In[102]:


# 3- Writing the Rare encoder.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()  # Make a copy to avoid modifying the original dataframe.
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]  # If there is any class with a frequency lower than the rare percentage, consider it as rare columns.
    for var in rare_columns:  # Iterate through rare columns.
        tmp = temp_df[var].value_counts() / len(temp_df)  # Calculate class ratios.
        rare_labels = tmp[tmp < rare_perc].index  # Select labels with ratios lower than the rare percentage.
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])  # Replace the rare labels with 'Rare'.
    return temp_df

# Apply rare encoder to create a new dataframe.
new_df = rare_encoder(dff, 0.01)
# Analyze the relationship with the target variable.
rare_analyzer(new_df, "TARGET", cat_cols)


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>Feature Scaling</font></h3>

# In[103]:


# Resolving the measurement differences between variables to approach under equal conditions
###################
# StandardScaler: Classical standardization. Subtract the mean, divide by the standard deviation. z = (x - u) / s
###################
# Titanic dataset
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


# In[104]:


# RobustScaler: Subtract the median, divide by the interquartile range (IQR).
###################
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T


# In[105]:


# MinMaxScaler: Transform features by scaling each feature to a given range.
###################
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T


# In[106]:


#getting columsn with age 
age_cols = [col for col in df.columns if "Age" in col]
#gettin graph
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)


# In[107]:


#Numeric to Categorical: Converting Numerical Variables to Categorical Variables
# Binning
###################
df["Age_qcut"] = pd.qcut(df['Age'], 5)


# # <p style="border-radius:10px; border:#DEB887 solid; padding:25px; background-color: #FFFAF0; font-size:100%;color:#52017A;text-align:center;"> Feature Extraction</p>

# In[108]:


#Binary Features: Flag, Bool, True-False

#Importing the Titanic dataset
df= pd.read_csv("/kaggle/input/titanic/train.csv")
#Filling missing values with 0 for non-null values and 1 for null values in the "Cabin" column
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

#Analyzing the survival rate based on the newly created variable
#It appears that the survival rate is higher for passengers with missing cabin numbers.
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


# In[109]:


#Testing the significance of the distribution of the newly created feature
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#The p-value is 0, which rejects the null hypothesis, suggesting a significant difference between the distributions.


# In[110]:


#Creating a new feature by summing siblings and parents, then binary encoding for being alone or not
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

#Checking survival rates based on the new feature
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


# In[111]:


#Testing for a statistically significant difference
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0], df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Rejecting the null hypothesis indicates a significant difference.


# <div style="border-radius:10px; border:#D0C2F0 solid; padding: 15px; background-color: #FFF0F4; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#5E5273'>Feature Engineering through Text Analysis</font></h3>

# In[112]:


#Letter Count

#Counting the number of letters in a name
df["NEW_NAME_COUNT"] = df["Name"].str.len()


# In[113]:


#Word Count

#Counting the number of words in a name / convert to strings, split by spaces, then count the number of words using len
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))


# In[114]:


#Identifying Special Structures

#Trying to identify names with "Dr." / split and check if "Dr" is present in each name
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))


# In[115]:


#Grouping by the presence of "Dr" and checking the average survival rate, there are 10 doctors but the survival rate is high for them.
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})


# In[116]:


#Deriving Variables with Regex
###################

#Titles may be important for us, so we extract those titles with regex by providing the pattern including spaces, dots, lowercase, and 
#uppercase letters, etc.
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+).', expand=False)

#Grouping by the new title according to the three categories below and checking the mean survival rate for survived and age.
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})


# In[117]:


df6 = pd.read_csv("/kaggle/input/course-reviewsdataset/course_reviews.csv")
df6.head()


# In[121]:


# Converting the 'Timestamp' variable to datetime
df6['Timestamp'] = pd.to_datetime(df6["Timestamp"], format="%Y-%m-%d %H:%M:%S")


# In[122]:


# Extracting date components
# year
df6['year'] = df6['Timestamp'].dt.year


# In[123]:


# month
df6['month'] = df6['Timestamp'].dt.month


# In[124]:


# year difference (difference between the current year and the year in the timestamp)
df6['year_diff'] = date.today().year - df6['Timestamp'].dt.year


# In[125]:


# month difference (the difference in months between two dates): year difference + month difference
df6['month_diff'] = (date.today().year - df6['Timestamp'].dt.year) * 12 + date.today().month - df6['Timestamp'].dt.month


# In[126]:


# day name
df6['day_name'] = df6['Timestamp'].dt.day_name()


# In[127]:


df6.head()


# In[129]:


df= pd.read_csv("/kaggle/input/titanic/train.csv")
# Multiplying age by pclass to represent a combination of age and class as an indicator of wealth.
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]


# In[130]:


# Calculating the total number of family members on board, including the person themselves.
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1


# In[132]:


# Categorizing males based on age: young males (age <= 21), mature males (21 < age < 50), and senior males (age >= 50).
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'


# In[133]:


#Categorizing females based on age: young females (age <= 21), mature females (21 < age < 50), and senior females (age >= 50).
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


# In[134]:


# Analyzing the average survival rate based on the new sex categories.
# When we look at the average survival rate by sex category, mature females have a high probability of survival.
df.groupby("NEW_SEX_CAT")["Survived"].mean()

