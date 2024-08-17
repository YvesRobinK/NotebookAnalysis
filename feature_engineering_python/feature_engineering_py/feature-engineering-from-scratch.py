#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://www.vidora.com/wp-content/uploads/2018/06/Screen-Shot-2018-06-22-at-2.53.26-PM.png"></center>

# <center><h1 style="font-size:280%; font-family:cursive; background:#ff6666; color:black; border-radius:10px 10px; padding:10px;">Introduction</h1></center>
# 
# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;"><b>What is Feature Engineering ?</b></h1>
# <p style="font-size:150%; font-family:cursive;">Feature engineering is the process of transforming raw data to provide your algorithms with the most predictive inputs possible. Before seeing data, an algorithm knows nothing of the problem at hand. Humans, though, can inject prior knowledge into the equation. If, for example, we know from experience that changes in user activity are more predictive than raw activity itself, we can introduce this information. Given the infinite number of potential features, it’s often not computationally feasible for even the most sophisticated algorithms to do this on their own.</p>

# <center><h1 style="font-size:280%; font-family:cursive; background:#ff6666; color:black; border-radius:10px 10px; padding:10px;">Table of Contents</h1></center>
# 
# <ol>
#     <li style="font-size:200%; font-family:cursive;">Handling Missing Values</li>
#     <br>
#     <ul>
#            <li style="font-size:180%; font-family:cursive;">Types of Missing Values:</li>
#            <ol>
#                <li style="font-size:150%; font-family:cursive;">Missing Completely at Random</li>
#                <li style="font-size:150%; font-family:cursive;">Missing at Random</li>
#                <li style="font-size:150%; font-family:cursive;">Missing Not at Random</li>
#            </ol>
#            <li style="font-size:180%; font-family:cursive;">Techniuqes to Handle Missing Values:</li>
#            <ol>
#                <li style="font-size:150%; font-family:cursive;">Mean / Median Imputation</li>
#                <li style="font-size:150%; font-family:cursive;">Random Sample Imputation</li>
#                <li style="font-size:150%; font-family:cursive;">Capturing NaN with new feature</li>
#                <li style="font-size:150%; font-family:cursive;">End of Distribution</li>
#                <li style="font-size:150%; font-family:cursive;">Arbitrary Imputation</li>
#                <li style="font-size:150%; font-family:cursive;">Frequency Category Imputation(Mode)</li>
#                <li style="font-size:150%; font-family:cursive;">Regression Imputation</li>
#                <li style="font-size:150%; font-family:cursive;">KNN Imputation</li>
#            </ol>
#     </ul>
#     <br>
#     <li style="font-size:200%; font-family:cursive;">Outliers</li>
#     <li style="font-size:200%; font-family:cursive;">Handling Categorical Variables</li>
#     <br>
#     <ol>
#         <li style="font-size:150%; font-family:cursive;">Ordinal Encoding</li>
#         <li style="font-size:150%; font-family:cursive;">One Hot Encoding</li>
#         <li style="font-size:150%; font-family:cursive;">Count / Frequency Encoding</li>
#         <li style="font-size:150%; font-family:cursive;">Target Guided Ordinal Encoding</li>
#         <li style="font-size:150%; font-family:cursive;">Mean Encoding</li>
#     </ol>
#     <br>
#     <li style="font-size:200%; font-family:cursive;">Feature Transformation Techniques</li>
#     <li style="font-size:200%; font-family:cursive;">Feature Selection Techniques</li>
# </ol>

# ----

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">1. HANDLING MISSING VALUES</h1></center>

# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>1.1 Types of Missing Values</b></h1></center>

# <p style="font-size:200%; font-family:cursive;">There are 3 major types of missing values to be concerned about.</p>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Missing Completely at Random</li>
#   <li style="font-size:150%; font-family:cursive;">Missing at Random</li>
#   <li style="font-size:150%; font-family:cursive;">Missing Not at Random</li>
# </ul>

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.1.1 MISSING COMPLETELY AT RANDOM</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">MCAR occurs when the probability of missing values in a variable is the same for all samples.</li>
#   <li style="font-size:150%; font-family:cursive;">For example, when a survey is conducted, and values were just randomly missed when being entered in the computer or a respondent chose not to respond to a question.</li>
# </ul>
# 
# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.1.2 MISSING AT RANDOM</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">The probability of missing values, at random, in a variable depends only on the available information in other predictors.</li>
#   <li style="font-size:150%; font-family:cursive;">For example, when men and women respond to the question “have you ever taken parental leave?”, men would tend to ignore the question at a different rate compared to women.</li>
#   <li style="font-size:150%; font-family:cursive;">MARs are handled by using the information in the other predictors to build a model and impute a value for the missing entry.</li>
# </ul>
# 
# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.1.3 MISSING NOT AT RANDOM</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">The probability of missing values, not at random, depends on information that has not been recorded, and this information also predicts the missing values.</li>
#   <li style="font-size:150%; font-family:cursive;">For example, in a survey, cheaters are less likely to respond when asked if they have ever cheated.</li>
#   <li style="font-size:150%; font-family:cursive;">MNARs are almost impossible to handle.</li>
#   <li style="font-size:150%; font-family:cursive;">Luckily there shouldn’t be any effect of MNAR on inferences made by a model trained on such data.</li>  
# </ul>
# 
# 
# 
# 
# 

# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>1.2 TECHNIQUES TO HANDLE MISSING VALUES</b></h1></center>
# <br>
# <center><h1 style="font-size:180%; font-family:cursive; color:black;">[ IMPLEMENTATION OF MISSING VALUES USING PYTHON ]</h1></center>

# In[1]:


#import the libraries:

import pandas as pd
import numpy as np


# In[2]:


#read the dataset:

df = pd.read_csv('../input/titanic/train.csv')


# In[3]:


#check for missing values:

df.isnull().sum()


# ----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.1 Mean / Median Imputation</h1>

# <p style="font-size:180%; font-family:cursive;"><b>When to use mean/median imputation?</b></p>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Data is missing completely at random.</li>
#   <li style="font-size:150%; font-family:cursive;"> No more than 5% of the variable contains missing data.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>Assumptions</b></p>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Data is missing completely at random (MCAR)</li>
#   <li style="font-size:150%; font-family:cursive;">The missing observations, most likely look like the majority of the observations in the variable (aka, the mean/median)</li>
#   <li style="font-size:150%; font-family:cursive;">If data is missing completely at random, then it is fair to assume that the missing values are most likely very close to the value of the mean or the median of the distribution, as these represent the most frequent/average observation.</li>
# </ul>

# In[4]:


#select only columns which have missing values:
df=pd.read_csv('../input/titanic/train.csv',usecols=['Age','Fare','Survived'])
df.head()


# In[5]:


## Lets get the percentage of missing values
df.isnull().mean()


# In[6]:


# small function to impute the missing values with median of the col:
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)


# In[7]:


#calculate the median of the Age feature:
median=df.Age.median()

#call the function:
impute_nan(df,'Age',median)


# In[8]:


#check whether the dataframe is updated or not:
df.head()


# In[9]:


#VIZUALIZATION OF THE CHANGE:
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">Fast way of obtaining complete datasets.</li>
#   <li style="font-size:150%; font-family:cursive;">Can be integrated into production (during model deployment)..</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Distortion of the original variable distribution.</li>
#   <li style="font-size:150%; font-family:cursive;">Distortion of the original variance.</li>
# </ul>

# ----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.1 Random Sample Imputation</h1>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Random sample imputation consists of taking random observation from the dataset and we use this observation to replace the nan values</li>
# </ul>
# <p style="font-size:180%; font-family:cursive;"><b>Assumptions</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">It assumes that the data are missing completely at random(MCAR)</li>
# </ul>
# 

# In[10]:


import pandas as pd
df=pd.read_csv('../input/titanic/train.csv', usecols=['Age','Fare','Survived'])
df.head()


# In[11]:


#Function to impute misisng values using RANDOM SAMPLE IMPUTATION:

def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample
    
    


# In[12]:


#median calculation:
median=df.Age.median()

#Call the function:

impute_nan(df, "Age", median)


# In[13]:


#check for the updation:

df.head()


# In[14]:


#Visualization:
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
df.Age_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">There is less distortion in variance.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">In every situation randomness wont work.</li>
# </ul>

# -----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.3 Capturing NaN with New Features</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">This technique suits well, when the data is not missing at random (MNAR). We will be capturing Nan values and further it will be replaced with our new features which we will compute on desired column or values.</li>
# </ul>

# In[15]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Age','Fare','Survived'])


# In[16]:


df['Age_NaN'] = np.where(df['Age'].isnull(), 1, 0)
df.head()


# <ul>
#   <li style="font-size:150%; font-family:cursive;">Here simply we are creating new column named as “Age_NAN” where we are taking the null values of the column Age and replacing it with 1 if the value is NaN and with 0 if it’s not Nan value. The where () function is a very handy function using which you can use the functionality of where clause from SQL.</li>
# </ul>

# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">Able to capture the importance of missing values</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Creates additional features causing Curse of Dimensionality (CoD). This might happen in the case where you have huge amount of data. This means that this technique will work perfectly where you have sparse data.</li>
# </ul>

# -------------

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.4 End of Distribution</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">This method is a bit tricky as it’s not that straightforward as it sounds. Talking about the basic definition then in this technique we will take values which are far away from the distribution aka end of the distribution.Lets this see using visualization.</li>
# </ul>

# In[17]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Age','Fare','Survived'])


# In[18]:


df.Age.hist(bins=50)


# <ul>
#   <li style="font-size:150%; font-family:cursive;">Here, these values(70-80) are the one which are at the end of this distribution. This histogram is of Age column as we are going to do computations on this column. So we are going to replace our Nan values with these values.</li>
# </ul>

# In[19]:


extreme=df.Age.mean()+3*df.Age.std()


# In[20]:


import seaborn as sns
sns.boxplot('Age',data=df)


# <ul>
#   <li style="font-size:150%; font-family:cursive;">The very first cell in the figure given above is the calculated standard deviation where we took the values which lies far end of distribution after 3rd standard deviation. </li>
# </ul>

# In[21]:


#Here we are creating our function with passing parameters of df(data frame), variable (in our case Age is our variable), median(Calculated median)
#extreme(our standard deviation value)

def impute_nan(df,variable,median,extreme):
    df[variable+"_end_distribution"]=df[variable].fillna(extreme)
    df[variable].fillna(median,inplace=True)


# In[22]:


impute_nan(df,'Age',df.Age.median(),extreme)


# In[23]:


df.head()


# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">Able to capture the importance of missing values</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">It will masks the Outliers because of the distribution.</li>
# </ul>

# ----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.5 Arbitrary Imputation</h1>
# 
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Arbitrary value imputation consists of replacing all occurrences of missing values (NA) within a variable with an arbitrary value. The arbitrary value should be different from the mean or median and not within the normal values of the variable.</li>
#   <li style="font-size:150%; font-family:cursive;">This technique was derived from kaggle competition.</li>
#   <li style="font-size:150%; font-family:cursive;">We can use arbitrary values such as 0, 999, -999 (or other combinations of 9s) or -1 (if the distribution is positive).</li>
#   <li style="font-size:150%; font-family:cursive;">This method is suitable for numerical and categorical variables.</li>
# </ul>
# <br>
# <p style="font-size:180%; font-family:cursive;"><b>Assumptions</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Data is not missing at random.</li>
# </ul>

# In[24]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Age','Fare','Survived'])


# In[25]:


def impute_nan(df,variable):
    df[variable+'_zero']=df[variable].fillna(0)
    df[variable+'_hundred']=df[variable].fillna(100)


# In[26]:


impute_nan(df, "Age")


# In[27]:


df.head()


# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">It’s a fast way to obtain complete datasets.</li>
#   <li style="font-size:150%; font-family:cursive;">It can be used in production, i.e during model deployment.</li>
#   <li style="font-size:150%; font-family:cursive;">It captures the importance of a value being “missing”, if there is one.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Distortion of the original variable distribution and variance.</li>
#   <li style="font-size:150%; font-family:cursive;">Distortion of the covariance with the remaining dataset variables.</li>
#   <li style="font-size:150%; font-family:cursive;">If the arbitrary value is at the end of the distribution, it may mask or create outliers.</li>
#   <li style="font-size:150%; font-family:cursive;">We need to be careful not to choose an arbitrary value too similar to the mean or median (or any other typical value of the variable distribution).</li>
# </ul>

# ----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.6 Frequency Count Imputation (Mode)</h1>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Frequent category imputation—or mode imputation—consists of replacing all occurrences of missing values (NA) within a variable with the mode, or the most frequent value.</li>
#   <li style="font-size:150%; font-family:cursive;">This method is suitable for numerical and categorical variables, but in practice, we use this technique with categorical variables.</li>
#   <li style="font-size:150%; font-family:cursive;">You can use this method when data is missing completely at random, and no more than 5% of the variable contains missing data.</li>
# </ul>
# <br>
# <p style="font-size:180%; font-family:cursive;"><b>Assumptions</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Data is missing at random.</li>
#   <li style="font-size:150%; font-family:cursive;">The missing observations most likely look like the majority of the observations (i.e. the mode).</li>
# </ul>

# In[28]:


df=pd.read_csv('../input/big-mart-sales/train_v9rqX0R.csv', usecols=['Outlet_Size', 'Item_Outlet_Sales'])


# In[29]:


df.head()


# In[30]:


#Compute the frequency of each type:

df['Outlet_Size'].value_counts().plot.bar()


# In[31]:


def impute_nan(df,variable):
    most_frequent_category=df[variable].mode()[0]
    df[variable].fillna(most_frequent_category,inplace=True)


# In[32]:


impute_nan(df,'Outlet_Size')


# In[33]:


df.head()


# <p style="font-size:180%; font-family:cursive;"><b>ADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Easy to implement.</li>
#   <li style="font-size:150%; font-family:cursive;">It’s a fast way to obtain complete datasets.</li>
#   <li style="font-size:150%; font-family:cursive;">It can be used in production, i.e during model deployment.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">It distorts the relation of the most frequent label with other variables within the dataset.</li>
#   <li style="font-size:150%; font-family:cursive;">May lead to an over-representation of the most frequent label if there is are a lot of missing observations.</li>
# </ul>

# ----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.7 Regression Imputation</h1>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Mean, median or mode imputation only look at the distribution of the values of the variable with missing entries. If we know there is a correlation between the missing value and other variables, we can often get better guesses by regressing the missing variable on other variables.</li>
# </ul>

# In[34]:


#Sample Dataset to implement this:

import seaborn as sns

tips = sns.load_dataset('tips')

df = tips.loc[:, ['total_bill', 'size', 'tip']]

#introduce the nan values:

df.loc[0:20, 'size'] = np.nan
df.loc[220:230, 'total_bill'] = np.nan


# In[35]:


corr = df.corr()
corr.style.background_gradient(cmap = 'coolwarm').set_precision(2)


# <ul>
#   <li style="font-size:150%; font-family:cursive;">As we can see, in our example data, tip and total_bill have the highest correlation. Thus, we can use a simple linear model regressing total_bill on tip to fill the missing values in total_bill.</li>
# </ul>

# In[36]:


#create a subset of data where there are no missing values in the features

df_bill_tip = df.dropna(axis=0, subset = ['total_bill', 'tip'])
df_bill_tip = df_bill_tip.loc[:, ['total_bill', 'tip']]

#find the entries with total_bill_missing
missing_bill = df['total_bill'].isnull()

#extract the tips of observations with total_bill_missing
tip_misbill = pd.DataFrame(df['tip'][missing_bill])


# In[37]:


X = df_bill_tip[['tip']]
y = df_bill_tip['total_bill']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train, y_train)

bill_pred = lm.predict(tip_misbill)


# In[38]:


import matplotlib.pyplot as plt

plt.scatter(tip_misbill, tips['total_bill'][missing_bill], color='gray')
plt.plot(tip_misbill, bill_pred, color='royalblue', linewidth=2)
plt.xlabel("tip")
plt.ylabel("total_bill")
plt.show()


# <ul>
#   <li style="font-size:150%; font-family:cursive;">As we can see, the imputed total_bill from a simple linear model from tips does not exactly recover the truth but capture the general trend (and is better a single value imputation such as mean imputation). We can, of course, use more variables in the regression model to get better imputation.</li>
# </ul>

# -----

# <h1 style="font-size:180%; font-family:cursive; color:#ff6666;">1.2.8 KNN IMPUTER</h1>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Besides model-based imputation like regression imputation, neighbour-based imputation can also be used. K-nearest neighbour (KNN) imputation is an example of neighbour-based imputation. For a discrete variable, KNN imputer uses the most frequent value among the k nearest neighbours and, for a continuous variable, use the mean or mode.</li>
#   <li style="font-size:150%; font-family:cursive;">To use KNN for imputation, first, a KNN model is trained using complete data. For continuous data, commonly used distance metric include Euclidean, Mahapolnis, and Manhattan distance and, for discrete data, hamming distance is a frequent choice.</li>
# </ul>

# In[39]:


from sklearn.impute import KNNImputer
import numpy as np

X = [ [3, np.NaN, 5], [1, 0, 0], [3, 3, 3] ]
print("X: ", X)
print("===========")


imputer = KNNImputer(n_neighbors= 1)
impute_with_1 = imputer.fit_transform(X)

print("\nImpute with 1 Neighbour: \n", impute_with_1)



imputer = KNNImputer(n_neighbors= 2)
impute_with_2 = imputer.fit_transform(X)

print("\n Impute with 2 Neighbours: \n", impute_with_1)


# ----

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">2. Outliers</h1></center>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>What is an Outlier?</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">An outlier is a data point that diverges from an overall pattern in a sample.</li>
# </ul>
# 
# <p style="font-size:150%; font-family:cursive; color:#ff6666;"><b>Most common causes of outliers on a data set:</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Data entry errors (human errors)</li>
#   <li style="font-size:150%; font-family:cursive;">Measurement errors (instrument errors)</li>
#   <li style="font-size:150%;font-family:cursive;">Intentional (dummy outliers made to test detection methods)</li>
#   <li style="font-size:150%;font-family:cursive;">Data processing errors (data manipulation or data set unintended mutations)</li>
#   <li style="font-size:150%;font-family:cursive;">Sampling errors (extracting or mixing data from wrong or various sources)</li>
#   <li style="font-size:150%;font-family:cursive;">Natural (not an error, novelties in data)</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>Most common causes of outliers on a data set:</b></p>
# <ul>
#   <li style="font-size:150%;font-family:cursive;">Box Plot</li>
#   <li style="font-size:150%;font-family:cursive;">Scatter Plot</li>
#   <li style="font-size:150%;font-family:cursive;">Z-Score</li>
#   <li style="font-size:150%; font-family:cursive;">IQR Score</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>Methods to Handle Outliers</b></p>
# <ul>
#   <li style="font-size:150%;font-family:cursive;">Z-Score</li>
#   <li style="font-size:150%;font-family:cursive;">IQR Score</li>
# </ul>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;">Algorithms that are <b>NOT sentitive</b> to outliers</p>
# <ul>
#   <li style="font-size:150%;font-family:cursive;">Naive Bayes</li>
#   <li style="font-size:150%;font-family:cursive;">SVM</li>
#   <li style="font-size:150%;font-family:cursive;">Decision Trees</li>
#   <li style="font-size:150%;font-family:cursive;">Random Forest</li>
#   <li style="font-size:150%;font-family:cursive;">XGBoost, GBM</li>
#   <li style="font-size:150%;font-family:cursive;">KNN</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;">Algorithms that are <b>sentitive</b> to outliers</p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Linear Regression</li>
#   <li style="font-size:150%;font-family:cursive;">Logistic Regression</li>
#   <li style="font-size:150%;font-family:cursive;">K-Means Clustering</li>
#   <li style="font-size:150%;font-family:cursive;">Hierarchical Clustering</li>
#   <li style="font-size:150%;font-family:cursive;">PCA</li>
#   <li style="font-size:150%;font-family:cursive;">Neural Networks</li>
# </ul>

# ----

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">3. Handling Categorical Features</h1></center>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>What is Categorical Encoding?</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Categorical encoding is a process of converting categories to numbers.</li>
# </ul>

# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>3.1 Ordinal Encoding</b></h1></center>

# In[40]:


df = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High", "Low", "Medium","High", "Low"]})
print(df)


# In[41]:


scale_mapper = {"Low":1, "Medium":2, "High":3}
df["Scale"] = df["Score"].replace(scale_mapper)


# In[42]:


print(df)


# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>3.2 One Hot Encoding</b></h1></center>

# In[43]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Embarked'])


# In[44]:


df.head()


# In[45]:


df['Embarked'].unique()


# In[46]:


df.dropna(inplace=True)


# In[47]:


pd.get_dummies(df,drop_first=True).head()


# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>3.3 Count Or Frequency Encoding</b></h1></center>
# <br>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Replace the categories by the count of the observations that show that category in the dataset. Similarly, we can replace the category by the frequency -or percentage- of observations in the dataset. That is, if 10 of our 100 observations show the color blue, we would replace blue by 10 if doing count encoding, or by 0.1 if replacing by the frequency.</li>
# </ul>

# In[48]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Sex', 'Embarked', 'Cabin', 'Survived'])


# In[49]:


df['Cabin'] = df['Cabin'].fillna('Missing')
df['Cabin'] = df['Cabin'].str[0]


# In[50]:


df.head()


# In[51]:


# create the dictionary
count_map_sex = df['Sex'].value_counts().to_dict()
count_map_cabin = df['Cabin'].value_counts().to_dict()
count_map_embark = df['Embarked'].value_counts().to_dict()
# Map the column with dictionary
df['Sex'] = df['Sex'].map(count_map_sex)
df['Cabin'] = df['Cabin'].map(count_map_cabin)
df['Embarked'] = df['Embarked'].map(count_map_embark)
df.head()


# <p style="font-size:180%; font-family:cursive;"><b>DISADVANTAGES</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">If two different categories appear the same amount of times in the dataset, that is, they appear in the same number of observations, they will be replaced by the same number,hence, may lose valuable information.</li>
# </ul>

# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>3.4 Target Guided Ordinal Encoding</b></h1></center>

# In[52]:


import numpy as np
import pandas as pd

df=pd.read_csv('../input/titanic/train.csv', usecols=['Cabin','Survived'])


# In[53]:


df['Cabin'].fillna('Missing',inplace=True)
df['Cabin']=df['Cabin'].astype(str).str[0]


# In[54]:


df.head()


# In[55]:


df.Cabin.unique()


# In[56]:


df.groupby(['Cabin'])['Survived'].mean()


# In[57]:


ordinal_labels=df.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_labels


# In[58]:


enumerate(ordinal_labels,0)


# In[59]:


ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
ordinal_labels2


# In[60]:


df['Cabin_ordinal_labels']=df['Cabin'].map(ordinal_labels2)
df.head()


# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>3.5 Mean Encoding</b></h1></center>

# In[61]:


mean_ordinal=df.groupby(['Cabin'])['Survived'].mean().to_dict()


# In[62]:


df['mean_ordinal_encode']=df['Cabin'].map(mean_ordinal)
df.head()


# -----------

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">4. Feature Transformation Techniques</h1></center>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>What is Feature Transformations</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Feature transformation is the process of modifying your data but keeping the information. These modifications will make Machine Learning algorithms understanding easier, which will deliver better results.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>Types of Transformations</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Standard Scaler</li>
#   <li style="font-size:150%; font-family:cursive;">Min Max Scaler</li>
#   <li style="font-size:150%; font-family:cursive;">Robust Scaler</li>
#   <li style="font-size:150%; font-family:cursive;">logarithmic Transformation</li>
#   <li style="font-size:150%; font-family:cursive;">reciprocal Transformation</li>
#   <li style="font-size:150%; font-family:cursive;">square root Transformation</li>
#   <li style="font-size:150%; font-family:cursive;">exponential Transformation</li>
#   <li style="font-size:150%; font-family:cursive;">boxcox Transformation</li>
# </ul>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>1. Standard Scaler</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">For each feature, the Standard Scaler scales the values such that the mean is 0 and the standard deviation is 1(or the variance).</li>
#   <li style="font-size:150%; font-family:cursive;">FORMULA: x_scaled = x – mean/std_dev</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>2. Min-Max Scaler</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">The MinMax scaler is one of the simplest scalers to understand.  It just scales all the data between 0 and 1.</li>
#   <li style="font-size:150%; font-family:cursive;">FORMULA: x_scaled = (x – x_min)/(x_max – x_min)</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>3. Robust Scaler</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Removes the median from the data & Scales the data by the InterQuartile Range(IQR)</li>
#   <li style="font-size:150%; font-family:cursive;">FORMULA: IQR = Q3 – Q1</li>
# </ul>

# In[63]:


# Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
matplotlib.style.use('fivethirtyeight')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
  
# data
x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal
  
scaler = RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])
  
scaler = StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])
  
scaler = MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])
  
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))
ax1.set_title('Before Scaling')
  
sns.kdeplot(x['x1'], ax = ax1, color ='r')
sns.kdeplot(x['x2'], ax = ax1, color ='b')
ax2.set_title('After Robust Scaling')
  
sns.kdeplot(robust_df['x1'], ax = ax2, color ='red')
sns.kdeplot(robust_df['x2'], ax = ax2, color ='blue')
ax3.set_title('After Standard Scaling')
  
sns.kdeplot(standard_df['x1'], ax = ax3, color ='black')
sns.kdeplot(standard_df['x2'], ax = ax3, color ='g')
ax4.set_title('After Min-Max Scaling')
  
sns.kdeplot(minmax_df['x1'], ax = ax4, color ='black')
sns.kdeplot(minmax_df['x2'], ax = ax4, color ='g')
plt.show()


# -------------

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">5. Feature Selection Techniques</h1></center>

# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>What is Feature Selection</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Feature Selection is one of the core concepts in machine learning which hugely impacts the performance of your model. The data features that you use to train your machine learning models have a huge influence on the performance you can achieve.</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>Feature Selection Methods:</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Univariate Selection</li>
#   <li style="font-size:150%; font-family:cursive;">Feature Importance</li>
#   <li style="font-size:150%; font-family:cursive;">Correlation Matrix with Heatmap</li>
# </ul>
# 
# <p style="font-size:180%; font-family:cursive; color:#ff6666;"><b>Benefits of performing feature selection before modeling your data</b></p>
# <ul>
#   <li style="font-size:150%; font-family:cursive;"> Reduces Overfitting</li>
#   <li style="font-size:150%; font-family:cursive;">Improves Accuracy</li>
#   <li style="font-size:150%; font-family:cursive;">Reduces Training Time</li>
# </ul>

# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>5.1 Univariate Selection</b></h1></center>
# <ul>
#   <li style="font-size:150%; font-family:cursive;"> Statistical tests can be used to select those features that have the strongest relationship with the output variable.</li>
#   <li style="font-size:150%; font-family:cursive;">The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.</li>
# </ul>

# In[64]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[65]:


data = pd.read_csv('../input/mobile-price-classification/train.csv')


# In[66]:


X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features


# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>5.2 Feature Importance</b></h1></center>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.</li>
#   <li style="font-size:150%; font-family:cursive;">Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.</li>
# </ul>

# In[67]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# <center><h1 style="font-size:200%; font-family:cursive; color:#ff6666;"><b>5.3 Correlation Matrix</b></h1></center>
# <ul>
#   <li style="font-size:150%; font-family:cursive;">Correlation states how the features are related to each other or the target variable.</li>
#   <li style="font-size:150%; font-family:cursive;">Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable).</li>
#   <li style="font-size:150%; font-family:cursive;">Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.</li>
# </ul>
# 

# In[68]:


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ------------

# <center><h1 style="font-size:300%; font-family:cursive; color:white; background:#ff6666; padding:11px; border-radius: 20px 20px;">6. Conclusion</h1></center>

# <p style="font-size:150%; font-family:cursive; color:#ff6666;">In this kernel I have discussed all the steps in feature engineering (According to my knowledge). If I have missed any technique / step, please let me know in the comment section - I'll add it ASAP.</p>
# <br>
# <p style="font-size:180%; font-family:cursive; color:green;">If you find this kernel helpful, please give an upvote. Thank you !!!!</p>
