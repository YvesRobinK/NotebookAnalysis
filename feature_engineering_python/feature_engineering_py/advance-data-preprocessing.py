#!/usr/bin/env python
# coding: utf-8

#  <h1 style="text-align: center;" class="list-group-item list-group-item-action active">Table of Contents</h1>
# 
# <ul style="list-style-type:none;">
#         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#1" role="tab"
#                 aria-controls="settings">1. Introduction<span class="badge badge-primary badge-pill">1</span></a>
#         </li>
#         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2" role="tab"
#                 aria-controls="settings">2. Handling Missing Values<span
#                     class="badge badge-primary badge-pill">2</span></a>
#             <ul style="list-style-type:none;">
#                 <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.1" role="tab"
#                         aria-controls="settings">2.1 Visualizing Missing Data <span
#                             class="badge badge-primary badge-pill">3</span></a>
#                     <ul style="list-style-type:none;">
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.1.1"
#                                 role="tab" aria-controls="settings">2.1.1 Matrix <span
#                                     class="badge badge-primary badge-pill">4</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.1.2"
#                                 role="tab" aria-controls="settings">2.1.2 Correlation Heatmap <span
#                                     class="badge badge-primary badge-pill">5</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.1.3"
#                                 role="tab" aria-controls="settings">2.1.3 Dendrogram <span
#                                     class="badge badge-primary badge-pill">6</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.1.4"
#                                 role="tab" aria-controls="settings">2.1.4 Simple Numeric Summaries<span
#                                     class="badge badge-primary badge-pill">7</span></a></li>
#                     </ul>
#                 </li>
#                 <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.2" role="tab"
#                         aria-controls="settings">2.2 Method to Handle Missing Data<span
#                             class="badge badge-primary badge-pill">8</span></a>
#                     <ul style="list-style-type:none;">
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.2.1"
#                                 role="tab" aria-controls="settings">2.2.1 Deletion of Data <span
#                                     class="badge badge-primary badge-pill">9</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.2.2"
#                                 role="tab" aria-controls="settings">2.2.2 Encoding Missingness<span
#                                     class="badge badge-primary badge-pill">10</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#2.2.3"
#                                 role="tab" aria-controls="settings">2.2.3 Imputation Methods<span
#                                     class="badge badge-primary badge-pill">11</span></a></li>
#                     </ul>
#                 </li>
#             </ul>
#         </li>
#         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3" role="tab"
#                 aria-controls="settings">3. Encoding Categorical Attributes<span
#                     class="badge badge-primary badge-pill">12</span></a>
#             <ul style="list-style-type:none;">
#                 <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3.1" role="tab"
#                         aria-controls="settings">3.1 Supervised Encoding Methods <span
#                             class="badge badge-primary badge-pill">13</span></a>
#                     <ul style="list-style-type:none;">
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3.1.1"
#                                 role="tab" aria-controls="settings">3.1.1 Likelihood Encoding <span
#                                     class="badge badge-primary badge-pill">14</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3.1.2"
#                                 role="tab" aria-controls="settings">3.1.2 Target Encoding <span
#                                     class="badge badge-primary badge-pill">15</span></a></li>
#                         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3.1.3"
#                                 role="tab" aria-controls="settings">3.1.3 Deep Learning Methods <span
#                                     class="badge badge-primary badge-pill">16</span></a></li>
#                     </ul>
#                 </li>
#                 <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#3.2" role="tab"
#                         aria-controls="settings">3.2 Approaches for Novel Categories<span
#                             class="badge badge-primary badge-pill">17</span></a>
#             </ul>
#         </li>
#         <li><a class="list-group-item list-group-item-action" data-toggle="list" href="#4" role="tab"
#             aria-controls="settings">4. References <span
#                 class="badge badge-primary badge-pill">18</span></a> </li>
# 
# 
#  </ul>
# 

# 
# <h1  style="text-align: center" class="list-group-item list-group-item-action active">1. Introduction</h1><a id = "1" ></a>
# 
# 
# 
# We all know the Importance of good features for machine learning models. In Machine learning task we have features which we need to process to make them good and this is done by data preprocessing  tasks.
# 
# **Data Preprocessing** : Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model. When creating a machine learning project, it is not always a case that we come across the clean and formatted data. And while doing any operation with data, it is mandatory to clean it and put in a formatted way. So for this, we use data preprocessing task.
# 
# A real-world data generally contains noises, missing values, and maybe in an unusable format which cannot be directly used for machine learning models. Data preprocessing is required tasks for cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model.
# 
# Data Preprocessing involves below steps:
# 
# - Getting the dataset
# - Importing libraries
# - Importing datasets
# - Finding Missing Data
# - Encoding Categorical Data
# 
# Many of us know traditional approaches for above listed steps but in this notebook I will discuss some different approaches which could be game changer for your next project. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("../input/loan-default-dataset/Loan_Default.csv")
df.head()


# In[3]:


df.info()


# <h1  style="text-align: center" class="list-group-item list-group-item-action active">2. Handling Missing Values</h1><a id = "2" ></a>
# 
# Missing data are not rare in real data sets. In fact, the chance that at least one data point is missing increases as the data set size increases. Missing data can occur any number of ways, some of which include the following.
# 
# - Merging of source data sets
# - Random events
# - Failures of measurement
# 
# 

# <h2  style="text-align: center" class="list-group-item list-group-item-success"> 2.1 Visualizing Missing Data</h2><a id = "2.1" ></a>
# 
# 
# Visualizations as well as numeric summaries are the first step in understanding the challenge of missing information in a data set. For small to moderate data (100s of samples and 100s of attributes), several techniques are available that allow the visualization of all of the samples and Attributes simultaneously.
# 
# In this notebook I'll Cover Following visualizations for missing values:-
# - Matrix
# - Correlation Heatmap
# - Dendrogram
# - Simple numerical summaries
# 
# Question may arise that why we need Visualizations?
# Because it is wise to explore relationships within the attributes that might be related to missingness. 

# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.1.1 Matrix</h3><a id = "2.1.1" ></a>
# 
# It is the nullity matrix that allows us to see the distribution of data across all columns in the whole dataset. It also shows a sparkline (or, in some cases, a striped line) that emphasizes rows in a dataset with the highest and lowest nullity.

# In[4]:


import missingno as msno


# In[5]:


msno.matrix(df)
plt.figure(figsize = (15,9))
plt.show()


# From the above plot we can interpret our dataset has lots of missing values in it 

# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.1.2 Correlation Heatmap </h3><a id = "2.1.2" ></a>
# 
# Correlation heatmap measures nullity correlation between columns of the dataset. It shows how strongly the presence or absence of one feature affects the other.
# 
# Nullity correlation ranges from(-1 to 1):
# - -1 means if one column(attribute) is present, the other is almost certainly absent.
# - 0 means there is no dependence between the columns(attributes).
# - 1 means if one column(attributes) is present, the other is also certainly present.
# 
# Unlike in a familiar correlation heatmap, if you see here, many columns are missing. Those columns which are always full or always empty have no meaningful correlation and are removed from the visualization.
# 
# The heatmap is helpful for identifying data completeness correlations between attribute pairs, but it has the limited explanatory ability for broader relationships and no special support for really big datasets.

# In[6]:


msno.heatmap(df, labels = True)


# From above visualization we can easily interpret missingness of attribute rate_of_interest and upfront_charges is dependent on each other(correlation value = 1) means if one will be present another will be present. 

# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.1.3 Dendrogram </h3><a id = "2.1.3" ></a>
# 
# 
# The dendrogram shows the hierarchical nullity relationship between columns. The dendrogram uses a hierarchical clustering algorithm against one another by their nullity correlation.
# 
# [More about Hierarchical Clustering Algorithm](http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

# In[7]:


# Columns having missing values
missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
missing_columns


# In[8]:


msno.dendrogram(df[missing_columns])


# We interpret the dendrogram based on a top-down approach, i.e., to focus on the height at which any two columns are joined together with matters of nullity. More will be the height less will be the relation and vice versa is also True. 
# 
# For example if we see pair of attributes LTV and property value has height 0 implies they are highly correlated in case of nullity. Similarly attribute LTV and rate_of_interest have maximum height implies they are less correlated with each other.  

# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.1.4 Simple Numerical Summaries </h3><a id = "2.1.4" ></a>
# 
# 
# Moving Forward lets try to analyse numerical summary of missing attributes. Simple numerical summaries are effective at identifying problematic predictors and samples when the data become too large to visually inspect.

# In[9]:


def get_numerical_summary(df):
    total = df.shape[0]
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        per = (null_count/total) * 100
        missing_percent[col] = per
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    return missing_percent


# In[10]:


missing_percent = get_numerical_summary(df)


# Now I guess visualization part is done lets move forward to methods which we can use to handle missing values.

# 
# <h2  style="text-align: center" class="list-group-item list-group-item-success"> 2.2 Methods to Handle Missing Data</h2><a id = "2.2" ></a>
# 
# As we Know if our data has missing values than our model will not train except few models which can tolerate them like some tree based models but the point is we want to handle this and how can we handle them. So, in this notebook to handle missing data I will discuss following techniques :-
# 
# - Deletion of Data 
# - Encoding Missingness
# - Imputation Methods
# 

# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.2.1 Deletion of Data </h3><a id = "2.2.1" ></a>
# 
# 
# 
# The simplest approach for dealing with missing values is to remove entire attribute(s) and/or sample(s) that contain missing values. However, one must carefully consider a number of aspects of the data prior to taking this approach. For example, missing values could be eliminated by removing all predictors that contain at least one missing value. Similarly, missing values could be eliminated by removing all samples with any missing values.
# 
# **Note: When it is difficult to obtain samples or when the data contain a small number of samples (i.e., rows), then it is not desirable to remove samples from the data.**
# 
# Consider this small intuition shown below
# 
# Let M = Number of Samples(rows).\
# and Let N = Number of Attributes(columns).
# 
# 
# Case 1: Deletion of Attributes
# 
# If N has range of [1-10]\
# Then don't delete the attribute that contain missing values but if that attribute has missing values around 80-90% then deletion of that attribute will be good option instead of just predicting values of those 80-90% data based on that 10-20% data. 
# 
# Case 2: Deletion of Samples
# 
# If M is a large number according to your task\
# Then deletion of sample can be a Good step but if that sample has few missing values with respect to attribute, then you should consider methods to fill those missing values.
# 
# Lets move on to the implementation part, I will just show how to delete data for both cases but you can interpret more according to your tasks.

# **Deletion of an Attribute**
# 
# According to Simple numerical Summaries the attribute Upfront_charges has largest missing values percentage of (26.664%) which is not ideal percentage to remove a feature but just for sake of implementation I will remove that feature.

# In[11]:


df_temp = df.copy()


# In[12]:


# Threshold to remove attribute having missing values greater than threshold
ATTRIBUTE_THRESHOLD = 25 #25% in this case 

for col, per in missing_percent.items():
    if per > ATTRIBUTE_THRESHOLD:
        df_temp.drop(col, axis = 1, inplace = True)


# By generating numerical summary of df_temp we can see now attribute Upfont_chargers being removed from the dataset as it has missing values percentage greater than threshold we defined

# In[13]:


_ = get_numerical_summary(df_temp)


# In[14]:


del df_temp


# **Deletion of the Samples**
# 
# We will try to delete those samples having missing values in more than 5 attributes  

# In[15]:


df_temp = df.copy()


# In[16]:


# Getting Missing count of each sample            

for idx in range(df_temp.shape[0]):
    df_temp.loc[idx, 'missing_count'] = df_temp.iloc[idx, :].isnull().sum()  


# In[17]:


# Threshold to remove samples having missing values greater than threshold
SAMPLE_THRESHOLD = 5

print("Samples Before Removal : {}".format(df_temp.shape[0]))

df_temp.drop(df_temp[df_temp['missing_count'] > SAMPLE_THRESHOLD].index, axis = 0, inplace = True)

print("Samples After Removal : {}".format(df_temp.shape[0]))


# In[18]:


del df_temp


# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.2.2 Encoding Missingness </h3><a id = "2.2.2" ></a>
# 
# 
# 
# When an attribute is discrete in nature, missingness can be directly encoded into the attribute as if it were a naturally occurring category. For example in this dataset the attribute loan_limit has 3344 missing values so we can assign some new category to these missing values. 

# In[19]:


cat_missing_cols = [col for col in missing_columns if df[col].dtype == 'object']
cat_missing_cols


# In[20]:


df.loan_limit.value_counts()


# In[21]:


df[cat_missing_cols] = df[cat_missing_cols].fillna('Missing')
df.loan_limit.value_counts()


# In[22]:


df[cat_missing_cols].info()


# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 2.2.3 Imputation Methods</h3><a id = "2.2.3" ></a>
# 
# 
# 
# Another approach to handling missing values is to impute or estimate them. Imputation uses information and relationships among the non-missing
# attributes to provide an estimate to fill in the missing value.
# 
# In this section we will work on imputation models which will help us impute missing values by extracting interesting patterns from attributes which don't have missing values at that point on time.
# 
# 
# - Within a sample data point, other variables may also be missing. For this reason, an imputation method should be tolerant of other missing data.
# 
# - Imputation creates a model embedded within another model. There is a prediction equation associated with every attribute in the training set that might have missing data. It is desirable for the imputation method to be fast and have a compact prediction equation.
# 
# - Many data sets often contain both numeric and discrete attributes. Rather than generating dummy variables for discrete attributes, a useful imputation method would be able to use attributes of various types as inputs.
# 
# - The model for predicting missing values should be numerically stable and not be overly influenced by outlying data points.
# 
# Virtually any machine learning model could be used to impute the data. Here, the focus will be on several that are good candidates to consider.
# 
# Question arise if an attribute has missing values around 50-60% then can we use imputation methods? And the answer is it depends upon datasets which we are using because an attribute with 60% missing values may has very good correlation with some other attribute which can be helpful to fill those missing values on the other side if let say some column like ID column which is independent of all columns has missing values around 10% using imputation methods we may not get results we wanted. 

# In this notebook we are gonna work on following imputation methods:-
# 
# - KNN for Imputation
# - Tree Based Imputation
# - Linear Models for Imputation   

# <h3>(a) K-Nearest Neighbors(KNN) for Imputation</h3>
# 
# When the training set is small or moderate in size, K-nearest neighbors can be a quick and effective method for imputing missing values. This procedure identifies a sample with one or more missing values. Then it identifies the K most similar samples in the training data that are complete (i.e., have no missing values in some columns). Similarity of samples for this method is defined by a distance metric. When all of the predictors are
# numeric, standard Euclidean distance is commonly used as the similarity metric. 
# 
# After computing the distances, the K closest samples to the sample with the missing value are identified and the average value of the predictor of interest is calculated. This value is then used to replace the missing value of the sample.

# In[23]:


from sklearn.impute import KNNImputer

df_temp = df.copy()


# As we haven't done categorical encoding yet (We'll cover it in next section) so, for time being lets impute on numerical data only later we will impute on full data after encoding.

# In[24]:


num_cols = [col for col in df_temp.columns if df_temp[col].dtype != 'object']
print(num_cols)
df_temp = df_temp[num_cols]


# In[25]:


# Initializing KNNImputer
knn = KNNImputer(n_neighbors = 3)

knn.fit(df_temp)


# In[26]:


X = knn.transform(df_temp)


# In[27]:


df_temp = pd.DataFrame(X, columns = num_cols)
df.info()


# In[28]:


del df_temp


# <h3>(b) Trees</h3>
# 
# Tree-based models are a reasonable choice for an imputation technique since a tree can be constructed in the presence of other missing data. While a single tree could be used as an imputation technique, it is known to produce results that have low bias but high variance. And we all know who kills the bias its Ensembles of trees. 
# 
# Random forests is one such technique and has been studied for this purpose. However, there are a couple of notable drawbacks when using this technique in a predictive modeling setting. First and foremost, the random selection of predictors at each split necessitates a large number of trees (500 to 2000) to achieve a stable, reliable model. This can present a challenge as the number of attributes with missing data increases since a separate model must be built and retained for each predictor. Also Random forest will have heavy computations.
# 
# A good alternative that has a smaller computational footprint is a bagged tree. A bagged tree is constructed in a similar fashion to a random forest.
# The primary difference is that in a bagged model, all attributes are evaluated at each split in each tree. The performance of a bagged tree using 25–50 trees is generally in the ballpark of the performance of a random forest model. And the smaller number of trees is a clear advantage when the goal is to find reasonable imputed values for missing data.
# 
# 
# 

# In[29]:


missing_columns


# In[30]:


from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor()


# In[31]:


income = df['income']


# As we have to encode categorical variables into numerical data to use sklearn's tree based models so for the time being I am encoding categorical variables using Label Encoding Method

# In[32]:


df_temp = df.copy()


# In[33]:


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

cat_cols = [col for col in df.columns if df[col].dtype == 'object']

for col in cat_cols:
    df_temp[col] = lb.fit_transform(df_temp[col])


# In[34]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


def tree_imputation(df):
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    non_missing_cols = [col for col in df.columns if df[col].isnull().sum() == 0]
    # num_cols = [col for col in missing_cols if df[col].dtype != 'object']

    # df = df[num_cols]
    for col in missing_cols:

        # Defining a new bagging model for each attribute  
        model = BaggingRegressor(DecisionTreeRegressor(), n_estimators = 40, max_samples = 1.0, max_features = 1.0, bootstrap = False, n_jobs = -1)

        col_missing = df[df[col].isnull()]
        temp = df.drop(df[df[col].isnull()].index, axis = 0)

        # print(temp.columns)
        # X = temp.drop(col, axis = 1)
        X = temp.loc[:, non_missing_cols]
        y = temp[col]

        model.fit(X, y)

        y_pred = model.predict(col_missing[non_missing_cols])
        # col_missing[col] = y_pred

        df.loc[col_missing.index, col] = y_pred
        
    return df
    


# In[35]:


df_new = tree_imputation(df_temp)
df_new.info()


# In[36]:


msno.bar(df_new)
plt.show()


# We can see all missing values from the dataset are gone. Now as we temporarily encode categorical variables because we will encode them in later section so lets decode them.

# In[37]:


df_new = pd.concat([df[cat_cols], df_new.drop(cat_cols, axis = 1)], axis = 1)
df_new.head()


# In[38]:


df_new.info()


# <h3>(c) Linear Methods</h3>
# 
# When a complete Attribute shows a strong linear relationship with a attribute that requires imputation, a straightforward linear model may be the best approach. Linear models can be computed very quickly. Linear regression can be used for a numeric attribute that requires imputation.
# Similarly, logistic regression is appropriate for a categorical attribute that requires imputation.
# 
# Let say feature rate_of_interest and Interest_rate_spread are dependent features means one feature can be defined using other. If feature rate_of_interest has missing values than it can be imputed using simple linear model trained on Interest_rate_spread.

# <h1  style="text-align: center" class="list-group-item list-group-item-action active">3. Encoding Categorical Attributes</h1><a id = "3" ></a>
# 
# 
# Categorical Features are those that contain qualitative data.This Section focuses primarily on methods that encode categorical data to numeric values.
# 
# Categorical variables/features are any feature type can be classified into three major types:
# 
# - Nominal
# - Ordinal
# - Binary
# 
# **Nominal variables** are variables that have two or more categories which do not have any kind of order associated with them. For Example if our dataset has any 4 types of colors, i.e. Red, Blue, Orange, Green it can be considered as a nominal variable.
# 
# **Ordinal variables** on the other hand, have “levels” or categories with a particular order associated with them. For example, an ordinal categorical variable can be a feature with three different levels: low, medium and high. Order is important.
# 
# **Binary Variables** are same as nominal variables but with only categories For example, if gender is classified into two groups, i.e. male and female.
# 
# For Nominal Variables We generally uses Label Encoding Scheme in which we encode each category by just converting it to some integer values this kind of encoding can work in case of Ordinal variables but **for label encoding it has the disadvantage that the numeric values can be misinterpreted by algorithms as having some sort of hierarchy/order in them**. This ordering issue is addressed in another common alternative approach called 'One-Hot Encoding'.
# 
# One-Hot-Encoding has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space. **The disadvantage is that for high cardinality, the feature space can really blow up quickly and you start fighting with the curse of dimensionality.**
# 
# Another big issue with encoding schemes is new category or while splitting data in train/validation/test set all samples of the rare classes may split into validation/test set then during it will raise error while predicting.  
# 

# Because of few potential issues with traditional approaches we now need to search for some unique approaches some of them are listed below:-
# 
# - Supervised Encoding Methods
# - Approaching for Novel Categories 
# 
# Starting from Supervised Encoding Methods lets define it first 
# 
# 
# <h2  style="text-align: center" class="list-group-item list-group-item-success"> 3.1 Supervised Encoding Methods</h2><a id = "3.1" ></a>
# 
# 
# There are several methods of encoding categorical variable to numeric columns using the output data as a guide (so that they are supervised methods). In Supervised Techniques we will discuss following methods to encode categorical variable:-
# 
# - Effect or Likelihood Encoding  
# - Target Encoding 
# - Deep Learning Methods
# 
# 
# 
# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 3.1.1 Likelihood Encoding</h3><a id = "3.1.1" ></a>
# 
# 
# 
# The effect of the factor level on the output data is measured and this effect is used as the numeric encoding. Here effect of that particular category on output data can be calculated using simple linear models or mean, mode and median methods. 
# 
# For classification problems, a simple logistic regression model can be used to measure the effect between the categorical outcome and the categorical predictor. After Computing Effects we will compute log-odds of those effects, If the effect is p, the odds of that event are defined as p/(1 − p) and log odds by simply taking log of odds. After when we get the log odds of each category we than encode each category with them using map function.     
# 

# In[39]:


df_temp = df_new.copy()


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


## Again we have to temporarily encode variables
lb = LabelEncoder()

cat_cols = [col for col in df_temp.columns if df_temp[col].dtype == 'object']

for col in cat_cols:
    df_temp[col] = lb.fit_transform(df_temp[col])


# In[41]:


def likelihood_encoding(df, cat_cols, target_variable = "Status"):
    # cat_cols.remove(target_variable)
    df_temp = df.copy()
    for col in cat_cols:
        effect = {}
        print(col)
        for category in df[col].unique():
            print(category)

            try:
                temp = df[df[col] == category]
                lr = LogisticRegression()
                X = temp.drop(target_variable, axis = 1, inplace = False)
                y = temp[target_variable]
                # print(temp.drop(target_variable, axis = 1).isnull().sum())
                lr.fit(X, y)

                effect[category] = accuracy_score(y, lr.predict(X))
            except Exception as E:
                print(E)
        
        for key, value in effect.items():
            effect[key] = np.log(effect[key] / (1 - effect[key] + 1e-6))
            
        df_temp.loc[:, col] = df_temp.loc[:, col].map(effect)
    return df_temp


# In[42]:


df_temp = likelihood_encoding(df_temp, cat_cols)


# In[43]:


df_temp.head()


# In[44]:


df_temp.info()


# In[45]:


del df_temp


# Implementation part is done While very fast, it has drawbacks. For example, what happens when a factor level has a single value? Theoretically, the log-odds should be infinite in the appropriate direction i.e. p/(1 - p) tends to infinity if p = 1. And numerically, it is usually capped at a large (and inaccurate) value.
# 
# For example in above implementation for column construction_type their are 2 categories 1 has around 148637 values and other has 33 values so for category mh which has 33 values p becomes 1 and it raises the error. Then This lead us to move to next technique known as target encoding which is simpler that likelihood encoding. 
# 
# 
# 
# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 3.1.2 Target Encoding</h3><a id = "3.1.2" ></a>
# 
# 
# It is same as likelihood encoding but the difference is we use average of output variable for that particular category to encode values inplace of some linear model. Lets move to the implementation.

# In[46]:


def target_encoding(df, cat_cols, target_variable = "Status"):

    for col in cat_cols:
        weight = 7
        feat = df.groupby(col)[target_variable].agg(["mean", "count"])
        mean = feat['mean']
        count = feat['count']
        
        smooth = (count * mean + weight * mean) / (weight + count)

        df.loc[:, col] = df.loc[:, col].map(smooth)

    return df


# In[47]:


df_temp = df_new.copy()


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


## Again we have to temporarily encode variables
lb = LabelEncoder()

cat_cols = [col for col in df_temp.columns if df_temp[col].dtype == 'object']

for col in cat_cols:
    df_temp[col] = lb.fit_transform(df_temp[col])


# In[49]:


df_temp = target_encoding(df_temp, cat_cols)


# In[50]:


df_temp.head()


# - Target Encoding could be good choice for binary classification but for regression it is not, because it ignores intra-category variation of the target variable. This is addressed in Bayesian Target Encoding.
# 
# - Target encoding has a tendency to overfit due to the target leakage.
# 
# - Another problem is that some of the categories have few training examples, and the mean target value for these categories may assume extreme values, so encoding these values with mean may reduce the model performance.
# 
# These issues are addressed in Bayesian Target Encoding. Which you can read from this informational blog [Target Encoding and Bayesian Target Encoding](https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c)
# 

# In[51]:


df['age'].value_counts()


# <h3  style="text-align: center" class="list-group-item list-group-item-warning"> 3.1.3 Deep Learning Methods</h3><a id = "3.1.3" ></a>
# 
# 
# Another supervised approach comes from the deep learning literature on the analysis of textual data. In this case, large amounts of text can be cut up into individual words. Rather than making each of these words into its own indicator variable, word embedding or entity embedding approaches have been developed. Similar to the dimension reduction methods, the idea is to estimate a smaller set of numeric features that can be used to adequately represent the categorical predictors.
# 
# In addition to the dimension reduction, there is the possibility that these methods can estimate semantic relationships between words so that words with similar themes (e.g., “dog”, “pet”, etc.) have similar values in the new encodings. This technique is not limited to text data and can be used to encode any type of qualitative variable.
# 
# The idea is well very simple do not extract features manually use neural network to do the hard part and just wait for the results.

# <h2  style="text-align: center" class="list-group-item list-group-item-success"> 3.2 Approaches for Novel Categories</h2><a id = "3.2" ></a>
# 
# 
# 
# What if some new category introduce to some attribute in future how will we encode that variable then? If there is a possibility of encountering a new category in the future, one strategy would be to use the "other" category to capture new values. 
# 
# While this approach may not be the most effective at extracting predictive information relative to the response for this specific category, it does enable the original model to be applied to new data without completely refitting and we do need to ensure that the "other" category is present in the training/testing data.
# 
# After assigning "other" category to novel category than we can do all kinds of encodings we studied above

# This is it for this notebook in upcoming notebooks will try to cover more data preprocessing techniques like Binning for reducing noise, application of PCA in feature engineering, advance feature selection techniques and many more. 

# <h1 style="text-align: center" class="list-group-item list-group-item-action active">References</h1><a id = "" ></a>
# 
# - [Data Preprocessing in Machine Learning](https://www.javatpoint.com/data-preprocessing-machine-learning)
# - [Easy Way of Finding and Visualizing Missing Data in Python](https://medium.datadriveninvestor.com/easy-way-of-finding-and-visualizing-missing-data-in-python-bf5e3f622dc5)
# - [Visualizing Missing Values in Python is Shockingly Easy](https://towardsdatascience.com/visualizing-missing-values-in-python-is-shockingly-easy-56ed5bc2e7ea)
# - [Encoding categorical variables using likelihood estimation](https://datascience.stackexchange.com/questions/11024/encoding-categorical-variables-using-likelihood-estimation)
# - [Target Encoding and Bayesian Target Encoding](https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c)

# In[ ]:




