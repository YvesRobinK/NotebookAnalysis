#!/usr/bin/env python
# coding: utf-8

# <h2>Work under progress!!</h2>
# 
# <h2>Quote:</h2> If you investigate the data long enough, sooner or later it will confess!

# In[1]:


from IPython.core.display import display, HTML, Javascript
import IPython.display


# This is one of my learning competitions. Throughout this notebook I will be experimenting with a lot different techniques in order to clear out my own doubts in most feasible way  possible. 
# By all means this kernel is not directed towards optimisation. Rather, it is focused more towards implementation of different techniques presented in Machine Learning. I will try to highlight the insights that I have learned from this competion. 
# In addtition, I will try to explore the advantages and disadvantages of each model in my experience. 
# <h4>Through out this kernel. I will be touching some topic about statistical learning and its significance in the world. I have been reading few of the data science books and with the help of this kernel I will be sharing some usefull information that you might like.</h4>
# <img src = "https://i.imgur.com/FCXWhmy.png" width="500"/>

# <h2># What is Statistical Learning?</h2>
# <strong>It refers to set of tools for understanding the data. These tools can be classified as supervised or unsupervised.</strong>
#     
#  a. **Supervised**: It involves building statistical models for **predicting or estimating** an output based on one or more input variables/features. 
# 
# b. **Unsupervised**: And, in a breif description in these problems. There are input varaibles but no supervising output. We gernerally don't have labeled data and we try to group together the similar data points based on their characteristics/features
# 
# 
# <h2># How we can use statistical learning?</h2>
# Our motive is to understand our objective. <strong>Stastistical Learning can be used for two reason creating inference  or predictions models.</strong> Machine learning is not always about making predictions but rather it can be majorly used for building inference about features and finding the relationship among features and target variable (<strong>Let's say function (f)</strong>).
# <p><strong>Function f:</strong> f is a fixed but unknown functions of features X1,...Xp with the realtion to our target variable y. <strong>f functions presents the systematic information that features(X) provide about Y. As mentioned earlier our two main reasons that we may to wish to estimate f are prediction and inference.</strong>. And, before standing diving in any project we should be interested in answering the following questions:</p>
# <ul> 
#     <strong><li>Which predictors are assciated with our target varaible and why?</li></strong>
#   <strong>  <li>Which predictors are assciated with our target varaible and why?</li></strong>
#    <strong> <li>Can the relationship between Y and each predictor be adequately summarized using a linear equation, or it is more complicated?</li></strong>
#     </ul>
# 

# ## **[Structure consists of 7 stages:](#0)**<a id ="0"></a></br>
# ## [1. Importing the Libraries](#1)
#     a. Pandas, Numpy
#     b. Matplotlib, Seaborn, Plotly
#     c. Scikit Learn 
#     
# ## [2. Importing and understanding the Dataset:](#2)
#     a. Understanding the Data and its Datatypes
#     b. Basic Statistical Analysis 
#     c. Datatypes and its distrinution
#     
# ## [3. Exploratory Data Analysis:](#3)
#     a. Univariate Analysis 
#     b. Bivaraite Analysis 
#     c. Multivariate Analysis 
#     d. Target Varaible Characteristics
#     
# ## [4. Data Preprocessing:](#4)
#     a. Target Variable 
#     b. Outliers
#     c. Missing Values
#     d. Feature Engineering 
#     e. Feature Scaling 
#     f. Encoding 
#     g. Cross-Validation
#     
# ## [5. Model Selections:](#5)
#     a. Regressions 
#     b. Neural Networks 
#     
# ## [6. Model Evaluation:](#6)
#     a. Metrics: RMSE
#     
# ## [7. Model Tuning](#7)
#     a. Hyperparameters 

# <h1>1. Importing the Libraries</h1><strong>

# In[2]:


# General Essential Libraries:
import numpy as np 
import pandas as pd 

import seaborn as sns 
sns.set(style = "whitegrid")
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

from IPython.core.display import display, HTML, Javascript
import IPython.display


# In[3]:


# Libraries for interactive visualisation: 
import plotly.figure_factory as ff 
import  plotly.offline as py
import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs,init_notebook_mode, iplot, plot
from plotly import tools 
py.init_notebook_mode(connected = True)

import cufflinks as cf 
cf.go_offline()


# In[4]:


# Libraries for Machine Learning Algroithyms:
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder 

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[5]:





# <h1>2. Importing and Understanding the Data Set<h2>

# In[5]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[6]:


# Header of Training set 
df_train.head(3)


# In[7]:


# Header of Test Set 
df_test.head(3)


# In[8]:





# In[8]:


# Information of the Training set: 
#df_train.info()


# In[9]:


# Information of the Test set: 
#df_test.info()


# In[10]:





# <h1>2.b Concatenating both the data set (Training and Testing)</h1>

# In[10]:


# General Description of Numberical Columns of the Entire Set (Training and Testing)
df = pd.concat([df_train,df_test], axis = 0)
df.drop("SalePrice", axis =1, inplace=True )
df.describe()


# <h3>Estimating the location for our target variable</h3>

# In[11]:


round(df_train["SalePrice"].describe(),2)


# In[12]:





# <h1>2.c) Datatypes and its distribution</h1>

# In[12]:


df_types = df.dtypes.value_counts()
print(df_types)

plt.figure(figsize = (14,4))
sns.barplot(x = df_types.index, y = df_types.values)
plt.title("Data Type Distribution")


# Creating 2 new variable for each data type. This will help us to manipulate the data in an effecient manner. 

# In[13]:


num_col = df.select_dtypes(include=("float64", "int64"))
cat_col = df.select_dtypes(include=("object"))


# <h3>Key Notes:</h3>
# <strong>As we can see  we have 37 integer and float data type columns which can be used for  predicting the target variable and we have 43 object data type columns which we can use to identify relationship between the feature. However, in order to use these columns for our Machine Learning algorithms we need to Encode them later in our Data Preprocessing stage</strong>

# <img src = "https://i.imgur.com/FCXWhmy.png" width="500"/>

# # 3.Exploratory Data Analysis[^](#0) <a id = "1"></a> <br>

# ## ** What is EDA (Exploratory Data Analysis)?** 
# <strong>It is considered to be 1st step towards data science. It is a comparatively new area of Statistics. Classical statistics focuses on exclusively on **inference**. While EDA is an approach to identify the characteristics of the data itself. A sometimes complex set of procedures are being followed for drawing conclusions about large populations based on small samples by the means of EDA.</strong>
# ###  **Interested in getting deeper understanding about EDA?** 
# **Read Book Exploratory Data Analysis [Tukey-1977]**
# 
# <img src="https://i.imgur.com/RPtnBxf.png" width="300"/>

# <h1>3. a. Correlation matric for numerical data</h1>

# In[14]:


# Correalation plot in order to identify the relationship between the Numberical Features:
plt.figure(figsize=(20,10))
sns.heatmap(df_train.corr(), linewidths=.1, annot=True, cmap='magma')
df_train.corr()["SalePrice"].sort_values(ascending = False).head(5)


# ## **Why Correlation Matrix?**
# **It helps to get a better understanding of all variable and their relation with each other. Features which have higher linearity either positive or negative tends to have high positive and high negative Person's correlation coefficient. However, the above matrix can not be used for inference as there might a lot of outliears in the data set which does not capture the true characteristics of the data and moreover correlation coefficient are very sensitive to outliers which we have not removed or replaced yet! Nonetheless, matrix in general helps to get an understanding of missing values and anomalies in the data set in early stage.**
# 
# ### Top 10 features correlated with our target variable:
# 1. Over All Quality (OverallQual) with correlation = 0.79
# 2. OverallQual = 0.790982
# 3. GarageCars = 0.640409
# 4. GarageArea  = 0.62343
# 5. TotalBsmtSF = 0.613581
# 6. 1stFlrSF = 0.605852
# 7. FullBath = 0.560664
# 8. TotRmsAbvGrd = 0.533723
# 9. YearBuilt = 0.5228974
# 

# <h2># Before moving further let's take a look at missing values in Numberical and Categorical columns:</h2>

# In[15]:


fig, (ax1, ax2) =plt.subplots(nrows=2, ncols=1, figsize = (15,10))

sns.heatmap(cat_col.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False, ax=ax1)
plt.title("Missing Values in Categorical Columns")
sns.heatmap(num_col.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False)
plt.title("Missing Values in Numberical Columns")
plt.tight_layout()


# <h3>**All the light color represents the missing values in the respective columns**</h3>

# In[16]:





# ##  **Important Key Terms:** 
# **Before we get in further detail and start our brief Exploartory Data Analysis. I would like to mention some key terms for data types**. 
# ### **There are two type of basic structured data: numberic and categorical and they are further sub categories which are very important to understand.**
#  **1. Continous**: Data that can take on any value in an interval
#  **2. Discrete:** Which can take only interger values. 
#  **3. Categorical**: Data which can take only set of possible. 
#  **4. Binary**: It is a special case of categorical data that **take only one or two vales**
#  **5. Ordinal**: Categorical data that a intituitive ordering. These are very important to be identified during the EDA because a feature having **Ordinal Data** tends to have more relationship with another features as the values increases by. 
#  
#  ### ** Let's take look at all the above mentioned data types in our data set to get a deeper understanding**
# 

# ## # **Scaterplot Matrix w.r.t Sales Price**

# In[16]:


mat = df_train[["SalePrice", "LotFrontage", "TotalBsmtSF","GrLivArea","OverallQual" ]]
mat["index"] = np.arange(len(mat))

fig = ff.create_scatterplotmatrix(mat, diag="box", index="index", colormap_type="seq", colormap="Jet", 
                                 height = 900, width = 1100)
py.iplot(fig)


# ## **As we can see there are few outiers in the dataset. Which we will be removing in the data preprocessing stage:**

# ## # **Distribution of the Target variable (Sales Price)**

# In[17]:


hist_data = [df_train["SalePrice"]]
label = ["Sales Price"]
color = ["navy"]

fig = ff.create_distplot(hist_data, label, colors = color, show_hist=False)
fig["layout"].update(title ="Sale Price Distribution") 
py.iplot(fig)


# ## **# What is Central Limit Theorem and how it is implimented?**:
# ### ** CLT: As the sample size increase, the sample mean approaches a normal distribution (Gaussian Curve). It is far most important theorem in Statistics because in *Statistics or Machine Learning* we assume that our distribuiton is normally distributed.** 
# **A-lot MAchine Learning algorithms such as Linear Regression assume that the sample/population distribution is normally distributed.** Therefore, it is always a good practive to  check the distribution of the given data features. 
# 
# ### **#Which Leads to a term called Parametric and Non-Parametric**
# **Parametric: I assume that the population follows a distribution based on fixed parameters. While Non-Parametric opposite.**

# ## # **Creating functions for effecient data visualisation:**
# <strong>
# 

# In[18]:


for i in cat_col.columns:
    print(cat_col[i].value_counts(), "/n")


# In[19]:


def uni(col):
    out = []
    for i in col:
        if i not in out:
            out.append(i)
    return(out)
""""""
colors1 = ["#a9fcca","#d6a5ff", "#639af2", "#fca6da", "#f4d39c", "orange", "#7af9ad","green", "maroon", "#3498db", "#95a5a6", "#e74c3c", "#34495e","#df6a84","#ad2543","#223f08", "#DF3A01", "#045FB4","#088A4B","#FE2E64" ]
''''''
def bar_pie (col, colors = colors1,
            main_title = "Main Title", x_label = "X Label", y = "Y label", do_x = [.6,.9], do_y = [.9,.2]):
    
    col_count = df[col].value_counts()
    
    trace = go.Bar(x = col_count.index, y = col_count.values, marker=dict(color = colors))
    
    trace1 = go.Pie(labels= col_count.index, values=col_count.values, hole= 0.6, textposition="outside", marker=dict(colors = colors),
               domain = dict(x = do_x, y = do_y), name = "title", )
    
    data = [trace, trace1]
    layout = go.Layout(title= main_title)
    fig = go.Figure(data =data, layout = layout)
    iplot(fig)

""""""
def price(col):
    if col in range(0, 150000):
        return("Low")
    elif col in range(15000, 300000):
        return("Medium")
    else:
        return("High")
df_train["Price"] =df_train["SalePrice"].apply(lambda x: price(x))
df_train.head(4)

''''''


# In[20]:


neig_hood= uni(df["Neighborhood"])
sale_cond = uni(df["SaleCondition"])
qual = uni(df["OverallQual"])
me = df_train.groupby("Neighborhood").agg({"SalePrice":np.mean}).reset_index()


# <h2>Part 1: Visualising the Categorical Varibles:</h2>
# <strong> # **As mentioned earlier as we know by know generally a data has two data types  numberical and categorical.** </strong>
# **Let's dive in and explore some characteristics of the categorical data in our dataset**. 
# 
# **Step 1** Identifying the total number of categorical columns we have in our data set!

# In[21]:


cat_col.columns.values


# In[22]:


num_col.columns.values


# **Therefore, we have 43 categorical columns in our dataset. Some of them are Binary, General Categorical and Ordinal types**
# 
# **I will be exploring each one of them to get a better understanding and will try to explain there distinguish characteristics best in my knowledge in this sections**
# 
# ## # **So what plots are mostly use for Categorical Columns?**
# **Generally Categorical columns are those which are assigned to a specific category (name,type, locaction etc) and the related to some other features with values**.
# **We can visualise Categorical Columns by following (the most common ones)**:
# 1. **Box Plots** (These are generally the most comman used plots for Categorical Columns)
# 2. **Violin Plots**: These are same as box plot but **they have kernel desnsity distribution on both the side**. 
# 3.**Bar Plots**: Basic bar plots
# 4. **Swarn Plots** 
# 5.  **Strip Pltots**: Similar to swarn plots but without the spread. 
# 
# **I will try to use each one of them in this stage along with explaination of the data type present in those plots**.
# 

# <h2> # ** 1. Bar Plots: House Style Frequency**</h2>
# **Let's take a look at House Style columns and make a Bar Plot of it**

# In[23]:


bar_pie(col="HouseStyle", main_title="House Style Frequency")


# **As we can see on x-axis we have a categorical values which are the type of houses in the data set. While on the y-axis are the total number of observations for those respective types.**
# **It seems to look like highest number of listing are for 1 story apartments 1471 followed to 2 stories with value of 872. Which makes sense as majority of the appartments tends to be small due to higher requirement and high return on investment.**

# In[24]:





# <h2> <strong># Number of Houses Sold every Month</strong></h2>

# In[24]:


mon_d = {1:"Jan", 2:"Feb",3:"Mar",4:"Aprl",5:"May",6:"Jun",7:"July",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dex"}
df["Month"] = df["MoSold"].map(mon_d)

bar_pie(col="Month", main_title="Number of Houses sold very Month", do_x= [.9,.9] ,do_y= [.9,.4])


# <strong>Please Note:</strong> The new columns which I have created in this cell is in my main Data frame (which I am using for visualisation), not  in the train/test set. Hence, we do not have to worry about the addition redundant column we have created.
# <h3>Observation:</h3>
# <strong>Tends to look like in the middle of the year May, June, July  house are more likely to be sold, which makes sense as alot of people usually like to invest by the end of fiscal year cycle and during the holidays. INTERESTING</strong>

# ## #**2. Box plots: Neighborhood**
# ### # **One of the main factor in property evaluation is the Location. Let's see how Sales Prices varies from place to place.**

# In[25]:


data = []
for i in neig_hood:
    data.append(go.Box(y = df_train[df_train["Neighborhood"]==i]["SalePrice"], name = i))

layout = go.Layout(title = 'Sales Price based on Neighborhood', 
                   xaxis = dict(title = 'Neighborhood'), 
                   yaxis = dict(title = 'Sale Price'))
fig = dict(data = data, layout = layout)
py.iplot(fig)


#   ## **#Estimates of Locations: where STATISTICS comes in place**
#   ### **Variables with measured data always have thousand of distinct values. A basic step in exploring the data is getting a "TYPICAL VALUE" for each feature: an estimate of where the most of the data is located.**
#   
#   **How can we estimate the location of a varaible? (By location I mean, where the majority of the data is being placed in terms of values)**
#   ***Mean, Median, Outliers etc they all help us to get better understanding about the location of the values of the data. Therefore, Box Plots are very usefull because they helps to identifying the location of the values in a given feature.**

# In[26]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
sns.violinplot( x= df_train["Exterior1st"], y=df_train["SalePrice"], ax=ax[0])
plt.title("Sales Price VS Exterior Distibution")
plt.xticks(rotation =90)

sns.boxplot( x= df_train["Exterior2nd"], y=df_train["SalePrice"], ax=ax[1])
plt.xticks(rotation =90)
plt.title("Sales Price VS Exterior Distibution")
plt.tight_layout()


# In[27]:


data = []
for i in qual:
    data.append(go.Box(y = df_train[df_train["OverallQual"]==i]["SalePrice"], name = i,  boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))
layout = go.Layout(title = 'Sales Price based on Overall Quality', xaxis=dict(title ="Ouality grade"), yaxis=dict(title ="Sales Price"))

fig = dict(data = data, layout = layout)
py.iplot(fig)


# <strong>As the plot shows, it is clear as the quality of the houses improves the sales prices goes proportionally.</strong>

# <h2> <em> #A Comparison between Violin Plot and Box Plots</h2></em>
# <h2> <strong># Sales Prices based on Year built</strong></h2>

# In[28]:


yr_built = uni(df["YearBuilt"])
data = []
for i in yr_built:
    data.append(go.Box(y = df_train[df_train["YearBuilt"]==i]["SalePrice"], name = i,  boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))
layout = go.Layout(title = 'Sales Price based on Year Built', xaxis=dict(title ="Years"), yaxis=dict(title ="Sales Price"))

fig = dict(data = data, layout = layout)
py.iplot(fig)


# <h3>Observation:</h3> What we can observe from this plot?
# <p>•  We can clearly see, the number of houses which were built after 2000 are more likely to be expensive. So, we can assume because of factors like; they might have more amenities or these houses could be more modernize based on current market trends etc. Nonetheless, we can get a decent picture of the trend here. 
#  

# In[29]:


yr_rem = uni(df["YearRemodAdd"])
data = []
for i in yr_rem:
    data.append(go.Violin(y = df_train[df_train["YearRemodAdd"]==i]["SalePrice"], name = i))
layout = go.Layout(title = 'Sales Price of Renovated Houses', xaxis=dict(title ="Years"), yaxis=dict(title ="Sales Price"))

fig = dict(data = data, layout = layout)
py.iplot(fig)


# <h2> Key Points</h2>
#  <p>• In Boxplots the top and bottom represents 75th and 25th perecentile resp., which can give us an idea of the distribution of the varaible and can be used in side by side display to compare distribution</p>
#  <p>•  Vioplots can be considered as enhancement of the boxplots and the density estimate because it show the density on the y-axis. The advantage of Violin plot is that it can show nuance in the distribution which can not be interpreted properly from the boxplots. However, the boxplots clearly shows the outliers in the data. 
#  <p>• EDA of all the variables is one of the key factors for efficient analysis, as it helps to get a better understanding of our data.</p>
#  
#  
#  <p>•  We can go on and on and explore more characteristics of the Categorical variables but for this kernel, I decided to touch some very basic of Categorical Variable Visualisation.</p>

# <h2> # Let's understand the characteristics of our Numberical Columns</h2>
# <h2> Part 2 EDA: Visualising the Numberical Columns</h2>

# In[30]:


num_col.columns


# In[31]:


df_train.corr()["SalePrice"].sort_values()


# In[32]:


price_uni = uni(df_train["Price"])
qual = uni(df["OverallQual"])
buil_type = uni(df_train["BldgType"])


# In[33]:


data1 = []
for item, colors in zip(buil_type, ["lime","deepskyblue","#d6a5ff", "#639af2", "#fca6da", "#f4d39c", "orange", "#7af9ad"]):
    
    tem_df = df_train[df_train["BldgType"]== item]

    data1.append(go.Scatter(x = tem_df["LotArea"], y = tem_df["SalePrice"], name=item, mode= "markers",opacity = 0.75,
                             marker = dict(line = dict(color = 'black', width = 0.5))))
layout = go.Layout(title = 'Sales Price vs Lot Area ', xaxis = dict(title = 'Lot Area'), 
                   yaxis = dict(title = 'Sales Price'))

fig = go.Figure(data = data1, layout = layout)
py.iplot(fig)


# In[34]:





# In[34]:


df_train.corr()["SalePrice"].sort_values(ascending=False).head(10)


# In[35]:


fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(nrows=2,ncols=2, figsize=(20,8))
sns.scatterplot(x= df_train["LotArea"], y = df_train["SalePrice"], ax= ax1, hue = df_train["Price"])
sns.scatterplot(x= df_train["GrLivArea"], y = df_train["SalePrice"], ax=ax2,hue = df_train["Price"])
sns.scatterplot(x= df_train["GarageArea"], y = df_train["SalePrice"], ax=ax3,hue = df_train["Price"])
sns.scatterplot(x= df_train["TotalBsmtSF"], y = df_train["SalePrice"], ax=ax4, hue = df_train["Price"])
plt.tight_layout()


# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:





# In[36]:


df_train["Utilities"].value_counts()


# In[37]:


plt.figure(figsize = (18,9))
sns.swarmplot(x= df_train["GarageCars"], y = df_train["SalePrice"])


# In[38]:


# Total Houses sold in dollars:
df_train.groupby("YrSold")["SalePrice"].sum()


# In[39]:


# Boxplot of Sales Price Vs Years
#sns.boxplot(data = df_train, x = "YrSold", y ="SalePrice")

trace = go.Box( y = df_train[df_train["YrSold"]==2006]["SalePrice"], 
              name = "2006")
trace1 = go.Box( y = df_train[df_train["YrSold"]==2007]["SalePrice"],
               name = "2007")
trace2 = go.Box( y = df_train[df_train["YrSold"]==2008]["SalePrice"],
               name = "2008")
trace3 = go.Box( y = df_train[df_train["YrSold"]==2009]["SalePrice"],
               name = "2009")
trace4 = go.Box( y = df_train[df_train["YrSold"]==2010]["SalePrice"],
               name = "2010")

layout = go.Layout(title = "Yearly Sale Prices", 
                   yaxis=dict(title = "Sales Price"), 
                  xaxis=dict(title = "Years"))

data = [trace, trace1, trace2, trace3, trace4]

fig = go.Figure(data= data, layout=layout)
py.iplot(fig)


# In[40]:


sns.boxplot(data = df_train, x = "MoSold", y ="SalePrice")


# In[41]:


df_train.head(1)


# In[42]:


sns.boxplot(data = df_train, x = "HouseStyle", y ="SalePrice")


# In[43]:


plt.figure(figsize=(20,15))
sns.boxplot(data = df_train, x = "YearBuilt", y ="SalePrice")
plt.xticks(rotation = "90")


# In[44]:


df["YearBuilt"].value_counts()


# In[45]:





# In[45]:





#  <h1><strong>4 Data Preprocessing</h1></strong>

# <h2><stong>4.a  the Missing Values:</h2></strong>
# 

#  <strong>Step 1:</strong> <p><strong>Identifying the total percentage of the missing values in both the data set, exculding the target variable</p></strong>

# In[45]:


tot_cel = np.product(df.shape)
tot_cel
miss_cel = df.isnull().sum().sum()
total_missing = (miss_cel/tot_cel)*100
print(f"Total percent of missing values in the data is: {round(total_missing)}%")


# <strong>Total percentage of missing values in the both the dataset (training, testing) is 6%. Which is comparitely very less as compare to the problems I have done in past. Hence, we don't have to worry about in finding the most optimal technique to replace the missing values, we can simply impute the missing values with any possible technique without any high impact on the accuracy.</strong>
# 

# <strong>Step 2:</strong> <strong><p>Identifying the columns with missing values:</strong></p>

# In[46]:


print(df.isnull().any().value_counts(),"\nTherefore, the total columns having missing values are 34")


# <h3><strong>Step 3:</h3></strong><p><strong>Creating a table of columns with maximum missing values</p></strong>
# 

# In[47]:


# Data Frame of all the features having missing values with percentage:
total = df_train.isnull().sum().sort_values(ascending= False)

perc = df_train.isnull().sum() / df_train.isnull().count()*100
perc1 = (round(perc, 2).sort_values(ascending = False))

missing_data = pd.concat([total, perc1, df_train.isnull().sum(), df_test.isnull().sum()], axis=1,  keys=["Total Missing Values", "Percantage %", "Missing values in Train", "Missing values in Test"])
missing_data.sort_values(by="Total Missing Values", ascending=False).head(20)


# <h3><strong>Step 4:</h3></strong><p><strong>Visualising the missing values along with there respective coulmns</p></strong>
# 

# In[48]:


plt.figure(figsize=(20,5))
sns.heatmap(df.isnull(),cbar= False, yticklabels=False, cmap = "cividis")

# Ploting the top features based on their missing values
trace1 = go.Bar(x = missing_data.index, y = missing_data["Total Missing Values"].values,
               marker = dict(color = df["YearRemodAdd"],
                            colorscale = "Picnic"))

layout = go.Layout(title="Total Missing Values Plot", 
                   yaxis= dict(title ="Percatnage (%)"))

data = [trace1]

fig = go.Figure(data= data , layout= layout)
py.iplot(fig)


# In[49]:





# In[49]:





# In[49]:





# In[49]:




