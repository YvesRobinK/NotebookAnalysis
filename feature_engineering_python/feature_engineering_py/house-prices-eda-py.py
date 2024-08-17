#!/usr/bin/env python
# coding: utf-8

# <br>
# <section style="font-family:Times New Roman"><br>
#     <strong style= "font-weight: bold; color:#0e1a40; font-size:35px;"> <i>House-Prices, EDA.</strong><br><br>
#     <b style= "font-weight: bold; color:#000000; font-size:20px;"> By: <i>Kheirallah Samaha</b><br><br>
#     <b style= "font-weight: bold; color:#000000; font-size:17px;"> Date: 2023-Jan-09</b><br>
# </section>
# <br>

# 
#  # Table of Contents
# * [1. Introduction](#1.-Introduction)   
# * [2. Load libraries](#2.-Load-libraries)    
# * [3. Import data](#3.-Import-data)   
# * [4. Checking data Types](#4.-Checking-data-Types)   
#     * [4.1 Convert object data type to category](#4.1-Convert-object-data-type-to-category)
# * [5. Missing Data](#5.-Missing-Data)   
#     * [5.1 Missing values distribution](#5.1-Missing-values-distribution)   
#     * [5.2 Removing columns have more than 50% missing values](#5.2-Removing-columns-have-more-than-50pct-missing-values)
# * [6. Correlation matrix](#6.-Correlation-matrix)   
#     * [6.1 Correlation matrix Calculations](#6.1-Correlation-matrix-Calculations)   
#     * [6.2 Correlation matrix Heatmap](#6.2-Correlation-matrix-Heatmap)
# * [7. Feature Engineering](#7.-Feature-Engineering)   
#     * [7.1 Total Square Feet](#7.1-Total-Square-Feet)   
#     * [7.2 Total number of Bathrooms](#7.2-Total-number-of-Bathrooms)   
#     * [7.3 House Age](#7.3-House-Age)   
#     * [7.4 Five 5 Neighborhood](#7.4-Five-5-Neighborhood)   
# * [8. Analyzing the Target variable "SalePrice"](#8.-Analyzing-the-Target-variable-"SalePrice")   
#     * [8.1 Target Variable "SalePrice" Summary](#8.1-Target-Variable-"SalePrice"-Summary)   
#     * [8.2 SalePrice Distribution (Histogram)](#8.2-SalePrice-Distribution-(Histogram))  
#     * [8.3 Checking Normality for "SalePrice" and "SalePriceLog"](#8.3-Checking-Normality-for-"SalePrice"-and-"SalePriceLog")   
#     * [8.4 Build a function to check the Normality in various Variables](#8.4-Build-a-function-to-check-the-Normality-in-various-Variables)
#         * [8.4.1 Normality "SalePrice" and "SalePriceLog"](#8.4.1-Normality-"SalePrice"-and-"SalePriceLog")   
#         * [8.4.2 Normality "SalePrice" and "GrLivArea"](#8.4.2-Normality-"SalePrice"-and-"GrLivArea")        
#         * [8.4.3 Normality "GrLivArea" and "GrLivAreaLog"](#8.4.3-Normality-"GrLivArea"-and-"GrLivAreaLog")   
# * [9. Relationship with numerical variables](#9.-Relationship-with-numerical-variables)
#     * [9.1 Build a function for plotting numerical and categorical variables](#9.1-Build-a-function-for-plotting-numerical-and-categorical-variables)
#     * [9.2 "SalePrice" vs "Total_Square_Feet" by "SaleCondition"](#9.2-"SalePrice"-vs-"Total_Square_Feet"-by-"SaleCondition")  
#     * [9.3 "SalePrice" vs "House_Age"](#9.3-"SalePrice"-vs-"House_Age")
#     * [9.4 "SalePrice" vs "Total_Square_Feet" by "Neighborhood"](#9.4-"SalePrice"-vs-"Total_Square_Feet"-by-"Neighborhood")
# * [10. Relationship with categorical variables](#10.-Relationship-with-categorical-variables)
#     * [10.1 "SalePrice" vs "SaleCondition" by "RoofStyle"](#10.1-"SalePrice"-vs-"SaleCondition"-by-"RoofStyle")
#     * [10.2 "SalePrice" vs "OverallQual" by "SaleCondition"](#10.2-"SalePrice"-vs-"OverallQual"-by-"SaleCondition")
#     * [10.3 "SalePrice" vs "KitchenQual" by "Electrical"](#10.3-"SalePrice"-vs-"KitchenQual"-by-"Electrical")
# * [11. "SalePrice" relationship with various variable using FacetGrid Scatter Plot](#11.-"SalePrice"-relationship-with-various-variable-using-FacetGrid-Scatter-Plot)
#     * [11.1 Build a function for FacetGrid](#11.1-Build-a-function-for-FacetGrid)
#     * [11.2 "SalePrice" vs "Total_Square_Feet" by "Foundation"](#11.2-"SalePrice"-vs-"Total_Square_Feet"-by-"Foundation")
#     * [11.3 "SalePrice" vs "Total_Square_Feet" by "RoofStyle"](#11.3-"SalePrice"-vs-"Total_Square_Feet"-by-"RoofStyle")
#     * [11.4 "SalePrice" vs "Total_Square_Feet" by "Heating"](#11.4-"SalePrice"-vs-"Total_Square_Feet"-by-"Heating")
# * [12. Pair Plot](#12.-Pair-Plot)
#     * [12.1 Select Six variables](#12.1-Select-Six-variables)
#     * [12.2 Plot pairwise relationships of the selected Six variables in the dataset](#12.2-Plot-pairwise-relationships-of-the-selected-Six-variables-in-the-dataset)
# * [13. Thank you!](#13.-Thank-you!)

# <a id='1.-Introduction'></a>
# <h1 style= "font-weight: bold; color:#492634;">1. Introduction</h1>
# 
# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">This notebook demonstrates the usage of the panda’s package in exploring Houses prices on a dataset uploaded to Kaggle and used for competition purpose.<br><br>
# To this day, many notebooks have been published covering a lot of EDA, machine learning, and visualization method, approach and strategies that give the Kaggle community various ideas and ways of thinking about how to deal with this kind of data and unleash the potential hiding behind it.<br><br>
# Although too hard to add new ideas to this competition, I tried to learn how to use the panda’s package to explore and examine some of the relationships between the various features. And consequently, there will be no attempt to make a model because, you know, there are a lot of great notebooks connected to this dataset that have done various types of solid models.</p><br>
# 
# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">So, let us start by loading some libraries...</p>

# <a id='2.-Load-libraries'></a>
# <h1 style= "font-weight: bold; color:#492634;">2. Load libraries</h1>

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# plt.style.use("fivethirtyeight")
plt.style.use("fast")


# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">I have imported the stats module from scipy library just to use shapiro() function to check the Normality of various features.</p>

# In[2]:


my_palette = [
    "#ED0000",
    "#374E55",
    "#BC3C29",
    "#0072B5",
    "#E18727",
    "#20854E",
    "#7876B1",
    "#6F99AD",
    "#FFDC91",
    "#EE4C97",
]


# <a id='3.-Import-data'></a>
# <h1 style= "font-weight: bold; color:#492634;">3. Import data</h1>

# In[3]:


usa_housing_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")


# <a id='4.-Checking-data-Types'>
# </a> <h1 style= "font-weight: bold; color:#492634;">4. Checking data Types</h1>

# In[4]:


usa_housing_df.info(verbose=False)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u> 
#     <dl style= "font-weight: lighter; color:#28140b;">
#         <dt>* 1460 rows.</dt>
#         <dt>* 81 Variables.</dt>
#         <dd>- Three(3) vars float64.</dd>
#         <dd>- 35 vars int64.</dd>
#         <dd>- 43 vars objects.</dd><br>
#     </dl>
#     <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px"> I think converting the variables with an objects data type to a category data type is a good thing to consider for this EDA as well as for building a model, for now, let's check the columns of each data type.</p>
# </section>

# In[5]:


usa_housing_df.select_dtypes("int").columns


# In[6]:


usa_housing_df.select_dtypes("float").columns


# In[7]:


usa_housing_df.select_dtypes("object").columns


# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:14px"> Let's convert object data type to category data type.</p>

# <a id='4.1-Convert-object-data-type-to-category'></a>
# <h2 style= "font-weight: bold; color:#492634;">4.1 Convert object data type to category</h2>

# In[8]:


usa_housing_df[
    usa_housing_df.select_dtypes("object").columns
] = usa_housing_df.select_dtypes("object").astype("category")
usa_housing_df.select_dtypes("category").columns


# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">I'm going to select the features that I have considered important features that to not have a wide table as an output of describe function.</p>

# In[9]:


cols_sum_stat = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "FullBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "SalePrice",
]

usa_housing_df[cols_sum_stat].describe().round()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u> 
#     <dl style= "font-weight: lighter; color:#28140b;">
#         <dt>* <b>Count</b> row tells us that there are missing values in this dataset, "LotFrontage" and "MasVnrArea" and apparently other features.</dt>
#         <dt>* <b>Many features</b> have zero(0) as min values, so we have to remember that while doing EDA .</dt>
#         <dt>* <b>SalePrice</b> ranges from 34900 to 755000, mean of 180921 and median or 50% of 163000 we have to check the <b>NORMALITY!</b></dt>
#         <dt>* <b>GrLivArea</b> ranges from 334 to 5642.</dt>
#         <dt>* <b>LotArea</b> ranges from 1300 to 215245.</dt>
#         <dt>* <b>BedroomAbvGr</b> ranges from zero(0) to eight(8).</dt><br>
#     </dl>
# </section>
# 
# 
# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">Let's check the proportion of missing values</p>

# <a id='5.-Missing-Data'></a>
# <h1 style= "font-weight: bold; color:#24273A;">5. Missing Data</h1>

# In[10]:


na_usa_housing_df = usa_housing_df.isna().sum().to_frame().reset_index()
na_usa_housing_df = na_usa_housing_df.rename(
    columns={"index": "variable_name", 0: "total_na"}
)
na_usa_housing_df = (
    na_usa_housing_df.query("total_na > 0")
    .copy()
    .sort_values(by="total_na", ascending=False)
).copy()

na_usa_housing_df.nlargest(columns="total_na", n=5)


# <a id='5.1-Missing-values-distribution'></a>
# <h2 style= "font-weight: bold; color:#24273A;">5.1 Missing values distribution</h2>

# In[11]:


fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(
    data=na_usa_housing_df,
    x="total_na",
    y="variable_name",
    palette="Blues_r",
    orient="h",
)

values = ax.containers[0].datavalues
labels = ["{:0.1f}%".format(val) for val in values / 1460 * 100]
ax.bar_label(ax.containers[0], labels=labels, size=8, padding=3)
ax.set_title(
    "Percentage of Missing Values occured in each Variable",
    fontsize=15,
    loc="left",
    pad=20,
)
ax.tick_params(axis="x", labelsize=9, pad=10)
ax.tick_params(axis="y", labelsize=9, pad=10)
ax.set_xlabel("Total NAs", fontsize=14)
ax.set_ylabel("Variable", fontsize=14)

plt.show()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u>
#     <dl style= "font-weight: lighter; color:#28140b;">
#         <dt>* <b>PoolQC:</b><strong style= "color:#ee0000;"> 99.5%</strong></dt>
#         <dt>* <b>MiscFeature:</b><strong style= "color:#ee0000;"> 96.3%</strong></dt>
#         <dt>* <b>Alley:</b><strong style= "color:#ee0000;"> 93.8%</strong></dt>
#         <dt>* <b>Fence:</b><strong style= "color:#ee0000;"> 80.8%</strong></dt>
#         <dt>* <b>FireplaceQu:</b><strong style= "color:#e4d00a;"> 47.3%</strong></dt>
#         <dt>* <b>LotFrontage:</b><strong style= "color:#228b22;"> 17.7%</strong></dt><br>
#     </dl>
# </section>
# 
# <p style="font-family:Lucida Sans Typewriter;font-size:13px;"> We are going to remove the columns that have more than <b>50%</b> missing values <b>'PoolQC', 'MiscFeature', 'Alley', and 'Fence'</b></p>

# <a id="5.2-Removing-columns-have-more-than-50pct-missing-values"></a>
# <h2 style= "font-weight: bold; color:#24273A;">5.2 Removing columns have more than 50&percnt; missing values</h2>

# In[12]:


cols_na_more_than_50_percent = (
    na_usa_housing_df.query("total_na > 700").variable_name.copy().to_list()
)
cols_na_more_than_50_percent


# In[13]:


usa_housing_df = usa_housing_df.drop(cols_na_more_than_50_percent, axis=1)


# In[14]:


usa_housing_df.info(verbose=False)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u>
#     <dl style= "font-weight: lighter; color:#28140b;">
#         <dt>* 1460 rows.</dt>
#         <dt>* 77 Variables.</dt>
#         <dd>- three(3) vars float64.</dd>
#         <dd>- 35 vars int64.</dd>
#         <dd>- 39 vars category.</dd><br>
#     </dl>
#      <p style= "font-size:13px;">This figure is a result of removing the variables with more than 50% NAs: 'PoolQC', 'MiscFeature', 'Alley', and 'Fence'</p>   
# </section>

# <a id='6.-Correlation-matrix'></a>
# <h1 style= "font-weight: bold; color:#005582;">6. Correlation matrix</h1>

# In[15]:


cols_corr_out = [
    "Id",
    "MSSubClass",
    "OverallCond",
    "Fireplaces",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
    "BsmtHalfBath",
    "BsmtFinSF2",
    "LowQualFinSF",
    "EnclosedPorch",
]


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Note: </u><br><br>
#     <p style= "font-size:13px;">To not have a crowded Correlation Matrix, we are to eliminate the features in "cols_corr_out" list, i think it will be more readable!</p>   
# </section>

# <a id='6.1-Correlation-matrix-Calculations'></a>
# <h2 style= "font-weight: bold; color:#005582;">6.1 Correlation matrix Calculations</h2>

# In[16]:


corr_mat = (
    usa_housing_df.loc[:, ~usa_housing_df.columns.isin(cols_corr_out)]
    .select_dtypes("int64")
    .corr()
)
corr_mat[["SalePrice"]].sort_values("SalePrice", ascending=False).style.format(
    precision=3
).background_gradient(cmap="Greys")


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <p style= "color:#000000; font-size:13px;"> Correlation between target variable and other variables: </p>
#     <dl>
#         <dt><b>OverallQual:</b><strong style= "color:#0047ab;"> 0.790</strong></dt>
#         <dt><b>GrLivArea:</b><strong style= "color:#0047ab;">   0.708</strong></dt>
#         <dt><b>GarageCars:</b><strong style= "color:#0047ab;">  0.640</strong></dt>
#         <dt><b>GarageArea:</b><strong style= "color:#0047ab;">  0.623</strong></dt>
#         <dt><b>TotalBsmtSF:</b><strong style= "color:#0047ab;"> 0.613</strong></dt>
#         <dt><b>1stFlrSF:</b><strong style= "color:#0047ab;">    0.605</strong></dt>
#         <dt><b>FullBath:</b><strong style= "color:#0047ab;">    0.560</strong></dt>
#         <dt><b>TotRmsAbvGrd:</b><strong style= "color:#0047ab;">0.533</strong></dt>
#         <dt><b>YearBuilt:</b><strong style= "color:#0047ab;">   0.522</strong></dt>
#         <dt><b>YearRemodAdd:</b><strong style= "color:#0047ab;">0.507</strong></dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>

# <a id='6.2-Correlation-matrix-Heatmap'></a>
# <h2 style= "font-weight: bold; color:#005582;">6.2 Correlation matrix Heatmap</h2>

# In[17]:


mask = np.triu(np.ones_like(corr_mat.corr()))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr_mat,
    square=False,
    cmap="Greys",
    mask=mask,
    annot=True,
    annot_kws={"size": 8},
    fmt=".2f",
)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()


# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <p style= "color:#000000; font-size:13px;"> OK! this Matrix has a lot of information to give, many variables are correlated together, and most of them are practically obvious to be correlated such like:</p>
#     <dl>
#         <dt>* "1stFlSf" and "TotalBsmtSF": <strong style= "color:#a11826;"> 0.82</strong></dt><br>
#         <dt>* "TotalRmsAbvGrd" and "GrLivArea": <strong style= "color:#a11826;"> 0.83</strong></dt><br>
#         <dt>* "GarageArea" and  "GarageCars": <strong style= "color:#a11826;"> 0.88</strong></dt><br>
#         <dt>* "GrLivArea" and  "2ndFlrSF": <strong style= "color:#a11826;"> 0.69</strong></dt><br>
#         <dt>* "TotalRmsAbvGrd" and  "BedroomAbvGr": <strong style= "color:#a11826;"> 0.68</strong></dt><br>
#         <dt>* "BsmtFullBath" and  "BsmtFinSF1": <strong style= "color:#a11826;"> 0.65</strong></dt><br>
#         <dt>* "FullBath" and  "GrLivArea": <strong style= "color:#a11826;"> 0.63</strong></dt><br>
#         <dt>* "YearBuilt" and  "OverallQual": <strong style= "color:#a11826;"> 0.57</strong></dt><br>
#         <dt>* "YearRemodAdd" and  "YearBuilt": <strong style= "color:#a11826;"> 0.59</strong></dt><br>
#         <dt>* "FullBath" and  "YearBuilt":<strong style= "color:#a11826;">  0.59</strong></dt><br>
#     </dl>
#      <p style= "font-size:13px;">It is a very informative Matrix, if we going to build a model, in any way let's move to Feature-Engineering section to see how we're going to manage it!</p>   
# </section>

# <a id='7.-Feature-Engineering'></a>
# <h2 style= "font-weight: bold; color:#005582;">7. Feature-Engineering</h2>

# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">Although I'm not going to build a model in this notebook, I will perform feature engineering just as a preprocessing task in case I decide in the future to build a model, but in any way, this section tends to be short and simple!!</p>

# <a id='7.1-Total-Square-Feet'></a>
# <h2 style= "font-weight: bold; color:#005582;">7.1 Total Square Feet</h2>

# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">We all know that the total living area is very important factor when people decide to buy a house, therefore, I will add up the living areas that above and below ground level.</p>

# In[18]:


usa_housing_df["Total_Square_Feet"] = (
    usa_housing_df["GrLivArea"] + usa_housing_df["TotalBsmtSF"]
)
usa_housing_df["Total_Square_Feet"]


# <a id='7.2-Total-number-of-Bathrooms'></a>
# <h2 style= "font-weight: bold; color:#005582;">7.2 Total number of Bathrooms</h2>
# 

# 
# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">There are four(4) Features of bathrooms, Full bathrooms, and Half Bathrooms, however, I will consider them as one type of "Bathroom" and add them together!</p>

# In[19]:


usa_housing_df["Total_number_of_Bathrooms"] = (
    usa_housing_df["FullBath"]
    + (usa_housing_df["HalfBath"])
    + usa_housing_df["BsmtFullBath"]
    + (usa_housing_df["BsmtHalfBath"])
)

usa_housing_df["Total_number_of_Bathrooms"].describe().round()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* min number of bathrooms: <strong style= "color:#a11826;"> one(1)</strong></dt><br>
#         <dt>* max number of bathrooms: <strong style= "color:#a11826;"> Six(6)</strong></dt><br>
#     </dl>
#      <p style= "font-size:13px;">It is OK! let's move on...</p>   
# </section>

# <a id='7.3-House-Age'></a>
# <h2 style= "font-weight: bold; color:#005582;">7.3 House Age</h2>

# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">We can observe from the correlation Matrix that there are two(2) variables that are highly connected to the Age of a house; YearBuilt and YearRemodAdd. YearRemodAdd is equal to YearBuilt if there has been no house renovation, any way, i will be using YearBuilt and YearSold to find out the House Age, however I will create another house Age variable using YearRemodAdd.</p>

# In[20]:


usa_housing_df["House_Age"] = usa_housing_df["YrSold"] - usa_housing_df["YearBuilt"]

usa_housing_df["House_Age_YearRemodAdd"] = (
    usa_housing_df["YrSold"] - usa_housing_df["YearRemodAdd"]
)

usa_housing_df[["House_Age"]].describe().round(), usa_housing_df[
    ["House_Age_YearRemodAdd"]
].describe().round()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* mean house age: <strong style= "color:#a11826;"> 23.0</strong></dt><br>
#         <dt>* median house age: <strong style= "color:#a11826;"> 35.0</strong></dt><br>
#         <dt>* min house age: <strong style= "color:#ee0000;"> Zero(0)</strong></dt><br>
#         <dt>* max house age: <strong style= "color:#a11826;"> 136.0</strong></dt><br>
#     </dl>
#      <p style= "font-size:13px;">Let us check the min of -One(-1) in House_Age_YearRemodAdd. <br>I assume this house has been sold while the house is under construction or renovation. let us check!</p>   
# </section>

# In[21]:


usa_housing_df[usa_housing_df["House_Age"] == 0].loc[
    :, ["YearBuilt", "YearRemodAdd", "YrSold", "House_Age", "House_Age_YearRemodAdd"]
].sort_values("House_Age_YearRemodAdd").head()


# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">We do not know whether this house with -One(-1) has been sold while it was under renovation, or is a data entry issue.<br> OK Let us move on...</p>

# <a id='7.4-Five-5-Neighborhood'></a>
# <h2 style= "font-weight: bold; color:#005582;">7.4 Five(5) Neighborhood
# </h2>

# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Note: </u><br><br>
#      <p style= "font-size:13px;">For plotting purpose, I am going to select five(5) out of the observations in "Neighborhood" feature, and lump the other observations under one category named as "other".</p>   
# </section>

# In[22]:


# (
#     usa_housing_df[["Neighborhood"]]
#     .astype("object")
#     .value_counts()
#     .nlargest(20)
#     .reset_index()
#     .loc[:, "Neighborhood"]
#     .to_list()
# )

top_5_Neighborhood = [
    "Edwards",
    "NridgHt",
    "NoRidge",
    "OldTown",
    "CollgCr"
]


# In[23]:


usa_housing_df["Neighborhood"] = np.where(
    usa_housing_df["Neighborhood"].isin(top_5_Neighborhood),
    usa_housing_df["Neighborhood"],
    "Other",
)


# <a id='8.-Analyzing-the-Target-variable-"SalePrice"'></a>
# <h1 style= "font-weight: bold; color:#000000; ">8. Analyzing the Target variable "SalePrice"</h1>

# <a id='8.1-Target-Variable-"SalePrice"-Summary'></a>
# <h2 style= "font-weight: bold; color:#770099; ">8.1 Target Variable "SalePrice" Summary</h2>

# In[24]:


usa_housing_df[["SalePrice"]].describe().round()


# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">Let's use Histogram plot to have more interpretation! </p>

# In[25]:


mean_SalePrice = usa_housing_df[["SalePrice"]].mean().squeeze()
median_SalePrice = usa_housing_df[["SalePrice"]].median().squeeze()

plt.figure(figsize=(10, 5))
sns.set_context("paper")

histplt = sns.histplot(
    data=usa_housing_df,
    x="SalePrice",
    color="#4f758f",
    bins=60,
    alpha=0.5,
    lw=2,
)
histplt.set_title("SalePrice Distribution", fontsize=12)
histplt.set_xlabel("SalePrice", fontsize=12)

plt.axvline(x=mean_SalePrice, color="#14967f", ls="--", lw=1.5)
plt.axvline(x=median_SalePrice, color="#9b0f33", ls="--", lw=1.5)
plt.text(mean_SalePrice + 5000, 175, "Mean SalePrice", fontsize=9, color="#14967f")
plt.text(
    median_SalePrice - 115000, 175, "Median SalePrice", fontsize=9, color="#9b0f33"
)
histplt.xaxis.set_major_formatter(ticker.EngFormatter())
plt.ylim(0, 200)
plt.show()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* Data is positively skewed. The mean is greater than the median</dt><br>
#         <dt>* The outliers located in the right side, although we do not know how many outliers are there. <strong style= "color:#a11826;"> 35.0</strong></dt><br>
#         <dt>* We may use log transformation to reduces the skewness<strong style= "color:#ee0000;"></strong></dt><br>
#         <dt>* The majority of data points are clustered between 100K and 200K <strong style= "color:#a11826;"> 136.0</strong></dt><br>
#     </dl>
#      <p style= "font-size:13px;">let's check the the Normality...</p>   
# </section>

# <a id='8.3 Checking Normality for "SalePrice" and "SalePriceLog"'></a>
# <h2 style= "font-weight: bold; color:#770099; ">8.3 Checking Normality for "SalePrice" and "SalePriceLog"</h2>

# In[26]:


usa_housing_df["SalePriceLog"] = np.log(usa_housing_df["SalePrice"])
usa_housing_df["GrLivAreaLog"] = np.log(usa_housing_df["GrLivArea"])
usa_housing_df["LotAreaLog"] = np.log(usa_housing_df["LotArea"])
usa_housing_df[
    ["SalePrice", "SalePriceLog", "GrLivArea", "GrLivAreaLog", "LotArea", "LotAreaLog"]
]


# <p style="font-family:Lucida Sans Typewriter;font-size:13px;">The above code is to use log transformation that is to see the difference in the two distributions.<br> first we are going to build a function!</p>

# <a id='8.4-Build-a-function-to-check-the-Normality-in-various-Variables'></a>
# <h2 style= "font-weight: bold; color:#770099; ">8.4 Build a function to check the Normality in various Variables</h2>

# In[27]:


def prob_plot(tq_var1, tq_var2):

    plt.subplot(2, 2, 1)
    tq_var1_hist = sns.histplot(
        usa_housing_df[tq_var1], kde=True, color="#03396c", line_kws={"lw": 2}
    )
    tq_var1_hist.xaxis.set_major_formatter(ticker.EngFormatter())
    tq_var1_hist.set_title("(" + tq_var1 + ") " + "Distributions", fontsize=10)
    ######################
    ax = plt.subplot(2, 2, 2)
    stats.probplot(usa_housing_df[tq_var1], plot=plt)
    ax.get_lines()[0].set_marker("o")
    ax.get_lines()[0].set_markerfacecolor("#343d46")
    ax.get_lines()[0].set_markeredgecolor("#343d46")
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_color("#f01c58")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_title("Probability Plot " + "(" + tq_var1 + ")", fontsize=10)
    #####################
    plt.subplot(2, 2, 3)
    tq_var1_hist = sns.histplot(
        usa_housing_df[tq_var2], kde=True, color="#03396c", line_kws={"lw": 2}
    )
    tq_var1_hist.xaxis.set_major_formatter(ticker.EngFormatter())
    tq_var1_hist.set_title("(" + tq_var2 + ") " + "Distributions", fontsize=10)
    #####################
    ax = plt.subplot(2, 2, 4)
    stats.probplot(usa_housing_df[tq_var2], plot=plt)
    ax.get_lines()[0].set_marker("o")
    ax.get_lines()[0].set_markerfacecolor("#343d46")
    ax.get_lines()[0].set_markeredgecolor("#343d46")
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_linewidth(3.0)
    ax.get_lines()[1].set_color("#f01c58")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_title("Probability Plot " + "(" + tq_var2 + ")", fontsize=10)

    plt.tight_layout(pad=1)
    shapiro_test_tq_var1 = stats.shapiro(usa_housing_df[tq_var1])
    shapiro_test_tq_var2 = stats.shapiro(usa_housing_df[tq_var2])
    return print("(" + tq_var1 + ")--->", shapiro_test_tq_var1), print(
        "(" + tq_var2 + ")--->", shapiro_test_tq_var2
    )


# <a id='8.4.1-Normality-"SalePrice"-and-"SalePriceLog"'></a>
# <h3 style= "font-weight: bold; color:#770099; ">8.4.1 Normality "SalePrice" and "SalePriceLog"</h3>

# In[28]:


plt.figure(figsize=(9, 6))
prob_plot("SalePrice", "SalePriceLog")
plt.show()


# 
# <h3 style= "font-weight: light; color:#36802d; font-size:14px;"> * Shapiro Result for SalePrice || Statistic=0.869671642780304, pvalue=3.206247534576162e-33) and </h3>
# <h3 style= "font-weight: light; color:#36802d; font-size:14px;">  ** Shapiro Result for SalePriceLog || Statistic=0.9912067651748657, pvalue=1.1490678986092462e-07) </h3><br>
# 
# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">** We have done a log transformation "SalePriceLog" to the target variable "SalePrice" which reduces the Skewness. And we can say that it became more or less a normally distributed variable, however Since the p-value in Shapiro Result is less than .05, so we can reject the null hypothesis of the Shapiro-Wilk test, this means we have a piece of sufficient evidence to consider that the sample data does not come from a normal distribution.</p>

# <a id='8.4.2-Normality-"SalePrice"-and-"GrLivArea"'></a>
# <h3 style= "font-weight: bold; color:#770099; ">8.4.2 Normality "SalePrice" and "GrLivArea"</h3>

# In[29]:


plt.figure(figsize=(9, 6))
prob_plot("SalePrice", "GrLivArea")
plt.show()


# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">** do not be confused; here I wanted to see the distributions of "SalePrice" and "GrLivArea" (remember the correlation coefficient is 0.708.<br> Again, the p-value in Shapiro Result is less than .05 for both variables, so again, we can reject the null hypothesis.</p>

# <a id='8.4.3-Normality-"GrLivArea"-and-"GrLivAreaLog"'></a>
# <h3 style= "font-weight: bold; color:#770099; ">8.4.3 Normality "GrLivArea" and "GrLivAreaLog"</h3>

# In[30]:


plt.figure(figsize=(9, 6))
prob_plot("GrLivArea", "GrLivAreaLog")
plt.show()


# 
# <h3 style= "font-weight: light; color:#36802d; font-size:14px;"> * Shapiro Result for GrLivArea || statistic=0.9279828071594238, pvalue=6.598091159538852e-26 and </h3>
# <h3 style= "font-weight: light; color:#36802d; font-size:14px;">  ** Shapiro Result for GrLivAreaLog || statistic=0.9960905313491821, pvalue=0.0008570468635298312</h3><br>
# 
# <p style= "font-family:Lucida Sans Typewriter; color:#492634; font-size:13px">** Somehow, we have the same distribution of SlaePrice variable, however the pvalue of GrLivAreaLog is somehow improved but not enough to consider it as a normal</p>

# 

# <a id='9.-Relationship-with-numerical-variables'></a>
# <h2 style= "font-weight: bold; color:#449933; ">9. Relationship with numerical variables</h2>

# <a id='9.1-Build-a-function-for-plotting-numerical-and-categorical-variables'></a>
# <h2 style= "font-weight: bold; color:#449933; ">9.1 Build a function for plotting numerical and categorical variables</h2>

# In[31]:


date_var = ["YearRemodAdd", "YearBuilt", "YrSold", "MoSold"]


def to_plot(df, x_var, y_var, hue_var=None, palette_set=None):

    if pd.api.types.is_numeric_dtype(df[x_var]):
        if hue_var is None or palette_set is None:
            lmplt = sns.lmplot(
                data=df,
                x=x_var,
                y=y_var,
                ci=0,
                height=5,
                aspect=1.8,
                legend=False,
                # hue=hue_var,
                # palette=palette_set,
                line_kws={"lw": 2, "linestyle": "--", "alpha": 0.8},
                scatter_kws={"marker": "o", "s": 20, "alpha": 0.7},
            )

            lmplt.set_xlabels(x_var, fontsize=14)
            lmplt.set_ylabels(y_var, fontsize=14)
            lmplt.set_titles("Sale Price by " + str(x_var), fontsize=16)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            # plt.legend().set_visible(False)

            if x_var in date_var:
                for ax in lmplt.axes.flat:
                    ax.yaxis.set_major_formatter(ticker.EngFormatter())
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
            else:
                for ax in lmplt.axes.flat:
                    ax.yaxis.set_major_formatter(ticker.EngFormatter())
                    ax.xaxis.set_major_formatter(ticker.EngFormatter())

        else:
            lmplt = sns.lmplot(
                data=df,
                x=x_var,
                y=y_var,
                ci=0,
                height=5,
                aspect=1.8,
                legend=False,
                hue=hue_var,
                palette=palette_set,
                line_kws={"lw": 2, "linestyle": "--", "alpha": 0.8},
                scatter_kws={"marker": "o", "s": 20, "alpha": 0.7},
            )

            lmplt.set_xlabels(x_var, fontsize=14)
            lmplt.set_ylabels(y_var, fontsize=14)
            lmplt.set_titles("Sale Price by " + str(x_var), fontsize=16)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(fontsize=10, title=hue_var, title_fontsize=10)

            if x_var in date_var:
                for ax in lmplt.axes.flat:
                    ax.yaxis.set_major_formatter(ticker.EngFormatter())
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
            else:
                for ax in lmplt.axes.flat:
                    ax.yaxis.set_major_formatter(ticker.EngFormatter())
                    ax.xaxis.set_major_formatter(ticker.EngFormatter())

    elif pd.api.types.is_categorical_dtype(df[x_var]):

        plt.figure(figsize=(10, 5))
        bxplt = sns.boxplot(
            data=df,
            x=x_var,
            y=y_var,
            hue=hue_var,
            palette=palette_set,
            saturation=0.9,
            width=0.9,
            fliersize=2,
            linewidth=1,
            whis=2,
        )

        bxplt.set_ylabel(y_var, fontsize=14)
        bxplt.set_xlabel(x_var, fontsize=14)
        bxplt.set_title("Sale Price by " + str(x_var), fontsize=16)
        bxplt.yaxis.set_major_formatter(ticker.EngFormatter())
        plt.legend(fontsize=10, title=hue_var, title_fontsize=10)

    else:

        print("Please try again!")

    return plt.show()


# <a id='9.2-"SalePrice"-vs-"Total_Square_Feet"-by-"SaleCondition"'></a>
# <h2 style= "font-weight: bold; color:#449933; ">9.2 "SalePrice" vs "Total_Square_Feet" by "SaleCondition"</h2>

# In[32]:


to_plot(
    df=usa_housing_df,
    x_var="Total_Square_Feet",
    y_var="SalePrice",
    hue_var="SaleCondition",
    palette_set=my_palette,
)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">SaleCondition: Condition of sale: </u> 
#     <ul style= "font-weight: lighter; color:#28140b;">
#         <li>Normal: Normal Sale.</li>
#         <li>Abnorml: Abnormal Sale -  trade, foreclosure, short sale.</li>
#         <li>AdjLand: Adjoining Land Purchase).</li>
#         <li>Alloca: Allocation - two linked properties with separate deeds, typically condo with a garage unit.</li>
#         <li>Family: Sale between family members.</li>
#         <li>Family: Partial: Home was not completed when last assessed (associated with New Homes).</li><br>
#     </ul>
# </section>
# <br>
# 
# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* The majority of sales condition is <b>"Normal"</b> sales condition</dt><br>
#         <dt>* We have extreme two(2) outliers related to Partial sales condition that we need to check them</dt><br>
#         <dt>* Mainly house area Square Feet is one of the important features to determine the sale price </dt><br>
#         <dt>* We know from the correlation matrix that <b>GrLivAra</b> and <b>SalePrice</b> vars are highly correlated <b>0.709</b></dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>

# <a id='9.3-"SalePrice"-vs-"House_Age"'></a>
# <h2 style= "font-weight: bold; color:#449933; ">9.3 "SalePrice"-vs-"House_Age"</h2>

# In[33]:


to_plot(df=usa_housing_df, x_var="House_Age", y_var="SalePrice")


# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* We can notice kind of clusters</dt>
#         <dd>- From ~0 to ~20.</dd>
#         <dd>- From ~25 to ~60.</dd>
#         <dd>- From ~80 to ~100.</dd><br>
#         <dt>* We have many data points above ~100 years with a high price, that is assumingly connected to another correlated feature like:</dt>
#         <dd>- OverallQual-SalePrice: 0.791</dd>
#         <dd>- GrLivAra-SalePrice: 0.709</dd>
#         <dd>- YearBuilt-SalePrice: 0.523</dd>
#         <dd>- YearRemodAdd-SalePrice: 0.507</dd>
#     </dl>
#      <p style= "font-size:13px;"> Now! let us check the Neighborhood var and its relationship with the Sale Price</p>   
# </section>

# <a id='9.4-"SalePrice"-vs-"Total_Square_Feet"-by-"Neighborhood"'></a>
# <h2 style= "font-weight: bold; color:#449933; ">9.4 "SalePrice" vs "Total_Square_Feet" by "Neighborhood"</h2>

# In[34]:


to_plot(
    df=usa_housing_df,
    x_var="Total_Square_Feet",
    y_var="SalePrice",
    hue_var="Neighborhood",
    palette_set=my_palette,
)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* It is clear that the Neighborhood feature has an important role to play in determining the sale price</dt><br>
#         <dt>* The two(2) extreme data points are related to <b>Edwards</b> Neighborhood category</dt><br>
#         <dt>* The data points representing the highest price are related to <b>NoRidge</b> Neighborhood category </dt><br>
#     </dl>
#      <p style= "font-size:13px;">Let us create a table to have more information! </p>   
# </section>

# In[35]:


usa_housing_df[
    [
        "SalePrice",
        "GrLivArea",
        "LotArea",
        "PoolArea",
        "Total_Square_Feet",
        "YearBuilt",
        "FullBath",
        "KitchenQual",
        "OverallQual",
        "ExterCond",
        "MSSubClass",
        "Neighborhood",
        "Utilities",
        "Foundation",
    ]
].nlargest(columns="Total_Square_Feet", n=10).style.background_gradient()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* Through the above table we see that mainly the <b>NoRidge</b> Neighborhood category has defeats the <b>Edwards</b> category, although the <b>PoolArea</b> and <b>FullBath</b> have a little influence over that matter as well.</dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>

# <a id='10.-Relationship-with-categorical-variables'></a>
# <h2 style= "font-weight: bold; color:#115599; ">10. Relationship with categorical variables</h2>

# <a id='10.1-"SalePrice"-vs-"SaleCondition"-by-"RoofStyle"'></a>
# <h3 style= "font-weight: bold; color:#115599; ">10.1 "SalePrice" vs "SaleCondition" by "RoofStyle"</h3>

# In[36]:


to_plot(
    df=usa_housing_df,
    x_var="SaleCondition",
    y_var="SalePrice",
    hue_var="RoofStyle",
    palette_set=my_palette,
)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* Regardless of the "Normal" Sales Condition proportion, we can notice that "Gable" and "Hip" roof types are the most selected Roof Type, although we did not attempt to find the proportions of each type</dt><br>
#     </dl>   
# </section>

# <a id='10.2-"SalePrice"-vs-"OverallQual"-by-"SaleCondition"'></a>
# <h3 style= "font-weight: bold; color:#115599; ">10.2 "SalePrice" vs "OverallQual" by "SaleCondition"</h3>

# In[37]:


usa_housing_df[["OverallQual"]].value_counts(normalize=True).round(
    3
).reset_index().rename(columns={0: "Percentage"}).style.background_gradient()


# In[38]:


to_plot(
    df=usa_housing_df,
    x_var="OverallQual",
    y_var="SalePrice",
)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* About 70% of the data points in the <b>OverallQual</b> var are <b>five(5), Six(6) and Seven(7) collectively</b></dt><br>
#         <dt>* We can see without doubt the strong relationship between the <b>OverallQual</b> Feature and the target variable <b>SalePrice</b>, especially from Six(6) up to 10.</dt><br>
#     </dl>   
# </section>

# <a id='10.3-"SalePrice"-vs-"KitchenQual"-by-"Electrical"'></a>
# <h3 style= "font-weight: bold; color:#115599; ">10.3 "SalePrice" vs "KitchenQual" by "Electrical"</h3>

# In[39]:


to_plot(
    df=usa_housing_df,
    x_var="KitchenQual",
    y_var="SalePrice",
    hue_var="Electrical",
    palette_set=my_palette,
)


# 
# <br><p style="font-family:Consolas;font-size:14px;"> Note: We can see that excellent and Good Kitchens Quality have equipped with Standard Circuit Breaker based Power Protection System, and we know that the Kitchen quality is one of the most important feature that all buyer looking for!</p>
# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations, Electrical system: </u><br><br>
#     <ul style= "font-weight: lighter; color:#28140b;">
#         <li>SBrkr: Standard Circuit Breakers &amp; Romex.</li>
#         <li>FuseA: Fuse Box over 60 AMP and all Romex wiring (Average).</li>
#         <li>FuseF: 60 AMP Fuse Box and mostly Romex wiring (Fair).</li>
#         <li>FuseP: 60 AMP Fuse Box and mostly knob &amp; tube wiring (poor.</li>
#         <li>Mix: Mixed.</li><br>
#     </ul>
#     <p style="font-size:13px;"> Standard Circuit Breaker based power protection system is more desirable by the buyer, and if we go deeply into the analysis we would find that all modern houses have equipped with this kind of power protection system!</p>
# </section>
# 

# <a id='11.-"SalePrice"-relationship-with-various-variable-using-FacetGrid-Scatter-Plot'></a>
# <h1 style= "font-weight: bold; color:#005500; ">11. "SalePrice" relationship with various variable using FacetGrid Scatter Plot</h1>

# <a id='11.1-Build-a-function-for-FacetGrid'></a>
# <h2 style= "font-weight: bold; color:#005500; ">11.1 Build a function for FacetGrid</h2>

# In[40]:


def grid_plot(df, col_var, x_var, y_var, paint):

    f_grid = sns.FacetGrid(
        df,
        col=col_var,
        col_wrap=3,
        height=3,
        aspect=1.3,
        sharex=True,
        sharey=True,
    )

    f_grid.map_dataframe(
        sns.scatterplot,
        x=x_var,
        y=y_var,
        color=paint,
        alpha=0.5,
    ).set(ylim=(-0, None), xlim=(-0, None))

    f_grid.set_titles(col_template="{col_var} ({col_name})", size=10, c="#386087")
    f_grid.set_yticklabels(fontsize=8)
    f_grid.set_xticklabels(fontsize=8)
    for ax in f_grid.axes.flat:
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
        ax.xaxis.set_major_formatter(ticker.EngFormatter())

    f_grid.set_xlabels(label=x_var + " (square feet)", size=10)
    f_grid.set_ylabels(label="$ " + y_var, size=10)
    return plt.show()


# <a id='11.2-"SalePrice"-vs-"Total_Square_Feet"-by-"Foundation"'></a>
# <h2 style= "font-weight: bold; color:#005500; ">11.2 "SalePrice" vs "Total_Square_Feet" by "Foundation"</h2>

# In[41]:


grid_plot(
    usa_housing_df,
    col_var="Foundation",
    x_var="Total_Square_Feet",
    y_var="SalePrice",
    paint="#008080",
)


# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Foundation: Type of foundation </u><br><br>
#     <dl>
#         <dt>* BrkTil: Brick and Tile</dt><br>
#         <dt>* CBlock: Cinder Block</dt><br>
#         <dt>* PConc: poured concrete</dt><br>
#         <dt>* Slab: Slab</dt><br>
#         <dt>* Stone: Stone</dt><br>
#         <dt>* Wood: Wood</dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>
# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* All houses which their prices are above $500K are built using <b>"PConc: poured concrete"</b>, although they are few, it is the most desirable type of foundation in general.</dt><br>
#         <dt>* <b>"CBlock: Cinder Block"</b> Type of foundation is the second most desirable! followed by <b>"BrkTil: Brick and Tile"</b> </dt><br>
#         <dt>* The <b>"Slab", "Stone" and "Wood"</b> types of foundation almost do not have any contribution, as if they do not exist!</dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>

# <a id='11.3-"SalePrice"-vs-"Total_Square_Feet"-by-"RoofStyle"'></a>
# <h2 style= "font-weight: bold; color:#005500; ">11.3 "SalePrice" vs "Total_Square_Feet" by "RoofStyle"</h2>

# In[42]:


grid_plot(
    usa_housing_df,
    col_var="RoofStyle",
    x_var="Total_Square_Feet",
    y_var="SalePrice",
    paint="orange",
)


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* We can say the same about "Gable" Type of roof and "Hip" as they are the most desirable type of roof</dt><br>
#         <dt>* The <b>"Flat", "Gambrel" "Shed" and "Mansard"</b> types of foundation almost do not exist</dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>

# <a id='11.4-"SalePrice"-vs-"Total_Square_Feet"-by-"Heating"'></a>
# <h2 style= "font-weight: bold; color:#005500; ">11.4 "SalePrice" vs "Total_Square_Feet" by "Heating"</h2>

# In[43]:


grid_plot(
    usa_housing_df,
    col_var="Heating",
    x_var="Total_Square_Feet",
    y_var="SalePrice",
    paint="#064273",
)


# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Heating: Type of heating </u><br><br>
#     <dl>
#         <dt>* Floor: Floor Furnace</dt><br>
#         <dt>* GasA: Gas forced warm air furnace</dt><br>
#         <dt>* GasW: Gas hot water or steam heat</dt><br>
#         <dt>* Grav: Gravity furnace	</dt><br>
#         <dt>* OthW: Hot water or steam heat other than gas</dt><br>
#         <dt>* Wall: Wall furnace</dt><br>
#     </dl>
#      <p style= "font-size:13px;"></p>   
# </section>
# 
# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Observations: </u><br><br>
#     <dl>
#         <dt>* <b> "Gas forced warm air furnace"</b> almost dominate all types of heating mentioned in this feature.</dt><br>
#         <dt>* <b>18</b> data points for <b>"Gas hot water or steam heat"</b> and <b>one(1)</b> data point for <b>"Floor Furnace"</b> </dt><br>
#     </dl>   
# </section>

# <a id='12.-Pair-Plot'></a>
# <h1 style= "font-weight: bold; color:#823322; ">12. Pair Plot</h1>

# <p style="font-family:Lucida Sans Typewriter; font-size:13px;"> Lastly, I am going to end this notebook by apply a pair plot to this six variables just to see the relationship between them which I think they are the most important features.</p>

# <h1 style= "font-weight: bold; color:#823322; ">12.1 Select Six variables</h1>

# In[44]:


pair_cols = [
    "SalePrice",
    "Total_Square_Feet",
    "OverallQual",
    "TotalBsmtSF",
    "FullBath",
    "MasVnrArea",
]


# <a id='12.2-Plot-pairwise-relationships-of-the-selected-Six-variables-in-the-dataset'></a>
# <h1 style= "font-weight: bold; color:#823322; ">12.2 Plot pairwise relationships of the selected Six variables in the dataset</h1>

# In[45]:


sns.pairplot(
    usa_housing_df[pair_cols],
    hue="OverallQual",
    palette=my_palette,
    kind="scatter",
    diag_kind="kde",
    height=1.5,
    aspect=1.5,
    plot_kws={"alpha": 0.7},
)
plt.show()


# <br>
# <section style="font-family:Lucida Sans Typewriter; font-size:12px;">
#     <u style= "font-weight: lighter; color:#000000; font-size:12px;">Note: </u><br>
#     <dl>
#         <dt>* If we can deal with the outliers in a proper way, And consider the imbalanced proportions of observations in may features, we would have a big chance to build a solid model, however, we know that the house price is not related to the feature of this dataset, and the economy system and government decisions and legislation play a crucial role in this mater.</dt><br>
#     </dl>   
# </section>

# <a id='13.-Thank-you!'></a>
# <h1 style= "font-weight: bold; color:#000000; ">13.Thank you!</h1>

# <p style="font-family:Lucida Sans Typewriter; font-size:13px;">Knowing that this dataset has a many features which need more EDA work, I hope this EDA work was informative and somehow plain and got your appreciation,  and your comments in this matter are keys to better development!<br><br> THANK YOU!</p>
