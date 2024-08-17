#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#46b983;font-size:65px;font-family:Georgia;text-align:center;"><strong>Welcome <strong style="color:black;font-size:60px;font-family:Georgia;">To <strong style="color:#46b983;font-size:65px;font-family:Georgia;">House <strong style="color:black;font-size:60px;font-family:Georgia;">Price <strong style="color:#46b983;font-size:65px;font-family:Georgia;">Prediction</strong></strong></strong></strong></strong></h1>
# 
# 
# 
# ![](https://imagesvc.meredithcorp.io/v3/mm/gif?q=85&c=sc&poi=%5B832%2C672%5D&w=1600&h=800&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F23%2F2021%2F06%2F24%2Fhouse-price.gif)

# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>What <strong style="color:black;font-size:50px;font-family:Georgia;">Is <strong style="color:#46b983;font-size:55px;font-family:Georgia;">This <strong style="color:black;font-size:50px;font-family:Georgia;">Kernel <strong style="color:#46b983;font-size:55px;font-family:Georgia;">About?</strong></strong></strong></strong></strong></h2>
# 
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:110%;text-align:center;border-radius:10px 10px">Hey Everyone,<br>
# Today I am sharing this notebook which divides into three parts - EDA On Training Dataset + EDA On Testing Dataset + Model To Predict The House Price.<br>Here,we have two different datasets-one is training dataset and another one is testing dataset.<br>
# So,whenever i work on any dataset,my major focus will on a visualizations part because their you will actually grab the audience attention.Here also,i tried representing simple plot with some attractive add on features.<br>
# I just hope you will enjoy reading it.Please UPVOTE if you like it.<br>
#     Let get started...<br></p>
# 

# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>Table <strong style="color:black;font-size:50px;font-family:Georgia;">Of <strong style="color:#46b983;font-size:55px;font-family:Georgia;">The <strong style="color:black;font-size:50px;font-family:Georgia;">Contents <strong style="color:#46b983;font-size:55px;font-family:Georgia;">:-</strong></strong></strong></strong></strong></h2>
# 
# 
# * [1.Introduction](#1)
# * [2.Problem Statement](#2)
# * [3.Importing The Dataset](#3)
# * [4.Loading The Dataset](#4)
# * [5.About The Dataset](#5)
# * [6.Exploratory Data Analysis On The Training Dataset](#6)
# * [7.Exploratory Data Analysis On The Testing Dataset](#7)
# * [8.Creating the final dataset](#8)
# * [9.Model Creation Which Will Help Us To Predict🤔🤔 The Price💸💸 Of The 🏡🏡?](#9)
# * [10.Conclusion](#10)

# <a id="1"></a>
# <h1 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>1. <strong style="color:BLACK;font-size:50px;font-family:Georgia;">Introduction </strong></strong></h1>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px">The Housing Dataset, consisting of 1460 observations of residential properties sold between 2006-2010 in Ames, Iowa, was compiled by Dean de Cock in 2011<br>  A total of 80 predictors--23 nominal, 23 ordinal, 14 discrete, and 20 continuous describe aspects of the residential homes on the market during that period, as well as sale conditions.<br>
# <br>
# In 2016, Kaggle opened a housing price prediction competition, utilizing this dataset. Participants were provided with a training set and test set--consisting of 1460 and 1459 observations, respectively--and requested to submit sale price predictions on the test set.<br>
# Intended as practice in feature selection/engineering and machine learning modeling, the competition has been unfolding continuously over a nearly three-year span, and no cash prizes have been awarded.<br></p>
# 

# <a id="2"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>2. <strong style="color:black;font-size:50px;font-family:Georgia;">Problem <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Statement </strong></strong></strong></h2>
# 
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px">The main aim of this project is to predict the house price based on various features of the dataset.</p>
# 

# In[1]:


from IPython.core.display import display, HTML, Javascript

# ----- Notebook Theme -----#1e77b3
color_map = ['#52a57f', '#76bc9c', '#bbddce','#cce6da', '#cce6da','#eef7f3']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = ''' 

div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s; /* light orange */
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s; /* light orange */
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s; /* rgba(229, 143, 101, 0.1); light orange [exactly #E58F65] */
    border-top: 2px solid %s; /* light orange */
}

div.input_prompt {
color: %s; /* light blue */
}

div.output_prompt {
color: %s; /* strong orange */
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s; /* light orange */
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s; /* light orange */
}

.edit_mode div.cell.selected:before {
background: %s; /* light orange */
}

.edit_mode div.cell.selected {
border-color: %s; /* light orange */

}
'''
def to_rgb(h): 
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba, main_color,  main_color, prompt, main_color, main_color, main_color, main_color))

def nb(): 
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# <a id="3"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>4. <strong style="color:black;font-size:50px;font-family:Georgia;">Importing <strong style="color:#46b983;font-size:55px;font-family:Georgia;">The <strong style="color:black;font-size:50px;font-family:Georgia;">Libraries </strong></strong></strong></strong></h2>

# In[2]:


#IMPORT THE LIBRARIES....
import numpy as np # linear algebra....
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)....
from matplotlib import pyplot as plt #Visualization of the data....
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px">Looks like we have two different dataset -Training dataset and Testing dataset.</p>
# 
# 

# <a id="4"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>5. <strong style="color:black;font-size:50px;font-family:Georgia;">Loading <strong style="color:#46b983;font-size:55px;font-family:Georgia;">The <strong style="color:black;font-size:50px;font-family:Georgia;">Dataset </strong></strong></strong></strong></h2>

# In[4]:


df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_train.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# <a id="5"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>3. <strong style="color:black;font-size:50px;font-family:Georgia;">About <strong style="color:#46b983;font-size:55px;font-family:Georgia;">The <strong style="color:black;font-size:50px;font-family:Georgia;">Dataset  </strong></strong></strong></strong></h2>
# 
# 
# 
# 
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">
# 1.SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.<br>
# 2.MSSubClass: The building class.<br>
# 3.MSZoning: The general zoning classification.<br>
# 4.LotFrontage: Linear feet of street connected to property.<br>
# 5.LotArea: Lot size in square feet.<br>
# 6.Street: Type of road access.<br>
# 7.Alley: Type of alley access.<br>
# 8.LotShape: General shape of property.<br>
# 9.LandContour: Flatness of the property.<br>
# 10.Utilities: Type of utilities available.<br>
# 11.LotConfig: Lot configuration.<br>
# 12.LandSlope: Slope of property.<br>
# 13.Neighborhood: Physical locations within Ames city limits.<br>
# 14.Condition1: Proximity to main road or railroad.<br>
# 15.Condition2: Proximity to main road or railroad (if a second is present).<br>
# 16.BldgType: Type of dwelling.<br>
# 17.HouseStyle: Style of dwelling.<br>
# 18.OverallQual: Overall material and finish quality.<br>
# 19.OverallCond: Overall condition rating.<br>
# 20.YearBuilt: Original construction date.<br>
# 21.YearRemodAdd: Remodel date.<br>
# 22.RoofStyle: Type of roof.<br>
# 23.RoofMatl: Roof material.<br>
# 24.Exterior1st: Exterior covering on house.<br>
# 25.Exterior2nd: Exterior covering on house (if more than one material).<br>
# 26.MasVnrType: Masonry veneer type.<br>
# 27.MasVnrArea: Masonry veneer area in square feet.<br>
# 28.ExterQual: Exterior material quality.<br>
# 29.ExterCond: Present condition of the material on the exterior.<br>
# 30.Foundation: Type of foundation.<br>
# 31.BsmtQual: Height of the basement.<br>
# 32.BsmtCond: General condition of the basement.<br>
# 33.BsmtExposure: Walkout or garden level basement walls.<br>
# 34.BsmtFinType1: Quality of basement finished area.<br>
# 35.BsmtFinSF1: Type 1 finished square .<br>
# 36.BsmtFinType2: Quality of second finished area (if present).<br>
# 37.BsmtFinSF2: Type 2 finished square feet.<br>
# 38.BsmtUnfSF: Unfinished square feet of basement area.<br>
# 39.TotalBsmtSF: Total square feet of basement area.<br>
# 40.Heating: Type of heating.<br>
# 41.HeatingQC: Heating quality and condition.<br>
# 42.CentralAir: Central air conditioning.<br>
# 43.Electrical: Electrical system.<br>
# 44.1stFlrSF: First Floor square feet.<br>
# 45.2ndFlrSF: Second floor square feet.<br>
# 46.LowQualFinSF: Low quality finished square feet (all floors).<br>
# 47.GrLivArea: Above grade (ground) living area square feet.<br>
# 48.BsmtFullBath: Basement full bathrooms.<br>
# 49.BsmtHalfBath: Basement half bathrooms.<br>
# 50.FullBath: Full bathrooms above grade.<br>
# 51.HalfBath: Half baths above grade.<br>
# 52.Bedroom: Number of bedrooms above basement level.<br>
# 53.Kitchen: Number of kitchens.<br>
# 54.KitchenQual: Kitchen quality.<br>
# 55.TotRmsAbvGrd: Total rooms above grade (does not include bathrooms).<br>
# 56.Functional: Home functionality rating.<br>
# 57.Fireplaces: Number of fireplaces.<br>
# 58.FireplaceQu: Fireplace quality.<br>
# 59.GarageType: Garage location.<br>
# 60.GarageYrBlt: Year garage was built.<br>
# 61.GarageFinish: Interior finish of the garage.<br>
# 62.GarageCars: Size of garage in car capacity.<br>
# 63.GarageArea: Size of garage in square feet.<br>
# 64.GarageQual: Garage quality.<br>
# 65.GarageCond: Garage condition.<br>
# 66.PavedDrive: Paved driveway.<br>
# 67.WoodDeckSF: Wood deck area in square feet.<br>
# 68.OpenPorchSF: Open porch area in square feet.<br>
# 69.EnclosedPorch: Enclosed porch area in square feet.<br>
# 70.3SsnPorch: Three season porch area in square feet.<br>
# 71.ScreenPorch: Screen porch area in square feet.<br>
# 72.PoolArea: Pool area in square feet.<br>
# 73.PoolQC: Pool quality.<br>
# 74.Fence: Fence quality.<br>
# 75.MiscFeature: Miscellaneous feature not covered in other categories.<br>
# 76.MiscVal: Value of miscellaneous feature.<br>
# 77.MoSold: Month Sold.<br>
# 78.YrSold: Year Sold.<br>
# 79.SaleType: Type of sale.<br>
# 80.SaleCondition: Condition of sale.<br></p>
# 

# <a id="6"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>6. <strong style="color:black;font-size:50px;font-family:Georgia;">Exploratory <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Data <strong style="color:black;font-size:50px;font-family:Georgia;">Analysis <strong style="color:#46b983;font-size:55px;font-family:Georgia;">On <strong style="color:black;font-size:50px;font-family:Georgia;">Training <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Dataset</strong></strong></strong></strong></strong></strong></strong></h2>
# 

# In[5]:


print("The shape of the Training Dataset is:",df_train.shape)


# In[6]:


print("The Dimensions of the Training Dataset is:",df_train.ndim)


# In[7]:


print("Column names in the Training Dataset are:\n",df_train.columns)


# In[8]:


df_train.info()


# In[9]:


#Actual memory size...
memory_usage = df_train.memory_usage(deep=True) / 1024 ** 2
print('memory usage of features: \n', memory_usage.head(7))
print('memory usage sum: ',memory_usage.sum())


# In[10]:


#Memory after reduction...
def reduce_memory_usage(df_train, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df_train.memory_usage().sum() / 1024 ** 2
    for col in df_train.columns:
        col_type = df_train[col].dtypes
        if col_type in numerics:
            c_min = df_train[col].min()
            c_max = df_train[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_train[col] = df_train[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_train[col] = df_train[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_train[col] = df_train[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df_train[col] = df_train[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df_train[col] = df_train[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df_train[col] = df_train[col].astype(np.float32)
                else:
                    df_train[col] = df_train[col].astype(np.float64)
    end_mem = df_train.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df_train

df_train = reduce_memory_usage(df_train, verbose=True)


# In[11]:


df_train.describe().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[12]:


#Visualizing the missing values
import missingno as mn
mn.matrix(df_train,color=(0,0,0))


# In[13]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in df_train.columns if df_train[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(df_train[feature].isnull().mean(), 4),  ' % missing values.\n')


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px">Since they are many missing values, we need to find the relationship between missing values and Sales Price.</p>
# 
# 

# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px;"><b><u>Let's plot some diagram for this relationship:-</u></b></p>
# 
# 

# In[14]:


for feature in features_with_na:
    data = df_train.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the MEDIAN SalePrice where the information is missing or present
    colors = ["#FF6347","#98FB98"]
    data.groupby(feature)['SalePrice'].median().plot.barh(color= colors, edgecolor = "black",linewidth=3.5)
    plt.title(feature + " Vs Saleprice",fontsize = 22) 
    plt.ylabel(feature,fontsize = 15)
    plt.xlabel("SalePrice",fontsize = 15)
    plt.figure(figsize=(15,10)) 
    plt.show()   


# In[15]:


print("Id of Houses {}".format(len(df_train.Id)))


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Numerical Variables Analysis:-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">Generally we have two types of numerical variables - <br>
# 1.Continuous Variables.<br>
# 2.Discrete Variables.    
# <br></p>
# 

# In[16]:


# list of numerical variables............
numerical_features = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))
print('\n')
# visualise the numerical variables........
df_train[numerical_features].head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Temporal Variables(Eg: Datetime Variables):-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold.<br></p>
# 
# 
# 
# 
# 

# In[17]:


# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

print("Variables which include year is :-", year_feature)


# In[18]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, df_train[feature].unique())
    print('\n')


# In[19]:


## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

df_train.groupby('YrSold')['SalePrice'].median().plot(color = "tomato",linestyle = "--",linewidth=3)
plt.xlabel('Year Sold',fontsize = 15)
plt.ylabel('Median House Price',fontsize =15)
plt.title("House Price Vs YearSold",fontsize=22)
plt.figure(figsize=(30,12))


# In[20]:


print(year_feature)


# In[21]:


## Here we will compare the difference between All years feature with SalePrice
for feature in year_feature:
    if feature!='YrSold':
        data=df_train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]
        # Creating figure
        fig = plt.figure(figsize = (30, 12))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(data[feature],data['SalePrice'],color="tomato")
        plt.title(feature  +  " Vs SalePrice",fontsize = 22)
        plt.xlabel(feature,fontsize = 15)
        plt.ylabel('SalePrice',fontsize = 15)
        plt.grid(color="palegreen")
        plt.show()


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Discrete Variables:-</u></b></p>

# In[22]:


discrete_feature=[feature for feature in numerical_features if len(df_train[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[23]:


print(discrete_feature)


# In[24]:


df_train[discrete_feature].head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[25]:


## Lets Find the realtionship between them and Sale PRice
for feature in discrete_feature:
    data=df_train.copy()
    colors = ["#FF6347","#98FB98"]
    data.groupby(feature)['SalePrice'].median().plot.bar(color=colors,edgecolor = "black",linewidth=1.5)
    plt.xlabel(feature,fontsize=15)
    plt.ylabel('SalePrice',fontsize =15)
    plt.title(feature +" Vs SalePrice",fontsize = 22)
    plt.figure(figsize = (30, 12))
    plt.show()


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">* There is a relationship between variable number and SalePrice.
# <br></p>

# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Continuous  Variables:-</u></b></p>

# In[26]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[27]:


print(continuous_feature)


# In[28]:


## Lets analyse the continuous values by creating histograms to understand the distribution
for feature in continuous_feature:
    data=df_train.copy()
    data[feature].hist(bins=25,color = "tomato",edgecolor = "black",linewidth = 1.75)
    plt.xlabel(feature + "",fontsize=15)
    plt.ylabel("Count",fontsize=15)
    plt.title("Frequency of " + feature,fontsize = 22) 
    plt.grid(color = "palegreen")
    plt.figure(figsize = (30, 12))
    plt.show()
    print("The Skew value of "+feature,data[feature].skew())
    print('\n')


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">We will be using logarithmic transformation.</b>
# 
# </p>

# In[29]:


for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        # Creating figure
        fig = plt.figure(figsize = (30, 12))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(data[feature],data['SalePrice'],color = "tomato")
        plt.xlabel(feature,fontsize=15)
        plt.ylabel('SalesPrice',fontsize = 15)
        plt.title(feature + " Vs SalePrice",fontsize=22)
        plt.grid(color = "palegreen")
        plt.show()


# In[30]:


#Basic statistic on saleprice..
df_train['SalePrice'].describe()


# In[31]:


#Average sales price...
print("The Average SalePrice is :-",df_train['SalePrice'].mean())


# In[32]:


#Minimum sales price...
print("The Minimum SalePrice is :-",df_train['SalePrice'].min())


# In[33]:


#Maximum sales price...
print("The Maximum SalePrice is :-",df_train['SalePrice'].max())


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Let's Detect The Outliers:-</u></b></p>

# In[34]:


for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature,color = 'tomato')
        plt.ylabel(feature,fontsize = 15)
        plt.title("BoxPlot of " + feature,fontsize = 22)
        plt.grid(color = "palegreen")
        plt.figure(figsize=(30,12))
        plt.show()


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Categorical Variables:-</u></b></p>

# In[35]:


categorical_features=[feature for feature in df_train.columns if df_train[feature].dtypes=='O']
print("categorical feature Count {}".format(len(categorical_features)))


# In[36]:


print(categorical_features)


# In[37]:


df_train[categorical_features].head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[38]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}.'.format(feature,len(df_train[feature].unique())))
    print('\n')


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Find out the relationship between categorical variable and dependent feature SalesPrice:-</u></b></p>

# In[39]:


for feature in categorical_features:
    data=df_train.copy()
    colors = ["#FF6347","#98FB98"]
    data.groupby(feature)['SalePrice'].median().plot.bar(color = colors,edgecolor = "black",linewidth = 1.5)
    plt.xlabel(feature, fontsize = 15)
    plt.ylabel('SalePrice',fontsize = 15)
    plt.title(feature + " Vs SalePrice ",fontsize = 22)
    plt.figure(figsize=(30,12))
    plt.show()


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Let's Find The Missing values:-</u></b></p>

# In[40]:


#PERCENTAGE OF THE MISSING VALUES IN THE DATAFRAME..
def missing_data(df_train):
    total = df_train.isnull().sum().sort_values(ascending = False)
    Percentage = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
    Dtypes =df_train.dtypes
    return pd.concat([total, Percentage,Dtypes], axis=1, keys=['Total', 'Percentage','Dtypes'])
missing_data(df_train).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[41]:


print("Unique value in PoolQC are",df_train['PoolQC'].unique())
print("\n")
print("Unique value in MiscFeature are",df_train['MiscFeature'].unique())
print("\n")
print("Unique value in Alley are",df_train['Alley'].unique())
print("\n")
print("Unique value in Fence are",df_train['Fence'].unique())


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">NOTE:-While finding the relationship between the independent and dependent variables before treating the missing values, we found that there is some relationship between a variable that has more missing values(above 80%).<br>
# I don't want to drop those variables as it has a positive relationship with the dependent variable(Saleprice),<br>
# <br></p> 

# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Fill the missing values:-</u></b></p>

# In[42]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>1 and df_train[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(df_train[feature].isnull().mean(),4)))


# In[43]:


df_train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[44]:


df_train.head(10).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[45]:


# Dealing with Numerical missing values
df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].median())
df_train['MasVnrArea']=df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].median())
df_train['GarageYrBlt']=df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].median())


# In[46]:


#Dealing with ctegorical Missing values
df_train['FireplaceQu']=df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])
df_train['GarageCond']=df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])
df_train['GarageType']=df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])
df_train['GarageFinish']=df_train['GarageFinish'].fillna(df_train['GarageFinish'].mode()[0])
df_train['GarageQual']=df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])
df_train['BsmtFinType2']=df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].mode()[0])
df_train['BsmtExposure']=df_train['BsmtExposure'].fillna(df_train['BsmtExposure'].mode()[0])
df_train['BsmtQual']=df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])
df_train['BsmtCond']=df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])
df_train['BsmtFinType1']=df_train['BsmtFinType1'].fillna(df_train['BsmtFinType1'].mode()[0])
df_train['MasVnrType']=df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])
df_train['Electrical']=df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])


# In[47]:


#PERCENTAGE OF THE MISSING VALUES IN THE DATAFRAME..
def missing_data(df_train):
    total = df_train.isnull().sum().sort_values(ascending = False)
    Percentage = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
    Dtypes =df_train.dtypes
    return pd.concat([total, Percentage,Dtypes], axis=1, keys=['Total', 'Percentage','Dtypes'])
missing_data(df_train).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[48]:


df_train.head(10).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Temporal Variables (Date Time Variables):-</u></b></p>

# In[49]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df_train[feature]=df_train['YrSold']-df_train[feature]


# In[50]:


df_train.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[51]:


df_train[['YearBuilt','YearRemodAdd','GarageYrBlt']].head(10).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Treatment of the outliers:-</u></b></p>

# In[52]:


df_train.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[53]:


import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    df_train[feature]=np.log(df_train[feature])


# In[54]:


print("The Skew value of "+feature,df_train['SalePrice'].skew())


# In[55]:


df_train.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[56]:


print("The Dimensions of the Training Dataset is:",df_train.ndim)


# In[57]:


print("The shape of the Training Dataset is:",df_train.shape)


# In[58]:


print("Column names in the Training Dataset are:\n",df_train.columns)


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Correlation of training dataset:-</u></b></p>

# In[59]:


import seaborn as sns
plt.figure(figsize=(35,20))
sns.heatmap(df_train.corr(),annot=True,cmap="Greys",linewidth = 1,linecolor = "white")


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">-We can see that there isn’t much correlation among the input features . Thus there is no Multicollinearity among the features. This is good.<br>
# -Few Features have greater than 0.5 Pearson Correlation with output feature.<br>
# -A value closer to 0 implies weaker correlation (exact 0 implying no correlation).<br>
# -A value closer to 1 implies stronger positive correlation.<br>
# -A value closer to -1 implies stronger negative correlation.<br>
# </p>

# In[60]:


df_train.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[61]:


df_train.drop(['Id'],axis = 1,inplace =True)


# In[62]:


df_train.shape


# <p style= "background-color:#98FB98;font-family:Georgia;color:#000000;font-size:150%;text-align:center;border-radius:10px 10px"><b><u>Insights From Training Dataset:-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:140%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><br>1. Number of data points in the dataset : 1460.<br>
#     <br>
# 2. Number of features in the dataset : 81.<br>
#     <br>
# 3. Shape of the dataset is : (1460,81).<br>
#     <br>
# 4. Dimensions of the dataset is : 2.<br>
#     <br>
# 5. Column names : ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig','LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1','BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating','HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType','GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual','GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC','Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition', 'SalePrice']<br>
#     <br>
#  6.Highest missing values are in Alley,PoolQC,'Fence', 'MiscFeature' which is near to 80+%.<br>
#     <br>
#  7.After analyising the relationship between missing value and dependent variable we found that [LotFrontage,Alley,MasVnrType,MasVnrArea,Fence,MiscFeature] and [SalePrice] has a positive relation.<br>
#     <br>
# 8.In the dataset we have Numerical Variables(Continous and Discrete Variables) & Categorical Variables(Ordinal and Nominal Varaibles)<br>
#     <br>
# ------Numerical Variables------<br>
#     Total we have 38 Numerical Variables in the dataset.<br>
#     <br>
# ------Continous Variables------<br>
#    (i)As per the dataset we have total 16 variables.<br>
#    (ii)Column name :-'LotFrontage','LotArea', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea',
#     'WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch','SalePrice'<br>
#    (iii)When detecting the outlier - Skew value of SalePrice :-The Skew value of SalePrice is 1.8828757597682129.<br>
#    (iv)The Average SalePrice is :- 180921.19589041095<br>
#    (v)The Minimum SalePrice is :- 34900<br>
#    (vi)The Maximum SalePrice is :- 755000<br>
#    (vii)From Box plot we can clearly see that LotFrontage,LotArea,1stFlrSF,GrLivArea,SalePrice has outliers.<br>
#    (viii)After treatment of the outlier - Skew value of SalePrice :-The Skew value of SalePrice is 0.12133506220520406<br>
#     <br>
# ------Discrete Variables------<br>
#    (i)As per the dataset we have total 17 variables.<br>
#    (ii)Column name :-  MSSubClass','OverallQual','OverallCond',
#    'LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
#     'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','3SsnPorch','PoolArea','MiscVal','MoSold'<br>  
#     <br>
# ------Categorical Variables------<br>
#    (i)Total we have 43 Categorical Variables in the dataset.<br>
#     (ii)Column name :-'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
#     'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
#     'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
#     'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',
#     'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'.<br>  
#     <br>
# ------Temporal Variables(Date Time variables)------<br> 
#  (i)Total we have 4 temporal variables in the dataset<br>
#  (ii) Column name:-'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'<br>
#  (iii)After the analysis we can say that there is a negative relation between YrSold and SalePrice.<br></p>

# <a id="7"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>7. <strong style="color:black;font-size:50px;font-family:Georgia;">Exploratory <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Data <strong style="color:black;font-size:50px;font-family:Georgia;">Analysis <strong style="color:#46b983;font-size:55px;font-family:Georgia;">On <strong style="color:black;font-size:50px;font-family:Georgia;">Testing <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Dataset </strong></strong></strong></strong></strong></strong></strong></h2>
# 

# In[63]:


df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_test.head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[64]:


print("The shape of the Testing Dataset is:",df_test.shape)


# In[65]:


print("The Dimensions of the Testing Dataset is:",df_test.ndim)


# In[66]:


print("Column names in the Testing Dataset are:\n",df_test.columns)


# In[67]:


df_test.info()


# In[68]:


#Actual memory size....
memory_usage = df_test.memory_usage(deep=True) / 1024 ** 2
print('memory usage of features: \n', memory_usage.head(7))
print('memory usage sum: ',memory_usage.sum())


# In[69]:


#Memory after reduction...
def reduce_memory_usage(df_test, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df_test.memory_usage().sum() / 1024 ** 2
    for col in df_test.columns:
        col_type = df_test[col].dtypes
        if col_type in numerics:
            c_min = df_test[col].min()
            c_max = df_test[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_test[col] = df_test[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_test[col] = df_test[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_test[col] = df_test[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df_test[col] = df_test[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df_test[col] = df_test[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df_test[col] = df_test[col].astype(np.float32)
                else:
                    df_test[col] = df_test[col].astype(np.float64)
    end_mem = df_test.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df_test

df_test = reduce_memory_usage(df_test, verbose=True)


# In[70]:


df_test.describe().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[71]:


#Visualizing the missing values
import missingno as mn
mn.matrix(df_test,color=(0,0,0))


# In[72]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in df_test.columns if df_test[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(df_test[feature].isnull().mean(), 4),  ' % missing values.')
    print('\n')


# In[73]:


print("Id of Houses {}".format(len(df_test.Id)))


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Numerical Variables Analysis:-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">Generally we have two types of numerical variables - <br>
# 1.Continuous Variables.<br>
# 2.Discrete Variables.   
# <br></p>
# 

# In[74]:


# list of numerical variables............
numerical_features = [feature for feature in df_test.columns if df_test[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))
print('\n')
# visualise the numerical variables........
df_test[numerical_features].head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Temporal Variables(Eg: Datetime Variables):-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold.<br></p>

# In[75]:


# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

print(year_feature)


# In[76]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, df_test[feature].unique())
    print('\n')


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Discrete Variables:-</u></b></p>

# In[77]:


discrete_feature=[feature for feature in numerical_features if len(df_test[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[78]:


print(discrete_feature)


# In[79]:


df_test[discrete_feature].head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[80]:


## Frequency of each discrete feature........
for feature in discrete_feature:
    data=df_test.copy()
    colors = ["#FF6347","#98FB98"]
    data[feature].value_counts().plot.bar(color=colors,edgecolor = "black",linewidth=1.5)
    plt.xlabel(feature,fontsize=15)
    plt.ylabel('Frequency',fontsize =15)
    plt.title("Frequency of " + feature,fontsize = 22)
    plt.figure(figsize = (30, 12))
    plt.show()


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Continuous  Variables:-</u></b></p>

# In[81]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {} ".format(len(continuous_feature)))


# In[82]:


print(continuous_feature)


# In[83]:


## Lets analyse the continuous values by creating histograms to understand the distribution
for feature in continuous_feature:
    data=df_test.copy()
    data[feature].hist(bins=25,color = "tomato",edgecolor = "black",linewidth = 1.75)
    plt.xlabel(feature + "",fontsize=15)
    plt.ylabel("Frequency",fontsize=15)
    plt.title("Frequency of " + feature,fontsize = 22) 
    plt.grid(color = "palegreen")
    plt.figure(figsize = (30, 12))
    plt.show()
    print("The Skew value of "+feature,data[feature].skew())
    print('\n')     


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Let's Find The Outliers:-</u></b></p>

# In[84]:


for feature in continuous_feature:
    data=df_test.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature,color = 'tomato')
        plt.ylabel(feature,fontsize = 15)
        plt.title("BoxPlot of " + feature,fontsize = 22)
        plt.grid(color = "palegreen")
        plt.figure(figsize=(30,12))
        plt.show()


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Categorical Variables:-</u></b></p>

# In[85]:


categorical_features=[feature for feature in df_test.columns if df_test[feature].dtypes=='O']
print("categorical_features  Count {} ".format(len(categorical_features)))


# In[86]:


print(categorical_features)


# In[87]:


df_test[categorical_features].head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[88]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}.'.format(feature,len(df_test[feature].unique())))
    print('\n')


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Let's Find The Missing values:-</u></b></p>

# In[89]:


#PERCENTAGE OF THE MISSING VALUES IN THE DATAFRAME..
def missing_data(df_test):
    Total = df_test.isnull().sum().sort_values(ascending = False)
    Percentage = (df_test.isnull().sum()/df_test.isnull().count()*100).sort_values(ascending = False)
    Dtypes =df_test.dtypes
    return pd.concat([Total,Percentage,Dtypes], axis=1, keys=['Total', 'Percentage','Dtypes'])
missing_data(df_test).style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[90]:


print("Unique value in PoolQC are",df_test['PoolQC'].unique())
print("\n")
print("Unique value in MiscFeature are",df_test['MiscFeature'].unique())
print("\n")
print("Unique value in Alley are",df_test['Alley'].unique())
print("\n")
print("Unique value in Fence are",df_test['Fence'].unique())


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Fill the missing values:-</u></b></p>

# In[91]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in df_test.columns if df_test[feature].isnull().sum()>1 and df_test[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(df_test[feature].isnull().mean(),4)))
    print("\n")


# In[92]:


df_test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[93]:


df_test.head(10).style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[94]:


# Dealing with Numerical missing values
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].median())
df_test['GarageYrBlt']=df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].median())
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].median())
df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].median())
df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].median())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].median())
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].median())
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].median())
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].median())
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].median())


# In[95]:


#Dealing with categorical Missing values
df_test['Exterior1st']=df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['KitchenQual']=df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['SaleType']=df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

df_test['Functional']=df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['Utilities']=df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['MSZoning']=df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])

df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mode()[0])
df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtQual']=df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])

df_test['BsmtCond']=df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageCond']=df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['GarageQual']=df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])

df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['FireplaceQu']=df_test['FireplaceQu'].fillna(df_test['FireplaceQu'].mode()[0])
df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])


# In[96]:


#PERCENTAGE OF THE MISSING VALUES IN THE DATAFRAME..
def missing_data(df_test):
    total = df_test.isnull().sum().sort_values(ascending = False)
    Percentage = (df_test.isnull().sum()/df_test.isnull().count()*100).sort_values(ascending = False)
    Dtypes =df_test.dtypes
    return pd.concat([total,Percentage,Dtypes], axis=1, keys=['Total','Percentage','Dtypes'])
missing_data(df_test).style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[97]:


df_test.head(10).style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Temporal Variables (Date Time Variables):-</u></b></p>

# In[98]:


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df_test[feature]=df_test['YrSold']-df_test[feature]


# In[99]:


df_test.head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[100]:


df_test[['YearBuilt','YearRemodAdd','GarageYrBlt']].head(10).style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Treatment of the outliers:-</u></b></p>

# In[101]:


df_test.head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[102]:


import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_features:
    df_test[feature]=np.log(df_test[feature])


# In[103]:


df_test.head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[104]:


print("The Dimensions of the Testing Dataset is:",df_test.ndim)


# In[105]:


print("The shape of the Testing Dataset is:",df_test.shape)


# In[106]:


print("Column names in the Testing Dataset are:\n",df_test.columns)


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b><u>Correlation of testing dataset:-</u></b></p>

# In[107]:


import seaborn as sns
plt.figure(figsize=(35,20))
sns.heatmap(df_test.corr(),annot=True,cmap="twilight_shifted_r",linewidth = 1,linecolor = "white")


# In[108]:


df_test.head().style.set_properties(**{"background-color": "#FF6347","color": "black", "border-color": "black"})


# In[109]:


df_test.drop(['Id'],axis=1,inplace=True)


# In[110]:


print("The shape of the Testing Dataset is:",df_test.shape)


# <p style= "background-color:#FF6347;font-family:Georgia;color:#000000;font-size:150%;text-align:center;border-radius:10px 10px"><b><u>Insights From Testing Dataset:-</u></b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:140%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px"><br>1. Number of data points in the dataset : 1459.<br>
#     <br>
# 2. Number of features in the dataset : 80.<br>
#     <br>
# 3. Shape of the dataset is : (1459,80).<br>
#     <br>
# 4. Dimensions of the dataset is : 2.<br>
#     <br>
# 5. Column names : ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
#        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
#        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
#        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
#        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
#        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
#        'SaleCondition']<br>
#     <br>
#  6.Highest missing values are in Alley,PoolQC,'Fence', 'MiscFeature' which is near to 80+%.<br>
#     <br>
# 7.In the dataset we have Numerical Variables(Continous and Discrete Variables) & Categorical Variables(Ordinal and Nominal Varaibles)<br>
#     <br>
# ------Numerical Variables------<br>
#     Total we have 37 Numerical Variables in the dataset.<br>
#     <br>
# ------Continous Variables------<br>
#    (i)As per the dataset we have total 16 variables.<br>
#    (ii)Column name :-''LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal'<br>
# (iii)From Box plot we can clearly see that LotFrontage,LotArea,1stFlrSF,GrLivArea,SalePrice has outliers.<br>
#  <br>
# ------Discrete Variables------<br>
#    (i)As per the dataset we have total 16 variables.<br>
#    (ii)Column name :- 'MSSubClass','OverallQual','OverallCond','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces','GarageCars','3SsnPorch','PoolArea','MoSold'<br>  
#     <br>
# ------Categorical Variables------<br>
#    (i)Total we have 43 Categorical Variables in the dataset.<br>
#     (ii)Column name :-'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'.<br>  
#     <br>
# ------Temporal Variables(Date Time variables)------<br> 
#  (i)Total we have 4 temporal variables in the dataset<br>
#  (ii) Column name:-'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'<br>
# </p>

# <a id="8"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>8. <strong style="color:black;font-size:50px;font-family:Georgia;">Creating <strong style="color:#46b983;font-size:55px;font-family:Georgia;">The <strong style="color:black;font-size:50px;font-family:Georgia;">Final <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Dataset </strong></strong></strong></strong></strong></h2>
# 

# In[111]:


# Concatenating both train and test data (This is done to handle all the different category value possibilites )
df = pd.concat([df_train, df_test], axis=0)
df.shape


# In[112]:


df.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[113]:


# Creating the dummy variables for all the categorical features

cat_cols = list(df.select_dtypes("object"))

for col in cat_cols:
    dummy_cols = pd.get_dummies(df[col], drop_first=True, prefix=col)
    df = pd.concat([df,dummy_cols],axis=1)
    df.drop(columns=col, axis=1, inplace=True)


# In[114]:


df.shape


# In[115]:


df.head().style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[116]:


#Cheacking the duplicates if any?
df[df.duplicated()]


# In[117]:


#Rearrangement of the columns......
df = df[['MSSubClass',
'LotFrontage',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'YearRemodAdd',
'MasVnrArea',
'BsmtFinSF1',
'BsmtFinSF2',
'BsmtUnfSF',
'TotalBsmtSF',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'BsmtFullBath',
'BsmtHalfBath',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'GarageYrBlt',
'GarageCars',
'GarageArea',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold',
'MSZoning_FV',
'MSZoning_RH',
'MSZoning_RL',
'MSZoning_RM',
'Street_Pave',
'LotShape_IR2',
'LotShape_IR3',
'LotShape_Reg',
'LandContour_HLS',
'LandContour_Low',
'LandContour_Lvl',
'Utilities_NoSeWa',
'LotConfig_CulDSac',
'LotConfig_FR2',
'LotConfig_FR3',
'LotConfig_Inside',
'LandSlope_Mod',
'LandSlope_Sev',
'Neighborhood_Blueste',
'Neighborhood_BrDale',
'Neighborhood_BrkSide',
'Neighborhood_ClearCr',
'Neighborhood_CollgCr',
'Neighborhood_Crawfor',
'Neighborhood_Edwards',
'Neighborhood_Gilbert',
'Neighborhood_IDOTRR',
'Neighborhood_MeadowV',
'Neighborhood_Mitchel',
'Neighborhood_NAmes',
'Neighborhood_NPkVill',
'Neighborhood_NWAmes',
'Neighborhood_NoRidge',
'Neighborhood_NridgHt',
'Neighborhood_OldTown',
'Neighborhood_SWISU',
'Neighborhood_Sawyer',
'Neighborhood_SawyerW',
'Neighborhood_Somerst',
'Neighborhood_StoneBr',
'Neighborhood_Timber',
'Neighborhood_Veenker',
'Condition1_Feedr',
'Condition1_Norm',
'Condition1_PosA',
'Condition1_PosN',
'Condition1_RRAe',
'Condition1_RRAn',
'Condition1_RRNe',
'Condition1_RRNn',
'Condition2_Feedr',
'Condition2_Norm',
'Condition2_PosA',
'Condition2_PosN',
'Condition2_RRAe',
'Condition2_RRAn',
'Condition2_RRNn',
'BldgType_2fmCon',
'BldgType_Duplex',
'BldgType_Twnhs',
'BldgType_TwnhsE',
'HouseStyle_1.5Unf',
'HouseStyle_1Story',
'HouseStyle_2.5Fin',
'HouseStyle_2.5Unf',
'HouseStyle_2Story',
'HouseStyle_SFoyer',
'HouseStyle_SLvl',
'RoofStyle_Gable',
'RoofStyle_Gambrel',
'RoofStyle_Hip',
'RoofStyle_Mansard',
'RoofStyle_Shed',
'RoofMatl_CompShg',
'RoofMatl_Membran',
'RoofMatl_Metal',
'RoofMatl_Roll',
'RoofMatl_Tar&Grv',
'RoofMatl_WdShake',
'RoofMatl_WdShngl',
'Exterior1st_AsphShn',
'Exterior1st_BrkComm',
'Exterior1st_BrkFace',
'Exterior1st_CBlock',
'Exterior1st_CemntBd',
'Exterior1st_HdBoard',
'Exterior1st_ImStucc',
'Exterior1st_MetalSd',
'Exterior1st_Plywood',
'Exterior1st_Stone',
'Exterior1st_Stucco',
'Exterior1st_VinylSd',
'Exterior1st_Wd Sdng',
'Exterior1st_WdShing',
'Exterior2nd_AsphShn',
'Exterior2nd_Brk Cmn',
'Exterior2nd_BrkFace',
'Exterior2nd_CBlock',
'Exterior2nd_CmentBd',
'Exterior2nd_HdBoard',
'Exterior2nd_ImStucc',
'Exterior2nd_MetalSd',
'Exterior2nd_Other',
'Exterior2nd_Plywood',
'Exterior2nd_Stone',
'Exterior2nd_Stucco',
'Exterior2nd_VinylSd',
'Exterior2nd_Wd Sdng',
'Exterior2nd_Wd Shng',
'MasVnrType_BrkFace',
'MasVnrType_None',
'MasVnrType_Stone',
'ExterQual_Fa',
'ExterQual_Gd',
'ExterQual_TA',
'ExterCond_Fa',
'ExterCond_Gd',
'ExterCond_Po',
'ExterCond_TA',
'Foundation_CBlock',
'Foundation_PConc',
'Foundation_Slab',
'Foundation_Stone',
'Foundation_Wood',
'BsmtQual_Fa',
'BsmtQual_Gd',
'BsmtQual_TA',
'BsmtCond_Gd',
'BsmtCond_Po',
'BsmtCond_TA',
'BsmtExposure_Gd',
'BsmtExposure_Mn',
'BsmtExposure_No',
'BsmtFinType1_BLQ',
'BsmtFinType1_GLQ',
'BsmtFinType1_LwQ',
'BsmtFinType1_Rec',
'BsmtFinType1_Unf',
'BsmtFinType2_BLQ',
'BsmtFinType2_GLQ',
'BsmtFinType2_LwQ',
'BsmtFinType2_Rec',
'BsmtFinType2_Unf',
'Heating_GasA',
'Heating_GasW',
'Heating_Grav',
'Heating_OthW',
'Heating_Wall',
'HeatingQC_Fa',
'HeatingQC_Gd',
'HeatingQC_Po',
'HeatingQC_TA',
'CentralAir_Y',
'Electrical_FuseF',
'Electrical_FuseP',
'Electrical_Mix',
'Electrical_SBrkr',
'KitchenQual_Fa',
'KitchenQual_Gd',
'KitchenQual_TA',
'Functional_Maj2',
'Functional_Min1',
'Functional_Min2',
'Functional_Mod',
'Functional_Sev',
'Functional_Typ',
'FireplaceQu_Fa',
'FireplaceQu_Gd',
'FireplaceQu_Po',
'FireplaceQu_TA',
'GarageType_Attchd',
'GarageType_Basment',
'GarageType_BuiltIn',
'GarageType_CarPort',
'GarageType_Detchd',
'GarageFinish_RFn',
'GarageFinish_Unf',
'GarageQual_Fa',
'GarageQual_Gd',
'GarageQual_Po',
'GarageQual_TA',
'GarageCond_Fa',
'GarageCond_Gd',
'GarageCond_Po',
'GarageCond_TA',
'PavedDrive_P',
'PavedDrive_Y',
'SaleType_CWD',
'SaleType_Con',
'SaleType_ConLD',
'SaleType_ConLI',
'SaleType_ConLw',
'SaleType_New',
'SaleType_Oth',
'SaleType_WD',
'SaleCondition_AdjLand',
'SaleCondition_Alloca',
'SaleCondition_Family',
'SaleCondition_Normal',
'SaleCondition_Partial','SalePrice']]
df.head(2).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[118]:


df.shape


# In[119]:


# Splitting into df_train and df_test

df_train = df.iloc[:1460,:]
df_test = df.iloc[1460:,:]

print(df_train.shape)
print(df_test.shape)


# In[120]:


df_train.head(2).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[121]:


df_test.head(2).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[122]:


df_test.drop(['SalePrice'],axis=1,inplace=True)


# <a id="9"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>9. <strong style="color:black;font-size:50px;font-family:Georgia;">Model <strong style="color:#46b983;font-size:55px;font-family:Georgia;">Creation </strong></strong></strong></h2>
# 

# In[123]:


# Splitting the data into train and test
X_train=df_train.drop(['SalePrice'],axis=1)
y_train=df_train['SalePrice']


# In[124]:


#Model comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[125]:


#Building piplines for model selection

lr=Pipeline([("pca1",PCA(n_components=50)),("LR",LinearRegression())])

dt=Pipeline([("pca2",PCA(n_components=50)),("DT",DecisionTreeRegressor())])

rf=Pipeline([("pca3",PCA(n_components=50)),("RF",RandomForestRegressor())])

knn=Pipeline([("pca4",PCA(n_components=50)),("KNN",KNeighborsRegressor())])

ada=Pipeline([("pca5",PCA(n_components=50)),("ADA",AdaBoostRegressor())])

bag=Pipeline([("pca6",PCA(n_components=50)),("BAG",BaggingRegressor())])

ext=Pipeline([("pca7",PCA(n_components=50)),("EXT",ExtraTreesRegressor())])

grad=Pipeline([("pca4",PCA(n_components=50)),("XGB",GradientBoostingRegressor())])


#List of all the pipelines
pipelines = [lr, dt, rf, knn, ada,bag, ext, grad]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "AdaBoosting",5:"Bagging",6:"ExtraTree",7:"GradientBoosting"}


# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

#Getting CV scores    
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=20)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:140%;text-align:center;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b>The Best Model is Linear Regression 🏆🥳🏆</b></p>

# In[126]:


#Since linear regression has least scores.

lr.fit(X_train,y_train)


# In[127]:


#predicting the test set
y_pred =lr.predict(df_test)


# In[128]:


#Since we have converted the data using log function-
#Here i am trying to convert the log value into a number.
y_pred = np.exp(y_pred)
print(y_pred)


# In[129]:


#Metrics to  evaluate the model....
from sklearn.metrics import r2_score, mean_squared_error

y_pred = lr.predict(X_train)
print("R2 Score for our training data: " + str(r2_score(y_train,y_pred)))
print("RMSE for our training data: " + str(np.sqrt(mean_squared_error(y_train,y_pred))))


# In[130]:


#Create a sample submission file and submit
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
data = pd.concat([sub_df['Id'],pred],axis = 1)
data.columns = ['Id','SalePrice']
data['SalePrice']=np.exp(data['SalePrice'])
data.head(10).style.set_properties(**{"background-color": "#98FB98","color": "black", "border-color": "black"})


# In[131]:


#Saving the predicted result in csv format..
data.to_csv("Sample__submission.csv",index = False)


# <a id="10"></a>
# <h2 style="color:#46b983;font-size:55px;font-family:Georgia;text-align:center;"><strong>10. <strong style="color:black;font-size:50px;font-family:Georgia;">Conclusion <strong style="color:#46b983;font-size:55px;font-family:Georgia;">:- </strong></strong></strong></h2>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:140%;text-align:center;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:center;border-radius:10px 10px"><b>The Linear Regression has the best performance Root Mean Square Error(RMSE):-0.13916244817342852</b></p>
# 
# <p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:140%;text-align:left;border-radius:10px 10px"><p style= "background-color:#ffffff;font-family:Georgia;color:#000000;font-size:130%;text-align:left;border-radius:10px 10px">If you liked this Notebook, please do upvote.<br>
# If you Fork this Notebook, please do upvote.<br>
# If you have any questions, feel free to comment!<br>
# <b>BEST OF LUCK.</b><br>
# <b>HAPPY LEARNING😊😊</b><br></p>
