#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:30px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>✨ Predicting House Price ✨</b></div>

# <h3 align="center" style="font-size: 35px; color: #800080; font-family: Georgia;">
#     <span style="color: #008080;"> Author:</span> 
#     <span style="color: black;">Kumod Sharma .📄🖋️</span>
# </h3>

# <img src="https://media.licdn.com/dms/image/C5612AQF-4JihSLXkjw/article-cover_image-shrink_600_2000/0/1639905437564?e=2147483647&v=beta&t=dpD207ru5kxp4ZZecfHuXLr9AdenVCeu7TqP27ZLnG0" style='width: 1000px; height: 450px;'>

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 🎬 Introduction 🎬</b></div>

# <div style="border-radius:10px;border:black solid;padding: 15px;background-color:white;font-size:110%;text-align:left">
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:17px">
# 
#    
#   
# <h3 align="left"><font color=purple>📝 Project Objective:</font></h3><br> 
# 
#     
# 1. The aim of this project is to <b>train a Machine Learning Model</b> which can predict the <b>House Sale Price</b> using various relevant features.<br>
# 2. This project is completely based for <b>House Prices - Advanced Regression Techniques</b> Kaggle Competition.<br>
# 3. With <b>79 explanatory variables</b> describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges us to predict the final price of each home.<br>
# 4. Dataset Link:- <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data">Click to get the Dataset</a>
# <br>
#     
#     
#     
# <h3 align="left"><font color=purple>🌟 Business Understanding:</font></h3><br> 
#  
# 1. In the dynamic real estate market, the significance of <b>accurate house price predictions</b> is increasing significantly, as they have the potential to empower homeowners, buyers, and real estate professionals by providing valuable insights into property values and facilitating informed decision-making.<br>
# 
#     
# 2. As a result, the <b>real estate industry faces the crucial task of determining the appropriate pricing for houses</b> before listing them on the market. This is achieved through a comprehensive analysis of various property attributes such as location, size, amenities, condition, market trends, and more.<br>
# 
#     
# 3. Analyzing <b>house attributes to determine pricing</b> helps the real estate industry strike a balance between fair market value for sellers and affordability for buyers. It ensures that house prices align with their unique characteristics, desirability, and overall value proposition. This approach also fosters <b>transparency and facilitates fair competition</b> among properties, allowing buyers to make well-informed decisions based on their specific needs, preferences, and budget constraints. Moreover, accurate house price predictions enable homeowners to assess their property's worth and make informed choices regarding selling, refinancing, or investment opportunities.<br>
# </div>
# </div>

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 📝 Project Contents 📝</b></div>

# <div style="border-radius:10px;border:black solid;padding: 15px;background-color:white;font-size:110%;text-align:left">
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:16px">
# 
# <h3 align="left"><font color=purple>📊 Table of Contents:</font></h3><br>
#     
#     
# 1. <b>📚 Importing Libraries:</b> - To perform <b>Data Manipulation,Visualization & Model Building.</b><br>
#     
#     
# 2. <b>⏳ Loading Dataset:</b> - Load the dataset into a <b>suitable data structure using pandas.</b><br>
# 
#     
# 3. <b>🧠 Basic Understaning of Data:</b> - Generate basic informations about the data.<br>
#     
#     
# 4. <b>🧹 Data Preprocessing Part-1:</b> - To <b>clean, transform, and restructure</b> the data in order to make it suitable for analysis.<br>
#  
#     
# 5. <b>📊 Exploatory Data Analysis:</b> -  To  identify <b>trends, patterns, and relationships</b> among the variabels.<br>
# 
#     
# 6. <b>📈 Feature Engineering:</b> -  To create <b>new relevant features</b> for model building.<br>
#     
#     
# 7. <b>⚙️ Data Preprocessing Part-2:</b> - To transform data for creating more accurate & robust model.<br>
#     
#     
# 8. <b>🎯 Model building:</b>- To build <b>predictive models</b>, using various algorithms.<br>
#     
#     
# 9. <b>⚡️ Model evaluation:</b> - To analyze the Model performance using metrics.<br>
#     
# 
# 10. <b>🌟 Hyper-Parameter Tunning:</b> - Optimiging model using best parameters.<br>
#     
#     
# 11. <b>🍀 Stacking Model:</b>- To develop a stacked model using the top performing models.<br>
#  
#  
# 11. <b>🎈 Conclusion:</b> - Conclude the project by summarizing the <b>key findings.</b><br>
#     
# </div>

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 📚 Importing Libraries 📚</b></div>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)

from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⏳ Loading Datset ⏳</b></div>

# In[2]:


df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[3]:


df_train.head()


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🧠 Basic Understanding of Data 🧠</b></div>

# ### 1. Checking Dimension of the Datasets.

# In[4]:


print("Train Dataset has ",df_train.shape[0],"Records/Rows and ",df_train.shape[1],"attributes/columns.")
print("Test Dataset has ",df_test.shape[0],"Records/Rows and ",df_test.shape[1],"attributes/columns.")


# ---

# ### 2. Generating Basic Information of Train Data.

# In[5]:


df_train.info(verbose=False)


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * There is total **1460 records/rows** and **81 attributes/columns.**
# * Out of **81 columns**, **38 columns are numerical** and **43 columns are categorical.**

# ---

# ### 3. Performing Descriptive Statistical Analysis on Categorical Features.

# In[6]:


df_train.describe(include="object")


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **None of the categorical features** are having **high cardinality.**
# * **Features** like **`Neighborhood`**, **`Exterior1st`**, **`Exterior2nd`** are having **little bit of high cardinality** but that can be **manged** using different techniques of **encoding.**

# ---

# ### 4. Performing Descriptive Statistical Analysis on Numerical Features.

# In[7]:


df_train.describe(include=[int,float])


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **Numerical Features** like **`LotArea`**, **`BsmtFinSF1`**,**`BsmtUnfSF`**, **`TotalBsmtSF`**,**`GrLivArea`**, **`MiscVal`** and even the target feature **SalePrice** is having **very high deviation** values which can **lead to bias, Overfitting, and can affect the accuracy of the model.**
# * So we have to use **different transformation technique** to reduce the deviation between the data-points. 

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⚙️ Data Preprocessing - Part 1 ⚙️</b></div>

# ### 1. Showing Random Sample of the Dataset.

# In[8]:


df_train.sample(5)


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * We can observe that feature named **Id** represnt a **index number** to each records in the dataset.
# * This feature **doesn't seem relevant** for the **analysis**, so we can simply **drop this feature.**

# ### 2. Dropping "Id" Feature.

# In[9]:


test_id = df_test["Id"]    ##Storing test id because we need it for subission file.
df_train.drop(columns="Id",inplace=True)
df_test.drop(columns="Id",inplace=True)


# ---

# ### 3. Computing Features with Missing Values More Than 45%.

# In[10]:


null_df = round(df_train.isnull().sum()/len(df_train)*100,2).sort_values().to_frame().rename(columns=
                                                                                    {0:"Train % of Missing Values"})
null_df["Test % of Missing Values"] = round(df_test.isnull().sum()/len(df_train)*100,2)


# In[11]:


null_df[(null_df["Train % of Missing Values"]>45) | (null_df["Test % of Missing Values"]>45)]


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features like **`PoolQC`**,**`MiscFeature`**, **`Alley`** & **`Fence`** are **having large number of missing values.**
# * **Data Imputation** in features with **large scale of misisng values** can lead to **bias & noise** in the dataset.
# * So we can simply **drop** those features with **large scale of missing values.**

# ---

# ### 4. Dropping Features with more than 45% of Missing Values.

# In[12]:


cols = ["FireplaceQu","Fence","Alley","MiscFeature","PoolQC"]

df_train.drop(columns=cols, inplace=True)
df_test.drop(columns=cols, inplace=True)


# ---

# ### 5. Combining Train & Test Dataset for Easier Analysis.

# In[13]:


target = df_train[["SalePrice"]].reset_index(drop=True)

df_train.drop(columns=["SalePrice"],inplace=True)

df = pd.concat([df_train,df_test]).reset_index(drop=True)


# In[14]:


df.shape


# ---

# ### 6. Computing Total Missing Values and % of Misisng Values.

# In[15]:


null_df = df.isnull().sum()[df.isnull().sum()>0].sort_values().to_frame().rename(columns={0:"Total Missing values"})
null_df["% of Missing Values"] = round(null_df["Total Missing values"]/len(df)*100,2)
null_df["Feature Data Type"] = df[null_df.index.tolist()].dtypes


# In[16]:


null_df


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Total **29 columns** are still having **missing values.**
# * We will **fill missing values separately** in categorical and Numerical Columns.

# ---

# ### 7. Filling Missing Values in Features realted to Garage & Basement.

# In[17]:


for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    df[col] = df[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna('None')

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    df[col] = df[col].fillna('None')


# ---

# ### 8. Filling Missing Values in Categorical Columns.

# In[18]:


df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[19]:


cat_cols = ['Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical',
            'KitchenQual','Functional','SaleType']

imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imputer.fit_transform(df[cat_cols])


# ---

# ### 9. Filling Missing Values in Numerical Columns.

# In[20]:


df["LotFrontage"] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df["MasVnrArea"]  = df.groupby("MasVnrType")["MasVnrArea"].transform(lambda x: x.fillna(x.median()))
df["BsmtFinSF1"]  = df.groupby("BsmtFinType1")["BsmtFinSF1"].transform(lambda x: x.fillna(x.median()))
df["BsmtFinSF2"]  = df.groupby("BsmtFinType2")["BsmtFinSF2"].transform(lambda x: x.fillna(x.median()))


# In[21]:


df["BsmtFullBath"] = df["BsmtFullBath"].fillna(0.0)
df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0.0)
df["TotalBsmtSF"]  = df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["BsmtUnfSF"]    = df["BsmtUnfSF"].fillna(df["BsmtUnfSF"].median())


# ---

# ### 10. Confirming Filling of Missing Values.

# In[22]:


print("Total Missing Values Left is:",df.isnull().sum().sum())


# ---

# ### 11. Separating Train and Test Datframe.

# In[23]:


train_df = pd.concat([df.iloc[:len(target["SalePrice"]),:],target],axis=1)
test_df = df.iloc[len(target["SalePrice"]):,:]


# In[24]:


print("Dimension of train data is:",train_df.shape)
print("Dimension of test data is:",test_df.shape)


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>📊 Exploratory Data Analysis 📊</b></div>

# ### 1. Analyzing & Visualizing Target Varibele (SalePrice).

# In[25]:


train_df["SalePrice"].describe().to_frame().T


# In[26]:


plt.figure(figsize=(13,6))

plt.subplot(1,2,1)
sns.histplot(train_df["SalePrice"],color="purple",kde=True)
plt.title("SalePrice Distribution Plot",fontweight="black",pad=20,size=18)

plt.subplot(1,2,2)
sns.boxplot(train_df["SalePrice"],color="purple")
plt.title("SalePrice Outliers Detection",fontweight="black",pad=20,size=18)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * The target feature is having a **right-skewed distribution** due to presence of **positive outliers.**
# * It is apparent that **SalePrice doesn't follow normal distribution**, so before performing regression it has to be transformed.
# * To achieve a **Normal Distribution** we can use different **transformation techniques** like:
#     * **`Johnsonsu Transformation`**, **`Norm Transformation`** or **`Log Noraml Transformation`**
#     * From these three tansformation which ever **gives best fit** we can **use that transformation.**

# ----

# ### 2. Visualizing Different Transformation Techniques on "SalePrice" Attribute.

# In[27]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
sns.distplot(train_df["SalePrice"],kde=False, fit=stats.johnsonsu,color="red")
plt.title("Johnson Transformation",fontweight="black",size=20,pad=20)

plt.subplot(1,3,2)
sns.distplot(train_df["SalePrice"],kde=False, fit=stats.norm,color="green")
plt.title("Normal Transformation",fontweight="black",size=20,pad=20)

plt.subplot(1,3,3)
sns.distplot(train_df["SalePrice"],kde=False,fit=stats.lognorm,color="blue")
plt.title("Log Normal Transformation",fontweight="black",size=18,pad=20)
plt.tight_layout()
plt.show()


# 
# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **After applying** different transformation techniques the best result were given by **`Unbounded Johnson Transformation`.**
#     
# * But the **`Log Normal Transformation`** has also done a good job to achieve a **`normal distribution.`**

# ---

# ### 3. Visualizing Distribution of Continous Numerical Features.

# In[28]:


con_cols = ["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
            "1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF",
            "EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"]


# In[29]:


plt.figure(figsize=(25,20))
for index,column in enumerate(con_cols):
    plt.subplot(5,4,index+1)
    sns.histplot(train_df[column],bins=10,kde=True,color="red")
    plt.title(f"{column} Distribution",fontweight="black",size=20,pad=10)
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **GargeArea Feature** is having a **kind of Normal Distribution.**
# * **None of the other featurs** is having a **normal distribution** and all the features is **right-skewed.**
# * We know that **Linear regression** models assume a **linear relationship** between the **predictors and the response variable.**
# * Since the **relationship is non-linear, transforming** the variables can **help capture and represent the underlying non-linear relationship more accurately.**

# ---

# ### 4. Visualizing the Skewness of Continous Numerical Features.

# In[30]:


skewness = df[con_cols].skew().sort_values()

plt.figure(figsize=(14,6))
sns.barplot(skewness.index, skewness, palette=sns.color_palette("Reds",19))
for i, v in enumerate(skewness):
    plt.text(i, v, f"{v:.1f}", ha="center", va="bottom",size=15,fontweight="black")

plt.ylabel("Skewness")
plt.xlabel("Columns")
plt.xticks(rotation=90)
plt.title("Skewness of Continous Numerical Columns",fontweight="black",size=20,pad=10)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features like **`3SsnPorch`**,**`LowQualFinSF`**,**`LotArea`**,**`PoolArea`** and **`MiscVal`** are having **extremly high skewness** which can **create model-complexity.**
# * We know that **skewness** should be **near to zero** for a **normal distrbution** to achieve that we can use **different transformations.**

# ---

# ### 5. Visualizing the Correlation of Continous Numerical Features.

# In[31]:


con_cols.append("SalePrice")


# In[32]:


corr = train_df[con_cols].corr(method="spearman")["SalePrice"].sort_values()

plt.figure(figsize=(15,6))
sns.barplot(corr.index, corr, palette=["lightcoral" if v < 0 else "lightgreen" for v in corr])
for i, v in enumerate(corr):
    plt.text(i, v, f"{v:.1f}", ha="center", va="bottom",size=15,fontweight="black")

plt.title("Coorelation of Continous features w.r.t SalePrice",fontweight="black",size=20,pad=10)
plt.xticks(rotation=90)
plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features like **`1stFlrSF`**,**`GrLivArea`**,and **`GarageArea`** are having **strong relation** with the target variable.
#     
# * Features like **`WoodDeckSF`**,**`LotDrontage`**,and **`MasVnrArea`** are having **modearte relation** with the target varible.
#     
# * Features like **`LowQualFinSF`**,**`MiscVal`**,**`BsmtFinSF2`**,**`PoolArea`**,**`3SsnPorch`**,and **`ScreenPorch`** are having **very low relation** with the target variable. So if required we can **drop this features,**

# ---

# ### 6. Visualizing Categorical Features w.r.t SalePrice.

# In[33]:


cat_cols = train_df.select_dtypes(include="object").columns.tolist()


# In[34]:


def boxplot(col_list):
    plt.figure(figsize=(22,12))
    for index,column in enumerate(col_list):
        plt.subplot(2,4,index+1)
        sns.boxplot(x=column, y="SalePrice", data=train_df)
        plt.title(f"{column} vs SalePrice",fontweight="black",pad=10,size=20)
        plt.xticks(rotation=90)
        plt.tight_layout()


# In[35]:


boxplot(cat_cols[0:8])


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features like **`Utilities`** and **`Street`** are having **very high class imbalance.** So we can **simply drop this features.**
# * **Neighborhood** feature is having **high cardinality**, So we have to **perform Target Encoding** on this feature.

# #### Dropping Columns with High Class-Imbalance.

# In[36]:


train_df.drop(columns=["Utilities","Street"],inplace=True)
test_df.drop(columns=["Utilities","Street"],inplace=True)


# ---

# In[37]:


boxplot(cat_cols[8:16])


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features like **`Condition2`** and **`RoofMatl`** are having **very high class imbalance.**
# * We can **drop RooftMatl** feature and we will fo **feature engineering** on **condition2** to **reduce the class-imbalance.** 

# #### Dropping feature with high class-imbalance.

# In[38]:


train_df.drop(columns=["RoofMatl"],inplace=True)
test_df.drop(columns=["RoofMatl"],inplace=True)


# ---

# In[39]:


boxplot(cat_cols[16:24])


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **All** the features in above visualization **seems very useful** for predicting the price of house.
# * The only thing is that we have to perform **enconding** before model training.

# ---

# In[40]:


boxplot(cat_cols[24:32])


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Features **`Heating`** is having **a class imbalance.** But heating is a **important feature** for people buying house.
# * Hence we can perform **feature engineering** to reduce the class-imbalance.
# * Rest **all** the other features **seems useful** for house price prediction.

# ---

# In[41]:


plt.figure(figsize=(22,12))
for index,column in enumerate(cat_cols[32:]):
    plt.subplot(2,3,index+1)
    sns.boxplot(x=column, y="SalePrice", data=train_df)
    plt.title(f"{column} vs SalePrice",fontweight="black",pad=10,size=20)
    plt.xticks(rotation=90)
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **All** the features in above visualization **seems very useful** for predicting the price of house.
# * The only thing is that we have to perform **enconding** before model training.

# ---

# ### 7. Visualizing Discrete Numerical Features w.r.t Average "SalePrice".

# In[42]:


dis_cols = ["OverallQual","OverallCond","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr",
            "KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","MoSold"]


# In[43]:


plt.figure(figsize=(22,14))
for index,column in enumerate(dis_cols):
    data = train_df.groupby(column)["SalePrice"].mean()
    plt.subplot(3,4,index+1)
    sns.barplot(data.index, data,)
    plt.title(f"{column} vs Avg. Sale Price",fontweight="black",size=15,pad=10)
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * **`OverallQual`**,**`FullBath`**,**`TotRmsAbvGrd`**, **`FirePlaces`** and **`GarageCars`** are having **strong positive relation** with the **SalePrice.**
#     
# * **`KitchenAbvGr`** is having a **negative correlation** with **SalePrice.**
# * So **all these above features** seems **useful** for predicting **Saleprice**.

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🛠️ Feature Engineering 🛠️ </b></div>

# ## 1. Creating Two New Features "RenovationStatus" and "AgeAtSale" of the House.

# * **YearBuilt:** It shows the Original construction date*
# * **YrSold:** It shows the original Year Sold (YYYY)
# * **YearRemodAdd:** It shows Remodel date (same as construction date if no remodeling or additions).
# 
# * **Note:**
#     * **First** we can create a **binary feature** that **indicates whether** the **house underwent construction or not.**
#     * **Second** we can create a **Discrete numerical feature** that **indicates** the **age of house**.
#     * This features can **provide valuable information about the remodeling history** of the property and potentially impact the sale price. 

# In[44]:


train_df['RenovationStatus'] = (train_df['YearBuilt'] != train_df['YearRemodAdd']).astype(int)
test_df['RenovationStatus']  = (test_df['YearBuilt'] != test_df['YearRemodAdd']).astype(int)


# In[45]:


train_df['AgeAtSale'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['AgeAtSale'] = test_df['YrSold'] - test_df['YearBuilt']


# #### Dropping the Unwanted Features.

# In[46]:


train_df.drop(columns=["YearBuilt","YrSold","YearRemodAdd"],inplace=True)
test_df.drop(columns=["YearBuilt","YrSold","YearRemodAdd"],inplace=True)


# ### Visualizing the New Features Created.

# In[47]:


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
data=train_df.groupby("RenovationStatus")["SalePrice"].mean()
sns.barplot(data.index,data,palette="Set2")
plt.title("RenovationStatus vs Avg. SalePrice",pad=10,size=20,fontweight="black")
plt.subplot(1,2,2)
sns.regplot(train_df["AgeAtSale"],train_df["SalePrice"],color="black", scatter_kws={'s': 70, 'alpha': 0.5}, 
                line_kws={'color': 'red', 'lw': 3})
plt.title("AgeAtScale vs SalePrice",pad=10,size=20,fontweight="black")
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * The **SalePrice** for both the **RenovationStatus** cateegory is **approxiamately same.**
# * There is a **negative correlation** between **AgeAtScale** & **SalePrice.** So this new feature **seems very useful** for model training.

# ---

# ## 2. Creating a New Feature using all the columns storing "Bathroom Values."

# * **FullBath:** It shows total no. of Full bathrooms above grade.
# * **HalfBath:** It shows total no. of Half bathrooms above grade.
# * **BsmtFullBath:** It shows total no. of Basement full bathrooms.
# * **BsmtHalfBath:** It shows total no. of Basement half bathrooms.
# 
# * **Note:**
#     * **By adding** all these feature values we can create a new feature **Total bathrooms.**

# In[48]:


train_df["Total_Bathrooms"] = (train_df["FullBath"] + (0.5 * train_df["HalfBath"]) + 
                               train_df["BsmtFullBath"] + (0.5 * train_df["BsmtHalfBath"]))


# In[49]:


test_df["Total_Bathrooms"] = (test_df["FullBath"] + (0.5 * test_df["HalfBath"]) + 
                               test_df["BsmtFullBath"] + (0.5 * test_df["BsmtHalfBath"]))


# ### Visualiing "Total_Bathrooms" w.r.t  Average "SalePrice".

# In[50]:


plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.boxplot(train_df["Total_Bathrooms"], train_df["SalePrice"], palette="Set2")
plt.title("Total Batroom vs Sale Price",fontweight="black",size=20,pad=10)

plt.subplot(1,2,2)
avg = train_df.groupby("Total_Bathrooms")["SalePrice"].mean()
sns.barplot(avg.index,avg,palette="Set2")
plt.title("Total Batroom vs Avg. Sale Price",fontweight="black",size=20,pad=10)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * We can clearly observe a **strong positive correlation** between **Total Bathrooms** and **SalePrice**.
# * There's a **class-imbalance** because there are only **1 record** for **Total Bathrooms 5 & 6**.
# * But still this **feature seems important** very important for **predicting Saleprice.**

# ---

# ## 3. Creatng a New Feature using all the columns related to "porch".

# * **WoodDeckSF:**  Wood deck area in square feet.
# * **OpenPorchSF:** Open porch area in square feet.
# * **EnclosedPorch:** Enclosed porch area in square feet.
# * **3SsnPorch:** Three season porch area in square feet.
# * **ScreenPorch:** Screen porch area in square feet.
# * **Note:-**
#     * A **porch** is a covered outdoor **living space attached to a house**, typically used for relaxation or socializing.
#     * So we can create a new feature **Total_Porch_SF** to indicate the **total porch Sqaure Feet** available.

# In[51]:


train_df['Total_Porch_SF'] = (train_df['OpenPorchSF'] + train_df['3SsnPorch'] +train_df['EnclosedPorch'] +
                              train_df['ScreenPorch'] + train_df['WoodDeckSF'])


# In[52]:


test_df['Total_Porch_SF'] = (test_df['OpenPorchSF'] + test_df['3SsnPorch'] +test_df['EnclosedPorch'] +
                              test_df['ScreenPorch'] + test_df['WoodDeckSF'])


# ### Visualizing All These Features.

# In[53]:


cols = ["OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF","Total_Porch_SF"]

plt.figure(figsize=(22,12))
for index,column in enumerate(cols):
    plt.subplot(2,3,index+1)
    sns.regplot(train_df[column],train_df["SalePrice"],color="black", scatter_kws={'s': 70, 'alpha': 0.5}, 
                line_kws={'color': 'red', 'lw': 3})
    corr = round(train_df[[column,"SalePrice"]].corr()["SalePrice"][0],2)
    plt.title(f"Correlation value is {corr}",pad=10,size=20,fontweight="black")
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * Feature like **`OpenPorchSF`**,**`WoodDeckSF`** and **`Total_Porch_SF`** are having **moderate correlation.**So these are **useful features.**
#     
# * Feature like **`3SsnPorch`**,**`EnclosedPorch`**, and **`ScreenPorch`** are having **weak correlation.**So we can simply **drop this features.**

# ### Dropping Features with Weak Correlation.

# In[54]:


cols = ["3SsnPorch","EnclosedPorch","ScreenPorch"]

train_df.drop(columns=cols,inplace=True)
test_df.drop(columns=cols,inplace=True)


# ---

# ## 4. Creating a New Feature Using "Sqaure Footage".

# * **BsmtFinSF1:** Type 1 finished square feet.
# * **BsmtFinSF2:** Type 2 finished square feet.
# * **BsmtUnfSF:** Unfinished square feet of basement area.
# * **TotalBsmtSF:** Total square feet of basement area.
# * **1stFlrSF:** First Floor square feet.
# * **2ndFlrSF:** Second floor square feet.
# * **Note:-**
#     * By **adding all these square footage** values we can create a new feature **Total_sqr_footage.** indicating the total **square footage of house.**
# 

# In[55]:


train_df['Total_sqr_footage']=(train_df['BsmtFinSF1']+train_df['BsmtFinSF2']+train_df['1stFlrSF']+train_df['2ndFlrSF'])
test_df['Total_sqr_footage'] =(test_df['BsmtFinSF1']+test_df['BsmtFinSF2']+test_df['1stFlrSF']+test_df['2ndFlrSF'])


# ### Visualizing All These Features.

# In[56]:


cols = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","Total_sqr_footage"]

plt.figure(figsize=(24,12))
for index,column in enumerate(cols):
    plt.subplot(2,4,index+1)
    sns.regplot(train_df[column],train_df["SalePrice"],color="black", scatter_kws={'s': 70, 'alpha': 0.5}, 
                line_kws={'color': 'red', 'lw': 3})
    corr = round(train_df[[column,"SalePrice"]].corr()["SalePrice"][0],2)
    plt.title(f"Correlation value is {corr}",pad=10,size=20,fontweight="black")
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * The new feature **`Total_sqr_footage`** and **`1stFlrSF`** are having **very high correlation** with the target varibale.
# 
# * Features like **`BsmtFinSF1`**,**`TotalBsmtSF`**, and **`2ndFlrSF`** are having **modearte correlation** with the target variable.
#     
# * Features like **`BsmtFinSF2`** and **`BsmtUnfSF`** are having very **weak correaltion** witht the target variable. So we can simply **drop those featues.**

# ### Dropping Features with weak correaltion.

# In[57]:


cols = ["BsmtFinSF2","BsmtUnfSF"]
train_df.drop(columns=cols,inplace=True)
test_df.drop(columns=cols,inplace=True)


# ---

# ## 5. Creatining a New Feature using "Condition1" & "Condition2".

# * **Condition1:** Proximity to various conditions.
# * **Condition2:** Proximity to various conditions (if more than one is present).
# * **Note:-**
#     * In the **EDA** section we found that **Condition2 feature** wah having **class-imbalace.**
#     * Instead of **keeping two different conditions** we can **combine both** condition1 and condition2.
#     * **After Combining** we can create a new **boolean feature ProximityStatus** indicating **No if its Norm** or else **Yes if there's any proximity.**

# In[58]:


def condition(df):
    df["Condition2"] = df["Condition2"].replace({"Norm":""}) #Norm means normal which indicates there's no second condition
    combined_condition = []
    for val1,val2 in zip(df["Condition1"],df["Condition2"]):
        if val2 == "":
            combined_condition.append(val1)
        elif val1==val2:
            combined_condition.append(val1)
        else:
            combined_condition.append(val1+val2)
            
    df["Combined_Condition"] = combined_condition
    df["ProximityStatus"] = (df["Combined_Condition"] == "Norm").astype(int)


# In[59]:


condition(train_df)
condition(test_df)


# ### Dropping Columns Which are not Required Anymore.

# In[60]:


train_df.drop(columns=["Condition1","Condition2","Combined_Condition"],inplace=True)
test_df.drop(columns=["Condition1","Condition2","Combined_Condition"],inplace=True)


# ---

# ## 6. Creating New Feature using "Heating" Feature.

# * **Heating:** Type of heating.
# * **HeatingQC:** Heating quality and condition.
# * **Note:**
#     * While performing **EDA** we found **huge class-imbalance** in **Heating** Feature.
#     * To **reduce class-imbalance** we can create a new feature by **concatenating both the features.**

# In[61]:


train_df["HeatingQuality"] = train_df["Heating"] + "-" + train_df["HeatingQC"]
test_df["HeatingQuality"] = test_df["Heating"] + "-" + test_df["HeatingQC"]


# #### Dropping Features which are not required anymore.

# In[62]:


train_df.drop(columns=["Heating","HeatingQC"],inplace=True)
test_df.drop(columns=["Heating","HeatingQC"],inplace=True)


# In[63]:


test_df["HeatingQuality"].replace({"Wall-Po":"Wall-TA"},inplace=True)


# ---

# ## 7. Creating Some New Boolean Features.

# In[64]:


def boolean_feature(df):
    df["Has2ndFloor"] = (df['2ndFlrSF'] != 0).astype(int)
    df["HasGarage"]  = (df["GarageArea"] !=0).astype(int)
    df["HasBsmt"]    = (df["TotalBsmtSF"]!=0).astype(int)
    df["HasFirePlace"] = (df["Fireplaces"]!=0).astype(int) 


# In[65]:


boolean_feature(train_df)
boolean_feature(test_df)


# ### Visualizing All These New Boolean Features.

# In[66]:


plt.figure(figsize=(22,6))
for index,column in enumerate(["Has2ndFloor","HasGarage","HasBsmt","HasFirePlace"]):
    plt.subplot(1,4,index+1)
    sns.boxplot(x=column, y="SalePrice", data=train_df, palette="Set2")
    plt.title(f"{column} vs SalePrice",fontweight="black",pad=10,size=20)
    plt.xticks(rotation=90)
    plt.tight_layout()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * All these features **seems very useful** for **model training.**

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⚙️ Data Preprocessing - Part 2 ⚙️</b></div>

# ### 1. Performing Log Transformation on Target variable.

# In[67]:


z = train_df["SalePrice"]

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])


# In[68]:


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
sns.histplot(z,color="red",kde=True)
plt.title("SalePrice Before Transformation",size=20,pad=10,fontweight="black")

plt.subplot(1,2,2)
sns.histplot(train_df["SalePrice"],color="blue",kde=True)
plt.title("SalePrice Before Transformation",size=20,pad=10,fontweight="black")
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * We can clearly onserve that **SalePrice** has been transformed to a **normal distribution**.
#     
# * This will help model in **`Homoscedasticity`**,**`Interpretability`** and **`Model Performance`**.

# ---

# ### 2. Applying Box-Cox Transformation on Continous Numerical Features to Reduce Skewness.

# In[69]:


con_cols = ["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF",
            "GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","PoolArea","MiscVal","AgeAtSale","Total_Porch_SF",
            "Total_sqr_footage"]


# In[70]:


train_df[con_cols].skew().sort_values().to_frame().rename(columns={0:"Skewness"}).T


# In[71]:


for feature in con_cols:
    train_df[feature] = boxcox1p(train_df[feature], boxcox_normmax(train_df[feature] + 1))
    test_df[feature] = boxcox1p(test_df[feature] + 1, boxcox_normmax(train_df[feature] + 1))


# In[72]:


train_df[con_cols].skew().sort_values().to_frame().rename(columns={0:"Skewness"})


# ---

# ### 3. Dropping Features with High Skewness Values.

# In[73]:


cols = ["MiscVal","LowQualFinSF","PoolArea"]

train_df.drop(columns=cols, inplace=True)
test_df.drop(columns=cols, inplace=True)


# ---

# ### 4. Performing Target Encoding on Categorical Features with High Cardinality.

# In[74]:


cols = ["Neighborhood","Exterior1st","Exterior2nd","HeatingQuality"]
for column in cols:
    data = train_df.groupby(column)["SalePrice"].mean()
    for value in data.index:
        train_df[column] = train_df[column].replace({value:data[value]})
        test_df[column] = test_df[column].replace({value:data[value]})


# ---

# ### 5. Performing Label Encoding on Other Features.

# In[75]:


cols = ["HouseStyle","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","Electrical","KitchenQual",
        "GarageQual","GarageCond"]


# In[76]:


encoder = LabelEncoder()

train_df[cols] = train_df[cols].apply(encoder.fit_transform)
test_df[cols] = test_df[cols].apply(encoder.fit_transform)


# ---

# ### 6. Applying One-Hot Encoding on Nominal Categorical Columns.

# In[77]:


cols = train_df.select_dtypes(include="object").columns


# In[78]:


train_df = pd.get_dummies(train_df, columns=cols)
test_df = pd.get_dummies(test_df,columns=cols)


# In[79]:


train_df.shape


# In[80]:


test_df.shape


# ---

# ### 7. Segregating Features and Labels For Model Training.

# In[81]:


X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"]


# ---

# ### 8. Feature Scaling using RobustScaler.

# In[82]:


scaler =RobustScaler()


# In[83]:


X_scaled = scaler.fit_transform(X)
test_df = scaler.fit_transform(test_df)


# ---

# ### 9. Splitting Data For Model Training & Testing.

# In[84]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=0)


# In[85]:


print("Dimension of x_train:=>",x_train.shape)
print("Dimension of x_test:=>",x_test.shape)
print("Dimension of y_train:=>",y_train.shape)
print("Dimension of y_test:=>",y_test.shape)


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🎯 Model Creation & Evaluation 🎯</b></div>

# ## Creating  a Function to Train Model using Different Regression Algorithms.

# In[86]:


r2_value = []
adjusted_r2_value = []
mae_value = []
mse_value = []
rmse_value = []


# In[87]:


def model_evaluation(model):
    model.fit(x_train, y_train)
    y_train_pred= model.predict(x_train)
    y_test_pred = model.predict(x_test)

    #Metrics Calculation.
    mae = mean_absolute_error(y_test,y_test_pred)
    mse = mean_squared_error(y_test,y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_test_pred)
    adjusted_r2 = 1 - ((1-r2)*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))
   
    mae_value.append(mae)
    mse_value.append(mse)
    rmse_value.append(rmse)
    r2_value.append(r2)
    adjusted_r2_value.append(adjusted_r2) 
    
    print(f"R2 Score of the {model} model is=>",r2)
    print(f"Adjusted R2 Score of the {model} model is=>",adjusted_r2)
    print()
    print(f"MAE of {model} model is=>",mae)
    print(f"MSE of {model} model is=>",mse)
    print(f"RMSE of {model} model is=>",rmse)
    

    # Scatter plot.
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)    
    plt.scatter(y_train, y_train_pred, color='blue', label='Train')
    plt.scatter(y_test, y_test_pred, color='red', label='Test')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.title('Scatter Plot',fontweight="black",size=20,pad=10)
    
    # Residual plot.
    plt.subplot(1,2,2)
    plt.scatter(y_train_pred, y_train_pred - y_train, color='blue', label='Train')
    plt.scatter(y_test_pred, y_test_pred - y_test, color='red', label='Test')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.title('Residual Plot',fontweight="black",size=20,pad=10)
    plt.show()


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">1. Creating Linear Regression Model.

# In[88]:


model_evaluation(LinearRegression())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 1px; color:black; font-size:200%; text-align:center;padding: 0px;">2. Creating Support vector Regressor Model.

# In[89]:


model_evaluation(SVR())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">3. Creating Random Forest Regressor Model.

# In[90]:


model_evaluation(RandomForestRegressor())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">4. Creating AdaBoost Regressor Model.

# In[91]:


model_evaluation(AdaBoostRegressor())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">5. Creating Gradient Boosting Regressor Model.

# In[92]:


model_evaluation(GradientBoostingRegressor())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">6. Creating LGBM Regressor Model.

# In[93]:


model_evaluation(LGBMRegressor())


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">7. Creating XGBRegressor Model.

# In[94]:


model_evaluation(XGBRegressor())


# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 2px; color:black; font-size:200%; text-align:center;padding: 0px;">8. Creating CatBoost Regressor Model.

# In[95]:


model_evaluation(CatBoostRegressor(verbose=False))


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⚖️ Model's Performance Comparison ⚖️</b></div>

# In[96]:


algos = ["LinearRegression","SVR","RandomForestRegresor","AdaBoostRegressor","GradientBosstRegressor",
         "LGBMRegressor","XGBosstRegressor","CatBoostRegressor"]


# In[97]:


new_df = pd.DataFrame({"Model":algos,"R2_Score":r2_value,"Adjusted_R2_Score":adjusted_r2_value,
                       "MAE":mae_value,"MSE":mse_value,"RMSE":rmse_value})


# In[98]:


new_df


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
# 
# * The **best performing** model is **CatBoostRegressor** with **highest R2 & Adjusted_R2 Scores** and **lowest MAE,MSE,RMSE** values.
#     
# * The **second & third best performing model** is **GradientBoostingRegressor** & **LGBMRegressor** models.
# * So we will perform **Hyper-Parameter-Tunning** on this three model **to obatain more accurate results.**

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⚖️ Model Hyper-Parameter Tunnin ⚖️</b></div>

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 1px; color:black; font-size:150%; text-align:center;padding: 0px;">1. Hyper-Parameter Tunning of CatBoost Regressor Model.

# In[99]:


catboost_model = CatBoostRegressor(verbose=False)


# In[100]:


parameters1 = {"n_estimators":[50,100,150],
               "random_state":[0,42,50],
               "learning_rate":[0.1,0.3,0.5,1.0]}


# In[101]:


grid_search = GridSearchCV(catboost_model, parameters1 , cv=5, n_jobs=-1)
grid_search.fit(x_train,y_train)


# In[102]:


best_parameters = grid_search.best_params_
best_parameters


# ### Creating CatBoost Regressor model using Best Parameters.

# In[103]:


catboost_model = CatBoostRegressor(**best_parameters, verbose=False)


# In[104]:


catboost_model.fit(x_train,y_train)


# In[105]:


y_pred = catboost_model.predict(x_test)


# In[106]:


print("R2_Score of model is:",r2_score(y_test,y_pred))
print("RMSE Score of model is:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Adjusted_R2_Score of model is:",1-((1-r2_score(y_test,y_pred))*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1)))


# ---

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 1px; color:black; font-size:150%; text-align:center;padding: 0px;">2. Hyper-Parameter Tunning of Gradient Boosting Regressor Model.

# In[107]:


gradient_model = GradientBoostingRegressor()


# In[108]:


parameters2 = {"loss":['squared_error', 'absolute_error', 'huber', 'quantile'],
               "learning_rate":[0.1,0.3,0.5,1.0],
               "n_estimators":[50,100,150],
               "random_state":[0,42,45,50]}


# In[109]:


grid_search_2 = GridSearchCV(gradient_model, parameters2, cv=5)
grid_search_2.fit(x_train,y_train)


# In[110]:


best_parameters2 = grid_search_2.best_params_
best_parameters2


# 
# ### Creating GradientBossting Regressor Model Using Best-Parameters.

# In[111]:


gradient_model = GradientBoostingRegressor(**best_parameters)


# In[112]:


gradient_model.fit(x_train,y_train)


# In[113]:


y_pred2 = gradient_model.predict(x_test)


# In[114]:


print("R2_Score of model is:",r2_score(y_test,y_pred2))
print("RMSE Score of model is:",np.sqrt(mean_squared_error(y_test,y_pred2)))
print("Adjusted_R2_Score of model is:",1-((1-r2_score(y_test,y_pred2))*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1)))


# ----

# <div style="border-radius:10px; border:black solid; padding: 15px; background-color: aliceblue; font-size:100%; text-align:left">
# <p style="font-family:Georgia; font-weight:bold; letter-spacing: 1px; color:black; font-size:150%; text-align:center;padding: 0px;">3. Hyper-Parameter Tunning of LGBM Regressor Model.

# In[115]:


lgbm_model = LGBMRegressor()


# In[116]:


parameters3 = {"boosting_type":['gbdt','dart','goss','rf'],
               "learning_rate":[0.1,0.3,0.5,1.0],
               "random_state":[0,42,45,50]}


# In[117]:


grid_search_3 = GridSearchCV(lgbm_model, parameters3, cv=5)
grid_search_3.fit(x_train,y_train)


# In[118]:


best_parameters3 = grid_search_3.best_params_
best_parameters3


# ### Creating LGBM Model using Best Parameters.

# In[119]:


lgbm_model = LGBMRegressor(**best_parameters3)


# In[120]:


lgbm_model.fit(x_train,y_train)


# In[121]:


y_pred3 = lgbm_model.predict(x_test)


# In[122]:


print("R2_Score of model is:",r2_score(y_test,y_pred3))
print("RMSE Score of model is:",np.sqrt(mean_squared_error(y_test,y_pred3)))
print("Adjusted_R2_Score of model is:",1-((1-r2_score(y_test,y_pred3))*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1)))


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🍀 Creating Stacked Model 🍀</b></div>

# In[123]:


stack_model = StackingCVRegressor(regressors=(catboost_model,gradient_model,lgbm_model),
                                  meta_regressor = catboost_model,
                                  use_features_in_secondary=True)


# In[124]:


stack_model.fit(x_train,y_train)


# In[125]:


y_pred = stack_model.predict(x_test)


# In[126]:


print("R2_Score of model is:",r2_score(y_test,y_pred))
print("RMSE Score of model is:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Adjusted_R2_Score of model is:",1-((1-r2_score(y_test,y_pred))*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1)))


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>📊 Inference:</font></h3>
#     
# * The model demonstrates a strong correlation **(R2 Score) of 0.876** between predicted and actual house prices.
# * The **RMSE Score of 0.137** indicates a low average error in the model's predictions.
# * The **Adjusted R2 Score of 0.756** accounts for the number of predictors in the model, providing a reliable measure of its performance.
# * These results highlight the model's **high accuracy and reliability** in predicting house prices.
# * The model's performance can guide **homeowners, buyers, and real estate professionals in making informed decisions** regarding property values.

# ---

# # <div style="padding:20px;color:white;margin:0;font-size:32px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🌼  Predicting Test Datset using Stacked Model 🌼 </b></div>

# In[127]:


test_preds = stack_model.predict(test_df)


# In[128]:


sdf = test_id.to_frame()
sdf["SalePrice"] = np.floor(np.expm1(test_preds))


# In[129]:


sdf


# In[130]:


sdf.to_csv("C:\\Users\\kumod sharma\\Desktop\\Sub2.csv",index=False)


# ---

# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🎈 Conclusion 🎈</b></div>

# <div style="border-radius:10px;border:black solid;padding: 15px;background-color:white;font-size:110%;text-align:left">
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:17px">
# 
# 
# <h3 align="left"><font color=purple>📝 Key Findings:</font></h3><br> 
#     
# 1. Features like <b>`1stFlrSF`</b>,<b>`GrLivArea`</b>,and <b>`GarageArea`</b> are having <b>strong relation</b> with the target variable.<br>
# 2. The <b>best performing</b> model is <b>CatBoostRegressor</b> with <b>highest R2 & Adjusted_R2 Scores</b> and <b>lowest MAE,MSE,RMSE</b> values.<br>
# 3. The <b>second & third best performing model</b> is <b>GradientBoostingRegressor</b> & <b>LGBMRegressor</b> models.<br>
# 4. The <b>stacked model performance</b> was impressive becuase of <b>hight accuracy and low error rates</b>.<br>
# 5. The project developed a house price prediction model with <b>strong performance metrics.</b><br>
# 6. The project effectively addresses the task of house price prediction and contributes as a <b>valuable tool</b> in the dynamic real estate industry.<br>

# ---
