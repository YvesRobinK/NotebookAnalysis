#!/usr/bin/env python
# coding: utf-8

# ## <div style="text-align: center">LEARN FEATURE ENGINEERING AND FEATURE SELECTION TECHNIQUES </div>
# <div style="text-align:center"><img src="https://brainstation-23.com/wp-content/uploads/2018/12/ML-real-state.png"></div>

# PC - BRAIN STATION 23

# ## I hope this kernel helpful and some <font color='red'><b>UPVOTES</b></font> would be very much appreciated

# <a id='top'></a> <br>
# ## NOTEBOOK CONTENT
# 1. [IMPORTS](#1)
# 1. [LOAD DATA](#2)
# 1. [DATA SNEAK PEAK](#3)
#     1. [UNIQUE WAY TO SEE MISSING VALUES](#3-1)
# 1. [DATA SCIENCE WORKFLOW](#4)
# 1. [PROFILE REPORT](#5)
# 1. [FEATURE SELECTION TECHNIQUES FOR NUMERICAL VARIABLES](#6)
# 1. [DATA CLEANING](#7)
# 1. [UNIVARIATE SELECTION](#8)
# 1. [FEATURE IMPORTANCE](#9)
# 1. [RANDOM FOREST](#10)
# 1. [WORKING WITH TEST DATA](#11)
# 1. [TO MAKE CSV FILE FOR SUBMISSION](#12)
# 1. [CATEGORICAL DATA](#13)
#     1. [HANDLING MISSING CATEGORICAL DATA](#13-1)
# 1. [DATA VISUALISATION FOR CATEGORICAL VARIABLES](#14)
# 1. [One Hot Encoding Categorical Columns](#15)

# <a id="1"></a> <br>
# ## 1-IMPORTS

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for data visualzation
import seaborn as sns
import matplotlib.pyplot as plt

import pandas_profiling # LIBRARY TO see all the details of data
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="2"></a> <br>
# ## 2-LOAD DATA

# In[2]:


train_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')


# In[3]:


test_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
# test_id=test_data.pop('Id')
test_data.shape


# <a id=3></a><br>
# ## 3-DATA SNEAK PEAK

# In[4]:


print(train_data.dtypes.unique())
print(len(list(train_data.columns))) # we have total 81 columns with 1 target column and 80 variables
# train_data.columns
# train_id=train_data.pop('Id') # since Id iss not going to be used in Prediction so etter to pop it

num_col=train_data.select_dtypes(exclude='object')
cat_col=train_data.select_dtypes(exclude=['int64','float64'])

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
num_col.describe(include = 'all')


# <a id=3-1></a> <br>
# ## 3.1-UNIQUE TECHNIQUE TO SEE MISSING VALUES

# In[5]:


# HEATMAP TO SEE MISSING VALUES
plt.figure(figsize=(15,5))
sns.heatmap(num_col.isnull(),yticklabels=0,cbar=False,cmap='viridis')


# So the heatmap shows **LotFrontage**,**MasVnrArea**,**GarageYrBlt** have the missing values

# <a id=4></a> <br>
# ## 4-DATA SCIENCE WORKFLOW
# 
# There is no template for solving a data science problem. The roadmap changes with every new dataset and new problem. But we do see similar steps in many different projects. I wanted to make a clean workflow to serve as an example to aspiring data scientists. 
# 
# <img src="https://miro.medium.com/max/2000/1*3FQbrDoP1w1oibNPj9YeDw.png"> 
# 
# ### Overview:
# 
# - Source the Data 
# - Data Processing
# - Modeling
# - Model Deployment
# - Model Monitoring 
# - Exploration and reporting
# 
# ## In this notebook we will focus mainly on <font color='red'><b>STEP-2(DATA CLEANING)</b></font>
# 
# 

# <a id=5></a> <br>
# ## 5-PROFILE REPORT  (PERSONAL FAV.)

# <font color='red'>WARNING</font> The Implementation of this cell takes resources

# In[6]:


# num_col.profile_report() # this will show profile report only for numerical variables because we are using dataframe having only numerical variables


# <a id=6></a> <br>
# # 6-FEATURE SELECTION TECHNIQUES FOR NUMERICAL VARIABLES
# ### 1-UNIVARIATE SELECTION
# ### 2-FEATURE IMPORTANCE
# ### 3-CORRELATION MATRIX
# 

# <a id=7></a> <br>
# ## 7-DATA CLEANING

# In[7]:


X=num_col.copy() #  all numerical variables
y=X.pop('SalePrice') # storing target variable in y
X.isnull().sum()


# In[8]:


X.isna().sum().reset_index() 


# Very Important Link
# [replace pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)

# 1.**MasVnrArea**

# Delaing with **MasVnrArea**
#     So MasVnrArea is  **Masonry veneer area** of house so we can take mean of **MasVnrArea** and replace all nun values of column
# 

# In[9]:


#correlation map
f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(X.corr().iloc[7:8,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
# this shows that MasVnrArea is not highly corelated to any other feature


# In[10]:


sns.kdeplot(X.MasVnrArea,Label='MasVnrArea',color='g')


# This and profilereport above shows that most of the values (nearly 60%) values of MasVnrArea have **zero** value so replace nan values here with **ZERO**

# In[11]:


X.MasVnrArea.replace({np.nan:0},inplace=True)


# 2. **GarageYrBlt**

# In[12]:


f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(X.corr().iloc[24:25,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)


# This shows that **GarageYrBlt** is highly corelated to **YearBuilt**
# 
# And One Important Reason that we cant drop **GarageYrBlt** because it has significant corelation with our predicctor variable **SalePrice** 

# In[13]:


sns.kdeplot(X.GarageYrBlt,Label='GarageYrBlt',color='g')


# In[14]:


X.GarageYrBlt.describe()


# Looking at the Kdeplot for **GarageYrBlt** and Description we find that data in this column is not spread enough so we can use **mean** of this column to fill its Missing Values

# In[15]:


X['GarageYrBlt'].replace({np.nan:X.GarageYrBlt.mean()},inplace=True)


# 3. **LotFrontage**

# In[16]:


f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(X.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)


# **LotFrontage**: Linear feet of street connected to property
# 
# And One Important Reason that we cant drop **LotFrontage** because it has significant corelation with our predicctor variable **SalePrice** 
# 

# In[17]:


sns.kdeplot(X.LotFrontage,Label='LotFrontage',color='g')


# In[18]:


X.LotFrontage.describe()


# Looking at the Kde Plot and Description  of **LotFrontage** we can replace Nan Values of this column either by **Mean** or **Median** Because data is almost Normal distribution
# 
# I would Choose Mean to replace NaN values

# In[19]:


X['LotFrontage'].replace({np.nan:X.LotFrontage.mean()},inplace=True)


# So we are done with **data cleaning** part 

# <a id=8></a> <br>
# ## 8-UNIVARIATE SELECTION
# **CHI2** Test for Univariate Selection
# ### 1. CHI2 Test only applies for Positive Value
# ### 2. CHI2 should only be applied to columns that do not have any NAN values

# In[20]:


from sklearn.feature_selection import SelectKBest # SELECT K  BEST  is used to first top k features from variables list
from sklearn.feature_selection import chi2 # import chi1 function


# In[21]:


# # apply SelectKBest class to extract top 30 best features
bestfeatures = SelectKBest(score_func=chi2, k=30)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(30,'Score'))  #print  TOP 30 best features


# <a id=9></a> <br>
# ## 9-FEATURE IMPORTANCE
# ### Extra Tree Classifier also uses positive values and is not applicable for NaN,Infinite values

# In[22]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh',figsize=(30,10))


# In[23]:


features_tree=set(list(feat_importances.nlargest(30).index)) # top 35 features by tree classifier
features_chi=set(list(featureScores.Specs[:30]))# top 30 features by chi square test


# ### Selcting Union of Features from both the ways

# In[24]:


union_fe=features_chi.union(features_tree)
union_fe=list(union_fe)


# In[25]:


X=X[union_fe] # SELCTING TOP fEATURES FROM FEATURE SET


# <a id=10></a> <br>
# ## 10-RANDOM FOREST
# Random Forest is a trademark term for an ensemble of decision trees. In Random Forest, we’ve collection of decision trees (so known as “Forest”). To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

# In[26]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tt
from sklearn.ensemble import RandomForestRegressor as rr
train_X,val_X,train_Y,val_Y=tt(X,y,random_state=23)
forest_model=rr(random_state=12,max_depth=9,n_estimators=200)
forest_model.fit(X,y)

prediction=forest_model.predict(val_X)
print(mae(val_Y,prediction))


# In[ ]:





# <a id=11></a> <br>
# ## 11-WORKING WITH TEST DATA

# In[27]:


test_X=test_data.select_dtypes(exclude=['object']) # way to select all numerical variables
test_X.head()


# ## HANDLE MISSING VALUES OF <font color='red'>Test Data</font>

# 1. <font color='red'>**LotFrontage**</font>
# 
# Use the same criteria that we used to handle missing values of <font color='red'>LotFrontage</font> in training data

# In[28]:


test_X['LotFrontage'].replace({np.nan:test_X.LotFrontage.mean()},inplace=True)


# 2. <font color='red'>**MasVnrArea**</font>
# 
# Use the same criteria that we used to handle missing values of <font color='red'>MasVnrArea</font> in training data

# In[29]:


test_X.MasVnrArea.replace({np.nan:0},inplace=True)


# 3. <font color='red'>**GarageYrBlt**</font>
# 
# Use the same criteria that we used to handle missing values of <font color='red'>GarageYrBlt</font> in training data

# In[30]:


test_X['GarageYrBlt'].replace({np.nan:test_X.GarageYrBlt.mean()},inplace=True)


# 4. <font color='red'>**TotalBsmtSF**</font>
# 
# We have only 1 missing values in this column so just replace it with mean of the column
# Although data is varying alot but still the median and mean of data are nearly same so i chosse mean

# In[31]:


sns.kdeplot(test_X.TotalBsmtSF,label='TotalBsmtSF')
print(test_X.TotalBsmtSF.describe())


# In[32]:


test_X['TotalBsmtSF'].replace({np.nan:test_X.TotalBsmtSF.mean()},inplace=True)


# 
# 5. <font color='red'> **BsmtFinSF1**</font>
# 
# We can see from description of this column that there is lot of gap between 75th percentile and max value so that is the reason why mean is so high even 25 percentile is equal to zero.
# So better to use median to replace NaN values
# 

# In[33]:


sns.kdeplot(test_X.BsmtFinSF1,label='BsmtFinSF1')
print(test_X.BsmtFinSF1.describe())


# In[34]:


test_X['BsmtFinSF1'].replace({np.nan:test_X.BsmtFinSF1.median()},inplace=True)


# 6. <font color='red'>**BsmtFinSF2**</font>
# 
# As we see 75 percent of the value are zero so better to replace missing value with 0

# In[35]:


sns.kdeplot(test_X.BsmtFinSF2,label='BsmtFinSF2')
print(test_X.BsmtFinSF2.describe())


# In[36]:


test_X['BsmtFinSF2'].replace({np.nan:0},inplace=True)


# 7. <font color='red'>**GarageArea**</font>
# 
# We have only 1 missing values in this column so just replace it with mean of the column

# In[37]:


sns.kdeplot(test_X.GarageArea,label='GarageArea')
print(test_X.GarageArea.describe())


# In[38]:


test_X['GarageArea'].replace({np.nan:test_X.GarageArea.mean()},inplace=True)


# 8. <font color='red'>**BsmtUnfSF**</font>
# 
# As it is highly varied data so to use median to replace NaN value

# In[39]:


sns.kdeplot(test_X.BsmtUnfSF,label='BsmtUnfSF')
print(test_X.BsmtUnfSF.describe())


# In[40]:


# BsmtUnfSF
test_X['BsmtUnfSF'].replace({np.nan:test_X.BsmtUnfSF.median()},inplace=True)


# 9. <font color='red'>**BsmtFullBath**</font>
# 
# 75 percentile of values are 0 so replace NaN with 0

# In[41]:


sns.kdeplot(test_X.BsmtFullBath,label='BsmtUnfSF')
print(test_X.BsmtFullBath.describe())


# In[42]:


test_X['BsmtFullBath'].replace({np.nan:0},inplace=True)


# 10. <font color='red'>**BsmtHalfBath**</font>

# In[43]:


# BsmtHalfBath
sns.kdeplot(test_X.BsmtHalfBath,label='BsmtHalfBath')
print(test_X.BsmtHalfBath.describe())


# In[44]:


test_X['BsmtHalfBath'].replace({np.nan:0},inplace=True)


# 11. <font color='red'>**GarageCars**</font>
# 

# In[45]:


sns.kdeplot(test_X.GarageCars,label='BsmtHalfBath')
print(test_X.GarageCars.describe())


# In[46]:


test_X['GarageCars'].replace({np.nan:test_X.GarageCars.median()},inplace=True)


# In[47]:


# test_X.isnull().sum().sort_values(ascending=False)


# In[48]:


test_X.columns


# In[49]:


train_X.columns


# ## Data cleaning for test data done

# In[50]:


test_X.shape


# In[51]:


f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(test_X.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)


# In[52]:


# test_X=test_X[union_fe]
# make predictions which we will submit. 
test_preds = forest_model.predict(test_X)
test_preds.shape
# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission12.csv', index=False)
# test_X.columns


# <a id=12></a> <br>
# ## 12-TO MAKE CSV FILE FOR SUBMISSION

# In[53]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = output

# create a link to download the dataframe
create_download_link(df)


# <a id=13></a> <br>
# ## 13-CATEGORICAL DATA

# In[54]:


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

cat_col_name=[cname for cname in cat_col.columns if cat_col[cname].nunique() < 10 and 
                        cat_col[cname].dtype == "object"]


# In[55]:


cat_col_name


# In[56]:


num_col_name=list(X.columns)
num_col_name


# In[57]:


features=cat_col_name+num_col_name
features


# In[58]:


train_data=train_data[features]
train_data=pd.get_dummies(train_data)


# <a id=13-1></a> <br>
# ## 13.1-Handling Missing  Values in Categorical data

# In[59]:


#idxmax() function returns index of first occurrence of maximum over requested axis.
#While finding the index of the maximum value across any index, all NA/null values are excluded.

#These columns has only few value missing as compared to total number so i choose to replace the NaN values with the most frequent values

#MasVnrType
cat_col['MasVnrType'].replace({np.nan:cat_col.MasVnrType.value_counts().idxmax()},inplace=True)

#BsmtQual
cat_col['BsmtQual'].replace({np.nan:cat_col.BsmtQual.value_counts().idxmax()},inplace=True)

#BsmtCond
cat_col['BsmtCond'].replace({np.nan:cat_col.BsmtCond.value_counts().idxmax()},inplace=True)

#BsmtExposure
cat_col['BsmtExposure'].replace({np.nan:cat_col.BsmtExposure.value_counts().idxmax()},inplace=True)

#BsmtFinType1
cat_col['BsmtFinType1'].replace({np.nan:cat_col.BsmtFinType1.value_counts().idxmax()},inplace=True)

#BsmtFinType2
cat_col['BsmtFinType2'].replace({np.nan:cat_col.BsmtFinType2.value_counts().idxmax()},inplace=True)

#Electrical
cat_col['Electrical'].replace({np.nan:cat_col.Electrical.value_counts().idxmax()},inplace=True)

#GarageType
cat_col['GarageType'].replace({np.nan:cat_col.GarageType.value_counts().idxmax()},inplace=True)

#GarageFinish
cat_col['GarageFinish'].replace({np.nan:cat_col.GarageFinish.value_counts().idxmax()},inplace=True)

#GarageQual
cat_col['GarageQual'].replace({np.nan:cat_col.GarageQual.value_counts().idxmax()},inplace=True)

#GarageCond
cat_col['GarageCond'].replace({np.nan:cat_col.GarageCond.value_counts().idxmax()},inplace=True)




# In[60]:


# Now filling Missing Value for FireplaceQu column
print(cat_col.FireplaceQu.value_counts())
# and Missing Values in this column are 690 so we will Replace nan with 'Unknown'
cat_col.FireplaceQu.replace({np.nan:'Unknown'},inplace=True)


# ### So we have finally filled all the Missing Values in Categorical columns

# <a id=14></a> <br>
# ## 14-DATA VISUALISATION FOR CATEGORICAL VARIABLES

# ## Street vs SalePrice

# In[61]:


f, axes = plt.subplots(1, 2,figsize=(12, 8))
g = sns.swarmplot(x=cat_col.Street,y=y,ax=axes[0]) # y is Saleprice
g = g.set_ylabel("Sale Price for Diffrent Streets")

labels=['Pave','Grvl']
slices=[cat_col.loc[cat_col.Street=="Pave"].shape[0],cat_col.loc[cat_col.Street=="Grvl"].shape[0]]
plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0.5,0.7),autopct='%1.2f%%',colors=['#99ff99','#ffcc99'])
plt.show()


# This clearly shows that **Pave** street has more Saleprices as compared to **Grvl**
# and very Interesting thing 
# Most of the people(99.59%) prefer **Pave** Street Access as compared to **Grvl**
# and in **Pave** section also most of the Houses cost under **400,000 $**

# In[62]:


list_of_col=list(cat_col.columns)
dict_of_col={i:list_of_col[i] for i in range(len(list_of_col))}
dict_of_col


# Observation of First 5 Variables with **SalePrice**

# In[63]:


temp_list=list(i for i in range(5))
f,axes=plt.subplots(1, 5,figsize=(20,8))
f.subplots_adjust(hspace=0.5)

for j in temp_list:
        g = sns.swarmplot(x=cat_col[dict_of_col[j]],y=y,ax=axes[j]) # y is Saleprice
        g.set_title(label=dict_of_col[j].upper(),fontsize=20)
        g.set_xlabel(str(dict_of_col[j]),fontsize=25)
        g.set_ylabel('SalePrice',fontsize=25)
        plt.tight_layout() # to increase gapping between subplots

    


# 1. RL is the most choosen street zone from all 5 zones and has  the maximum prices
# 1. Pave is Highly Preferred Street access than Grvl
# 1. IR1 and Reg are highly preferred LotShape

# Observing next 2 **variables**

# In[64]:


f,axes=plt.subplots(1, 2,figsize=(15,8))
f.subplots_adjust(hspace=0.5)
for j,i in zip(temp_list,range(5,7)):
        g = sns.swarmplot(x=cat_col[dict_of_col[i]],y=y,ax=axes[j]) # y is Saleprice
        g.set_title(label=dict_of_col[i].upper(),fontsize=20)
        g.set_xlabel(str(dict_of_col[i]),fontsize=25)
        g.set_ylabel('SalePrice',fontsize=25)
        plt.tight_layout() # to increase gapping between subplots


# Similarly we can observe each **Categorical variables**

# ## Next we will look how to Handle and use these categorical variables in our model

# Looking at all the **diffrent** values in each column

# In[65]:


for i in cat_col.columns:
    print(i,'\t',cat_col[str(i)].unique(),'\n')


# In[66]:


from sklearn.preprocessing import LabelEncoder as le,OneHotEncoder as oe


# ## Label Encoding one Column with Sklearn 

# In[67]:


st_le=le()
street_labels=st_le.fit_transform(cat_col.Street) # meaning of fit_transform is that it will asign appropriate numbers for each categorical variables
street_mapping={index:label for index,label in enumerate(st_le.classes_)}
street_mapping


# In[68]:


temp_col=cat_col.copy() # doing our work on a copy of datframe


# In[69]:


temp_col['Street_Labels']=street_labels
temp_col.head()


# ## One Hot encoding a Column ith Sklearn

# In[70]:


st_oe=oe(handle_unknown='ignore')
st_feature_arr=st_oe.fit_transform(temp_col[['Street_Labels']]).toarray()
st_feature_labels=list(st_le.classes_)
st_features=pd.DataFrame(st_feature_arr,columns=st_feature_labels)


# In[71]:


temp_X_train=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
temp_X_test=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')


# Fitting a label encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them. Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn', but these don't appear in the training data -- thus, if we try to use a label encoder with scikit-learn, the code will throw an error.
# 
# This is a common problem that you'll encounter with real-world data, and there are many approaches to fixing this issue. For instance, you can write a custom label encoder to deal with new categories. The simplest approach, however, is to drop the problematic categorical columns.
# 
# Run the code cell below to save the problematic columns to a Python list bad_label_cols. Likewise, columns that can be safely label encoded are stored in good_label_cols.
# 
# **Bad_Label_cols** are those columns that the values are not the same between the 2 dataset. In this case the **training** and **testing**
# 
# 

# In[72]:


# All categorical columns
object_cols = [col for col in temp_X_train.columns if temp_X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(temp_X_train[col]) == set(temp_X_test[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


# So finally we will use **cat_col** dataframe and **good_label_cols** for one hot encoding and later will be used for prediction
# First of all we will remove **Alley,PoolQC,Fence,MiscFeature** from good labels list because these columns have more than 80% values as **NaN**
# and then we will use **good_label_cols** as list of features  for cat_col dataframe

# In[73]:


good_label_cols=list(set(good_label_cols)-set(['Alley','PoolQC','Fence','MiscFeature']))
print(len(cat_col.columns))
cat_col=cat_col[good_label_cols]
print(len(cat_col.columns))


# 
# 
# [GO to top](#top)
