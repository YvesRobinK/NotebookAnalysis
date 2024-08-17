#!/usr/bin/env python
# coding: utf-8

# # *Welcome Kagglers*

# # *EDA  , Feature Engineering and Prediction*

# **It will helpfull for beginners & Intermediate**
# **By this notebook you will get a idia how things work**

# The aim of this project is to build a Machine Learning model in order to predict the appropriate price of a house given a set of features. We decided to divide our analysis into 5 parts:
# 
# 
#    * First look at the problem and general understanding of the variables;
#    * Study the main variable ("SalePrice");
#    * Study how the main variable is related to the other feature;
#    * Data Preprocessing: make some cleaning on our training data set in order to better visualize and estimate;
#    * Build a model in order to predict SalePrice
#    * Explorty data Analysis
# 

# In[1]:


#importing all usefull lib

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error, make_scorer 
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import math
from sklearn.preprocessing import StandardScaler
import warnings as wr
wr.filterwarnings("ignore")


# In[2]:


#uploading training and test data

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[3]:


train_data.head()


# In[4]:


#saving outcome in Sale_Price

Sale_Price=train_data.iloc[:,80]
Sale_Price.shape


# In[5]:


train_data.shape


# In[6]:


#droping SalePrice column
train=train_data.drop(["SalePrice"],axis=1)
train.head()


# In[7]:


test.head()


# In[8]:


test.shape


# # Data Preposesing 
# 
# **combining training & testing data for preposesing after that we do not write same code for test**
# 

# In[9]:


data= pd.concat([train,test], keys=['x', 'y'])#here X is training data and Y testing data
data=data.drop(["Id"],axis=1)


# In[10]:


data.shape


# # Dealing with null values
# 
# **Now our goal is to deal with null values and try to understand for each one what can we do: maybe we can replace them or maybe we can just skip them.**

# In[11]:


plt.figure(figsize=(20,6))
sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='mako')


# In[12]:


total_null = data.isnull().sum().sort_values(ascending=False) #First sum and order all null values for each variable
percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False) #Get the percentage
missing_data = pd.concat([total_null, percentage], axis=1, keys=['Total', 'Percentage'])
missing_data.head(20)


# We have to do some considerations. 
# Let's divide our null values into 2 groups:
#  - __PoolQC__, __MiscFeature__, __Alley__, __Fence__, __FireplaceQu__ and __LotFrontage__.
# These are all variables which presents many null values. In general, by common opinion, we can discourage variables which have more than 15% of missing values. 
# These are not vital information for someone who wants to buy an house, such as __FireplaceQu__ or, for example, many houses doesn't have an __Alley__ access. We can drop them.
# 
# The second group:
#  - __GarageX__ properties
# If we look carefully, all of these variables have the same number of null values! Maybe this can be a strange coincidence, or just that they all refer to the same variable Garage, in which "Na" means "There is no Garage". The same occurs for __BsmtX__ and MasVnr__, which means that we will have to deal with them afterwards.

# In[13]:


data = data.drop((missing_data[missing_data["Percentage"] > 0.05]).index,1) #Drop All Var. with null values > 1

data.isnull().sum()


# **finding numeric column from data**

# In[14]:


num_col=data._get_numeric_data().columns.tolist()
num_col


# **finding catogorical features**

# In[15]:


cat_col=set(data.columns)-set(num_col)
cat_col


# # *filling numrical missing value using fillna*

# In[16]:


for col in num_col:
    data[col].fillna(data[col].mean(),inplace=True)


# # *filling catgorical missing value*

# In[17]:


for col in cat_col:

    data[col].fillna(data[col].mode()[0],inplace=True)


# In[18]:


#count total value in every catgorical feature
for i in cat_col:
    print(data[i].value_counts())


# In[19]:


#droping some unnecessary cat_features bcoz they have 80% + same value and 20% - defertnt values so they can't effect score
df=data.drop(["RoofMatl","Heating","Condition2","BsmtCond","CentralAir","Functional","Electrical",
              "LandSlope","ExterCond","Condition1","GarageArea","BsmtUnfSF","3SsnPorch","MiscVal",
              "BsmtFinType2","Utilities","Street","Exterior2nd","Neighborhood"],axis=1) 


# # EDA

# In[20]:


corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(9,9))
g = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[21]:


var = train_data[train_data.columns[1:]].corr()['SalePrice'][:]
var.sort_values(ascending=False)


# In[22]:


#droping low version feature
df=df.drop(["MoSold","BsmtFinSF2","BsmtHalfBath","OverallCond","YrSold",
            "MSSubClass","EnclosedPorch","KitchenAbvGr","ScreenPorch","2ndFlrSF","OverallQual","GrLivArea"],axis=1)


# In[23]:


df.shape


# In[24]:


#here we checking data summury
df.describe()


# # Our initial considerations 
# Looking forward to our columns, we found some variables which can have an high correlation with our main variable SalePrice:
# - __Year Built__
# - __TotalBsmtSF__
# - __GrLivArea__
# - __PoolArea__
# 
# These are variables related to the conditions of the building, its age and some "extra luxury" features such as __PoolArea__. 
# In principle they are all characteristics which can rise the price of an abitation. 
# Another theory we suggested was to consider mainly the "inner" part of the house, such as __KitchenQual__ or __CentralAir__, but these could be too general features which mainly all the houses can have.
# 
# Now, with these prior hypotesis, let's dive into the "__SalePrice__" analysis.

# In[25]:


#sale price analysis

sns.distplot(train_data['SalePrice']);
print("Skewness coeff. is: %f" % train_data['SalePrice'].skew())
print("Kurtosis coeff. is: %f" % train_data['SalePrice'].kurt())


# These measures of symmetry are useful in order to understand the symmetry of the distribution of our main variable.
# Our distribution is highly skewed and present a longer tail on the right. 
# The high value of kurtosis can determine an higher probability of outliers values.

# In[26]:


sns.kdeplot(data=train_data,x='SalePrice',hue="MoSold",fill=True,common_norm=False,palette="husl")


# **Sale Price Analysis on YearBuilt**

# In[27]:


data_year_trend = pd.concat([train_data['SalePrice'], train_data['YearBuilt']], axis=1)
data_year_trend.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0,800000));


# **Sale price Analysis on TotalBsmtSf**

# In[28]:


data_bsmt_trend = pd.concat([train_data['SalePrice'], train_data['TotalBsmtSF']], axis=1)
data_bsmt_trend.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));


# Sale Price analysis on PoolArea

# In[29]:


data_PoolArea_trend = pd.concat([train_data['SalePrice'], train_data['PoolArea']], axis=1)
data_PoolArea_trend.plot.scatter(x='PoolArea', y='SalePrice', ylim=(0,800000));


# by the above chart be can essly find out the outliers

# saleprice analysis on QverallQual

# In[30]:


data = pd.concat([train_data['SalePrice'], train_data['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# **By these analysis** 
# 
# we discovered that our previsions were quite correct.
# 
# __Year Built__ seems to have a slight relation with our main variable, and people, as we thought, tend to buy newer houses. 
# 
# Instead, for __TotalBsmtSF__ and __GrLivArea__ there seems be a stronger relation with __SalePrice__. 

# # Heatmap Correlation Matrix

# In[31]:


corr_matrix = df.corr()
f, ax1 = plt.subplots(figsize=(12,9)) 
ax1=sns.heatmap(corr_matrix,vmax = 0.9); 


# In[32]:


df.shape


# # Outliers

# In[33]:


#Here we extract the numerical variables, this will come in handy later on

n_features = df.select_dtypes(exclude = ["object"]).columns


# In[ ]:





# In[34]:


#for i in df[n_features]:
    #sns.boxplot(x=df[i])
    #plt.show()


# **here we use one hot encoading to encoad cat_features**

# In[35]:


X=pd.get_dummies(df)
X.shape


# **here we use minmax scaler for scaling numeric fields**

# In[36]:


#scalerX = MinMaxScaler(feature_range=(0, 1))
#X[X.columns] = scalerX.fit_transform(X[X.columns])
scaler=StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])





# In[37]:


#Training data after preproscing

Train_data=X.loc["x"]
Train_data.shape


# In[38]:


#Testing data after preproscing
Test_data=X.loc["y"]
Test_data.shape


# In[39]:


#here we add salePrice column in traning data

Train_data.insert(2,column="SalePrice",value=Sale_Price)
Train_data.head()


# # here we split data in input(x) and output(y)

# In[40]:


x=Train_data.drop(["SalePrice"],axis=True)
y=Train_data["SalePrice"]


# # Model Building using train Data

# **spliting Training data for traning model and cheak score**

# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=40)


# # here we use Random Forest Regressor for model building

# In[42]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators = 50,random_state=40,
                          min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001,min_samples_split=5)
rfr.fit(x_train,y_train)
y_predictrfr = rfr.predict(x_test)

#here we can check our model score
print(rfr.score(x_test,y_test))


# In[43]:


rmse = math.sqrt(mean_squared_error(y_test, rfr.predict(x_test)))

print("mear squares error :",rmse)


# # here we use Decision Tree algo

# In[44]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=140,min_samples_split=5,min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001)
dtr.fit(x_train,y_train)
y_predictdtr = dtr.predict(x_test)
#u can also use GridSearchCV / random Searchcv for hyperperameter tuning
print(dtr.score(x_test,y_test))


# In[45]:


rmse = math.sqrt(mean_squared_error(y_test, dtr.predict(x_test)))
print("RMSE:",rmse)


# # G Boosting

# In[46]:


GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.04,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
#RMSE estimated through the partition of the train set
GBoost.fit(x_train, y_train)
y_predictGB = GBoost.predict(x_test)
print("RMSE: %.4f" % rmse)


# In[47]:


print(GBoost.score(x_test,y_test))


# In[48]:


import numpy as np
red = plt.scatter(np.arange(0,80,5),y_predictGB[0:80:5],color = "red")
green = plt.scatter(np.arange(0,80,5),y_predictrfr[0:80:5],color = "green")
blue = plt.scatter(np.arange(0,80,5),y_predictdtr[0:80:5],color = "blue")
black = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "black")
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Index of Candidate")
plt.ylabel("Home Price")
plt.legend((red,green,blue,black),('GBoost', 'RFR', 'DTR', 'REAL'))
plt.show()


# # Prediction On Testing Data

# In[49]:


#here we see test data here one column is missing that is Saleprice bcoz that is need to predict
Test_data.head()


# In[50]:


Test_data.shape


# In[51]:


#here we predict SalePrice using RFR model
y_model_prerfc = GBoost.predict(Test_data)


# In[52]:


#Here we can See predict Sale Price
y_model_prerfc=np.around(y_model_prerfc,2)
y_model_prerfc


# In[53]:


prediction=np.array(y_model_prerfc).tolist()
test.head()


# In[54]:


test.insert(1,column="SalePrice",value=prediction)
test.head()


# In[55]:


predict_sub=test.drop(test.iloc[:,2:],axis=1)
predict_sub.head()


# In[56]:


predict_sub.shape


# In[57]:


predict_sub.to_csv('Home_predictionsGB.csv',index=False)



# # Please do an up vote if you find useful
# ### Thank you
