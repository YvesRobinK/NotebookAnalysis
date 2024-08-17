#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# In this kernel, I have deeply explained all numeric features with visualization. Many of the features play a significant role to predict SalesPrice of home. There are also some features exist in data, which may not alone work, but if we concat that feature with other feature. It performs outstanding. 
# 
# I highly suggest checking out [this lernel](https://www.kaggle.com/bhavikapanara/how-to-detect-an-outlier)
# to understand Outlier detection techniques.

#     Let's load Libraries

# In[1]:


from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew 
from scipy.special import boxcox1p
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 100) 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os


# In[2]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.shape,test.shape


#     Let's get the list of numeric features.

# In[3]:


num_feat = [col for col in train.columns if train[col].dtype != 'object']
print("Number of numeric feature in data :",len(num_feat))


# # Data Analysis for one by one feature

# In[4]:


def num_analysis(i=False, feat_name=False):
    
    if (i):
        feat = num_feat[i]
    
    elif (feat_name):
        feat = feat_name
        
    print("Feature: {}  & Correlation with target: {}".format(feat,train[feat].corr(train['SalePrice'])))
            
    fig=plt.figure(figsize=(20,4))

    ax=fig.add_subplot(1,3,1)
    ax.hist(train[feat])
    ax.set_title("train", fontsize = 20)
    ax.set_xlabel(feat,fontsize=20)
    ax.tick_params(labelsize=13)
    
    ax=fig.add_subplot(1,3,2)
    ax.scatter(train[feat],train['SalePrice'])
    ax.set_title(feat+" vs SalesPrice", fontsize = 20)
    ax.set_xlabel(feat,fontsize=20)
    ax.set_ylabel("SalesPrice",fontsize=20)
    ax.tick_params(labelsize=13)
    
    ax=fig.add_subplot(1,3,3)
    ax.hist(test[feat])
    ax.set_title("test", fontsize = 20)
    ax.set_xlabel(feat,fontsize=20)
    ax.tick_params(labelsize=13)
    
    plt.show()


# In[5]:


k=1
num_analysis(i = k)


# Actually, MSSubClass feature is a categorical feature. To make this feature more senseful we would apply   a label encoding

# In[6]:


train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

lbl = LabelEncoder() 
lbl.fit(list(train['MSSubClass'].values) + list(test['MSSubClass'].values)) 

train['MSSubClass'] = lbl.transform(list(train['MSSubClass'].values))
test['MSSubClass'] = lbl.transform(list(test['MSSubClass'].values))

num_analysis(feat_name='MSSubClass')


# In[7]:


k= k +1
num_analysis(i = k)


# Here, May be these two outlier exists with LotFrontage > 300 and SalesProce < 250000

# Let's see the details of these two outlier

# In[8]:


dff = train.loc[(train['LotFrontage'] >= 300) & (train['SalePrice'] < 200000)]
dff


#     Let's find some helpful insights from this house.

#     LotFrontage : 300 | LotArea:63887 | OverallQual:10 | BsmtQual:Ex | ExterQual:Ex | TotalBsmtSF:6110 | GrLivArea:5642 | KitchenQual:Ex | TotRmsAbvGrd:12 | GarageArea:1418 | PoolArea:480 | SaleType:New
# 
#     YearBuilt:2008  | YrSold:2008  (It's new House)

# By Observing the house feature, can conclude that the house's Salesprice is low with respect to all of its features.

# In[9]:


k = k+1
num_analysis(i = k)


#     This feature hasn't uniform distribution. Let's check skewness of LotArea  feature

# # Skewness & Kurtosis

# Skewness is a measure of the symmetry in a distribution. 
# 
# A symmetrical dataset will have a skewness equal to 0.  So, a normal distribution will have a skewness of 0.   Skewness essentially measures the relative size of the two tails. 
# 
# If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
# 
# If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed
# 
# If the skewness is less than -1 or greater than 1, the data are highly skewed
# 
# Kurtosis is a measure of the combined sizes of the two tails.  It measures the amount of probability in the tails.  
# 
# The value is often compared to the kurtosis of the normal distribution, which is equal to 3.  If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).  If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution (less in the tails).  

# In[10]:


from IPython.display import Image
Image("../input/skew-img/skew_kurt.png")


# In[11]:


print("Skewness: %f" % train['LotArea'].skew())
print("Kurtosis: %f" % train['LotArea'].kurt())


# The distribution of LotArea feature is highly skewed.
# 
# So, we should apply logithmic function to  LotArea feature to make it less skeded. Let's do it.

# In[12]:


train["LotArea"] = np.log1p(train["LotArea"])
test["LotArea"] = np.log1p(test["LotArea"])
num_analysis(i = k)


#     Now, It's look much better and also correaltion with SalesPrice(target) is also increased from  0.2638 to 0.3835.

# Let's move to next feature 

# In[13]:


k = k + 1
num_analysis(i =k)


#     OverallQual feature looks good as it has higher correaltion with taregt SalesPrice.

# Next Feature:

# In[14]:


k = k+ 1
num_analysis(i=k)


#     OverallQual: Rates the overall material and finish of the house
# 
#     OverallCond: Rates the overall condition of the house
# 
#     OverallQual feature positivly correaltion with target variable SalesPrice. means higher OverallQual value of house sold at higher price. 
#  
#     But, OverallCond hasn't any relationship with target variable(SalesPrice). 
# 
#     Above scatter plot depicts that, some house with OverallCond value 5 has sold at higher price. Hence, this feature alone isn't helpful.

# In[15]:


train.loc[(train['OverallCond'] == 6) & (train['SalePrice'] > 600000)]


#     Let's make new feature of OverallCond & OverallQual

# In[16]:


train['cond*qual'] = (train['OverallCond'] * train['OverallQual']) / 100.0
test['cond*qual'] = (test['OverallCond'] * test['OverallQual']) / 100.0

num_analysis(feat_name='cond*qual')


#     This generated new feature is very useful. When I had trained the LightGBM model, this feature took the fist position in feature importance graph. trust me.. It works.
# 
#     Also, you should remove the OverallCond & OverallQual feature.

# In[17]:


k = k+ 1
num_analysis(i=k)


#     YearBuilt feature has somewhat linear relationship with SalesPrice.
# 
#     If you remember, the data also contains one feature YrSold (year sold). 
#     Using these two feature YearBuilt and YrSold, we can generate new feature home age. It means how the house was old when it Sold.

# In[18]:


train['home_age_when_sold'] = train['YrSold'] - train['YearBuilt']
test['home_age_when_sold'] = test['YrSold'] - test['YearBuilt']


# In[19]:


num_analysis(feat_name='home_age_when_sold')


#     Now, you need to remove YrSold & YearBuilt features, as we have already create one feature using it.  

# Let's move to next feature

# In[20]:


k = k +1 
num_analysis(i=k)


#     We can also generate one new feature using YearRemodAdd feature.
# 
#     YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 
#     new_feature is Home age after renovation

# In[21]:


train['after_remodel_home_age_when_sold'] = train['YrSold'] - train['YearRemodAdd']
test['after_remodel_home_age_when_sold'] = test['YrSold'] - test['YearRemodAdd']


# In[22]:


num_analysis(feat_name='after_remodel_home_age_when_sold')


#     If you use this new feature, you must remove YearRemodAdd feature. Otherwise It create redundancy.

# Next Feature:

# In[23]:


k = k + 1
num_analysis(i=k)


# In[24]:


print("Skewness: %f" % train['MasVnrArea'].skew())
print("Kurtosis: %f" % train['MasVnrArea'].kurt())


# In[25]:


train["MasVnrArea"] = np.log1p(train["MasVnrArea"])
test["MasVnrArea"] = np.log1p(test["MasVnrArea"])
num_analysis(i = k)


# In[26]:


k = k +1
num_analysis(i=k)


#     BsmtFinSF1: Type 1 (basement) finished square feet
# 
#     In train data,  one home has more are of basement and sold at low price.
#     Let's see details of the this feature

# In[27]:


train.loc[(train['BsmtFinSF1'] > 4500) & (train['SalePrice'] < 200000)]


#     Again the same home (index = 1298).
#     
#     we have already discover this home above.

# In[28]:


k = k +1
num_analysis(i=k)


#     There are lots of home with 0 value of BsmtFinSF2 feature.

# In[29]:


print("Number of home which has BsmtFinSF2 in train : {}%" .format(((train['BsmtFinSF2']!=0).sum() / train.shape[0])*100))
print("Number of home which has BsmtFinSF2 in test  : {}%" .format(((test['BsmtFinSF2']!=0).sum() / test.shape[0])*100))


#     BsmtFinSF2 feature is not looking useful.
#     
#     We could make new feature using BsmtFinSF1 & BsmtFinSF2
#     
#     BsmtFinSF1: Type 1 finished square feet
#     BsmtFinSF2: Type 2 finished square feet

# In[30]:


train['BsmtFinSF1+BsmtFinSF2'] = train['BsmtFinSF1'] + train['BsmtFinSF2']
test['BsmtFinSF1+BsmtFinSF2'] = test['BsmtFinSF1'] + test['BsmtFinSF2']

num_analysis(feat_name='BsmtFinSF1+BsmtFinSF2')


#     If we use this new feature, we must remove BsmtFinSF1 & BsmtFinSF2 feature as we have already use it.

# In[31]:


k = k +1
num_analysis(i=k)


# In[32]:


k = k +1
num_analysis(i=k)


#     One Outlier may be detected with TotalBsmtSF>6000 and SalesPrice is low. Let's see details of that home

# In[33]:


train.loc[(train['TotalBsmtSF']>=6000 ) &(train['SalePrice'] < 200000)]


#     Again the same house (index = 1298)
#     We already discover this home above.

# In[34]:


k = k +1
num_analysis(i=k)


#     1stFlrSF: First Floor square feet
# 
#     Outlier exists. 1stFlrSF value > 4000 & salesPrice < 100000
# 
#     Let's find it

# In[35]:


train.loc[(train['1stFlrSF'] >= 4000) & (train['SalePrice'] < 200000)]


#     One more time index=1298 home detected as outlier

# In[36]:


k = k +1
num_analysis(i=k)


#     2ndFlrSF feature is linearly correlation with target SalesPrice.
#     
#     Using 1stFlrSF, 2ndFlrSF and TotalBsmtSF features, we would generate new feature total_SF
#     
#     TotalBsmtSF: Total square feet of basement area
#     1stFlrSF: First Floor square feet
#     2ndFlrSF: Second floor square feet

# In[37]:


train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

num_analysis(feat_name='TotalSF')


#     this created new TotalSF feature is highly correalted with Salesprice. Great!

# In[38]:


k = k +1
num_analysis(i=k)


#     This feature LowQualFinSF alone may not be useful, as many of house doesn't have value for this feature. But, we could use this feature with another feature and create new feature.

# In[39]:


k = k +1
num_analysis(i=k)


# GrLivArea: Above grade (ground) living area square feet
# 
# From above plot, there may be a Outlier exists with GrLivArea > 5000 and salesPrice is too low. 
# 
# Let's find it.

# In[40]:


train.loc[(train['GrLivArea'] >= 5000) & (train['SalePrice'] < 200000)]


#     again same house detected (index=1298), which we have already discovered ambiguity above.

# In[41]:


k = k +1
num_analysis(i=k)


# In[42]:


k = k +1
num_analysis(i=k)


# In[43]:


k = k +1
num_analysis(i=k)


# In[44]:


k = k +1
num_analysis(i=k)


#     we can generate one new feature using these all bathoom related features.
#     
#     BsmtFullBath: Basement full bathrooms
# 
#     BsmtHalfBath: Basement half bathrooms
# 
#     FullBath: Full bathrooms above grade
# 
#     HalfBath: Half baths above grade

# In[45]:


train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))
test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

num_analysis(feat_name='Total_Bathrooms')


#     This new generated Total_Bathrooms feature has good correlation with target Salesprice feature.

# In[46]:


k = k +1
num_analysis(i=k)


# In[47]:


k = k +1
num_analysis(i=k)


#     Something Interesting!!
# 
#     There is a house in train data which have 3 kitchens and its Sold at very low price.
# 
#     Let's find that house

# In[48]:


train.loc[(train['KitchenAbvGr'] == 3) & (train['SalePrice'] < 150000)]


#     SalesPrice of these house are less as they are too old.

# In[49]:


k = k +1
num_analysis(i=k)


#     This feature has uniform distribution. It's good
# 
#     In train data, one house exist with 14 total bedrooms. Let's see all details of this home.

# In[50]:


train.loc[(train['TotRmsAbvGrd'] == 14) & (train['SalePrice'] <= 200000)]


#     This home is also too old when it sold (93 year old home)

# In[51]:


k = k +1
num_analysis(i=k)


#     Observing Graph plot, there are such home exists with 3 fireplaces and there SalesPrice not much.
# 
#     Let's see these home

# In[52]:


train.loc[(train['Fireplaces'] == 3)]


#     There are such 5 home exists.
# 
#     means, number of Fireplace is alone not important, it's also depends on the quality of the Fireplace.
# 
#     So, we can generate one new feature from these two feature Fireplaces & FireplaceQu, which make more sense.

#     Let's see the feature FireplaceQu(It's categorical) feature

# FireplaceQu: Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace

#     To make it more senseful, we should encode this catecorical feature in descending order
# 
#     like, Ex:5 | Gd:4 | TA:3 | Fa:2 | Po:1 | NA:0

#     Let's check missing values in these two feature.

# In[53]:


print("Missing value count in Fireplaces  :",train['Fireplaces'].isna().sum())
print("Missing value count in FireplaceQu :",train['FireplaceQu'].isna().sum())


#     Let's fill missing value in FireplaceQu with NA.

# In[54]:


train["FireplaceQu"] = train["FireplaceQu"].fillna("NA")
test["FireplaceQu"] = test["FireplaceQu"].fillna("NA")


# In[55]:


def FireplaceQu_encode(x):
    if x=='Ex':
        return 5
    elif x=='Gd':
        return 4
    elif x == 'TA':
        return 3
    elif x=='Fa':
        return 2
    elif x=='Po':
        return 1
    elif x=='NA':
        return 0


# In[56]:


train['FireplaceQu'] = train['FireplaceQu'].apply(lambda x: FireplaceQu_encode(x))
test['FireplaceQu'] = test['FireplaceQu'].apply(lambda x: FireplaceQu_encode(x))


# In[57]:


num_analysis(feat_name='FireplaceQu')


# # Generate New Feature

# In[58]:


train['FirePlace*FireplaceQu'] = train['Fireplaces']*train['FireplaceQu']
test['FirePlace*FireplaceQu'] = test['Fireplaces']*test['FireplaceQu']


# In[59]:


num_analysis(feat_name='FirePlace*FireplaceQu')


# In[60]:


k = k +1
num_analysis(i=k)


#     Something look surprise in histogram of test data.
# 
#     There is a home which GarageYrBlt value > 2100
# 
#     Let's see deatils of it

# In[61]:


test.loc[(test['GarageYrBlt'] >= 2100)]


# # one more outlier Detected

#     this home details says, Garage was built in year 2207... Not possible at all.

# In[62]:


k = k +1
num_analysis(i=k)


#     GarageCars: Size of garage in car capacity
#     It's also helpful feature

# In[63]:


k = k +1
num_analysis(i=k)


#     GarageArea feature look uniform distribution and linearly correaltion with target SalesPrice.

# In[64]:


k = k +1
num_analysis(i=k)


# In[65]:


k = k +1
num_analysis(i=k)


# In[66]:


k = k +1
num_analysis(i=k)


# In[67]:


k = k +1
num_analysis(i=k)


# In[68]:


k = k +1
num_analysis(i=k)


#     Let's create one combined feature "total_porch_area"

# In[69]:


train['total_porch_area'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['total_porch_area'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']

num_analysis(feat_name='total_porch_area')


# In[70]:


k = k +1
num_analysis(i=k)


#     Most of the home doesn't have pool.
#     
#     Let's check howmany house has pool in train & test

# In[71]:


print("Number of house which has Pool in train :",(train['PoolArea'] != 0).sum())
print("Number of house which has Pool in test  :",(test['PoolArea'] != 0).sum())


#     You can discard this feature as 99% of home doesn't have pool

# In[72]:


k = k +1
num_analysis(i=k)


#     From histogram of test data, there is a home exist with more MiscVal value.
# 
#     Let's check test data

# In[73]:


test.loc[test['MiscVal'] >= 15000]


# # One more Outlier Detected

#     the above home in test data have MiscVal cost $17000 and NaN at MiscFeature.
# 
#     In MiscFeature, NaN means No Miscellaneous feature
# 
#     How is it possible?
# 

# In[74]:


k = k +1
num_analysis(i=k)


#     MoSold feature doesn't have any correaltion realation with Salesprice.
# 
#     You can remove this feature. not helpful for prediction

# In[75]:


k = k +1
num_analysis(i=k)


#     We have already made feature home age while sold using YrSold, so you should remove YrSold feature

#     More to come...
#     
#     Stay tuned!

# If You found this kernel helpful to you, Please upvote it. And also If you found any wrong information in kernel, Please comment it.
#     
#     Thanks for reading!

# In[ ]:




