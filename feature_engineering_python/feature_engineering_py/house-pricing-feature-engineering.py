#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>Making necessary imports</h3>

# In[2]:


import pandas as pd
import numpy as np

import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


# <h1> Train dataset </h1>

# <h3>Reading the data</h3>

# In[3]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[4]:


data.head(10)


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


plt.hist(data.SalePrice)
plt.title('Distribution of the target colum Sale Price')
plt.xlabel('Price')
plt.ylabel('Number of houses')
plt.show()


# In[8]:


outliers_price = data[data.SalePrice > 450000].index


# In[9]:


data= data.drop(outliers_price)


# In[10]:


plt.hist(data.TotalBsmtSF)


# ## Removing outliers

# In[11]:


outliers_Gr = data[data.GrLivArea >3000].index


# In[12]:


data= data.drop(outliers_Gr)


# In[13]:


outliers_Lot =data[data.LotFrontage > 140].index


# In[14]:


data = data.drop(outliers_Lot)


# In[15]:


outliers_Mas = data[data.MasVnrArea > 600].index


# In[16]:


data = data.drop(outliers_Mas)


# In[17]:


outliers_Bf2 = data[data.BsmtFinSF2 > 400].index


# In[18]:


data = data.drop(outliers_Bf2)


# In[19]:


outliers_Tb=data[data.TotalBsmtSF > 2000].index


# In[20]:


data= data.drop(outliers_Tb)


# In[21]:


data.shape


# <h3>Verification for null and nan values</h3>

# In[22]:


pd.set_option('display.max_rows', 500) 
missing_values_container =data.isna().sum()


# In[23]:


msno.heatmap(data)
plt.show()


# In[24]:


counter = 0
for i in range(len(missing_values_container)):
    if missing_values_container[i] != 0:
        counter +=1
        print(f'Number of missing values in column {data.columns[i]} : {missing_values_container[i]}' '\n')
print(f'Total number of colums with nan of null values {counter}.')
     


# <h3> Data cleaning: columns with nan or null values </h3>

# <h4> MiscFeature </h4>

# In[25]:


data.drop('MiscFeature', axis=1, inplace=True)


# <h4> Alley: Type of alley access to property</h4>

# In[26]:


data.drop('Alley', axis=1, inplace=True)


# <h4> PoolQC: Pool quality </h4>

# In[27]:


data.drop('PoolQC', axis=1, inplace=True)


# <h4>Fence: Fence quality </h4>

# In[28]:


data.drop('Fence', axis=1, inplace=True)


# <h4>LotFrontage: Linear feet of street connected to property</h4>

# In[29]:


plt.hist(data.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')
plt.show()


# In[30]:


print(f'The average distance between the building and the road is {data.LotFrontage.median()}m.')


# In[31]:


missing_lotFrontage = data[data.LotFrontage.isnull()]


# In[32]:


missing_lotFrontage.head(10)


# In[33]:


print(f'The mean housing sales price of the columns with LotFrontage missing values is {missing_lotFrontage.SalePrice.median()}.' '\n')
print(f'The mean sales price in the total dataset is {data.SalePrice.median()}.')


# In[34]:


"""I will take that piece of the data as refence"""

reference_for_LotFrontage_mode = data[(data.SalePrice > 150000) & (data.SalePrice < 250000)]


# In[35]:


print(f'The mode in the chosen part of the dataset of the LotFrontage column is {int(reference_for_LotFrontage_mode.LotFrontage.median())}.')


# In[36]:


data.LotFrontage.fillna(int(reference_for_LotFrontage_mode.LotFrontage.median()), inplace=True)


# In[37]:


data.LotFrontage.isna().sum()


# <h4> FireplaceQu: Fireplace quality </h4>
# 

# In[38]:


data.FireplaceQu.value_counts(sort=False).plot.bar(rot=0)
plt.show()


# In[39]:


reference_for_FireplaceQu =data[(data.FireplaceQu == 'TA') | (data.FireplaceQu =='Gd')]


# In[40]:


print(f'Number of houses with good or average Fireplace quality : {reference_for_FireplaceQu.shape[0]}.')


# In[41]:


print(f'Number of total rows of the dataset with information about the Fireplace quality: {data.shape[0] - data.FireplaceQu.isnull().sum()}')


# In[42]:


print(f'The mean price of the house where the information about the Firequality is missing is {round(data[data.FireplaceQu.isnull()].SalePrice.mean())}. Which is a bit less than the general average price.')


# In[43]:


data.drop('FireplaceQu', axis=1, inplace=True)


# I`m dropping the column because of the big number missing values and the deviation problem

# <h4> Garage linked columns: 5 columns</h4>
# 

# In[44]:


print(f'The average type of garage in the dataset is { data.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is { data.GarageFinish .mode()}.')
print()
print(f'The average garage quality in the dataset is { data.GarageQual .mode()}.')
print()
print(f'The average year when the garages were build in the dataset is { int(data.GarageYrBlt .mode())}.')
print()
print(f'The average condition of the garages  in the dataset is { data.GarageCond  .mode()}.')
print()


# In[45]:


data.GarageType.fillna('Attchd', inplace= True)
data.GarageFinish.fillna('Unf', inplace=True)
data.GarageQual.fillna('TA', inplace = True)
data.GarageYrBlt.fillna(int(data.GarageYrBlt .mode()), inplace=True)
data.GarageCond.fillna('TA', inplace=True)


# In[46]:


data.GarageType.isna().sum()




# In[47]:


data.GarageFinish.isna().sum()


# In[48]:


data.GarageQual.isna().sum()


# In[49]:


data.GarageYrBlt.isna().sum()


# In[50]:


data.GarageCond.isna().sum()


# <h4> Basement linked columns</h4>

# In[51]:


print(f'The average basement condition in the dataset is { data.BsmtCond.mode()}.')
print()
print(f'The average basement quality in the dataset is { data.BsmtQual.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is { data.BsmtFinType1.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is { data.BsmtFinType2.mode()}.')
print()
print(f'The average  type 1 finished square feet in the dataset is { int(data.BsmtFinSF1.median())}.')
print()
print(f'The average  type 2 finished square feet in the dataset is { int(data.BsmtFinSF2.median())}.')
print()
print(f'The average  refers to walkout or garden level walls in the dataset is { data.BsmtExposure.mode()}.')
print()


# In[52]:


data.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')
plt.show()
data.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')
plt.show()
data.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')
plt.show()
data.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')
plt.show()
plt.hist(data.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')
plt.show()
plt.hist(data.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')
plt.show()
data.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')
plt.show()


# In[53]:


data.BsmtCond.fillna('TA', inplace=True)
data.BsmtQual.fillna('TA', inplace=True)
data.BsmtFinType1.fillna('Unf', inplace=True)
data.BsmtFinType2.fillna('Unf', inplace=True)
data.BsmtFinSF1.fillna(int(data.BsmtFinSF1.median()), inplace=True)
data.BsmtFinSF2.fillna(int(data.BsmtFinSF2.median()), inplace=True)
data.BsmtExposure.fillna('No', inplace=True)


# In[54]:


data.isna().sum()


# <h4> Masonry veneer type, Masonry veneer area, Electrical columns </h4>

# In[55]:


data.Electrical.mode()


# In[56]:


data.Electrical.fillna('SBrkr',inplace=True)


# In[57]:


data.MasVnrArea.fillna(int(data.MasVnrArea.median()), inplace=True)


# In[58]:


data.MasVnrType.mode()


# In[59]:


data.MasVnrType.fillna('None', inplace=True)


# In[60]:


data.isna().sum()


# In[61]:


data.shape


# In[ ]:





# In[62]:


"""Not necessary but better"""
drop_cols = ['BsmtFinSF1','BsmtFinSF2','LowQualFinSF','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
             'MiscVal','MoSold','YrSold','1stFlrSF','2ndFlrSF' ,'BsmtUnfSF', 'YearBuilt','YearRemodAdd', 'BldgType','Neighborhood','BsmtQual','Street',
             'LandSlope','RoofMatl','LotConfig','RoofStyle','BsmtHalfBath','Functional','Heating']


# In[63]:


data.drop(drop_cols,axis=1,inplace =True)


# In[ ]:





# In[ ]:





# <h1> Test dataset </h1>
# <h5> The same column should be dropped from the test dataset </h5>

# In[64]:


test_data= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[65]:


test_data.head(5)


# In[66]:


test_data.shape


# <h3>Verification for null and nan values</h3>

# In[67]:


msno.heatmap(test_data)
plt.show()


# In[68]:


test_data.isna().sum()


# In[69]:


test_data.drop('MiscFeature', axis=1, inplace=True)


# In[70]:


test_data.drop('Alley', axis=1, inplace=True)


# In[71]:


test_data.drop('PoolQC', axis=1, inplace=True)


# In[72]:


test_data.drop('Fence', axis=1, inplace=True)


# In[73]:


test_data.drop('FireplaceQu', axis=1, inplace=True)


# <h3> Garage columns </h3>

# In[74]:


print(f'The average type of garage in the dataset is { test_data.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is { test_data.GarageFinish.mode()}.')
print()
print(f'The average garage quality in the dataset is { test_data.GarageQual.mode()}.')
print()
print(f'The average year when the garages were build in the dataset is { int(test_data.GarageYrBlt.mode())}.')
print()
print(f'The average size of garage in car capacity in the dataset is { int(test_data.GarageCars.mode())}.')
print()
print(f'The average size of garage in square feet in the dataset is { int(test_data.GarageArea.mode())}.')
print()

print(f'The average condition of the garages  in the dataset is { test_data.GarageCond  .mode()}.')
print()


# In[75]:


test_data.GarageType.fillna('Attchd', inplace= True)
test_data.GarageFinish.fillna('Unf', inplace=True)
test_data.GarageQual.fillna('TA', inplace = True)
test_data.GarageYrBlt.fillna(int(data.GarageYrBlt.mode()), inplace=True)
test_data.GarageCars.fillna(int(data.GarageCars.mode()), inplace=True)
test_data.GarageArea.fillna(int(data.GarageArea.mode()), inplace=True)
test_data.GarageCond.fillna('TA', inplace=True)


# <h4>LotFrontage: Linear feet of street connected to property </h4>

# In[76]:


plt.hist(test_data.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')
plt.show()


# In[77]:


avg_lotf= int(test_data.LotFrontage.median())


# In[78]:


test_data.LotFrontage.fillna(avg_lotf, inplace=True)


# <h4> Basement  columns:  </h4>

# In[79]:


test_data.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')
plt.show()
test_data.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')
plt.show()
test_data.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')
plt.show()
test_data.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')
plt.show()
plt.hist(test_data.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')
plt.show()
plt.hist(test_data.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')
plt.show()
test_data.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')
plt.show()
plt.hist(test_data.BsmtUnfSF)
plt.title('Unfinished square feet of basement area')
plt.show()
plt.hist(test_data.TotalBsmtSF)
plt.title('Total square feet of basement area')
plt.show()
plt.hist(test_data.BsmtFullBath)
plt.title('Full bathrooms above grade')
plt.show()
plt.hist(test_data.BsmtHalfBath)
plt.title('Half bathrooms above grade')
plt.show()


# In[80]:


test_data.BsmtCond.fillna('TA', inplace=True)
test_data.BsmtQual.fillna('TA', inplace=True)
test_data.BsmtFinType1.fillna('GLQ', inplace=True)
test_data.BsmtFinType2.fillna('Unf', inplace=True)
test_data.BsmtFinSF1.fillna(int(test_data.BsmtFinSF1.median()), inplace=True)
test_data.BsmtFinSF2.fillna(int(test_data.BsmtFinSF2.median()), inplace=True)
test_data.BsmtExposure.fillna('No', inplace=True)
test_data.BsmtUnfSF.fillna(int(test_data.BsmtUnfSF.median()), inplace=True)
test_data.TotalBsmtSF.fillna(int(test_data.TotalBsmtSF.median()), inplace=True)
test_data.BsmtFullBath.fillna(int(test_data.BsmtFullBath.median()), inplace=True)
test_data.BsmtHalfBath.fillna(int(test_data.BsmtHalfBath.median()), inplace=True)


# <h4> Masonry veneer type, Masonry veneer area,General zoning classification, Utilities,  Exterior covering on house, Kitchen quality, Functionality and Sales type columns </h4>

# In[81]:


test_data.MSZoning.mode()
test_data.MSZoning.fillna('RL', inplace=True)


# In[82]:


test_data.Utilities.mode()
test_data.Utilities.fillna('AllPub',inplace=True)


# In[83]:


test_data.Exterior1st.mode()
test_data.Exterior1st.fillna('VinylSd', inplace=True)


# In[84]:


test_data.Exterior2nd.mode()
test_data.Exterior2nd.fillna('VinylSd', inplace=True)


# In[85]:


test_data.MasVnrArea.fillna(int(test_data.MasVnrArea.mean()), inplace=True)


# In[86]:


test_data.MasVnrType.mode()
test_data.MasVnrType.fillna('None', inplace=True)


# In[87]:


test_data.KitchenQual.mode()
test_data.KitchenQual.fillna('TA',inplace=True)


# In[88]:


test_data.Functional.mode()
test_data.Functional.fillna('Typ', inplace=True)


# In[89]:


test_data.SaleType.mode()
test_data.SaleType.fillna('WD', inplace=True)


# In[90]:


test_data.isna().sum()


# In[91]:


test_data.drop(drop_cols,axis=1,inplace=True)


# In[92]:


data.shape, test_data.shape


# In[93]:


print(f'Train data has {data.shape[1]} including the target column, Test data has {test_data.shape[1]} . The data is ready for training ')


# <h2> Exporting the data </h2>

# In[94]:


data.to_csv('train_data.csv', index=False)


# In[95]:


test_data.to_csv('test_data.csv', index=False)

