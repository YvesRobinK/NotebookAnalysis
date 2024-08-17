#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


len(df.select_dtypes(include='object').columns)


# In[5]:


df.head()


# In[6]:


df.shape,df1.shape


# In[7]:


df.dtypes


# In[8]:


na = df.isna().sum()/len(df)


# In[9]:


na[na>.50]


# In[10]:


df = df.drop(columns=['PoolQC','Id'])
df1 = df1.drop(columns=['PoolQC','Id'])


# In[11]:


df['Alley'] = df['Alley'].fillna('check')
print(df.groupby('Alley')['SalePrice'].mean())
df['Alley'] = df['Alley'].replace({'check':0,'Grvl':1,'Pave':2})

df1['Alley'] = df1['Alley'].fillna('check')
df1['Alley'] = df1['Alley'].replace({'check':0,'Grvl':1,'Pave':2})


# In[12]:


df['Fence'] = df['Fence'].fillna('check')
print(df.groupby('Fence')['SalePrice'].mean())
df['Fence'] = df['Fence'].replace({'check':0,'MnPrv':1,'MnWw':1,'GdPrv':2,'GdWo':2})

df1['Fence'] = df1['Fence'].fillna('check')
df1['Fence'] = df1['Fence'].replace({'check':0,'MnPrv':1,'MnWw':1,'GdPrv':2,'GdWo':2})


# In[13]:


df['MiscFeature'] = df['MiscFeature'].fillna('check')
print(df.groupby('MiscFeature')['SalePrice'].mean())
df['MiscFeature'] = df['MiscFeature'].replace({'check':0,'Othr':1,'Shed':2,'Gar2':3,'TenC':4})

df1['MiscFeature'] = df1['MiscFeature'].fillna('check')
df1['MiscFeature'] = df1['MiscFeature'].replace({'check':0,'Othr':1,'Shed':2,'Gar2':3,'TenC':4})


# In[14]:


df['Street'] = df['Street'].fillna('check')
print(df.groupby('Street')['SalePrice'].mean())
df['Street'] = df['Street'].replace({'Grvl':0,'Pave':1})

df1['Street'] = df1['Street'].fillna('check')
df1['Street'] = df1['Street'].replace({'Grvl':0,'Pave':1})


# In[15]:


df['Utilities'] = df['Utilities'].fillna('check')
print(df.groupby('Utilities')['SalePrice'].mean())
df['Utilities'] = df['Utilities'].replace({'NoSeWa':0,'AllPub':1})

df1['Utilities'] = df1['Utilities'].fillna('check')
df1['Utilities'] = df1['Utilities'].replace({'NoSeWa':0,'AllPub':1,'check':0})


# In[16]:


df['CentralAir'] = df['CentralAir'].fillna('check')
print(df.groupby('CentralAir')['SalePrice'].mean())
df['CentralAir'] = df['CentralAir'].replace({'N':0,'Y':1})

df1['CentralAir'] = df1['CentralAir'].replace({'N':0,'Y':1})


# In[17]:


df['LandSlope'] = df['LandSlope'].fillna('check')
print(df.groupby('LandSlope')['SalePrice'].mean())
df['LandSlope'] = df['LandSlope'].replace({'Gtl':0,'Mod':1,'Sev':2})

df1['LandSlope'] = df1['LandSlope'].fillna('check')
df1['LandSlope'] = df1['LandSlope'].replace({'Gtl':0,'Mod':1,'Sev':2})


# In[18]:


df['GarageFinish'] = df['GarageFinish'].fillna('check')
print(df.groupby('GarageFinish')['SalePrice'].mean())
df['GarageFinish'] = df['GarageFinish'].replace({'check':0,'Unf':1,'RFn':2,'Fin':3})

df1['GarageFinish'] = df1['GarageFinish'].fillna('check')
df1['GarageFinish'] = df1['GarageFinish'].replace({'check':0,'Unf':1,'RFn':2,'Fin':3})


# In[19]:


df['PavedDrive'] = df['PavedDrive'].fillna('check')
print(df.groupby('PavedDrive')['SalePrice'].mean())
df['PavedDrive'] = df['PavedDrive'].replace({'N':0,'P':1,'Y':2})

df1['PavedDrive'] = df1['PavedDrive'].fillna('check')
df1['PavedDrive'] = df1['PavedDrive'].replace({'N':0,'P':1,'Y':2})


# In[20]:


df['LandContour'] = df['LandContour'].fillna('check')
print(df.groupby('LandContour')['SalePrice'].mean())
df['LandContour'] = df['LandContour'].replace({'Bnk':0,'Lvl':1,'Low':2,'HLS':3})

df1['LandContour'] = df1['LandContour'].fillna('check')
df1['LandContour'] = df1['LandContour'].replace({'Bnk':0,'Lvl':1,'Low':2,'HLS':3})


# In[21]:


df['LotShape'] = df['LotShape'].fillna('check')
print(df.groupby('LotShape')['SalePrice'].mean())
df['LotShape'] = df['LotShape'].replace({'Reg':0,'IR1':1,'IR3':2,'IR2':3})

df1['LotShape'] = df1['LotShape'].fillna('check')
df1['LotShape'] = df1['LotShape'].replace({'Reg':0,'IR1':1,'IR3':2,'IR2':3})


# In[22]:


df['ExterQual'] = df['ExterQual'].fillna('check')
print(df.groupby('ExterQual')['SalePrice'].mean())
df['ExterQual'] = df['ExterQual'].replace({'Fa':0,'TA':1,'Gd':2,'Ex':3})

df1['ExterQual'] = df1['ExterQual'].fillna('check')
df1['ExterQual'] = df1['ExterQual'].replace({'Fa':0,'TA':1,'Gd':2,'Ex':3})


# In[23]:


df['MasVnrType'] = df['MasVnrType'].fillna('check')
print(df.groupby('MasVnrType')['SalePrice'].mean())
df['MasVnrType'] = df['MasVnrType'].replace({'BrkCmn':0,'None':1,'BrkFace':2,'check':3,'Stone':4})

df1['MasVnrType'] = df1['MasVnrType'].fillna('check')
df1['MasVnrType'] = df1['MasVnrType'].replace({'BrkCmn':0,'None':1,'BrkFace':2,'check':3,'Stone':4})


# In[24]:


df['MSZoning'] = df['MSZoning'].fillna('check')
print(df.groupby('MSZoning')['SalePrice'].median())
df['MSZoning'] = df['MSZoning'].replace({'C (all)':0,'RM':1,'RH':2,'RL':3,'FV':4})

df1['MSZoning'] = df1['MSZoning'].fillna('check')
df1['MSZoning'] = df1['MSZoning'].replace({'C (all)':0,'RM':1,'RH':2,'RL':3,'FV':4,'check':2})


# In[25]:


df['LotConfig'] = df['LotConfig'].fillna('check')
print(df.groupby('LotConfig')['SalePrice'].median())
df['LotConfig'] = df['LotConfig'].replace({'Corner':0,'FR2':0,'Inside':0,'CulDSac':1,'FR3':1})

df1['LotConfig'] = df1['LotConfig'].fillna('check')
df1['LotConfig'] = df1['LotConfig'].replace({'Corner':0,'FR2':0,'Inside':0,'CulDSac':1,'FR3':1})


# In[26]:


df['BldgType'] = df['BldgType'].fillna('check')
print(df.groupby('BldgType')['SalePrice'].median())
df['BldgType'] = df['BldgType'].replace({'Twnhs':0,'2fmCon':0,'Duplex':0,'1Fam':1,'TwnhsE':1})

df1['BldgType'] = df1['BldgType'].fillna('check')
df1['BldgType'] = df1['BldgType'].replace({'Twnhs':0,'2fmCon':0,'Duplex':0,'1Fam':1,'TwnhsE':1})


# In[27]:


df['BsmtQual'] = df['BsmtQual'].fillna('check')
print(df.groupby('BsmtQual')['SalePrice'].median())
df['BsmtQual'] = df['BsmtQual'].replace({'check':0,'Fa':0,'TA':1,'Gd':2,'Ex':3})

df1['BsmtQual'] = df1['BsmtQual'].fillna('check')
df1['BsmtQual'] = df1['BsmtQual'].replace({'check':0,'Fa':0,'TA':1,'Gd':2,'Ex':3})


# In[28]:


df['RoofStyle'] = df['RoofStyle'].fillna('check')
print(df.groupby('RoofStyle')['SalePrice'].mean())
df['RoofStyle'] = df['RoofStyle'].replace({'Gambrel':0,'Gable':1,'Mansard':1,'Flat':1,'Shed':2,'Hip':2})

df1['RoofStyle'] = df1['RoofStyle'].fillna('check')
df1['RoofStyle'] = df1['RoofStyle'].replace({'Gambrel':0,'Gable':1,'Mansard':1,'Flat':1,'Shed':2,'Hip':2})


# In[29]:


df['BsmtCond'] = df['BsmtCond'].fillna('check')
print(df.groupby('BsmtCond')['SalePrice'].mean())
df['BsmtCond'] = df['BsmtCond'].replace({'Po':0,'check':1,'Fa':1,'TA':2,'Gd':2})

df1['BsmtCond'] = df1['BsmtCond'].fillna('check')
df1['BsmtCond'] = df1['BsmtCond'].replace({'Po':0,'check':1,'Fa':1,'TA':2,'Gd':2})


# In[30]:


df['BsmtExposure'] = df['BsmtExposure'].fillna('check')
print(df.groupby('BsmtExposure')['SalePrice'].mean())
df['BsmtExposure'] = df['BsmtExposure'].replace({'No':1,'check':0,'Mn':1,'Av':1,'Gd':2})

df1['BsmtExposure'] = df1['BsmtExposure'].fillna('check')
df1['BsmtExposure'] = df1['BsmtExposure'].replace({'No':1,'check':0,'Mn':1,'Av':1,'Gd':2})


# In[31]:


df['KitchenQual'] = df['KitchenQual'].fillna('check')
print(df.groupby('KitchenQual')['SalePrice'].mean())
df['KitchenQual'] = df['KitchenQual'].replace({'Fa':0,'TA':0,'Gd':1,'Ex':2})

df1['KitchenQual'] = df1['KitchenQual'].fillna('check')
df1['KitchenQual'] = df1['KitchenQual'].replace({'Fa':0,'TA':0,'Gd':1,'Ex':2,'check':0})


# In[32]:


df['ExterCond'] = df['ExterCond'].fillna('check')
print(df.groupby('ExterCond')['SalePrice'].mean())
df['ExterCond'] = df['ExterCond'].replace({'Fa':0,'Po':0,'Gd':1,'TA':1,'Ex':2})

df1['ExterCond'] = df1['ExterCond'].fillna('check')
df1['ExterCond'] = df1['ExterCond'].replace({'Fa':0,'Po':0,'Gd':1,'TA':1,'Ex':2})


# In[33]:


df['Electrical'] = df['Electrical'].fillna('check')
print(df.groupby('Electrical')['SalePrice'].mean())
df['Electrical'] = df['Electrical'].replace({'FuseP':0,'Mix':0,'FuseA':1,'FuseF':1,'SBrkr':2,'check':2})

df1['Electrical'] = df1['Electrical'].fillna('check')
df1['Electrical'] = df1['Electrical'].replace({'FuseP':0,'Mix':0,'FuseA':1,'FuseF':1,'SBrkr':2,'check':2})


# In[34]:


df['Heating'] = df['Heating'].fillna('check')
print(df.groupby('Heating')['SalePrice'].mean())
df['Heating'] = df['Heating'].replace({'Floor':0,'Grav':0,'Wall':1,'OthW':2,'GasW':3,'GasA':4})

df1['Heating'] = df1['Heating'].fillna('check')
df1['Heating'] = df1['Heating'].replace({'Floor':0,'Grav':0,'Wall':1,'OthW':2,'GasW':3,'GasA':4})


# In[35]:


df['HeatingQC'] = df['HeatingQC'].fillna('check')
print(df.groupby('HeatingQC')['SalePrice'].mean())
df['HeatingQC'] = df['HeatingQC'].replace({'Po':0,'Fa':1,'TA':2,'Gd':2,'Ex':3})

df1['HeatingQC'] = df1['HeatingQC'].fillna('check')
df1['HeatingQC'] = df1['HeatingQC'].replace({'Po':0,'Fa':1,'TA':2,'Gd':2,'Ex':3})


# In[36]:


df['FireplaceQu'] = df['FireplaceQu'].fillna('check')
print(df.groupby('FireplaceQu')['SalePrice'].mean())
df['FireplaceQu'] = df['FireplaceQu'].replace({'Po':0,'check':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

df1['FireplaceQu'] = df1['FireplaceQu'].fillna('check')
df1['FireplaceQu'] = df1['FireplaceQu'].replace({'Po':0,'check':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})


# In[37]:


df['GarageQual'] = df['GarageQual'].fillna('check')
print(df.groupby('GarageQual')['SalePrice'].mean())
df['GarageQual'] = df['GarageQual'].replace({'Po':0,'check':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

df1['GarageQual'] = df1['GarageQual'].fillna('check')
df1['GarageQual'] = df1['GarageQual'].replace({'Po':0,'check':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})


# In[38]:


df['GarageCond'] = df['GarageCond'].fillna('check')
print(df.groupby('GarageCond')['SalePrice'].mean())
df['GarageCond'] = df['GarageCond'].replace({'Po':0,'check':0,'Fa':1,'TA':2,'Gd':2,'Ex':1})

df1['GarageCond'] = df1['GarageCond'].fillna('check')
df1['GarageCond'] = df1['GarageCond'].replace({'Po':0,'check':0,'Fa':1,'TA':2,'Gd':2,'Ex':1})


# In[39]:


df['Foundation'] = df['Foundation'].fillna('check')
print(df.groupby('Foundation')['SalePrice'].mean())
df['Foundation'] = df['Foundation'].replace({'Slab':0,'BrkTil':1,'CBlock':2,'Stone':3,'Wood':4,'PConc':5})

df1['Foundation'] = df1['Foundation'].fillna('check')
df1['Foundation'] = df1['Foundation'].replace({'Slab':0,'BrkTil':1,'CBlock':2,'Stone':3,'Wood':4,'PConc':5})


# In[40]:


df['BsmtFinType1'] = df['BsmtFinType1'].fillna('check')
print(df.groupby('BsmtFinType1')['SalePrice'].mean())
df['BsmtFinType1'] = df['BsmtFinType1'].replace({'check':0,'Rec':1,'BLQ':1,'LwQ':1,'ALQ':2,'Unf':2,'GLQ':3})

df1['BsmtFinType1'] = df1['BsmtFinType1'].fillna('check')
df1['BsmtFinType1'] = df1['BsmtFinType1'].replace({'check':0,'Rec':1,'BLQ':1,'LwQ':1,'ALQ':2,'Unf':2,'GLQ':3})


# In[41]:


df['BsmtFinType2'] = df['BsmtFinType2'].fillna('check')
print(df.groupby('BsmtFinType2')['SalePrice'].mean())
df['BsmtFinType2'] = df['BsmtFinType2'].replace({'check':0,'Rec':1,'BLQ':1,'LwQ':1,'ALQ':3,'Unf':2,'GLQ':3})

df1['BsmtFinType2'] = df1['BsmtFinType2'].fillna('check')
df1['BsmtFinType2'] = df1['BsmtFinType2'].replace({'check':0,'Rec':1,'BLQ':1,'LwQ':1,'ALQ':3,'Unf':2,'GLQ':2})


# In[42]:


df['GarageType'] = df['GarageType'].fillna('check')
print(df.groupby('GarageType')['SalePrice'].mean())
df['GarageType'] = df['GarageType'].replace({'check':0,'CarPort':0,'Detchd':1,'2Types':2,'Basment':3,'Attchd':4,'BuiltIn':5})

df1['GarageType'] = df1['GarageType'].fillna('check')
df1['GarageType'] = df1['GarageType'].replace({'check':0,'CarPort':0,'Detchd':1,'2Types':2,'Basment':3,'Attchd':4,'BuiltIn':5})


# In[43]:


df['SaleCondition'] = df['SaleCondition'].fillna('check')
print(df.groupby('SaleCondition')['SalePrice'].mean())
df['SaleCondition'] = df['SaleCondition'].replace({'AdjLand':0,'Abnorml':1,'Family':1,'Alloca':2,'Normal':2,'Partial':3})

df1['SaleCondition'] = df1['SaleCondition'].fillna('check')
df1['SaleCondition'] = df1['SaleCondition'].replace({'AdjLand':0,'Abnorml':1,'Family':1,'Alloca':2,'Normal':2,'Partial':3})


# In[44]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[45]:


df[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']] = df[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']].fillna('aaaaaa')
df[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']] = df[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']].astype(str)


# In[46]:


df1[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']] = df1[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']].fillna('aaaaaa')
df1[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']] = df1[['Neighborhood','Exterior1st','Exterior2nd','SaleType','Condition1','Condition2','HouseStyle','RoofMatl','Functional']].astype(str)


# In[47]:


df['Neighborhood']= label_encoder.fit_transform(df['Neighborhood'])
df['Exterior1st']= label_encoder.fit_transform(df['Exterior1st'])
df['Exterior2nd']= label_encoder.fit_transform(df['Exterior2nd'])
df['SaleType']= label_encoder.fit_transform(df['SaleType'])
df['Condition1']= label_encoder.fit_transform(df['Condition1'])
df['Condition2']= label_encoder.fit_transform(df['Condition2'])
df['HouseStyle']= label_encoder.fit_transform(df['HouseStyle'])
df['RoofMatl']= label_encoder.fit_transform(df['RoofMatl'])
df['Functional']= label_encoder.fit_transform(df['Functional'])

df1['Neighborhood']= label_encoder.fit_transform(df1['Neighborhood'])
df1['Exterior1st']= label_encoder.fit_transform(df1['Exterior1st'])
df1['Exterior2nd']= label_encoder.fit_transform(df1['Exterior2nd'])
df1['SaleType']= label_encoder.fit_transform(df1['SaleType'])
df1['Condition1']= label_encoder.fit_transform(df1['Condition1'])
df1['Condition2']= label_encoder.fit_transform(df1['Condition2'])
df1['HouseStyle']= label_encoder.fit_transform(df1['HouseStyle'])
df1['RoofMatl']= label_encoder.fit_transform(df1['RoofMatl'])
df1['Functional']= label_encoder.fit_transform(df1['Functional'])


# In[48]:


for i in df.select_dtypes(include='object').columns:
    print('____________________________________________________')
    print(i,df[i].nunique())
    


# In[49]:


from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# In[50]:


model = CatBoostRegressor()
#model = XGBRegressor()


# In[51]:


X = df.drop(columns=['SalePrice'])
y = df['SalePrice']


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[53]:


model.fit(X_train,y_train,verbose=False)
model.score(X_train,y_train)


# In[54]:


model.score(X_test,y_test)


# In[55]:


y_pred = model.predict(X_test)


# In[56]:


from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error(y_test, y_pred))


# In[57]:


model.fit(X,y,verbose=False)
model.score(X,y)


# In[58]:


y_pred = model.predict(df1)


# In[59]:


sub = pd.DataFrame({'id':range(1461,1461+len(df1)),'SalePrice':y_pred})
sub.to_csv('submission.csv',index=False)

