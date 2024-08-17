#!/usr/bin/env python
# coding: utf-8

# # Regression
# 
# ### Housing Price Prediction
# 
# The goal of this competition entry is apply concepts I've learnt so far, and gain more practise in feature engineering and stacked models.
# 
# As the missing features of this dataset are rather sparse, engineering them will mainly mean:
# 
# * Imputing missing values
# * Dealing with outliers
# * Transforming variables
# * Label Encoding
# * Stabilizing variance/skewness
# * Aquiring dummy variables
# 
# Model will be a sklearn base + XGBoost + LightGBM stack predicting the **SalePrice**.

# ### Credits & Inspiration
# * [Stacked Regression by Serigne](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook) - Absolutely stunning notebook that I've used as the template for my own explorations here. Taught me a lot about feature engineering and stacked modelling.
# * [Stacking Ensemble Machine Learning With Python by Jason Brownlee](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) - Great tutorial on stacking models
# * [Comprehensive data exploration with Python by Pedro Marcelino](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) - In-depth EDA on this dataset
# * [A study on Regression applied to the Ames dataset by juliencs](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset) - Great notebook using linear regression on this dataset
# * [Missing Values, Ordinal data and stories
#  by mitra mirshafiee](https://www.kaggle.com/mitramir5/missing-values-ordinal-data-and-stories) - Illustrative and educational notebook on handling missing values

# In[1]:


#Imports
import numpy as np
import pandas as pd

#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [12,6]
sns.set(style="darkgrid")

#Statistics
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

#Float Output to 2 decimals
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

#Models
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#Quality of life
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importing datasets
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


#First look training set
train.head()


# In[4]:


#First look test set
test.head()


# In[5]:


#Saving then dropping Id column as it does not impact our prediction
train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# ### Outliers

# In[6]:


#visualizing outliers, seen in bottom right.
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice');


# In[7]:


#the vast difference between 2 values of GrLivArea over 4000 and SalePrice under 200000 and the rest of datapoints makes me believe these outliers are safe to remove from the training set
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)


# In[8]:


#visualizing outliers, seen in bottom right.
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice');


# ### Univariate Analysis SalePrice

# In[9]:


sns.distplot(train['SalePrice'] , fit=norm);

#Fitted parameters used by function
(mu, sigma) = norm.fit(train['SalePrice'])

#Visualize Normal Distribution
plt.legend(['Normal dist. ($\mu=$ {:.0f} and $\sigma=$ {:.0f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Visualize Quantile-Quantile plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# The target variable is right skewed, which is a problem for our model as it needs to be normally distributed. This means we need to apply a Log transformation.

# ### Log transformation

# In[10]:


#Using numpy log1p fuction, which applies log(1+x) to elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train['SalePrice'] , fit=norm);

#Fitted parameters used by function
(mu, sigma) = norm.fit(train['SalePrice'])

#Visualize Normal Distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Visualize Quantile-Quantile plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# The SalePrice has been transformed to be normally distributed.

# ### Data Correlation

# In[11]:


#Correlation map to understand how other features relate to SalePrice
corrmat = train.corr()
mask = np.zeros_like(corrmat)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, mask=mask, ax=ax, cbar_kws={"shrink": .82},vmax=.9, cmap='coolwarm', square=True)


# We can immediately spot how the OverallQual and GrLivArea features have the high positive correlation with SalePrice.

# ### Feature Engineering
# Concatenating training and test set into same dataframe for preprocessing. Our feature engineering mainly revolves around harmonizing values so they more accurately reflect reality, whilst increasing predictions.
# 
# *version 6 note: **data leakage will occur** since we impute values based on test+train datasets, leading to a too optimistic score. In future version we will calculate compute missing values and apply Box-Cox seperately.*

# In[12]:


#Creating index for splitting datasets
i_train = train.shape[0]
i_test = test.shape[0]

#Setting y_train
y_train = train.SalePrice.values

#Concatenating datasets
df_concat = pd.concat((train, test)).reset_index(drop=True)
df_concat.drop(['SalePrice'], axis=1, inplace=True)

#Printing shape of datasets
print(train.shape)
print(test.shape)
print(df_concat.shape)


# ### Missing Data

# In[13]:


#Creating dataframe with percentages of missing values
df_concat_na = (df_concat.isnull().sum() / len(df_concat)) * 100

#Sorting by percentages of missing values
df_concat_na = df_concat_na.sort_values(ascending=False)[:20] #values of missing percentages after index 20 drop to around 0.03.. 

missing_values = pd.DataFrame({'Missing %' :df_concat_na})
missing_values.head()


# In[14]:


#Visualizing missing values by feauture
sns.barplot(x=df_concat_na.index, y=df_concat_na)
plt.xticks(rotation='90')
plt.xlabel('Features', fontsize=15)
plt.ylabel('% of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15);


# We clearly see on where we need to focus our work.
# 1. Coming up with a strategy to the missing values with over 80% NAs
# 2. Defining approach in imputing other missing values 

# ### Imputing Missing Values
# We will handle missing values by descending order of %NA. For each feature we'll first learn what we can from the description provided with the datasets, and secondly, provide our method for dealing with the missing values.

# **PoolQC:** Pool quality
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
#        
# As NA means 'No Pool' we can safely assume a missing value equates to None

# In[15]:


df_concat['PoolQC'].fillna('None', inplace=True)


# **MiscFeature**: Miscellaneous feature not covered in other categories
# 		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
# 
# Equally here, NA refers to lack of this feature. No misc. features means we can again apply a 'None' fill.

# In[16]:


df_concat['MiscFeature'].fillna('None', inplace=True)


# **Alley**: Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
#        
# NA means lack of this feature, so we fill missing values with None 

# In[17]:


df_concat['Alley'].fillna('None', inplace=True)


# Fence: Fence quality
# 		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
# 
# NA means lack of this feature, so we fill missing values with None 

# In[18]:


df_concat['Fence'].fillna('None', inplace=True)


# **FireplaceQu**: Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
#        
# NA means lack of this feature, so we fill missing values with None       

# In[19]:


df_concat['FireplaceQu'].fillna('None', inplace=True)


# **LotFrontage**: Linear feet of street connected to property
# 
# We assume here that neighborhoods contrain similar houses to fill in missing values by median LotFrontage per Neighborhood. Computing median has to be split per dataset, so as to not cause data leakage.

# In[20]:


df_concat["LotFrontage"] = df_concat.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[21]:


'''
v6 - Possible direction for future notebooks to compute missing values seperately

train_copy = train.copy(deep=True)
train_copy.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

LotF_median_train = train_copy['LotFrontage']

df_concat.iloc[:i_train]['LotFrontage'] = LotF_median_train
'''


# **GarageType**: Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# 		
# **GarageFinish**: Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage
# 
# **GarageQual**: Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# **GarageCond**: Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
#        
# NA means lack of this feature, so we fill missing values with None       

# In[22]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_concat[col].fillna('None', inplace=True)


# **GarageYrBlt**: Year garage was built
# 
# **GarageCars**: Size of garage in car capacity
# 
# **GarageArea**: Size of garage in square feet
# 
# Missing numerical values will be replaced by 0

# In[23]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_concat[col].fillna(0, inplace=True)


# **BsmtQual**: Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# **BsmtCond**: Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# **BsmtExposure**: Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# **BsmtFinType1**: Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 		
# **BsmtFinType2**: Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
#        
# NA means lack of this feature, so we fill missing values with None.       

# In[24]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):    
    df_concat[col].fillna('None', inplace=True)


# **BsmtFinSF1**: Type 1 finished square feet
# 
# **BsmtFinSF2**: Type 2 finished square feet
# 
# **BsmtUnfSF**: Unfinished square feet of basement area
# 
# **TotalBsmtSF**: Total square feet of basement area
# 
# **BsmtFullBath**: Basement full bathrooms
# 
# **BsmtHalfBath**: Basement half bathrooms
# 
# Missing numerical values will be replaced by 0

# In[25]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtHalfBath', 'BsmtFullBath'):
    df_concat[col].fillna(0, inplace=True)


# **MasVnrType**: Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
#        
# Missing values will be categorized as None       

# In[26]:


df_concat['MasVnrType'].fillna('None', inplace=True)


# **MasVnrArea** : Masonry veneer area in square feet
# As an architect I'm astonished this feature even exists. We assume missing values mean no masonry veneer, meaning 0.

# In[27]:


df_concat['MasVnrArea'].fillna(0, inplace=True)


# **MSZoning**: Identifies the general zoning classification of the sale.
# 		
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
#        
# As RL is by far the most common, we take the mode and fill in missing values with RL.       

# In[28]:


df_concat['MSZoning'].value_counts()


# In[29]:


df_concat['MSZoning'].fillna('RL', inplace=True)


# **Utilities**: Type of utilities available
# 		
#        AllPub	All public Utilities (E,G,W,& S)	
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only
#        
# The value of this feauture is AllPub for most of the dataset, with only 1 NoSeWa feauture. We can fately assume this feauture will not be helpful in predictive modelling, and will remove it.      

# In[30]:


df_concat['Utilities'].value_counts()


# In[31]:


df_concat.drop(['Utilities'], axis=1, inplace=True)


# **Functional**: Home functionality (Assume typical unless deductions are warranted)
# 
#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only
#        
# Any missing values will be transformed to Typ.     

# In[32]:


df_concat['Functional'].fillna('Typ', inplace=True)


# **Electrical**: Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
#        
# It's only NA value will be set to SBrkr.       

# In[33]:


df_concat['Electrical'].fillna('SBrkr', inplace=True)


# **KitchenQual**: Kitchen quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        
# We replace its only missing value by the mode of TA       

# In[34]:


df_concat['KitchenQual'].fillna('TA', inplace=True)


# **Exterior1st**: Exterior covering on house
# **Exterior2nd**: Exterior covering on house (if more than one material)
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast	
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 
# Since both have a single missing value we again replace the most common string as the missing value.

# In[35]:


df_concat['Exterior1st'].fillna('TA', inplace=True)
df_concat['Exterior2nd'].fillna('TA', inplace=True)


# **SaleType**: Type of sale
# 		
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
#        
# Missing values will be filled with most frequent type WD       

# In[36]:


df_concat['SaleType'].fillna('WD', inplace=True)


# **Missing Values Check**

# In[37]:


#Creating dataframe with percentages of missing values
df_concat_na = (df_concat.isnull().sum() / len(df_concat)) * 100

#Sorting by percentages of missing values
df_concat_na = df_concat_na.sort_values(ascending=False) 

missing_values = pd.DataFrame({'Missing %' :df_concat_na})
missing_values.head()


# ### Label Encoding
# Simple (simplified) encoding.
# 
# Future version: Transforming ordinal and nominal features

# In[38]:


'''
#Ordinal features
ordinal_to_encode=['LandSlope','YearBuilt','YearRemodAdd','CentralAir','GarageYrBlt','PavedDrive','YrSold']

#Encoding features for model with LabelEncoder
for field in ordinal_to_encode:
    le = LabelEncoder()
    df_concat[field] = le.fit_transform(df_concat[field].values)
'''


# In[39]:


#ordinal_ready = ['OverallQual','OverallCond','MoSold','FullBath','KitchenAbvGr','TotRmsAbvGrd']


# In[40]:


'''ordinal_mappings = {
    "MSSubClass": ['None', 20, 30,40,45,50,60,70,75,80,85, 90,120,150,160,180,190], 
    "ExterQual": ['None','Fa','TA','Gd','Ex'], 
    "LotShape": ['None','Reg','IR1' ,'IR2','IR3'], 
    "BsmtQual": ['None','Fa','TA','Gd','Ex'], 
    "BsmtCond": ['None','Po','Fa','TA','Gd','Ex'], 
    "BsmtExposure": ['None','No','Mn','Av','Gd'], 
    "BsmtFinType1": ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ], 
    "BsmtFinType2": ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ], 
    "HeatingQC": ['None','Po','Fa','TA','Gd','Ex'], 
    "Functional": ['None','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'], 
    "FireplaceQu": ['None','Po','Fa','TA','Gd','Ex'], 
    "KitchenQual": ['None','Fa','TA','Gd','Ex'], 
    "GarageFinish": ['None','Unf','RFn','Fin'], 
    "GarageQual": ['None','Po','Fa','TA','Gd','Ex'], 
    "GarageCond": ['None','Po','Fa','TA','Gd','Ex'], 
    "PoolQC": ['None','Fa','Gd','Ex'], 
    "Fence": ['None','MnWw','GdWo','MnPrv','GdPrv'],
}

# transform to a suitable format for ce.OrdinalEncoder
ce_ordinal_mappings = []
for col, unique_values in ordinal_mappings.items():
    local_mapping = {val:idx for idx, val in enumerate(unique_values)}
    ce_ordinal_mappings.append({"col":col, "mapping":local_mapping})
    
encoder = ce.OrdinalEncoder(mapping=ce_ordinal_mappings, return_df=True)
encoder.fit_transform(train_X)
encoder.transform(test)
'''


# In[41]:


#MSSubClass
df_concat['MSSubClass'].astype(str)


#Changing OverallCond into a categorical variable
df_concat['OverallCond'] = df_concat['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
df_concat['YrSold'] = df_concat['YrSold'].astype(str)
df_concat['MoSold'] = df_concat['MoSold'].astype(str)


# In[42]:


#List of features to encode
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

#Applying LabelEncoder to categorical features
for i in cols:
    le = LabelEncoder() 
    le.fit(list(df_concat[i].values)) 
    df_concat[i] = le.transform(list(df_concat[i].values))


# **Adding a Total sqf feature**
# 
# Since we've seen the correlation between square footage on SalePrice, adding a feature combing all seperate parameters together seems logical as a great predictor. 

# In[43]:


df_concat['TotalSF'] = df_concat['TotalBsmtSF'] + df_concat['1stFlrSF'] + df_concat['2ndFlrSF']


# ### Skewness and Box Cox Transformation
# Calculating degree of skewness and computing Box Cox transformation of 1 + x
# 
# *Note: applying Box Cox transformation to both datasets at once will lead to data leakage and will be addressed in future versions.*

# In[44]:


numeric_feats = df_concat.dtypes[df_concat.dtypes != "object"].index

#Check the skew of all numerical features
skewed_feats = df_concat[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[45]:


skewness = skewness[abs(skewness.Skew)>0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #df_concat[feat] += 1
    df_concat[feat] = boxcox1p(df_concat[feat], lam)


# **Dummy Variables**

# In[46]:


df_concat = pd.get_dummies(df_concat)


# **Creating new training and test sets**

# In[47]:


train = df_concat[:i_train]
test = df_concat[i_train:]


# ### Modelling

# In[48]:


#Creating cross validation strategy with shuffle
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# **LASSO Regression**

# In[49]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=0))


# **Elastic Net Regression**

# In[50]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0))


# **Kernel Ridge Regression**

# In[51]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# **Gradient Boosting**

# In[52]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =0)



# **XGBoost**

# In[53]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =0, nthread = -1,
                             verbosity = 0)


# **LightGBM**

# In[54]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                              verbose_eval = -1)


# ### Model Scores

# In[55]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[56]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[57]:


score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[58]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[59]:


score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[60]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# ### Stacking Models
# 
# 
# **Average Base Models**
# 
# Building a new scaleable class to extend scikit-learn with our model

# In[61]:


class AverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    #Define clones of original models to fit the data
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        #Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Predictions for cloned models and get average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 


# **Average Base Models Score**

# In[62]:


averaged_models = AverageModels(models = (lasso, ENet, GBoost, KRR))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Even the simplest stacking approach offers great improvement to our score, and encouragement to keep exploring further options.

# **Meta-model**
# 
# In this model of models we use our base models to train a new meta-model. That's a lot of models.
# 
# 1. Create train and holdout sets
# 2. Train base models on train set
# 3. Test base models on holdout
# 4. Use predictions as output to train meta-model

# **Stacking Averaged Base Models**

# In[63]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    #Fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        #Train the cloned meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Predictions of all base models on test data and use the averaged predictions as 
    #meta-features for the final prediction
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# **Stacking Averaged Models Score**

# In[64]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# **Ensembling StackedRegressor, XGBoost and LightGBM** 

# In[65]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# **Stacked Regressor**

# In[66]:


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# **XGBoost**

# In[67]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# **LightGBM**

# In[68]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


# **RMSLE score**

# In[69]:


print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# **Ensemble Prediction**

# In[70]:


ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15


# In[71]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)


# ### Credits & Inspiration
# * [Stacked Regression by Serigne](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook) - Absolutely stunning notebook that I've used as the template for my own explorations here. Taught me a lot about feature engineering and stacked modelling.
# * [Stacking Ensemble Machine Learning With Python by Jason Brownlee](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) - Great tutorial on stacking models
# * [Comprehensive data exploration with Python by Pedro Marcelino](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) - In-depth EDA on this dataset
# * [A study on Regression applied to the Ames dataset by juliencs](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset) - Great notebook using linear regression on this dataset
# * [Missing Values, Ordinal data and stories
#  by mitra mirshafiee](https://www.kaggle.com/mitramir5/missing-values-ordinal-data-and-stories) - Illustrative and educational notebook on handling missing values
