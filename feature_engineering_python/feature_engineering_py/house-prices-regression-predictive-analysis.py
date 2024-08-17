#!/usr/bin/env python
# coding: utf-8

# # STEPS FOR REGRESSION PREDICTIVE ANALYSIS
# 
# **********************************
# 
# * **LOADING DATA**
# 
#     - Loading both train and test dataset and applying preprocessing steps together
#     
# ********************
# 
# * **SUMMARY OF DATA**
# 
#     - Total Samples
#     - Total Features
#     - Total Categorical Features
#     - Total Numerical Features
#     - Stats of Numerical Features
#     - Value Count of Categorical Features
#     - Unique Values DataFrame
#     - Null Values DataFrame
#     
# ***********************
# 
# *  **PREPROCESSING**
# 
#     * Drop Duplicates
#     * Drop Columns with more than 80% null values
#     * Drop uninformative columns
#     * Drop Columns with single unique values
#     * Inpute Null Values
#     * Create New Features
#     * Outlier Analysis and Removal
#     * Drop Columns with single unique values again after outlier analysis
#     
# *************************
# 
# * **VISUALIZE**
# 
#     - Scatterplot of numerical features
#     - Distribution of numerical features
#     - BarCharts of categorical features
#     - Box plots to check the outliers
#     
# ***********************
# 
# * **FEATURE TRANSFORMATION**
# 
#     - Changing the distribution of numerical features to Gaussian (Normal)
#     
# ****************
# 
# *  **ENCODING**
# 
#     - Some of the categorical features are nominal and some are ordinal. We need to encode them separately.
#     - For **ordinal** features, we will do **label encoding**
#     - For **nominal** features, we will do **dummy encoding**
#     
# *************
# 
# * **MODEL TRAINING & EVALUATION**
# 
#     - Perform Scaling
#     
#         - MinMax Scaling
#         - Variance Scaling (Standard Scaler)
#         
#     - Fitting Different Regression Models
#     
#         - Linear Regression
#         - Polynomial Regression (with interaction features)
#         - Ridge Regression
#         - Lasso Regression
#         - SGD Regression
#         - Elastic Regression
#         - Bayesian Ridge
#         - Huber Regression (robust to outliers)
#         - RANSAC Regression (robust to outliers)
#         - XGB Regressor
#         - Ensemble Regressor 
#             - Random Forest
#             - Gradient Boosting
#             - AdaBoosting
#             - Bagging Regressor
#             - ExtraTreesRegressor
#          
#        
# *******************
# 
# * **FEATURE SELECTION**
# 
#     * Selecting strong numerical features using Pearson’s Correlation Coefficient
#     * Selecting strong categorical using ANOVA 
# 
# *****************************
# 
# * **FEATURE EXTRACTION**
# 
#     * Using PCA to perform dimensionality reduction.
#     * Don't forget to scale your data before doing PCA.
#     
# ******************
# 
# * **MODEL TRAINING & EVALUATION WITH STRONG FEATURES ONLY**
# 
#     - Using the same models as stated above.
#     
# *******************
# 
# * **CONCLUSION**
# 
#     - Which model performed the best one with using all the features or the one with the strong features only ?
#     
# *************
# 
# * **HYPERPARAMETER TUNING**
# 
#     - Tuning the parameters of the best model.
#     
# ******************************
# 
# * **PREDICTION**
# 
#     - Prediction on the test dataset using the top scorer model
#     - Saving the results in submission.csv
#     
# ************************
# 
# * **FEATURE ENGINEERING ANALYSIS**
# 
#     - Comparison of the scores of the different feature engineering steps.
# **************************
# 
# * **RESULT ANALYSIS**
# 
#     - Analysis of the results given by the model.
#     
# ****************
# 
# * **STORY TELLING FROM THE RESULT ANALYSIS**
# 
#     - Simple interpretation of the results in layman language.
#     
# *****************
# ****************

# # LOADING DATA 

# In[1]:


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[2]:


train_df


# # SUMMARY OF DATA
# *******************
# - Total Samples
# - Total Features
# - Total Categorical Features
# - Total Numerical Features
# - Stats of Numerical Features
# - Value Count of Categorical Features
# - Unique Values DataFrame
# - Null Values DataFrame

# In[3]:


def get_cat_num_features(df):
    
    num_features = []
    cat_features = []
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            num_features.append(col)
                
        if is_object_dtype(df[col]):
            cat_features.append(col)
            
    return num_features, cat_features

def get_unique_df(features):
    unique_df = pd.DataFrame(columns=['Feature', 'Unique', 'Count'])
    for col in features.columns:
        v = features[col].unique()
        l = len(v)
        unique_df = unique_df.append({'Feature':col, 
                                     'Unique':v,
                                     'Count':l}, ignore_index=True)
    return unique_df

def get_null_df(features):
    col_null_df = pd.DataFrame(columns = ['Column', 'Type', 'Total NaN', '%'])
    col_null = features.columns[features.isna().any()].to_list()
    L = len(features)
    for col in col_null:
        T = 0
        if is_numeric_dtype(features[col]):
            T = "Numerical"  
        else:
            T = "Categorical"
        nulls = len(features[features[col].isna() == True][col])   
        col_null_df = col_null_df.append({'Column': col, 
                                          'Type': T,
                                          'Total NaN': nulls,
                                          '%': (nulls / L)*100
                                         }, ignore_index=True)
        
    return col_null_df

def summary(data):
    
    print("Samples --> ", len(data))
    print()
    target = data['SalePrice']
    features = data.drop(['SalePrice'], axis=1)
    print("Features --> ", len(features.columns))
    print("\n",features.columns)
    
    num_features, cat_features = get_cat_num_features(features)
      
    print()
    print("\nNumerical Features --> ", len(num_features))
    print()
    print(num_features)
    print()
    print("Categorical Features -->", len(cat_features))
    print()
    print(cat_features)
    print()
    print("*************************************************")
    stats = features.describe().T
    
    print()
    print("Value counts of each categorical feature\n")
    for col in cat_features:
        print(col)
        print(features[col].value_counts())
        print()
        
    unique_df = get_unique_df(features)
    
    col_null_df = get_null_df(features)
    
    return {'features':features, 
            'target': target, 
            'stats': stats, 
            'unique_df':unique_df,
            'col_null_df': col_null_df}


# In[4]:


df_summary = summary(train_df)


# In[5]:


# Features with only 1 unique value
df_summary['unique_df'][df_summary['unique_df']['Count'] == 1]


# In[6]:


# Features with null values
df_summary['col_null_df']


# In[7]:


# stats of the numerical feature
df_summary['stats']


# In[8]:


target = df_summary['target']
target


# # PREPROCESSING
# *************************
# 
# * Drop Duplicates
# * Drop Columns with more than 80% null values
# * Drop uninformative columns
# * Drop Columns with single unique values
# * Inpute Null Values
# * Create New Features
# * Outlier Analysis and Removal
# * Drop Columns with single unique values again after outlier analysis

# ### Drop Duplicates

# In[9]:


df = df_summary['features']

cleaned_df = df.drop_duplicates(subset=['Id'])
new_test_df = test_df.drop_duplicates(subset=['Id'])

print("Total Duplicates were ", len(df) - len(cleaned_df))


# ### Drop Columns with more than 80% null values
# *******************
# 
# * In this notebook version 2, I am not dropping the values with 80% null values.
# 
# * I dropped them in the version 1. There was not much change in the scores obtained by the model. 
# 
# * Thats why I have commented out the below code snippets

# In[10]:


# col_null_df = df_summary['col_null_df']
# col_null_df[col_null_df['%']>=80]


# In[11]:


# null_cols = col_null_df[col_null_df['%']>=80]['Column'].to_list()

# cleaned_df.drop(null_cols, axis=1, inplace=True)
# new_test_df.drop(null_cols, axis=1, inplace=True)

# print(cleaned_df.shape)
# print(new_test_df.shape)


# ### Drop uninformative columns

# In[12]:


cleaned_df.drop(['Id'], axis=1, inplace=True)
new_test_df.drop(['Id'], axis=1, inplace=True)

print(cleaned_df.shape)
print(new_test_df.shape)


# ### Drop columns with single unique value

# In[13]:


df_summary['unique_df'][df_summary['unique_df']['Count']==1]


# ### Impute NaN Values

# In[14]:


# col_null_df[col_null_df['%'] < 80]
col_null_df = df_summary["col_null_df"]


# In[15]:


# null values in test_df

test_null_df = get_null_df(new_test_df)
test_null_df


# In[16]:


def imputation(null_df, df):
    
    for ind, row in null_df.iterrows():
        col = row['Column']
        if row['Type'] == 'Categorical':
            df[col].fillna('NotAvail', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

# null_df = col_null_df[col_null_df['%'] < 80]
null_df = col_null_df

cleaned_df = imputation(null_df, cleaned_df)
new_test_df = imputation(test_null_df, new_test_df)

print(cleaned_df.shape)
print(new_test_df.shape)


# In[17]:


cleaned_df.columns[cleaned_df.isna().any()]


# In[18]:


new_test_df.columns[new_test_df.isna().any()]


# ### Create New Features
# ********************
# 
# * There are columns with the year. Years are not informative, so we need to transform it.
# 
# * We will perform **binning** on those year column to divide it into decades and will make it categorical. 
# 
# * We will also calculate the age of house on the basis of different columns.

# In[19]:


print("Min & Max of YearBuilt ", cleaned_df['YearBuilt'].min(), cleaned_df['YearBuilt'].max())
print("\nMin & Max of GarageYrBlt ", cleaned_df['GarageYrBlt'].min(), cleaned_df['GarageYrBlt'].max())
print("\nMin & Max of YrSold ", cleaned_df['YrSold'].min(), cleaned_df['YrSold'].max())
print("\nMin & Max of YearRemodAdd ", cleaned_df['YearRemodAdd'].min(), cleaned_df['YearRemodAdd'].max())


# In[20]:


cleaned_df[cleaned_df['YearBuilt']<1900]['YearBuilt'].count()


# In[21]:


cleaned_df[cleaned_df['YearBuilt']==1900]['YearBuilt'].count()


# * We will make categories 
#     - 1800-1900
#     - 1900-1910
#     - 1910-1920
#     
#     ...
#     - 2000-2010
#     
# * Column 'YrSold' has all the values between 2006 to 2010 so all of its will fall into a single category 2000-2010. 
# 
# * Hence we will delete YrSold in further steps coz it won't be giving us any information (because of no variance)

# In[22]:


bins =  [1800] + [i for i in range(1900, 2020, 10)]
bins


# In[23]:


def create_new_features(cleaned_df):
    cleaned_df['Remod_Built_Age'] = cleaned_df['YearRemodAdd'] - cleaned_df['YearBuilt']
    cleaned_df['Sold_Built_Age'] = cleaned_df['YrSold'] - cleaned_df['YearBuilt']
    cleaned_df['Remod_Sold_Age'] = cleaned_df['YrSold'] - cleaned_df['YearRemodAdd']

    cleaned_df['MSSubClass'].replace({20:"1-STORY 1946 & NEWER",
                                   30:"1-STORY 1945 & OLDER",
                                   40:"1-STORY W/FINISHED",
                                   45:"1-1/2 STORY - UNFINISHED",
                                   50:"1-1/2 STORY FINISHED",
                                   60:"2-STORY 1946 & NEWER",
                                   70:"2-STORY 1945 & OLDER",
                                   75:"2-1/2 STORY ALL AGES",
                                   80:"SPLIT OR MULTI-LEVEL",
                                   85:"SPLIT FOYER",
                                   90:"DUPLEX",
                                   120:"1-STORY PUD",
                                   150:"1-1/2 STORY PUD",
                                   160:"2-STORY PUD",
                                   180:"PUD - MULTILEVEL",
                                   190:"2 FAMILY CONVERSION"                         
                                  },inplace=True)

    cleaned_df['MoSold'].replace({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                            ,inplace=True)

    return cleaned_df

# Function for binning
def binning(df, col, bins, test=None):
    df[col] = df[col].astype(int)
    binned_values = list()
    for ind, i in enumerate(df[col]):
        for j, k in enumerate(bins):
            if i <= k and i!= 1800:
                binned_values.append('{}-{}'.format(bins[j-1], k))
                if col == 'GarageYrBlt' and ind==1458 and test!= None:
                    binned_values.append('{}-{}'.format(bins[j-1], k))
                    
                    
                break
            if i <= k and i == 1800:
                binned_values.append('{}'.format(k))
                break
    df[col] = binned_values
    return df


# In[24]:


create_new_features(cleaned_df)
create_new_features(new_test_df)

yr_cols = ['YearBuilt', 'GarageYrBlt', 'YrSold', 'YearRemodAdd']

for col in yr_cols:
    binning(cleaned_df, col, bins)
    binning(new_test_df, col, bins, 1)

print(cleaned_df.shape)
print(new_test_df.shape)


# ### Outlier Analysis and Removal
# 
# *******************************
# 
# * Using IQR 

# In[25]:


def calc_interquartile(df, column):
    
    #calculating the first and third quartile
    first_quartile, third_quartile = np.percentile(df[column], 25), np.percentile(df[column], 75)
    
    #calculate the interquartilerange
    iqr = third_quartile - first_quartile
    
    # outlier cutoff (1.5 is a generally taken as a threshold thats why i am also taking it)
    cutoff = iqr*1.5
    
    #calculate the lower and upper limits
    lower, upper = first_quartile - cutoff , third_quartile + cutoff
    
    #remove the outliers from the columns
    upper_outliers = df[df[column] > upper]
    lower_outliers = df[df[column] < lower]
    
    return lower, upper, lower_outliers.shape[0]+upper_outliers.shape[0]


def get_outliers(df, num_feat):
    
    outlier_df = pd.DataFrame(columns=['Feature', 'Total Outliers','Upper limit', 'Lower limit'])
    
    for col in num_feat:
        lower, upper, total = calc_interquartile(df, col)
        if total != 0 and (upper !=0 and lower!=0):
            outlier_df = outlier_df.append({'Feature':col, 'Total Outliers': total,
                                       'Upper limit': upper, 'Lower limit':lower}, ignore_index=True)
        
    return outlier_df

num_feat, _ = get_cat_num_features(cleaned_df)

outlier_df = get_outliers(cleaned_df, num_feat)
outlier_df


# In[26]:


def remove_outliers(df, outlier_df, num_feat):
    
    for col in outlier_df['Feature'].to_list():
        upper = outlier_df[outlier_df['Feature']== col ]['Upper limit'].values[0]
        lower = outlier_df[outlier_df['Feature']== col ]['Lower limit'].values[0]
        
        df[col] = np.where(df[col]>upper, upper, df[col])
        df[col] = np.where(df[col]<lower, lower, df[col])
        
    return df

cleaned_df = remove_outliers(cleaned_df, outlier_df, num_feat)


# In[27]:


get_outliers(cleaned_df, num_feat)


# In[28]:


print(cleaned_df.shape)
print(new_test_df.shape)


# ### Drop Columns with single unique values again after outlier analysis

# In[29]:


unique_df = get_unique_df(cleaned_df)
unique_df


# In[30]:


# columns with single unique values
unique_df[unique_df['Count']==1]


# In[31]:


cols = unique_df[unique_df['Count']==1]['Feature'].to_list()

cleaned_df.drop(cols, axis=1, inplace=True)
new_test_df.drop(cols, axis=1, inplace=True)

get_unique_df(cleaned_df)[get_unique_df(cleaned_df)['Count']==1]


# In[32]:


print(cleaned_df.shape)
print(new_test_df.shape)


# # FEATURE TRANSFORMATION
# 
# ***************************
# 
# * Changing the distribution of numerical features to Gaussian (Normal)
# * We will apply power transform (Yeo-Johnson) on the features. 

# In[33]:


# BOX COX TRANSFORMATION

# from scipy import stats
# trans_df = cleaned_df.copy()
# num_feat, _ = get_cat_num_features(cleaned_df)

# def transformed_feat(trans_df, num_feat):
    
#     for col in num_feat:
#         try:
#             trans_df[col], _ = stats.boxcox(cleaned_df[col])
#         except:
#             # if there are observations which 0 or negative, shit their values to 0.001 to make them above 0
#             trans_df[col] = np.where(trans_df[col]<=0, 0.001, trans_df[col])
#             trans_df[col], _ = stats.boxcox(trans_df[col])
        
#     return trans_df

# trans_df = transformed_feat(trans_df, num_feat)
# new_test_df = transformed_feat(new_test_df, num_feat)


# In[34]:


from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')


# In[35]:


trans_df = cleaned_df.copy()
num_feat, _ = get_cat_num_features(cleaned_df)

def transformed_feat(trans_df, new_test_df, num_feat):
    
    for col in num_feat:
        t = scaler.fit_transform(np.array(cleaned_df[col]).reshape(-1,1))
        trans_df[col] = t.reshape(-1)
        t = scaler.transform(np.array(new_test_df[col]).reshape(-1, 1))
        new_test_df[col] = t.reshape(-1)
        
    return trans_df, new_test_df

trans_df, new_test_df = transformed_feat(trans_df, new_test_df, num_feat)


# In[36]:


print(trans_df.shape)
print(new_test_df.shape)


# # VISUALIZE
# 
# *************************
# 
# - Scatterplot and distribution of numerical features
# - BarCharts of categorical features
# - Box plots to check the outliers

# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# In[38]:


num_feat, cat_feat = get_cat_num_features(trans_df)


# In[39]:


#  Scatterplot for numerical features

plt.figure(figsize=(20,90))
for i in range(len(num_feat)):
    plt.subplot(12, 3, i+1)
    sns.scatterplot(x=trans_df[num_feat[i]], y=target)

plt.show()


# In[40]:


#  Bar Plot for Categorical Features
for col in cat_feat:
    plt.figure(figsize=(30, 10))
    sns.barplot(x=trans_df[col], y=target)
    plt.show()


# In[41]:


# Box Plot of Numerical Features

plt.figure(figsize=(20,90))
for i in range(len(num_feat)):
    plt.subplot(12, 3, i+1)
    sns.boxplot(y=trans_df[num_feat[i]])

plt.show()


# In[42]:


# Distribution Plots
for i in num_feat:
    sns.displot(x=trans_df[i], kde=True)
    plt.show()


# # ENCODING
# ****************
# 
# * Some of the categorical features are nominal and some are ordinal. We need to encode them separately.
# 
# * For ordinal data, use label encoding and for nominal data, use dummy encoding.
# 
# * In dummy encoding, we create separate columns for each category in a feature. Hence we need to make sure that two different features does not contain the same categories. Otherwise, there will be multiple columns with the same name.
# 
# * Features like `Condition1 and Condition2` , `Exterior1st and Exterior2nd` are nominal features and contains the same categories.
# 
# * In that case, while performing dummy encoding we will change the column name by putting a prefix of the original column.

# In[43]:


from sklearn.preprocessing import LabelEncoder


# In[44]:


yr_cols.remove('YrSold')
yr_cols


# In[45]:


ordinal_feat = ['LotShape','LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
               'GarageQual', 'GarageCond' , 'Utilities', "PoolQC"]

nominal_feat = ['MSSubClass', 'MSZoning','Street', 'LotConfig', 'Neighborhood',
               'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
               'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
               'GarageType', 'GarageFinish', 'PavedDrive','SaleType', 'SaleCondition'
               , 'LandContour', 'MoSold', "Alley", "Fence", "MiscFeature"]


# ### Dictionary of all the ordinal categorical features

# In[46]:


ord_dict = {"LotShape": ['Reg','IR1','IR2','IR3', 'NotAvail'],
            "LandSlope" : ["Gtl", "Mod", "Sev",'NotAvail' ],
            "ExterQual": [  "Ex", "Gd", "TA", "Fa", "Po", 'NotAvail' ],
            "ExterCond": [  "Ex", "Gd", "TA", "Fa", "Po", 'NotAvail' ],
            "BsmtQual": [  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "BsmtCond":[  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "BsmtExposure": ["Gd", "Av", "Mn", "No", "NA", 'NotAvail'],
            "BsmtFinType1":[ "GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA", 'NotAvail'],
            "BsmtFinType2":[ "GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA", 'NotAvail'],
            "HeatingQC": [  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "KitchenQual": [  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "FireplaceQu":[  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "GarageQual":[  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "GarageCond": [  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ],
            "Utilities":  [ "AllPub", "NoSewr", "NoSeWa","ELO", "NotAvail"],
            "PoolQC":[  "Ex", "Gd", "TA", "Fa", "Po", "NA", 'NotAvail' ]
           
           }


# In[47]:


enc_df = trans_df.copy()
test_enc_df = new_test_df.copy()

def encode_feat(nom_feat, ord_feat, yr_cols, df, t_df=pd.DataFrame()):
    
    # Label encoding ordinal features
    le = LabelEncoder()
    
    for col in ord_feat:
        le.fit(ord_dict[col])
        df[col] = le.transform(df[col])
        if len(t_df) != 0:
            t_df[col] = le.transform(t_df[col])
            
    for col in yr_cols:
        df[col] = le.fit_transform(df[col])
        if len(t_df) != 0:
            t_df[col] = le.transform(t_df[col])
    
    # dummy encoding nominal features
    for col in nom_feat:
        dum = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dum], axis=1)
        df.drop([col], axis=1, inplace=True)
        
        if len(t_df) != 0:
            t_df = pd.concat([t_df, dum], axis=1)
            t_df.drop([col], axis=1, inplace=True)
     
    if len(t_df) != 0:
        return df, t_df
    else:
        return df


enc_df, test_enc_df = encode_feat(nominal_feat, ordinal_feat, yr_cols, enc_df, test_enc_df) 


# In[48]:


print(enc_df.shape)
print(test_enc_df.shape)


# In[49]:


test_enc_df = test_enc_df.iloc[:-1, :]
test_enc_df


# In[50]:


print(enc_df.shape)
print(test_enc_df.shape)


# # MODEL TRAINING & EVALUATION

# In[51]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, BayesianRidge,RANSACRegressor,HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline


# In[52]:


sc = ('Scaler', StandardScaler())
est =[]
est.append(('LinearRegression', Pipeline([sc, ('LinearRegression', LinearRegression())])))
est.append(('Ridge', Pipeline([sc, ('Ridge', Ridge())])))
est.append(('Lasso', Pipeline([sc, ('Lasso', Lasso())])))
est.append(('BayesianRidge', Pipeline([sc, ('BayesianRidge', BayesianRidge())])))
est.append(('ElasticNet', Pipeline([sc,('Elastic', ElasticNet())])))
est.append(('SGD', Pipeline([sc,('SGD', SGDRegressor())])))
est.append(('Huber', Pipeline([sc,('Huber', HuberRegressor())])))
est.append(('RANSAC', Pipeline([sc,('RANSAC', RANSACRegressor())])))
est.append(('GradientBoosting', Pipeline([sc,('GradientBoosting',GradientBoostingRegressor())])))
est.append(('AdaBoost', Pipeline([sc, ('AdaBoost', AdaBoostRegressor())])))
est.append(('ExtraTree', Pipeline([sc,('ExtraTrees', ExtraTreesRegressor())])))
est.append(('RandomForest', Pipeline([sc,('RandomForest', RandomForestRegressor())]))) 
est.append(('Bagging', Pipeline([sc,('Bagging', BaggingRegressor())])))
est.append(('KNeighbors', Pipeline([sc,('KNeighbors', KNeighborsRegressor())])))
est.append(('DecisionTree', Pipeline([sc,('DecisionTree', DecisionTreeRegressor())])))
est.append(('XGB', Pipeline([sc,('XGB', XGBRegressor())])))


# In[53]:


import warnings
warnings.filterwarnings(action='ignore')
seed = 4
splits = 7
models_score ={}
for i in est:
    kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
    results = cross_val_score(i[1], enc_df, target, cv=kfold)
    models_score.update({i[0] : results.mean()})
    
sorted(models_score.items(), key= lambda v:v[1], reverse=True)


# In[54]:


base_model_scores = sorted(models_score.items(), key= lambda v:v[1], reverse=True)


# # FEATURE SELECTION
# *****************************
# 
# * Selecting strong numerical features using Pearson’s Correlation Coefficient
# * Selecting strong categorical using ANOVA 

# In[55]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,  f_classif


# In[56]:


num_feat, cat_feat = get_cat_num_features(trans_df)
num_df = trans_df[num_feat]
cat_df = trans_df[cat_feat]

print("Total Numerical Features = ", len(num_feat))
print("Total Categorical Features = ", len(cat_feat))


# ### Selecting best 20 numerical features

# In[57]:


# define feature selection
num_fs = SelectKBest(score_func=f_regression, k=20)
# apply feature selection
num_fs.fit(num_df, target)
# get the column indices
cols  = num_fs.get_support(indices=True)
best_num_df = num_df.iloc[:,cols]

best_num_df


# ### Select best 30 categorical features

# ### Encoding categorical features

# In[58]:


import warnings
warnings.filterwarnings(action='ignore')

def lab_encode_feat(df):
    
    # Label encoding ordinal features
    le = LabelEncoder()
    
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
        
    return df

lab_encode_feat(cat_df) 
cat_df


# In[59]:


cat_fs = SelectKBest(score_func=f_classif, k=30)
cat_fs.fit(cat_df, target)
cols = cat_fs.get_support(indices=True)
best_cat_df = cat_df.iloc[:, cols]

best_cat_df


# In[60]:


best_cols = best_num_df.columns.to_list() + best_cat_df.columns.to_list()
best_cols


# In[61]:


best_feat_df = trans_df[best_cols]
best_feat_df


# # MODEL TRAINING WITH STRONG FEATURES

# In[62]:


best_cat_df.columns.to_list()


# In[63]:


# strong nominal and ordinal columns
nom_cols = []
ord_cols = []
for col in best_cat_df.columns.to_list():
    if col in nominal_feat:
        nom_cols.append(col)
    else:
        ord_cols.append(col)


# In[64]:


nom_cols


# In[65]:


ord_cols


# In[66]:


ord_cols.remove('YearBuilt')
ord_cols.remove('YearRemodAdd')
ord_cols.remove('GarageYrBlt')


# In[67]:


best_feat_df = encode_feat(nom_cols, ord_cols, yr_cols, best_feat_df)
best_feat_df


# In[68]:


import warnings
warnings.filterwarnings(action='ignore')
seed = 4
splits = 7
models_score ={}
for i in est:
    kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
    results = cross_val_score(i[1], best_feat_df, target, cv=kfold)
    models_score.update({i[0] : results.mean()})
    
sorted(models_score.items(), key= lambda v:v[1], reverse=True)


# # FEATURE EXTRACTION
# ***********
# 
# * Using PCA to perform dimensionality reduction.
# * Don't forget to scale your data before doing PCA.

# In[69]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaled_df = StandardScaler().fit_transform(enc_df)
pca = PCA(n_components=0.99, svd_solver='full')
pca_enc_df = pca.fit_transform(scaled_df)

pca_enc_df.shape


# In[70]:


est =[]
est.append(('LinearRegression', Pipeline([('LinearRegression', LinearRegression())])))
est.append(('Ridge', Pipeline([('Ridge', Ridge())])))
est.append(('Lasso', Pipeline([ ('Lasso', Lasso())])))
est.append(('BayesianRidge', Pipeline([('BayesianRidge', BayesianRidge())])))
est.append(('ElasticNet', Pipeline([('Elastic', ElasticNet())])))
est.append(('SGD', Pipeline([('SGD', SGDRegressor())])))
est.append(('Huber', Pipeline([('Huber', HuberRegressor())])))
est.append(('RANSAC', Pipeline([('RANSAC', RANSACRegressor())])))
est.append(('GradientBoosting', Pipeline([('GradientBoosting',GradientBoostingRegressor())])))
est.append(('AdaBoost', Pipeline([('AdaBoost', AdaBoostRegressor())])))
est.append(('ExtraTree', Pipeline([('ExtraTrees', ExtraTreesRegressor())])))
est.append(('RandomForest', Pipeline([('RandomForest', RandomForestRegressor())]))) 
est.append(('Bagging', Pipeline([('Bagging', BaggingRegressor())])))
est.append(('KNeighbors', Pipeline([('KNeighbors', KNeighborsRegressor())])))
est.append(('DecisionTree', Pipeline([('DecisionTree', DecisionTreeRegressor())])))
est.append(('XGB', Pipeline([('XGB', XGBRegressor())])))


# In[71]:


import warnings
warnings.filterwarnings(action='ignore')
seed = 4
splits = 7
models_score ={}
for i in est:
    kfold = KFold(n_splits=splits, random_state=seed, shuffle=True)
    results = cross_val_score(i[1], pca_enc_df, target, cv=kfold)
    models_score.update({i[0] : results.mean()})
    
sorted(models_score.items(), key= lambda v:v[1], reverse=True)


# # CONCLUSION
# *****************
# 
# * We performed both **Feature Selection** and **Feature Extraction** and we can see, when we used all the features we got better results as compared to using only strong features.
# 
# * Feature Selection was better than the Feature Extraction (PCA). Hence feature selection through f-test ANOVA and pearson correlation test is best here to select strong features than the dimensionality reduction method using PCA.
# 
# * Although the difference between the results of both (feature selection and with all the features) isn't much, so we can use only the strong features also. Here I am taking all the features.
# 
# * Hence we will use all the features and the top model with the best score to predict house prices   

# In[72]:


# Top model with scores
base_model_scores[0]


# # HYPERPARAMETER TUNING
# ******************
# 
# * Our best model is gradient boosting classifier.
# 
# * We will tune its parameter that is `learning_rate` , `max_depth` and `n_estimators`.
# 
# * Well the default values of the parameters on which we got the results are following:-
# 
#     * learning_rate=0.1       
#     * n_estimators=100        
#     * max_depth=3.
#     * min_samples_split=2.
#     * min_samples_leaf=1.
#     * subsample=1.0
#     
#     
# * Well tuning the min_samples_split, min_samples_leaf and subsample results to nothing. Hence we won't tune these parameters

# In[73]:


from sklearn.model_selection import GridSearchCV


# ***************
# ### UPDATED
# ***************
# * In the previous version (6th) , I have used the best parameters to train the model, but it seemed to me that the tree kind of overfitted and didn't generalise well on the test set because my score which was `0.23` before became `0.26` on the scoreboard.
# 
# * In the 11th version, I sticked to the default parameters of the Gradient Boosting but applied yeo-johnson feature transformation and my score improved from `0.23` to `0.16`.
# 
# * Hence I am commenting out the following code.

# In[74]:


# %%time

# sc = ('Scaler', StandardScaler())
# h_est = []
# h_est.append(('GBR', Pipeline([sc,('GBR',GradientBoostingRegressor())])))

# best = []
# seed = 4

# parameters = {
              
#               'GBR': {'GBR__learning_rate': [0.1, 0.01],
#                          'GBR__max_depth': [4,6,8],
#                       'GBR__n_estimators': [400, 500, 600]}
#              }


# for i in h_est:
#     kfold = KFold(n_splits=3, random_state=seed, shuffle=True)
#     grid = GridSearchCV(estimator=i[1], param_grid = parameters[i[0]], cv = kfold, n_jobs=-1)
#     grid.fit(enc_df, target)
#     best.append((i[0], grid.best_score_,  grid.best_params_))


# In[75]:


# best


# In[76]:


# param_dict = best[0][2]
# param = {}
# for i in param_dict:
#     key = i.split('__')[1]
#     param.update({key:param_dict[i]})
    
# param


# ### Training the model with best parameters

# In[77]:


# std_scaler = StandardScaler()
# std_scaler.fit(enc_df)
# scaled_df = std_scaler.transform(enc_df)

# model = GradientBoostingRegressor(learning_rate= param['learning_rate'], 
#                                   max_depth= param['max_depth'], 
#                                   n_estimators=param['n_estimators'])
# model.fit(scaled_df, target)

# top_model = XGBRegressor(learning_rate=0.1,  
#                                   n_estimators= 1000)
# top_model.fit(scaled_df, target)


# In[78]:


top_model_name = base_model_scores[0][0]
top_model = dict(est)[top_model_name][0]

std_scaler = StandardScaler()
std_scaler.fit(enc_df)
scaled_df = std_scaler.transform(enc_df)

top_model.fit(scaled_df, target)


# # PREDICTION ON TEST DATA
# 
# *****************************
# 
# We already performed the same preprocessing steps on test dataset **( test_enc_df )** along with the training dataset. Hence we will just pass it through the top_model .

# In[79]:


test_scaled_df = std_scaler.transform(test_enc_df)
predictions = top_model.predict(test_scaled_df)


# In[80]:


submission = pd.DataFrame(columns=['Id', 'SalePrice'])
submission['Id'] = test_df['Id']
submission['SalePrice'] = predictions

submission.to_csv('submission.csv', index=False)


# # FEATURE ENGINEERING ANALYSIS
# ********************
# 
# * Feature engineering is an art and requires different combinations of techniques to attain better performance.
# 
# * Following cells shows different analysis that I made from the different combinations of techniques.

# ## VERSION 6 NOTEBOOK FEATURE ENGINEERING ANALYSIS
# 
# ************************
# I have performed various combinations of feature engineering and evaluated their results. The following results are on the basis of **IQR outlier removal and BoxCox feature transformation**
# 
# 
# 
# * **NO FEATURE TRANSFORMATION & NO OUTLIER REMOVAL**
# 
# 
#       ('GradientBoosting', 0.8608838908601799),
# 
#       ('ExtraTree', 0.8513076801447276),
# 
#       ('RandomForest', 0.8483765546873913)
#           
#           
# * **WITH FEATURE TRANSFORMATION & NO OUTLIER REMOVAL**
# 
# 
#       ('ExtraTree', 0.8516058854676427),
# 
#       ('GradientBoosting', 0.8491665618535139),
# 
#       ('RandomForest', 0.8462603323781929)
#          
#      
# * **NO FEATURE TRANSFORMATION & WITH OUTLIER REMOVAL**
# 
# 
#       ('ExtraTree', 0.8709907177784123),
# 
#       ('GradientBoosting', 0.8695859492594143),
# 
#       ('RandomForest', 0.8579828682121151)
#          
#          
# * **WITH FEATURE TRANSFORMATION & WITH OUTLIER REMOVAL**
# 
# 
#       ('GradientBoosting', 0.8735160942186653),
# 
#       ('ExtraTree', 0.8655515278510605),
# 
#       ('RandomForest', 0.8585987968429459)
#      
#  *************************
#  
# * Ensemble Trees have given the best results followed by the linear models with regularization (ridge, lasso etc.).
# 
# * Simple linear models have performed very poor. 
# 
# * On the basis of the r2 scores of different models, we can see we got **better results when we removed the outliers** although feature transformation didn't put much impact. This is because the trees doesn't take normality (gaussian distribution) into consideration. So even if you don't scale or normalize your data and directly put it into the emsemble trees, it would behave the same. Thats why **feature transformation** had a very minor impact.
# 
# * Also in the initial version of this notebook, I didn't perform the binning on year related features. In this notebook I performed it and kind of analysed the results and I found that it also had no impact on the results. The results were same. This could mean that year related columns were not impacting the prices (target variable) much.
# 
# *****************************

# ## LATEST VERSION NOTEBOOK FEATURE ENGINEERING ANALYSIS 

# * Intially I used **BOXCOX** transformation that requires all the values to be greater than 0, so I shifted the value to 0.02 in the columns and then applied **BOXCOX** transformation to it. It gave me a score of `0.23` in the scoreboard *(lesser the score, the better the model behaved on test set)*.
# 
# * But when I applied **Yeo-Johnson Transformation** which can work on both negative and positive values, my score became `0.13` in the scoreboard which is great. That means **TRANSFORMATION** did play a great role in better generalization (performance on unknown dataset). 
# 
# * Hence we can say that a **combination of outlier removal (IQR), performing binning on year columns and yeo-johnson feature transformation** has led to better result that the earlier combination of *IQR outlier removal, binning and boxcox feature transformation* .
# 
# * Let's see which features really helped in making decisions.

# # RESULT ANALYSIS 
# 
# ***************
# 
#    - Let's analyse the results from the training data

# ### Top 10 features with their scores

# In[81]:


feat_imp_df = pd.DataFrame(columns=['Features', 'Value'])
feat_imp_df['Features'] = enc_df.columns.to_list()
feat_imp_df['Value'] = top_model.feature_importances_

feat_imp_df[feat_imp_df['Value']>0].sort_values(by=['Value'], ascending=False).head(10)


# * From the above dataframe, we can see the top 10 features which are impacting the house prices.
# 
#     * **OverallQual**  - Rates the overall material and finish of the house. Discrete feature (0 to 10)
#     
#     * **GrLivArea** - Above grade (ground) living area square feet
#     
#     * **TotalBsmtSF** - Total square feet of basement area
#     
#     * **GarageCars** - Size of garage in car capacity
#     
#     * **BsmtFinSF1** - Basement finished square feet area
#     
#     * **2ndFlrSF** - Second floor square feet
#     
#     * **1stFlrSF** - First floor square feet
#     
#     * **Remod_Sold_Age** - This is the feature we created. Difference between YrSold and YearRemodAdd
#     
#          - YrSold - Year in which the house is sold
#     
#          - YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#          
#     * **LotArea** - Lot size in square feet. A lot area is the total area of a property.
#     
#     * **GarageArea** - Size of the garage

# ### Visualising these columns with respect to the price

# In[82]:


cols = feat_imp_df[feat_imp_df['Value']>0].sort_values(by=['Value'], ascending=False).head(10)['Features'].to_list()


# In[83]:


#  Scatterplot for numerical features

plt.figure(figsize=(20,90))
for i in range(len(cols)):
    plt.subplot(12, 3, i+1)
    sns.scatterplot(x=enc_df[cols[i]], y=target)

plt.show()


# # STORY TELLING FROM THE ABOVE PLOTS

# * Houses with a good overall condition, with a greater living room which is above ground along with a greater basement area are expensive houses. 
# 
# * People are preferring those houses that has furnished basement with a greater area with a garage having capacity of storing more cars.
# 
# * It seems that most of the people prefer to buy two story houses, with greater floor area. 
# 
# * Big houses or a mansions with a greater area are even more expensive.
# 
# * It seems that people are preferring the houses that has been recently remodeled or reconstructed.
# 
# ***************
# ## MORAL OF THE STORY
# 
# * Buy a **big 2-floored** house with a **good overall condition**, with a **greater furnished basement area** along with a **wide garage** and which has been remodelled recently.

# In[ ]:




