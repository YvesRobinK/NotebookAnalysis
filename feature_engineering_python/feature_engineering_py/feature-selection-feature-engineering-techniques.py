#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

from sklearn import metrics
import xgboost as xgb
from scipy import stats
from scipy.stats import norm, skew

from IPython.display import display


# In[2]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


df_train.shape


# In[4]:


df_test.shape


# In[5]:


df_train.info()


# In[6]:


df_train.describe()


# In[7]:


pd.options.display.max_rows = None
display(100-(df_train.isnull().sum()*100/len(df_train)))


# In[8]:


pd.options.display.max_rows = None
display(100-(df_test.isnull().sum()*100/len(df_test)))


# There are features with a lot of missing values.

# In[9]:


df_train.columns


# # Exploratory Data Analysis(EDA)

# In[10]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('House Price Distribution Plot')
sns.distplot(df_train.SalePrice,color ='red')

plt.subplot(1,2,2)
plt.title('House Price Spread')
ax = sns.boxplot(y = df_train.SalePrice ,color = 'pink' )

plt.show()


# In[11]:


plt.figure(figsize = (30, 30))
sns.heatmap(df_train.corr(), annot = True, cmap="PiYG")
plt.show()


# Variable that are highly correlated to the Target variable: 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'Heating', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea'

# # Data Preprocessing
# 

# ## Feature Engineering 
# 
# Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. A feature is a property shared by independent units on which analysis or prediction is to be done. Features are used by predictive models and influence results. 
# 
# ### Handling Missing Data
# There can be 4 ways to handle the missing data:
# - Deleting the rows (loss of information)
# - Replace with the most frequent values
# - Apply classifier/regressor model to predict missing values
# - Apply unsupervised machine learning to predict (clustering)

# In[12]:


sol = df_test['Id']


# In[13]:


#Dropping columns with less than 60% missing data
df_train = df_train.drop(columns=['Id'])
df_train = df_train.drop(columns=['Alley'])
df_train = df_train.drop(columns=['FireplaceQu'])
df_train = df_train.drop(columns=['PoolQC'])
df_train = df_train.drop(columns=['Fence'])
df_train = df_train.drop(columns=['MiscFeature'])

df_test = df_test.drop(columns=['Id'])
df_test = df_test.drop(columns=['Alley'])
df_test = df_test.drop(columns=['FireplaceQu'])
df_test = df_test.drop(columns=['PoolQC'])
df_test = df_test.drop(columns=['Fence'])
df_test = df_test.drop(columns=['MiscFeature'])


# In[14]:


#Imputing numerical data with mean value
df_train["LotFrontage"] = df_train["LotFrontage"].replace(np.NaN, df_train["LotFrontage"].mean())
df_train["MasVnrArea"] = df_train["MasVnrArea"].replace(np.NaN, df_train["MasVnrArea"].mean())
df_train["GarageYrBlt"] = df_train["GarageYrBlt"].replace(np.NaN, df_train["GarageYrBlt"].mean())

df_test["LotFrontage"] = df_test["LotFrontage"].replace(np.NaN, df_test["LotFrontage"].mean())
df_test["MasVnrArea"] = df_test["MasVnrArea"].replace(np.NaN, df_test["MasVnrArea"].mean())
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].replace(np.NaN, df_test["GarageYrBlt"].mean())
df_test["BsmtFinSF1"] = df_test["BsmtFinSF1"].replace(np.NaN, df_test['BsmtFinSF1'].mean())
df_test["BsmtFinSF2"] = df_test["BsmtFinSF2"].replace(np.NaN, df_test['BsmtFinSF2'].mean())
df_test["BsmtFullBath"] = df_test["BsmtFullBath"].replace(np.NaN, df_test['BsmtFullBath'].mean())
df_test["BsmtHalfBath"] = df_test["BsmtHalfBath"].replace(np.NaN, df_test['BsmtHalfBath'].mean())
df_test["GarageCars"] = df_test["GarageCars"].replace(np.NaN, df_test['GarageCars'].mean())
df_test["GarageArea"] = df_test["GarageArea"].replace(np.NaN, df_test["GarageArea"].mean())
df_test["BsmtUnfSF"] = df_test["BsmtUnfSF"].replace(np.NaN, df_test['BsmtUnfSF'].mean())
df_test["TotalBsmtSF"] = df_test["TotalBsmtSF"].replace(np.NaN, df_test['TotalBsmtSF'].mean())


# In[15]:


#Imputin missing categorical data values with mode value
df_train["BsmtQual"] = df_train["BsmtQual"].replace(np.NaN, df_train["BsmtQual"].mode()[0][:])
df_train["BsmtExposure"] = df_train["BsmtExposure"].replace(np.NaN, df_train["BsmtExposure"].mode()[0][:])
df_train["BsmtFinType1"] = df_train["BsmtFinType1"].replace(np.NaN, df_train["BsmtFinType1"].mode()[0][:])
df_train["BsmtCond"] = df_train["BsmtCond"].replace(np.NaN, df_train["BsmtCond"].mode()[0][:])
df_train["BsmtFinType2"] = df_train["BsmtFinType2"].replace(np.NaN, df_train["BsmtFinType2"].mode()[0][:])
df_train["Electrical"] = df_train["Electrical"].replace(np.NaN, df_train["Electrical"].mode()[0][:])
df_train["GarageType"] = df_train["GarageType"].replace(np.NaN, df_train["GarageType"].mode()[0][:])
df_train["GarageFinish"] = df_train["GarageFinish"].replace(np.NaN, df_train["GarageFinish"].mode()[0][:])
df_train["GarageQual"] = df_train["GarageQual"].replace(np.NaN, df_train["GarageQual"].mode()[0][:])
df_train["GarageCond"] = df_train["GarageCond"].replace(np.NaN, df_train["GarageCond"].mode()[0][:])
df_train["MasVnrType"] = df_train["MasVnrType"].replace(np.NaN, df_train['MasVnrType'].mode()[0][:])

df_test["BsmtQual"] = df_test["BsmtQual"].replace(np.NaN, df_test["BsmtQual"].mode()[0][:])
df_test["BsmtExposure"] = df_test["BsmtExposure"].replace(np.NaN, df_test["BsmtExposure"].mode()[0][:])
df_test["BsmtFinType1"] = df_test["BsmtFinType1"].replace(np.NaN, df_test["BsmtFinType1"].mode()[0][:])
df_test["BsmtCond"] = df_test["BsmtCond"].replace(np.NaN, df_test["BsmtCond"].mode()[0][:])
df_test["BsmtFinType2"] = df_test["BsmtFinType2"].replace(np.NaN, df_test["BsmtFinType2"].mode()[0][:])
df_test["Electrical"] = df_test["Electrical"].replace(np.NaN, df_test["Electrical"].mode()[0][:])
df_test["GarageType"] = df_test["GarageType"].replace(np.NaN, df_test["GarageType"].mode()[0][:])
df_test["GarageFinish"] = df_test["GarageFinish"].replace(np.NaN, df_test["GarageFinish"].mode()[0][:])
df_test["GarageQual"] = df_test["GarageQual"].replace(np.NaN, df_test["GarageQual"].mode()[0][:])
df_test["GarageCond"] = df_test["GarageCond"].replace(np.NaN, df_test["GarageCond"].mode()[0][:])
df_test["MasVnrType"] = df_test["MasVnrType"].replace(np.NaN, df_test['MasVnrType'].mode()[0][:])
df_test["Utilities"] = df_test["Utilities"].replace(np.NaN, df_test['Utilities'].mode()[0][:])
df_test["MSZoning"] = df_test["MSZoning"].replace(np.NaN, df_test['MSZoning'].mode()[0][:])
df_test["SaleType"] = df_test["SaleType"].replace(np.NaN, df_test['SaleType'].mode()[0][:])
df_test["Exterior1st"] = df_test["Exterior1st"].replace(np.NaN, df_test['Exterior1st'].mode()[0][:])
df_test["Exterior2nd"] = df_test["Exterior2nd"].replace(np.NaN, df_test['Exterior2nd'].mode()[0][:])
df_test["KitchenQual"] = df_test["KitchenQual"].replace(np.NaN, df_test['KitchenQual'].mode()[0][:])
df_test["Functional"] = df_test["Functional"].replace(np.NaN, df_test['Functional'].mode()[0][:])


# In[16]:


pd.options.display.max_rows = None
display(100-(df_test.isnull().sum()*100/len(df_test)))
#no missing data


# In[17]:


pd.options.display.max_rows = None
display(100-(df_train.isnull().sum()*100/len(df_train)))
#no missing data


# ### Encoding Categorical Data
# Types of Encoding: 
# 1. Nominal Encoding : (Categorical features where rank is not important)
#     - One Hot encoding
#     - One Hot encodng with many categories - We take 10(or more) most frequently occuring categories and group them into 1       category.
#     - Mean encoding - Replace the label with mean Eg; Pincode
# 2. Ordinal Encoding : (Categorical features where rank is important)
#     - Label encoding
#     - Target guided ordinal encoding
# 3. Count/Frequency Encoding: Can be used for both nominal and ordinal features

# In[18]:


cat_features = list(df_train.select_dtypes(include='object').columns)
print(cat_features)


# In[19]:


for col in cat_features:
    print(col,"   ", df_train[col].unique())


# Ordinal Categorical Features: 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond' 
# 
# Nominal Categorical features: 'MSZoning','Street','LotShape', 'LandContour', 'Utilities', 'LotConfig',
#  'LandSlope', 'BldgType', 'RoofStyle', 'MasVnrType', 'Foundation', 'BsmtFinType1',
#  'BsmtFinType2', 'Heating', 'CentralAir','Electrical','Functional',
#  'GarageType', 'PavedDrive', 'SaleCondition'

# 1. One Hot Encoding for Features with large number of Categories

# In[20]:


for col in cat_features:
    print(col," : ",len(df_train[col].unique())," labels")


# Features with large number of categories: 'Neighourhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1',
#     'Condition2', 'HouseStyle', 'RoofMatl'

# In[21]:


top_12 = [x for x in df_train['Neighborhood'].value_counts().sort_values(ascending=False).head(12).index]
top_12


# In[22]:


#Applying one hot encoding on the most frequently occuring top 12 results and labeling 1 if the condition is true
for label in top_12:
    df_train[label] = np.where(df_train['Neighborhood']==label , 1, 0)
    df_test[label] = np.where(df_test['Neighborhood']==label , 1, 0)



# In[23]:


def one_hot_top_x(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable+'_'+label] = np.where(df[variable]==label , 1, 0)

        
one_hot_top_x(df_train, 'Neighborhood',top_12)
df_train = df_train.drop(columns=['Neighborhood'])

one_hot_top_x(df_test, 'Neighborhood',top_12)
df_test = df_test.drop(columns=['Neighborhood'])

df_train.head()


# In[24]:


#Applying one hot encoding on the most frequently occuring top 5 results and labeling 1 if the condition is true
feat_top_5 = [ 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl']

for feat in feat_top_5:
    top_5 = [x for x in df_train[feat].value_counts().sort_values(ascending=False).head(5).index]
    one_hot_top_x(df_train, feat ,top_5)
    one_hot_top_x(df_test, feat ,top_5)

df_train = df_train.drop(columns=feat_top_5)
df_test = df_test.drop(columns=feat_top_5)


# In[25]:


df_train.shape


# In[26]:


df_test.shape


# 2. One Hot Encoding for ordinary categorical Features

# In[27]:


feat_one_hot=['MSZoning','Street','LotShape', 'LandContour', 'Utilities', 'LotConfig',
 'LandSlope', 'BldgType', 'RoofStyle', 'MasVnrType', 'Foundation', 'BsmtFinType1',
 'BsmtFinType2', 'Heating', 'CentralAir','Electrical','Functional',
 'GarageType', 'PavedDrive', 'SaleCondition','GarageFinish']


def dummies(x,df):
    temp = pd.get_dummies(df[x], prefix = x , drop_first = True).astype('int32')
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

for ft in feat_one_hot:
    df_train = dummies(ft,df_train)
    df_test = dummies(ft,df_test)


# In[28]:


df_train.shape


# In[29]:


df_test.shape


# In[30]:


for x in df_train.columns:
        if x not in df_test.columns:
            print(x)


# In[31]:


df_train.drop(columns = ['Utilities_NoSeWa','Heating_GasA','Heating_OthW','Electrical_Mix'], inplace=True)


# 3. Encoding Ordinal features using Label encoding 

# In[32]:


df_train['BsmtExposure'].unique()


# In[33]:


ordinal_feat = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']

qual_map={
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}

#Here, Ex:'Excellent' Gd:'Good' Fa:'Fair' TA:'Typical' Po:'Poor'

def ord_encode(df):
    for qual in ordinal_feat:
        df[qual+'_ord'] = df[qual].map(qual_map)
        df.drop(columns=[qual], inplace=True)
    
ord_encode(df_train)
ord_encode(df_test)


# In[34]:


qual_map={
    'Gd': 5,
    'Av': 4,
    'Mn': 3,
    'No': 2,
    'NB': 1
}

#Here, Gd:'Good Exposure' Av:'Average Exposure' Mn:'Minimum Exposure' No:'No Exposure' NB:'No Basement'

def ord_encode_Expo(df):
        df['BsmtExposure'+'_ord'] = df['BsmtExposure'].map(qual_map)
        df.drop(columns=['BsmtExposure'], inplace=True)
    
ord_encode_Expo(df_train)
ord_encode_Expo(df_test)


# In[35]:


df_train.shape


# In[36]:


df_test.shape


# ## Feature Selection for numerical Data

# In[37]:


num_features= ['MSSubClass',
'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
'HalfBath', 'BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch',
'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] 


# 1. Removing constant features using Variance Threshold
# 

# In[38]:


# Threshold=0 means it will remove all the low variance features
var_thres = VarianceThreshold(threshold=0)
var_thres.fit(df_train[num_features])


# In[39]:


constant_columns = [x for x in num_features
                    if x not in df_train[num_features].columns[var_thres.get_support()]]

print(len(constant_columns))


# In[40]:


print(var_thres.get_support())


# - No Duplicate or constant features present in the dataset.

# 2. Dropping Features Using Pearson Correlation

# In[41]:


df_x = df_train[num_features]
y_train = df_train['SalePrice']


# In[42]:


#Pearson Correlation Chart
plt.figure(figsize = (30, 30))
sns.heatmap(df_x.corr(), annot = True, cmap="PiYG")
plt.show()


# In[43]:


# Finding out how many features are correlated to each other to avoid having 
# duplicate features in our model.

def correlation(df, thres):
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i,j] > thres:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_feat = correlation(df_x, 0.7)
len(set(corr_feat))


# In[44]:


# Duplicate Features
print(corr_feat)


# In[45]:


df_train = df_train.drop(corr_feat, axis=1)
df_test = df_test.drop(corr_feat, axis=1)


# 3. Using Information gain - mutual information for Feature Selection
# 
# I(X ; Y) = H(X) – H(X | Y) and IG(S, a) = H(S) – H(S | a)
# 
# As such, mutual information is sometimes used as a synonym for information gain. Technically, they calculate the same quantity if applied to the same data.

# In[46]:


mutual_info = mutual_info_regression(df_x, y_train)
mutual_info


# In[47]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = df_x.columns
mutual_info.sort_values(ascending=False)


# In[48]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))


# In[49]:


selected_top_columns = SelectPercentile(mutual_info_regression, percentile=40)
selected_top_columns.fit(df_x, y_train)


# In[50]:


selected_top_columns.get_support()


# In[51]:


df_x.columns[selected_top_columns.get_support()]


# In[52]:


unimp_columns = [x for x in num_features
                    if x not in df_x.columns[selected_top_columns.get_support()]]
print(unimp_columns)


# In[53]:


for x in unimp_columns:
    df_train.drop(columns=[x], axis=1, inplace=True)
    df_test.drop(columns=[x], axis=1, inplace=True)


# ### Outlier Detection
# 

# In[54]:


num_vars = ['MSSubClass', 'LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
        'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'ExterQual_ord', 
        'ExterCond_ord', 'BsmtQual_ord', 'BsmtCond_ord','BsmtExposure_ord',
        'HeatingQC_ord', 'KitchenQual_ord', 'GarageQual_ord', 'GarageCond_ord']


# In[55]:


z = np.abs(stats.zscore(df_train[num_vars]))
threshold = 3
print(np.where(z > 3))


# In[56]:


df = pd.concat((df_train.drop(columns=['SalePrice']), df_test))


# ### Transforming the skewed features
# 
# Normally distributed features are an assumption in Statistical algorithms. Deep learning & Regression-type algorithms also benefit from normally distributed data.
# Transformation is required to treat the skewed features and make them normally distributed. Right skewed features can be transformed to normality with Square Root/ Cube Root/ Logarithm transformation.

# In[57]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":df_train["SalePrice"], "log(price + 1)":np.log1p(df_train["SalePrice"])})
prices.hist()


# In[58]:


numeric_cols = ['MSSubClass', 'LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
        'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars']

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

skewed_feats = df_train[numeric_cols].apply(lambda x: skew(x)) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df[skewed_feats] = np.log1p(df[skewed_feats])


# ### Feature Scaling
# Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization. When to use feature scaling:
# - Algorithms which use gradient descent or euclidean distance for eg: Knn or K means clustering etc
# 
# When to not use feature scaling:
# - Algorithms like Decision Tree, Random Forest, or XGBoost etc.
# 
# NOTE: We are gonna skip scaling in this model as we have applied logarithmic transformation to treat skewed features. Also, we mostly are gonna use xgboost model and it is recommended not to use feature scaling along with XGB model.

# ### Preparing the dataset for model fitting and Splitting the data into training set and test set

# In[59]:


x_train = df[:df_train.shape[0]]
x_test = df[df_train.shape[0]:]
y_train = df_train.SalePrice


# In[60]:


x_test.head()


# # Model Building
# 
# Now we are going to use regularized linear regression models from the scikit learn module. I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning parameter.

# In[61]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# This problem uses the mean squared error, mse as the scoring metric. However, this metric returns negative values. Therefore, we need to use abs(mse) to get positive values.
# 
# The mse takes the errors: difference between the actual values and those predicted by the model, and find the mean of the squares.
# 
# It isn’t null and the negative sign does not make it ineffective. A high mse means that the error is large.

# ## 1. Multiple linear Regression Model

# In[95]:


mlr = LinearRegression()
mlr.fit(x_train,y_train)
mlr_preds = mlr.predict(x_test)


# In[93]:


rmse_cv(mlr).min()


# ## 2. Ridge Regression

# The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# In[65]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[66]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation Curve")
plt.xlabel("alpha")
plt.ylabel("rmse")


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 3 is about right based on the plot above.

# In[67]:


cv_ridge.min()


# In[87]:


ridge = Ridge(alpha = 3).fit(x_train, y_train)


# In[88]:


ridge_preds = np.expm1(ridge.predict(x_test))


# ## 3. Lasso Regression

# In[68]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_train, y_train)model_ridge


# In[69]:


rmse_cv(model_lasso).mean()


# In[70]:


lasso_preds = np.expm1(model_lasso.predict(x_test))


# ## 4. XGBoost Model

# In[71]:


dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[72]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[73]:


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(x_train, y_train)


# In[74]:


xgb_preds = np.expm1(model_xgb.predict(x_test))


# In[89]:


preds = 0.7*lasso_preds + 0.3*xgb_preds


# In[96]:


solution = pd.DataFrame({"id":sol, "SalePrice":preds})
solution.to_csv("solution.csv", index = False)

