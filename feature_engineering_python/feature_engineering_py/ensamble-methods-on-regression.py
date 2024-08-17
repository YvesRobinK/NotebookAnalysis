#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn-images-1.medium.com/max/1200/1*surWujKv0sqceD1kOYpqtQ.png)

# Goal is to showcase different ensamble methods when performing regression.

# Necessary libraries/tools

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warnings (as always)


from scipy import stats
from scipy.stats import norm, skew #to know where to apply Box-cox transformation


# Read in the dataset

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head(5)


# Standard step: Remove id columns from both data sets since we are not going to need when modeling.

# In[3]:


#Save it
train_id = train['Id']
test_id = test['Id']

#drop it
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# **Data-Processing**

# When performing regression it is always important to aim for normality, that is does our dependent variable follows normal distribution (theoretically it is important demand). Other than that we should also (in general) look for outliers to remove them from data. Either we have some sort of documentation about the data that specifically indicates that like [http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt] (http://) (under special notes) or we can do a joint plot of all of the variables versus the dependent variables. And get a hunch where the outlier might be hidden.

# In[4]:


#g = sns.pairplot(data=train, palette = 'seismic',
                #size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) ) #problem is the size of the dataset. Execution is to long, that is why we are going to focus only on notes and look at the problematical variable


# In[5]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Remove

# In[6]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# **NOTE** We should be careful with outliers, always removing them is not the best choice. It could happen that we have also outliers in the test set. Than removel from training set will not give desired predictions. We should rather opt for another choice and that is scaling/transformations in order to make our model robust to the outliers

# *Normality Assumption* AS already mentioned we ought to make normality assumptions when dealing with regression. First we need to check whether it is neccesary.

# In[7]:


sns.distplot(train['SalePrice'] , fit=norm); #Informal plot, where blue line is  our data and the black is the theoretical normal distribution

# Fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Legend
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Formally with QQ-plot we can determine wether distribution is normal
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# Log transformation is just a special case of Box-Cox transformation, let us apply it on our prices data to make it more normal.

# In[8]:


#log1p applies log(1+x) (because of 0 values) transformation
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# **Feature engineering**

# In[9]:


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True) #data to be predicted
print("all_data size is : {}".format(all_data.shape))
y_train = train.SalePrice.values


# Standard approach, missing data, scaling, imputating etc...

# In[10]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# Plot the distribution of missing values

# In[11]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# Lets go variable by variable and inpute the values (if possible/reasonable)! Question that we need to ask ourselves what does NaN stand for this specific feature and then imupte it accordingly.

# PoolQC: NA means "No Pool"

# In[12]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("No pool")


# MiscFeature :  NA means "no misc feature"

# In[13]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")


# Now there are a couple of them where missing values indicites None (meaning it does not exist) so we can jsut write a for loop for these columns:
# 

# In[14]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',"FireplaceQu","Fence","Alley"):
    all_data[col] = all_data[col].fillna('None')


# *LotFrontage*: Linear feet of street connected to property. Now this property of the house is most likely going to be similiar to the other ones in the neighbourhood. ´SO let us group and impute with the median (due to potential outliers)

# In[15]:


all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# Another thing that we can notice is that some variables will inherit the imputed value due to the fact that we do not have the object at hand. For example havign no garage implies for the following 3 variables that NaN means 0

# In[16]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# Same for basement

# In[17]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# Likewise for categorical values of basement NaN will imply "None"

# In[18]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# We go further down the list with our missing features. This is the most boring part when we have this many variables but the procedure is the same. What does NaN most likely mean for this variable, and does it even make sence to impute it?

#  NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
# 

# In[19]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# MSZoning: since "RL" is the most common values, we are going to use mode to impute it

# In[20]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# Utilities : This categorical variable has almost all of the observations as "AllPub", except for one "NoSeWa" and 2 NA . No predictive power!---remove

# In[21]:


all_data = all_data.drop(['Utilities'], axis=1)


# Functional: Read the data descritpion or documentation, there we can see that NA means typical

# In[22]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# 
# We are moving futher down our list
# Electrical has one NA value. Since this feature has mostly 'SBrkr' (but not only), we can set that for the missing value.

# In[23]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# KitchenQual: Same as for electrical, we can see only one missing values and we are going to impute it with the most occuring one

# In[24]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd and SaleType have just one missing values (both of them are strings!) so we are just going to impute with the most common string!

# In[25]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# In[26]:


# The test example with ID 666 has GarageArea, GarageCars, and GarageType 
# but none of the other fields, so use the mode and median to fill them in.
test.loc[666, "GarageQual"] = "TA"
test.loc[666, "GarageCond"] = "TA"
test.loc[666, "GarageFinish"] = "Unf"
test.loc[666, "GarageYrBlt"] = "1980"

# The test example 1116 only has GarageType but no other information. We'll 
# assume it does not have a garage.
test.loc[1116, "GarageType"] = np.nan

# For imputing missing values: fill in missing LotFrontage values by the median
# LotFrontage of the neighborhood.
lot_frontage_by_neighborhood = train["LotFrontage"].groupby(train["Neighborhood"])


# That should be it with NaN, let us check

# In[27]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# **Data Correlation**

# In[28]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# **How to think about this correlation heat-map**
# 
# The correlation coefficient is bound between -1 and 1 and tells you the linear relationship between these two variables. A coefficient close to 1 means a strong and positive associantion between the two variables (when one of them grows, the other does, also, and when one of them decreases, the other one does the same).
# A coefficient close to -1 means strong negative association between the two variables, this is, observations with a large value in one of the variables tend to have a small value in the other variable or vice-versa.
# A coeffcient close to 0 means no linear relation between the two variables.
# Yo have to be careful with the following matters:
# 1) Association does not mean necessarily a causal relation between both variables. For example, there might be a third variable you have not cosidered and this third variable might be the explanation for the behaviour of the other two.
# 2) Even if there is a causal relationship between the variables, the correlation coefficient does not tell you which variable is the cause and which is the effect.
# 3) If the coefficient is clse to 0, it does not necessarily mean that there is no relation between the two variables. It means there is'nt a LINEAR relationship, but there might be another type of functional relationship (for example, quadratic or exponential).

# So what is our next step in data processing? We should note that ML algorithms can only process numerical data (it needs to be encoded in numerical format), hence we ought to labelEncode (if there is hierarchy in the independent variable, i.e. good, better best---0 ,1 ,2) or Dummy encode (0 and 1 vectors). Next we should also think about what variables are actually categorical even tough they numerical or strings. For fully numerical values we should also think about skewnes (how can we remedy it?). Is there any text that we can extract info from? And finally what about scaling, we need to make our data comparable and robust to outliers!

# For example, our first column MSSUbClass should be actually categorical, and not only that but with some hiararchy also (since there is difference whether it is 120 or 20)
# 
# **NOTE** if we do not labelENCODE numerical variables BEFORE we apply dummy encoding, than these variables will never be encoded. Since dummy encoding works only on categorical variables.

# In[29]:


# First we need to make sure they are strings in order to make sure labelEncoding can be applied
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[30]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# New feature, total sqaure footage:

# In[31]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# AS soon as we have numerical variables, check for skeewnes and correct it

# In[32]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# Coreect it with Box-Cox method

# In[33]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# **Dummy Encoding**

# In[34]:


all_data = pd.get_dummies(all_data)

ntrain = train.shape[0]
ntest = test.shape[0]
train = all_data[:ntrain]#newdata
test = all_data[ntrain:]


# **MODELING**

# First couple of "simple" models than we are going to stack them (using Python classes) into more powerful and accurate model.

# In[35]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline #using a pipeline we can chain together modeling and scaling where it is neccesary
from sklearn.preprocessing import RobustScaler #scaling in pipeline (robust to outliers!)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[36]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) #shuffle=True, shuffles the whole data set before each cross-validation split
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# **Models**
# But before we proceede we should note that we did not scale anythin until now. Scaling should take place before PCA or kNN for example because different metrics will affect the results. Also when performing gradient descent not scaling might slow-down the speed of algorithm. In Lasso and Ridge regression penalise the outliers, hence we need to make sure outliers are on the same scale (across all of the features).

# **Lasso**

# In[37]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# **Elastic Net Regression :**

# In[38]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# **Kernel Ridge Regression :**

# In[39]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# **Gradient Boosting Regression :**

# In[40]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# **XGBoost :**

# In[41]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# **LightGBM :**

# In[42]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# **Base models scores**

# In[43]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# # Ensamble learning methods: 
# All of the methods here presented aim in the end to optimise following function:

# ![](data:image/png;base64,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)

# 
# 
# **Simplest ensambling approach : Averaging base models**
# We begin with this simple approach of averaging base models. We build a new class to extend scikit-learn with our model and also to laverage encapsulation and code reuse (inheritance). A way to look at classes is that they are now same type as our model. Meaning we define which functionalities they will have. Very simply said, **avaraging base method** uses predictions (y_prediction) avarage values of all the algorithms as our final prediction. In more complicated ensembling (further below) WE add a meta-model which is trained on the predictions from the previous ones. (Please differentiate betweent boosting (ensamle method) where we "stack"-build up on residuals of the former-weaker models!

# In[44]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# Let us average all of them :)

# In[45]:


averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso,model_xgb,model_lgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# We first define a rmsle evaluation function

# In[46]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# # Stacking
# It is an ensambling method were we use predictions of other models to make the final prediction. Please do consult the link in the begininng to understand the concept. In short what happens is that we have moved one step toward the true solution (here with averaging of bagging method) and now for the final model (here it is xgboost) we will be using actually predictions as predictor variables. So now if we have had on the initial M models n predictors that are used to make predictions. Than on the second model we will have M predictors (for M models) and the same number of rows in other words observations. But here is the crucial part, what observations? **k-fold cross validation** has to be used when training the first set of models. In order to make sure that we really exploit weaknesses and strengths of different models we need to know were are they stron or weak, if we were not to use k-fold cross validation than we would not find out. Predictions from a validation (hold-out) set are going to be the new features of the final model

# In[47]:


averaged_models.fit(train.values, y_train)
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15


# Submit

# In[48]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)

