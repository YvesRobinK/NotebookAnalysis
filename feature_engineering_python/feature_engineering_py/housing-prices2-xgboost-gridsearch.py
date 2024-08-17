#!/usr/bin/env python
# coding: utf-8

# # **Housing Prices Competition**

# ## Contents:
# 1. **[X, y Summary](#DF)**
#     * Import Libraries
#     * Read the data
#     * X, y info
#     * Correlation between features and target on heatmap
# 2. **[Data Cleaning](#Clean)**
#     * Remove the columns with more than half missing values
#     * Drop columns with most of the rows having only one category
# 3. **[Feature Engineering](#Feature)**
#     * Create New Numerical Features
#     * Create New Boolean Features
#     * Replace ordered categories with numbers
#     * Create features using mathematical transformations
#     * Create feature using count
#     * Create feature using group transforms
#     * Drop high cordinality categorical columns
#     * Handle rare categorical values
#     * Create features using feature interactions
#     * Impute numerical columns
# 4. **[Data Visualization](#Viz)**
#     * Distribution of top 5 features correlated with Sales Price
# 5. **[Feature Selection](#Select)**
#     * Selecte Features
# 6. **[Model Creation](#Model)**
#     * Grid Search
# 7. **[Training and Testing Model](#Test)**
#     * Best Parameters
#     * Feature Importance

# # 1) X, y Summary<a id="DF"></a>

# ## Import Libraries

# In[1]:


# Data Analytics Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

# Machine Learning Libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder #LabelEncoder #OneHotEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from math import ceil

# Update some default parameters for plotting throughout the notebook
plt.rcParams.update({'font.size': 12, 'xtick.labelsize':15, 'ytick.labelsize':15, 'axes.labelsize':15, 'axes.titlesize':20})


# In[2]:


# List all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Create X, y DataFrames

# In[3]:


# Read the data
Xy = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

# Remove rows with missing target
Xy = Xy.dropna(axis=0, subset=['SalePrice'])

# Separate target from predictors
X = Xy.drop(['SalePrice'], axis=1)
y = Xy.SalePrice


# ## X, y Info

# In[4]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)


# In[5]:


info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info


# In[6]:


X.dtypes.value_counts()


# In[7]:


y.describe()


# ## Correlation between features and target on heatmap

# In[8]:


correlation_matrix = Xy.corr()

# Returns copy of array with upper part of the triangle (which will be masked/hidden)
mask = np.triu(correlation_matrix.corr())

sns.set(font_scale=1.1)
plt.figure(figsize=(20, 20), dpi=140)
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm', 
            square=True, mask=mask, linewidths=1, cbar=False)
plt.show()


# # 2) Data Cleaning<a id="Clean"></a>

# In[9]:


# Total rows/observations/houses in Training data and Test data
print(len(X),len(X_test))


# ## Remove the columns with more than half missing values

# In[10]:


# Making function so that we can reuse it in later stages as well
def show_null_values(X, X_test):
    # Missing values in each column of Training and Testing data
    null_values_train = X.isnull().sum()
    null_values_test = X_test.isnull().sum()

    # Making DataFrame for combining training and testing missing values
    null_values = pd.DataFrame(null_values_train)
    null_values['Test Data'] = null_values_test.values
    null_values.rename(columns = {0:'Train Data'}, inplace = True)

    # Showing only columns having missing values and sorting them
    null_values = null_values.loc[(null_values['Train Data']!=0) | (null_values['Test Data']!=0)]
    null_values = null_values.sort_values(by=['Train Data','Test Data'],ascending=False)
    
    print("Total miising values:",null_values.sum(),sep='\n')
    
    return null_values


# In[11]:


show_null_values(X, X_test)


# In[12]:


# Columns with missing values in more than half number of rows
null_cols = [col for col in X.columns if X[col].isnull().sum() > len(X)/2]
null_cols


# In[13]:


X.drop(null_cols,axis=1,inplace=True)
X_test.drop(null_cols,axis=1,inplace=True)


# In[14]:


# Total missing values after removing columns with more than half missing values
print("Total missing values:")
print("Training data\t",X.isnull().sum().sum())
print("Testing data\t",X_test.isnull().sum().sum())


# ## Data Visualization (Categorical Data)

# In[15]:


object_cols = X.select_dtypes('object').columns
len(object_cols)


# In[16]:


fig, ax = plt.subplots(nrows=ceil(len(object_cols) / 4), ncols=4, figsize=(22, 1.4*len(object_cols)), sharey=True, dpi=120)

for col, subplot in zip(object_cols, ax.flatten()):
    freq = X[col].value_counts()
    subplot.ticklabel_format(style='plain')
    plt.ylim([0, 800000])
    plt.subplots_adjust(wspace=.1,hspace=.4)
    for tick in subplot.get_xticklabels():
        tick.set_rotation(45)
    sns.violinplot(data=X, x=col, y=y, order=freq.index, ax=subplot)


# ## Drop columns with most of the rows having only one category

# In[17]:


# From above violin plots, 'Utilities' feature seems to have mostly one category.
# Lets confirm that using value_counts for each of its categories.
X.Utilities.value_counts()


# In[18]:


X_test.Utilities.value_counts()


# In[19]:


X.drop('Utilities',axis=1,inplace=True)
X_test.drop('Utilities',axis=1,inplace=True)


# # 3) Feature Engineering<a id="Feature"></a>

# In[20]:


# Merge the datasets so we can process them together
df = pd.concat([X, X_test])


# ## 3.1) Creating New Numerical Features

# In[21]:


df1 = pd.DataFrame()  # dataframe to hold new features

# Age of House when sold
df1['Age'] = df['YrSold']-df['YearBuilt']

# Years between Remodeling and sales
df1['AgeRemodel'] = df['YrSold']-df['YearRemodAdd']

Years = ['YrSold','YearBuilt','YearRemodAdd']
year_cols = ['YrSold','YearBuilt','AgeRemodel', 'Age']
df_1 = pd.concat([df, df1], axis=1).loc[:,year_cols]
X_1 = df_1.loc[X.index, :]
X_1.sample()


# In[22]:


sns.set(style='whitegrid')
# sns.set_context("paper", rc={"font.size":20,"axes.titlesize":25,"axes.labelsize":20}) 
sns.set(rc={'figure.figsize':(11.7,8.27),"font.size":20,"axes.titlesize":20,"axes.labelsize":20},style="white")
# sns.set(font_scale=1.1)
fig, ax = plt.subplots(1, 4, figsize=(20, 6), dpi=100)

# scatterplot
for col,i in zip(year_cols, [0,1,2,3]):
    sns.scatterplot(x=X_1.loc[:,col], y=y, ax=ax[i], hue=X.ExterQual, palette='pastel')

fig.tight_layout()
fig.text(0.5, 1, 'Distribution of SalesPrice with respect to years columns', size=20, ha="center", va="center")
plt.show()


# In[23]:


# Correlation of year columns with SalePrice
X_1.corrwith(y)


# ## 3.2) Creating New Boolean Features
# **Remodel column:**
# * False (for 764 houses having Remodel date same as construction date i.e. no modeling or additions)
# * True (for 696 houses with modeling or additions done)
# 
# **Garage column:**
# * False (for 81 rows having missing values in columns GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond)
# 
# **Fireplace column:**
# * False (for 690 rows having missing values in column FireplaceQu)
# 
# **Basement column:**
# * False (for 37 rows having missing values in columns BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2)
# 
# **Masonry veneer column:**
# * False (for 8 rows having missing values in columns MasVnrType, MasVnrArea)

# In[24]:


df2 = pd.DataFrame()  # dataframe to hold new features

df2['Remodel'] = df['YearRemodAdd']!=df['YearBuilt']
df2['Garage'] = df['GarageQual'].notnull()
df2['Fireplace'] = df['FireplaceQu'].notnull()
df2['Bsmt'] = df['BsmtQual'].notnull()
df2['Masonry'] = df['MasVnrType'].notnull()

# Converting boolean columns [False,True] into numerical columns [0,1]
df2 = df2.replace([False,True], [0,1])
df2.sample()


# ## Ordered Categorical Columns

# ### Replacing ordered categories with numbers

# In[25]:


object_cols = df.select_dtypes(include=['object']).columns
# Categorical Columns with number of unuque categoies in them 
df[object_cols].nunique().sort_values()


# Read 'data_description.txt' from the input files. Some columns were having **levels of quality, condition or finish** in words (i.e. string format). These columns can be used as ordinal columns (i.e. ordered category columns).

# In[26]:


ordinal_cols = [i for i in object_cols if ('QC' in i) or ('Qu' in i) or ('Fin' in i) or ('Cond' in i) and ('Condition' not in i)]
df.loc[:,ordinal_cols] = df.loc[:,ordinal_cols].fillna('NA')
print("Column Names: [Unique Categories in each column]")
{col:[*df[col].unique()] for col in ordinal_cols}


# In[27]:


# 1] Columns with similar ordered categories [Poor<Fair<Typical/Average<Good<Excellent]
ordinal_cols1 = [i for i in object_cols if ('QC' in i) or ('Qu' in i) or ('Cond' in i) and ('Condition' not in i)]
df.loc[:,ordinal_cols1] = df.loc[:,ordinal_cols1].replace(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4,5])

# 2] Columns with similar ordered categories [No Garage/Basement<Unfinished<Rough Finished<Finished,etc]
ordinal_cols2 = ['BsmtFinType1', 'BsmtFinType2']
df.loc[:,ordinal_cols2] = df.loc[:,ordinal_cols2].replace(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [0,1,2,3,4,5,6])

# 3] Column with ordered categories [No Basement<No Exposure<Mimimum Exposure<Average Exposure<Good Exposure]
ordinal_cols3 = ['BsmtExposure']
df.loc[:,ordinal_cols3] = df.loc[:,ordinal_cols3].fillna('NA')
df.loc[:,ordinal_cols3] = df.loc[:,ordinal_cols3].replace(['NA', 'No', 'Mn', 'Av', 'Gd'], [0,1,2,3,4])

# 4] Column with ordered categories [Regular<Slightly irregular<Moderately Irregular<Irregular]
ordinal_cols4 = ['LotShape']
df.loc[:,ordinal_cols4] = df.loc[:,ordinal_cols4].replace(['Reg', 'IR1', 'IR2', 'IR3'], [0,1,2,3])

# 5] Column with ordered categories [No Garage<Unfinished<Rough Finished<Finished]
ordinal_cols5 = ['GarageFinish']
df.loc[:,ordinal_cols5] = df.loc[:,ordinal_cols5].replace(['NA', 'Unf', 'RFn', 'Fin'], [0,1,2,3])

# 6] Home functionality Column
ordinal_cols6 = ['Functional']
df.loc[:,ordinal_cols3] = df.loc[:,ordinal_cols3].fillna('Mod')
df.loc[:,ordinal_cols6] = df.loc[:,ordinal_cols6].replace(["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], list(range(8)))


# In[28]:


o_columns = ordinal_cols1+ordinal_cols2+ordinal_cols3+ordinal_cols4+ordinal_cols5+ordinal_cols6
df.loc[:,o_columns].dtypes.value_counts()


# ## 3.3) Creating features using mathematical transformations

# In[29]:


Bath_cols = [i for i in df.columns if 'Bath' in i]
Bath_cols


# In[30]:


SF_cols = ['TotalBsmtSF','1stFlrSF','2ndFlrSF']
df[SF_cols+Bath_cols] = df[SF_cols+Bath_cols].fillna(0)


# In[31]:


df3 = pd.DataFrame()  # dataframe to hold new features

df3["Liv_Qual"] = (df.OverallQual + df.OverallCond/3) * df.GrLivArea
df3["GarageArea_Qual"] = (df.GarageQual + df.GarageCond/3) * df.GarageArea * df.GarageCars
df3['BsmtArea_Qual'] = (df.BsmtQual * df.BsmtCond/3) * df.TotalBsmtSF
df3["LivLotRatio"] = df.GrLivArea / df.LotArea
df3["Spaciousness"] = (df['1stFlrSF'] + df['2ndFlrSF']) / df.TotRmsAbvGrd
df3['TotalSF'] = df[SF_cols].sum(axis = 1)
df3['TotalBath'] = df.FullBath + df.BsmtFullBath + df.HalfBath/2 + df.BsmtHalfBath/2
# df3["Garage_Spaciousness"] = df.GarageArea / (df.GarageCars+1)
# df3["BsmtQual_SF"] = ((df.BsmtQual + df.BsmtCond/2 + df.BsmtExposure/3) * df.TotalBsmtSF) + (df.BsmtFinType1 * df.BsmtFinSF1) + (df.BsmtFinType2 * df.BsmtFinSF2)


# In[32]:


df3.sample()


# ### 3.4) Creating features using count

# In[33]:


df4 = pd.DataFrame()

Porches = ["WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]
df4["PorchTypes"] = df[Porches].gt(0.0).sum(axis=1)


# In[34]:


df4.sample()


# ### 3.5) Creating features using group transforms

# In[35]:


df5 = pd.DataFrame()
df5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")


# In[36]:


df5.sample()


# ### 3.6) Creating features using feature interactions

# In[37]:


df6 = pd.DataFrame()  # dataframe to hold new features

df6 = pd.get_dummies(df.BldgType, prefix="Bldg")
df6 = df6.mul(df.GrLivArea, axis=0)


# ### 3.7) Dropping Categorical Columns with high cordinality

# In[38]:


before = df.shape[1]

# Drop categorical columns with high cardinality (number of unique values in a column)
cat_columns_to_drop = [cname for cname in X.select_dtypes(["object","category","bool"]).columns
                    if X[cname].nunique() > 10]
df.drop(cat_columns_to_drop, axis=1, inplace=True)

after = df.shape[1]

print(f'Number of columns reduced from {before} to {after}')


# ### 3.8) Handling rare categorical values
# Credit: https://medium.com/gett-engineering/handling-rare-categorical-values-in-pandas-d1e3f17475f0

# In[39]:


cat_columns = list(df.select_dtypes('object').columns)
before = df[cat_columns].nunique().sum()


# In[40]:


# For categorical columns, some unique categories are occuring very rarely. 
# Let see an example of 'HouseStyle' column.
df.HouseStyle.value_counts(normalize=True)


# We can see from above that, few categories are having count less than 1% (arbitrarily selected threshold).

# In[41]:


for col in cat_columns:
    df[col]=df[col].mask(df[col].map(df[col].value_counts(normalize=True)) < 0.01, 'Other')


# By combining rare categories and categorising them in one category 'Other', number of categories will be decreased by around 30%.

# In[42]:


after = df[cat_columns].nunique().sum()
print(f'Number of unique categories reduced from {before} to {after}')


# In[43]:


df.HouseStyle.value_counts(normalize=True)*100


# ### 3.9) Nominative (unordered) categorical features
# For Tree based ML models, Label Encodeing can be used for categorical variables instead of OneHot Encoding. 

# In[44]:


df.dtypes.value_counts()


# In[45]:


# Note: `MSSubClass` feature is read as an `int` type, but is actually a (nominative) categorical.

features_nom = ["MSSubClass"] + cat_columns

# Cast each of the above 21 columns into 'category' DataType
for name in features_nom:
    df[name] = df[name].astype("category")
    
    # Add a None category for missing values
    if "NA" not in df[name].cat.categories:
        df[name] = df[name].cat.add_categories("NA")


# In[46]:


# Label encoding for categoricals
for colname in df.select_dtypes(["category"]):
    df[colname] = df[colname].cat.codes


# In[47]:


df.dtypes.value_counts() #'object' dtype converted into 'int8'


# ### Concat Created Features with Original Features

# In[48]:


df.shape


# In[49]:


df.drop(Years+Porches,axis=1, inplace=True)
df = pd.concat([df,df1,df2,df3,df4,df5,df6], axis=1)
df.sample()


# In[50]:


df.shape


# In[51]:


# Reform splits
X = df.loc[X.index, :]
X_test = df.loc[X_test.index, :]


# In[52]:


print(X.shape,X_test.shape,sep='\n')


# ## Imputing Numerical Columns

# In[53]:


my_imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value=0)
 
# Fitting the data to the imputer object
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

# Imputation removed column names and indices; put them back
imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns
imputed_X.index = X.index
imputed_X_test.index = X_test.index
 
# Using original names of DataSets
X = imputed_X
X_test = imputed_X_test


# In[54]:


show_null_values(X, X_test)


# # 4) Data Visualization<a id="Viz"></a>

# In[55]:


X_y = X.copy()
X_y['SalesPrice'] = y
X_y.sample()


# Now, we will create a function to automate plotting 3 types of plot for a single numerical variable.

# In[56]:


def univariate_numerical_plot(df, x):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
    
    # 0) histogram
    sns.histplot(data=df, x=x, kde=True, ax=ax[0], bins=min(df[x].nunique(),10), kde_kws={'bw_adjust':3})
    sns.despine(bottom=True, left=True)
    ax[0].set_title('histogram')
    ax[0].set_xlabel(xlabel=x)
    
    # 1) box plot
    sns.boxplot(data=df, x=x, ax=ax[1])
    ax[1].set_title('boxplot')
    ax[1].set_ylabel(ylabel=x)
    
    # 2) scatterplot
    sns.scatterplot(x=df[x], y=y, ax=ax[2], hue=y ,palette='coolwarm')
    plt.legend([],[], frameon=False)
    
    # To add border
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor('cornflowerblue')
    
    fig.tight_layout()
    fig.text(0.5, 1, f'Distribution of {x}', size=25, ha="center", va="center")
    plt.show()


# In[57]:


# Check distribution of target variable
univariate_numerical_plot(X_y,'SalesPrice')


# As many features are available for plotting, we won't plot all of them. We can focus on features having higher correaltion (which is calculated in the section below).

# ## Correlation of X with y

# In[58]:


def make_mi_scores(X, y):
    X = X.copy()
    # All discrete features should now have integer dtypes
    # discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X.select_dtypes('number'), y, random_state=0)
    mi_scores = pd.DataFrame(mi_scores.round(2), columns=["MI_Scores"], index=X.select_dtypes('number').columns)
    return mi_scores


# In[59]:


mi_scores = make_mi_scores(X, y)
linear_corr = pd.DataFrame(X.corrwith(y).round(2), columns=['Lin_Correlation'])

corr_with_price = pd.concat([mi_scores, linear_corr], axis=1)
corr_with_price = corr_with_price.sort_values('MI_Scores',ascending=False)

corr_with_price


# ## Distribution of top 5 features correlated with Sales Price

# In[60]:


top_features = corr_with_price.index[1:6]


# In[61]:


for feature in top_features:
    univariate_numerical_plot(X,feature)


# # 5) Feature Selection<a id="Select"></a>

# In[62]:


before = X.shape[1]
X.dtypes.value_counts()


# In[63]:


# Numerical columns with large correlation with Sales Price
threshold = 0.01
numerical_cols = [cname for cname in X.select_dtypes('number').columns
                  if corr_with_price.MI_Scores[cname] > threshold]

# Keep selected columns only
selected_cols = numerical_cols
X = X[selected_cols]
X_test = X_test[selected_cols]
after = X.shape[1]

# Selected Features for Model Training or Fitting
print(f'Out of {before} features, {after} fetures are having MI_Scores more than {threshold}.')


# In[64]:


# To see which columns were selected according to min correlation condition
info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info.sort_values(['Dtype', 'Unique'])


# # 6) Model Creation<a id="Model"></a>

# In[65]:


# Create object of class XGBRegressor
xgb = XGBRegressor(eval_metric='rmse')


# ## Grid Search & Cross Validation

# In[66]:


param_grid = [
    {'subsample': [0.5], 'n_estimators': [1400], 
     'max_depth': [5], 'learning_rate': [0.02],
     'colsample_bytree': [0.4], 'colsample_bylevel': [0.5],
     'reg_alpha':[1], 'reg_lambda': [1], 'min_child_weight':[2]}
]
grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=1, scoring='neg_root_mean_squared_error')


# # 7) Training and Testing Model<a id="Test"></a>

# In[67]:


grid_search.fit(X, np.log(y));


# In[68]:


# Top 5 hyper-parameter combinations
cv_results = pd.DataFrame(grid_search.cv_results_)
display(cv_results.sort_values('rank_test_score')[:6])


# ## Best Parameters:

# In[69]:


grid_search.best_params_


# ## Best score in grid search

# **Public leader board will have little less RMSLE than that of on validation data**, because below error is on validation data when model is trained/fitted on 67% of the training data because of 3-fold cross validation.
# Public leaderboard will have error on test data when model is retrained/refitted on 100% of the training data.

# In[70]:


print("RMSLE on training data:",round(-grid_search.score(X, np.log(y)),4))
print("RMSLE on validation data:",round(-grid_search.best_score_,4))


# ## To avoid overfitting:
# Difference in the performance (Root Mean Squared Log Error) of model on training data and validation data should be minimized.

# ## Feature Importance

# In[71]:


Feature_Imp = grid_search.best_estimator_.feature_importances_
Feature_Imp_sorted_series = pd.Series(Feature_Imp,X.columns).sort_values(ascending=True)

# Plot horizaontal bar plot
plt.figure(figsize=(14,20), dpi=120)
palette = sns.color_palette("coolwarm", len(X.columns)).as_hex()
ax = Feature_Imp_sorted_series.plot.barh(width=0.8 ,color=palette)
ax.set_title('Feature Importance in XgBoost')
plt.show()


# ## Generate test predictions

# In[72]:


y_preds = np.exp(grid_search.predict(X_test))


# ## Save output to CSV file

# In[73]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': y_preds.round()})
output.to_csv('submission.csv', index=False)


# ## Check output format and submit results

# In[74]:


output.sample(2)


# ## **Reference Kaggle courses:**
# * [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)
# * [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

# **Note:**
# 
# * This is the copy of my notebook "[Housing Prices (XGBoost, GridSearch, Pipeline)](https://www.kaggle.com/code/maheshnanavare/housing-prices-xgboost-gridsearch-pipeline)" from similar compitition "[Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)".
# * I changed evaluation method for XGBoost from 'mean absolute error' to '*root mean squared log error*'. Used *log(y)* instead of y.
# * I also updated, added some things and removed pipeline in this notebook.
