#!/usr/bin/env python
# coding: utf-8

# # **Housing Prices Competition**

# ## Contents:
# 1. **X, y Dataframes Creation**
#     * Import Libraries
#     * Read the data
# 2. **X, y Summary**
#     * Correlation between features and target on heatmap
# 3. **Data Cleaning**
#     * Remove the columns with more than half missing values
#     * Drop columns with most of the rows having only one category
# 4. **Feature Engineering**
#     * Creating feature for 'Age' of house when sold
#     * Adding features 'Garage', 'Fireplace' and 'Basement'
#     * Replacing ordered categories with numbers
# 5. **Data Visualization**
#     * Distribution of top 5 features correlated with Sales Price
# 6. **Feature Selection**
#     * Selected Features
# 7. **Model Creation**
#     * Preprocessing
#     * Visualize Pipeline
#     * Grid Search
# 8. **Training and Testing Model**
#     * Best Parameters

# # 1) X, y Dataframes Creation

# ## Import Libraries

# In[1]:


# Data Analytics Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype


# In[2]:


# Machine Learning Libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from math import ceil


# In[3]:


# Update some default parameters for plotting throughout the notebook
plt.rcParams.update({'font.size': 12, 'xtick.labelsize':15, 'ytick.labelsize':15, 'axes.labelsize':15, 'axes.titlesize':20})


# In[4]:


# List all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


# Read the data
Xy = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target
Xy = Xy.dropna(axis=0, subset=['SalePrice'])

# Separate target from predictors
X = Xy.drop(['SalePrice'], axis=1)
y = Xy.SalePrice


# # 2) X, y Summary

# In[6]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)


# In[7]:


info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info


# In[8]:


X.dtypes.value_counts()


# In[9]:


y.describe()


# ## Correlation between features and target on heatmap

# In[10]:


correlation_matrix = Xy.corr()

# Returns copy of array with upper part of the triangle (which will be masked/hidden)
mask = np.triu(correlation_matrix.corr())

sns.set(font_scale=1.1)
plt.figure(figsize=(20, 20), dpi=140)
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm', 
            square=True, mask=mask, linewidths=1, cbar=False)
plt.show()


# # 3) Data Cleaning

# In[11]:


# Total rows/observations/houses in Training data and Test data
print(len(X),len(X_test))


# ## Remove the columns with more than half missing values

# In[12]:


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


# In[13]:


show_null_values(X, X_test)


# In[14]:


# Columns with missing values in more than half number of rows
null_cols = [col for col in X.columns if X[col].isnull().sum() > len(X)/2]
null_cols


# In[15]:


X.drop(null_cols,axis=1,inplace=True)
X_test.drop(null_cols,axis=1,inplace=True)


# In[16]:


# Total missing values after removing columns with more than half missing values
print("Total missing values:")
print("Training data\t",X.isnull().sum().sum())
print("Testing data\t",X_test.isnull().sum().sum())


# ## Data Visualization (Categorical Data)

# In[17]:


object_cols = X.select_dtypes('object').columns
len(object_cols)


# In[18]:


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

# In[19]:


# From above violin plots, 'Utilities' feature seems to have mostly one category.
# Lets confirm that using value_counts for each of its categories.
X.Utilities.value_counts()


# In[20]:


X_test.Utilities.value_counts()


# In[21]:


X.drop('Utilities',axis=1,inplace=True)
X_test.drop('Utilities',axis=1,inplace=True)


# # 4) Feature Engineering

# In[22]:


# Merge the datasets so we can process them together
df = pd.concat([X, X_test])


# ## Ordered Categorical Columns

# ### Replacing ordered categories with numbers

# In[23]:


object_cols = df.select_dtypes(include=['object']).columns
# Categorical Columns with number of unuque categoies in them 
df[object_cols].nunique().sort_values()


# Read 'data_description.txt' from the input files. Some columns were having **levels of quality, condition or finish** in words (i.e. string format). These columns can be used as ordinal columns (i.e. ordered category columns).

# In[24]:


ordinal_cols = [i for i in object_cols if ('QC' in i) or ('Qu' in i) or ('Fin' in i) or ('Cond' in i) and ('Condition' not in i)]
print("Column Names: [Unique Categories in each column]")
{col:[*df[col].unique()] for col in ordinal_cols}


# In[25]:


df.loc[:,ordinal_cols] = df.loc[:,ordinal_cols].fillna('NA')


# In[26]:


df.Functional.isnull().sum()


# In[27]:


# 1] Columns with similar ordered categories [Poor<Fair<Typical/Average<Good<Excellent]
ordinal_cols1 = [i for i in object_cols if ('QC' in i) or ('Qu' in i) or ('Cond' in i) and ('Condition' not in i)]
df.loc[:,ordinal_cols1] = df.loc[:,ordinal_cols1].replace(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0,1,2,3,4,5])

# 2] Columns with similar ordered categories [No Garage/Basement<Unfinished<Rough Finished<Finished,etc]
ordinal_cols2 = ['BsmtFinType1', 'BsmtFinType2']
df.loc[:,ordinal_cols2] = df.loc[:,ordinal_cols2].replace(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [0,1,2,3,4,5,6])

# 3] Column with ordered categories [No Basement<No Exposure<Mimimum Exposure<Average Exposure<Good Exposure]
ordinal_cols3 = ['BsmtExposure']
df.loc[:,ordinal_cols3] = df.loc[:,ordinal_cols3].replace(['NA', 'No', 'Mn', 'Av', 'Gd'], [0,1,2,3,4])

# 4] Column with ordered categories [Regular<Slightly irregular<Moderately Irregular<Irregular]
ordinal_cols4 = ['LotShape']
df.loc[:,ordinal_cols4] = df.loc[:,ordinal_cols4].replace(['Reg', 'IR1', 'IR2', 'IR3'], [0,1,2,3])

# 5] Column with ordered categories [No Garage<Unfinished<Rough Finished<Finished]
ordinal_cols5 = ['GarageFinish']
df.loc[:,ordinal_cols5] = df.loc[:,ordinal_cols5].replace(['NA', 'Unf', 'RFn', 'Fin'], [0,1,2,3])

# 6] Home functionality Column
ordinal_cols6 = ['Functional']
df.loc[:,ordinal_cols5] = df.loc[:,ordinal_cols5].replace(["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], list(range(8)))


# ## 4.1) Creating New Numerical Features

# In[28]:


df1 = pd.DataFrame()  # dataframe to hold new features

# Age of House when sold
df1['Age'] = df['YrSold']-df['YearBuilt']

# Years between Remodeling and sales
df1['AgeRemodel'] = df['YrSold']-df['YearRemodAdd']

year_cols = ['YrSold','YearBuilt','AgeRemodel', 'Age']
df_1 = pd.concat([df, df1], axis=1).loc[:,year_cols]
X_1 = df_1.loc[X.index, :]
X_1.head(2)


# In[29]:


fig, ax = plt.subplots(1, 4, figsize=(20, 6), dpi=100)

# scatterplot
for col,i in zip(year_cols, [0,1,2,3]):
    sns.scatterplot(x=X_1.loc[:,col], y=y, ax=ax[i])

fig.tight_layout()
fig.text(0.5, 1, 'Distribution of SalesPrice with respect to years columns', size=25, ha="center", va="center")
plt.show()


# In[30]:


# Correlation of year columns with SalePrice
X_1.corrwith(y)


# ## 4.2) Creating New Boolean Features
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

# In[31]:


df2 = pd.DataFrame()  # dataframe to hold new features

df2['Remodel'] = df['YearRemodAdd']!=df['YearBuilt']
df2['Garage'] = df['GarageQual'].notnull()
df2['Fireplace'] = df['FireplaceQu'].notnull()
df2['Bsmt'] = df['BsmtQual'].notnull()
df2['Masonry'] = df['MasVnrType'].notnull()

# Converting boolean columns [False,True] into numerical columns [0,1]
df2 = df2.replace([False,True], [0,1])
df2.head(2)


# ## 4.3) Creating features using mathematical transformations

# In[32]:


df3 = pd.DataFrame()  # dataframe to hold new features

df3["Liv_Qual"] = (df.OverallQual + df.OverallCond/2) * df.GrLivArea
# df4["LivLotRatio"] = df.GrLivArea / df.LotArea
df3["Spaciousness"] = (df['1stFlrSF'] + df['2ndFlrSF']) / df.TotRmsAbvGrd
df3["GarageQual_Area"] = (df.GarageQual + df.GarageCond) * df.GarageArea
# df3["Garage_Spaciousness"] = df.GarageArea / (df.GarageCars+1)
df3["BsmtQual_SF"] = ((df.BsmtQual + df.BsmtCond/2 + df.BsmtExposure/3) * df.TotalBsmtSF) + (df.BsmtFinType1 * df.BsmtFinSF1) + (df.BsmtFinType2 * df.BsmtFinSF2)


# In[33]:


df3.head(2)


# ### 4.4) Creating features using count

# In[34]:


df4 = pd.DataFrame()

df4["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch"]].gt(0.0).sum(axis=1)


# In[35]:


df4.head(2)


# ### 4.5) Creating features using group transforms

# In[36]:


df5 = pd.DataFrame()
df5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")


# In[37]:


df5.head(2)


# ### 4.6) Creating features using feature interactions

# In[38]:


df6 = pd.DataFrame()  # dataframe to hold new features

df6 = pd.get_dummies(df.BldgType, prefix="Bldg")
df6 = df6.mul(df.GrLivArea, axis=0)


# ### Concat Created Features with Original Features

# In[39]:


df.shape


# In[40]:


df = pd.concat([df,df1,df2,df3,df4,df5,df6], axis=1)
df.head(2)


# In[41]:


df.shape


# In[42]:


# Reform splits
X = df.loc[X.index, :]
X_test = df.loc[X_test.index, :]


# In[43]:


print(X.shape,X_test.shape,sep='\n')


# # 5) Data Visualization

# In[44]:


X_y = X.copy()
X_y['SalesPrice'] = y
X_y.head(2)


# Now, we will create a function to automate plotting 4 types of plot for a single numerical variable.

# In[45]:


def univariate_numerical_plot(df, x):
    fig, ax = plt.subplots(1, 4, figsize=(20, 6), dpi=100)
    
    # 0) histogram
    sns.histplot(data=df, x=x, kde=True, ax=ax[0], bins=min(df[x].nunique(),10), kde_kws={'bw_adjust':3})
    sns.despine(bottom=True, left=True)
    ax[0].set_title('histogram')
    ax[0].set_xlabel(xlabel=x)
    
    # 1) box plot
    sns.boxplot(data=df, y=x, ax=ax[1])
    ax[1].set_title('boxplot')
    ax[1].set_ylabel(ylabel=x)
    
    # 2) probability plot
    plt.sca(ax[2])
    stats.probplot(df[x], dist = "norm", plot = plt)
    
    # 3) scatterplot
    sns.scatterplot(x=df[x], y=y, ax=ax[3])
    
    fig.tight_layout()
    fig.text(0.5, 1, f'Distribution of {x}', size=25, ha="center", va="center")
    plt.show()


# In[46]:


# Check distribution of target variable
univariate_numerical_plot(X_y,'SalesPrice')


# As many features are available for plotting, we won't plot all of them. We can focus on features having higher linear correaltion (which is calculated in the section below).

# ## Correlation of X with y

# In[47]:


corr_s = X_y.corr(method='spearman')
corr_with_price = pd.DataFrame(corr_s['SalesPrice'])

corr_with_price.rename(columns = {'SalesPrice':'spearman'}, inplace = True)

corr_p = X_y.corr(method='pearson')
corr_with_price['pearson']=corr_p['SalesPrice']

corr_with_price = corr_with_price.sort_values('pearson',ascending=False)
round(corr_with_price,2)


# ## Distribution of top 5 features correlated with Sales Price

# In[48]:


top_features = corr_with_price.index[1:6]


# In[49]:


for feature in top_features:
    univariate_numerical_plot(X,feature)


# # 6) Feature Selection

# In[50]:


X.dtypes.value_counts()


# In[51]:


# Categorical columns with low cardinality (number of unique values in a column)
categorical_cols = [cname for cname in X.select_dtypes(["object","category","bool"]).columns
                    if X[cname].nunique() < 30]

# Numerical columns with large correlation with Sales Price
numerical_cols = [cname for cname in X.select_dtypes(['int64', 'float64']).columns
                  if abs(corr_with_price.pearson[cname]) > 0]

# Keep selected columns only
selected_cols = categorical_cols + numerical_cols
X = X[selected_cols]
X_test = X_test[selected_cols]


# ## Selected Features

# In[52]:


# Out of 79 available feature, following number of features will be used for regression
print(len(categorical_cols),len(numerical_cols))


# In[53]:


X.dtypes.value_counts()


# In[54]:


# To see which object columns were selected according to max cardinality condition 
# and which numerical columns were selected according to min correlation condition
info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info.sort_values(['Dtype', 'Unique'])


# In[55]:


show_null_values(X, X_test)


# Note: These null values will be imputed by preprocessor in next section

# # 7) Model Creation

# ## Preprocessing

# In[56]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[57]:


# Create object of class XGBRegressor
xgb = XGBRegressor()

# Bundle preprocessing and modeling code in a pipeline
regressor = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', xgb)
                     ])


# ## Visualize Pipeline

# In[58]:


from sklearn import set_config
set_config(display='diagram')
regressor


# ## Grid Search & Cross Validation

# In[59]:


param_grid = [
    {'model__subsample': [0.5], 'model__n_estimators': [1400], 
     'model__max_depth': [5], 'model__learning_rate': [0.02],
     'model__colsample_bytree': [0.4], 'model__colsample_bylevel': [0.5],
     'model__reg_alpha':[2], 'model__reg_lambda': [1]}
]
grid_search = GridSearchCV(regressor, param_grid, cv=3, verbose=1, scoring='neg_mean_absolute_error')


# # 8) Training and Testing Model

# In[60]:


grid_search.fit(X, y);


# In[61]:


pd.DataFrame(grid_search.cv_results_)


# ## Best Parameters:

# In[62]:


grid_search.best_params_


# ## Best score in grid search

# **Public leader board will have little less Mean Absolute Error than below**, because below error is on validation data when model is trained/fitted on 80% of the training data because of 5-fold cross validation.
# Public leaderboard will have error on test data when model is retrained/refitted on 100% of the training data.

# In[63]:


print("Mean Absolute Error on validation data:",-grid_search.best_score_)


# ## Perforamnce Evaluation on Training data

# In[64]:


y_train_preds = grid_search.predict(X)


# Following scores are regarding training data. So, not much useful.

# In[65]:


print("Best XGBoost on whole trained data:")
print("Mean Absolute Error:",-grid_search.score(X, y))
RMSE = mean_squared_error(y, y_train_preds, squared=False)
print('Root Mean Squared Error:',round(RMSE))
r2 = r2_score(y, y_train_preds)
print('R² or the coefficient of determination:',round(r2,3))


# ## Generate test predictions

# In[66]:


y_preds = grid_search.predict(X_test)


# ## Save output to CSV file

# In[67]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': y_preds.round()})
output.to_csv('submission.csv', index=False)


# ## Check output format and submit results

# In[68]:


output.head(2)

