#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing Libraries and Datasets

# ### 1.1 Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.preprocessing import OrdinalEncoder
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


# ### 1.2 Loading Datasets

# In[2]:


df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df_train.head()


# We can see some missing values in this dataset. Let's explore the dataset to handle this.

# ## 2. Feature Engineering

# Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used to create a predictive model using machine learning or statistical modeling. 

# ### 2.1 Missing Value Handling

# #### 2.1.1 Checking Missing Values

# In[3]:


train_null = df_train.isna().sum()
test_null = df_test.isna().sum()
missing = pd.DataFrame(
              data=[train_null, train_null/df_train.shape[0]*100,
                    test_null, test_null/df_test.shape[0]*100],
              columns=df_train.columns,
              index=["Train Null", "Train Null (%)", "Test Null", "Test Null (%)"]
          ).T.sort_values(["Train Null", "Test Null"], ascending=False)

# Filter only columns with missing values
missing = missing.loc[(missing["Train Null"] > 0) | (missing["Test Null"] > 0)]
missing.style.background_gradient('summer_r')


# Wow, there are a lot of missing values in this dataset. The missing values are not only found in the training data, but also in the test data. 

# In[4]:


# Plot variables with more than 5% missing values
missing.loc[missing["Train Null (%)"] > 5, ["Train Null (%)", "Test Null (%)"]].iloc[::-1].plot.barh(figsize=(8,6))
plt.title("Variables With More Than 5% Missing Values")
plt.show()


# As we can see, we have 11 variables with more than 5% of missing values in it. These variables are PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage, GarageYrBlt, GarageFinish, GarageQual, GarageCond, and GarageType. The ratio of missing values for each variable in the train data and test data looks quite balanced. We will handle this problem. First, let's group each variable with missing values based on their data type.

# #### 2.1.2 Grouping Variable with Missing Values

# In[5]:


df_missing = df_train[missing.index]
missing_cat = df_missing.loc[:, df_missing.dtypes == "object"].columns
missing_num = df_missing.loc[:, df_missing.dtypes != "object"].columns

print(f"number of categorical variables with missing values: {len(missing_cat)}")
print(f"number of numerical variables with missing values: {len(missing_num)}")


# As we can see, we have 23 categorical variables and 11 numerical variables with missing values. Now let's check their distribution.

# #### 2.1.3 Checking The Distribution of Variables with Missing Values

# #### Categorical Variables

# In[6]:


fig, ax = plt.subplots(6, 4, figsize=(20, 24))
ax = ax.flatten()

for i, var in enumerate(missing_cat):
    count = sns.countplot(data=df_train, x=var, ax=ax[i])
    for bar in count.patches:
        count.annotate(format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2,
            bar.get_height()), ha='center', va='center',
            size=11, xytext=(0, 8),
            textcoords='offset points')
        
    ax[i].set_title(f"{var} Distribution")
    if df_train[var].nunique() > 6:
        ax[i].tick_params(axis='x', rotation=45)
    
plt.subplots_adjust(hspace=0.5)
plt.show()


# #### Numerical Variables

# In[7]:


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
ax = ax.flatten()

for i, var in enumerate(missing_num):
    sns.histplot(data=df_train, x=var, kde=True, ax=ax[i])
    ax[i].set_title(f"{var} Distribution")

plt.subplots_adjust(hspace=0.5)
plt.show()


# #### 2.1.4 Filling Missing Values

# The most important thing to do to be able to handle missing values in this dataset is understanding the description of each variable first. To do this, we can read **data_description.txt** file provided in this competition. 

# #### Categorical

# After reading data description file, i decided to fill in the missing values as follows:
# 1. Fill with **None**: PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageFinish, GarageQual, GarageCond, GarageType
# 2. Fill with **NB** (No Basement): BsmtExposure, BsmtFinType2, BsmtCond, BsmtQual, BsmtFinType1
# 3. Fill with **Mode**: Electrical, Functional, KitchenQual, Exterior1st, Exterior2nd, MSZoning, SaleType, MasVnrType
# 
# For Utilities variable, we will just remove it because this variable does not provide information at all. As we can see in the bar chart above, this variable is only dominated by 1 value.

# In[8]:


cat_none_var = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "GarageType"]
cat_nb_var = ["BsmtExposure", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtFinType1"]
cat_mode_var = ["Electrical", "Functional", "KitchenQual", "Exterior1st", "Exterior2nd", "MSZoning", "SaleType", "MasVnrType"]
cat_mode = {var: df_train[var].mode()[0] for var in cat_mode_var}

# Categorical
for df in [df_train, df_test]:
    # Fill with None (because null means no building/object)
    df[cat_none_var] = df[cat_none_var].fillna("None")
    
    # Fill with NB (because null means no basememt)
    df[cat_nb_var] = df[cat_nb_var].fillna("NB")
    
    # Fill other categorical variables with mode
    for var in cat_mode_var:
        df[var].fillna(cat_mode[var], inplace=True)
        
    # Drop variable because no information
    df.drop("Utilities", axis=1, inplace=True)

missing_cat = missing_cat.drop("Utilities")
print(f"Categorical variable missing values in train data: {df_train[missing_cat].isna().sum().sum()}")
print(f"Categorical variable missing values in test data: {df_test[missing_cat].isna().sum().sum()}")


# #### Numerical

# For numerical variables, we will just fill the missing values with 0 because most of this variables are related to buildings or objects which may not exist. But for LotFrontage variable, we will fill the missing values with the average value of the variable based on the type of road access to property (Street).

# In[9]:


mean_LF = df_train.groupby("Street")["LotFrontage"].mean()
num_zero_var = missing_num.drop("LotFrontage")

for df in [df_train, df_test]:
    df.loc[(df["LotFrontage"].isna()) & (df["Street"] == "Grvl"), "LotFrontage"] = mean_LF["Grvl"]
    df.loc[(df["LotFrontage"].isna()) & (df["Street"] == "Pave"), "LotFrontage"] = mean_LF["Pave"]
    
    for var in num_zero_var:
        df[var].fillna(0, inplace=True)
        
print(f"Numerical variable missing values in train data: {df_train[missing_num].isna().sum().sum()}")
print(f"Numerical variable missing values in test data: {df_test[missing_num].isna().sum().sum()}")


# Done. We don't have any missing values now.

# ### 2.2 Categorical Variable Encoding

# Categorical variables need to be converted into numerical format so that the data with converted categorical values can be provided to the predictive models. There are two types of categorical variable, which is nominal and ordinal. Nominal variable has no intrinsic ordering to its categories, while ordinal variable has a clear ordering. There are different ways to encode these variables depending on their data type. Let's check our categorical variable values first.

# In[10]:


# Get numerical and categorical variable
cat_var = df_train.loc[:, df_train.dtypes == "object"].nunique() # Get variable names and number of unique values
num_var = df_train.loc[:, df_train.dtypes != "object"].columns # Get variable names


# #### 2.2.1 Get Unique Values For Each Categorical Variable 

# In[11]:


# Count sorted unique values (alphabetically) for each categorical variable
cat_var_unique = {var: sorted(df_train[var].unique()) for var in cat_var.index}

# Add "-" for each values to replace none in the DataFrame (25 is highest len of unique values)
for key, val in cat_var_unique.items():
    cat_var_unique[key] += ["-" for x in range(25-len(val))]

pd.DataFrame.from_dict(cat_var_unique, orient="index").sort_values([x for x in range(25)])


# From the table above, we can see the unique values of each categorical variable in this dataset. Now we can determine which variables have nominal values and which variables have ordinal values, and then decide what method we will use to encode that variable.

# #### 2.2.2 Ordinal Encoding

# For ordinal variables, we will use ordinal encoding to encode these variables. Ordinal encoding converts each label into integer values and the encoded data represents the sequence of labels. We will need to group each ordinal variable based on their unique values.

# In[12]:


ord_var1 = ["ExterCond", "HeatingQC"]
ord_var1_cat = ["Po", "Fa", "TA", "Gd", "Ex"]

ord_var2 = ["ExterQual", "KitchenQual"]
ord_var2_cat = ["Fa", "TA", "Gd", "Ex"]

ord_var3 = ["FireplaceQu", "GarageQual", "GarageCond"]
ord_var3_cat = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

ord_var4 = ["BsmtQual"]
ord_var4_cat = ["NB", "Fa", "TA", "Gd", "Ex"]

ord_var5 = ["BsmtCond"]
ord_var5_cat = ["NB", "Po", "Fa", "TA", "Gd"]

ord_var6 = ["BsmtExposure"]
ord_var6_cat = ["NB", "No", "Mn", "Av", "Gd"]

ord_var7 = ["BsmtFinType1", "BsmtFinType2"]
ord_var7_cat = ["NB", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]

# Put all in one array for easier iteration
ord_var = [ord_var1, ord_var2, ord_var3, ord_var4, ord_var5, ord_var6, ord_var7]
ord_var_cat = [ord_var1_cat, ord_var2_cat, ord_var3_cat, ord_var4_cat, ord_var5_cat, ord_var6_cat, ord_var7_cat]
ord_all = ord_var1 + ord_var2 + ord_var3 + ord_var4 + ord_var5 + ord_var6 + ord_var7 

for i in range(len(ord_var)):
    enc = OrdinalEncoder(categories=[ord_var_cat[i]])
    for var in ord_var[i]:
        df_train[var] = enc.fit_transform(df_train[[var]])
        df_test[var] = enc.transform(df_test[[var]])


# This is the result.

# In[13]:


df_train[ord_all]


# #### 2.2.3 One-Hot Encoding

# Next we will encode the nominal variable. For this type of variable, we will use one-hot encoding. With one-hot encoding, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. We will only use this method on variables with a number of unique values less than 6.

# In[14]:


cat_var = cat_var.drop(ord_all)
onehot_var = cat_var[cat_var < 6].index

df_train = pd.get_dummies(df_train, prefix=onehot_var, columns=onehot_var)
df_test = pd.get_dummies(df_test, prefix=onehot_var, columns=onehot_var)


# In[15]:


# Get encoded variables name that do not yet exist in the test data
add_var = [var for var in df_train.columns if var not in df_test.columns]

# Add new columns in the test data with value of 0
for var in add_var:
    if var != "SalePrice":
        df_test[var] = 0

# Reorder test data column so it is the same order as the train data
df_test = df_test[df_train.columns.drop("SalePrice")]


# #### 2.2.4 Target Encoding

# The problem with One-Hot Encoding is that the more unique values in a variable, the more new columns will be created. In that case, it can lead to high memory consumption and increase the computational cost. Therefore, we will use target encoding for variables with 6 or more unique values. 
# 
# Actually, we missed something before. There is a variable which is actually categorical, but that variable in this dataset has a numerical value (int). That variable is **MoSold**, which contains the value of what month the house was sold. So, we will add that variable when we do the target encoding

# In[16]:


cat_var = cat_var.drop(onehot_var)
X_train = df_train.drop("SalePrice", axis=1)
y_train = df_train["SalePrice"]

te = MEstimateEncoder(cols=df_train[cat_var.index.append(pd.Index(["MoSold"]))]) # Add MoSold variable to the encoder
X_train = te.fit_transform(X_train, y_train)
df_test = te.transform(df_test)

df_train = pd.concat([X_train, y_train], axis=1)


# ### 2.3 Creating and Deleting Variables

# In this process, we will create some new variables from existing data. We need to do this mainly because we don't want any multicollinearity in our data. Multicollinearity happens when independent variables in the regression model are highly correlated to each other. It makes it hard to interpret of model and also creates an overfitting problem. Another reason we want to do this is because we can create new features that might be useful for predicting targets.

# #### 2.3.1 Checking Correlation

# Let's check the correlation between variables in our data first.

# In[17]:


plt.figure(figsize=(12,12))
sns.heatmap(df_train[num_var].drop("Id", axis=1).corr())
plt.show()


# As we can see, some variables are highly correlated with other variables. Brighter color means they have positive correlation, meaning both variables move in the same direction, while darker color means they have negative correlation, meaning that when one variable's value increases, the other variables' values decrease.
# 
# We can't clearly see the correlation in the graph above because there are too many variables, so let's sort by the highest correlation. We will only display variables with more than 0.5 correlation coefficient (including negative correlation).

# In[18]:


df_train_corr = df_train[num_var].corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

high_corr = df_train_corr['Correlation Coefficient'] > 0.5
df_train_corr[high_corr].reset_index(drop=True).style.background_gradient('summer_r')


# There we go. Now let's also check which variables have a high correlation with the target.

# In[19]:


df_train_corr[high_corr].loc[(df_train_corr["Feature 1"]=="SalePrice") | (df_train_corr["Feature 2"]=="SalePrice")].reset_index(drop=True).style.background_gradient('summer_r')


# #### 2.3.2 Creating New Variables

# Now we will create some new variables based on variables that have a high correlation in the data above to avoid collinearity. We will also create new features that might be useful.

# In[20]:


for df in [df_train, df_test]:
    df["GarAreaPerCar"] = (df["GarageArea"] / df["GarageCars"]).fillna(0)
    df["GrLivAreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"]
    df["TotalHouseSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalFullBath"] = df["FullBath"] + df["BsmtFullBath"]
    df["TotalHalfBath"] = df["HalfBath"] + df["BsmtHalfBath"]
    df["InitHouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodHouseAge"] = df["InitHouseAge"] - (df["YrSold"] - df["YearRemodAdd"])
    df["IsRemod"] = (df["YearRemodAdd"] - df["YearBuilt"]).apply(lambda x: 1 if x > 0 else 0)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).apply(lambda x: 0 if x > 2000 else x)
    df["IsGarage"] = df["GarageYrBlt"].apply(lambda x: 1 if x > 0 else 0)
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df["AvgQualCond"] = (df["OverallQual"] + df["OverallCond"]) / 2


# #### 2.3.3 Deleting Variables

# We are going to remove some variables since we no longer need them. This will also reduce memory consumption because the features we will use are also reduced.

# In[21]:


for df in [df_train, df_test]:
    df.drop([
        "GarageArea", "GarageCars", "GrLivArea", 
        "TotRmsAbvGrd", "TotalBsmtSF", "1stFlrSF", 
        "2ndFlrSF", "FullBath", "BsmtFullBath", "HalfBath", 
        "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd",
        "GarageYrBlt", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "OverallQual", "OverallCond"
    ], axis=1, inplace=True)


# ## 3. Model Building

# #### 3.1 Splitting Dataset

# In[22]:


X_train = df_train.drop(["Id", "SalePrice"], axis=1)
y_train = df_train.SalePrice

X_test = df_test.drop("Id", axis=1)


# #### 3.2 Feature Scaling

# In[23]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_log = np.log10(y_train)


# #### 3.3 Selecting Best Model

# In[24]:


regressors = {
    "XGB Regressor": XGBRegressor(),
    "LGBM Regressor": LGBMRegressor(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "SVR": SVR(),
    "GB Regressor": GradientBoostingRegressor(random_state=0)
}

results = pd.DataFrame(columns=["Regressor", "Avg_RMSE"])
for name, reg in regressors.items():
    model = reg
    cv_results = cross_validate(
        model, X_train_scaled, y_train_log, cv=10,
        scoring=(['neg_root_mean_squared_error'])
    )

    results = results.append({
        "Regressor": name,
        "Avg_RMSE": np.abs(cv_results['test_neg_root_mean_squared_error']).mean()
    }, ignore_index=True)

results = results.sort_values("Avg_RMSE", ascending=True)
results.reset_index(drop=True)


# In[25]:


plt.figure(figsize=(12, 6))
sns.barplot(data=results, x="Avg_RMSE", y="Regressor")
plt.title("Average RMSE CV Score")
plt.show()


# We will use 3 best algorithms from the cross validation test above, which is GB Regressor, LGBM Regressor, and XGB Regressor. Let's tune their hyperparameters using Grid Search Cross Validation.

# #### 3.4 Hyperparameter Tuning

# #### Gradient Boosting Regressor

# In[26]:


gbr = GradientBoostingRegressor(random_state=0)
params = {
    "loss": ("squared_error", "absolute_error"),
    "learning_rate": (1.0, 0.1, 0.01),
    "n_estimators": (50, 100, 200)
}
reg1 = GridSearchCV(gbr, params, cv=10)
reg1.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg1.best_params_)


# In[27]:


y_pred = reg1.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")


# #### XGB Regressor

# In[28]:


xgb = XGBRegressor(random_state=0)
params = {
    "max_depth": (3, 6, 9),
    "learning_rate": (0.3, 0.1, 0.05),
    "n_estimators": (50, 100, 200)
}
reg2 = GridSearchCV(xgb, params, cv=10)
reg2.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg2.best_params_)


# In[29]:


y_pred = reg2.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")


# #### LGBM Regressor

# In[30]:


lgbm = LGBMRegressor(random_state=0)
params = {
    "num_leaves": (11, 31, 51),
    "learning_rate": (0.5, 0.1, 0.05),
    "n_estimators": (50, 100, 200)
}
reg3 = GridSearchCV(lgbm, params, cv=10)
reg3.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg3.best_params_)


# In[31]:


y_pred = reg3.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")


# #### 3.5 Stacking 3 Regressors

# In[32]:


def sm_predict(X):
    return (3 * reg1.predict(X) + 2 * reg2.predict(X) + 5 * reg3.predict(X)) / 10

y_pred_stack = sm_predict(X_train_scaled)
print(f"Train RMSE with Stacking: {mean_squared_error(y_train_log, y_pred_stack, squared=False)}")


# ## Making Submission

# In[33]:


y_pred = sm_predict(X_test_scaled)
y_pred_inv = 10 ** y_pred

submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_pred_inv})
submission.to_csv('submission.csv', index=False)


# In[ ]:




