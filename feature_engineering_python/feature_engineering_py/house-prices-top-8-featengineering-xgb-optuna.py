#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# This notebook outlines the step-by-step process of creating a house price prediction model—it includes data pre-processing, feature engineering, model training, hyperparameter tuning, and model explainability. The prediction model generated currently ranks in the top 8% of Kaggle's [House Price Prediction Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) leaderboard and top 1% of [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course) leaderboard (as of 10/29/2021).
# 
# If you're interested in learning more about model deployment (as an interactive [web app](https://share.streamlit.io/ruthgn/ames-housing-price-prediction/main/ames-house-ml-app.py)), check out [this notebook](https://www.kaggle.com/ruthgn/top-1-model-interpretation-deployment).

# # Part 1 - Preliminaries

# ## Imports and Configuration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
import optuna
import shap
import pickle

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from pathlib import Path


# ## Data Preprocessing

# In[2]:


def load_data():
    # Read data
    data_dir = Path("../input/house-prices-advanced-regression-techniques/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    # Preprocessing steps
    df = clean(df)
    df = encode(df)
    df = impute_plus(df)
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test


# ### Clean Data

# A closer look at the dataset make it clear that there are categorical features with typos in the categories:

# In[3]:


data_dir = Path("../input/house-prices-advanced-regression-techniques/")
df = pd.read_csv(data_dir / "train.csv", index_col="Id")

df.Exterior2nd.unique()


# We will create a function to make corrections on several detected issues within the dataset:

# In[4]:


def clean(df):
    # Correct typo on Exterior2nd
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
    # Some values of GarageYrBlt are corrupt, so we'll replace them with the year house was built
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    # Name beginning with numbers are awkward to work with
    df.rename(columns={
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF',
        '3SsnPorch': 'Threeseasonporch'
        }, inplace=True)
    return df


# ### Encode the Statistical Data Type

# Next, we will encode each feature with its correct data type to ensure each feature is treated appropriately by whatever functions we use moving forward.
# 
# The numeric features in our particular dataset are already encoded correctly (`float` for continuous and `int` for discrete features). What we need to pay closer attention to is the categorical features. For instance, note in particular, that the 'MSSubClass' feature is read as an `int` type, but is actually a nominative categorical.

# In[5]:


# The nominative (unordered) categorical features
features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", 
                "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", 
                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", 
                "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", 
                "SaleType", "SaleCondition"]

# The ordinal (ordered) categorical features 

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()}

def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df


# ### Handle Missing Values

# We'll impute 0 for missing numeric values and "None" for missing categorical values. Additionally, we will create "missing value" indicator columns--these columns will contain boolean values indicating whether a particular feature value was imputed for a sample.

# In[6]:


def impute_plus(df):
    # Get names of columns with missing values
    cols_with_missing = [col for col in df.columns if col != 'SalePrice' and df[col].isnull().any()]
    # Make new columns indicating imputed features (`SalePrice` column exluded)
    for col in cols_with_missing:
        df[col + '_was_missing'] = df[col].isnull()
        df[col + '_was_missing'] = (df[col + '_was_missing']) * 1
    # Impute 0 for missing numeric values
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    # Impute "None" for missing categorical values
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df


# ## Load Data

# Now we can call the data loader and get the processed data splits--let's take a quick look:

# In[7]:


df_train, df_test = load_data()


# In[8]:


# Peek at the values
display(df_train)
display(df_test)

# # Display information about dtypes and missing values
# display(df_train.info())
# display(df_test.info())


# ## Establish Baseline

# Before we delve into feature engineering, we're going to establish a baseline score to judge our upcoming feature sets against. We will make our predictions with an XGBoost model and create a function to compute the cross-validated *Root Mean Squared Error* (RMSE) score for each feature set our model trains on. 
# 
# XGBoost minimizes *Mean Squared Error* (MSE), but we want to minimize Root Mean Squared Error (RMSE) specifically, requiring us to "reshape" our target feature (`Sale Price`) using log transformation for training and later applying exponential transform to the predictions. Mathematically, this makes sense because we typically use the log scale for variables that change multiplicatively with other factors. How do we know when variables should be modeled as changing multiplicatively? 
# - Every day language (surprise, surprise!). Examples include prices ("foreclosed homes sell at a 20% to 30% discount"), and sales ("your yoy sales are up 20% accross models").
# - More generally, variables that are strictly non-negative (e.g, volatility, counts of errors or events, rainfall) are often treated as changing linearly in a log scale.

# In[9]:


# My default XGB parameters

xgb_params = dict(
    max_depth=3,                           # maximum depth of each tree - try 2 to 10
    learning_rate=0.1,                     # effect of each tree - try 0.0001 to 0.1
    n_estimators=100,                      # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=1,                    # minimum number of houses in a leaf - try 1 to 10
    colsample_bytree=1,                    # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=1,                           # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0,                           # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=1,                          # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,                   # set > 1 for boosted random forests
)


# In[10]:


def score_dataset(X, y, model=XGBRegressor(**xgb_params)):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Use RMSLE (Root Mean Squared Log Error) instead of MSE (Mean Squared Error)as evaluation metric
    # (So, we need to log-transform y to train and exp-transform the predictions)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring='neg_mean_squared_error'
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Let's run our new function on the processed data and get a baseline score:

# In[11]:


X = df_train.copy()
y = X.pop("SalePrice")

baseline_score = score_dataset(X, y)
print(f"Baseline score: {baseline_score:.5f} RMSE")


# This baseline score should help us in knowing whether some set of features we've assembled in our experimentation actually led to any improvement or not.

# # Part 2 - Feature Engineering

# ## Feature Utility Scores

# It's time for us to take a closer look at the features we have in our dataset. We will analyze how much potential a feature has by computing its *utility score*.

# In[12]:


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores): 
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# Our feature scores are listed below:

# In[13]:


mi_scores = make_mi_scores(X, y)

# Show Mutual Information (MI) score plot
plt.figure(dpi=120, figsize=(8, 20))
plot_mi_scores(mi_scores)


# We have a number of features that are highly informative and several that don't seem to be informative at all (at least by themselves). Therefore, we will focus our efforts on the top scoring features. Training on uninformative features can lead to overfitting as well, so features with 0.0 MI scores are going to be dropped entirely.

# In[14]:


def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]


# Let's check if removing these features actually lead to a performance gain.

# In[15]:


drop_uninformative(X, mi_scores)


# In[16]:


drop_uninformative(X, mi_scores).info()


# In[17]:


X = df_train.copy()
y = X.pop("SalePrice")
X = drop_uninformative(X, mi_scores)

# Check out if this results in any improvement from the baseline score
score_dataset(X, y)


# Nice! Removing our uninformative features does lead to a slight performance gain. We will add our new `drop_uninformative` function to our feature-creation pipeline.

# ## Create Features

# To make our feature engineering workflow more modular, we'll define a function that will take a prepared dataframe and pass it through a pipeline of transformations to get the final feature set.

# Let's go ahead and define one transformation now--a label encoding for the categorical features. 
# 
# *Note that we're specifically using label encoding for our unordered categories because we are using a tree-ensemble, XGBoost, particularly. If instead we decide to try using a linear regression model, we're going to have to use one-hot encoding for features with unordered categories.*

# In[18]:


def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    return X


# ### Create Features with Pandas (Data Wrangling)

# #### Mathematical Transforms (Ratios)

# Ratios seem to be difficult for most models to learn, so creating new features expressing ratio combinations can often lead to some easy performance gains. We're going to create two new features expressing important ratios using mathematical transformation:
# 
# - `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
# - `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
# 

# In[19]:


def mathematical_transforms(df):
    X = pd.DataFrame() # Just a dataframe to hold new features
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X


# #### Interactions

# During an exploratory analysis of our data, something interesting came up:

# In[20]:


# Check out interaction between `BldgType` and `GrLivArea`
feature = "GrLivArea"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df_train, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);


# The trend lines being significantly different from one category to the next indicates an interaction effect between `GrLivArea` and `BldgType` that relates to a home's `SalePrice`. 
# 
# Below are several other detected interaction effects between categorical and numerical variables:

# In[21]:


# Check out interaction between `BsmtCond` and `TotalBsmtSF`
feature = "TotalBsmtSF"

sns.lmplot(
    x=feature, y="SalePrice", hue="BsmtCond", col="BsmtCond",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);


# In[22]:


# Check out interaction between `GarageQual` and `GarageArea`
feature = "GarageArea"

sns.lmplot(
    x=feature, y="SalePrice", hue="GarageQual", col="GarageQual",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);


# In[23]:


def interactions(df):
    # BldgType interaction
    X_inter_1 = pd.get_dummies(df.BldgType, prefix='Bldg')
    X_inter_1 = X_inter_1.mul(df.GrLivArea, axis=0)
    # Bsmt interaction
    X_inter_2 = pd.get_dummies(df.BsmtCond, prefix='BsmtCond')
    X_inter_2 = X_inter_2.mul(df.TotalBsmtSF, axis=0)
    # Garage interaction
    X_inter_3 = pd.get_dummies(df.GarageQual, prefix='GarageQual')
    X_inter_3 = X_inter_3.mul(df.GarageArea, axis=0)
    # Combine into one DataFrame
    X = X_inter_1.join(X_inter_2)
    #X = X.join(X_inter_3)
    return X


# #### Counts

# Let's create a new feature called `PorchTypes` that describes how many kinds of outdoor areas a dwelling has. We will count how many of the following are greater than 0.0:
# 
# -`WoodDeckSF`
# 
# -`OpenPorchSF`
# 
# -`EnclosedPorch`
# 
# -`Threeseasonporch`
# 
# -`ScreenPorch`

# And then we're going to create another new feature `TotalHalfBath` that contains the sum of half-bathrooms within the property.

# Additionally, we will also sum up the total number of rooms (including full and half bathrooms) in each property and store them in a new feature called `TotalRoom`.

# In[24]:


def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF',
                        'OpenPorchSF',
                        'EnclosedPorch',
                        'Threeseasonporch',
                        'ScreenPorch'
                        ]].gt(0.0).sum(axis=1)
    X['TotalHalfBath'] = df.BsmtFullBath + df.BsmtHalfBath
    X['TotalRoom'] = df.TotRmsAbvGrd + df.FullBath + df.HalfBath
    return X


# #### Grouped Transform

# The value of a home often depends on how it compares to typical homes in its neighborhood. Therefore, let's create a new feature called `MedNhbdArea` that describes the *median* of `GrLivArea` grouped on `Neighborhood`.

# In[25]:


def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbdArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    return X


# ### k-Means Clustering

# Let's use the help of an unsupervised algorithm (k-mean clustering) to create new features. We've selected the following features to determine what the clusters are based on.

# In[26]:


cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]


# First, we will use the cluster labels generated by the algorithm as a new feature.

# In[27]:


def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new


# In[28]:


# # Optimize number of clusters
# # by comparing cross validation scores
# # (Will take some time--like HOURS)

# for n in list(range(1,21)):
#     X_orig = df_train.copy().drop("SalePrice", axis=1)
#     X = cluster_labels(df_train, cluster_features, n_clusters=n)
#     X = X_orig.join(X)
#     score = score_dataset(X, y, xgb)
#     print("Cross-validation score:", score, 
#         "\n Value used for n_clusters (number of clusters) for labeling:n=", n)


# Next, we will use the *distance* of the observations to each cluster as another new feature.

# In[29]:


def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd


# In[30]:


# # Optimize number of clusters
# # by comparing cross validation scores
# # (Will take some time--like HOURS)

# for n in list(range(1,21)):
#     X_orig = df_train.copy().drop("SalePrice", axis=1)
#     X = cluster_distance(df_train, cluster_features, n_clusters=n)
#     X = X_orig.join(X)
#     score = score_dataset(X, y, xgb)
#     print("Cross-validation score:", score, 
#         "\n Value used for n_clusters (number of clusters) for labeling:n=", n)


# ### Principal Component Analysis

# This time we'll use PCA, another unsupervised learning method, to create more new features.
# 
# *Note: We are not including missing value indicator columns when assessing correlations because the combinations of rows having many 0s in these indicator columns will yield a NaN result when the given numerator and denominator are equal to 0 during the calculation.*

# In[31]:


def corrplot(df, method="pearson", annot=True, **kwargs):
    sns.clustermap(
        df.corr(method),
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method="complete",
        annot=annot,
        **kwargs,
    )

corrplot(df_train.iloc[:,:80], annot=None)


# In[32]:


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,        # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,          # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


# Let's pick a subset of features for PCA:

# In[33]:


pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]


# The PCA algorithm gives us *loadings* which describe each *component* of variation, and also the components which were the transformed datapoints. The loadings can suggest features to create. Additionally, we can use the components as features directly.

# In[34]:


X_temp = X.loc[:, pca_features]

# `apply_pca`, defined above
pca, X_pca, loadings = apply_pca(X_temp)
print(loadings)


# In[35]:


# Plot explained variance based on components from PCA
plot_variance(pca)


# Since our goal right now is to discover as many useful features as possible, let's create features inspired by the PCA loadings while also using the exact components as a different set of features:

# In[36]:


def pca_inspired(df):
    X = pd.DataFrame()
    X["GrLivAreaPlusBsmtSF"] = df.GrLivArea + df.TotalBsmtSF
    X["RecentRemodLargeBsmt"] = df.YearRemodAdd * df.TotalBsmtSF
    return X

def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


# #### PCA Application - Indicate Outliers

# Applying PCA can also help us determine houses that are outliers or houses with values that are not well represented by the rest of the data set.

# In[37]:


sns.catplot(
    y="value",
    col="variable",
    data=X_pca.melt(),
    kind='boxen',
    sharey=False,
    col_wrap=2,
);


# As you can see, for each of the components there are several points lying at the extreme ends of the distributions -- they are outliers. Let's see those houses that sit at the extremes of a component:

# In[38]:


# Can change PC1 to PC2, PC3, or PC4
component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index

df_train[["SalePrice", "Neighborhood", "SaleCondition"] + pca_features].iloc[idx]


# Notice that there are several dwellings listed as `Partial` sales in the `Edwards` neighborhood that stand out. A partial sale is what occurs when there are multiple owners of a property and one or more of them sell their "partial" ownership of the property. These kinds of sales often happen during the settlement of a family estate or the dissolution of a business and aren't advertised publicly, making these cases true outliers, especially within our supposed research context--houses on the open market.

# Some models can benefit from having these outliers indicated, which is what this next transform will do.

# In[39]:


def indicate_outliers(df):
    X_new = pd.DataFrame()
    X_new["Outlier"] = (df.Neighborhood == "Edwards") & (df.SaleCondition == "Partial")
    return X_new


# ### Target Encoding

# We're going to use target encoding using the following steps:
# 1. Split the data into folds, each fold having two splits of the dataset.
# 2. Train the encoder on one split but transform the values of the other.
# 3. Repeat for all the splits.
# 
# The next cell contains a wrapper we can use with any target encoder:

# In[40]:


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


# Note:
# 
# To use, follow code sample below:
# 
# `encoder = CrossFoldEncoder(MEstimateEncoder, m=1)`
# 
# `X_encoded = encoder.fit_transform(X, y, cols=["MSSubClass"]))`
# 
# 

# ### Create Final Feature Set

# Time to combine everything together.

# In[41]:


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop('SalePrice')
    mi_scores = make_mi_scores(X, y)
    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Step 1: Drop features with low Mutual Information scores
    X = drop_uninformative(X, mi_scores)

    # Step 2: Add features from mathematical transforms 
    ######## (`LivLotRatio`, `Spaciousness`)
    X = X.join(mathematical_transforms(X))

    # Step 3: Add features from known interaction effects 
    ######## (categorical-`BldgType`and continuous-`GrLivArea`)
    #X = X.join(interactions(X))

    # Step 4: Add new feature from counts 
    ######## (`PorchTypes`, `TotalHalfBath`, `TotalRoom`)
    X = X.join(counts(X))

    # Step 5: Add new feature from group transform
    ######## (median `GrLivArea` by `neighborhood`)
    X = X.join(group_transforms(X))

    # Step 6: Add features from k-means clustering 
    ######## (cluster labels, cluster distance)
    #X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    #X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Step 7: Add features from PCA
    ######## (loadings-inspired features , PCA components, & outlier indicators)
    X = X.join(pca_inspired(X))
    #X = X.join(pca_components(X, pca_features))
    #X = X.join(indicate_outliers(X))
  
    # Label encoding for the categorical features
    X = label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Step 8: Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))

    if df_test is not None:
        return X, X_test
    else:
        return X


# Putting the transformations into separate functions makes it easier to experiment with various combination.
# 
# We can modify any of these transformations or come up with some more ideas to add to the pipeline. At this stage, we have left the ones that gave the best results uncommented.

# In[42]:


df_train, df_test = load_data()
X_train = create_features(df_train)
y_train = df_train.loc[:, 'SalePrice']

score_dataset(X_train, y_train)


# # Part 3 - Hyperparameter Tuning

# Now that we are done creating out final set of features, it's time to do some hyperparameter tuning with XGBoost to optimize our model performance further.

# In[43]:


X_train = create_features(df_train)
y_train = df_train.loc[:, "SalePrice"]

xgb_params = dict(
    max_depth=4,                           # maximum depth of each tree - try 2 to 10
    learning_rate=0.0058603076512435655,    # effect of each tree - try 0.0001 to 0.1
    n_estimators=5045,                     # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=2,                    # minimum number of houses in a leaf - try 1 to 10
    colsample_bytree=0.22556099175248345,   # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=0.5632348136091383,          # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0.09888625622197889,        # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=0.00890758697724437,         # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,                   # set > 1 for boosted random forests
)

xgb = XGBRegressor(**xgb_params)
score_dataset(X_train, y_train, xgb)


# Rather than just tuning them by hand, we're going to use a tuning library, Optuna, with XGBoost:

# In[44]:


# def objective(trial):
#     xgb_params = dict(
#         max_depth=trial.suggest_int("max_depth", 2, 10),
#         learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
#         min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
#         colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
#         subsample=trial.suggest_float("subsample", 0.2, 1.0),
#         reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
#         reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
#     )
#     xgb = XGBRegressor(**xgb_params)
#     return score_dataset(X_train, y_train, xgb)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=40)
# xgb_params = study.best_params


# Optuna's recommended parameter values:
# 
# 
# ```
# Parameters: 
# {'max_depth': 4, 'learning_rate': 0.0058603076512435655, 'n_estimators': 5045, 'min_child_weight': 2, 'colsample_bytree': 0.22556099175248345, 'subsample': 0.5632348136091383, 'reg_alpha': 0.09888625622197889, 'reg_lambda': 0.00890758697724437}. 
# Score: 0.11442743288078303.
# ```
# 
# 

# # Part 4 - Train Model and Create Predictions

# To create our final predictions, we will take the following steps:
# * create your feature set from the original data
# * train XGBoost on the training data
# * use the trained model to make predictions from the test set
# * save the predictions to a CSV file
# 
# 

# In[45]:


X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "SalePrice"]

xgb = XGBRegressor(**xgb_params)
# XGB minimizes MSE, but we want to minimize RMSLE
# So, we need to log-transform y to train and exp-transform the predictions
xgb.fit(X_train, np.log(y))
predictions = np.exp(xgb.predict(X_test))

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})


# In[46]:


# Save Submission
output.to_csv('my_submission.csv', index=False)
print("Your predictions are successfully saved!")


# In[47]:


# Save the XGB model
filename = 'ames_house_xgb_model.pkl'
pickle.dump(xgb, open(filename, 'wb'))

# Save processed test data
X_test.to_csv('df_test_processed.csv', index=False)


# # Part 5 - Model Interpretation

# Our final model landed in the top 8% of Kaggle House Prices Prediction leaderboard (as of 10/24/2021). Groovy! However, it's important for us to take it a step further.
# 
# > Many people say machine learning models are "black boxes", in the sense that they can make good predictions but you can't understand the logic behind those predictions. This statement is true in the sense that most data scientists don't know how to extract insights from models yet.
# 
# There is an increasing need for data scientists who are able to extract insights from sophisticated machine learning models to help inform human decision-making. Some decisions are made automatically by models like the ones we have just built, but many important decisions are made by humans. For these decisions, insights can be more valuable than predictions. Beyond informing human decision-making, insights extracted from machine learning models have many other uses, including:
# - Debugging
# - Informing feature engineering
# - Directing future data collection
# - Building Trust
# 
# Right now, we want to answer the following questions about our model:
# * What features in the data did the model think are most important?
# * For any single prediction from a model, how did each feature in the data affect that particular prediction?
# * How does each feature affect the model's predictions in a big-picture sense (what is its typical effect when considered over a large number of possible predictions)?

# First, we'll use SHAP Values to explain individual predictions. Afterwards, we will look at model-level insights.

# In[48]:


# Pick an arbitrary row (first row starts at 0)
row_to_show = 42
data_for_prediction = X_test.iloc[[row_to_show]]

# Generate prediction
y_sample = np.exp(xgb.predict(data_for_prediction))

# Create object that can calculate Shap values
explainer = shap.TreeExplainer(xgb)

# Calculate Shap values from prediction
shap_values = explainer.shap_values(data_for_prediction)


# **For a single prediction, what features in the data did the model think are most important?**

# In[49]:


plt.title('Feature importance based on SHAP values?')
shap.summary_plot(shap_values, data_for_prediction, plot_type="bar")


# **How did each feature in the data affect that particular prediction?**

# In[50]:


plt.title('Feature impact on model output (feature impact in details below)')
shap.summary_plot(shap_values, data_for_prediction)


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)


# Now that we've seen the inner workings of our model in making an individual prediction, let's aggregate all the information into powerful model-level insights.

# In[51]:


# Use test set to get predictions
data_for_prediction = X_test

# Generate predictions
y_sample = np.exp(xgb.predict(data_for_prediction))

# Create object that can calculate Shap values
explainer = shap.TreeExplainer(xgb)

# Calculate Shap values from predictions
shap_values = explainer.shap_values(data_for_prediction)


# **How does each feature affect the model's predictions in a big-picture sense? In other words, what is its typical effect when considered over a large number of possible predictions?**

# In[52]:


plt.title('Feature impact on overall model output (feature impact in details below)')
shap.summary_plot(shap_values, data_for_prediction)


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)


# _____
# 
# ## Acknowledgement
# 
# Steps taken throughout the model-building process in this notebook are inspired by [this Kaggle notebook](https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices) by Ryan Holbrook and Alexis Cook (modified for better performance). Check out their notebook for more ideas to improve the prediction model.
# 
# Some text in the beginning of the Model Interpretation section is copied from Kaggle's fantastic [Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) course.
# 
# Other quoted sources include [Business Data Science](https://www.amazon.com/Business-Data-Science-Combining-Accelerate/dp/1260452778) by Matt Taddy.
