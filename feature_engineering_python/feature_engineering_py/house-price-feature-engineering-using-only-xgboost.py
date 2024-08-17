#!/usr/bin/env python
# coding: utf-8

# ## Imports and Configuration ##

# In[1]:


import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') # Mute warnings (thrown from Optuna/XGBoost)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


# ## Introduction ##
# 
# This is FilterJoe's final version of the House Price Learning competition for the Feature Engineering Course on Kaggle. The base notebook that came from the bonus lesson of this course was extensively modified/expanded. The goal was to learn as much as possible about feature engineering, using only XGBoost. If you want to do the same, I recommend you do NOT just run this notebook . . . but rather start with the bonus lesson as your base code and gradually experiment over the course of a week or two like I did, perhaps glancing at this notebook from time to time to get inspired with new ideas.
# 
# After weeks of experimentation with many forms of feature engineering, this notebook ended up with a very good score, possibly the best score on the leaderboard that uses only XGBoost. If getting the best score were the goal, next steps would be to experiment with other types of models besides XGboost. Then, you'd want to blend and/or stack several models together.
# 
# ## Data Preprocessing ##
# 
# Before we can do any feature engineering, we need to *preprocess* the data to get it in a form suitable for analysis. The feature engineering course data was simpler than this competition data. For the *Ames* competition dataset, we'll need to:
# - **Load** the data from CSV files
# - **Clean** the data to fix any errors or inconsistencies
# - **Encode** the statistical data type (numeric, categorical)
# - **Impute** any missing values
# 
# These steps are wrapped in a function, which makes it easy to get a fresh dataframe when needed. After reading the CSV file, we'll apply three preprocessing steps, `clean`, `encode`, and `impute`, and then create the data splits: one (`df_train`) for training the model, and one (`df_test`) for making the predictions for the leaderboard test set.
# 
# The base code for all this was provided in the bonus lesson (from Kaggle's feature engineering course), but was extensively modified (Kaggle user FilterJoe) to include:
# 
# - verbose flags: summary of each step in preprocessing and feature engineering is displayed when called with verbose=True
# - one-hot encoding function
# - feature dropping function
# - added/customized features (kept only those that boosted score)
# - improvements to imputation strategies
# - experimented with Boruta-SHAP feature selection (commented out as it always hurt XGBoost score a little)
# - feature importances output from XGBoost (used this info to test dropping features of low importance to XGBoost)
# - I did a bunch of EDA, but it's in a different notebook to keep this notebook smaller
# 
# 
# ## Load and Clean ##

# In[2]:


# Preprocessing steps collected together into this one function, then defined seperately in following cells

def preprocess_data(verbose=False):
    if verbose:
        print("*** BEGIN PREPROCESSING ***")
    data_directory = Path("../input/house-prices-advanced-regression-techniques/")
    df, train_indices, test_indices = load(data_directory, verbose)
    df = clean(df, verbose)
    df = order_ordinals(df, verbose)
    df = encode_nominative_categoricals(df, verbose)
    df = impute(df, verbose)

    df_train = df.loc[train_indices, :] # Reform splits: train
    df_test = df.loc[test_indices, :] # Reform splits: test
    
    if verbose:
        print("*** END PREPROCESSING ***\n")
    return df_train, df_test


# In[3]:


# LOAD: read and merge

def load(data_dir, verbose):
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")  # read_csv assumes first column is index, which you can name w/index_col
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")  # use the (better be!) unique Ids to merge and split as needed
    df = pd.concat([df_train, df_test]) # Merge splits, so can preprocess (clean, encode, impute) together
    if verbose:
        print("Train and test splits read from CSV and merged.")
        print("Missing values before any preprocessing: ", df.isnull().values.sum())
    return df, df_train.index, df_test.index


# In[4]:


# CLEAN: Examine data_description.txt further ... Any more cleaning to add here?

def clean(df, verbose):
    cleaned_features = ["Exterior2nd", "GarageYrBlt"]
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})  # df.Exterior2nd.unique() displayed multiple names for BrkComm
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Wd Shng": "WdShing"})  # to match same name in Exterior1st
    df["Exterior2nd"] = df["Exterior2nd"].replace({"CmentBd": "CemntBd"})  # to match same name in Exterior1st
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt) # replace nonsensical GarageYrBlt values with year house was built
    
    # Names beginning with numbers make for messy Python
    name_pairs = {"1stFlrSF": "FirstFlrSF",
                  "2ndFlrSF": "SecondFlrSF",
                  "3SsnPorch": "ThreeSeaPorch",}
    df.rename(columns=name_pairs, inplace=True)
    if verbose:
        print("Cleaned: ", cleaned_features, sep='\n    ')
        print("Renamed (From, To): ", *name_pairs.items(), sep='\n    ')
    return df


# ## Encode ##
# 
# Convert to expected formats so functions, transformations, and models work correctly/consistently.
# 
# Numeric features already encoded correctly (`float` for continuous, `int` for discrete)
# Categoricals need work. Some int type classes (i.e. MSSubClass) are really nominative categoricals.
# * ordinal - ordered such as poor, fair, typical/average, good, excellent
# * nominative - unordered such as eye color (whether or not initially coded with strings or ints)
# 

# In[5]:


# ENCODE ordinal categorical features 

five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

# map each ordinal category to its ORDERED range of possible values (called levels in Pandas)
ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels, # one-hot encoding for XGBoost makes score worse
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
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], # swapped Maj2 and Maj1 to match data description 
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"], # all but 1 row have all utilities - yet keeping this feature helps both val and lb scores significantly!
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"], # IMHO, Mix better than FuseP but only 1 row has it so doesn't matter
    "Fence": ["MnPrv", "MnWw", "GdWo", "GdPrv"], # one-hot-encoding didn't help wood-quality/privacy-level mix, but moving MnPrv to bottom in order DID help
}
ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()} # Add a None level, which will be assigned for missing values

def print_n_per_line(items, n):
    for idx, item in enumerate(items):
        print("    " + item + ''.join([' ' for i in range(14 - len(item))]), end='')
        if idx % n == n-1:
            print()
    print()

def order_ordinals(df, verbose):
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    if verbose:
        print("Created ordinal categories and ordered with levels: ") # a display alternative: , *list(ordered_levels.keys()), sep='\n    ')
        print_n_per_line(list(ordered_levels.keys()), 5)
    return df


# In[6]:


# ENCODE nominative categorical features

features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]

def encode_nominative_categoricals(df, verbose):
    for name in features_nom: # Nominal categories
        df[name] = df[name].astype("category")
        if "None" not in df[name].cat.categories: # Add None category for missing values
            df[name].cat.add_categories("None", inplace=True)
    if verbose:
        print("Created nominative categories (mostly conversions from numericals): ") # , *features_nom, sep='\n    '
        print_n_per_line(features_nom, 5)
    return df


# ## Impute ##
# 
# Handle Missing Values with Imputation
# 
# The "starter" imputation function (last few lines of cell below)
# * imputes 0 for missing numeric values
# * imputes "None" for missing categorical values
# 
# Improvements:
# * using mode for some missing values of categorical features
# * using mean for same neighborhood for features where it seems inutitively warranted
# 
# Potential further improvements:
# * creating missing value indicators (1 when value imputed, 0 otherwise)

# In[7]:


def impute(df, verbose):
    if verbose:
        print("\nMissing values before imputing:")
        sums_df = pd.DataFrame(df.isnull().sum(), columns = ['Missing Values'])
        print(sums_df[sums_df['Missing Values'] > 0])
        total_missing_values = df.isnull().values.sum()
        print("Total missing values: ", total_missing_values)
    
    # Some experiments for better imputing (if commented out, they didn't help with XGBoost):
    df['MSZoning'] = df.groupby("Neighborhood")['MSZoning'].transform(lambda x: x.fillna(x.mode()))
#     df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) # didn't much help or hurt
#     df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
#     df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
#     df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    
    # the baseline generic strategy for remaining missing values
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    if verbose:
        print("\nmissing values imputed: ", total_missing_values, )
        print("missing values remaining after imputing: ", df.isnull().values.sum())
    return df


# ## Preprocess Data ##
# 
# Everything is defined so we can now call the data preprocessor to get the processed data splits. Then we examine:

# In[8]:


df_train, df_test = preprocess_data(verbose=False) # load, clean, encode, impute


# In[9]:


# Peek at data - uncomment and run individual lines in this cell if you'd like to see what df's contain.
# Extensive EDA is in a different notebook

# display(df_train)
# display(df_test)

# Display information about dtypes and missing values
# display(df_train.info())
# display(df_test.info())

display(df_train.Functional.unique()) # Note how order is displayed using "<" for an ordered category such as 'Functional'
display(df_train.Functional.value_counts())

display(df_train.head().T.head(40))
display(df_train.head().T.tail(40))


# ## Establish Baseline ##
# 
# Finally, let's establish a baseline score to judge our feature engineering against.
# 
# Here is the function created in Lesson 1 that computes cross-validated RMSLE score for a feature set. XGBoost was used throughout, but this framework can certainly be used to experiment with other models.
# 

# In[10]:


def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    
    log_y = np.log(y) # RMSLE (Root Mean Squared Log Error) is metric for Housing competition
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error")
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# We can reuse this scoring function anytime we want to try out a new feature set. We'll run it now on the processed data with no additional features and get a baseline score:

# In[11]:


X = df_train.copy()
y = X.pop("SalePrice")

baseline_score = score_dataset(X, y) # , model=lass
print(f"Baseline score: {baseline_score:.5f} RMSLE")


# This baseline score helps us to know whether some set of features we've assembled has actually led to any improvement or not.
# 
# # Step 2 - Feature Utility Scores #
# 
# In Lesson 2 we saw how to use mutual information to compute a *utility score* for a feature, giving you an indication of how much potential the feature has. This hidden cell defines the two utility functions we used, `make_mi_scores` and `plot_mi_scores`: 

# In[12]:


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# Let's look at our feature scores again:

# In[13]:


X = df_train.copy()
y = X.pop("SalePrice")

mi_scores = make_mi_scores(X, y)
mi_scores


# You can see that we have a number of features that are highly informative and also some that don't seem to be informative at all (at least by themselves). As we talked about in Tutorial 2, the top scoring features will usually pay-off the most during feature development, so it could be a good idea to focus your efforts on those. On the other hand, training on uninformative features can lead to overfitting. So, the features with 0.0 scores we'll drop entirely:

# In[14]:


def drop_uninformative(df, mi_scores, verbose=False):
    if verbose:
        print('Dropping the following features with low mi_scores:')
        print(df.loc[:, mi_scores == 0.0].columns)
    return df.loc[:, mi_scores > 0.0]


# Removing them does lead to a modest performance gain:

# In[15]:


X = df_train.copy()
y = X.pop("SalePrice")
X = drop_uninformative(X, mi_scores, verbose=True)

score_dataset(X, y)


# Later, we'll add the `drop_uninformative` function to our feature-creation pipeline. However . . .
# 
# Mutual information is not the best way to decide which features to drop, and that this gets impacted by model selection and hyperparameter optimization, as well as other feature development. After much experimentation, I found that with XGBoost, the best guide to deciding which features to drop was XGBoost itself - getting feature importances from it's .get_booster method and experimenting with dropping the least important features according to this XGBoost ranking.
# 
# I found that a better use of Mutual information was looking at the highest ranking features and spending the majority of my feature engineering time on those top 5-10 features.
# 
# # Step 3 - Create Features #
# 
# Now we'll start developing our feature set.
# 
# To make our feature engineering workflow more modular, we'll define a function that will take a prepared dataframe and pass it through a pipeline of transformations to get the final feature set. It will look something like this:
# 
# ```
# def create_features(df):
#     X = df.copy()
#     y = X.pop("SalePrice")
#     X = X.join(create_features_1(X))
#     X = X.join(create_features_2(X))
#     X = X.join(create_features_3(X))
#     # ...
#     return X
# ```
# 
# Let's go ahead and define one transformation now, a [label encoding](https://www.kaggle.com/alexisbcook/categorical-variables) for the categorical features:

# In[16]:


def label_encode_these(df, cols=[], verbose=False):
    # Label encoding works great for XGBoost and RandomForest. Models like Lasso
    # or Ridge work better with one-hot, especially nominal categories.
    # Categories not passed to label_encode_these will be automatically one-hot encoded.
    X = df.copy()
    for colname in cols:
        X[colname] = X[colname].cat.codes # holds category labels (alternative: sklearn's LabelEncoder)
    if verbose:
        print(len(cols), "categorical features label encoded:", )
        print_n_per_line(cols, 5)
    return X

# The above more fleixbile function works in conjunction with the one_hot_encode_except function
# that follows, and replaces the function that was initially supplied:
    
# def label_encode_all(df, verbose=False):
#     X = df.copy()
#     category_columns = X.select_dtypes(["category"])
#     for colname in category_columns:
#         X[colname] = X[colname].cat.codes
#     return X


# In[17]:


def one_hot_encode_except(df, label_encoded_cols=[], verbose=False):
    # label_encoded_cols is supplied as a parameter because don't want
    # to one hot encode columns that have already been label encoded. This is because either:
    # 1) the label encoded columns have a natural order to them (ordinal) OR
    # 2) through experimentation, found label encoding did better than one-hot (i.e. with XGBoost)
    X = df.copy()
    remaining_col_names = []
    for col_name in list(X):
        if X[col_name].dtype.name == 'category' and col_name not in label_encoded_cols:
            remaining_col_names.append(col_name)
    X = pd.get_dummies(X, columns=remaining_col_names)
    
    if verbose:
        print("One Hot encoding applied to remaining", len(remaining_col_names), "categorical features:")
        print_n_per_line(remaining_col_names, 5)
    return X


# A label encoding is okay for any kind of categorical feature when you're using a tree-ensemble like XGBoost, even for unordered categories. If you wanted to try a linear regression model (also popular in this competition), you would instead want to use a one-hot encoding, especially for the features with unordered categories.
# 
# ## Create Features with Pandas ##
# 
# This cell reproduces the work you did in Exercise 3, where you applied strategies for creating features in Pandas. Modify or add to these functions to try out other feature combinations.

# In[18]:


def mathematical_transforms(df, verbose=False):
    X = pd.DataFrame()  # dataframe to hold new features
    X["LivLotRatio"] = df.GrLivArea / df.LotArea
    X["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    X["TotalSF"] = df.TotalBsmtSF + df.FirstFlrSF + df.SecondFlrSF
    # following experiments tested poorly with XGboost:
#     X['YrBuiltPlusRemod'] = df.YearBuilt + df.YearRemodAdd
#     X['Total_Bathrooms'] = df.FullBath + df.BsmtFullBath + .5 * df.HalfBath + .5 * df.BsmtHalfBath
#     X["TotalRooms"] = df.TotRmsAbvGrd + df.FullBath + df.HalfBath + df.BsmtFullBath + df.BsmtHalfBath + df.BedroomAbvGr + df.KitchenAbvGr
#     X["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch
    new_features = ["LivLotRatio", "Spaciousness", "TotalSF"]
    if verbose and new_features:
        print(len(new_features), "new features created via mathematical transforms: ")
        print("    ", new_features)
    return X

def interactions(df, verbose=False):
    # interactions with GrLivArea that did NOT help include HouseStyle, Neighborhood, OverallQual, MSZoning, LotConfig, YearBuilt, ExterQual, TotRmsAbvGrd
    # interactions with GrLivArea that DID help according to cross validation, but not leaderboard: BedroomAbvGr
    new_interactions = ["Bldg with GrLivArea"]
    columns_to_interact_with_GrLivArea = ['BldgType'] # if you add to this list, also add to lists in adjacent lines (above and below)
    new_prefixes = ['Bldg']
    X = pd.get_dummies(df[columns_to_interact_with_GrLivArea], columns=columns_to_interact_with_GrLivArea, prefix=new_prefixes)
    X = X.mul(df.GrLivArea, axis=0)

    if verbose and new_interactions:
        print(len(new_interactions), "new interaction features created: ")
        print("    ", new_interactions, " :" )
        for prefix in new_prefixes:
            print_n_per_line(list(X.columns[X.columns.str.startswith(prefix)]), 5)
    return X

def counts(df, verbose=False):
    X = pd.DataFrame()
    X["PorchTypes"] = df[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "ThreeSeaPorch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)
    new_counted_features = ["PorchTypes"]
    if verbose and new_counted_features:
        print(len(new_counted_features), 'new "feature count" type features created: ')
        print("    ", new_counted_features)
    return X

def break_down(df, verbose=False):
# This function made sense in Creating Features Exercise 3, but
# it makes no sense with this data set because
# 1) in this data set, MSSubClass is a number, not text
# 2) Even if we replace the number with the text description . . .
# 3) The extra info is already covered by several other features: HouseStyle, YearBuilt, BldgType
    X = pd.DataFrame()
    X["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0] 
    return X

def group_transforms(df, verbose=False):
    X = pd.DataFrame()
    X["MedNeighLivArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    X["MedNeighLotArea"] = df.groupby("Neighborhood")["LotArea"].transform("median")
    X["MedQualLivArea"] = df.groupby("OverallQual")["GrLivArea"].transform("median")
#     X["MedQualLotArea"] = df.groupby("OverallQual")["LotArea"].transform("median")
#     X["MedCondLivArea"] = df.groupby("OverallCond")["GrLivArea"].transform("median")
#     X["MedYearBuiltLivArea"] = df.groupby("YearBuilt")["GrLivArea"].transform("median")
#     X["MedGarageCarArea"] = df.groupby("GarageCars")["GarageArea"].transform("median")
#     X["MedGarageCarArea"] = df.groupby("GarageCars")["GarageArea"].transform("median")

    new_transforms = ["MedNeighLivArea", "MedNeighLotArea", "MedQualLivArea"]
    if verbose and new_transforms:
        print(len(new_transforms), "new features created via groupby transform of a feature: ")
        print("    ", new_transforms)
    return X


# Here are some ideas for other transforms you could explore:
# - Interactions between the quality `Qual` and condition `Cond` features. `OverallQual`, for instance, was a high-scoring feature. You could try combining it with `OverallCond` by converting both to integer type and taking a product.
# - Square roots of area features. This would convert units of square feet to just feet.
# - Logarithms of numeric features. If a feature has a skewed distribution, applying a logarithm can help normalize it.
# - Interactions between numeric and categorical features that describe the same thing. You could look at interactions between `BsmtQual` and `TotalBsmtSF`, for instance.
# - Other group statistics in `Neighboorhood`. We did the median of `GrLivArea`. Looking at `mean`, `std`, or `count` could be interesting. You could also try combining the group statistics with other features. Maybe the *difference* of `GrLivArea` and the median is important?
# 
# ## k-Means Clustering ##
# 
# The first unsupervised algorithm we used to create features was k-means clustering. We saw that you could either use the cluster labels as a feature (a column with `0, 1, 2, ...`) or you could use the *distance* of the observations to each cluster. We saw how these features can sometimes be effective at untangling complicated spatial relationships.

# In[19]:


# This feature set is often in the 1-3 line descrption of a home for sale on info sheets
# developed by real estate agents. But didn't help:
# cluster_features = [
#     "BedroomAbvGr",
#     "FullBath",
#     "HalfBath",
#     "GrLivArea",
# ]

cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]

def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new


def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd


# ## Principal Component Analysis ##
# 
# PCA was the second unsupervised model we used for feature creation. We saw how it could be used to decompose the variational structure in the data. The PCA algorithm gave us *loadings* which described each component of variation, and also the *components* which were the transformed datapoints. The loadings can suggest features to create and the components we can use as features directly.
# 
# Here are the utility functions from the PCA lesson:

# In[20]:


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
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
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


# And here are transforms that produce the features from the Exercise 5. You might want to change these if you came up with a different answer.
# 

# In[21]:


def pca_inspired(df, verbose=False):
    X = pd.DataFrame()
    X["TotSqFt"] = df.GrLivArea + df.TotalBsmtSF  # adding in + df.GarageArea makes worse
    X["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
#     X["NewSqFt"] = ((df.GrLivArea + df.TotalBsmtSF) / df.YearRemodAdd.subtract(2011.5).mul(-1))  # my feature from ex. 5 ... adding GarageArea makes things worse
#     X["BsmtSF_to_AboveGrSF_ratio"] = df.TotalBsmtSF / df.GrLivArea  # my feature from ex. 5 ... hurt more than helped
    return X

def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca

pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",]


# These are only a couple ways you could use the principal components. You could also try clustering using one or more components. One thing to note is that PCA doesn't change the distance between points -- it's just like a rotation. So clustering with the full set of components is the same as clustering with the original features. Instead, pick some subset of components, maybe those with the most variance or the highest MI scores.
# 
# For further analysis, look at a correlation matrix for the dataset, as groups of highly correlated features often yield interesting loadings. EDA for this data set is done in a seperate notebook, which includes a correlation matrix and heat map.

# ### PCA Application - Indicate Outliers ###
# 
# In Exercise 5, you applied PCA to determine houses that were **outliers**, that is, houses having values not well represented in the rest of the data. You saw that there was a group of houses in the `Edwards` neighborhood having a `SaleCondition` of `Partial` whose values were especially extreme.
# 
# Some models can benefit from having these outliers indicated, which is what this next transform will do.

# In[22]:


def indicate_outliers(df, verbose=False):
    X_new = pd.DataFrame()
    X_new["Outlier"] = (df.Neighborhood == "Edwards") & (df.SaleCondition == "Partial")
    return X_new


# You could also consider applying some sort of robust scaler from scikit-learn's `sklearn.preprocessing` module to the outlying values, especially those in `GrLivArea`. [Here](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) is a tutorial illustrating some of them. Another option could be to create a feature of "outlier scores" using one of scikit-learn's [outlier detectors](https://scikit-learn.org/stable/modules/outlier_detection.html).

# ## Target Encoding ##
# 
# Needing a separate holdout set to create a target encoding is rather wasteful of data. In *Tutorial 6* we used 25% of our dataset just to encode a single feature, `Zipcode`. The data from the other features in that 25% we didn't get to use at all.
# 
# There is, however, a way you can use target encoding without having to use held-out encoding data. It's basically the same trick used in cross-validation:
# 1. Split the data into folds, each fold having two splits of the dataset.
# 2. Train the encoder on one split but transform the values of the other.
# 3. Repeat for all the splits.
# 
# This way, training and transformation always take place on independent sets of data, just like when you use a holdout set but without any data going to waste.
# 
# In the next hidden cell is a wrapper you can use with any target encoder. Use it like:
# 
# ```
# encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
# X_encoded = encoder.fit_transform(X, y, cols=["MSSubClass"]))
# ```
# 
# You can turn any of the encoders from the [`category_encoders`](http://contrib.scikit-learn.org/category_encoders/) library into a cross-fold encoder. The [`CatBoostEncoder`](http://contrib.scikit-learn.org/category_encoders/catboost.html) would be worth trying. It's similar to `MEstimateEncoder` but uses some tricks to better prevent overfitting. Its smoothing parameter is called `a` instead of `m`.

# In[23]:


# The advantages of using CrossFoldEncoder as opposed to something with no validation like:
# X["MedNeighPrice"] = X.groupby("Neighborhood")["SalePrice"].transform("median") )
# is that:
    # 1) you get crossfold validation which prevents target leakage
    # 2) you get to choose m, the smoothing factor which makes the model deal better with tiny amounts of data for a specific Neighborhood (for example)

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


# ## Drop Features ##
# Drop features that, for a given model, have very low importance or negative impact.<br>
# Mutual Information is a crude method that doesn't capture interactions between feature.<br>Better methods include:
# * Intuitition developed from EDA
# * feature selection algorithm such as Boruta-SHAP
# * feature importances given after fitting a particular model, such as using XGBoost's feature importances

# In[24]:


def drop_unhelpful_features(df, features_to_drop=None, verbose=False):
    if features_to_drop is None:
        return df
    else:
        if verbose:
            print(len(features_to_drop), "features dropped: ")
            print("    ", features_to_drop)
        return df.drop(features_to_drop, axis=1)

#     def drop_uninformative(df, mi_scores):
#     print('Dropping the following features with low mi_scores:')
#     print(df.loc[:, mi_scores == 0.0].columns)
#     return df.loc[:, mi_scores > 0.0]

# features Boruta-SHAP suggests with XGBoost (must modify this after each time changing feature set or big change to XGBoost HyperParameters):

#     features_to_drop = ['Exterior1st', 'LowQualFinSF', 'Condition2', 'BedroomAbvGr', 'LotShape', 'Electrical', 'BldgType', 'OpenPorchSF', 'WoodDeckSF', 'Condition1', 'BsmtFinType2', 'LotConfig', 'HalfBath', 'Bldg_2fmCon', 'YearBuilt_encoded', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'ScreenPorch', 'Functional', 'BsmtFullBath', 'HouseStyle', 'BsmtCond', 'TotRmsAbvGrd', 'MasVnrType', 'Foundation', 'GarageQual', 'Bldg_TwnhsE', 'Heating', 'Street', 'LivLotRatio', 'Threeseasonporch', 'Bldg_Duplex', 'GarageCond', 'LandSlope', 'MSZoning', 'Exterior2nd', 'MSZoning_encoded', 'BsmtHalfBath', 'GarageYrBlt', 'PoolArea', 'Bldg_None', 'SecondFlrSF', 'SaleType', 'MSSubClass', 'LotFrontage', 'MasVnrArea', 'EnclosedPorch', 'MedNeighLotArea', 'BsmtFinSF2', 'KitchenAbvGr', 'Alley', 'Bldg_Twnhs', 'Neighborhood', 'LandContour', 'Fence', 'TotalBsmtSF', 'Utilities', 'Spaciousness', 'ExterCond']
#     features_to_drop = ['ExterCond', 'LotFrontage', 'Alley', 'SaleType', 'BedroomAbvGr', 'LandSlope', 'Bldg_Twnhs', 'LotShape', 'LandContour', 'Exterior1st', 'Neighborhood', 'BsmtFinSF2', 'Bldg_2fmCon', 'MSSubClass', 'NewSqFt_encoded', 'CentralAir', 'OpenPorchSF', 'GarageArea', 'BsmtHalfBath', 'LivLotRatio', 'EnclosedPorch', 'Foundation', 'LotConfig', 'BldgType', 'RoofStyle', 'Threeseasonporch', 'RoofMatl', 'Bldg_TwnhsE', 'Spaciousness', 'Fence', 'GarageYrBlt', 'WoodDeckSF', 'Bldg_Duplex', 'Functional', 'SecondFlrSF', 'PavedDrive', 'Bldg_None', 'TotSqFt_encoded', 'TotRmsAbvGrd', 'Electrical', 'Condition2', 'MSZoning_encoded', 'MSZoning', 'BsmtFinType2', 'HalfBath', 'HouseStyle', 'BsmtCond', 'KitchenAbvGr', 'GarageQual', 'GarageCond', 'MedNeighLotArea', 'ScreenPorch', 'YearBuilt_encoded', 'Exterior2nd', 'Condition1', 'MasVnrType', 'BsmtFullBath', 'Heating', 'MasVnrArea', 'Street', 'PoolArea', 'LowQualFinSF', 'BsmtSF_to_AboveGrSF_ratio', 'Utilities']
#     features_to_drop = ['ExterCond', 'LotFrontage', 'Alley', 'SaleType', 'BedroomAbvGr', 'HeatingQC', 'LandSlope', 'Bldg_Twnhs', 'LotShape', 'LandContour', 'Exterior1st', 'BsmtFinSF2', 'Bldg_2fmCon', 'OverallCond_encoded', 'MSSubClass', 'Neighborhood', 'NewSqFt_encoded', 'CentralAir', 'BsmtHalfBath', 'LivLotRatio', 'EnclosedPorch', 'Foundation', 'LotConfig', 'BldgType', 'RoofStyle', 'Threeseasonporch', 'RoofMatl', 'Bldg_TwnhsE', 'Fence', 'GarageYrBlt', 'Bldg_Duplex', 'PavedDrive', 'Bldg_None', 'TotSqFt_encoded', 'Electrical', 'Condition2', 'MSZoning', 'BsmtFinType2', 'HalfBath', 'HouseStyle', 'BsmtCond', 'KitchenAbvGr', 'GarageQual', 'GarageCond', 'MSZoning_encoded', 'ScreenPorch', 'Condition1', 'Exterior2nd', 'MasVnrType', 'Heating', 'MasVnrArea', 'Street', 'PoolArea', 'LowQualFinSF', 'BsmtSF_to_AboveGrSF_ratio', 'Utilities']
#     features_to_drop = ['BsmtFullBath', 'LivLotRatio', 'LotShape', 'PavedDrive', 'BsmtSF_to_AboveGrSF_ratio', 'Spaciousness', 'EnclosedPorch', 'MSZoning', 'SaleType', 'RoofMatl', 'MSZoning_encoded', 'MSSubClass', 'Utilities', 'GarageQual', 'TotRmsAbvGrd', 'YearBuilt_encoded', 'LotConfig', 'RoofStyle', 'Bldg_Twnhs', 'Bldg_Duplex', 'MasVnrArea', 'HouseStyle', 'Bldg_2fmCon', 'BsmtFinType2', 'Condition1', 'LotFrontage', 'MasVnrType', 'Exterior1st', 'Foundation', 'HalfBath', 'GarageYrBlt', 'Fence', 'NewSqFt_encoded', 'KitchenAbvGr', 'BldgType', 'ExterCond', 'MedNeighLotArea', 'Exterior2nd', 'LandSlope', 'ScreenPorch', 'Bldg_TwnhsE', 'SecondFlrSF', 'Heating', 'Neighborhood', 'YearRemodAdd', 'LowQualFinSF', 'GarageCond', 'BsmtFinSF2', 'PoolArea', 'BedroomAbvGr', 'WoodDeckSF', 'TotSqFt_encoded', 'Threeseasonporch', 'LandContour', 'Condition2', 'BsmtHalfBath', 'Alley', 'BsmtCond', 'Street', 'Bldg_None', 'Electrical'] # 20 iterations BS
#     features_to_drop =['Bldg_Duplex', 'Bldg_TwnhsE', 'Exterior1st', 'MasVnrArea', 'BsmtCond', 'Electrical', 'BedroomAbvGr', 'TotSqFt_encoded', 'TotRmsAbvGrd', 'EnclosedPorch', 'LivLotRatio', 'ExterCond', 'MSZoning', 'PorchTypes', 'Condition2', 'MedNeighLotArea', 'LotConfig', 'LowQualFinSF', 'LotFrontage', 'Foundation', 'Threeseasonporch', 'Utilities', 'BsmtHalfBath', 'WoodDeckSF', 'Neighborhood', 'ScreenPorch', 'SaleType', 'GarageQual', 'Bldg_Twnhs', 'LotShape', 'Street', 'Alley', 'Condition1', 'HalfBath', 'BsmtFinType2', 'MSSubClass', 'RoofStyle', 'BldgType', 'Exterior2nd', 'PavedDrive', 'LandContour', 'HouseStyle', 'Heating', 'LandSlope', 'KitchenAbvGr', 'NewSqFt_encoded', 'Bldg_None', 'OpenPorchSF', 'Fence', 'MasVnrType', 'FullBath', 'Spaciousness', 'BsmtFinSF2', 'GarageCond', 'BsmtSF_to_AboveGrSF_ratio', 'PoolArea', 'Bldg_2fmCon', 'BsmtFullBath', 'RoofMatl'] # 12 iterations BS, RS=2

# hand picked by JG
#     features_to_drop = ['GarageCond', 'RoofStyle'] # hand picked by JG - didn't help

# XGBoost's low feature importances
# features_to_drop = ['Street', 'Threeseasonporch', 'PoolQC', 'Condition2', 'PoolArea', 'Heating', 'Bldg_2fmCon'] # 7 least important features according to XGBoost importances
# features_to_drop = ['Street', 'Threeseasonporch', 'Condition2'] # these 3 of the 4 features XGboost rated has having low importances gave best results


# ## Create Final Feature Pipeline Function ##
# 
# Combine all preprocessing and feature engineering into one pipeline function. Comment in or out individual lines to see whether it hurts or helps a given model. Also modify or add other types of feature engineering or transformations, keeping it tidy within small, testible functions.
# 
# After much experimentintation, the best performing combination will remain.

# In[25]:


def create_features(df, df_test=None, verbose=False):
    if verbose:
        print("*** BEGIN FEATURE ENGINEERING PIPELINE ***\n")
    X = df.copy()
    y = X.pop("SalePrice")
    mi_scores = make_mi_scores(X, y)

    # Combine splits (so test data gets feature engineering too)
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Lesson 2 - Mutual Information
    # X = drop_uninformative(X, mi_scores, verbose)  # XGboost makes use of some of the info in these "uninformative" features

    # Lesson 3 - Transformations
    X = X.join(mathematical_transforms(X, verbose))
    X = X.join(interactions(X, verbose))
    X = X.join(counts(X, verbose))
    # X = X.join(break_down(X, verbose))
    X = X.join(group_transforms(X, verbose))

    # Lesson 4 - Clustering
    # X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    # X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Lesson 5 - PCA
    X = X.join(pca_inspired(X, verbose))
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))
    
    # Reform splits (so target encoding can use entire data set for cross validation)
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)
    
    # Lesson 6 - Target Encoder  
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    columns_to_encode = ["MSSubClass", "Neighborhood", "YrSold", ]
    # tested "MSZoning","OverallCond", "YearBuilt", "TotSqFt", "NewSqFt" : led to worse results w XGBoost
    # "MoSold" tested: important but no score change w XGBoost
    X = X.join(encoder.fit_transform(X, y, cols=columns_to_encode))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))
    if verbose:
        print(len(columns_to_encode), "features target encoded (with crossfold validation): ")
        print("    ", columns_to_encode)
        
    # Combine splits (so test data gets feature engineering too)
    if df_test is not None:
        X_test = X_test.copy()
        X = pd.concat([X, X_test])
        
    # Mutual Information is usually too crude. So ....
    # drop features based on EDA, a feature selection algorithm, or (post fit) model feature importances
    # This is late in pipeline in case an earlier step extracts info from a feature to later be dropped.
    X = drop_unhelpful_features(X, features_to_drop=['Street', 'ThreeSeaPorch', 'Condition2'], verbose=verbose)

    # choose to label encode 0, some, or all category features depending on which algorithm will train data
    # label_encode_columns = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond'] # can't label enclode 'YrSold' and 'MoSold' with cat.codes unless I make them into categories first
    # label_encode_columns = [] # label encoding for no category features - all will be one hot encoded.
    label_encode_columns = list(X.select_dtypes(["category"])) # label encoding for all category features - good for tree ensemble algos such as XGBoost
    # label_encode_columns.remove('Fence') #one-hot encode just this one
    # final_list = list(set(item_list) - set(list_to_remove)) # if need to remove a bunch of features, do it this way
    X = label_encode_these(X, cols=label_encode_columns, verbose=verbose) 
    X = one_hot_encode_except(X, label_encoded_cols=label_encode_columns, verbose=verbose) # use for linear regression, SVM, etc. especially for nominal (unordered) categories

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    if verbose:
        print("*** END FEATURE ENGINEERING PIPELINE ***\n")
        
    if df_test is not None:
        return X, X_test
    else:
        return X


# In[26]:


df_train, df_test = preprocess_data(verbose=True) # load, clean, encode, impute
# # drop outliers here. Example:
# df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
    
X_train, X_test = create_features(df_train, df_test, verbose=True)
y_train = df_train.loc[:, "SalePrice"]

print("\nX_train shape: ", X_train.shape)
print("\nX_test shape: ", X_test.shape)
print("cross validiation score: ", score_dataset(X_train, y_train))
# X_train.head()
# X_train.filter(like='Neigh', axis=1).head()


# In[27]:


# before any feature engineering, mi_scores were:

# OverallQual     0.571457
# Neighborhood    0.526220
# GrLivArea       0.430395
# YearBuilt       0.407974
# LotArea         0.394468
#                   ...   
# PoolQC          0.000000
# MiscFeature     0.000000
# MiscVal         0.000000
# MoSold          0.000000
# YrSold          0.000000
# Name: MI Scores, Length: 79, dtype: float64

# post feature enginnering mi_scores:
mi_scores = make_mi_scores(X_train, y)
mi_scores


# In[28]:


# !pip install BorutaShap


# In[29]:


# import pprint
# import joblib
# from functools import partial

# from BorutaShap import BorutaShap


# In[30]:


# %%time
# # params = dict([('colsample_bytree', 0.11807135201147481),
# #                ('learning_rate', 0.03628302216953097),
# #                ('max_depth', 3),
# #                ('n_estimators', 1000), # tried 10,000 but took way too long! 1000 is 3m/iteration so feasible - could even bump to 2000
# #                ('reg_alpha', 23.13181079976304),
# #                ('reg_lambda', .0008746338866473539),
# #                ('subsample', 0.7875490025178415)])
# params = {'max_depth': 4, 'learning_rate': 0.008756709153431472, 'n_estimators': 3508, 'min_child_weight': 2, 'colsample_bytree': 0.2050378195385253, 'subsample': 0.40369887914955715, 'reg_alpha': 0.3301567121037565, 'reg_lambda': 0.046181862052743}

# model = XGBRegressor(random_state=2, objective='reg:squarederror', **params)
# Feature_Selector = BorutaShap(model=model,
#                               importance_measure='shap', 
#                               classification=False)

# Feature_Selector.fit(X=X_train, y=np.log(y_train), n_trials=12, random_state=2)  # tried both y=np.log(y_train) and y=y_train . . . log version worked slightly better


# In[31]:


# Feature_Selector.plot(which_features='all', figsize=(48,12))


# In[32]:


# Feature_Selector.Subset()


# # Step 4 - Hyperparameter Tuning #
# 
# At this stage, you might like to do some hyperparameter tuning with XGBoost before creating your final submission. Uncomment code block below to try some hyperparameter tuning using Optuna.

# In[33]:


# %%time
# import optuna

# X_train = create_features(df_train)
# y_train = df_train.loc[:, "SalePrice"]

# def objective(trial):
#     xgb_params = dict(
#         max_depth=trial.suggest_int("max_depth", 3, 5),
#         learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         n_estimators=trial.suggest_int("n_estimators", 2000, 5000),
#         min_child_weight=trial.suggest_int("min_child_weight", 1, 3),
#         colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, .7),
#         subsample=trial.suggest_float("subsample", 0.2, 1.0),
#         reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
#         reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
#     )
#     xgb = XGBRegressor(**xgb_params, random_state = 42) # , tree_method = 'gpu_hist'
#     return score_dataset(X_train, y_train, xgb)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20)
# xgb_params = study.best_params
# xgb_params


# The next hidden cell shows a log of many different experiments performed. If you want to develop intuition about feature engineering and a good experimentation mindset, it's great to take advantage of being able to submit up to 10 times per day. And you don't have to submit - if you're validation score gets a lot worse from adding or changing a feature, than you probably won't want to bother seeing if the score also gets worse on the leaderboard.
# 
# Following this hidden cell is the final set of hyperparameters for XGBoost, which is fed data that has already been through the feature engineering pipeline. It took weeks of experimentation to get to a pure XGBoost model (no blending) to score so well:

# In[34]:


# xgb_params = dict(
#     max_depth=4,           # maximum depth of each tree - try 2 to 10
#     learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
#     n_estimators=2800,     # number of trees (that is, boosting rounds) - try 1000 to 8000
#     min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
#     colsample_bytree=0.5,  # fraction of features (columns) per tree - try 0.2 to 1.0
#     subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
#     reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
#     reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
#     num_parallel_tree=1,   # set > 1 for boosted random forests
# )
        
# v9 trial 3/10, 4508 n_est, val 0.11651, leaderboard = 0.11999 (val slightly better, leaderboard same as before I put random state 42 - changing random state doesn't have much impact)
# v10 trial 3/10, 3508 n_est, val 0.11655, leaderboard = 0.11988
# v19 trial 3/10, 3300 n_est, val 0.11657, leaderboard = 0.11993
# v20 trial 3/10, 3800 n_est, val 0.11649, leaderboard = 0.11994 (m=1 for target encoding on this and prior models)
# v21 trial 3/10, 3800 n_est, val 0.11644, leaderboard = 0.12030 (m=2 for target encoding)
# v22 trial 3/10, 3800 n_est, val 0.11662, leaderboard = 0.12024 (m=0 for target encoding) - so looks like m=1 is best
# v25 trial 3/10, 3508 n_est, val 0.11655, leaderboard = 0.11972 (m=1 for target encoding), random state fixed
# v26 same as v25 except DF sorted by SaleCondition first val 0.11655, leaderboard=.44368
# why v26 sort drastic  impact on leaderboard? Because joining than resplitting test/train screws up test rows order if train rows order changed,
# which causes house values to be assigned to the wrong rows in submission file
# v29 add some target encodings, trial 3/10, 3508 n_est, val 0.11630, leaderboard = 0.12144
# v31 after Boruta-SHAP feature selection(YES log y), trial 3/10, 3508 n_est, val 0.11992, leaderboard = 0.12373 pruning feautures hurt score with existing hyperP.
# v35 after Boruta-SHAP feature selection (NO log y), trial 3/10, 3508 n_est, val 0.12299, leaderboard = 0.12387 little difference between using log (or not) of y
# v37 trial 3/10 after BS (YES log y, 20 BS trials), val .12053, leaderboard: 0.12219
# v38 trial 3/10 after BS (YES log y, 12 BS trials, dropped 59 features), val 0.11862, leaderboard: 12420
# v43 trial 3/10 similar to v25 with a couple of added veatures, val: 11779, leaderboard: 0.12101
# v44 trial 3/10 almost same as v25 (no neighorhood median sales price), val: 0.11643, leaderboard=12169
# v45 t 3/10 (with neigh median sales price) val 0.11631, leaderboard=0.12132 so something else is different from best score
# v46 t 3/10 same as v25 except kept the 5 features I have always been dropping due to mutal information score 0.0, val = 0.11719, leaderboard=0.11906 NEW RECORD!
# v47 t 3/10 but only features with few values are dropped: ['PoolQC', 'MiscFeature', 'Fence'], val = 0.11724, leaderboard=0.11957
# v48 t 3/10 but only 2 of 3 features with few values are dropped: ['PoolQC', 'MiscFeature'], val = 0.11712, leaderboard=0.11978
# v49 t 3/10 but only 'Fence' feature dropped, val = 0.11752, leaderboard=0.11883 (perhaps 'fence' throws off XGboost slightly, at least for leaderboard)
# v50 drop only ['Fence', 'Street', 'Threeseasonporch'] as inspired by XGBoost importances, val=0.11736, leaderboard=0.11854 ANOTHER RECORD !!
# v51 drop only ['Street', 'Threeseasonporch', 'PoolQC'], inspired by XGBoost import., val=.11704, leaderboard: 0.11873
# v52 drop only ['Street', 'Threeseasonporch'], val: 0.11755, leaderboard=0.11854 (TIED - so dropping Fence doesn't hurt or help. Weird given that it's of modest importance)
# v53 drop 7 features that XGBoost ranks least important, val: 0.11794, leaderboard: 0.11913 a hair worse than leave all features in
# v54 drop only ['Street', 'Threeseasonporch', 'Condition2'], val: 0.11706, leaderboard: 0.11838 BEAT PRIOR RECORD - using 3 of the 4 least important features according to XGboost
# v57 same as v54 but dropped 2 outliers, val: 0.11256, leaderboard: 0.12210 hugely worse. WHY??? had to change y to y_train before final fit - does that have anything to do with it?
# v58 added new feature TotalSF val: 11653, leaderboard: 0.11773 FEATURE ADD is BIG WINNER!
# v59 added TotalRooms feature val: 11750, leaderboard: 0.11918 this feature made things a lot worse. out it goes.
# v60 tested TotalOutsideSF feature val: 11722, leaderboard: 0.11882 another bad feature for XGboost
# v61 test Total_Bathrooms feature val: 0.11664, leaderboard: 0.11824 nope
# v62 test YrBuiltPlusRemod feature val: 0.11730, leaderboard: 0.11859 nope

# new notebook
# v4 val: 11703 instead of val 11653 with exact same setup as v58. The only thing I changed was that I called create_features() with both train and test.
# This causes validation score to come out differently, even though submission to leaderboard is identical and leaderboard score is identical.
# Why? Very confused - is this due to a change in random state? Is it possible that all validations scores going forward will come out closer to LB score?
# v5 one-hot encoding makes XGBoost worse with features going from 92-->377, val: 11988, leaderboard: 0.12599 though maybe just needs more estimators to deal with so many features?
# v8 test lot frontage by neighborhood imputation, val 11708, lb 0.11781 - so hurt a tiny bit; enough to be noise
# v9 test imputing mode for a bunch of features, val 11698, lb 0.11775 - hardly changed
# v10 tests imputing modes for 2 exterior features, val 0.11703 lb 0.11770
# v11 impute mode ext1 only, val 0.11703, lb 0.11771
# v12 impute mode ext2 only, val 0.11703, lb 0.11773
# v13 impute mode 2 Exts, SaleType, val 0.11703, lb 0.11770
# v15 25 label encoded features b4 one hot, val: 0.11825, lb 0.12210 an improvement over v5 when the 25 features were not label encoded
# v17 different method for label encoding 25 features b4 one-hot, val .11773, lb 0.12015 which is pretty good for 256 features on XGBoost!
# what is weird though is why should this be any different from v15? does method of doing label encoding (pandas vs scikit learn) actually matter?
# v19 impute neighborhood mode for MSZoning NAs val: 11703, lb 0.11770 (no impact?)
# v20 fix ordering of Functional feature's values: val 11703, lb 0.11735 - wow, even though no val change, lb improved a signficantly from minor change!
# v21 dropped Utilities val 11743 lb 0.11847 - wow! has just 1 row without full utilities yet has dramatic impact!
# v22 hot encode BsmtQual val 11703 lb 0.11735 - whoops - coding error didn't one hot encode
# v23 hot encode BsmtCond val 11703 lb 0.11735 - whoops - coding error didn't one hot encode
# v25 hot encode BsmtQual val 11722 lb 0.11934 so one hot encode was way worse
# v26 changed Electrical order slightly val 11703 lb 0.11735 - no difference which is not surprising as only 1 row had mixed
# v27 one-hot encode Fence val 0.11668, 0.11873 - lb got worse even though validation was better. odd.
# v28 changed Fence order val 0.11702 lb 0.11725 a tiny boost
# xgb_params = {'max_depth': 4, 'learning_rate': 0.0088, 'n_estimators': 3400, 'min_child_weight': 2, 'colsample_bytree': 0.205, 'subsample': 0.4037, 'reg_alpha': 0.3301567121037565, 'reg_lambda': 0.046181862052743}  
# v42 tweaked favorite (just above) val 0.11701 lb 0.11746 slightly worse on lb
# optuna: 11807 (with GPU)
# # the one I've used for over a week:
# xgb_params = {'max_depth': 4, 'learning_rate': 0.008756709153431472, 'n_estimators': 3508, 'min_child_weight': 2, 'colsample_bytree': 0.2050378195385253, 'subsample': 0.40369887914955715, 'reg_alpha': 0.3301567121037565, 'reg_lambda': 0.046181862052743}  



# xgb_params = {'max_depth': 3, 'learning_rate': 0.02114934913088701, 'n_estimators': 2523, 'min_child_weight': 2, 'colsample_bytree': 0.5547814801304602, 'subsample': 0.5239731194871952, 'reg_alpha': 0.00026074805799699554, 'reg_lambda': 0.006531563710312441}
# trial 2/10 optuna: 0.11713 without generating test set
# v29 new hyperP 2/10 val: 0.11699, lb: 0.12258 worse than expected lb given good val. Maybe the prior one is overfitted to lb? This was just out of 10 trials so I'll see what happens with 200 trials overnight.

# v30 optuna 0.11336 trial 164/200, val 0.11394, lb 0.11935
# xgb_params ={'max_depth': 4, 'learning_rate': 0.006698471590173297, 'n_estimators': 4567, 'min_child_weight': 2, 'colsample_bytree': 0.24129246054168077, 'subsample': 0.44513118412794983, 'reg_alpha': 0.0004519705422846089, 'reg_lambda': 0.00020474392345395197}

# v31 optuna 0.11363 Trial 155/200, val 0.11404, lb 0.11936
# xgb_params = {'max_depth': 4, 'learning_rate': 0.006831727298305696, 'n_estimators': 4621, 'min_child_weight': 2, 'colsample_bytree': 0.2566112440621832, 'subsample': 0.4447205088335255, 'reg_alpha': 0.0007900851888091328, 'reg_lambda': 0.00013972524409906255}

# v32 optuna 0.11386 Trial 124/200, val 0.11385, lb 0.11950
# xgb_params = {'max_depth': 4, 'learning_rate': 0.005195780138622257, 'n_estimators': 4886, 'min_child_weight': 2, 'colsample_bytree': 0.2208822800528527, 'subsample': 0.6144261364605068, 'reg_alpha': 0.00421214459632269, 'reg_lambda': 0.000591878759903201}

# v33 optuna 0.11492 trial 27/200 val 0.11562, lb 0.11945
# xgb_params = {'max_depth': 4, 'learning_rate': 0.012805027367186405, 'n_estimators': 4794, 'min_child_weight': 2, 'colsample_bytree': 0.20165415726310923, 'subsample': 0.47523199921737347, 'reg_alpha': 0.00033877374644280514, 'reg_lambda': 0.31708616538063455}

# # v34 optuna 0.11488 trial 44/200 val 11450, lb 0.12194
# xgb_params = {'max_depth': 4, 'learning_rate': 0.017081472889643453, 'n_estimators': 4589, 'min_child_weight': 2, 'colsample_bytree': 0.27559556105601635, 'subsample': 0.5631124749718487, 'reg_alpha': 0.0052208767846268775, 'reg_lambda': 0.07070285925896815}

# v35 optuna 0.11436 trial 63/200 val 0.11430, lb 0.12015
# xgb_params = {'max_depth': 4, 'learning_rate': 0.008067194679663887, 'n_estimators': 4917, 'min_child_weight': 2, 'colsample_bytree': 0.23380166692436696, 'subsample': 0.6182032462404947, 'reg_alpha': 0.0030756455512833247, 'reg_lambda': 0.00848093006044966}

# v36 optuna 0.11437 trial 70/200 val 0.11453 lb 0.12017 
# xgb_params = {'max_depth': 4, 'learning_rate': 0.004152609847352432, 'n_estimators': 4928, 'min_child_weight': 2, 'colsample_bytree': 0.24500537489579363, 'subsample': 0.7162925206124766, 'reg_alpha': 0.00935417891464248, 'reg_lambda': 0.0017248422134794907}

# v37 optuna 0.11392 trial 129/200 val  0.11412 lb 0.11981
# xgb_params = {'max_depth': 4, 'learning_rate': 0.007247015583102812, 'n_estimators': 4910, 'min_child_weight': 2, 'colsample_bytree': 0.2186797801083439, 'subsample': 0.5730531634592627, 'reg_alpha': 0.001428890225504756, 'reg_lambda': 0.0006153299687041702}

# v38 optuna 0.11350 trial 198/200 val 0.11361, lb 0.11907
# xgb_params = {'max_depth': 4, 'learning_rate': 0.007081042884678392, 'n_estimators': 4491, 'min_child_weight': 2, 'colsample_bytree': 0.2233667118614051, 'subsample': 0.4572573800301871, 'reg_alpha': 0.00033322844976968274, 'reg_lambda': 0.00016251307640631252}

# v39 optuna .11364 trial 193/200 val 0.11379, lb 0.11855 finally one that is getting close - should tweak?
# xgb_params = {'max_depth': 4, 'learning_rate': 0.0064678641044469114, 'n_estimators': 4460, 'min_child_weight': 2, 'colsample_bytree': 0.22290140130032263, 'subsample': 0.45774929968395833, 'reg_alpha': 0.0003479070886226114, 'reg_lambda': 0.00017416758313069236}

# v40 trial 193/200 hyperP tweaks val .11357, lb 0.11889 made lb score worse
# xgb_params = {'max_depth': 4, 'learning_rate': 0.0068, 'n_estimators': 3800, 'min_child_weight': 2, 'colsample_bytree': 0.22290140130032263, 'subsample': 0.45774929968395833, 'reg_alpha': 0.0003479070886226114, 'reg_lambda': 0.00017416758313069236}

# v41 optuna 0.11355 trial 183/200 val 0.11370, lb 0.11915
# xgb_params = {'max_depth': 4, 'learning_rate': 0.007263975459755825, 'n_estimators': 4657, 'min_child_weight': 2, 'colsample_bytree': 0.22275346430359858, 'subsample': 0.4412986244746556, 'reg_alpha': 0.0001939620035983133, 'reg_lambda': 0.00010185983943986789}

# xgb_params = {'max_depth': 4, 'learning_rate': 0.00602, 'n_estimators': 4100, 'min_child_weight': 2, 'colsample_bytree': 0.22, 'subsample': 0.45, 'reg_alpha': 0.00035, 'reg_lambda': 0.0002}
# v43 trial 193/200 hyperP tweaks val 0.11408, optuna .11364, lb 0.11893 slightly worse
# v44 trial 193/200 hyperP tweaks val 0.11397, optuna .11364, lb 0.11828 slightly better with lower learning rate of .0062
# v45 trial 193/200 hyperP tweaks val 0.11423, optuna .11364, lb 0.11811 with lower learning rate of .0058, n_est 3600
# v46 trial 193/200 hyperP tweaks val 0.11401, optuna .11364, lb 0.11825 with learning_rate .0058, n_est 4600
# v47 trial 193/200 hyperP tweaks val 0.11423, optuna .11364, lb 0.11802 with learning_rate .0059, n_est 3700
# v48 trial 193/200 hyperP tweaks val 0.11435, optuna .11364, lb 0.11806 with learning_rate .0059, n_est 3300
# v49 trial 193/200 hyperP tweaks val 0.11503, optuna .11364, lb 0.11908 with learning_rate .0051, n_est 3300
# v50 trial 193/200 hyperP tweaks val 0.11413, optuna .11364, lb 0.11805 with learning_rate .00605, n_est 3800
# v51 trial 193/200 hyperP tweaks val 0.11410, optuna .11364, lb 0.11795 with learning_rate .00602, n_est 4100

# # the great Hyperparameters I've used for over a week and can't seem to beat:
# xgb_params = {'max_depth': 4, 'learning_rate': 0.008756709153431472, 'n_estimators': 3508, 'min_child_weight': 2, 'colsample_bytree': 0.2050378195385253, 'subsample': 0.40369887914955715, 'reg_alpha': 0.3301567121037565, 'reg_lambda': 0.046181862052743}  
# v28 changed Fence order val 0.11702 lb 0.11725 a tiny boost

# v52 favorite hyperP n_est 3450 val 11706 lb 0.11727
# v53 favorite hyperP n_est 3530 val 11703 lb 0.11726
# v54 favorite hyperP n_est 3495 val 0.11702 lb 0.11725
# v55 favorite hyperP n_est 3480 val 0.11702 lb 0.11726
# v56 favorite hyperP n_est 3515 val 0.11702 lb 0.11724 tiny improvement, so 3515 is new n_est
# v57 favorite hyperP lrn_rate .00875 val 0.11684 0.11684104079783438    lb 0.11717 NEW RECORD !
# v58 favorite hyperP lrn_rate .00873 val 0.11676 lb 0.11723
# v59 favorite hyperP subsample .4 val 0.11666 lb 0.11779
# v60 favorite hyperP subsample .41 val 0.11655 lb 0.11745
# v61 favorite hyperP subsample .4037 val 11682 lb 0.11722
# v62 favorite hyperP dropped PoolQC, PoolArea val 0.11698 lb 0.11833
# v63 favorite hyperP dropped Fence val 0.11703 lb 0.11776
# v64 removed interaction features val 11729 lb 0.11827
# v65 2 interaction features val 0.11742 lb 0.11730 slightly worse
# v66 HouseType interaction features val 0.11663 lb 0.11848 bit worse - likes other interaction better
# v67 NH, Bldg interaction features val 0.11709 lb 0.11799
# v68 MSZ, Bldg interaction features val 0.11689 lb 0.11909
# v69 BRAbvGr, Bldg interaction features val 0.11546 lb 0.11775 weird that saw big improvement with val, but worse with lb
# v70 is v69 less 3 least used BRAG ftrs val .11600, lb 0.11813
# v71 added OverallQual group transform val 0.11670, lb 0.11677 *** NEW RECORD ***
# v72 added OverallCond group transform val 0.11626, lb 0.11705
# v74 is v71 with more data cleaning val 0.11670 lb 0.11677 (as expected, data clean change had no effect)
# v75 MedQualLotArea group transform val 0.11626 lb 0.11770
# v76 MedYearBuiltLivArea group transform val 0.11678 lb 0.11725
# v77 MedGarageCarArea group transform val 0.11678 lb 0.11734
# v78 MoSold target encoding val 0.11671 lb 0.11681 very high importance yet was a (very) slight hit to score. Should I keep?
# v79 is v78 with slight tinkering HyperP val .11652 lb 0.11680 'learning_rate': 0.00878, 'n_estimators': 3400
# v80 specified early_stopping_rounds=300 val 0.11670 lb 0.11677 whoops - added fit_params in score_dataset, but not elsewhere when fitting model
# v81 specified early_stopping_rounds=50 val 0.11670 lb 0.11677 whoops - added fit_params in score_dataset, but not elsewhere when fitting model
# v82 early_stopping_rounds: 300 val 0.11670 lb b 0.11677 - was same score anyway after doing it correctly
# v83 early_stopping_rounds: 50 val 0.11670 lb 0.11677 seemed identical home prices. Not sure if I got everything right with whether to do rmse or rmsle or log_y, etc.
# v84 cluster_labels Kmeans n=20 val 0.11634 lb 0.11718
# v85 cluster_labels Kmeans n=10 val 0.11618, lb 0.11751
# v86 cluster_labels Kmeans n=30 val 0.11621 lb 0.11700
# v87 cluster_labels Kmeans n=50 val 0.11649 lb 0.11738
# v88 cluster_labels Kmeans n=14 val 0.11615 lb 0.11701
# v89 MoSold_enc replaces MoSold val 0.11577 lb 0.11703 another instance of improving val, worse lb
# v90 YrSold_enc val 0.11721 lb 0.11653 NEW RECORD
# v91 YrSold_enc replaces YrSold val 0.11644 lb 0.11667 a hair worse
# v92 OverallQual_enc val .11714 lb 0.11657
# v93 OverallQual_enc replaces OverallQual val 0.11707 lb 0.11747
# v94 OverallCond_enc val 0.11751 lb 0.11752
# v95 cluster_labels Bed/Bath/SQFT n=20 val 0.11721 lb whoops - cluster was commented out
# v96 cluster_labels Bed/Bath/SQFT n=20 val 0.11721 lb 0.11756


# In[35]:


get_ipython().run_cell_magic('time', '', '\ndf_train, df_test = preprocess_data() # load, clean, encode, impute\nX_train, X_test = create_features(df_train, df_test)\ny_train = df_train.loc[:, "SalePrice"]\n\nxgb_params = {\'max_depth\': 4, \'learning_rate\': 0.00875, \'n_estimators\': 3515, \'min_child_weight\': 2, \'colsample_bytree\': 0.2050378195385253, \'subsample\': 0.40369887914955715, \'reg_alpha\': 0.3301567121037565, \'reg_lambda\': 0.046181862052743}\n\nxgb = XGBRegressor(**xgb_params, random_state=42)\n\nprint("\\nX_train shape: ", X_train.shape)\nprint("X_test shape: ", X_test.shape)\n\nprint("score: ", score_dataset(X_train, y_train, xgb))\n')


# In[36]:


# fit the model to entire dataset and then save to submission file so it can be submitted to leaderboard

df_train, df_test = preprocess_data() # load, clean, encode, impute
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "SalePrice"]

xgb = XGBRegressor(**xgb_params, random_state=42)
# XGB minimizes MSE, but competition loss is RMSLE
# So, we need to log-transform y to train and exp-transform the predictions

xgb.fit(X_train, np.log(y_train))
predictions = np.exp(xgb.predict(X_test))

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[37]:


output.head()


# In[38]:


# feature importances from the XGBoost algorithm

feature_important = xgb.get_booster().get_score(importance_type='weight')  # 'weight' seems to produce most sensible results, 'gain' is nearly as good, while 'cover' is horrible
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.plot(kind='barh', figsize = (6,24))


# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/221677) to chat with other Learners.*
