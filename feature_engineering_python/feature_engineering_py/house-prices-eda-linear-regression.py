#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">House Prices: EDA + Linear Regression</h1>
# </div>
# 
# 
# ## Problem Type
# 
# Linear Regression
# 
# ## Evaluation Metric
# 
# RMSE - [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

# 

# - https://www.kaggle.com/code/nkitgupta/feature-engineering-and-feature-selection
# - https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection
# - https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition
# - https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1
# - https://www.kaggle.com/code/ravi20076/houseprice-custom-pipelines/notebook

# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Import Libraries</h1>
# </div>

# In[1]:


from typing import List, Set, Dict, Tuple, Optional

import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import impute
from sklearn import preprocessing
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn import model_selection, metrics

# from xgboost import XGBRegressor

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Visualization Libraries
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import missingno as msno


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Configuration</h1>
# </div>

# In[2]:


class Config:
    path:str = "../input/house-prices-advanced-regression-techniques"
    gpu:bool = False
    model_type:str = "xgb"  # (xgb, cat, lgbm, keras)
    model_name:str = "xgb1"
    debug:bool = False
    competition:str = "Housing Prices"
    calc_probability:bool = False
    seed:int = 42
    N_ESTIMATORS:int = 100  # 5000
    N_FOLDS:int = 5
    EPOCHS:int = 10


# In[3]:


TARGET = "SalePrice"


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Theme</h1>
# </div>
# 
# ### Generate Color Palette
# - https://coolors.co
# - https://color.adobe.com/create/color-wheel

# In[4]:


from itertools import cycle

plt.style.use("fivethirtyeight")  # ggplot, fivethirtyeight
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# In[5]:


mpl.rcParams["font.size"] = 16

theme_colors = ["#ffd670", "#70d6ff", "#ff4d6d", "#8338ec", "#90cf8e"]
theme_palette = sns.set_palette(sns.color_palette(theme_colors))

sns.palplot(sns.color_palette(theme_colors), size=1.5)
plt.tick_params(axis="both", labelsize=0, length=0)


# In[6]:


my_colors = ["#CDFC74", "#F3EA56", "#EBB43D", "#DF7D27", "#D14417", "#B80A0A", "#9C0042"]
sns.palplot(sns.color_palette(my_colors), size=0.8)


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Library</h1>
# </div>
# 
# Create a personal library of reusable functions

# In[7]:


def read_data(path:str, analyze:bool=True):
    data_dir = Path(path)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")

    if analyze:
        print("=== Shape of Data ===")
        print(f" train data: Rows={train.shape[0]}, Columns={train.shape[1]}")
        print(f" test data : Rows={test.shape[0]}, Columns={test.shape[1]}")

        print("\n=== Train Data: First 5 Rows ===\n")
        display(train.head())
        print("\n=== Train Column Names ===\n")
        display(train.columns)
        print("\n=== Features/Explanatory Variables ===\n")
        eval_features(train)
        print("\n === Skewness ===\n")
        check_skew(train)
    return train, test, submission_df

def create_submission(model_name:str, target:str, preds:List[float], seed:int=42, nfolds:int=5) -> pd.DataFrame:
    sample_submission[target] = preds

    if len(model_name) > 0:
        fname = f"submission_{model_name}_k{nfolds}_s{seed}.csv"
    else:
        fname = "submission.csv"

    sample_submission.to_csv(fname, index=False)

    return sample_submission

def show_classification_scores(gt:List[int], yhat:List[int]) -> None:
    accuracy = accuracy_score(gt, yhat)
    precision = precision_score(gt, yhat)
    recall = recall_score(gt, yhat)
    f1 = f1_score(gt, yhat)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
    

def label_encoder(train:pd.DataFrame, test:pd.DataFrame, columns:List[str]) -> (pd.DataFrame,pd.DataFrame):
    for col in columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        train[col] = preprocessing.LabelEncoder().fit_transform(train[col])
        test[col] = preprocessing.LabelEncoder().fit_transform(test[col])
    return train, test   

def create_folds(df:pd.DataFrame, TARGET:str, n_folds:int=5, seed:int=42) -> pd.DataFrame:
    print(f"TARGET={TARGET}, n_folds={n_folds}, seed={seed}")
    df["fold"] = -1

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # kf = GroupKFold(n_splits=Config.N_FOLDS)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, df[TARGET])):
        df.loc[valid_idx, "fold"] = fold

    # df.to_csv(f"train_fold{num_folds}.csv", index=False)
    return df

def show_fold_scores(scores:List[float]) -> (float, float):
    cv_score = np.mean(scores)  # Used in filename
    std_dev = np.std(scores)
    print(
        f"Scores -> Adjusted: {np.mean(scores) - np.std(scores):.8f} , mean: {np.mean(scores):.8f}, std: {np.std(scores):.8f}"
    )
    return cv_score, std_dev


def feature_distribution_types(df:pd.DataFrame, display:bool=True) -> (List[str], List[str]):
    continuous_features = list(df.select_dtypes(include=['int64', 'float64', 'uint8']).columns)
    categorical_features = list(df.select_dtypes(include=['object', 'bool']).columns)
    if display:
        print(f"Continuous Features={continuous_features}\n")
        print(f"Categorical Features={categorical_features}")
    return continuous_features, categorical_features   

def show_cardinality(df:pd.DataFrame, features:List[str]) -> None:
    print("=== Cardinality ===")
    print(df[features].nunique())

## === Model Support ===    

from scipy.stats import mode


def merge_test_predictions(final_test_predictions:List[float], calc_probability:bool=True):

    if Config.calc_probability:
        print("Mean")
        result = np.mean(np.column_stack(final_test_predictions), axis=1)
    else:
        print("Mode")
        mode_result = mode(np.column_stack(final_test_predictions), axis=1)
        result = mode_result[0].ravel()

    return result    


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">EDA Library</h1>
# </div>
# 
# Functions specific to an EDA kernel

# In[8]:


def show_missing_features(df:pd.DataFrame) -> None:
    missing_vals = df.isna().sum().sort_values(ascending=False)
    print(missing_vals[missing_vals > 0])


def show_duplicate_records(df:pd.DataFrame) -> None:
    dups = df.duplicated()
    print(dups.sum())


def eval_features(df:pd.DataFrame) -> None:
    ## Separate Categorical and Numerical Features
    categorical_features = list(
        df.select_dtypes(include=["category", "object"]).columns
    )
    continuous_features = list(df.select_dtypes(include=["number"]).columns)

    print(f"Continuous features: {continuous_features}")
    print(f"Categorical features: {categorical_features}")
    print("\n --- Cardinality of Categorical Features ---\n")

    for feature in categorical_features:
        cardinality = df[feature].nunique()
        if cardinality < 10:
            print(f"{feature}: cardinality={cardinality}, {df[feature].unique()}")
        else:
            print(f"{feature}: cardinality={cardinality}")
    all_features = categorical_features + continuous_features
    return all_features, categorical_features, continuous_features


def show_feature_importance(feature_importance_lst) -> None:
    fis_df = pd.concat(feature_importance_lst, axis=1)

    fis_df.sort_values("0_importance", ascending=True).head(40).plot(
        kind="barh", figsize=(12, 12), title="Feature Importance Across Folds"
    )
    plt.show()


def show_feature_target_crosstab(df:pd.DataFrame, feature_lst:List[str], target:str) -> None:
    for feature in feature_lst:
        print(f"\n=== {feature} vs {target} ===\n")
        display(
            pd.crosstab(df[feature], df[target], margins=True)
        )  # display keeps bold formatting


def show_cardinality(df:pd.DataFrame, features:List[str]) -> None:
    print("=== Cardinality ===")
    print(df[features].nunique())


def show_unique_features(df:pd.DataFrame, features:List[str]) -> None:
    for col in features:
        print(col, sorted(df[col].dropna().unique()))


def feature_distribution_types(df:pd.DataFrame, display:bool=True) -> (List[str], List[str]):
    continuous_features = list(
        df.select_dtypes(include=["int64", "float64", "uint8"]).columns
    )
    categorical_features = list(df.select_dtypes(include=["object", "bool"]).columns)
    if display:
        print(f"Continuous Features={continuous_features}\n")
        print(f"Categorical Features={categorical_features}")
    return continuous_features, categorical_features

def describe(X:pd.DataFrame) -> None:
    desc = X.describe()
    desc.loc['var'] = X.var(numeric_only=True).tolist()
    desc.loc['skew'] = X.skew(numeric_only=True).tolist()
    desc.loc['kurt'] = X.kurtosis(numeric_only=True).tolist()

    with pd.option_context('display.precision', 2):
        style = desc.transpose().style.background_gradient(cmap='coolwarm') #.set_precision(4)
    display(style)
    
def check_skew(df:pd.DataFrame) -> None:
    skew = df.skew(skipna=True,numeric_only=True).sort_values(ascending=False)
    print(skew)


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Load Train/Test Data and Analyze</h1>
# </div>
# 
# Support datatable for large datasets.
# Using datatable is described in [Tutorial on reading datasets](https://www.kaggle.com/hiro5299834/tutorial-on-reading-datasets)

# In[9]:


get_ipython().run_cell_magic('time', '', 'train, test, sample_submission = read_data(Config.path, analyze=True)\n')


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Explore the Data</h1>
# </div>

# In[10]:


train.columns


# In[11]:


# train_df.head(20)
train.sample(5)


# ## Skewness

# In[12]:


skew = train.skew(skipna=True, numeric_only=True).sort_values(ascending=False)
skew


# In[13]:


kurtosis = train.kurtosis(skipna=True, numeric_only=True).sort_values(ascending=False)
kurtosis


# ## Some numerical variables may actually be categorical

# ## Determine Cardinality of Categorical Variables

# In[14]:


ax = sns.scatterplot(
    x="LotArea",
    y="SalePrice",
    hue="YearBuilt",
    data=train,
)
ax.set_title("LotArea vs. SalePrice")
plt.show()


# In[15]:


missing_vals = train.isna().sum()
missing_vals[missing_vals > 0]


# In[16]:


corr = train[["YearBuilt", "LotArea"]].dropna().corr()
corr


# In[17]:


# plt.figure(figsize=(15, 15))
# sns.heatmap(train.corr(), annot=True, cmap="PuBuGn")
# plt.show()


# In[18]:


train.select_dtypes(exclude=["object","bool"]).corr()


# In[19]:


get_ipython().run_cell_magic('time', '', 'num_feats = list(train.select_dtypes(exclude=["object","bool"]))\n\ncorr = train[num_feats].corr()\nmask = np.triu(train[num_feats].corr())\n\nsns.set(font_scale=1.1)\nplt.figure(figsize=(20, 20), dpi=140)\nsns.heatmap(corr, annot=True, fmt=\'.1f\', \n            cmap=\'coolwarm\', \n            square=True, \n            mask=mask, \n            linewidths=1,\n            cbar=False)\nplt.show()\n')


# In[20]:


cat_list = train.nunique()[train.nunique() < 30][:-1]
cat_list


# In[21]:


def get_skewed_distributions(df:pd.DataFrame):
    pass


# In[22]:


skew_cols = train.select_dtypes(exclude="object").skew().sort_values(ascending=False)
skew_cols = pd.DataFrame(skew_cols.loc[skew_cols > 0.75]).rename(
    columns={0: "Skew before"}
)
skew_cols


# ## Box-cox transformation
# 
# Impute or remove null values first

# In[23]:


# Box-cox transformation
def box_cox_transform(df:pd.DataFrame) -> None:
    t = df.copy()
    for i in skew_cols.index.tolist():
        t[i] = boxcox1p(t[i], boxcox_normmax(t[i] + 1))

    skew_df = pd.concat([skew_cols, t[skew_cols.index].skew()], axis=1).rename(
        columns={0: "After"}
    )
    skew_df.head()


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Preprocessing</h1>
# </div>

# In[24]:


cont_features, cat_features = feature_distribution_types(train, display=True)
show_cardinality(train, cat_features)

cont_features.remove(TARGET) # SalePrice


# ### Impute Categorical Features

# In[25]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

train[cat_features] = imputer.fit_transform(train[cat_features])
test[cat_features] = imputer.transform(test[cat_features])


# ### Impute Numerical Features

# In[26]:


# imputer = SimpleImputer(strategy="mean")
imputer = SimpleImputer(strategy="median")  # median is more robust to outliers

train[cont_features] = imputer.fit_transform(train[cont_features])
test[cont_features] = imputer.transform(test[cont_features])


# ### Encode Categorical Features

# In[27]:


train, test = label_encoder(train, test, cat_features)


# In[28]:


FEATURES = cont_features + cat_features


# In[29]:


train = create_folds(train, TARGET, Config.N_FOLDS)


# <div style="background-color:rgba(255, 87, 51, 0.9);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Train Model with Cross Validation</h1>
# </div>

# In[30]:


def train_cv_model(
    df:pd.DataFrame,
    test:pd.DataFrame,
    get_model_fn,
    FEATURES:List[str],
    TARGET:str,
    calc_probability:bool,
    rowid,
    params,
    n_folds:int=5,
    seed:int=42,
):

    final_test_predictions = []
    final_valid_predictions = {}
    fold_scores = []  # Scores of Validation Set
    feature_importance_lst = []

    test = test[FEATURES].copy()

    for fold in range(n_folds):
        print(10 * "=", f"Fold {fold+1}/{n_folds}", 10 * "=")

        start_time = time.time()

        xtrain = df[df.fold != fold].reset_index(
            drop=True
        )  # Everything not in validation fold
        xvalid = df[df.fold == fold].reset_index(drop=True)
        xtest = test.copy()

        valid_ids = xvalid.Id.values.tolist()  # Id's of everything in validation fold

        ytrain = xtrain[TARGET]
        yvalid = xvalid[TARGET]

        xtrain = xtrain[FEATURES]
        xvalid = xvalid[FEATURES]
        xtrain.head()

        model = get_model_fn()

        model.fit(
            xtrain,
            ytrain,
        )

        # Mean of the predictions
        #         preds_valid = model.predict(xvalid)
        #         test_preds = model.predict(xtest)
        if calc_probability:
            preds_valid = model.predict_proba(xvalid)[:, 1]
            test_preds = model.predict_proba(xtest)[:, 1]
        else:
            preds_valid = model.predict(xvalid)
            test_preds = model.predict(xtest)

        preds_valid_class = model.predict(xvalid)

        final_test_predictions.append(test_preds)
        final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))

#         fold_score = metrics.accuracy_score(yvalid, preds_valid)  # Validation Set Score
#         fold_score = metrics.roc_auc_score(yvalid, preds_valid)  # Validation Set Score
        fold_score = metrics.mean_absolute_error(
            yvalid, preds_valid
        )  # Validation Set 
        fold_scores.append(fold_score)
        #         importance_list.append(model.coef_.ravel())

        # Feature importance
        fi = pd.DataFrame(
            index=FEATURES,
            data=model.coef_.ravel(),
            columns=[f"{fold}_importance"],
        )
        feature_importance_lst.append(fi)

        run_time = time.time() - start_time

        print(f"fold: {fold+1}, Accuracy: {fold_score}, Run Time: {run_time:.2f}")

    return (
        model,
        feature_importance_lst,
        fold_scores,
        final_valid_predictions,
        final_test_predictions,
    )


# In[31]:


model_name = "lreg"


# In[32]:


def get_linear_regression_model():
#     model = LogisticRegression(
#         max_iter=1000, C=0.0001, penalty="l2", solver="newton-cg"
#     )
    model = linear_model.LinearRegression()
    return model


# In[33]:


get_ipython().run_cell_magic('time', '', '(\n    model,\n    feature_importance_lst,\n    fold_scores,\n    final_valid_predictions,\n    final_test_predictions,\n) = train_cv_model(\n    train,\n    test,\n    get_linear_regression_model,\n    FEATURES,\n    TARGET,\n    Config.calc_probability,\n    "id",\n    {},\n    Config.N_FOLDS,\n    Config.seed,\n)\n')


# In[34]:


fold_scores[:5]


# In[35]:


show_fold_scores(fold_scores)


# In[36]:


test_preds = merge_test_predictions(final_test_predictions)


# In[37]:


print(f"=== Model: {model_name} ===")
create_submission(model_name, TARGET, test_preds)


# ### Ridge Regression

# In[38]:


model_name="ridge"


# In[39]:


def get_ridge_regression_model():
#     model = LogisticRegression(
#         max_iter=1000, C=0.0001, penalty="l2", solver="newton-cg"
#     )
    model = linear_model.Ridge()
    return model


# In[40]:


get_ipython().run_cell_magic('time', '', '(\n    model,\n    feature_importance_lst,\n    fold_scores,\n    final_valid_predictions,\n    final_test_predictions,\n) = train_cv_model(\n    train,\n    test,\n    get_ridge_regression_model,\n    FEATURES,\n    TARGET,\n    Config.calc_probability,\n    "id",\n    {},\n    Config.N_FOLDS,\n    Config.seed,\n)\n')


# In[41]:


show_fold_scores(fold_scores)
test_preds = merge_test_predictions(final_test_predictions)
print(f"=== Model: {model_name} ===")
create_submission(model_name, TARGET, test_preds)

