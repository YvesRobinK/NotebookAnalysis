#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill">
#     <h1 style="text-align: center;padding: 12px 0px 12px 0px;">TPS Aug 22: XGBoost+LightGBM+CatBoost+Optuna</h1>
# </div>
# 
# 
# ## References
# 
# - https://www.kaggle.com/code/desalegngeb/tps08-logisticregression-and-some-fe
# - https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense
# - https://www.kaggle.com/code/samuelcortinhas/tps-aug-22-failure-prediction
# - https://www.kaggle.com/code/kartushovdanil/tps-aug-22-advanced-eda-modeling
# - https://www.kaggle.com/code/pourchot/hunting-for-missing-values
# - https://www.kaggle.com/code/takanashihumbert/tps-aug22-lb-0-59013

# In[1]:


# Black formatter https://black.readthedocs.io/en/stable/

# ! pip install nb-black > /dev/null

# %load_ext lab_black


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install feature-engine\n')


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Import Libraries</h1>
# </div>

# In[3]:


import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression, HuberRegressor

from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Visualization Libraries
import matplotlib.pylab as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from itertools import cycle

plt.style.use("ggplot")  # ggplot, fivethirtyeight
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Configuration</h1>
# </div>

# In[4]:


class Config:
    path = "../input/tabular-playground-series-aug-2022"
    gpu = False
    debug = False
    optimize = False
    n_optuna_trials = 10
    model_type = "tf"  # (xgb, cat, lgbm, keras)
    model_name = "tf1"
    competition = "tps-aug-2022"
    calc_probability = True
    seed = 42
    N_ESTIMATORS = 500  # 100, 300, 2000, 5000 GBDT

    batch_size = 64
    epochs = 25
    N_FOLDS = 5  # 5,10,15
    SEED_LENGTH = 1  # 5,10


# ### The target/dependent variable in the dataset

# In[5]:


TARGET = "failure"


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Library</h1>
# </div>
# 
# Creating a few functions that will be reused in each project.
# 
# I need to be better with [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) on Kaggle.

# In[6]:


def read_data(path):
    data_dir = Path(path)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")

    print(f"train data: Rows={train.shape[0]}, Columns={train.shape[1]}")
    print(f"test data : Rows={test.shape[0]}, Columns={test.shape[1]}")
    return train, test, submission_df


# In[7]:


def create_submission(model_name, target, preds):
    sample_submission[target] = preds

    if len(model_name) > 0:
        fname = f"submission_{model_name}.csv"
    else:
        fname = "submission.csv"
    print(f"Saving submission: {fname}")

    sample_submission.to_csv(fname, index=False)

    return sample_submission


# In[8]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def show_scores(gt, yhat):
    accuracy = accuracy_score(gt, yhat)
    precision = precision_score(gt, yhat)
    recall = recall_score(gt, yhat)
    f1 = f1_score(gt, yhat)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")


# In[9]:


from sklearn.preprocessing import LabelEncoder


def label_encoder(train, test, columns):
    for col in columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        train[col] = LabelEncoder().fit_transform(train[col])
        test[col] = LabelEncoder().fit_transform(test[col])
    return train, test


# In[10]:


from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder(train, test, columns):
    for col in columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        train[col] = OneHotEncoder().fit_transform(train[col])
        test[col] = OneHotEncoder().fit_transform(test[col])
    return train, test


# In[11]:


def show_missing_features(df):
    missing_vals = df.isna().sum().sort_values(ascending=False)
    print(missing_vals[missing_vals > 0])


# In[12]:


def show_duplicate_records(df):
    dups = df.duplicated()
    print(dups.sum())


# In[13]:


def create_folds(df, TARGET, n_folds=5, seed=42):
    print(f"TARGET={TARGET}, n_folds={n_folds}, seed={seed}")
    df["fold"] = -1

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    #     kf = GroupKFold(n_splits=Config.N_FOLDS)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, df[TARGET])):
        df.loc[valid_idx, "fold"] = fold

    # df.to_csv(f"train_fold{num_folds}.csv", index=False)
    return df


# In[14]:


def create_group_kfolds(df, TARGET, n_folds=5, seed=42):
    print(f"TARGET={TARGET}, n_folds={n_folds}, seed={seed}")
    df["fold"] = -1

    #     kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kf = GroupKFold(n_splits=Config.N_FOLDS)

    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(df, df[TARGET], df.product_code)
    ):
        df.loc[valid_idx, "fold"] = fold

    # df.to_csv(f"train_fold{num_folds}.csv", index=False)
    return df


# In[15]:


def show_fold_scores(scores):
    cv_score = np.mean(scores)  # Used in filename
    std_dev = np.std(scores)
    print(
        f"Scores -> Adjusted: {np.mean(scores) - np.std(scores):.8f} , mean: {np.mean(scores):.8f}, std: {np.std(scores):.8f}"
    )
    return cv_score, std_dev


# ## Optuna Objective Functions

# In[16]:


def objective_xgb(trial, X_train, X_valid, y_train, y_valid):

    xgb_params = {
        #         "objective": trial.suggest_categorical("objective", ["multi:softmax"]),
        #         "eval_metric": "mlogloss",
        #         "objective": "multi:softmax",
        "eval_metric": "auc",  # auc, rmse, mae
        "objective": "binary:logistic",
        #         "enable_categorical": trial.suggest_categorical("use_label_encoder", [True]),
        "use_label_encoder": trial.suggest_categorical("use_label_encoder", [False]),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 5000, 100),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 0.25),
        "subsample": trial.suggest_float("subsample", 0.1, 1, step=0.01),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 1, 20),  # 10
        "gamma": trial.suggest_float("gamma", 0, 100, step=0.1),
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "tree_method": trial.suggest_categorical(
            "tree_method", ["gpu_hist"]
        ),  # hist, gpu_hist
        "predictor": "gpu_predictor",
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 100),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 100),
        "random_state": trial.suggest_categorical("random_state", [42]),
        "n_jobs": trial.suggest_categorical("n_jobs", [4]),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-1, 1e3),
        # "min_child_weight": trial.suggest_categorical("min_child_weight", [256]),
    }

    # Model loading and training
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        early_stopping_rounds=5000,
        verbose=False,
    )

    print(f"Number of boosting rounds: {model.best_iteration}")
    #     oof = model.predict_proba(X_valid)[:, 1] # Probability
    oof = model.predict(X_valid)  # Classification: 0,1

    return accuracy_score(y_valid, oof)


# ### LightGBM

# In[17]:


def objective_lgbm(trial, X_train, X_valid, y_train, y_valid):

    params = {
        "boosting_type": "gbdt",
        # "objective": trial.suggest_categorical("objective", ["mae", "rmse"]),
        #         "objective": trial.suggest_categorical("objective", ["multi:softprob"]),
        #         "n_estimators": trial.suggest_categorical("n_estimators", [1_000]),
        #         "n_estimators": trial.suggest_categorical("n_estimators", [5000]),
        "n_estimators": trial.suggest_int("n_estimators", 700, 1000),
        "importance_type": "gain",
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1, step=0.01),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1000),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
        "subsample": trial.suggest_float("subsample", 0.1, 1, step=0.01),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 0.25),
        "max_depth": trial.suggest_int("max_depth", 1, 100),
        "random_state": trial.suggest_categorical("random_state", [42]),
        "n_jobs": trial.suggest_categorical("n_jobs", [4]),
        #         'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-1, 1e3),
        # "min_child_weight": trial.suggest_categorical("min_child_weight", [256]),
    }
    if Config.gpu:
        params["device_type"] = "gpu"

    # Model loading and training
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        # eval_metric="mae",
        callbacks=[
            lgb.log_evaluation(500),
            lgb.early_stopping(500, False, True),
        ],
    )

    #     print(f"Number of boosting rounds: {model.best_iteration}")
    oof = model.predict(X_valid)

    #     return accuracy_score(y_valid, oof)
    return roc_auc_score(y_valid, oof)


# ### Catboost

# In[18]:


def objective_cb(trial, X_train, X_valid, y_train, y_valid):

    cb_params = {
        "iterations": 10,  # 1000
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.1, 1.0),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 100),
        "bagging_temperature": trial.suggest_loguniform(
            "bagging_temperature", 0.1, 20.0
        ),
        "random_strength": trial.suggest_float("random_strength", 1.0, 2.0),
        "depth": trial.suggest_int("depth", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
        "use_best_model": True,
        #         "task_type": "GPU",
        "random_seed": 42,
    }

    # Model loading and training
    model = cb.CatBoostClassifier(**cb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        # eval_metric="accuracy",
        early_stopping_rounds=500,
        verbose=False,
    )

    # print(f"Number of boosting rounds: {model.best_iteration}")
    # oof = model.predict_proba(X_valid)[:, 1]
    oof = model.predict(X_valid)  # Classification

    return accuracy_score(y_valid, oof)


# In[19]:


# Save OOF Results
if not os.path.exists("results"):
    os.makedirs("results")


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Load Train/Test Data</h1>
# </div>
# 
# ## Load the following files
# 
#  - train.csv - Data used to build our machine learning model
#  - test.csv - Data used to build our machine learning model. Does not contain the target variable
#  - sample_submission.csv - A file in the proper format to submit test predictions

# In[20]:


train, test, sample_submission = read_data(Config.path)


# In[21]:


train.head()


# In[22]:


train.columns


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Preprocessing</h1>
# </div>

# Preprocessing taken from these notebooks:
# 
# - https://www.kaggle.com/code/thedevastator/tps-aug-simple-baseline/notebook?scriptVersionId=103379436
# - https://www.kaggle.com/code/pourchot/hunting-for-missing-values

# In[23]:


target, groups = train["failure"], train["product_code"]
train.drop("failure", axis=1, inplace=True)


# In[24]:


from sklearn.impute import KNNImputer
from feature_engine.encoding import WoEEncoder


# In[25]:


def preprocessing(df_train, df_test):
    data = pd.concat([df_train, df_test])

    data["m3_missing"] = data["measurement_3"].isnull().astype(np.int8)
    data["m5_missing"] = data["measurement_5"].isnull().astype(np.int8)
    data["area"] = data["attribute_2"] * data["attribute_3"]

    feature = [
        f for f in df_test.columns if f.startswith("measurement") or f == "loading"
    ]

    # dictionnary of dictionnaries (for the 11 best correlated measurement columns),
    # we will use the dictionnaries below to select the best correlated columns according to the product code)
    # Only for 'measurement_17' we make a 'manual' selection :
    full_fill_dict = {}
    full_fill_dict["measurement_17"] = {
        "A": ["measurement_5", "measurement_6", "measurement_8"],
        "B": ["measurement_4", "measurement_5", "measurement_7"],
        "C": ["measurement_5", "measurement_7", "measurement_8", "measurement_9"],
        "D": ["measurement_5", "measurement_6", "measurement_7", "measurement_8"],
        "E": ["measurement_4", "measurement_5", "measurement_6", "measurement_8"],
        "F": ["measurement_4", "measurement_5", "measurement_6", "measurement_7"],
        "G": ["measurement_4", "measurement_6", "measurement_8", "measurement_9"],
        "H": [
            "measurement_4",
            "measurement_5",
            "measurement_7",
            "measurement_8",
            "measurement_9",
        ],
        "I": ["measurement_3", "measurement_7", "measurement_8"],
    }

    # collect the name of the next 10 best measurement columns sorted by correlation (except 17 already done above):
    col = [col for col in df_test.columns if "measurement" not in col] + [
        "loading",
        "m3_missing",
        "m5_missing",
    ]
    a = []
    b = []
    for x in range(3, 17):
        corr = np.absolute(
            data.drop(col, axis=1).corr()[f"measurement_{x}"]
        ).sort_values(ascending=False)
        a.append(
            np.round(np.sum(corr[1:4]), 3)
        )  # we add the 3 first lines of the correlation values to get the "most correlated"
        b.append(f"measurement_{x}")
    c = pd.DataFrame()
    c["Selected columns"] = b
    c["correlation total"] = a
    c = c.sort_values(by="correlation total", ascending=False).reset_index(drop=True)
    print(f"Columns selected by correlation sum of the 3 first rows : ")
    display(c.head(10))

    for i in range(10):
        measurement_col = (
            "measurement_" + c.iloc[i, 0][12:]
        )  # we select the next best correlated column
        fill_dict = {}
        for x in data.product_code.unique():
            corr = np.absolute(
                data[data.product_code == x].drop(col, axis=1).corr()[measurement_col]
            ).sort_values(ascending=False)
            measurement_col_dic = {}
            measurement_col_dic[measurement_col] = corr[1:5].index.tolist()
            fill_dict[x] = measurement_col_dic[measurement_col]
        full_fill_dict[measurement_col] = fill_dict

    feature = [f for f in data.columns if f.startswith("measurement") or f == "loading"]
    nullValue_cols = [
        col for col in df_train.columns if df_train[col].isnull().sum() != 0
    ]

    for code in data.product_code.unique():
        total_na_filled_by_linear_model = 0
        print(f"\n-------- Product code {code} ----------\n")
        print(f"filled by linear model :")
        for measurement_col in list(full_fill_dict.keys()):
            tmp = data[data.product_code == code]
            column = full_fill_dict[measurement_col][code]
            tmp_train = tmp[column + [measurement_col]].dropna(how="any")
            tmp_test = tmp[
                (tmp[column].isnull().sum(axis=1) == 0)
                & (tmp[measurement_col].isnull())
            ]

            model = HuberRegressor(epsilon=1.9)
            model.fit(tmp_train[column], tmp_train[measurement_col])
            data.loc[
                (data.product_code == code)
                & (data[column].isnull().sum(axis=1) == 0)
                & (data[measurement_col].isnull()),
                measurement_col,
            ] = model.predict(tmp_test[column])
            print(f"{measurement_col} : {len(tmp_test)}")
            total_na_filled_by_linear_model += len(tmp_test)

        # others NA columns:
        NA = data.loc[data["product_code"] == code, nullValue_cols].isnull().sum().sum()
        model1 = KNNImputer(n_neighbors=3)
        data.loc[data.product_code == code, feature] = model1.fit_transform(
            data.loc[data.product_code == code, feature]
        )
        print(f"\n{total_na_filled_by_linear_model} filled by linear model ")
        print(f"{NA} filled by KNN ")

    data["measurement_avg"] = data[[f"measurement_{i}" for i in range(3, 17)]].mean(
        axis=1
    )
    df_train = data.iloc[: df_train.shape[0], :]
    df_test = data.iloc[df_train.shape[0] :, :]

    woe_encoder = WoEEncoder(variables=["attribute_0"])
    woe_encoder.fit(df_train, target)
    df_train = woe_encoder.transform(df_train)
    df_test = woe_encoder.transform(df_test)

    features = [
        "loading",
        "attribute_0",
        "measurement_17",
        "measurement_0",
        "measurement_1",
        "measurement_2",
        "area",
        "m3_missing",
        "m5_missing",
        "measurement_avg",
    ]

    return df_train, df_test, features


def scale(train_data, val_data, test_data, feats):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[feats])
    scaled_val = scaler.transform(val_data[feats])
    scaled_test = scaler.transform(test_data[feats])
    new_train = train_data.copy()
    new_val = val_data.copy()
    new_test = test_data.copy()
    new_train[feats] = scaled_train
    new_val[feats] = scaled_val
    new_test[feats] = scaled_test
    return new_train, new_val, new_test


train, test, features = preprocessing(train, test)
train["failure"] = target


# In[ ]:





# In[26]:


def create_features_old(df):
    df["attribute_2*3"] = df["attribute_2"] * df["attribute_3"]

    df["m_3_missing"] = df.measurement_3.isna()
    df["m_5_missing"] = df.measurement_5.isna()

    # From https://www.kaggle.com/code/heyspaceturtle/feature-selection-is-all-u-need-2
    meas_gr1_cols = [
        f"measurement_{i:d}" for i in list(range(3, 5)) + list(range(9, 17))
    ]
    df["meas_gr1_avg"] = np.mean(df[meas_gr1_cols], axis=1)
    df["meas_gr1_std"] = np.std(df[meas_gr1_cols], axis=1)
    meas_gr2_cols = [f"measurement_{i:d}" for i in list(range(5, 9))]
    df["meas_gr2_avg"] = np.mean(df[meas_gr2_cols], axis=1)

    return df


# ## Categorical/Numerical Variables

# In[27]:


## Separate Categorical and Numerical Features
cat_features = list(train.select_dtypes(include=["category", "object"]).columns)
num_features = list(test.select_dtypes(include=["number"]).columns)
num_features.remove("id")

FEATURES = cat_features + num_features
FEATURES


# In[28]:


train[FEATURES].head()


# # Extract Target and Drop Unused Columns

# In[29]:


# y = train[TARGET]

# X = train_df.drop(columns=["id", TARGET], axis=1).copy()
# X = train[FEATURES].copy()


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Missing Values</h1>
# </div>

# In[30]:


show_missing_features(train)


# ### At this point we no longer have missing values

# In[31]:


show_missing_features(train)


# # Encoding Categorical Features
# 
# Need to convert categorical features into numerical features.

# In[32]:


cat_features


# In[33]:


train, test = label_encoder(train, test, cat_features)
# X_test = pd.get_dummies(test[FEATURES], drop_first=True)

train.head()


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Feature Engineering</h1>
# </div>

# In[34]:


# train = create_features(train)
# test = create_features(test)


# In[35]:


features


# In[36]:


## Separate Categorical and Numerical Features
cat_features = list(train.select_dtypes(include=["category", "object"]).columns)
num_features = list(test.select_dtypes(include=["number"]).columns)
num_features.remove("id")

FEATURES = cat_features + num_features
FEATURES


# In[37]:


FEATURES = features


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Scale Data</h1>
# </div>
# 
# - https://www.kaggle.com/code/samuelcortinhas/tps-aug-22-failure-prediction

# ### Scaling code replaced with a better preprocesing function

# In[38]:


y = train[TARGET]
X = train[FEATURES].copy()

X_test = test[FEATURES].copy()


# In[39]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=Config.seed
)


# In[40]:


# train = create_folds(train, TARGET, Config.N_FOLDS)
train = create_group_kfolds(train, TARGET, Config.N_FOLDS)


# In[41]:


submission_df = test[["id"]].copy().astype(int)
submission_df.head()


# In[42]:


#
oof = train[["id", TARGET, "fold"]].copy().reset_index(drop=True).copy()
oof.set_index("id", inplace=True)
oof.head()


# ## Train Model

# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Train Model with Cross Validation</h1>
# </div>

# ### Some common functions

# In[43]:


all_cv_scores = pd.DataFrame(
    {
        "Model": pd.Series(dtype="str"),
        "Score": pd.Series(dtype="float"),
        "StdDev": pd.Series(dtype="float"),
    }
)


# In[44]:


def process_valid_predictions(final_valid_predictions, train_id, model_name):
    model = f"pred_{model_name}"
    final_valid_predictions_df = pd.DataFrame.from_dict(
        final_valid_predictions, orient="index"
    ).reset_index()
    final_valid_predictions_df.columns = [train_id, model]
    final_valid_predictions_df.set_index(train_id, inplace=True)
    final_valid_predictions_df.sort_index(inplace=True)
    final_valid_predictions_df.to_csv(f"train_pred_{model_name}.csv", index=True)

    return final_valid_predictions_df


# In[45]:


from scipy.stats import mode


def merge_test_predictions(final_test_predictions, calc_probability=True):

    if Config.calc_probability:
        print("Mean")
        result = np.mean(np.column_stack(final_test_predictions), axis=1)
    else:
        print("Mode")
        mode_result = mode(np.column_stack(final_test_predictions), axis=1)
        result = mode_result[0].ravel()

    return result


# In[46]:


def show_feature_importance(feature_importance_lst):
    fis_df = pd.concat(feature_importance_lst, axis=1)

    fis_df.sort_values("0_importance", ascending=True).head(40).plot(
        kind="barh", figsize=(12, 12), title="Feature Importance Across Folds"
    )
    plt.show()


# In[47]:


def save_oof_predictions(model_name, final_valid_predictions, oof):
    final_valid_predictions_df = process_valid_predictions(
        final_valid_predictions, "id", model_name
    )
    display(final_valid_predictions_df.head())
    oof[f"pred_{model_name}"] = final_valid_predictions_df[f"pred_{model_name}"]

    return oof


# In[48]:


def save_test_predictions(model_name):
    result = merge_test_predictions(final_test_predictions, Config.calc_probability)
    # result[:20]
    submission_df[f"target_{model_name}"] = result
    #     submission_df.head(10)
    ss = submission_df[["id", f"target_{model_name}"]].copy().reset_index(drop=True)
    ss.rename(columns={f"target_{model_name}": TARGET}, inplace=True)
    ss.to_csv(
        f"submission_{model_name}.csv", index=False
    )  # Can submit the individual model
    ss.head(10)


# In[49]:


def train_xgb_model(
    df,
    test,
    get_model_fn,
    FEATURES,
    TARGET,
    calc_probability,
    rowid,
    params,
    n_folds=5,
    seed=42,
):

    final_test_predictions = []
    final_valid_predictions = {}
    fold_scores = []  # Scores of Validation Set
    feature_importance_lst = []

    test = test[FEATURES].copy()

    # oof_preds = np.zeros((df.shape[0],)) # Zero array
    # print(f"oof_preds size={df.shape[0]}")
    #     print(
    #         f"\n===== XGBoost Estimators: {params['n_estimators']}, Random State: {seed} ====="
    #     )

    for fold in range(n_folds):
        print(10 * "=", f"Fold {fold+1}/{n_folds}", 10 * "=")

        start_time = time.time()

        xtrain = df[df.fold != fold].reset_index(
            drop=True
        )  # Everything not in validation fold
        xvalid = df[df.fold == fold].reset_index(drop=True)
        xtest = test.copy()

        valid_ids = xvalid.id.values.tolist()  # Id's of everything in validation fold

        ytrain = xtrain[TARGET]
        yvalid = xvalid[TARGET]

        xtrain = xtrain[FEATURES]
        xvalid = xvalid[FEATURES]
        xtrain.head()
        #         print(f"{yvalid}")
        model = get_model_fn(params)

        model.fit(
            xtrain,
            ytrain,
            eval_set=[(xvalid, yvalid)],
            #             eval_metric="acc",  # auc
            verbose=False,
            #             early_stopping_rounds=3000,
            #             callbacks=[
            #                 xgb.log_evaluation(0),
            #                 xgb.early_stopping(500, False, True),
            #             ],
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

        #         fold_score = accuracy_score(yvalid, preds_valid_class)  # Validation Set Score
        fold_score = roc_auc_score(yvalid, preds_valid)  # Validation Set Score

        fold_scores.append(fold_score)

        # Feature importance
        fi = pd.DataFrame(
            index=FEATURES,
            data=model.feature_importances_,
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


# In[50]:


xgb_params = {
    "enable_categorical": True,
    "objective": "binary:logistic",
    "eval_metric": "auc",  # auc, logloss
    "n_estimators": Config.N_ESTIMATORS,
    "learning_rate": 0.021138659045230178,
    "subsample": 0.4,
    "colsample_bytree": 0.91,
    "max_depth": 17,
    "gamma": 0.6000000000000001,
    "booster": "gbtree",
    "tree_method": "hist",
    "reg_lambda": 0.005882742898970815,
    "reg_alpha": 0.0014501578157205654,
    "random_state": 42,
    "n_jobs": 4,
    "min_child_weight": 5.567082153821453,
}


# In[51]:


def get_xgb_clf_model(params):
    #     model = xgb.XGBClassifier(n_estimators=1000)
    model = xgb.XGBClassifier(**params)
    return model


# In[52]:


# https://lightgbm.readthedocs.io/en/latest/Parameters.html

lgbm_params = {
    "n_estimators": Config.N_ESTIMATORS,
    #     "device_type": "gpu",
    #     "objective": "multiclass",
    #     "metric": "multi_logloss",
    "objective": "binary",
    "metric": "auc",  # auc, accuracy
    "lambda_l1": 0.009130931198077825,
    "lambda_l2": 3.530680683338868e-05,
    #     "reg_alpha": 0.009130931198077825,
    #     "reg_lambda": 3.530680683338868e-05,
    "num_leaves": 430,
    "importance_type": "split",
    #     "learning_rate": 0.029330486500731102,
    "learning_rate": 0.1,
    "feature_fraction": 0.8757445736567416,
    "bagging_fraction": 0.9989307214277753,
    "bagging_freq": 10,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
}
if Config.gpu:
    lgbm_params["device_type"] = "gpu"


# In[53]:


def get_lgbm_clf_model(params):
    model = lgb.LGBMClassifier(**params)
    return model


# In[54]:


cb_params = {
    "learning_rate": 0.3277295792305584,
    "l2_leaf_reg": 3.1572972266001518,
    "bagging_temperature": 0.6799604234141348,
    "random_strength": 1.99590400593318,
    "depth": 6,
    "min_data_in_leaf": 93,
    "iterations": Config.N_ESTIMATORS,  # 10000
    "use_best_model": True,
    #     "task_type": "GPU",
    "random_seed": 42,
}

if Config.gpu:
    cb_params["task_type"] = "GPU"


# In[55]:


def get_cat_clf_model(params):
    model = cb.CatBoostClassifier(**params)
    return model


# ## Begin

# In[56]:


model_name = "xgb1"  # Make the code more generic


# In[57]:


time_limit = 3600 * 3

if Config.optimize:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, X_valid, y_train, y_valid),
        n_trials=Config.n_optuna_trials,
        # timeout=time_limit,  # this or n_trials
    )

if Config.optimize:
    print("Number of finished trials:", len(study.trials))
    print("Best trial parameters:", study.best_trial.params)
    print("Best score:", study.best_value)

best_params = xgb_params


# In[58]:


(
    model,
    feature_importance_lst,
    fold_scores,
    final_valid_predictions,
    final_test_predictions,
) = train_xgb_model(
    train,
    test,
    get_xgb_clf_model,
    FEATURES,
    TARGET,
    Config.calc_probability,
    "id",
    best_params,
    Config.N_FOLDS,
    Config.seed,
)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Scores</h1>
# </div>
# 
# CV, or Cross Validation, Score.
# 
# We average the means and the standard deviations.
# 
# The Adjusted Score is the average of the means minus the average of standard deviation. Do this to attempt to get one number to evaluate the score when comparing different models.

# In[59]:


cv_score, std_dev = show_fold_scores(fold_scores)
score_dict = {"Model": model_name, "Score": cv_score, "StdDev": std_dev}
all_cv_scores = all_cv_scores.append(score_dict, ignore_index=True)
all_cv_scores.sort_values(by=["Score"], ascending=False)


# In[60]:


oof = save_oof_predictions(model_name, final_valid_predictions, oof)
oof.head()


# In[61]:


save_test_predictions(model_name)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Feature Importance</h1>
# </div>

# In[62]:


show_feature_importance(feature_importance_lst)


# ## LGBM Model

# In[63]:


model_name = "lgbm1"  # Make the code more generic


# In[64]:


time_limit = 3600 * 3

if Config.optimize:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_lgbm(trial, X_train, X_valid, y_train, y_valid),
        n_trials=Config.n_optuna_trials,
        # timeout=time_limit,  # this or n_trials
    )

if Config.optimize:
    print("Number of finished trials:", len(study.trials))
    print("Best trial parameters:", study.best_trial.params)
    print("Best score:", study.best_value)

best_params = lgbm_params


# In[65]:


(
    model,
    feature_importance_lst,
    fold_scores,
    final_valid_predictions,
    final_test_predictions,
) = train_xgb_model(
    train,
    test,
    get_lgbm_clf_model,
    FEATURES,
    TARGET,
    Config.calc_probability,
    "id",
    best_params,
    Config.N_FOLDS,
    Config.seed,
)


# In[66]:


cv_score, std_dev = show_fold_scores(fold_scores)
score_dict = {"Model": model_name, "Score": cv_score, "StdDev": std_dev}
all_cv_scores = all_cv_scores.append(score_dict, ignore_index=True)
all_cv_scores.sort_values(by=["Score"], ascending=False)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Save OOF Predictions</h1>
# </div>
# 
# This is unused for this example but needed later for [Blending](https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483).
# 
# **General idea**: The values will be use to create new features in a blended model.
# 
# - [Stacking and Blending — An Intuitive ExplanationStacking and Blending — An Intuitive Explanation](https://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413chttps://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413c)

# In[67]:


oof = save_oof_predictions(model_name, final_valid_predictions, oof)
oof.head()


# In[68]:


save_test_predictions(model_name)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Feature Importance</h1>
# </div>

# In[69]:


show_feature_importance(feature_importance_lst)


# ## CatBoost

# In[70]:


model_name = "cat1"


# In[71]:


time_limit = 3600 * 3

if Config.optimize:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_cb(trial, X_train, X_valid, y_train, y_valid),
        n_trials=Config.n_optuna_trials,
        # timeout=time_limit,  # this or n_trials
    )

if Config.optimize:
    print("Number of finished trials:", len(study.trials))
    print("Best trial parameters:", study.best_trial.params)
    print("Best score:", study.best_value)

best_params = cb_params


# In[72]:


(
    model,
    feature_importance_lst,
    fold_scores,
    final_valid_predictions,
    final_test_predictions,
) = train_xgb_model(
    train,
    test,
    get_cat_clf_model,
    FEATURES,
    TARGET,
    Config.calc_probability,
    "id",
    best_params,
    Config.N_FOLDS,
    Config.seed,
)


# In[73]:


cv_score, std_dev = show_fold_scores(fold_scores)
score_dict = {"Model": model_name, "Score": cv_score, "StdDev": std_dev}
all_cv_scores = all_cv_scores.append(score_dict, ignore_index=True)
all_cv_scores.sort_values(by=["Score"], ascending=False)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Save OOF Predictions</h1>
# </div>

# In[74]:


oof = save_oof_predictions(model_name, final_valid_predictions, oof)
oof.head()


# In[75]:


save_test_predictions(model_name)


# In[76]:


show_feature_importance(feature_importance_lst)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Non-GBDT Models: Logistic Regression, etc</h1>
# </div>

# In[77]:


def train_cv_model(
    df,
    test,
    get_model_fn,
    FEATURES,
    TARGET,
    calc_probability,
    rowid,
    params,
    n_folds=5,
    seed=42,
):

    final_test_predictions = []
    final_valid_predictions = {}
    fold_scores = []  # Scores of Validation Set
    feature_importance_lst = []

    test = test[FEATURES].copy()

    # oof_preds = np.zeros((df.shape[0],)) # Zero array
    # print(f"oof_preds size={df.shape[0]}")
    #     print(
    #         f"\n===== XGBoost Estimators: {params['n_estimators']}, Random State: {seed} ====="
    #     )

    for fold in range(n_folds):
        print(10 * "=", f"Fold {fold+1}/{n_folds}", 10 * "=")

        start_time = time.time()

        xtrain = df[df.fold != fold].reset_index(
            drop=True
        )  # Everything not in validation fold
        xvalid = df[df.fold == fold].reset_index(drop=True)
        xtest = test.copy()

        valid_ids = xvalid.id.values.tolist()  # Id's of everything in validation fold

        ytrain = xtrain[TARGET]
        yvalid = xvalid[TARGET]

        xtrain = xtrain[FEATURES]
        xvalid = xvalid[FEATURES]
        xtrain.head()
        #         print(f"{yvalid}")
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

        #         fold_score = accuracy_score(yvalid, preds_valid_class)  # Validation Set Score
        fold_score = roc_auc_score(yvalid, preds_valid)  # Validation Set Score

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


# ## Logistic Regression

# In[78]:


model_name = "lr"


# In[79]:


from sklearn.linear_model import LogisticRegression


def get_logistic_regression_model():
    model = LogisticRegression(
        max_iter=1000, C=0.0001, penalty="l2", solver="newton-cg"
    )
    return model


# In[80]:


(
    model,
    feature_importance_lst,
    fold_scores,
    final_valid_predictions,
    final_test_predictions,
) = train_cv_model(
    train,
    test,
    get_logistic_regression_model,
    FEATURES,
    TARGET,
    Config.calc_probability,
    "id",
    {},
    Config.N_FOLDS,
    Config.seed,
)


# In[81]:


cv_score, std_dev = show_fold_scores(fold_scores)
score_dict = {"Model": model_name, "Score": cv_score, "StdDev": std_dev}
all_cv_scores = all_cv_scores.append(score_dict, ignore_index=True)
all_cv_scores.sort_values(by=["Score"], ascending=False)


# In[82]:


oof = save_oof_predictions(model_name, final_valid_predictions, oof)
oof.head()


# In[83]:


save_test_predictions(model_name)


# In[84]:


show_feature_importance(feature_importance_lst)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Weighted Blend Based on OOF</h1>
# </div>

# In[85]:


all_cv_scores.sort_values(by=["Score"], ascending=False)


# In[86]:


oof.head()


# In[87]:


fig, axs = plt.subplots(1, 2, figsize=(12, 5))
oof.plot(
    x="pred_lgbm1",
    y="pred_xgb1",
    kind="scatter",
    title="Compare OOF predictions for two models",
    ax=axs[0],
)

submission_df.plot(
    x="target_lgbm1",
    y="target_xgb1",
    kind="scatter",
    title="Compare Test predictions for two models",
    ax=axs[1],
)
plt.show()


# In[88]:


oof.head()


# In[89]:


def get_oof_accuracy_score(weight, oof, pred_model1, pred_model2):
    blend_pred = (oof[pred_model1] * weight) + (oof[pred_model2] * (1 - weight))
    blend_int = np.rint(blend_pred).astype(int)
    score = accuracy_score(oof[TARGET], blend_int)
    return score


# In[90]:


def get_oof_roc_score(weight, oof, pred_model1, pred_model2):
    blend_pred = (oof[pred_model1] * weight) + (oof[pred_model2] * (1 - weight))
    score = roc_auc_score(oof[TARGET], blend_pred)
    return score


# In[91]:


myscores = {}
best = 0
best_weight = 0

for weight in range(100):
    weight /= 100
    score = get_oof_roc_score(weight, oof, "pred_xgb1", "pred_lgbm1")
    if score > best:
        best = score
        best_weight = weight
        print(f"Best Weight: {best_weight},Score {best}")
    myscores[weight] = score


# In[92]:


blend_results = pd.DataFrame(myscores, index=["score"]).T


# In[93]:


ax = blend_results.plot(title="Weight vs. OOF Score")
ax.set_ylabel("OOF Score")
ax.set_xlabel("Weight %")

plt.show()


# ### The Blended Weight is ...

# In[94]:


blend_results.loc[blend_results["score"] == blend_results["score"].max()]


# In[95]:


blend_score = blend_results["score"].max()
print(f"Blended Score: {blend_score:.8f}")


# In[96]:


w = blend_results.loc[blend_results["score"] == blend_results["score"].max()]
w


# In[97]:


wt = w.first_valid_index()
wt


# In[98]:


blend_results["score"]


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Blended Submission File</h1>
# </div>

# In[99]:


submission_df.head()


# In[100]:


print(f"Weights=({wt}, {1-wt})")

sample_submission[TARGET] = (submission_df["target_xgb1"] * wt) + (
    submission_df["target_lgbm1"] * (1 - wt)
)
sample_submission.head(8)


# In[101]:


sample_submission.to_csv("submission_blend_xgb_lgbm.csv", index=False)
sample_submission.head(8)


# In[ ]:





# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Submission File</h1>
# </div>

# In[102]:


submission_df.head()


# In[103]:


xgb_wt = 0.25
cb_wt = 0.25
lgbm_wt = 0.25  # 1 - (xgb_wt + cb_wt)
lr_wt = 0.25
# print(f"Weights=({wt}, {1-wt})")


sample_submission[TARGET] = (
    (submission_df["target_xgb1"] * xgb_wt)
    + (submission_df["target_lgbm1"] * lgbm_wt)
    + (submission_df["target_cat1"] * cb_wt)
    + (submission_df["target_lr"] * lr_wt)
)


# In[104]:


sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(8)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Blended with Regression</h1>
# </div>

# In[105]:


train_df, test_df, sample_submission = read_data(Config.path)


# In[106]:


df1 = pd.read_csv("../working/train_pred_lr.csv")
df1 = df1.rename(columns={"pred_lr": "pred_1"})

df2 = pd.read_csv("../working/train_pred_cat1.csv")
df2 = df2.rename(columns={"pred_cat1": "pred_2"})

df3 = pd.read_csv("../working/train_pred_xgb1.csv")
df3 = df3.rename(columns={"pred_xgb1": "pred_3"})

df4 = pd.read_csv("../working/train_pred_lgbm1.csv")
df4 = df4.rename(columns={"pred_lgbm1": "pred_4"})

display(df1.head(2))
display(df2.head(2))
display(df3.head(2))
display(df4.head(2))


# In[107]:


train_df = train_df.merge(df1, on="id", how="left")
train_df = train_df.merge(df2, on="id", how="left")
train_df = train_df.merge(df3, on="id", how="left")
train_df = train_df.merge(df4, on="id", how="left")


# In[108]:


train_df.head()


# ## Load Test Predictions

# In[109]:


df_test1 = pd.read_csv("../working/submission_lr.csv")
df_test2 = pd.read_csv("../working/submission_cat1.csv")
df_test3 = pd.read_csv("../working/submission_xgb1.csv")
df_test4 = pd.read_csv("../working/submission_lgbm1.csv")

display(df_test1.head(2))

df_test1 = df_test1.rename(columns={TARGET: "pred_1"})
df_test2 = df_test2.rename(columns={TARGET: "pred_2"})
df_test3 = df_test3.rename(columns={TARGET: "pred_3"})
df_test4 = df_test4.rename(columns={TARGET: "pred_4"})


display(df_test1.head(2))


# In[110]:


test_df = test_df.merge(df_test1, on="id", how="left")
test_df = test_df.merge(df_test2, on="id", how="left")
test_df = test_df.merge(df_test3, on="id", how="left")
test_df = test_df.merge(df_test4, on="id", how="left")


# In[111]:


test_df.head()


# ## Only Using pred_1, ..., pred_n

# In[112]:


useful_features = ["pred_1", "pred_2", "pred_3", "pred_4"]
# useful_features = ["pred_1", "pred_2"]

# test_df = test_df[useful_features]


# In[113]:


plt.figure(figsize=(10, 8))
sns.heatmap(train_df[useful_features].corr(), annot=True, fmt=".2f", cmap="RdBu_r")  # PuBuGn RdBu_r
shape = np.triu(train.corr())
plt.title('Model Correlation')
plt.tight_layout()
plt.show()


# In[114]:


from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge


# In[115]:


def run_lr(useful_features, train_df, test_df):
    final_predictions = []
    scores = []

    kfold = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.seed)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        xtrain = train_df.iloc[train_idx].reset_index(drop=True)
        xvalid = train_df.iloc[valid_idx].reset_index(drop=True)

        xtest = test_df[useful_features].copy()

        ytrain = xtrain[TARGET]
        yvalid = xvalid[TARGET]

        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

#         model = LogisticRegression()
        # Smaller C means more regularization; default=1.0
        # 2947.0517025518097
#         model = LogisticRegression(max_iter=500, C=2947.0517025518097, penalty='l2',solver='newton-cg')
        model = LogisticRegression(C = 2947.0517025518097,
                        max_iter = 500,
                        penalty = 'l2',
                        solver = 'liblinear')
        model.fit(xtrain, ytrain)

        preds_valid = model.predict_proba(xvalid)[:,-1]
        test_preds = model.predict_proba(xtest)[:,-1]

        final_predictions.append(test_preds)
        score = roc_auc_score(yvalid, preds_valid)
        print(f"Fold={fold}, Score={score}")
        scores.append(score)
    return scores, final_predictions


# In[116]:


fold_scores, final_predictions = run_lr(useful_features, train_df, test_df)
test_preds = np.mean(np.column_stack(final_predictions), axis=1)
cv_score, std_dev = show_fold_scores(fold_scores)
create_submission("level1_lr", TARGET, test_preds)


# In[117]:


all_cv_scores.sort_values(by=["Score"], ascending=False)

