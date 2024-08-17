#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This model demonstrates some of the concepts that I discussed in the [4th place entry](https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222791) in the February 2021 Tabular Playground series. In particular, this kernel stacks XGBoost, LightGBM, CatBoost, Ridge, SGD, and HistGradientBoosting classification models. It uses 10-fold cross validation to build each model, and makes both test and training predictions out-of-fold. Those results are then fed into the level 2 Ridge model, where 10-fold cross validation is used again to make out-of-fold predictions for the submission result. Note that no feature engineering has been performed at all outside of categorical encodings. Note as well that first glances at EDAs suggest a lot of possibilities for categorical data. If you like the model, please consider upvoting!

# # Load Data

# In[1]:


import pandas
import numpy

train = pandas.read_csv("../input/tabular-playground-series-mar-2021/train.csv")
test = pandas.read_csv("../input/tabular-playground-series-mar-2021/test.csv")
train


# # Define Features

# In[2]:


cont_features = [
    "cont0", "cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7",
    "cont8", "cont9", "cont10",
]
cat_features = [
    "cat0", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7",
    "cat8", "cat9", "cat10", "cat11", "cat12", "cat13", "cat14", "cat15",
    "cat16", "cat17", "cat18"
]
target = train["target"]


# # Encode Features

# In[3]:


from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder

xgb_cat_features = []
lgb_cat_features = []
cb_cat_features = []
ridge_cat_features = []
sgd_cat_features = []
hgbc_cat_features = []

loo_features = []
le_features = []

def label_encode(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column].unique().tolist() + test_df[column].unique().tolist())
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    return new_feature

def loo_encode(train_df, test_df, column):
    loo = LeaveOneOutEncoder()
    new_feature = "{}_loo".format(column)
    loo.fit(train_df[column], train_df["target"])
    train_df[new_feature] = loo.transform(train_df[column])
    test_df[new_feature] = loo.transform(test_df[column])
    return new_feature

for feature in cat_features:
    loo_features.append(loo_encode(train, test, feature))
    le_features.append(label_encode(train, test, feature))
    
xgb_cat_features.extend(loo_features)
lgb_cat_features.extend(le_features)
cb_cat_features.extend(cat_features)
ridge_cat_features.extend(loo_features)
sgd_cat_features.extend(loo_features)
hgbc_cat_features.extend(loo_features)


# # Generate Level 1 Models

# In[4]:


import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

random_state = 2021
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

xgb_train_preds = numpy.zeros(len(train.index), )
xgb_test_preds = numpy.zeros(len(test.index), )
xgb_features = xgb_cat_features + cont_features

lgb_train_preds = numpy.zeros(len(train.index), )
lgb_test_preds = numpy.zeros(len(test.index), )
lgb_features = lgb_cat_features + cont_features

cb_train_preds = numpy.zeros(len(train.index), )
cb_test_preds = numpy.zeros(len(test.index), )
cb_features = cb_cat_features + cont_features

ridge_train_preds = numpy.zeros(len(train.index), )
ridge_test_preds = numpy.zeros(len(test.index), )
ridge_features = ridge_cat_features + cont_features

sgd_train_preds = numpy.zeros(len(train.index), )
sgd_test_preds = numpy.zeros(len(test.index), )
sgd_features = sgd_cat_features + cont_features

hgbc_train_preds = numpy.zeros(len(train.index), )
hgbc_test_preds = numpy.zeros(len(test.index), )
hgbc_features = hgbc_cat_features + cont_features

for fold, (train_index, test_index) in enumerate(k_fold.split(train, target)):
    print("--> Fold {}".format(fold + 1))
    y_train = target.iloc[train_index]
    y_valid = target.iloc[test_index]

    xgb_x_train = pandas.DataFrame(train[xgb_features].iloc[train_index])
    xgb_x_valid = pandas.DataFrame(train[xgb_features].iloc[test_index])

    lgb_x_train = pandas.DataFrame(train[lgb_features].iloc[train_index])
    lgb_x_valid = pandas.DataFrame(train[lgb_features].iloc[test_index])

    cb_x_train = pandas.DataFrame(train[cb_features].iloc[train_index])
    cb_x_valid = pandas.DataFrame(train[cb_features].iloc[test_index])

    ridge_x_train = pandas.DataFrame(train[ridge_features].iloc[train_index])
    ridge_x_valid = pandas.DataFrame(train[ridge_features].iloc[test_index])

    sgd_x_train = pandas.DataFrame(train[sgd_features].iloc[train_index])
    sgd_x_valid = pandas.DataFrame(train[sgd_features].iloc[test_index])

    hgbc_x_train = pandas.DataFrame(train[hgbc_features].iloc[train_index])
    hgbc_x_valid = pandas.DataFrame(train[hgbc_features].iloc[test_index])

    xgb_model = XGBClassifier(
        seed=random_state,
        n_estimators=10000,
        verbosity=1,
        eval_metric="auc",
        tree_method="gpu_hist",
        gpu_id=0,
        alpha=7.105038963844129,
        colsample_bytree=0.25505629740052566,
        gamma=0.4999381950212869,
        reg_lambda=1.7256912198205319,
        learning_rate=0.011823142071967673,
        max_bin=338,
        max_depth=8,
        min_child_weight=2.286836198630466,
        subsample=0.618417952155855,
    )
    xgb_model.fit(
        xgb_x_train,
        y_train,
        eval_set=[(xgb_x_valid, y_valid)], 
        verbose=0,
        early_stopping_rounds=200
    )

    train_oof_preds = xgb_model.predict_proba(xgb_x_valid)[:,1]
    test_oof_preds = xgb_model.predict_proba(test[xgb_features])[:,1]
    xgb_train_preds[test_index] = train_oof_preds
    xgb_test_preds += test_oof_preds / n_folds
    print(": XGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    lgb_model = LGBMClassifier(
        cat_feature=[x for x in range(len(lgb_cat_features))],
        random_state=random_state,
        cat_l2=25.999876242730252,
        cat_smooth=89.2699690675538,
        colsample_bytree=0.2557260109926193,
        early_stopping_round=200,
        learning_rate=0.00918685483594994,
        max_bin=788,
        max_depth=81,
        metric="auc",
        min_child_samples=292,
        min_data_per_group=177,
        n_estimators=1600000,
        n_jobs=-1,
        num_leaves=171,
        reg_alpha=0.7115353581785044,
        reg_lambda=5.658115293998945,
        subsample=0.9262904583735796,
        subsample_freq=1,
        verbose=-1,
    )
    lgb_model.fit(
        lgb_x_train,
        y_train,
        eval_set=[(lgb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = lgb_model.predict_proba(lgb_x_valid)[:,1]
    test_oof_preds = lgb_model.predict_proba(test[lgb_features])[:,1]
    lgb_train_preds[test_index] = train_oof_preds
    lgb_test_preds += test_oof_preds / n_folds
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    cb_model = CatBoostClassifier(
        verbose=0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_state=random_state,
        num_boost_round=20000,
        od_type="Iter",
        od_wait=200,
        task_type="GPU",
        devices="0",
        cat_features=[x for x in range(len(cb_cat_features))],
        bagging_temperature=1.288692494969795,
        grow_policy="Depthwise",
        l2_leaf_reg=9.847870133539244,
        learning_rate=0.01877982653902465,
        max_depth=8,
        min_data_in_leaf=1,
        penalties_coefficient=2.1176668909602734,
    )
    cb_model.fit(
        cb_x_train,
        y_train,
        eval_set=[(cb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = cb_model.predict_proba(cb_x_valid)[:,1]
    test_oof_preds = cb_model.predict_proba(test[cb_features])[:,1]
    cb_train_preds[test_index] = train_oof_preds
    cb_test_preds += test_oof_preds / n_folds
    print(": CB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ridge_model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state),
        cv=3,
    )
    ridge_model.fit(
        ridge_x_train,
        y_train,
    )

    train_oof_preds = ridge_model.predict_proba(ridge_x_valid)[:,-1]
    test_oof_preds = ridge_model.predict_proba(test[ridge_features])[:,-1]
    ridge_train_preds[test_index] = train_oof_preds
    ridge_test_preds += test_oof_preds / n_folds
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    sgd_model = CalibratedClassifierCV(
        SGDClassifier(
            random_state=random_state,
            n_jobs=-1,
            loss="squared_hinge",
        ),
        cv=3,
    )
    sgd_model.fit(
        sgd_x_train,
        y_train,
    )

    train_oof_preds = sgd_model.predict_proba(sgd_x_valid)[:,-1]
    test_oof_preds = sgd_model.predict_proba(test[sgd_features])[:,-1]
    sgd_train_preds[test_index] = train_oof_preds
    sgd_test_preds += test_oof_preds / n_folds
    print(": SGD - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    hgbc_model = HistGradientBoostingClassifier(
        l2_regularization=1.766059063693552,
        learning_rate=0.10675193678150449,
        max_bins=128,
        max_depth=31,
        max_leaf_nodes=185,
        random_state=2021
    )
    hgbc_model.fit(
        hgbc_x_train,
        y_train,
    )

    train_oof_preds = hgbc_model.predict_proba(hgbc_x_valid)[:,-1]
    test_oof_preds = hgbc_model.predict_proba(test[hgbc_features])[:,-1]
    hgbc_train_preds[test_index] = train_oof_preds
    hgbc_test_preds += test_oof_preds / n_folds
    print(": HGBC - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": XGB - ROC AUC Score = {}".format(roc_auc_score(target, xgb_train_preds, average="micro")))
print(": LGB - ROC AUC Score = {}".format(roc_auc_score(target, lgb_train_preds, average="micro")))
print(": CB - ROC AUC Score = {}".format(roc_auc_score(target, cb_train_preds, average="micro")))
print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(target, ridge_train_preds, average="micro")))
print(": SGD - ROC AUC Score = {}".format(roc_auc_score(target, sgd_train_preds, average="micro")))
print(": HGBC - ROC AUC Score = {}".format(roc_auc_score(target, hgbc_train_preds, average="micro")))


# # Build Level 2 Model

# In[5]:


from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

random_state = 2021
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

l1_train = pandas.DataFrame(data={
    "xgb": xgb_train_preds.tolist(),
    "lgb": lgb_train_preds.tolist(),
    "cb": cb_train_preds.tolist(),
    "ridge": ridge_train_preds.tolist(),
    "sgd": sgd_train_preds.tolist(),
    "hgbc": hgbc_train_preds.tolist(),
    "target": target.tolist()
})
l1_test = pandas.DataFrame(data={
    "xgb": xgb_test_preds.tolist(),
    "lgb": lgb_test_preds.tolist(),
    "cb": cb_test_preds.tolist(),
    "sgd": sgd_test_preds.tolist(),
    "ridge": ridge_test_preds.tolist(),    
    "hgbc": hgbc_test_preds.tolist(),
})

train_preds = numpy.zeros(len(l1_train.index), )
test_preds = numpy.zeros(len(l1_test.index), )
features = ["xgb", "lgb", "cb", "ridge", "sgd", "hgbc"]

for fold, (train_index, test_index) in enumerate(k_fold.split(l1_train, target)):
    print("--> Fold {}".format(fold + 1))
    y_train = target.iloc[train_index]
    y_valid = target.iloc[test_index]

    x_train = pandas.DataFrame(l1_train[features].iloc[train_index])
    x_valid = pandas.DataFrame(l1_train[features].iloc[test_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]
    train_preds[test_index] = train_oof_preds
    test_preds += test_oof_preds / n_folds
    print(": ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": ROC AUC Score = {}".format(roc_auc_score(target, train_preds, average="micro")))


# # Save Predictions

# In[6]:


submission = pandas.read_csv("../input/tabular-playground-series-mar-2021/sample_submission.csv")
submission["target"] = test_preds.tolist()
submission.to_csv("submission.csv", index=False)

