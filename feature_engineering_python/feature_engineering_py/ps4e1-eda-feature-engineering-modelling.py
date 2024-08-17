#!/usr/bin/env python
# coding: utf-8

# <div style="padding: 20px; background-color: #000080; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
#     <div style="border: 2px solid #000080; padding: 20px; text-align: center; border-radius: 10px; background-color: #ffffff;">
#         <h1 style="color: #00000; font-size: 32px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;">Easy EDA & modelling with BlueCast</h1>
#         <div><em>
#        If you like the content please consider an upvote. It is a great motivator to keep sharing code and ideas.
#         Thank you!!!
#     </em></div>
# </div>

# <h1 style="background-color: #000080; color: #ffff00;">Introducing BlueCast</h1>
# 
# Bluecast is an automl framework that offers a lightweight library with EDA, automl and xperiment tracking capabilities.
# It offers many options for customization. Check out the repo to see many examples:
# https://github.com/ThomasMeissnerDS/BlueCast

# <h1 style="background-color: #000080; color: #ffff00;">Table of contents</h1>
# 
# * [Load the data](#1)
# * [EDA with BlueCast](#2)
#     * [Feature type detection](#2.1)
#     * [Univariate plots](#2.2)    
#     * [Bivariate plots](#2.3) 
#     * [Correlation to target](#2.4)  
#     * [Correlation heatmap](#2.5) 
#     * [Mutual informtion score](#2.6)
#     * [Dimensionality reduction using PCA](#2.7) 
#     * [Dimensionality reduction using t-SNE](#2.8) 
#     * [Map of associations between categorical features](#2.9)
#     * [Missing values](#2.10)
#     * [Do we have columns with high cardinality?](#2.11) 
# * [Leakage detection](#3)
#     * [Leakage detection for numerical columns](#3.1)
#     * [Leakage detection for categorical columns](#3.2)  
# * [Building the pipeline in a few lines of code](#4)
# * [Plot decision trees](#5)
# * [Predict on new data](#6) 
# * [Accessing the inbuilt experiment tracker](#7) 
#     * [Understand most impactful parameters across all hyperparameters tests and model trainings](#7.1)
#     * [Don't lose your progress!](#7.2) 
# * [Submission time](#8)

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install numpy==1.23\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install bluecast --no-index --find-links=file:/kaggle/input/bluecast/bluecast-0.80-py3-none-any.whl\n')


# <h1 style="background-color: #000080; color: #ffff00;">Load the data</h1>

# In[3]:


import numpy as np
import pandas as pd
import re
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from xgboost import plot_tree

import warnings
warnings.filterwarnings("ignore")


from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.general_utils.general_utils import save_to_production, load_for_production

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer, LabelEncoder


# In[4]:


train = pd.read_csv('../input/playground-series-s4e1/train.csv')
train_original = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")
test = pd.read_csv('../input/playground-series-s4e1/test.csv')
submission = pd.read_csv('../input/playground-series-s4e1/sample_submission.csv')
target = "Exited"

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


# In[5]:


train


# In[6]:


train_original = train_original.rename(columns = {"RowNumber": "Id"})
train_original


# In[7]:


train = pd.concat([train, train_original])
train = train.reset_index(drop=True)


# In[8]:


test


# In[9]:


# from here: https://www.kaggle.com/competitions/playground-series-s4e1/discussion/465192
def age_tr(df) : 
    df['Age_Category'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    return df

def cred_score_tr(df) : 
    df['Credit_Score_Range'] = pd.cut(df['CreditScore'], bins=[0, 300, 600, 700, 800, 900], labels=['0-300', '300-600', '600-700', '700-800', '900+'])
    return df

def geo_gender_tr(df) : 
    df['Geo_Gender'] = df['Geography'] + '_' + df['Gender']
    return df

train = age_tr(train)
test = age_tr(test)

train = cred_score_tr(train)
test = cred_score_tr(test)

train = geo_gender_tr(train)
test = geo_gender_tr(test)


# In[10]:


train["source"] = 0
test["source"] = 1
all_data = pd.concat([train, test]).reset_index(drop=True) # reset is needed, because train and test have overlapping indices


# In[11]:


train.info()


# In[12]:


all_data.loc[all_data["CustomerId"] == 15815645]


# In[13]:


sort_columns = ["CustomerId", "Surname", "id", "Tenure"]
sort_order = [True, True, True, True]


# In[14]:


# add running count of occurances
all_data["running_occurance_per_customer"] = all_data.sort_values(sort_columns, ascending=sort_order).groupby(['CustomerId', 'Surname']).cumcount() + 1
all_data.loc[all_data["CustomerId"] == 15815645]


# In[15]:


# add running balance
all_data["running_balance"] = all_data.sort_values(sort_columns, ascending=sort_order).groupby(['CustomerId', 'Surname'])[["Balance"]].cumsum()
all_data["running_balance_mean"] = all_data["running_balance"] / all_data["running_occurance_per_customer"]
all_data.loc[all_data["CustomerId"] == 15815645]


# In[16]:


# add last credit score and tenure
all_data["last_credit_score"] = all_data.sort_values(sort_columns, ascending=sort_order).groupby(['CustomerId', 'Surname'])[["CreditScore"]].shift(1).fillna(train["CreditScore"].mean())
all_data["last_tenure"] = all_data.sort_values(["CustomerId", "Surname", "id"], ascending=[True, True, True]).groupby(['CustomerId', 'Surname'])[["Tenure"]].shift(1).fillna(0)
all_data.loc[all_data["CustomerId"] == 15815645]


# In[17]:


# onehot encode gender
all_data['Gender'] = np.where(all_data['Gender']=="Male", 1, 0) # alphabetic
all_data.loc[all_data["CustomerId"] == 15815645]


# In[18]:


all_data = all_data.reset_index(drop=True)
train = all_data.loc[all_data["source"] == 0].copy()
test = all_data.loc[all_data["source"] == 1].copy()

train = train.reset_index(drop=True)

test = test.drop(target, axis=1)
test = test.reset_index(drop=True)

# drop ids
train = train.drop(["id", "CustomerId", "source"], axis=1)
test = test.drop(["id", "CustomerId", "source"], axis=1)


# In[19]:


train


# In[20]:


test


# In[21]:


train.isna().sum()


# In[22]:


train.info()


# In[23]:


target = "Exited"


# In[24]:


train[target].value_counts()


# <h1 style="background-color: #000080; color: #ffff00;">EDA with BlueCast</h1>
# 
# Here you can get an overview of the data and its distribution.

# In[25]:


from bluecast.eda.analyse import (
    bi_variate_plots,
    correlation_heatmap,
    correlation_to_target,
    plot_pca,
    plot_theil_u_heatmap,
    plot_tsne,
    univariate_plots,
    check_unique_values,
    plot_null_percentage,
    mutual_info_to_target
)

from bluecast.preprocessing.feature_types import FeatureTypeDetector


# <h2 style="background-color: #000080; color: #ffff00;">Feature type detection</h2>

# In[26]:


feat_type_detector = FeatureTypeDetector()
train_data = feat_type_detector.fit_transform_feature_types(train)

len(feat_type_detector.num_columns)


# <h2 style="background-color: #000080; color: #ffff00;">Univariate plots</h2>

# In[27]:


univariate_plots(
        train_data.loc[
            :, feat_type_detector.num_columns
        ],
        target,
    )


# <h2 style="background-color: #000080; color: #ffff00;">Bivariate plots</h2>

# In[28]:


bi_variate_plots(
        train.loc[
            :, feat_type_detector.num_columns
        ],
        target,
    )


# <h2 style="background-color: #000080; color: #ffff00;">Correlation to target</h2>

# In[29]:


# show correlation to target
correlation_to_target(
    train.loc[:, feat_type_detector.num_columns],
      target,
      )


# <h2 style="background-color: #000080; color: #ffff00;">Correlation heatmap</h2>

# In[30]:


correlation_heatmap(train_data.loc[
            :, feat_type_detector.num_columns])


# <h2 style="background-color: #000080; color: #ffff00;">Mutual informtion score</h2>

# In[31]:


# show mutual information of categorical features to target
# features are expected to be numerical format
extra_params = {"random_state": 30}
mutual_info_to_target(train_data.loc[:, feat_type_detector.num_columns].fillna(0), target, class_problem="binary", **extra_params)


# <h2 style="background-color: #000080; color: #ffff00;">Dimensionality reduction using PCA</h2>

# In[32]:


# show feature space after principal component analysis
plot_pca(train_data.loc[
            :, feat_type_detector.num_columns
        ].fillna(0), target)


# <h2 style="background-color: #000080; color: #ffff00;">Dimensionality reduction using t-SNE</h2>

# In[33]:


# show feature space after t-SNE
#plot_tsne(train_data.loc[
#            :, feat_type_detector.num_columns
#        ].fillna(0), target, perplexity=1000, random_state=0)


# <h2 style="background-color: #000080; color: #ffff00;">Map of associations between categorical features</h2>

# In[34]:


# show a heatmap of assocations between categorical variables
theil_matrix = plot_theil_u_heatmap(train_data, feat_type_detector.cat_columns)
theil_matrix


# <h2 style="background-color: #000080; color: #ffff00;">Missing values</h2>

# In[35]:


# plot the percentage of Nulls for all features
if train_data.loc[:, feat_type_detector.num_columns].isna().sum().sum() > 0:
    plot_null_percentage(
       train_data.loc[:, feat_type_detector.num_columns],
        )
else:
    print("This dataset does not have any missing values")


# <h2 style="background-color: #000080; color: #ffff00;">Do we have columns with high cardinality?</h2>

# In[36]:


# detect columns with a very high share of unique values
many_unique_cols = check_unique_values(train_data, feat_type_detector.cat_columns, threshold=0.90)
many_unique_cols


# <h1 style="background-color: #000080; color: #ffff00;">Leakage detection</h1>
# 
# With big data and complex pipelines data leakage can easily sneak in.
# To detect leakage BlueCast offers two functions:

# In[37]:


from bluecast.eda.data_leakage_checks import (
    detect_categorical_leakage,
    detect_leakage_via_correlation,
)


# <h2 style="background-color: #000080; color: #ffff00;">Leakage detection for numerical columns</h2>

# In[38]:


# Detect leakage of numeric columns based on correlation
numresult = detect_leakage_via_correlation(
        train.loc[:, feat_type_detector.num_columns], target, threshold=0.9 # target column is part of detected numerical columns here
    )


# <h2 style="background-color: #000080; color: #ffff00;">Leakage detection for categorical columns</h2>

# In[39]:


# Detect leakage of categorical columns based on Theil's U
result = detect_categorical_leakage(
        train_data.loc[:, feat_type_detector.cat_columns + [target]], target, threshold=0.9
    )


# This could also be a false positive when these features are contant.

# <h1 style="background-color: #000080; color: #ffff00;">Building the pipeline in a few lines of code</h1>

# In[40]:


from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

# We give more depth
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 1000
xgboost_param_config.max_depth_max = 7

# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.global_random_state = 333
train_config.hypertuning_cv_folds = 5
train_config.hyperparameter_tuning_rounds = 1000
train_config.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 2
train_config.enable_grid_search_fine_tuning = True # enable param refinement
train_config.use_full_data_for_final_model = True
train_config.precise_cv_tuning = False
train_config.gridsearch_nb_parameters_per_grid = 5

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1987)



# In[41]:


automl = BlueCastCV(
        class_problem="binary", # also multiclass is possible
        stratifier=skf,
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
        #custom_in_fold_preprocessor=custom_infold_preproc,
        #custom_preprocessor=custom_preprocessor,
        #ml_model=custom_model_tab,
        )


# In[42]:


try:
    automl.fit_eval(train.copy(), target_col=target)
except Exception as e:
    print(e)


# <h1 style="background-color: #000080; color: #ffff00;">Plot decision trees</h1>

# In[43]:


for model in automl.bluecast_models:
    plot_tree(model.ml_model.model)
    fig = plt.gcf()
    fig.set_size_inches(150, 80)
    plt.show()


# <h1 style="background-color: #000080; color: #ffff00;">Predict on new data</h1>

# In[44]:


probs, classes = automl.predict(test)


# <h1 style="background-color: #000080; color: #ffff00;">Accessing the inbuilt experiment tracker</h1>
# 
# BlueCast also keep track of your experiments! Let's see how we can access them.

# In[45]:


# access the experiment tracker if needed
tracker = automl.experiment_tracker
tracker


# In[46]:


# see all stored information as a Pandas DataFrame
tracker_df = tracker.retrieve_results_as_df()
tracker_df


# In[47]:


tracker_df.info()


# Let us try to find out what the most important hyperparameters and settings have been so far

# <h2 style="background-color: #000080; color: #ffff00;">Understand most impactful parameters across all hyperparameters tests and model trainings</h2>

# In[48]:


from sklearn.ensemble import RandomForestRegressor
import shap 

cols = [
    "shuffle_during_training",
    "global_random_state",
    "early_stopping_rounds",
    "autotune_model",
    "enable_feature_selection",
    "train_split_stratify",
    "use_full_data_for_final_model",
    "eta",
    "max_depth",
    "alpha",
    "lambda",
    "gamma",
    "max_leaves",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel"
]

regr = RandomForestRegressor(max_depth=4, random_state=0)

tracker_df = tracker_df.loc[tracker_df["score_category"] == "oof_score"]

experiment_feats_df, experiment_feats_target = tracker_df.loc[:, cols], tracker_df.loc[:, "eval_scores"]

regr.fit(experiment_feats_df.fillna(0), experiment_feats_target.fillna(99))

explainer = shap.TreeExplainer(regr)


shap_values = explainer.shap_values(experiment_feats_df)
shap.summary_plot(shap_values, experiment_feats_df)


# <h2 style="background-color: #000080; color: #ffff00;">Don't lose your progress!</h2>
# 
# BlueCast offers simple utilities to save and load your pipeline (including the tracker)

# In[49]:


# save pipeline including tracker
save_to_production(automl, "/kaggle/working/", "bluecast_cv_pipeline")

# in production or for further experiments this can be loaded again
automl_loaded = load_for_production("/kaggle/working/", "bluecast_cv_pipeline")


# <h1 style="background-color: #000080; color: #ffff00;">Submission time</h1>

# In[50]:


probs


# In[51]:


submission[target] = probs
submission.to_csv('submission.csv', index=False)


# In[52]:


submission

