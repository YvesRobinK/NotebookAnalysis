#!/usr/bin/env python
# coding: utf-8

# ## MLJAR AutoML starter
# 
# This is my first time I start using AutoML for finding fast solution and first model to test (I will see in model leaderboard to choose appropriate direction). The most interesting part for now is to find the first good model - which one is the most suitable for this competition. Certainly it could change (often it change) but ... good start is important.
# 
# #### The goal of this notebook is to explain how to build AutoML pipeline using MLJAR (explain arguments/parameters to build customized pipeline).
# 
# STEPS:
# 1. Create model Leaderboard using AutoML
# 2. Play with EDA and feature engeneering (I know this should be first step but ... we will do it later after first check)
# 3. One more time create leaderboard using AutoML - looking for improvements
# 4. Look for custom solutions based on learning from AutoML - we can use learnings from AutoML (just look inside AutoML logs :) - here you can find many tips)
# 5. Tuning (blending, mixing, optimizing ... etc.)

# <div class="alert alert-info">
#     <strong>MLJAR</strong>
# 
#  State-of-the-art Automated Machine Learning for tabular data. mljar builds a complete Machine Learning Pipeline
#     <ul>
# <li>advanced feature engineering</li>
# <li>algorithms selection and tuning</li>
# <li>automatic documentation</li>
# <li>ML explanations</li>
#     </ul>
#     <div><br/></div>
#  <strong>Important links</strong>
#     <ul>
#         <li><a href="https://mljar.com/">MLJAR home</a></li>
#         <li><a href="https://github.com/mljar/mljar-supervised">MLJAR GitHub</a></li>
#     </ul>
# </div>
# 
# 

# In[1]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@dev')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

import seaborn as sns
sns.set(font_scale= 1.25)

from supervised.automl import AutoML # mljar-supervised


# In[3]:


train = pd.read_csv('../input/tabular-playground-series-jun-2021/train.csv', index_col = 'id')
test = pd.read_csv('../input/tabular-playground-series-jun-2021/test.csv', index_col = 'id')


# In[4]:


x_cols = train.columns[0:-1].tolist()
y_col = train.columns[-1]


# As you can see I have not used any feature engeneering techniques - just use data as they are ... to make first check. We will do feature engeneering later ... 

# # 1. EDA - BEFORE WE START TRAINING MODEL - LET'S LOOK INTO DATA

# ## TARGET DISTRIDUTION

# In[5]:


fig, ax = plt.subplots(figsize = (10,5))
plt.xticks(rotation=45)
ax = sns.countplot(x='target', data=train, order=sorted(train['target'].unique()), ax=ax)
ax.set_title('Target Distribution')
plt.show()


# ## TRAIN DATA 

# In[6]:


train.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# ## TEST DATA

# In[7]:


test.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# ## CROSS CHECK

# In[8]:


# This snipplet was taken from great notebook: https://www.kaggle.com/subinium/tps-may-categorical-eda

def diff_color(x):
    color = 'red' if x<0 else ('green' if x > 0 else 'black')
    return f'color: {color}'

(train.describe() - test.describe())[test.columns].T.iloc[:,1:].style\
        .bar(subset=['mean', 'std'], align='mid', color=['#d65f5f', '#5fba7d'])\
        .applymap(diff_color, subset=['min', 'max'])


# ## FEATURE DISTRIBUTION - CLASS

# In[9]:


import plotly.express as px

target_column = 'target'
num_rows, num_cols = 15,5
f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 60))

for index, column in enumerate(x_cols):
    i,j = (index // num_cols, index % num_cols)

    sns.kdeplot(train.loc[train[target_column] == 'Class_1', column], color=px.colors.qualitative.G10[1], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_2', column], color=px.colors.qualitative.G10[2], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_3', column], color=px.colors.qualitative.G10[9], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_4', column], color=px.colors.qualitative.G10[4], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_5', column], color=px.colors.qualitative.G10[5], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_6', column], color=px.colors.qualitative.G10[6], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_7', column], color=px.colors.qualitative.G10[7], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_8', column], color=px.colors.qualitative.G10[8], shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train[target_column] == 'Class_9', column], color=px.colors.qualitative.G10[3], shade=True, ax=axes[i,j])
plt.tight_layout()
plt.show()


# ## TEATURE DISTRIBUTION - TOTAL 

# In[10]:


df_all = pd.concat([train.drop('target', axis = 1), test], axis = 0)
unique_df = pd.DataFrame(df_all.nunique()).reset_index()
unique_df.columns=['features','count']


fig, feat_bar = plt.subplots(figsize = (15,30))
feat_bar = sns.barplot(y="features", x="count", data = unique_df, palette="crest", orient='h')


# ## ZEROS IN DATA
# 
# ### ZEROS IN TRAIN

# In[11]:


fig, zero_train = plt.subplots(figsize = (15,30))
plt.xticks(rotation=45)
zero_train = sns.barplot(data = pd.DataFrame((train[x_cols]==0).mean()).T, palette="crest", orient='h')

zero_train.axvline((train[x_cols]==0).mean().mean(), color ='red')
zero_train.text((train[x_cols]==0).mean().mean()+0.01, 1, "mean: {}".format((train[x_cols]==0).mean().mean()), size = 20, alpha = 1)
#fig.suptitle('Zero distribution in TRAIN dataset', fontsize = 25, fontweight = 'bold')


# ### ZEROS IN TEST

# In[12]:


fig, zero_train = plt.subplots(figsize = (15,30))
plt.xticks(rotation=45)
zero_test = sns.barplot(data = pd.DataFrame((test[x_cols]==0).mean()).T, palette="crest", orient='h')

zero_test.axvline((test[x_cols]==0).mean().mean(), color ='red')
zero_test.text((test[x_cols]==0).mean().mean()+0.01, 1, "mean: {}".format((test[x_cols]==0).mean().mean()), size = 20, alpha = 1)
#fig.suptitle('Zero distribution in TRAIN dataset', fontsize = 25, fontweight = 'bold')


# ## OUTLIERS

# In[13]:


fig, out_fig = plt.subplots(figsize = (15,30))
plt.xticks(rotation=45)
out_fig = sns.boxplot(data = train, orient="h", palette="crest")


# ## DIMENTIONALITY REDUCTION

# ### TSNE

# In[14]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

train_sub = train.sample(10000, random_state= 42)
model = TSNE(n_components=2, random_state=0, perplexity= 50, n_iter=3000)
tsne_data = model.fit_transform(StandardScaler().fit_transform(train_sub.drop('target', axis = 1).astype(float)))
tsne_data = np.vstack((tsne_data.T, train_sub.target)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("D1", "D2", "target"))

sns.FacetGrid(tsne_df, hue="target", height=6).map(plt.scatter, 'D1', 'D2').add_legend()
plt.title('Perplexity= 50, n_iter=3000')
plt.show()


# ### LDA

# In[15]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

train_sub = train.sample(10000, random_state= 42)
lda_data = LDA(n_components=2).fit_transform(train_sub.drop(columns='target'),train_sub.target)
plt.figure(figsize=(10,10))
sns.scatterplot(x = lda_data[:, 0], y = lda_data[:, 1], hue = 'target', data=train_sub)


# ### UMAP

# In[16]:


import umap

train_sub = train.sample(10000, random_state= 42)
embedding_2d = umap.UMAP(random_state = 42 ,n_components=2).fit_transform(train_sub.drop(columns='target').to_numpy())
embedding_3d = umap.UMAP(random_state = 42 ,n_components=3).fit_transform(train_sub.drop(columns='target').to_numpy())


# In[17]:


plt.figure(figsize=(10,10))
sns.scatterplot(x = embedding_2d[:, 0], y = embedding_2d[:, 1], hue='target', data=train_sub)


# ### UMAP 3D

# In[18]:


plt.figure(figsize=(50,30))
umap_3d = px.scatter_3d(
    embedding_3d, x=0, y=1, z=2,
    labels={'color': 'target'},
    color= train_sub.target,
    color_discrete_sequence=['red', 'seagreen', 'gold', 'black'],
)

umap_3d.update_traces(marker_size=2)
umap_3d.show()


# # AUTOML - MLJAR TRAINING
# 
# How amazing is to run over 60 models and their combination writing only ..... one line of code!!! This is really rapid development. Cool!

# In[19]:


OPTUNA = False


# ### MLJAR - Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).
# 
# Arguments:
# 
# <code>def __init__(
#         self,
#         results_path=None,
#         total_time_limit=60 * 60,
#         mode="Explain",
#         ml_task="auto",
#         model_time_limit=None,
#         algorithms="auto",
#         train_ensemble=True,
#         stack_models="auto",
#         eval_metric="auto",
#         validation_strategy="auto",
#         explain_level="auto",
#         golden_features="auto",
#         features_selection="auto",
#         start_random_models="auto",
#         hill_climbing_steps="auto",
#         top_models_to_improve="auto",
#         boost_on_errors="auto",
#         kmeans_features="auto",
#         mix_encoding="auto",
#         max_single_prediction_time=None,
#         optuna_time_budget=None,
#         optuna_init_params={},
#         optuna_verbose=True,
#         n_jobs=-1,
#         verbose=1,
#         random_state=1234,
#     )</code>

# ### ARGUMENTS EXPLANATION 
# Docs: https://github.com/mljar/mljar-supervised/blob/master/supervised/automl.py
# 
# - **results_path (str)**: The path with results. If None, then the name of directory will be generated with the template: AutoML_{number}, where the number can be from 1 to 1,000 - depends which direcory name will be available. If the `results_path` will point to directory with AutoML results (`params.json` must be present),then all models will be loaded.
# - **total_time_limit (int)**: The total time limit in seconds for AutoML training. It is not used when `model_time_limit` is not `None`.
# - **mode (str)**: Can be (`Explain`, `Perform`, `Compete`, `Optuna`). This parameter defines the goal of AutoML and how intensive the AutoML search will be.
#     - `Explain` : To to be used when the user wants to explain and understand the data. Uses 75%/25% train/test split. Uses the following models: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `XGBoost`, `Neural Network`, and `Ensemble`. Has full explanations in reports: learning curves, importance plots, and SHAP plots.
#     - `Perform` : To be used when the user wants to train a model that will be used in real-life use cases. Uses 5-fold CV (Cross-Validation). Uses the following models: `Linear`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`, `Neural Network`, and `Ensemble`. Has learning curves and importance plots in reports.
#     - `Compete` : To be used for machine learning competitions (maximum performance). Uses 80/20 train/test split, or 5-fold CV, or 10-fold CV (Cross-Validation) - it depends on `total_time_limit`. If not set directly, AutoML will select validation automatically. Uses the following models: `Decision Tree`, `Random Forest`, `Extra Trees`, `LightGBM`,  `XGBoost`, `CatBoost`, `Neural Network`, `Nearest Neighbors`, `Ensemble`, and `Stacking`. It has only learning curves in the reports.
#     - `Optuna` : To be used for creating highly-tuned machine learning models. Uses 10-fold CV (Cross-Validation). It tunes with Optuna the following algorithms: `Random Forest`, `Extra Trees`, `LightGBM`, `XGBoost`, `CatBoost`, `Neural Network`. It applies `Ensemble` and `Stacking` for trained models. It has only learning curves in the reports.
# - **ml_task (str)**: Can be {"auto", "binary_classification", "multiclass_classification", "regression"}. If left `auto` AutoML will try to guess the task based on target values. If there will be only 2 values in the target, then task will be set to `"binary_classification"`. If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`. In all other casses, the task is set to `"regression"`.
# - **model_time_limit (int)**: The time limit for training a single model, in seconds. If `model_time_limit` is set, the `total_time_limit` is not respected. The single model can contain several learners. The time limit for subsequent learners is computed based on `model_time_limit`.
# - **algorithms (list of str)**: The list of algorithms that will be used in the training. The algorithms can be: `Baseline`,`Linear`,`Decision Tree`,`Random Forest`, `Extra Trees`, `LightGBM`,`Xgboost`, `CatBoost`,`Neural Network`,`Nearest Neighbors`,
#  
# - **train_ensemble (boolean)**: Whether an ensemble gets created at the end of the training.
# - **stack_models (boolean)**: Whether a models stack gets created at the end of the training. Stack level is 1.
# - **eval_metric (str)**: The metric to be used in early stopping and to compare models. for binary classification: `logloss`, `auc`, `f1`, `average_precision`, `accuracy` - default is logloss (if left "auto"). for mutliclass classification: `logloss`, `f1`, `accuracy` - default is `logloss` (if left "auto"). for regression: `rmse`, `mse`, `mae`, `r2`, `mape`, `spearman`, `pearson` - default is `rmse` (if left "auto")
# 
# - **validation_strategy (dict)**: Dictionary with validation type. Right now train/test split and cross-validation are supported.
# - **explain_level (int)**: The level of explanations included to each model: if `explain_level` is `0` no explanations are produced. if `explain_level` is `1` the following explanations are produced: importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients. if `explain_level` is `2` the following explanations are produced: the same as `1` plus SHAP explanations. If left `auto` AutoML will produce explanations based on the selected `mode`.
# - **golden_features (boolean or int)**: Whether to use golden features (and how many should be added). If left `auto` AutoML will use golden features based on the selected `mode`: If `mode` is "Explain", `golden_features` = False. If `mode` is "Perform", `golden_features` = True. If `mode` is "Compete", `golden_features` = True. If `boolean` value is set then the number of Golden Features is set automatically. It is set to min(100, max(10, 0.1*number_of_input_features)). If `int` value is set, the number of Golden Features is set to this value.
# - **features_selection (boolean)***: Whether to do features_selection. If left `auto` AutoML will do feature selection based on the selected `mode`: If `mode` is "Explain", `features_selection` = False. If `mode` is "Perform", `features_selection` = True. If `mode` is "Compete", `features_selection` = True.
# - **start_random_models (int)**: Number of starting random models to try. If left `auto` AutoML will select it based on the selected `mode`: If `mode` is "Explain", `start_random_models` = 1. If `mode` is "Perform", `start_random_models` = 5. If `mode` is "Compete", `start_random_models` = 10.
# - **hill_climbing_steps (int)**: Number of steps to perform during hill climbing. If left `auto` AutoML will select it based on the selected `mode`: If `mode` is "Explain", `hill_climbing_steps` = 0. If `mode` is "Perform", `hill_climbing_steps` = 2. If `mode` is "Compete", `hill_climbing_steps` = 2.
# - **top_models_to_improve (int)**: Number of best models to improve in `hill_climbing` steps. If left `auto` AutoML will select it based on the selected `mode`: If `mode` is "Explain", `top_models_to_improve` = 0. If `mode` is "Perform", `top_models_to_improve` = 2. If `mode` is "Compete", `top_models_to_improve` = 3.
# - **boost_on_errors (boolean)**: Whether a model with boost on errors from previous best model should be trained. By default available in the `Compete` mode.
# - **kmeans_features (boolean)**: Whether a model with k-means generated features should be trained. By default available in the `Compete` mode.
# - **mix_encoding (boolean)**: Whether a model with mixed encoding should be trained. Mixed encoding is the encoding that uses label encoding for categoricals with more than 25 categories, and one-hot binary encoding for other categoricals. It is only applied if there are categorical features with cardinality smaller than 25. By default it is available in the `Compete` mode.
# - **max_single_prediction_time (int or float)**: The limit for prediction time for single sample. Use it if you want to have a model with fast predictions. Ideal for creating ML pipelines used as REST API. Time is in seconds. By default (`max_single_prediction_time=None`) models are not optimized for fast predictions, except the mode `Perform`. For the mode `Perform` the default is `0.5` seconds.
# - **optuna_time_budget (int)**: The time in seconds which should be used by Optuna to tune each algorithm. It is time for tuning single algorithm. If you select two algorithms: Xgboost and CatBoost, and set optuna_time_budget=1000, then Xgboost will be tuned for 1000 seconds and CatBoost will be tuned for 1000 seconds. What is more, the tuning is made for each data type, for example for raw data and for data with inserted Golden Features. This parameter is only used when `mode="Optuna"`. If you set `mode="Optuna"` and forget to set this parameter, it will be set to 3600 seconds.
# - **optuna_init_params (dict)**: If you have already tuned parameters from Optuna you can reuse them by setting this parameter. This parameter is only used when `mode="Optuna"`. The dict should have structure and params as specified in the MLJAR AutoML .
# - **optuna_verbose (boolean)**: If true the Optuna tuning details are displayed. Set to `True` by default.
# - **n_jobs (int)**: Number of CPU cores to be used. By default is set to `-1` which means using  all processors.
# - **verbose (int)**: Controls the verbosity when fitting and predicting. Note: Still not implemented, please left `1`
# - **random_state (int)**: Controls the randomness of the `AutoML`
# 

# In[20]:


if OPTUNA:
    automl = AutoML(
        mode="Optuna",
        optuna_time_budget=600,
        total_time_limit=6*3600,
        golden_features=True,
        boost_on_errors=True,
        optuna_verbose=False)
    
else:
     automl = AutoML(
        mode="Compete", 
        algorithms=["CatBoost", "Xgboost", "LightGBM"], # I check in one run that only these 3 gb tree algorithms play roles in this competition
        total_time_limit=3*3600)

## You can highly customize MLJAR experiments
'''
    automl = AutoML(
        mode = "Compete",
        algorithms=["CatBoost", "Xgboost", "LightGBM"],
        total_time_limit=4*3600,
        start_random_models=10,
        hill_climbing_steps=3,
        top_models_to_improve=3,
        golden_features=True,
        features_selection=True,
        stack_models=True,
        train_ensemble=True,
        explain_level=1,
        ml_task = 'multiclass_classification',
        eval_metric='logloss',
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 5,
            "shuffle": False,
            "stratify": True,
        })
'''

automl.fit(train[x_cols], train[y_col])


# And publish score .... 

# In[21]:


preds = automl.predict_proba(test)

sub = pd.read_csv("../input/tabular-playground-series-jun-2021/sample_submission.csv")
sub[sub.columns[1:]] = preds

sub.to_csv("MLJAR_submission.csv", index=False)


# ## LOOK INSIDE SUBMISSION FILE

# In[22]:


palette = itertools.cycle(sns.color_palette())

plt.figure(figsize=(16, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    c = next(palette)
    sns.histplot(sub, x = f'Class_{i+1}', color=c)
plt.suptitle("Class prediction distribution")


# In[23]:


sub.drop("id", axis=1).describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[24]:


automl.report()

