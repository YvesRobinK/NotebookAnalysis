#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Machine Failure CatBoost Baseline</p>
#           
# The aim of this notebook is simply to create a baseline CatBoost model. As of this writing the baseline model from this notebook results in a top 10% score on the public leaderboard. 

# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Table of Contents</p>
# 
# * [Feature Engineering](#section-one)
# * [Hyperparameter Tuning](#section-two)
# * [Feature Importance](#section-three)
# * [Submission](#section-four)

# In[1]:


# Installs
get_ipython().system('pip install polars')
get_ipython().system('pip install lets-plot')

# Imports
import statistics
import optuna
import plotly
import polars as pl
import numpy as np
import plotly.figure_factory as ff

from lets_plot import *
from lets_plot.bistro.corr import *
from lets_plot.mapping import as_discrete
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.model_selection import train_test_split

# So the plots look nice
LetsPlot.setup_html()
plotly.offline.init_notebook_mode(connected = True)


# In[2]:


# Read in the data
df_original = pl.read_csv("/kaggle/input/machine-failure-predictions/machine failure.csv").drop('UDI')
df_train = pl.read_csv("/kaggle/input/playground-series-s3e17/train.csv").drop('id')
df_test = pl.read_csv("/kaggle/input/playground-series-s3e17/test.csv").drop('id')

# Stack training and original data
df_train = pl.concat([df_train, df_original], how='vertical')


# <a id="section-one"></a>
# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Feature Engineering</p>

# In[3]:


df_train = df_train.with_columns(
    (pl.col('Torque [Nm]')*pl.col('Rotational speed [rpm]')).alias('Power')
)

df_test = df_test.with_columns(
    (pl.col('Torque [Nm]')*pl.col('Rotational speed [rpm]')).alias('Power')
)


# <a id="section-two"></a>
# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Hyperparameter Tuning</p>

# In[4]:


# # Optuna tune
# def objective_cat(trial):
    
#     # Initalizing stuff
#     nfolds = 5

#     # Params
#     params = {'loss_function': 'Logloss',
#               'eval_metric': 'AUC',
#               'verbose': False,
#               'random_seed': 19970507,
#               'learning_rate': trial.suggest_float("learning_rate", 1e-2, 0.10, log = True),
#               'iterations': trial.suggest_int("iterations", 900, 1300),
#               'depth': trial.suggest_int("depth", 2, 6),
#               'subsample': trial.suggest_float('subsample', 0.7, 1.0)
#                }

#     # Add folds indicator
#     df = df_train.with_columns(pl.lit(-1).alias('fold')) # Create folds column
#     kf = model_selection.StratifiedKFold(n_splits = nfolds, shuffle = True)

#     for fold, (train_idx, valid_idx) in enumerate(kf.split(X = df.drop("Machine failure"), y = df.get_column("Machine failure"))):
#         df[valid_idx, "fold"] = fold 
      
#     # Initialize list to store auc metrics
#     auc = []

#     for fold in range(0, nfolds):
#         tfold = df.filter(pl.col('fold') != fold).drop('fold') 
#         vfold = df.filter(pl.col('fold') == fold).drop('fold') 
        
#         feature_names = tfold.drop('Machine failure').columns
    
#         xtrain = tfold.drop('Machine failure').to_numpy()
#         xvalid = vfold.drop('Machine failure').to_numpy()

#         ytrain = tfold.get_column('Machine failure').to_numpy()
#         yvalid = vfold.get_column('Machine failure').to_numpy()    
    
#         tpool = Pool(xtrain, ytrain, feature_names = feature_names, cat_features = ['Product ID', 'Type'])
#         vpool = Pool(xvalid, yvalid, feature_names = feature_names, cat_features = ['Product ID', 'Type'])

#         model = CatBoostClassifier(**params)
#         model.fit(tpool, eval_set = vpool)

#         preds = model.predict_proba(vpool)

#         auc.append(roc_auc_score(yvalid, preds[:,1]))

#     print(f'Average validation AUC from {nfolds}-fold CV: {statistics.mean(auc)}, std: {np.std(auc)}')
#     return(statistics.mean(auc))
    
# # Run the study
# study_cat = optuna.create_study(direction = 'maximize')
# study_cat.optimize(objective_cat, n_trials = 20)


# In[5]:


# # Table of best hyperparameters
# cat_table = [["Parameter", "Optimal Value from Optuna"],
#             ["Iterations (num_boost_rounds)", study_cat.best_params['iterations']],
#             ['Learning Rate (eta)', round(study_cat.best_params['learning_rate'], 3)],
#             ['Max Depth (max_depth)', round(study_cat.best_params['depth'], 3)],
#             ['Subsample (subsample)', round(study_cat.best_params['subsample'], 3)]]
             
# colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
# print(f'AUC from CV for the final hyperparameters: {study_cat.best_value}')

# ff.create_table(cat_table, colorscale = colorscale, height_constant = 10)


# <a id="section-three"></a>
# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Final Baseline Model</p>

# In[6]:


# Fit the CatBoost model 
params = {'loss_function': 'Logloss',
          'eval_metric': 'AUC',
          'verbose': False,
          'random_seed': 19970507,
          'learning_rate': 0.021, 
          'iterations': 1384, 
          'depth': 7, 
          'subsample': 0.984}

df = df_train

feature_names = df.drop('Machine failure').columns

xtrain = df.drop('Machine failure').to_numpy()
ytrain = df.get_column('Machine failure').to_numpy()
   
tpool = Pool(xtrain, ytrain, feature_names = feature_names, cat_features = ['Product ID', 'Type'])

cat_model = CatBoostClassifier(**params)
cat_model.fit(tpool, eval_set=tpool)

# Get test data into the correct format
xtest = df_test.to_numpy()
feature_names = df_test.columns
test_pool = Pool(xtest, feature_names=feature_names, cat_features = ['Product ID', 'Type'])

# Get final predictions
preds_test = cat_model.predict_proba(test_pool)[:,1]


# <a id="section-four"></a>
# <p style="font-family: monospace; 
#           font-weight: bold; 
#           letter-spacing: 1px; 
#           color: black; 
#           font-size: 200%; 
#           text-align: left;
#           padding: 0px; 
#           border-bottom: 5px solid #FF9999" >Submission</p>

# In[7]:


# Submit
sub = pl.read_csv('/kaggle/input/playground-series-s3e17/sample_submission.csv')
sub = sub.with_columns(pl.Series(name='Machine failure', values=preds_test))
sub.write_csv('submission.csv')
sub.head()

