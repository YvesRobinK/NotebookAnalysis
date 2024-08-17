#!/usr/bin/env python
# coding: utf-8

# # TPS Nov 2022
# The November Tabular Playground is the chance to practice this skill on blending predictions trying to get better results than using the output of a single model

# <img src="https://i.postimg.cc/Zqg9M9F1/stacking-drawio.png"/>

# In[1]:


import os
import pandas as pd
import numpy as np
import scipy
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
get_ipython().system('pip install mrmr_selection')
from mrmr import mrmr_classif

from tqdm import tqdm
#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 
import random 
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


from sklearn.metrics import log_loss

path = Path('/kaggle/input/tabular-playground-series-nov-2022/')


# In[2]:


# SEED 
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


# ### Files
# - submission_files/ - a folder containing binary model predictions
# - train_labels.csv - the ground truth labels for the first half of the rows in the submission files
# - sample_submission.csv - a sample submission file in the correct format, only containing the row ids for the second half of each file in the submissions folder; your task is to blend together submissions that achieve the improvements in the score.

# In[3]:


submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index

# the ids of the labeled rows (useful later)
gt_ids = labels.index 

# list of files in the submission folder
subs = sorted(os.listdir(path / 'submission_files'))


# # Labels distribution
# The ground truth for these rows are provided in the file train_labels.csv. 

# In[4]:


labels.head(5)


# In[5]:


f, ax = plt.subplots(1, 2, figsize = (15, 7))
labels_names = [f"{p:.2f}%" for p in labels.value_counts()/labels.value_counts().sum()*100]
labels.value_counts().plot(kind='pie', ax=ax[0], labels=labels_names, colors=("r", "b"), ylabel="label")
labels.value_counts().plot(kind='bar', ax=ax[1], color=("r", "b"))


# # Submissions files
# Each file name in the submissions folder corresponds to the logloss score of the the first half of the prediction rows (20k rows) against the ground truth labels in that file. This is effectively the "training" set.

# In[6]:


s0 = pd.read_csv(path / 'submission_files' / subs[0], index_col='id')

score = log_loss(labels, s0.loc[gt_ids])

# Notice the score of the labeled rows matches the file name
print(subs[0],' log_loss:', f'{score:.10f}')


# In[7]:


subs[0:10]


# # Loading all submission files
# we are going to load all submission files into a dataframe with final shape 40k x 5k. Each Submission file will go into a column of the dataframe
# 
# We also calculate the ROC-AUC for each submission file in order to use this metric later

# In[8]:


X_train_orig = np.zeros((s0.shape[0], len(subs)))
roc_auc_scores = {}


for i, name in  tqdm(enumerate(subs)):
    sub = pd.read_csv(path / 'submission_files' / name, index_col='id')
    X_train_orig[:,i] = sub.pred.values
    auc_score=roc_auc_score(labels.label[0:20000], X_train_orig[0:20000,i])
    roc_auc_scores[name]=auc_score
X_train_orig = pd.DataFrame(X_train_orig, columns=subs)


# In[9]:


X_train_orig.head(10)


# # Removing Strange submission files
# we are going to remove strange submission files with values >1 or <0
# As result we remove 108 submission files
# 
# - Instead of removing strange submission files we try to clip all
# - we try to test some results coming from my AutoML notebook https://www.kaggle.com/code/infrarosso/automl-autoviz-pycaret
#  

# In[10]:


# instead of remove features with strange values I'll try to clip
X_train_orig = X_train_orig.loc[:, X_train_orig.max()<=1]
X_train_orig = X_train_orig.loc[:, X_train_orig.min()>=0]
# X_train_orig = X_train_orig.clip(0,1)

# testing some results coming from my AutoML notebook
# ref. https://www.kaggle.com/code/infrarosso/automl-autoviz-pycaret
# 2 variables removed since they were low-information variables
# DROP_LOW_INFO_FEATURES =  ['0.6933054832.csv', '0.6933472206.csv']
# X_train_orig = X_train_orig[list(set(X_train_orig.columns)-set(DROP_LOW_INFO_FEATURES))]


# # Calibration of Train set
# Our train set is composed of several models' predictions output, so it's a good idea to calibrate their output before training a blended model with these
# 
# ref. https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/363778

# In[11]:


# example of an uncalibrated model prediction on train set
X_train_orig[:20000]['0.6222863195.csv']
X_train_orig[:]['0.6223807245.csv'].plot(kind='hist', bins=100)
CalibrationDisplay.from_predictions(labels.label[0:20000], X_train_orig[0:20000]['0.6223807245.csv'], n_bins=20,
                                        strategy='quantile', color='#ffd700')


# In[12]:


# train set calibration 
# ref. https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/363778
roc_auc_scores_calibrated = {}
X_train_calibrated = X_train_orig.copy()
for i, c in tqdm(enumerate(X_train_orig.columns)):
    x_model_calibration = np.zeros(40000)
    model_calibration = IsotonicRegression(out_of_bounds='clip')
    x_model_calibration[:20000] = model_calibration.fit_transform(X_train_orig[:20000][c], labels.label).clip(0.001, 0.999)
    x_model_calibration[20000:] = model_calibration.transform(X_train_orig[20000:][c]).clip(0.001, 0.999)
    X_train_calibrated[c] = x_model_calibration
    auc_score=roc_auc_score(labels.label[0:20000], x_model_calibration[0:20000])
    roc_auc_scores_calibrated[c]=auc_score
pd.DataFrame(X_train_calibrated[:20000][X_train_calibrated.columns[0]]).plot(kind='hist', bins=100)
CalibrationDisplay.from_predictions(labels.label[0:20000], X_train_calibrated[:20000][X_train_calibrated.columns[0]], n_bins=20,
                                        strategy='quantile', color='#ffd700')


# # Dimensionality Reduction with ROC-AUC? Why not!
# 
# <img src="https://i.imgflip.com/6z74zp.jpg"/>
# 
# ## Idea for Feature Selection using ROC-AUC
# A bad classifier can be identified by the ROC curve which looks very similar, if not identical, to the diagonal of the graph, representing the performance of a purely random classifier; ROC-AUC scores close to 0.5 are considered near-random results.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Roc_curve.svg/1024px-Roc_curve.svg.png" width="500px"/>
# 
# Starting from the concepts behind ROC-AUC explained above, we are going to use only submissions where the ROC-AUC is major of a threshold
# 
# 

# In[13]:


#AUC_TH = 0.797
AUC_TH = 0.79
fig = plt.figure(figsize=(30,5))
fig.suptitle("ROC-AUC Distribution")
out = plt.hist(roc_auc_scores_calibrated.values(), bins=200, color=("b"))
plt.plot([AUC_TH,AUC_TH],[0,500], linestyle='dotted', c="r")


# In[14]:


SELECTED_FEATURES = []
TOP_LOG_LOSS_TH = 500
for k, v in roc_auc_scores_calibrated.items():
    if v>=AUC_TH:
        if k in X_train_orig.columns:
            SELECTED_FEATURES.append(k)
print(len(SELECTED_FEATURES), SELECTED_FEATURES[:TOP_LOG_LOSS_TH])
X_train_reduced = X_train_calibrated[SELECTED_FEATURES[:TOP_LOG_LOSS_TH]]
print(X_train_reduced.shape)


# # Feature Selection with MRMR (optional)
# Maximum Relevance — Minimum Redundancy” (aka MRMR) is an algorithm used by Uber’s machine learning platform for finding the “minimal-optimal” subset of features.
# 
# - Enabling FEATURE_SELECTION_ENABLED the train will be done with the features selection algorithm
# - Disabling FEATURE_SELECTION_ENABLED the train will be done with all features availables in the train set (all columns)

# In[15]:


FEATURE_SELECTION_ENABLED = True
# FEATURE SELECTION MRMR
def feature_selection(X, y, k):
    if not FEATURE_SELECTION_ENABLED:
        return X.columns
    out = mrmr_classif(X, y, k)
    print("Features selection:", out)
    return out

if FEATURE_SELECTION_ENABLED:
    FEATURES_SELECTED = feature_selection(X_train_reduced[0:20000], labels.label, 50)
    X_train_reduced = X_train_reduced[FEATURES_SELECTED]


# # Feature Engineering
# ## Adding Unsupervised new feature
# 
# we try to add an unsupervised new feature calculated with kmeans and others as combination of features and mean

# In[16]:


from sklearn.cluster import KMeans

#
X_train_reduced['cluster'] = KMeans(n_clusters=2, random_state=seed).fit_predict(X_train_reduced)
X_train_reduced['mean'] = X_train_reduced.mean(axis=1)
# compose a new feature as combination of two best features
X_train_reduced['compose_feature'] = (X_train_reduced['0.6778730537.csv'] + X_train_reduced['0.6702783631.csv'])/X_train_reduced['mean']


# # Adversarial Validation
# 
# ## Warning! I'm going to give you a spoiler of some results! 🖖
# Here you can find a brief introduction and I'll give you a spoiler about results of the *Adversarial Validation* applied to this competition. 
# To go more in deep on this topic and understand how this technique work, I invite you to read my [dedicated notebook](https://www.kaggle.com/code/infrarosso/tps-nov-2022-adversarial-validation), and if like it **upvote it** 😎
# 
# -----------------------------------------------------------------------------
# 
# <img src="https://i.postimg.cc/wjTT17yX/adversarial-validation-drawio.png" style="float:right" width=300/>
# In a real-world project we have to generalize at all costs, instead a Kaggle competition is more focused on having a model that works on the given test set, especially the private one 😎
# 
# We often expect test data to have the same distribution as training data, but in reality, this is not always the case because if differ, you need to understand if there is any possibility to mitigate the different distributions between training and test data and build a model that works on that set of tests. If you focus on this idea, you will have a better chance to discovering the best validation strategy, that will help you to rank higher in the competition.
# 
# The **adversarial validation** is a technique that allows you to estimate the degree of difference between training and test data
# 
# ## How it work
# To see how it work and ROC-AUC role read my [dedicated notebook](https://www.kaggle.com/code/infrarosso/tps-nov-2022-adversarial-validation)
# 
# ### And Here what we are going to do ?
# The **Adversarial Validation** technique applied to the raw dataset of this competition (TPS Nov 2022) got a ROC-AUC of 0.50414901, so as it is close to 0.5, it means that the training and test data are not easily distinguishable and appear to come from the same distribution (great! 👌)
# 
# But in our notebooks before training our models, we start from the original dataset and we do some pre-processing stages before training, like feature selections, feature engineering and other pipeline elaborations.
# 
# The question running through my head is … **Will our processed train and test data, always be not easily distinguishable?**
# 
# In order to response of this question, here we are going to apply adversarial validation to the processed train/test dataset (X_train_reduced in my case).
# 
# So let's start!

# In[17]:


# all dataset train + test
X_adversarial_validation = X_train_reduced
# 0 label for train dataset (first 20k rows) + 1 label for test dataset (last 20k rows) 
y_adversarial_validation = [0] * 20000 + [1] * 20000

# training all!
adversarial_model = RandomForestClassifier()
adversarial_cv_preds = cross_val_predict(adversarial_model, X_adversarial_validation, y_adversarial_validation, cv=5, n_jobs=-1, method='predict_proba')

# ROC-AUC as adversarial validation score
print(roc_auc_score(y_true=y_adversarial_validation, y_score=adversarial_cv_preds[:, 1]))


# # Result 0.50358424 🚀
# ROC-AUC on raw dataset and on processed dataset are around 0.5, it means that the processed training and test data that we are going to use to modelling are safe because are not easily distinguishable and appear to come from the same distribution.
# 
# <img src="https://i.imgflip.com/7188ac.jpg" width=200/>

# # Blending/Stacking Model
# 

# In[18]:


FOLDS = 15
k_fold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
y = labels
X = X_train_reduced[0:20000]
X_test = X_train_reduced[20000:]


# # RandomSearchCV
# To find the best hyperparameters, we will use the RandomSearchCV. Random search is a technique more faster than GridSearchCV which calculates all possible combinations

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.01, shuffle=True, random_state= seed, stratify=y)
X_train = X_train.reset_index().drop("index", axis=1)
X_validation = X_validation.reset_index().drop("index", axis=1)
y_train = y_train.reset_index()['label']
y_validation = y_validation.reset_index()['label']


# In[20]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


lgbm_fit_params = {
    "eval_metric" : 'binary_logloss', 
    "eval_set" : [(X_validation, y_validation)],   
    'verbose':0
}

lgbm_param_test = {
    'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
    'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
    'num_leaves': sp_randint(6, 50, 100), 
    'min_child_samples': sp_randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': sp_uniform(loc=0.2, scale=0.8), 
    'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
}

cat_fit_params = {    
}

cat_param_test = {
    'depth'         : sp_randint(1, 10),
    'learning_rate' : sp_uniform(),
    'iterations'    : sp_randint(10, 300)
}


# In[21]:


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

lgbm_rs = LGBMClassifier(metric= 'binary_logloss', 
                         random_state=seed, 
                         silent=True, 
                         n_jobs=-1
                        )

cat_rs = CatBoostClassifier(random_seed=seed, 
                                   eval_metric='Logloss',
                                   logging_level='Silent'
                                   )

random_search = RandomizedSearchCV(
    estimator=cat_rs, 
    param_distributions=cat_param_test,     
    scoring='neg_log_loss',
    cv=k_fold,
    refit=True,
    random_state=seed,
    verbose=100, n_iter=50
)

lgbm_opt_parameters = {
    'colsample_bytree': 0.45066268483824545,
    'learning_rate': 0.02,
    'max_depth': 5,
    'min_child_samples': 285,
    'min_child_weight': 0.01,
    'n_estimators': 300,
    'num_leaves': 116,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'subsample': 0.532329735064063
}

cat_opt_parameters = {
   'depth': 3, 'iterations': 102, 'learning_rate': 0.15043761084876184
}


RS_FIT = False # enable it to use RandomSearchCV
if RS_FIT:
    random_search.fit(X, y, **cat_fit_params)
    opt_parameters =  random_search.best_params_
    print(opt_parameters)


# # Hybrid Model Class
# The following class permit us to encapsulate the logic of ensembling N models making a weighted average of their predictions (Soft-Voting)
# 
# We are going to choose several models in order to ensemble them and try to get better results
# 
# ## Hybrid Model Schema
# 
# <img src="https://i.postimg.cc/prVWrkCg/hybrid-model-drawio.png"/>

# In[22]:


class EnsembleHybrid:
   def __init__(self, models=[], weights=[]):
       self.models = models
       self.weights = weights

   def fit(self, X, y):
       # Train models
       for m in self.models:
           print(f"Training {m}...")
           m.fit(X, y)

   def predict_proba(self, X_test):
       y_pred = pd.Series(np.zeros(X_test.shape[0]), index=X_test.index)
       for i, m in enumerate(self.models):
           y_pred += pd.Series(m.predict_proba(X_test)[:,1], index=X_test.index) * self.weights[i]
       return y_pred


# In[23]:


def folds_model(debug=True):
    # several models in order to stacking
    lgbm = LGBMClassifier(**lgbm_opt_parameters, objective='binary', silent=True,
                              random_state=seed,
                              metric='binary_logloss')
    
    cat_boost = CatBoostClassifier(**cat_opt_parameters, random_seed=seed,
                                   eval_metric='Logloss',
                                   logging_level='Silent')


    xgbm = XGBClassifier(objective='binary:logistic',
                     random_state=seed,
                     learning_rate=0.1,
                     n_estimators=100,
                     max_depth=8, 
                     #tree_method='gpu_hist'
                    )
    
    models = [lgbm, cat_boost]
    weights=[0.2, 0.8]
    Y_validations, ensemble_val_preds, ensemble_test_preds, scores=[],[],[],[]
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
        if debug:
            print("\nFold {}".format(fold+1))
        X_fold_val, Y_fold_val = X.iloc[val_idx,:], y.label[val_idx]
        X_fold_train, Y_fold_train = X.iloc[train_idx,:], y.label[train_idx]
        if debug:
            print("Train shape: {}, {}, Valid shape: {}, {}".format(
            X_fold_train.shape, Y_fold_train.shape, X_fold_val.shape, Y_fold_val.shape))

        ensemble_model = EnsembleHybrid(models=models, weights=weights)
        ensemble_model.fit(X_fold_train, Y_fold_train)
        model_prob = ensemble_model.predict_proba(X_fold_val)        

        ensemble_prob = ensemble_model.predict_proba(X_fold_val)
        Y_validations.append(Y_fold_val)
        ensemble_val_preds.append(ensemble_prob)
        ensemble_test_preds.append(ensemble_model.predict_proba(X_test))        

        score=log_loss(Y_fold_val, ensemble_prob)
        scores.append(score)
        if debug:
            print(f"Fold {fold+1} Validation Score = {score:.4f}")

        del X_fold_train, Y_fold_train, X_fold_val, Y_fold_val 
    mix_score = sum(scores)/FOLDS
    if debug:
        print("Total Score (Mixing Folds Predictions) = {:.4f}".format(mix_score))
    return mix_score, ensemble_test_preds

score, ensemble_test_preds = folds_model()


# # Final Submission

# In[24]:


folds_blend = np.zeros(X_test.shape[0])
for j in range(FOLDS):
    folds_blend += ensemble_test_preds[j]
folds_blend = folds_blend/FOLDS

# testing final clip in order to improve logloss
th = 0.01
folds_blend_clipped = np.clip(folds_blend, th, 1-th)
submission['pred'] = folds_blend_clipped
submission.to_csv("final_submission_folds_mix.csv")


# -------------------------------------------------------------------------
# <div style="text-align: center;">
#     <h3>Thanks for watching till the end ;)  </h3>
#     <h2>If you liked this notebook, upvote it ! </h2>
# <img src="https://i.postimg.cc/SsChkSJv/upvote.png" width="70"/>
# </div>
# 
# 
# 
# 
# ---
