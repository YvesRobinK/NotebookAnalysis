#!/usr/bin/env python
# coding: utf-8

# # EDA which makes sense ⭐️⭐️⭐️⭐️⭐️
# 
# ([Playground Series - Season 3, Episode 23](https://www.kaggle.com/competitions/playground-series-s3e23/): Binary Classification with a Software Defects Dataset)
# 
# This notebook shows
# - a little bit of EDA (this competition doesn't need much of an EDA)
# - how to cross-validate correctly
# - how to preprocess the data for various classifiers
# - how to tune the most important hyperparameters of some models
# - how feature selection can improve the score
# - how an ensemble performs better than any single model

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier

np.set_printoptions(linewidth=195, edgeitems=5)


# # Reading the data

# In[2]:


result_list = []
train = pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/software-defect-prediction/jm1.csv',
                       na_values=['?'])
with pd.option_context("display.min_rows", 6):
    display(train)


# # EDA

# How much of an EDA can we do here?
# 
# - `defects` is the target column, the 21 other columns are the features. All features are numerical.
# - There are no missing values (print `train.isna().sum().sum()` to verify).
# - There are no duplicates (print `train.duplicated().sum()` to verify).
# - The dataset is somewhat unbalanced: Only 23 % of the samples belong to the positive class (i.e., have defect==True). Print `train.defects.mean()` to verify(). We'll use a `StratifiedKFold` for cross-validation. There is no need for upsampling, downsampling or similar voodoo.
# - The dataset has \>100000 rows. For some algorithms (e.g., kernel methods or Neighborhood Components Analysis), this is too much.
# - \>100000 rows with 21 features is abundant data. We don't need to worry much about overfitting.
# 
# In this competition, we don't need a fancy EDA with a lot of colorful diagrams. We can learn much more about the data by tuning and cross-validating a few machine learning models.
# 
# Two points, however, are worth mentioning:
# 1. There are significant differences between the original and the synthetic data. I wouldn't include the original data in training.
# 1. All 21 features are nonnegative and their distributions are right-skewed, resembling an exponential distribution. A log-transformation is recommended.

# ## Differences between original and synthetic data
# 
# `train` and`test` of this competition consist of synthetic data, which was generated based on the real-world dataset `original`. The distributions are not the same, however. 
# 
# The description of the [original dataset] says that nine columns of the dataset are "derived measures":
# - P = volume = V = N * log2(mu) (the number of mental comparisons needed to write a program of length N)
# - V* = volume on minimal implementation = (2 + mu2')*log2(2 + mu2')
# - L  = program length = V*/N
# - D  = difficulty = 1/L
# - L' = 1/D 
# - I  = intelligence = L'*V'
# - E  = effort to write program = V/L 
# - T  = time to write program = E/18 seconds
# 
# Two examples (in the form of diagrams) suffice to show that these relationships don't hold for the synthetic data, and the ratio of defects differs between the datasets anyway:

# In[3]:


_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.scatter(train.l, train.d, s=3, label='synthetic train')
ax1.scatter(original.l, original.d, s=3, label='original')
ax1.set_xlabel('L')
ax1.set_ylabel('D')
ax1.set_title('D = 1/L')
ax1.legend()
ax2.scatter(train.e, train.t, s=3, label='synthetic train')
ax2.scatter(original.e, original.t, s=3, label='original')
ax2.set_xlabel('E')
ax2.set_ylabel('T')
ax2.set_title('T = E/18')
ax2.legend()
bars1 = ax3.bar([0], [train.defects.mean()*100])
bars2 = ax3.bar([1], [original.defects.mean()*100])
ax3.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax3.bar_label(bars1, fmt='{:.0f}%')
ax3.bar_label(bars2, fmt='{:.0f}%')
ax3.set_xticks([0, 1], ['synthetic\ntrain', 'original'])
ax3.set_title('Defect ratio')
plt.tight_layout()
plt.suptitle('Original vs. synthetic datasets', y=1.03)
plt.show()


# **Insight:** In view of these differences, I recommend not to use the original dataset for training models.

# ## A log-transformation is recommended
# 
# All 21 features are nonnegative and their histograms are right-skewed, resembling an exponential distribution.

# In[4]:


_, axs = plt.subplots(7, 3, figsize=(12, 12))
for col, ax in zip(test.columns, axs.ravel()):
    if train[col].dtype == float:
        ax.hist(train[col], bins=100, color='chocolate')
    else: #int
        vc = train[col].value_counts()
        ax.bar(vc.index, vc, color='chocolate')
    ax.set_xlabel(col)
plt.tight_layout()
plt.suptitle('Feature distributions', y=1.02, fontsize=20)
plt.show()


# **Insight:** With this kind of distributions, many models profit if we log-transform the data by prepending a `FunctionTransformer(np.log1p)` to the pipeline.

# In[5]:


_, axs = plt.subplots(7, 3, figsize=(12, 12))
for col, ax in zip(test.columns, axs.ravel()):
    ax.hist(np.log1p(train[col]), bins=100, color='magenta')
    ax.set_xlabel(col)
plt.tight_layout()
plt.suptitle('Feature distributions after log transformation', y=1.02, fontsize=20)
plt.show()


# # Cross-validation

# In[6]:


def cross_val(model, label):
    """Cross-validate the model with a StratifiedKFold
    
    The cross-validation score is printed and added to the global result_list"""
    start_time = datetime.now()
    kf = StratifiedKFold(shuffle=True, random_state=1)
    oof = np.full(len(train), np.nan)
    auc_list = []
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.defects)):
        X_tr = train.iloc[idx_tr]
        X_va = train.iloc[idx_va]
        y_tr = X_tr.pop('defects')
        y_va = X_va.pop('defects')
        model.fit(X_tr, y_tr)
#         print(np.round(model[-1].coef_, 2), np.round(model[-1].intercept_, 2))
        try:
            y_va_pred = model.predict_proba(X_va)[:, 1]
        except AttributeError: # 'LinearSVC' object has no attribute 'predict_proba'
            y_va_pred = model.decision_function(X_va)
        oof[idx_va] = y_va_pred
        auc = roc_auc_score(y_va, y_va_pred)
        auc_list.append(auc)
    auc = np.array(auc_list).mean()
    execution_time = datetime.now() - start_time
    print(f"# AUC {auc:.5f}   time={str(execution_time)[-15:-7]}   {label}")
    result_list.append((auc, label, execution_time))
#     plt.figure(figsize=(6, 2))
#     plt.hist(oof, bins=200, density=True)
#     plt.show()
    return auc
    


# In[7]:


def plot_score_list(label, parameter, xscale='linear'):
    """Show a scatterplot of the scores in the global variable score_list"""
    plt.figure(figsize=(6, 3))
    plt.scatter([p for p, s in score_list],
                [s for p, s in score_list])
    plt.xscale(xscale)
    plt.xlabel(f'{parameter}{" (log-scale)" if xscale == "log" else ""}')
    plt.ylabel('AUC score')
    plt.title(label)
    plt.show()


# # A few models
# ## LinearSVC and LogisticRegression
# 
# Good to know:
# - LinearSVC and LogisticRegression are usually used in a pipeline after a StandardScaler.
# - Linear models are often improved if we add polynomial features. As the dataset has only 21 features, we can afford to add all products of two features by inserting a `PolynomialFeatures(include_bias=False)` into the pipeline before the StandardScaler. With this addition, the dataset is blown up to 252 features.
# - The documentation says for both algorithms: Prefer dual=False when n_samples > n_features.
# - Regularization is controlled by C (low C = high regularization).
# - For logistic regression, `solver='newton-cholesky'` is the fastest solver for the given dataset.
# - For our dataset, LogisticRegression gives good scores with `class_weight='balanced`. All predicted probabilities will be too high, but this doesn't hurt the auc score.
# 
# The plots show that the hyperparameter `C` does matter. You'll rarely get good results without tuning `C`.

# In[8]:


# LinearSVC
score_list = []
for C in np.logspace(-4, -1, 4):
    auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                  PolynomialFeatures(2, include_bias=False),
                                  StandardScaler(),
                                  LinearSVC(dual=False, C=C)),
                    f'Poly-LinearSVC {C=:.2g}')
    score_list.append((C, auc))
plot_score_list('Poly-LinearSVC', 'C', 'log')


# In[9]:


# LogisticRegression
score_list = []
for C in np.logspace(-2, 1, 9):
    auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                  PolynomialFeatures(2, include_bias=False),
                                  StandardScaler(),
                                  LogisticRegression(dual=False, C=C,
                                                     class_weight='balanced',
                                                     max_iter=1500,
                                                     random_state=1,
                                                     solver='newton-cholesky')),
                    f'Poly-LogisticRegression {C=:.2g}')
    score_list.append((C, auc))
plot_score_list('Poly-LogisticRegression', 'C', 'log')


# # Kernel approximation
# 
# **Kernel methods** are somewhat neglected on Kaggle, but they will play a role in this competition.
# 
# They have been developed based on two observations:
# 1. Linear machine learning models are limited because most machine learning tasks require a nonlinear solution.
# 2. Linear machine learning algorithms typically only apply linear operations in a vector space (vector addition, multiplication by a scalar, and scalar products) to the training and test data.
# 
# The math behind kernel methods takes some time to digest, but the main idea is that the data is mapped nonlinearly into a higher-dimensional vector space, and the linear classifier is applied to the higher-dimensional data. Because of the higher dimension, the linear classifier has more degrees of freedom to find a good decision surface. One can as well think of kernel methods as a form of automated feature engineering which generates nonlinear features for the linear classifier.
# 
# The drawback of kernel methods is their complexity: They work with a kernel matrix of size n_samples\*n_samples, and calculating a 100000\*100000 matrix for a 100000-row dataset takes too much time and memory. This is where **kernel approximation** enters the scene. Kernel approximation has almost the same benefits as the full kernel method, but at lower cost.
# 
# The following notebook cell cross-validates a linear classifier (logistic regression) with kernel approximation (Nyström approximation). Notice how the pipeline steps match the insight gained from EDA:
# - We start with a log transformation because our EDA has shown the right-skewed nonnegative feature distributions.
# - We then apply the Nyström approximation because we know that the dataset is too large to compute the full kernel matrix.
# 
# **TLDR:** The pipeline is the same as before, except that we have replaced `PolynomialFeatures` by `Nystroem`, and the score is substantially better.

# In[10]:


# Kernel approximation for logistic regression
score_list = []
n_components = 400
for C in np.logspace(-3, -2, 9):
    auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                  Nystroem(n_components=n_components, random_state=10), # gamma=1/21
                                  StandardScaler(),
                                  LogisticRegression(dual=False, C=C,
                                                     class_weight='balanced',
                                                     max_iter=1500,
                                                     random_state=1,
                                                     solver='newton-cholesky')),
                    f'Nyström-LogisticRegression {n_components=} {C=:.2g}')
    score_list.append((C, auc))
plot_score_list('Nyström LogisticRegression', 'C', 'log')


# Further reading about the topic:
# - [Kernel method](https://en.wikipedia.org/wiki/Kernel_method) on Wikipedia
# - [Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) in the scikit-learn User Guide
# - [Low-rank matrix approximations](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations) on Wikipedia

# ## ExtraTreesClassifier
# 
# Good to know:
# - The most important hyperparameter of ExtraTreesClassifier is min_samples_leaf.
#   - High min_samples_leaf: Estimator is fast, but underfits.
#   - Low min_samples_leaf: Estimator is slow and overfits.
# - Setting max_features to 1.0 usually gives better auc scores than the default of max_features='sqrt', but slows down the training process.

# In[11]:


# ExtraTreesClassifier
score_list = []
for min_samples_leaf in [10, 20, 50, 100, 150]:
    auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                  ExtraTreesClassifier(n_estimators=100,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=1.0,
                                         random_state=1)),
                    f"ET {min_samples_leaf=}")
    score_list.append((min_samples_leaf, auc))
plot_score_list('ExtraTreesClassifier', 'min_samples_leaf')


# The plot shows how the ExtraTreesClassifier scores depend on the hyperparameter `min_samples_leaf`. A value of 100 seems to be the optimum.
# 
# In [A comparison of hyperparameter settings for ExtraTreesClassifier](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/446078), @broccolibeef shows that running ExtraTreesClassifier on a subset of the features improves the score:

# In[12]:


auc = cross_val(make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])],
                                                remainder='passthrough'),
                              FunctionTransformer(np.log1p),
                              ExtraTreesClassifier(n_estimators=100,
                                                   min_samples_leaf=100,
                                                   max_features=1.0,
                                                   random_state=1)),
                f"Feature-selection-ET")


# ## RandomForestClassifier
# 
# Good to know:
# - The most important hyperparameter of a random forest is min_samples_leaf.
#   - High min_samples_leaf: Estimator is fast, but underfits.
#   - Low min_samples_leaf: Estimator is slow and overfits.
# - Setting max_features to 1.0 usually gives better auc scores than the default of max_features='sqrt', but slows down the training process.

# In[13]:


# RandomForestClassifier
score_list = []
for min_samples_leaf in [100, 150, 200, 250, 300]:
    auc = cross_val(RandomForestClassifier(n_estimators=100,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=1.0,
                                           random_state=1),
                    f"RF {min_samples_leaf=}")
    score_list.append((min_samples_leaf, auc))
plot_score_list('RandomForestClassifier', 'min_samples_leaf')

# AUC 0.78969   time=0:05:33   RF min_samples_leaf=50
# AUC 0.79057   time=0:04:50   RF min_samples_leaf=100
# AUC 0.79104   time=0:04:12   RF min_samples_leaf=200
# AUC 0.79074   time=0:03:43   RF min_samples_leaf=300
# AUC 0.79058   time=0:03:29   RF min_samples_leaf=400


# The plot shows how the random forest scores depend on `min_samples_leaf`. A value of 150 to 200 seems to be the optimum.

# ## KNeighborsClassifier
# 
# Good to know:
# - KNeighborsClassifier is usually used in a pipeline after a StandardScaler.
# - The most important hyperparameter is n_neighbors.
# 

# In[14]:


# KNeighborsClassifier
score_list = []
for n_neighbors in range(200, 800, 100):
    auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                  StandardScaler(),
                                  KNeighborsClassifier(n_neighbors=n_neighbors,
                                                       weights='distance')),
                    f"KNN {n_neighbors=}")
    score_list.append((n_neighbors, auc))
plot_score_list('KNeighborsClassifier', 'n_neighbors')


# The plot shows that the optimum of `n_neighbors` lies around 400. Large datasets (>100k samples in this competition) usually go together with a high `n_neighbors`.

# ## HistGradientBoostingClassifier
# 
# Good to know:
# - If the sample size is larger than 10000, HistGradientBoostingClassifier uses 10 % of its training data as an internal validation set for early stopping. This means that we don't need to tune `max_iter`.

# In[15]:


# HistGradientBoostingClassifier
auc = cross_val(HistGradientBoostingClassifier(random_state=1),
                f"HistGradientBoostingClassifier")


# HistGradientBoostingClassifier gives a good result without any hyperparameter tuning! We can expect even better scores if we tune its hyperparameters.

# # Ensemble
# 
# We ensemble three models, `HistGradientBoostingClassifier`, `RandomForestClassifier` and logistic regression with kernel approximation: 

# In[16]:


ensemble = VotingClassifier(
    [('hgb', HistGradientBoostingClassifier(random_state=1)),
     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', 
                                              ['iv(g)', 't', 'b', 'n', 'lOCode', 'v',
                                               'branchCount', 'e', 'i', 'lOComment'])],
                                            remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=100,
                                               min_samples_leaf=100,
                                               max_features=1.0,
                                               random_state=1))),
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                                      Nystroem(n_components=400, random_state=1),
                                      StandardScaler(),
                                      LogisticRegression(dual=False, C=0.0032,
                                                         max_iter=1500,
                                                         random_state=1)))],
    voting='soft',
    weights=[0.4, 0.4, 0.2])
auc = cross_val(ensemble, 'Ensemble(HGB+ET+NY)')


# # Final comparison
# 
# To easily compare the merits of the tuned classifiers, a diagram is helpful:
# 

# In[17]:


result_df = pd.DataFrame(result_list, columns=['auc', 'label', 'time'])
result_df['time'] = result_df.time.dt.seconds
result_df['model'] = result_df.label.str.split(expand=True).iloc[:,0]
result_df = result_df.sort_values('auc', ascending=False)
# with pd.option_context("display.precision", 5): display(result_df)
result_df = result_df.drop_duplicates('model', keep='first')
plt.figure(figsize=(6, len(result_df) * 0.4))

def color_map(row):
    if row['label'].startswith('Ensemble'): return 'green'
    if row['auc'] > 0.79050: return 'lightgreen'
    return 'yellow'

colors = result_df.apply(color_map, axis=1)
bars = plt.barh(np.arange(len(result_df)), result_df.auc, color=colors)
plt.gca().bar_label(bars, fmt='%.5f')
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.xlim(0.785, 0.795)
plt.xticks([0.785, 0.79, 0.795])
plt.gca().invert_yaxis()
plt.xlabel('AUC score (higher is better)')
plt.show()


# **Insight:**
# - The ensemble (dark green) obviously wins.
# - HistGradientBoosting and ExtraTrees have the best scores of the single models. Other gradient-boosting methods (LightGBM, XGBoost, Catboost) will have similarly good scores. In all cases, you'll need to tune the hyperparameters.
# - Random forest and Nyström logistic regression belong to the top group as well. All models with a light green bar in the chart must be included in the final ensemble.
# - Logistic regression and LinearSVC have lower scores. This means that the class boundary of the given dataset is nonlinear (and not even quadratic). If you want to get a good score with a linear model, you need to engineer features which account for the nonlinearity of the data.
# - KNeighborsClassifier ends up last.

# # AUC score explained
# 
# (Skip this section if you already know what the AUC score is.)
# 
# Many binary classification competitions on Kaggle are evaluated on *area under the ROC curve*.
# 
# Classification models usually predict a probability (`predict_proba`) or a decision function value for every sample. To convert these numbers into class predictions (`predict`), the predicted probability is compared to a threshold. If the probability is above the threshold, the model predicts the positive class; if the probability is below the threshold, the model predicts the negative class.
# 
# By varying the threshold, we can choose different trade-offs between false positives and false negatives. The receiver operating curve plots the true positive rate against the false positive rate while varying the threshold. 
# 

# In[18]:


def plot_roc_curve(model):
    """Plot the ROC curve for the first fold"""
    kf = StratifiedKFold(shuffle=True, random_state=1)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.defects)):
        X_tr = train.iloc[idx_tr]
        X_va = train.iloc[idx_va]
        y_tr = X_tr.pop('defects')
        y_va = X_va.pop('defects')
        model.fit(X_tr, y_tr)
        y_va_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_va_pred)

        plt.figure(figsize=(4, 4))
        fpr, tpr, _ = roc_curve(y_va, y_va_pred)
        plt.plot(fpr, tpr, color='r', lw=2)
        plt.fill_between(fpr, tpr, color='r', alpha=0.2)
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.gca().set_aspect('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver operating characteristic\nArea under the curve = {auc:.5f}")
        plt.show()
        break

plot_roc_curve(ensemble)


# - The lower left corner of the diagram corresponds to a very high threshold so that the model always predicts the negative class. There are zero positive predictions, i.e. 0 true positives and 0 false positives.
# - The upper right corner of the diagram corresponds to a very low threshold so that the model always predicts the positive class. The model predicts all positive samples correctly (recall = true positive rate = 1.0), and it predicts all negative samples as false positives (false positive rate = 1.0).
# 
# The AUC score is the area under the red curve. It usually is between 0.5 (area of the lower triangle) and 1.0 (area of the square).
# - With a perfect classification model, the curve would pass through the upper left corner (recall = 1.0 and no false positives) for an area under the curve of 1.0.
# - With a dummy classification model which makes random predictions or always predicts the same class, the curve follows the diagonal of the square for an area under the curve of 0.5.
# 
# As @iqbalsyahakbar notes [here](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444690), with auc scoring you need to call the model's `predict_proba()` function rather than `predict()`.
# 
# See the [Wikipedia article](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for the long explanation.
# 
# # Submission

# In[19]:


get_ipython().run_cell_magic('time', '', "ensemble.fit(train.iloc[:, :-1], train.defects)\ny_pred = ensemble.predict_proba(test)[:, 1]\nsubmission = pd.Series(y_pred, index=test.index, name='defects')\nsubmission.to_csv('submission.csv')\n!head submission.csv\n")


# In[ ]:




