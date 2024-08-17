#!/usr/bin/env python
# coding: utf-8

# # 20 Burning XGBoost FAQs Answered to Use the Library Like a Pro
# ## Gradient-boost your XGBoost knowledge by learning these crucial lessons
# ![](https://miro.medium.com/max/2000/1*n6aa_ZbeL5c4O5vvKKJMVg.jpeg)
# <figcaption style="text-align: center;">
#     <strong>
#         Photo by 
#         <a href='https://unsplash.com/@haithemfrd_off?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText'>Haithem Ferdi</a>
#         on 
#         <a href='https://unsplash.com/s/photos/boost?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText'>Unsplash.</a> All images are by the author unless specified otherwise.
#     </strong>
# </figcaption>

# ## Setup

# In[1]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

rcParams["font.size"] = 15

iris = sns.load_dataset("iris").dropna()
penguins = sns.load_dataset("penguins").dropna()

i_input, i_target = iris.drop("species", axis=1), iris[["species"]]
p_input, p_target = penguins.drop("body_mass_g", axis=1), penguins[["body_mass_g"]]

p_input = pd.get_dummies(p_input)

le = LabelEncoder()
i_target = le.fit_transform(i_target.values.ravel())

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    i_input, i_target, test_size=0.2, random_state=1121218
)


X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    p_input, p_target, test_size=0.2, random_state=1121218
)


# XGBoost is a real beast.
# 
# It is a tree-based power horse that is behind the winning solutions of many tabular competitions and datathons. Currently, it is the “hottest” ML framework of the “sexiest” job in the world.
# 
# While basic modeling with XGBoost can be straightforward, you need to master the nitty-gritty to achieve maximum performance.
# 
# With that said, I present you this article, which is the result of
# - hours of reading the documentation (it wasn't fun)
# - crying through some awful but useful Kaggle kernels
# - hundreds of Google keyword searches
# - completely exhausting my Medium membership by reading a lotta articles
# 
# The post answers 20 most burning questions on XGBoost and its API. These should be enough to make you look like you have been using the library forever.

# ## 1. Which API should I choose - Scikit-learn or the core learning API?

# XGBoost in Python has two APIs — Scikit-learn compatible (estimators have the familiar `fit/predict` pattern) and the core XGBoost-native API (a global `train` function for both classification and regression).
# 
# The majority of the Python community, including Kagglers and myself, use the Scikit-learn API.

# In[2]:


import xgboost as xgb

# Regression
reg = xgb.XGBRegressor()
# Classification
clf = xgb.XGBClassifier()


# ```python
# reg.fit(X_train, y_train)
# 
# clf.fit(X_train, y_train)
# ```

# This API enables you to integrate XGBoost estimators into your familiar workflow. The benefits are (and are not limited to):
# 
# - the ability to pass core XGB algorithms into [Sklearn pipelines](https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d?source=your_stories_page-------------------------------------)
# - using a more efficient cross-validation workflow
# - avoiding the hassles that come with learning a new API, etc.

# ## 2. How do I completely control the randomness in XGBoost?

# > The rest of the references to XGBoost algorithms mainly imply the Sklearn-compatible XGBRegressor and XGBClassifier (or similar) estimators.
# 
# The estimators have the `random_state` parameter (the alternative seed has been deprecated but still works). However, running XGBoost with default parameters will yield identical results even with different seeds.

# In[3]:


reg1 = xgb.XGBRegressor(random_state=1).fit(X_train_p, y_train_p)
reg2 = xgb.XGBRegressor(random_state=2).fit(X_train_p, y_train_p)

reg1.score(X_test_p, y_test_p) == reg2.score(X_test_p, y_test_p)


# This behavior is because XGBoost induces randomness only when `subsample` or any other parameter that starts with `colsample_by*` prefix is used. As the names suggest, these parameters have a lot to do with [random sampling](https://towardsdatascience.com/why-bootstrap-sampling-is-the-badass-tool-of-probabilistic-thinking-5d8c7343fb67?source=your_stories_page-------------------------------------).

# ## 3. What are objectives in XGBoost and how to specify them for different tasks?

# Both regression and classification tasks have different types. They change depending on the objective function, the distributions they can work with, and their loss function.
# 
# You can switch between these implementations with the `objective` parameter. It accepts special code strings provided by XGBoost.
# 
# Regression objectives have the `reg:` prefix while classification starts either with `binary:` or `multi:`.
# I will leave it to you to explore the full list of objectives from [this documentation page](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters) as there are quite a few.
# 
# Also, specifying the correct objective and metric gets rid of that unbelievably annoying warning you get when fitting XGB classifiers.

# ## 4. Which booster should I use in XGBoost - gblinear, gbtree, dart?

# > XGBoost has 3 types of gradient boosted learners - these are gradient boosted (GB) linear functions, GB trees and DART trees. You can switch the learners using the `booster` parameter.
# 
# If you ask Kagglers, they will choose boosted trees over linear functions on any day (as do I). The reason is that trees can capture non-linear, complex relationships that linear functions cannot.
# 
# So, the only question is which tree booster should you pass to the `booster` parameter - `gbtree` or `dart`?
# 
# I won’t bother you with the full differences here. The thing you should know is that XGBoost uses an ensemble of decision tree-based models when used with gbtree booster.
# 
# DART trees are an improvement (to be yet validated) where they introduce random dropping of the subset of the decision trees to prevent overfitting.
# 
# In the few small experiments I did with default parameters for `gbtree` and `dart`, I got slightly better scores with dart when I set the `rate_drop` between 0.1 and 0.3.
# 
# For more details, I refer you to [this page](https://xgboost.readthedocs.io/en/latest/tutorials/dart.html) of the XGB documentation to learn about the nuances and additional hyperparameters.

# ## 5. Which tree method should I use in XGBoost?

# There are 5 types of algorithms that control tree construction. You should pass `hist` to `tree_method` if you are doing distributed training.
# 
# For other scenarios, the default (and recommended) is `auto` which changes from `exact` for small-to-medium datasets to `approx.` for large datasets.

# ## 6. What is a boosting round in XGBoost?

# As we said, XGBoost is an ensemble of gradient boosted decision trees. Each tree in the ensemble is called a base or weak learner. A weak learner is any algorithm that performs slightly better than random guessing.
# 
# By combining the predictions of multiples of weak learners, XGBoost yields a final, robust prediction (skipping a lot of details now).
# 
# Each time we fit a tree to the data, it is called a single boosting round.
# 
# So, to specify the number of trees to be built, pass an integer to `num_boost_round` of the Learning API or to `n_estimators` of the Sklearn API.
# 
# Typically, too few trees lead to underfitting, and a too large number of trees lead to overfitting. You will normally tune this parameter with hyperparameter optimization.

# ## 7. What is `early_stopping_rounds` in XGBoost?

# From one boosting round to the next, XGBoost builds upon the predictions of the last tree.
# 
# If the predictions do not improve after a sequence of rounds, it is sensible to stop training even if we are not at a hard stop for `num_boost_round` or `n_estimators`.
# 
# To achieve this, XGBoost provides `early_stopping_rounds` parameter. For example, setting it to 50 means we stop the training if the predictions have not improved for the last 50 rounds.
# 
# It is a good practice to set a higher number for `n_estimators` and change early stopping accordingly to achieve better results.
# 
# Before I show an example of how it is done in code, there are two other XGBoost parameters to discuss.

# ## 8. What are `eval_set`s in XGBoost?

# Early stopping is only enabled when you pass a set of evaluation data to the `fit` method. These evaluation sets are used to keep track of the ensemble's performance from one round to the next.
# 
# A tree is trained on the passed training sets at each round, and to see if the score has been improving, it makes predictions on the passed evaluation sets. Here is what it looks like in code:

# In[4]:


reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)

reg = reg.fit(
    X_train_p,
    y_train_p,
    eval_set=[(X_test_p, y_test_p)],
    early_stopping_rounds=5,
)


# > Set `verbose` to False to get rid of the log messages.

# After the 14th iteration, the score starts decreasing. So the training stops at the 19th iteration because 5 rounds of early stopping is applied.
# 
# It is also possible to pass multiple evaluation sets to `eval_set` as a tuple, but only the last pair will be used when used alongside early stopping.
# 
# > Check out [this post](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/) to learn more about early stopping and evaluation sets. 

# ## 9. When do evaluation metrics have effect in XGBoost?

# You can specify various evaluation metrics using the `eval_metric` of the fit method. Passed metrics only affect internally - for example, they are used to assess the quality of the predictions during early stopping.
# 
# You should change the metric according to the objective you choose. You can find the full list of objectives and their supported metrics on [this page](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters) of the documentation. 
# 
# Below is an example of an XGBoost classifier with multi-class log loss and ROC AUC as metrics:

# In[5]:


clf = xgb.XGBClassifier(
    objective="multi:softprob", n_estimators=200, use_label_encoder=False
)

eval_set = [(X_test_i, y_test_i)]

_ = clf.fit(
    X_train_i,
    y_train_i,
    eval_set=eval_set,
    eval_metric=["auc", "mlogloss"],
    early_stopping_rounds=5,
)


# No matter what metric you pass to `eval_metric`, it only affects the fit function. So, when you call `score()` on the classifier, it will still yield accuracy, which is the default in Sklearn:

# ## 10. What is learning rate (eta) in XGBoost?

# Each time XGBoost adds a new tree to the ensemble, it is used to correct the residual errors of the last group of trees.
# 
# The problem is that this approach is fast and powerful, making the algorithm quickly learn and overfit the training data. So, XGBoost or any other gradient boosting algorithm has a `learning_rate` parameter that controls the speed of fitting and combats overfitting.
# 
# Typical values for `learning_rate` range from 0.1 to 0.3, but it is possible to go beyond these, especially towards 0.
# 
# Whatever value passed to `learning_rate`, it plays as a weighting factor for the corrections made by new trees. So, a lower learning rate means we place less importance on the corrections of the new trees, hence avoiding overfitting.
# 
# A good practice is to set a low number for `learning_rate` and use early stopping with a larger number of estimators (`n_estimators`):

# In[6]:


reg = xgb.XGBRegressor(
    objective="reg:squaredlogerror", n_estimators=1000, learning_rate=0.01
)

eval_set = [(X_test_p, y_test_p)]
_ = reg.fit(
    X_train_p,
    y_train_p,
    eval_set=eval_set,
    early_stopping_rounds=10,
    eval_metric="rmsle",
    verbose=False,
)


# You will immediately see the effect of slow `learning_rate` because early stopping will be applied much later during training (in the above case, after the 430th iteration).
# 
# However, each dataset is different, so you need to tune this parameter with hyperparameter optimization.
# 
# > Check out [this post](https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/) on how to tune learning rate. 

# ## 11. Should you let XGBoost deal with missing values?

# For this, I will give the advice I've got from two different Kaggle Competition Grandmasters.
# 
# 1. If you give `np.nan` to tree-based models, then, at each node split, the missing values are either send to the left child or the right child of the node, depending on what's best. So, at each split, missing values get special treatment, which may lead to overfitting. A simple solution that works pretty well with trees is to fill in nulls with a value different than the rest of the samples, like -999.
# 
# 2. Even though packages like XGBoost and LightGBM can treat nulls without preprocessing, it is always a good idea to develop your own imputation strategy.
# 
# For real-world datasets, you should always investigate the type of missingness (MCAR, MAR, MNAR) and choose an imputation strategy (value-based [mean, median, mode] or model-based [KNN imputers or tree-based imputers]).
# 
# If you are not familiar with these terms, I got you covered [here](https://towardsdatascience.com/going-beyond-the-simpleimputer-for-missing-data-imputation-dd8ba168d505?source=your_stories_page-------------------------------------).

# ## 12. What is the best way of doing cross-validation with XGBoost?

# Even though XGBoost comes with built-in CV support, always go for the Sklearn CV splitters.
# 
# When I say Sklearn, I don't mean the basic utility functions like `cross_val_score` or `cross_validate`.
# No one cross-validates that way in 2021 (well, at least not on Kaggle).
# 
# The method that gives more flexibility and control over the CV process is to use the `.split` function of Sklearn CV splitters and implement your own CV logic inside a `for` loop.
# 
# Here is what a 5-fold CV looks like in code:

# In[7]:


from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1121218,
)

fold = 0
scores = np.empty(5)
for train_idx, test_idx in cv.split(p_input, p_target):
    print(f"Started fold {fold}...")
    # Create the training sets from training indices
    X_cv_train, y_cv_train = p_input.iloc[train_idx], p_target.iloc[train_idx]
    # Create the test sets from test indices
    X_cv_test, y_cv_test = p_input.iloc[test_idx], p_target.iloc[test_idx]
    # Init/fit XGB
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=10000, learning_rate=0.05
    )
    model.fit(
        X_cv_train,
        y_cv_train,
        eval_set=[(X_cv_test, y_cv_test)],
        early_stopping_rounds=50,
        verbose=False,
    )
    # Generate preds, evaluate
    preds = model.predict(X_cv_test)
    rmsle = np.sqrt(mean_squared_log_error(y_cv_test, preds))
    print("RMSLE of fold {}: {:.4f}\n".format(fold, rmsle))
    scores[fold] = rmsle
    fold += 1

print("Overall RMSLE: {:.4f}".format(np.mean(scores)))


# Doing CV inside a `for` loop enables you to use evaluation sets and early stopping, while simple functions like `cross_val_score` does not.

# ## 13. How to use XGBoost in [Sklearn Pipelines](https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d)?

# If you use the Sklearn API, you can include XGBoost estimators as the last step to the pipeline (just like other Sklearn classes):

# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make a simple pipeline
xgb_pipe = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("clf", xgb.XGBClassifier(objective="multi:softmax", use_label_encoder=False)),
    ]
)


# If you want to use `fit` parameters of XGBoost within pipelines, you can easily pass them to the pipeline's `fit` method. The only difference is that you should use the `stepname__parameter` syntax:

# In[9]:


_ = xgb_pipe.fit(
    X_train_i.values,
    y_train_i,  # Make sure to pass the rest after the data
    clf__eval_set=[(X_test_i.values, y_test_i)],
    clf__eval_metric="mlogloss",
    clf__verbose=False,
    clf__early_stopping_rounds=10,
)


# Since we named the XGBoost step as `clf` in the pipeline, every fit parameter should be prefixed with `clf__` for the pipeline to work properly.
# 
# > Also, since `StandardScaler` removes the column names of the Pandas DataFrames, XGBoost kept throwing errors because of the mismatch between `eval_set`s and the training data. So, I used `.values` on both sets to avoid that.

# ## 14. How to improve the default score significantly?

# After establishing a base performance with default XGBoost settings, what can you do to boost the score significantly?
# 
# Many hastily move on to hyperparameter tuning, but it does not always give that huge jump in the score you want. Often, the improvements from parameter optimization can be marginal.
# 
# In practice, any substantial score increase mostly comes from proper feature engineering and using techniques like model blending or stacking.
# 
# You should spend most of your time feature engineering- effective FE comes from doing proper EDA and having a deep understanding of the dataset. Especially, creating domain-specific features might do wonders to the performance.
# 
# Then, try combining multiple models as part of an ensemble. What tends to work reasonably well on Kaggle is to stack the big three - XGBoost, CatBoost, and LightGBM.

# ## 15. What are the most important hyperparameters in XGBoost?

# Hyperparameter tuning should always be the last step in your project workflow.
# 
# If you are short on time, you should prioritize to tune XGBoost's hyperparameters that control overfitting. These are:
# - `n_estimators`: the number of trees to train
# - `learning_rate`: step shrinkage or `eta`, used to prevent overfitting
# - `max_depth`: the depth of each tree
# - `gamma`: complexity control - pseudo-regularization parameter
# - `min_child_weight`: another parameter to control tree depth
# - `reg_alpha`: L1 regularization term (as in LASSO regression)
# - `reg_lambda`: L2 regularization term (as in Ridge regression)

# ## 16. How to tune max_depth in XGBoost?

# `max_depth` is the longest length between the root node of the tree and the leaf node. It is one of the most important parameters to control overfitting.
# 
# The typical values range is 3–10, but it rarely needs to be higher than 5–7. Also, using deeper trees make XGBoost extremely memory-consuming.

# ## 17. How to tune min_child_weight in XGBoost?

# `min_child_weight` controls the sum of weights of all samples in the data when creating a new node. When this value is small, each node will group a smaller and smaller number of samples in each node.
# 
# If it is small enough, the trees will be highly likely to overfit the peculiarities in the training data. So, set a high value for this parameter to avoid overfitting.
# 
# The default value is 1, and its value is only limited to the number of rows in the training data. However, a good range to try for tuning is 2–10 or up to 20.

# ## 18. How to tune gamma in XGBoost?

# A more challenging parameter is `gamma`, and for laypeople like me, you can think of it as the complexity control of the model. The higher the gamma, the more regularization is applied.
# 
# It can range from 0 to infinity - so, tuning it can be tough. Also, it is highly dependent on the dataset and other hyperparameters. This means there can be multiple optimal gammas for a single model.
# 
# Most often, you can find the best gamma within 0–20.

# ## 19. How to tune reg_alpha and reg_lambda in XGBoost?

# These parameters refer to regularization strength on feature weights. In other words, increasing them will make the algorithm more conservative by placing less importance on features with low coefficients (or weights).
# 
# `reg_alpha` refers to L1 regularization of Lasso regression and `reg_lambda` for Ridge regression.
# Tuning them can be a real challenge since their values can also range from 0 to infinity.
# 
# First, choose a wide interval such as [1e5, 1e2, 0.01, 10, 100]. Then, depending on the optimum value returned from this range, choose a few other nearby values.

# ## 20. How to tune random sampling hyperparameters in XGBoost?

# After the above 6 parameters, it is highly recommended to tune those that control random sampling. Indeed, random sampling is another method applied in algorithms to further prevent overfitting.
# 
# - `subsample`: recommended values [0.5 - 0.9]. The proportion of all training samples to be randomly sampled (without replacement) for each boosting round.
# - `colsample_by*`: parameters that start with this prefix refers to the proportion of columns to be randomly selected for
#   - `colsample_bytree`: each boosting round
#   - `colsample_bylevel`: each depth level achieved in a tree
#   - `colsample_bynode`: each node created or in each split
# 
# Like `subsample`, the recommended range is [0.5 - 0.9].

# ## Summary

# Finally, this painfully long but hopefully useful article has come to an end.
# 
# We have covered a lot - how to choose the right API, correct objective and metrics for the task, most important parameters of the fit and some valuable XGBoost best practices collected from constantly updated sources such as Kaggle.
# 
# If you have further questions on XGBoost, post them in the comments. I will try to answer faster than dudes on StackExchange sites (a favor not received while writing this article😉).
