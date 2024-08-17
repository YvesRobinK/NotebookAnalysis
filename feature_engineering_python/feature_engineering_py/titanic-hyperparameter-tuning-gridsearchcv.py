#!/usr/bin/env python
# coding: utf-8

# # 1. Importing libraries and loading datasets

# In[1]:


import numpy as np
import pandas as pd

# Modelling
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB

# KNeighbors
from sklearn.neighbors import KNeighborsClassifier

# Perceptron
from sklearn.linear_model import Perceptron

# Support Vector Machines
from sklearn.svm import SVC

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# XGBoost
from xgboost import XGBClassifier

# LightGBM
from lightgbm import LGBMClassifier


# In[2]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


# # 2. Explore data

# In[3]:


train_data


# In[4]:


train_data.describe()


# In[5]:


print("Columns: \n{0} ".format(train_data.columns.tolist()))


# # 3. Basic data check

# ## Missing values

# In[6]:


missing_values = train_data.isna().any()
print('Columns which have missing values: \n{0}'.format(missing_values[missing_values == True].index.tolist()))


# In[7]:


print("Percentage of missing values in `Age` column: {0:.2f}".format(100.*(train_data.Age.isna().sum()/len(train_data))))
print("Percentage of missing values in `Cabin` column: {0:.2f}".format(100.*(train_data.Cabin.isna().sum()/len(train_data))))
print("Percentage of missing values in `Embarked` column: {0:.2f}".format(100.*(train_data.Embarked.isna().sum()/len(train_data))))


# ## Check for duplicates

# In[8]:


duplicates = train_data.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates))


# ## Categorical variables

# In[9]:


categorical = train_data.nunique().sort_values(ascending=True)
print('Categorical variables in train data: \n{0}'.format(categorical))


# # 4. Data cleaning

# In[10]:


def clean_data(data):
    # Too many missing values
    data.drop(['Cabin'], axis=1, inplace=True)
    
    # Probably will not provide some useful information
    data.drop(['Name', 'Ticket', 'Fare', 'Embarked'], axis=1, inplace=True)
    
    return data
    
train_data = clean_data(train_data)
test_data = clean_data(test_data)


# In[11]:


train_data.tail()


# # 5. Feature engineering
# 
# Although I have eliminated most of the columns for simplicity, in the future I am planning to recover those columns. They may contain some useful information.  
# For now encoding the `Sex` column and filling `Age` column is enough to run a model.

# In[12]:


train_data['Sex'].replace({'male':0, 'female':1}, inplace=True)
test_data['Sex'].replace({'male':0, 'female':1}, inplace=True)

# Merge two data to get the average Age and fill the column
all_data = pd.concat([train_data, test_data])
average = all_data.Age.median()
print("Average Age: {0}".format(average))
train_data.fillna(value={'Age': average}, inplace=True)
test_data.fillna(value={'Age': average}, inplace=True)


# In[13]:


train_data.tail()


# # 6. Modelling
# 
# Try different models with different parameters to understand which models give better results.

# In[14]:


# Set X and y
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']
test_X = test_data.drop(['PassengerId'], axis=1)


# In[15]:


# To store models created
best_models = {}

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def print_best_parameters(hyperparameters, best_parameters):
    value = "Best parameters: "
    for key in hyperparameters:
        value += str(key) + ": " + str(best_parameters[key]) + ", "
    if hyperparameters:
        print(value[:-2])

def get_best_model(estimator, hyperparameters, fit_params={}):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=estimator, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring="accuracy")
    best_model = grid_search.fit(train_X, train_y, **fit_params)
    best_parameters = best_model.best_estimator_.get_params()
    print_best_parameters(hyperparameters, best_parameters)
    return best_model

def evaluate_model(model, name):
    print("Accuracy score:", accuracy_score(train_y, model.predict(train_X)))
    best_models[name] = model


# In[16]:


print("Features: \n{0} ".format(X.columns.tolist()))


# ## [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 
# Tune the logistic regression model by changing some of its parameters.
# 
# Logistic regression parameters:  
# 
# * **solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’**  
#     * Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
#         * For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
#         * For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
#         * ‘liblinear’ is limited to one-versus-rest schemes.
# 
# > **Warning**  
# > The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:  
# > * ‘newton-cg’ - [‘l2’, ‘none’]  
# > * ‘lbfgs’ - [‘l2’, ‘none’]  
# > * ‘liblinear’ - [‘l1’, ‘l2’]  
# > * ‘sag’ - [‘l2’, ‘none’]  
# > * ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, ‘none’]  
# 
# * **penalty: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’**  
#     * Specify the norm of the penalty:
#         * 'none': no penalty is added;
#         * 'l2': add a L2 penalty term and it is the default choice;
#         * 'l1': add a L1 penalty term;
#         * 'elasticnet': both L1 and L2 penalty terms are added.
# 
# > **Warning**  
# > Some penalties may not work with some solvers. See the parameter solver below, to know the compatibility between the penalty and solver. 
# 
# * **C: float, default=1.0**  
#     Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# 

# In[17]:


# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
hyperparameters = {
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty' : ['l2'],
    'C'       : [100, 10, 1.0, 0.1, 0.01]
}
estimator = LogisticRegression(random_state=1)
best_model_logistic = get_best_model(estimator, hyperparameters)


# In[18]:


evaluate_model(best_model_logistic.best_estimator_, 'logistic')


# ## [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

# ### [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
# 
# * **var_smoothing: float, default=1e-9**  
#     Portion of the largest variance of all features that is added to variances for calculation stability.

# In[19]:


# https://www.analyticsvidhya.com/blog/2021/01/gaussian-naive-bayes-with-hyperpameter-tuning/
hyperparameters = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
estimator = GaussianNB()
best_model_gaussian_nb = get_best_model(estimator, hyperparameters)


# In[20]:


evaluate_model(best_model_gaussian_nb.best_estimator_, 'gaussian_nb')


# ### [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
# 
# * **alpha: float, default=1.0**  
#     Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
#     
# * **fit_prior: bool, default=True**  
#     Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

# In[21]:


# https://medium.com/@kocur4d/hyper-parameter-tuning-with-pipelines-5310aff069d6
hyperparameters = {
    'alpha'     : [0.5, 1.0, 1.5, 2.0, 5],
    'fit_prior' : [True, False],
}
estimator = MultinomialNB()
best_model_multinominal_nb = get_best_model(estimator, hyperparameters)


# In[22]:


evaluate_model(best_model_multinominal_nb.best_estimator_, 'multinominal_nb')


# ### [Complement Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
# 
# * **alpha: float, default=1.0**  
#     Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
# 
# * **fit_prior: bool, default=True**  
#     Only used in edge case with a single class in the training set.
# 
# * **norm: bool, default=False**  
#     Whether or not a second normalization of the weights is performed. The default behavior mirrors the implementations found in Mahout and Weka, which do not follow the full algorithm described in Table 9 of the paper.

# In[23]:


hyperparameters = {
    'alpha'     : [0.5, 1.0, 1.5, 2.0, 5],
    'fit_prior' : [True, False],
    'norm'      : [True, False]
}
estimator = ComplementNB()
best_model_complement_nb = get_best_model(estimator, hyperparameters)


# In[24]:


evaluate_model(best_model_complement_nb.best_estimator_, 'complement_nb')


# ### [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
# 
# * **alpha: float, default=1.0**  
#     Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
#     
# * **fit_prior: bool, default=True**  
#     Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

# In[25]:


hyperparameters = {
    'alpha'     : [0.5, 1.0, 1.5, 2.0, 5],
    'fit_prior' : [True, False],
}
estimator = BernoulliNB()
best_model_bernoulli_nb = get_best_model(estimator, hyperparameters)


# In[26]:


evaluate_model(best_model_bernoulli_nb.best_estimator_, 'bernoulli_nb')


# ## [K-nearest neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# 
# Tune k-nearest neighbors model by changing some of its parameters.
# 
# * **n_neighbors: int, default=5**  
#     Number of neighbors to use by default for kneighbors queries.
# 
# 
# * **weights: {‘uniform’, ‘distance’} or callable, default=’uniform’**  
#     * Weight function used in prediction. Possible values:
#         * ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#         * ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
#         * [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
# 
# 
# * **algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’**  
#     * Algorithm used to compute the nearest neighbors:  
#         * ‘ball_tree’ will use BallTree
#         * ‘kd_tree’ will use KDTree
#         * ‘brute’ will use a brute-force search.
#         * ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
#         
# > Note: fitting on sparse input will override the setting of this parameter, using brute force.
# 
# 
# * **leaf_size: int, default=30**  
#     Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
#     
# * **p: int, default=2**  
#     Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
# 
# * **n_neighbors: int, default=5**  
#     Number of neighbors to use by default for kneighbors queries.

# In[27]:


# https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f
hyperparameters = {
    'n_neighbors' : list(range(1,5)),
    'weights'     : ['uniform', 'distance'],
    'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size'   : list(range(1,10)),
    'p'           : [1,2]
}
estimator = KNeighborsClassifier()
best_model_kneighbors = get_best_model(estimator, hyperparameters)


# In[28]:


evaluate_model(best_model_kneighbors.best_estimator_, 'kneighbors')


# ## [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
# 
# * **penalty: {‘l2’,’l1’,’elasticnet’}, default=None**  
#     The penalty (aka regularization term) to be used.
# 
# * **max_iter: int, default=1000**  
#     The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
#     
# * **eta0: double, default=1**  
#     Constant by which the updates are multiplied.

# In[29]:


# https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/
# https://machinelearningmastery.com/manually-optimize-hyperparameters/
hyperparameters = {
    'penalty'  : ['l1', 'l2', 'elasticnet'],
    'eta0'     : [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_iter' : list(range(50, 200, 50))
}
estimator = Perceptron(random_state=1)
best_model_perceptron = get_best_model(estimator, hyperparameters)


# In[30]:


evaluate_model(best_model_perceptron.best_estimator_, 'perceptron')


# ## [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# 
# * **C: float, default=1.0**  
#     Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# 
# * **kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’**  
#     Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
# 
# 
# * **gamma{‘scale’, ‘auto’} or float, default=’scale’**  
#     Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#     * if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#     * if ‘auto’, uses 1 / n_features.

# In[31]:


# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167
hyperparameters = {
    'C'      : [0.1, 1, 10, 100],
    'gamma'  : [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel' : ['rbf']
}
estimator = SVC(random_state=1)
best_model_svc = get_best_model(estimator, hyperparameters)


# In[32]:


evaluate_model(best_model_svc.best_estimator_, 'svc')


# ## [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
# 
# * **loss: str, default=’hinge’**  
#     The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.  
#     The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.  
#     The ‘log’ loss gives logistic regression, a probabilistic classifier. ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates. ‘squared_hinge’ is like hinge but is quadratically penalized. ‘perceptron’ is the linear loss used by the perceptron algorithm. The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
# 
# * **penalty: {‘l2’, ‘l1’, ‘elasticnet’}, default=’l2’**  
#     The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
#     
# * **alpha: float, default=0.0001**  
#     Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when set to learning_rate is set to ‘optimal’.

# In[33]:


# https://towardsdatascience.com/how-to-make-sgd-classifier-perform-as-well-as-logistic-regression-using-parfit-cc10bca2d3c4
# https://www.knowledgehut.com/tutorials/machine-learning/hyperparameter-tuning-machine-learning
hyperparameters = {
    'loss'    : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'penalty' : ['l1', 'l2', 'elasticnet'],
    'alpha'   : [0.01, 0.1, 1, 10]
}
estimator = SGDClassifier(random_state=1, early_stopping=True)
best_model_sgd = get_best_model(estimator, hyperparameters)


# In[34]:


evaluate_model(best_model_sgd.best_estimator_, 'sgd')


# ## [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
# 
# * **loss: {‘deviance’, ‘exponential’}, default=’deviance’**  
#     The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
# 
# * **learning_rate: float, default=0.1**  
#     Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
# 
# * **n_estimators: int, default=100**  
#     The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# 
# * **subsample: float, default=1.0**  
#     The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
# 
# * **max_depth: int, default=3**  
#     The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.

# In[35]:


# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
hyperparameters = {
    'loss'          : ['deviance', 'exponential'],
    'learning_rate' : [0.01, 0.1, 0.2, 0.3],
    'n_estimators'  : [50, 100, 200],
    'subsample'     : [0.1, 0.2, 0.5, 1.0],
    'max_depth'     : [2, 3, 4, 5]
}
estimator = GradientBoostingClassifier(random_state=1)
best_model_gbc = get_best_model(estimator, hyperparameters)


# In[36]:


evaluate_model(best_model_gbc.best_estimator_, 'gbc')


# ## [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
# 
# * **n_estimators: int, default=50**  
#     The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
#     
# * **learning_rate: float, default=1.0**  
#     Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters.

# In[37]:


# https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20
hyperparameters = {
    'n_estimators'  : [10, 50, 100, 500],
    'learning_rate' : [0.001, 0.01, 0.1, 1.0]
}
estimator = AdaBoostClassifier(random_state=1)
best_model_adaboost = get_best_model(estimator, hyperparameters)


# In[38]:


evaluate_model(best_model_adaboost.best_estimator_, 'adaboost')


# ## [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# 
# Tune decision tree classifier model by changing some of its parameters.
# 
# * **criterion: {“gini”, “entropy”}, default=”gini”**  
#     The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
# 
# * **splitter: {“best”, “random”}, default=”best”**  
#     The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
# 
# * **max_depth: int, default=None**  
#     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#     
# 
# * **min_samples_split: int or float, default=2**  
#     * The minimum number of samples required to split an internal node:
#         * If int, then consider min_samples_split as the minimum number.
#         * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
# 
# 
# * **min_samples_leaf: int or float, default=1**  
#    The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.  
#     * If int, then consider min_samples_leaf as the minimum number.
#     * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

# In[39]:


# https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
# https://www.kaggle.com/gauravduttakiit/hyperparameter-tuning-in-decision-trees
hyperparameters = {
    'criterion'         : ['gini', 'entropy'],
    'splitter'          : ['best', 'random'],
    'max_depth'         : [None, 1, 2, 3, 4, 5],
    'min_samples_split' : list(range(2,5)),
    'min_samples_leaf'  : list(range(1,5))
}
estimator = DecisionTreeClassifier(random_state=1)
best_model_decision_tree = get_best_model(estimator, hyperparameters)


# In[40]:


evaluate_model(best_model_decision_tree.best_estimator_, 'decision_tree')


# ## [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
# 
# * **n_estimators: int, default=100**  
#     The number of trees in the forest.
# 
# 
# * **max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”**  
#     * The number of features to consider when looking for the best split:
#         * If int, then consider max_features features at each split.
#         * If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
#         * If “auto”, then max_features=sqrt(n_features).
#         * If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
#         * If “log2”, then max_features=log2(n_features).
#         * If None, then max_features=n_features.
# 
# > Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
# 
# * **criterion: {“gini”, “entropy”}, default=”gini”**  
#     The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.
# 
# * **max_depth: int, default=None**  
#     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#     
#     
# * **min_samples_split: int or float, default=2**  
#     * The minimum number of samples required to split an internal node:
#         * If int, then consider min_samples_split as the minimum number.
#         * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
# 
# 
# * **min_samples_leaf: int or float, default=1**  
#     The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.  
#      * If int, then consider min_samples_leaf as the minimum number.
#      * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

# In[41]:


# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
hyperparameters = {
    'n_estimators'      : list(range(10, 50, 10)),
    'max_features'      : ['auto', 'sqrt', 'log2'],
    'criterion'         : ['gini', 'entropy'],
    'max_depth'         : [None, 1, 2, 3, 4, 5],
    'min_samples_split' : list(range(2,5)),
    'min_samples_leaf'  : list(range(1,5))
}
estimator = RandomForestClassifier(random_state=1)
best_model_random_forest = get_best_model(estimator, hyperparameters)


# In[42]:


evaluate_model(best_model_random_forest.best_estimator_, 'random_forest')


# ## [XGBClassifier](https://xgboost.readthedocs.io/en/stable/parameter.html)
# 
# * **eta [default=0.3, alias: learning_rate]**  
#     * Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
#     * range: [0,1]
# 
# 
# * **gamma [default=0, alias: min_split_loss]**  
#     * Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
#     * range: [0,∞]
# 
# 
# * **max_depth [default=6]**  
#     * Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
#     * range: [0,∞] (0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist)
# 
# 
# * **lambda [default=1, alias: reg_lambda]**  
#     L2 regularization term on weights. Increasing this value will make model more conservative.
# 
# * **alpha [default=0, alias: reg_alpha]**  
#     L1 regularization term on weights. Increasing this value will make model more conservative.

# In[43]:


# https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d
hyperparameters = {
    'learning_rate' : [0.3, 0.4, 0.5],
    'gamma'         : [0, 0.4, 0.8],
    'max_depth'     : [2, 3, 4],
    'reg_lambda'    : [0, 0.1, 1],
    'reg_alpha'     : [0.1, 1]
}
fit_params = {
    'verbose'               : False,
    'early_stopping_rounds' : 40,
    'eval_metric'           : 'logloss',
    'eval_set'              : [(val_X, val_y)]
}
estimator = XGBClassifier(seed=1, tree_method='gpu_hist', predictor='gpu_predictor', use_label_encoder=False)
best_model_xgb = get_best_model(estimator, hyperparameters, fit_params)


# In[44]:


evaluate_model(best_model_xgb.best_estimator_, 'xgb')


# ## [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
# 
# * **boosting_type: (str, optional (default='gbdt'))**  
#     * ‘gbdt’, traditional Gradient Boosting Decision Tree.
#     * ‘dart’, Dropouts meet Multiple Additive Regression Trees.
#     * ‘goss’, Gradient-based One-Side Sampling.
#     * ‘rf’, Random Forest.
# 
# 
# * **num_leaves: (int, optional (default=31))**  
#     Maximum tree leaves for base learners.
# 
# * **learning_rate: (float, optional (default=0.1))**  
#     Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback. Note, that this will ignore the learning_rate argument in training.
# 
# * **n_estimators: (int, optional (default=100))**  
#     Number of boosted trees to fit.
# 
# * **reg_alpha: (float, optional (default=0.))**  
#     L1 regularization term on weights.
# 
# * **reg_lambda: (float, optional (default=0.))**  
#     L2 regularization term on weights.

# In[45]:


# https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
hyperparameters = {
    'boosting_type' : ['gbdt', 'dart', 'goss'],
    'num_leaves'    : [4, 8, 16, 32],
    'learning_rate' : [0.01, 0.1, 1],
    'n_estimators'  : [25, 50, 100],
    'reg_alpha'     : [0, 0.1, 1],
    'reg_lambda'    : [0, 0.1, 1],
}
estimator = LGBMClassifier(random_state=1, device='gpu')
best_model_lgbm = get_best_model(estimator, hyperparameters)


# In[46]:


evaluate_model(best_model_lgbm.best_estimator_, 'lgbm')


# # WORK IN PROGRESS

# # 7. Submission

# In[47]:


# Get predictions for each model and create submission files
for model in best_models:
    predictions = best_models[model].predict(test_X)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission_' + model + '.csv', index=False)

