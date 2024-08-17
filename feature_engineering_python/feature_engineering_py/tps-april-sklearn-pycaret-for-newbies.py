#!/usr/bin/env python
# coding: utf-8

# - [Overview](#overview)
#     - [Etiquette](#etiquette)
#     
# - [Feature Engineering](#feature_engineering)
#     - [(1) Data Import](#data_import)
#     - [(2) Data Combine](#data_combine)
#     - [(3) Handling Missing Values](#handling_missing_values)
#     - [(4) Feature Encoding](#feature_encoding)
#     - [(5) Split Data](#split_data)
#     - [(6) Limitation](#limitation)
#     
# - [Scikit Learn](#scikit_learn)
#     - [(1) Data Split](#data_split)
#     - [(2) Base Model - Decision Tree](#base_model_tree)
#     - [(3) Create Helper Class and Submission Function](#helper_class)
#         * [(A) DecisionTreeClassifier](#DecisionTreeClassifier)
#         * [(B) RandomForestClassifier](#RandomForestClassifier)
#         * [(C) LightGBM](#lightgbm)
#         * [(D) Feature Importance](#feature_importance)
#         * [(E) Submission](#submission) 
# 
# - [PyCaret](#pycaret)
#     - [(1) Intro](#intro)
#     - [(2) PyCaret Tutorials](#pycaret_tutorials)
#     - [(3) Base Model](#base_model)
#         + [(A) Initialize Setup](#initialize_setup)
#         + [(B) Comparing All Models](#compare_models)
#         + [(C) Create Model](#create_pycaret_model)
#         + [(D) Tune Model](#tune_pycaret_model)
#         + [(E) Plot Model](#plot_pycaret_model)
#         + [(F) Predictions and Submissions](#preds_submissions)
#     - [(4) Review PyCaret](#review_pycaret)
# 
#         
# > If you want to know how to create table of contents in Kaggle Notebooks, please check this article [Create Table of Contents in a Notebook](https://www.kaggle.com/dcstang/create-table-of-contents-in-a-notebook) by David Tang

# <a id="overview"></a>
# ## Overview
# - This is my personal tutorial sharing with my students as example. 
# - The whole processes will be shared from EDA to Modeling and Evaluation, Finally Submission. 
#     + Let's Check My [EDA Code](https://www.kaggle.com/j2hoon85/2021-april-play-ground-eda-for-kaggle-newbies)
# - The well-known notebooks shared will be enough for students to learn Kaggle as an entry level. 
# 
# > Happy to Code

# <a id='etiquette'></a>
# ### Etiquette
# - When students get codes and ideas from other notebooks, then please make sure to leave a reference and upvote it as well. 👆👆👆

# <a id="feature_engineering"></a>
# ## Feature Engineering
# - After EDA, it's time to conduct Feature Engineering. 
# - If you are not familiar with this concept, then please read a book
# 
# ![Feature Engineering for Machine Learning](https://learning.oreilly.com/library/cover/9781491953235/250w/)
#     
# - And If you need a short summary about feature engineering, then please check this article as well. 
#     + [@Chris Deotte Feature Engineering Techniques](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575)

# <a id='data_import'></a>
# ### (1) Data Import
# - Let's get datasets

# In[1]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import os

print("Version Pandas", pd.__version__)
print("Version Matplotlib", matplotlib.__version__)
print("Version Numpy", np.__version__)
print("Version Seaborn", sb.__version__)

os.listdir('../input/tabular-playground-series-apr-2021/')


# In[2]:


BASE_DIR = '../input/tabular-playground-series-apr-2021/'
train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')
sample_submission = pd.read_csv(BASE_DIR + 'sample_submission.csv')

train.shape, test.shape, sample_submission.shape


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


sample_submission.head()


# <a id='data_combine'></a>
# ### (2) Data Combine
# - Let's combine train with test as all_df

# In[6]:


all_df = pd.concat([train, test])
all_df.shape


# <a id='handling_missing_values'></a>
# ### (3) Handling Missing Values
# - Let's fill with some value in each column.
# > *Important Note:* This idea is from [TPS Apr 2021 LightGBM CV](https://www.kaggle.com/jmargni/tps-apr-2021-lightgbm-cv). Thank you. 
# 

# In[7]:


# Start
print("Before Handling:", all_df.shape)

# Age
age_dict = all_df[['Age', 'Pclass']].dropna().groupby('Pclass').mean().round(0).to_dict()
print("Avg. Mean of Age by Pclass:", age_dict)
all_df['Age'] = all_df['Age'].fillna(all_df.Pclass.map(age_dict['Age']))

# Cabin
all_df["Cabin"].fillna("No Cabin", inplace = True)
print("Values from Cabin: ", all_df["Cabin"].unique())
all_df['Cabin_Code'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())
print("Values from Cabin Code: ", all_df["Cabin_Code"].unique())

# Fare
print("Avg. Mean:", np.round(all_df['Fare'].mean(), 2))
all_df['Fare'] = all_df['Fare'].fillna(round(all_df['Fare'].mean(), 2))

# Embarked
all_df["Embarked"].fillna("X", inplace = True)
print("Values from Embarked: ", all_df["Embarked"].unique())

# Delete Columns
all_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
print("After Handling:", all_df.shape)


# <a id='feature_encoding'></a>
# ### (4) Feature Encoding
# - Let's check each column's data type

# In[8]:


all_df.info()


# - Dataset will be divided into two groups - categorical variables and numerical variables
# 

# In[9]:


cat_cols = ['Pclass', 'Sex', 'Cabin_Code', 'Embarked']
num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived']

onehot_df = pd.get_dummies(all_df[cat_cols])
print("onehot_df Shape:", onehot_df.shape)

num_df = all_df[num_cols]
print("num_df Shape:", num_df.shape)

all_cleansed_df = pd.concat([num_df, onehot_df], axis=1)
print("all_cleansed_df Shape:", all_df.shape)


# > Important note: When conducting feature encoding, Newbies must understand difference between ordinal encoding, label encoding, and one-hot encoding. See. https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
# 

# <a id='split_data'></a>
# ### (5) Split Data
# - Based on Feature Engineering, the final task is to re-split all data into independent variables and dependent variables. 
# 
# 
# 
# 

# In[10]:


X = all_cleansed_df[:train.shape[0]]
print("X Shape is:", X.shape)
y = X['Survived']
X.drop(['Survived'], axis=1, inplace=True)
test_data = all_cleansed_df[train.shape[0]:].drop(columns=['Survived'])
test_data.info()


# In[11]:


X.shape, y.shape


# In[12]:


test_data


# <a id="limitation"></a>
# ### (6) Limitation
# - What I missed here is not to create new variable so-called wealthy class and others, yet. My assumption is wealthy people were more survived than other group. This will be compared baseline model with the more upgraded model, reflecting new feature. If some readers get this idea, then please implement it. Hope to see a better model. 

# <a id="scikit_learn"></a>
# ## Scikit Learn
# - Let's make simple model based on Scikit Learn Framework.
# - URL: https://scikit-learn.org/stable/
# 
# ![](https://scikit-learn.org/stable/_images/scikit-learn-logo-notext.png)

# <a id="data_split"></a>
# ### (1) Data Split
# - We know test data exists as final testset, so we create validation set from sklearn module. 
# - We will use [Stratified Sampling](https://medium.com/@411.codebrain/train-test-split-vs-stratifiedshufflesplit-374c3dbdcc36). 

# In[13]:


get_ipython().system('pip install scikit-learn==0.23.2')


# In[14]:


import sklearn
print(sklearn.__version__)


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, stratify = X[['Pclass']], random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# <a id="base_model_tree"></a>
# ### (2) Base Model - Decision Tree
# - Let's create Base Model
# - Model Evaluation is [Accuracy](https://www.kaggle.com/c/tabular-playground-series-apr-2021/overview/evaluation)
#     + If you want to know more, please see https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification

# In[16]:


from sklearn.metrics import accuracy_score
def acc_score(y_true, y_pred, **kwargs):
    return accuracy_score(y_true, (y_pred > 0.5).astype(int), **kwargs)


# - Let's evaluate of base model using validation set
# - AUC & Accurarcy are measured at this moment. 

# In[17]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train, y_train)
predictions = tree_model.predict_proba(X_val)
AUC = roc_auc_score(y_val, predictions[:,1])
ACC = acc_score(y_val, predictions[:,1])
print("Model AUC:", AUC)
print("Model Accurarcy:", ACC)
print("\n")

fpr, tpr, _ = roc_curve(y_val, predictions[:,1])

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(fpr, tpr)
ax.text(x = 0.3, 
        y = 0.4, 
        s = "Model AUC is {}\n\nModel Accuracy is {}".format(np.round(AUC, 2), np.round(ACC, 2)), 
        fontsize=16, bbox=dict(facecolor='gray', alpha=0.3))
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC curve')

plt.show()


# - Good, great. 
# - Now, we finally submit file to competition. 
# - `.5` Threshold could be different, depending upon your assumption. 
#     + If you are not familar with the concept Threshold, then please read this article. https://developers.google.com/machine-learning/crash-course/classification/thresholding

# In[18]:


final_preds = tree_model.predict(test_data)
binarizer = np.vectorize(lambda x: 1 if x >= .5 else 0)
prediction_binarized = binarizer(final_preds)
submission = pd.concat([sample_submission,pd.DataFrame(prediction_binarized)], axis=1).drop(columns=['Survived'])
submission.columns = ['PassengerId', 'Survived']
submission.to_csv('tree_base_submission.csv', index=False)


# <a id="helper_class"></a>
# ### (3) Create Helper Class and Submission Function
# - we need to create helper class with common tasks such as model, train, predict, fit, feature_importance, and even ROC Curve Graph
# - 
# 

# In[19]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

SEED = 0 # for Reproducibility

# class 
class sk_helper(object):
    def __init__(self, model, seed = 0, params={}):
        params['random_state'] = seed
        self.model = model(**params)
        self.model_name = str(model).split(".")[-1][:-2]
        
    # train
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    # predict
    def predict(self, y_val):
        return self.model.predict(y_val)
    
    # inner fit
    def fit(self, x, y):
        return self.model.fit(x, y)
    
    # feature importance
    def feature_importances(self, X_train, y_train):
        return self.model.fit(X_train, y_train).feature_importances_
        
    # roc_curve
    def roc_curve_graph(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        
        print("model_name:", self.model_name)
        model_name = self.model_name
        preds_proba = self.model.predict_proba(X_val)
        preds = (preds_proba[:, 1] > 0.5).astype(int)
        auc = roc_auc_score(y_val, preds_proba[:, 1])
        acc = accuracy_score(y_val, preds)
        confusion = confusion_matrix(y_val, preds)
        print('Confusion Matrix')
        print(confusion)
        print("Model AUC: {0:.3f}, Model Accuracy: {1:.3f}\n".format(auc, acc))
        fpr, tpr, _ = roc_curve(y_val, predictions[:,1])
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(fpr, tpr)
        ax.text(x = 0.3, 
                y = 0.4, 
                s = "Model AUC is {}\n\nModel Accuracy is {}".format(np.round(auc, 2), np.round(acc, 2)), 
                fontsize=16, bbox=dict(facecolor='gray', alpha=0.3))
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC curve of {}'.format(model_name), fontsize=16)

        plt.show()


# - Now, Let's test if this works or not

# <a id="DecisionTreeClassifier"></a>
# #### (A) DecisionTreeClassifier
# - This is DecisionTreeClassifier Model from [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). 

# In[20]:


get_ipython().run_cell_magic('time', '', "tree_params = {'max_depth': 6}\n\ntree_model = sk_helper(model=DecisionTreeClassifier, seed=SEED, params=tree_params)\ntree_model.roc_curve_graph(X_train, y_train, X_val, y_val)\n")


# <a id="RandomForestClassifier"></a>
# #### (B) RandomForestClassifier
# - This is RandomForestClassifier Model from [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

# In[21]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestClassifier\n\nrf_params = {\n    'n_jobs': -1,\n    'n_estimators': 500,\n     'warm_start': True, \n     #'max_features': 0.2,\n    'max_depth': 6,\n    'min_samples_leaf': 2,\n    'max_features' : 'sqrt',\n    'verbose': 1\n}\n\nrf_model = sk_helper(model=RandomForestClassifier, seed=SEED, params=rf_params)\nrf_model.roc_curve_graph(X_train, y_train, X_val, y_val)\n")


# <a id="lightgbm"></a>
# #### (C) LightGBM
# - Let's implement LightGBM with best parameters. You can found it here: https://www.kaggle.com/jmargni/tps-apr-2021-lightgbm-optuna

# In[22]:


get_ipython().run_cell_magic('time', '', "\nimport lightgbm\nfrom lightgbm import LGBMClassifier\nprint(lightgbm.__version__)\nlgb_params = {\n    'metric': 'auc',\n    'n_estimators': 10000,\n    'objective': 'binary',\n}\n\nlgb_model = sk_helper(model=LGBMClassifier, seed=SEED, params=lgb_params)\nlgb_model.roc_curve_graph(X_train, y_train, X_val, y_val)\n")


# <a id="feature_importance"></a>
# #### (D) Feature Importance 
# - Let's draw graph feature importance plot

# In[23]:


tree_features = tree_model.feature_importances(X_train, y_train)
rf_features = rf_model.feature_importances(X_train, y_train)
lgb_features = lgb_model.feature_importances(X_train, y_train)


# In[24]:


cols = X.columns.values
feature_df = pd.DataFrame({'features': cols, 
                          'Decision Tree': tree_features, 
                          'RandomForest': rf_features, 
                          'LightGBM': lgb_features})

feature_df


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sb
import matplotlib.pyplot as plt

width = 0.3
x = np.arange(0, len(feature_df.index))

## ax[0] graph
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (16, 16)) # Option sharex=True
ax[0].bar(x - width/2, feature_df['Decision Tree'], color = "#0095FF", width = width)
ax[0].bar(x + width/2, feature_df['RandomForest'], color = "#E6C0B1", width = width)
ax[0].set_xticks(x)
ax[0].set_xticklabels(feature_df['features'], rotation=90)

## ax[0] legend
colors = {'Decision Tree':'#0095FF', 'RandomForest':'#E6C0B1'} 
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax[0].legend(handles, labels, bbox_to_anchor = (0.95, 0.95))
ax[0].set_title("Feature Importance between Decision Tree and RandomForest", fontsize=20)

## ax[1] graph
ax[1].bar(x, feature_df['LightGBM'], color = "#60F09E")
ax[1].set_xticks(x)
ax[1].set_xticklabels(feature_df['features'], rotation=90)
ax[1].set_title("Feature Importance of LightGBM", fontsize=20)

## plt manage
## plt.xticks(x, feature_df['features'], rotation=90)
plt.tight_layout()
plt.show()


# - Each algorithm provides different feature important values. 
#     + If you want to know this concept further, please read this article: [How to Calculate Feature Importance With Python](https://machinelearningmastery.com/calculate-feature-importance-with-python/)
# - The main goal here is to improve a predictive model, deleting some features. For example, feature SisSp and Parch in both algorithm are not quite important. So, At this moment, we can delete them.
#     + This code will be worked with feature engineering section. 
# 

# ### 

# <a id="submission"></a>
# #### (E) Submission
# - Now, we create submission function. 

# In[26]:


import numpy as np
from datetime import datetime

version = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

def final_submission(model, data, version):
    final_preds = model.predict(data)
    binarizer = np.vectorize(lambda x: 1 if x >= .5 else 0)
    prediction_binarized = binarizer(final_preds)
    submission = pd.concat([sample_submission,pd.DataFrame(prediction_binarized)], axis=1).drop(columns=['Survived'])
    submission.columns = ['PassengerId', 'Survived']
    submission.to_csv('Sklearn of Submit Date {} Submission.csv'.format(version), index=False)
    
final_submission(lgb_model, test_data, version)


# - Now, We will move on `PyCaret` Framework. 

# <a id='pycaret'></a>
# ## PyCaret
# ![](https://miro.medium.com/max/2048/1*Cku5-rqmqSIuhUyFkIAdIA.png)
# 
# - PyCaret.. Caret in R?
#     + My 1st reaction on this.. when got heard .. was "is it copy of [caret](https://cran.r-project.org/web/packages/caret/vignettes/caret.html) package in R?"
#     
# - Let's look at this framework. 
# 

# <a id="intro"></a>
# ### (1) Intro
# - URL: https://pycaret.org/about/
# > It's an open source low-code machine learning library that aims to reduce cycle time from hypothesis to insights. 
# 
# - Point 1. Simple and Easy to use
# > All the operations performed in PyCaret are automatically stored in a custom `Pipeline` that is fully orchestrated for `deployment`. 
# - Point 2. Python Wrapper
# > Around several machine learning libraries and frameworks such as scikit-learn, XGBoost, Microsoft LightGBM, spaCy and many more. 
# - Point 3. Train Multiple Models 
# > It trains multiple models SIMULTANEOUSLY.. (interesting!) and outputs a table comparing performaces of each model you developed. 
# - Point 4. [PyCaret on GPU](https://pycaret.readthedocs.io/en/latest/installation.html)
# > `PyCaret >= 2.2` provides the option to use GPU for select model training and hyperparameter tuning. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default slim version or the full version. The following estimators can be trained on GPU.

# <a id="pycaret_tutorials"></a>
# ### (2) PyCaret Tutorials 
# - If you want to learn some basic tutorials, then please visit here: https://pycaret.readthedocs.io/en/latest/tutorials.html
#      + I will skip out introducing some basic codes here. 

# <a id="base_model"></a>
# ### (3) Base Model
# - Unfortunately, if you want to use this framework, then you should install it with following command. 
# > !pip install pycaret
# 
# - It will take a few minutes.

# - We will downgrade it this since we are going to use PyCaret. Some issues are reported.
#     + https://github.com/pycaret/pycaret/issues/1140

# In[27]:


# !pip uninstall scikit-learn -y
get_ipython().system('pip install pycaret==2.2.3')


# - Let's check version

# In[28]:


# check version
from pycaret.utils import version
import sklearn
print("pycaret version:", version())
print("sklearn version:", sklearn.__version__)


# <a id="initialize_setup"></a>
# #### (A) Initialize Setup
# - Before you train, you need to setup with following code. 
# - Here, We need to combine two dataframe, X and y and stored it as train_data
# 

# In[29]:


all_df_pycaret = pd.concat([X, y], axis=1)
all_df_pycaret['Survived'] = all_df_pycaret['Survived'].astype('int64')
all_df_pycaret.info()


# - This code is simple but very powerful. 
# - If you want to study more setting up more options such as Data Preparation, Scale and Transformation, and Feature Engineering, Feature Selection. Then please visit here: https://pycaret.org/train-test-split/
# 

# In[30]:


from pycaret.classification import *

setup(data = all_df_pycaret, 
      target = 'Survived', 
      fold = 3,
      silent = True
     )


# <a id="compare_models"></a>
# #### (B) Comparing All Models
# - This is starting point to recommend a model, evaluating performaces of all models when the setup is completed. 
# - This function trains all models in the model library and scores them using stratified cross validation for metric evaluation.

# In[31]:


get_ipython().run_cell_magic('time', '', "\nbest_model = compare_models(sort = 'Accuracy', n_select = 3)\n")


# In[32]:


print(best_model)


# - To me, the best model is `Gradient Boosting Classifier`
# - So, We will create GBC model in this tutorial. 

# <a id="create_pycaret_model"></a>
# #### (C) Create Model
# - This function creates a model and scores it using Stratified Cross Validation.The output prints a score grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold (default = 10 Fold). This function returns a trained model object. 
# - You can create different models here. 
#     + Examples: https://pycaret.org/classification/
# 
# ```python
# # train logistic regression model
# lr = create_model('lr') #lr is the id of the model
# # check the model library to see all models
# models()
# # train rf model using 5 fold CV
# rf = create_model('rf', fold = 5)
# # train svm model without CV
# svm = create_model('svm', cross_validation = False)
# # train xgboost model with max_depth = 10
# xgboost = create_model('xgboost', max_depth = 10)
# # train xgboost model on gpu
# xgboost_gpu = create_model('xgboost', tree_method = 'gpu_hist', gpu_id = 0) #0 is gpu-id
# # train multiple lightgbm models with n learning_rate<br>import numpy as np
# lgbms = [create_model('lightgbm', learning_rate = i) for i in np.arange(0.1,1,0.1)]
# # train custom model
# from gplearn.genetic import SymbolicClassifier
# symclf = SymbolicClassifier(generation = 50)
# sc = create_model(symclf)
# ```

# In[33]:


get_ipython().run_cell_magic('time', '', "gbc_model = create_model('gbc')\n")


# <a id="tune_pycaret_model"></a>
# #### (D) Tune Model
# - This function tunes the hyperparameters of a model and scores it using Stratified Cross Validation. The output prints a score grid that shows Accuracy, AUC, Recall Precision, F1, Kappa, and MCCby fold (by default = 10 Folds). This function returns a trained model object.
#     + Examples: https://pycaret.org/classification/#tune-model
#     
# ```python
# # train a decision tree model with default parameters
# dt = create_model('dt')
# 
# # tune hyperparameters of decision tree
# tuned_dt = tune_model(dt)
# 
# # tune hyperparameters with increased n_iter
# tuned_dt = tune_model(dt, n_iter = 50)
# 
# # tune hyperparameters to optimize AUC
# tuned_dt = tune_model(dt, optimize = 'AUC') #default is 'Accuracy'
# 
# # tune hyperparameters with custom_grid
# params = {"max_depth": np.random.randint(1, (len(data.columns)*.85),20),
#           "max_features": np.random.randint(1, len(data.columns),20),
#           "min_samples_leaf": [2,3,4,5,6],
#           "criterion": ["gini", "entropy"]
#           }
# 
# tuned_dt_custom = tune_model(dt, custom_grid = params)
# 
# # tune multiple models dynamically
# top3 = compare_models(n_select = 3)
# tuned_top3 = [tune_model(i) for i in top3]
# ```

# In[34]:


get_ipython().run_cell_magic('time', '', 'tuned_gbc = tune_model(gbc_model, n_iter = 50)\n')


# <a id = "plot_pycaret_model"></a>
# #### (E) Plot Model
# - This function takes a trained model object and returns a plot based on the test / hold-out set. The process may require the model to be re-trained in certain cases. See list of plots supported below. Model must be created using create_model() or tune_model().
#     + Examples: https://pycaret.org/classification/#plot-model
# 
# ```python
# #create a model
# lr = create_model('lr')
# #plot a model
# plot_model(lr)
# ```
# 
# - Many options are avaiable. 

# In[35]:


plot_model(tuned_gbc, plot = 'confusion_matrix')


# In[36]:


plot_model(tuned_gbc, plot = 'feature_all')


# In[37]:


plot_model(tuned_gbc, plot = 'auc')


# <a id="preds_submissions"></a>
# #### (F) Predictions and Submissions
# - Now, it's time to predict and submit it. 
# 

# In[38]:


predictions = predict_model(tuned_gbc, data = test_data)
predictions.info()


# In[39]:


submission = pd.read_csv(BASE_DIR + 'sample_submission.csv')
submission['Survived'] = predictions['Label']
submission.to_csv('PyCaret Submission.csv', index=False)
submission.head()


# - Finally Done. 

# <a id="review_pycaret"></a>
# ### (4) Review PyCaret 
# > Generalization of Machine Learning for Non-coders. 
# 
# - The phrase is everything for Newbies. 
# - Well, Scikit-Learn library is very wonderful framework. but, it's somehow difficult to compare each models. To do this, it needs to implement seperately. 
# - But, it's simple to compare with one line code. 
# ```python
# compare_models()
# ```
# - And even, model, visualization, evaluation, and so on. It becomes so easy to do it. 
# - I guess that this kind of framework will become trend in Machine Learning World. 
# - Found Good Intro Article of PyCaret. Let's read it more. [PyCaret: Better Machine Learning with Python](https://towardsdatascience.com/pycaret-better-machine-learning-with-python-58b202806d1e)
# 
# > Happy To code.

# ### 

# #### 
