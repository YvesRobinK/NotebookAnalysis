#!/usr/bin/env python
# coding: utf-8

# # ***ICR classification problem***

# Binary classification problem that is focued on predicting if a person have any of the defined medical conditions. For evaluation metric we are using balanced log loss.

# # Installing and importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
pd.set_option('display.max_columns', 100)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#installing featurewiz
get_ipython().system('pip -q install featurewiz --no-index --find-links=file:///kaggle/input/featurewiz')


# # Preprocessing data

# In[3]:


data = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv', index_col='Id')
data


# There is a very low number of missing values for the most of features that contains it. How ever there is features like BQ and EL that have 60 missing values (about 10% of all values). For others we will use simple imputer and impute missing values with their medians (mostly of features doesnt have distribution that is simular to normal), for the BQ we will impute 0 (there was discusion about that all missing values of BQ indicate that person belongs to class 0, so there is big correlation between those two and because I suppose that BQ is some kind of feature that is existential for disease to happend, 0 will be a good value to use over missing values) and for EL we will make regression model that will predict values based on other features (at first we made model for both EL and BQ, after that I have found out about correlation between class and BQ and made changes for solving missing values of BQ).

# In[4]:


#quantifying categorical variable EJ
data.EJ=data.EJ.map({'A':0, 'B':1})

#filling missing values of features that have a few of them, for EL we will make regression model for prediction
data_na = data.drop(['EL', 'BQ'], axis=1)
imputer = SimpleImputer(strategy='median').fit(data_na)
data_na = pd.DataFrame(imputer.transform(data_na),columns=data_na.columns, index=data_na.index)
data_na = pd.concat([data_na, data[['BQ','EL']]], axis=1)
data_post = data_na.copy()
data_post.BQ = data_post.BQ.fillna(0)
data_post


# In[5]:


##BQ datasets
#train_bq_df = data_post[~data_post.BQ.isna()]
#X_train_bq_df=train_bq_df.drop(['BQ', 'EL', 'Class'], axis=1)
#y_train_bq_df=train_bq_df.BQ

# test_bq_df = data_post[data_post.BQ.isna()]
# X_test_bq_df=test_bq_df.drop(['BQ', 'Class'], axis=1)

##making grid for hyperparamters optimization for feature selection
#bq_grid_fs = GridSearchCV(XGBRegressor(), param_grid={'n_estimators':[50,80,100], 'eta': [0.001, 0.005, 0.01, 0.03, 0.1, 1]},
#                           n_jobs=-1, cv=10, verbose=1,scoring='neg_mean_squared_error')
#bq_grid_fs.fit(X_train_bq_df, y_train_bq_df)

##making model for feature selection
#bq_model_fs = XGBRegressor(n_estimators=bq_grid_fs.best_params_['n_estimators'],
#                           eta=bq_grid_fs.best_params_['eta'])
#bq_model_fs.fit(X_train_bq_df, y_train_bq_df)

##chosing 10 most important features
#feature_importances = bq_model_fs.feature_importances_
#sorted_indices = np.argsort(feature_importances)[::-1]
#features_bq = sorted_indices[:10]

#X_train_bq_df = X_train_bq_df.iloc[:,features_bq]
#X_test_bq_df = X_test_bq_df.iloc[:,features_bq]

##making grid for hyperparamters optimization for prediction
#bq_grid = GridSearchCV(XGBRegressor(), param_grid={'n_estimators':[50,80,100], 'eta': [0.001, 0.005, 0.01, 0.03, 0.1, 1]},
#                           n_jobs=-1, cv=10, verbose=1,scoring='neg_mean_squared_error')
#bq_grid.fit(X_train_bq_df, y_train_bq_df)

##making model for prediction
#bq_model = XGBRegressor(n_estimators=bq_grid.best_params_['n_estimators'],
#                           eta=bq_grid.best_params_['eta'])
#bq_model.fit(X_train_bq_df, y_train_bq_df)

##predicting missing values and imputing it in dataset
#bq_pred=bq_model.predict(X_test_bq_df)
#data_post.loc[data_post.BQ.isna(), 'BQ']=bq_pred


# In[6]:


#EL datasets
train_el_df = data_post[~data_post.EL.isna()]
X_train_el_df=train_el_df.drop(['BQ', 'EL', 'Class'], axis=1)
y_train_el_df=train_el_df.EL

test_el_df = data_post[data_post.EL.isna()]
X_test_el_df=test_el_df.drop(['EL', 'Class'], axis=1)


#making grid for hyperparamters optimization for feature selection
el_grid_fs = GridSearchCV(XGBRegressor(), param_grid={'n_estimators':[50,80,100], 'eta': [0.001, 0.005, 0.01, 0.03, 0.1, 1]},
                           n_jobs=-1, cv=10, verbose=1,scoring='neg_mean_squared_error')
el_grid_fs.fit(X_train_el_df, y_train_el_df)

#making model for features selection
el_model_fs = XGBRegressor(n_estimators=el_grid_fs.best_params_['n_estimators'], eta=el_grid_fs.best_params_['eta'])
el_model_fs.fit(X_train_el_df, y_train_el_df)

#chosing 10 most important features
feature_importances = el_model_fs.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
features_el = sorted_indices[:10]

X_train_el_df = X_train_el_df.iloc[:,features_el]
X_test_el_df = X_test_el_df.iloc[:,features_el]

#making grid for hyperparamters optimization for prediction
el_grid = GridSearchCV(XGBRegressor(), param_grid={'n_estimators':[50,80,100], 'eta': [0.001, 0.005, 0.01, 0.03, 0.1, 1]},
                           n_jobs=-1, cv=10, verbose=1,scoring='neg_mean_squared_error')
el_grid.fit(X_train_el_df, y_train_el_df)

#making model for prediction
el_model = XGBRegressor(n_estimators=el_grid.best_params_['n_estimators'], eta=el_grid.best_params_['eta'])
el_model.fit(X_train_el_df, y_train_el_df)

el_pred=el_model.predict(X_test_el_df)
data_post.loc[data_post.EL.isna(), 'EL']=el_pred


# In[7]:


data_post


# Because we dont know how much missing values test set have, we will use simple imputer and impute medians for missing values.

# In[8]:


#importing test set and preprocessing it
test=pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
test_helper = test[['BQ', 'EL']]
test.EJ=test.EJ.map({'A':0, 'B':1})
test=test.drop(['BQ', 'EL'], axis=1)
new_test = test.drop('Id', axis=1)
new_test = pd.concat([new_test, test_helper], axis=1)
new_test.BQ = new_test.BQ.fillna(0)
new_test.fillna(new_test.median(), inplace=True)
new_test


# # Creating new features

# For feature creation we will use polynomial features of 2 degree. That will make every combination of features with max polynomial degree of 2 (products of each combination of features and features sqaured).

# In[9]:


poly_train = PolynomialFeatures()
data_transform = data_post.drop('Class', axis=1).copy()
data_poly = pd.DataFrame(poly_train.fit_transform(data_transform),
                         columns=poly_train.get_feature_names_out(), index=data_transform.index)
data_poly=data_poly.drop('1', axis=1)
data_poly=pd.concat([data_poly, data_post[['Class']]], axis=1)
data_poly.Class = data_poly.Class.astype('int64')
data_poly


# In[10]:


#we will make same changes to test set
poly_test = PolynomialFeatures()
test_poly = pd.DataFrame(poly_test.fit_transform(new_test), columns=poly_test.get_feature_names_out())
test_poly = test_poly.drop('1', axis=1)
test_poly


# # Feature selection

# For feature selection i prefer using featurewiz. It uses SULOV method that drops highly correlated features based on their mutual information score and then uses XGBoost for more precise feature selection.

# In[11]:


from featurewiz import featurewiz
out1, _ = featurewiz(data_poly, 'Class', corr_limit=0.5, verbose=2)


# In[12]:


data_sel = data_poly[out1]
data_sel['Class']=data_poly.Class
data_sel


# # Hyperparameter optimization

# Because competition evaluation metric is balanced log loss that is slightly changed, i created both metrics with scorer maker and use both of for evaluating models. Besides this two, accuracy is always useful to have during evaulation.

# In[13]:


from sklearn.metrics import make_scorer

def competition_log_loss(y_true, y_pred):
    
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    
    return (log_loss_0 + log_loss_1)/2

def balanced_log_loss(y_true, y_pred):

    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)

    w_0 = 1 / N_0
    w_1 = 1 / N_1

    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1

    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))

    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)

    return balanced_log_loss/(N_0+N_1)

cll = make_scorer(competition_log_loss, greater_is_better=False, needs_proba=True)
bll = make_scorer(balanced_log_loss, greater_is_better=False, needs_proba=True)


# At first i started with algorithms like tree, LGBM, XGBoost, logistic regression and ensemble them with stacking. Results were not so great and i found out that LGBM have better result them stacked models and all other models by themself, so i choosed only LGBM for future modeling. For evaluation i am using full dataset and cross-validation for testing. I am not dividing it into classic train/test because there is not enough data for that split (in my opinion of course). For LGBM models i used chain grid search, where i dont use all hyperparameters in one grid, rather making more grids for few hyperparameters. This does not assure best tunned model, but saves a lot of time. For scoring i used log loss instead of self made metrics beacuse it gave better results than both of those metrics (bll and cll). Hyperparameters are tunned using 5-Fold CV.

# In[14]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate


X = data_sel.drop('Class', axis=1)
y = data_sel['Class']


# In[15]:


# gridTree = GridSearchCV(DecisionTreeClassifier(), {'max_depth':[2,3,4,5,6,7,8], 'min_samples_leaf': [2,3,4,5,6,7,8],
#                                                    'min_samples_split': [2,3,4,5,6,7,8]},cv=5, n_jobs=-1, scoring='accuracy')
# gridTree.fit(X, y)
# gridTree.best_params_

# gridGB = GridSearchCV(GradientBoostingClassifier(), {'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.3, 0.5, 1],
#                                                      'n_estimators': [50,80,100,200],
#                                                      'max_leaf_nodes': [8, 10, 15, 20, 14, 30]},cv=5, n_jobs=-1,
#                       scoring='accuracy' ,verbose=1)
# gridGB.fit(X, y)
# gridGB.best_params_

# gridX = GridSearchCV(XGBClassifier(), {'learning_rate': [0.01, 0.05, 0.1, 0.26, 0.3, 0.5, 1],
#                                        'n_estimators': [50,80,100,200],'gamma':[0.1, 0.3, 0.5, 0.7, 1],
#                                        'max_leaves': [8,10,14,20,26,32]},
#                                        cv=5, n_jobs=-1, scoring='accuracy',verbose=1)
# gridX.fit(X, y)
# gridX.best_params_

# gridRandom = GridSearchCV(RandomForestClassifier(), {'n_estimators': [30,50,80, 100, 120, 170, 200, 300],
#                                                     'criterion' : ['gini', 'log_loss']},
#                          n_jobs=-1, cv=5, scoring='accuracy', verbose=1)
# gridRandom.fit(X, y)
# gridRandom.best_params_

# modelX = XGBClassifier(n_estimators=gridX.best_params_['n_estimators'],
#                        learning_rate=gridX.best_params_['learning_rate'],
#                        gamma=gridX.best_params_['gamma'],
#                        max_leaves=gridX.best_params_['max_leaves'], eval_metric='error',
#                        random_state=12)
# modelGB = GradientBoostingClassifier(n_estimators=gridGB.best_params_['n_estimators'],
#                                      learning_rate=gridGB.best_params_['learning_rate'],
#                                      max_leaf_nodes=gridGB.best_params_['max_leaf_nodes'],
#                                      random_state=12)
# modelRandom = RandomForestClassifier(min_samples_leaf=gridTree.best_params_['min_samples_leaf'],
#                                      max_depth=gridTree.best_params_['max_depth'],
#                                      min_samples_split=gridTree.best_params_['min_samples_split'],
#                                      n_estimators=gridRandom.best_params_['n_estimators'],
#                                      criterion=gridRandom.best_params_['criterion'], random_state=12)
# modelTree = DecisionTreeClassifier(min_samples_leaf=gridTree.best_params_['min_samples_leaf'],
#                                      max_depth=gridTree.best_params_['max_depth'],
#                                      min_samples_split=gridTree.best_params_['min_samples_split'],
#                                    criterion='log_loss', random_state=12)
# stack = StackingClassifier([('xgboost', modelX), ('lbgm', modelL),
#                             ('rf', modelRandom), ('tree', modelTree), ('gb', modelGB)], cv=5, n_jobs=-1)

# modelTree.fit(X, y)
# modelRandom.fit(X, y)
# modelGB.fit(X, y)
# modelX.fit(X, y)
# stack.fit(X, y)


# results_tree = cross_val_score(modelTree, X, y, cv=10, n_jobs=-1, scoring=bll).mean()
# results_random = cross_val_score(modelRandom, X, y, cv=10, n_jobs=-1, scoring=bll).mean()
# results_gb = cross_val_score(modelGB, X, y, cv=10, n_jobs=-1, scoring=bll).mean()
# results_xgb = cross_val_score(modelX, X, y, cv=10, n_jobs=-1, scoring=bll).mean()
# results_stack = cross_val_score(stack, X, y, cv=10, n_jobs=-1, scoring=bll).mean()

# results_tree_acc = cross_val_score(modelTree, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()
# results_random_acc = cross_val_score(modelRandom, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()
# results_gb_acc = cross_val_score(modelGB, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()
# results_xgb_acc = cross_val_score(modelX, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()
# results_stack_acc = cross_val_score(stack, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()


# print(f'Tree: bll = {results_tree}, accuracy = {results_tree_acc}')
# print(f'Random forest: bll = {results_random}, accuracy = {results_random_acc}')
# print(f'Gradient boost: bll = {results_gb}, accuracy = {results_gb_acc}')
# print(f'XGBoost: bll = {results_xgb}, accuracy = {results_xgb_acc}')
# print(f'Stack: bll = {results_stack}, accuracy = {results_stack_acc}')


# In[16]:


# gridLGBM_1 = GridSearchCV(LGBMClassifier(), {'learning_rate': [0.01, 0.05, 0.1, 0.26, 0.3, 0.5, 1],
#                                         'n_estimators': [50,80,100,200], 'num_leaves': [8,10,14,20,26,31],
#                                        'max_depth': [-1,2,4,6,8,10]},
#                                         cv=KFold(n_splits=5, shuffle=True, random_state=12), n_jobs=-1,
#                         scoring='neg_log_loss', verbose=1)
# gridLGBM_1.fit(X, y)
# print(f'Best hyperparameters: {gridLGBM_1.best_params_}')

# modelLGBM_1 = LGBMClassifier(n_estimators=gridLGBM_1.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1.best_params_['max_depth'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[17]:


#making few indepedent grids for LGBM to reduce time spent on searching for best hyperparams fits
# gridLGBM_2 = GridSearchCV(modelLGBM_1, {'subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1], 
#                                           'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
#                                        'subsample_freq': [0,1,3,5,8,10]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_2.fit(X, y)
# print(f'Best hyperparameters: {gridLGBM_2.best_params_}')

# modelLGBM_2 = LGBMClassifier(n_estimators=gridLGBM_1.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1.best_params_['max_depth'],
#                         subsample = gridLGBM_2.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2.best_params_['subsample_freq'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[18]:


# gridLGBM_3 = GridSearchCV(modelLGBM_2, {'min_split_gain': [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1],
#                                           'min_child_weight' : [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
#                                           'min_child_samples':[5,10,15,20, 25,30]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_3.fit(X, y)
# print(f'Best hyperparameters: {gridLGBM_3.best_params_}')

# modelLGBM_3 = LGBMClassifier(n_estimators=gridLGBM_1.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1.best_params_['max_depth'],
#                         subsample = gridLGBM_2.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2.best_params_['subsample_freq'], 
#                         min_split_gain = gridLGBM_3.best_params_['min_split_gain'],
#                         min_child_weight = gridLGBM_3.best_params_['min_child_weight'],
#                         min_child_samples = gridLGBM_3.best_params_['min_child_samples'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[19]:


# gridLGBM_4 = GridSearchCV(modelLGBM_3, {'reg_lambda': [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1],
#                                           'reg_alpha' : [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_4.fit(X, y)
# print(f'Best hyperparameters: {gridLGBM_4.best_params_}')


# # Modeling and evaluation

# Because there will always be same results using grid search, i comment it out so notebook doesn't need to run for long period every time.

# In[20]:


# modelLGBM_4 = LGBMClassifier(n_estimators=gridLGBM_1.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1.best_params_['max_depth'],
#                         subsample = gridLGBM_2.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2.best_params_['subsample_freq'],
#                         min_split_gain = gridLGBM_3.best_params_['min_split_gain'],
#                         min_child_weight = gridLGBM_3.best_params_['min_child_weight'],
#                         min_child_samples = gridLGBM_3.best_params_['min_child_samples'],
#                         reg_lambda = gridLGBM_4.best_params_['reg_lambda'],
#                         reg_alpha = gridLGBM_4.best_params_['reg_alpha'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)

modelLGBM_4 = LGBMClassifier(n_estimators=80,
                        learning_rate=0.1,
                        num_leaves=10,
                        max_depth=4,
                        subsample = 0.9,
                        colsample_bytree = 0.3,
                        subsample_freq = 3,
                        min_split_gain = 0.1,
                        min_child_weight = 0.0005,
                        min_child_samples = 20,
                        reg_lambda = 0.01,
                        reg_alpha = 0.1,
                        objective = 'binary',
                        class_weight = 'balanced',
                        random_state=12)


# In[21]:


modelLGBM_4.fit(X, y)

# results_lgbm = cross_val_score(modelLGBM_4, X, y, cv=10, n_jobs=-1, scoring=bll).mean()
# results_lgbm_acc = cross_val_score(modelLGBM_4, X, y, cv=10, n_jobs=-1, scoring='accuracy').mean()

scorings = {'acc' : 'accuracy', 'bll' : bll}

results = cross_validate(modelLGBM_4, X, y, cv=10, n_jobs=-1, scoring=scorings)

print(f"LGBM: bll = {results['test_bll'].mean()}, accuracy = {results['test_acc'].mean()}")


# # Feature importance analysis

# Because results were not so great it is always useful to lower down problems dimesnion by further feature selection. For that i used LGBM feature importance and choosed about 30 features (all features that have feature importance value higher or equal than 12, beacuse that threshold gave best results). 

# In[22]:


feat_importances = pd.DataFrame(modelLGBM_4.feature_importances_,
                                index = modelLGBM_4.feature_name_,
                                columns = ['Importance'])
plt.figure(figsize=(30, 30))
ax = sns.barplot(y = feat_importances.index, x='Importance',
              data = feat_importances.sort_values('Importance', ascending=False))
ax.bar_label(ax.containers[0])
plt.show()


# In[23]:


#choosing features that have importance score higher than 8
X_new = X.iloc[:,modelLGBM_4.feature_importances_>=8]
X_new


# # Feature transformation

# Still in progress...

# In[24]:


from sklearn.preprocessing import power_transform
X_tr = pd.DataFrame(power_transform(X_new), index=X_new.index, columns=X_new.columns)
X_tr


# # Creating new models based on feature importance analysis

# Because now features differ realtive to before secound feature selection, we will use grid again to tune hyperparameters for new model.

# In[25]:


# gridLGBM_1_new = GridSearchCV(LGBMClassifier(),
#                               {'learning_rate': [0.01, 0.05, 0.1, 0.26, 0.3, 0.5, 1],
#                                 'num_leaves': [8,10,14,20,26,31],
#                                 'max_depth': [-1,2,4,6,8,10],
#                               'n_estimators' : [50,80,100,200,300]},
#                                  cv=KFold(n_splits=5, shuffle=True, random_state=12), n_jobs=-1,
#                                  scoring='neg_log_loss', verbose=1)
# gridLGBM_1_new.fit(X_tr, y)
# print(f'Best hyperparameters: {gridLGBM_1_new.best_params_}')

# modelLGBM_1_new = LGBMClassifier(n_estimators=gridLGBM_1_new.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1_new.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1_new.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1_new.best_params_['max_depth'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[26]:


# gridLGBM_2_new = GridSearchCV(modelLGBM_1_new, {'subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1], 
#                                           'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
#                                        'subsample_freq': [0,1,3,5,8,10]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_2_new.fit(X_tr, y)
# print(f'Best hyperparameters: {gridLGBM_2_new.best_params_}')

# modelLGBM_2_new = LGBMClassifier(n_estimators=gridLGBM_1_new.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1_new.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1_new.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1_new.best_params_['max_depth'],
#                         subsample = gridLGBM_2_new.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2_new.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2_new.best_params_['subsample_freq'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[27]:


# gridLGBM_3_new = GridSearchCV(modelLGBM_2_new, {'min_split_gain': [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1],
#                                           'min_child_weight' : [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
#                                           'min_child_samples':[5,10,15,20, 25,30]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_3_new.fit(X_tr, y)
# print(f'Best hyperparameters: {gridLGBM_3_new.best_params_}')

# modelLGBM_3_new = LGBMClassifier(n_estimators=gridLGBM_1_new.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1_new.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1_new.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1_new.best_params_['max_depth'],
#                         subsample = gridLGBM_2_new.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2_new.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2_new.best_params_['subsample_freq'], 
#                         min_split_gain = gridLGBM_3_new.best_params_['min_split_gain'],
#                         min_child_weight = gridLGBM_3_new.best_params_['min_child_weight'],
#                         min_child_samples = gridLGBM_3_new.best_params_['min_child_samples'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[28]:


# gridLGBM_4_new = GridSearchCV(modelLGBM_3_new, {'reg_lambda': [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1],
#                                           'reg_alpha' : [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1]},
#                           cv = KFold(n_splits=5, shuffle=True, random_state=12),
#                           n_jobs=-1, scoring='neg_log_loss', verbose=1)
# gridLGBM_4_new.fit(X_tr, y)
# print(f'Best hyperparameters: {gridLGBM_4_new.best_params_}')

# modelLGBM_4_tr = LGBMClassifier(n_estimators=gridLGBM_1_new.best_params_['n_estimators'],
#                         learning_rate=gridLGBM_1_new.best_params_['learning_rate'],
#                         num_leaves=gridLGBM_1_new.best_params_['num_leaves'],
#                         max_depth=gridLGBM_1_new.best_params_['max_depth'],
#                         subsample = gridLGBM_2_new.best_params_['subsample'],
#                         colsample_bytree = gridLGBM_2_new.best_params_['colsample_bytree'],
#                         subsample_freq = gridLGBM_2_new.best_params_['subsample_freq'],
#                         min_split_gain = gridLGBM_3_new.best_params_['min_split_gain'],
#                         min_child_weight = gridLGBM_3_new.best_params_['min_child_weight'],
#                         min_child_samples = gridLGBM_3_new.best_params_['min_child_samples'],
#                         reg_lambda = gridLGBM_4_new.best_params_['reg_lambda'],
#                         reg_alpha = gridLGBM_4_new.best_params_['reg_alpha'],
#                         objective = 'binary',
#                         class_weight = 'balanced',
#                         random_state=12)


# In[29]:


modelLGBM_4_tr = LGBMClassifier(n_estimators=200,
                        learning_rate=0.1,
                        num_leaves=8,
                        max_depth=2,
                        subsample = 0.7,
                        colsample_bytree = 0.9,
                        subsample_freq = 10,
                        min_split_gain = 0.1,
                        min_child_weight = 0.03,
                        min_child_samples = 5,
                        reg_lambda = 0.01,
                        reg_alpha = 0.01,
                        objective = 'binary',
                        class_weight = 'balanced',
                        random_state=12)


# In[30]:


modelLGBM_4_new = LGBMClassifier(n_estimators=200,
                        learning_rate=0.1,
                        num_leaves=8,
                        max_depth=2,
                        subsample = 0.7,
                        colsample_bytree = 0.5,
                        subsample_freq = 10,
                        min_split_gain = 0.1,
                        min_child_weight = 0.0005,
                        min_child_samples = 20,
                        reg_lambda = 0.1,
                        reg_alpha = 0.05,
                        objective = 'binary',
                        class_weight = 'balanced',
                        random_state=12)


# As i said, i am using both self made metrics, accuracy and log loss to evaulate this model. Evalauting is done using 10-Fold CV.

# In[31]:


modelLGBM_4_new.fit(X_new, y)

final_scorings = {'acc': 'accuracy', 'cll' : cll, 'bll' : bll, 'll' : 'neg_log_loss'}
final_results = cross_validate(modelLGBM_4_new, X_new, y, cv=10, n_jobs=-1, scoring=final_scorings)


print('LGBM:')
print('------------------')
print(f"BLL = {final_results['test_bll'].mean()}")
print('------------------')
print(f"CLL = {final_results['test_cll'].mean()}")
print('------------------')
print(f"Accuracy = {final_results['test_acc'].mean()}")
print('------------------')
print(f"LL = {final_results['test_ll'].mean()}")


# In[32]:


modelLGBM_4_tr.fit(X_tr, y)

final_scorings = {'acc': 'accuracy', 'cll' : cll, 'bll' : bll, 'll' : 'neg_log_loss'}
final_results = cross_validate(modelLGBM_4_tr, X_tr, y, cv=10, n_jobs=-1, scoring=final_scorings)


print('LGBM:')
print('------------------')
print(f"BLL = {final_results['test_bll'].mean()}")
print('------------------')
print(f"CLL = {final_results['test_cll'].mean()}")
print('------------------')
print(f"Accuracy = {final_results['test_acc'].mean()}")
print('------------------')
print(f"LL = {final_results['test_ll'].mean()}")


# Best results so far were -0.135 for BLL, -0.235 for CLL, 0.946 for accuracy and 0.147 for LL. I am still focusing on making better results by adapting model to new hyperparamter combinations and better feature engineering.

# # Predicting test data and submiting results

# In[33]:


test_sel=test_poly[out1]
test_sel_new=test_sel.iloc[:,modelLGBM_4.feature_importances_>=8]
test_sel_new


# In[34]:


test_sel_tr = pd.DataFrame(power_transform(test_sel_new),
                           index=test_sel_new.index, columns=test_sel_new.columns)
test_sel_tr


# In[35]:


pred = modelLGBM_4_new.predict_proba(test_sel_tr)
submission=test[['Id']]
submission['class_0']=pred[:,0]
submission['class_1']=pred[:,1]
submission


# In[36]:


submission.to_csv("/kaggle/working/submission.csv",index=False)

